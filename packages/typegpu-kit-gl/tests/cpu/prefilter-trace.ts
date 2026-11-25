import { cubeSampleEnvelope, cubeSampleModel } from './cube-model.ts';
import { compile } from '../../src/cpu/index.ts';
import { textureOps, encodeHalf, decodeHalf } from '../../src/cpu/textures.ts';
import type { CpuResources, CpuTexture } from '../../src/cpu/types.ts';
import type { ReplaySnapshot } from './replay.ts';
import { readBuffer } from './example-comparison.ts';

const ROWS = 512;
interface Row {
  tag: number;
  value: number[];
}
export async function tracePrefilter(
  device: GPUDevice,
  wgsl: string,
  groups: Map<number, GPUBindGroupEntry[]>,
  snapshot: ReplaySnapshot,
  resources: CpuResources,
  texels: number[],
  side: number,
) {
  let records: Row[][] = texels.map(() => []);
  let selected = -1,
    iteration = 0;
  let injected: number[][][] | undefined;
  let texture: CpuTexture, sampler: GPUSamplerDescriptor;
  const select = (id: number[]) => {
    selected = texels.indexOf(id[0]! + side * (id[1]! + side * id[2]!));
    iteration = 0;
    return selected >= 0;
  };
  const record = (tag: number, value: number[]) => {
    records[selected]!.push({ tag, value: Array.from(value) });
    if (tag === 1) iteration = value[3]!;
  };
  const sample = (t: CpuTexture, s: GPUSamplerDescriptor, c: number[], l: number, v: number[]) => {
    texture = t;
    sampler = s;
    record(2, [...c, l]);
    record(3, v);
    return injected?.[selected]?.[iteration] ?? v;
  };
  const scope = globalThis as typeof globalThis & {
    cpuPrefilterTrace?: { select: typeof select; record: typeof record; sample: typeof sample };
  };
  const previous = scope.cpuPrefilterTrace;
  let cpu = snapshot.code.replace(
    'function prefilter(gid) {',
    'function prefilter(gid) {if(!globalThis.cpuPrefilterTrace.select(gid))return;',
  );
  cpu = cpu.replace(
    /(let lod = [^\n]+;)/,
    '$1globalThis.cpuPrefilterTrace.record(0,[cosTheta,sinTheta,nl,spread]);globalThis.cpuPrefilterTrace.record(1,[density,share,lod,i]);',
  );
  cpu = cpu.replace(
    /((?:let|const) nl = [^;\n]+;)/,
    '$1globalThis.cpuPrefilterTrace.record(4,[i,nl,cosTheta,sinTheta]);',
  );
  cpu = `const prefSample=$textures.textureSampleLevel,prefStore=$textures.textureStore;$textures={...$textures,textureSampleLevel:(t,s,c,l)=>globalThis.cpuPrefilterTrace.sample(t,s,c,l,prefSample(t,s,c,l)),textureStore:(t,c,l,v)=>{globalThis.cpuPrefilterTrace.record(112,v);return prefStore(t,c,l,v);}};\n${cpu}`;
  const run = () => {
    scope.cpuPrefilterTrace = { select, record, sample };
    try {
      compile(cpu).dispatch(...snapshot.workgroups, resources);
    } finally {
      if (previous) scope.cpuPrefilterTrace = previous;
      else delete scope.cpuPrefilterTrace;
    }
  };
  run();
  const original = records;
  const group = Math.max(...groups.keys()) + 1,
    entryStart = wgsl.indexOf('@compute'),
    open = wgsl.indexOf('{', entryStart);
  let shader =
    wgsl.slice(0, open + 1) +
    `let linear=gid.x+${side}u*(gid.y+${side}u*gid.z);${texels.map((t, i) => `if(linear==${t}u){pref_slot=${i}u;}`).join('')}if(pref_slot==0xffffffffu){return;}` +
    wgsl.slice(open + 1);
  shader = shader
    .replace(
      /(      let lod = [^\n]+;)/,
      '$1pref_record(0u,vec4f(cosTheta,sinTheta,nl,spread));pref_record(1u,vec4f(density,share,lod,f32(i)));',
    )
    .replace(/(let nl = [^;\n]+;)/, '$1pref_record(4u,vec4f(f32(i),nl,cosTheta,sinTheta));')
    .replaceAll('textureSampleLevel(', 'pref_sample(')
    .replaceAll('textureStore(', 'pref_store(');
  shader += `\nvar<private> pref_slot:u32=0xffffffffu;var<private> pref_row:u32=0u;
@group(${group}) @binding(0) var<storage,read_write> pref_output:array<vec4f>;
fn pref_record(tag:u32,v:vec4f){if(pref_row>=${ROWS}u){return;}let at=(pref_slot*${ROWS}u+pref_row)*2u;pref_output[at]=vec4f(f32(tag),1,0,0);pref_output[at+1u]=v;pref_row++;}
fn pref_sample(t:texture_cube<f32>,s:sampler,c:vec3f,l:f32)->vec4f{let v=textureSampleLevel(t,s,c,l);pref_record(2u,vec4f(c,l));pref_record(3u,v);return v;}
fn pref_store(t:texture_storage_2d_array<rgba16float,write>,c:vec2u,l:u32,v:vec4f){pref_record(112u,v);textureStore(t,c,l,v);}`;
  const output = device.createBuffer({
    size: texels.length * ROWS * 32,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  try {
    const pipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: shader }) },
    });
    const encoder = device.createCommandEncoder(),
      pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    for (const [g, entries] of groups)
      pass.setBindGroup(
        g,
        device.createBindGroup({ layout: pipeline.getBindGroupLayout(g), entries }),
      );
    pass.setBindGroup(
      group,
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(group),
        entries: [{ binding: 0, resource: { buffer: output } }],
      }),
    );
    pass.dispatchWorkgroups(...snapshot.workgroups);
    pass.end();
    device.queue.submit([encoder.finish()]);
    const words = new Float32Array(await readBuffer(device, output));
    const dawn = texels.map((_texel, i) =>
      Array.from({ length: ROWS }, (_, row) => {
        const at = (i * ROWS + row) * 8;
        return words[at + 1] === 1
          ? { tag: words[at]!, value: Array.from(words.subarray(at + 4, at + 8)) }
          : undefined;
      }).filter((row): row is Row => !!row),
    );
    injected = dawn.map((rows) => {
      const values: number[][] = [];
      let at = 0;
      for (const row of rows) {
        if (row.tag === 1) at = row.value[3]!;
        if (row.tag === 3) values[at] = row.value;
      }
      return values;
    });
    records = texels.map(() => []);
    run();
    return texels.map((texel, i) => {
      const originalStore = original[i]!.find((row) => row.tag === 112)!.value,
        sameStore = records[i]!.find((row) => row.tag === 112)!.value,
        expectedStore = dawn[i]!.find((row) => row.tag === 112)!.value;
      const samples = dawn[i]!.filter((row) => row.tag === 2).map((row, j) => ({
        coordinates: row.value,
        cpu: textureOps.textureSampleLevel(
          texture,
          sampler,
          row.value.slice(0, 3),
          row.value[3]!,
        ) as number[],
        dawn: dawn[i]!.filter((item) => item.tag === 3)[j]!.value,
        envelope: cubeSampleEnvelope(texture, sampler, row.value.slice(0, 3), row.value[3]!),
        cornerModel: cubeSampleModel(texture, sampler, row.value.slice(0, 3), row.value[3]!),
        mip6Model: cubeSampleModel(
          texture,
          sampler,
          row.value.slice(0, 3),
          Math.floor(decodeHalf(encodeHalf(row.value[3]!)) * 64) / 64,
        ),
        cornerTruncatedLod: cubeSampleModel(
          texture,
          sampler,
          row.value.slice(0, 3),
          row.value[3]!,
          true,
          true,
        ),
        cornerOnly: cubeSampleModel(texture, sampler, row.value.slice(0, 3), row.value[3]!, false),
      }));
      return {
        texel,
        cell: [texel % side, Math.floor(texel / side) % side, Math.floor(texel / (side * side))],
        branchFlips: original[i]!.filter((row) => row.tag === 4).flatMap((row) => {
          const other = dawn[i]!.find((item) => item.tag === 4 && item.value[0] === row.value[0]);
          return other && row.value[1]! > 0 !== other.value[1]! > 0
            ? [
                {
                  iteration: row.value[0],
                  cpu: row.value,
                  dawn: other.value,
                  delta: Math.abs(row.value[1]! - other.value[1]!),
                },
              ]
            : [];
        }),
        originalStore,
        sameStore,
        expectedStore,
        sameSamplesMaxHalfUlps: Math.max(
          ...sameStore.map((v, c) => Math.abs(encodeHalf(v) - encodeHalf(expectedStore[c]!))),
        ),
        cpu: original[i]!,
        dawn: dawn[i]!,
        samples,
      };
    });
  } finally {
    output.destroy();
  }
}
