import { compile } from '../../src/cpu/index.ts';
import type { CpuResources } from '../../src/cpu/types.ts';
import type { ReplaySnapshot } from './replay.ts';
import { readBuffer } from './example-comparison.ts';
import { encodeHalf } from '../../src/cpu/textures.ts';

const ROWS = 512;
interface Row {
  tag: number;
  value: number[];
}
export async function traceSky(
  device: GPUDevice,
  wgsl: string,
  groups: Map<number, GPUBindGroupEntry[]>,
  snapshot: ReplaySnapshot,
  resources: CpuResources,
  texels: number[],
  side: number,
) {
  let records: Row[][] = texels.map(() => []);
  let selected = -1;
  let inputs: Row[][] | undefined;
  const cursors = new Map<number, number>();
  const select = (id: number[]) => {
    selected = texels.indexOf(id[0]! + side * (id[1]! + side * id[2]!));
    cursors.clear();
    return selected >= 0;
  };
  const value = (tag: number, data: number[]) => {
    records[selected]!.push({ tag, value: Array.from(data) });
    if (!inputs) return data;
    const index = cursors.get(tag) ?? 0;
    cursors.set(tag, index + 1);
    return inputs[selected]!.filter((row) => row.tag === tag)[index]!.value;
  };
  const scope = globalThis as typeof globalThis & {
    cpuSkyTrace?: { select: typeof select; value: typeof value };
  };
  const previous = scope.cpuSkyTrace;
  let cpu = snapshot.code;
  const entry = JSON.parse(/^\/\/#entry (.*)$/m.exec(cpu)![1]!) as { name: string };
  cpu = cpu.replace(
    `function ${entry.name}(gid) {`,
    `function ${entry.name}(gid) {if(!globalThis.cpuSkyTrace.select(gid))return;`,
  );
  cpu = cpu.replace(/(let height = [^\n]+;)/g, (line) =>
    line.includes('(radius)')
      ? `${line}const skyView=globalThis.cpuSkyTrace.value(1,[radius,height,length_1,t]);radius=skyView[0];height=skyView[1];`
      : `${line}height=globalThis.cpuSkyTrace.value(0,[height+6360,height,r,mu])[1];`,
  );
  cpu = cpu.replace(
    'let length_1 = Math.fround(((ground) >= (0)) ? ground : toTop(Math.fround(r),Math.fround(cpuComponent_10)));',
    'let length_1 = Math.fround(((top)=>ground>=0?ground:top)(toTop(Math.fround(r),Math.fround(cpuComponent_10))));',
  );
  if (process.env.CPU_SKY_DISCRIMINANT === '1')
    cpu = cpu.replace(
      /(let disc = [^\n]+;)/,
      '$1globalThis.cpuSkyTrace.value(4,[Math.fround(r*r),Math.fround(Math.fround(mu*mu)-1),Math.fround(Math.fround(r*r)*Math.fround(Math.fround(mu*mu)-1)),disc]);',
    );
  cpu = cpu
    .replace('function toGround(r,mu)', 'function skyGround(r,mu)')
    .replace('function toTop(r,mu)', 'function skyTop(r,mu)');
  cpu += `\nfunction toGround(r,mu){const v=skyGround(r,mu);return globalThis.cpuSkyTrace.value(2,[r,mu,v,0])[2];}function toTop(r,mu){const v=skyTop(r,mu);return globalThis.cpuSkyTrace.value(3,[r,mu,v,0])[2];}`;
  cpu = `const skyStore=$textures.textureStore;$textures={...$textures,textureStore:(t,c,l,v)=>{globalThis.cpuSkyTrace.value(112,v);return skyStore(t,c,l,v);}};\n${cpu}`;
  const run = () => {
    scope.cpuSkyTrace = { select, value };
    try {
      compile(cpu).dispatch(...snapshot.workgroups, resources);
    } finally {
      if (previous) scope.cpuSkyTrace = previous;
      else delete scope.cpuSkyTrace;
    }
  };
  run();
  const original = records;
  const group = Math.max(...groups.keys()) + 1;
  const entryStart = wgsl.indexOf('@compute'),
    open = wgsl.indexOf('{', entryStart);
  let shader =
    wgsl.slice(0, open + 1) +
    `let linear=gid.x+${side}u*(gid.y+${side}u*gid.z);${texels.map((t, i) => `if(linear==${t}u){sky_slot=${i}u;}`).join('')}if(sky_slot==0xffffffffu){return;}` +
    wgsl.slice(open + 1);
  shader = shader.replace(/(    let height = [^\n]+;)/g, (line) =>
    line.includes('radius')
      ? `${line}sky_value(1u,vec4f(radius,height,length_1,t));`
      : `${line}sky_value(0u,vec4f(height+6360,height,r,mu));`,
  );
  if (process.env.CPU_SKY_DISCRIMINANT === '1')
    shader = shader.replace(
      /(  let disc = [^\n]+;)/,
      '$1sky_value(4u,vec4f(r*r,mu*mu-1,(r*r)*(mu*mu-1),disc));',
    );
  shader = shader
    .replace('fn toGround(', 'fn skyGround(')
    .replace('fn toTop(', 'fn skyTop(')
    .replaceAll('textureStore(', 'sky_store(');
  shader += `\nvar<private> sky_slot:u32=0xffffffffu;var<private> sky_row:u32=0u;
@group(${group}) @binding(0) var<storage,read_write> sky_output:array<vec4f>;
fn sky_value(tag:u32,v:vec4f){if(sky_row>=${ROWS}u){return;}let at=(sky_slot*${ROWS}u+sky_row)*2u;sky_output[at]=vec4f(f32(tag),1,0,0);sky_output[at+1u]=v;sky_row++;}
fn toGround(r:f32,mu:f32)->f32{let v=skyGround(r,mu);sky_value(2u,vec4f(r,mu,v,0));return v;}
fn toTop(r:f32,mu:f32)->f32{let v=skyTop(r,mu);sky_value(3u,vec4f(r,mu,v,0));return v;}
fn sky_store(t:texture_storage_2d_array<rgba16float,write>,c:vec2u,l:u32,v:vec4f){sky_value(112u,v);textureStore(t,c,l,v);}`;
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
    for (const [index, entries] of groups)
      pass.setBindGroup(
        index,
        device.createBindGroup({ layout: pipeline.getBindGroupLayout(index), entries }),
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
    inputs = dawn;
    records = texels.map(() => []);
    run();
    const bits = new Uint32Array(1),
      float = new Float32Array(bits.buffer);
    const fbits = (v: number) => {
      float[0] = v;
      return bits[0]!;
    };
    return texels.map((texel, i) => {
      const radii = original[i]!.filter((row) => row.tag === 0 || row.tag === 1),
        others = dawn[i]!.filter((row) => row.tag === 0 || row.tag === 1);
      const a = records[i]!.find((row) => row.tag === 112)!.value,
        b = dawn[i]!.find((row) => row.tag === 112)!.value;
      return {
        texel,
        cell: [texel % side, Math.floor(texel / side) % side, Math.floor(texel / (side * side))],
        radiusMaxUlps: Math.max(
          ...radii.map((row, j) => Math.abs(fbits(row.value[0]!) - fbits(others[j]!.value[0]!))),
        ),
        sameInputsMaxHalfUlps: Math.max(
          ...a.map((v, c) => Math.abs(encodeHalf(v) - encodeHalf(b[c]!))),
        ),
        cpu: original[i]!,
        dawn: dawn[i]!,
        sameInputs: records[i]!,
      };
    });
  } finally {
    output.destroy();
  }
}
