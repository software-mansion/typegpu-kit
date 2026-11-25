import { compile } from '../../src/cpu/index.ts';
import { textureOps, encodeHalf, decodeHalf } from '../../src/cpu/textures.ts';
import type { CpuResources, CpuTexture } from '../../src/cpu/types.ts';
import type { ReplaySnapshot } from './replay.ts';
import { readBuffer } from './example-comparison.ts';

const ROWS = 2048;
interface TraceRow {
  tag: number;
  value: number[];
}
export async function traceAdvection(
  device: GPUDevice,
  wgsl: string,
  groups: Map<number, GPUBindGroupEntry[]>,
  snapshot: ReplaySnapshot,
  resources: CpuResources,
  texels: number[],
  size: number[],
) {
  let records: TraceRow[][] = texels.map(() => []);
  let selected = -1;
  let injectedPositions: number[][] | undefined;
  const positionInput = () => injectedPositions?.[selected];
  const record = (tag: number, value: number[]) => {
    if (selected >= 0) records[selected]!.push({ tag, value: Array.from(value) });
  };
  const select = (id: number[]) => {
    selected = texels.indexOf(id[0]! + size[0]! * (id[1]! + size[1]! * id[2]!));
    sampleIndex = 0;
    return selected >= 0;
  };
  const sample = (texture: CpuTexture, coords: number[], result: number[]) => {
    if (selected < 0) return;
    const level = texture.levels[0]!;
    const dimensions = [level.width, level.height, level.depthOrArrayLayers];
    const position = coords.map((v, axis) => Math.fround(Math.fround(v * dimensions[axis]!) - 0.5));
    const base = position.map(Math.floor);
    record(100, coords);
    record(101, position);
    record(
      102,
      position.map((v, axis) => Math.fround(v - base[axis]!)),
    );
    for (let z = 0; z < 2; z++)
      for (let y = 0; y < 2; y++)
        for (let x = 0; x < 2; x++) {
          const at = [x, y, z].map((v, axis) =>
            Math.min(dimensions[axis]! - 1, Math.max(0, base[axis]! + v)),
          );
          record(103 + x + 2 * y + 4 * z, textureOps.textureLoad(texture, at, 0) as number[]);
        }
    record(111, result);
  };
  let modeled = false;
  let injected: number[][][] | undefined;
  let sampleIndex = 0;
  const filter = (
    texture: CpuTexture,
    sampler: GPUSamplerDescriptor,
    coords: number[],
    lod: number,
    value: number[],
  ) => {
    if (injected) return injected[selected]![sampleIndex++]!;
    if (!modeled) return value;
    const level = texture.levels[0]!;
    const dimensions = [level.width, level.height, level.depthOrArrayLayers];
    const snapped = coords.map(
      (v, i) => (Math.round((v * dimensions[i]! - 0.5) * 256) / 256 + 0.5) / dimensions[i]!,
    );
    return (textureOps.textureSampleLevel(texture, sampler, snapped, lod) as number[]).map((v) =>
      decodeHalf(encodeHalf(v)),
    );
  };
  const scope = globalThis as typeof globalThis & {
    cpuAdvectionTrace?: {
      select: typeof select;
      record: typeof record;
      sample: typeof sample;
      filter: typeof filter;
      position: typeof positionInput;
    };
  };
  const previous = scope.cpuAdvectionTrace;
  scope.cpuAdvectionTrace = { select, record, sample, filter, position: positionInput };
  let cpu = snapshot.code;
  const entry = JSON.parse(/^\/\/#entry (.*)$/m.exec(cpu)![1]!) as { name: string };
  cpu = cpu.replace(
    `function ${entry.name}(id) {`,
    `function ${entry.name}(id) {if(!globalThis.cpuAdvectionTrace.select(id)) return;`,
  );
  const begin = cpu.indexOf('let dt =', cpu.indexOf(`function ${entry.name}(id)`));
  const end = cpu.indexOf('  $textures.textureStore', begin);
  const lines = cpu.slice(begin, end).trimEnd().split('\n');
  if (lines.length !== 16)
    throw new Error(`Advection entry shape changed: ${lines.length} statements`);
  cpu =
    cpu.slice(0, begin) +
    lines
      .map((line, i) => {
        const locals = [...line.matchAll(/(?:let|const) (cpuComponent(?:_\d+)?|dt)=?/g)].map(
          (m) => m[1]!,
        );
        const values =
          i === 13 || i === 15 ? ['cpuComponent_72', 'cpuComponent_73', 'cpuComponent_74'] : locals;
        return `${line}\nglobalThis.cpuAdvectionTrace.record(${i},[${values.join(',')}]);${i === 10 ? 'const auditPosition=globalThis.cpuAdvectionTrace.position();if(auditPosition){cpuComponent_89=auditPosition[0];cpuComponent_90=auditPosition[1];cpuComponent_91=auditPosition[2];}' : ''}`;
      })
      .join('\n') +
    '\n' +
    `globalThis.cpuAdvectionTrace.record(113,[Math.fround(Math.exp(Math.fround(-smoke_settings_f32[6]*dt))),spongeAt(cpuComponent_7,cpuComponent_8,cpuComponent_9),Math.fround(1-Math.fround(Math.exp(Math.fround(-cpuComponent_149*dt))))]);\n` +
    cpu.slice(end);
  cpu = `const auditSample=$textures.textureSampleLevel; const auditStore=$textures.textureStore;
$textures={...$textures,textureSampleLevel:(t,s,c,l)=>{const v=globalThis.cpuAdvectionTrace.filter(t,s,c,l,auditSample(t,s,c,l));globalThis.cpuAdvectionTrace.sample(t,c,v);return v;},textureStore:(t,c,v)=>{globalThis.cpuAdvectionTrace.record(112,v);return auditStore(t,c,v);}};\n${cpu}`;
  const originalRecords = records;
  try {
    compile(cpu).dispatch(...snapshot.workgroups, resources);
    modeled = true;
    records = texels.map(() => []);
    compile(cpu).dispatch(...snapshot.workgroups, resources);
  } finally {
    if (previous) scope.cpuAdvectionTrace = previous;
    else delete scope.cpuAdvectionTrace;
  }

  const binding = Math.max(...groups.keys()) + 1;
  const output = device.createBuffer({
    size: texels.length * ROWS * 32,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const entryStart = wgsl.indexOf('@compute');
  const open = wgsl.indexOf('{', entryStart);
  const selectCode = texels
    .map((texel, i) => `if(linear==${texel}u){audit_slot=${i}u;}`)
    .join('\n');
  let shader =
    wgsl.slice(0, open + 1) +
    `\nlet linear=id.x+${size[0]}u*(id.y+${size[1]}u*id.z);${selectCode}if(audit_slot==0xffffffffu){return;}\n` +
    wgsl.slice(open + 1);
  const start = shader.indexOf('  let dt =', open);
  const finish = shader.indexOf('  textureStore', start);
  const statements = shader.slice(start, finish).trimEnd().split('\n');
  const names = [
    'dt',
    'at',
    'back',
    'restored',
    'moved',
    'held',
    'swirl',
    'rise',
    'toward',
    'confinement',
    'position',
    'lift',
    'pushed',
    'moved',
    'dragged',
    'moved',
  ];
  if (statements.length !== names.length) throw new Error('WGSL advection entry shape changed');
  shader =
    shader.slice(0, start) +
    statements
      .map(
        (line, i) =>
          `${line}\naudit_write(${i}u,${i === 0 ? 'vec4f(dt,0,0,0)' : i === 5 ? 'vec4f(held,0,0)' : i === 14 ? 'dragged' : `vec4f(${names[i]},0)`});`,
      )
      .join('\n') +
    '\n' +
    'audit_write(113u,vec4f(exp(-smoke_settings.drag*dt),spongeAt(at),1-exp(-dragged.w*dt),0));\n' +
    shader.slice(finish);
  shader = shader
    .replaceAll('textureSampleLevel(', 'audit_sample(')
    .replaceAll('textureStore(', 'audit_store(');
  shader += `
var<private> audit_slot:u32=0xffffffffu;
var<private> audit_row:u32=0u;
@group(${binding}) @binding(0) var<storage,read_write> audit_output:array<vec4f>;
fn audit_write(tag:u32,v:vec4f){if(audit_slot>=${texels.length}u || audit_row>=${ROWS}u){return;}let at=(audit_slot*${ROWS}u+audit_row)*2u;audit_output[at]=vec4f(f32(tag),1,0,0);audit_output[at+1u]=v;audit_row++;}
fn audit_sample(t:texture_3d<f32>,s:sampler,c:vec3f,l:f32)->vec4f{
let dimensions=textureDimensions(t,0);let position=c*vec3f(dimensions)-0.5;let base=vec3i(floor(position));
audit_write(100u,vec4f(c,0));audit_write(101u,vec4f(position,0));audit_write(102u,vec4f(fract(position),0));
for(var z=0;z<2;z++){for(var y=0;y<2;y++){for(var x=0;x<2;x++){let at=clamp(base+vec3i(x,y,z),vec3i(0),vec3i(dimensions)-1);audit_write(u32(103+x+2*y+4*z),textureLoad(t,at,0));}}}
let value=textureSampleLevel(t,s,c,l);audit_write(111u,value);return value;}
fn audit_store(t:texture_storage_3d<rgba16float,write>,c:vec3u,v:vec4f){audit_write(112u,v);textureStore(t,c,v);}`;
  try {
    const pipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: shader }) },
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    for (const [group, entries] of groups)
      pass.setBindGroup(
        group,
        device.createBindGroup({ layout: pipeline.getBindGroupLayout(group), entries }),
      );
    pass.setBindGroup(
      binding,
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(binding),
        entries: [{ binding: 0, resource: { buffer: output } }],
      }),
    );
    pass.dispatchWorkgroups(...snapshot.workgroups);
    pass.end();
    device.queue.submit([encoder.finish()]);
    const values = new Float32Array(await readBuffer(device, output));
    const dawnRecords = texels.map((_texel, i) =>
      Array.from({ length: ROWS }, (_, row) => {
        const at = (i * ROWS + row) * 8;
        return values[at + 1] === 1
          ? { tag: values[at]!, value: Array.from(values.subarray(at + 4, at + 8)) }
          : undefined;
      }).filter((row): row is TraceRow => !!row),
    );
    const modelRecords = records;
    injected = dawnRecords.map((rows) =>
      rows.filter((row) => row.tag === 111).map((row) => row.value),
    );
    records = texels.map(() => []);
    scope.cpuAdvectionTrace = { select, record, sample, filter, position: positionInput };
    try {
      compile(cpu).dispatch(...snapshot.workgroups, resources);
    } finally {
      if (previous) scope.cpuAdvectionTrace = previous;
      else delete scope.cpuAdvectionTrace;
    }
    const injectedRecords = records;
    injectedPositions = dawnRecords.map((rows) => rows.find((row) => row.tag === 10)!.value);
    records = texels.map(() => []);
    scope.cpuAdvectionTrace = { select, record, sample, filter, position: positionInput };
    try {
      compile(cpu).dispatch(...snapshot.workgroups, resources);
    } finally {
      if (previous) scope.cpuAdvectionTrace = previous;
      else delete scope.cpuAdvectionTrace;
    }
    const decorate = (rows: TraceRow[]) =>
      rows.map((row) => ({
        ...row,
        ...(row.tag >= 103 && row.tag <= 110 ? { halfBits: row.value.map(encodeHalf) } : {}),
      }));
    return texels.map((texel, i) => ({
      texel,
      cell: [
        texel % size[0]!,
        Math.floor(texel / size[0]!) % size[1]!,
        Math.floor(texel / (size[0]! * size[1]!)),
      ],
      cpu: decorate(originalRecords[i]!),
      modeled: decorate(modelRecords[i]!),
      injectedSamples: decorate(injectedRecords[i]!),
      injectedPosition: decorate(records[i]!),
      dawn: decorate(dawnRecords[i]!),
    }));
  } finally {
    output.destroy();
  }
}

const orderedHalf = (bits: number) => (bits & 0x8000 ? -(bits & 0x7fff) : bits);
const halfGap = (a: number, b: number) =>
  Math.abs(orderedHalf(encodeHalf(a)) - orderedHalf(encodeHalf(b)));
export function summarizeAdvection(
  traces: Awaited<ReturnType<typeof traceAdvection>>,
  stored: { storedDawn: number[] }[],
) {
  let samples = 0,
    sampleModelMaxHalfUlps = 0,
    injectedSampleMaxHalfUlps = 0,
    injectedPositionMaxHalfUlps = 0,
    coordinateMaxUlps = 0;
  const bits = new Uint32Array(1),
    floats = new Float32Array(bits.buffer);
  const floatBits = (value: number) => {
    floats[0] = value;
    return bits[0]!;
  };
  const residuals = [];
  for (let t = 0; t < traces.length; t++) {
    const trace = traces[t]!;
    const expected = stored[t]!.storedDawn.map(decodeHalf);
    const sampleStore = trace.injectedSamples.find((row) => row.tag === 112)!.value;
    const positionStore = trace.injectedPosition.find((row) => row.tag === 112)!.value;
    const sampleGap = Math.max(...sampleStore.map((value, c) => halfGap(value, expected[c]!)));
    const positionGap = Math.max(...positionStore.map((value, c) => halfGap(value, expected[c]!)));
    injectedSampleMaxHalfUlps = Math.max(injectedSampleMaxHalfUlps, sampleGap);
    injectedPositionMaxHalfUlps = Math.max(injectedPositionMaxHalfUlps, positionGap);
    residuals.push({ cell: trace.cell, sampleGap, positionGap });
    const sameInputs = trace.injectedSamples.filter((row) => row.tag === 100);
    let sampleIndex = 0;
    for (let i = 0; i < trace.dawn.length; i++) {
      if (trace.dawn[i]!.tag !== 100) continue;
      samples++;
      const coords = trace.dawn[i]!.value;
      for (let c = 0; c < 3; c++)
        coordinateMaxUlps = Math.max(
          coordinateMaxUlps,
          Math.abs(floatBits(coords[c]!) - floatBits(sameInputs[sampleIndex]!.value[c]!)),
        );
      sampleIndex++;
      const weights = trace.dawn[i + 2]!.value.slice(0, 3).map((v) => Math.round(v * 256) / 256);
      for (let c = 0; c < 4; c++) {
        let value = 0;
        for (let tap = 0; tap < 8; tap++) {
          let weight = 1;
          for (let axis = 0; axis < 3; axis++)
            weight *= tap & (1 << axis) ? weights[axis]! : 1 - weights[axis]!;
          value += trace.dawn[i + 3 + tap]!.value[c]! * weight;
        }
        sampleModelMaxHalfUlps = Math.max(
          sampleModelMaxHalfUlps,
          halfGap(value, trace.dawn[i + 11]!.value[c]!),
        );
      }
    }
  }
  return {
    texels: traces.length,
    samples,
    sampleModelMaxHalfUlps,
    coordinateMaxUlps,
    injectedSampleMaxHalfUlps,
    injectedPositionMaxHalfUlps,
    residuals,
  };
}
