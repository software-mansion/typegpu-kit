import { compile } from '../../src/cpu/index.ts';
import type { CpuResources } from '../../src/cpu/types.ts';
import type { ReplaySnapshot } from './replay.ts';
import { readBuffer } from './example-comparison.ts';

const ROWS = 4096;
interface Row {
  tag: number;
  value: number[];
}
export async function traceMaterial(
  device: GPUDevice,
  wgsl: string,
  groups: Map<number, GPUBindGroupEntry[]>,
  snapshot: ReplaySnapshot,
  resources: CpuResources,
  texels: number[],
  size: number[],
) {
  let records: Row[][] = texels.map(() => []);
  let selected = -1;
  let centers: number[][] | undefined;
  const centerInput = () => centers?.[selected];
  const select = (id: number[]) => {
    selected = texels.indexOf(id[0]! + size[0]! * (id[1]! + size[1]! * id[2]!));
    return selected >= 0;
  };
  const record = (tag: number, value: number[]) => {
    if (selected >= 0) records[selected]!.push({ tag, value: Array.from(value) });
  };
  const scope = globalThis as typeof globalThis & {
    cpuMaterialTrace?: {
      select: typeof select;
      record: typeof record;
      centerInput: typeof centerInput;
    };
  };
  const previous = scope.cpuMaterialTrace;
  scope.cpuMaterialTrace = { select, record, centerInput };
  let cpu = snapshot.code;
  const entry = JSON.parse(/^\/\/#entry (.*)$/m.exec(cpu)![1]!) as { name: string };
  cpu = cpu.replace(
    `function ${entry.name}(gid) {`,
    `function ${entry.name}(gid) {if(!globalThis.cpuMaterialTrace.select(gid))return;`,
  );
  const start = cpu.indexOf('function groundDistance('),
    end = cpu.indexOf('\n}', start);
  const body = cpu.slice(start, end);
  const args = /function groundDistance\(([^)]+)\)/.exec(body)![1]!.split(',');
  const [px, py, pz] = args;
  const n = body.indexOf('let n =');
  const centreLine = body.slice(0, n).trimEnd().split('\n').at(-1)!;
  const centre = [...centreLine.matchAll(/let (\w+)=/g)].map((match) => match[1]!);
  if (centre.length !== 2) throw new Error('Ground-distance centre shape changed');
  const afterN = body.indexOf('\n', n);
  const audit = `\nglobalThis.cpuMaterialTrace.record(0,[${px},${py},${pz},reach]);globalThis.cpuMaterialTrace.record(1,[h,gx,gz,estimate]);globalThis.cpuMaterialTrace.record(2,[side,best,step_1,n]);globalThis.cpuMaterialTrace.record(3,[Math.fround(Math.fround(Math.fround(${px}/Math.fround(0.1))-0.5)/Math.fround(step_1)),Math.fround(Math.fround(Math.fround(${pz}/Math.fround(0.1))-0.5)/Math.fround(step_1)),${centre.join(',')}]);const auditCenter=globalThis.cpuMaterialTrace.centerInput();if(auditCenter){${centre[0]}=auditCenter[2];${centre[1]}=auditCenter[3];}`;
  let instrumented = body.slice(0, afterN) + audit + body.slice(afterN);
  instrumented = instrumented.replace(
    /(\(best\) =[^\n]+);/g,
    '$1;globalThis.cpuMaterialTrace.record(4,[dx,dz,across,gap]);globalThis.cpuMaterialTrace.record(5,[best]);',
  );
  instrumented = instrumented.replace(
    '  return Math.fround((side)*(best));',
    '  globalThis.cpuMaterialTrace.record(6,[side*best]);return Math.fround((side)*(best));',
  );
  cpu = cpu.slice(0, start) + instrumented + cpu.slice(end);
  cpu = `const materialStore=$textures.textureStore;$textures={...$textures,textureStore:(t,c,v)=>{globalThis.cpuMaterialTrace.record(112,v);return materialStore(t,c,v);}};\n${cpu}`;
  try {
    compile(cpu).dispatch(...snapshot.workgroups, resources);
  } finally {
    if (previous) scope.cpuMaterialTrace = previous;
    else delete scope.cpuMaterialTrace;
  }
  const group = Math.max(...groups.keys()) + 1;
  let shader = wgsl;
  const entryStart = shader.indexOf('@compute'),
    open = shader.indexOf('{', entryStart);
  shader =
    shader.slice(0, open + 1) +
    `let linear=gid.x+${size[0]}u*(gid.y+${size[1]}u*gid.z);${texels.map((t, i) => `if(linear==${t}u){audit_slot=${i}u;}`).join('')}if(audit_slot==0xffffffffu){return;}` +
    shader.slice(open + 1);
  shader = shader.replace(
    /(  let n =[^\n]+;)/,
    '$1\naudit_write(0u,vec4f(p,reach));audit_write(1u,vec4f(h,gx,gz,estimate));audit_write(2u,vec4f(side,best,f32(step_1),f32(n)));audit_write(3u,vec4f((p.xz/0.1-0.5)/f32(step_1),vec2f(centre)));',
  );
  shader = shader.replace(
    /(      best =[^\n]+;)/,
    '$1\naudit_write(4u,vec4f(f32(dx),f32(dz),across,gap));audit_write(5u,vec4f(best,0,0,0));',
  );
  shader = shader.replace(
    '  return (side * best);',
    '  audit_write(6u,vec4f(side*best,0,0,0));return (side * best);',
  );
  shader = shader.replaceAll('textureStore(', 'audit_store(');
  const format = /texture_storage_3d<([^,]+),/.exec(shader)![1];
  shader += `\nvar<private> audit_slot:u32=0xffffffffu;var<private> audit_row:u32=0u;
@group(${group}) @binding(0) var<storage,read_write> audit_output:array<vec4f>;
fn audit_write(tag:u32,v:vec4f){if(audit_row>=${ROWS}u){return;}let at=(audit_slot*${ROWS}u+audit_row)*2u;audit_output[at]=vec4f(f32(tag),1,0,0);audit_output[at+1u]=v;audit_row++;}
fn audit_store(t:texture_storage_3d<${format},write>,c:vec3u,v:vec4f){audit_write(112u,v);textureStore(t,c,v);}`;
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
    const dawnRows = texels.map((_texel, i) =>
      Array.from({ length: ROWS }, (_, row) => {
        const at = (i * ROWS + row) * 8;
        return words[at + 1] === 1
          ? { tag: words[at]!, value: Array.from(words.subarray(at + 4, at + 8)) }
          : undefined;
      }).filter((row): row is Row => !!row),
    );
    const original = records;
    records = texels.map(() => []);
    centers = dawnRows.map((rows) => rows.find((row) => row.tag === 3)!.value);
    scope.cpuMaterialTrace = { select, record, centerInput };
    try {
      compile(cpu).dispatch(...snapshot.workgroups, resources);
    } finally {
      if (previous) scope.cpuMaterialTrace = previous;
      else delete scope.cpuMaterialTrace;
    }
    return texels.map((texel, i) => ({
      texel,
      cell: [
        texel % size[0]!,
        Math.floor(texel / size[0]!) % size[1]!,
        Math.floor(texel / (size[0]! * size[1]!)),
      ],
      cpu: original[i]!,
      sameCenter: records[i]!,
      dawn: dawnRows[i]!,
    }));
  } finally {
    output.destroy();
  }
}
