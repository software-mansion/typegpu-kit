import { counters, sorter, workgroup } from '@typegpu/kit';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';
import { indexedValues } from './indexed.ts';

async function composite(root: TgpuRoot): Promise<number[]> {
  const Row = d.struct({ position: d.vec3f, velocity: d.vec3f });
  const rows = root.createMutable(d.arrayOf(Row, 16));
  const lanes = tgpu.workgroupVar(d.arrayOf(d.u32, 8));
  const values = root.createMutable(d.arrayOf(d.u32, 16));
  const counts = counters.create(root, ['hits']);
  const kernel = tgpu.computeFn({
    workgroupSize: [8],
    in: { gid: d.builtin.globalInvocationId, lid: d.builtin.localInvocationIndex },
  })(({ gid, lid }) => {
    'use gpu';
    const row = rows.$[gid.x];
    row.velocity = d.vec3f(d.f32(gid.x), 2, 3);
    let copy = d.vec3f(row.velocity);
    copy.x = 7;
    row.position = row.velocity * 2 + copy;
    lanes.$[lid] = gid.x + d.u32(1);
    std.workgroupBarrier();
    values.$[gid.x] = lanes.$[(lid + d.u32(1)) % d.u32(8)];
    counts.add.hits(3);
  });
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(2);
  const result = [
    ...(await rows.read()).flatMap((row) => [
      row.position.x,
      row.position.y,
      row.position.z,
      row.velocity.x,
      row.velocity.y,
      row.velocity.z,
    ]),
    ...(await values.read()),
    ...(await counts.buffer.read()),
  ];
  counts.destroy();
  return result;
}

async function sorted(root: TgpuRoot): Promise<number[]> {
  const keys = root
    .createBuffer(
      d.arrayOf(d.u32, 257),
      Array.from({ length: 257 }, (_, i) => (i * 7919) % 613),
    )
    .$usage('storage');
  const sort = sorter.create(root, keys, { keyBits: 16 });
  const total = workgroup.reduce(workgroup.sum(d.u32), 16);
  const output = root.createMutable(d.arrayOf(d.u32, 16));
  const kernel = tgpu.computeFn({
    workgroupSize: [16],
    in: { id: d.builtin.localInvocationIndex },
  })(({ id }) => {
    'use gpu';
    output.$[id] = total(id + d.u32(1), id);
  });
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
  const encoder = root.createCommandEncoder();
  const pass = encoder.beginComputePass();
  sort.sort(pass);
  pass.end();
  encoder.submit();
  const result = [...(await keys.read()), ...(await output.read())];
  sort.destroy();
  return result;
}

async function privateBarriers(root: TgpuRoot): Promise<number[]> {
  const privateState = tgpu.privateVar(d.vec2u);
  const output = root.createMutable(d.arrayOf(d.vec2u, 16));
  const kernel = tgpu.computeFn({ workgroupSize: [8], in: { id: d.builtin.globalInvocationId } })(({
    id,
  }) => {
    'use gpu';
    privateState.$ = d.vec2u(id.x, id.x + d.u32(7));
    std.workgroupBarrier();
    privateState.$ *= d.u32(3);
    std.workgroupBarrier();
    output.$[id.x] = d.vec2u(privateState.$);
  });
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(2);
  return (await output.read()).flatMap((v) => [v.x, v.y]);
}

for (const [name, run] of [
  ['composite vectors, barrier exchange, counters', composite],
  ['Kit radix sorter and workgroup reduction', sorted],
  ['private vector state across barriers', privateBarriers],
  ['indexed vector, matrix, struct and array writes', indexedValues],
] as const) {
  it(`matches dawn: ${name}`, async () => {
    const gpu = await gpuRoot();
    const cpu = cpuRoot();
    try {
      const expected = await run(gpu);
      const actual = await run(cpu);
      expect(actual.length).toBe(expected.length);
      for (let i = 0; i < expected.length; i++)
        expect(actual[i], `${name}: word ${i}, CPU=${actual[i]}, dawn=${expected[i]}`).toBe(
          expected[i],
        );
    } finally {
      gpu.destroy();
      cpu.destroy();
    }
  });
}
