import { d, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';

it('clamps float conversions while preserving integer bit reinterpretation', async () => {
  const root = cpuRoot();
  const input = root.createReadonly(d.arrayOf(d.vec4f, 3), [
    d.vec4f(-1, 4294967296, 2147483648, -2147483904),
    d.vec4f(-0.5, 1.9, 255.9, 42),
    d.vec4f(0),
  ]);
  root.device.queue.writeBuffer(
    root.unwrap(input.buffer),
    32,
    new Float32Array([NaN, Infinity, -Infinity, 0]),
  );
  const calls = tgpu.privateVar(d.u32);
  const readValue = (index: number) => {
    'use gpu';
    calls.$++;
    return d.vec4f(input.$[index]);
  };
  const unsigned = root.createMutable(d.arrayOf(d.vec4u, 3));
  const signed = root.createMutable(d.arrayOf(d.vec4i, 3));
  const scalars = root.createMutable(d.arrayOf(d.u32, 5));
  root
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [1], in: { id: d.builtin.globalInvocationId } })(
        ({ id }) => {
          'use gpu';
          unsigned.$[id.x] = d.vec4u(readValue(id.x));
          signed.$[id.x] = d.vec4i(input.$[id.x]);
          scalars.$[0] = d.u32(input.$[0].x);
          scalars.$[1] = d.u32(input.$[0].y);
          scalars.$[2] = d.u32(d.i32(input.$[0].z));
          scalars.$[3] = d.u32(d.i32(-1));
          scalars.$[4] = d.u32(d.i32(d.u32(4294967295)));
        },
      ),
    })
    .dispatchWorkgroups(3);
  expect((await unsigned.read()).map((v) => [...v])).toEqual([
    [0, 4294967040, 2147483648, 0],
    [0, 1, 255, 42],
    [0, 4294967040, 0, 0],
  ]);
  expect((await signed.read()).map((v) => [...v])).toEqual([
    [-1, 2147483520, 2147483520, -2147483648],
    [0, 1, 255, 42],
    [0, 2147483520, -2147483648, 0],
  ]);
  expect(await scalars.read()).toEqual([0, 4294967040, 2147483520, 4294967295, 4294967295]);
  root.destroy();
});
