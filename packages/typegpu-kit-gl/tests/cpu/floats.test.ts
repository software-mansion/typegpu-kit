import { d, std, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';

it('rounds literals, conversions and arithmetic at every f32 step', async () => {
  let code = '';
  const root = cpuRoot({
    onModule: (source) => {
      code = source;
    },
  });
  const input = root.createReadonly(d.vec4f, d.vec4f(0.1, 0.3, 1e-3, 16777216));
  const output = root.createMutable(d.arrayOf(d.vec4f, 64));
  try {
    const kernel = tgpu.computeFn({
      workgroupSize: [64],
      in: { id: d.builtin.globalInvocationId },
    })(({ id }) => {
      'use gpu';
      const seed = d.f32(id.x) * 1e-3;
      const sum = (input.$.x + input.$.y) * input.$.z;
      let accumulated = d.f32(input.$.w);
      accumulated += 1;
      accumulated -= input.$.w;
      const converted = d.f32(id.x + d.u32(16777216));
      output.$[id.x] = d.vec4f(seed, sum, accumulated, converted);
    });
    root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
    const values = await output.read();
    const round = Math.fround;
    for (let i = 0; i < values.length; i++) {
      expect([...values[i]!]).toEqual([
        round(round(i) * round(0.001)),
        round(round(round(0.1) + round(0.3)) * round(0.001)),
        0,
        round(i + 16777216),
      ]);
    }
    expect(code).toContain('0.0010000000474974513');
  } finally {
    root.destroy();
  }
});

it('rounds dot, mix and fract intermediate operations', async () => {
  const root = cpuRoot();
  const a = d.vec3f(0.1, 0.3, -0.7),
    b = d.vec3f(-0.2, 0.7, 0.9);
  const input = root.createReadonly(d.arrayOf(d.vec3f, 2), [a, b]);
  const output = root.createMutable(d.vec4f);
  try {
    const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      output.$ = d.vec4f(
        std.dot(input.$[0], input.$[1]),
        std.mix(input.$[0].x, input.$[1].x, 0.3),
        std.fract(input.$[0].z),
        std.sqrt(input.$[0].y),
      );
    });
    root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
    const r = Math.fround,
      left = [...a].map(r),
      right = [...b].map(r);
    const dot = left.map((value, i) => r(value * right[i]!)).reduce((sum, value) => r(sum + value));
    expect([...(await output.read())]).toEqual([
      dot,
      r(r(left[0]! * r(1 - r(0.3))) + r(right[0]! * r(0.3))),
      r(left[2]! - Math.floor(left[2]!)),
      r(Math.sqrt(left[1]!)),
    ]);
  } finally {
    root.destroy();
  }
});

it('rounds abstract literals on either side of comparisons and converts named integers', async () => {
  const root = cpuRoot();
  const input = root.createReadonly(d.f32, 0.1);
  const integer = root.createReadonly(d.u32, 16777217);
  const output = root.createMutable(d.vec4f);
  try {
    const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      const e = integer.$;
      output.$ = d.vec4f(
        std.select(0, 1, 0.1 === input.$),
        std.select(0, 1, input.$ === 0.1),
        std.select(0, 1, 0.1 < input.$),
        d.f32(e),
      );
    });
    root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
    expect([...(await output.read())]).toEqual([1, 1, 0, 16777216]);
  } finally {
    root.destroy();
  }
});
