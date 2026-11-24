import { d, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';

it('runs ordinary TypeGPU compute pipelines and reads their output', async () => {
  const root = cpuRoot();
  const output = root.createMutable(d.arrayOf(d.u32, 16));
  const kernel = tgpu.computeFn({ workgroupSize: [8], in: { id: d.builtin.globalInvocationId } })(({
    id,
  }) => {
    'use gpu';
    output.$[id.x] = id.x * d.u32(7);
  });
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(2);
  expect(await output.read()).toEqual(Array.from({ length: 16 }, (_, i) => i * 7));
  root.destroy();
});

it('multiplies matrix columns with a vector', async () => {
  const root = cpuRoot();
  const output = root.createMutable(d.vec2f);
  const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    const matrix = d.mat2x2f(d.vec2f(1, 2), d.vec2f(3, 4));
    output.$ = matrix * d.vec2f(2, 3);
  });
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
  const result = await output.read();
  expect([result.x, result.y]).toEqual([11, 16]);
  root.destroy();
});

it('keeps scalarized vector locals scoped to their shader function', async () => {
  const root = cpuRoot();
  const source = root.createUniform(d.vec2f, d.vec2f(3.5, -2.5));
  const output = root.createMutable(d.vec2f);
  const twice = tgpu.fn(
    [d.vec2f],
    d.vec2f,
  )((value) => {
    'use gpu';
    const copy = d.vec2f(value);
    return copy * 2;
  });
  const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    const value = d.vec2f(source.$);
    const changed = twice(value);
    output.$ = changed + value;
  });
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
  const value = await output.read();
  expect([value.x, value.y]).toEqual([10.5, -7.5]);
  root.destroy();
});

it('scopes buffer address aliases to their declaring helper', async () => {
  const root = cpuRoot();
  const Row = d.struct({ value: d.vec2f });
  const source = root.createUniform(Row, { value: d.vec2f(3, 7) });
  const output = root.createMutable(d.arrayOf(d.vec2f, 2));
  const read = tgpu.fn(
    [Row],
    d.vec2f,
  )((view) => {
    'use gpu';
    return d.vec2f(view.value);
  });
  const nested = tgpu.fn(
    [],
    d.vec2f,
  )(() => {
    'use gpu';
    const view = source.$;
    return read(view);
  });
  const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    output.$[0] = nested();
    output.$[1] = read(Row({ value: d.vec2f(11, 13) }));
  });
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
  expect((await output.read()).map((v) => [v.x, v.y])).toEqual([
    [3, 7],
    [11, 13],
  ]);
  root.destroy();
});

it('preserves TypeGPU reference assignment validation', () => {
  const root = cpuRoot();
  const privateState = tgpu.privateVar(d.vec2u);
  const output = root.createMutable(d.vec2u);
  const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    output.$ = privateState.$;
  });
  expect(() => root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1)).toThrow(
    'references cannot be assigned',
  );
  root.destroy();
});
