import { d, std, tgpu, type TgpuRoot } from 'typegpu';

export async function callValues(root: TgpuRoot): Promise<number[]> {
  const count = tgpu.privateVar(d.u32);
  const output = root.createMutable(d.arrayOf(d.vec4u, 8));
  const scalar = tgpu.fn(
    [],
    d.u32,
  )(() => {
    'use gpu';
    count.$ += d.u32(1);
    return count.$;
  });
  const vector = tgpu.fn(
    [],
    d.vec2u,
  )(() => {
    'use gpu';
    count.$ += d.u32(1);
    return d.vec2u(count.$, count.$ * d.u32(10));
  });
  const together = tgpu.fn(
    [d.u32, d.vec2u, d.u32],
    d.vec4u,
  )((first, middle, last) => {
    'use gpu';
    return d.vec4u(first, middle, last);
  });
  const synchronized = tgpu.fn(
    [d.vec2u],
    d.vec2u,
  )((value) => {
    'use gpu';
    std.workgroupBarrier();
    return value * d.u32(2);
  });
  const kernel = tgpu.computeFn({ workgroupSize: [8], in: { id: d.builtin.localInvocationIndex } })(
    ({ id }) => {
      'use gpu';
      const combined = together(scalar(), vector(), scalar());
      const doubled = synchronized(vector());
      output.$[id] = d.vec4u(combined.x, combined.y, combined.w, doubled.y);
    },
  );
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
  return (await output.read()).flatMap((value) => [value.x, value.y, value.z, value.w]);
}

export async function callSnapshot(root: TgpuRoot): Promise<number[]> {
  const Row = d.struct({ value: d.vec2u });
  const source = root.createMutable(Row, { value: d.vec2u(4, 5) });
  const output = root.createMutable(d.vec2u);
  const snapshot = tgpu.fn(
    [Row],
    d.vec2u,
  )((before) => {
    'use gpu';
    source.$.value = d.vec2u(99);
    return d.vec2u(before.value);
  });
  const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    output.$ = snapshot(source.$);
  });
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
  const result = await output.read();
  return [result.x, result.y];
}

export async function callSwizzles(root: TgpuRoot): Promise<number[]> {
  const output = root.createMutable(d.arrayOf(d.vec4f, 2));
  const make = tgpu.fn(
    [d.f32],
    d.vec4f,
  )((value) => {
    'use gpu';
    return d.vec4f(value, 2, 3, 4);
  });
  const exchange = tgpu.fn([d.ptrFn(d.vec4f)])((value) => {
    'use gpu';
    value.$.yx = d.vec2f(value.$.xy);
  });
  const kernel = tgpu.computeFn({ workgroupSize: [2], in: { id: d.builtin.localInvocationIndex } })(
    ({ id }) => {
      'use gpu';
      let values = d.arrayOf(d.vec4f, 2)([d.vec4f(1, 2, 3, 4), d.vec4f(5, 6, 7, 8)]);
      values[id].yx = d.vec2f(values[id].xy);
      let copy = d.vec4f(values[id]);
      exchange(d.ref(copy));
      copy.z = make(9).x;
      copy.w = std.dot(make(3).yx, d.vec2f(1));
      output.$[id] = d.vec4f(copy);
    },
  );
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
  return (await output.read()).flatMap((value) => [value.x, value.y, value.z, value.w]);
}

export async function callLocalSnapshot(root: TgpuRoot): Promise<number[]> {
  const Row = d.struct({ value: d.vec2u });
  const output = root.createMutable(d.vec2u);
  const mutate = tgpu.fn(
    [d.ptrFn(Row)],
    d.u32,
  )((value) => {
    'use gpu';
    value.$.value = d.vec2u(99);
    return d.u32(0);
  });
  const take = tgpu.fn(
    [Row, d.u32],
    d.vec2u,
  )((value, ignored) => {
    'use gpu';
    return value.value + d.vec2u(ignored);
  });
  const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    let local = Row({ value: d.vec2u(4, 5) });
    output.$ = take(local, mutate(d.ref(local)));
  });
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
  const result = await output.read();
  return [result.x, result.y];
}

export async function vectorPrecedence(root: TgpuRoot): Promise<number[]> {
  const input = root.createReadonly(d.arrayOf(d.vec3f, 2), [d.vec3f(2, 3, 5), d.vec3f(7, 11, 13)]);
  const output = root.createMutable(d.arrayOf(d.vec3f, 3));
  const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    const a = input.$[0];
    const b = input.$[1];
    output.$[0] = std.cross(a, b) * 2;
    output.$[1] = 3 * std.cross(a, b);
    output.$[2] = std.reflect(a, d.vec3f(0, 1, 0)) * 4;
  });
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
  return (await output.read()).flatMap((v) => [v.x, v.y, v.z]);
}

export async function callEffects(root: TgpuRoot): Promise<number[]> {
  const state = tgpu.privateVar(d.u32);
  const source = root.createMutable(d.vec2u, d.vec2u(4, 5));
  const output = root.createMutable(d.arrayOf(d.u32, 3));
  const load = tgpu.fn(
    [],
    d.vec2u,
  )(() => {
    'use gpu';
    return d.vec2u(source.$);
  });
  const change = tgpu.fn(
    [],
    d.u32,
  )(() => {
    'use gpu';
    state.$ += d.u32(1);
    source.$ = d.vec2u(99);
    return d.u32(0);
  });
  const take = tgpu.fn(
    [d.vec2u, d.u32],
    d.vec2u,
  )((value, _unused) => {
    'use gpu';
    return d.vec2u(value);
  });
  const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    const before = take(load(), change());
    output.$[0] = before.x;
    output.$[1] = before.y;
    output.$[2] = state.$;
  });
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
  return output.read();
}
