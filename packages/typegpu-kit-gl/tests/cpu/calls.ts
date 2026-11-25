import { d, std, tgpu, type TgpuRoot } from 'typegpu';

export async function recordFields(root: TgpuRoot): Promise<number[]> {
  const Row = d.struct({ position: d.vec3f, member: d.u32, unused: d.vec4f });
  const source = root.createMutable(d.arrayOf(Row, 2), [
    { position: d.vec3f(1, 2, 3), member: 7, unused: d.vec4f(11) },
    { position: d.vec3f(4, 5, 6), member: 8, unused: d.vec4f(12) },
  ]);
  const count = tgpu.privateVar(d.u32);
  const next = tgpu.fn(
    [],
    d.u32,
  )(() => {
    'use gpu';
    const index = count.$;
    count.$ += d.u32(1);
    return index;
  });
  const make = tgpu.fn(
    [d.u32],
    Row,
  )((index) => {
    'use gpu';
    let result = Row();
    result.position = d.vec3f(source.$[index].position);
    result.member = source.$[index].member;
    result.unused = d.vec4f(source.$[index].unused);
    return result;
  });
  const forwarded = tgpu.fn(
    [d.u32],
    Row,
  )((index) => {
    'use gpu';
    return make(index);
  });
  const effectful = tgpu.fn(
    [d.f32],
    Row,
  )((value) => {
    'use gpu';
    return Row({ position: d.vec3f(value), member: next(), unused: d.vec4f(100) });
  });
  const consume = tgpu.fn(
    [Row],
    d.vec4f,
  )((row) => {
    'use gpu';
    return d.vec4f(row.position.x, d.f32(row.member), row.position.yz);
  });
  const output = root.createMutable(d.arrayOf(d.vec4f, 5));
  const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    const before = make(next());
    source.$[0].position.x = 90;
    output.$[0] = consume(before);
    const selected = forwarded(next()).position;
    output.$[1] = d.vec4f(selected, d.f32(count.$));
    const affected = effectful(2).position;
    output.$[2] = d.vec4f(affected, d.f32(count.$));
    output.$[3] = consume(make(0));
    const local = make(0);
    source.$[0].position.y = 95;
    output.$[4] = d.vec4f(local.position, d.f32(local.member));
  });
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
  return (await output.read()).flatMap((value) => [value.x, value.y, value.z, value.w]);
}

export async function matrixLocals(root: TgpuRoot): Promise<number[]> {
  const source = root.createMutable(d.mat2x2f, d.mat2x2f(d.vec2f(1, 2), d.vec2f(3, 4)));
  const output = root.createMutable(d.arrayOf(d.vec4f, 2));
  const count = tgpu.privateVar(d.u32);
  const make = tgpu.fn(
    [],
    d.mat2x2f,
  )(() => {
    'use gpu';
    count.$ += d.u32(1);
    return d.mat2x2f(source.$);
  });
  const inspect = tgpu.fn(
    [d.mat2x2f],
    d.vec4f,
  )((matrix) => {
    'use gpu';
    let copy = d.mat2x2f(matrix);
    let rows = std.transpose(matrix);
    copy.columns[0].x += 3;
    return d.vec4f(copy.columns[0].x, copy.columns[1].y, rows.columns[0].y, rows.columns[1].x);
  });
  const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    let snapshot = std.transpose(make());
    source.$.columns[0].y = 91;
    output.$[0] = inspect(snapshot);
    output.$[1] = d.vec4f(
      snapshot.columns[0].y,
      snapshot.columns[1].x,
      d.f32(count.$),
      source.$.columns[0].y,
    );
  });
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
  return (await output.read()).flatMap((value) => [value.x, value.y, value.z, value.w]);
}

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

export async function vectorResults(root: TgpuRoot): Promise<number[]> {
  const state = tgpu.privateVar(d.u32);
  const output = root.createMutable(d.arrayOf(d.vec4u, 8));
  const next = tgpu.fn(
    [d.u32],
    d.vec2u,
  )((value) => {
    'use gpu';
    state.$ += d.u32(1);
    const x = value + state.$;
    return d.vec2u(x, x * d.u32(3));
  });
  const forward = tgpu.fn(
    [d.u32],
    d.vec2u,
  )((value) => {
    'use gpu';
    return next(value);
  });
  const kernel = tgpu.computeFn({ workgroupSize: [4], in: { id: d.builtin.globalInvocationId } })(({
    id,
  }) => {
    'use gpu';
    let accumulated = d.vec2u();
    for (let i = d.u32(0); i < 3; i++) {
      const a = forward(id.x);
      const b = forward(id.x + d.u32(10));
      accumulated += a + b;
      accumulated += forward(id.x);
    }
    const before = next(id.x);
    std.workgroupBarrier();
    const after = next(id.x);
    output.$[id.x] = d.vec4u(accumulated, before.x, after.x);
  });
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(2);
  return (await output.read()).flatMap((v) => [v.x, v.y, v.z, v.w]);
}

export async function workgroupReturns(root: TgpuRoot): Promise<number[]> {
  const output = root.createMutable(d.arrayOf(d.vec2u, 16));
  const shared = tgpu.workgroupVar(d.arrayOf(d.u32, 4));
  const kernel = tgpu.computeFn({
    workgroupSize: [4],
    in: {
      gid: d.builtin.globalInvocationId,
      wid: d.builtin.workgroupId,
      lid: d.builtin.localInvocationIndex,
    },
  })(({ gid, wid, lid }) => {
    'use gpu';
    if (wid.x === 1) return;
    shared.$[lid] = gid.x + d.u32(1);
    std.workgroupBarrier();
    output.$[gid.x].x = shared.$[(lid + 1) % 4];
    if (wid.x === 2) return;
    std.workgroupBarrier();
    output.$[gid.x].y = shared.$[(lid + 2) % 4];
  });
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(4);
  return (await output.read()).flatMap((v) => [v.x, v.y]);
}

export async function recordValues(root: TgpuRoot): Promise<number[]> {
  const Row = d.struct({ value: d.vec2u, index: d.u32 });
  const source = root.createMutable(d.arrayOf(d.vec2u, 2), [d.vec2u(3, 4), d.vec2u(8, 9)]);
  const output = root.createMutable(d.arrayOf(d.u32, 7));
  const state = tgpu.privateVar(d.u32);
  const next = tgpu.fn(
    [],
    d.u32,
  )(() => {
    'use gpu';
    const current = state.$;
    state.$ += d.u32(1);
    return current;
  });
  const make = tgpu.fn(
    [d.u32],
    Row,
  )((index) => {
    'use gpu';
    return Row({ value: source.$[index], index });
  });
  const forward = tgpu.fn(
    [d.u32],
    Row,
  )((index) => {
    'use gpu';
    return make(index);
  });
  const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    const first = forward(next());
    const second = forward(next());
    first.value.x += d.u32(100);
    source.$[0] = d.vec2u(99);
    output.$[0] = first.value.x;
    output.$[1] = first.value.y;
    output.$[2] = second.value.x;
    output.$[3] = second.value.y;
    output.$[4] = first.index;
    output.$[5] = second.index;
    output.$[6] = state.$;
  });
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
  return output.read();
}

export async function uniformArguments(root: TgpuRoot): Promise<number[]> {
  const View = d.struct({ value: d.vec4u });
  const views = root.createUniform(d.arrayOf(View, 2), [
    { value: d.vec4u(2, 3, 4, 5) },
    { value: d.vec4u(7, 8, 9, 10) },
  ]);
  const layer = tgpu.privateVar(d.u32);
  const output = root.createMutable(d.arrayOf(d.vec2u, 8));
  const index = tgpu.fn(
    [],
    d.u32,
  )(() => {
    'use gpu';
    return layer.$;
  });
  const view = tgpu.fn(
    [],
    View,
  )(() => {
    'use gpu';
    return View(views.$[index()]);
  });
  const consume = tgpu.fn(
    [View],
    d.vec2u,
  )((camera) => {
    'use gpu';
    return d.vec2u(camera.value.xy);
  });
  const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    layer.$ = 0;
    output.$[0] = consume(view());
    layer.$ = 1;
    output.$[1] = consume(view());
    const local = view();
    local.value.x = 99;
    output.$[2] = consume(view());
    layer.$ = 0;
    output.$[3] = consume(view());
    output.$[4] = consume(views.$[0]);
    output.$[5] = consume(views.$[1]);
    output.$[6] = consume(views.$[layer.$]);
    layer.$ = 1;
    output.$[7] = consume(views.$[layer.$]);
  });
  const pipeline = root.createComputePipeline({ compute: kernel });
  pipeline.dispatchWorkgroups(1);
  const first = (await output.read()).flatMap((v) => [v.x, v.y]);
  views.write([{ value: d.vec4u(12, 13, 14, 15) }, { value: d.vec4u(17, 18, 19, 20) }]);
  pipeline.dispatchWorkgroups(1);
  return [...first, ...(await output.read()).flatMap((v) => [v.x, v.y])];
}

export async function literalFields(root: TgpuRoot): Promise<number[]> {
  const Row = d.struct({ used: d.u32, unused: d.vec4u });
  const source = root.createMutable(d.vec4u, d.vec4u(3, 4, 5, 6));
  const output = root.createMutable(d.arrayOf(d.u32, 5));
  const state = tgpu.privateVar(d.u32);
  const next = tgpu.fn(
    [],
    d.vec4u,
  )(() => {
    'use gpu';
    state.$ += d.u32(1);
    return d.vec4u(state.$);
  });
  const pure = tgpu.fn(
    [],
    Row,
  )(() => {
    'use gpu';
    return Row({ used: source.$.x, unused: source.$ });
  });
  const effectful = tgpu.fn(
    [],
    Row,
  )(() => {
    'use gpu';
    return Row({ used: source.$.y, unused: next() });
  });
  const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    const snapshot = pure();
    source.$.x = 99;
    const other = effectful();
    output.$[0] = snapshot.used;
    output.$[1] = other.used;
    output.$[2] = state.$;
    const mutable = pure();
    mutable.used += d.u32(1);
    output.$[3] = mutable.used;
    if (source.$.x === 99) {
      const sibling = pure();
      output.$[4] = sibling.used;
    } else {
      const sibling = pure();
      output.$[4] = sibling.used + d.u32(2);
    }
  });
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
  return output.read();
}

export async function addressReturns(root: TgpuRoot): Promise<number[]> {
  const rows = root.createMutable(d.arrayOf(d.vec4u, 1), [d.vec4u(4, 5, 6, 7)]);
  const output = root.createMutable(d.arrayOf(d.vec4u, 2));
  const read = tgpu.fn(
    [d.u32],
    d.vec4u,
  )((index) => {
    'use gpu';
    const row = rows.$[index];
    return d.vec4u(row);
  });
  const mutate = tgpu.fn(
    [],
    d.u32,
  )(() => {
    'use gpu';
    rows.$[0] = d.vec4u(99);
    return 0;
  });
  const consume = tgpu.fn(
    [d.vec4u, d.u32],
    d.vec4u,
  )((before, _unused) => {
    'use gpu';
    return d.vec4u(before);
  });
  const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    output.$[0] = consume(read(0), mutate());
    output.$[1] = read(0);
  });
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
  return (await output.read()).flatMap((row) => [...row]);
}

export async function conditionalValues(root: TgpuRoot): Promise<number[]> {
  const output = root.createMutable(d.arrayOf(d.vec4f, 4));
  const count = tgpu.privateVar(d.u32);
  const next = tgpu.fn(
    [],
    d.u32,
  )(() => {
    'use gpu';
    count.$ += 1;
    return d.u32(count.$);
  });
  const pipeline = root.createComputePipeline({
    compute: tgpu.computeFn({ workgroupSize: [4], in: { id: d.builtin.globalInvocationId } })(
      ({ id }) => {
        'use gpu';
        const burning = id.x % 2 === 0;
        const spreading = burning ? 0.3 : 0.1;
        const vector = burning ? d.vec2f(2, 3) : d.vec2f(4, 5);
        const nested = burning ? spreading : id.x > 2 ? d.f32(8) : d.f32(6);
        const both = std.select(next(), next(), burning);
        output.$[id.x] = d.vec4f(vector, nested, d.f32(both * 10 + count.$));
      },
    ),
  });
  pipeline.dispatchWorkgroups(1);
  return (await output.read()).flatMap((value) => [...value]);
}
