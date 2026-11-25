import { d, tgpu, type TgpuRoot } from 'typegpu';

export async function scopedLocals(root: TgpuRoot) {
  const S = d.struct({ a: d.f32, b: d.f32 });
  const rows = root.createReadonly(d.arrayOf(S, 2), [
    { a: 1, b: 2 },
    { a: 3, b: 5 },
  ]);
  const other = root.createReadonly(d.arrayOf(S, 2), [
    { a: 10, b: 20 },
    { a: 30, b: 50 },
  ]);
  const sibling = tgpu.fn(
    [d.u32, d.bool],
    d.f32,
  )((i, first) => {
    'use gpu';
    let sum = d.f32(0);
    if (first) {
      const item = rows.$[i];
      sum += item.b;
    } else {
      const item = S(other.$[i]);
      sum += item.b;
    }
    if (first) {
      const v = rows.$[i];
      sum += v.a;
    } else {
      const v = other.$[i].a;
      sum += v;
    }
    return sum;
  });
  const loop = tgpu.fn(
    [d.bool],
    d.f32,
  )((first) => {
    'use gpu';
    let sum = d.f32(0);
    for (let i = 0; i < 2; i++) {
      if (first) {
        const item = rows.$[i];
        sum += item.b;
      } else {
        const item = S(other.$[i]);
        sum += item.b;
      }
    }
    return sum;
  });
  const nested = tgpu.fn(
    [d.bool],
    d.f32,
  )((first) => {
    'use gpu';
    const item = rows.$[1];
    let sum = item.a;
    if (first) {
      // oxlint-disable-next-line no-shadow -- deliberate lexical shadowing regression
      const item = S(other.$[1]);
      sum += item.b;
    }
    sum += item.b;
    return sum;
  });
  const cases = tgpu.fn(
    [d.u32],
    d.f32,
  )((i) => {
    'use gpu';
    let sum = d.f32(0);
    switch (i) {
      case 0: {
        const item = rows.$[1];
        sum += item.b;
        break;
      }
      case 1:
      case 2: {
        const item = S(other.$[1]);
        sum += item.b;
        break;
      }
      default: {
        sum = -1;
      }
    }
    return sum;
  });
  const vectors = tgpu.fn(
    [d.bool],
    d.f32,
  )((first) => {
    'use gpu';
    let sum = d.f32(0);
    if (first) {
      const value = d.vec2f(2, 3);
      sum += value.x;
    } else {
      const value = d.f32(7);
      sum += value;
    }
    if (first) {
      const value = d.vec2f(5, 6);
      sum += value.y;
    } else {
      const value = d.vec2f(11, 13);
      sum += value.y;
    }
    return sum;
  });
  const output = root.createMutable(d.arrayOf(d.f32, 10));
  root
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [1], in: { id: d.builtin.globalInvocationId } })(
        ({ id }) => {
          'use gpu';
          const first = id.x === 0;
          output.$[id.x] = sibling(1, first);
          output.$[id.x + 2] = loop(first);
          output.$[id.x + 4] = nested(first);
          output.$[id.x + 6] = cases(id.x);
          output.$[id.x + 8] = vectors(first);
        },
      ),
    })
    .dispatchWorkgroups(2);
  return output.read();
}
