import {
  perlin2d,
  randf,
} from '../../../typegpu-kit-examples/node_modules/@typegpu/noise/index.mjs';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';

export async function scaledSeeds(root: TgpuRoot): Promise<number[]> {
  const output = root.createMutable(d.arrayOf(d.f32, 64));
  const kernel = tgpu.computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })(
    ({ gid }) => {
      'use gpu';
      randf.seed2(d.vec2f(d.f32(gid.x), d.f32(gid.x)) * 1e-3);
      output.$[gid.x] = randf.sample();
    },
  );
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
  return [...(await output.read())];
}

export async function perlinCache(root: TgpuRoot): Promise<number[]> {
  const cache = perlin2d.staticCache({ root, size: d.vec2u(64, 64) });
  const output = root.createMutable(d.arrayOf(d.vec2f, 64 * 64));
  try {
    const kernel = tgpu.computeFn({
      workgroupSize: [8, 8],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      output.$[gid.y * 64 + gid.x] = cache.getJunctionGradient(d.vec2i(gid.xy));
    });
    root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(8, 8);
    return (await output.read()).flatMap((value) => [value.x, value.y]);
  } finally {
    cache.destroy();
  }
}

export async function randomValues(root: TgpuRoot): Promise<number[]> {
  const output = root.createMutable(d.arrayOf(d.vec2f, 4));
  const kernel = tgpu.computeFn({ workgroupSize: [4], in: { gid: d.builtin.globalInvocationId } })(
    ({ gid }) => {
      'use gpu';
      randf.seed2(d.vec2f(d.f32(gid.x), 0.5));
      output.$[gid.x] = d.vec2f(randf.sample(), randf.sample());
    },
  );
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
  return (await output.read()).flatMap((v) => [v.x, v.y]);
}

export async function storedVectors(root: TgpuRoot): Promise<number[]> {
  const Row = d.struct({ vector: d.vec4u });
  const state = tgpu.privateVar(Row);
  const shared = tgpu.workgroupVar(d.arrayOf(Row, 4));
  const table = tgpu.const(d.arrayOf(d.vec4u, 2), [d.vec4u(1, 2, 3, 4), d.vec4u(5, 6, 7, 8)]);
  const output = root.createMutable(d.arrayOf(d.vec4u, 4));
  const make = tgpu.fn(
    [d.u32],
    d.vec4u,
  )((i) => {
    'use gpu';
    return d.vec4u(table.$[i % 2]);
  });
  const kernel = tgpu.computeFn({ workgroupSize: [4], in: { id: d.builtin.localInvocationIndex } })(
    ({ id }) => {
      'use gpu';
      state.$.vector = make(id);
      state.$.vector.xy = d.vec2u(state.$.vector[1], state.$.vector[0]);
      shared.$[id].vector = d.vec4u(state.$.vector);
      std.workgroupBarrier();
      shared.$[id].vector.zw = d.vec2u(shared.$[id].vector[3], shared.$[id].vector[2]);
      output.$[id] = d.vec4u(shared.$[id].vector);
    },
  );
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
  return (await output.read()).flatMap((v) => [v.x, v.y, v.z, v.w]);
}
