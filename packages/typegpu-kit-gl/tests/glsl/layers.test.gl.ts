import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { afterAll, beforeAll, expect, it } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { roots } from '../device/compare.ts';

let gpu: TgpuRoot;
let gl: TgpuRoot;
beforeAll(async () => {
  ({ gpu, gl } = await roots());
});
afterAll(() => {
  gpu.destroy();
  gl.destroy();
});

const W = 6;
const H = 5;

const vec = (v: d.v4f) => [v.x, v.y, v.z, v.w].map((c) => Number(c.toFixed(3)));

/** Fills a 3d texture, then sums it along z a column per invocation, and reads both back. */
async function volume(root: TgpuRoot, depth: number) {
  const make = () =>
    root
      .createTexture({ size: [W, H, depth], format: 'rgba16float', dimension: '3d' })
      .$usage('sampled', 'storage');
  const source = make();
  const summed = make();
  const fill = tgpu.bindGroupLayout({
    out: { storageTexture: d.textureStorage3d('rgba16float', 'write-only') },
  });
  const sum = tgpu.bindGroupLayout({
    from: { texture: d.texture3d(d.f32) },
    into: { storageTexture: d.textureStorage3d('rgba16float', 'write-only') },
  });
  const read = tgpu.bindGroupLayout({ from: { texture: d.texture3d(d.f32) } });
  const out = root.createMutable(d.arrayOf(d.vec4f, W * H * depth * 2));
  const gid = { id: d.builtin.globalInvocationId };
  root
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [2, 2, 2], in: gid })(({ id }) => {
        'use gpu';
        if (id.x >= W || id.y >= H || id.z >= depth) return;
        const value = d.vec4f(d.f32(id.x) * 0.25, d.f32(id.y) * 0.5, d.f32(id.z), 1);
        std.textureStore(fill.$.out, id, value);
      }),
    })
    .with(root.createBindGroup(fill, { out: source }))
    .dispatchWorkgroups(Math.ceil(W / 2), Math.ceil(H / 2), Math.ceil(depth / 2));
  root
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [4, 4, 1], in: gid })(({ id }) => {
        'use gpu';
        if (id.x >= W || id.y >= H) return;
        let total = d.vec4f();
        for (let k = d.u32(0); k < depth; k++) {
          total += std.textureLoad(sum.$.from, d.vec3u(id.xy, k), 0);
          std.textureStore(sum.$.into, d.vec3u(id.xy, k), total);
        }
      }),
    })
    .with(root.createBindGroup(sum, { from: source, into: summed }))
    .dispatchWorkgroups(Math.ceil(W / 4), Math.ceil(H / 4), 1);
  const copy = (texture: typeof source, at: number) =>
    root
      .createComputePipeline({
        compute: tgpu.computeFn({ workgroupSize: [1] })(() => {
          'use gpu';
          for (let z = d.u32(0); z < depth; z++) {
            for (let y = d.u32(0); y < H; y++) {
              for (let x = d.u32(0); x < W; x++) {
                const i = (z * H + y) * W + x + at;
                out.$[i] = std.textureLoad(read.$.from, d.vec3u(x, y, z), 0);
              }
            }
          }
        }),
      })
      .with(root.createBindGroup(read, { from: texture }))
      .dispatchWorkgroups(1);
  copy(source, 0);
  copy(summed, W * H * depth);
  return (await out.read()).map(vec);
}

/** Writes a 2d-array texture's layers, whole, through a range of layers, or at its second level. */
async function layers(root: TgpuRoot, view: 'whole' | 'range' | 'level') {
  const texture = root
    .createTexture({
      size: [W, H, 7],
      format: 'rgba16float',
      mipLevelCount: view === 'level' ? 2 : 1,
    })
    .$usage('sampled', 'storage');
  const layout = tgpu.bindGroupLayout({
    into: { storageTexture: d.textureStorage2dArray('rgba16float', 'write-only') },
  });
  const read = tgpu.bindGroupLayout({ from: { texture: d.texture2dArray(d.f32) } });
  const level = view === 'level' ? 1 : 0;
  const [w, h] = [W >> level, H >> level];
  const count = view === 'range' ? 3 : 7;
  root
    .createComputePipeline({
      compute: tgpu.computeFn({
        workgroupSize: [2, 2, 1],
        in: { id: d.builtin.globalInvocationId },
      })(({ id }) => {
        'use gpu';
        if (id.x >= w || id.y >= h) return;
        std.textureStore(
          layout.$.into,
          id.xy,
          id.z,
          d.vec4f(d.f32(id.x), d.f32(id.y), d.f32(id.z), 1),
        );
      }),
    })
    .with(
      root.createBindGroup(layout, {
        into: texture.createView(d.textureStorage2dArray('rgba16float', 'write-only'), {
          baseMipLevel: level,
          mipLevelCount: 1,
          baseArrayLayer: view === 'range' ? 2 : 0,
          arrayLayerCount: count,
        }),
      }),
    )
    .dispatchWorkgroups(Math.ceil(w / 2), Math.ceil(h / 2), count);
  const out = root.createMutable(d.arrayOf(d.vec4f, w * h * 7));
  root
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [1] })(() => {
        'use gpu';
        for (let z = d.u32(0); z < 7; z++) {
          for (let y = d.u32(0); y < h; y++) {
            for (let x = d.u32(0); x < w; x++) {
              out.$[(z * h + y) * w + x] = std.textureLoad(read.$.from, d.vec2u(x, y), z, level);
            }
          }
        }
      }),
    })
    .with(root.createBindGroup(read, { from: texture.createView(d.texture2dArray(d.f32)) }))
    .dispatchWorkgroups(1);
  return (await out.read()).map(vec);
}

it.each([1, 3, 4, 5, 9])('writes a 3d texture of %i slices as WebGPU does', async (depth) => {
  expect(await volume(gl, depth)).toEqual(await volume(gpu, depth));
  const kernels = [...(gl.device as unknown as GLDevice).stats.kernels.keys()];
  expect(kernels.filter((label) => label.endsWith('(gpu)')).length).toBeGreaterThan(0);
});

it.each(['whole', 'range', 'level'] as const)(
  'writes the layers of a 2d-array texture as WebGPU does (%s)',
  async (view) => {
    expect(await layers(gl, view)).toEqual(await layers(gpu, view));
  },
);
