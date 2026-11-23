import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';

type Vec3 = readonly [number, number, number];

export const WORKGROUP = [8, 4, 2] as const;
/** A compute entry point over the cells of a box. */
export const kernel = tgpu.computeFn({
  workgroupSize: [...WORKGROUP],
  in: { id: d.builtin.globalInvocationId },
});
/** The workgroups over a box of cells, or over one colour's half of its rows. */
export const groups = (size: Vec3, halved = false) => [
  Math.ceil((halved ? Math.ceil(size[0] / 2) : size[0]) / WORKGROUP[0]),
  Math.ceil(size[1] / WORKGROUP[1]),
  Math.ceil(size[2] / WORKGROUP[2]),
];
/** Whether a cell lies in a box of `size` cells. */
export const inside = (cell: d.v3i, size: d.v3i) => {
  'use gpu';
  return std.all(std.ge(cell, d.vec3i())) && std.all(std.lt(cell, size));
};
// Spreading would allocate on every dispatch, which Hermes collects in frames
export const dispatch = (pass: TgpuComputePass, counts: number[]) =>
  pass.dispatchWorkgroups(counts[0], counts[1], counts[2]);
/** A 3D texture that kernels store to and sample, with a view for each. */
export const volume = (
  root: TgpuRoot,
  size: Vec3,
  format: 'rgba16float' | 'rg16float',
  name: string,
) => {
  const texture = root
    .createTexture({ size: [...size], format, dimension: '3d' })
    .$usage('storage', 'sampled')
    .$name(name);
  return {
    texture,
    read: texture.createView(d.texture3d(d.f32)),
    write: texture.createView(d.textureStorage3d(format, 'write-only')),
  };
};
