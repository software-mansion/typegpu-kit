import { memo } from '@typegpu/kit/memo';
import { d, std, tgpu, type TgpuRoot, type TgpuTexture } from 'typegpu';

const Picture = tgpu.bindGroupLayout({
  picture: { texture: d.texture2d(d.f32) },
  smooth: { sampler: 'filtering' },
});

/** Samples the bound picture at `uv`, filtered. */
export const sample = (uv: d.v2f) => {
  'use gpu';
  return std.textureSampleLevel(Picture.$.picture, Picture.$.smooth, uv, 0);
};

/**
 * Returns the bind group through which `sample` reads a picture, made again only when a resized
 * target gives a new texture.
 */
export function pictureGroups(root: TgpuRoot) {
  const smooth = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  return memo((picture: TgpuTexture & { usableAsSampled: true }) =>
    root.createBindGroup(Picture, { picture, smooth }),
  );
}
