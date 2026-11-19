import { targets } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';
import { expect, it } from 'vitest';
import { exposure, finish, fxaa, post, taa } from '../../src/index.ts';

const texture = (width: number, height: number) =>
  ({ props: { size: [width, height] } }) as unknown as post.FloatTexture;

it("reads a source of either form, the region option before a rendered source's own", () => {
  const picture = texture(64, 36);
  const half = { width: 32, height: 18 };
  const quarter = { width: 16, height: 9 };
  const rendered: post.Rendered = { output: picture, region: half };

  expect(post.textureOf(() => picture)()).toBe(picture);
  expect(post.textureOf(rendered)()).toBe(picture);
  expect(post.regionOf(() => picture, undefined)).toBeUndefined();
  expect(post.regionOf(rendered, undefined)).toBe(half);
  expect(post.regionOf(rendered, quarter)).toBe(quarter);
});

it('fits a region inside its texture and reports only a change', () => {
  const picture = texture(64, 36);
  const size = [0, 0];

  expect(post.fitRegion(size, picture, { width: 32, height: 100 })).toBe(true);
  expect(size).toEqual([32, 36]);
  expect(post.fitRegion(size, picture, { width: 32, height: 100 })).toBe(false);
  expect(post.fitRegion(size, picture, undefined)).toBe(true);
  expect(size).toEqual([64, 36]);
});

it('names the pipelines of TAA, exposure, FXAA and the finish for the debug breakdown', async () => {
  const { root } = recordingRoot();
  const labels: (string | undefined)[] = [];
  const device = root.device;
  for (const key of ['createRenderPipelineAsync', 'createComputePipelineAsync'] as const) {
    const create = device[key].bind(device) as (descriptor: GPUPipelineDescriptorBase) => unknown;
    Object.assign(device, {
      [key]: (descriptor: GPUPipelineDescriptorBase) => {
        labels.push(descriptor.label);
        return create(descriptor);
      },
    });
  }
  const screen = targets.create(
    root,
    { width: 8, height: 8 },
    {
      hdr: { format: 'rgba16float', sampled: true },
    },
  );
  const source = () => screen.textures.hdr;
  const made = [
    taa.create(root, { source, depth: null }),
    exposure.create(root, { source }),
    fxaa.create(root, { format: 'rgba8unorm', source }),
    finish.create(root, { format: 'rgba8unorm', source }),
  ];
  await Promise.all(made.map((each) => each.initAsync()));

  expect(labels).toEqual(['taa', 'exposure', 'fxaa', 'finish']);
});
