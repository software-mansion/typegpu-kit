import { cameras, targets } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';
import type { TgpuCommandEncoder } from 'typegpu';
import { expect, it } from 'vitest';
import { ao, bloom, dof, exposure, finish, fxaa, post, taa } from '../../src/index.ts';

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

it('takes a rendered source, its region and a depth once per record', () => {
  const { root } = recordingRoot();
  const size = { width: 64, height: 32 };
  const screen = targets.create(root, size, {
    hdr: { format: 'rgba16float', sampled: true },
    shown: { format: 'rgba8unorm' },
    depth: { format: 'depth32float', sampled: true },
  });
  const view = cameras.orbit(root, { size, target: [0, 0, 0], distance: 6 });
  // The texture stays while the region alternates between two objects
  const regions = [size, { width: 32, height: 16 }];
  const reads = { output: 0, region: 0, depth: 0 };
  const source: post.Rendered = {
    get output() {
      reads.output++;
      return screen.textures.hdr;
    },
    get region() {
      return regions[reads.region++ % 2];
    },
  };
  const depth = () => (reads.depth++, screen.textures.depth);

  const blur = dof.create(root, view, { source, depth });
  const occlusion = ao.create(root, view, { depth });
  const glow = bloom.create(root, { source });
  const eye = exposure.create(root, { source });
  const smooth = fxaa.create(root, { format: 'rgba8unorm', source });
  const last = finish.create(root, { format: 'rgba8unorm', source });
  const resolve = taa.create(root, { source, depth: null });
  const shown = screen.views.shown;
  const records: ((encoder: TgpuCommandEncoder) => void)[] = [
    (encoder) => glow.record(encoder),
    (encoder) => eye.record(encoder),
    (encoder) => smooth.record(encoder, shown),
    (encoder) => last.record(encoder, shown),
    (encoder) => resolve.record(encoder),
    (encoder) => blur.record(encoder),
  ];

  for (const record of records) {
    const encoder = root.createCommandEncoder();
    Object.assign(reads, { output: 0, region: 0 });
    record(encoder);
    expect(reads).toMatchObject({ output: 1, region: 1 });
  }
  // The region the last record read
  blur.record(root.createCommandEncoder());
  expect(blur.region).toMatchObject(regions[1]);

  reads.depth = 0;
  occlusion.record(root.createCommandEncoder());
  expect(reads.depth).toBe(1);
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
