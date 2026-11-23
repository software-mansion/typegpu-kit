import type { SampledFlag, TgpuTexture } from 'typegpu';
import { expect, it } from 'vitest';
import { recordingRoot } from '../src/testing.ts';
import * as mipmaps from '../src/mipmaps.ts';
import { recorded } from './recorded.ts';

const passes = {
  storage: ['begin-compute', 'dispatch', 'dispatch', 'end-compute'],
  render: Array.from({ length: 4 }, () => ['begin-render', 'draw', 'end-render']).flat(),
};

it('records nothing for a single level with no source', () => {
  const { root, log } = recordingRoot();
  const texture = root
    .createTexture({ size: [8, 8], format: 'rgba16float' })
    .$usage('sampled', 'storage');
  const levels = mipmaps.create(root, texture, { combine: mipmaps.mean });
  const encoder = root.createCommandEncoder();

  levels.record(encoder);
  encoder.submit();

  expect(log.events).toEqual(['submit']);
});

it.each(['storage', 'render'] as const)(
  'records a dispatch per level, or a pass per level and layer, through %s usage',
  (usage) => {
    const { root, log } = recordingRoot();
    const texture = root
      .createTexture({ size: [8, 8, 2], format: 'rgba16float', mipLevelCount: 3 })
      .$usage('sampled', usage);
    const levels = mipmaps.create(root, texture, { combine: mipmaps.mean });
    const record = () => {
      const encoder = root.createCommandEncoder();
      levels.record(encoder);
      encoder.submit();
    };
    record();
    const counts = log.counts();

    expect(recorded(log, record)).toEqual([...passes[usage], 'submit']);
    expect(log.counts()).toEqual(counts);
  },
);

it('names the property that keeps a texture or source out of a fill route', () => {
  const { root } = recordingRoot();
  // Props the types reject still reach `create` through a texture typed `any`
  const texture = (props: object): TgpuTexture & SampledFlag =>
    root
      .createTexture({ size: [8, 8], format: 'rgba16float', mipLevelCount: 2, ...props })
      .$usage('sampled', 'storage');
  const combine = mipmaps.mean;
  const levels = mipmaps.create(root, texture({}), { combine });

  expect(() => mipmaps.create(root, texture({ format: 'r32uint' }), { combine })).toThrow(
    "requires a format with float texels, not 'r32uint'",
  );
  expect(() => mipmaps.create(root, texture({ dimension: '3d' }), { combine })).toThrow(
    "requires dimension '2d', not '3d'",
  );
  expect(() => levels.of(texture({}), texture({ dimension: '3d' }))).toThrow(
    "A mipmaps source requires dimension '2d', not '3d'",
  );
  expect(() => levels.of(texture({}), texture({ size: [8, 8, 2] }))).toThrow(
    'The mipmaps source has 2 array layers. It must match the destination texture, which has 1.',
  );
  expect(() => levels.of(texture({ format: 'rgba8unorm' }))).toThrow(
    "format 'rgba16float' and must be created with 'storage' usage",
  );
});
