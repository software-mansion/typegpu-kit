import { expect, it } from 'vitest';
import { recordingRoot } from '../src/testing.ts';
import * as mipmaps from '../src/mipmaps.ts';

const passes = {
  storage: ['begin-compute', 'dispatch', 'dispatch', 'end-compute'],
  render: Array.from({ length: 4 }, () => ['begin-render', 'draw', 'end-render']).flat(),
};

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
    log.events.length = 0;

    record();

    expect(log.events).toEqual([...passes[usage], 'submit']);
    expect(log.counts()).toEqual(counts);
  },
);
