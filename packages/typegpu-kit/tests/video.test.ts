import { expect, it } from 'vitest';
import { frames, video } from '../src/index.ts';
import { recordingRoot } from '../src/testing.ts';

it('fits each new frame once, turned upright, to cover or be contained', () => {
  const { root, log } = recordingRoot();
  const covered = video.create(root, { size: [4, 4] });
  const contained = video.create(root, { size: [4, 4], fit: 'contain' });
  const frame = {
    texture: root
      .unwrap(root.createTexture({ size: [1, 1], format: 'rgba8unorm' }).$usage('sampled'))
      .createView() as unknown as GPUExternalTexture,
    width: 8,
    height: 4,
    rotation: 90 as const,
    serial: 1,
  };
  const source = {
    latest: (_device: GPUDevice, seen?: number) => (frame.serial === seen ? undefined : frame),
  };
  covered.attach(source);
  contained.attach(source);
  const loop = frames.create(root, (each) => {
    covered.update(each);
    contained.update(each);
  });
  const draws = () => log.events.filter((event) => event === 'draw').length;

  loop.run(0);
  loop.run(1);
  expect(draws()).toBe(2);
  expect(covered.frameSize).toEqual([4, 8]);
  expect(covered.crop).toEqual({ origin: [0, 0.25], extent: [1, 0.5] });
  expect(contained.crop).toEqual({ origin: [-0.5, 0], extent: [2, 1] });

  frame.serial = 2;
  loop.run(2);
  expect([draws(), covered.frames, contained.frames]).toEqual([4, 2, 2]);
});
