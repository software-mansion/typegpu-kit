/// <reference types="node" />
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { frames } from '@typegpu/kit';
import { offscreenSurface } from '@typegpu/kit/testing';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { expect, it, vi } from 'vitest';
import { cardAt, FINISHES } from 'typegpu-kit-pull/card.ts';
import { COLORS } from 'typegpu-kit-pull/dither.ts';
import { pulledBox } from 'typegpu-kit-pull/box.ts';
import { LAUNCH } from 'typegpu-kit-pull/timeline.ts';
import { pulled, resting } from '../src/chest/box.ts';
import { create as face } from '../src/chest/face.ts';

const fonts = path.resolve(import.meta.dirname, '..');
const fontAt = async (name: 'inter' | 'mono') => {
  const at =
    name === 'inter'
      ? path.resolve(fonts, '../../packages/typegpu-kit/fonts/inter')
      : path.resolve(fonts, 'assets/fonts/mono');
  const [json, atlas] = await Promise.all([readFile(`${at}.json`, 'utf8'), readFile(`${at}.rgba`)]);
  return { json: JSON.parse(json) as unknown, atlas: new Uint8Array(atlas).buffer };
};

const surfaceOn = (root: Awaited<ReturnType<typeof gpuRoot>>, width: number, height: number) =>
  offscreenSurface(root, { width, height, pixelRatio: 3, format: 'bgra8unorm' });

it('rests, pulls and turns the card over on a real device', async () => {
  const root = await gpuRoot();
  const pressed = { value: false };
  const over = { value: 0 };
  const rest = resting(root, surfaceOn(root, 124, 93), { pressed, over });
  const stage = { width: 340, height: 476, slot: [195, 400] as const, size: 46 };
  const exit = pulledBox(LAUNCH.box + 500, 0.3, [195, 520], 46, 0, COLORS.accent[0], [1, 1, 1]);
  const box = pulled(root, surfaceOn(root, 130, 280), { box: { value: exit.input } });
  const pose = { value: cardAt('box', LAUNCH.box + 100, exit, stage, 1, 0.5) };
  // dawn.node copies no images, so a stand-in bitmap checks that the card's texture takes one
  const copy = vi
    .spyOn(root.device.queue, 'copyExternalImageToTexture')
    .mockImplementation(() => undefined);
  const card = face(root, surfaceOn(root, 390, 844), {
    poster: { width: 160, height: 120, close() {} } as ImageBitmap,
    fonts: { display: await fontAt('inter'), mono: await fontAt('mono') },
    size: [340, 476],
    entry: {
      key: 'jelly-world',
      title: 'A title long enough to wrap onto a second line and be cut',
      line: 'A line long enough to run past three lines of the card, so the caption cuts it with an ellipsis where it stops.',
      category: 'Worlds',
      number: 7,
      lines: 1234,
      rarity: 'legendary',
      tier: 'Legendary',
      tint: [0.97, 0.7, 0.31],
      finish: FINISHES.legendary,
    },
    pose,
  });

  const loop = frames.create(root, (frame) => {
    rest.frame(frame);
    box.frame(frame);
    card.frame(frame);
  });
  loop.run(0);
  pressed.value = true;
  // The card's back, then its face mid-flip, then landed
  pose.value = { ...pose.value, ry: 180 };
  loop.run(0.2);
  pose.value = { ...pose.value, ry: -30, rx: -12, rz: 2, rim: undefined };
  loop.run(0.4);
  pose.value = { ...pose.value, x: 195, y: 400, z: 0, scale: 1, rx: 0, ry: 0, rz: 0 };
  loop.run(0.6);
  expect(over.value).toBeGreaterThan(0.9);
  const [, destination] = copy.mock.calls[0];
  expect(destination.texture.usage & GPUTextureUsage.RENDER_ATTACHMENT).toBeTruthy();

  rest.destroy?.();
  box.destroy?.();
  card.destroy?.();
});
