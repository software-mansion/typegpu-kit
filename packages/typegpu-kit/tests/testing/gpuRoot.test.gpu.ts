import { expect, it, onTestFinished } from 'vitest';
import { gpuRoot } from '../../src/testing/dawn.ts';

const invalid = () => ({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.STORAGE });

// Inverted: passes only when the error left at the end fails the test.
it.fails('fails a test that leaves an uncaptured error', async () => {
  const root = await gpuRoot();
  root.device.createBuffer(invalid());
  root.destroy();
});

it('passes a test that takes the errors it expects', async () => {
  const root = await gpuRoot();
  root.device.createBuffer(invalid());

  expect(await root.takeErrors()).toEqual([
    expect.stringContaining('Buffer usages (BufferUsage::(MapRead|Storage)) is invalid.'),
  ]);
  expect(await root.takeErrors()).toEqual([]);
  root.destroy();
});

it('destroys its device when the test ends', async () => {
  let device: GPUDevice | undefined;
  // Registered before gpuRoot's, so it runs after it: vitest runs these in reverse.
  onTestFinished(async () => {
    expect((await device!.lost).reason).toBe('destroyed');
  });

  device = (await gpuRoot()).device;
});

it('throws when the adapter lacks a requested feature', async () => {
  const bufferUsage = Object.getOwnPropertyDescriptor(globalThis, 'GPUBufferUsage');
  const gpu = Object.getOwnPropertyDescriptor(navigator, 'gpu');
  onTestFinished(() => {
    expect(Object.getOwnPropertyDescriptor(globalThis, 'GPUBufferUsage')).toEqual(bufferUsage);
    expect(Object.getOwnPropertyDescriptor(navigator, 'gpu')).toEqual(gpu);
  });
  await expect(gpuRoot({ features: ['no-such-feature' as GPUFeatureName] })).rejects.toThrow(
    'The adapter lacks no-such-feature.',
  );
});

it('leaves out an optional feature the adapter lacks', async () => {
  const root = await gpuRoot({ optionalFeatures: ['no-such-feature' as GPUFeatureName] });
  expect(root.device.features.has('no-such-feature')).toBe(false);
  root.destroy();
});
