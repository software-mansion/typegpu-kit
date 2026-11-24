/**
 * Finds where a scene first differs between WebGPU and the WebGL device: records every texture each
 * device creates, runs the same frames on both, then reads every readable texture back and compares
 * them in creation order.
 */

import { frames } from '@typegpu/kit';
import { offscreenSurface } from '@typegpu/kit/testing';
import type { TgpuRoot } from 'typegpu';
import { loading } from '../../../../apps/typegpu-kit-docs/src/examples/files.ts';
import { begin, filesOf } from '../../../typegpu-kit-examples/src/hosts.ts';
import { exampleOf } from '../../../typegpu-kit-examples/src/index.ts';
import { compare, readTexture, values } from './compare.ts';

interface Made {
  readonly descriptor: GPUTextureDescriptor;
  readonly texture: GPUTexture;
  /** The kit or example source line that created it. */
  readonly origin: string;
}

/** The first kit or example frame of the current stack, as `file:line`. */
function origin(): string {
  const lines = (new Error().stack ?? '').split('\n');
  const line = lines.find(
    (each) =>
      (each.includes('/typegpu-kit/src/') || each.includes('/typegpu-kit-examples/src/')) &&
      !each.includes('/targets.ts'),
  );
  const match = line && /\/(typegpu-kit(?:-examples)?\/src\/[^?:]+)[^:]*:(\d+)/.exec(line);
  return match ? `${match[1]}:${match[2]}` : '';
}

const READABLE = new Set<GPUTextureFormat>([
  'r8unorm',
  'rg8unorm',
  'rgba8unorm',
  'rgba8unorm-srgb',
  'r16float',
  'rg16float',
  'rgba16float',
  'r32float',
  'rg32float',
  'rgba32float',
  'r32uint',
  'rgba32uint',
  'rg11b10ufloat',
  'rgb10a2unorm',
  'depth32float',
  'depth16unorm',
]);

/** Records the textures `device` creates from now on, with COPY_SRC added so most read back. */
function record(device: GPUDevice): Made[] {
  const made: Made[] = [];
  const create = device.createTexture.bind(device);
  (device as { createTexture: GPUDevice['createTexture'] }).createTexture = (descriptor) => {
    // WebGPU rejects copies from multisampled textures, so leave those as they are
    const single = (descriptor.sampleCount ?? 1) === 1;
    const usage = single ? descriptor.usage | GPUTextureUsage.COPY_SRC : descriptor.usage;
    const widened = { ...descriptor, usage };
    const texture = create(widened);
    made.push({ descriptor, texture, origin: origin() });
    return texture;
  };
  return made;
}

interface MadeBuffer {
  readonly descriptor: GPUBufferDescriptor;
  readonly buffer: GPUBuffer;
}

/** Records the buffers `device` creates from now on, with COPY_SRC where WebGPU allows it. */
function recordBuffers(device: GPUDevice): MadeBuffer[] {
  const made: MadeBuffer[] = [];
  const create = device.createBuffer.bind(device);
  (device as { createBuffer: GPUDevice['createBuffer'] }).createBuffer = (descriptor) => {
    const mappable = descriptor.usage & (GPUBufferUsage.MAP_READ | GPUBufferUsage.MAP_WRITE);
    const usage = mappable ? descriptor.usage : descriptor.usage | GPUBufferUsage.COPY_SRC;
    const buffer = create({ ...descriptor, usage });
    if (!mappable) made.push({ descriptor, buffer });
    return buffer;
  };
  return made;
}

async function readBuffer(device: GPUDevice, buffer: GPUBuffer): Promise<Uint8Array> {
  const size = Math.ceil(buffer.size / 4) * 4;
  const staging = device.createBuffer({
    size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const bytes = new Uint8Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return bytes;
}

export interface BufferDifference {
  readonly index: number;
  readonly label: string;
  readonly size: number;
  readonly usage: number;
  /** Share of 32-bit words that differ, compared as floats within 1e-3 relative. */
  readonly mismatched: number;
  /** The first differing word: index, WebGPU's and WebGL's value as u32 and f32. */
  readonly first: string;
}

function compareWords(a: Uint8Array, b: Uint8Array): { mismatched: number; first: string } {
  const ua = new Uint32Array(a.buffer, 0, a.byteLength / 4);
  const ub = new Uint32Array(b.buffer, 0, b.byteLength / 4);
  const fa = new Float32Array(a.buffer, 0, a.byteLength / 4);
  const fb = new Float32Array(b.buffer, 0, b.byteLength / 4);
  let far = 0;
  let first = '';
  for (let i = 0; i < ua.length; i++) {
    if (ua[i] === ub[i]) continue;
    const close = Math.abs(fa[i]! - fb[i]!) <= 1e-3 * Math.max(1, Math.abs(fa[i]!));
    if (close) continue;
    far++;
    if (!first) first = `word ${i}: gpu ${ua[i]} (${fa[i]}) gl ${ub[i]} (${fb[i]})`;
  }
  return { mismatched: Number((far / Math.max(1, ua.length)).toFixed(4)), first };
}

async function run(root: TgpuRoot, key: string, count: number, size: [number, number]) {
  const made = record(root.device);
  const buffers = recordBuffers(root.device);
  const surface = offscreenSurface(root, { width: size[0], height: size[1], format: 'rgba8unorm' });
  const { createScene } = await exampleOf(key).scene();
  const files = await filesOf(key, loading());
  const scene = await begin(await createScene(root, surface, files));
  await scene.ready;
  const loop = frames.create(root, scene.frame);
  for (let i = 0; i < count; i++) {
    loop.run(i * 0.04);
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  await root.device.queue.onSubmittedWorkDone();
  return { made, buffers, surface, scene, loop };
}

export interface TextureDifference {
  readonly index: number;
  readonly label: string;
  readonly format: GPUTextureFormat;
  readonly size: string;
  readonly layer: number;
  readonly level: number;
  /** Share of values more than 4/255 (or 1 % of the range) apart. */
  readonly mismatched: number;
  readonly maxDifference: number;
  /** Mean of each side's values, and their first values, for a glance at which side is off. */
  readonly expected: string;
  readonly actual: string;
}

function summary(readback: Parameters<typeof values>[0]): string {
  const all = values(readback);
  let sum = 0;
  for (const value of all) sum += value;
  const head = Array.from(all.subarray(0, 4), (v) => Number(v.toFixed(3)));
  return `mean ${(sum / all.length).toFixed(3)} first ${head.join(',')}`;
}

/** Runs `key` on both roots for `count` frames and compares every texture both made alike. */
export async function diffScene(
  gpu: TgpuRoot,
  gl: TgpuRoot,
  key: string,
  options: { frames?: number; size?: [number, number]; layers?: number; levels?: number } = {},
): Promise<{ textures: TextureDifference[]; buffers: BufferDifference[] }> {
  const count = options.frames ?? 3;
  const size = options.size ?? [320, 240];
  const expected = await run(gpu, key, count, size);
  const actual = await run(gl, key, count, size);
  const differences: TextureDifference[] = [];
  const pairs = Math.min(expected.made.length, actual.made.length);
  for (let index = 0; index < pairs; index++) {
    const a = expected.made[index]!;
    const b = actual.made[index]!;
    const { texture } = a;
    if (
      texture.format !== b.texture.format ||
      texture.width !== b.texture.width ||
      texture.height !== b.texture.height ||
      texture.sampleCount > 1 ||
      texture.dimension !== '2d' ||
      !READABLE.has(texture.format)
    ) {
      continue;
    }
    const layers = Math.min(texture.depthOrArrayLayers, options.layers ?? 6);
    const levels = Math.min(texture.mipLevelCount, options.levels ?? 1);
    for (let at = 0; at < layers * levels; at++) {
      const layer = at % layers;
      const level = Math.floor(at / layers);
      try {
        const left = await readTexture(gpu.device, a.texture, { layer, level });
        const right = await readTexture(gl.device, b.texture, { layer, level });
        const tolerance = texture.format.includes('float') ? 0.01 : 4 / 255;
        const result = compare(left, right, tolerance);
        differences.push({
          index,
          label: `${a.descriptor.label ?? ''} ${a.origin}`,
          format: texture.format,
          size: `${texture.width}x${texture.height}x${texture.depthOrArrayLayers}`,
          layer,
          level,
          mismatched: Number(result.mismatched.toFixed(4)),
          maxDifference: Number(result.maxDifference.toFixed(4)),
          expected: summary(left),
          actual: summary(right),
        });
      } catch (error) {
        differences.push({
          index,
          label: `${a.descriptor.label ?? ''} (${(error as Error).message.slice(0, 80)})`,
          format: texture.format,
          size: `${texture.width}x${texture.height}`,
          layer,
          level,
          mismatched: -1,
          maxDifference: -1,
          expected: '',
          actual: '',
        });
      }
    }
  }
  const buffers: BufferDifference[] = [];
  const bufferPairs = Math.min(expected.buffers.length, actual.buffers.length);
  for (let index = 0; index < bufferPairs; index++) {
    const a = expected.buffers[index]!;
    const b = actual.buffers[index]!;
    if (a.buffer.size !== b.buffer.size) continue;
    const words = compareWords(
      await readBuffer(gpu.device, a.buffer),
      await readBuffer(gl.device, b.buffer),
    );
    if (words.mismatched === 0) continue;
    buffers.push({
      index,
      label: a.descriptor.label ?? '',
      size: a.buffer.size,
      usage: a.descriptor.usage,
      ...words,
    });
  }
  for (const side of [expected, actual]) {
    side.loop.destroy();
    side.scene.destroy?.();
    side.surface.destroy();
  }
  return { textures: differences, buffers };
}
