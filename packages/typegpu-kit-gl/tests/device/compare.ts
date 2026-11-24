/**
 * Draws the same thing on WebGPU and on the WebGL device and compares the pixels. Runs in the
 * browser tests, where both APIs exist.
 *
 * ```ts
 * const { gpu, gl } = await roots();
 * const draw = (root: TgpuRoot) => { ...; return texture; };   // a GPUTexture or TgpuTexture
 * const result = await compareRoots(gpu, gl, draw);
 * expect(result.mismatched).toBeLessThan(0.01);
 * ```
 */

import { tgpu, type TgpuRoot } from 'typegpu';
import { createDevice, type DeviceOptions, type GLDevice } from '../../src/device/index.ts';
import { KitGenerator } from '../../src/generator.ts';

/** A WebGPU root and a WebGL root. Destroy both when done. */
export async function roots(options: DeviceOptions = {}): Promise<{ gpu: TgpuRoot; gl: TgpuRoot }> {
  const gpu = await tgpu.init();
  return { gpu, gl: glRoot(options) };
}

/** A root on the WebGL device with the fallback's shader generator. */
export function glRoot(options: DeviceOptions = {}): TgpuRoot {
  const device = createDevice(options);
  return tgpu.initFromDevice({ device, unstable_shaderGeneratorClass: KitGenerator });
}

export function glDevice(options: DeviceOptions = {}): GLDevice {
  return createDevice(options);
}

/** Bytes per texel of the formats `readTexture` reads. */
const TEXEL_BYTES: Partial<Record<GPUTextureFormat, number>> = {
  r8unorm: 1,
  rg8unorm: 2,
  rgba8unorm: 4,
  'rgba8unorm-srgb': 4,
  bgra8unorm: 4,
  r16float: 2,
  rg16float: 4,
  rgba16float: 8,
  r32float: 4,
  rg32float: 8,
  rgba32float: 16,
  r32uint: 4,
  rgba32uint: 16,
  rg11b10ufloat: 4,
  rgb10a2unorm: 4,
  depth32float: 4,
  depth16unorm: 2,
};

export interface Readback {
  readonly width: number;
  readonly height: number;
  readonly format: GPUTextureFormat;
  /** Texels in the format's own layout, rows top to bottom, no padding. */
  readonly bytes: Uint8Array;
}

/**
 * Reads one level and layer of a texture with `copyTextureToBuffer`, on either device. Accepts a
 * raw `GPUTexture` or anything with `unwrap` that the root can unwrap.
 */
export async function readTexture(
  device: GPUDevice,
  texture: GPUTexture,
  options: { level?: number; layer?: number; aspect?: GPUTextureAspect } = {},
): Promise<Readback> {
  const level = options.level ?? 0;
  const width = Math.max(1, texture.width >> level);
  const height = Math.max(1, texture.height >> level);
  const texel = TEXEL_BYTES[texture.format];
  if (!texel) throw new Error(`readTexture cannot read '${texture.format}'`);
  const tight = width * texel;
  const bytesPerRow = Math.ceil(tight / 256) * 256;
  const buffer = device.createBuffer({
    size: bytesPerRow * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer(
    {
      texture,
      mipLevel: level,
      origin: [0, 0, options.layer ?? 0],
      ...(options.aspect && { aspect: options.aspect }),
    },
    { buffer, bytesPerRow },
    [width, height, 1],
  );
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const padded = new Uint8Array(buffer.getMappedRange());
  const bytes = new Uint8Array(tight * height);
  for (let row = 0; row < height; row++) {
    bytes.set(padded.subarray(row * bytesPerRow, row * bytesPerRow + tight), row * tight);
  }
  buffer.unmap();
  buffer.destroy();
  return { width, height, format: texture.format, bytes };
}

function halfToFloat(half: number): number {
  const sign = half & 0x8000 ? -1 : 1;
  const exponent = (half >>> 10) & 0x1f;
  const mantissa = half & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 0x1f) return mantissa ? NaN : sign * Infinity;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

/** The readback's values as numbers, normalized formats from 0 to 1. */
export function values(readback: Readback): Float64Array {
  const { bytes, format } = readback;
  const { buffer, byteOffset, byteLength } = bytes;
  if (format.includes('8unorm')) return Float64Array.from(bytes, (v) => v / 255);
  if (format.endsWith('16float')) {
    return Float64Array.from(new Uint16Array(buffer, byteOffset, byteLength / 2), halfToFloat);
  }
  if (format.endsWith('32float'))
    return Float64Array.from(new Float32Array(buffer, byteOffset, byteLength / 4));
  if (format.endsWith('32uint'))
    return Float64Array.from(new Uint32Array(buffer, byteOffset, byteLength / 4));
  if (format === 'depth16unorm') {
    return Float64Array.from(new Uint16Array(buffer, byteOffset, byteLength / 2), (v) => v / 65535);
  }
  return Float64Array.from(bytes);
}

export interface Comparison {
  /** The largest difference of one value. */
  readonly maxDifference: number;
  /** The share of values that differ by more than the tolerance. */
  readonly mismatched: number;
  /** The first value that differs by more than the tolerance, as `x, y, component`. */
  readonly first:
    | { x: number; y: number; component: number; expected: number; actual: number }
    | undefined;
}

export function compare(expected: Readback, actual: Readback, tolerance = 2 / 255): Comparison {
  if (expected.width !== actual.width || expected.height !== actual.height) {
    throw new Error(
      `Sizes differ: ${expected.width}x${expected.height} and ${actual.width}x${actual.height}`,
    );
  }
  const a = values(expected);
  const b = values(actual);
  const perTexel = a.length / (expected.width * expected.height);
  let maxDifference = 0;
  let far = 0;
  let first: Comparison['first'];
  for (let i = 0; i < a.length; i++) {
    const difference = Math.abs(a[i]! - b[i]!);
    if (!(difference <= tolerance) && !(Number.isNaN(a[i]) && Number.isNaN(b[i]))) {
      far++;
      if (!first) {
        const texel = Math.floor(i / perTexel);
        first = {
          x: texel % expected.width,
          y: Math.floor(texel / expected.width),
          component: i % perTexel,
          expected: a[i]!,
          actual: b[i]!,
        };
      }
    }
    if (difference > maxDifference) maxDifference = difference;
  }
  return { maxDifference, mismatched: far / a.length, first };
}

type Drawn = GPUTexture | { readonly [key: string]: unknown };

function rawTexture(root: TgpuRoot, drawn: Drawn): GPUTexture {
  if ('createView' in drawn && 'format' in drawn && 'width' in drawn) return drawn as GPUTexture;
  return root.unwrap(drawn as never) as unknown as GPUTexture;
}

/**
 * Runs `draw` on both roots, reads back the texture each returns and compares them. `draw` returns
 * a raw `GPUTexture` or a TypeGPU texture; it may submit work itself.
 */
export async function compareRoots(
  gpu: TgpuRoot,
  gl: TgpuRoot,
  draw: (root: TgpuRoot) => Drawn | Promise<Drawn>,
  options: { tolerance?: number; level?: number; layer?: number } = {},
): Promise<Comparison & { expected: Readback; actual: Readback }> {
  const expected = await readTexture(gpu.device, rawTexture(gpu, await draw(gpu)), options);
  const actual = await readTexture(gl.device, rawTexture(gl, await draw(gl)), options);
  return { ...compare(expected, actual, options.tolerance), expected, actual };
}

/** The texel at `x, y` of an 8-bit RGBA readback, as bytes. */
export function pixel(readback: Readback, x: number, y: number): number[] {
  const texel = readback.bytes.length / (readback.width * readback.height);
  const at = (y * readback.width + x) * texel;
  return [...readback.bytes.subarray(at, at + texel)];
}

/** Collects `uncapturederror` messages a device reports while `run` runs. */
export async function errorsOf(device: GPUDevice, run: () => unknown): Promise<string[]> {
  const messages: string[] = [];
  const listener = (event: Event) => {
    messages.push((event as GPUUncapturedErrorEvent).error.message);
  };
  device.addEventListener('uncapturederror', listener);
  try {
    await run();
  } finally {
    device.removeEventListener('uncapturederror', listener);
  }
  return messages;
}
