import { packedEqual } from './encoded-fields.ts';
import { frames } from '@typegpu/kit';
import { offscreenSurface } from '@typegpu/kit/testing';
import type { TgpuRoot } from 'typegpu';
import type { ExampleScene } from '../../../typegpu-kit-examples/src/contract.ts';
import { exampleOf } from '../../../typegpu-kit-examples/src/index.ts';
import { begin, filesOf } from '../../../typegpu-kit-examples/src/hosts.ts';
import { fromDisk, settle } from '../../../typegpu-kit-examples/tests/utils/examples.ts';
import type { Layout } from '../../src/cpu/memory.ts';
import { decodeHalf } from '../../src/cpu/textures.ts';
import { observeScene, type Resource } from './scene-observer.ts';

export const smallScenes: Record<string, object> = {
  forest: { trees: 40 },
  stampede: { count: 64 },
  skeletons: { count: 8 },
  'quiver-trees': { count: 32 },
  'star-map': { count: 256 },
  'slime-mould': { grid: 32, count: 256 },
};

export interface CapturedResource extends Omit<Resource, 'handle' | 'reasons' | 'order'> {
  reasons: string[];
  order: string[];
  levels: { width: number; height: number; depth: number; bytes: ArrayBuffer }[];
}
export interface SceneCapture {
  resources: CapturedResource[];
  skipped: { key: string; reasons: string[] }[];
  frames: number;
}

export function seeded(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), state | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export async function readBuffer(device: GPUDevice, source: GPUBuffer): Promise<ArrayBuffer> {
  const staging = device.createBuffer({
    size: source.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, staging, 0, source.size);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    return staging.getMappedRange().slice(0);
  } finally {
    staging.unmap();
    staging.destroy();
  }
}

export async function readTexture(
  device: GPUDevice,
  source: GPUTexture,
): Promise<CapturedResource['levels']> {
  const channels =
    source.format === 'rgb10a2unorm'
      ? 1
      : /^(?:rgba|bgra)/.test(source.format)
        ? 4
        : source.format.startsWith('rg')
          ? 2
          : 1;
  const bits = source.format === 'rgb10a2unorm' ? 32 : Number(/(8|16|32)/.exec(source.format)?.[1]);
  if (!bits || /depth|stencil|rg11/.test(source.format))
    throw new Error(`texture format ${source.format} has no comparison decoder`);
  const levels: CapturedResource['levels'] = [];
  for (let level = 0; level < source.mipLevelCount; level++) {
    const width = Math.max(1, source.width >> level),
      height = Math.max(1, source.height >> level);
    const depth =
      source.dimension === '3d'
        ? Math.max(1, source.depthOrArrayLayers >> level)
        : source.depthOrArrayLayers;
    const row = (width * channels * bits) / 8,
      bytesPerRow = Math.ceil(row / 256) * 256;
    const staging = device.createBuffer({
      size: bytesPerRow * height * depth,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = device.createCommandEncoder();
      encoder.copyTextureToBuffer(
        { texture: source, mipLevel: level },
        { buffer: staging, bytesPerRow, rowsPerImage: height },
        [width, height, depth],
      );
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const padded = new Uint8Array(staging.getMappedRange()),
        bytes = new Uint8Array(row * height * depth);
      for (let y = 0; y < height * depth; y++)
        bytes.set(padded.subarray(y * bytesPerRow, y * bytesPerRow + row), y * row);
      levels.push({ width, height, depth, bytes: bytes.buffer });
    } finally {
      staging.unmap();
      staging.destroy();
    }
  }
  return levels;
}

export async function captureScene(
  root: TgpuRoot,
  observer: ReturnType<typeof observeScene>,
  key: string,
  read = true,
): Promise<SceneCapture> {
  const random = Math.random;
  Math.random = seeded(1);
  const surface = offscreenSurface(root, { width: 64, height: 36, format: 'rgba8unorm' });
  let scene: ExampleScene | undefined;
  let loop: ReturnType<typeof frames.create> | undefined;
  try {
    const example = exampleOf(key);
    scene = await begin(
      await (
        await example.scene()
      ).createScene(root, surface, {
        ...(await filesOf(key, fromDisk)),
        ...example.fallback,
        ...smallScenes[key],
        random: seeded(1),
      }),
    );
    const state: { ready: boolean; failure?: unknown } = { ready: scene.ready === undefined };
    void scene.ready?.then(
      () => {
        state.ready = true;
      },
      (error: unknown) => {
        state.failure = error;
      },
    );
    // Async compilation and asset loading do not advance simulation time.
    for (let i = 0; i < 100 && !state.ready && !state.failure; i++) await settle();
    if (state.failure) throw state.failure;
    loop = frames.create(root, scene.frame);
    for (let frame = 0; frame < 2; frame++) {
      loop.run(frame * 0.04);
      await root.device.queue.onSubmittedWorkDone();
      await settle();
    }
    if (state.failure) throw state.failure;
    if (!state.ready) throw new Error(`${key}: scene did not become ready before capture`);
    if (!read) return { resources: [], skipped: [], frames: 2 };
    const resources: CapturedResource[] = [],
      skipped: SceneCapture['skipped'] = [];
    for (const resource of observer.resources.slice()) {
      const reasons = [...resource.reasons];
      if (key === 'ocean' && resource.kind === 'buffer' && resource.label === 'lattice')
        reasons.push(
          'backend-specific FFT scratch: WebGPU retains row transforms, fallback retains column transforms; final wave maps are compared',
        );
      if (!resource.written)
        reasons.push('no compute writes in these two frames or initialization');
      if (resource.destroyed) reasons.push('destroyed before capture');
      if (reasons.length) {
        skipped.push({ key: resource.key, reasons });
        continue;
      }
      let levels: CapturedResource['levels'];
      if (resource.kind === 'buffer') {
        levels = [
          {
            width: 0,
            height: 0,
            depth: 0,
            bytes: await readBuffer(root.device, resource.handle as GPUBuffer),
          },
        ];
      } else {
        const cpu = observer.cpuTextures.get(resource.handle);
        if (cpu)
          levels = cpu.levels.map((level) => ({
            width: level.width,
            height: level.height,
            depth: level.depthOrArrayLayers,
            bytes: new Uint8Array(
              level.data.buffer,
              level.data.byteOffset,
              level.data.byteLength,
            ).slice().buffer,
          }));
        else {
          try {
            levels = await readTexture(root.device, resource.handle as GPUTexture);
          } catch (error) {
            skipped.push({ key: resource.key, reasons: [String(error)] });
            continue;
          }
        }
      }
      const { handle: _handle, ...metadata } = resource;
      resources.push({ ...metadata, reasons: [], order: [...resource.order], levels });
    }
    return { resources, skipped, frames: 2 };
  } finally {
    Math.random = random;
    loop?.destroy();
    scene?.destroy?.();
    surface.destroy();
  }
}

export function fieldAt(
  layout: Layout,
  offset: number,
  prefix = '',
): { path: string; type: string; vectorOffset?: number; vectorSize?: number } | undefined {
  if (offset < 0) return undefined;
  if (layout.fields) {
    for (const [name, field] of Object.entries(layout.fields))
      if (offset >= field.offset && offset < field.offset + field.layout.size)
        return fieldAt(field.layout, offset - field.offset, `${prefix}.${name}`);
    return undefined;
  }
  if (layout.element && layout.stride) {
    const index = Math.floor(offset / layout.stride);
    if (layout.count && index >= layout.count) return undefined;
    return fieldAt(layout.element, offset % layout.stride, `${prefix}[${index}]`);
  }
  return offset < layout.size ? { path: prefix || '$', type: layout.type } : undefined;
}

function vectorRange(layout: Layout, offset: number, base = 0): [number, number] {
  if (layout.type.startsWith('vec')) return [base, layout.size];
  if (layout.fields)
    for (const field of Object.values(layout.fields)) {
      if (offset >= field.offset && offset < field.offset + field.layout.size)
        return vectorRange(field.layout, offset - field.offset, base + field.offset);
    }
  if (layout.element && layout.stride) {
    const shift = Math.floor(offset / layout.stride) * layout.stride;
    return vectorRange(layout.element, offset - shift, base + shift);
  }
  return [base, 4];
}

export interface ExampleDifference {
  key: string;
  words: number;
  first: string[];
  order: string[];
}
export function compareCaptures(expected: SceneCapture, actual: SceneCapture) {
  const differences: ExampleDifference[] = [],
    skipped = [
      ...expected.skipped.map((item) => ({ ...item, side: 'dawn' })),
      ...actual.skipped.map((item) => ({ ...item, side: 'CPU' })),
    ];
  let compared = 0;
  const matched = new Set<string>();
  for (const cpu of actual.resources) {
    const gpu = expected.resources.find((resource) => resource.key === cpu.key);
    if (!gpu) {
      skipped.push({
        key: cpu.key,
        side: 'CPU',
        reasons: [
          'no matching readable resource on WebGPU; backend-specific allocation or excluded counterpart',
        ],
      });
      continue;
    }
    matched.add(cpu.key);
    const first: string[] = [];
    let words = 0;
    let readLevel = false;
    const order = [...new Set([...cpu.order, ...gpu.order])];
    if (cpu.levels.length !== gpu.levels.length) {
      words++;
      first.push('mip level counts differ');
    }
    for (let level = 0; level < Math.min(cpu.levels.length, gpu.levels.length); level++) {
      const a = gpu.levels[level]!,
        b = cpu.levels[level]!;
      if (a.bytes.byteLength !== b.bytes.byteLength) {
        first.push(
          `level ${level}: byte lengths CPU=${b.bytes.byteLength}, dawn=${a.bytes.byteLength}`,
        );
        words++;
        continue;
      }
      if (cpu.kind === 'buffer' && !cpu.layout) {
        skipped.push({
          key: cpu.key,
          side: 'CPU',
          reasons: ['no typed binding layout; refusing to reinterpret integer words as floats'],
        });
        continue;
      }
      const bits =
        cpu.kind === 'buffer' || cpu.format === 'rgb10a2unorm'
          ? 32
          : Number(/(8|16|32)/.exec(cpu.format!)?.[1]);
      const ua =
        bits === 8
          ? new Uint8Array(a.bytes)
          : bits === 16
            ? new Uint16Array(a.bytes)
            : new Uint32Array(a.bytes);
      const ub =
        bits === 8
          ? new Uint8Array(b.bytes)
          : bits === 16
            ? new Uint16Array(b.bytes)
            : new Uint32Array(b.bytes);
      const fa = bits === 32 ? new Float32Array(a.bytes) : undefined,
        fb = bits === 32 ? new Float32Array(b.bytes) : undefined;
      readLevel = true;
      for (let word = 0; word < ua.length; word++) {
        const field =
          cpu.kind === 'buffer'
            ? fieldAt(cpu.layout!, word * 4 - (cpu.layoutOffset ?? 0))
            : {
                type: cpu.format!.includes('float') ? 'f32' : 'u32',
                path: `mip[${level}].component[${word}]`,
              };
        if (!field || ua[word] === ub[word]) continue;
        if (cpu.format?.includes('norm')) {
          if (bits === 8 && Math.abs(ua[word]! - ub[word]!) <= 1) continue;
          if (
            cpu.format === 'rgb10a2unorm' &&
            [0, 10, 20, 30].every(
              (shift) =>
                Math.abs(
                  ((ua[word]! >>> shift) & (shift === 30 ? 3 : 1023)) -
                    ((ub[word]! >>> shift) & (shift === 30 ? 3 : 1023)),
                ) <= 1,
            )
          )
            continue;
        }
        let av = ua[word]!,
          bv = ub[word]!;
        const encoding = (gpu.encoded ?? cpu.encoded)?.find(
          (item) => word % item.components === item.component,
        )?.encoding;
        if (encoding && encoding !== 'ordered-f32' && packedEqual(ua[word]!, ub[word]!, encoding))
          continue;
        if (field.type === 'f32' || encoding === 'ordered-f32') {
          av = bits === 16 ? decodeHalf(av) : fa![word]!;
          bv = bits === 16 ? decodeHalf(bv) : fb![word]!;
          let leftBits = ua[word]!,
            rightBits = ub[word]!;
          if (encoding === 'ordered-f32') {
            leftBits = (leftBits ^ (leftBits >= 0x80000000 ? 0x80000000 : 0xffffffff)) >>> 0;
            rightBits = (rightBits ^ (rightBits >= 0x80000000 ? 0x80000000 : 0xffffffff)) >>> 0;
            av = new Float32Array(Uint32Array.of(leftBits).buffer)[0]!;
            bv = new Float32Array(Uint32Array.of(rightBits).buffer)[0]!;
          }
          const ulps = Math.abs(leftBits - rightBits);
          if (av === bv || (Number.isNaN(av) && Number.isNaN(bv))) continue;
          let magnitude = Math.max(Math.abs(av), Math.abs(bv));
          if (cpu.layout && field.type === 'f32' && bits === 32) {
            const [at, size] = vectorRange(
              cpu.layout,
              word * 4 - (cpu.layoutOffset ?? 0),
              cpu.layoutOffset ?? 0,
            );
            for (let component = at / 4; component < (at + size) / 4; component++)
              magnitude = Math.max(magnitude, Math.abs(fa![component]!), Math.abs(fb![component]!));
          }
          if (
            Number.isFinite(av) &&
            Number.isFinite(bv) &&
            ((Math.sign(av) === Math.sign(bv) && ulps <= (bits === 16 ? 1 : 4)) ||
              Math.abs(av - bv) <= Math.max(1e-6, magnitude * 1e-5))
          )
            continue;
        }
        words++;
        if (first.length < 20)
          first.push(
            `word ${word} ${field.path}: CPU=${bv} (0x${ub[word]!.toString(16)}), dawn=${av} (0x${ua[word]!.toString(16)})`,
          );
      }
    }
    if (readLevel || words) compared++;
    if (words) differences.push({ key: cpu.key, words, first, order });
  }
  for (const gpu of expected.resources)
    if (!matched.has(gpu.key))
      skipped.push({
        key: gpu.key,
        side: 'dawn',
        reasons: [
          'no matching readable CPU resource; backend-specific allocation or excluded counterpart',
        ],
      });
  const allocationOrder = differences.filter((difference) => difference.order.length > 0);
  const review = differences.filter((difference) => difference.order.length === 0);
  const decoded = expected.resources.flatMap((resource) =>
    resource.encoded?.length
      ? [
          {
            key: resource.key,
            fields: resource.encoded,
            reason:
              'typed integer storage encodes float values; octahedral seams have multiple equivalent packed representations',
          },
        ]
      : [],
  );
  return {
    decoded,
    compared,
    differences,
    skipped,
    summary: {
      passesWithinPolicy: compared - differences.length,
      differsReview: review.length,
      excludedAllocationOrder: allocationOrder.length,
      excludedRenderDependent: skipped.filter((item) =>
        item.reasons.some((reason) => reason.includes('render pass')),
      ).length,
      excludedOther: skipped.filter(
        (item) => !item.reasons.some((reason) => reason.includes('render pass')),
      ).length,
    },
  };
}
