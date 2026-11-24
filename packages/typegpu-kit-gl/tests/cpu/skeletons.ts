import { frames } from '@typegpu/kit';
import { offscreenSurface } from '@typegpu/kit/testing';
import type { TgpuRoot } from 'typegpu';
import { begin, filesOf } from '../../../typegpu-kit-examples/src/hosts.ts';
import { createScene } from '../../../typegpu-kit-examples/src/skeletons/scene.ts';
import { fromDisk } from '../../../typegpu-kit-examples/tests/utils/examples.ts';
import { Pose } from '../../../typegpu-kit/src/render/skinning.ts';
import { layoutOf, type Layout } from '../../src/cpu/memory.ts';

export interface CapturedBuffer {
  label: string;
  bytes: ArrayBuffer;
}

export async function skeletons(root: TgpuRoot): Promise<CapturedBuffer[]> {
  const buffers: GPUBuffer[] = [];
  const device = root.device;
  const createBuffer = device.createBuffer.bind(device);
  device.createBuffer = (descriptor) => {
    const buffer = createBuffer(descriptor);
    buffers.push(buffer);
    return buffer;
  };
  const random = Math.random;
  let seed = 12345;
  Math.random = () => {
    seed = (seed + 0x6d2b79f5) >>> 0;
    let value = Math.imul(seed ^ (seed >>> 15), seed | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
  const surface = offscreenSurface(root, { width: 64, height: 36, format: 'rgba8unorm' });
  let scene: Awaited<ReturnType<typeof createScene>> | undefined;
  let loop: ReturnType<typeof frames.create> | undefined;
  try {
    scene = await begin(
      await createScene(root, surface, { ...(await filesOf('skeletons', fromDisk)), count: 32 }),
    );
    loop = frames.create(root, scene.frame);
    for (let i = 0; i < 100; i++) {
      loop.run(i * 0.04);
      await device.queue.onSubmittedWorkDone();
    }
    const result: CapturedBuffer[] = [];
    for (const buffer of buffers.filter((value) =>
      /^(Pose_device|poses_joints|poses_spans)/.test(value.label),
    )) {
      const staging = createBuffer({
        size: buffer.size,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      result.push({ label: buffer.label, bytes: staging.getMappedRange().slice(0) });
      staging.unmap();
      staging.destroy();
    }
    return result;
  } finally {
    device.createBuffer = createBuffer;
    Math.random = random;
    loop?.destroy();
    scene?.destroy();
    surface.destroy();
  }
}

function fieldAt(layout: Layout, offset: number): string {
  if (layout.fields) {
    const member = Object.entries(layout.fields).find(
      ([, field]) => offset >= field.offset && offset < field.offset + field.layout.size,
    );
    return member
      ? `${member[0]}${fieldAt(member[1].layout, offset - member[1].offset)}`
      : '<padding>';
  }
  if (layout.element)
    return `[${Math.floor(offset / layout.stride!)}]${fieldAt(layout.element, offset % layout.stride!)}`;
  return '';
}

export function poseField(label: string, word: number): string {
  if (label.startsWith('Pose_device')) {
    const layout = layoutOf(Pose['~parts'].device!);
    return `row[${Math.floor((word * 4) / layout.size)}].${fieldAt(layout, (word * 4) % layout.size)}`;
  }
  if (label.startsWith('poses_joints'))
    return `jointRow[${Math.floor(word / 4)}].${'xyzw'[word % 4]}`;
  return `span[${Math.floor(word / 3)}].${['first', 'count', 'duration'][word % 3]}`;
}
