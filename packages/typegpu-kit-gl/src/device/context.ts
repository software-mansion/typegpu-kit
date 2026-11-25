/** What every part of the device shares: the GL context, its state cache and bookkeeping. */

import type { Errors } from './errors.ts';
import type { Extensions } from './extensions.ts';
import type { FormatTable } from './formats.ts';
import type { GLState } from './state.ts';

/** Where the device spent its time since the last `reset`, for finding slow frames. */
export interface DeviceStats {
  /** Milliseconds and dispatches per compute pipeline, by label. */
  readonly kernels: Map<string, { ms: number; dispatches: number }>;
  /** Texture reads from GL into CPU copies for kernels. */
  readonly readback: { ms: number; textures: number };
  /** Readback per texture, by label with its format and size, for finding the costly ones. */
  readonly readbacks: Map<string, { ms: number; levels: number }>;
  /**
   * Per buffer, by label and size: reads of GL's copy back to the CPU after GPU forms wrote it, and
   * uploads of CPU writes to the copies GL draws or samples from.
   */
  readonly buffers: Map<string, BufferStats>;
  /** Draw calls, and the instances they drew. */
  readonly draws: { count: number; instances: number };
  /** Uploads of the CPU copies kernels wrote, before GL uses them. */
  readonly upload: { ms: number; textures: number };
  /**
   * Why a compute pipeline runs on the CPU, by label: the generator wrote no GPU form, the form
   * failed to link, or the last dispatch did not fit it. `reset` keeps these.
   */
  readonly fallbacks: Map<string, string>;
  /**
   * Where each compute pipeline with a GPU form runs and why, by label, as its readbacks and
   * costs decided. `reset` keeps these.
   */
  readonly placement: Map<string, string>;
  reset(): void;
}

export interface BufferStats {
  readMs: number;
  reads: number;
  uploadMs: number;
  uploads: number;
  uploadBytes: number;
}

export function createStats(): DeviceStats {
  const stats: DeviceStats = {
    kernels: new Map(),
    readback: { ms: 0, textures: 0 },
    readbacks: new Map(),
    buffers: new Map(),
    draws: { count: 0, instances: 0 },
    upload: { ms: 0, textures: 0 },
    fallbacks: new Map(),
    placement: new Map(),
    reset() {
      stats.kernels.clear();
      Object.assign(stats.readback, { ms: 0, textures: 0 });
      stats.readbacks.clear();
      stats.buffers.clear();
      Object.assign(stats.draws, { count: 0, instances: 0 });
      Object.assign(stats.upload, { ms: 0, textures: 0 });
    },
  };
  return stats;
}

export interface Ctx {
  readonly gl: WebGL2RenderingContext;
  readonly canvas: HTMLCanvasElement | OffscreenCanvas;
  readonly ext: Extensions;
  readonly state: GLState;
  readonly formats: FormatTable;
  readonly errors: Errors;
  /** Lines added after `#version` in every render module the device parses. */
  readonly defines: readonly string[];
  readonly maxTextureSize: number;
  readonly maxTextureUnits: number;
  readonly uniformAlignment: number;
  readonly maxUniformBindings: number;
  /** A sampler that filters nothing, for buffers and texel loads. */
  readonly nearest: WebGLSampler;
  /** Framebuffers the device uses for copies, blits and readback. */
  readonly scratch: { readonly read: WebGLFramebuffer; readonly draw: WebGLFramebuffer };
  /** Unique numbers for resource identity in cache keys. */
  id(): number;
  /**
   * Frees a GL object when the device is destroyed, unless its owner frees it first. With
   * `owner`, also once the owner is garbage collected, as WebGPU frees what a program drops:
   * `free` must then reach no object that reaches the owner.
   */
  track(free: () => void, owner?: object): () => void;
  /** Called after a submit wrote to a canvas texture, to present it. */
  presented(texture: object): void;
  readonly stats: DeviceStats;
  /** Whether compute pipelines may run their GPU forms. */
  readonly gpuCompute: boolean;
  lost: boolean;
}

/** The key under which a WebGPU-shaped object keeps its device-side record. */
export const IMPL: unique symbol = Symbol('@typegpu/kit-gl/impl');

export function implOf<T>(value: unknown, what: string): T {
  const impl = (value as { [IMPL]?: T } | null | undefined)?.[IMPL];
  if (impl === undefined) {
    throw new Error(`@typegpu/kit-gl: the ${what} does not belong to a WebGL device.`);
  }
  return impl;
}

export const BUFFER_USAGE = {
  MAP_READ: 0x1,
  MAP_WRITE: 0x2,
  COPY_SRC: 0x4,
  COPY_DST: 0x8,
  INDEX: 0x10,
  VERTEX: 0x20,
  UNIFORM: 0x40,
  STORAGE: 0x80,
  INDIRECT: 0x100,
  QUERY_RESOLVE: 0x200,
} as const;

export const TEXTURE_USAGE = {
  COPY_SRC: 0x1,
  COPY_DST: 0x2,
  TEXTURE_BINDING: 0x4,
  STORAGE_BINDING: 0x8,
  RENDER_ATTACHMENT: 0x10,
} as const;

export const SHADER_STAGE = { VERTEX: 0x1, FRAGMENT: 0x2, COMPUTE: 0x4 } as const;

export const MAP_MODE = { READ: 0x1, WRITE: 0x2 } as const;

export const COLOR_WRITE = { RED: 0x1, GREEN: 0x2, BLUE: 0x4, ALPHA: 0x8, ALL: 0xf } as const;
