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
  /** Uploads of the CPU copies kernels wrote, before GL uses them. */
  readonly upload: { ms: number; textures: number };
  reset(): void;
}

export function createStats(): DeviceStats {
  const stats: DeviceStats = {
    kernels: new Map(),
    readback: { ms: 0, textures: 0 },
    upload: { ms: 0, textures: 0 },
    reset() {
      stats.kernels.clear();
      Object.assign(stats.readback, { ms: 0, textures: 0 });
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
  /** Frees a GL object when the device is destroyed, unless its owner frees it first. */
  track(free: () => void): () => void;
  /** Called after a submit wrote to a canvas texture, to present it. */
  presented(texture: object): void;
  readonly stats: DeviceStats;
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
