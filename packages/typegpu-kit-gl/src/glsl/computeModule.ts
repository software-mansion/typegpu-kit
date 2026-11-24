/**
 * The GPU form of a compute pipeline: the kernel as WebGL 2 fragment passes, which a CPU module
 * carries next to its JavaScript when the kernel can run that way. The CPU module stays the
 * authoritative fallback; the device uses the GPU form for a dispatch only when everything below
 * says it may, and the device depends only on the types here and on `parseCompute`.
 *
 * ## How a dispatch runs
 *
 * Each invocation of a kernel the GPU form accepts writes only its own cell of the resources it
 * writes: element `global_invocation_id.x` of one array in a storage buffer, or texel
 * `global_invocation_id.xy` (`.xyz` for 2d-array and 3d) of a storage texture. A pass renders one
 * such target: every fragment runs the whole kernel for the invocation that owns its cell and
 * outputs the cell's new value. Writes go to per-invocation copies of the written cells, so reads
 * after writes inside an invocation see them; every other read sees the contents from before the
 * dispatch.
 *
 * For a dispatch of `x * y * z` workgroups, the device:
 *
 * 1. Uses the GPU form only if all of these hold, and runs the CPU module otherwise:
 *    - the invocation counts `invocations = (x, y, z) * workgroupSize` are 1 along every axis the
 *      module's `grid` leaves out (`grid` 1: `y` and `z`; `grid` 2: `z`);
 *    - every texture target's format is color-renderable on this context (float formats need
 *      `EXT_color_buffer_float`);
 *    - `bindings` agrees with the CPU module's `bindings`: every entry here exists there with the
 *      same group and binding and kind (`texture` here matches `texture` or `storageTexture`
 *      there), and every target is a binding the CPU module writes;
 *    - no target's memory is bound twice in the dispatch: not two targets in one buffer, and not a
 *      target buffer also bound at another binding, whose reads would see the old contents.
 * 2. Binds the module's `buffers`, `blocks`, `textures` and `levels` as for a render module, all
 *    with their contents from before the dispatch. A target is read through its own entry there,
 *    named by the target's `source`: the data texture of a buffer target, or for a texture target
 *    an unfiltered sampler uniform over the storage texture's view.
 * 3. Sets the uniforms `kit_invocations` (`uvec3`, the invocation counts) and `kit_workgroups`
 *    (`uvec3`, `x`, `y` and `z`) in every pass.
 * 4. Runs each pass into a new texture of the target's size and format: for a buffer target, a
 *    copy of the shape of the buffer's data texture (`RGBA32UI`, same width and height); for a
 *    texture target, the view's mip level, once per layer or slice of a 2d-array or 3d target with
 *    `kit_layer` (`int`) set to its index, and 0 otherwise. It draws `PASS_VERTEX` as three
 *    vertices with the viewport covering the whole target, no blending, no depth or stencil and no
 *    flip: `gl_FragCoord.xy` is the cell's column and row in memory, row 0 first. Every fragment
 *    is written, so the new texture needs no initial contents.
 * 5. After the last pass, replaces each target's contents with its new texture, and treats the
 *    CPU copies of those resources as stale.
 *
 * A buffer target's pass covers the whole data texture. A word outside the target array, past the
 * bound range, in padding inside an element, or of an element at or past `kit_invocations.x`
 * keeps its old value, as does a cell outside the invocation counts in a texture target.
 */

import {
  bindingsOf,
  chunksOf,
  int,
  reachable,
  source,
  type Bindings,
  type ChunkStage,
} from './renderModule.ts';

export const COMPUTE_HEADER = '//#kit-gl gpu-compute';
/** The line a CPU module carries its GPU form on, followed by the form's code as JSON. */
export const GPU_FORM = '//#kit-gl gpu ';
/** The line a CPU module carries instead when its kernel cannot run as fragment passes. */
export const CPU_ONLY = '//#kit-gl cpu-only ';
export const COMPUTE = '//#compute';
export const TARGET = '//#target';

/** Uniforms every pass may declare, which the device sets as step 3 and 4 of a dispatch say. */
export const INVOCATIONS = 'kit_invocations';
export const WORKGROUPS = 'kit_workgroups';
export const LAYER = 'kit_layer';

/** The vertex stage of every pass: a triangle that covers the viewport, without varyings. */
export const PASS_VERTEX = `#version 300 es
void main() {
  vec2 corner = vec2(float((gl_VertexID & 1) << 2), float((gl_VertexID & 2) << 1)) - 1.0;
  gl_Position = vec4(corner, 0.0, 1.0);
}`;

/** One array of a storage buffer, written element by element. */
export interface BufferTarget {
  readonly kind: 'buffer';
  readonly group: number;
  readonly binding: number;
  /** The `usampler2D` uniform of the buffer's data texture, also listed in `buffers`. */
  readonly source: string;
  /** The array's first byte, from the start of the bound range. */
  readonly offset: number;
  /** Bytes from one element to the next. */
  readonly stride: number;
  /** An element's size in bytes, without the padding up to `stride`. */
  readonly size: number;
}

/** A storage texture, written texel by texel. */
export interface TextureTarget {
  readonly kind: 'texture';
  readonly group: number;
  readonly binding: number;
  /** The unfiltered sampler uniform over the texture's view, also listed in `textures`. */
  readonly source: string;
  readonly dimension: '1d' | '2d' | '2d-array' | '3d';
  readonly format: GPUTextureFormat;
}

export type ComputeTarget = BufferTarget | TextureTarget;

export interface ComputePass {
  readonly target: ComputeTarget;
  /** Complete GLSL ES 3.00 fragment source. Its one output, location 0, is the target's cell. */
  readonly fragment: string;
}

/** A binding the GPU form uses, to compare with the CPU module's `bindings`. */
export interface ComputeBinding {
  readonly group: number;
  readonly binding: number;
  readonly kind: 'buffer' | 'texture' | 'sampler';
  readonly writes: boolean;
}

export interface ComputeModule extends Bindings {
  readonly kind: 'gpu-compute';
  readonly workgroupSize: readonly [number, number, number];
  /** How many axes of the invocation grid the passes cover, from `x`: see step 1. */
  readonly grid: 1 | 2 | 3;
  readonly passes: readonly ComputePass[];
  readonly bindings: readonly ComputeBinding[];
}

export interface ParseComputeOptions {
  /** Lines the device adds after `#version`. Default none. */
  readonly defines?: readonly string[];
}

/** The code of the GPU form a CPU module carries, if it carries one. */
export function gpuFormOf(code: string): string | undefined {
  const line = code.split('\n').find((each) => each.startsWith(GPU_FORM));
  return line && (JSON.parse(line.slice(GPU_FORM.length)) as string);
}

/** Why a CPU module carries no GPU form, if the generator said. */
export function cpuOnlyReason(code: string): string | undefined {
  const line = code.split('\n').find((each) => each.startsWith(CPU_ONLY));
  return line && (JSON.parse(line.slice(CPU_ONLY.length)) as string);
}

function targetOf(line: string): ComputeTarget {
  const [, kind, group, binding, uniform, ...rest] = line.trim().split(/\s+/);
  if (kind === 'buffer') {
    const [offset, stride, size] = rest;
    return {
      kind,
      group: int(group, line),
      binding: int(binding, line),
      source: uniform as string,
      offset: int(offset, line),
      stride: int(stride, line),
      size: int(size, line),
    };
  }
  if (kind === 'texture') {
    const [dimension, format] = rest;
    return {
      kind,
      group: int(group, line),
      binding: int(binding, line),
      source: uniform as string,
      dimension: dimension as TextureTarget['dimension'],
      format: format as GPUTextureFormat,
    };
  }
  throw new Error(`@typegpu/kit-gl: malformed directive '${line}'`);
}

/**
 * Reads the GPU form a CPU module carries: one fragment source per target and the bindings they
 * read. Returns `undefined` when the module carries none, and the dispatch runs on the CPU.
 */
export function parseCompute(
  code: string,
  options: ParseComputeOptions = {},
): ComputeModule | undefined {
  const form = gpuFormOf(code);
  if (form === undefined) return undefined;
  const lines = form.split('\n');
  if (lines[0] !== COMPUTE_HEADER) throw new Error('@typegpu/kit-gl: not a GPU compute form');
  const meta = lines.find((line) => line.startsWith(`${COMPUTE} `)) as string;
  const [, wx, wy, wz, grid] = meta.split(/\s+/);

  const chunks = chunksOf(lines.slice(1).join('\n'));
  const passes: ComputePass[] = [];
  const used = new Set<(typeof chunks)[number]>();
  for (const chunk of chunks) {
    if (!chunk.stage.startsWith('pass')) continue;
    const line = chunk.directives.find((each) => each.startsWith(`${TARGET} `));
    if (!line) throw new Error('@typegpu/kit-gl: a pass without a target');
    const included = reachable(chunks, chunk.stage as ChunkStage);
    for (const each of included) used.add(each);
    passes.push({ target: targetOf(line), fragment: source(included, options.defines ?? []) });
  }

  const bindings = bindingsOf(used);
  const targets = new Set(passes.map(({ target }) => `${target.group}:${target.binding}`));
  const table = new Map<string, ComputeBinding>();
  const add = (group: number, binding: number, kind: ComputeBinding['kind']) => {
    const key = `${group}:${binding}`;
    table.set(key, { group, binding, kind, writes: targets.has(key) });
  };
  for (const each of [...bindings.buffers, ...bindings.blocks]) {
    add(each.group, each.binding, 'buffer');
  }
  for (const each of bindings.textures) {
    add(each.group, each.binding, 'texture');
    if (each.sampler) add(each.sampler.group, each.sampler.binding, 'sampler');
  }
  for (const each of bindings.levels) add(each.group, each.binding, 'texture');

  return {
    kind: 'gpu-compute',
    workgroupSize: [int(wx, meta), int(wy, meta), int(wz, meta)],
    grid: int(grid, meta) as 1 | 2 | 3,
    passes,
    ...bindings,
    bindings: [...table.values()],
  };
}
