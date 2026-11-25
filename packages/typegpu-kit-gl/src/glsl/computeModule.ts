/**
 * The GPU form of a compute pipeline: the kernel as WebGL 2 fragment passes, which a CPU module
 * carries next to its JavaScript when the kernel can run that way. The CPU module stays the
 * authoritative fallback; the device uses the GPU form for a dispatch only when everything below
 * says it may, and the device depends only on the types here and on `parseCompute`.
 *
 * ## How a dispatch runs
 *
 * Each invocation of a kernel the GPU form accepts writes only its own cell of the resources it
 * writes: element `global_invocation_id.x` of one array in a storage buffer (or, with strides and
 * offsets every invocation shares, an element no other invocation's index reaches), or texel
 * `global_invocation_id.xy` (`.xyz` for 2d-array and 3d) of a storage texture, or, in a form whose
 * `grid` is 2, the column of a 2d-array or 3d texture at `global_invocation_id.xy`, every layer or
 * slice of it. A buffer target may also be a whole binding that at most one invocation writes, as
 * values every invocation sees alike choose; its pass works out which. A pass renders one such
 * target: every fragment runs the whole kernel for the invocation that owns its cell and outputs
 * the cell's new value. Writes go to per-invocation copies of the written cells, so reads
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
 *    texture target, the view's mip level. A 2d-array or 3d target's pass has `LAYERS_PER_DRAW`
 *    outputs, at locations 0 and up, which hold layers or slices `kit_layer` (`int`) and up; the
 *    device draws it once per that many layers with `kit_layer` set to the first, and leaves the
 *    outputs past the last layer unattached. Any other pass has one output, with `kit_layer` 0.
 *    It draws `PASS_VERTEX` as three vertices with the viewport covering the whole target, no
 *    blending, no depth or stencil and no flip: `gl_FragCoord.xy` is the cell's column and row in
 *    memory, row 0 first. Every fragment is written, so the new texture needs no initial contents.
 * 5. After the last pass, replaces each target's contents with its new texture, and treats the
 *    CPU copies of those resources as stale.
 *
 * Some forms need more of the dispatch, which only devices that check it see: see
 * `ParseComputeOptions.conditions`.
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
/** Marks a pass of points: `//#scatter <sites> <words> [lanes]`. */
export const SCATTER = '//#scatter';
/** Marks a pass of points that adds to a buffer: `//#accumulate <slots>`. */
export const ACCUMULATE = '//#accumulate';
/**
 * The most records a pass of sums may sum into one word in a dispatch. Each float channel sums
 * one nibble of every amount, so it stays exact up to 2^24 / 15.
 */
export const MOST_ADDED = Math.floor(2 ** 24 / 15);
/** The most invocations a dispatch that takes may have, so ranking stays a few hundred reads. */
export const MOST_TAKEN = 65_536;

/** Uniforms every pass may declare, which the device sets as step 3 and 4 of a dispatch say. */
export const INVOCATIONS = 'kit_invocations';
export const WORKGROUPS = 'kit_workgroups';
export const LAYER = 'kit_layer';
/** Layers of a 2d-array or 3d target that one draw of its pass writes, as step 4 says. */
export const LAYERS_PER_DRAW = 4;

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
  /**
   * Present when element `x + y * width + z * width * height` belongs to invocation `(x, y, z)`.
   * The form may run only when the dispatch's x invocation count is at most `width`.
   */
  readonly width?: number;
  /** Present with `width`: the form may run only when the y invocation count is at most this. */
  readonly height?: number;
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
  /**
   * Complete GLSL ES 3.00 fragment source. Its outputs, from location 0, are the target's cells,
   * as step 4 says; in a pass of points, the cell its point covers.
   */
  readonly fragment: string;
  /** Present for a pass of points, as `Scatter` describes. */
  readonly scatter?: Scatter;
}

/** A field of a write site's record: its name and GLSL type, `uint`, `ivec3` or a 4-vector. */
export interface ScatterField {
  readonly name: string;
  readonly type: 'uint' | 'ivec3' | 'uvec4' | 'ivec4' | 'vec4';
}

/**
 * A pass of points, for a target the kernel writes at cells it computes.
 *
 * First the device runs `capture`, a vertex stage, as `POINTS` with the rasterizer discarded, one
 * vertex per invocation (`gl_VertexID`, `x` fastest, then `y`, then `z`), with the uniforms of
 * step 3 and the bindings of step 2. Transform feedback records, interleaved, for each site from
 * 0 to `sites - 1` and each of `fields` in order, the varying `kit_c<site>_<name>`: what the
 * invocation wrote at that write site, with field `hit` 0 where it did not reach it.
 *
 * Then it starts the target's new contents as a copy of the old ones, and draws `vertex` with
 * `fragment` as `POINTS`, one vertex per invocation, with the state of step 4, reading one site's
 * record as the attributes `kit_a_<name>`. It draws once per site. For a buffer target it also
 * draws once per word of an element, `kit_part` (`int`) from 0 to `words - 1`, and per channel of
 * a data texel, `kit_phase` (`int`) from 0 to 3, writing only that channel. For a 2d-array or 3d
 * target it draws once per layer, `kit_layer` set to it, into that layer alone. It then replaces
 * the target as step 5 says.
 */
export interface Scatter {
  readonly capture: string;
  readonly vertex: string;
  readonly fields: readonly ScatterField[];
  readonly sites: number;
  readonly words: number;
  /**
   * For a buffer target, 1, or the words a draw writes together, 2 or 4: then each draw writes a
   * whole element, `kit_part` stays 0, and `kit_phase` runs from 0 to `3 / lanes`, writing the
   * `words` channels from `kit_phase * lanes` on.
   */
  readonly lanes: number;
  /**
   * Set for a pass that adds to a buffer, as `Accumulating` describes. Its fields are `hit`, `at`
   * and `value`, all `uint`, `sites` is its slots, and `words` and `lanes` are 1.
   */
  readonly accumulate?: true;
  /**
   * How a pass that accumulates combines its records with each word, as `combiningFragment`
   * describes; absent, it adds them.
   */
  readonly operation?: Exclude<Operation, 'add'>;
  /**
   * Set for a counter whose values the kernel takes, as `Taking` describes: the `kit_aim` of its
   * `kit_bases<aim>` sampler.
   */
  readonly takes?: number;
}

/**
 * A counter the kernel takes values of: atomic additions of 1 to one whole `u32` whose results it
 * uses. The kernel qualifies when how many values an invocation takes does not depend on a value it
 * took: no taken value, nor a variable made from one, decides a branch or a loop, is returned or is
 * passed to a function of the kernel. Such a counter is also a pass of sums, which adds the takes.
 *
 * Before any pass, the device runs that pass's capture stage with `kit_ranked` (`int`) 0, and
 * counts each invocation's hits. It renders, into a `R32UI` texture 256 texels wide, each
 * invocation's base: the takes of the invocations before it, in the order `gl_VertexID` numbers
 * them. It binds that texture to `kit_bases<aim>` in every program of the form and sets
 * `kit_ranked` to 1. A take then returns the counter's value before the dispatch, plus the
 * invocation's base, plus its own takes before it: one order WebGPU's atomics may give. A dispatch
 * of more than `MOST_TAKEN` invocations runs on the CPU.
 */
export const TAKE_WIDTH = 256;

/**
 * A pass of points that adds to a buffer, for atomic additions whose results the kernel drops on
 * cells other invocations may add to too. The kernel qualifies when every atomic on the buffer is
 * an `atomicAdd` or `atomicSub` whose value its statement drops, it neither reads nor writes the
 * buffer otherwise, except with an `atomicLoad` that no addition to the buffer can precede on any
 * path through the kernel, which reads the contents from before the dispatch, and no addition, its
 * own or in a function it calls, is inside a runtime loop: then an invocation makes at most
 * `sites` additions, its slots, the most any one path makes.
 *
 * The capture stage records, for each slot from 0 to `sites - 1`, whether the invocation made that
 * many additions (`hit`), the word it added to (`at`) and the amount as `uint` bits (`value`,
 * negated for a subtraction). The device splits each amount into its eight nibbles. It clears
 * eight `RGBA32F` textures of the data texture's size to 0, and draws `vertex` with
 * `accumulateFragment(targets)` as `POINTS` once per slot with additive blending (`ONE`, `ONE`),
 * each point on its word's data texel, carrying `kit_value` and `kit_lane`, the word's channel.
 * Texture `k` sums nibble `k` of the amounts in each channel. A device of eight draw buffers draws
 * into all eight at once; one of four draws every slot twice, into four textures each, with
 * `kit_group` (`int`) 0 for nibbles 0 to 3 and 1 for nibbles 4 to 7. Last it renders the target's
 * new contents with `ACCUMULATED_FRAGMENT`: each word plus the sum of its nibbles at their places,
 * in `uint` arithmetic, which wraps as the atomic does.
 *
 * The sums are exact while no word takes more than `MOST_ADDED` records in a dispatch, so the
 * device runs a dispatch of more invocations times slots on the CPU. The order of additions does
 * not change a sum.
 */
export function accumulateFragment(targets: 4 | 8): string {
  const outputs = Array.from(
    { length: targets },
    (_, k) => `layout(location = ${k}) out vec4 kit_part${k};`,
  );
  const first = targets === 8 ? '0' : '4 * kit_group';
  const sums = Array.from(
    { length: targets },
    (_, k) => `  kit_part${k} = lane * float((kit_value >> uint(4 * (${first} + ${k}))) & 15u);`,
  );
  return `#version 300 es
precision highp float;
precision highp int;
flat in uint kit_value;
flat in uint kit_lane;
${targets === 8 ? '' : 'uniform int kit_group;\n'}${outputs.join('\n')}
void main() {
  vec4 lane = vec4(equal(uvec4(kit_lane), uvec4(0u, 1u, 2u, 3u)));
${sums.join('\n')}
}`;
}

/** How a pass that accumulates records into a buffer's words combines them. */
export type Operation = 'add' | 'min' | 'max' | 'or' | 'and';

/** Operations other than `add`, by the word the accumulate directive names them with. */
export const OPERATIONS: readonly Exclude<Operation, 'add'>[] = ['min', 'max', 'or', 'and'];

/**
 * A pass of points that combines records with a buffer's words by `min`, `max`, `or` or `and`,
 * for those atomics whose results the kernel drops on cells other invocations share, as
 * `Accumulating` describes for additions. Each record is `value` at word `at`. None of these has a
 * bound on how many records a word takes, and none depends on their order.
 *
 * `min` and `max` on `u32` run in two rounds into one `RGBA32F` texture each, blended with
 * `MIN` or `MAX`, which compares without arithmetic, so each channel holds exactly the least or
 * greatest of the values drawn into it. Round 0 (`kit_round` 0) draws `value >> 16`, and round 1
 * draws `value & 0xffff` for the records whose high half equals the one round 0 kept for their
 * word, read from `kit_high`; both halves are integers below 2^24, so floats hold them exactly. The
 * order of `u32` is that of (high half, low half), so the two rounds find the least or greatest
 * value. Channels no record reaches keep the cleared `kit_neutral`, 65536 for `min` and -1 for
 * `max`, which no half equals. `MINMAX_FRAGMENT` then renders each word combined with the result.
 *
 * `or` and `and` run as bit-planes into `RGBA8` textures blended with `MAX`, one bit of every word
 * per texture: a record writes 1 where its bit is set, for `and` where it is clear, so a channel
 * holds whether any record set (or cleared) that bit. A round draws `targets` bits, `kit_group`
 * times `targets` to that plus `targets - 1`, and `BITS_FRAGMENT` renders each word with those
 * bits set, or for `and` cleared; the next round starts from that result.
 */
export function combiningFragment(operation: Exclude<Operation, 'add'>, targets: 4 | 8): string {
  if (operation === 'min' || operation === 'max') {
    return `#version 300 es
precision highp float;
precision highp int;
flat in uint kit_value;
flat in uint kit_lane;
uniform int kit_round;
uniform float kit_neutral;
uniform highp sampler2D kit_high;
layout(location = 0) out vec4 kit_part0;
void main() {
  vec4 lane = vec4(equal(uvec4(kit_lane), uvec4(0u, 1u, 2u, 3u)));
  float high = float(kit_value >> 16u);
  float piece = high;
  if (kit_round == 1) {
    if (dot(texelFetch(kit_high, ivec2(gl_FragCoord.xy), 0), lane) != high) discard;
    piece = float(kit_value & 65535u);
  }
  kit_part0 = mix(vec4(kit_neutral), vec4(piece), lane);
}`;
  }
  const outputs = Array.from(
    { length: targets },
    (_, k) => `layout(location = ${k}) out vec4 kit_part${k};`,
  );
  const bits = Array.from(
    { length: targets },
    (_, k) => `  kit_part${k} = lane * float((bits >> uint(${targets} * kit_group + ${k})) & 1u);`,
  );
  return `#version 300 es
precision highp float;
precision highp int;
flat in uint kit_value;
flat in uint kit_lane;
uniform int kit_group;
${outputs.join('\n')}
void main() {
  vec4 lane = vec4(equal(uvec4(kit_lane), uvec4(0u, 1u, 2u, 3u)));
  uint bits = ${operation === 'and' ? '~kit_value' : 'kit_value'};
${bits.join('\n')}
}`;
}

/** Renders each word of `kit_old` with the least or greatest value two rounds of `min` or `max` kept. */
export const MINMAX_FRAGMENT = `#version 300 es
precision highp float;
precision highp int;
uniform highp usampler2D kit_old;
uniform highp sampler2D kit_high;
uniform highp sampler2D kit_low;
uniform int kit_max;
layout(location = 0) out uvec4 kit_result;
void main() {
  ivec2 at = ivec2(gl_FragCoord.xy);
  uvec4 old = texelFetch(kit_old, at, 0);
  vec4 high = texelFetch(kit_high, at, 0);
  vec4 low = texelFetch(kit_low, at, 0);
  uvec4 kept = (uvec4(clamp(high, 0.0, 65535.0)) << 16u) | uvec4(clamp(low, 0.0, 65535.0));
  bvec4 reached = kit_max == 1 ? greaterThanEqual(high, vec4(0.0)) : lessThan(high, vec4(65536.0));
  uvec4 combined = kit_max == 1 ? max(old, kept) : min(old, kept);
  kit_result = uvec4(
    reached.x ? combined.x : old.x,
    reached.y ? combined.y : old.y,
    reached.z ? combined.z : old.z,
    reached.w ? combined.w : old.w);
}`;

/** Renders each word of `kit_old` with the bits one round of `or` or `and` drew set or cleared. */
export function bitsFragment(targets: 4 | 8): string {
  const planes = Array.from({ length: targets }, (_, k) => `uniform highp sampler2D kit_bit${k};`);
  const sum = Array.from(
    { length: targets },
    (_, k) =>
      `(uvec4(greaterThan(texelFetch(kit_bit${k}, at, 0), vec4(0.5))) << uint(${targets} * kit_group + ${k}))`,
  );
  return `#version 300 es
precision highp float;
precision highp int;
uniform highp usampler2D kit_old;
${planes.join('\n')}
uniform int kit_group;
uniform int kit_and;
layout(location = 0) out uvec4 kit_result;
void main() {
  ivec2 at = ivec2(gl_FragCoord.xy);
  uvec4 old = texelFetch(kit_old, at, 0);
  uvec4 bits = ${sum.join('\n    | ')};
  kit_result = kit_and == 1 ? old & ~bits : old | bits;
}`;
}

/** Renders a buffer's data texture plus the nibbles' sums a pass of sums drew, at their places. */
export const ACCUMULATED_FRAGMENT = `#version 300 es
precision highp float;
precision highp int;
uniform highp usampler2D kit_old;
${Array.from({ length: 8 }, (_, k) => `uniform highp sampler2D kit_sum${k};`).join('\n')}
layout(location = 0) out uvec4 kit_result;
void main() {
  ivec2 at = ivec2(gl_FragCoord.xy);
  uvec4 added = ${Array.from({ length: 8 }, (_, k) => `(uvec4(texelFetch(kit_sum${k}, at, 0)) << ${4 * k}u)`).join('\n    + ')};
  kit_result = texelFetch(kit_old, at, 0) + added;
}`;

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
  /**
   * True when one invocation owns whole bindings: the form may run only when the dispatch is
   * exactly one invocation, all three counts 1.
   */
  readonly single?: boolean;
}

/**
 * A condition on the dispatch, beyond those of step 1, that some GPU forms need the device to
 * check: `limits`, a buffer target's `width` and `height`; `single`, the module's `single`.
 */
export type ComputeCondition = 'limits' | 'single';

export interface ParseComputeOptions {
  /** Lines the device adds after `#version`. Default none. */
  readonly defines?: readonly string[];
  /**
   * The conditions the device checks. A form that needs another one parses as `undefined`, so
   * the dispatch stays on the CPU. Default none.
   */
  readonly conditions?: readonly ComputeCondition[];
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
    const [offset, stride, size, width, height] = rest;
    return {
      kind,
      group: int(group, line),
      binding: int(binding, line),
      source: uniform as string,
      offset: int(offset, line),
      stride: int(stride, line),
      size: int(size, line),
      ...(width !== undefined && { width: int(width, line) }),
      ...(height !== undefined && { height: int(height, line) }),
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

/** The fragment stage of a pass of points: it writes the value its point carries. */
function pointFragment(target: ComputeTarget, defines: readonly string[]): string {
  const type =
    target.kind === 'buffer'
      ? 'uvec4'
      : /uint/.test(target.format)
        ? 'uvec4'
        : /sint/.test(target.format)
          ? 'ivec4'
          : 'vec4';
  return [
    '#version 300 es',
    ...defines,
    'precision highp float;\nprecision highp int;',
    `flat in ${type} kit_value;`,
    `layout(location = 0) out ${type} kit_result;`,
    'void main() {\n  kit_result = kit_value;\n}',
  ].join('\n');
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
  const [, wx, wy, wz, grid, single] = meta.split(/\s+/);
  const conditions = new Set(options.conditions ?? []);
  if (single === '1' && !conditions.has('single')) return undefined;

  const chunks = chunksOf(lines.slice(1).join('\n'));
  const passes: ComputePass[] = [];
  const used = new Set<(typeof chunks)[number]>();
  for (const chunk of chunks) {
    if (!chunk.stage.startsWith('pass')) continue;
    const line = chunk.directives.find((each) => each.startsWith(`${TARGET} `));
    if (!line) throw new Error('@typegpu/kit-gl: a pass without a target');
    const included = reachable(chunks, chunk.stage as ChunkStage);
    for (const each of included) used.add(each);
    const target = targetOf(line);
    if (target.kind === 'buffer' && target.width !== undefined && !conditions.has('limits')) {
      return undefined;
    }
    const accumulate = chunk.directives.find((each) => each.startsWith(`${ACCUMULATE} `));
    const scatter = accumulate ?? chunk.directives.find((each) => each.startsWith(`${SCATTER} `));
    if (scatter) {
      const parts = scatter.trim().split(/\s+/);
      const [, sites, words = '1', lanes = '1'] = accumulate ? [ACCUMULATE, parts[1]] : parts;
      const taken =
        accumulate && parts[2] === 'take' ? int(parts[3] as string, scatter) : undefined;
      const named = accumulate ? parts[parts.indexOf('op') + 1] : undefined;
      const operation =
        accumulate && parts.includes('op') ? OPERATIONS.find((each) => each === named) : undefined;
      if (accumulate && parts.includes('op') && !operation) {
        throw new Error(`@typegpu/kit-gl: an accumulating pass with operation '${named}'`);
      }
      const stage = `points${chunk.stage.slice('pass'.length)}`;
      const draw = chunks.find((each) => each.stage === stage);
      if (!draw) throw new Error(`@typegpu/kit-gl: a pass of points without its '${stage}'`);
      const drawing = reachable(chunks, stage as ChunkStage);
      for (const each of drawing) used.add(each);
      const fields = [...draw.code.matchAll(/^in (\w+) kit_a_(\w+);$/gm)].map(([, type, name]) => ({
        name: name as string,
        type: type as ScatterField['type'],
      }));
      passes.push({
        target,
        fragment: operation
          ? combiningFragment(operation, 8)
          : accumulate
            ? accumulateFragment(8)
            : pointFragment(target, options.defines ?? []),
        scatter: {
          capture: source(included, options.defines ?? []),
          vertex: source(drawing, options.defines ?? []),
          fields,
          sites: int(sites as string, scatter),
          words: int(words, scatter),
          lanes: int(lanes, scatter),
          ...(accumulate && { accumulate: true as const }),
          ...(operation && { operation }),
          ...(taken !== undefined && { takes: taken }),
        },
      });
      continue;
    }
    passes.push({ target, fragment: source(included, options.defines ?? []) });
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
    single: single === '1',
  };
}
