/**
 * What a linked program reads and where: texture units for buffers and textures, uniform block
 * binding points and level-count uniforms. Render pipelines and compute passes bind through it.
 */

import type { Bindings } from '../glsl/renderModule.ts';
import { type Bound, bufferEntry } from './binding.ts';
import { bindUniformRange, dataTexture } from './buffer.ts';
import type { Ctx } from './context.ts';
import type { Program } from './program.ts';
import { sampled } from './texture.ts';

export interface BufferUse {
  readonly group: number;
  readonly binding: number;
  readonly unit: number;
  readonly range: WebGLUniformLocation | null;
}

export interface TextureUse {
  readonly group: number;
  readonly binding: number;
  readonly unit: number;
  readonly sampler: { readonly group: number; readonly binding: number } | undefined;
  readonly comparison: boolean;
  readonly depth: boolean;
}

export interface BlockUse {
  readonly group: number;
  readonly binding: number;
  readonly point: number;
  readonly size: number;
}

export interface LevelsUse {
  readonly group: number;
  readonly binding: number;
  readonly location: WebGLUniformLocation;
}

export interface Uses {
  readonly buffers: readonly BufferUse[];
  readonly textures: readonly TextureUse[];
  readonly blocks: readonly BlockUse[];
  readonly levels: readonly LevelsUse[];
}

/**
 * Assigns texture units and uniform block binding points on a program that just linked. Returns an
 * error message when the program needs more of either than the context has.
 */
export function connectUses(
  ctx: Ctx,
  label: string,
  bindings: Bindings,
  program: Program,
): Uses | string {
  const { gl, state } = ctx;
  const units = bindings.buffers.length + bindings.textures.length;
  if (units > ctx.maxTextureUnits - 1) {
    return `'${label}' binds ${bindings.buffers.length} buffers and ${bindings.textures.length} textures, which need ${units} texture units; WebGL 2 here has ${ctx.maxTextureUnits - 1} for shaders.`;
  }
  state.program(program.program);
  const buffers: BufferUse[] = [];
  const textures: TextureUse[] = [];
  let unit = 0;
  const seen = new Set<string>();
  for (const binding of bindings.buffers) {
    if (seen.has(binding.texture)) continue;
    seen.add(binding.texture);
    const location = program.uniforms.get(binding.texture);
    if (!location) continue;
    gl.uniform1i(location, unit);
    buffers.push({
      group: binding.group,
      binding: binding.binding,
      unit,
      range: program.uniforms.get(binding.range) ?? null,
    });
    unit++;
  }
  for (const binding of bindings.textures) {
    if (seen.has(binding.uniform)) continue;
    seen.add(binding.uniform);
    const location = program.uniforms.get(binding.uniform);
    if (!location) continue;
    gl.uniform1i(location, unit);
    textures.push({
      group: binding.group,
      binding: binding.binding,
      unit,
      sampler: binding.sampler,
      comparison: binding.comparison,
      depth: binding.sampleType === 'depth',
    });
    unit++;
  }
  const blocks: BlockUse[] = [];
  for (const binding of bindings.blocks) {
    const index = gl.getUniformBlockIndex(program.program, binding.block);
    if (index === gl.INVALID_INDEX) continue;
    const point = blocks.length;
    if (point >= ctx.maxUniformBindings) {
      return `'${label}' reads ${bindings.blocks.length} uniform blocks; WebGL 2 here binds ${ctx.maxUniformBindings}.`;
    }
    gl.uniformBlockBinding(program.program, index, point);
    blocks.push({ group: binding.group, binding: binding.binding, point, size: binding.size });
  }
  const levels = bindings.levels.flatMap((binding) => {
    const location = program.uniforms.get(binding.uniform);
    return location ? [{ group: binding.group, binding: binding.binding, location }] : [];
  });
  return { buffers, textures, blocks, levels };
}

const ranges = new WeakMap<WebGLUniformLocation, [number, number]>();
const ints = new WeakMap<WebGLUniformLocation, number>();

/** Sets an `int` uniform of the program in use, unless it holds the value already. */
export function setInt(
  gl: WebGL2RenderingContext,
  location: WebGLUniformLocation | null,
  value: number,
): void {
  if (!location || ints.get(location) === value) return;
  ints.set(location, value);
  gl.uniform1i(location, value);
}

/**
 * Binds what a program reads from `groups`, with the program in use. Returns what is missing, or
 * undefined once everything is bound.
 */
/** Formats each device reported sampling with filtering it cannot do, so it reports each once. */
const unfilteredFormats = new WeakMap<Ctx, Set<GPUTextureFormat>>();

function reportUnfiltered(ctx: Ctx, format: GPUTextureFormat): void {
  let reported = unfilteredFormats.get(ctx);
  if (!reported) unfilteredFormats.set(ctx, (reported = new Set()));
  if (reported.has(format)) return;
  reported.add(format);
  // WebGPU rejects this too, without the `float32-filterable` feature this device then lacks
  ctx.errors.report(
    'validation',
    `A filtering sampler reads a '${format}' texture, which needs OES_texture_float_linear on WebGL 2; it samples with nearest filtering.`,
  );
}

export function bindUses(
  ctx: Ctx,
  uses: Uses,
  program: Program,
  groups: readonly (Bound | undefined)[],
): string | undefined {
  const { gl, state } = ctx;
  for (const use of uses.buffers) {
    const entry = bufferEntry(groups[use.group], use.binding);
    if (!entry) return `buffer @group(${use.group}) @binding(${use.binding})`;
    const data = dataTexture(ctx, entry.buffer);
    state.texture(use.unit, gl.TEXTURE_2D, data.texture);
    state.sampler(use.unit, ctx.nearest);
    if (use.range) {
      const words = entry.offset >>> 2;
      const last = ranges.get(use.range);
      if (!last || last[0] !== words || last[1] !== entry.size) {
        state.program(program.program);
        gl.uniform2ui(use.range, words, entry.size);
        ranges.set(use.range, [words, entry.size]);
      }
    }
  }
  for (const use of uses.textures) {
    const entry = groups[use.group]?.group.entries.get(use.binding);
    if (entry?.kind !== 'texture') return `texture @group(${use.group}) @binding(${use.binding})`;
    const { target, texture } = sampled(ctx, entry.view);
    state.texture(use.unit, target, texture);
    let sampler = ctx.nearest;
    if (use.sampler) {
      const samplerEntry = groups[use.sampler.group]?.group.entries.get(use.sampler.binding);
      if (samplerEntry?.kind !== 'sampler') {
        return `sampler @group(${use.sampler.group}) @binding(${use.sampler.binding})`;
      }
      const impl = samplerEntry.sampler;
      const { info } = entry.view.texture;
      sampler =
        use.comparison || (info.filterable && !use.depth) ? impl.sampler : impl.unfiltered();
      if (impl.filtering && !info.filterable && info.kind === 'float' && !use.comparison) {
        reportUnfiltered(ctx, entry.view.texture.format);
      }
    }
    state.sampler(use.unit, sampler);
  }
  for (const use of uses.blocks) {
    const entry = bufferEntry(groups[use.group], use.binding);
    if (!entry) return `uniform buffer @group(${use.group}) @binding(${use.binding})`;
    bindUniformRange(ctx, entry.buffer, use.point, entry.offset, use.size);
  }
  for (const use of uses.levels) {
    const entry = groups[use.group]?.group.entries.get(use.binding);
    if (entry?.kind !== 'texture') continue;
    state.program(program.program);
    setInt(gl, use.location, entry.view.mipLevelCount);
  }
  return undefined;
}
