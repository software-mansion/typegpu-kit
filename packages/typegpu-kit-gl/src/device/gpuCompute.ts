/**
 * Compute dispatches as fragment passes, for kernels whose generator wrote a GPU form: each pass
 * renders a target's new contents into a fresh texture from the contents before the dispatch, and
 * the new textures replace the targets once every pass ran. `src/glsl/computeModule.ts` states
 * when a dispatch may run this way; any other dispatch runs on the CPU.
 */

import type { CpuModule } from '../cpu/types.ts';
import {
  type ComputeModule,
  type ComputeTarget,
  INVOCATIONS,
  LAYER,
  PASS_VERTEX,
  WORKGROUPS,
} from '../glsl/computeModule.ts';
import { type Bound, bufferEntry } from './binding.ts';
import { type BufferImpl, dataTexture, replaceData } from './buffer.ts';
import type { Ctx } from './context.ts';
import { link, type Program } from './program.ts';
import type { BlendState } from './state.ts';
import {
  blitLevel,
  glWritten,
  mipSize,
  onGl,
  type TextureImpl,
  type ViewImpl,
  writeEverything,
} from './texture.ts';
import { bindUses, connectUses, setInt, type Uses } from './uses.ts';

interface GpuPass {
  readonly target: ComputeTarget;
  readonly program: Program;
  readonly uses: Uses;
  readonly invocations: WebGLUniformLocation | null;
  readonly workgroups: WebGLUniformLocation | null;
  readonly layer: WebGLUniformLocation | null;
}

export interface GpuCompute {
  readonly module: ComputeModule;
  readonly passes: readonly GpuPass[];
}

/** Checks the GPU form against the CPU module and links its passes, or says why it cannot run. */
export function prepareGpu(
  ctx: Ctx,
  label: string,
  cpu: CpuModule,
  module: ComputeModule,
): GpuCompute | string {
  for (const binding of module.bindings) {
    const twin = cpu.bindings.find(
      (other) => other.group === binding.group && other.binding === binding.binding,
    );
    const kinds =
      binding.kind === 'texture' ? ['texture', 'storageTexture'] : [binding.kind as string];
    if (!twin || !kinds.includes(twin.kind)) {
      return `@group(${binding.group}) @binding(${binding.binding}) differs from the CPU module's bindings`;
    }
    if (binding.writes && !twin.writes) {
      return `the CPU module does not write target @group(${binding.group}) @binding(${binding.binding})`;
    }
  }
  const passes: GpuPass[] = [];
  for (const pass of module.passes) {
    const linked = link(ctx, PASS_VERTEX, pass.fragment, `a pass of '${label}'`);
    if (!linked.ok) return linked.message;
    const { program } = linked;
    const uses = connectUses(ctx, label, module, program);
    if (typeof uses === 'string') return uses;
    passes.push({
      target: pass.target,
      program,
      uses,
      invocations: program.uniforms.get(INVOCATIONS) ?? null,
      workgroups: program.uniforms.get(WORKGROUPS) ?? null,
      layer: program.uniforms.get(LAYER) ?? null,
    });
  }
  return { module, passes };
}

const NO_BLEND: BlendState = {
  enabled: false,
  colorOp: 0,
  colorSrc: 0,
  colorDst: 0,
  alphaOp: 0,
  alphaSrc: 0,
  alphaDst: 0,
};

const pools = new WeakMap<Ctx, Map<string, WebGLTexture[]>>();
/**
 * Textures a pass may render into are kept for the next dispatch, up to this many per size, or
 * one per layer of the largest layered target of that size.
 */
const POOLED = 4;

function poolOf(ctx: Ctx): Map<string, WebGLTexture[]> {
  let pool = pools.get(ctx);
  if (!pool) {
    const made = new Map<string, WebGLTexture[]>();
    pool = made;
    pools.set(ctx, made);
    ctx.track(() => {
      for (const list of made.values()) for (const texture of list) ctx.gl.deleteTexture(texture);
      made.clear();
    });
  }
  return pool;
}

function take(ctx: Ctx, internal: number, width: number, height: number): WebGLTexture {
  const reused = poolOf(ctx).get(`${internal}:${width}:${height}`)?.pop();
  if (reused) return reused;
  const { gl, state } = ctx;
  const texture = gl.createTexture();
  state.scratchTexture(gl.TEXTURE_2D, texture);
  gl.texStorage2D(gl.TEXTURE_2D, 1, internal, width, height);
  return texture;
}

function give(
  ctx: Ctx,
  internal: number,
  width: number,
  height: number,
  texture: WebGLTexture,
  keep = POOLED,
) {
  const key = `${internal}:${width}:${height}`;
  const pool = poolOf(ctx);
  const list = pool.get(key) ?? [];
  pool.set(key, list);
  if (list.length < Math.max(keep, POOLED)) list.push(texture);
  else ctx.gl.deleteTexture(texture);
}

type Resolved =
  | { readonly kind: 'buffer'; readonly buffer: BufferImpl }
  | { readonly kind: 'texture'; readonly view: ViewImpl };

type Memory =
  | {
      readonly kind: 'buffer';
      readonly buffer: BufferImpl;
      readonly from: number;
      readonly to: number;
    }
  | { readonly kind: 'texture'; readonly view: ViewImpl };

/** The memory a binding of the dispatch holds: a buffer's byte range or a view's subresources. */
function memoryAt(
  groups: readonly (Bound | undefined)[],
  group: number,
  binding: number,
): Memory | undefined {
  const buffer = bufferEntry(groups[group], binding);
  if (buffer) {
    const from = buffer.offset;
    return { kind: 'buffer', buffer: buffer.buffer, from, to: from + buffer.size };
  }
  const entry = groups[group]?.group.entries.get(binding);
  return entry?.kind === 'texture' ? { kind: 'texture', view: entry.view } : undefined;
}

function overlaps(a: Memory, b: Memory): boolean {
  if (a.kind === 'buffer') {
    return b.kind === 'buffer' && a.buffer === b.buffer && a.from < b.to && b.from < a.to;
  }
  if (b.kind === 'buffer') return false;
  const [x, y] = [a.view, b.view];
  if (x.texture !== y.texture) return false;
  const levels =
    x.baseMipLevel < y.baseMipLevel + y.mipLevelCount &&
    y.baseMipLevel < x.baseMipLevel + x.mipLevelCount;
  const layers =
    x.texture.dimension === '3d' ||
    (x.baseArrayLayer < y.baseArrayLayer + y.arrayLayerCount &&
      y.baseArrayLayer < x.baseArrayLayer + x.arrayLayerCount);
  return levels && layers;
}

/**
 * Runs the dispatch as fragment passes. Returns why it cannot, in which case nothing ran and the
 * caller runs the CPU module.
 */
export function runGpu(
  ctx: Ctx,
  gpu: GpuCompute,
  groups: readonly (Bound | undefined)[],
  x: number,
  y: number,
  z: number,
): string | undefined {
  const { gl, state } = ctx;
  const { module } = gpu;
  const [wx, wy, wz] = module.workgroupSize;
  const invocations = [x * wx, y * wy, z * wz] as const;
  if (module.grid < 2 && invocations[1] !== 1) return 'the dispatch spans y';
  if (module.grid < 3 && invocations[2] !== 1) return 'the dispatch spans z';
  if (module.single && invocations[0] * invocations[1] * invocations[2] !== 1) {
    return 'the form needs a dispatch of one invocation';
  }
  for (const { target } of gpu.passes) {
    if (target.kind !== 'buffer') continue;
    if (target.width !== undefined && invocations[0] > target.width) {
      return `the dispatch is wider than ${target.width} invocations`;
    }
    if (target.height !== undefined && invocations[1] > target.height) {
      return `the dispatch is taller than ${target.height} invocations`;
    }
  }

  const resolved: Resolved[] = [];
  const written = new Set<BufferImpl>();
  for (const { target } of gpu.passes) {
    if (target.kind === 'buffer') {
      const entry = bufferEntry(groups[target.group], target.binding);
      if (!entry) return `no buffer at @group(${target.group}) @binding(${target.binding})`;
      // A buffer target's pass replaces the whole buffer.
      if (written.has(entry.buffer)) return 'two targets share their memory';
      written.add(entry.buffer);
      resolved.push({ kind: 'buffer', buffer: entry.buffer });
    } else {
      const entry = groups[target.group]?.group.entries.get(target.binding);
      if (entry?.kind !== 'texture') {
        return `no texture at @group(${target.group}) @binding(${target.binding})`;
      }
      const { texture } = entry.view;
      if (!texture.info.renderable || texture.sampleCount > 1) {
        return `'${texture.format}' is not color-renderable here`;
      }
      resolved.push({ kind: 'texture', view: entry.view });
    }
  }
  for (const { target } of gpu.passes) {
    const memory = memoryAt(groups, target.group, target.binding) as Memory;
    for (const other of module.bindings) {
      if (other.group === target.group && other.binding === target.binding) continue;
      const elsewhere = memoryAt(groups, other.group, other.binding);
      if (elsewhere && overlaps(memory, elsewhere)) return 'a target is bound twice';
    }
  }

  const outputs: (() => void)[] = [];
  for (const [i, pass] of gpu.passes.entries()) {
    const resource = resolved[i]!;
    state.program(pass.program.program);
    const missing = bindUses(ctx, pass.uses, pass.program, groups);
    if (missing) {
      ctx.errors.report('validation', `The dispatch is missing ${missing}.`);
      return undefined;
    }
    if (pass.invocations) gl.uniform3ui(pass.invocations, ...invocations);
    if (pass.workgroups) gl.uniform3ui(pass.workgroups, x, y, z);

    let width: number;
    let height: number;
    let layers = 1;
    let internal: number;
    if (resource.kind === 'buffer') {
      const data = dataTexture(ctx, resource.buffer);
      width = data.width;
      height = data.height;
      internal = gl.RGBA32UI;
    } else {
      const { view } = resource;
      const [w, h, depth] = mipSize(onGl(ctx, view.texture), view.baseMipLevel);
      width = w;
      height = h;
      if (view.dimension === '2d-array') layers = view.arrayLayerCount;
      if (view.dimension === '3d') layers = depth;
      internal = view.texture.info.internal;
    }

    const made: WebGLTexture[] = [];
    for (let layer = 0; layer < layers; layer++) {
      const texture = take(ctx, internal, width, height);
      state.drawFramebuffer(ctx.scratch.draw);
      gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      state.drawBuffers([gl.COLOR_ATTACHMENT0]);
      state.program(pass.program.program);
      setInt(gl, pass.layer, layer);
      state.vertexArray(null);
      state.viewport(0, 0, width, height);
      writeEverything(ctx);
      for (const cap of [
        gl.SCISSOR_TEST,
        gl.DEPTH_TEST,
        gl.STENCIL_TEST,
        gl.CULL_FACE,
        gl.RASTERIZER_DISCARD,
        gl.SAMPLE_ALPHA_TO_COVERAGE,
        gl.POLYGON_OFFSET_FILL,
      ]) {
        state.enable(cap, false);
      }
      // Through `blend`: disabling BLEND as a cap turns off every draw buffer's blending behind
      // the per-buffer cache that render passes rely on
      state.blend(0, NO_BLEND);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0);
      made.push(texture);
    }

    if (resource.kind === 'buffer') {
      const size = [internal, width, height] as const;
      outputs.push(() => give(ctx, ...size, replaceData(resource.buffer, made[0]!)));
    } else {
      const { view } = resource;
      outputs.push(() => {
        const texture = view.texture;
        made.forEach((value, layer) => {
          const source: TextureImpl = {
            ...texture,
            target: gl.TEXTURE_2D,
            texture: value,
            renderbuffer: null,
          };
          blitLevel(
            ctx,
            { texture: source, level: 0, layer: 0, x: 0, y: 0 },
            {
              texture,
              level: view.baseMipLevel,
              layer: view.dimension === '3d' ? layer : view.baseArrayLayer + layer,
              x: 0,
              y: 0,
            },
            width,
            height,
          );
          give(ctx, internal, width, height, value, made.length);
        });
        glWritten(texture);
      });
    }
  }
  for (const output of outputs) output();
  return undefined;
}
