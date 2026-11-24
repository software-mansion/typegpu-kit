/** Running a recorded render pass on GL: its framebuffer, clears, bindings and draws. */

import { type Bound, bufferEntry } from './binding.ts';
import { type BufferImpl, dataTexture, indexBuffer, vertexBuffer } from './buffer.ts';
import type { Ctx } from './context.ts';
import type { RenderPipelineImpl } from './pipeline.ts';
import {
  type Attachment,
  framebuffer,
  glWritten,
  mipSize,
  sampled,
  type TextureImpl,
  type ViewImpl,
  viewImpl,
} from './texture.ts';

export interface ColorTarget {
  readonly attachment: Attachment;
  readonly resolve: Attachment | undefined;
  readonly clear: readonly [number, number, number, number] | undefined;
  readonly store: boolean;
}

/** A render pass descriptor resolved when it is recorded, since callers reuse descriptors. */
export interface PassTargets {
  readonly colors: readonly (ColorTarget | null)[];
  readonly depth:
    | {
        readonly attachment: Attachment;
        readonly clearDepth: number | undefined;
        readonly clearStencil: number | undefined;
        readonly depthReadOnly: boolean;
        readonly stencilReadOnly: boolean;
        readonly store: boolean;
      }
    | undefined;
}

export interface RenderPass {
  readonly ctx: Ctx;
  readonly targets: PassTargets;
  readonly fbo: WebGLFramebuffer;
  readonly width: number;
  readonly height: number;
  pipeline: RenderPipelineImpl | undefined;
  groups: (Bound | undefined)[];
  vertex: ({ buffer: BufferImpl; offset: number } | undefined)[];
  index: { buffer: BufferImpl; format: GPUIndexFormat; offset: number } | undefined;
  viewport: [number, number, number, number, number, number];
  scissor: [number, number, number, number];
  stencilReference: number;
  blendConstant: [number, number, number, number];
  /** The pipeline whose bindings are bound now; reset by any binding change. */
  bound: RenderPipelineImpl | undefined;
}

function color(value: GPUColor | undefined): [number, number, number, number] {
  if (!value) return [0, 0, 0, 0];
  if (Symbol.iterator in value) {
    const [r = 0, g = 0, b = 0, a = 0] = Array.from(value as Iterable<number>);
    return [r, g, b, a];
  }
  return [value.r, value.g, value.b, value.a];
}

function attachmentOf(view: GPUTextureView | GPUTexture, depthSlice?: number): Attachment {
  const impl: ViewImpl =
    'createView' in view ? viewImpl((view as GPUTexture).createView()) : viewImpl(view);
  const texture = impl.texture;
  return {
    texture,
    level: impl.baseMipLevel,
    layer: texture.dimension === '3d' ? (depthSlice ?? 0) : impl.baseArrayLayer,
  };
}

export function resolveTargets(descriptor: GPURenderPassDescriptor): PassTargets {
  const colors = Array.from(descriptor.colorAttachments, (attachment): ColorTarget | null => {
    if (!attachment) return null;
    return {
      attachment: attachmentOf(attachment.view, attachment.depthSlice),
      resolve: attachment.resolveTarget ? attachmentOf(attachment.resolveTarget) : undefined,
      clear: attachment.loadOp === 'clear' ? color(attachment.clearValue) : undefined,
      store: attachment.storeOp !== 'discard',
    };
  });
  const depthStencil = descriptor.depthStencilAttachment;
  return {
    colors,
    depth: depthStencil
      ? {
          attachment: attachmentOf(depthStencil.view),
          clearDepth:
            depthStencil.depthLoadOp === 'clear' ? (depthStencil.depthClearValue ?? 0) : undefined,
          clearStencil:
            depthStencil.stencilLoadOp === 'clear'
              ? (depthStencil.stencilClearValue ?? 0)
              : undefined,
          depthReadOnly: depthStencil.depthReadOnly ?? false,
          stencilReadOnly: depthStencil.stencilReadOnly ?? false,
          store: depthStencil.depthStoreOp !== 'discard',
        }
      : undefined,
  };
}

export function beginRender(ctx: Ctx, targets: PassTargets): RenderPass {
  const { gl, state } = ctx;
  const attachments = targets.colors.map((target) => target?.attachment ?? null);
  const fbo = framebuffer(ctx, attachments, targets.depth?.attachment ?? null);
  const first = attachments.find((a) => a !== null) ?? targets.depth?.attachment;
  const [width, height] = first ? mipSize(first.texture, first.level) : [1, 1];

  state.drawFramebuffer(fbo);
  state.enable(gl.SCISSOR_TEST, false);
  state.enable(gl.RASTERIZER_DISCARD, false);
  targets.colors.forEach((target, i) => {
    if (!target?.clear) return;
    state.colorMask(i, 0xf);
    const kind = target.attachment.texture.info.kind;
    if (kind === 'uint') gl.clearBufferuiv(gl.COLOR, i, new Uint32Array(target.clear));
    else if (kind === 'sint') gl.clearBufferiv(gl.COLOR, i, new Int32Array(target.clear));
    else gl.clearBufferfv(gl.COLOR, i, target.clear);
  });
  const depth = targets.depth;
  if (depth) {
    const { info } = depth.attachment.texture;
    const clearDepth = info.hasDepth ? depth.clearDepth : undefined;
    const clearStencil = info.hasStencil ? depth.clearStencil : undefined;
    if (clearDepth !== undefined) state.depthMask(true);
    if (clearStencil !== undefined) state.stencilWriteMask(0xff);
    if (clearDepth !== undefined && clearStencil !== undefined) {
      gl.clearBufferfi(gl.DEPTH_STENCIL, 0, clearDepth, clearStencil);
    } else if (clearDepth !== undefined) {
      gl.clearBufferfv(gl.DEPTH, 0, [clearDepth]);
    } else if (clearStencil !== undefined) {
      gl.clearBufferiv(gl.STENCIL, 0, [clearStencil]);
    }
  }

  return {
    ctx,
    targets,
    fbo,
    width,
    height,
    pipeline: undefined,
    groups: [],
    vertex: [],
    index: undefined,
    viewport: [0, 0, width, height, 0, 1],
    scissor: [0, 0, width, height],
    stencilReference: 0,
    blendConstant: [0, 0, 0, 0],
    bound: undefined,
  };
}

function written(ctx: Ctx, texture: TextureImpl): void {
  glWritten(texture);
  if (texture.canvas) ctx.presented(texture.canvas);
}

export function endRender(pass: RenderPass): void {
  const { ctx, targets } = pass;
  const { gl, state } = ctx;
  const discards: number[] = [];
  targets.colors.forEach((target, i) => {
    if (!target) return;
    if (target.resolve) {
      const resolve = framebuffer(ctx, [target.resolve], null);
      state.readFramebuffer(pass.fbo);
      gl.readBuffer(gl.COLOR_ATTACHMENT0 + i);
      state.drawFramebuffer(resolve);
      state.enable(gl.SCISSOR_TEST, false);
      gl.blitFramebuffer(
        0,
        0,
        pass.width,
        pass.height,
        0,
        0,
        pass.width,
        pass.height,
        gl.COLOR_BUFFER_BIT,
        gl.NEAREST,
      );
      written(ctx, target.resolve.texture);
    }
    if (target.store) written(ctx, target.attachment.texture);
    else discards.push(gl.COLOR_ATTACHMENT0 + i);
  });
  if (targets.depth) {
    if (targets.depth.store) written(ctx, targets.depth.attachment.texture);
    else {
      const { info } = targets.depth.attachment.texture;
      discards.push(info.hasStencil ? gl.DEPTH_STENCIL_ATTACHMENT : gl.DEPTH_ATTACHMENT);
    }
  }
  if (discards.length > 0) {
    state.drawFramebuffer(pass.fbo);
    gl.invalidateFramebuffer(gl.DRAW_FRAMEBUFFER, discards);
  }
}

const ranges = new WeakMap<WebGLUniformLocation, [number, number]>();
const ints = new WeakMap<WebGLUniformLocation, number>();

function setInt(gl: WebGL2RenderingContext, location: WebGLUniformLocation | null, value: number) {
  if (!location || ints.get(location) === value) return;
  ints.set(location, value);
  gl.uniform1i(location, value);
}

function missing(pass: RenderPass, pipeline: RenderPipelineImpl, what: string): false {
  pass.ctx.errors.report(
    'validation',
    `The draw with pipeline '${pipeline.label}' is missing ${what}.`,
  );
  return false;
}

/** Binds the pipeline's buffers and textures to their units. False when one is missing. */
function bindResources(pass: RenderPass, pipeline: RenderPipelineImpl): boolean {
  const { ctx } = pass;
  const { gl, state } = ctx;
  for (const use of pipeline.buffers) {
    const entry = bufferEntry(pass.groups[use.group], use.binding);
    if (!entry)
      return missing(pass, pipeline, `buffer @group(${use.group}) @binding(${use.binding})`);
    const data = dataTexture(ctx, entry.buffer);
    state.texture(use.unit, gl.TEXTURE_2D, data.texture);
    state.sampler(use.unit, ctx.nearest);
    if (use.range) {
      const words = entry.offset >>> 2;
      const last = ranges.get(use.range);
      if (!last || last[0] !== words || last[1] !== entry.size) {
        state.program(pipeline.program!.program);
        gl.uniform2ui(use.range, words, entry.size);
        ranges.set(use.range, [words, entry.size]);
      }
    }
  }
  for (const use of pipeline.textures) {
    const entry = pass.groups[use.group]?.group.entries.get(use.binding);
    if (entry?.kind !== 'texture') {
      return missing(pass, pipeline, `texture @group(${use.group}) @binding(${use.binding})`);
    }
    const { target, texture } = sampled(ctx, entry.view);
    state.texture(use.unit, target, texture);
    let sampler = ctx.nearest;
    if (use.sampler) {
      const samplerEntry = pass.groups[use.sampler.group]?.group.entries.get(use.sampler.binding);
      if (samplerEntry?.kind !== 'sampler') {
        return missing(
          pass,
          pipeline,
          `sampler @group(${use.sampler.group}) @binding(${use.sampler.binding})`,
        );
      }
      const impl = samplerEntry.sampler;
      const filterable = entry.view.texture.info.filterable;
      sampler = use.comparison || (filterable && !use.depth) ? impl.sampler : impl.unfiltered();
    }
    state.sampler(use.unit, sampler);
  }
  for (const use of pipeline.levels) {
    const entry = pass.groups[use.group]?.group.entries.get(use.binding);
    if (entry?.kind !== 'texture') continue;
    state.program(pipeline.program!.program);
    setInt(gl, use.location, entry.view.mipLevelCount);
  }
  return true;
}

function applyState(pass: RenderPass, pipeline: RenderPipelineImpl): void {
  const { gl, state } = pass.ctx;
  const s = pipeline.state;
  state.enable(gl.CULL_FACE, s.cull !== 0);
  if (s.cull !== 0) state.cullFace(s.cull);
  state.frontFace(s.frontFace);

  const depth = pass.targets.depth;
  const info = depth?.attachment.texture.info;
  state.enable(gl.DEPTH_TEST, s.depthTest && info?.hasDepth === true);
  if (s.depthTest) {
    state.depthFunc(s.depthFunc);
    state.depthMask(s.depthWrite && !depth?.depthReadOnly);
  }
  const stencil = s.stencilTest && info?.hasStencil === true;
  state.enable(gl.STENCIL_TEST, stencil);
  if (stencil) {
    state.stencil(s.stencilFront, s.stencilBack, pass.stencilReference, s.stencilRead);
    state.stencilWriteMask(depth?.stencilReadOnly ? 0 : s.stencilWrite);
  }
  state.enable(gl.POLYGON_OFFSET_FILL, s.bias !== undefined);
  if (s.bias) state.polygonOffset(s.bias.slope, s.bias.constant);

  const count = pass.targets.colors.length;
  for (let i = 0; i < count; i++) {
    const target = s.targets[i];
    if (target) {
      state.blend(i, target.blend);
      state.colorMask(i, target.mask);
    } else {
      state.colorMask(i, 0);
    }
  }
  state.enable(gl.SAMPLE_ALPHA_TO_COVERAGE, s.alphaToCoverage);
  state.enable(gl.RASTERIZER_DISCARD, false);
  state.enable(gl.SCISSOR_TEST, true);
  const [sx, sy, sw, sh] = pass.scissor;
  state.scissor(sx, sy, sw, sh);
  const [x, y, w, h, near, far] = pass.viewport;
  state.viewport(Math.round(x), Math.round(y), Math.round(w), Math.round(h));
  state.depthRange(near, far);
  const [r, g, b, a] = pass.blendConstant;
  state.blendColor(r, g, b, a);
}

/**
 * Sets everything a draw reads. `vertexOffset` and `instanceOffset` count elements the vertex
 * and instance attributes skip, as WebGL 2 draws cannot.
 */
function prepare(
  pass: RenderPass,
  vertexOffset: number,
  instanceOffset: number,
): RenderPipelineImpl | undefined {
  const { ctx } = pass;
  const { gl, state } = ctx;
  const pipeline = pass.pipeline;
  if (!pipeline?.program) return undefined;
  if (pass.bound !== pipeline) {
    if (!bindResources(pass, pipeline)) return undefined;
    pass.bound = pipeline;
  }
  state.drawFramebuffer(pass.fbo);
  state.program(pipeline.program.program);
  state.vertexArray(pipeline.vao);
  for (const slot of pipeline.slots) {
    const bound = pass.vertex[slot.slot];
    if (!bound) {
      missing(pass, pipeline, `vertex buffer ${slot.slot}`);
      return undefined;
    }
    const buffer = vertexBuffer(ctx, bound.buffer);
    const base = bound.offset + (slot.instance ? instanceOffset : vertexOffset) * slot.stride;
    if (slot.buffer === buffer && slot.base === base) continue;
    for (const attribute of slot.attributes) {
      if (attribute.integer) {
        gl.vertexAttribIPointer(
          attribute.location,
          attribute.size,
          attribute.type,
          slot.stride,
          base + attribute.offset,
        );
      } else {
        gl.vertexAttribPointer(
          attribute.location,
          attribute.size,
          attribute.type,
          attribute.normalized,
          slot.stride,
          base + attribute.offset,
        );
      }
    }
    slot.buffer = buffer;
    slot.base = base;
  }
  setInt(gl, pipeline.firstInstance, instanceOffset);
  applyState(pass, pipeline);
  return pipeline;
}

export function draw(
  pass: RenderPass,
  vertexCount: number,
  instanceCount = 1,
  firstVertex = 0,
  firstInstance = 0,
): void {
  if (vertexCount === 0 || instanceCount === 0) return;
  const pipeline = prepare(pass, 0, firstInstance);
  if (!pipeline) return;
  setInt(pass.ctx.gl, pipeline.baseVertex, 0);
  pass.ctx.gl.drawArraysInstanced(pipeline.state.mode, firstVertex, vertexCount, instanceCount);
}

export function drawIndexed(
  pass: RenderPass,
  indexCount: number,
  instanceCount = 1,
  firstIndex = 0,
  baseVertex = 0,
  firstInstance = 0,
): void {
  if (indexCount === 0 || instanceCount === 0) return;
  const index = pass.index;
  if (!index) {
    pass.ctx.errors.report('validation', 'drawIndexed without an index buffer.');
    return;
  }
  const pipeline = prepare(pass, baseVertex, firstInstance);
  if (!pipeline) return;
  const { gl } = pass.ctx;
  setInt(gl, pipeline.baseVertex, baseVertex);
  indexBuffer(pass.ctx, index.buffer);
  const wide = index.format === 'uint32';
  gl.drawElementsInstanced(
    pipeline.state.mode,
    indexCount,
    wide ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT,
    index.offset + firstIndex * (wide ? 4 : 2),
    instanceCount,
  );
}

export function drawIndirect(pass: RenderPass, buffer: BufferImpl, offset: number): void {
  const at = offset >>> 2;
  const words = buffer.words;
  draw(pass, words[at]!, words[at + 1]!, words[at + 2]!, words[at + 3]!);
}

export function drawIndexedIndirect(pass: RenderPass, buffer: BufferImpl, offset: number): void {
  const at = offset >>> 2;
  const words = buffer.words;
  drawIndexed(pass, words[at]!, words[at + 1]!, words[at + 2]!, words[at + 3]! | 0, words[at + 4]!);
}

export function setBindGroup(pass: RenderPass, index: number, bound: Bound | undefined): void {
  pass.groups[index] = bound;
  pass.bound = undefined;
}

export function setPipeline(pass: RenderPass, pipeline: RenderPipelineImpl): void {
  if (pass.pipeline !== pipeline) pass.bound = undefined;
  pass.pipeline = pipeline;
}

/** Forgets what bundles set, as a pass does after `executeBundles`. */
export function resetState(pass: RenderPass): void {
  pass.pipeline = undefined;
  pass.groups = [];
  pass.vertex = [];
  pass.index = undefined;
  pass.bound = undefined;
}
