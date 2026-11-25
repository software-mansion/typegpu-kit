/** Running a recorded render pass on GL: its framebuffer, clears, bindings and draws. */

import type { Bound } from './binding.ts';
import { type BufferImpl, indexBuffer, vertexBuffer } from './buffer.ts';
import type { Ctx } from './context.ts';
import type { RenderPipelineImpl, RenderState } from './pipeline.ts';
import { bindUses, setInt } from './uses.ts';
import {
  type Attachment,
  framebuffer,
  glWritten,
  mipSize,
  type TextureImpl,
  type ViewImpl,
  viewImpl,
  writeEverything,
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
        /** Whether the pass keeps the depth it wrote; false with `depthStoreOp: 'discard'`. */
        readonly storeDepth: boolean;
        readonly storeStencil: boolean;
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
          storeDepth: depthStencil.depthStoreOp !== 'discard',
          storeStencil: depthStencil.stencilStoreOp !== 'discard',
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
  // A draw without OES_draw_buffers_indexed may have left out the targets it did not write
  const buffers = attachments.map((attachment, i) =>
    attachment ? gl.COLOR_ATTACHMENT0 + i : gl.NONE,
  );
  state.drawBuffers(buffers.length > 0 ? buffers : [gl.NONE]);
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
      state.readBuffer(gl.COLOR_ATTACHMENT0 + i);
      state.drawFramebuffer(resolve);
      writeEverything(ctx);
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
  const depth = targets.depth;
  if (depth) {
    const { info } = depth.attachment.texture;
    const discardDepth = info.hasDepth && !depth.storeDepth;
    const discardStencil = info.hasStencil && !depth.storeStencil;
    if ((info.hasDepth && !discardDepth) || (info.hasStencil && !discardStencil)) {
      written(ctx, depth.attachment.texture);
    }
    // Each aspect is discarded alone: invalidating a combined attachment loses both
    if (discardDepth && discardStencil) discards.push(gl.DEPTH_STENCIL_ATTACHMENT);
    else if (discardDepth) discards.push(gl.DEPTH_ATTACHMENT);
    else if (discardStencil) discards.push(gl.STENCIL_ATTACHMENT);
  }
  if (discards.length > 0) {
    state.drawFramebuffer(pass.fbo);
    gl.invalidateFramebuffer(gl.DRAW_FRAMEBUFFER, discards);
  }
}

/** Binds the pipeline's buffers and textures to their units. False when one is missing. */
function bindResources(pass: RenderPass, pipeline: RenderPipelineImpl): boolean {
  const missing = bindUses(pass.ctx, pipeline, pipeline.program!, pass.groups);
  if (missing === undefined) return true;
  pass.ctx.errors.report(
    'validation',
    `The draw with pipeline '${pipeline.label}' is missing ${missing}.`,
  );
  return false;
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

  const colors = pass.targets.colors;
  if (pass.ctx.ext.drawBuffersIndexed) {
    for (let i = 0; i < colors.length; i++) {
      const target = s.targets[i];
      if (target) {
        state.blend(i, target.blend);
        state.colorMask(i, target.mask);
      } else {
        state.colorMask(i, 0);
      }
    }
  } else {
    targetGroup(pass, s, 0);
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
 * Without `OES_draw_buffers_indexed`, points the draw buffers at the targets of one of the
 * pipeline's groups and sets their shared blend state and mask.
 */
function targetGroup(pass: RenderPass, s: RenderState, group: number): void {
  const { gl, state } = pass.ctx;
  const indices = s.shared[group];
  const buffers = pass.targets.colors.map((attached, i) =>
    attached && indices?.includes(i) ? gl.COLOR_ATTACHMENT0 + i : gl.NONE,
  );
  const shared = indices && s.targets[indices[0]!];
  if (shared) state.blend(0, shared.blend);
  state.colorMaskAll(shared?.mask ?? 0);
  state.drawBuffers(buffers.length > 0 ? buffers : [gl.NONE]);
}

/**
 * Issues a draw, once for each group of targets that blend and mask alike when the context has
 * one blend state for every draw buffer. Each later draw repeats the first one's fragments: it
 * writes no depth or stencil, and a pipeline that wrote depth keeps the fragments that won.
 */
function issue(pass: RenderPass, pipeline: RenderPipelineImpl, call: () => void): void {
  const s = pipeline.state;
  if (!s.writes) return;
  call();
  if (pass.ctx.ext.drawBuffersIndexed || s.shared.length < 2) return;
  const { gl, state } = pass.ctx;
  state.depthMask(false);
  if (s.depthWrite && s.depthTest) state.depthFunc(gl.EQUAL);
  for (let group = 1; group < s.shared.length; group++) {
    targetGroup(pass, s, group);
    call();
  }
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
      ctx.errors.report(
        'validation',
        `The draw with pipeline '${pipeline.label}' is missing vertex buffer ${slot.slot}.`,
      );
      return undefined;
    }
    const buffer = vertexBuffer(ctx, bound.buffer);
    const base = bound.offset + (slot.instance ? instanceOffset : vertexOffset) * slot.stride;
    if (slot.buffer === buffer && slot.base === base) continue;
    const first = Math.min(...slot.attributes.map((attribute) => attribute.offset));
    if (base + first < 0) {
      // baseVertex moves the attribute pointers, which cannot start before the buffer
      ctx.errors.report(
        'validation',
        `The draw with pipeline '${pipeline.label}' has baseVertex ${vertexOffset}, which starts vertex buffer ${slot.slot} ${-(base + first)} bytes before its bound range; WebGL 2 cannot draw it.`,
      );
      return undefined;
    }
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

function counted(pass: RenderPass, instances: number): void {
  const { draws } = pass.ctx.stats;
  draws.count++;
  draws.instances += instances;
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
  counted(pass, instanceCount);
  issue(pass, pipeline, () =>
    pass.ctx.gl.drawArraysInstanced(pipeline.state.mode, firstVertex, vertexCount, instanceCount),
  );
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
  counted(pass, instanceCount);
  issue(pass, pipeline, () =>
    gl.drawElementsInstanced(
      pipeline.state.mode,
      indexCount,
      wide ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT,
      index.offset + firstIndex * (wide ? 4 : 2),
      instanceCount,
    ),
  );
}

export function drawIndirect(pass: RenderPass, buffer: BufferImpl, offset: number): void {
  const at = offset >>> 2;
  buffer.sync();
  const words = buffer.words;
  draw(pass, words[at]!, words[at + 1]!, words[at + 2]!, words[at + 3]!);
}

export function drawIndexedIndirect(pass: RenderPass, buffer: BufferImpl, offset: number): void {
  const at = offset >>> 2;
  buffer.sync();
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
