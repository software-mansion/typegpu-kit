/**
 * Command encoders, passes and render bundles. Encoding records closures; `queue.submit` runs
 * them in order, so writes made on the queue before the submit apply first, as in WebGPU.
 */

import { bind } from './binding.ts';
import { type BufferImpl, bufferImpl, copyLater, written } from './buffer.ts';
import { type ComputePass, dispatch, dispatchIndirect, workgroupCount } from './compute.ts';
import { BUFFER_USAGE, type Ctx, IMPL } from './context.ts';
import { computePipelineImpl, renderPipelineImpl } from './pipeline.ts';
import {
  beginRender,
  draw,
  drawIndexed,
  drawIndexedIndirect,
  drawIndirect,
  endRender,
  type RenderPass,
  resetState,
  resolveTargets,
  setBindGroup,
  setPipeline,
} from './render.ts';
import { textureImpl } from './texture.ts';
import {
  copyBufferToTexture,
  copyTextureToBuffer,
  copyTextureToBufferLater,
  copyTextureToTexture,
} from './transfer.ts';

export interface Exec {
  render: RenderPass | undefined;
  compute: ComputePass | undefined;
}

export type Command = (exec: Exec) => void;
type RenderCommand = (pass: RenderPass) => void;

export interface CommandBufferImpl {
  readonly commands: readonly Command[];
  readonly label: string;
}

export function commandBufferImpl(buffer: GPUCommandBuffer): CommandBufferImpl {
  return (buffer as unknown as { [IMPL]: CommandBufferImpl })[IMPL];
}

const noop = () => {};

function layoutOf(info: GPUTexelCopyBufferInfo): GPUTexelCopyBufferLayout {
  const { offset, bytesPerRow, rowsPerImage } = info;
  return {
    ...(offset !== undefined && { offset }),
    ...(bytesPerRow !== undefined && { bytesPerRow }),
    ...(rowsPerImage !== undefined && { rowsPerImage }),
  };
}

/** The recording methods render passes and render bundles share. */
function renderRecorder(record: (command: RenderCommand) => void) {
  return {
    setPipeline(pipeline: GPURenderPipeline) {
      const impl = renderPipelineImpl(pipeline);
      record((pass) => setPipeline(pass, impl));
    },
    setBindGroup(
      index: number,
      group: GPUBindGroup | null,
      offsets?: Iterable<number> | Uint32Array,
      start?: number,
      length?: number,
    ) {
      const bound = group ? bind(group, offsets, start, length) : undefined;
      record((pass) => setBindGroup(pass, index, bound));
    },
    setVertexBuffer(slot: number, buffer: GPUBuffer | null, offset = 0) {
      const impl = buffer ? bufferImpl(buffer) : undefined;
      record((pass) => {
        pass.vertex[slot] = impl ? { buffer: impl, offset } : undefined;
      });
    },
    setIndexBuffer(buffer: GPUBuffer, format: GPUIndexFormat, offset = 0) {
      const impl = bufferImpl(buffer);
      record((pass) => {
        pass.index = { buffer: impl, format, offset };
      });
    },
    draw(
      vertexCount: number,
      instanceCount?: number,
      firstVertex?: number,
      firstInstance?: number,
    ) {
      record((pass) => draw(pass, vertexCount, instanceCount, firstVertex, firstInstance));
    },
    drawIndexed(
      indexCount: number,
      instanceCount?: number,
      firstIndex?: number,
      baseVertex?: number,
      firstInstance?: number,
    ) {
      record((pass) =>
        drawIndexed(pass, indexCount, instanceCount, firstIndex, baseVertex, firstInstance),
      );
    },
    drawIndirect(buffer: GPUBuffer, offset: number) {
      const impl = bufferImpl(buffer);
      record((pass) => drawIndirect(pass, impl, offset));
    },
    drawIndexedIndirect(buffer: GPUBuffer, offset: number) {
      const impl = bufferImpl(buffer);
      record((pass) => drawIndexedIndirect(pass, impl, offset));
    },
    pushDebugGroup: noop,
    popDebugGroup: noop,
    insertDebugMarker: noop,
  };
}

interface BundleImpl {
  readonly commands: readonly RenderCommand[];
}

export function createRenderBundleEncoder(
  descriptor: GPURenderBundleEncoderDescriptor,
): GPURenderBundleEncoder {
  const commands: RenderCommand[] = [];
  return {
    __brand: 'GPURenderBundleEncoder',
    label: descriptor.label ?? '',
    ...renderRecorder((command) => commands.push(command)),
    finish(finished?: GPURenderBundleDescriptor) {
      return {
        [IMPL]: { commands } satisfies BundleImpl,
        __brand: 'GPURenderBundle',
        label: finished?.label ?? descriptor.label ?? '',
      } as unknown as GPURenderBundle;
    },
  } as unknown as GPURenderBundleEncoder;
}

export function createCommandEncoder(
  ctx: Ctx,
  descriptor: GPUCommandEncoderDescriptor = {},
): GPUCommandEncoder {
  const commands: Command[] = [];
  let open: 'render' | 'compute' | undefined;
  const record = (command: Command) => commands.push(command);
  const fail = (message: string) => record(() => ctx.errors.report('validation', message));

  const presentIfCanvas = (texture: GPUTexture) => {
    const impl = textureImpl(texture);
    if (impl.canvas) ctx.presented(impl.canvas);
  };

  const encoder = {
    __brand: 'GPUCommandEncoder',
    label: descriptor.label ?? '',

    beginRenderPass(passDescriptor: GPURenderPassDescriptor): GPURenderPassEncoder {
      if (open) fail(`beginRenderPass while a ${open} pass is open.`);
      open = 'render';
      const targets = resolveTargets(passDescriptor);
      record((exec) => {
        exec.render = beginRender(ctx, targets);
      });
      const inPass = (command: RenderCommand) =>
        record((exec) => {
          if (exec.render) command(exec.render);
        });
      return {
        __brand: 'GPURenderPassEncoder',
        label: passDescriptor.label ?? '',
        ...renderRecorder(inPass),
        setViewport(
          x: number,
          y: number,
          width: number,
          height: number,
          minDepth: number,
          maxDepth: number,
        ) {
          inPass((pass) => {
            pass.viewport = [x, y, width, height, minDepth, maxDepth];
          });
        },
        setScissorRect(x: number, y: number, width: number, height: number) {
          inPass((pass) => {
            pass.scissor = [x, y, width, height];
          });
        },
        setBlendConstant(value: GPUColor) {
          const color =
            Symbol.iterator in value
              ? (Array.from(value as Iterable<number>) as [number, number, number, number])
              : ([value.r, value.g, value.b, value.a] as [number, number, number, number]);
          inPass((pass) => {
            pass.blendConstant = color;
          });
        },
        setStencilReference(reference: number) {
          inPass((pass) => {
            pass.stencilReference = reference;
          });
        },
        beginOcclusionQuery: noop,
        endOcclusionQuery: noop,
        executeBundles(bundles: Iterable<GPURenderBundle>) {
          const list = Array.from(
            bundles,
            (bundle) => (bundle as unknown as { [IMPL]: BundleImpl })[IMPL],
          );
          inPass((pass) => {
            resetState(pass);
            for (const bundle of list) {
              for (const command of bundle.commands) command(pass);
              resetState(pass);
            }
          });
        },
        end() {
          open = undefined;
          record((exec) => {
            if (exec.render) endRender(exec.render);
            exec.render = undefined;
          });
        },
      } as unknown as GPURenderPassEncoder;
    },

    beginComputePass(passDescriptor: GPUComputePassDescriptor = {}): GPUComputePassEncoder {
      if (open) fail(`beginComputePass while a ${open} pass is open.`);
      open = 'compute';
      record((exec) => {
        exec.compute = { pipeline: undefined, groups: [] };
      });
      const inPass = (command: (pass: ComputePass) => void) =>
        record((exec) => {
          if (exec.compute) command(exec.compute);
        });
      return {
        __brand: 'GPUComputePassEncoder',
        label: passDescriptor.label ?? '',
        setPipeline(pipeline: GPUComputePipeline) {
          const impl = computePipelineImpl(pipeline);
          inPass((pass) => {
            pass.pipeline = impl;
          });
        },
        setBindGroup(
          index: number,
          group: GPUBindGroup | null,
          offsets?: Iterable<number> | Uint32Array,
          start?: number,
          length?: number,
        ) {
          const bound = group ? bind(group, offsets, start, length) : undefined;
          inPass((pass) => {
            pass.groups[index] = bound;
          });
        },
        dispatchWorkgroups(x: number, y = 1, z = 1) {
          const [cx, cy, cz] = [x, y, z].map(workgroupCount) as [number, number, number];
          inPass((pass) => dispatch(ctx, pass, cx, cy, cz));
        },
        dispatchWorkgroupsIndirect(buffer: GPUBuffer, offset: number) {
          const impl = bufferImpl(buffer);
          inPass((pass) => dispatchIndirect(ctx, pass, impl, offset));
        },
        pushDebugGroup: noop,
        popDebugGroup: noop,
        insertDebugMarker: noop,
        end() {
          open = undefined;
          record((exec) => {
            exec.compute = undefined;
          });
        },
      } as unknown as GPUComputePassEncoder;
    },

    copyBufferToBuffer(
      ...args: [GPUBuffer, GPUBuffer, number?] | [GPUBuffer, number, GPUBuffer, number, number?]
    ) {
      const [from, sourceOffset, to, destinationOffset, given] = (
        typeof args[1] === 'number' ? args : [args[0], 0, args[1], 0, args[2]]
      ) as [GPUBuffer, number, GPUBuffer, number, number?];
      const source = bufferImpl(from);
      const destination = bufferImpl(to);
      const size = given ?? source.size - sourceOffset;
      record(() => {
        // Into a buffer only the CPU reads, a copy of what GL alone holds need not wait for it
        const later =
          destination.usage & BUFFER_USAGE.MAP_READ &&
          source.gpuNewer &&
          source.data &&
          !source.destroyed &&
          !destination.destroyed;
        if (later) {
          copyLater(ctx, source, sourceOffset, destination, destinationOffset, size);
          return;
        }
        source.sync();
        destination.sync();
        destination.bytes.set(
          source.bytes.subarray(sourceOffset, sourceOffset + size),
          destinationOffset,
        );
        written(destination, destinationOffset, destinationOffset + size);
      });
    },

    copyBufferToTexture(
      source: GPUTexelCopyBufferInfo,
      destination: GPUTexelCopyTextureInfo,
      size: GPUExtent3DStrict,
    ) {
      const buffer = bufferImpl(source.buffer);
      const layout = layoutOf(source);
      const target = { ...destination };
      record(() => {
        copyBufferToTexture(ctx, buffer, layout, target, size);
        presentIfCanvas(target.texture);
      });
    },

    copyTextureToBuffer(
      source: GPUTexelCopyTextureInfo,
      destination: GPUTexelCopyBufferInfo,
      size: GPUExtent3DStrict,
    ) {
      const buffer: BufferImpl = bufferImpl(destination.buffer);
      const layout = layoutOf(destination);
      const from = { ...source };
      record(() => {
        const later =
          buffer.usage & BUFFER_USAGE.MAP_READ &&
          copyTextureToBufferLater(ctx, from, buffer, layout, size);
        if (!later) copyTextureToBuffer(ctx, from, buffer, layout, size);
      });
    },

    copyTextureToTexture(
      source: GPUTexelCopyTextureInfo,
      destination: GPUTexelCopyTextureInfo,
      size: GPUExtent3DStrict,
    ) {
      const from = { ...source };
      const to = { ...destination };
      record(() => {
        copyTextureToTexture(ctx, from, to, size);
        presentIfCanvas(to.texture);
      });
    },

    clearBuffer(buffer: GPUBuffer, offset = 0, size?: number) {
      const impl = bufferImpl(buffer);
      const length = size ?? impl.size - offset;
      record(() => {
        impl.sync();
        impl.bytes.fill(0, offset, offset + length);
        written(impl, offset, offset + length);
      });
    },

    resolveQuerySet(
      _set: GPUQuerySet,
      first: number,
      count: number,
      destination: GPUBuffer,
      offset: number,
    ) {
      const impl = bufferImpl(destination);
      record(() => {
        impl.sync();
        impl.bytes.fill(0, offset, offset + count * 8);
        written(impl, offset, offset + count * 8);
        void first;
      });
    },

    writeTimestamp: noop,
    pushDebugGroup: noop,
    popDebugGroup: noop,
    insertDebugMarker: noop,

    finish(finished?: GPUCommandBufferDescriptor): GPUCommandBuffer {
      if (open) fail(`The command encoder was finished while a ${open} pass was open.`);
      return {
        [IMPL]: {
          commands,
          label: finished?.label ?? descriptor.label ?? '',
        } satisfies CommandBufferImpl,
        __brand: 'GPUCommandBuffer',
        label: finished?.label ?? '',
      } as unknown as GPUCommandBuffer;
    },
  };
  return encoder as unknown as GPUCommandEncoder;
}

/** Runs a command buffer's commands. */
export function execute(ctx: Ctx, buffer: CommandBufferImpl): void {
  const exec: Exec = { render: undefined, compute: undefined };
  for (const command of buffer.commands) {
    try {
      command(exec);
    } catch (error) {
      ctx.errors.report('validation', (error as Error).message);
    }
  }
}
