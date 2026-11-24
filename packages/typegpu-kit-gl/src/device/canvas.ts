/**
 * Canvas contexts. Each holds an ordinary device texture sized to its canvas; after a submit
 * renders into it, it is presented once per task with a flipped blit: straight to the GL
 * context's own canvas, or through it to another canvas as a bitmap.
 */

import { type Ctx, TEXTURE_USAGE } from './context.ts';
import { attachLevel, textureImpl, writeEverything } from './texture.ts';

export interface Presenter {
  configure(
    canvas: HTMLCanvasElement | OffscreenCanvas,
    configuration: Omit<GPUCanvasConfiguration, 'device'>,
    device: GPUDevice,
  ): GPUCanvasContext;
  /** Queues a context for presentation at the end of the task. */
  presented(record: object): void;
  destroy(): void;
}

interface Record {
  readonly canvas: HTMLCanvasElement | OffscreenCanvas;
  configuration: (GPUCanvasConfiguration & { device: GPUDevice }) | undefined;
  texture: GPUTexture | undefined;
  /** How frames reach a canvas other than the GL context's own. */
  sink:
    | ImageBitmapRenderingContext
    | OffscreenCanvasRenderingContext2D
    | CanvasRenderingContext2D
    | undefined;
}

export function createPresenter(
  ctx: Ctx,
  createTexture: (descriptor: GPUTextureDescriptor) => GPUTexture,
): Presenter {
  const { gl, state } = ctx;
  const own = ctx.canvas;
  const pending = new Set<Record>();
  const configured = new Set<Record>();
  let scheduled = false;

  const sinkOf = (record: Record) => {
    if (record.sink) return record.sink;
    const canvas = record.canvas as HTMLCanvasElement;
    // A bitmap moves without a copy when the GL canvas is offscreen
    record.sink =
      (own instanceof OffscreenCanvas
        ? (canvas.getContext('bitmaprenderer') as ImageBitmapRenderingContext | null)
        : null) ??
      (canvas.getContext('2d') as CanvasRenderingContext2D | null) ??
      undefined;
    if (!record.sink) {
      throw new Error('@typegpu/kit-gl: the canvas already has a context of another kind.');
    }
    return record.sink;
  };

  const blit = (record: Record, width: number, height: number) => {
    const texture = textureImpl(record.texture!);
    state.readFramebuffer(ctx.scratch.read);
    attachLevel(ctx, gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, texture, 0, 0);
    state.readBuffer(gl.COLOR_ATTACHMENT0);
    state.drawFramebuffer(null);
    writeEverything(ctx);
    // A fresh drawing buffer is cleared lazily by the next clear or draw, not by a blit, so clear
    // it now rather than after the blit
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    // Texture rows run top to bottom and the default framebuffer's bottom to top
    gl.blitFramebuffer(0, 0, width, height, 0, height, width, 0, gl.COLOR_BUFFER_BIT, gl.NEAREST);
    gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0);
    if ((record.configuration?.alphaMode ?? 'opaque') === 'opaque') {
      state.colorMaskAll(0x8);
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
  };

  const present = (record: Record) => {
    const texture = record.texture;
    if (!texture || !record.configuration) return;
    const impl = textureImpl(texture);
    if (impl.destroyed || !impl.texture) return;
    const { width, height } = impl;
    if (record.canvas === own) {
      blit(record, width, height);
      return;
    }
    const ownConfigured = [...configured].some((other) => other.canvas === own);
    if (ownConfigured) {
      throw new Error(
        "@typegpu/kit-gl: another canvas cannot be presented while the device's own canvas is configured too.",
      );
    }
    if (own.width !== width) own.width = width;
    if (own.height !== height) own.height = height;
    blit(record, width, height);
    const sink = sinkOf(record);
    if ('transferFromImageBitmap' in sink) {
      sink.transferFromImageBitmap((own as OffscreenCanvas).transferToImageBitmap());
    } else {
      sink.globalCompositeOperation = 'copy';
      sink.drawImage(own, 0, 0);
    }
  };

  const flush = () => {
    scheduled = false;
    if (ctx.lost) return;
    const records = [...pending];
    pending.clear();
    for (const record of records) present(record);
  };

  return {
    presented(record) {
      pending.add(record as Record);
      if (!scheduled) {
        scheduled = true;
        queueMicrotask(flush);
      }
    },

    configure(canvas, configuration, device) {
      const record: Record = {
        canvas,
        configuration: undefined,
        texture: undefined,
        sink: undefined,
      };
      const context = {
        __brand: 'GPUCanvasContext',
        canvas,
        configure(next: GPUCanvasConfiguration) {
          record.texture?.destroy();
          record.texture = undefined;
          record.configuration = { ...next, device: next.device ?? device };
          configured.add(record);
        },
        unconfigure() {
          record.texture?.destroy();
          record.texture = undefined;
          record.configuration = undefined;
          configured.delete(record);
          pending.delete(record);
        },
        getConfiguration() {
          return record.configuration ?? null;
        },
        getCurrentTexture(): GPUTexture {
          const config = record.configuration;
          if (!config)
            throw new DOMException('The canvas context is not configured.', 'InvalidStateError');
          const width = Math.max(1, canvas.width);
          const height = Math.max(1, canvas.height);
          const current = record.texture;
          if (current && current.width === width && current.height === height) return current;
          current?.destroy();
          const texture = createTexture({
            size: [width, height],
            format: config.format,
            usage:
              (config.usage ?? TEXTURE_USAGE.RENDER_ATTACHMENT) |
              TEXTURE_USAGE.RENDER_ATTACHMENT |
              TEXTURE_USAGE.COPY_SRC,
            ...(config.viewFormats && { viewFormats: config.viewFormats }),
            label: 'canvas texture',
          });
          textureImpl(texture).canvas = record;
          record.texture = texture;
          return texture;
        },
      };
      context.configure({ ...configuration, device } as GPUCanvasConfiguration);
      return context as unknown as GPUCanvasContext;
    },

    destroy() {
      for (const record of configured) record.texture?.destroy();
      configured.clear();
      pending.clear();
    },
  };
}
