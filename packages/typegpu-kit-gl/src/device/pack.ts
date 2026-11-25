/**
 * Reads that need not wait: `readPixels` into a pixel pack buffer copies on the GPU, as the
 * command stream reaches it, and a fence says when the bytes are there. Reading the buffer back
 * before the fence signals waits for the GPU, as a plain `readPixels` does.
 */

import type { Ctx } from './context.ts';

export interface Packed {
  readonly buffer: WebGLBuffer;
  readonly fence: WebGLSync;
  readonly bytes: number;
}

/**
 * Starts reading `width` by `height` pixels of the bound read framebuffer into a new pack buffer,
 * as `format` and `type`, `bytes` long.
 */
export function packPixels(
  ctx: Ctx,
  x: number,
  y: number,
  width: number,
  height: number,
  format: number,
  type: number,
  bytes: number,
): Packed {
  const { gl } = ctx;
  const buffer = gl.createBuffer()!;
  gl.bindBuffer(gl.PIXEL_PACK_BUFFER, buffer);
  gl.bufferData(gl.PIXEL_PACK_BUFFER, bytes, gl.STREAM_READ);
  gl.readPixels(x, y, width, height, format, type, 0);
  gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
  const fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0)!;
  // A fence signals only once the commands before it reach the GPU
  gl.flush();
  return { buffer, fence, bytes };
}

/** Whether the GPU has finished the read. WebGL updates this only between tasks. */
export function signalled(ctx: Ctx, packed: Packed): boolean {
  const { gl } = ctx;
  return ctx.lost || gl.getSyncParameter(packed.fence, gl.SYNC_STATUS) === gl.SIGNALED;
}

/** Copies the read bytes into `into`, waiting for the GPU if they are not there yet, and frees them. */
export function unpack(ctx: Ctx, packed: Packed, into: ArrayBufferView): void {
  const { gl } = ctx;
  if (!ctx.lost) {
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, packed.buffer);
    gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, into);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
  }
  discard(ctx, packed);
}

/** Frees a read without copying it out. */
export function discard(ctx: Ctx, packed: Packed): void {
  ctx.gl.deleteBuffer(packed.buffer);
  ctx.gl.deleteSync(packed.fence);
}
