import { d, readFromArrayBuffer, type TgpuBuffer, type TgpuRoot } from 'typegpu';
import { afterEncoderSubmit } from './encoderSubmit.ts';
import type { Frame } from './frames.ts';

export interface StagingOptions<S extends d.AnyWgslData, V = d.Infer<S>> {
  /** Defines the schema of copied buffer values. */
  readonly schema: S;
  readonly slots: number;
  /**
   * Passes `read` the mapped bytes instead of the decoded value. They are valid only during `read`,
   * which runs before the slot unmaps. Default false.
   */
  readonly bytes?: boolean;
  /**
   * Copies `size` bytes from `offset` in the source, into staging buffers that size. Default the
   * whole buffer.
   */
  readonly range?: { readonly offset: number; readonly size: number } | undefined;
  /** Receives data from slot `i`, which is free again once `read` returns. */
  readonly read: (i: number, value: V) => void;
  /**
   * Receives a mapping rejection or an error from `read` for slot `i`. Not called after `destroy`.
   */
  readonly failed: (i: number, error: unknown) => void;
  /**
   * Reports a slot whose copy was never submitted. That slot holds no copied data.
   * Default none.
   */
  readonly dropped?: (i: number) => void;
}

export interface Staging<S extends d.AnyWgslData> {
  /**
   * Returns a free slot for this frame, or -1 while every slot is mapping. A second call in the
   * same frame returns -1 so a second capture records nothing.
   */
  take(frame: Frame): number;
  /**
   * Copies `source` into slot `i` on the frame encoder and reads it after submission. With `size`,
   * copies and maps only that many bytes from the start of the range.
   */
  copy(frame: Frame, i: number, source: TgpuBuffer<S>, size?: number): void;
  destroy(): void;
}

export function staging<S extends d.AnyWgslData, V = d.Infer<S>>(
  root: TgpuRoot,
  options: StagingOptions<S, V>,
): Staging<S> {
  const { schema, slots, bytes = false, range, read, failed, dropped } = options;
  const states: ('free' | 'copied' | 'mapping')[] = Array.from({ length: slots }, () => 'free');
  const sizes: (number | undefined)[] = Array.from({ length: slots }, () => undefined);
  let taken: Frame | undefined;
  let destroyed = false;

  // MAP_READ permits only COPY_DST alongside it. Allocate all buffers at setup so steady frames
  // create nothing
  const held = range ? d.arrayOf(d.u32, range.size / 4) : schema;
  const buffers = Array.from({ length: slots }, () => {
    const buffer = root.createBuffer<d.AnyWgslData>(held).$addFlags(GPUBufferUsage.MAP_READ);
    root.unwrap(buffer);
    return buffer;
  });

  // Bytes stay borrowed from the mapping, and a schema reads its value out, before the unmap
  const view = bytes
    ? (mapped: ArrayBuffer) => mapped
    : (mapped: ArrayBuffer) => readFromArrayBuffer(mapped, held);

  const reads = buffers.map((buffer, i) => {
    const mapped = root.unwrap(buffer);

    const unmap = () => {
      if (destroyed) return;
      try {
        read(i, view(mapped.getMappedRange(0, sizes[i])) as V);
      } catch (error) {
        failed(i, error);
      } finally {
        if (!destroyed) mapped.unmap();
        states[i] = 'free';
      }
    };

    const reject = (error: unknown) => {
      if (destroyed) return; // destroying a buffer rejects its pending map
      states[i] = 'free';
      failed(i, error);
    };

    return () => {
      if (destroyed) return;
      states[i] = 'mapping';
      mapped.mapAsync(GPUMapMode.READ, 0, sizes[i]).then(unmap, reject);
    };
  });

  return {
    take(frame) {
      if (frame === taken) return -1;
      taken = frame;

      let free = -1;
      for (let i = 0; i < slots; i++) {
        if (states[i] === 'copied') {
          states[i] = 'free'; // its encoder was never submitted, so the copy never ran
          dropped?.(i);
        }

        if (free < 0 && states[i] === 'free') free = i;
      }

      return free;
    },

    copy(frame, i, source, size) {
      sizes[i] = size;
      if (range || size !== undefined) {
        root
          .unwrap(frame.encoder)
          .copyBufferToBuffer(
            root.unwrap(source),
            range?.offset ?? 0,
            root.unwrap(buffers[i]),
            0,
            size ?? range!.size,
          );
      } else {
        buffers[i].copyFrom(source, frame.encoder);
      }
      states[i] = 'copied';
      afterEncoderSubmit(frame, reads[i]);
    },

    destroy() {
      destroyed = true;
      for (const buffer of buffers) buffer.destroy();
    },
  };
}
