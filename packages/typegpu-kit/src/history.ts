/**
 * Keeps the previous frame's state for motion vectors and reprojection. Code compiled with
 * `previously` reads the previous values supplied by `temporal` fills. `history.create` keeps
 * buffer state for those fills.
 *
 * @module
 * @owns 3 buffers: current state, previous state and pending writes
 * @records `record` once a frame, before the passes and kernels that read or move the state.
 */

import {
  d,
  tgpu,
  writeToArrayBuffer,
  type StorageFlag,
  type TgpuAccessor,
  type TgpuBuffer,
  type TgpuCommandEncoder,
  type TgpuGenericFn,
  type TgpuMutableAccessor,
  type TgpuRoot,
  type UniformFlag,
} from 'typegpu';
import { afterEncoderSubmit } from './encoderSubmit.ts';
import type { Frame } from './frames.ts';
import { byStage } from './inject.ts';

/** Selects which frame resolved code reads. `previously` resolves code with `'previous'`. */
export const epoch = tgpu.slot<'current' | 'previous'>('current');

/**
 * Supplies `current` to an accessor, or `previous` in code compiled with `previously`. Use it as
 * `cfg.with(offsets, temporal(now, before))`. Accepts any accessor input, such as a current
 * constant and a previous uniform.
 *
 * @param current what the accessor holds in code compiled as is
 * @param previous the value supplied in code compiled with `previously`
 * @returns a fill for `cfg.with`
 */
export const temporal = <T>(current: T, previous: T): T =>
  // Accessors evaluate function fills while resolving the shader
  (() => (epoch.$ === 'previous' ? previous : current)) as T;

type AnyFn = (...args: never[]) => unknown;

/**
 * Compiles `callback` to read previous values from accessors with `temporal` fills. Call it from
 * any `'use gpu'` function in any stage.
 *
 * @param callback a `'use gpu'` function that reads accessors `temporal` fills
 * @returns a function that reads them as they were last frame
 */
export const previously = <F extends AnyFn>(callback: F): TgpuGenericFn<F> =>
  tgpu.fn(callback).with(epoch, 'previous');

type Usage = 'uniform' | 'storage';

export interface Options<T extends d.AnyWgslData, U extends Usage> {
  /**
   * Selects the buffer usage. Use `uniform` for small CPU-written state, such as a clock, or
   * `storage` for arrays that kernels can also update. Default `storage`.
   */
  readonly usage?: U;
  /** Sets the initial state for both the current and previous frame. Default zeros. */
  readonly initial?: d.InferInput<T>;
}

type Flag<U extends Usage> = U extends 'uniform' ? UniformFlag : StorageFlag;

export interface History<T extends d.AnyWgslData, U extends Usage = 'storage'> extends Disposable {
  /**
   * Holds the current state, including pending writes applied by `record` and changes made by
   * kernels.
   */
  readonly current: TgpuBuffer<T> & Flag<U>;
  /** Holds the state from the previous frame, copied from `current` by the last `record`. */
  readonly previous: TgpuBuffer<T> & Flag<U>;
  /**
   * Fills an accessor with `current`, or `previous` in code compiled with `previously`. Storage
   * binds read-only in render pipelines and read-write in compute pipelines. This fill also works
   * with mutable accessors that kernels write through.
   */
  readonly reading: U extends 'uniform'
    ? TgpuAccessor.In<T>
    : TgpuAccessor.In<T> & TgpuMutableAccessor.In<T>;
  /** Queues a copy of `value` for the next `record` to apply. */
  write(value: d.InferInput<T>): void;
  /**
   * Copies the current state to `previous`, then applies pending writes. Call once per frame
   * before passes and kernels that read or update the state. A write without a following `record`
   * never becomes current, and a frame that does not submit leaves it for the next. A write after
   * `record` leaves the recorded frame as it was and waits for a record in a later submit.
   */
  record(frame: Frame): void;
  /** Destroys the buffers. */
  destroy(): void;
}

const uniformly = (current: TgpuBuffer<d.AnyWgslData> & UniformFlag, previous: typeof current) =>
  temporal(current.as('uniform'), previous.as('uniform'));

const byStages = (current: TgpuBuffer<d.AnyWgslData> & StorageFlag, previous: typeof current) => {
  // Accessors evaluate function fills while resolving the shader
  const now = byStage(current) as unknown as () => unknown;
  const then = byStage(previous) as unknown as () => unknown;
  return () => (epoch.$ === 'previous' ? then() : now());
};

/**
 * Creates buffers for the current and previous state. `write` queues values until `record`
 * records copies into the frame encoder. This preserves the previous state, because CPU writes
 * reach the queue before submitted commands. A uniform history names its usage in the options,
 * as `{ usage: 'uniform' }`.
 *
 * @param schema the state, such as `d.arrayOf(Transform, 256)`
 * @returns a history you destroy
 */
export function create<T extends d.AnyWgslData>(
  root: TgpuRoot,
  schema: T,
  options?: Options<T, 'storage'>,
): History<T, 'storage'>;
export function create<T extends d.AnyWgslData, U extends Usage>(
  root: TgpuRoot,
  schema: T,
  options: Options<T, U> & { readonly usage: U },
): History<T, U>;
export function create<T extends d.AnyWgslData, U extends Usage>(
  root: TgpuRoot,
  schema: T,
  options: Options<T, U> = {},
): History<T, U> {
  const { usage = 'storage', initial } = options;

  // TypeScript cannot resolve TypeGPU's conditional usage and copy checks for generic T. Create
  // and copy buffers as general data, then expose their specific types
  const make = () => {
    const buffer = root.createBuffer(schema as d.AnyWgslData, initial);
    return (usage === 'uniform'
      ? buffer.$usage('uniform')
      : buffer.$usage('storage')) as unknown as TgpuBuffer<T> & UniformFlag & StorageFlag;
  };
  const current = make();
  const previous = make();
  const staged = make();
  const destroy = () => {
    current.destroy();
    previous.destroy();
    staged.destroy();
  };

  const reading = usage === 'uniform' ? uniformly(current, previous) : byStages(current, previous);
  const stagedGpu = root.unwrap(staged);
  const queue = root.device.queue;

  // Writes land in `bytes` and reach `staged` at a record, so a write after the record leaves the
  // recorded copy alone. Each write is a generation; one stays pending until the encoder that
  // copies it submits
  // Sized as the buffer, since a write takes whole words and halves can end mid-word
  const bytes = new ArrayBuffer(stagedGpu.size);
  let written = 0;
  let uploaded = 0;
  let recorded = 0;
  let landed = 0;
  // The encoder whose copy reads `staged`, until it submits
  let reader: TgpuCommandEncoder | undefined;
  const submitted = () => {
    landed = recorded;
    reader = undefined;
  };

  return {
    current,
    previous,
    // TypeScript cannot resolve the conditional fill types for generic T
    reading: reading as History<T, U>['reading'],

    write(value) {
      writeToArrayBuffer(bytes, schema, value);
      written += 1;
    },

    record(frame) {
      const { encoder } = frame;
      previous.copyFrom(current as never, encoder);
      if (landed === written) return;
      // An earlier copy in this encoder still reads `staged`, so a newer write waits for its submit
      if (reader !== encoder && uploaded !== written) {
        queue.writeBuffer(stagedGpu, 0, bytes);
        uploaded = written;
      }
      reader = encoder;
      current.copyFrom(staged as never, encoder);
      recorded = uploaded;
      afterEncoderSubmit(frame, submitted);
    },

    destroy,
    [Symbol.dispose]: destroy,
  };
}
