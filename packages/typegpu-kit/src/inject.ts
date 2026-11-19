/**
 * Provides helpers for writing modules. `unfilled` reports a missing fill, `byStage` selects
 * storage buffer access for each shader stage, and `Checked` checks an options literal whose type
 * a factory infers.
 *
 * @module
 */

import {
  std,
  type Configurable,
  type d,
  type StorageFlag,
  type TgpuAccessor,
  type TgpuBuffer,
} from 'typegpu';

/**
 * Fills module accessors for pipelines created from the configuration it is piped into. Modules
 * return it from `inject()`, as TypeGPU caches do. Use `root.pipe(table.inject())`.
 */
export type Inject = (cfg: Configurable) => Configurable;

/** Provides a module fill through `root.pipe(module.inject())`. */
export interface Injectable {
  inject(): Inject;
}

/**
 * Creates an accessor default that reports a missing `inject()` fill. A shader that reads it fails
 * to resolve with the module's name and the fill it needs. For example,
 * ``tgpu.accessor(d.arrayOf(d.u32), unfilled("a neighbour grid's cells", 'cells', 'the
 * `grid.create` it walks'))``.
 *
 * @param what what the accessor holds, as the error names it
 * @param instance what the caller calls the module, in `root.pipe(instance.inject())`
 * @param origin what made that module
 * @returns an accessor's default that throws when a shader reads it
 */
export const unfilled = (what: string, instance: string, origin: string) => (): never => {
  throw new Error(
    `Nothing filled ${what}: create the pipeline from \`root.pipe(${instance}.inject())\`, where \`${instance}\` is ${origin}.`,
  );
};

/**
 * Fills a storage accessor with read-write access in compute shaders and read-only access in other
 * stages. WebGPU forbids read-write storage in vertex stages, and a dispatch cannot bind the same
 * buffer both ways. Selects access while the shader stage resolves.
 *
 * @param buffer a storage buffer
 * @returns a fill for an accessor of the buffer's schema
 */
export const byStage = <T extends d.AnyWgslData>(
  buffer: TgpuBuffer<T> & StorageFlag,
): TgpuAccessor.In<T> => {
  const readonly = buffer.as('readonly');
  const mutable = buffer.as('mutable');

  // TypeScript cannot resolve the conditional type In<T> for generic T
  return (() => (std.getShaderStage() === 'compute' ? mutable : readonly)) as TgpuAccessor.In<T>;
};

type KeysOf<U> = U extends unknown ? keyof U : never;

// Module handles, such as a camera given as an `eye`, are values, not options to check
type Known<T, S> = T extends
  | readonly unknown[]
  | ((...args: never) => unknown)
  | Disposable
  | { inject(): unknown }
  ? T
  : {
      readonly [K in keyof T]: K extends KeysOf<NonNullable<S>>
        ? Known<T[K], NonNullable<S>[K & keyof NonNullable<S>]>
        : `${K & string} is not an option here`;
    };

/**
 * Types an options parameter whose literal type `O` a factory infers, against the options type
 * `S`, so a mistake is an error at the literal and never changes the type of what the factory
 * returns. A value of the wrong type is reported against `S`, and a key `S` lacks is reported at
 * the key as the type `'<key> is not an option here'`, such as `'smapled is not an option here'`.
 */
export type Checked<O, S> = O extends S ? (O extends Known<O, S> ? O : Known<O, S>) : S;
