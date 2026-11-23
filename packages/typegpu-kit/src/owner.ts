/**
 * Collects what a scene or module makes as it makes it, to compile and free it all in one call
 * each. It is optional: a scene that makes a few modules lists them for `pipelines.compile` and
 * frees them in its own `destroy`.
 *
 * @module
 */

import { freeEach } from './owned.ts';
import type { Warmable } from './pipelines.ts';

/** What an owner frees: anything with `destroy()`, or a fixed resource such as a uniform. */
export type Freeable = { destroy(): void } | { readonly buffer: { destroy(): void } };

/** A function an owner runs at `destroy`, such as one freeing a resource replaced on resize. */
export type Cleanup = () => void;

/**
 * What an owner takes: something to free or dispose, a cleanup, or something that only compiles,
 * such as a pipeline.
 */
export type Ownable = Freeable | Cleanup | Disposable | Warmable;

export interface Owner extends Disposable {
  /**
   * Takes `item` to free at `destroy` and returns it. Passes `undefined` through. Frees a fixed
   * resource, such as one from `root.createUniform`, through its `buffer`, and runs a cleanup
   * function. Lists an item with `initAsync` in `warmables`. After `destroy`, frees `item` at
   * once.
   */
  <T extends Ownable | undefined>(item: T): T;
  /** Reports whether `destroy` has run. */
  readonly destroyed: boolean;
  /**
   * Lists what it took that has `initAsync`, in the order taken, until `destroy`.
   * `pipelines.compile(own)` compiles them.
   */
  readonly warmables: readonly Warmable[];
  /**
   * Throws the owner's message if it is destroyed. Call it first in a method of your own module
   * that should refuse to run after `destroy`.
   */
  usable(): void;
  /**
   * Frees everything it took, the last taken first, once; later calls do nothing. When one throws,
   * frees the rest, then throws its error, or an `AggregateError` of all of them with the first
   * one's message. Also `[Symbol.dispose]`.
   */
  destroy(): void;
}

/**
 * Collects what a scene or module makes, to compile and free it all in one call each. Pass each
 * thing through it as you make it, pipelines and pass descriptions too, and return its `destroy`
 * as your own. Take a resource that is replaced during the module's life, such as a texture that
 * follows the canvas size, as one cleanup that frees the current one. What it takes after
 * `destroy`, such as a model that finished loading late, it frees at once.
 *
 * ```ts
 * const own = owner('The swarm was destroyed.');
 * const screen = own(targets.create(root, surface, { hdr }));
 * const settings = own(root.createUniform(Settings));
 * const kernel = own(root.createComputePipeline({ compute: step }));
 * own(() => {
 *   for (const buffer of reads) buffer.destroy();
 * });
 * return {
 *   initAsync: () => pipelines.compile(own),
 *   step(pass) { own.usable(); … },
 *   destroy: own.destroy,
 *   [Symbol.dispose]: own.destroy,
 * };
 * ```
 *
 * @param message what `usable` throws, naming the module: "The table was destroyed."
 * @returns an owner you pass each thing to and destroy
 */
export function owner(message = 'It was destroyed.'): Owner {
  const items: Ownable[] = [];
  const warmables: Warmable[] = [];
  let destroyed = false;

  const own = <T extends Ownable | undefined>(item: T): T => {
    if (item === undefined) return item;
    if (destroyed) free(item);
    else {
      items.push(item);
      if (typeof (item as Partial<Warmable>).initAsync === 'function') {
        warmables.push(item as Warmable);
      }
    }
    return item;
  };

  const destroy = () => {
    if (destroyed) return;
    destroyed = true;

    const frees: (() => void)[] = [];
    for (let i = items.length - 1; i >= 0; i--) {
      const item = items[i];
      frees.push(() => free(item));
    }
    items.length = 0;
    warmables.length = 0;
    freeEach(frees);
  };

  return Object.defineProperties(own, {
    destroyed: { get: () => destroyed },
    warmables: { value: warmables },
    usable: {
      value: () => {
        if (destroyed) throw new Error(message);
      },
    },
    destroy: { value: destroy },
    [Symbol.dispose]: { value: destroy },
  }) as Owner;
}

// An owner is itself a function with `destroy`, so `destroy` comes before the call
const free = (item: Ownable) => {
  const held = item as Partial<{ destroy(): void; buffer: { destroy(): void } } & Disposable>;
  if (held.destroy) held.destroy();
  else if (typeof item === 'function') item();
  else if (held.buffer) held.buffer.destroy();
  else held[Symbol.dispose]?.();
};
