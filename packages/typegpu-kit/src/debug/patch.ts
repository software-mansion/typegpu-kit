type Method<T, K extends keyof T> = Extract<T[K], (...args: never[]) => unknown>;
type Args<T, K extends keyof T> = Parameters<Method<T, K>>;
type Result<T, K extends keyof T> = ReturnType<Method<T, K>>;

/** This function type installs a replacement for a method on `target`. */
export type Patch = <T extends object, K extends keyof T>(
  target: T,
  key: K,
  handler: (self: T, args: Args<T, K>, next: (...args: Args<T, K>) => Result<T, K>) => Result<T, K>,
  active?: () => boolean,
) => void;

/**
 * Creates reversible method patches. A replacement calls `handler`, which can call `next` to run
 * the original method. When `active` returns false, only the original method runs. `restore`
 * removes patches in reverse order and preserves other methods.
 */
export function patches(): { readonly patch: Patch; restore(): void } {
  const restores: (() => void)[] = [];

  const patch: Patch = (target, key, handler, active = () => true) => {
    type T = typeof target;
    type K = typeof key;
    const original = target[key] as Method<T, K>;

    target[key] = function (this: T, ...args: Args<T, K>): Result<T, K> {
      if (!active()) return Reflect.apply(original, this, args) as Result<T, K>;
      return handler(this, args, (...forwarded) => Reflect.apply(original, this, forwarded));
    } as T[K];

    restores.push(() => {
      target[key] = original;
    });
  };

  return {
    patch,
    restore() {
      const undo = restores.splice(0);
      undo.reverse();
      for (const restore of undo) restore();
    },
  };
}
