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

/** Each removed replacement that a newer one still wraps, and the property it replaced. */
const retired = new WeakMap<object, PropertyDescriptor | undefined>();

/**
 * Creates reversible method patches. A replacement calls `handler`, which can call `next` to run
 * the method it replaced on the same receiver. When `active` returns false, only that method runs.
 * `restore` removes patches in reverse order and puts back the target's own property, or its
 * absence. A replacement that a newer patch has wrapped since stays, running only the method it
 * replaced, until the newer patch is removed too.
 */
export function patches(): { readonly patch: Patch; restore(): void } {
  const restores: (() => void)[] = [];

  const patch: Patch = (target, key, handler, active = () => true) => {
    type T = typeof target;
    type K = typeof key;
    const before = Object.getOwnPropertyDescriptor(target, key);
    const original = target[key] as Method<T, K>;

    const replacement = function (this: T, ...args: Args<T, K>): Result<T, K> {
      if (retired.has(replacement) || !active()) {
        return Reflect.apply(original, this, args) as Result<T, K>;
      }
      return handler(this, args, (...forwarded) => Reflect.apply(original, this, forwarded));
    };
    target[key] = replacement as T[K];

    restores.push(() => {
      if (Object.getOwnPropertyDescriptor(target, key)?.value !== replacement) {
        retired.set(replacement, before);
        return;
      }
      let restored = before;
      while (restored && retired.has(restored.value)) restored = retired.get(restored.value);
      if (restored) Object.defineProperty(target, key, restored);
      else delete target[key];
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
