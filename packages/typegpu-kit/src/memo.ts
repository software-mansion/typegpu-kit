/** Limiting calls to four arguments avoids allocating an argument array. */
type Args = readonly unknown[] & { readonly length: 0 | 1 | 2 | 3 | 4 };

interface Entry<R> {
  ready: boolean;
  a: unknown;
  b: unknown;
  c: unknown;
  d: unknown;
  result: R | undefined;
}

const entry = <R>(): Entry<R> => ({
  ready: false,
  a: undefined,
  b: undefined,
  c: undefined,
  d: undefined,
  result: undefined,
});

/**
 * Caches the last two results of `fn`, comparing arguments by identity. Calls `fn` only when
 * neither cached argument list matches. For example, create a bind group with `const group =
 * memo((hdr) => root.createBindGroup(layout, { hdr }))`, then call `group(screen.textures.hdr)`
 * each frame. Alternating between two textures, such as a history pair, calls `fn` only twice.
 * Does not destroy cached results.
 *
 * @param fn a function of up to four arguments
 * @returns `fn`, remembering its last two results
 */
export function memo<const A extends Args, R>(fn: (...args: A) => R): (...args: A) => R {
  const run = fn as unknown as (a?: unknown, b?: unknown, c?: unknown, d?: unknown) => R;

  let newer = entry<R>();
  let older = entry<R>();

  const holds = (kept: Entry<R>, a: unknown, b: unknown, c: unknown, d: unknown) =>
    kept.ready && a === kept.a && b === kept.b && c === kept.c && d === kept.d;

  // Destructuring would allocate an array on every call
  const swap = () => {
    const kept = newer;
    newer = older;
    older = kept;
  };

  const call = (a?: unknown, b?: unknown, c?: unknown, d?: unknown) => {
    if (holds(newer, a, b, c, d)) return newer.result as R;
    swap();
    if (holds(newer, a, b, c, d)) return newer.result as R;

    // Replace the older entry after swapping
    newer.result = run(a, b, c, d);
    newer.ready = true;
    newer.a = a;
    newer.b = b;
    newer.c = c;
    newer.d = d;

    return newer.result;
  };

  return call as unknown as (...args: A) => R;
}

/**
 * Describes a render pass on one view, kept and reused so a frame allocates no new one.
 *
 * @hidden
 */
export interface PassDescriptor {
  colorAttachments: {
    view: GPUTextureView;
    clearValue?: GPUColor;
    loadOp: GPULoadOp;
    storeOp: GPUStoreOp;
  };
  timestampWrites?: GPURenderPassTimestampWrites | undefined;
}

/**
 * Makes a memo of passes that clear a view to `clearValue` and store what they draw.
 *
 * @hidden
 */
export const passOf = (clearValue: GPUColor = [0, 0, 0, 0]) =>
  memo((view: GPUTextureView): PassDescriptor => ({
    colorAttachments: { view, clearValue, loadOp: 'clear', storeOp: 'store' },
  }));
