/**
 * These `'use gpu'` functions combine one value from each invocation of a workgroup: `reduce`
 * gives every invocation the combination of all of them, and `scan` gives each the combination of
 * those before it.
 *
 * @module
 * @kind shader
 */

import { d, std, tgpu } from 'typegpu';

/**
 * This type combines values of one schema. `combine` must be associative, and also commutative
 * for `reduce`. Combining a value with `identity()` must leave it unchanged. Both functions are
 * `'use gpu'` and also run on the CPU, so a test can fold the same values.
 */
export interface Monoid<T extends d.AnyWgslData = d.AnyWgslData> {
  /** Holds the schema of the values it combines. */
  readonly schema: T;
  /** Returns the value that leaves another unchanged when combined with it. */
  readonly identity: () => d.Infer<T>;
  /** Combines two values into one. */
  readonly combine: (a: d.Infer<T>, b: d.Infer<T>) => d.Infer<T>;
}

/** This type is a scalar or vector schema of numbers, which `min`, `max` and `bounds` compare. */
export type Ordered =
  | d.F32
  | d.F16
  | d.I32
  | d.U32
  | d.Vec2f
  | d.Vec3f
  | d.Vec4f
  | d.Vec2h
  | d.Vec3h
  | d.Vec4h
  | d.Vec2i
  | d.Vec3i
  | d.Vec4i
  | d.Vec2u
  | d.Vec3u
  | d.Vec4u;

/** Tests that a schema is a numeric scalar or vector, or a struct or array of them. */
type Numeric<T> = T extends Ordered
  ? true
  : T extends d.Decorated<infer I>
    ? Numeric<I>
    : T extends d.WgslArray<infer E>
      ? Numeric<E>
      : T extends d.WgslStruct<infer P>
        ? false extends { [K in keyof P]: Numeric<P[K]> }[keyof P]
          ? false
          : true
        : false;

type Combine = (a: never, b: never) => unknown;
type Fields = Record<string | number, unknown>;

const added = <T>(a: T, b: T): T => {
  'use gpu';
  return ((a as number) + (b as number)) as T;
};

const NUMERIC = /^(f32|f16|i32|u32|vec[234][fhiu])$/;

const smaller = <T>(a: T, b: T): T => {
  'use gpu';
  return std.min(a as never, b as never) as T;
};

const larger = <T>(a: T, b: T): T => {
  'use gpu';
  return std.max(a as never, b as never) as T;
};

/** Adds structs and arrays field by field, down to `added` on each scalar or vector. */
function addOf(schema: d.BaseData): Combine {
  let fields: { key: string | number; add: Combine }[];
  if (d.isDecorated(schema)) return addOf(schema.inner);
  if (d.isWgslArray(schema)) {
    const add = addOf(schema.elementType);
    fields = Array.from({ length: schema.elementCount }, (_, key) => ({ key, add }));
  } else if (d.isWgslStruct(schema)) {
    fields = Object.entries(schema.propTypes).map(([key, field]) => ({
      key,
      add: addOf(field),
    }));
  } else if (NUMERIC.test(schema.type)) {
    return added;
  } else {
    throw new TypeError(`workgroup.sum adds numbers and vectors, not ${schema.type}.`);
  }
  const fieldwise = (a: Fields, b: Fields) => {
    'use gpu';
    const total = std.copy(a);
    for (const field of tgpu.unroll(fields)) {
      total[field.key] = field.add(a[field.key] as never, b[field.key] as never);
    }
    return total;
  };
  return fieldwise as Combine;
}

/**
 * Adds values from zero, per component of a vector, on the GPU and the CPU alike. A struct or
 * array adds field by field, so one reduction sums several quantities at once. Throws for a field
 * that is not a number or a vector of them, such as a `bool` or a matrix.
 *
 * @param schema a numeric scalar or vector, or a struct or array of them
 * @returns a monoid for `reduce` and `scan`
 */
export function sum<T extends d.AnyWgslData>(
  schema: T & (Numeric<T> extends true ? unknown : 'adds numbers and vectors only'),
): Monoid<T> {
  const value = d.isDecorated(schema) ? schema.inner : schema;
  const create = value as unknown as () => d.Infer<T>;
  const zero = () => {
    'use gpu';
    return create();
  };
  return { schema, identity: zero, combine: addOf(schema) as Monoid<T>['combine'] };
}

const HIGHEST = { f32: 3.4028234663852886e38, f16: 65504, i32: 2147483647, u32: 4294967295 };
const LOWEST = { f32: -HIGHEST.f32, f16: -HIGHEST.f16, i32: -HIGHEST.i32 - 1, u32: 0 };

function extreme<T extends Ordered>(schema: T, extremes: typeof HIGHEST): () => d.Infer<T> {
  const component = ('primitive' in schema ? schema.primitive : schema).type;
  const value = extremes[component as keyof typeof HIGHEST];
  const create = schema as unknown as (value: number) => d.Infer<T>;
  const filled = () => {
    'use gpu';
    return create(value);
  };
  // WGSL has no literal for the lowest i32, so it is the complement of the highest
  const lowestI32 = () => {
    'use gpu';
    return create(~d.i32(HIGHEST.i32));
  };
  return component === 'i32' && extremes === LOWEST ? lowestI32 : filled;
}

/**
 * Keeps the smaller value, per component of a vector, from the highest value the schema holds.
 *
 * @param schema a scalar or vector of numbers
 * @returns a monoid for `reduce` and `scan`
 */
export function min<T extends Ordered>(schema: T): Monoid<T> {
  return { schema, identity: extreme(schema, HIGHEST), combine: smaller };
}

/**
 * Keeps the larger value, per component of a vector, from the lowest value the schema holds.
 *
 * @param schema a scalar or vector of numbers
 * @returns a monoid for `reduce` and `scan`
 */
export function max<T extends Ordered>(schema: T): Monoid<T> {
  return { schema, identity: extreme(schema, LOWEST), combine: larger };
}

/**
 * Keeps the box around points: `low` is their `min` and `high` their `max`. Give each invocation's
 * point as `schema({ low: point, high: point })`.
 *
 * @param schema a scalar or vector of numbers
 * @returns a monoid of `{ low, high }` for `reduce` and `scan`
 */
export function bounds<T extends Ordered>(schema: T): Monoid<d.WgslStruct<{ low: T; high: T }>> {
  const Bounds = d.struct({ low: schema, high: schema });
  const low = min(schema);
  const high = max(schema);
  const empty = () => {
    'use gpu';
    return Bounds({ low: low.identity(), high: high.identity() });
  };
  const enclosing = (a: d.Infer<typeof Bounds>, b: d.Infer<typeof Bounds>) => {
    'use gpu';
    return Bounds({ low: low.combine(a.low, b.low), high: high.combine(a.high, b.high) });
  };
  return { schema: Bounds, identity: empty, combine: enclosing };
}

/** Declares one value of the monoid's schema per invocation, in workgroup memory. */
const lanesOf = <T extends d.AnyWgslData>(monoid: Monoid<T>, size: number) =>
  tgpu.workgroupVar(d.arrayOf(monoid.schema, size)) as unknown as { $: d.Infer<T>[] };

function checked(name: string, size: number) {
  if (!(size >= 1 && Number.isInteger(Math.log2(size)))) {
    throw new RangeError(
      `workgroup.${name} requires \`size\` to be a power of two. Received ${size}.`,
    );
  }
  return size;
}

/**
 * Returns a `'use gpu'` function `(value, lid)` that combines `value` from each of the `size`
 * invocations of a workgroup and returns the result to every one of them. `lid` is the
 * invocation's `localInvocationIndex`. Every invocation must call it, before any of them returns:
 * one with nothing to add passes `monoid.identity()`. Each call to `reduce` declares its own
 * workgroup array, and the function can run several times in one kernel.
 *
 * @param monoid how to combine values, such as `workgroup.sum(d.f32)`
 * @param size the workgroup's invocation count, a power of two
 * @returns a `'use gpu'` function for kernels of that workgroup size
 */
export function reduce<T extends d.AnyWgslData>(
  monoid: Monoid<T>,
  size: number,
): (value: d.Infer<T>, lid: number) => d.Infer<T> {
  const lanes = lanesOf(monoid, checked('reduce', size));
  const strides = Array.from({ length: Math.log2(size) }, (_, i) => size >> (i + 1));
  const reduced = (value: d.Infer<T>, lid: number) => {
    'use gpu';
    lanes.$[lid] = std.copy(value);
    for (const stride of tgpu.unroll(strides)) {
      std.workgroupBarrier();
      if (lid < stride) lanes.$[lid] = monoid.combine(lanes.$[lid], lanes.$[lid + stride]);
    }
    std.workgroupBarrier();
    const result = std.copy(lanes.$[0]);
    // The next call writes the lanes again
    std.workgroupBarrier();
    return result;
  };
  return reduced;
}

/**
 * Returns a `'use gpu'` function `(value, lid)` that gives each of the `size` invocations of a
 * workgroup the combination of the values of the invocations before it in `lid` order, and the
 * first `monoid.identity()`. Combine the result with `value` to include the invocation's own.
 * Calls follow the rules of `reduce`.
 *
 * @param monoid how to combine values, such as `workgroup.sum(d.u32)` for offsets to write at
 * @param size the workgroup's invocation count, a power of two
 * @returns a `'use gpu'` function for kernels of that workgroup size
 */
export function scan<T extends d.AnyWgslData>(
  monoid: Monoid<T>,
  size: number,
): (value: d.Infer<T>, lid: number) => d.Infer<T> {
  const lanes = lanesOf(monoid, checked('scan', size));
  const reaches = Array.from({ length: Math.log2(size) }, (_, i) => 1 << i);
  const scanned = (value: d.Infer<T>, lid: number) => {
    'use gpu';
    lanes.$[lid] = std.copy(value);
    std.workgroupBarrier();
    for (const reach of tgpu.unroll(reaches)) {
      let earlier = monoid.identity();
      if (lid >= reach) earlier = std.copy(lanes.$[lid - reach]);
      std.workgroupBarrier();
      lanes.$[lid] = monoid.combine(earlier, lanes.$[lid]);
      std.workgroupBarrier();
    }
    let before = monoid.identity();
    if (lid > 0) before = std.copy(lanes.$[lid - 1]);
    // The next call writes the lanes again
    std.workgroupBarrier();
    return before;
  };
  return scanned;
}
