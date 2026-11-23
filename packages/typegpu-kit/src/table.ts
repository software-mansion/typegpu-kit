/**
 * Stores per-object rows in GPU buffers with stable ids. The CPU and kernels each write their own
 * fields.
 *
 * @module
 * @injects definition its definition's rows, live bits and span, for code written over any table
 * of the definition; each table's `gpu` members carry its own
 * @binds 1 storage: a field, per part, host or device, such as `Boid.position(i)`; `row(i)` binds
 * both parts
 * @binds 1 storage, 1 uniform: `alive(i)`, the live bits and the span; `span()` and `slotOf(id)`
 * bind only the uniform
 * @owns 3 or 4 buffers: the host and device rows, the live bits and the span; a part without fields
 * has no buffer. With `history`, a part's buffer holds last frame's rows after its own, and 2 more
 * per part hold the uploads and a copy on the way
 * @records `flush()` once a frame, before submit: it uploads the frame's changes with
 * `queue.writeBuffer`, which land before every later submit. With `history`, `flush(frame)` also
 * records copies that keep last frame's rows, before the passes that read the table.
 */

import {
  d,
  std,
  tgpu,
  type Configurable,
  type StorageFlag,
  type TgpuAccessor,
  type TgpuBuffer,
  type TgpuCommandEncoder,
  type TgpuGenericFn,
  type TgpuRoot,
  type TgpuUniform,
} from 'typegpu';
import { afterEncoderSubmit } from './encoderSubmit.ts';
import { encoderOf, type Frame } from './frames.ts';
import {
  byteViews,
  componentsOf,
  offsetsOf,
  readScalar,
  scalarOf,
  writerOf,
  writeScalar,
  type ByteViews,
  type Writer,
} from './bytes.ts';
import { temporal } from './history.ts';
import { byStage, unfilled, type Inject } from './inject.ts';
import { named } from './named.ts';

export type Props = Record<string, d.AnyWgslData>;
export type Rows<P extends Props> = d.WgslArray<d.WgslStruct<P>>;

declare const rowBrand: unique symbol;
/**
 * Identifies a row with its slot in the low bits and its generation in the remaining bits. The
 * slot uses as many bits as capacity requires. Fits in a `u32` and is never 0. The definition
 * brands the id type, as in `RowId<typeof Enemy>`.
 */
export type RowId<T extends AnyTableDef = AnyTableDef> = number & { readonly [rowBrand]: T };

const MAX_CAPACITY = 2 ** 20;
/** Marks every device field of a row as written. */
const ALL = 0xffffffff;
/** Host uploads bridge gaps of fewer clean rows than this. */
const HOST_GAP = 16;
/** A new row waits for its first flush: `FRESH`, or for the frame that recorded it: `HELD`. */
const FRESH = 1;
const HELD = 2;

/** Shaders need the span and bitmask to extract a slot from an id. */
const Index = d.struct({ span: d.u32, mask: d.u32 }).$name('index');

declare const schemaOf: unique symbol;
/**
 * Reads a field in a shader. Its type carries the field schema, so a `u32` getter cannot pass for
 * an `f32` getter.
 */
export interface Getter<T> extends TgpuGenericFn<(i: number) => d.InferGPU<T>> {
  readonly [schemaOf]?: T;
  pipe(transform: (cfg: Configurable) => Configurable): Getter<T>;
}

/** Provides one getter per field so shaders load only the fields they use. */
export type Getters<P extends Props> = { readonly [K in keyof P]: Getter<P[K]> };

type Reserved = 'host' | 'device' | 'row' | 'alive' | 'span' | 'slotOf';
type Unreserved = { readonly [K in Reserved]?: 'is reserved by table.define' };
type NotIn<H> = { readonly [K in keyof H]?: 'is a host field already' };

interface Parts<N extends string, H extends Props, D extends Props> {
  /** Labels types and WGSL. It is not a lookup key. */
  readonly name: N;
  readonly host: d.WgslStruct<H> | undefined;
  readonly device: d.WgslStruct<D> | undefined;
  /** Describes the whole row that `row(i)` returns. */
  readonly row: d.WgslStruct<H & D>;
  readonly live: TgpuAccessor<d.WgslArray<d.U32>>;
  readonly index: TgpuAccessor<typeof Index>;
  /** Offsets slots to last frame's rows: `capacity` in code compiled with `previously`, else 0. */
  readonly back: TgpuAccessor<d.U32>;
  readonly writers: Readonly<Record<string, Writer>>;
  /** Counts the tables created from the definition, so each starts its generations elsewhere. */
  tables: number;
}

/** Defines a table layout and its shader functions without a root. */
export type TableDef<N extends string, H extends Props, D extends Props> = Getters<H> &
  Getters<D> & {
    /** Provides CPU-written rows with read-only shader access. */
    readonly host: TgpuAccessor<Rows<H>>;
    /**
     * Provides GPU-written rows with read-write access in compute pipelines and read-only access
     * in render pipelines.
     */
    readonly device: TgpuAccessor<Rows<D>>;
    /**
     * Returns a copy of the whole row, including every field, for helpers that take a row value.
     */
    readonly row: TgpuGenericFn<(i: number) => d.InferGPU<d.WgslStruct<H & D>>>;
    /** Tests whether slot `i` holds a row. Returns false beyond the span. */
    readonly alive: TgpuGenericFn<(i: number) => boolean>;
    /** Returns the highest live slot plus one. */
    readonly span: TgpuGenericFn<() => number>;
    /**
     * Extracts a slot from a row id, such as an id stored in a host field. Does not check the
     * generation.
     */
    readonly slotOf: TgpuGenericFn<(id: number) => number>;
    readonly '~parts': Parts<N, H, D>;
  };

// oxlint-disable-next-line typescript/no-explicit-any -- matches a definition of any layout
export type AnyTableDef = TableDef<string, any, any>;

type NameOf<T extends AnyTableDef> = T['~parts']['name'];

type HostOf<T extends AnyTableDef> = NonNullable<T['~parts']['host']>['propTypes'];

type DeviceOf<T extends AnyTableDef> = NonNullable<T['~parts']['device']>['propTypes'];

/**
 * Gives a definition's shader members bound to one table: its getters, `row`, `alive`, `span`,
 * `slotOf`, `host` and `device`, which read that table in any pipeline without a fill.
 */
type GpuMembers<T extends AnyTableDef> = Omit<
  TableDef<NameOf<T>, HostOf<T>, DeviceOf<T>>,
  '~parts'
>;

/** Gives the whole row type as read by shaders, for helpers that take a row value. */
export type RowOf<T extends AnyTableDef> = d.InferGPU<d.WgslStruct<HostOf<T> & DeviceOf<T>>>;

/**
 * Creates an array schema for host or device rows, for a buffer or history you manage. Throws if
 * capacity is not an integer in [1, 2^20] or the selected part has no fields.
 *
 * @param def the definition, from `table.define`
 * @param part `'host'` or `'device'`
 * @param capacity how many rows the array holds
 * @returns an array schema of the part's rows
 */
export function rows<T extends AnyTableDef, P extends 'host' | 'device'>(
  def: T,
  part: P,
  capacity: number,
): Rows<P extends 'host' ? HostOf<T> : DeviceOf<T>> {
  if (!Number.isInteger(capacity) || capacity < 1 || capacity > MAX_CAPACITY) {
    throw new RangeError(`A table's capacity must be an integer in [1, 2^20], got ${capacity}.`);
  }
  const schema = def['~parts'][part];
  if (!schema) throw new Error(`The table '${def['~parts'].name}' has no ${part} part.`);
  return d.arrayOf(schema, capacity);
}

function getter(source: TgpuAccessor<Rows<Props>>, back: TgpuAccessor<d.U32>, key: string) {
  return (i: number) => {
    'use gpu';
    return std.copy(source.$[d.u32(i) + back.$][key]);
  };
}

const RESERVED = new Set<string>([
  'host',
  'device',
  'row',
  'alive',
  'span',
  'slotOf',
] satisfies Reserved[]);

/**
 * Declares CPU-written `host` fields and GPU-written `device` fields using the records accepted by
 * `d.struct`. Throws if no fields are declared, a field
 * appears in both parts, the CPU writer cannot write a field type such as `f16`, or a field uses a
 * reserved member name such as `alive`.
 *
 * `name` labels structs, buffers, WGSL functions and errors. It is not a lookup key. The
 * definition itself brands row ids, as `RowId<typeof Enemy>`.
 *
 * @param name a label for ids and WGSL
 * @param parts the `host` fields the CPU writes and the `device` fields the GPU writes
 * @returns a definition whose getters shaders read and that `table.create` allocates
 */
export function define<
  const N extends string,
  H extends Props & Unreserved = {},
  D extends Props & Unreserved & NotIn<H> = {},
>(name: N, parts: { host?: H; device?: D }): TableDef<N, H, D> {
  const hostFields: Props = parts.host ?? {};
  const deviceFields: Props = parts.device ?? {};
  const hostKeys = Object.keys(hostFields);
  const deviceKeys = Object.keys(deviceFields);

  if (hostKeys.length + deviceKeys.length === 0) throw new Error('A table needs a field.');
  for (const key of deviceKeys) {
    if (Object.hasOwn(hostFields, key)) {
      throw new Error(`'${key}' is both a host and a device field; each field has one writer.`);
    }
  }

  const writers: Record<string, Writer> = {};
  for (const [key, schema] of Object.entries({ ...hostFields, ...deviceFields })) {
    if (RESERVED.has(key)) throw new Error(`'${key}' is reserved by table.define.`);
    writers[key] = writerOf(schema, key);
  }

  // TypeGPU cannot lay out empty structs, so omit parts with no fields
  const hostStruct = hostKeys.length > 0 ? d.struct(hostFields).$name(`${name}Host`) : undefined;
  const deviceStruct =
    deviceKeys.length > 0 ? d.struct(deviceFields).$name(`${name}Device`) : undefined;

  const empty = (accessor: string) =>
    unfilled(`\`${name}_${accessor}\``, 'rows', `a \`table.create\` of \`${name}\``);

  // Leave missing parts unfilled so shaders reading them fail with `Missing value`
  const host = tgpu
    .accessor(d.arrayOf(hostStruct ?? d.u32), hostStruct ? empty('host') : undefined)
    .$name(`${name}_host`) as unknown as TgpuAccessor<Rows<H>>;
  const device = tgpu
    .accessor(d.arrayOf(deviceStruct ?? d.u32), deviceStruct ? empty('device') : undefined)
    .$name(`${name}_device`) as unknown as TgpuAccessor<Rows<D>>;
  const live = tgpu.accessor(d.arrayOf(d.u32), empty('live')).$name(`${name}_live`);
  const index = tgpu.accessor(Index, empty('index')).$name(`${name}_index`);
  const back = tgpu.accessor(d.u32, 0).$name(`${name}_back`);

  // Building getters by key loses H and D, so restore the declared types with casts
  const fields: Record<string, unknown> = {};
  for (const key of hostKeys) {
    fields[key] = named(`${name}_${key}`, getter(host as never, back, key));
  }
  for (const key of deviceKeys) {
    fields[key] = named(`${name}_${key}`, getter(device as never, back, key));
  }

  const whole = d.struct({ ...hostFields, ...deviceFields }).$name(`${name}Row`);
  const hostRows = host as unknown as TgpuAccessor<Rows<Props>>;
  const deviceRows = device as unknown as TgpuAccessor<Rows<Props>>;

  const row = (i: number) => {
    'use gpu';
    const result = whole() as Record<string, unknown>;
    const at = d.u32(i) + back.$;
    for (const key of tgpu.unroll(hostKeys)) result[key] = std.copy(hostRows.$[at][key]);
    for (const key of tgpu.unroll(deviceKeys)) result[key] = std.copy(deviceRows.$[at][key]);

    return result;
  };

  return {
    ...(fields as Getters<H> & Getters<D>),
    host,
    device,
    row: named(`${name}_row`, row) as unknown as TableDef<N, H, D>['row'],

    alive: named(`${name}_alive`, (i: number) => {
      'use gpu';
      return i < index.$.span && ((live.$[i >>> 5] >>> (i & 31)) & 1) === 1;
    }),

    span: named(`${name}_span`, () => {
      'use gpu';
      return index.$.span;
    }),

    slotOf: named(`${name}_slotOf`, (id: number) => {
      'use gpu';
      return id & index.$.mask;
    }),

    '~parts': {
      name,
      host: hostStruct as d.WgslStruct<H> | undefined,
      device: deviceStruct as d.WgslStruct<D> | undefined,
      row: whole as unknown as d.WgslStruct<H & D>,
      live,
      index,
      back,
      writers,
      tables: 0,
    },
  };
}

type Scalar = d.F32 | d.U32 | d.I32;
type Vec2 = d.Vec2f | d.Vec2i | d.Vec2u;
type Vec3 = d.Vec3f | d.Vec3i | d.Vec3u;
type Vec4 = d.Vec4f | d.Vec4i | d.Vec4u;

/**
 * Gives the value a field of schema `T` takes. A vector also takes any array of numbers, such as a
 * `Float32Array` from a math library, and reads its first components.
 */
export type FieldInput<T> = T extends Vec2 | Vec3 | Vec4
  ? d.InferInput<T> | ArrayLike<number>
  : d.InferInput<T>;

type Schemas<T extends AnyTableDef> = HostOf<T> & DeviceOf<T>;

/** Gives the values `add` and `write` take: any fields of either part. */
export type Values<T extends AnyTableDef> = {
  readonly [K in keyof Schemas<T>]?: FieldInput<Schemas<T>[K]>;
};

type Components<T> = T extends Scalar
  ? [value: number]
  : T extends Vec2
    ? [x: number, y: number]
    : T extends Vec3
      ? [x: number, y: number, z: number]
      : T extends Vec4
        ? [x: number, y: number, z: number, w: number]
        : never;

interface FieldWrite<T, D extends AnyTableDef> {
  /**
   * Writes the whole field. A vector takes any array of numbers and reads its first components.
   * Reuse typed inputs to avoid allocating.
   */
  write(id: RowId<D>, value: FieldInput<T>): void;
}

type FieldSet<T, D extends AnyTableDef> = [Components<T>] extends [never]
  ? {}
  : { set(id: RowId<D>, ...components: Components<T>): void };

type FieldGet<T, D extends AnyTableDef> = T extends Scalar
  ? { get(id: RowId<D>): number }
  : [Components<T>] extends [never]
    ? {}
    : {
        /** Copies the field into `out`, a vector or an array of numbers, and returns `out`. */
        get<O extends d.Infer<T> | Float32Array | number[]>(id: RowId<D>, out: O): O;
      };

/** Provides CPU access to a host field. Writes update the CPU copy and reach the GPU at `flush`. */
export type HostField<T, D extends AnyTableDef> = FieldWrite<T, D> &
  FieldSet<T, D> &
  FieldGet<T, D>;

/**
 * Provides CPU writes to a device field. At `flush`, these writes overwrite values written by
 * kernels.
 */
export type DeviceField<T, D extends AnyTableDef> = FieldWrite<T, D> & FieldSet<T, D>;

export type Fields<T extends AnyTableDef> = {
  readonly [K in keyof HostOf<T>]: HostField<HostOf<T>[K], T>;
} & { readonly [K in keyof DeviceOf<T>]: DeviceField<DeviceOf<T>[K], T> };

export interface Options<T extends AnyTableDef> {
  /**
   * Sets the table capacity for its lifetime. Must be an integer in [1, 2^20], with slots from `0`
   * to `capacity - 1`. A row id uses enough bits for the slot and the rest for its generation. The
   * generation has 29 bits at 8 rows and 12 bits at 2^20 rows.
   */
  readonly capacity: number;
  /**
   * Sets values for fields omitted by `add`. Without defaults, every field starts at zero. Default
   * none.
   */
  readonly defaults?: Values<T>;
  /**
   * Keeps last frame's rows, which code compiled with `history.previously` reads, such as a
   * drawing's motion. `create` then returns a `HistoryTable`, whose `flush` takes the frame or its
   * encoder: it copies the rows before the frame's changes, so a row added this frame starts where
   * it is. Default false.
   */
  readonly history?: boolean;
}

/**
 * Provides the capacity and live test for a module that runs over table slots. Spread it into that
 * module's options.
 */
export interface Live {
  /** Reports the table capacity. */
  readonly capacity: number;
  /** Provides the table's `gpu.alive`. */
  readonly alive: (i: number) => boolean;
  /** Returns the table's `span`, so a module dispatches no thread above it. */
  readonly span?: () => number;
}

export interface Table<T extends AnyTableDef> extends Disposable {
  /** Reports the table capacity. */
  readonly capacity: number;
  /** Counts live rows. */
  readonly count: number;
  /** Reports the highest live slot plus one. */
  readonly span: number;
  /**
   * Provides the definition's shader members bound to this table, such as `reds.gpu.position(i)`
   * and `reds.gpu.device.$[i]`. A shader reads several tables of one definition through their
   * `gpu` members, with nothing piped.
   */
  readonly gpu: GpuMembers<T>;
  /**
   * Fills the definition's own members with this table's buffers, for code written over any table
   * of the definition, such as a kind's hooks.
   */
  inject(): Inject;
  /**
   * Returns the capacity and live test for live lists, a grid, a picker or state machines. Use
   * `liveLists.create(root, { ...coins.live(), draws })`.
   */
  live(): Live;
  /**
   * Provides one handle per field. Keep a handle outside a loop instead of reading `fields.x`
   * inside it.
   */
  readonly fields: Fields<T>;

  /** Holds the rows; with `history`, a part's buffer holds last frame's rows from slot `capacity`. */
  readonly buffers: {
    readonly host: (TgpuBuffer<Rows<HostOf<T>>> & StorageFlag) | undefined;
    readonly device: (TgpuBuffer<Rows<DeviceOf<T>>> & StorageFlag) | undefined;
    readonly live: TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag;
    /** Holds the span and bitmask that shaders use to extract slots from ids. */
    readonly index: TgpuUniform<typeof Index>;
  };
  /**
   * Adds a row in the lowest free slot. Uses defaults for omitted fields. Throws when the table is
   * full, or when a value fails to write, which leaves the slot free.
   */
  add(values?: Values<T>): RowId<T>;
  /** Frees the row slot and makes its id stale. */
  remove(id: RowId<T>): void;
  /** Removes every row. */
  clear(): void;
  /** Tests whether `id` names a live row, without throwing for a stale one. */
  has(id: RowId<T>): boolean;
  /** Writes values to several fields. Device field writes overwrite kernel results at `flush`. */
  write(id: RowId<T>, values: Values<T>): void;
  /** Returns the slot for a row id. Throws if the id is stale. */
  slotOf(id: RowId<T>): number;
  /**
   * Returns the id of the row in `slot`, such as a slot read back from the GPU. Throws if the slot
   * is free.
   */
  idAt(slot: number): RowId<T>;
  /**
   * Uploads changes since the last flush with `queue.writeBuffer`. Writes reach the queue before
   * the next submit. Call once per frame before passes that read the table. A `HistoryTable` takes
   * the frame or its encoder and throws without one.
   */
  flush(target?: Frame | TgpuCommandEncoder): void;
  /** Destroys the storage buffers. */
  destroy(): void;
}

/** A table created with `history`, which keeps last frame's rows. */
export interface HistoryTable<T extends AnyTableDef> extends Omit<Table<T>, 'flush'> {
  /**
   * Uploads changes since the last flush and records into `target`, a frame or an encoder, the
   * copies that keep last frame's rows. Call once per frame before passes that read the table. A
   * frame that throws leaves its copies to the next flush; an encoder can't, so a dropped one
   * loses them.
   */
  flush(target: Frame | TgpuCommandEncoder): void;
}

interface Part {
  readonly struct: d.WgslStruct<Props>;
  readonly stride: number;
  readonly buffer: TgpuBuffer<d.WgslArray> & StorageFlag;
  readonly gpu: GPUBuffer;
  readonly cpu: ArrayBuffer;
  readonly views: ByteViews;
  readonly template: Uint8Array<ArrayBuffer>;
  /** Holds last frame's rows and the frame's uploads, with `history`. */
  readonly kept: Kept | undefined;
}

interface Kept {
  readonly staged: TgpuBuffer<d.WgslArray> & StorageFlag;
  /** Carries the rows to last frame's, since WebGPU copies within a buffer only through another. */
  readonly scratch: TgpuBuffer<d.WgslArray> & StorageFlag;
  readonly stagedGpu: GPUBuffer;
  readonly scratchGpu: GPUBuffer;
  /**
   * Byte ranges uploaded since the last submitted flush, as start and end pairs, to copy into the
   * current rows. Those before `recorded` are recorded into an encoder that has not submitted.
   */
  ranges: Uint32Array;
  count: number;
  recorded: number;
}

interface FieldInfo {
  readonly part: Part;
  readonly schema: d.AnyWgslData;
  readonly offset: number;
  readonly device: boolean;
  /** Each device field has one bit in its row's dirty mask: `bit` of the mask's word `word`. */
  readonly word: number;
  readonly bit: number;
  readonly write: Writer;
}

/**
 * Allocates GPU buffers and a CPU copy of host rows. Throws unless capacity is an integer in [1,
 * 2^20].
 *
 * @param def the definition its shaders were written against, from `table.define`
 * @returns a table you destroy
 */
export function create<T extends AnyTableDef>(
  root: TgpuRoot,
  def: T,
  options: Options<T> & { readonly history: true },
): HistoryTable<T>;
export function create<T extends AnyTableDef>(
  root: TgpuRoot,
  def: T,
  options: Options<T>,
): Table<T>;
export function create<T extends AnyTableDef>(
  root: TgpuRoot,
  def: T,
  options: Options<T>,
): Table<T> {
  const { capacity, history = false } = options;
  if (!Number.isInteger(capacity) || capacity < 1 || capacity > MAX_CAPACITY) {
    throw new RangeError(`A table's capacity must be an integer in [1, 2^20], got ${capacity}.`);
  }

  const slotBits = Math.max(1, 32 - Math.clz32(capacity - 1));
  const slotMask = (1 << slotBits) - 1;
  const lastGeneration = -1 >>> slotBits;

  const parts = def['~parts'];
  const { name } = parts;
  const defaults = (options.defaults ?? {}) as Record<string, unknown>;

  const makePart = (struct: d.WgslStruct<Props> | undefined, label: string): Part | undefined => {
    if (!struct) return undefined;

    // Cast the generic struct so TypeGPU can check its storage usage
    const make = (called: string, length: number) =>
      root
        .createBuffer(d.arrayOf(struct as unknown as d.WgslStruct<{ x: d.U32 }>, length))
        .$usage('storage')
        .$name(called) as unknown as Part['buffer'];
    // With `history`, last frame's rows follow the current ones in the same buffer
    const buffer = make(label, history ? capacity * 2 : capacity);

    const stride = d.sizeOf(struct);
    const cpu = new ArrayBuffer(stride * capacity);

    let kept: Kept | undefined;
    if (history) {
      const staged = make(`${label}_staged`, capacity);
      const scratch = make(`${label}_scratch`, capacity);
      kept = {
        staged,
        scratch,
        stagedGpu: root.unwrap(staged),
        scratchGpu: root.unwrap(scratch),
        ranges: new Uint32Array(64),
        count: 0,
        recorded: 0,
      };
    }

    return {
      struct,
      stride,
      buffer,
      gpu: root.unwrap(buffer),
      cpu,
      views: byteViews(cpu),
      template: new Uint8Array(stride),
      kept,
    };
  };

  const host = makePart(parts.host, `${name}_host`);
  const device = makePart(parts.device, `${name}_device`);

  const words = Math.ceil(capacity / 32);
  const liveBits = new Uint32Array(words);
  const fullWords = new Uint32Array(Math.ceil(words / 32));
  let firstOpenGroup = 0;

  // Start generations above 0 so zeroed fields never hold live ids, and at a different value in
  // each table of a definition so an id from another table reads as stale
  const firstGeneration = 1 + ((Math.imul(parts.tables++, 0x9e3779b9) >>> 0) % lastGeneration);
  const generations = new Uint32Array(capacity).fill(firstGeneration);

  const liveBuffer = root
    .createBuffer(d.arrayOf(d.u32, words))
    .$usage('storage')
    .$name(`${name}_live`);
  const liveGpu = root.unwrap(liveBuffer);

  const indexUniform = root
    .createUniform(Index, { span: 0, mask: slotMask })
    .$name(`${name}_index`);
  const indexGpu = root.unwrap(indexUniform.buffer);
  const spanValue = new Uint32Array(1);

  let count = 0;
  let high = 0;

  const destroy = () => {
    for (const part of [host, device]) {
      part?.buffer.destroy();
      part?.kept?.staged.destroy();
      part?.kept?.scratch.destroy();
    }
    liveBuffer.destroy();
  };

  // Host rows written since the last flush. Clean host rows match the GPU, so a flush uploads
  // runs that bridge gaps under `HOST_GAP` rows, bounding the number of `writeBuffer` calls
  const hostDirty = new Uint8Array(host ? capacity : 0);
  const hostSlots = new Uint32Array(host ? capacity : 0);
  let hostCount = 0;
  let liveLow = words;
  let liveHigh = -1;
  let spanDirty = true;

  // Each row has one dirty bit per device field, in as many 32-bit words as the fields need. A new
  // row marks every field, so it uploads whole
  const deviceFields = device ? Object.keys(device.struct.propTypes).length : 0;
  const maskWords = Math.ceil(deviceFields / 32);
  const full = new Uint32Array(maskWords).fill(ALL);
  if (deviceFields % 32 !== 0) full[maskWords - 1] = ALL >>> (32 - (deviceFields % 32));
  const dirty = new Uint32Array(capacity * maskWords);
  // A device field's bytes run to the next field's start, so written fields side by side upload
  // as one run across their padding. The last field runs to the end of the row
  const bounds = new Uint32Array(deviceFields + 1);
  // With `history`, rows added since the last flush start last frame where they are now. Rows a
  // flush recorded are held until its encoder submits, so a frame that throws leaves them to the
  // next
  const fresh = new Uint32Array(history ? capacity : 0);
  const held = new Uint32Array(history ? capacity : 0);
  const listed = new Uint8Array(history ? capacity : 0);
  let freshCount = 0;
  let heldCount = 0;
  // Host rows change only through uploads, so last frame's copy is needed only after one
  let hostMoved = false;
  let hostRecorded = false;
  // The frame whose flush recorded copies, until it finishes
  let recording: Frame | undefined;
  const dirtySlots = new Uint32Array(device ? capacity : 0);
  let dirtyCount = 0;

  const fieldInfo: Record<string, FieldInfo> = {};
  for (const [part, isDevice] of [
    [host, false],
    [device, true],
  ] as const) {
    if (!part) continue;
    const offsets = offsetsOf(part.struct);
    const template = byteViews(part.template.buffer);

    Object.entries(part.struct.propTypes).forEach(([key, schema], index) => {
      const info: FieldInfo = {
        part,
        schema,
        offset: offsets[key],
        device: isDevice,
        word: index >>> 5,
        bit: 1 << (index & 31),
        write: parts.writers[key],
      };

      fieldInfo[key] = info;
      if (defaults[key] !== undefined) info.write(template, info.offset, defaults[key]);
      if (isDevice) bounds[index] = info.offset;
    });
  }
  if (device) bounds[deviceFields] = device.stride;

  const isLive = (slot: number) => ((liveBits[slot >>> 5] >>> (slot & 31)) & 1) === 1;

  const liveSlot = (id: number) => {
    const slot = id & slotMask;
    return slot < capacity && generations[slot] === id >>> slotBits && isLive(slot) ? slot : -1;
  };

  const slotOfId = (id: number) => {
    const slot = liveSlot(id);
    if (slot < 0) {
      throw new Error(`Row id ${id} is stale: its row was removed, or it is from another table.`);
    }

    return slot;
  };

  const idOf = (slot: number) => (((generations[slot] << slotBits) | slot) >>> 0) as RowId<T>;

  const markLive = (word: number) => {
    if (word < liveLow) liveLow = word;
    if (word > liveHigh) liveHigh = word;
  };

  const markHost = (slot: number) => {
    if (hostDirty[slot] !== 0) return;
    hostDirty[slot] = 1;
    hostSlots[hostCount++] = slot;
  };

  /** The nearest written host row within `HOST_GAP` rows past `slot` in direction `step`, or -1. */
  const nearHost = (slot: number, step: 1 | -1) => {
    for (let k = 1; k <= HOST_GAP; k++) {
      const next = slot + k * step;
      if (next < 0 || next >= capacity) return -1;
      if (hostDirty[next] !== 0) return next;
    }
    return -1;
  };

  const isClean = (slot: number) => {
    for (let at = slot * maskWords, end = at + maskWords; at < end; at++) {
      if (dirty[at] !== 0) return false;
    }
    return true;
  };

  const markDevice = (slot: number, word: number, bit: number) => {
    if (isClean(slot)) dirtySlots[dirtyCount++] = slot;
    dirty[slot * maskWords + word] |= bit;
  };

  // The upload being gathered. A run that starts where it ends joins it, so an upload never
  // rewrites a field that a kernel owns
  let runStart = 0;
  let runEnd = 0;
  const run = (start: number, end: number) => {
    if (start !== runEnd) {
      if (runEnd > runStart) upload(device!, runStart, runEnd);
      runStart = start;
    }
    runEnd = end;
  };

  /** Adds the runs of written device fields in `slot`, and marks the row clean. */
  const runsOf = (slot: number, stride: number) => {
    const row = slot * stride;
    for (let word = 0; word < maskWords; word++) {
      const at = slot * maskWords + word;
      let mask = dirty[at];
      dirty[at] = 0;
      while (mask !== 0) {
        const low = 31 - Math.clz32(mask & -mask);
        const above = ~(mask >>> low);
        const length = above === 0 ? 32 : 31 - Math.clz32(above & -above);
        const field = word * 32 + low;
        run(row + bounds[field], row + bounds[field + length]);
        mask = low + length < 32 ? mask & (-1 << (low + length)) : 0;
      }
    }
  };

  const writeFields = (slot: number, values: Record<string, unknown>) => {
    for (const key in values) {
      const value = values[key];
      const info = fieldInfo[key];
      if (value === undefined || !info) continue;

      const offset = slot * info.part.stride + info.offset;
      info.write(info.part.views, offset, value);

      if (!info.device) markHost(slot);
      else markDevice(slot, info.word, info.bit);
    }
  };

  const allocate = () => {
    for (let s = firstOpenGroup; s < fullWords.length; s++) {
      const notFull = ~fullWords[s];
      if (notFull === 0) continue;
      const word = (s << 5) | (31 - Math.clz32(notFull & -notFull));
      if (word >= words) break;

      const free = ~liveBits[word];
      const slot = (word << 5) | (31 - Math.clz32(free & -free));
      if (slot >= capacity) break;
      liveBits[word] |= 1 << (slot & 31);

      if (liveBits[word] === 0xffffffff || (word === words - 1 && slot === capacity - 1)) {
        fullWords[s] |= 1 << (word & 31);
      }

      firstOpenGroup = fullWords[s] === 0xffffffff ? s + 1 : s;
      markLive(word);
      return slot;
    }

    throw new Error(`The table is full (capacity ${capacity}).`);
  };

  const release = (slot: number) => {
    const word = slot >>> 5;
    const group = word >>> 5;
    liveBits[word] &= ~(1 << (slot & 31));
    fullWords[group] &= ~(1 << (word & 31));
    firstOpenGroup = Math.min(firstOpenGroup, group);
    generations[slot] = generations[slot] === lastGeneration ? 1 : generations[slot] + 1;
    markLive(word);
    count -= 1;

    if (slot + 1 === high) {
      while (high > 0 && !isLive(high - 1)) high--;
      spanDirty = true;
    }
  };

  /** Uploads `[start, end)` of a part's CPU copy, through its staging buffer with `history`. */
  const upload = (part: Part, start: number, end: number) => {
    const { kept } = part;
    queue.writeBuffer(kept ? kept.stagedGpu : part.gpu, start, part.cpu, start, end - start);
    if (!kept) return;
    if (kept.count * 2 === kept.ranges.length) {
      const grown = new Uint32Array(kept.ranges.length * 2);
      grown.set(kept.ranges);
      kept.ranges = grown;
    }
    kept.ranges[kept.count * 2] = start;
    kept.ranges[kept.count * 2 + 1] = end;
    kept.count += 1;
  };

  const kept = [host, device].filter(
    (part): part is Part & { kept: Kept } => part?.kept !== undefined,
  );
  const list = (slot: number) => {
    if ((listed[slot] & FRESH) !== 0) return;
    listed[slot] |= FRESH;
    fresh[freshCount++] = slot;
  };

  /** Settles the copies a flush recorded, once their encoder submits or at once for an encoder. */
  const sent = () => {
    for (const part of kept) part.kept.count = part.kept.recorded = 0;
    for (let i = 0; i < heldCount; i++) listed[held[i]] &= ~HELD;
    heldCount = 0;
  };

  /** Starts the next frame's copy of last frame's rows, once the frame that flushed finishes. */
  const finished = () => {
    hostMoved = hostRecorded;
    hostRecorded = false;
    recording = undefined;
  };

  /** Lists again the uploads and new rows of a frame that never finished. */
  const unfinished = () => {
    for (const part of kept) part.kept.recorded = 0;
    for (let i = 0; i < heldCount; i++) {
      listed[held[i]] &= ~HELD;
      list(held[i]);
    }
    heldCount = 0;
    // An early submit may have sent its host uploads
    hostMoved ||= hostRecorded;
    hostRecorded = false;
    recording = undefined;
  };

  /**
   * Copies the current rows to last frame's, unless this frame did already, then the uploads not
   * yet recorded to the current rows, and new rows to both. Host rows that nothing uploaded last
   * frame already match.
   */
  const keep = (raw: GPUCommandEncoder, again: boolean) => {
    if (host?.kept && host.kept.count > 0) hostRecorded = true;
    for (const part of kept) {
      const { stagedGpu, scratchGpu, ranges } = part.kept;
      const back = capacity * part.stride;
      const span = high * part.stride;
      if (!again && span > 0 && (part === device || hostMoved)) {
        raw.copyBufferToBuffer(part.gpu, 0, scratchGpu, 0, span);
        raw.copyBufferToBuffer(scratchGpu, 0, part.gpu, back, span);
      }
      for (let i = part.kept.recorded; i < part.kept.count; i++) {
        const start = ranges[i * 2];
        const end = ranges[i * 2 + 1];
        raw.copyBufferToBuffer(stagedGpu, start, part.gpu, start, end - start);
      }
      part.kept.recorded = part.kept.count;

      // Copy runs of new rows that sit side by side as one
      for (let i = 0; i < freshCount;) {
        const first = fresh[i];
        let last = first;
        while (++i < freshCount && fresh[i] === last + 1) last++;
        const start = first * part.stride;
        raw.copyBufferToBuffer(
          stagedGpu,
          start,
          part.gpu,
          back + start,
          (last + 1) * part.stride - start,
        );
      }
    }
    for (let i = 0; i < freshCount; i++) {
      const slot = fresh[i];
      if ((listed[slot] & HELD) === 0) held[heldCount++] = slot;
      listed[slot] = HELD;
    }
    freshCount = 0;
  };

  const handles: Record<string, unknown> = {};
  for (const [key, info] of Object.entries(fieldInfo)) {
    handles[key] = fieldHandle(info, slotOfId, markHost, markDevice);
  }

  // Restore the definition types after creating part buffers with a general struct type
  const liveRows = liveBuffer.as('readonly');
  const hostRows = host && (host.buffer.as('readonly') as never);
  const deviceRows = device && (byStage(device.buffer) as never);
  const back = temporal(0, capacity);

  const queue = root.device.queue;

  const fill: Inject = (cfg) => {
    const counted = cfg.with(parts.live, liveRows).with(parts.index, indexUniform);
    const result = history ? counted.with(parts.back, back) : counted;
    const withHost = hostRows ? result.with(def.host, hostRows) : result;
    return deviceRows ? withHost.with(def.device, deviceRows) : withHost;
  };
  // Accessors filled with the same values as the definition's, so a pipeline that reads both
  // binds each buffer once
  const members = def as unknown as Record<string, { pipe(fill: Inject): unknown }>;
  const gpu: Record<string, unknown> = {
    host: tgpu.accessor(def.host.schema, hostRows).$name(`${name}_host`),
    device: tgpu.accessor(def.device.schema, deviceRows).$name(`${name}_device`),
  };
  for (const key of [...Object.keys(fieldInfo), 'row', 'alive', 'span', 'slotOf']) {
    gpu[key] = members[key].pipe(fill);
  }
  const bound = gpu as GpuMembers<T>;
  const live: Live = { capacity, alive: bound.alive, span: () => high };

  const table: Table<T> = {
    capacity,

    get count() {
      return count;
    },

    get span() {
      return high;
    },

    gpu: bound,

    inject: () => fill,

    live: () => live,

    fields: handles as Fields<T>,
    buffers: {
      host: host?.buffer as Table<T>['buffers']['host'],
      device: device?.buffer as Table<T>['buffers']['device'],
      live: liveBuffer,
      index: indexUniform,
    },

    add(values) {
      const slot = allocate();
      count += 1;

      if (slot >= high) {
        high = slot + 1;
        spanDirty = true;
      }

      if (host) {
        host.views.bytes.set(host.template, slot * host.stride);
        markHost(slot);
      }

      if (device) {
        device.views.bytes.set(device.template, slot * device.stride);
        for (let word = 0; word < maskWords; word++) markDevice(slot, word, full[word]);
      }
      if (history) list(slot);

      if (values) {
        try {
          writeFields(slot, values as Record<string, unknown>);
        } catch (error) {
          release(slot);
          throw error;
        }
      }
      return idOf(slot);
    },

    remove(id) {
      release(slotOfId(id));
    },

    clear() {
      for (let slot = 0; slot < high; slot++) {
        if (isLive(slot)) release(slot);
      }
    },

    has(id) {
      return liveSlot(id) >= 0;
    },

    write(id, values) {
      writeFields(slotOfId(id), values as Record<string, unknown>);
    },

    slotOf: slotOfId,

    idAt(slot) {
      if (slot >= capacity || !isLive(slot)) {
        throw new Error(`Slot ${slot} holds no row.`);
      }
      return idOf(slot);
    },

    flush(target) {
      if (history && !target) {
        throw new Error(
          `The table '${name}' keeps history and needs the frame when you flush it. Call \`flush(frame)\`.`,
        );
      }
      const frame = target && 'encoder' in target ? target : undefined;
      if (recording && recording !== frame) unfinished();
      if (host) {
        for (let i = 0; i < hostCount; i++) {
          let first = hostSlots[i];
          if (hostDirty[first] === 0) continue;

          for (let near = nearHost(first, -1); near >= 0; near = nearHost(first, -1)) first = near;
          let last = first;
          for (let near = nearHost(last, 1); near >= 0; near = nearHost(last, 1)) last = near;

          hostDirty.fill(0, first, last + 1);
          upload(host, first * host.stride, (last + 1) * host.stride);
        }
        hostCount = 0;
      }

      if (device) {
        for (let i = 0; i < dirtyCount; i++) {
          let slot = dirtySlots[i];
          if (isClean(slot)) continue;

          // Walk written rows side by side in order, so their runs that meet upload as one
          while (slot > 0 && !isClean(slot - 1)) slot--;
          for (; slot < capacity && !isClean(slot); slot++) runsOf(slot, device.stride);
        }
        if (runEnd > runStart) upload(device, runStart, runEnd);
        runStart = runEnd = 0;
        dirtyCount = 0;
      }

      if (liveHigh >= liveLow) {
        const start = liveLow * 4;
        queue.writeBuffer(liveGpu, start, liveBits.buffer, start, (liveHigh + 1) * 4 - start);
        liveLow = words;
        liveHigh = -1;
      }

      if (spanDirty) {
        spanValue[0] = high;
        queue.writeBuffer(indexGpu, 0, spanValue.buffer, 0, 4);
        spanDirty = false;
      }

      if (history && target) {
        keep(root.unwrap(encoderOf(target)), recording !== undefined);
        if (!frame) {
          sent();
          finished();
        } else {
          afterEncoderSubmit(frame, sent);
          if (!recording) {
            recording = frame;
            frame.afterSubmit(finished);
          }
        }
      }
    },

    destroy,
    [Symbol.dispose]: destroy,
  };

  return table;
}

function fieldHandle(
  info: FieldInfo,
  slotOfId: (id: number) => number,
  markHost: (slot: number) => void,
  markDevice: (slot: number, word: number, bit: number) => void,
) {
  const { part, schema, word, bit } = info;
  const { views, stride } = part;
  const scalar = scalarOf(schema);
  const components = componentsOf(schema);

  const touch = info.device
    ? (slot: number) => markDevice(slot, word, bit)
    : (slot: number) => markHost(slot);

  const handle: Record<string, unknown> = {
    write(id: number, value: unknown) {
      const slot = slotOfId(id);
      const offset = slot * stride + info.offset;
      info.write(views, offset, value);
      touch(slot);
    },
  };

  if (!scalar) return handle;

  // Separate arities avoid allocating an argument array for each set
  if (components === 1) {
    handle.set = (id: number, x: number) => {
      const slot = slotOfId(id);
      const offset = slot * stride + info.offset;
      writeScalar(views, scalar, offset, x);
      touch(slot);
    };
  } else if (components === 2) {
    handle.set = (id: number, x: number, y: number) => {
      const slot = slotOfId(id);
      const offset = slot * stride + info.offset;
      writeScalar(views, scalar, offset, x);
      writeScalar(views, scalar, offset + 4, y);
      touch(slot);
    };
  } else if (components === 3) {
    handle.set = (id: number, x: number, y: number, z: number) => {
      const slot = slotOfId(id);
      const offset = slot * stride + info.offset;
      writeScalar(views, scalar, offset, x);
      writeScalar(views, scalar, offset + 4, y);
      writeScalar(views, scalar, offset + 8, z);

      touch(slot);
    };
  } else {
    handle.set = (id: number, x: number, y: number, z: number, w: number) => {
      const slot = slotOfId(id);
      const offset = slot * stride + info.offset;
      writeScalar(views, scalar, offset, x);
      writeScalar(views, scalar, offset + 4, y);
      writeScalar(views, scalar, offset + 8, z);
      writeScalar(views, scalar, offset + 12, w);

      touch(slot);
    };
  }

  if (info.device) return handle;

  const at = (id: number) => slotOfId(id) * stride + info.offset;
  handle.get =
    components === 1
      ? (id: number) => readScalar(views, scalar, at(id))
      : (id: number, out: Record<number, number>) => {
          const offset = at(id);
          for (let i = 0; i < components; i++) out[i] = readScalar(views, scalar, offset + i * 4);
          return out;
        };

  return handle;
}
