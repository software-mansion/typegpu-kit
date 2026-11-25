// oxlint-disable no-underscore-dangle -- the generator overrides WgslGenerator's protected hooks, named with one
import { NodeTypeCatalog as NODE, type Const, type Expression, type Statement } from 'tinyest';
import { d, tgpu } from 'typegpu';
import {
  getName,
  snip,
  UnknownData,
  withValue,
  type FunctionDefinitionOptions,
  type ResolutionCtx,
  type ResolvedSnippet,
  type ResolvedStatement,
  type Snippet,
  type VariableDefinitionOptions,
} from 'typegpu/~internal';
import {
  COMPUTE,
  COMPUTE_HEADER,
  CPU_ONLY,
  GPU_FORM,
  INVOCATIONS,
  LAYER,
  LAYERS_PER_DRAW,
  ACCUMULATE,
  type Operation,
  SCATTER,
  TARGET,
  WORKGROUPS,
} from './computeModule.ts';
import {
  attributeOf,
  KitGlslGenerator,
  memberName,
  membersOf,
  strideOf,
  undecorate,
  wordsOf,
  type BufferInfo,
  type Element,
  type Place,
  type TextureInfo,
} from './generator.ts';
import { ADD_MARK, flowOf, LOAD_MARK } from './flow.ts';
import { CHUNK } from './renderModule.ts';

type AnyFn = Parameters<KitGlslGenerator['call']>[0];

/** Why a kernel cannot run as fragment passes. The pipeline then runs on the CPU. */
export class Ineligible extends Error {
  constructor(reason: string) {
    super(`GPU compute: ${reason}`);
  }
}

/**
 * How the elements of a written array map to invocations: element
 * `x + y * width + z * width * height` belongs to invocation `(x, y, z)`. Without a width, element
 * `x` belongs to invocation `(x, 0, 0)`.
 */
export interface Layout {
  readonly width?: number;
  readonly height?: number;
  /**
   * Present when the index is that count times `stride`, plus a phase below the stride, so only
   * every `stride`-th element is the invocation's.
   */
  readonly stride?: number;
  /** GLSL `uint` text added to the index: every invocation sees it alike. */
  readonly phase?: string;
}

/** A buffer a kernel writes. */
export interface WrittenPlan {
  /** The invocation owns the whole binding, as the dispatch's only one. */
  readonly whole: boolean;
  /** The written array's first word, for a buffer written element by element. */
  readonly offset: number;
  readonly layout: Layout;
  /**
   * With `whole`, when one invocation chosen by values every invocation shares writes the binding:
   * its `x`, then `y` and `z` if the choice names them, as GLSL `uint` expressions.
   */
  readonly chosen?: readonly string[];
}

/** What a kernel writes: found by a first resolution, used by the second. */
export interface ComputePlan {
  readonly buffers: ReadonlyMap<string, WrittenPlan>;
  /** Each written storage texture's id. */
  readonly textures: ReadonlySet<string>;
  /** The written 3d and 2d-array textures whose invocations own the column at their `xy`. */
  readonly columns: ReadonlySet<string>;
  /** The written textures whose cells are not plainly the invocation's, by their texel maps. */
  readonly maps: ReadonlyMap<string, TexelMap>;
  /** The buffers and textures written at cells the kernel computes, a pass of points each. */
  readonly scatters: ReadonlySet<string>;
  /** The buffers the kernel only adds to with atomics whose results it drops, a pass each. */
  readonly adds?: ReadonlySet<string>;
  /** The most additions an invocation makes to each such buffer, which bounds its records. */
  readonly slots?: ReadonlyMap<string, number>;
  /** The most writes an invocation makes at computed cells of each such resource. */
  readonly records?: ReadonlyMap<string, number>;
  /** The counters the kernel takes values of, among `adds`. */
  readonly takes?: ReadonlySet<string>;
  /** The arrays invocations own constant blocks of, written through helpers. */
  readonly blocks?: ReadonlyMap<string, BlockPlan>;
}

/**
 * An array each invocation owns a constant 2-d block of: element `row * width + col` belongs to
 * invocation `g` at site `s` when `col = step[0] * g.x + sites[s][0] + phase[0]` and `row =
 * step[1] * g.y + sites[s][1] + phase[1]`, with `col` below `width`. Phases are GLSL `int`
 * text every invocation sees alike.
 */
export interface BlockPlan {
  readonly offset: number;
  readonly width: number;
  readonly step: readonly [number, number];
  readonly phase: readonly [string, string];
  readonly sites: readonly (readonly [number, number])[];
}

interface Block extends BlockPlan {
  readonly buffer: BufferInfo;
  readonly array: d.WgslArray;
}

/** A helper's write at `row * width + col` of a vector parameter, guarded to its row's width. */
interface BlockWrite {
  readonly param: number;
  readonly buffer: BufferInfo;
  readonly array: d.WgslArray;
  readonly offset: number;
  readonly width: number;
}

/** The most cells of one array an invocation may own as a block. */
const MOST_SITES = 8;

/** A resource written at computed cells: an array of a buffer, or a texture. */
type Scatter =
  | {
      readonly kind: 'buffer';
      readonly buffer: BufferInfo;
      readonly array: d.WgslArray;
      readonly offset: number;
    }
  | { readonly kind: 'texture'; readonly texture: TextureInfo };

/**
 * How a texture's cells map to invocations, coordinate by coordinate: coordinate `i` of the cell
 * invocation `gid` owns is `step * gid[i]`, plus `r.of & r.mask` when there is a residue, which
 * stays below `step` and does not depend on `gid[i]`.
 */
type TexelMap = readonly { readonly step: number; readonly r?: Affine['r'] }[];

/** The texel map own coordinates give, if every cell belongs to at most one invocation. */
function texelMapOf(own: Own): TexelMap | undefined {
  const map: { step: number; r?: Affine['r'] }[] = [];
  let residues = 0;
  for (const [i, each] of own.entries()) {
    const name = AXES[i] as 'x' | 'y' | 'z';
    const step = each[name];
    const others = [...AXES].filter((c) => c !== name) as ('x' | 'y' | 'z')[];
    if (step < 1 || each.k !== 0 || each.s || others.some((c) => each[c] !== 0)) return undefined;
    if (each.r) {
      if (each.r.mask >= step || dependsOn(each.r.of, name)) return undefined;
      residues++;
    }
    map.push({ step, ...(each.r && { r: each.r }) });
  }
  return residues <= 1 ? map : undefined;
}

const plainMap = (map: TexelMap) => map.every(({ step, r }) => step === 1 && !r);

const GID = 'kit_gid';
/** Which resource written at computed cells the pass of points draws, by its index. */
const AIM = 'kit_aim';
/** An invocation's additions so far, and their words and amounts. */
const ADDED = 'kit_added';
const ADD = 'kit_add';
/** Takes the next of a counter's values in invocation order; see `TAKES`. */
const TAKE = 'kit_take';
/** Whether the counts of earlier invocations' takes are known: 0 while the device counts them. */
const RANKED = 'kit_ranked';
/** Stands in for a read of a buffer the kernel writes at computed cells, which no pass can see. */
const SCATTERED = 'kit_scattered_';
/** The layer of a layered target the pass computes now, of the several one draw writes. */
const Z = 'kit_z';
const BEGIN = 'kit_begin';
const RESULT = 'kit_result';

/** Stands in for a read of a whole written array, which would see none of the invocation's writes. */
const STALE = 'kit_stale_';

/** The largest binding a single invocation may own whole, so its packing stays small. */
const MAX_WHOLE_BYTES = 1024;

/** Storage texture formats WebGL 2 renders to, the float ones with `EXT_color_buffer_float`. */
const RENDERABLE: ReadonlySet<string> = new Set([
  'rgba8unorm',
  'rgba8uint',
  'rgba8sint',
  'rgba16uint',
  'rgba16sint',
  'rgba16float',
  'r32uint',
  'r32sint',
  'r32float',
  'rg32uint',
  'rg32sint',
  'rg32float',
  'rgba32uint',
  'rgba32sint',
  'rgba32float',
]);

const AXES = 'xyz';

/** Atomic read-modify-writes, as the new value from the old one `o` and the operand `v`. */
const ATOMICS: Record<string, string> = {
  atomicAdd: 'o + v',
  atomicSub: 'o - v',
  atomicMax: 'max(o, v)',
  atomicMin: 'min(o, v)',
  atomicAnd: 'o & v',
  atomicOr: 'o | v',
  atomicXor: 'o ^ v',
  atomicExchange: 'v',
};

/**
 * The atomics whose results a kernel may drop on cells other invocations share, by how a pass of
 * points combines their records with the words: their codes in the records' helper, too.
 */
const COMBINED: Record<string, readonly [Operation, number]> = {
  atomicAdd: ['add', 0],
  atomicSub: ['add', 0],
  atomicMin: ['min', 1],
  atomicMax: ['max', 2],
  atomicOr: ['or', 3],
  atomicAnd: ['and', 4],
};

/** A term of an index that every invocation sees alike, as GLSL, added or subtracted. */
interface Term {
  readonly text: string;
  readonly sign: 1 | -1;
  /** The settled variables it reads, which must then never be reassigned. */
  readonly from: readonly string[];
}

/**
 * `x * gid.x + y * gid.y + z * gid.z + k`, in integers, plus `s`, terms every invocation sees
 * alike, and `r`, `of & mask`, a residue below `mask + 1` of another such form.
 */
interface Affine {
  readonly x: number;
  readonly y: number;
  readonly z: number;
  readonly k: number;
  readonly s?: readonly Term[];
  readonly r?: { readonly of: Affine; readonly mask: number };
}

/** A value that provably derives from `global_invocation_id`: one affine form per component. */
type Own = readonly Affine[];

/** A value every invocation sees alike, as the GLSL text it was read from. */
interface Shared {
  readonly shared: string;
  readonly from: readonly string[];
  /** Present when the value is known to stay below this, as a mask leaves it. */
  readonly below?: number;
}

/** The settled variables an own value's shared terms read. */
function fromOf(own: Own): string[] {
  return own.flatMap((each) => [
    ...(each.s ?? []).flatMap((term) => term.from),
    ...(each.r ? fromOf([each.r.of]) : []),
  ]);
}

const constant = (k: number): Affine => ({ x: 0, y: 0, z: 0, k });
const axis = (name: 'x' | 'y' | 'z'): Affine => ({ ...constant(0), [name]: 1 });
const ID: Own = [axis('x'), axis('y'), axis('z')];
const plain = (a: Affine) => a.s === undefined && a.r === undefined;

/** The axes an own value is, component by component, such as `xy`, when each is one plain axis. */
function axesOf(own: Own | undefined): string | undefined {
  if (!own) return undefined;
  let axes = '';
  for (const each of own) {
    const { x, y, z, k } = each;
    if (!plain(each) || k !== 0 || x + y + z !== 1 || [x, y, z].some((c) => c !== 0 && c !== 1)) {
      return undefined;
    }
    axes += x ? 'x' : y ? 'y' : 'z';
  }
  return axes;
}

/** An affine form over `gid` as GLSL `uint`. */
function affineText(a: Affine, gid: string): string {
  const parts: string[] = [];
  for (const [c, n] of [
    ['x', a.x],
    ['y', a.y],
    ['z', a.z],
  ] as const) {
    if (n === 0) continue;
    if (n < 0) return '';
    parts.push(n === 1 ? `${gid}.${c}` : `${gid}.${c} * ${n}u`);
  }
  let text = parts.length > 0 ? parts.join(' + ') : '0u';
  if (a.k > 0) text += ` + ${a.k}u`;
  if (a.k < 0) text += ` - ${-a.k}u`;
  for (const term of a.s ?? []) text += ` ${term.sign > 0 ? '+' : '-'} uint(${term.text})`;
  if (a.r) text += ` + ((${affineText(a.r.of, gid)}) & ${a.r.mask}u)`;
  return `(${text})`;
}

/** Whether an affine form depends on an axis of the invocation, through its residue too. */
function dependsOn(a: Affine, on: 'x' | 'y' | 'z'): boolean {
  return a[on] !== 0 || (a.r !== undefined && dependsOn(a.r.of, on));
}

/** The layout an own array index is, if it gives each invocation its own element. */
function layoutOf(own: Own | undefined): Layout | undefined {
  if (own?.length !== 1) return undefined;
  const a = own[0] as Affine;
  const { x: stride, y, z, k } = a;
  if (stride < 1 || y < 0 || z < 0 || y % stride !== 0) return undefined;
  if (a.r && (dependsOn(a.r.of, 'x') || dependsOn(a.r.of, 'y') || dependsOn(a.r.of, 'z'))) {
    return undefined;
  }
  // With a stride, the phase must stay below it, so no two invocations meet
  if (stride > 1) {
    const residue = a.r && a.r.mask < stride && k === 0;
    const offset = !a.r && k >= 0 && k < stride;
    if (a.s || !(residue || offset)) return undefined;
  }
  const phase = affineText({ ...a, x: 0, y: 0, z: 0 }, '');
  const shape: { stride?: number; phase?: string } = {
    ...(stride > 1 && { stride }),
    ...(phase !== '(0u)' && { phase }),
  };
  if (y === 0) return z === 0 ? shape : undefined;
  const width = y / stride;
  if (z === 0) return { width, ...shape };
  return z % y === 0 ? { width, height: z / y, ...shape } : undefined;
}

const sameLayout = (a: Layout, b: Layout) =>
  a.width === b.width && a.height === b.height && a.stride === b.stride && a.phase === b.phase;

/** The element index a layout gives the invocation `gid` stands for, as GLSL. */
function linear(layout: Layout, gid: string): string {
  let index: string;
  if (layout.width === undefined) index = `${gid}.x`;
  else {
    const rows = `${gid}.y * ${layout.width}u`;
    index =
      layout.height === undefined
        ? `(${gid}.x + ${rows})`
        : `(${gid}.x + ${rows} + ${gid}.z * ${layout.width * layout.height}u)`;
  }
  if (layout.stride !== undefined) index = `(${index} * ${layout.stride}u)`;
  return layout.phase === undefined ? index : `(${index} + ${layout.phase})`;
}

/** The invocation an element's index counts, without stride or phase, as GLSL. */
function counted(layout: Layout, element: string): string {
  const { width, height } = layout;
  if (width === undefined) return `uvec3(${element}, 0u, 0u)`;
  if (height === undefined) return `uvec3(${element} % ${width}u, ${element} / ${width}u, 0u)`;
  return `uvec3(${element} % ${width}u, (${element} / ${width}u) % ${height}u, ${element} / ${width * height}u)`;
}

/**
 * The invocation an element belongs to under a layout, as GLSL, or one past every dispatch when
 * no invocation owns it.
 */
function invocationOf(layout: Layout, element: string): string {
  // Wrapping like the kernel's own index arithmetic
  const local = layout.phase === undefined ? element : `(${element} - ${layout.phase})`;
  if (layout.stride === undefined) return counted(layout, local);
  return `((${local}) % ${layout.stride}u == 0u ? ${counted(layout, `((${local}) / ${layout.stride}u)`)} : uvec3(0xFFFFFFFFu))`;
}

type Value = Own | number | Shared;

const isShared = (value: Value): value is Shared => typeof value === 'object' && 'shared' in value;

/** A value as an own one, a shared value becoming a term. */
function ownOf(value: Own | number | Shared): Own {
  if (typeof value === 'number') return [constant(value)];
  if (isShared(value)) {
    const term: Affine = { ...constant(0), s: [{ text: value.shared, sign: 1, from: value.from }] };
    return value.below === undefined
      ? [term]
      : [{ ...constant(0), r: { of: term, mask: value.below - 1 } }];
  }
  return value;
}

/**
 * A value as an own one of `size` components: a vector's own as it is, a number or a shared
 * scalar for each component, and a shared vector one component each.
 */
function spread(value: Own | number | Shared, size: number): Own {
  if (Array.isArray(value) || size === 1) return ownOf(value);
  if (typeof value === 'number') return Array.from({ length: size }, () => constant(value));
  const shared = value as Shared;
  return Array.from({ length: size }, (_, i) => ({
    ...constant(0),
    s: [{ text: `(${shared.shared}).${'xyzw'[i]}`, sign: 1 as const, from: shared.from }],
  }));
}

function scale(value: Own, by: number): Own | undefined {
  if (by === 1) return value;
  const scaled: Affine[] = [];
  for (const each of value) {
    if (each.r) return undefined;
    if (each.s && by !== -1) return undefined;
    scaled.push({
      x: each.x * by,
      y: each.y * by,
      z: each.z * by,
      k: each.k * by,
      ...(each.s && { s: each.s.map((term) => ({ ...term, sign: -term.sign as 1 | -1 })) }),
    });
  }
  return scaled;
}

function add(left: Own, right: Own): Own | undefined {
  if (left.length !== right.length) return undefined;
  const sum: Affine[] = [];
  for (const [i, each] of left.entries()) {
    const other = right[i] as Affine;
    if (each.r && other.r) return undefined;
    const s = [...(each.s ?? []), ...(other.s ?? [])];
    const r = each.r ?? other.r;
    sum.push({
      x: each.x + other.x,
      y: each.y + other.y,
      z: each.z + other.z,
      k: each.k + other.k,
      ...(s.length > 0 && { s }),
      ...(r && { r }),
    });
  }
  return sum;
}

/** Whether a `for…of` iterates `tgpu.unroll(…)`, which TypeGPU unrolls as it generates. */
/**
 * The class of what `tgpu.unroll` gives a shader, which TypeGPU expands a `for…of` over: taken
 * from the real `tgpu.unroll` by calling its code generation once, since TypeGPU does not export
 * it. Undefined if that changes, and then no loop counts as unrolled.
 */
const UNROLLED: (abstract new (...args: never[]) => unknown) | undefined = (() => {
  const callable = tgpu.unroll as unknown as Record<symbol, unknown>;
  for (const key of Object.getOwnPropertySymbols(callable)) {
    const generate = callable[key];
    if (typeof generate !== 'function') continue;
    try {
      const made = (generate as (ctx: unknown, args: unknown[]) => { value?: unknown })(undefined, [
        snip(0, d.u32, 'constant'),
      ]);
      const kind = (made?.value as object | undefined)?.constructor;
      if (typeof kind === 'function' && kind.name === 'UnrollableIterable') {
        return kind as abstract new (...args: never[]) => unknown;
      }
    } catch {
      // Not the code generation
    }
  }
  return undefined;
})();

/** GLSL text with the settled variables of `frame` it names replaced by their values. */
function inlined(text: string, frame: Frame): string {
  return text.replace(/(?<![\w.])[A-Za-z_]\w*/g, (name) => {
    const known = frame.settled.get(name);
    return known ? `(${known.value})` : name;
  });
}

/** `text` split at `separator` where it is outside parentheses and brackets. */
function splitTop(text: string, separator: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < text.length; i++) {
    const c = text[i] as string;
    if (c === '(' || c === '[') depth++;
    else if (c === ')' || c === ']') depth--;
    else if (depth === 0 && text.startsWith(separator, i)) {
      const before = text[i - 1];
      const after = text[i + separator.length];
      // `==` must not be part of `<=`, `>=`, `!=` or `===`
      if (separator === '==' && (/[<>!=]/.test(before ?? '') || after === '=')) continue;
      parts.push(text.slice(start, i));
      start = i + separator.length;
      i = start - 1;
    }
  }
  parts.push(text.slice(start));
  return parts;
}

/** The terms of a GLSL disjunction, however its `||`s are grouped. */
function disjuncts(text: string): string[] {
  const parts = splitTop(unwrap(text), '||');
  return parts.length === 1 ? parts : parts.flatMap(disjuncts);
}

/** Whether a statement is a `return`, alone or as a block's only statement. */
function returns(statement: Statement): boolean {
  if (!Array.isArray(statement)) return false;
  if (statement[0] === NODE.return) return true;
  const body = statement[1] as readonly Statement[];
  return statement[0] === NODE.block && body.length === 1 && returns(body[0] as Statement);
}

/** The terms of a GLSL conjunction, however its `&&`s are grouped. */
function conjuncts(text: string): string[] {
  const parts = splitTop(unwrap(text), '&&');
  return parts.length === 1 ? parts : parts.flatMap(conjuncts);
}

/** `text` without parentheses around all of it. */
function unwrap(text: string): string {
  let inner = text;
  while (inner.startsWith('(') && inner.endsWith(')')) {
    let depth = 0;
    let closes = -1;
    for (let i = 0; i < inner.length; i++) {
      if (inner[i] === '(') depth++;
      else if (inner[i] === ')' && --depth === 0) {
        closes = i;
        break;
      }
    }
    if (closes !== inner.length - 1) break;
    inner = inner.slice(1, -1);
  }
  return inner;
}

/**
 * Reads GLSL integer arithmetic over `names` into affine forms: sums, differences, products with
 * constants, masks with constants, casts and vector constructors, and values `shared` accepts,
 * which every invocation sees alike. Anything else is not provably own.
 */
class Arithmetic {
  readonly #text: string;
  readonly #names: ReadonlyMap<string, Own>;
  readonly #shared: ((text: string) => readonly string[] | undefined) | undefined;
  #at = 0;

  constructor(
    text: string,
    names: ReadonlyMap<string, Own>,
    shared?: (text: string) => readonly string[] | undefined,
  ) {
    this.#text = text;
    this.#names = names;
    this.#shared = shared;
  }

  read(): Value | undefined {
    const value = this.#sum();
    return this.#at === this.#text.length ? value : undefined;
  }

  #peek(): string | undefined {
    return this.#text[this.#at];
  }

  /** The value of the text from `start`, read whole, when neither part is own. */
  #whole(start: number, a: Value, b: Value, combine: (a: number, b: number) => number): Value {
    if (typeof a === 'number' && typeof b === 'number') return combine(a, b);
    const from = [a, b].flatMap((each) => (isShared(each) ? each.from : []));
    return { shared: this.#text.slice(start, this.#at), from };
  }

  #sum(): Value | undefined {
    const start = this.#at;
    let value = this.#product();
    while (value !== undefined && (this.#peek() === '+' || this.#peek() === '-')) {
      const sign = this.#text[this.#at++] === '+' ? 1 : -1;
      const other = this.#product();
      if (other === undefined) return undefined;
      if (Array.isArray(value) || Array.isArray(other)) {
        const size = Math.max(
          Array.isArray(value) ? value.length : 1,
          Array.isArray(other) ? other.length : 1,
        );
        const right = scale(spread(other, size), sign);
        value = right && add(spread(value, size), right);
      } else value = this.#whole(start, value, other, (a, b) => a + sign * b);
    }
    return value;
  }

  #product(): Value | undefined {
    const start = this.#at;
    let value = this.#factor();
    while (value !== undefined && this.#peek() === '*') {
      this.#at++;
      const other = this.#factor();
      if (other === undefined) return undefined;
      if (Array.isArray(value) && typeof other === 'number') value = scale(value, other);
      else if (Array.isArray(other) && typeof value === 'number') value = scale(other, value);
      else if (Array.isArray(value) || Array.isArray(other)) return undefined;
      else value = this.#whole(start, value, other, (a, b) => a * b);
    }
    return value;
  }

  #factor(): Value | undefined {
    const start = this.#at;
    const rest = this.#text.slice(this.#at);
    const number = /^(\d+)u?(?![\w.])/.exec(rest);
    if (number) {
      this.#at += number[0].length;
      return Number(number[1]);
    }
    if (rest.startsWith('(')) {
      this.#at++;
      const value = this.#sum();
      if (value !== undefined && (this.#peek() === '&' || this.#peek() === '%')) {
        const op = this.#text[this.#at++];
        const from = this.#at;
        const right = this.#sum();
        const literal = this.#text.slice(from, this.#at);
        if (this.#peek() !== ')') return undefined;
        this.#at++;
        if (op === '&') return this.#masked(start, value, right);
        // A remainder by a power of two is a mask, in unsigned arithmetic, which a `u` literal forces
        if (typeof right !== 'number' || !/^\d+u$/.test(literal) || right < 1) return undefined;
        return (right & (right - 1)) === 0 ? this.#masked(start, value, right - 1) : undefined;
      }
      if (this.#peek() !== ')') return undefined;
      this.#at++;
      return value;
    }
    const call = /^(uint|int|[iu]vec[23])\(/.exec(rest);
    if (call) {
      this.#at += call[0].length;
      const parts: Value[] = [];
      for (;;) {
        const part = this.#sum();
        if (part === undefined) return undefined;
        parts.push(part);
        if (this.#peek() === ',') this.#at++;
        else break;
      }
      if (this.#peek() !== ')') return undefined;
      this.#at++;
      if (call[1] === 'uint' || call[1] === 'int') {
        if (parts.length !== 1) return undefined;
        const [part] = parts as [Value];
        return isShared(part)
          ? { shared: this.#text.slice(start, this.#at), from: part.from }
          : part;
      }
      const size = Number(call[1]?.at(-1));
      if (
        parts.every((part) => typeof part === 'number') &&
        call[1] !== 'uint' &&
        call[1] !== 'int'
      ) {
        // A vector of one number has it in every component
        const numbers = parts.length === 1 ? Array<number>(size).fill(parts[0] as number) : parts;
        if (numbers.length === size) return (numbers as number[]).map(constant);
      }
      if (parts.every((part) => !Array.isArray(part))) {
        return this.#atom(start, this.#text.slice(start, this.#at));
      }
      const own = parts.flatMap(ownOf);
      return own.length === Number(call[1]?.at(-1)) ? own : undefined;
    }
    const name = /^([A-Za-z_]\w*)(?:\.([xyzw]+))?(?![\w(.[])/.exec(rest);
    const own = name && this.#names.get(name[1] as string);
    if (name && own) {
      this.#at += name[0].length;
      if (!name[2]) return own;
      const picked = [...name[2]].map((c) => own['xyzw'.indexOf(c)]);
      return picked.every(Boolean) ? (picked as Affine[]) : undefined;
    }
    const length = atomLength(rest);
    if (length === 0) return undefined;
    this.#at += length;
    return this.#atom(start, rest.slice(0, length));
  }

  /** A value of the text that is not own: shared when `shared` accepts it. */
  #atom(start: number, text: string): Value | undefined {
    const from = this.#shared?.(text);
    if (!from) {
      this.#at = start;
      return undefined;
    }
    return { shared: text, from };
  }

  /** `(value & mask)`: a residue below `mask + 1`, when the mask is one less than a power of two. */
  #masked(start: number, value: Value, mask: Value | undefined): Value | undefined {
    if (typeof mask !== 'number' || mask < 0 || (mask & (mask + 1)) !== 0) return undefined;
    if (typeof value === 'number') return value & mask;
    if (isShared(value)) {
      return { shared: this.#text.slice(start, this.#at), from: value.from, below: mask + 1 };
    }
    if (value.length !== 1) return undefined;
    const of = value[0] as Affine;
    if (of.r) return undefined;
    return [{ ...constant(0), r: { of, mask } }];
  }
}

/** The length of an identifier with the calls, members and indices after it, or 0. */
function atomLength(text: string): number {
  const name = /^[A-Za-z_]\w*/.exec(text);
  if (!name) return 0;
  let at = name[0].length;
  for (;;) {
    const c = text[at];
    if (c === '(' || c === '[') {
      let depth = 0;
      for (; at < text.length; at++) {
        if (text[at] === '(' || text[at] === '[') depth++;
        else if ((text[at] === ')' || text[at] === ']') && --depth === 0) break;
      }
      if (depth !== 0) return 0;
      at++;
    } else if (c === '.') {
      const member = /^\.[A-Za-z_]\w*/.exec(text.slice(at));
      if (!member) return at;
      at += member[0].length;
    } else return at;
  }
}

interface WrittenBuffer extends WrittenPlan {
  readonly buffer: BufferInfo;
  /** The array written element by element, unless the binding is owned whole. */
  readonly array?: d.WgslArray;
}

/** What one function knows: the names that hold own values, and whether it is the entry. */
interface Frame {
  readonly name: string;
  readonly names: Map<string, Own>;
  readonly entry: boolean;
  /** Its parameters and local variables, which may differ between invocations. */
  readonly locals: Set<string>;
  /**
   * Variables first set to what every invocation sees alike, which may be reassigned later: the
   * others of them their value came from, and that value with theirs inlined.
   */
  readonly settled: Map<string, { readonly from: readonly string[]; readonly value: string }>;
  /** Variables it assigns after declaring them. */
  readonly assigned: Set<string>;
  /** The settled variables a choice of invocation depends on. */
  readonly choosing: Set<string>;
  /** Whether its code depends on its arguments being the invocation's own. */
  relies: boolean;
  /** The most additions to shared cells it makes per buffer, its callees' included. */
  readonly adds: Map<string, number>;
  /** The most writes at computed cells it makes per resource, its callees' included. */
  readonly records: Map<string, number>;
  /** Its parameters' names, in order. */
  readonly params: readonly string[];
  /** Conditions of early returns at its top, before anything else it does, that hold after. */
  readonly guards: string[];
  /** The `if` statements around its first call, which its own top is not inside. */
  readonly depth: number;
  /** Its writes of blocks through a vector parameter. */
  readonly blockWrites: BlockWrite[];
}

/**
 * Emits the GPU form of a compute pipeline: its kernel as one fragment pass per written resource,
 * in the format `computeModule.ts` reads. A first resolution without a plan finds what the kernel
 * writes and throws `Ineligible` for what fragment passes cannot do; a second with that plan emits
 * the passes.
 */
export class KitGlslComputeGenerator extends KitGlslGenerator {
  readonly #plan: ComputePlan | undefined;
  readonly #buffers = new Map<string, WrittenBuffer>();
  readonly #textures = new Map<string, TextureInfo>();
  readonly #columns = new Set<string>();
  readonly #maps = new Map<string, TexelMap>();
  readonly #scatters = new Map<string, Scatter>();
  /** Buffers the kernel adds to at shared cells, dropping the atomics' results. */
  readonly #adds = new Map<string, BufferInfo>();
  /** How each buffer in `#adds` combines its records, `add` unless another atomic said. */
  readonly #operations = new Map<string, Operation>();
  /** The most additions each generated function makes, its callees' included. */
  readonly #addsOf = new Map<string, ReadonlyMap<string, number>>();
  /** The most writes at computed cells each generated function makes, per resource. */
  readonly #recordsOf = new Map<string, ReadonlyMap<string, number>>();
  /** The entry's writes at computed cells per resource, which size the records of each. */
  #records: ReadonlyMap<string, number> = new Map();
  /** The call whose value the statement being generated drops, and whether it is being made. */
  #dropped: Expression | undefined;
  #dropping = false;
  /** The function a call is being made from, so a second route to the same call counts once. */
  #inCall: Frame | undefined;
  /** Counters whose values the kernel takes: atomic additions of 1 whose results it uses. */
  readonly #takes = new Set<string>();
  /** Arrays invocations own blocks of, with the sites the entry's calls give. */
  readonly #blocks = new Map<string, Block>();
  readonly #blockWritersOf = new Map<string, readonly BlockWrite[]>();
  readonly #codeOf = new Map<string, string>();
  /** Buffers the kernel loads atomically anywhere, in whatever order its functions resolve. */
  readonly #loads = new Set<string>();
  /** Every function's code, which the takes are checked over once the walk has seen it all. */
  readonly #codes: string[] = [];
  #slots = new Map<string, number>();
  /** Runtime loops around the statement being generated. */
  #loops = 0;
  /** The `for…of` being generated, and whether TypeGPU took its iterable as one to unroll. */
  #iterable: { readonly node: unknown; unrolled: boolean } | undefined;
  /** `if` statements around the statement being generated. */
  #branches = 0;
  /** The condition of an early `return` the entry starts with, before it writes anything. */
  #exit: Expression | undefined;
  /** The condition of an early `return` at the top of a function, which holds after it. */
  #guard: Expression | undefined;
  /** Bounds such returns put on the invocation, by axis: invocations at or past them return. */
  readonly #bounds = new Map<number, number>();
  /** Whether the write being checked assigns a whole value with `=`. */
  #plain = false;
  readonly #functions = new Set<string>();
  /** Functions without parameters that read only what every invocation sees alike. */
  readonly #pure = new Set<string>();
  /** The conditions around the statement being generated that choose one invocation by axis. */
  readonly #chosen: {
    readonly frame: Frame;
    readonly axes: ReadonlyMap<number, string>;
    /** The settled variables the choice reads. */
    readonly from: readonly string[];
  }[] = [];
  /** The condition of the `if` being generated, when it may choose an invocation. */
  #condition: Expression | undefined;
  /** Storage textures the kernel loads from. */
  readonly #loaded = new Set<string>();
  readonly #shadows = new Map<string, string>();
  readonly #privates: { id: string; dataType: d.BaseData; init: string | undefined }[] = [];
  readonly #frames: Frame[] = [];
  /** The own arguments of the call being made, for the function it resolves. */
  #pending: (Own | undefined)[] | undefined;
  /** The own arguments each function was generated for. */
  readonly #assumed = new Map<string, string>();
  /** Functions whose code depends on the own arguments they were generated for. */
  readonly #relies = new Set<string>();
  /** Whether the dispatch is one invocation, by the entry's workgroup size. */
  #single = false;
  #packers = 0;

  readonly #outer: Numbering | undefined;

  /**
   * `outer` is the resolution the form is attached to. Without it, the form numbers bindings as
   * its own resolution does, which differs where a catch-all group's order or a layout's automatic
   * index depends on the order of resolution.
   */
  constructor(plan?: ComputePlan, outer?: ResolutionCtx) {
    super();
    this.#plan = plan;
    this.#outer = outer as Numbering | undefined;
  }

  /** The group and binding `outer` gives a resource, as placeholders its resolution replaces. */
  #numbered(options: VariableDefinitionOptions): VariableDefinitionOptions {
    const outer = this.#outer;
    if (!outer || options.group === undefined || options.binding === undefined) return options;
    const inner = this.ctx as unknown as Numbering;
    if (options.group === CATCHALL) {
      const resource = inner.fixedBindings[options.binding]?.resource;
      const binding = outer.fixedBindings.findIndex((each) => each.resource === resource);
      if (binding < 0) throw new Error(`@typegpu/kit-gl: '${options.id}' is not bound outside`);
      return { ...options, group: `${OUTER}${CATCHALL.slice(1)}`, binding };
    }
    const layout = [...inner.bindGroupLayoutsToPlaceholderMap].find(
      ([, placeholder]) => placeholder === options.group,
    )?.[0];
    const placeholder = layout && outer.bindGroupLayoutsToPlaceholderMap.get(layout);
    if (!placeholder) throw new Error(`@typegpu/kit-gl: '${options.id}' is not bound outside`);
    return { ...options, group: `${OUTER}${placeholder.slice(1)}` };
  }

  /** What the kernel writes, once a resolution has run. */
  get plan(): ComputePlan {
    return {
      buffers: new Map(
        [...this.#buffers].map(([id, { whole, offset, layout, chosen }]) => [
          id,
          { whole, offset, layout, ...(chosen && { chosen }) },
        ]),
      ),
      textures: new Set(this.#textures.keys()),
      columns: new Set(this.#columns),
      maps: new Map(this.#maps),
      scatters: new Set(this.#scatters.keys()),
      adds: new Set(this.#adds.keys()),
      slots: new Map(this.#slots),
      records: new Map(this.#records),
      takes: new Set(this.#takes),
      blocks: new Map(
        [...this.#blocks].map(([id, { offset, width, step, phase, sites }]) => [
          id,
          { offset, width, step, phase, sites },
        ]),
      ),
    };
  }

  protected override header(): string {
    return COMPUTE_HEADER;
  }

  override initGenerator(ctx: ResolutionCtx): void {
    super.initGenerator(ctx);
    for (const name of [GID, BEGIN, RESULT, INVOCATIONS, WORKGROUPS, LAYER, Z, AIM]) {
      ctx.reserveIdentifier(name, 'global');
    }
    this.chunk(
      'shared',
      [GID, INVOCATIONS, WORKGROUPS, LAYER, Z, AIM],
      `uvec3 ${GID};\nuniform highp uvec3 ${INVOCATIONS};\nuniform highp uvec3 ${WORKGROUPS};\nuniform int ${LAYER};\nint ${Z};\nint ${AIM} = -1;`,
    );
  }

  get #frame(): Frame | undefined {
    return this.#frames.at(-1);
  }

  /** Marks the function being generated as depending on its own arguments. */
  #relyOnOwn(): void {
    const frame = this.#frame;
    if (frame && !frame.entry) frame.relies = true;
  }

  // ---------------------------------------------------------------------------------------------
  // The invocation's own index
  // ---------------------------------------------------------------------------------------------

  /** What a snippet provably is in terms of `global_invocation_id`, in the current function. */
  #own(snippet: Snippet | undefined): Own | undefined {
    const frame = this.#frame;
    if (!snippet || !frame || frame.names.size === 0 || snippet.dataType === UnknownData) {
      return undefined;
    }
    let text: string;
    try {
      text = this.ctx.resolveSnippet(snippet).value;
    } catch {
      return undefined;
    }
    const value = new Arithmetic(text.replace(/\s+/g, ''), frame.names, (part) =>
      this.#shared(part, frame),
    ).read();
    if (!Array.isArray(value)) return undefined;
    return value.map((each) => ({
      ...each,
      ...(each.s && {
        s: each.s.map((term: Term) => ({ ...term, text: inlined(term.text, frame) })),
      }),
      ...(each.r?.of.s && {
        r: {
          ...each.r,
          of: {
            ...each.r.of,
            s: each.r.of.s.map((term: Term) => ({ ...term, text: inlined(term.text, frame) })),
          },
        },
      }),
    }));
  }

  protected override ownIndex(index: Snippet): unknown {
    const own = this.#own(index);
    const layout = layoutOf(own);
    // A phase read through a settled variable holds only while it is never reassigned
    if (layout?.phase && own) for (const name of fromOf(own)) this.#frame?.choosing.add(name);
    return layout;
  }

  /**
   * TypeGPU reuses a local's name in a sibling block, so a declaration takes the name away from
   * the own values until a `const` gives it back.
   */
  override _emitVarDecl(
    keyword: Parameters<KitGlslGenerator['_emitVarDecl']>[0],
    name: string,
    dataType: d.BaseData,
    rhs: string,
  ): string {
    const frame = this.#frame;
    frame?.names.delete(name);
    frame?.settled.delete(name);
    // A copy of what every invocation sees alike is seen alike too, until it is reassigned
    const from = frame && this.#shared(rhs, frame);
    if (from) frame.settled.set(name, { from, value: inlined(rhs, frame) });
    else frame?.locals.add(name);
    return super._emitVarDecl(keyword, name, dataType, rhs);
  }

  /**
   * An alias declares the address it names as a local of its own, which is settled when every
   * invocation computes it alike, as a declaration's value is.
   */
  protected override _aliasConstStatement(
    rawId: string,
    eqNode: Expression,
    eq: Snippet,
  ): ResolvedStatement {
    const result = super._aliasConstStatement(rawId, eqNode, eq);
    const frame = this.#frame;
    if (!frame || typeof result.code !== 'string') return result;
    for (const [, name, rhs] of result.code.matchAll(/^\s*uint (\w+) = (.*);$/gm)) {
      if (!name || rhs === undefined) continue;
      frame.names.delete(name);
      frame.settled.delete(name);
      const from = this.#shared(rhs, frame);
      if (from) frame.settled.set(name, { from, value: inlined(rhs, frame) });
      else frame.locals.add(name);
    }
    return result;
  }

  protected override _constStatement(statement: Const): ResolvedStatement {
    const result = super._constStatement(statement);
    const [, idNode, eqNode] = statement;
    const frame = this.#frame;
    if (eqNode !== undefined && frame && frame.names.size > 0) {
      const id = typeof idNode === 'string' ? idNode : (idNode as unknown as [number, string])[1];
      const defined = this.ctx.getById(id);
      const own = this.#own(this._expression(eqNode));
      if (own && typeof defined?.value === 'string') frame.names.set(defined.value, own);
    }
    return result;
  }

  // ---------------------------------------------------------------------------------------------
  // Calls carry own values into helpers
  // ---------------------------------------------------------------------------------------------

  /**
   * TypeGPU generates a function once, for the call that resolves it first. Its parameters are own
   * where that call passed own values, so every later call must pass the same.
   */
  #calling<T extends Snippet | undefined>(args: readonly Snippet[], call: () => T): T {
    const owns = args.map((arg) => this.#own(arg));
    const outer = this.#pending;
    this.#pending = owns.some(Boolean) ? owns : undefined;
    // The generator makes a call through `call` and then `_callShellless` in the same function
    const nested = this.#inCall === this.#frame;
    const calling = this.#inCall;
    this.#inCall = this.#frame;
    let result: T;
    try {
      result = call();
    } finally {
      this.#pending = outer;
      this.#inCall = calling;
    }
    const name = typeof result?.value === 'string' ? /^(\w+)\(/.exec(result.value)?.[1] : undefined;
    if (!nested && name !== undefined) this.#blockCall(name, args);
    // A call reaches here once more through the generator's own call, which counts it
    const adds = !nested && name !== undefined ? this.#addsOf.get(name) : undefined;
    for (const [id, count] of adds ?? []) this.#added(id, count);
    const records = !nested && name !== undefined ? this.#recordsOf.get(name) : undefined;
    for (const [id, count] of records ?? []) this.#wrote(id, count);
    if (name === undefined || !this.#relies.has(name)) return result;
    if (this.#assumed.get(name) !== JSON.stringify(owns.some(Boolean) ? owns : [])) {
      throw new Ineligible(`calls '${name}' with other invocation indices than it was made for`);
    }
    this.#relyOnOwn();
    return result;
  }

  override call(callee: AnyFn, args: readonly Snippet[], calleeSnippet?: Snippet): Snippet {
    const name = getName(callee) ?? '';
    const dropped = this.#dropping;
    if (name === 'atomicLoad' || name === 'atomicStore' || name in ATOMICS) {
      return this.#atomic(name, args, dropped);
    }
    return this.#calling(args, () => super.call(callee, args, calleeSnippet));
  }

  /**
   * An atomic on a cell the invocation owns, or on a binding a single invocation owns whole, is
   * sequential: a read, or a read-modify-write of what the invocation holds, returning the old
   * value. Atomics on cells other invocations share stay on the CPU.
   */
  #atomic(name: string, args: readonly Snippet[], dropped: boolean): Snippet {
    const target = args[0];
    const place = target && this.placeOf(target);
    if (!target || !place) throw new Ineligible(`'${name}' outside a buffer`);
    const inner = (undecorate(place.schema) as d.Atomic).inner as d.BaseData;
    const value = this.ctx.resolveSnippet(target).value;
    const id = place.buffer.id;
    const taken = this.#takes.has(id) || this.#plan?.takes?.has(id) === true;
    if (taken && dropped) throw new Ineligible(`takes values of '${id}' and adds to it`);
    if (this.#adds.has(id) && !taken && name !== 'atomicLoad' && !(dropped && name in COMBINED)) {
      throw new Ineligible(`'${name}' on '${id}', which it adds to at shared cells`);
    }
    if (
      dropped &&
      name in COMBINED &&
      (this.#adds.has(id) ||
        (!place.element?.own &&
          !this.#single &&
          !this.#buffers.has(id) &&
          !this.#plan?.buffers.has(id) &&
          !this.#chosen.some((each) => each.frame === this.#frame)))
    ) {
      return this.#add(name, place, args[1], inner);
    }
    if (name === 'atomicLoad') {
      this.#loads.add(id);
      // Marked for the order of loads and additions; a summed buffer's load reads the contents
      // from before the dispatch
      if (!this.#plan) return snip(`${LOAD_MARK}${id}(${value})`, inner, 'runtime', false);
      return snip(value, inner, 'runtime', false);
    }
    if (
      name === 'atomicAdd' &&
      !dropped &&
      inner === d.u32 &&
      (this.#takes.has(id) ||
        this.#plan?.takes?.has(id) ||
        (!place.element && !this.#single && !this.#buffers.has(id) && !this.#adds.has(id)))
    ) {
      if (!d.isAtomic(undecorate(place.buffer.schema))) {
        throw new Ineligible(`takes from '${id}', which is more than one atomic`);
      }
      return this.#take(place, args[1]);
    }
    this.#recordPlace(place, `'${name}' on`);
    const operand = this.#valueAs(args[1], inner);
    if (name === 'atomicStore') return snip(`${value} = ${operand}`, d.Void, 'runtime', true);
    const helper = `kit_${name}`;
    this.helper(
      helper,
      ['uint', 'int']
        .map(
          (type) =>
            `${type} ${helper}(inout ${type} o, ${type} v) {\n  ${type} old = o;\n  o = ${(ATOMICS[name] as string).replaceAll('o', 'old')};\n  return old;\n}`,
        )
        .join('\n'),
    );
    return snip(`${helper}(${value}, ${operand})`, inner, 'runtime', true);
  }

  /** A snippet's value as GLSL of `type`, converting literals and other integer types. */
  #valueAs(snippet: Snippet | undefined, type: d.BaseData): string {
    if (!snippet) throw new Error('@typegpu/kit-gl: an atomic without an operand');
    if (typeof snippet.value === 'number') return this.numericLiteral(snippet.value, type).value;
    const value = this.ctx.resolveSnippet(snippet).value;
    return snippet.dataType === type ? value : `${this.ctx.resolve(type).value}(${value})`;
  }

  /** Counts writes at computed cells, which loops would leave unbounded. */
  #wrote(id: string, count: number): void {
    if (this.#loops > 0) throw new Ineligible(`writes '${id}' at computed cells inside a loop`);
    const frame = this.#frame;
    if (frame) frame.records.set(id, (frame.records.get(id) ?? 0) + count);
  }

  /** Counts additions the statement being generated makes, which loops would leave unbounded. */
  #added(id: string, count: number): void {
    if (this.#loops > 0) throw new Ineligible('adds to shared cells inside a loop');
    const frame = this.#frame;
    if (frame) frame.adds.set(id, (frame.adds.get(id) ?? 0) + count);
  }

  /**
   * An atomic addition, or `min`, `max`, `or` or `and`, whose result the kernel drops, on a cell
   * other invocations may combine into too: the invocation records the word and the operand, and a
   * pass combines every record with the cells by the buffer's one operation. `min` and `max` take
   * `u32` cells only, since the device compares their unsigned halves.
   */
  #add(name: string, place: Place, amount: Snippet | undefined, inner: d.BaseData): Snippet {
    const { buffer } = place;
    if (this.#buffers.has(buffer.id) || this.#scatters.has(buffer.id)) {
      throw new Ineligible(`adds to '${buffer.id}', which it also writes`);
    }
    const [operation, code] = COMBINED[name] as readonly [Operation, number];
    const before = this.#operations.get(buffer.id);
    if (before !== undefined && before !== operation) {
      throw new Ineligible(`'${name}' on '${buffer.id}', which it also combines by '${before}'`);
    }
    // Unsigned order is not signed order, and the device compares unsigned halves
    if ((operation === 'min' || operation === 'max') && inner !== d.u32) {
      throw new Ineligible(`'${name}' on '${buffer.id}', whose cells are not 'u32'`);
    }
    this.#operations.set(buffer.id, operation);
    this.#adds.set(buffer.id, buffer);
    this.#added(buffer.id, 1);
    const operand = this.#valueAs(amount, inner);
    const bits = inner === d.u32 ? operand : `uint(${operand})`;
    const signed = name === 'atomicSub' ? `0u - ${bits}` : bits;
    if (!this.#plan) return snip(`${ADD_MARK}${buffer.id}()`, d.Void, 'runtime', true);
    // The records are kept per pass, one buffer's additions each, in arrays the most of them fit
    const slots = Math.max(0, ...(this.#plan.slots?.values() ?? []));
    this.helper(
      ADD,
      `uint ${ADDED};\nuint ${ADDED}_at[${slots}];\nuint ${ADDED}_amount[${slots}];
void ${ADD}(int aim, uint at, uint amount, uint words, uint first, int op) {
  if (${AIM} != aim) return;
  // A buffer of a fixed size keeps a record per word; an access past its words may be discarded
  if (words > 0u) {
    if (at - first >= words) return;
    for (uint i = 0u; i < ${ADDED}; i++) {
      if (${ADDED}_at[i] == at) {
        uint kept = ${ADDED}_amount[i];
        ${ADDED}_amount[i] = op == 0 ? kept + amount : op == 1 ? min(kept, amount) : op == 2 ? max(kept, amount) : op == 3 ? kept | amount : kept & amount;
        return;
      }
    }
  }
  if (${ADDED} >= ${slots}u) return;
  ${ADDED}_at[${ADDED}] = at;
  ${ADDED}_amount[${ADDED}] = amount;
  ${ADDED} += 1u;
}`,
    );
    const merged = this.#merges(buffer);
    const range = merged ? `${this.#wordsOf(buffer)}u, ${buffer.range as string}.x` : '0u, 0u';
    return snip(
      `${ADD}(${this.#aim(buffer.id)}, uint(${place.address}), uint(${signed}), ${range}, ${code})`,
      d.Void,
      'runtime',
      true,
    );
  }

  /** The accumulate directive's operation for `id`, empty for additions. */
  #operationOf(id: string): string {
    const operation = this.#operations.get(id);
    return operation && operation !== 'add' ? ` op ${operation}` : '';
  }

  /**
   * Whether an invocation's additions to `buffer` share a record per word, so it keeps no more
   * records than the buffer has words: a buffer of a fixed size the kernel takes no values of,
   * whose takes are counted by records. Sums of a word's amounts wrap as their atomics do.
   */
  #merges(buffer: BufferInfo): boolean {
    return !this.#takes.has(buffer.id) && Number.isFinite(this.#wordsOf(buffer));
  }

  #wordsOf(buffer: BufferInfo): number {
    return Math.floor(d.sizeOf(buffer.schema as d.AnyData) / 4);
  }

  /**
   * An atomic addition of 1 to a whole counter, whose result the kernel uses: the counter's value
   * before the dispatch, plus the takes of the invocations before this one, plus this invocation's
   * own so far. The addition is recorded too, so the counter's pass of sums advances it.
   */
  #take(place: Place, amount: Snippet | undefined): Snippet {
    const { buffer } = place;
    if (amount?.value !== 1)
      throw new Ineligible(`adds other than 1 to '${buffer.id}' and uses it`);
    if (this.#adds.has(buffer.id) && !this.#takes.has(buffer.id)) {
      throw new Ineligible(`takes values of '${buffer.id}' and adds to it`);
    }
    this.#takes.add(buffer.id);
    const added = this.#add('atomicAdd', place, amount, d.u32);
    if (!this.#plan) return snip(`${TAKE}0()`, d.u32, 'runtime', true);
    const aim = this.#aim(buffer.id);
    const name = `${TAKE}${aim}`;
    const head = this.load(buffer.source, place.address, d.u32, buffer.id);
    this.helper(RANKED, `uniform int ${RANKED};`);
    this.helper(
      name,
      `uniform highp usampler2D kit_bases${aim};
uint kit_taken${aim};
uint ${name}() {
  uint base = 0u;
  if (${RANKED} == 1) {
    uint id = ${GID}.x + ${INVOCATIONS}.x * (${GID}.y + ${INVOCATIONS}.y * ${GID}.z);
    uint width = uint(textureSize(kit_bases${aim}, 0).x);
    base = texelFetch(kit_bases${aim}, ivec2(int(id % width), int(id / width)), 0).x;
  }
  ${added.value};
  uint taken = kit_taken${aim};
  kit_taken${aim} += 1u;
  return ${head} + base + taken;
}`,
    );
    return snip(`${name}()`, d.u32, 'runtime', true);
  }

  /**
   * Refuses a kernel where a taken value could change how many values an invocation takes: where
   * one, or a variable stored from one in any way, decides a branch or a loop, is returned, or is
   * passed to a function of the kernel. Indices and stored values may use it. Run over every
   * function's code once the walk is done, so the order functions resolve in does not matter.
   */
  #checkTakes(code: string): void {
    const body = code.slice(code.indexOf('{'));
    // A function that hands values back through its parameters could carry a taken one out
    if (/\b(inout|out)\s+\w/.test(body)) {
      throw new Ineligible('takes from a counter beside a function that writes its arguments');
    }
    const tainted = new Set<string>();
    // Every store taints the variable it stores into, through any index, member or swizzle, and
    // compound assignments too; the variables here span every function, as private ones do
    const stores =
      /\b([A-Za-z_]\w*)(?:\s*(?:\[[^\]]*\]|\.\w+))*\s*(?:<<|>>|[-+*/%&|^])?=(?!=)\s*([^;]*);/g;
    for (let grown = true; grown;) {
      grown = false;
      for (const [, name, rhs] of body.matchAll(stores)) {
        if (!name || rhs === undefined || tainted.has(name)) continue;
        const names = [TAKE, ...tainted];
        if (names.some((each) => new RegExp(`\\b${each}`).test(rhs))) {
          tainted.add(name);
          grown = true;
        }
      }
    }
    const reads = (text: string) =>
      [TAKE, ...tainted].some((each) => new RegExp(`\\b${each}`).test(text));
    const balanced = (from: number) => {
      let depth = 0;
      for (let i = from; i < body.length; i++) {
        if (body[i] === '(') depth++;
        if (body[i] === ')' && --depth === 0) return body.slice(from, i + 1);
      }
      return body.slice(from);
    };
    for (const match of body.matchAll(/\b(if|while|for|switch)\s*\(/g)) {
      if (reads(balanced((match.index ?? 0) + match[0].length - 1))) {
        throw new Ineligible('decides a branch on a value it takes from a counter');
      }
    }
    // A ternary's condition and a short-circuit's left operand decide what runs after them
    for (const match of body.matchAll(/\?|&&|\|\||\breturn\b[^;]*;/g)) {
      const chooses = match[0] === '?' || match[0] === '&&' || match[0] === '||';
      const text = chooses
        ? body.slice(Math.max(0, (match.index ?? 0) - 200), match.index)
        : match[0];
      if (!chooses ? reads(text) : reads(text.split(/[;{}]/).at(-1) ?? '')) {
        throw new Ineligible('returns or chooses by a value it takes from a counter');
      }
    }
    for (const name of this.#functions) {
      for (const match of body.matchAll(new RegExp(`\\b${name}\\(`, 'g'))) {
        if (reads(balanced((match.index ?? 0) + match[0].length - 1))) {
          throw new Ineligible(`passes a value it takes from a counter to '${name}'`);
        }
      }
    }
  }

  protected override _callShellless(
    callee: AnyFn,
    args: readonly Snippet[],
  ): ResolvedSnippet | undefined {
    return this.#calling(args, () => super._callShellless(callee, args));
  }

  // ---------------------------------------------------------------------------------------------
  // Writes
  // ---------------------------------------------------------------------------------------------

  protected override _statement(statement: Statement): ResolvedStatement {
    if (Array.isArray(statement) && statement[0] === NODE.call) {
      const outer = this.#dropped;
      this.#dropped = statement as Expression;
      try {
        return super._statement(statement);
      } finally {
        this.#dropped = outer;
      }
    }
    if (Array.isArray(statement)) {
      if (statement[0] === NODE.assignmentExpr) {
        this.#plain = statement[2] === '=';
        let place: Place | undefined;
        try {
          place = this.#checkWrite(statement[1] as Expression);
        } finally {
          this.#plain = false;
        }
        if (place && this.#plan?.scatters.has(place.buffer.id)) {
          return this.#scatterWrite(place, statement[3] as Expression);
        }
      }
      if (statement[0] === NODE.postUpdate) this.#checkWrite(statement[2] as Expression);
      if (statement[0] === NODE.for || statement[0] === NODE.while || statement[0] === NODE.forOf) {
        // A loop until TypeGPU resolves a `for…of`'s iterable to what `tgpu.unroll` gives, which
        // it then expands; decided by that value, never by how the call is spelled
        const outer = this.#iterable;
        const iterable = {
          node: statement[0] === NODE.forOf ? statement[2] : undefined,
          unrolled: false,
        };
        this.#iterable = iterable;
        this.#loops++;
        try {
          return super._statement(statement);
        } finally {
          if (!iterable.unrolled) this.#loops--;
          this.#iterable = outer;
        }
      }
      // Without an `else`, the condition holds for everything the `if` generates
      if (statement[0] === NODE.if && statement[3] === undefined) {
        const depth = this.#chosen.length;
        this.#condition = statement[1] as Expression;
        if (returns(statement[2] as Statement) && this.#beforeWrites()) {
          this.#exit = statement[1] as Expression;
        }
        const top = this.#branches === (this.#frame?.depth ?? 0);
        if (returns(statement[2] as Statement) && this.#loops === 0 && top) {
          this.#guard = statement[1] as Expression;
        }
        this.#branches++;
        try {
          return super._statement(statement);
        } finally {
          this.#branches--;
          this.#condition = undefined;
          this.#exit = undefined;
          this.#guard = undefined;
          this.#chosen.length = depth;
        }
      }
      // A switch case is a branch too: a return in it does not dominate what follows the switch
      if (statement[0] === NODE.if || statement[0] === NODE.switch) {
        this.#branches++;
        try {
          return super._statement(statement);
        } finally {
          this.#branches--;
        }
      }
    }
    return super._statement(statement);
  }

  /** Notes whether the call being made is the one whose value its statement drops. */
  protected override _expression(expression: Expression): Snippet {
    const outer = this.#dropping;
    this.#dropping = expression === this.#dropped;
    try {
      const result = super._expression(expression);
      const iterable = this.#iterable;
      if (
        iterable &&
        !iterable.unrolled &&
        expression === iterable.node &&
        UNROLLED !== undefined &&
        result.value instanceof UNROLLED
      ) {
        iterable.unrolled = true;
        this.#loops--;
      }
      return result;
    } finally {
      this.#dropping = outer;
    }
  }

  protected override _expressionWithTypeConversion(
    expression: Expression,
    expectedType: d.BaseData | d.BaseData[],
  ): Snippet {
    const result = super._expressionWithTypeConversion(expression, expectedType);
    const frame = this.#frame;
    if (expression === this.#exit && frame && typeof result.value === 'string') {
      this.#exit = undefined;
      this.#boundBy(result.value, frame);
    }
    if (expression === this.#guard && frame && typeof result.value === 'string') {
      this.#guard = undefined;
      frame.guards.push(result.value.replace(/\s+/g, ''));
    }
    if (expression === this.#condition && frame) {
      this.#condition = undefined;
      const choice =
        typeof result.value === 'string' ? this.#choiceIn(result.value, frame) : undefined;
      if (choice) this.#chosen.push({ frame, ...choice });
    }
    return result;
  }

  /**
   * The invocation a condition chooses, axis by axis: each top-level conjunct `gid.<axis> == e`
   * where `e` is the same for every invocation. Other conjuncts only narrow the choice.
   */
  #choiceIn(
    condition: string,
    frame: Frame,
  ): { axes: Map<number, string>; from: string[] } | undefined {
    const axes = new Map<number, string>();
    const from: string[] = [];
    for (const conjunct of conjuncts(condition.replace(/\s+/g, ''))) {
      const sides = splitTop(unwrap(conjunct), '==');
      if (sides.length !== 2) continue;
      for (const [own, other] of [sides, sides.toReversed()] as [string, string][]) {
        const value = new Arithmetic(own, frame.names).read();
        const named = Array.isArray(value) ? axesOf(value) : undefined;
        const settled = named?.length === 1 ? this.#shared(other, frame) : undefined;
        if (!settled) continue;
        axes.set('xyz'.indexOf(named as string), inlined(other, frame));
        from.push(...settled);
        break;
      }
    }
    return axes.size > 0 ? { axes, from } : undefined;
  }

  /**
   * Whether GLSL text reads only what every invocation sees alike: the settled variables it reads
   * if so, which must then not be reassigned.
   */
  #shared(text: string, frame: Frame): string[] | undefined {
    if (this.#varies(text)) return undefined;
    const settled: string[] = [];
    for (const [name] of text.matchAll(/(?<![\w.])[A-Za-z_]\w*/g)) {
      if (frame.locals.has(name) || frame.names.has(name)) return undefined;
      const known = frame.settled.get(name);
      if (known) settled.push(name, ...known.from);
    }
    return settled;
  }

  /**
   * Whether GLSL text reads what may differ between invocations, other than locals: the
   * invocation, private variables, written resources, or functions that may.
   */
  #varies(text: string): boolean {
    if (/texel|texture/.test(text)) return true;
    const varying = new Set([
      GID,
      ...this.#privates.map((each) => each.id),
      ...this.#buffers.keys(),
      ...this.#textures.keys(),
      ...(this.#plan ? [...this.#plan.buffers.keys(), ...this.#plan.textures] : []),
    ]);
    for (const [name] of text.matchAll(/(?<![\w.])[A-Za-z_]\w*/g)) {
      if (varying.has(name) || name.startsWith('kit_shadow')) return true;
      if (this.#functions.has(name) && !this.#pure.has(name)) return true;
    }
    return false;
  }

  /**
   * The invocation the conditions around the current statement choose, from `x` on. Using it
   * forbids reassigning the settled variables it reads.
   */
  #choice(): string[] | undefined {
    const frame = this.#frame;
    const axes = new Map<number, string>();
    const from: string[] = [];
    for (const each of this.#chosen) {
      if (each.frame !== frame) continue;
      for (const [index, text] of each.axes) axes.set(index, text);
      from.push(...each.from);
    }
    const chosen: string[] = [];
    for (let index = 0; axes.has(index); index++) chosen.push(`uint(${axes.get(index)})`);
    if (chosen.length === 0 || !frame) return undefined;
    for (const name of from) frame.choosing.add(name);
    return chosen;
  }

  #checkWrite(node: Expression): Place | undefined {
    const target = this._expression(node);
    const place = this.placeOf(target);
    const frame = this.#frame;
    if (!place) {
      const text = target.dataType === UnknownData ? '' : this.ctx.resolveSnippet(target).value;
      const root = /^\s*([A-Za-z_]\w*)/.exec(text)?.[1];
      if (root) frame?.assigned.add(root);
      if (root && frame?.names.has(root)) {
        throw new Ineligible(`modifies '${root}', which indices are derived from`);
      }
      return undefined;
    }
    this.#recordPlace(place, 'writes');
    return place;
  }

  /**
   * Whether the entry returns early for every invocation past a layout's width and height, so
   * invocations the layout would alias write nothing, and the dispatch may be wider.
   */
  #guarded(layout: Layout): boolean {
    if (layout.width === undefined) return false;
    const x = this.#bounds.get(0);
    const y = this.#bounds.get(1);
    return (
      x !== undefined &&
      x <= layout.width &&
      (layout.height === undefined || (y !== undefined && y <= layout.height))
    );
  }

  /** Whether the statement being generated is at the top of the entry, before any write. */
  #beforeWrites(): boolean {
    return (
      this.#frame?.entry === true &&
      this.#loops === 0 &&
      this.#branches === 0 &&
      this.#buffers.size + this.#textures.size + this.#scatters.size === 0
    );
  }

  /** Reads an early return's condition for `gid.<axis> >= constant` disjuncts. */
  #boundBy(condition: string, frame: Frame): void {
    for (const disjunct of disjuncts(condition.replace(/\s+/g, ''))) {
      const sides = /^(.*?)(>=|>)(.*)$/.exec(unwrap(disjunct));
      if (!sides) continue;
      const [, left, op, right] = sides as unknown as [string, string, string, string];
      const own = new Arithmetic(left, frame.names).read();
      const bound = new Arithmetic(right, frame.names).read();
      const named = Array.isArray(own) ? axesOf(own) : undefined;
      if (named?.length !== 1 || typeof bound !== 'number') continue;
      const index = AXES.indexOf(named);
      const limit = op === '>' ? bound + 1 : bound;
      this.#bounds.set(index, Math.min(this.#bounds.get(index) ?? limit, limit));
    }
  }

  /**
   * Whether a write the kernel makes may be a point of a scatter pass. The plan fixes which
   * resources are, so the second resolution finds no new ones.
   */
  #scatterable(): boolean {
    return this.#loops === 0 && !this.#plan;
  }

  /** A resource's index among those the kernel writes at computed cells, as `kit_aim` holds it. */
  #aim(id: string): number {
    return [...(this.#plan?.scatters ?? []), ...(this.#plan?.adds ?? [])].toSorted().indexOf(id);
  }

  /** The records a resource written at computed cells keeps per invocation. */
  #slotsOf(id: string): number {
    return this.#plan?.records?.get(id) ?? 0;
  }

  /**
   * The function a write at computed cells of `id` calls: it records where the invocation wrote
   * and what, in the next of the resource's slots, which a pass of points reads after the kernel.
   * A buffer's values are kept as their words.
   */
  #putter(id: string, where: 'uint' | 'ivec3', type: d.BaseData | string): string {
    const aim = this.#aim(id);
    const name = `kit_put${aim}`;
    const slots = this.#slotsOf(id);
    const prefix = `kit_p${aim}`;
    const declared = typeof type === 'string' ? type : `${this.ctx.resolve(type).value}`;
    const suffix = typeof type === 'string' ? '' : this.arraySuffix(type);
    let kept: string[];
    let stored: string[];
    if (where === 'uint') {
      const schema = type as d.BaseData;
      const words = Math.ceil(d.sizeOf(schema as d.AnyData) / 4);
      const pack = this.#packer(schema);
      const vectors = Math.ceil(words / 4);
      kept = Array.from({ length: vectors }, (_, j) => `uvec4 ${prefix}_v${j}[${slots}];`);
      stored = Array.from({ length: vectors }, (_, j) => {
        const parts = [0, 1, 2, 3].map((c) =>
          4 * j + c < words ? `${pack}(value, ${4 * j + c}u, 0u)` : '0u',
        );
        return `  ${prefix}_v${j}[${prefix}_n] = uvec4(${parts.join(', ')});`;
      });
    } else {
      kept = [`${declared} ${prefix}_value[${slots}];`];
      stored = [`  ${prefix}_value[${prefix}_n] = value;`];
    }
    this.helper(
      name,
      `uint ${prefix}_n;
${where} ${prefix}_at[${slots}];
${kept.join('\n')}
void ${name}(${where} at, ${declared} value${suffix}) {
  if (${AIM} != ${aim} || ${prefix}_n >= ${slots}u) return;
  ${prefix}_at[${prefix}_n] = at;
${stored.join('\n')}
  ${prefix}_n += 1u;
}`,
    );
    return name;
  }

  /** A whole element written at a computed index: captured when its site is the pass's. */
  #scatterWrite(place: Place, value: Expression): ResolvedStatement {
    const { buffer, element } = place;
    if (!element?.array || element.path !== '') {
      throw new Error(`@typegpu/kit-gl: the second resolution writes '${buffer.id}' anew`);
    }
    const type = element.array.elementType as d.BaseData;
    const stride = wordsOf(strideOf(type));
    this.#scatters.set(buffer.id, {
      kind: 'buffer',
      buffer,
      array: element.array,
      offset: element.offset,
    });
    const written = this.ctx.resolveSnippet(this._expressionWithTypeConversion(value, type)).value;
    const at = `${buffer.base} + ${element.offset}u + uint(${element.index}) * ${stride}u`;
    return {
      code: `${this.ctx.pre}${this.#putter(buffer.id, 'uint', type)}(${at}, ${written});`,
      definesInNearestScope: false,
    };
  }

  /** Records a write to a place, or rejects the kernel when the place is not provably owned. */
  #recordPlace(place: Place, verb: string): void {
    const { buffer, element } = place;
    const frame = this.#frame;
    if (this.#adds.has(buffer.id)) {
      throw new Ineligible(`${verb} '${buffer.id}', which it adds to at shared cells`);
    }
    if (verb === 'writes' && this.#blockWrite(place)) return;
    if (this.#plan?.blocks?.has(buffer.id) || this.#blocks.has(buffer.id)) {
      throw new Ineligible(`${verb} '${buffer.id}' other than at a block it owns`);
    }
    if (this.#plan?.scatters.has(buffer.id)) {
      if (verb !== 'writes' || !this.#plain) {
        throw new Ineligible(`${verb} '${buffer.id}', which it writes at computed cells`);
      }
      return;
    }
    if (
      verb === 'writes' &&
      this.#plain &&
      element?.path === '' &&
      !element.own &&
      !this.#single &&
      this.#scatterable() &&
      !this.#chosen.some((each) => each.frame === frame)
    ) {
      this.#scatters.set(buffer.id, {
        kind: 'buffer',
        buffer,
        array: element.array as d.WgslArray,
        offset: element.offset,
      });
      this.#wrote(buffer.id, 1);
      return;
    }
    this.#relyOnOwn();
    if (element?.own) {
      const layout = element.ownership as Layout;
      this.#record(buffer, { whole: false, offset: element.offset, layout }, element.array);
      return;
    }
    if (this.#single) {
      this.#record(buffer, { whole: true, offset: 0, layout: {} });
      return;
    }
    const chosen = this.#choice();
    if (chosen) {
      this.#record(buffer, { whole: true, offset: 0, layout: {}, chosen });
      return;
    }
    if (!element) throw new Ineligible(`${verb} '${buffer.id}' outside an array`);
    if (verb === 'writes' && this.#plain && element.path === '' && this.#scatterable()) {
      this.#scatters.set(buffer.id, {
        kind: 'buffer',
        buffer,
        array: element.array as d.WgslArray,
        offset: element.offset,
      });
      this.#wrote(buffer.id, 1);
      return;
    }
    if (verb === 'writes' && this.#plain && element.path === '' && this.#loops > 0) {
      throw new Ineligible(`writes '${buffer.id}' at computed cells inside a loop`);
    }
    if (!frame?.entry && !frame?.names.size) {
      throw new Ineligible(`a helper function ${verb} '${buffer.id}'`);
    }
    throw new Ineligible(`${verb} an element of '${buffer.id}' that is not the invocation's own`);
  }

  /**
   * Whether a write is a helper's write at `row * width + col` of one of its vector parameters,
   * after an early return unless a guard function proves `0 <= col < width`: it then records
   * itself on the helper, and each call names the block's cell it writes.
   */
  #blockWrite(place: Place): boolean {
    const { buffer, element } = place;
    const frame = this.#frame;
    if (!frame || frame.entry || !element?.array || element.path !== '' || !this.#plain) {
      return false;
    }
    const index = element.index.replace(/\s+/g, '');
    const shape = /^\(*\(uint\((\w+)\.y\)\*(\d+)u\)\+uint\(\1\.x\)\)*$/.exec(index);
    if (!shape) return false;
    const [, name, wide] = shape as unknown as [string, string, string];
    const param = frame.params.indexOf(name);
    if (param < 0 || this.#loops > 0) return false;
    const width = Number(wide);
    if (!frame.guards.some((guard) => this.#keepsColumn(guard, name, width))) {
      throw new Ineligible(`writes '${buffer.id}' at a column it does not keep below ${width}`);
    }
    frame.blockWrites.push({
      param,
      buffer,
      array: element.array,
      offset: element.offset,
      width,
    });
    return true;
  }

  /**
   * Whether an early return's condition, `!f(p)`, leaves `0 <= p.x < width`: `f` returns a
   * conjunction that includes `p.x >= 0`, or every component non-negative, and `p.x < width`.
   */
  #keepsColumn(guard: string, name: string, width: number): boolean {
    const call = /^!\(?(\w+)\((\w+)\)\)?$/.exec(guard);
    if (!call || call[2] !== name) return false;
    const code = this.#codeOf
      .get(call[1] as string)
      ?.replace(/^\/\/#.*$/gm, '')
      .replace(/\s+/g, '');
    const signature = code && /^bool\w+\(ivec2(\w+)\)\{return(.*);\}$/.exec(code);
    if (!signature) return false;
    const [, param, body] = signature as unknown as [string, string, string];
    const flatten = (text: string): string[] => {
      const parts = conjuncts(unwrap(text));
      return parts.length > 1 ? parts.flatMap(flatten) : [unwrap(text)];
    };
    const parts = flatten(body);
    const low = [`all(greaterThanEqual(${param},ivec2(0)))`, `${param}.x>=0`, `0<=${param}.x`];
    const high = [`${param}.x<${width}`, `${width}>${param}.x`];
    return parts.some((part) => low.includes(part)) && parts.some((part) => high.includes(part));
  }

  /** Names the block cells a call writes through a helper that writes blocks. */
  #blockCall(name: string, args: readonly Snippet[]): void {
    const writes = this.#blockWritersOf.get(name);
    if (!writes?.length) return;
    const frame = this.#frame;
    if (!frame?.entry || this.#loops > 0) {
      throw new Ineligible(`writes blocks through '${name}' other than from the entry, once`);
    }
    for (const write of writes) {
      const own = this.#own(args[write.param]);
      const [col, row] = own ?? [];
      const plainAxis = (a: Affine | undefined, along: 'x' | 'y') =>
        a &&
        !a.r &&
        a[along] > 0 &&
        (['x', 'y', 'z'] as const).every((c) => c === along || a[c] === 0);
      if (own?.length !== 2 || !plainAxis(col, 'x') || !plainAxis(row, 'y')) {
        throw new Ineligible(
          `writes '${write.buffer.id}' through '${name}' at no block of its own`,
        );
      }
      const phaseOf = (a: Affine) =>
        (a.s ?? []).map((term) => `${term.sign > 0 ? '+' : '-'} int(${term.text})`).join(' ') ||
        '+ 0';
      for (const term of [...(col?.s ?? []), ...(row?.s ?? [])]) {
        for (const settled of term.from) frame.choosing.add(settled);
      }
      const block: Block = {
        buffer: write.buffer,
        array: write.array,
        offset: write.offset,
        width: write.width,
        step: [(col as Affine).x, (row as Affine).y],
        phase: [phaseOf(col as Affine), phaseOf(row as Affine)],
        sites: [],
      };
      const known = this.#blocks.get(write.buffer.id) ?? block;
      if (
        known.width !== block.width ||
        known.offset !== block.offset ||
        String(known.step) !== String(block.step) ||
        String(known.phase) !== String(block.phase)
      ) {
        throw new Ineligible(`writes '${write.buffer.id}' at blocks of two shapes`);
      }
      const site = [(col as Affine).k, (row as Affine).k] as const;
      const sites = known.sites.some(([c, r]) => c === site[0] && r === site[1])
        ? known.sites
        : [...known.sites, site];
      this.#blocks.set(write.buffer.id, { ...known, sites });
    }
  }

  #record(buffer: BufferInfo, written: WrittenPlan, array?: d.WgslArray): void {
    const known = this.#buffers.get(buffer.id);
    if (
      known &&
      (known.chosen || written.chosen) &&
      (known.whole !== written.whole || String(known.chosen) !== String(written.chosen))
    ) {
      throw new Ineligible(`writes '${buffer.id}' from invocations chosen differently`);
    }
    if (written.whole || known?.whole) {
      const size = d.sizeOf(buffer.schema as d.AnyData);
      if (size === 0 || size > MAX_WHOLE_BYTES) {
        throw new Ineligible(
          `writes '${buffer.id}' whole, which is larger than ${MAX_WHOLE_BYTES} bytes`,
        );
      }
      const chosen = written.chosen ?? known?.chosen;
      this.#buffers.set(buffer.id, {
        buffer,
        whole: true,
        offset: 0,
        layout: {},
        ...(chosen && { chosen }),
      });
    } else if (known) {
      if (known.offset !== written.offset) {
        throw new Ineligible(`writes two arrays of '${buffer.id}'`);
      }
      if (!sameLayout(known.layout, written.layout)) {
        throw new Ineligible(`writes '${buffer.id}' at indices of two shapes`);
      }
    } else {
      this.#buffers.set(buffer.id, { buffer, ...written, ...(array && { array }) });
    }
    const planned = this.#plan?.buffers.get(buffer.id);
    const recorded = this.#buffers.get(buffer.id) as WrittenBuffer;
    if (
      this.#plan &&
      (!planned ||
        planned.whole !== recorded.whole ||
        planned.offset !== recorded.offset ||
        String(planned.chosen) !== String(recorded.chosen) ||
        !sameLayout(planned.layout, recorded.layout))
    ) {
      throw new Error(`@typegpu/kit-gl: the second resolution writes '${buffer.id}' differently`);
    }
  }

  /** The variable that holds what the invocation owns of a written buffer. */
  #bufferShadow(buffer: BufferInfo, written: WrittenPlan, array?: d.WgslArray): string {
    const known = this.#shadows.get(buffer.id);
    if (known) return known;
    const name = this.ctx.makeUniqueIdentifier(`kit_shadow_${buffer.id}`, 'global');
    const type = written.whole ? buffer.schema : ((array as d.WgslArray).elementType as d.BaseData);
    this.chunk(
      'shared',
      [name],
      `${this.ctx.resolve(type).value} ${name}${this.arraySuffix(type)};`,
    );
    this.#shadows.set(buffer.id, name);
    return name;
  }

  protected override placeValue(place: Place, load: string): string {
    if (this.#plan?.scatters.has(place.buffer.id))
      return `${SCATTERED}${place.buffer.id}_${this.#packers++}`;
    const block = this.#plan?.blocks?.get(place.buffer.id);
    if (block) {
      const element = place.element;
      if (!element || element.offset !== block.offset) {
        return `${STALE}${place.buffer.id}_${this.#packers++}`;
      }
      const { shadows, site } = this.#blockNames(place.buffer, block, element.array);
      return `${shadows}[${site}(uint(${element.index}))]${element.path}`;
    }
    const written = this.#plan?.buffers.get(place.buffer.id);
    if (!written) return load;
    const element = place.element;
    if (written.whole) {
      const path = element
        ? `${element.arrayPath}[${element.index}]${element.path}`
        : (place.path ?? '');
      return `${this.#bufferShadow(place.buffer, written)}${path}`;
    }
    if (!element) {
      const schema = undecorate(place.schema);
      const covers = d.isWgslArray(schema) || d.isWgslStruct(schema);
      return covers && place.offset !== undefined && place.offset <= written.offset
        ? `${STALE}${place.buffer.id}_${this.#packers++}`
        : load;
    }
    if (element.offset !== written.offset) return load;
    const shadow = `${this.#bufferShadow(place.buffer, written, element.array)}${element.path}`;
    const own = element.own && sameLayout(element.ownership as Layout, written.layout);
    if (own) {
      this.#relyOnOwn();
      return shadow;
    }
    return `((${element.index}) == ${linear(written.layout, GID)} ? ${shadow} : ${load})`;
  }

  override declareGlobalVar(options: VariableDefinitionOptions): ResolvedSnippet {
    const result = super.declareGlobalVar(this.#numbered(options));
    if (options.scope === 'private') {
      this.#privates.push({
        id: options.id,
        dataType: options.dataType,
        init: options.init && this.ctx.resolveSnippet(options.init).value,
      });
    }
    return result;
  }

  // ---------------------------------------------------------------------------------------------
  // Storage textures
  // ---------------------------------------------------------------------------------------------

  #textureShadow(texture: TextureInfo): string {
    const known = this.#shadows.get(texture.id);
    if (known) return known;
    const name = this.ctx.makeUniqueIdentifier(`kit_shadow_${texture.id}`, 'global');
    this.chunk('shared', [name], `${this.#texelType(texture)} ${name};`);
    this.#shadows.set(texture.id, name);
    return name;
  }

  #texelType(texture: TextureInfo): string {
    const { sampleType } = texture.kind;
    return sampleType === 'u32' ? 'uvec4' : sampleType === 'i32' ? 'ivec4' : 'vec4';
  }

  /** The cell `gid` stands for in a texture, as `texelFetch` takes it. */
  #cellOf(texture: TextureInfo, gid: string): string {
    const { dimension } = texture.kind;
    if (dimension === '1d') return `ivec2(int(${gid}.x), 0)`;
    if (dimension === '2d') return `ivec2(${gid}.xy)`;
    return `ivec3(${gid})`;
  }

  /** The cell texture call arguments name, as `texelFetch` takes it. */
  #cellIn(texture: TextureInfo, args: readonly Snippet[]): string {
    const { dimension } = texture.kind;
    if (dimension === '1d') return `ivec2(${this.intOf(args[1])}, 0)`;
    if (dimension === '2d') return this.ivecOf(args[1], 2);
    if (dimension === '3d') return this.ivecOf(args[1], 3);
    return `ivec3(${this.ivecOf(args[1], 2)}, ${this.intOf(args[2])})`;
  }

  #isStorage(texture: TextureInfo): boolean {
    return (texture.schema.type as string).startsWith('texture_storage');
  }

  #store(args: readonly Snippet[]): string {
    const texture = this.textureOf(args[0]);
    const { format } = texture.schema as unknown as { format: string };
    if (!RENDERABLE.has(format)) {
      throw new Ineligible(`stores to a '${format}' texture, which WebGL 2 cannot render to`);
    }
    const { dimension } = texture.kind;
    const coords = axesOf(this.#own(args[1]));
    const owns =
      dimension === '2d-array'
        ? coords === 'xy' && axesOf(this.#own(args[2])) === 'z'
        : coords === { '1d': 'x', '2d': 'xy', '3d': 'xyz' }[dimension as string];
    const column =
      !owns &&
      (dimension === '2d-array'
        ? coords === 'xy'
        : dimension === '3d' && this.#ownsColumn(args[1]));
    if (column) this.#columns.add(texture.id);
    const scattered = this.#plan?.scatters.has(texture.id);
    if (scattered) return this.#scatterStore(texture, args);
    const strided = !owns && !column && dimension !== '1d' && this.#strided(texture, args);
    if (!owns && !column && !strided && this.#scatterable()) {
      this.#scatters.set(texture.id, { kind: 'texture', texture });
      this.#wrote(texture.id, 1);
      return '';
    }
    if (!owns && !column && !strided) {
      const frame = this.#frame;
      if (!frame?.entry && !frame?.names.size) {
        throw new Ineligible(`a helper function stores to '${texture.id}'`);
      }
      throw new Ineligible(`stores to a texel of '${texture.id}' that is not the invocation's own`);
    }
    if (this.#plan && !this.#plan.textures.has(texture.id)) {
      throw new Error(`@typegpu/kit-gl: the second resolution stores to '${texture.id}' anew`);
    }
    this.#relyOnOwn();
    this.#textures.set(texture.id, texture);
    const value = this.valueOf(args[dimension === '2d-array' ? 3 : 2]);
    const shadow = this.#textureShadow(texture);
    if (owns && this.#maps.has(texture.id)) {
      throw new Ineligible(`stores to '${texture.id}' at texels of two shapes`);
    }
    if (!this.#plan?.columns.has(texture.id)) return `${shadow} = ${value}`;
    const layer = dimension === '3d' ? `${this.ivecOf(args[1], 3)}.z` : this.intOf(args[2]);
    return `${shadow} = (${layer}) == ${Z} ? ${value} : ${shadow}`;
  }

  /** A texel stored at computed coordinates: captured when its site is the pass's. */
  #scatterStore(texture: TextureInfo, args: readonly Snippet[]): string {
    const { dimension } = texture.kind;
    const point =
      dimension === '2d-array'
        ? `ivec3(${this.ivecOf(args[1], 2)}, ${this.intOf(args[2])})`
        : dimension === '3d'
          ? this.ivecOf(args[1], 3)
          : dimension === '2d'
            ? `ivec3(${this.ivecOf(args[1], 2)}, 0)`
            : `ivec3(${this.intOf(args[1])}, 0, 0)`;
    const value = this.valueOf(args[dimension === '2d-array' ? 3 : 2]);
    this.#scatters.set(texture.id, { kind: 'texture', texture });
    return `${this.#putter(texture.id, 'ivec3', this.#texelType(texture))}(${point}, ${value})`;
  }

  /** Whether a store's coordinates own a cell through a texel map; records the map if so. */
  #strided(texture: TextureInfo, args: readonly Snippet[]): boolean {
    const coords = this.#own(args[1]);
    const layer = texture.kind.dimension === '2d-array' ? this.#own(args[2]) : [];
    const map = coords && layer && texelMapOf([...coords, ...layer]);
    if (!map || plainMap(map)) return false;
    const known = this.#maps.get(texture.id);
    if (
      (known && JSON.stringify(known) !== JSON.stringify(map)) ||
      (this.#textures.has(texture.id) && !known)
    ) {
      throw new Ineligible(`stores to '${texture.id}' at texels of two shapes`);
    }
    this.#maps.set(texture.id, map);
    // A residue read through a settled variable holds only while it is never reassigned
    for (const name of fromOf([...coords, ...layer])) this.#frame?.choosing.add(name);
    return true;
  }

  /** The cell a texture's invocation `gid` owns, as `texelFetch` takes it. */
  #ownCell(texture: TextureInfo, gid: string): string {
    const map = this.#plan?.maps.get(texture.id) ?? this.#maps.get(texture.id);
    if (!map) return this.#cellOf(texture, gid);
    const parts = map.map(({ step, r }, i) => {
      const base = step === 1 ? `${gid}.${AXES[i]}` : `${gid}.${AXES[i]} * ${step}u`;
      return `int(${base}${r ? ` + ((${affineText(r.of, gid)}) & ${r.mask}u)` : ''})`;
    });
    return map.length === 2 ? `ivec2(${parts.join(', ')})` : `ivec3(${parts.join(', ')})`;
  }

  /** Whether a 3d texel coordinate is `uvec3(xy, z)` with `xy` the invocation's own, any `z`. */
  #ownsColumn(coordinate: Snippet | undefined): boolean {
    const frame = this.#frame;
    if (!coordinate || !frame || frame.names.size === 0) return false;
    const text = this.ctx.resolveSnippet(coordinate).value.replace(/\s+/g, '');
    const inner = /^[iu]vec3\((.*)\)$/.exec(text)?.[1];
    if (inner === undefined) return false;
    const parts: string[] = [];
    let depth = 0;
    let start = 0;
    for (let i = 0; i < inner.length; i++) {
      const c = inner[i];
      if (c === '(') depth++;
      else if (c === ')') depth--;
      else if (c === ',' && depth === 0) {
        parts.push(inner.slice(start, i));
        start = i + 1;
      }
      if (depth < 0) return false;
    }
    parts.push(inner.slice(start));
    const own = (part: string) => {
      const value = new Arithmetic(part, frame.names).read();
      return Array.isArray(value) ? axesOf(value) : undefined;
    };
    if (parts.length === 2) return own(parts[0] as string) === 'xy';
    if (parts.length === 3) return own(`uvec2(${parts[0]},${parts[1]})`) === 'xy';
    return false;
  }

  override emitCall(
    name: string,
    templateParams: readonly Snippet[],
    args: readonly Snippet[],
  ): string {
    if (name === 'textureStore') return this.#store(args);
    if (name === 'textureLoad' && args[0]) {
      const texture = this.textureOf(args[0]);
      if (this.#isStorage(texture)) this.#loaded.add(texture.id);
      if (this.#plan?.scatters.has(texture.id)) {
        throw new Ineligible(`reads '${texture.id}', which it writes at computed cells`);
      }
      if (this.#isStorage(texture) && this.#plan?.textures.has(texture.id)) {
        const cell = this.#cellIn(texture, args);
        const own = this.#ownCell(texture, GID);
        const old = `texelFetch(${this.samplerUniform(texture)}, ${cell}, 0)`;
        return `(all(equal(${cell}, ${own})) ? ${this.#textureShadow(texture)} : ${old})`;
      }
    }
    return super.emitCall(name, templateParams, args);
  }

  // ---------------------------------------------------------------------------------------------
  // Functions and passes
  // ---------------------------------------------------------------------------------------------

  override functionDefinition(options: FunctionDefinitionOptions): string {
    const frame: Frame = {
      name: options.name,
      names: new Map(),
      entry: options.functionType === 'compute',
      locals: new Set(options.args.map((arg) => arg.name)),
      settled: new Map(),
      assigned: new Set(),
      choosing: new Set(),
      relies: false,
      adds: new Map(),
      records: new Map(),
      params: options.args.map((arg) => arg.name),
      guards: [],
      depth: this.#branches,
      blockWrites: [],
    };
    this.#functions.add(options.name);
    if (options.functionType === 'normal') {
      const pointer = options.args.find((arg) => d.isPtr(arg.decoratedType));
      if (pointer) throw new Ineligible(`a helper function takes the pointer '${pointer.name}'`);
      const owns = this.#pending;
      this.#pending = undefined;
      owns?.forEach((own, i) => {
        const arg = options.args[i];
        if (own && arg) frame.names.set(arg.name, own);
      });
      this.#assumed.set(options.name, JSON.stringify(owns ?? []));
    }
    this.#frames.push(frame);
    let code: string;
    try {
      code = super.functionDefinition(options);
    } finally {
      this.#frames.pop();
    }
    if (frame.relies) this.#relies.add(options.name);
    this.#addsOf.set(options.name, frame.adds);
    this.#recordsOf.set(options.name, frame.records);
    this.#blockWritersOf.set(options.name, frame.blockWrites);
    this.#codeOf.set(options.name, code);
    const body = code.slice(code.indexOf('{'));
    if (options.functionType === 'normal' && options.args.length === 0 && !this.#varies(body)) {
      this.#pure.add(options.name);
    }
    for (const name of frame.choosing) {
      if (frame.assigned.has(name)) {
        throw new Ineligible(`finds its cells through '${name}', which it reassigns`);
      }
    }
    this.#codes.push(code);
    const scattered = code.indexOf(SCATTERED);
    if (scattered >= 0) {
      const id = /^(\w+?)_\d+/.exec(code.slice(scattered + SCATTERED.length))?.[1];
      throw new Ineligible(`reads '${id}', which it writes at computed cells`);
    }
    const stale = code.indexOf(STALE);
    if (stale >= 0) {
      const id = /^(\w+?)_\d+/.exec(code.slice(stale + STALE.length))?.[1];
      throw new Ineligible(`reads all of '${id}', which it writes`);
    }
    return code;
  }

  protected override computeEntry(options: FunctionDefinitionOptions): string {
    const size = [...(options.workgroupSize ?? []), 1, 1, 1].slice(0, 3) as [
      number,
      number,
      number,
    ];
    this.#single = size.every((each) => each === 1);
    for (const arg of options.args) {
      if (attributeOf(arg.decoratedType, '@builtin')?.[0] === 'global_invocation_id') {
        this.#frame?.names.set(arg.name, ID);
      }
    }
    const body = this._block(options.body, /* allowInlining */ false).code;
    options.determineReturnType();
    this.#slots = new Map(
      [...(this.#frame?.adds ?? [])].map(([id, count]) => {
        const buffer = this.#adds.get(id);
        const merged = buffer && this.#merges(buffer);
        return [id, merged ? Math.min(count, this.#wordsOf(buffer)) : count];
      }),
    );
    this.#records = new Map(this.#frame?.records);

    const wg = `uvec3(${size.join('u, ')}u)`;
    const prelude = options.args
      .filter((arg) => arg.used)
      .map((arg) => {
        const builtin = attributeOf(arg.decoratedType, '@builtin')?.[0];
        if (builtin === 'global_invocation_id') return `  uvec3 ${arg.name} = ${GID};`;
        if (builtin === 'local_invocation_id') return `  uvec3 ${arg.name} = ${GID} % ${wg};`;
        if (builtin === 'workgroup_id') return `  uvec3 ${arg.name} = ${GID} / ${wg};`;
        if (builtin === 'num_workgroups') return `  uvec3 ${arg.name} = ${WORKGROUPS};`;
        if (builtin === 'local_invocation_index') {
          const local = `(${GID} % ${wg})`;
          return `  uint ${arg.name} = ${local}.x + ${local}.y * ${size[0]}u + ${local}.z * ${size[0] * size[1]}u;`;
        }
        throw new Ineligible(`reads the '${String(builtin)}' builtin`);
      });
    const open = body.indexOf('\n');
    const kernel =
      body === ''
        ? `void ${options.name}() {}`
        : `void ${options.name}() ${body.slice(0, open + 1)}${prelude.map((line) => `${line}\n`).join('')}${body.slice(open + 1)}`;
    const kernelChunk = `${CHUNK} shared ${options.name}\n${kernel}`;

    for (const id of this.#scatters.keys()) {
      this.#buffers.delete(id);
      this.#textures.delete(id);
      this.#columns.delete(id);
      this.#maps.delete(id);
      if (this.#loaded.has(id)) {
        throw new Ineligible(`reads '${id}', which it writes at computed cells`);
      }
    }
    const buffers = [...this.#buffers.values()];
    if (!this.#plan) {
      if (
        buffers.length + this.#textures.size + this.#scatters.size + this.#adds.size === 0 &&
        this.#blocks.size === 0
      ) {
        throw new Ineligible('writes nothing');
      }
      for (const [id, block] of this.#blocks) {
        if (this.#buffers.has(id) || this.#scatters.has(id) || this.#adds.has(id)) {
          throw new Ineligible(`writes '${id}' other than at blocks it owns`);
        }
        if (block.sites.length > MOST_SITES) {
          throw new Ineligible(`owns more than ${MOST_SITES} cells of '${id}'`);
        }
        // Each axis's offsets lie within one step, so no two invocations share a cell
        for (const along of [0, 1] as const) {
          const offsets = block.sites.map((site) => site[along]);
          if (Math.max(...offsets) - Math.min(...offsets) >= block.step[along]) {
            throw new Ineligible(`owns blocks of '${id}' wider than their step`);
          }
        }
      }
      // Decided from the finished functions, not where the walk met them: a helper resolved
      // before the kernel's addition may still load after it
      const flow = flowOf(new Map([...this.#codeOf, [options.name, kernel]]), options.name);
      for (const id of this.#adds.keys()) {
        if (!this.#loads.has(id)) continue;
        if (this.#takes.has(id)) {
          throw new Ineligible(`reads '${id}' with atomicLoad, which it takes from`);
        }
        // Loads before every addition read what every invocation's loads would before any
        // addition, an order WebGPU's atomics may take
        if (flow.late.has(id)) {
          throw new Ineligible(`reads '${id}' with atomicLoad where it may have added to it`);
        }
      }
      // One path makes no more additions than its own
      for (const [id, count] of this.#slots) {
        const most = flow.most.get(id);
        if (!this.#takes.has(id) && most !== undefined && most > 0 && most < count) {
          this.#slots.set(id, most);
        }
      }
      if (this.#takes.size > 0) this.#checkTakes([...this.#codes, kernel].join('\n'));
      // Writes the counts missed, through a route other than a call, would overrun the records
      for (const id of this.#scatters.keys()) {
        if (!this.#records.get(id)) throw new Ineligible(`writes '${id}' where it cannot count`);
      }
      for (const id of this.#adds.keys()) {
        if (!this.#slots.get(id)) throw new Ineligible('adds where it cannot count');
      }
      for (const id of this.#columns) {
        if (this.#loaded.has(id)) {
          throw new Ineligible(`loads from '${id}', of which an invocation writes a whole column`);
        }
      }
      for (const { layout, whole, chosen } of buffers) {
        if (chosen && size.slice(chosen.length).some((each) => each !== 1)) {
          throw new Ineligible('chooses an invocation on fewer axes than its workgroup spans');
        }
        if (whole) continue;
        const covered = layout.width === undefined ? 1 : layout.height === undefined ? 2 : 3;
        if (size.slice(covered).some((each) => each !== 1)) {
          throw new Ineligible('writes a buffer from a workgroup larger than its indices cover');
        }
      }
      return kernelChunk;
    }
    const written = [...this.#buffers.keys(), ...this.#textures.keys()].toSorted().join();
    const planned = [...this.#plan.buffers.keys(), ...this.#plan.textures].toSorted().join();
    if (written !== planned) {
      throw new Error('@typegpu/kit-gl: the second resolution writes other resources');
    }

    const single = buffers.some((each) => each.whole && !each.chosen);
    // A pass of points covers every invocation, so it limits nothing
    const grid = single
      ? 1
      : Math.min(
          3,
          ...buffers.map(({ layout, chosen }) =>
            chosen
              ? chosen.length
              : layout.width === undefined
                ? 1
                : layout.height === undefined
                  ? 2
                  : 3,
          ),
          ...[...this.#textures.values()].map((texture) =>
            this.#columns.has(texture.id)
              ? 2
              : ({ '1d': 1, '2d': 2 }[texture.kind.dimension as string] ?? 3),
          ),
          ...[...this.#blocks.keys()].map(() => 2),
        );
    return [
      `${COMPUTE} ${size.join(' ')} ${grid} ${single ? 1 : 0}`,
      kernelChunk,
      this.#begin(),
      ...buffers.map((target, i) => this.#bufferPass(target, i, options.name)),
      ...[...this.#textures.values()].map((target, i) =>
        this.#texturePass(target, buffers.length + i, options.name),
      ),
      ...[...this.#scatters.values()].map((target, i) =>
        this.#scatterPass(target, buffers.length + this.#textures.size + i, options.name),
      ),
      ...[...this.#blocks.values()].map((block, i) =>
        this.#blockPass(
          block,
          buffers.length + this.#textures.size + this.#scatters.size + i,
          options.name,
        ),
      ),
      ...[...this.#adds.values()].map((buffer, i) =>
        this.#addPass(
          buffer,
          buffers.length + this.#textures.size + this.#scatters.size + this.#blocks.size + i,
          options.name,
        ),
      ),
    ].join('\n\n');
  }

  /** `kit_begin(gid)`: starts an invocation, with fresh private variables and own cells. */
  #begin(): string {
    const lines = [`  ${GID} = kit_g;`];
    for (const id of this.#takes) lines.push(`  kit_taken${this.#aim(id)} = 0u;`);
    for (const block of this.#blocks.values()) lines.push(...this.#blockBegin(block));
    for (const { id, dataType, init } of this.#privates) {
      lines.push(
        `  ${id} = ${init ?? this.ctx.resolveSnippet(this.typeInstantiation(dataType, [])).value};`,
      );
    }
    for (const written of this.#buffers.values()) {
      const { buffer, offset, array, layout } = written;
      const shadow = this.#bufferShadow(buffer, written, array);
      if (written.whole) {
        const load = this.load(buffer.source, buffer.base, buffer.schema, buffer.id);
        lines.push(`  ${shadow} = ${load};`);
        continue;
      }
      const element = (array as d.WgslArray).elementType as d.BaseData;
      const stride = wordsOf(strideOf(element));
      const address = `${buffer.base} + ${offset}u + ${linear(layout, 'kit_g')} * ${stride}u`;
      lines.push(`  ${shadow} = ${this.load(buffer.source, address, element, buffer.id)};`);
    }
    for (const texture of this.#textures.values()) {
      const old = this.samplerUniform(texture);
      const cell = this.#columns.has(texture.id)
        ? `ivec3(ivec2(kit_g.xy), ${Z})`
        : this.#ownCell(texture, 'kit_g');
      lines.push(`  ${this.#textureShadow(texture)} = texelFetch(${old}, ${cell}, 0);`);
    }
    return `${CHUNK} shared ${BEGIN}\nvoid ${BEGIN}(uvec3 kit_g) {\n${lines.join('\n')}\n}`;
  }

  /**
   * The shadows of a block array, one per site and a spare, and the function that names the
   * shadow an element index reads or writes: its site's, or the spare, loaded from the old contents.
   */
  #blockNames(
    buffer: BufferInfo,
    block: BlockPlan,
    array: d.WgslArray | undefined,
  ): { shadows: string; addresses: string; site: string } {
    const element = (array as d.WgslArray).elementType as d.BaseData;
    const type = `${this.ctx.resolve(element).value}`;
    const suffix = this.arraySuffix(element);
    const k = block.sites.length;
    const shadows = `kit_shadows_${buffer.id}`;
    const addresses = `kit_addresses_${buffer.id}`;
    const site = `kit_site_${buffer.id}`;
    const stride = wordsOf(strideOf(element));
    const load = this.load(
      buffer.source,
      `${buffer.base} + ${block.offset}u + i * ${stride}u`,
      element,
      buffer.id,
    );
    this.helper(
      site,
      `${type} ${shadows}[${k + 1}]${suffix};
uint ${addresses}[${k}];
uint ${site}(uint i) {
  for (int s = 0; s < ${k}; s++) {
    if (i == ${addresses}[s]) return uint(s);
  }
  ${shadows}[${k}] = ${load};
  return ${k}u;
}`,
    );
    return { shadows, addresses, site };
  }

  /** A block array's sites' cells for invocation `kit_g`, and their old contents, for `kit_begin`. */
  #blockBegin(block: Block): string[] {
    const { shadows, addresses } = this.#blockNames(block.buffer, block, block.array);
    const element = block.array.elementType as d.BaseData;
    const stride = wordsOf(strideOf(element));
    return block.sites.map(([c, r], s) => {
      const load = this.load(
        block.buffer.source,
        `${block.buffer.base} + ${block.offset}u + ${addresses}[${s}] * ${stride}u`,
        element,
        block.buffer.id,
      );
      return `  {
    int kit_c = int(kit_g.x) * ${block.step[0]} + (${c}) ${block.phase[0]};
    int kit_r = int(kit_g.y) * ${block.step[1]} + (${r}) ${block.phase[1]};
    ${addresses}[${s}] = kit_c >= 0 && kit_c < ${block.width} && kit_r >= 0 ? uint(kit_r * ${block.width} + kit_c) : 0xFFFFFFFFu;
    if (${addresses}[${s}] != 0xFFFFFFFFu) ${shadows}[${s}] = ${load};
  }`;
    });
  }

  /**
   * The pass of a block array: each element finds the site and invocation whose block holds it,
   * runs the kernel for that invocation and keeps that site's shadow.
   */
  #blockPass(block: Block, index: number, kernel: string): string {
    const { buffer, offset, array } = block;
    const range = buffer.range as string;
    const value = array.elementType as d.BaseData;
    const size = Math.ceil(d.sizeOf(value as d.AnyData) / 4);
    const stride = wordsOf(strideOf(value));
    const count =
      array.elementCount > 0
        ? `${array.elementCount}u`
        : `(${range}.y / 4u > ${offset}u ? (${range}.y / 4u - ${offset}u) / ${stride}u : 0u)`;
    const { shadows } = this.#blockNames(buffer, block, array);
    const pack = this.#packer(value);
    const target = [
      TARGET,
      'buffer',
      buffer.group,
      buffer.binding,
      buffer.id,
      offset * 4,
      stride * 4,
      d.sizeOf(value as d.AnyData),
    ].join(' ');
    const sites = block.sites
      .map(
        ([c, r], s) => `    {
      int kit_dx = kit_c - (${c}) - (0 ${block.phase[0]});
      int kit_dy = kit_r - (${r}) - (0 ${block.phase[1]});
      if (kit_dx >= 0 && kit_dy >= 0 && kit_dx % ${block.step[0]} == 0 && kit_dy % ${block.step[1]} == 0) {
        kit_g = uvec3(uint(kit_dx / ${block.step[0]}), uint(kit_dy / ${block.step[1]}), 0u);
        kit_s = ${s};
      }
    }`,
      )
      .join('\n');
    return `${CHUNK} pass${index} main
${target}
layout(location = 0) out uvec4 ${RESULT};
void main() {
  ivec2 kit_frag = ivec2(gl_FragCoord.xy);
  uvec4 kit_old = texelFetch(${buffer.id}, kit_frag, 0);
  uvec4 kit_new = kit_old;
  uint kit_first = ${range}.x + ${offset}u;
  uint kit_end = ${range}.x + ${range}.y / 4u;
  uint kit_count = ${count};
  uint kit_base = (uint(kit_frag.y) * uint(textureSize(${buffer.id}, 0).x) + uint(kit_frag.x)) * 4u;
  uint kit_last = 0xFFFFFFFFu;
  for (int kit_k = 0; kit_k < 4; kit_k++) {
    uint kit_at = kit_base + uint(kit_k);
    if (kit_at < kit_first || kit_at >= kit_end) continue;
    uint kit_element = (kit_at - kit_first) / ${stride}u;
    uint kit_inner = kit_at - kit_first - kit_element * ${stride}u;
    if (kit_element >= kit_count || kit_inner >= ${size}u) continue;
    int kit_c = int(kit_element % ${block.width}u);
    int kit_r = int(kit_element / ${block.width}u);
    uvec3 kit_g = uvec3(0xFFFFFFFFu);
    int kit_s = -1;
${sites}
    if (kit_s < 0 || any(greaterThanEqual(kit_g, ${INVOCATIONS}))) continue;
    if (kit_element != kit_last) {
      ${BEGIN}(kit_g);
      ${kernel}();
      kit_last = kit_element;
    }
    kit_new[kit_k] = ${pack}(${shadows}[kit_s], kit_inner, kit_old[kit_k]);
  }
  ${RESULT} = kit_new;
}`;
  }

  #bufferPass(written: WrittenBuffer, index: number, kernel: string): string {
    const { buffer, offset, array, layout } = written;
    const range = buffer.range as string;
    const value = written.whole
      ? buffer.schema
      : ((array as d.WgslArray).elementType as d.BaseData);
    const size = Math.ceil(d.sizeOf(value as d.AnyData) / 4);
    const stride = written.whole ? size : wordsOf(strideOf(value));
    const elementCount = (array as d.WgslArray | undefined)?.elementCount ?? 0;
    const count = written.whole
      ? '1u'
      : elementCount > 0
        ? `${elementCount}u`
        : `(${range}.y / 4u > ${offset}u ? (${range}.y / 4u - ${offset}u) / ${stride}u : 0u)`;
    const shadow = this.#bufferShadow(buffer, written, array);
    const pack = this.#packer(value);
    const target = [
      TARGET,
      'buffer',
      buffer.group,
      buffer.binding,
      buffer.id,
      offset * 4,
      stride * 4,
      d.sizeOf(value as d.AnyData),
      ...(this.#guarded(layout) || layout.width === undefined ? [] : [layout.width]),
      ...(this.#guarded(layout) || layout.height === undefined ? [] : [layout.height]),
    ].join(' ');
    const { chosen } = written;
    const gid = chosen
      ? `uvec3(${[0, 1, 2].map((i) => chosen[i] ?? '0u').join(', ')})`
      : written.whole
        ? 'uvec3(0u)'
        : invocationOf(layout, 'kit_element');
    // Locals start with `kit_`, so the kernel's functions and globals keep their names here
    return `${CHUNK} pass${index} main
${target}
layout(location = 0) out uvec4 ${RESULT};
void main() {
  ivec2 kit_frag = ivec2(gl_FragCoord.xy);
  uvec4 kit_old = texelFetch(${buffer.id}, kit_frag, 0);
  uvec4 kit_new = kit_old;
  uint kit_first = ${range}.x + ${offset}u;
  uint kit_end = ${range}.x + ${range}.y / 4u;
  uint kit_count = ${count};
  uint kit_base = (uint(kit_frag.y) * uint(textureSize(${buffer.id}, 0).x) + uint(kit_frag.x)) * 4u;
  uint kit_last = 0xFFFFFFFFu;
  for (int kit_k = 0; kit_k < 4; kit_k++) {
    uint kit_at = kit_base + uint(kit_k);
    if (kit_at < kit_first || kit_at >= kit_end) continue;
    uint kit_element = (kit_at - kit_first) / ${stride}u;
    uint kit_inner = kit_at - kit_first - kit_element * ${stride}u;
    if (kit_element >= kit_count || kit_inner >= ${size}u) continue;
    uvec3 kit_g = ${gid};
    if (any(greaterThanEqual(kit_g, ${INVOCATIONS}))) continue;
    if (kit_element != kit_last) {
      ${BEGIN}(kit_g);
      ${kernel}();
      kit_last = kit_element;
    }
    kit_new[kit_k] = ${pack}(${shadow}, kit_inner, kit_old[kit_k]);
  }
  ${RESULT} = kit_new;
}`;
  }

  #texturePass(texture: TextureInfo, index: number, kernel: string): string {
    const column = this.#columns.has(texture.id);
    const { format } = texture.schema as unknown as { format: string };
    const old = this.samplerUniform(texture);
    const target = [
      TARGET,
      'texture',
      texture.group,
      texture.binding,
      old,
      texture.kind.dimension,
      format,
    ].join(' ');
    const map = this.#maps.get(texture.id);
    const owner: string[] = [];
    if (map) {
      // A coordinate with a residue depends on the others, so it comes last
      const order = [...map.keys()].toSorted((a, b) => Number(!!map[a]?.r) - Number(!!map[b]?.r));
      for (const i of order) {
        const { step, r } = map[i] as TexelMap[number];
        const c = AXES[i];
        const local = r
          ? `(kit_cell.${c} - ((${affineText(r.of, 'kit_g')}) & ${r.mask}u))`
          : `kit_cell.${c}`;
        owner.push(`  kit_mine = kit_mine && ${local} % ${step}u == 0u;`);
        owner.push(`  kit_g.${c} = ${local} / ${step}u;`);
      }
    }
    const gid = column ? 'uvec3(kit_cell.xy, 0u)' : map ? 'uvec3(0u)' : 'kit_cell';
    const fallback = column ? `ivec3(ivec2(kit_cell.xy), ${Z})` : this.#cellOf(texture, 'kit_cell');
    const type = this.#texelType(texture);
    // A layered target's draw writes several layers, one output each
    const { dimension } = texture.kind;
    const layered = dimension === '2d-array' || dimension === '3d';
    const outputs = Array.from({ length: layered ? LAYERS_PER_DRAW : 1 }, (_, i) =>
      layered ? `${RESULT}${i}` : RESULT,
    );
    if (!layered) {
      return `${CHUNK} pass${index} main
${target}
layout(location = 0) out ${type} ${RESULT};
void main() {
  ${Z} = ${LAYER};
  uvec3 kit_cell = uvec3(uvec2(gl_FragCoord.xy), uint(${LAYER}));
  uvec3 kit_g = ${gid};
  bool kit_mine = true;
${owner.join('\n')}
  if (!kit_mine || any(greaterThanEqual(kit_g, ${INVOCATIONS}))) {
    ${RESULT} = texelFetch(${old}, ${fallback}, 0);
    return;
  }
  ${BEGIN}(kit_g);
  ${kernel}();
  ${RESULT} = ${this.#textureShadow(texture)};
}`;
    }
    return `${CHUNK} pass${index} main
${target}
${outputs.map((name, i) => `layout(location = ${i}) out ${type} ${name};`).join('\n')}
${type} kit_cellAt(int kit_layerHere) {
  ${Z} = kit_layerHere;
  uvec3 kit_cell = uvec3(uvec2(gl_FragCoord.xy), uint(kit_layerHere));
  uvec3 kit_g = ${gid};
  bool kit_mine = true;
${owner.join('\n')}
  if (!kit_mine || any(greaterThanEqual(kit_g, ${INVOCATIONS}))) {
    return texelFetch(${old}, ${fallback}, 0);
  }
  ${BEGIN}(kit_g);
  ${kernel}();
  return ${this.#textureShadow(texture)};
}
void main() {
${outputs.map((name, i) => `  ${name} = kit_cellAt(${LAYER} + ${i});`).join('\n')}
}`;
  }

  /**
   * A pass of points, in two vertex stages. The capture stage runs the kernel once per invocation
   * and writes what each write site wrote as transform feedback varyings, `kit_c<site>_<field>`;
   * the draw stage reads one site's record as attributes, `kit_a_<field>`, and puts its point on
   * the written cell.
   */
  /**
   * A pass that sums what the kernel adds to a buffer: each invocation records up to `slots`
   * additions, and every record becomes a point on its word, which the device accumulates.
   */
  #addPass(buffer: BufferInfo, index: number, kernel: string): string {
    const slots = this.#plan?.slots?.get(buffer.id) ?? 0;
    const outputs: string[] = [];
    const lines: string[] = [];
    for (let slot = 0; slot < slots; slot++) {
      for (const name of ['hit', 'at', 'value'])
        outputs.push(`flat out uint kit_c${slot}_${name};`);
      lines.push(
        `  kit_c${slot}_hit = ${slot}u < ${ADDED} ? 1u : 0u;`,
        `  kit_c${slot}_at = ${ADDED}_at[${slot}];`,
        `  kit_c${slot}_value = ${ADDED}_amount[${slot}];`,
      );
    }
    const range = buffer.range as string;
    const directive = [TARGET, 'buffer', buffer.group, buffer.binding, buffer.id, 0, 4, 4];
    return `${CHUNK} pass${index} main
${directive.join(' ')}
${ACCUMULATE} ${slots}${this.#takes.has(buffer.id) ? ` take ${this.#aim(buffer.id)}` : ''}${this.#operationOf(buffer.id)}
${outputs.join('\n')}
void main() {
  uint kit_id = uint(gl_VertexID);
  uvec3 kit_g = uvec3(kit_id % ${INVOCATIONS}.x, (kit_id / ${INVOCATIONS}.x) % ${INVOCATIONS}.y, kit_id / (${INVOCATIONS}.x * ${INVOCATIONS}.y));
  ${ADDED} = 0u;
  ${AIM} = ${this.#aim(buffer.id)};
  ${BEGIN}(kit_g);
  ${kernel}();
${lines.join('\n')}
  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}

${CHUNK} points${index} main
in uint kit_a_hit;
in uint kit_a_at;
in uint kit_a_value;
flat out uint kit_value;
flat out uint kit_lane;
void main() {
  gl_PointSize = 1.0;
  gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
  if (kit_a_hit == 0u) return;
  uint kit_w = kit_a_at;
  if (kit_w < ${range}.x || kit_w >= ${range}.x + ${range}.y / 4u) return;
  ivec2 kit_size = textureSize(${buffer.id}, 0);
  uint kit_t = kit_w >> 2u;
  vec2 kit_xy = vec2(float(kit_t % uint(kit_size.x)), float(kit_t / uint(kit_size.x)));
  gl_Position = vec4((kit_xy + 0.5) / vec2(kit_size) * 2.0 - 1.0, 0.0, 1.0);
  kit_value = kit_a_value;
  kit_lane = kit_w & 3u;
}`;
  }

  #scatterPass(target: Scatter, index: number, kernel: string): string {
    const id = target.kind === 'buffer' ? target.buffer.id : target.texture.id;
    const sites = this.#slotsOf(id);
    const prefix = `kit_p${this.#aim(id)}`;
    const capture = (fields: readonly [string, string][], copy: (site: number) => string[]) => {
      const outputs: string[] = [];
      const lines: string[] = [];
      for (let site = 0; site < sites; site++) {
        for (const [name, type] of fields) outputs.push(`flat out ${type} kit_c${site}_${name};`);
        lines.push(`  kit_c${site}_hit = ${site}u < ${prefix}_n ? 1u : 0u;`, ...copy(site));
      }
      const resets = [`  ${prefix}_n = 0u;`];
      return `${outputs.join('\n')}
void main() {
  uint kit_id = uint(gl_VertexID);
  uvec3 kit_g = uvec3(kit_id % ${INVOCATIONS}.x, (kit_id / ${INVOCATIONS}.x) % ${INVOCATIONS}.y, kit_id / (${INVOCATIONS}.x * ${INVOCATIONS}.y));
${resets.join('\n')}
  ${AIM} = ${this.#aim(id)};
  ${BEGIN}(kit_g);
  ${kernel}();
${lines.join('\n')}
  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}`;
    };
    if (target.kind === 'buffer') {
      const { buffer, array, offset } = target;
      const type = array.elementType as d.BaseData;
      const stride = wordsOf(strideOf(type));
      const words = Math.ceil(d.sizeOf(type as d.AnyData) / 4);
      const range = buffer.range as string;
      const vectors = Math.ceil(words / 4);
      const fields: [string, string][] = [
        ['hit', 'uint'],
        ['at', 'uint'],
        ...Array.from({ length: vectors }, (_, j): [string, string] => [`v${j}`, 'uvec4']),
      ];
      const directive = [
        TARGET,
        'buffer',
        buffer.group,
        buffer.binding,
        buffer.id,
        offset * 4,
        stride * 4,
        d.sizeOf(type as d.AnyData),
      ].join(' ');
      // An element that fits its data texel is written whole, one draw per place in the texel
      const lanes = (stride === 2 || stride === 4) && offset % stride === 0 ? stride : 1;
      const word = (k: string) =>
        vectors === 1
          ? `kit_a_v0[${k}]`
          : Array.from({ length: vectors }, (_, j) => j)
              .toReversed()
              .reduce(
                // Words below 4 (j + 1) are in vector j or an earlier one
                (rest, j) =>
                  rest
                    ? `(${k} < ${4 * (j + 1)}u ? kit_a_v${j}[${k} - ${4 * j}u] : ${rest})`
                    : `kit_a_v${j}[${k} - ${4 * j}u]`,
                '',
              );
      const drawn =
        lanes > 1
          ? `  uint kit_w = kit_a_at;
  if (kit_w < ${range}.x || kit_w + ${words - 1}u >= ${range}.x + ${range}.y / 4u || (kit_w & 3u) / ${lanes}u != uint(kit_phase)) return;
  uvec4 kit_v = uvec4(0u);
${Array.from({ length: words }, (_, k) => `  kit_v[(kit_w & 3u) + ${k}u] = kit_a_v${k >> 2}[${k & 3}];`).join('\n')}`
          : `  uint kit_w = kit_a_at + uint(kit_part);
  if (kit_w < ${range}.x || kit_w >= ${range}.x + ${range}.y / 4u || (kit_w & 3u) != uint(kit_phase)) return;
  uvec4 kit_v = uvec4(${word('uint(kit_part)')});`;
      return `${CHUNK} pass${index} main
${directive}
${SCATTER} ${sites} ${words} ${lanes}
${capture(fields, (site) => [
  `  kit_c${site}_at = ${prefix}_at[${site}];`,
  ...Array.from({ length: vectors }, (_, j) => `  kit_c${site}_v${j} = ${prefix}_v${j}[${site}];`),
])}

${CHUNK} points${index} main
${fields.map(([name, t]) => `in ${t} kit_a_${name};`).join('\n')}
uniform int kit_part;
uniform int kit_phase;
flat out uvec4 kit_value;
void main() {
  gl_PointSize = 1.0;
  gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
  if (kit_a_hit == 0u) return;
${drawn}
  ivec2 kit_size = textureSize(${buffer.id}, 0);
  uint kit_t = kit_w >> 2u;
  vec2 kit_xy = vec2(float(kit_t % uint(kit_size.x)), float(kit_t / uint(kit_size.x)));
  gl_Position = vec4((kit_xy + 0.5) / vec2(kit_size) * 2.0 - 1.0, 0.0, 1.0);
  kit_value = kit_v;
}`;
    }
    const { texture } = target;
    const { format } = texture.schema as unknown as { format: string };
    const old = this.samplerUniform(texture);
    const type = this.#texelType(texture);
    const { dimension } = texture.kind;
    const layered = dimension === '2d-array' || dimension === '3d';
    const size = layered
      ? `textureSize(${old}, 0)`
      : dimension === '1d'
        ? `ivec3(textureSize(${old}, 0).x, 1, 1)`
        : `ivec3(textureSize(${old}, 0), 1)`;
    const fields: [string, string][] = [
      ['hit', 'uint'],
      ['at', 'ivec3'],
      ['value', type],
    ];
    const directive = [TARGET, 'texture', texture.group, texture.binding, old, dimension, format];
    return `${CHUNK} pass${index} main
${directive.join(' ')}
${SCATTER} ${sites} 1
${capture(fields, (site) => [
  `  kit_c${site}_at = ${prefix}_at[${site}];`,
  `  kit_c${site}_value = ${prefix}_value[${site}];`,
])}

${CHUNK} points${index} main
${fields.map(([name, t]) => `in ${t} kit_a_${name};`).join('\n')}
flat out ${type} kit_value;
void main() {
  gl_PointSize = 1.0;
  gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
  if (kit_a_hit == 0u) return;
  ivec3 kit_size = ${size};
  if (any(lessThan(kit_a_at, ivec3(0))) || any(greaterThanEqual(kit_a_at, kit_size))) return;
  if (kit_a_at.z != ${LAYER}) return;
  gl_Position = vec4((vec2(kit_a_at.xy) + 0.5) / vec2(kit_size.xy) * 2.0 - 1.0, 0.0, 1.0);
  kit_value = kit_a_value;
}`;
  }

  /**
   * A function that gives word `i` of a value as WGSL lays it out in memory, or `old` for padding.
   * It mirrors the loaders.
   */
  #packer(schema: d.BaseData): string {
    const words: string[] = [];
    const flatten = (type: d.BaseData, value: string, word: number): void => {
      const data = undecorate(type);
      const kind = data.type as string;
      if (d.isAtomic(data)) return flatten(data.inner as d.BaseData, value, word);
      if (kind === 'f32') words[word] = `floatBitsToUint(${value})`;
      else if (kind === 'i32') words[word] = `uint(${value})`;
      else if (kind === 'u32') words[word] = value;
      else if (/^vec[234][fiu]$/.test(kind)) {
        const size = Number(kind[3]);
        const scalar = { f: d.f32, i: d.i32, u: d.u32 }[kind[4] as 'f' | 'i' | 'u'];
        for (let c = 0; c < size; c++) flatten(scalar, `${value}.${AXES[c] ?? 'w'}`, word + c);
      } else if (/^mat[234]x[234]f$/.test(kind)) {
        const columns = Number(kind[3]);
        const rows = Number(kind[5]);
        const column = d[`vec${rows}f` as 'vec4f'];
        for (let c = 0; c < columns; c++) {
          flatten(column, `${value}[${c}]`, word + c * (rows === 2 ? 2 : 4));
        }
      } else if (d.isWgslStruct(data)) {
        for (const member of membersOf(data)) {
          flatten(
            member.type,
            `${value}.${memberName(member.name)}`,
            word + wordsOf(member.offset),
          );
        }
      } else if (d.isWgslArray(data) && data.elementCount > 0) {
        const element = data.elementType as d.BaseData;
        const stride = wordsOf(strideOf(element));
        for (let i = 0; i < data.elementCount; i++) {
          flatten(element, `${value}[${i}]`, word + i * stride);
        }
      } else {
        throw new Ineligible(`writes a '${kind}', which has no GLSL layout`);
      }
    };
    flatten(schema, 'v', 0);
    const name = this.ctx.makeUniqueIdentifier(`kit_pack`, 'global');
    const type = this.ctx.resolve(schema).value;
    const cases = words
      .map((value, i) => (value === undefined ? '' : `  if (i == ${i}u) return ${value};\n`))
      .join('');
    this.chunk(
      'shared',
      [name],
      `uint ${name}(${type} v${this.arraySuffix(schema)}, uint i, uint old) {\n${cases}  return old;\n}`,
    );
    return name;
  }
}

/**
 * What numbers a resolution's bindings: its catch-all group's resources in binding order, and the
 * placeholders it replaces with group indices once it is done.
 */
interface Numbering {
  readonly fixedBindings: readonly { readonly resource: unknown }[];
  readonly bindGroupLayoutsToPlaceholderMap: ReadonlyMap<object, string>;
}

/** TypeGPU's placeholder for the catch-all group's index, which a resolution replaces at its end. */
const CATCHALL = '#CATCHALL#';
/** Starts a placeholder of the outer resolution, which the inner one leaves alone. */
const OUTER = '#KIT_OUTER_';

/** The reason an error gives for a kernel's GPU form failing, with the function it is in. */
function reasonOf(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  const own = /GPU compute: (.*)$/m.exec(message);
  const kit = /@typegpu\/kit-gl: (.*)$/m.exec(message);
  const reason = own?.[1] ?? kit?.[1] ?? message.split('\n').at(-1) ?? message;
  const trace = [...message.matchAll(/^- (?:fn\*?|computeFn|var):(\w+)/gm)].at(-1)?.[1];
  return trace && !own ? `${reason} (in '${trace}')` : reason;
}

/**
 * A compute pipeline's GPU form, or why it has none. With `outer`, the resolution the form is
 * attached to, the form names groups with `outer`'s placeholders, which its resolution replaces.
 */
export function computeForm(
  item: object,
  outer?: ResolutionCtx,
): { code: string } | { reason: string } {
  type Options = Parameters<typeof tgpu.resolveWithContext>[1];
  try {
    const analysis = new KitGlslComputeGenerator(undefined, outer);
    tgpu.resolveWithContext([item as never], { unstable_shaderGenerator: analysis } as Options);
    const generator = new KitGlslComputeGenerator(analysis.plan, outer);
    const { code } = tgpu.resolveWithContext([item as never], {
      unstable_shaderGenerator: generator,
    } as Options);
    return { code: code.replaceAll(OUTER, '#') };
  } catch (error) {
    return { reason: reasonOf(error) };
  }
}

/**
 * Adds the GPU form of a compute pipeline to the code a resolution with `ctx` returns, when
 * `isCompute` says the pipeline was one, as a line the CPU backend reads as a comment.
 */
export function attachComputeForm(ctx: ResolutionCtx, isCompute: () => boolean): void {
  const resolve = ctx.resolve.bind(ctx) as (item: unknown, schema?: unknown) => ResolvedSnippet;
  let first = true;
  (ctx as { resolve: unknown }).resolve = (item: unknown, schema?: unknown) => {
    if (!first) return resolve(item, schema);
    first = false;
    const result = resolve(item, schema);
    if (!isCompute()) return result;
    const form = computeForm(item as object, ctx);
    const line =
      'code' in form
        ? `${GPU_FORM}${JSON.stringify(form.code)}`
        : `${CPU_ONLY}${JSON.stringify(form.reason)}`;
    return withValue(`${result.value}\n${line}`, result);
  };
}

export type { Element };
