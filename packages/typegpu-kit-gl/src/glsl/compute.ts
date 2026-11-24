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
}

/** A buffer a kernel writes. */
export interface WrittenPlan {
  /** The invocation owns the whole binding, as the dispatch's only one. */
  readonly whole: boolean;
  /** The written array's first word, for a buffer written element by element. */
  readonly offset: number;
  readonly layout: Layout;
}

/** What a kernel writes: found by a first resolution, used by the second. */
export interface ComputePlan {
  readonly buffers: ReadonlyMap<string, WrittenPlan>;
  /** Each written storage texture's id. */
  readonly textures: ReadonlySet<string>;
  /** The written 3d and 2d-array textures whose invocations own the column at their `xy`. */
  readonly columns: ReadonlySet<string>;
}

const GID = 'kit_gid';
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

/** `x * gid.x + y * gid.y + z * gid.z + k`, in integers. */
interface Affine {
  readonly x: number;
  readonly y: number;
  readonly z: number;
  readonly k: number;
}

/** A value that provably derives from `global_invocation_id`: one affine form per component. */
type Own = readonly Affine[];

const constant = (k: number): Affine => ({ x: 0, y: 0, z: 0, k });
const axis = (name: 'x' | 'y' | 'z'): Affine => ({ ...constant(0), [name]: 1 });
const ID: Own = [axis('x'), axis('y'), axis('z')];

/** The axes an own value is, component by component, such as `xy`, when each is one plain axis. */
function axesOf(own: Own | undefined): string | undefined {
  if (!own) return undefined;
  let axes = '';
  for (const { x, y, z, k } of own) {
    if (k !== 0 || x + y + z !== 1 || [x, y, z].some((c) => c !== 0 && c !== 1)) return undefined;
    axes += x ? 'x' : y ? 'y' : 'z';
  }
  return axes;
}

/** The layout an own array index is, if it gives each invocation one element. */
function layoutOf(own: Own | undefined): Layout | undefined {
  if (own?.length !== 1) return undefined;
  const { x, y, z, k } = own[0] as Affine;
  if (x !== 1 || k !== 0 || y < 0 || z < 0) return undefined;
  if (y === 0) return z === 0 ? {} : undefined;
  if (z === 0) return { width: y };
  return z % y === 0 ? { width: y, height: z / y } : undefined;
}

const sameLayout = (a: Layout, b: Layout) => a.width === b.width && a.height === b.height;

/** The element index a layout gives the invocation `gid` stands for, as GLSL. */
function linear(layout: Layout, gid: string): string {
  if (layout.width === undefined) return `${gid}.x`;
  const rows = `${gid}.y * ${layout.width}u`;
  if (layout.height === undefined) return `(${gid}.x + ${rows})`;
  return `(${gid}.x + ${rows} + ${gid}.z * ${layout.width * layout.height}u)`;
}

/** The invocation an element belongs to under a layout, as GLSL. */
function invocationOf(layout: Layout, element: string): string {
  const { width, height } = layout;
  if (width === undefined) return `uvec3(${element}, 0u, 0u)`;
  if (height === undefined) return `uvec3(${element} % ${width}u, ${element} / ${width}u, 0u)`;
  return `uvec3(${element} % ${width}u, (${element} / ${width}u) % ${height}u, ${element} / ${width * height}u)`;
}

type Value = Own | number;

function scale(value: Value, by: number): Value {
  if (typeof value === 'number') return value * by;
  return value.map(({ x, y, z, k }) => ({ x: x * by, y: y * by, z: z * by, k: k * by }));
}

function add(a: Value, b: Value): Value | undefined {
  if (typeof a === 'number' && typeof b === 'number') return a + b;
  const left = typeof a === 'number' ? undefined : a;
  const right = typeof b === 'number' ? undefined : b;
  if (left && right) {
    if (left.length !== right.length) return undefined;
    return left.map((each, i) => {
      const other = right[i] as Affine;
      return { x: each.x + other.x, y: each.y + other.y, z: each.z + other.z, k: each.k + other.k };
    });
  }
  const own = (left ?? right) as Own;
  const k = (typeof a === 'number' ? a : b) as number;
  return own.length === 1 ? [{ ...(own[0] as Affine), k: (own[0] as Affine).k + k }] : undefined;
}

/**
 * Reads GLSL integer arithmetic over `names` into affine forms: sums, differences, products with
 * constants, casts and vector constructors. Anything else is not provably own.
 */
class Arithmetic {
  readonly #text: string;
  readonly #names: ReadonlyMap<string, Own>;
  #at = 0;

  constructor(text: string, names: ReadonlyMap<string, Own>) {
    this.#text = text;
    this.#names = names;
  }

  read(): Value | undefined {
    const value = this.#sum();
    return this.#at === this.#text.length ? value : undefined;
  }

  #peek(): string | undefined {
    return this.#text[this.#at];
  }

  #sum(): Value | undefined {
    let value = this.#product();
    while (value !== undefined && (this.#peek() === '+' || this.#peek() === '-')) {
      const sign = this.#text[this.#at++] === '+' ? 1 : -1;
      const other = this.#product();
      value = other === undefined ? undefined : add(value, scale(other, sign));
    }
    return value;
  }

  #product(): Value | undefined {
    let value = this.#factor();
    while (value !== undefined && this.#peek() === '*') {
      this.#at++;
      const other = this.#factor();
      if (other === undefined) return undefined;
      if (typeof value === 'number') value = scale(other, value);
      else if (typeof other === 'number') value = scale(value, other);
      else return undefined;
    }
    return value;
  }

  #factor(): Value | undefined {
    const rest = this.#text.slice(this.#at);
    const number = /^(\d+)u?(?![\w.])/.exec(rest);
    if (number) {
      this.#at += number[0].length;
      return Number(number[1]);
    }
    if (rest.startsWith('(')) {
      this.#at++;
      const value = this.#sum();
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
      if (call[1] === 'uint' || call[1] === 'int') return parts.length === 1 ? parts[0] : undefined;
      const own = parts.flatMap((part) => (typeof part === 'number' ? [constant(part)] : part));
      return own.length === Number(call[1]?.at(-1)) ? own : undefined;
    }
    const name = /^([A-Za-z_]\w*)(?:\.([xyzw]+))?/.exec(rest);
    if (name) {
      const own = this.#names.get(name[1] as string);
      if (!own) return undefined;
      this.#at += name[0].length;
      if (!name[2]) return own;
      const picked = [...name[2]].map((c) => own['xyzw'.indexOf(c)]);
      return picked.every(Boolean) ? (picked as Affine[]) : undefined;
    }
    return undefined;
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
  /** Whether its code depends on its arguments being the invocation's own. */
  relies: boolean;
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
        [...this.#buffers].map(([id, { whole, offset, layout }]) => [
          id,
          { whole, offset, layout },
        ]),
      ),
      textures: new Set(this.#textures.keys()),
      columns: new Set(this.#columns),
    };
  }

  protected override header(): string {
    return COMPUTE_HEADER;
  }

  override initGenerator(ctx: ResolutionCtx): void {
    super.initGenerator(ctx);
    for (const name of [GID, BEGIN, RESULT, INVOCATIONS, WORKGROUPS, LAYER]) {
      ctx.reserveIdentifier(name, 'global');
    }
    this.chunk(
      'shared',
      [GID, INVOCATIONS, WORKGROUPS, LAYER],
      `uvec3 ${GID};\nuniform highp uvec3 ${INVOCATIONS};\nuniform highp uvec3 ${WORKGROUPS};\nuniform int ${LAYER};`,
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
    const value = new Arithmetic(text.replace(/\s+/g, ''), frame.names).read();
    return typeof value === 'object' ? value : undefined;
  }

  protected override ownIndex(index: Snippet): unknown {
    return layoutOf(this.#own(index));
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
    this.#frame?.names.delete(name);
    return super._emitVarDecl(keyword, name, dataType, rhs);
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
    let result: T;
    try {
      result = call();
    } finally {
      this.#pending = outer;
    }
    const name = typeof result?.value === 'string' ? /^(\w+)\(/.exec(result.value)?.[1] : undefined;
    if (name === undefined || !this.#relies.has(name)) return result;
    if (this.#assumed.get(name) !== JSON.stringify(owns.some(Boolean) ? owns : [])) {
      throw new Ineligible(`calls '${name}' with other invocation indices than it was made for`);
    }
    this.#relyOnOwn();
    return result;
  }

  override call(callee: AnyFn, args: readonly Snippet[], calleeSnippet?: Snippet): Snippet {
    const name = getName(callee) ?? '';
    if (name === 'atomicLoad' || name === 'atomicStore' || name in ATOMICS) {
      return this.#atomic(name, args);
    }
    return this.#calling(args, () => super.call(callee, args, calleeSnippet));
  }

  /**
   * An atomic on a cell the invocation owns, or on a binding a single invocation owns whole, is
   * sequential: a read, or a read-modify-write of what the invocation holds, returning the old
   * value. Atomics on cells other invocations share stay on the CPU.
   */
  #atomic(name: string, args: readonly Snippet[]): Snippet {
    const target = args[0];
    const place = target && this.placeOf(target);
    if (!target || !place) throw new Ineligible(`'${name}' outside a buffer`);
    const inner = (undecorate(place.schema) as d.Atomic).inner as d.BaseData;
    const value = this.ctx.resolveSnippet(target).value;
    if (name === 'atomicLoad') return snip(value, inner, 'runtime', false);
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
    if (Array.isArray(statement)) {
      if (statement[0] === NODE.assignmentExpr) this.#checkWrite(statement[1] as Expression);
      if (statement[0] === NODE.postUpdate) this.#checkWrite(statement[2] as Expression);
    }
    return super._statement(statement);
  }

  #checkWrite(node: Expression): void {
    const target = this._expression(node);
    const place = this.placeOf(target);
    const frame = this.#frame;
    if (!place) {
      const text = target.dataType === UnknownData ? '' : this.ctx.resolveSnippet(target).value;
      const root = /^\s*([A-Za-z_]\w*)/.exec(text)?.[1];
      if (root && frame?.names.has(root)) {
        throw new Ineligible(`modifies '${root}', which indices are derived from`);
      }
      return;
    }
    this.#recordPlace(place, 'writes');
  }

  /** Records a write to a place, or rejects the kernel when the place is not provably owned. */
  #recordPlace(place: Place, verb: string): void {
    const { buffer, element } = place;
    const frame = this.#frame;
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
    if (!element) throw new Ineligible(`${verb} '${buffer.id}' outside an array`);
    if (!frame?.entry && !frame?.names.size) {
      throw new Ineligible(`a helper function ${verb} '${buffer.id}'`);
    }
    throw new Ineligible(`${verb} an element of '${buffer.id}' that is not the invocation's own`);
  }

  #record(buffer: BufferInfo, written: WrittenPlan, array?: d.WgslArray): void {
    const known = this.#buffers.get(buffer.id);
    if (written.whole || known?.whole) {
      const size = d.sizeOf(buffer.schema as d.AnyData);
      if (size === 0 || size > MAX_WHOLE_BYTES) {
        throw new Ineligible(
          `writes '${buffer.id}' whole, which is larger than ${MAX_WHOLE_BYTES} bytes`,
        );
      }
      this.#buffers.set(buffer.id, { buffer, whole: true, offset: 0, layout: {} });
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
    if (!owns && !column) {
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
    if (!this.#plan?.columns.has(texture.id)) return `${shadow} = ${value}`;
    const layer = dimension === '3d' ? `${this.ivecOf(args[1], 3)}.z` : this.intOf(args[2]);
    return `${shadow} = (${layer}) == ${LAYER} ? ${value} : ${shadow}`;
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
      return typeof value === 'object' ? axesOf(value) : undefined;
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
      if (this.#isStorage(texture) && this.#plan?.textures.has(texture.id)) {
        const cell = this.#cellIn(texture, args);
        const own = this.#cellOf(texture, GID);
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
      relies: false,
    };
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

    const buffers = [...this.#buffers.values()];
    if (!this.#plan) {
      if (buffers.length + this.#textures.size === 0) throw new Ineligible('writes nothing');
      for (const id of this.#columns) {
        if (this.#loaded.has(id)) {
          throw new Ineligible(`loads from '${id}', of which an invocation writes a whole column`);
        }
      }
      for (const { layout, whole } of buffers) {
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

    const single = buffers.some((each) => each.whole);
    const grid = single
      ? 1
      : Math.min(
          ...buffers.map(({ layout }) =>
            layout.width === undefined ? 1 : layout.height === undefined ? 2 : 3,
          ),
          ...[...this.#textures.values()].map((texture) =>
            this.#columns.has(texture.id)
              ? 2
              : ({ '1d': 1, '2d': 2 }[texture.kind.dimension as string] ?? 3),
          ),
        );
    return [
      `${COMPUTE} ${size.join(' ')} ${grid} ${single ? 1 : 0}`,
      kernelChunk,
      this.#begin(),
      ...buffers.map((target, i) => this.#bufferPass(target, i, options.name)),
      ...[...this.#textures.values()].map((target, i) =>
        this.#texturePass(target, buffers.length + i, options.name),
      ),
    ].join('\n\n');
  }

  /** `kit_begin(gid)`: starts an invocation, with fresh private variables and own cells. */
  #begin(): string {
    const lines = [`  ${GID} = gid;`];
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
      const address = `${buffer.base} + ${offset}u + ${linear(layout, 'gid')} * ${stride}u`;
      lines.push(`  ${shadow} = ${this.load(buffer.source, address, element, buffer.id)};`);
    }
    for (const texture of this.#textures.values()) {
      const old = this.samplerUniform(texture);
      const cell = this.#columns.has(texture.id)
        ? `ivec3(ivec2(gid.xy), ${LAYER})`
        : this.#cellOf(texture, 'gid');
      lines.push(`  ${this.#textureShadow(texture)} = texelFetch(${old}, ${cell}, 0);`);
    }
    return `${CHUNK} shared ${BEGIN}\nvoid ${BEGIN}(uvec3 gid) {\n${lines.join('\n')}\n}`;
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
      ...(layout.width === undefined ? [] : [layout.width]),
      ...(layout.height === undefined ? [] : [layout.height]),
    ].join(' ');
    const gid = written.whole ? 'uvec3(0u)' : invocationOf(layout, 'element');
    return `${CHUNK} pass${index} main
${target}
layout(location = 0) out uvec4 ${RESULT};
void main() {
  ivec2 texel = ivec2(gl_FragCoord.xy);
  uvec4 old = texelFetch(${buffer.id}, texel, 0);
  uvec4 result = old;
  uint first = ${range}.x + ${offset}u;
  uint end = ${range}.x + ${range}.y / 4u;
  uint count = ${count};
  uint base = (uint(texel.y) * uint(textureSize(${buffer.id}, 0).x) + uint(texel.x)) * 4u;
  uint current = 0xFFFFFFFFu;
  for (int k = 0; k < 4; k++) {
    uint word = base + uint(k);
    if (word < first || word >= end) continue;
    uint element = (word - first) / ${stride}u;
    uint inner = word - first - element * ${stride}u;
    if (element >= count || inner >= ${size}u) continue;
    uvec3 gid = ${gid};
    if (any(greaterThanEqual(gid, ${INVOCATIONS}))) continue;
    if (element != current) {
      ${BEGIN}(gid);
      ${kernel}();
      current = element;
    }
    result[k] = ${pack}(${shadow}, inner, old[k]);
  }
  ${RESULT} = result;
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
    return `${CHUNK} pass${index} main
${target}
layout(location = 0) out ${this.#texelType(texture)} ${RESULT};
void main() {
  uvec3 gid = uvec3(uvec2(gl_FragCoord.xy), ${column ? '0u' : `uint(${LAYER})`});
  if (any(greaterThanEqual(gid, ${INVOCATIONS}))) {
    ${RESULT} = texelFetch(${old}, ${column ? `ivec3(ivec2(gid.xy), ${LAYER})` : this.#cellOf(texture, 'gid')}, 0);
    return;
  }
  ${BEGIN}(gid);
  ${kernel}();
  ${RESULT} = ${this.#textureShadow(texture)};
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
