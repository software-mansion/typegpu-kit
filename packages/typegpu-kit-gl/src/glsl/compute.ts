// oxlint-disable no-underscore-dangle -- the generator overrides WgslGenerator's protected hooks, named with one
import { NodeTypeCatalog as NODE, type Const, type Expression, type Statement } from 'tinyest';
import { d, tgpu } from 'typegpu';
import {
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

/** Why a kernel cannot run as fragment passes. The pipeline then runs on the CPU. */
export class Ineligible extends Error {
  constructor(reason: string) {
    super(`GPU compute: ${reason}`);
  }
}

/** What a kernel writes: found by a first resolution, used by the second. */
export interface ComputePlan {
  /** Each written buffer's id, with the first word of the array it writes. */
  readonly buffers: ReadonlyMap<string, number>;
  /** Each written storage texture's id. */
  readonly textures: ReadonlySet<string>;
}

const GID = 'kit_gid';
const BEGIN = 'kit_begin';
const RESULT = 'kit_result';

/** Stands in for a read of a whole written array, which would see none of the invocation's writes. */
const STALE = 'kit_stale_';

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

/** The arguments of a call's text, split at top-level commas. */
function argumentsOf(text: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (char === '(' || char === '[') depth++;
    else if (char === ')' || char === ']') depth--;
    else if (char === ',' && depth === 0) {
      parts.push(text.slice(start, i));
      start = i + 1;
    }
  }
  parts.push(text.slice(start));
  return parts;
}

interface WrittenBuffer {
  readonly buffer: BufferInfo;
  readonly offset: number;
  readonly array: d.WgslArray;
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
  readonly #shadows = new Map<string, string>();
  readonly #privates: { id: string; dataType: d.BaseData; init: string | undefined }[] = [];
  /** Names that hold `global_invocation_id` in the entry. */
  readonly #ids = new Set<string>();
  /** Locals of the entry that hold axes of `global_invocation_id`, with the axes they hold. */
  readonly #ownLocals = new Map<string, string>();
  #packers = 0;

  constructor(plan?: ComputePlan) {
    super();
    this.#plan = plan;
  }

  /** What the kernel writes, once a resolution has run. */
  get plan(): ComputePlan {
    return {
      buffers: new Map([...this.#buffers].map(([id, written]) => [id, written.offset])),
      textures: new Set(this.#textures.keys()),
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

  get #inEntry(): boolean {
    return this.ctx.topFunctionScope?.functionType === 'compute';
  }

  // ---------------------------------------------------------------------------------------------
  // The invocation's own index
  // ---------------------------------------------------------------------------------------------

  /** The axes of `global_invocation_id` a snippet provably is, in order, such as `xy`. */
  #own(snippet: Snippet): string | undefined {
    if (!this.#inEntry || snippet.dataType === UnknownData) return undefined;
    return this.#ownText(this.ctx.resolveSnippet(snippet).value.replace(/\s+/g, ''));
  }

  #ownText(text: string): string | undefined {
    const swizzle = /^(\w+)(?:\.([xyzw]+))?$/.exec(text);
    if (swizzle) {
      const [, name, components] = swizzle as unknown as [string, string, string | undefined];
      const axes = this.#ids.has(name) ? AXES : this.#ownLocals.get(name);
      if (!axes) return undefined;
      if (!components) return axes;
      const picked = [...components].map((c) => axes['xyzw'.indexOf(c)]);
      return picked.every(Boolean) ? picked.join('') : undefined;
    }
    const call = /^(uint|int|[iu]vec[234])\((.*)\)$/.exec(text);
    if (call) {
      const parts = argumentsOf(call[2] as string).map((part) => this.#ownText(part));
      if (parts.some((part) => part === undefined)) return undefined;
      const axes = parts.join('');
      return AXES.includes(axes) ? axes : undefined;
    }
    return undefined;
  }

  protected override indexIsOwn(index: Snippet): boolean {
    return this.#own(index) === 'x';
  }

  protected override _constStatement(statement: Const): ResolvedStatement {
    const result = super._constStatement(statement);
    const [, idNode, eqNode] = statement;
    if (eqNode !== undefined && this.#inEntry && this.#ids.size > 0) {
      const id = typeof idNode === 'string' ? idNode : (idNode as unknown as [number, string])[1];
      const defined = this.ctx.getById(id);
      const axes = this.#own(this._expression(eqNode));
      if (axes && typeof defined?.value === 'string') this.#ownLocals.set(defined.value, axes);
    }
    return result;
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
    if (!place) {
      const text = target.dataType === UnknownData ? '' : this.ctx.resolveSnippet(target).value;
      const root = /^\s*([A-Za-z_]\w*)/.exec(text)?.[1];
      if (root && (this.#ids.has(root) || this.#ownLocals.has(root))) {
        throw new Ineligible(`modifies '${root}', which indices are derived from`);
      }
      return;
    }
    const id = place.buffer.id;
    if (!this.#inEntry) throw new Ineligible(`a helper function writes to '${id}'`);
    if (!place.element) throw new Ineligible(`writes '${id}' outside an array`);
    if (!place.element.own) {
      throw new Ineligible(`writes an element of '${id}' that is not the invocation's own`);
    }
    const known = this.#buffers.get(id);
    if (known && known.offset !== place.element.offset) {
      throw new Ineligible(`writes two arrays of '${id}'`);
    }
    if (this.#plan && this.#plan.buffers.get(id) !== place.element.offset) {
      throw new Error(`@typegpu/kit-gl: the second resolution writes '${id}' differently`);
    }
    this.#buffers.set(id, {
      buffer: place.buffer,
      offset: place.element.offset,
      array: place.element.array,
    });
  }

  /** The variable that holds the invocation's own element of a written array. */
  #bufferShadow(buffer: BufferInfo, array: d.WgslArray): string {
    const known = this.#shadows.get(buffer.id);
    if (known) return known;
    const name = this.ctx.makeUniqueIdentifier(`kit_shadow_${buffer.id}`, 'global');
    const element = array.elementType as d.BaseData;
    this.chunk(
      'shared',
      [name],
      `${this.ctx.resolve(element).value} ${name}${this.arraySuffix(element)};`,
    );
    this.#shadows.set(buffer.id, name);
    return name;
  }

  protected override placeValue(place: Place, load: string): string {
    const offset = this.#plan?.buffers.get(place.buffer.id);
    if (offset === undefined) return load;
    const element = place.element;
    if (!element) {
      const schema = undecorate(place.schema);
      const covers = d.isWgslArray(schema) || d.isWgslStruct(schema);
      return covers && place.offset !== undefined && place.offset <= offset
        ? `${STALE}${place.buffer.id}_${this.#packers++}`
        : load;
    }
    if (element.offset !== offset) return load;
    const shadow = `${this.#bufferShadow(place.buffer, element.array)}${element.path}`;
    return element.own ? shadow : `((${element.index}) == ${GID}.x ? ${shadow} : ${load})`;
  }

  override declareGlobalVar(options: VariableDefinitionOptions): ResolvedSnippet {
    const result = super.declareGlobalVar(options);
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
    if (!this.#inEntry) throw new Ineligible(`a helper function stores to '${texture.id}'`);
    const { format } = texture.schema as unknown as { format: string };
    if (!RENDERABLE.has(format)) {
      throw new Ineligible(`stores to a '${format}' texture, which WebGL 2 cannot render to`);
    }
    const { dimension } = texture.kind;
    const coords = args[1] && this.#own(args[1]);
    const owns =
      dimension === '2d-array'
        ? coords === 'xy' && args[2] !== undefined && this.#own(args[2]) === 'z'
        : coords === { '1d': 'x', '2d': 'xy', '3d': 'xyz' }[dimension as string];
    if (!owns) {
      throw new Ineligible(`stores to a texel of '${texture.id}' that is not the invocation's own`);
    }
    if (this.#plan && !this.#plan.textures.has(texture.id)) {
      throw new Error(`@typegpu/kit-gl: the second resolution stores to '${texture.id}' anew`);
    }
    this.#textures.set(texture.id, texture);
    const value = args[dimension === '2d-array' ? 3 : 2];
    return `${this.#textureShadow(texture)} = ${this.valueOf(value)}`;
  }

  override emitCall(
    name: string,
    templateParams: readonly Snippet[],
    args: readonly Snippet[],
  ): string {
    if (name === 'textureStore') return this.#store(args);
    if (name === 'textureLoad' && args[0]) {
      const texture = this.textureOf(args[0]);
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
    if (options.functionType === 'normal') {
      const pointer = options.args.find((arg) => d.isPtr(arg.decoratedType));
      if (pointer) throw new Ineligible(`a helper function takes the pointer '${pointer.name}'`);
    }
    const code = super.functionDefinition(options);
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
    for (const arg of options.args) {
      if (attributeOf(arg.decoratedType, '@builtin')?.[0] === 'global_invocation_id') {
        this.#ids.add(arg.name);
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

    if (!this.#plan) {
      if (this.#buffers.size + this.#textures.size === 0) throw new Ineligible('writes nothing');
      if (this.#buffers.size > 0 && size[1] * size[2] !== 1) {
        throw new Ineligible('writes a buffer from a workgroup taller than 1 invocation');
      }
      return kernelChunk;
    }
    const written = [...this.#buffers.keys(), ...this.#textures.keys()].toSorted().join();
    const planned = [...this.#plan.buffers.keys(), ...this.#plan.textures].toSorted().join();
    if (written !== planned) {
      throw new Error('@typegpu/kit-gl: the second resolution writes other resources');
    }

    const grid = Math.min(
      ...[...this.#buffers.values()].map(() => 1),
      ...[...this.#textures.values()].map(
        (texture) => ({ '1d': 1, '2d': 2 })[texture.kind.dimension as string] ?? 3,
      ),
    );
    return [
      `${COMPUTE} ${size.join(' ')} ${grid}`,
      kernelChunk,
      this.#begin(),
      ...[...this.#buffers.values()].map((target, i) => this.#bufferPass(target, i, options.name)),
      ...[...this.#textures.values()].map((target, i) =>
        this.#texturePass(target, this.#buffers.size + i, options.name),
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
    for (const { buffer, offset, array } of this.#buffers.values()) {
      const element = array.elementType as d.BaseData;
      const address = `${buffer.base} + ${offset}u + gid.x * ${wordsOf(strideOf(element))}u`;
      const shadow = this.#bufferShadow(buffer, array);
      lines.push(`  ${shadow} = ${this.load(buffer.source, address, element, buffer.id)};`);
    }
    for (const texture of this.#textures.values()) {
      const old = this.samplerUniform(texture);
      lines.push(
        `  ${this.#textureShadow(texture)} = texelFetch(${old}, ${this.#cellOf(texture, 'gid')}, 0);`,
      );
    }
    return `${CHUNK} shared ${BEGIN}\nvoid ${BEGIN}(uvec3 gid) {\n${lines.join('\n')}\n}`;
  }

  #bufferPass({ buffer, offset, array }: WrittenBuffer, index: number, kernel: string): string {
    const element = array.elementType as d.BaseData;
    const stride = wordsOf(strideOf(element));
    const size = Math.ceil(d.sizeOf(element as d.AnyData) / 4);
    const range = buffer.range as string;
    const count =
      array.elementCount > 0
        ? `${array.elementCount}u`
        : `(${range}.y / 4u > ${offset}u ? (${range}.y / 4u - ${offset}u) / ${stride}u : 0u)`;
    const shadow = this.#bufferShadow(buffer, array);
    const pack = this.#packer(element);
    const target = [
      TARGET,
      'buffer',
      buffer.group,
      buffer.binding,
      buffer.id,
      offset * 4,
      stride * 4,
      d.sizeOf(element as d.AnyData),
    ].join(' ');
    return `${CHUNK} pass${index} main
${target}
layout(location = 0) out uvec4 ${RESULT};
void main() {
  ivec2 texel = ivec2(gl_FragCoord.xy);
  uvec4 old = texelFetch(${buffer.id}, texel, 0);
  uvec4 result = old;
  uint first = ${range}.x + ${offset}u;
  uint end = ${range}.x + ${range}.y / 4u;
  uint count = min(${count}, ${INVOCATIONS}.x);
  uint base = (uint(texel.y) * uint(textureSize(${buffer.id}, 0).x) + uint(texel.x)) * 4u;
  uint current = 0xFFFFFFFFu;
  for (int k = 0; k < 4; k++) {
    uint word = base + uint(k);
    if (word < first || word >= end) continue;
    uint element = (word - first) / ${stride}u;
    uint inner = word - first - element * ${stride}u;
    if (element >= count || inner >= ${size}u) continue;
    if (element != current) {
      ${BEGIN}(uvec3(element, 0u, 0u));
      ${kernel}();
      current = element;
    }
    result[k] = ${pack}(${shadow}, inner, old[k]);
  }
  ${RESULT} = result;
}`;
  }

  #texturePass(texture: TextureInfo, index: number, kernel: string): string {
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
  uvec3 gid = uvec3(uvec2(gl_FragCoord.xy), uint(${LAYER}));
  if (any(greaterThanEqual(gid, ${INVOCATIONS}))) {
    ${RESULT} = texelFetch(${old}, ${this.#cellOf(texture, 'gid')}, 0);
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

/** The reason an error gives for a kernel's GPU form failing. */
function reasonOf(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  const own = /GPU compute: (.*)$/m.exec(message);
  if (own) return own[1] as string;
  const lines = message.split('\n');
  return (lines.at(-1) ?? message).replace(/^- [^:]*: /, '');
}

/** A compute pipeline's GPU form, or why it has none. */
export function computeForm(item: object): { code: string } | { reason: string } {
  type Options = Parameters<typeof tgpu.resolveWithContext>[1];
  try {
    const analysis = new KitGlslComputeGenerator();
    tgpu.resolveWithContext([item as never], { unstable_shaderGenerator: analysis } as Options);
    const generator = new KitGlslComputeGenerator(analysis.plan);
    const { code } = tgpu.resolveWithContext([item as never], {
      unstable_shaderGenerator: generator,
    } as Options);
    return { code };
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
    const form = computeForm(item as object);
    const line =
      'code' in form
        ? `${GPU_FORM}${JSON.stringify(form.code)}`
        : `${CPU_ONLY}${JSON.stringify(form.reason)}`;
    return withValue(`${result.value}\n${line}`, result);
  };
}

export type { Element };
