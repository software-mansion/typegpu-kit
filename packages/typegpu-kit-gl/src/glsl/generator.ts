// oxlint-disable no-underscore-dangle -- the generator overrides WgslGenerator's protected hooks, named with one
import { CrossShaderStageState, GlslGenerator } from '@typegpu/gl';
import { NodeTypeCatalog as NODE, type Expression, type Return } from 'tinyest';
import { d, std } from 'typegpu';
import {
  abstractInt,
  getName,
  snip,
  UnknownData,
  withValue,
  type ConstantDefinitionOptions,
  type FunctionDefinitionOptions,
  type ResolutionCtx,
  type ResolvedSnippet,
  type ResolvedStatement,
  type Snippet,
  type VariableDefinitionOptions,
} from 'typegpu/~internal';
import {
  BASE_VERTEX,
  BLOCK,
  BUFFER,
  CHUNK,
  CLIP_ZERO_TO_ONE,
  FIRST_INSTANCE,
  HEADER,
  LEVELS,
  TEXTURE,
  type ChunkStage,
  type TextureBinding,
} from './renderModule.ts';

type AnyFn = Parameters<GlslGenerator['call']>[0];
type BufferAccess = 'uniform' | 'readonly' | 'mutable';

const BUFFER_ORIGINS: ReadonlySet<string> = new Set(['uniform', 'readonly', 'mutable']);

/** Marks a texture or sampler schema passed as an argument with the global it stands for. */
const HANDLE = Symbol('kit-gl handle');

/** Stands in for a texture or sampler argument in a call, removed once the call is built. */
const HANDLE_ARG = '\u0001kit-handle\u0001';

/** Stands in for an entry function's output variable until the output's location is known. */
const OUT = (prop: string) => `\u0002${prop}\u0002`;
// oxlint-disable-next-line no-control-regex -- the placeholder's delimiters cannot occur in GLSL
const OUT_PATTERN = /\u0002([A-Za-z0-9_$]*)\u0002/g;

/** Stands in for a load the shader cannot make, so the function that makes it fails clearly. */
const UNLOADABLE = 'kit_unloadable_';

/** Struct member names GLSL ES 3.00 reserves, renamed with a trailing underscore. */
const RESERVED_MEMBERS = new Set([
  'active',
  'asm',
  'cast',
  'class',
  'common',
  'enum',
  'extern',
  'external',
  'filter',
  'fixed',
  'goto',
  'half',
  'inline',
  'input',
  'interface',
  'long',
  'namespace',
  'noinline',
  'output',
  'partition',
  'public',
  'sample',
  'short',
  'sizeof',
  'static',
  'superp',
  'template',
  'this',
  'typedef',
  'union',
  'unsigned',
  'using',
  'volatile',
  'attribute',
  'varying',
  'resource',
  'patch',
  'subroutine',
  'precise',
  'buffer',
  'shared',
  'coherent',
  'restrict',
  'readonly',
  'writeonly',
  'lowp',
  'mediump',
  'highp',
  'precision',
  'invariant',
  'smooth',
  'flat',
  'centroid',
  'layout',
  'uniform',
  'in',
  'out',
  'inout',
  'struct',
  'texture',
]);

/** Names the kit's conventions use; user code never gets them. */
const KIT_NAMES = [
  'main',
  FIRST_INSTANCE,
  BASE_VERTEX,
  'kit_texel',
  'kit_word',
  'kit_word2',
  'kit_sampleMask',
  'kit_sampleClamped',
];

export function memberName(prop: string): string {
  return RESERVED_MEMBERS.has(prop) || prop.startsWith('gl_') ? `${prop}_` : prop;
}

const unsupported = (feature: string): Error =>
  new Error(`@typegpu/kit-gl: ${feature} is not available in WebGL 2 render stages.`);

/** Reads a buffer's words, which come four to a `uvec4`. */
interface Source {
  /** Loaders are made once per key. */
  readonly key: string;
  /** The parameter loaders for this source take, if any, and what a caller passes for it. */
  readonly parameter: string;
  readonly argument: string;
  /** The source as code inside a loader sees it. */
  inLoader(): Source;
  texel(index: string): string;
  word(address: string): string;
  word2(address: string): string;
}

interface BufferInfo {
  readonly id: string;
  /** The `uvec2` range uniform of a buffer read from a data texture. */
  readonly range?: string;
  readonly access: BufferAccess;
  readonly schema: d.BaseData;
  readonly source: Source;
  /** The first word of the bound range. */
  readonly base: string;
}

/**
 * The largest uniform buffer read from a uniform block, the size WebGL 2 guarantees. Blocks keep
 * texture units free, of which WebGL 2 guarantees 16 to a stage.
 */
const MAX_BLOCK_SIZE = 16384;

interface HandleInfo {
  readonly id: string;
  readonly group: string;
  readonly binding: number;
  readonly schema: d.BaseData;
}

/**
 * A value in a buffer. Its snippet holds the expression that reads it whole, and member and index
 * access narrow it to a smaller read instead.
 */
interface Place {
  readonly buffer: BufferInfo;
  /** A GLSL `uint` expression: the place's first word in the buffer's data texture. */
  readonly address: string;
  readonly schema: d.BaseData;
}

interface TextureKind {
  readonly dimension: TextureBinding['dimension'];
  readonly sampleType: TextureBinding['sampleType'];
  readonly multisampled: boolean;
}

function textureKind(schema: d.BaseData): TextureKind {
  const type = schema.type as string;
  if (type === 'texture_external') {
    return { dimension: '2d', sampleType: 'f32', multisampled: false };
  }
  if (type.startsWith('texture_storage_')) {
    const { format, dimension } = schema as unknown as { format: string; dimension: string };
    const sampleType = format.endsWith('uint') ? 'u32' : format.endsWith('sint') ? 'i32' : 'f32';
    return { dimension: dimension as TextureKind['dimension'], sampleType, multisampled: false };
  }
  const texture = schema as d.WgslTexture;
  return {
    dimension: texture.dimension as TextureKind['dimension'],
    sampleType: type.startsWith('texture_depth')
      ? 'depth'
      : (texture.sampleType.type as TextureKind['sampleType']),
    multisampled: texture.multisampled,
  };
}

function samplerType(kind: TextureKind, shadow: boolean): string {
  if (kind.multisampled) throw unsupported('A multisampled texture');
  const prefix = kind.sampleType === 'i32' ? 'i' : kind.sampleType === 'u32' ? 'u' : '';
  const dimension = {
    '1d': '2D',
    '2d': '2D',
    '2d-array': '2DArray',
    cube: 'Cube',
    '3d': '3D',
  }[kind.dimension as string];
  if (!dimension) throw unsupported(`A '${kind.dimension}' texture`);
  if (shadow && kind.dimension === '3d') throw unsupported('A 3D comparison texture');
  return `${prefix}sampler${dimension}${shadow ? 'Shadow' : ''}`;
}

const isHandleType = (data: unknown): boolean => {
  const type = (data as { type?: unknown } | undefined)?.type;
  return typeof type === 'string' && (type.startsWith('texture_') || type.startsWith('sampler'));
};

const roundUp = (value: number, to: number) => Math.ceil(value / to) * to;

function undecorate(data: d.BaseData): d.BaseData {
  return d.isDecorated(data) ? (data.inner as d.BaseData) : data;
}

function attributeOf(data: d.BaseData, kind: string): unknown[] | undefined {
  if (!d.isDecorated(data)) return undefined;
  const attribs = data.attribs as { type: string; params: unknown[] }[];
  return attribs.find((attrib) => attrib.type === kind)?.params;
}

/** Each member of a struct with its byte offset, as WGSL lays it out. */
function membersOf(struct: d.WgslStruct): { name: string; type: d.BaseData; offset: number }[] {
  let offset = 0;
  return Object.entries(struct.propTypes as Record<string, d.BaseData>).map(([name, type]) => {
    offset = roundUp(offset, d.alignmentOf(type as d.AnyData));
    const member = { name, type: undecorate(type), offset };
    offset += d.sizeOf(type as d.AnyData);
    return member;
  });
}

const strideOf = (element: d.BaseData) =>
  roundUp(d.sizeOf(element as d.AnyData), d.alignmentOf(element as d.AnyData));

function wordsOf(bytes: number): number {
  if (bytes % 4 !== 0) throw unsupported('A buffer member that is not 4-byte aligned');
  return bytes / 4;
}

/** Reinterprets raw words as `f`, `i` or `u` components. */
const cast = (raw: string, scalar: string, size: number) =>
  scalar === 'f'
    ? `uintBitsToFloat(${raw})`
    : scalar === 'i'
      ? `${size === 1 ? 'int' : `ivec${size}`}(${raw})`
      : raw;

const plus = (address: string, words: number) => (words === 0 ? address : `${address} + ${words}u`);

/** `common.fullScreenTriangle`, the one WGSL string function TypeGPU Kit draws with. */
const FULL_SCREEN_TRIANGLE =
  /^@vertex fn (\w+)\(@builtin\(vertex_index\) (\w+): uint\) -> (\w+) \{([\s\S]*)\}$/;

const squeeze = (text: string) => text.replace(/\s+/g, '');

const fullScreenTriangleBody = (index: string, output: string) =>
  squeeze(`
  const pos = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
  const uv = array<vec2f, 3>(vec2f(0, 1), vec2f(2, 1), vec2f(0, -1));
  return ${output}(vec4f(pos[${index}], 0, 1), uv[${index}]);`);

/** Ends every vertex stage: WebGPU's clip space in GL's, with rows running top to bottom. */
function vertexEpilogue(linear: readonly string[]): string[] {
  return [
    ...linear.map((name) => `  ${name} *= gl_Position.w;`),
    '  gl_Position.y = -gl_Position.y;',
    `#ifndef ${CLIP_ZERO_TO_ONE}`,
    '  gl_Position.z = 2.0 * gl_Position.z - gl_Position.w;',
    '#endif',
  ];
}

interface Entry {
  readonly stage: 'vertex' | 'fragment';
}

/**
 * Emits a render module: both stages of a pipeline as GLSL ES 3.00, with the directives
 * `renderModule.ts` reads.
 */
export class KitGlslGenerator extends GlslGenerator {
  /** Each place by the expression that reads it, which its snippet holds. */
  readonly #places = new Map<string, Place>();
  readonly #textures = new Map<string, HandleInfo>();
  readonly #samplers = new Map<string, HandleInfo>();
  readonly #combined = new Map<string, string>();
  readonly #levels = new Map<string, string>();
  readonly #loaders = new Map<string, Map<d.BaseData, string>>();
  readonly #helpers = new Set<string>();
  readonly #variants = new Map<unknown, Map<string, unknown[]>>();
  readonly #tags = new Map<d.BaseData, Map<string, d.BaseData>>();
  readonly #structs = new Map<string, d.WgslStruct>();
  #entry: Entry | undefined;
  #temporaries = 0;

  constructor() {
    super('neutral', new CrossShaderStageState());
  }

  override initGenerator(ctx: ResolutionCtx): void {
    super.initGenerator(ctx);
    for (const name of KIT_NAMES) ctx.reserveIdentifier(name, 'global');
    ctx.addDeclaration(HEADER);
    // TypeGPU adds functions written as WGSL strings without calling the generator
    const add = ctx.addDeclaration.bind(ctx);
    ctx.addDeclaration = (code: string, name?: string) =>
      add(code.startsWith('//#') ? code : this.#foreign(code, name), name);
  }

  /** GLSL for a declaration TypeGPU wrote in WGSL itself. */
  #foreign(code: string, name: string | undefined): string {
    const triangle = FULL_SCREEN_TRIANGLE.exec(code.trim());
    if (triangle) {
      const [, , index, output, body] = triangle as unknown as string[];
      const expected = fullScreenTriangleBody(index as string, output as string);
      if (squeeze(body as string) === expected) {
        const struct = this.#structs.get(output as string);
        const uv = struct && (struct.propTypes as Record<string, d.BaseData>).uv;
        const location = (uv && (attributeOf(uv, '@location')?.[0] as number)) ?? 0;
        const varying = `kit_vary${location}`;
        return [
          `${CHUNK} vertex`,
          `out vec2 ${varying};`,
          `uniform int ${BASE_VERTEX};`,
          '',
          `${CHUNK} vertex main`,
          'void main() {',
          '  vec2 pos[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));',
          '  vec2 uv[3] = vec2[3](vec2(0.0, 1.0), vec2(2.0, 1.0), vec2(0.0, -1.0));',
          `  int i = gl_VertexID + ${BASE_VERTEX};`,
          '  gl_Position = vec4(pos[i], 0.0, 1.0);',
          `  ${varying} = uv[i];`,
          ...vertexEpilogue([]),
          '}',
        ].join('\n');
      }
    }
    throw new Error(
      `@typegpu/kit-gl: '${name ?? code.slice(0, 40)}' is written in WGSL, which the GLSL generator cannot translate.`,
    );
  }

  #chunk(stage: ChunkStage, names: readonly string[], code: string, directives: string[] = []) {
    this.ctx.addDeclaration(
      [`${CHUNK} ${stage} ${names.join(' ')}`.trimEnd(), ...directives, code].join('\n'),
    );
  }

  #helper(name: string, code: string) {
    if (this.#helpers.has(name)) return;
    this.#helpers.add(name);
    this.#chunk('shared', [name], code);
  }

  protected override _declareHelper(name: string, code: string): void {
    this.#helper(name, code);
  }

  // ---------------------------------------------------------------------------------------------
  // Types and declarations
  // ---------------------------------------------------------------------------------------------

  #suffix(data: d.BaseData): string {
    const inner = undecorate(data);
    if (!d.isWgslArray(inner)) return '';
    if (d.isWgslArray(undecorate(inner.elementType as d.BaseData))) {
      throw unsupported('An array of arrays');
    }
    if (inner.elementCount === 0) throw unsupported('A runtime-sized array outside a buffer');
    return `[${inner.elementCount}]`;
  }

  override emitTypeAnnotation(data: d.BaseData): string {
    if (d.isDecorated(data)) return this.ctx.resolve(data.inner as d.BaseData).value;
    if (d.isAtomic(data)) return this.ctx.resolve(data.inner as d.BaseData).value;
    if (isHandleType(data)) {
      throw new Error(`@typegpu/kit-gl: '${data.type}' has no GLSL type of its own.`);
    }
    if (d.isWgslStruct(data)) {
      const id = this.ctx.makeUniqueIdentifier(getName(data), 'global');
      const members = Object.entries(data.propTypes as Record<string, d.BaseData>).map(
        ([prop, type]) =>
          `  ${this.ctx.resolve(undecorate(type)).value} ${memberName(prop)}${this.#suffix(type)};`,
      );
      this.#chunk('shared', [id], `struct ${id} {\n${members.join('\n')}\n};`);
      this.#structs.set(id, data);
      return id;
    }
    if (d.isWgslArray(data)) this.#suffix(data);
    return super.emitTypeAnnotation(data);
  }

  override declareGlobalConst(options: ConstantDefinitionOptions): ResolvedSnippet {
    const init = this.ctx.resolveSnippet(options.init).value;
    const type = this.ctx.resolve(options.dataType).value;
    this.#chunk(
      'shared',
      [options.id],
      `const ${type} ${options.id}${this.#suffix(options.dataType)} = ${init};`,
    );
    return snip(options.id, options.dataType, 'constant-immutable-def');
  }

  override declareGlobalVar(options: VariableDefinitionOptions): ResolvedSnippet {
    const { scope, id, dataType } = options;
    if (scope === 'handle') {
      const info = {
        id,
        group: String(options.group),
        binding: options.binding ?? 0,
        schema: dataType,
      };
      const type = dataType.type as string;
      (type.startsWith('sampler') ? this.#samplers : this.#textures).set(id, info);
      return snip(id, dataType, 'handle');
    }
    if (scope === 'uniform' || scope === 'readonly' || scope === 'mutable') {
      return this.#declareBuffer(options, scope);
    }
    if (scope === 'private') {
      const init = options.init ? ` = ${this.ctx.resolveSnippet(options.init).value}` : '';
      const type = this.ctx.resolve(dataType).value;
      this.#chunk('shared', [id], `${type} ${id}${this.#suffix(dataType)}${init};`);
      return snip(id, dataType, 'private');
    }
    throw unsupported(`A '${scope}' variable`);
  }

  // ---------------------------------------------------------------------------------------------
  // Buffers, read from data textures
  // ---------------------------------------------------------------------------------------------

  #declareBuffer(options: VariableDefinitionOptions, access: BufferAccess): ResolvedSnippet {
    const { id, dataType } = options;
    const size = d.sizeOf(dataType as d.AnyData);
    let buffer: BufferInfo;
    if (access === 'uniform' && size > 0 && size <= MAX_BLOCK_SIZE) {
      const block = this.ctx.makeUniqueIdentifier(`${id}_block`, 'global');
      const vectors = Math.ceil(size / 16);
      this.#chunk(
        'shared',
        [block, id],
        `layout(std140) uniform ${block} {\n  uvec4 ${id}[${vectors}];\n};`,
        [`${BLOCK} ${options.group} ${options.binding} ${block} ${vectors * 16}`],
      );
      buffer = { id, access, schema: dataType, source: this.#blockSource(id), base: '0u' };
    } else {
      const range = this.ctx.makeUniqueIdentifier(`${id}_range`, 'global');
      this.#chunk(
        'shared',
        [id, range],
        `uniform highp usampler2D ${id};\nuniform highp uvec2 ${range};`,
        [`${BUFFER} ${options.group} ${options.binding} ${access} ${id} ${range}`],
      );
      buffer = {
        id,
        range,
        access,
        schema: dataType,
        source: this.#textureSource(id),
        base: `${range}.x`,
      };
    }
    const root = this.#placeSnippet({ buffer, address: buffer.base, schema: dataType });
    return snip(root.value as string, dataType, access) as ResolvedSnippet;
  }

  /** Words from a data texture: `texture` is the global, or a loader's `t` parameter. */
  #textureSource(texture: string): Source {
    this.#helper(
      'kit_texel',
      `uvec4 kit_texel(highp usampler2D t, uint i) {
  uint width = uint(textureSize(t, 0).x);
  return texelFetch(t, ivec2(int(i % width), int(i / width)), 0);
}`,
    );
    this.#helper(
      'kit_word',
      `uint kit_word(highp usampler2D t, uint a) {
  uvec4 texel = kit_texel(t, a >> 2u);
  return texel[a & 3u];
}`,
    );
    this.#helper(
      'kit_word2',
      `uvec2 kit_word2(highp usampler2D t, uint a) {
  uvec4 texel = kit_texel(t, a >> 2u);
  return (a & 2u) == 0u ? texel.xy : texel.zw;
}`,
    );
    return {
      key: 'texture',
      parameter: 'highp usampler2D t',
      argument: texture,
      inLoader: () => this.#textureSource('t'),
      texel: (i) => `kit_texel(${texture}, ${i})`,
      word: (a) => `kit_word(${texture}, ${a})`,
      word2: (a) => `kit_word2(${texture}, ${a})`,
    };
  }

  /** Words from the `uvec4` array of a uniform block, which loaders read by name. */
  #blockSource(words: string): Source {
    this.#helper('kit_pick', 'uint kit_pick(uvec4 v, uint a) {\n  return v[a & 3u];\n}');
    this.#helper(
      'kit_pick2',
      'uvec2 kit_pick2(uvec4 v, uint a) {\n  return (a & 2u) == 0u ? v.xy : v.zw;\n}',
    );
    const source: Source = {
      key: words,
      parameter: '',
      argument: '',
      inLoader: () => source,
      texel: (i) => `${words}[${i}]`,
      word: (a) => `kit_pick(${words}[(${a}) >> 2u], ${a})`,
      word2: (a) => `kit_pick2(${words}[(${a}) >> 2u], ${a})`,
    };
    return source;
  }

  /** A GLSL expression that reads `schema` from `source` at word `address`. */
  #load(source: Source, address: string, schema: d.BaseData, owner: string): string {
    const data = undecorate(schema);
    const type = data.type as string;
    if (d.isAtomic(data)) return this.#load(source, address, data.inner as d.BaseData, owner);
    if (d.isWgslArray(data) && data.elementCount === 0) {
      return `${UNLOADABLE}${owner}_${this.#temporaries++}`;
    }
    if (type === 'f16' || type.endsWith('h')) throw unsupported('f16 in a buffer');
    if (type === 'bool' || (type.startsWith('vec') && type.endsWith('b'))) {
      throw unsupported('A bool in a buffer');
    }
    if (type === 'f32' || type === 'i32' || type === 'u32') {
      return cast(source.word(address), type[0] as string, 1);
    }
    const vec = /^vec([234])([fiu])$/.exec(type);
    if (vec) {
      const size = Number(vec[1]);
      const raw =
        size === 2
          ? source.word2(address)
          : `${source.texel(`(${address}) >> 2u`)}${size === 3 ? '.xyz' : ''}`;
      return cast(raw, vec[2] as string, size);
    }
    const loader = this.#loader(data, source);
    return `${loader}(${[source.argument, address].filter(Boolean).join(', ')})`;
  }

  /** The name of a function that reads a matrix, struct or fixed-size array from a source. */
  #loader(data: d.BaseData, outer: Source): string {
    let byType = this.#loaders.get(outer.key);
    if (!byType) {
      byType = new Map();
      this.#loaders.set(outer.key, byType);
    }
    const known = byType.get(data);
    if (known) return known;

    const source = outer.inLoader();
    const type = this.ctx.resolve(data).value;
    const mat = /^mat([234])x([234])f$/.exec(data.type as string);
    let suffix = '';
    let base: string;
    let body: string;
    if (mat) {
      const columns = Number(mat[1]);
      const rows = Number(mat[2]);
      const stride = rows === 2 ? 2 : 4;
      const column = (i: number) =>
        this.#load(source, plus('a', i * stride), d[`vec${rows}f` as 'vec4f'], '');
      base = `kit_load_${data.type}`;
      body = `  return ${type}(${Array.from({ length: columns }, (_, i) => column(i)).join(', ')});`;
    } else if (d.isWgslStruct(data)) {
      const members = membersOf(data).map((member) =>
        this.#load(source, plus('a', wordsOf(member.offset)), member.type, ''),
      );
      base = `kit_load_${type}`;
      body = `  return ${type}(${members.join(', ')});`;
    } else if (d.isWgslArray(data)) {
      const element = data.elementType as d.BaseData;
      const stride = wordsOf(strideOf(element));
      const load = this.#load(source, `a + i * ${stride}u`, element, '');
      suffix = this.#suffix(data);
      base = 'kit_load_array';
      body = `  ${type} r${suffix};
  for (uint i = 0u; i < ${data.elementCount}u; i++) {
    r[i] = ${load};
  }
  return r;`;
    } else {
      throw unsupported(`Reading '${data.type}' from a buffer`);
    }
    const name = this.ctx.makeUniqueIdentifier(
      source.parameter ? base : `${base}_${source.key}`,
      'global',
    );
    const parameters = [source.parameter, 'uint a'].filter(Boolean).join(', ');
    this.#chunk('shared', [name], `${type}${suffix} ${name}(${parameters}) {\n${body}\n}`);
    byType.set(data, name);
    return name;
  }

  #placeSnippet(place: Place, sideEffects = false): Snippet {
    const load = this.#load(place.buffer.source, place.address, place.schema, place.buffer.id);
    this.#places.set(load, place);
    return snip(load, undecorate(place.schema), place.buffer.access, sideEffects);
  }

  /** The buffer place a snippet stands for, when it reads a buffer the generator declared. */
  #placeOf(snippet: Snippet): Place | undefined {
    const value = snippet.value;
    if (typeof value === 'string') return this.#places.get(value);
    if (!BUFFER_ORIGINS.has(snippet.origin) || snippet.dataType === UnknownData) return undefined;
    if (typeof value !== 'object' || value === null) return undefined;
    return this.#places.get(this.ctx.resolveSnippet(snippet).value);
  }

  #uint(index: Snippet): string {
    const value = this.ctx.resolveSnippet(index).value;
    if (index.dataType !== UnknownData && index.dataType.type === 'u32') return value;
    if (typeof index.value === 'number') return `${index.value}u`;
    return `uint(${value})`;
  }

  #arrayLength(place: Place): Snippet {
    const array = undecorate(place.schema) as d.WgslArray;
    if (array.elementCount > 0) return snip(array.elementCount, abstractInt, 'constant', false);
    const stride = strideOf(array.elementType as d.BaseData);
    const range = place.buffer.range as string;
    return snip(
      `((${range}.y - 4u * ((${place.address}) - ${range}.x)) / ${stride}u)`,
      d.u32,
      'runtime',
      false,
    );
  }

  /** Narrows a place by a member or index access, or returns undefined to read it first. */
  #narrow(place: Place, expression: Expression): Snippet | undefined {
    const schema = undecorate(place.schema);
    const node = expression as [number, Expression, Expression];
    if (node[0] === NODE.memberAccess) {
      const prop = node[2] as unknown as string;
      if (d.isWgslStruct(schema)) {
        const member = membersOf(schema).find((each) => each.name === prop);
        if (!member) return undefined;
        const address = plus(place.address, wordsOf(member.offset));
        return this.#placeSnippet({ buffer: place.buffer, address, schema: member.type });
      }
      if (d.isWgslArray(schema) && prop === 'length') return this.#arrayLength(place);
      return undefined;
    }
    if (d.isWgslArray(schema)) {
      const index = this._expression(node[2]);
      const stride = wordsOf(strideOf(schema.elementType as d.BaseData));
      const offset = stride === 1 ? this.#uint(index) : `${this.#uint(index)} * ${stride}u`;
      const address = `${place.address} + ${offset}`;
      const element = schema.elementType as d.BaseData;
      return this.#placeSnippet(
        { buffer: place.buffer, address, schema: element },
        index.possibleSideEffects,
      );
    }
    return undefined;
  }

  /** Lets the base generator continue from an already generated snippet. */
  #bind(snippet: Snippet): string {
    const id = `$kit${this.#temporaries++}`;
    this.ctx.setBlockExternals({ [id]: snippet });
    return id;
  }

  protected override _expression(expression: Expression): Snippet {
    if (
      !Array.isArray(expression) ||
      (expression[0] !== NODE.memberAccess && expression[0] !== NODE.indexAccess)
    ) {
      return super._expression(expression);
    }
    const [kind, targetNode, propertyNode] = expression as [number, Expression, Expression];
    let target = this._expression(targetNode);
    const place = this.#placeOf(target);
    if (place) {
      const narrowed = this.#narrow(place, expression);
      if (narrowed) return narrowed;
    }
    if (
      kind === NODE.memberAccess &&
      target.dataType !== UnknownData &&
      d.isWgslStruct(target.dataType)
    ) {
      const prop = propertyNode as unknown as string;
      const type = (target.dataType.propTypes as Record<string, d.BaseData>)[prop];
      if (type) {
        return snip(
          `${this.ctx.resolveSnippet(target).value}.${memberName(prop)}`,
          undecorate(type),
          target.origin,
          target.possibleSideEffects,
        );
      }
    }
    return super._expression([kind, this.#bind(target), propertyNode] as unknown as Expression);
  }

  protected override _aliasConstStatement(
    rawId: string,
    eqNode: Expression,
    eq: Snippet,
  ): ResolvedStatement {
    const place = this.#placeOf(eq);
    if (!place) return super._aliasConstStatement(rawId, eqNode, eq);
    const name = this.ctx.makeUniqueIdentifier(`${rawId}_at`, 'block');
    const alias = this.#placeSnippet({ buffer: place.buffer, address: name, schema: place.schema });
    this.ctx.defineVariable(rawId, snip(alias.value, alias.dataType, eq.origin, false));
    return { code: `${this.ctx.pre}uint ${name} = ${place.address};`, definesInNearestScope: true };
  }

  override call(callee: AnyFn, args: readonly Snippet[], calleeSnippet?: Snippet): Snippet {
    if ((callee as unknown) === std.atomicLoad || (callee as unknown) === std.arrayLength) {
      const place = args[0] && this.#placeOf(args[0]);
      if (!place) throw unsupported(`'${getName(callee)}' outside a buffer`);
      if ((callee as unknown) === std.arrayLength) return this.#arrayLength(place);
      const inner = (undecorate(place.schema) as d.Atomic).inner as d.BaseData;
      return snip(
        this.#load(place.buffer.source, place.address, inner, place.buffer.id),
        inner,
        'runtime',
        false,
      );
    }
    const name = getName(callee);
    if (name?.startsWith('atomic') || name === 'workgroupBarrier' || name === 'storageBarrier') {
      throw unsupported(`'${name}'`);
    }
    return super.call(callee, args, calleeSnippet as Snippet);
  }

  // ---------------------------------------------------------------------------------------------
  // Textures and samplers
  // ---------------------------------------------------------------------------------------------

  #tagged(schema: d.BaseData, id: string): d.BaseData {
    let byId = this.#tags.get(schema);
    if (!byId) {
      byId = new Map();
      this.#tags.set(schema, byId);
    }
    let tagged = byId.get(id);
    if (!tagged) {
      tagged = Object.create(schema, { [HANDLE]: { value: id } }) as d.BaseData;
      byId.set(id, tagged);
    }
    return tagged;
  }

  #handleId(snippet: Snippet): string {
    const tag = (snippet.dataType as unknown as Record<symbol, string | undefined>)[HANDLE];
    return tag ?? this.ctx.resolveSnippet(snippet).value;
  }

  /**
   * Calls a function that takes textures or samplers with a copy of it for those exact globals, as
   * GLSL samples a texture and a sampler through one uniform.
   */
  protected override _callShellless(
    callee: AnyFn,
    args: readonly Snippet[],
  ): ResolvedSnippet | undefined {
    if (!args.some((arg) => isHandleType(arg.dataType))) return super._callShellless(callee, args);
    const ids = args.map((arg) => (isHandleType(arg.dataType) ? this.#handleId(arg) : ''));
    const key = ids.join('|');
    const tagged = args.map((arg, i) =>
      isHandleType(arg.dataType)
        ? snip(
            HANDLE_ARG,
            this.#tagged(arg.dataType as d.BaseData, ids[i] as string),
            arg.origin,
            false,
          )
        : arg,
    );

    // The repository keeps one variant per argument types, and handles of one type look alike, so
    // the lookup for this call sees a list of variants for these globals only.
    const repo = this.ctx.shelllessRepo as unknown as { cache: Map<unknown, unknown[]> };
    const real = repo.cache;
    const variants = this.#variants;
    repo.cache = {
      get(fn: unknown) {
        repo.cache = real;
        let byKey = variants.get(fn);
        if (!byKey) {
          byKey = new Map();
          variants.set(fn, byKey);
        }
        let list = byKey.get(key);
        if (!list) {
          list = [];
          byKey.set(key, list);
        }
        return list;
      },
    } as unknown as Map<unknown, unknown[]>;
    let result: ResolvedSnippet | undefined;
    try {
      result = super._callShellless(callee, tagged);
    } finally {
      repo.cache = real;
    }
    if (!result) return undefined;
    const value = (result.value as string)
      .replaceAll(`${HANDLE_ARG}, `, '')
      .replaceAll(`, ${HANDLE_ARG}`, '')
      .replaceAll(HANDLE_ARG, '');
    return withValue(value, result);
  }

  #texture(snippet: Snippet | undefined): HandleInfo & { kind: TextureKind } {
    if (!snippet) throw new Error('@typegpu/kit-gl: a texture call without a texture');
    const id = this.#handleId(snippet);
    const info = this.#textures.get(id);
    if (!info) throw new Error(`@typegpu/kit-gl: '${id}' is not a texture this shader declared`);
    return { ...info, kind: textureKind(info.schema) };
  }

  /** The GLSL sampler uniform that reads `texture` through `sampler`, declared on first use. */
  #uniform(texture: HandleInfo & { kind: TextureKind }, samplerSnippet?: Snippet): string {
    let sampler: HandleInfo | undefined;
    if (samplerSnippet) {
      const id = this.#handleId(samplerSnippet);
      sampler = this.#samplers.get(id);
      if (!sampler)
        throw new Error(`@typegpu/kit-gl: '${id}' is not a sampler this shader declared`);
    }
    const shadow = sampler?.schema.type === 'sampler_comparison';
    const key = `${texture.id}|${sampler?.id ?? ''}`;
    const known = this.#combined.get(key);
    if (known) return known;
    const name = this.ctx.makeUniqueIdentifier(
      sampler ? `${texture.id}_${sampler.id}` : texture.id,
      'global',
    );
    const { kind } = texture;
    const directive = [
      TEXTURE,
      texture.group,
      texture.binding,
      name,
      kind.dimension,
      kind.sampleType,
      shadow ? 1 : 0,
      kind.multisampled ? 1 : 0,
      ...(sampler ? [sampler.group, sampler.binding] : []),
    ].join(' ');
    this.#chunk('shared', [name], `uniform ${samplerType(kind, shadow)} ${name};`, [directive]);
    this.#combined.set(key, name);
    return name;
  }

  /** Any uniform of `texture`, for size queries, which work through every sampler type. */
  #anyUniform(texture: HandleInfo & { kind: TextureKind }): string {
    for (const [key, name] of this.#combined) {
      if (key.startsWith(`${texture.id}|`)) return name;
    }
    return this.#uniform(texture);
  }

  #value(snippet: Snippet | undefined): string {
    if (!snippet) throw new Error('@typegpu/kit-gl: a texture call is missing an argument');
    return this.ctx.resolveSnippet(snippet).value;
  }

  #float(snippet: Snippet | undefined): string {
    const value = this.#value(snippet);
    const type = (snippet as Snippet).dataType;
    return type !== UnknownData && type.type === 'f32' ? value : `float(${value})`;
  }

  #int(snippet: Snippet | undefined): string {
    const value = this.#value(snippet);
    const type = (snippet as Snippet).dataType;
    return type !== UnknownData && type.type === 'i32' ? value : `int(${value})`;
  }

  #ivec(snippet: Snippet | undefined, size: number): string {
    const value = this.#value(snippet);
    const type = (snippet as Snippet).dataType;
    if (size === 1) return this.#int(snippet);
    return type !== UnknownData && type.type === `vec${size}i` ? value : `ivec${size}(${value})`;
  }

  #textureCall(name: string, args: readonly Snippet[]): string | undefined {
    if (!name.startsWith('texture')) return undefined;
    if (name === 'textureStore') throw unsupported("'textureStore'");
    if (name.startsWith('textureGather')) throw unsupported(`'${name}'`);
    if (name === 'textureNumSamples') throw unsupported("'textureNumSamples'");

    const texture = this.#texture(args[0]);
    const { kind } = texture;
    const arrayed = kind.dimension === '2d-array' || kind.dimension === 'cube-array';
    const depth = kind.sampleType === 'depth';

    if (name === 'textureDimensions') {
      const level = args[1] ? this.#int(args[1]) : '0';
      const size = `textureSize(${this.#anyUniform(texture)}, ${level})`;
      if (kind.dimension === '1d') return `uint(${size}.x)`;
      if (kind.dimension === '3d') return `uvec3(${size})`;
      return arrayed ? `uvec2(${size}.xy)` : `uvec2(${size})`;
    }
    if (name === 'textureNumLayers') return `uint(textureSize(${this.#anyUniform(texture)}, 0).z)`;
    if (name === 'textureNumLevels') {
      let uniform = this.#levels.get(texture.id);
      if (!uniform) {
        uniform = this.ctx.makeUniqueIdentifier(`${texture.id}_levels`, 'global');
        this.#chunk('shared', [uniform], `uniform int ${uniform};`, [
          `${LEVELS} ${texture.group} ${texture.binding} ${uniform}`,
        ]);
        this.#levels.set(texture.id, uniform);
      }
      return `uint(${uniform})`;
    }
    if (name === 'textureLoad') {
      const uniform = this.#uniform(texture);
      const storage = (texture.schema.type as string).startsWith('texture_storage');
      const external = texture.schema.type === 'texture_external';
      let coords: string;
      if (kind.dimension === '1d') coords = `ivec2(${this.#int(args[1])}, 0)`;
      else if (kind.dimension === '3d') coords = this.#ivec(args[1], 3);
      else if (arrayed) coords = `ivec3(${this.#ivec(args[1], 2)}, ${this.#int(args[2])})`;
      else coords = this.#ivec(args[1], 2);
      const levelArg = storage || external ? undefined : args[arrayed ? 3 : 2];
      if (kind.multisampled) throw unsupported('Loading from a multisampled texture');
      const level = levelArg ? this.#int(levelArg) : '0';
      return `texelFetch(${uniform}, ${coords}, ${level})${depth ? '.r' : ''}`;
    }

    const sampler = args[1];
    const shadow = name.startsWith('textureSampleCompare');
    const uniform = this.#uniform(texture, sampler);
    const coordsValue = this.#value(args[2]);
    let rest = args.slice(3);
    let layer: string | undefined;
    if (arrayed) {
      layer = this.#float(rest[0]);
      rest = rest.slice(1);
    }
    const coords =
      kind.dimension === '1d'
        ? `vec2(${coordsValue}, 0.5)`
        : layer !== undefined
          ? `vec3(${coordsValue}, ${layer})`
          : coordsValue;

    if (name === 'textureSampleBaseClampToEdge') {
      this.#helper(
        'kit_sampleClamped',
        `vec4 kit_sampleClamped(highp sampler2D t, vec2 c) {
  vec2 h = 0.5 / vec2(textureSize(t, 0));
  return textureLod(t, clamp(c, h, 1.0 - h), 0.0);
}`,
      );
      return `kit_sampleClamped(${uniform}, ${coordsValue})`;
    }

    if (shadow) {
      const reference = this.#float(rest[0]);
      const offset = rest[1] ? this.#value(rest[1]) : undefined;
      const point =
        kind.dimension === '2d'
          ? `vec3(${coordsValue}, ${reference})`
          : `vec4(${coords}, ${reference})`;
      if (name === 'textureSampleCompare') {
        if (offset && kind.dimension !== '2d')
          throw unsupported('A comparison offset on this texture');
        return offset
          ? `textureOffset(${uniform}, ${point}, ${offset})`
          : `texture(${uniform}, ${point})`;
      }
      // Only 2D shadow samplers have textureLod; zero gradients select level 0 on the others.
      if (kind.dimension === '2d') {
        return offset
          ? `textureLodOffset(${uniform}, ${point}, 0.0, ${offset})`
          : `textureLod(${uniform}, ${point}, 0.0)`;
      }
      const zero = kind.dimension === 'cube' ? 'vec3(0.0)' : 'vec2(0.0)';
      return offset
        ? `textureGradOffset(${uniform}, ${point}, ${zero}, ${zero}, ${offset})`
        : `textureGrad(${uniform}, ${point}, ${zero}, ${zero})`;
    }

    const channel = depth ? '.r' : '';
    if (name === 'textureSample') {
      const offset = rest[0] ? this.#value(rest[0]) : undefined;
      return offset
        ? `textureOffset(${uniform}, ${coords}, ${offset})${channel}`
        : `texture(${uniform}, ${coords})${channel}`;
    }
    if (name === 'textureSampleBias') {
      const bias = this.#float(rest[0]);
      const offset = rest[1] ? this.#value(rest[1]) : undefined;
      return offset
        ? `textureOffset(${uniform}, ${coords}, ${offset}, ${bias})${channel}`
        : `texture(${uniform}, ${coords}, ${bias})${channel}`;
    }
    if (name === 'textureSampleLevel') {
      const level = this.#float(rest[0]);
      const offset = rest[1] ? this.#value(rest[1]) : undefined;
      return offset
        ? `textureLodOffset(${uniform}, ${coords}, ${level}, ${offset})${channel}`
        : `textureLod(${uniform}, ${coords}, ${level})${channel}`;
    }
    if (name === 'textureSampleGrad') {
      const dx = this.#value(rest[0]);
      const dy = this.#value(rest[1]);
      const offset = rest[2] ? this.#value(rest[2]) : undefined;
      return offset
        ? `textureGradOffset(${uniform}, ${coords}, ${dx}, ${dy}, ${offset})${channel}`
        : `textureGrad(${uniform}, ${coords}, ${dx}, ${dy})${channel}`;
    }
    throw unsupported(`'${name}'`);
  }

  override emitCall(
    name: string,
    templateParams: readonly Snippet[],
    args: readonly Snippet[],
  ): string {
    return this.#textureCall(name, args) ?? super.emitCall(name, templateParams, args);
  }

  // ---------------------------------------------------------------------------------------------
  // Functions and stage IO
  // ---------------------------------------------------------------------------------------------

  override functionDefinition(options: FunctionDefinitionOptions): string {
    const outer = this.#entry;
    this.#entry = undefined;
    try {
      let code: string;
      if (options.functionType === 'vertex' || options.functionType === 'fragment') {
        code = this.#entryDefinition(options, options.functionType);
      } else if (options.functionType === 'compute') {
        throw new Error('@typegpu/kit-gl: compute entry functions go to the CPU generator.');
      } else {
        const args = options.args.filter((arg) => !isHandleType(arg.decoratedType));
        code = `${CHUNK} shared ${options.name}\n${super.functionDefinition({ ...options, args })}`;
      }
      const unloadable = code.indexOf(UNLOADABLE);
      if (unloadable >= 0) {
        const owner = /^\w+/.exec(code.slice(unloadable + UNLOADABLE.length))?.[0];
        throw new Error(
          `@typegpu/kit-gl: '${options.name}' reads the whole runtime-sized array in '${owner}'. Read its elements instead.`,
        );
      }
      return code;
    } finally {
      this.#entry = outer;
    }
  }

  override _return(statement: Return): string {
    const entry = this.#entry;
    const node = statement[1];
    if (!entry || node === undefined) return super._return(statement);

    const expected = this.ctx.topFunctionReturnType;
    const expr = expected
      ? this._expressionWithTypeConversion(node, expected)
      : this._expression(node);
    if (expr.dataType === UnknownData) return super._return(statement);
    this.ctx.reportReturnType(expr.dataType as d.BaseData);

    const type = undecorate(expr.dataType as d.BaseData);
    const pre = this.ctx.pre;
    const value = this.ctx.resolveSnippet(expr).value;
    if (!d.isWgslStruct(type)) {
      return `${pre}{\n${pre}  ${OUT('')} = ${value};\n${pre}  return;\n${pre}}`;
    }
    const temporary = this.ctx.makeUniqueIdentifier('kit_out', 'block');
    const lines = [`${pre}  ${this.ctx.resolve(type).value} ${temporary} = ${value};`];
    for (const prop of Object.keys(type.propTypes)) {
      lines.push(`${pre}  ${OUT(prop)} = ${temporary}.${memberName(prop)};`);
    }
    return `${pre}{\n${lines.join('\n')}\n${pre}  return;\n${pre}}`;
  }

  #entryDefinition(options: FunctionDefinitionOptions, stage: 'vertex' | 'fragment'): string {
    this.#entry = { stage };
    const body = this._block(options.body, /* allowInlining */ false).code;
    const returnType = options.determineReturnType();

    const declarations: string[] = [];
    const linear: string[] = [];
    let sampleMask = false;

    const interpolation = (type: d.BaseData, name: string) => {
      const inner = undecorate(type);
      const [mode = 'perspective', sampling = ''] = String(
        attributeOf(type, '@interpolate')?.[0] ?? '',
      )
        .split(',')
        .map((part) => part.trim());
      const integer = /^(i32|u32|vec[234][iu])$/.test(inner.type as string);
      if (integer || mode === 'flat') return 'flat ';
      if (mode === 'linear') linear.push(name);
      return sampling === 'centroid' ? 'centroid ' : '';
    };

    // Outputs
    const outputs = new Map<string, string>();
    const output = (prop: string, type: d.BaseData) => {
      const builtin = attributeOf(type, '@builtin')?.[0] as string | undefined;
      if (builtin === 'position' || (stage === 'vertex' && !builtin && prop === '')) {
        outputs.set(prop, 'gl_Position');
        return;
      }
      if (builtin === 'frag_depth') {
        outputs.set(prop, 'gl_FragDepth');
        return;
      }
      if (builtin === 'sample_mask') {
        sampleMask = true;
        outputs.set(prop, 'kit_sampleMask');
        return;
      }
      if (builtin) throw unsupported(`The '${builtin}' output`);
      const location = (attributeOf(type, '@location')?.[0] as number | undefined) ?? 0;
      const glsl = this.ctx.resolve(undecorate(type)).value;
      if (stage === 'vertex') {
        const name = `kit_vary${location}`;
        declarations.push(`${interpolation(type, name)}out ${glsl} ${name};`);
        outputs.set(prop, name);
      } else {
        const name = `kit_out${location}`;
        declarations.push(`layout(location = ${location}) out ${glsl} ${name};`);
        outputs.set(prop, name);
      }
    };
    const returned = undecorate(returnType);
    if (d.isWgslStruct(returned)) {
      for (const [prop, type] of Object.entries(returned.propTypes as Record<string, d.BaseData>)) {
        output(prop, type);
      }
    } else if (returned.type !== 'void') {
      output('', returnType);
    }
    if (sampleMask) declarations.push('uint kit_sampleMask = 0xFFFFFFFFu;');

    // Inputs
    let baseVertex = false;
    let firstInstance = false;
    const input = (type: d.BaseData): string => {
      const builtin = attributeOf(type, '@builtin')?.[0] as string | undefined;
      if (builtin) {
        if (stage === 'vertex' && builtin === 'vertex_index') {
          baseVertex = true;
          return `uint(gl_VertexID + ${BASE_VERTEX})`;
        }
        if (stage === 'vertex' && builtin === 'instance_index') {
          firstInstance = true;
          return `uint(gl_InstanceID + ${FIRST_INSTANCE})`;
        }
        if (stage === 'fragment' && builtin === 'position') return 'gl_FragCoord';
        if (stage === 'fragment' && builtin === 'front_facing') return 'gl_FrontFacing';
        throw unsupported(`The '${builtin}' input`);
      }
      const location = (attributeOf(type, '@location')?.[0] as number | undefined) ?? 0;
      const glsl = this.ctx.resolve(undecorate(type)).value;
      if (stage === 'vertex') {
        const name = `kit_attr${location}`;
        declarations.push(`layout(location = ${location}) in ${glsl} ${name};`);
        return name;
      }
      const name = `kit_vary${location}`;
      const before = linear.length;
      declarations.push(`${interpolation(type, name)}in ${glsl} ${name};`);
      return linear.length > before ? `(${name} * gl_FragCoord.w)` : name;
    };

    const prelude: string[] = [];
    for (const arg of options.args) {
      if (!arg.used) continue;
      let argType = arg.decoratedType as d.BaseData;
      if ((argType as { type?: string }).type === 'auto-struct') {
        argType = (argType as unknown as { completeStruct: d.WgslStruct }).completeStruct;
      }
      if (d.isWgslStruct(argType)) {
        const struct = this.ctx.resolve(argType).value;
        const fields = Object.values(argType.propTypes as Record<string, d.BaseData>).map(input);
        prelude.push(`  ${struct} ${arg.name} = ${struct}(${fields.join(', ')});`);
      } else {
        prelude.push(
          `  ${this.ctx.resolve(undecorate(argType)).value} ${arg.name} = ${input(argType)};`,
        );
      }
    }
    if (baseVertex) declarations.push(`uniform int ${BASE_VERTEX};`);
    if (firstInstance) declarations.push(`uniform int ${FIRST_INSTANCE};`);
    if (declarations.length > 0) this.#chunk(stage, [], declarations.join('\n'));

    const withOutputs = body.replace(OUT_PATTERN, (_, prop: string) => {
      const name = outputs.get(prop);
      if (!name)
        throw new Error(`@typegpu/kit-gl: '${options.name}' returns unknown output '${prop}'`);
      return name;
    });
    const open = withOutputs.indexOf('\n');
    const fn =
      withOutputs === ''
        ? `void ${options.name}() {}`
        : `void ${options.name}() ${withOutputs.slice(0, open + 1)}${prelude.map((line) => `${line}\n`).join('')}${withOutputs.slice(open + 1)}`;

    const epilogue =
      stage === 'vertex'
        ? vertexEpilogue(linear)
        : sampleMask
          ? ['  if ((kit_sampleMask & 1u) == 0u) discard;']
          : [];
    const main = `void main() {\n  ${options.name}();\n${epilogue.map((line) => `${line}\n`).join('')}}`;
    return `${CHUNK} ${stage} ${options.name}\n${fn}\n\n${CHUNK} ${stage} main\n${main}`;
  }
}
