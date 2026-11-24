/**
 * The render module format: what `KitGlslGenerator` writes into a shader module's code and what the
 * device reads back. The device depends only on the types here and on `parse`.
 *
 * A module is a header line and a list of chunks. Each chunk starts with
 * `//#chunk <stage> <names…>`: `vertex` and `fragment` chunks belong to one stage, `shared` chunks
 * go to each stage that reaches one of their names from its own chunks. Directive lines after the
 * chunk line describe the bindings the chunk declares.
 */

/** A buffer binding a shader reads through its data texture. */
export interface BufferBinding {
  readonly group: number;
  readonly binding: number;
  readonly access: 'uniform' | 'readonly' | 'mutable';
  /** The `usampler2D` uniform that holds the buffer's words. */
  readonly texture: string;
  /** The `uvec2` uniform: `x` is the first word of the bound range, `y` its size in bytes. */
  readonly range: string;
}

/** One GLSL sampler uniform: a texture with the sampler a shader reads it through. */
export interface TextureBinding {
  readonly group: number;
  readonly binding: number;
  /** The GLSL sampler uniform. */
  readonly uniform: string;
  /** The sampler's binding. Absent when the shader only loads texels. */
  readonly sampler?: { readonly group: number; readonly binding: number };
  readonly dimension: '1d' | '2d' | '2d-array' | 'cube' | 'cube-array' | '3d';
  readonly sampleType: 'f32' | 'i32' | 'u32' | 'depth';
  /** True for a shadow sampler, which the device binds with a comparison function. */
  readonly comparison: boolean;
  readonly multisampled: boolean;
}

/**
 * A small uniform buffer a shader reads from a std140 uniform block: one `uvec4` array that holds
 * the bound range's words, so the WGSL layout is the only layout.
 */
export interface BlockBinding {
  readonly group: number;
  readonly binding: number;
  /** The uniform block's name. */
  readonly block: string;
  /** The block's size in bytes, the buffer's size rounded up to 16. The bound range covers it. */
  readonly size: number;
}

/** An `int` uniform the device sets to the mip level count of the view bound at a binding. */
export interface LevelCountBinding {
  readonly group: number;
  readonly binding: number;
  readonly uniform: string;
}

export interface RenderModule {
  readonly kind: 'render';
  /** Complete GLSL ES 3.00 vertex source, `#version` line included. */
  readonly vertex: string;
  /** Complete GLSL ES 3.00 fragment source. Absent for a depth-only pipeline's module. */
  readonly fragment: string | undefined;
  /** Buffers read through data textures. */
  readonly buffers: readonly BufferBinding[];
  /** Uniform buffers read through uniform blocks. `parse` always sets it. */
  readonly blocks?: readonly BlockBinding[];
  readonly textures: readonly TextureBinding[];
  /** Views whose level count a shader reads with `textureNumLevels`. */
  readonly levels: readonly LevelCountBinding[];
}

/** Uniforms with fixed names that the device sets when a program declares them. */
export const FIRST_INSTANCE = 'kit_firstInstance';
export const BASE_VERTEX = 'kit_baseVertex';
/** Defined by the device at the top of both stages when `EXT_clip_control` gives a 0 to 1 depth range. */
export const CLIP_ZERO_TO_ONE = 'KIT_CLIP_ZERO_TO_ONE';

export interface ParseOptions {
  /** Lines the device adds after `#version`, such as `#define KIT_CLIP_ZERO_TO_ONE`. Default none. */
  readonly defines?: readonly string[];
}

export const HEADER = '//#kit-gl render';
export const CHUNK = '//#chunk';
export const BUFFER = '//#buffer';
export const BLOCK = '//#block';
export const TEXTURE = '//#texture';
export const LEVELS = '//#levels';

export type ChunkStage = 'shared' | 'vertex' | 'fragment';

/** The precision every stage declares, so uniforms both stages read agree. */
const PRECISION = [
  'float',
  'int',
  'sampler2D',
  'sampler3D',
  'samplerCube',
  'sampler2DShadow',
  'samplerCubeShadow',
  'sampler2DArray',
  'sampler2DArrayShadow',
  'isampler2D',
  'isampler3D',
  'isamplerCube',
  'isampler2DArray',
  'usampler2D',
  'usampler3D',
  'usamplerCube',
  'usampler2DArray',
]
  .map((type) => `precision highp ${type};`)
  .join('\n');

/** Reports whether `code` is a module `KitGlslGenerator` wrote. */
export function isRenderModule(code: string): boolean {
  return code.includes(HEADER);
}

interface Chunk {
  readonly stage: ChunkStage;
  readonly names: readonly string[];
  readonly directives: readonly string[];
  readonly code: string;
  readonly uses: ReadonlySet<string>;
}

/** Identifiers, without the member names that follow a `.`, which name no declaration. */
const IDENTIFIER = /(?<![.\w])[A-Za-z_]\w*/g;

const STRUCT = /^struct \w+ \{/;
const MEMBER_TYPE = /^\s*(\w+)\s+\w+(?:\[\d+\])?;/gm;
/** A type followed by a name it declares: a parameter, a local or a global. */
const DECLARATION = /(?<![.\w])([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*(?=[=;[,)])/g;
const NOT_TYPES = new Set(['return', 'else', 'case', 'in', 'out', 'inout', 'flat', 'centroid']);

/**
 * The global names a chunk may refer to: its identifiers, less the members of a struct it declares
 * and the parameters and locals of a function, whose names can repeat a global's.
 */
function usesOf(code: string): Set<string> {
  if (STRUCT.test(code)) {
    return new Set(Array.from(code.matchAll(MEMBER_TYPE), (match) => match[1] as string));
  }
  const uses = new Set(code.match(IDENTIFIER) ?? []);
  const body = code.indexOf('{');
  if (body >= 0) {
    for (const [, type, name] of code.matchAll(DECLARATION)) {
      if (!NOT_TYPES.has(type as string)) uses.delete(name as string);
    }
  }
  return uses;
}

function chunksOf(code: string): Chunk[] {
  const chunks: Chunk[] = [];
  let current:
    | { stage: ChunkStage; names: string[]; directives: string[]; lines: string[] }
    | undefined;

  const close = () => {
    if (!current) return;
    const text = current.lines.join('\n').trim();
    const uses = usesOf(text.replace(/\/\/.*$/gm, ''));
    chunks.push({ ...current, code: text, uses });
  };

  for (const line of code.split('\n')) {
    if (line.startsWith(`${CHUNK} `)) {
      close();
      const [stage, ...names] = line
        .slice(CHUNK.length + 1)
        .trim()
        .split(/\s+/);
      current = { stage: stage as ChunkStage, names, directives: [], lines: [] };
    } else if (line.startsWith('//#')) {
      current?.directives.push(line);
    } else if (current) {
      current.lines.push(line);
    } else if (line.trim() !== '') {
      throw new Error(`@typegpu/kit-gl: render module text outside a chunk: '${line}'`);
    }
  }
  close();
  return chunks;
}

/** The chunks a stage needs: its own and the shared chunks they reach, in module order. */
function reachable(chunks: readonly Chunk[], stage: 'vertex' | 'fragment'): Chunk[] {
  const owner = new Map<string, Chunk>();
  for (const chunk of chunks) {
    if (chunk.stage === 'shared' || chunk.stage === stage) {
      for (const name of chunk.names) owner.set(name, chunk);
    }
  }
  const included = new Set<Chunk>();
  const pending = chunks.filter((chunk) => chunk.stage === stage);
  for (const chunk of pending) included.add(chunk);
  while (pending.length > 0) {
    const chunk = pending.pop() as Chunk;
    for (const use of chunk.uses) {
      const other = owner.get(use);
      if (other && !included.has(other)) {
        included.add(other);
        pending.push(other);
      }
    }
  }
  return chunks.filter((chunk) => included.has(chunk));
}

function source(chunks: readonly Chunk[], defines: readonly string[]): string {
  return [
    '#version 300 es',
    ...defines,
    PRECISION,
    ...chunks.map((chunk) => chunk.code).filter((code) => code !== ''),
  ].join('\n\n');
}

const int = (text: string | undefined, line: string) => {
  const value = Number(text);
  if (!Number.isInteger(value)) {
    throw new Error(`@typegpu/kit-gl: malformed directive '${line}'`);
  }
  return value;
};

/** Splits a render module into its stages and reads its bindings. */
export function parse(code: string, options: ParseOptions = {}): RenderModule {
  if (!isRenderModule(code)) {
    throw new Error('@typegpu/kit-gl: not a render module');
  }
  const defines = options.defines ?? [];
  const chunks = chunksOf(code.replace(HEADER, ''));
  const vertexChunks = reachable(chunks, 'vertex');
  const hasFragment = chunks.some((chunk) => chunk.stage === 'fragment');
  const fragmentChunks = hasFragment ? reachable(chunks, 'fragment') : [];

  const buffers = new Map<string, BufferBinding>();
  const blocks = new Map<string, BlockBinding>();
  const textures = new Map<string, TextureBinding>();
  const levels = new Map<string, LevelCountBinding>();
  for (const chunk of new Set([...vertexChunks, ...fragmentChunks])) {
    for (const line of chunk.directives) {
      const [kind, ...parts] = line.trim().split(/\s+/);
      if (kind === BUFFER) {
        const [group, binding, access, texture, range] = parts;
        buffers.set(texture as string, {
          group: int(group, line),
          binding: int(binding, line),
          access: access as BufferBinding['access'],
          texture: texture as string,
          range: range as string,
        });
      } else if (kind === BLOCK) {
        const [group, binding, block, size] = parts;
        blocks.set(block as string, {
          group: int(group, line),
          binding: int(binding, line),
          block: block as string,
          size: int(size, line),
        });
      } else if (kind === TEXTURE) {
        const [group, binding, uniform, dimension, sampleType, comparison, multisampled, sg, sb] =
          parts;
        textures.set(uniform as string, {
          group: int(group, line),
          binding: int(binding, line),
          uniform: uniform as string,
          ...(sg !== undefined && { sampler: { group: int(sg, line), binding: int(sb, line) } }),
          dimension: dimension as TextureBinding['dimension'],
          sampleType: sampleType as TextureBinding['sampleType'],
          comparison: comparison === '1',
          multisampled: multisampled === '1',
        });
      } else if (kind === LEVELS) {
        const [group, binding, uniform] = parts;
        levels.set(uniform as string, {
          group: int(group, line),
          binding: int(binding, line),
          uniform: uniform as string,
        });
      }
    }
  }

  return {
    kind: 'render',
    vertex: source(vertexChunks, defines),
    fragment: hasFragment ? source(fragmentChunks, defines) : undefined,
    buffers: [...buffers.values()],
    blocks: [...blocks.values()],
    textures: [...textures.values()],
    levels: [...levels.values()],
  };
}
