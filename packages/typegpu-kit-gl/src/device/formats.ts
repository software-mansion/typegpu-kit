/** How each WebGPU texture and vertex format maps onto WebGL 2. */

import type { Extensions } from './extensions.ts';

/** The texel's kind, which decides clears, readback and filtering. */
export type TexelKind = 'float' | 'unorm' | 'snorm' | 'uint' | 'sint' | 'depth' | 'depth-stencil';

export interface FormatInfo {
  readonly format: GPUTextureFormat;
  readonly internal: number;
  /** The upload format and type of `texSubImage*` in the format's own layout. */
  readonly upload: number;
  readonly type: number;
  /** Bytes per texel in the format's own layout. */
  readonly bytes: number;
  /** Components per texel in the format's own layout, packed formats counting as one. */
  readonly components: number;
  readonly kind: TexelKind;
  /** The typed array a texel row reads as in the format's own layout. */
  readonly array: 'u8' | 'i8' | 'u16' | 'i16' | 'f16' | 'u32' | 'i32' | 'f32';
  readonly renderable: boolean;
  readonly filterable: boolean;
  readonly blendable: boolean;
  readonly hasDepth: boolean;
  readonly hasStencil: boolean;
  /** Stored as RGBA with red and blue swapped on the way in and out. */
  readonly bgra: boolean;
  readonly srgb: boolean;
}

export type FormatTable = ReadonlyMap<GPUTextureFormat, FormatInfo>;

export function formatTable(gl: WebGL2RenderingContext, ext: Extensions): FormatTable {
  const floatRender = ext.colorBufferFloat !== null;
  const halfRender = floatRender || ext.colorBufferHalfFloat !== null;
  const floatLinear = ext.textureFloatLinear !== null;
  const floatBlend = ext.floatBlend !== null;
  const table = new Map<GPUTextureFormat, FormatInfo>();

  const add = (
    format: GPUTextureFormat,
    internal: number,
    upload: number,
    type: number,
    bytes: number,
    components: number,
    kind: TexelKind,
    array: FormatInfo['array'],
    renderable: boolean,
    filterable: boolean,
    extra: Partial<Pick<FormatInfo, 'bgra' | 'srgb' | 'blendable'>> = {},
  ) => {
    const depth = kind === 'depth' || kind === 'depth-stencil';
    table.set(format, {
      format,
      internal,
      upload,
      type,
      bytes,
      components,
      kind,
      array,
      renderable,
      filterable,
      blendable: extra.blendable ?? (renderable && (kind === 'float' || kind === 'unorm')),
      hasDepth: depth && format !== 'stencil8',
      hasStencil: kind === 'depth-stencil',
      bgra: extra.bgra ?? false,
      srgb: extra.srgb ?? false,
    });
  };

  const U8 = gl.UNSIGNED_BYTE;
  const I8 = gl.BYTE;
  const U16 = gl.UNSIGNED_SHORT;
  const I16 = gl.SHORT;
  const U32 = gl.UNSIGNED_INT;
  const I32 = gl.INT;
  const F16 = gl.HALF_FLOAT;
  const F32 = gl.FLOAT;

  add('r8unorm', gl.R8, gl.RED, U8, 1, 1, 'unorm', 'u8', true, true);
  add('r8snorm', gl.R8_SNORM, gl.RED, I8, 1, 1, 'snorm', 'i8', false, true);
  add('r8uint', gl.R8UI, gl.RED_INTEGER, U8, 1, 1, 'uint', 'u8', true, false);
  add('r8sint', gl.R8I, gl.RED_INTEGER, I8, 1, 1, 'sint', 'i8', true, false);
  add('rg8unorm', gl.RG8, gl.RG, U8, 2, 2, 'unorm', 'u8', true, true);
  add('rg8snorm', gl.RG8_SNORM, gl.RG, I8, 2, 2, 'snorm', 'i8', false, true);
  add('rg8uint', gl.RG8UI, gl.RG_INTEGER, U8, 2, 2, 'uint', 'u8', true, false);
  add('rg8sint', gl.RG8I, gl.RG_INTEGER, I8, 2, 2, 'sint', 'i8', true, false);
  add('rgba8unorm', gl.RGBA8, gl.RGBA, U8, 4, 4, 'unorm', 'u8', true, true);
  add('rgba8unorm-srgb', gl.SRGB8_ALPHA8, gl.RGBA, U8, 4, 4, 'unorm', 'u8', true, true, {
    srgb: true,
  });
  add('bgra8unorm', gl.RGBA8, gl.RGBA, U8, 4, 4, 'unorm', 'u8', true, true, { bgra: true });
  add('bgra8unorm-srgb', gl.SRGB8_ALPHA8, gl.RGBA, U8, 4, 4, 'unorm', 'u8', true, true, {
    bgra: true,
    srgb: true,
  });
  add('rgba8snorm', gl.RGBA8_SNORM, gl.RGBA, I8, 4, 4, 'snorm', 'i8', false, true);
  add('rgba8uint', gl.RGBA8UI, gl.RGBA_INTEGER, U8, 4, 4, 'uint', 'u8', true, false);
  add('rgba8sint', gl.RGBA8I, gl.RGBA_INTEGER, I8, 4, 4, 'sint', 'i8', true, false);

  add('r16uint', gl.R16UI, gl.RED_INTEGER, U16, 2, 1, 'uint', 'u16', true, false);
  add('r16sint', gl.R16I, gl.RED_INTEGER, I16, 2, 1, 'sint', 'i16', true, false);
  add('r16float', gl.R16F, gl.RED, F16, 2, 1, 'float', 'f16', halfRender, true);
  add('rg16uint', gl.RG16UI, gl.RG_INTEGER, U16, 4, 2, 'uint', 'u16', true, false);
  add('rg16sint', gl.RG16I, gl.RG_INTEGER, I16, 4, 2, 'sint', 'i16', true, false);
  add('rg16float', gl.RG16F, gl.RG, F16, 4, 2, 'float', 'f16', halfRender, true);
  add('rgba16uint', gl.RGBA16UI, gl.RGBA_INTEGER, U16, 8, 4, 'uint', 'u16', true, false);
  add('rgba16sint', gl.RGBA16I, gl.RGBA_INTEGER, I16, 8, 4, 'sint', 'i16', true, false);
  add('rgba16float', gl.RGBA16F, gl.RGBA, F16, 8, 4, 'float', 'f16', halfRender, true);

  add('r32uint', gl.R32UI, gl.RED_INTEGER, U32, 4, 1, 'uint', 'u32', true, false);
  add('r32sint', gl.R32I, gl.RED_INTEGER, I32, 4, 1, 'sint', 'i32', true, false);
  add('r32float', gl.R32F, gl.RED, F32, 4, 1, 'float', 'f32', floatRender, floatLinear, {
    blendable: floatRender && floatBlend,
  });
  add('rg32uint', gl.RG32UI, gl.RG_INTEGER, U32, 8, 2, 'uint', 'u32', true, false);
  add('rg32sint', gl.RG32I, gl.RG_INTEGER, I32, 8, 2, 'sint', 'i32', true, false);
  add('rg32float', gl.RG32F, gl.RG, F32, 8, 2, 'float', 'f32', floatRender, floatLinear, {
    blendable: floatRender && floatBlend,
  });
  add('rgba32uint', gl.RGBA32UI, gl.RGBA_INTEGER, U32, 16, 4, 'uint', 'u32', true, false);
  add('rgba32sint', gl.RGBA32I, gl.RGBA_INTEGER, I32, 16, 4, 'sint', 'i32', true, false);
  add('rgba32float', gl.RGBA32F, gl.RGBA, F32, 16, 4, 'float', 'f32', floatRender, floatLinear, {
    blendable: floatRender && floatBlend,
  });

  add(
    'rgb10a2unorm',
    gl.RGB10_A2,
    gl.RGBA,
    gl.UNSIGNED_INT_2_10_10_10_REV,
    4,
    1,
    'unorm',
    'u32',
    true,
    true,
  );
  add(
    'rgb10a2uint',
    gl.RGB10_A2UI,
    gl.RGBA_INTEGER,
    gl.UNSIGNED_INT_2_10_10_10_REV,
    4,
    1,
    'uint',
    'u32',
    true,
    false,
  );
  add(
    'rg11b10ufloat',
    gl.R11F_G11F_B10F,
    gl.RGB,
    gl.UNSIGNED_INT_10F_11F_11F_REV,
    4,
    1,
    'float',
    'u32',
    floatRender,
    true,
  );
  add(
    'rgb9e5ufloat',
    gl.RGB9_E5,
    gl.RGB,
    gl.UNSIGNED_INT_5_9_9_9_REV,
    4,
    1,
    'float',
    'u32',
    false,
    true,
  );

  add(
    'depth16unorm',
    gl.DEPTH_COMPONENT16,
    gl.DEPTH_COMPONENT,
    U16,
    2,
    1,
    'depth',
    'u16',
    true,
    false,
  );
  add(
    'depth24plus',
    gl.DEPTH_COMPONENT24,
    gl.DEPTH_COMPONENT,
    U32,
    4,
    1,
    'depth',
    'f32',
    true,
    false,
  );
  add(
    'depth24plus-stencil8',
    gl.DEPTH24_STENCIL8,
    gl.DEPTH_STENCIL,
    gl.UNSIGNED_INT_24_8,
    4,
    1,
    'depth-stencil',
    'u32',
    true,
    false,
  );
  add(
    'depth32float',
    gl.DEPTH_COMPONENT32F,
    gl.DEPTH_COMPONENT,
    F32,
    4,
    1,
    'depth',
    'f32',
    true,
    false,
  );
  add(
    'depth32float-stencil8',
    gl.DEPTH32F_STENCIL8,
    gl.DEPTH_STENCIL,
    gl.FLOAT_32_UNSIGNED_INT_24_8_REV,
    8,
    1,
    'depth-stencil',
    'f32',
    true,
    false,
  );
  add(
    'stencil8',
    gl.DEPTH24_STENCIL8,
    gl.DEPTH_STENCIL,
    gl.UNSIGNED_INT_24_8,
    4,
    1,
    'depth-stencil',
    'u8',
    true,
    false,
  );
  return table;
}

export function formatInfo(table: FormatTable, format: GPUTextureFormat): FormatInfo {
  const info = table.get(format);
  if (!info)
    throw new Error(`@typegpu/kit-gl: texture format '${format}' is not supported on WebGL 2.`);
  return info;
}

/** A vertex attribute's `vertexAttrib(I)Pointer` arguments. */
export interface VertexFormatInfo {
  readonly size: number;
  readonly type: number;
  readonly normalized: boolean;
  readonly integer: boolean;
  readonly bytes: number;
}

export function vertexFormatInfo(
  gl: WebGL2RenderingContext,
  format: GPUVertexFormat,
): VertexFormatInfo {
  const match = /^(uint|sint|unorm|snorm|float)(8|16|32)(?:x(\d))?$/.exec(format);
  if (match) {
    const [, kind, bitsText, countText] = match as unknown as [string, string, string, string?];
    const bits = Number(bitsText);
    const size = countText ? Number(countText) : 1;
    const signed = kind === 'sint' || kind === 'snorm';
    const type =
      kind === 'float'
        ? bits === 16
          ? gl.HALF_FLOAT
          : gl.FLOAT
        : bits === 8
          ? signed
            ? gl.BYTE
            : gl.UNSIGNED_BYTE
          : bits === 16
            ? signed
              ? gl.SHORT
              : gl.UNSIGNED_SHORT
            : signed
              ? gl.INT
              : gl.UNSIGNED_INT;
    return {
      size,
      type,
      normalized: kind === 'unorm' || kind === 'snorm',
      integer: kind === 'uint' || kind === 'sint',
      bytes: (bits / 8) * size,
    };
  }
  if (format === 'unorm10-10-10-2') {
    return {
      size: 4,
      type: gl.UNSIGNED_INT_2_10_10_10_REV,
      normalized: true,
      integer: false,
      bytes: 4,
    };
  }
  throw new Error(`@typegpu/kit-gl: vertex format '${format}' is not supported on WebGL 2.`);
}
