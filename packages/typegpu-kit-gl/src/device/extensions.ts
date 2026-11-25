/** The WebGL 2 extensions the device enables, each `null` where the browser lacks it. */

export interface ClipControl {
  readonly LOWER_LEFT_EXT: number;
  readonly ZERO_TO_ONE_EXT: number;
  clipControlEXT(origin: number, depth: number): void;
}

export interface DrawBuffersIndexed {
  enableiOES(target: number, index: number): void;
  disableiOES(target: number, index: number): void;
  blendEquationSeparateiOES(buf: number, modeRGB: number, modeAlpha: number): void;
  blendFuncSeparateiOES(
    buf: number,
    srcRGB: number,
    dstRGB: number,
    srcAlpha: number,
    dstAlpha: number,
  ): void;
  colorMaskiOES(buf: number, r: boolean, g: boolean, b: boolean, a: boolean): void;
}

export interface Anisotropic {
  readonly TEXTURE_MAX_ANISOTROPY_EXT: number;
  readonly MAX_TEXTURE_MAX_ANISOTROPY_EXT: number;
}

export interface TimerQuery {
  readonly TIME_ELAPSED_EXT: number;
  readonly TIMESTAMP_EXT: number;
  readonly QUERY_COUNTER_BITS_EXT: number;
  readonly GPU_DISJOINT_EXT: number;
  queryCounterEXT(query: WebGLQuery, target: number): void;
}

export interface Extensions {
  readonly colorBufferFloat: object | null;
  readonly colorBufferHalfFloat: object | null;
  readonly textureFloatLinear: object | null;
  readonly floatBlend: object | null;
  readonly anisotropic: Anisotropic | null;
  readonly clipControl: ClipControl | null;
  readonly drawBuffersIndexed: DrawBuffersIndexed | null;
  readonly parallelCompile: { readonly COMPLETION_STATUS_KHR: number } | null;
  readonly timerQuery: TimerQuery | null;
  readonly debugRendererInfo: {
    readonly UNMASKED_VENDOR_WEBGL: number;
    readonly UNMASKED_RENDERER_WEBGL: number;
  } | null;
}

export function enableExtensions(gl: WebGL2RenderingContext): Extensions {
  const get = <T>(name: string) => gl.getExtension(name) as T | null;
  return {
    colorBufferFloat: get('EXT_color_buffer_float'),
    colorBufferHalfFloat: get('EXT_color_buffer_half_float'),
    textureFloatLinear: get('OES_texture_float_linear'),
    floatBlend: get('EXT_float_blend'),
    anisotropic: get('EXT_texture_filter_anisotropic'),
    clipControl: get('EXT_clip_control'),
    drawBuffersIndexed: get('OES_draw_buffers_indexed'),
    parallelCompile: get('KHR_parallel_shader_compile'),
    timerQuery: get('EXT_disjoint_timer_query_webgl2'),
    debugRendererInfo: get('WEBGL_debug_renderer_info'),
  };
}
