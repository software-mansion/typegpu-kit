/**
 * A WebGL 2 weaker than the one tests run on: every `webgl2` context a canvas gives from now on
 * reports the limits WebGL 2 guarantees, and none of the extensions named. Calls go to the real
 * context, so a device that keeps to what the context reports keeps to the minimum, while one that
 * leans on the driver for a limit sees the real driver's larger one.
 */

export interface Restriction {
  /** Clamps the limits to WebGL 2's minimums. Default true. */
  readonly minimum?: boolean;
  /** `MAX_TEXTURE_SIZE` at the minimum. Default 4096; WebGL 2 guarantees 2048. */
  readonly textureSize?: number;
  /** Extensions `getExtension` refuses and `getSupportedExtensions` leaves out. */
  readonly hide?: readonly string[];
}

/** Every extension the device asks for, for hiding all at once. */
export const EXTENSIONS = [
  'EXT_color_buffer_float',
  'EXT_color_buffer_half_float',
  'EXT_float_blend',
  'OES_texture_float_linear',
  'EXT_clip_control',
  'OES_draw_buffers_indexed',
  'KHR_parallel_shader_compile',
  'EXT_texture_filter_anisotropic',
  'WEBGL_multi_draw',
  'EXT_disjoint_timer_query_webgl2',
  'WEBGL_debug_renderer_info',
] as const;

/**
 * Installs the restriction on `HTMLCanvasElement` and `OffscreenCanvas` and returns its undo. It
 * refers to nothing outside itself, so a page can run its source as an init script.
 */
export function install(restriction: Restriction): () => void {
  const { minimum = true, textureSize = 4096, hide = [] } = restriction;
  const hidden = new Set(hide);
  const limits: Record<string, number> = minimum
    ? {
        MAX_TEXTURE_SIZE: textureSize,
        MAX_CUBE_MAP_TEXTURE_SIZE: Math.min(textureSize, 2048),
        MAX_RENDERBUFFER_SIZE: Math.min(textureSize, 2048),
        MAX_3D_TEXTURE_SIZE: 256,
        MAX_ARRAY_TEXTURE_LAYERS: 256,
        MAX_DRAW_BUFFERS: 4,
        MAX_COLOR_ATTACHMENTS: 4,
        MAX_TEXTURE_IMAGE_UNITS: 16,
        MAX_VERTEX_TEXTURE_IMAGE_UNITS: 16,
        MAX_COMBINED_TEXTURE_IMAGE_UNITS: 32,
        MAX_UNIFORM_BUFFER_BINDINGS: 24,
        MAX_UNIFORM_BLOCK_SIZE: 16384,
        MAX_VERTEX_UNIFORM_BLOCKS: 12,
        MAX_FRAGMENT_UNIFORM_BLOCKS: 12,
        MAX_COMBINED_UNIFORM_BLOCKS: 24,
        MAX_VARYING_COMPONENTS: 60,
        MAX_VARYING_VECTORS: 15,
        MAX_VERTEX_ATTRIBS: 16,
        MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS: 64,
        MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS: 4,
        MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS: 4,
        MAX_SAMPLES: 4,
      }
    : {};
  const constants = WebGL2RenderingContext as unknown as Record<string, number>;
  const clamped = new Map(Object.entries(limits).map(([name, value]) => [constants[name], value]));
  const wrapped = new WeakMap<object, object>();

  const wrap = (gl: WebGL2RenderingContext) => {
    const bound = new Map<PropertyKey, unknown>();
    const overrides: Record<string, unknown> = {
      getParameter(name: number) {
        const value = gl.getParameter(name) as unknown;
        const limit = clamped.get(name);
        return limit !== undefined && typeof value === 'number' ? Math.min(value, limit) : value;
      },
      getExtension(name: string) {
        return hidden.has(name) ? null : gl.getExtension(name);
      },
      getSupportedExtensions() {
        return (gl.getSupportedExtensions() ?? []).filter((name) => !hidden.has(name));
      },
    };
    return new Proxy(gl, {
      get(target, key) {
        if (typeof key === 'string' && key in overrides) return overrides[key];
        if (bound.has(key)) return bound.get(key);
        const value = Reflect.get(target, key, target) as unknown;
        if (typeof value !== 'function') return value;
        const method = (value as (...args: unknown[]) => unknown).bind(target);
        bound.set(key, method);
        return method;
      },
    });
  };

  const undo: (() => void)[] = [];
  const kinds = [
    HTMLCanvasElement,
    typeof OffscreenCanvas === 'undefined' ? undefined : OffscreenCanvas,
  ];
  for (const kind of kinds) {
    if (!kind) continue;
    const prototype = kind.prototype as unknown as {
      getContext: (this: object, type: string, ...rest: unknown[]) => unknown;
    };
    const original = prototype.getContext;
    prototype.getContext = function (type, ...rest) {
      const context = original.call(this, type, ...rest);
      if (type !== 'webgl2' || !context) return context;
      let proxy = wrapped.get(context);
      if (!proxy) {
        proxy = wrap(context as WebGL2RenderingContext);
        wrapped.set(context, proxy);
      }
      return proxy;
    };
    undo.push(() => {
      prototype.getContext = original;
    });
  }
  return () => {
    for (const each of undo) each();
  };
}

/** The script that installs `restriction` in a page, for Playwright's `addInitScript`. */
export const installScript = (restriction: Restriction) =>
  `(${install.toString()})(${JSON.stringify(restriction)});`;
