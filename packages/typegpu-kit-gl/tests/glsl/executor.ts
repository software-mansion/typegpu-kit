/**
 * Runs a compute module's GPU form on raw WebGL 2, exactly as `computeModule.ts` tells the device
 * to: the reference the device follows. Resources are keyed by `group:binding`.
 */

import {
  INVOCATIONS,
  LAYER,
  PASS_VERTEX,
  WORKGROUPS,
  type ComputeModule,
} from '../../src/glsl/computeModule.ts';

export interface BufferResource {
  readonly kind: 'buffer';
  /** The whole buffer, bound from its first byte. */
  bytes: Uint8Array;
}

export interface TextureResource {
  readonly kind: 'texture';
  readonly format: 'rgba32float' | 'r32float' | 'rgba8unorm' | 'rgba32uint';
  readonly dimension: '2d' | '2d-array' | '3d';
  readonly size: readonly [number, number, number];
  /** Texels, rows top to bottom, layers in order. */
  data: Float32Array | Uint8Array | Uint32Array;
}

export type Resource = BufferResource | TextureResource;

/** Width of the data textures, small so that buffers span several rows. */
const DATA_WIDTH = 8;

interface Format {
  readonly internal: number;
  readonly format: number;
  readonly type: number;
  readonly channels: number;
}

function formatOf(gl: WebGL2RenderingContext, format: TextureResource['format']): Format {
  if (format === 'rgba32float') {
    return { internal: gl.RGBA32F, format: gl.RGBA, type: gl.FLOAT, channels: 4 };
  }
  if (format === 'r32float')
    return { internal: gl.R32F, format: gl.RED, type: gl.FLOAT, channels: 1 };
  if (format === 'rgba32uint') {
    return { internal: gl.RGBA32UI, format: gl.RGBA_INTEGER, type: gl.UNSIGNED_INT, channels: 4 };
  }
  return { internal: gl.RGBA8, format: gl.RGBA, type: gl.UNSIGNED_BYTE, channels: 4 };
}

function program(gl: WebGL2RenderingContext, fragment: string): WebGLProgram {
  const result = gl.createProgram() as WebGLProgram;
  for (const [type, text] of [
    [gl.VERTEX_SHADER, PASS_VERTEX],
    [gl.FRAGMENT_SHADER, fragment],
  ] as const) {
    const shader = gl.createShader(type) as WebGLShader;
    gl.shaderSource(shader, text);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(`${gl.getShaderInfoLog(shader)}\n${text}`);
    }
    gl.attachShader(result, shader);
  }
  gl.linkProgram(result);
  if (!gl.getProgramParameter(result, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(result) ?? 'link failed');
  }
  return result;
}

function targetOf(gl: WebGL2RenderingContext, dimension: TextureResource['dimension']): number {
  return dimension === '3d'
    ? gl.TEXTURE_3D
    : dimension === '2d-array'
      ? gl.TEXTURE_2D_ARRAY
      : gl.TEXTURE_2D;
}

interface Uploaded {
  readonly texture: WebGLTexture;
  readonly target: number;
  readonly width: number;
  readonly height: number;
  readonly depth: number;
  readonly format: Format;
}

/** A buffer's data texture: `RGBA32UI`, four words to a texel, rows of `DATA_WIDTH` texels. */
function uploadBuffer(gl: WebGL2RenderingContext, bytes: Uint8Array): Uploaded {
  const texels = Math.max(1, Math.ceil(bytes.length / 16));
  const height = Math.ceil(texels / DATA_WIDTH);
  const words = new Uint32Array(DATA_WIDTH * height * 4);
  new Uint8Array(words.buffer).set(bytes);
  const format = formatOf(gl, 'rgba32uint');
  return upload(gl, gl.TEXTURE_2D, format, [DATA_WIDTH, height, 1], words);
}

function upload(
  gl: WebGL2RenderingContext,
  target: number,
  format: Format,
  [width, height, depth]: readonly [number, number, number],
  data: ArrayBufferView | null,
): Uploaded {
  const texture = gl.createTexture() as WebGLTexture;
  gl.bindTexture(target, texture);
  gl.texParameteri(target, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(target, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  if (target === gl.TEXTURE_2D) {
    gl.texImage2D(target, 0, format.internal, width, height, 0, format.format, format.type, data);
  } else {
    gl.texImage3D(
      target,
      0,
      format.internal,
      width,
      height,
      depth,
      0,
      format.format,
      format.type,
      data,
    );
  }
  return { texture, target, width, height, depth, format };
}

/** Reads every layer of a texture back, as `readPixels` gives each format. */
function readBack(gl: WebGL2RenderingContext, uploaded: Uploaded): ArrayBufferView {
  const { width, height, depth, format, target } = uploaded;
  const layer = width * height * 4;
  const integer = format.type === gl.UNSIGNED_INT;
  const bytes = format.type === gl.UNSIGNED_BYTE;
  const all = integer
    ? new Uint32Array(layer * depth)
    : bytes
      ? new Uint8Array(layer * depth)
      : new Float32Array(layer * depth);
  const framebuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  for (let z = 0; z < depth; z++) {
    if (target === gl.TEXTURE_2D) {
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, target, uploaded.texture, 0);
    } else {
      gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, uploaded.texture, 0, z);
    }
    const view = all.subarray(z * layer, (z + 1) * layer);
    gl.readPixels(
      0,
      0,
      width,
      height,
      integer ? gl.RGBA_INTEGER : gl.RGBA,
      integer ? gl.UNSIGNED_INT : bytes ? gl.UNSIGNED_BYTE : gl.FLOAT,
      view,
    );
  }
  gl.deleteFramebuffer(framebuffer);
  return all;
}

/**
 * Runs one dispatch of `workgroups` with the GPU form `module` over `resources`, and writes the
 * results back into the resources the passes target.
 */
export function dispatch(
  gl: WebGL2RenderingContext,
  module: ComputeModule,
  resources: ReadonlyMap<string, Resource>,
  workgroups: readonly [number, number, number],
): void {
  if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('No float render targets');
  const resource = (group: number, binding: number) => {
    const found = resources.get(`${group}:${binding}`);
    if (!found) throw new Error(`Nothing bound at ${group}:${binding}`);
    return found;
  };

  // Every pass reads the contents from before the dispatch
  const old = new Map<string, Uploaded>();
  for (const [key, each] of resources) {
    if (each.kind === 'buffer') old.set(key, uploadBuffer(gl, each.bytes));
    else {
      const format = formatOf(gl, each.format);
      old.set(key, upload(gl, targetOf(gl, each.dimension), format, each.size, each.data));
    }
  }
  const invocations = workgroups.map((count, i) => count * (module.workgroupSize[i] as number));

  const results = new Map<string, Uploaded>();
  for (const pass of module.passes) {
    const shader = program(gl, pass.fragment);
    gl.useProgram(shader);
    const uniform = (name: string) => gl.getUniformLocation(shader, name);
    let unit = 0;
    for (const each of module.buffers) {
      const key = `${each.group}:${each.binding}`;
      const bound = resource(each.group, each.binding) as BufferResource;
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, (old.get(key) as Uploaded).texture);
      gl.uniform1i(uniform(each.texture), unit++);
      gl.uniform2ui(uniform(each.range), 0, bound.bytes.length);
    }
    for (const [index, each] of module.blocks.entries()) {
      const bound = resource(each.group, each.binding) as BufferResource;
      const contents = new Uint8Array(each.size);
      contents.set(bound.bytes.subarray(0, each.size));
      const buffer = gl.createBuffer();
      gl.bindBuffer(gl.UNIFORM_BUFFER, buffer);
      gl.bufferData(gl.UNIFORM_BUFFER, contents, gl.STATIC_DRAW);
      gl.uniformBlockBinding(shader, gl.getUniformBlockIndex(shader, each.block), index);
      gl.bindBufferBase(gl.UNIFORM_BUFFER, index, buffer);
    }
    for (const each of module.textures) {
      const texture = old.get(`${each.group}:${each.binding}`) as Uploaded;
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(texture.target, texture.texture);
      gl.uniform1i(uniform(each.uniform), unit++);
    }
    for (const each of module.levels) gl.uniform1i(uniform(each.uniform), 1);
    gl.uniform3ui(uniform(INVOCATIONS), ...(invocations as [number, number, number]));
    gl.uniform3ui(uniform(WORKGROUPS), ...workgroups);

    // A new texture shaped as the target's, rendered layer by layer, with no flip
    const { target } = pass;
    const key = `${target.group}:${target.binding}`;
    const before = old.get(key) as Uploaded;
    // On a unit no input uses, so creating the texture unbinds nothing
    gl.activeTexture(gl.TEXTURE0 + unit);
    const after = upload(
      gl,
      before.target,
      before.format,
      [before.width, before.height, before.depth],
      null,
    );
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.viewport(0, 0, after.width, after.height);
    gl.disable(gl.BLEND);
    for (let z = 0; z < after.depth; z++) {
      if (after.target === gl.TEXTURE_2D) {
        gl.framebufferTexture2D(
          gl.FRAMEBUFFER,
          gl.COLOR_ATTACHMENT0,
          gl.TEXTURE_2D,
          after.texture,
          0,
        );
      } else {
        gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, after.texture, 0, z);
      }
      gl.uniform1i(uniform(LAYER), z);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    gl.deleteFramebuffer(framebuffer);
    results.set(key, after);
    const error = gl.getError();
    if (error !== gl.NO_ERROR) throw new Error(`WebGL error ${error} in a pass`);
  }

  // After the last pass, the new textures replace the targets' contents
  for (const [key, after] of results) {
    const each = resources.get(key) as Resource;
    const pixels = readBack(gl, after);
    if (each.kind === 'buffer') {
      each.bytes = new Uint8Array(pixels.buffer).slice(0, each.bytes.length);
    } else {
      const channels = formatOf(gl, each.format).channels;
      const values = pixels as Float32Array | Uint8Array | Uint32Array;
      const texels = values.length / 4;
      const data = new (values.constructor as Float32ArrayConstructor)(texels * channels);
      for (let i = 0; i < texels; i++) {
        for (let c = 0; c < channels; c++) data[i * channels + c] = values[i * 4 + c] as number;
      }
      each.data = data;
    }
  }
}
