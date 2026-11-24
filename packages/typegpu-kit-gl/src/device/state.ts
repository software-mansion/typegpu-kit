/**
 * A cache of the GL state the device sets. Every state change goes through it, so a draw that
 * repeats the previous one's state makes no GL calls for it.
 */

import type { Extensions } from './extensions.ts';

export interface BlendState {
  readonly enabled: boolean;
  readonly colorOp: number;
  readonly colorSrc: number;
  readonly colorDst: number;
  readonly alphaOp: number;
  readonly alphaSrc: number;
  readonly alphaDst: number;
}

export interface StencilFace {
  readonly func: number;
  readonly fail: number;
  readonly depthFail: number;
  readonly pass: number;
}

export interface GLState {
  readonly gl: WebGL2RenderingContext;
  program(program: WebGLProgram | null): void;
  vertexArray(vao: WebGLVertexArrayObject | null): void;
  drawFramebuffer(framebuffer: WebGLFramebuffer | null): void;
  readFramebuffer(framebuffer: WebGLFramebuffer | null): void;
  arrayBuffer(buffer: WebGLBuffer | null): void;
  texture(unit: number, target: number, texture: WebGLTexture | null): void;
  sampler(unit: number, sampler: WebGLSampler | null): void;
  /** Binds `texture` on the unit the device keeps for uploads and parameter changes. */
  scratchTexture(target: number, texture: WebGLTexture | null): void;
  enable(cap: number, on: boolean): void;
  blend(index: number, state: BlendState): void;
  colorMask(index: number, mask: number): void;
  /** Sets every draw buffer's mask; used before clears. */
  colorMaskAll(mask: number): void;
  depthFunc(func: number): void;
  depthMask(on: boolean): void;
  stencil(front: StencilFace, back: StencilFace, ref: number, readMask: number): void;
  stencilWriteMask(mask: number): void;
  cullFace(face: number): void;
  frontFace(face: number): void;
  polygonOffset(factor: number, units: number): void;
  viewport(x: number, y: number, width: number, height: number): void;
  scissor(x: number, y: number, width: number, height: number): void;
  depthRange(near: number, far: number): void;
  blendColor(r: number, g: number, b: number, a: number): void;
  unpack(flipY: boolean, premultiply: boolean, rowLength: number, imageHeight: number): void;
  /** Forgets everything cached, after GL calls that bypassed the cache. */
  reset(): void;
  readonly scratchUnit: number;
}

const FULL_MASK = 0xf;

export function createState(gl: WebGL2RenderingContext, ext: Extensions): GLState {
  const indexed = ext.drawBuffersIndexed;
  const maxDrawBuffers = gl.getParameter(gl.MAX_DRAW_BUFFERS) as number;
  const scratchUnit = (gl.getParameter(gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS) as number) - 1;

  let program: WebGLProgram | null | undefined;
  let vao: WebGLVertexArrayObject | null | undefined;
  let drawFbo: WebGLFramebuffer | null | undefined;
  let readFbo: WebGLFramebuffer | null | undefined;
  let arrayBuffer: WebGLBuffer | null | undefined;
  let activeUnit = -1;
  const textures = new Map<number, WebGLTexture | null>();
  const samplers = new Map<number, WebGLSampler | null>();
  const caps = new Map<number, boolean>();
  let blends: (BlendState | undefined)[] = [];
  let masks: (number | undefined)[] = [];
  let depthFunc: number | undefined;
  let depthMask: boolean | undefined;
  let stencilKey = '';
  let stencilWrite: number | undefined;
  let cull: number | undefined;
  let front: number | undefined;
  let offset = '';
  let viewport = '';
  let scissor = '';
  let range = '';
  let constant = '';
  let unpackKey = '';

  const reset = () => {
    program = undefined;
    vao = undefined;
    drawFbo = undefined;
    readFbo = undefined;
    arrayBuffer = undefined;
    activeUnit = -1;
    textures.clear();
    samplers.clear();
    caps.clear();
    blends = [];
    masks = [];
    depthFunc = undefined;
    depthMask = undefined;
    stencilKey = '';
    stencilWrite = undefined;
    cull = undefined;
    front = undefined;
    offset = '';
    viewport = '';
    scissor = '';
    range = '';
    constant = '';
    unpackKey = '';
  };

  const unit = (index: number) => {
    if (activeUnit !== index) {
      activeUnit = index;
      gl.activeTexture(gl.TEXTURE0 + index);
    }
  };

  const texture = (index: number, target: number, value: WebGLTexture | null) => {
    const key = index * 0x10000 + target;
    if (textures.get(key) === value) return;
    unit(index);
    gl.bindTexture(target, value);
    textures.set(key, value);
  };

  const sameBlend = (a: BlendState | undefined, b: BlendState) =>
    a !== undefined &&
    a.enabled === b.enabled &&
    (!b.enabled ||
      (a.colorOp === b.colorOp &&
        a.colorSrc === b.colorSrc &&
        a.colorDst === b.colorDst &&
        a.alphaOp === b.alphaOp &&
        a.alphaSrc === b.alphaSrc &&
        a.alphaDst === b.alphaDst));

  const colorMask = (index: number, mask: number) => {
    if (masks[index] === mask) return;
    const r = (mask & 1) !== 0;
    const g = (mask & 2) !== 0;
    const b = (mask & 4) !== 0;
    const a = (mask & 8) !== 0;
    if (indexed) {
      indexed.colorMaskiOES(index, r, g, b, a);
      masks[index] = mask;
    } else {
      gl.colorMask(r, g, b, a);
      masks = Array.from({ length: maxDrawBuffers }, () => mask);
    }
  };

  return {
    gl,
    scratchUnit,
    reset,

    program(value) {
      if (program === value) return;
      program = value;
      gl.useProgram(value);
    },

    vertexArray(value) {
      if (vao === value) return;
      vao = value;
      gl.bindVertexArray(value);
    },

    drawFramebuffer(value) {
      if (drawFbo === value) return;
      drawFbo = value;
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, value);
    },

    readFramebuffer(value) {
      if (readFbo === value) return;
      readFbo = value;
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, value);
    },

    arrayBuffer(value) {
      if (arrayBuffer === value) return;
      arrayBuffer = value;
      gl.bindBuffer(gl.ARRAY_BUFFER, value);
    },

    texture,

    sampler(index, value) {
      if (samplers.get(index) === value) return;
      samplers.set(index, value);
      gl.bindSampler(index, value);
    },

    scratchTexture(target, value) {
      texture(scratchUnit, target, value);
      unit(scratchUnit);
    },

    enable(cap, on) {
      if (caps.get(cap) === on) return;
      caps.set(cap, on);
      if (on) gl.enable(cap);
      else gl.disable(cap);
    },

    blend(index, state) {
      if (sameBlend(blends[index], state)) return;
      if (indexed) {
        if (state.enabled) {
          indexed.enableiOES(gl.BLEND, index);
          indexed.blendEquationSeparateiOES(index, state.colorOp, state.alphaOp);
          indexed.blendFuncSeparateiOES(
            index,
            state.colorSrc,
            state.colorDst,
            state.alphaSrc,
            state.alphaDst,
          );
        } else {
          indexed.disableiOES(gl.BLEND, index);
        }
        blends[index] = state;
        caps.delete(gl.BLEND);
        return;
      }
      if (state.enabled) {
        gl.enable(gl.BLEND);
        gl.blendEquationSeparate(state.colorOp, state.alphaOp);
        gl.blendFuncSeparate(state.colorSrc, state.colorDst, state.alphaSrc, state.alphaDst);
      } else {
        gl.disable(gl.BLEND);
      }
      blends = Array.from({ length: maxDrawBuffers }, () => state);
      caps.set(gl.BLEND, state.enabled);
    },

    colorMask,

    colorMaskAll(mask) {
      if (indexed) {
        for (let i = 0; i < maxDrawBuffers; i++) colorMask(i, mask);
      } else {
        colorMask(0, mask);
      }
    },

    depthFunc(value) {
      if (depthFunc === value) return;
      depthFunc = value;
      gl.depthFunc(value);
    },

    depthMask(value) {
      if (depthMask === value) return;
      depthMask = value;
      gl.depthMask(value);
    },

    stencil(f, b, ref, readMask) {
      const key = `${f.func},${f.fail},${f.depthFail},${f.pass},${b.func},${b.fail},${b.depthFail},${b.pass},${ref},${readMask}`;
      if (stencilKey === key) return;
      stencilKey = key;
      gl.stencilFuncSeparate(gl.FRONT, f.func, ref, readMask);
      gl.stencilFuncSeparate(gl.BACK, b.func, ref, readMask);
      gl.stencilOpSeparate(gl.FRONT, f.fail, f.depthFail, f.pass);
      gl.stencilOpSeparate(gl.BACK, b.fail, b.depthFail, b.pass);
    },

    stencilWriteMask(value) {
      if (stencilWrite === value) return;
      stencilWrite = value;
      gl.stencilMask(value);
    },

    cullFace(value) {
      if (cull === value) return;
      cull = value;
      gl.cullFace(value);
    },

    frontFace(value) {
      if (front === value) return;
      front = value;
      gl.frontFace(value);
    },

    polygonOffset(factor, units) {
      const key = `${factor},${units}`;
      if (offset === key) return;
      offset = key;
      gl.polygonOffset(factor, units);
    },

    viewport(x, y, width, height) {
      const key = `${x},${y},${width},${height}`;
      if (viewport === key) return;
      viewport = key;
      gl.viewport(x, y, width, height);
    },

    scissor(x, y, width, height) {
      const key = `${x},${y},${width},${height}`;
      if (scissor === key) return;
      scissor = key;
      gl.scissor(x, y, width, height);
    },

    depthRange(near, far) {
      const key = `${near},${far}`;
      if (range === key) return;
      range = key;
      gl.depthRange(near, far);
    },

    blendColor(r, g, b, a) {
      const key = `${r},${g},${b},${a}`;
      if (constant === key) return;
      constant = key;
      gl.blendColor(r, g, b, a);
    },

    unpack(flipY, premultiply, rowLength, imageHeight) {
      const key = `${flipY},${premultiply},${rowLength},${imageHeight}`;
      if (unpackKey === key) return;
      unpackKey = key;
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, flipY);
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, premultiply);
      gl.pixelStorei(gl.UNPACK_ROW_LENGTH, rowLength);
      gl.pixelStorei(gl.UNPACK_IMAGE_HEIGHT, imageHeight);
    },
  };
}

export { FULL_MASK };
