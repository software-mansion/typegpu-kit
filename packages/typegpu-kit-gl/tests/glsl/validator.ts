import { chromium, type Browser, type Page } from 'playwright';
import type { RenderModule } from '../../src/glsl/renderModule.ts';

/** What WebGL 2 reported for a program: empty when both stages compiled and linked. */
export interface Report {
  readonly vertex: string;
  readonly fragment: string;
  readonly link: string;
}

export interface Validator {
  /** Compiles and links each module's stages in one WebGL 2 context. */
  check(modules: readonly RenderModule[]): Promise<Report[]>;
  /**
   * Draws three vertices into a `width` × 1 RGBA32F target with each buffer's bytes bound as the
   * device binds them, keyed by texture or block name, and returns the pixels.
   */
  run(module: RenderModule, width: number, buffers: Record<string, Uint8Array>): Promise<number[]>;
  close(): Promise<void>;
}

/** Stands in for a depth-only pipeline's missing fragment stage, as the device does. */
const EMPTY_FRAGMENT = '#version 300 es\nvoid main() {}';

/**
 * A validator backed by headless Chromium, whose WebGL 2 is ANGLE: the shader translator that
 * browsers run, so what passes here compiles where the fallback runs.
 */
export async function validator(): Promise<Validator> {
  const browser: Browser = await chromium.launch({ headless: true, channel: 'chromium' });
  const page: Page = await browser.newPage();
  return {
    async check(modules) {
      const sources = modules.map((module) => [module.vertex, module.fragment ?? EMPTY_FRAGMENT]);
      return page.evaluate((pairs) => {
        const gl = document.createElement('canvas').getContext('webgl2');
        if (!gl) throw new Error('No WebGL 2 context');
        const compile = (type: number, source: string) => {
          const shader = gl.createShader(type) as WebGLShader;
          gl.shaderSource(shader, source);
          gl.compileShader(shader);
          const log = gl.getShaderParameter(shader, gl.COMPILE_STATUS)
            ? ''
            : (gl.getShaderInfoLog(shader) ?? 'failed');
          return { shader, log };
        };
        return pairs.map(([vertexSource, fragmentSource]) => {
          const vertex = compile(gl.VERTEX_SHADER, vertexSource as string);
          const fragment = compile(gl.FRAGMENT_SHADER, fragmentSource as string);
          let link = '';
          if (!vertex.log && !fragment.log) {
            const program = gl.createProgram() as WebGLProgram;
            gl.attachShader(program, vertex.shader);
            gl.attachShader(program, fragment.shader);
            gl.linkProgram(program);
            if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
              link = gl.getProgramInfoLog(program) ?? 'failed';
            }
            gl.deleteProgram(program);
          }
          gl.deleteShader(vertex.shader);
          gl.deleteShader(fragment.shader);
          return { vertex: vertex.log, fragment: fragment.log, link };
        });
      }, sources);
    },
    async run(module, width, buffers) {
      const uploads = Object.fromEntries(
        Object.entries(buffers).map(([name, bytes]) => {
          const padded = new Uint8Array(Math.ceil(bytes.byteLength / 16) * 16 || 16);
          padded.set(bytes);
          return [name, { words: Array.from(new Uint32Array(padded.buffer)), size: bytes.length }];
        }),
      );
      return page.evaluate(
        ({ stages, columns, contents: uploaded }) => {
          const gl = document.createElement('canvas').getContext('webgl2');
          if (!gl) throw new Error('No WebGL 2 context');
          if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('No float targets');
          const program = gl.createProgram() as WebGLProgram;
          for (const [type, text] of [
            [gl.VERTEX_SHADER, stages.vertex],
            [gl.FRAGMENT_SHADER, stages.fragment],
          ] as const) {
            const shader = gl.createShader(type) as WebGLShader;
            gl.shaderSource(shader, text as string);
            gl.compileShader(shader);
            if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
              throw new Error(gl.getShaderInfoLog(shader) ?? 'compile failed');
            }
            gl.attachShader(program, shader);
          }
          gl.linkProgram(program);
          if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            throw new Error(gl.getProgramInfoLog(program) ?? 'link failed');
          }
          gl.useProgram(program);
          let unit = 0;
          for (const buffer of stages.buffers) {
            const { words, size } = uploaded[buffer.texture] ?? { words: [0, 0, 0, 0], size: 16 };
            const texels = words.length / 4;
            const texture = gl.createTexture();
            gl.activeTexture(gl.TEXTURE0 + unit);
            gl.bindTexture(gl.TEXTURE_2D, texture);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.texImage2D(
              gl.TEXTURE_2D,
              0,
              gl.RGBA32UI,
              texels,
              1,
              0,
              gl.RGBA_INTEGER,
              gl.UNSIGNED_INT,
              new Uint32Array(words),
            );
            gl.uniform1i(gl.getUniformLocation(program, buffer.texture), unit);
            gl.uniform2ui(gl.getUniformLocation(program, buffer.range), 0, size);
            unit++;
          }
          for (const [index, block] of (stages.blocks ?? []).entries()) {
            const { words } = uploaded[block.block] ?? { words: [] };
            const contents = new Uint32Array(block.size / 4);
            contents.set(words.slice(0, contents.length));
            const buffer = gl.createBuffer();
            gl.bindBuffer(gl.UNIFORM_BUFFER, buffer);
            gl.bufferData(gl.UNIFORM_BUFFER, contents, gl.STATIC_DRAW);
            gl.uniformBlockBinding(program, gl.getUniformBlockIndex(program, block.block), index);
            gl.bindBufferBase(gl.UNIFORM_BUFFER, index, buffer);
          }
          gl.uniform1i(gl.getUniformLocation(program, 'kit_baseVertex'), 0);
          gl.uniform1i(gl.getUniformLocation(program, 'kit_firstInstance'), 0);
          const target = gl.createTexture();
          gl.activeTexture(gl.TEXTURE0 + unit);
          gl.bindTexture(gl.TEXTURE_2D, target);
          gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, columns, 1);
          const framebuffer = gl.createFramebuffer();
          gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
          gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, target, 0);
          gl.viewport(0, 0, columns, 1);
          gl.drawArrays(gl.TRIANGLES, 0, 3);
          const pixels = new Float32Array(columns * 4);
          gl.readPixels(0, 0, columns, 1, gl.RGBA, gl.FLOAT, pixels);
          const error = gl.getError();
          if (error !== gl.NO_ERROR) throw new Error(`WebGL error ${error}`);
          return Array.from(pixels);
        },
        { stages: module, columns: width, contents: uploads },
      );
    },
    close: () => browser.close(),
  };
}

/** A report's problems as one line each, empty when the program compiled and linked. */
export function problems(report: Report): string[] {
  return [
    report.vertex && `vertex: ${report.vertex}`,
    report.fragment && `fragment: ${report.fragment}`,
    report.link && `link: ${report.link}`,
  ]
    .filter(Boolean)
    .map((line) => line.replaceAll('\u0000', '').trim());
}
