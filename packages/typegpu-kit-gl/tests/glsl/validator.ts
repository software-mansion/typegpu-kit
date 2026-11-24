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
