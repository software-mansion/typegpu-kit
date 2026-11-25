/** GLSL programs: compiled and linked once per pair of sources, with errors that show the source. */

import type { Ctx } from './context.ts';

export interface Program {
  readonly program: WebGLProgram;
  /** Every active uniform's location by name, array uniforms under their base name too. */
  readonly uniforms: ReadonlyMap<string, WebGLUniformLocation>;
}

export type Linked =
  | { readonly ok: true; readonly program: Program }
  | { readonly ok: false; readonly message: string };

/** A program the driver may still be compiling, which `settle` finishes at once if it is. */
interface Pending {
  readonly done: Promise<Linked>;
  settle(): Linked;
}

const caches = new WeakMap<Ctx, Map<string, Linked | Pending>>();

function cacheOf(ctx: Ctx) {
  let cache = caches.get(ctx);
  if (!cache) {
    cache = new Map();
    caches.set(ctx, cache);
  }
  return cache;
}

/** Numbers the lines of `source`, for error messages. */
export function numbered(source: string): string {
  const lines = source.split('\n');
  const width = String(lines.length).length;
  return lines.map((line, i) => `${String(i + 1).padStart(width)}: ${line}`).join('\n');
}

const DEPTH_ONLY_FRAGMENT = `#version 300 es
precision highp float;
void main() {}
`;

function compile(ctx: Ctx, type: number, source: string): WebGLShader {
  const { gl } = ctx;
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  return shader;
}

function finish(
  ctx: Ctx,
  program: WebGLProgram,
  shaders: [WebGLShader, WebGLShader],
  sources: [string, string],
  label: string,
): Linked {
  const { gl } = ctx;
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const parts: string[] = [];
    shaders.forEach((shader, i) => {
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const stage = i === 0 ? 'vertex' : 'fragment';
        parts.push(
          `The ${stage} shader of ${label} failed to compile:\n${gl.getShaderInfoLog(shader) ?? ''}\n${numbered(sources[i]!)}`,
        );
      }
    });
    if (parts.length === 0) {
      parts.push(
        `${label} failed to link:\n${gl.getProgramInfoLog(program) ?? ''}\n--- vertex ---\n${numbered(sources[0])}\n--- fragment ---\n${numbered(sources[1])}`,
      );
    }
    for (const shader of shaders) gl.deleteShader(shader);
    gl.deleteProgram(program);
    return { ok: false, message: parts.join('\n\n') };
  }
  for (const shader of shaders) {
    gl.detachShader(program, shader);
    gl.deleteShader(shader);
  }
  const uniforms = new Map<string, WebGLUniformLocation>();
  const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS) as number;
  for (let i = 0; i < count; i++) {
    const active = gl.getActiveUniform(program, i);
    if (!active) continue;
    const location = gl.getUniformLocation(program, active.name);
    if (!location) continue;
    uniforms.set(active.name, location);
    if (active.name.endsWith('[0]')) uniforms.set(active.name.slice(0, -3), location);
  }
  ctx.track(() => gl.deleteProgram(program));
  return { ok: true, program: { program, uniforms } };
}

function start(ctx: Ctx, vertex: string, fragment: string) {
  const { gl } = ctx;
  const sources: [string, string] = [vertex, fragment];
  const shaders: [WebGLShader, WebGLShader] = [
    compile(ctx, gl.VERTEX_SHADER, vertex),
    compile(ctx, gl.FRAGMENT_SHADER, fragment),
  ];
  const program = gl.createProgram()!;
  for (const shader of shaders) gl.attachShader(program, shader);
  gl.linkProgram(program);
  return { program, shaders, sources };
}

/** Links the program now. */
export function link(
  ctx: Ctx,
  vertex: string,
  fragment: string | undefined,
  label: string,
): Linked {
  const fragmentSource = fragment ?? DEPTH_ONLY_FRAGMENT;
  const key = `${vertex}\u0000${fragmentSource}`;
  const cache = cacheOf(ctx);
  const cached = cache.get(key);
  if (cached) return 'settle' in cached ? cached.settle() : cached;
  const { program, shaders, sources } = start(ctx, vertex, fragmentSource);
  const linked = finish(ctx, program, shaders, sources, label);
  cache.set(key, linked);
  return linked;
}

/**
 * Links the program without blocking where `KHR_parallel_shader_compile` exists: resolves once
 * the driver reports completion.
 */
export function linkAsync(
  ctx: Ctx,
  vertex: string,
  fragment: string | undefined,
  label: string,
): Promise<Linked> {
  const parallel = ctx.ext.parallelCompile;
  if (!parallel) return Promise.resolve(link(ctx, vertex, fragment, label));
  const fragmentSource = fragment ?? DEPTH_ONLY_FRAGMENT;
  const key = `${vertex}\u0000${fragmentSource}`;
  const cache = cacheOf(ctx);
  const cached = cache.get(key);
  if (cached) return 'settle' in cached ? cached.done : Promise.resolve(cached);
  const { gl } = ctx;
  const { program, shaders, sources } = start(ctx, vertex, fragmentSource);
  let linked: Linked | undefined;
  let resolve!: (linked: Linked) => void;
  const done = new Promise<Linked>((settled) => {
    resolve = settled;
  });
  // Asking for the link status waits for the driver, so a caller that cannot wait settles early
  const settle = () => {
    if (!linked) {
      linked = finish(ctx, program, shaders, sources, label);
      cache.set(key, linked);
      resolve(linked);
    }
    return linked;
  };
  const poll = () => {
    if (linked) return;
    if (ctx.lost || gl.getProgramParameter(program, parallel.COMPLETION_STATUS_KHR)) settle();
    else setTimeout(poll, 4);
  };
  cache.set(key, { done, settle });
  poll();
  return done;
}
