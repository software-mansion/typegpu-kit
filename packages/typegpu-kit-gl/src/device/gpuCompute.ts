/**
 * Compute dispatches as fragment passes, for kernels whose generator wrote a GPU form: each pass
 * renders a target's new contents into a fresh texture from the contents before the dispatch, and
 * the new textures replace the targets once every pass ran. `src/glsl/computeModule.ts` states
 * when a dispatch may run this way; any other dispatch runs on the CPU.
 */

import type { CpuModule } from '../cpu/types.ts';
import {
  type ComputeModule,
  type Scatter,
  type ScatterField,
  type ComputeTarget,
  INVOCATIONS,
  LAYER,
  LAYERS_PER_DRAW,
  PASS_VERTEX,
  WORKGROUPS,
} from '../glsl/computeModule.ts';
import { type Bound, bufferEntry } from './binding.ts';
import { type BufferImpl, dataTexture, replaceData } from './buffer.ts';
import type { Ctx } from './context.ts';
import { link, type Program } from './program.ts';
import type { BlendState } from './state.ts';
import {
  blitLevel,
  glWritten,
  mipSize,
  onGl,
  replaceStorage,
  type TextureImpl,
  type ViewImpl,
  writeEverything,
} from './texture.ts';
import { bindUses, connectUses, setInt, type Uses } from './uses.ts';

interface GpuPass {
  readonly target: ComputeTarget;
  readonly program: Program;
  readonly uses: Uses;
  readonly invocations: WebGLUniformLocation | null;
  readonly workgroups: WebGLUniformLocation | null;
  readonly layer: WebGLUniformLocation | null;
  /** Present for a pass of points: its capture stage and how its records are laid out. */
  readonly scatter: Points | undefined;
  /** For a pass of points, the uniforms that pick the word and channel it draws. */
  readonly word: WebGLUniformLocation | null;
  readonly phase: WebGLUniformLocation | null;
}

interface Points {
  readonly sites: number;
  readonly words: number;
  readonly lanes: number;
  readonly capture: Program;
  readonly captureUses: Uses;
  readonly invocations: WebGLUniformLocation | null;
  readonly workgroups: WebGLUniformLocation | null;
  /** The draw stage's attributes, with their place in one site's record. */
  readonly attributes: readonly {
    readonly location: number;
    readonly type: ScatterField['type'];
    readonly offset: number;
  }[];
  /** Bytes of one site's record, and of one invocation's records. */
  readonly siteBytes: number;
  readonly recordBytes: number;
  /** The records of the last capture, kept for the next dispatch. */
  records: { buffer: WebGLBuffer; bytes: number } | undefined;
  vertexArray: WebGLVertexArrayObject | undefined;
}

const FIELD_BYTES: Record<ScatterField['type'], number> = {
  uint: 4,
  ivec3: 12,
  uvec4: 16,
  ivec4: 16,
  vec4: 16,
};

const CAPTURE_FRAGMENT = `#version 300 es
precision highp float;
void main() {}
`;

const captures = new WeakMap<Ctx, Map<string, Program | string>>();

/** Links a capture stage, recording `varyings` interleaved; returns why it failed otherwise. */
function linkCapture(
  ctx: Ctx,
  vertex: string,
  varyings: readonly string[],
  label: string,
): Program | string {
  let cache = captures.get(ctx);
  if (!cache) {
    cache = new Map();
    captures.set(ctx, cache);
  }
  const key = `${vertex}\u0000${varyings.join(',')}`;
  const cached = cache.get(key);
  if (cached) return cached;
  const { gl } = ctx;
  const shaders = [
    [gl.VERTEX_SHADER, vertex],
    [gl.FRAGMENT_SHADER, CAPTURE_FRAGMENT],
  ].map(([type, source]) => {
    const shader = gl.createShader(type as number) as WebGLShader;
    gl.shaderSource(shader, source as string);
    gl.compileShader(shader);
    return shader;
  });
  const program = gl.createProgram();
  for (const shader of shaders) gl.attachShader(program, shader);
  gl.transformFeedbackVaryings(program, varyings as string[], gl.INTERLEAVED_ATTRIBS);
  gl.linkProgram(program);
  let linked: Program | string;
  if (gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const uniforms = new Map<string, WebGLUniformLocation>();
    const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS) as number;
    for (let i = 0; i < count; i++) {
      const active = gl.getActiveUniform(program, i);
      const location = active && gl.getUniformLocation(program, active.name);
      if (!active || !location) continue;
      uniforms.set(active.name, location);
      if (active.name.endsWith('[0]')) uniforms.set(active.name.slice(0, -3), location);
    }
    ctx.track(() => gl.deleteProgram(program));
    linked = { program, uniforms };
  } else {
    linked = `The capture stage of ${label} failed to link: ${gl.getShaderInfoLog(shaders[0] as WebGLShader) ?? ''} ${gl.getProgramInfoLog(program) ?? ''}`;
    gl.deleteProgram(program);
  }
  for (const shader of shaders) gl.deleteShader(shader);
  cache.set(key, linked);
  return linked;
}

const feedbacks = new WeakMap<Ctx, WebGLTransformFeedback>();

function feedbackOf(ctx: Ctx): WebGLTransformFeedback {
  let feedback = feedbacks.get(ctx);
  if (!feedback) {
    const made = ctx.gl.createTransformFeedback();
    feedback = made;
    feedbacks.set(ctx, made);
    ctx.track(() => ctx.gl.deleteTransformFeedback(made));
  }
  return feedback;
}

/** Prepares a pass of points: links its capture stage and lays out its records. */
function preparePoints(
  ctx: Ctx,
  label: string,
  module: ComputeModule,
  scatter: Scatter,
  draw: Program,
): Points | string {
  const { gl } = ctx;
  const varyings: string[] = [];
  for (let site = 0; site < scatter.sites; site++) {
    for (const field of scatter.fields) varyings.push(`kit_c${site}_${field.name}`);
  }
  const components =
    scatter.sites * scatter.fields.reduce((sum, f) => sum + FIELD_BYTES[f.type] / 4, 0);
  const limit = gl.getParameter(gl.MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS) as number;
  if (components > limit) {
    return `its writes need ${components} transform feedback components; WebGL 2 here records ${limit}`;
  }
  const capture = linkCapture(ctx, scatter.capture, varyings, `'${label}'`);
  if (typeof capture === 'string') return capture;
  const captureUses = connectUses(ctx, label, module, capture);
  if (typeof captureUses === 'string') return captureUses;
  let offset = 0;
  const attributes = scatter.fields.map((field) => {
    const at = {
      location: gl.getAttribLocation(draw.program, `kit_a_${field.name}`),
      type: field.type,
      offset,
    };
    offset += FIELD_BYTES[field.type];
    return at;
  });
  return {
    sites: scatter.sites,
    words: scatter.words,
    lanes: scatter.lanes,
    capture,
    captureUses,
    invocations: capture.uniforms.get(INVOCATIONS) ?? null,
    workgroups: capture.uniforms.get(WORKGROUPS) ?? null,
    attributes,
    siteBytes: offset,
    recordBytes: offset * scatter.sites,
    records: undefined,
    vertexArray: undefined,
  };
}

export interface GpuCompute {
  readonly module: ComputeModule;
  readonly passes: readonly GpuPass[];
}

/** Checks the GPU form against the CPU module and links its passes, or says why it cannot run. */
export function prepareGpu(
  ctx: Ctx,
  label: string,
  cpu: CpuModule,
  module: ComputeModule,
): GpuCompute | string {
  for (const binding of module.bindings) {
    const twin = cpu.bindings.find(
      (other) => other.group === binding.group && other.binding === binding.binding,
    );
    const kinds =
      binding.kind === 'texture' ? ['texture', 'storageTexture'] : [binding.kind as string];
    if (!twin || !kinds.includes(twin.kind)) {
      return `@group(${binding.group}) @binding(${binding.binding}) differs from the CPU module's bindings`;
    }
    if (binding.writes && !twin.writes) {
      return `the CPU module does not write target @group(${binding.group}) @binding(${binding.binding})`;
    }
  }
  const passes: GpuPass[] = [];
  for (const pass of module.passes) {
    const vertex = pass.scatter?.vertex ?? PASS_VERTEX;
    const linked = link(ctx, vertex, pass.fragment, `a pass of '${label}'`);
    if (!linked.ok) return linked.message;
    const { program } = linked;
    const uses = connectUses(ctx, label, module, program);
    if (typeof uses === 'string') return uses;
    const points = pass.scatter && preparePoints(ctx, label, module, pass.scatter, program);
    if (typeof points === 'string') return points;
    passes.push({
      target: pass.target,
      program,
      uses,
      invocations: program.uniforms.get(INVOCATIONS) ?? null,
      workgroups: program.uniforms.get(WORKGROUPS) ?? null,
      layer: program.uniforms.get(LAYER) ?? null,
      scatter: points,
      word: program.uniforms.get('kit_part') ?? null,
      phase: program.uniforms.get('kit_phase') ?? null,
    });
  }
  return { module, passes };
}

const NO_BLEND: BlendState = {
  enabled: false,
  colorOp: 0,
  colorSrc: 0,
  colorDst: 0,
  alphaOp: 0,
  alphaSrc: 0,
  alphaDst: 0,
};

const pools = new WeakMap<Ctx, Map<string, WebGLTexture[]>>();
/**
 * Textures a pass may render into are kept for the next dispatch, up to this many per size, or
 * one per layer of the largest layered target of that size.
 */
const POOLED = 4;

function poolOf(ctx: Ctx): Map<string, WebGLTexture[]> {
  let pool = pools.get(ctx);
  if (!pool) {
    const made = new Map<string, WebGLTexture[]>();
    pool = made;
    pools.set(ctx, made);
    ctx.track(() => {
      for (const list of made.values()) for (const texture of list) ctx.gl.deleteTexture(texture);
      made.clear();
    });
  }
  return pool;
}

function take(ctx: Ctx, internal: number, width: number, height: number): WebGLTexture {
  const reused = poolOf(ctx).get(`${internal}:${width}:${height}`)?.pop();
  if (reused) return reused;
  const { gl, state } = ctx;
  const texture = gl.createTexture();
  state.scratchTexture(gl.TEXTURE_2D, texture);
  gl.texStorage2D(gl.TEXTURE_2D, 1, internal, width, height);
  return texture;
}

function give(
  ctx: Ctx,
  internal: number,
  width: number,
  height: number,
  texture: WebGLTexture,
  keep = POOLED,
) {
  const key = `${internal}:${width}:${height}`;
  const pool = poolOf(ctx);
  const list = pool.get(key) ?? [];
  pool.set(key, list);
  if (list.length < Math.max(keep, POOLED)) list.push(texture);
  else ctx.gl.deleteTexture(texture);
}

/** A layered texture of a target's size, for a pass to render every layer of, from the pool. */
function takeLayered(
  ctx: Ctx,
  target: number,
  internal: number,
  size: readonly [number, number, number],
): WebGLTexture {
  const reused = poolOf(ctx)
    .get(`${target}:${internal}:${size.join(':')}`)
    ?.pop();
  if (reused) return reused;
  const { gl, state } = ctx;
  const texture = gl.createTexture();
  state.scratchTexture(target, texture);
  gl.texStorage3D(target, 1, internal, ...size);
  return texture;
}

function giveLayered(
  ctx: Ctx,
  target: number,
  internal: number,
  size: readonly [number, number, number],
  texture: WebGLTexture,
) {
  const key = `${target}:${internal}:${size.join(':')}`;
  const pool = poolOf(ctx);
  const list = pool.get(key) ?? [];
  pool.set(key, list);
  if (list.length < POOLED) list.push(texture);
  else ctx.gl.deleteTexture(texture);
}

/**
 * Whether a pass may render a view's whole texture into new storage that then replaces the old:
 * a single level, every layer, stored as the view's dimension.
 */
function replaceable(gl: WebGL2RenderingContext, view: ViewImpl): boolean {
  const { texture } = view;
  const layered = view.dimension === '3d' ? gl.TEXTURE_3D : gl.TEXTURE_2D_ARRAY;
  return (
    texture.mipLevelCount === 1 &&
    texture.sampleCount === 1 &&
    texture.canvas === undefined &&
    texture.target === layered &&
    (view.dimension === '3d' ||
      (view.baseArrayLayer === 0 && view.arrayLayerCount === texture.depthOrArrayLayers))
  );
}

/** Copies one level and layer of a texture into a 2d texture of its size, on the scratch unit. */
function copyInto(
  ctx: Ctx,
  source: { texture: WebGLTexture; target: number; level: number; layer: number },
  into: WebGLTexture,
  width: number,
  height: number,
): void {
  const { gl, state, scratch } = ctx;
  state.readFramebuffer(scratch.read);
  if (source.target === gl.TEXTURE_2D) {
    gl.framebufferTexture2D(
      gl.READ_FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      source.texture,
      source.level,
    );
  } else {
    gl.framebufferTextureLayer(
      gl.READ_FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      source.texture,
      source.level,
      source.layer,
    );
  }
  state.readBuffer(gl.COLOR_ATTACHMENT0);
  state.scratchTexture(gl.TEXTURE_2D, into);
  gl.copyTexSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 0, 0, width, height);
  gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0);
}

type Resolved =
  | { readonly kind: 'buffer'; readonly buffer: BufferImpl }
  | { readonly kind: 'texture'; readonly view: ViewImpl };

type Memory =
  | {
      readonly kind: 'buffer';
      readonly buffer: BufferImpl;
      readonly from: number;
      readonly to: number;
    }
  | { readonly kind: 'texture'; readonly view: ViewImpl };

/** The memory a binding of the dispatch holds: a buffer's byte range or a view's subresources. */
function memoryAt(
  groups: readonly (Bound | undefined)[],
  group: number,
  binding: number,
): Memory | undefined {
  const buffer = bufferEntry(groups[group], binding);
  if (buffer) {
    const from = buffer.offset;
    return { kind: 'buffer', buffer: buffer.buffer, from, to: from + buffer.size };
  }
  const entry = groups[group]?.group.entries.get(binding);
  return entry?.kind === 'texture' ? { kind: 'texture', view: entry.view } : undefined;
}

function overlaps(a: Memory, b: Memory): boolean {
  if (a.kind === 'buffer') {
    return b.kind === 'buffer' && a.buffer === b.buffer && a.from < b.to && b.from < a.to;
  }
  if (b.kind === 'buffer') return false;
  const [x, y] = [a.view, b.view];
  if (x.texture !== y.texture) return false;
  const levels =
    x.baseMipLevel < y.baseMipLevel + y.mipLevelCount &&
    y.baseMipLevel < x.baseMipLevel + x.mipLevelCount;
  const layers =
    x.texture.dimension === '3d' ||
    (x.baseArrayLayer < y.baseArrayLayer + y.arrayLayerCount &&
      y.baseArrayLayer < x.baseArrayLayer + x.arrayLayerCount);
  return levels && layers;
}

/**
 * Runs the dispatch as fragment passes. Returns why it cannot, in which case nothing ran and the
 * caller runs the CPU module.
 */
export function runGpu(
  ctx: Ctx,
  gpu: GpuCompute,
  groups: readonly (Bound | undefined)[],
  x: number,
  y: number,
  z: number,
): string | undefined {
  const { gl, state } = ctx;
  const { module } = gpu;
  const [wx, wy, wz] = module.workgroupSize;
  const invocations = [x * wx, y * wy, z * wz] as const;
  if (module.grid < 2 && invocations[1] !== 1) return 'the dispatch spans y';
  if (module.grid < 3 && invocations[2] !== 1) return 'the dispatch spans z';
  if (module.single && invocations[0] * invocations[1] * invocations[2] !== 1) {
    return 'the form needs a dispatch of one invocation';
  }
  for (const { target } of gpu.passes) {
    if (target.kind !== 'buffer') continue;
    if (target.width !== undefined && invocations[0] > target.width) {
      return `the dispatch is wider than ${target.width} invocations`;
    }
    if (target.height !== undefined && invocations[1] > target.height) {
      return `the dispatch is taller than ${target.height} invocations`;
    }
  }

  const resolved: Resolved[] = [];
  const written = new Set<BufferImpl>();
  for (const { target } of gpu.passes) {
    if (target.kind === 'buffer') {
      const entry = bufferEntry(groups[target.group], target.binding);
      if (!entry) return `no buffer at @group(${target.group}) @binding(${target.binding})`;
      // A buffer target's pass replaces the whole buffer.
      if (written.has(entry.buffer)) return 'two targets share their memory';
      written.add(entry.buffer);
      resolved.push({ kind: 'buffer', buffer: entry.buffer });
    } else {
      const entry = groups[target.group]?.group.entries.get(target.binding);
      if (entry?.kind !== 'texture') {
        return `no texture at @group(${target.group}) @binding(${target.binding})`;
      }
      const { texture } = entry.view;
      if (!texture.info.renderable || texture.sampleCount > 1) {
        return `'${texture.format}' is not color-renderable here`;
      }
      resolved.push({ kind: 'texture', view: entry.view });
    }
  }
  for (const { target } of gpu.passes) {
    const memory = memoryAt(groups, target.group, target.binding) as Memory;
    for (const other of module.bindings) {
      if (other.group === target.group && other.binding === target.binding) continue;
      const elsewhere = memoryAt(groups, other.group, other.binding);
      if (elsewhere && overlaps(memory, elsewhere)) return 'a target is bound twice';
    }
  }

  const outputs: (() => void)[] = [];
  for (const [i, pass] of gpu.passes.entries()) {
    const resource = resolved[i]!;
    state.program(pass.program.program);
    const missing = bindUses(ctx, pass.uses, pass.program, groups);
    if (missing) {
      ctx.errors.report('validation', `The dispatch is missing ${missing}.`);
      return undefined;
    }
    if (pass.invocations) gl.uniform3ui(pass.invocations, ...invocations);
    if (pass.workgroups) gl.uniform3ui(pass.workgroups, x, y, z);

    let width: number;
    let height: number;
    let layers = 1;
    let internal: number;
    if (resource.kind === 'buffer') {
      const data = dataTexture(ctx, resource.buffer);
      width = data.width;
      height = data.height;
      internal = gl.RGBA32UI;
    } else {
      const { view } = resource;
      const [w, h, depth] = mipSize(onGl(ctx, view.texture), view.baseMipLevel);
      width = w;
      height = h;
      if (view.dimension === '2d-array') layers = view.arrayLayerCount;
      if (view.dimension === '3d') layers = depth;
      internal = view.texture.info.internal;
    }

    if (pass.scatter) {
      const points = pass.scatter;
      const { sites, words, lanes } = points;
      const buffer = resource.kind === 'buffer';
      const count = invocations[0] * invocations[1] * invocations[2];
      const made: WebGLTexture[] = [];
      // The kernel runs once per invocation, into records every draw below reads
      const bytes = Math.max(count * points.recordBytes, 4);
      if (!points.records || points.records.bytes < bytes) {
        if (points.records) gl.deleteBuffer(points.records.buffer);
        const created = gl.createBuffer();
        gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, created);
        gl.bufferData(gl.TRANSFORM_FEEDBACK_BUFFER, bytes, gl.DYNAMIC_COPY);
        gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, null);
        points.records = { buffer: created, bytes };
        ctx.track(() => gl.deleteBuffer(created));
      }
      const records = points.records.buffer;
      if (count > 0) {
        state.program(points.capture.program);
        const lacking = bindUses(ctx, points.captureUses, points.capture, groups);
        if (lacking) {
          ctx.errors.report('validation', `The dispatch is missing ${lacking}.`);
          return undefined;
        }
        if (points.invocations) gl.uniform3ui(points.invocations, ...invocations);
        if (points.workgroups) gl.uniform3ui(points.workgroups, x, y, z);
        state.vertexArray(null);
        // A draw needs a complete framebuffer even with the rasterizer discarded
        state.drawFramebuffer(null);
        gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, feedbackOf(ctx));
        gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, records);
        state.enable(gl.RASTERIZER_DISCARD, true);
        gl.beginTransformFeedback(gl.POINTS);
        gl.drawArrays(gl.POINTS, 0, count);
        gl.endTransformFeedback();
        state.enable(gl.RASTERIZER_DISCARD, false);
        gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
        gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
        state.program(pass.program.program);
        bindUses(ctx, pass.uses, pass.program, groups);
      }
      points.vertexArray ??= gl.createVertexArray();
      const vertexArray = points.vertexArray;
      const pointAt = (site: number) => {
        state.vertexArray(vertexArray);
        state.arrayBuffer(records);
        for (const { location, type, offset } of points.attributes) {
          if (location < 0) continue;
          gl.enableVertexAttribArray(location);
          const at = site * points.siteBytes + offset;
          const size = FIELD_BYTES[type] / 4;
          if (type === 'vec4') {
            gl.vertexAttribPointer(location, size, gl.FLOAT, false, points.recordBytes, at);
          } else {
            const kind = type === 'uint' || type === 'uvec4' ? gl.UNSIGNED_INT : gl.INT;
            gl.vertexAttribIPointer(location, size, kind, points.recordBytes, at);
          }
        }
      };
      for (let layer = 0; layer < layers; layer++) {
        const texture = take(ctx, internal, width, height);
        // Cells no point covers keep their old contents
        if (resource.kind === 'buffer') {
          const data = dataTexture(ctx, resource.buffer);
          copyInto(
            ctx,
            { texture: data.texture, target: gl.TEXTURE_2D, level: 0, layer: 0 },
            texture,
            width,
            height,
          );
        } else {
          const { view: from } = resource;
          const stored = onGl(ctx, from.texture);
          copyInto(
            ctx,
            {
              texture: stored.texture as WebGLTexture,
              target: stored.target,
              level: from.baseMipLevel,
              layer: from.dimension === '3d' ? layer : from.baseArrayLayer + layer,
            },
            texture,
            width,
            height,
          );
        }
        made.push(texture);
        state.drawFramebuffer(ctx.scratch.draw);
        gl.framebufferTexture2D(
          gl.DRAW_FRAMEBUFFER,
          gl.COLOR_ATTACHMENT0,
          gl.TEXTURE_2D,
          texture,
          0,
        );
        state.drawBuffers([gl.COLOR_ATTACHMENT0]);
        state.program(pass.program.program);
        setInt(gl, pass.layer, layer);
        state.viewport(0, 0, width, height);
        writeEverything(ctx);
        for (const cap of [
          gl.SCISSOR_TEST,
          gl.DEPTH_TEST,
          gl.STENCIL_TEST,
          gl.CULL_FACE,
          gl.RASTERIZER_DISCARD,
          gl.SAMPLE_ALPHA_TO_COVERAGE,
          gl.POLYGON_OFFSET_FILL,
        ]) {
          state.enable(cap, false);
        }
        state.blend(0, NO_BLEND);
        const drawn = count > 0 ? sites : 0;
        for (let site = 0; site < drawn; site++) {
          pointAt(site);
          // A data texel holds four words; each draw writes the channels its points carry: one
          // word of an element, or a whole element where elements fit their texels
          const whole = buffer && lanes > 1;
          for (let word = 0; word < (buffer && !whole ? words : 1); word++) {
            setInt(gl, pass.word, word);
            for (let phase = 0; phase < (buffer ? 4 / lanes : 1); phase++) {
              setInt(gl, pass.phase, phase);
              const mask = whole ? ((1 << words) - 1) << (phase * lanes) : 1 << phase;
              state.colorMask(0, buffer ? mask : 0xf);
              gl.drawArrays(gl.POINTS, 0, count);
            }
          }
        }
        gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0);
      }
      state.vertexArray(null);
      if (resource.kind === 'buffer') {
        const size = [internal, width, height] as const;
        outputs.push(() => give(ctx, ...size, replaceData(resource.buffer, made[0]!)));
      } else {
        const { view: stored } = resource;
        outputs.push(() => {
          const texture = stored.texture;
          made.forEach((value, layer) => {
            const source: TextureImpl = {
              ...texture,
              target: gl.TEXTURE_2D,
              texture: value,
              renderbuffer: null,
            };
            blitLevel(
              ctx,
              { texture: source, level: 0, layer: 0, x: 0, y: 0 },
              {
                texture,
                level: stored.baseMipLevel,
                layer: stored.dimension === '3d' ? layer : stored.baseArrayLayer + layer,
                x: 0,
                y: 0,
              },
              width,
              height,
            );
            give(ctx, internal, width, height, value, made.length);
          });
          glWritten(texture);
        });
      }
      continue;
    }

    // A layered target's pass writes several layers a draw; a whole single-level texture is
    // rendered into new layered storage that replaces it, and anything else layer by layer
    const view = resource.kind === 'texture' ? resource.view : undefined;
    const layered = view?.dimension === '2d-array' || view?.dimension === '3d';
    const perDraw = layered ? LAYERS_PER_DRAW : 1;
    const whole = layered && replaceable(gl, view as ViewImpl);
    const shape = [width, height, layers] as const;
    const storage = whole
      ? takeLayered(ctx, (view as ViewImpl).texture.target, internal, shape)
      : undefined;
    const made: WebGLTexture[] = [];
    for (let first = 0; first < layers; first += perDraw) {
      state.drawFramebuffer(ctx.scratch.draw);
      const buffers: number[] = [];
      for (let slot = 0; slot < perDraw; slot++) {
        const point = gl.COLOR_ATTACHMENT0 + slot;
        const layer = first + slot;
        if (layer >= layers) {
          gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, point, gl.TEXTURE_2D, null, 0);
          buffers.push(gl.NONE);
        } else if (storage) {
          gl.framebufferTextureLayer(gl.DRAW_FRAMEBUFFER, point, storage, 0, layer);
          buffers.push(point);
        } else {
          const texture = take(ctx, internal, width, height);
          gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, point, gl.TEXTURE_2D, texture, 0);
          buffers.push(point);
          made.push(texture);
        }
      }
      state.drawBuffers(buffers);
      state.program(pass.program.program);
      setInt(gl, pass.layer, first);
      state.vertexArray(null);
      state.viewport(0, 0, width, height);
      writeEverything(ctx);
      for (const cap of [
        gl.SCISSOR_TEST,
        gl.DEPTH_TEST,
        gl.STENCIL_TEST,
        gl.CULL_FACE,
        gl.RASTERIZER_DISCARD,
        gl.SAMPLE_ALPHA_TO_COVERAGE,
        gl.POLYGON_OFFSET_FILL,
      ]) {
        state.enable(cap, false);
      }
      // Through `blend`: disabling BLEND as a cap turns off every draw buffer's blending behind
      // the per-buffer cache that render passes rely on
      for (let slot = 0; slot < perDraw; slot++) state.blend(slot, NO_BLEND);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      for (let slot = 0; slot < perDraw; slot++) {
        gl.framebufferTexture2D(
          gl.DRAW_FRAMEBUFFER,
          gl.COLOR_ATTACHMENT0 + slot,
          gl.TEXTURE_2D,
          null,
          0,
        );
      }
    }

    if (resource.kind === 'buffer') {
      const size = [internal, width, height] as const;
      outputs.push(() => give(ctx, ...size, replaceData(resource.buffer, made[0]!)));
    } else if (storage) {
      const { texture } = resource.view;
      const target = texture.target;
      outputs.push(() =>
        giveLayered(ctx, target, internal, shape, replaceStorage(ctx, texture, storage)),
      );
    } else {
      const { view: stored } = resource;
      outputs.push(() => {
        const texture = stored.texture;
        made.forEach((value, layer) => {
          const source: TextureImpl = {
            ...texture,
            target: gl.TEXTURE_2D,
            texture: value,
            renderbuffer: null,
          };
          blitLevel(
            ctx,
            { texture: source, level: 0, layer: 0, x: 0, y: 0 },
            {
              texture,
              level: stored.baseMipLevel,
              layer: stored.dimension === '3d' ? layer : stored.baseArrayLayer + layer,
              x: 0,
              y: 0,
            },
            width,
            height,
          );
          give(ctx, internal, width, height, value, made.length);
        });
        glWritten(texture);
      });
    }
  }
  for (const output of outputs) output();
  return undefined;
}
