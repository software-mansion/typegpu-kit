/** Render and compute pipelines: a linked program with its fixed-function state, or a CPU kernel. */

import type { CpuModule } from '../cpu/types.ts';
import { PASS_VERTEX } from '../glsl/computeModule.ts';
import { BASE_VERTEX, FIRST_INSTANCE } from '../glsl/renderModule.ts';
import { makeLayout, makePipelineLayout, pipelineLayoutGroups } from './binding.ts';
import { type Ctx, IMPL, SHADER_STAGE } from './context.ts';
import { formatInfo, vertexFormatInfo } from './formats.ts';
import { type Linked, link, linkAsync, type Program } from './program.ts';
import { compareFunc } from './sampler.ts';
import { type GpuCompute, prepareGpu } from './gpuCompute.ts';
import { moduleImpl, type Stages } from './shader.ts';
import type { BlendState, StencilFace } from './state.ts';
import {
  type BlockUse,
  type BufferUse,
  connectUses,
  type LevelsUse,
  type TextureUse,
} from './uses.ts';

export type { BlockUse, BufferUse, LevelsUse, TextureUse } from './uses.ts';

export interface Attribute {
  readonly location: number;
  readonly size: number;
  readonly type: number;
  readonly normalized: boolean;
  readonly integer: boolean;
  readonly offset: number;
}

export interface VertexSlot {
  readonly slot: number;
  readonly stride: number;
  readonly instance: boolean;
  readonly attributes: readonly Attribute[];
  /** The GL buffer and byte offset the attributes point at now, in the pipeline's VAO. */
  buffer: WebGLBuffer | null;
  base: number;
}

export interface RenderState {
  readonly mode: number;
  readonly cull: number;
  readonly frontFace: number;
  readonly targets: readonly { readonly blend: BlendState; readonly mask: number }[];
  readonly depthTest: boolean;
  readonly depthFunc: number;
  readonly depthWrite: boolean;
  readonly stencilTest: boolean;
  readonly stencilFront: StencilFace;
  readonly stencilBack: StencilFace;
  readonly stencilRead: number;
  readonly stencilWrite: number;
  readonly bias: { readonly constant: number; readonly slope: number } | undefined;
  readonly alphaToCoverage: boolean;
}

export interface RenderPipelineImpl {
  readonly label: string;
  /** Undefined while the program links or after it failed. */
  program: Program | undefined;
  error: string | undefined;
  buffers: readonly BufferUse[];
  textures: readonly TextureUse[];
  blocks: readonly BlockUse[];
  levels: readonly LevelsUse[];
  firstInstance: WebGLUniformLocation | null;
  baseVertex: WebGLUniformLocation | null;
  readonly vao: WebGLVertexArrayObject;
  readonly slots: readonly VertexSlot[];
  readonly state: RenderState;
}

export interface ComputePipelineImpl {
  readonly label: string;
  readonly module: CpuModule | undefined;
  readonly error: string | undefined;
  /** The kernel as fragment passes, when its module carries a GPU form that links. */
  readonly gpu: GpuCompute | undefined;
}

export function renderPipelineImpl(pipeline: GPURenderPipeline): RenderPipelineImpl {
  return (pipeline as unknown as { [IMPL]: RenderPipelineImpl })[IMPL];
}

export function computePipelineImpl(pipeline: GPUComputePipeline): ComputePipelineImpl {
  return (pipeline as unknown as { [IMPL]: ComputePipelineImpl })[IMPL];
}

function blendFactor(
  gl: WebGL2RenderingContext,
  factor: GPUBlendFactor | undefined,
  fallback: number,
): number {
  switch (factor) {
    case undefined:
      return fallback;
    case 'zero':
      return gl.ZERO;
    case 'one':
      return gl.ONE;
    case 'src':
      return gl.SRC_COLOR;
    case 'one-minus-src':
      return gl.ONE_MINUS_SRC_COLOR;
    case 'src-alpha':
      return gl.SRC_ALPHA;
    case 'one-minus-src-alpha':
      return gl.ONE_MINUS_SRC_ALPHA;
    case 'dst':
      return gl.DST_COLOR;
    case 'one-minus-dst':
      return gl.ONE_MINUS_DST_COLOR;
    case 'dst-alpha':
      return gl.DST_ALPHA;
    case 'one-minus-dst-alpha':
      return gl.ONE_MINUS_DST_ALPHA;
    case 'src-alpha-saturated':
      return gl.SRC_ALPHA_SATURATE;
    case 'constant':
      return gl.CONSTANT_COLOR;
    case 'one-minus-constant':
      return gl.ONE_MINUS_CONSTANT_COLOR;
    default:
      throw new Error(`@typegpu/kit-gl: blend factor '${factor}' is not supported on WebGL 2.`);
  }
}

function blendOp(gl: WebGL2RenderingContext, op: GPUBlendOperation | undefined): number {
  switch (op) {
    case 'subtract':
      return gl.FUNC_SUBTRACT;
    case 'reverse-subtract':
      return gl.FUNC_REVERSE_SUBTRACT;
    case 'min':
      return gl.MIN;
    case 'max':
      return gl.MAX;
    default:
      return gl.FUNC_ADD;
  }
}

function stencilOp(gl: WebGL2RenderingContext, op: GPUStencilOperation | undefined): number {
  switch (op) {
    case 'zero':
      return gl.ZERO;
    case 'replace':
      return gl.REPLACE;
    case 'invert':
      return gl.INVERT;
    case 'increment-clamp':
      return gl.INCR;
    case 'decrement-clamp':
      return gl.DECR;
    case 'increment-wrap':
      return gl.INCR_WRAP;
    case 'decrement-wrap':
      return gl.DECR_WRAP;
    default:
      return gl.KEEP;
  }
}

function stencilFace(
  gl: WebGL2RenderingContext,
  face: GPUStencilFaceState | undefined,
): StencilFace {
  return {
    func: compareFunc(gl, face?.compare ?? 'always'),
    fail: stencilOp(gl, face?.failOp),
    depthFail: stencilOp(gl, face?.depthFailOp),
    pass: stencilOp(gl, face?.passOp),
  };
}

function topology(gl: WebGL2RenderingContext, value: GPUPrimitiveTopology | undefined): number {
  switch (value) {
    case 'point-list':
      return gl.POINTS;
    case 'line-list':
      return gl.LINES;
    case 'line-strip':
      return gl.LINE_STRIP;
    case 'triangle-strip':
      return gl.TRIANGLE_STRIP;
    default:
      return gl.TRIANGLES;
  }
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

function renderState(ctx: Ctx, descriptor: GPURenderPipelineDescriptor): RenderState {
  const { gl } = ctx;
  const primitive = descriptor.primitive ?? {};
  const depthStencil = descriptor.depthStencil;
  const targets = Array.from(descriptor.fragment?.targets ?? []).map((target) => {
    if (!target) return { blend: NO_BLEND, mask: 0 };
    const blend = target.blend;
    const info = formatInfo(ctx.formats, target.format);
    const mask = target.writeMask ?? 0xf;
    // A target that cannot be rendered to was reported, with its extension, when it was created
    if (blend && info.renderable && !info.blendable && info.kind === 'float') {
      // WebGPU rejects this too, without the `float32-blendable` feature this device then lacks
      ctx.errors.report(
        'validation',
        `The pipeline '${descriptor.label ?? ''}' blends into '${target.format}', which needs EXT_float_blend on WebGL 2; it draws without blending.`,
      );
    }
    if (!blend || !info.blendable) return { blend: NO_BLEND, mask };
    return {
      blend: {
        enabled: true,
        colorOp: blendOp(gl, blend.color.operation),
        colorSrc: blendFactor(gl, blend.color.srcFactor, gl.ONE),
        colorDst: blendFactor(gl, blend.color.dstFactor, gl.ZERO),
        alphaOp: blendOp(gl, blend.alpha.operation),
        alphaSrc: blendFactor(gl, blend.alpha.srcFactor, gl.ONE),
        alphaDst: blendFactor(gl, blend.alpha.dstFactor, gl.ZERO),
      },
      mask,
    };
  });
  // Without OES_draw_buffers_indexed, a draw leaves out the targets that write nothing, and the
  // others share one state
  const written = targets.filter((target) => target.mask !== 0);
  if (!ctx.ext.drawBuffersIndexed && written.length > 1) {
    const first = JSON.stringify(written[0]);
    if (written.some((target) => JSON.stringify(target) !== first)) {
      ctx.errors.report(
        'validation',
        `The pipeline '${descriptor.label ?? ''}' blends or masks its targets differently, which needs OES_draw_buffers_indexed; every target uses the first one's state.`,
      );
    }
  }
  const front = stencilFace(gl, depthStencil?.stencilFront);
  const back = stencilFace(gl, depthStencil?.stencilBack);
  const usesStencil = [front, back].some(
    (face) =>
      face.func !== gl.ALWAYS ||
      face.fail !== gl.KEEP ||
      face.depthFail !== gl.KEEP ||
      face.pass !== gl.KEEP,
  );
  const depthFunc = compareFunc(gl, depthStencil?.depthCompare ?? 'always');
  const depthWrite = depthStencil?.depthWriteEnabled ?? false;
  const constant = depthStencil?.depthBias ?? 0;
  const slope = depthStencil?.depthBiasSlopeScale ?? 0;
  return {
    mode: topology(gl, primitive.topology),
    cull: primitive.cullMode === 'front' ? gl.FRONT : primitive.cullMode === 'back' ? gl.BACK : 0,
    // Vertex shaders negate y, which mirrors every triangle's winding
    frontFace: primitive.frontFace === 'cw' ? gl.CCW : gl.CW,
    targets,
    depthTest: depthStencil !== undefined && (depthFunc !== gl.ALWAYS || depthWrite),
    depthFunc,
    depthWrite,
    stencilTest: depthStencil !== undefined && usesStencil,
    stencilFront: front,
    stencilBack: back,
    stencilRead: depthStencil?.stencilReadMask ?? 0xff,
    stencilWrite: depthStencil?.stencilWriteMask ?? 0xff,
    bias: constant !== 0 || slope !== 0 ? { constant, slope } : undefined,
    alphaToCoverage: descriptor.multisample?.alphaToCoverageEnabled ?? false,
  };
}

function slotsOf(
  ctx: Ctx,
  buffers: Iterable<GPUVertexBufferLayout | null | undefined>,
): VertexSlot[] {
  const slots: VertexSlot[] = [];
  let index = 0;
  for (const layout of buffers) {
    const slot = index++;
    if (!layout) continue;
    const attributes = Array.from(layout.attributes).map((attribute) => {
      const info = vertexFormatInfo(ctx.gl, attribute.format);
      return {
        location: attribute.shaderLocation,
        size: info.size,
        type: info.type,
        normalized: info.normalized,
        integer: info.integer,
        offset: attribute.offset,
      };
    });
    if (attributes.length === 0) continue;
    slots.push({
      slot,
      stride: layout.arrayStride,
      instance: layout.stepMode === 'instance',
      attributes,
      buffer: null,
      base: -1,
    });
  }
  return slots;
}

/** Combines the stages of the vertex and fragment modules, which may be one module or two. */
function stagesOf(descriptor: GPURenderPipelineDescriptor): Stages | string {
  const vertex = moduleImpl(descriptor.vertex.module);
  if (vertex.kind === 'error') return vertex.message;
  if (vertex.kind !== 'render') return 'A render pipeline needs a render module.';
  const fragmentModule = descriptor.fragment ? moduleImpl(descriptor.fragment.module) : undefined;
  if (fragmentModule?.kind === 'error') return fragmentModule.message;
  if (fragmentModule && fragmentModule.kind !== 'render') {
    return 'A render pipeline needs a render module.';
  }
  const fragment = fragmentModule?.stages;
  if (!vertex.stages.vertex) return 'The vertex module has no vertex stage.';
  if (descriptor.fragment && !fragment?.fragment)
    return 'The fragment module has no fragment stage.';
  const same = fragment === vertex.stages;
  return {
    vertex: vertex.stages.vertex,
    fragment: descriptor.fragment ? fragment!.fragment : undefined,
    buffers:
      same || !fragment ? vertex.stages.buffers : [...vertex.stages.buffers, ...fragment.buffers],
    textures:
      same || !fragment
        ? vertex.stages.textures
        : [...vertex.stages.textures, ...fragment.textures],
    levels:
      same || !fragment ? vertex.stages.levels : [...vertex.stages.levels, ...fragment.levels],
    blocks:
      same || !fragment ? vertex.stages.blocks : [...vertex.stages.blocks, ...fragment.blocks],
  };
}

/** Assigns texture units and finds uniforms once the program linked. */
function connect(ctx: Ctx, impl: RenderPipelineImpl, stages: Stages, program: Program): void {
  const uses = connectUses(ctx, `The pipeline '${impl.label}'`, stages, program);
  if (typeof uses === 'string') {
    impl.error = uses;
    return;
  }
  impl.buffers = uses.buffers;
  impl.textures = uses.textures;
  impl.blocks = uses.blocks;
  impl.levels = uses.levels;
  impl.firstInstance = program.uniforms.get(FIRST_INSTANCE) ?? null;
  impl.baseVertex = program.uniforms.get(BASE_VERTEX) ?? null;
  impl.program = program;
}

/** A bind group layout for `layout: 'auto'`, from the bindings the stages read. */
function autoLayouts(stages: Stages): GPUBindGroupLayout[] {
  const groups: GPUBindGroupLayoutEntry[][] = [];
  const visibility = SHADER_STAGE.VERTEX | SHADER_STAGE.FRAGMENT;
  const add = (group: number, entry: GPUBindGroupLayoutEntry) => {
    const entries = (groups[group] ??= []);
    if (!entries.some((other) => other.binding === entry.binding)) entries.push(entry);
  };
  for (const binding of stages.buffers) {
    add(binding.group, {
      binding: binding.binding,
      visibility,
      buffer: { type: binding.access === 'uniform' ? 'uniform' : 'read-only-storage' },
    });
  }
  for (const binding of stages.textures) {
    add(binding.group, {
      binding: binding.binding,
      visibility,
      texture: {
        sampleType:
          binding.sampleType === 'depth'
            ? 'depth'
            : binding.sampleType === 'f32'
              ? 'float'
              : binding.sampleType === 'i32'
                ? 'sint'
                : 'uint',
        viewDimension: binding.dimension,
        multisampled: binding.multisampled,
      },
    });
    if (binding.sampler) {
      add(binding.sampler.group, {
        binding: binding.sampler.binding,
        visibility,
        sampler: { type: binding.comparison ? 'comparison' : 'filtering' },
      });
    }
  }
  return Array.from(groups, (entries) => makeLayout({ entries: entries ?? [] }));
}

/** Frees a pipeline's vertex array, made outside its scope so it captures nothing else. */
const deleting = (gl: WebGL2RenderingContext, vao: WebGLVertexArrayObject) => () =>
  gl.deleteVertexArray(vao);

function makeRenderPipeline(
  ctx: Ctx,
  descriptor: GPURenderPipelineDescriptor,
  linking: (stages: Stages, label: string) => Linked | Promise<Linked>,
): { pipeline: GPURenderPipeline; ready: Promise<void> } {
  const { gl, state } = ctx;
  const label = descriptor.label ?? '';
  const vao = gl.createVertexArray()!;
  let stages: Stages | string;
  let renderStateValue: RenderState;
  let slots: VertexSlot[] = [];
  try {
    stages = stagesOf(descriptor);
    renderStateValue = renderState(ctx, descriptor);
    slots = slotsOf(ctx, descriptor.vertex.buffers ?? []);
  } catch (error) {
    stages = (error as Error).message;
    renderStateValue = renderState(ctx, { layout: descriptor.layout, vertex: descriptor.vertex });
  }
  const impl: RenderPipelineImpl = {
    label,
    program: undefined,
    error: typeof stages === 'string' ? stages : undefined,
    buffers: [],
    textures: [],
    blocks: [],
    levels: [],
    firstInstance: null,
    baseVertex: null,
    vao,
    slots,
    state: renderStateValue,
  };
  ctx.track(deleting(gl, vao), impl);

  state.vertexArray(vao);
  for (const slot of slots) {
    for (const attribute of slot.attributes) {
      gl.enableVertexAttribArray(attribute.location);
      // A zero stride repeats the first element, which GL can only do as a huge divisor
      const divisor = slot.stride === 0 ? 0x7fffffff : slot.instance ? 1 : 0;
      if (divisor) gl.vertexAttribDivisor(attribute.location, divisor);
    }
  }
  state.vertexArray(null);

  const settle = (linked: Linked) => {
    if (typeof stages === 'string') return;
    if (!linked.ok) {
      impl.error = linked.message;
    } else {
      connect(ctx, impl, stages, linked.program);
    }
    if (impl.error) ctx.errors.report('validation', impl.error);
  };

  let ready: Promise<void>;
  if (typeof stages === 'string') {
    ctx.errors.report('validation', stages);
    ready = Promise.resolve();
  } else {
    const linked = linking(stages, label ? `'${label}'` : 'a render pipeline');
    if (linked instanceof Promise) ready = linked.then(settle);
    else {
      settle(linked);
      ready = Promise.resolve();
    }
  }

  const explicit =
    descriptor.layout === 'auto' ? undefined : pipelineLayoutGroups(descriptor.layout);
  let auto: GPUBindGroupLayout[] | undefined;
  const pipeline = {
    [IMPL]: impl,
    __brand: 'GPURenderPipeline',
    label,
    getBindGroupLayout(index: number): GPUBindGroupLayout {
      if (explicit) {
        const layout = explicit[index];
        if (layout) return layout;
      }
      auto ??= typeof stages === 'string' ? [] : autoLayouts(stages);
      return auto[index] ?? makeLayout({ entries: [] });
    },
  } as unknown as GPURenderPipeline;
  return { pipeline, ready };
}

export function createRenderPipeline(
  ctx: Ctx,
  descriptor: GPURenderPipelineDescriptor,
): GPURenderPipeline {
  return makeRenderPipeline(ctx, descriptor, (stages, label) =>
    link(ctx, stages.vertex!, stages.fragment, label),
  ).pipeline;
}

export async function createRenderPipelineAsync(
  ctx: Ctx,
  descriptor: GPURenderPipelineDescriptor,
): Promise<GPURenderPipeline> {
  const { pipeline, ready } = makeRenderPipeline(ctx, descriptor, (stages, label) =>
    linkAsync(ctx, stages.vertex!, stages.fragment, label),
  );
  await ready;
  return pipeline;
}

/**
 * Starts compiling the programs of a compute module's GPU form without waiting for them; resolves
 * once the driver has linked them all.
 */
function compileGpu(ctx: Ctx, descriptor: GPUComputePipelineDescriptor): Promise<unknown> {
  const module = moduleImpl(descriptor.compute.module);
  if (module.kind !== 'compute' || !module.gpu || !ctx.gpuCompute) return Promise.resolve();
  const label = `a pass of '${descriptor.label ?? ''}'`;
  return Promise.all(
    module.gpu.passes.map((pass) =>
      linkAsync(ctx, pass.scatter?.vertex ?? PASS_VERTEX, pass.fragment, label),
    ),
  );
}

/** Links the programs of a compute module's GPU form in the background, then makes the pipeline. */
export async function createComputePipelineAsync(
  ctx: Ctx,
  descriptor: GPUComputePipelineDescriptor,
): Promise<GPUComputePipeline> {
  await compileGpu(ctx, descriptor);
  return createComputePipeline(ctx, descriptor, true);
}

/**
 * Makes a compute pipeline. Its GPU form's programs start compiling here and, unless `eager`, are
 * linked at the first dispatch, so a pipeline made and not yet used costs no wait.
 */
export function createComputePipeline(
  ctx: Ctx,
  descriptor: GPUComputePipelineDescriptor,
  eager = false,
): GPUComputePipeline {
  const label = descriptor.label ?? '';
  const module = moduleImpl(descriptor.compute.module);
  let error: string | undefined;
  let cpu: CpuModule | undefined;
  if (module.kind === 'compute') cpu = module.module;
  else if (module.kind === 'error') error = module.message;
  else error = `The compute pipeline '${label}' was given a render module.`;
  if (error) ctx.errors.report('validation', error);
  if (module.kind === 'compute' && module.gpuError) ctx.stats.fallbacks.set(label, module.gpuError);
  let impl: ComputePipelineImpl = { label, module: cpu, error, gpu: undefined };
  if (module.kind === 'compute' && module.gpu && ctx.gpuCompute) {
    const form = module.gpu;
    if (!eager) void compileGpu(ctx, descriptor);
    let gpu: GpuCompute | undefined;
    let unprepared = true;
    const prepared = () => {
      if (unprepared) {
        unprepared = false;
        const made = prepareGpu(ctx, label, module.module, form);
        if (typeof made === 'string') ctx.stats.fallbacks.set(label, made);
        else gpu = made;
      }
      return gpu;
    };
    if (eager) prepared();
    impl = {
      label,
      module: cpu,
      error,
      get gpu() {
        return prepared();
      },
    };
  }
  return computePipeline(descriptor, label, cpu, impl);
}

function computePipeline(
  descriptor: GPUComputePipelineDescriptor,
  label: string,
  cpu: CpuModule | undefined,
  impl: ComputePipelineImpl,
): GPUComputePipeline {
  const explicit =
    descriptor.layout === 'auto' ? undefined : pipelineLayoutGroups(descriptor.layout);
  return {
    [IMPL]: impl,
    __brand: 'GPUComputePipeline',
    label,
    getBindGroupLayout(index: number): GPUBindGroupLayout {
      const layout = explicit?.[index];
      if (layout) return layout;
      const entries: GPUBindGroupLayoutEntry[] = (cpu?.bindings ?? [])
        .filter((binding) => binding.group === index)
        .map(
          (binding) =>
            ({
              binding: binding.binding,
              visibility: SHADER_STAGE.COMPUTE,
              ...(binding.kind === 'buffer'
                ? { buffer: { type: binding.writes ? 'storage' : 'read-only-storage' } }
                : binding.kind === 'sampler'
                  ? { sampler: {} }
                  : binding.kind === 'storageTexture'
                    ? { storageTexture: { format: 'rgba8unorm' } }
                    : { texture: {} }),
            }) as GPUBindGroupLayoutEntry,
        );
      return makeLayout({ entries });
    },
  } as unknown as GPUComputePipeline;
}

export { makePipelineLayout };
