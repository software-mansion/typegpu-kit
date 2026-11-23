/**
 * Compiles pipelines behind a loading screen, lists a pipeline's bindings per stage, and provides
 * blend states.
 *
 * @module
 */

import {
  isPipeline,
  tgpu,
  type ShaderStage,
  type TgpuComputePipeline,
  type TgpuLayoutEntry,
  type TgpuRenderPipeline,
} from 'typegpu';
import {
  declaredNames,
  type limitNames,
  nouns,
  overflowing,
  stagesIn,
  weight,
} from './bindingCounts.ts';
import type { Owner } from './owner.ts';

const ADD: GPUBlendComponent = Object.freeze({
  srcFactor: 'one',
  dstFactor: 'one',
  operation: 'add',
});
const OVER: GPUBlendComponent = Object.freeze({
  srcFactor: 'one',
  dstFactor: 'one-minus-src-alpha',
  operation: 'add',
});
const STRAIGHT: GPUBlendComponent = Object.freeze({
  srcFactor: 'src-alpha',
  dstFactor: 'one-minus-src-alpha',
  operation: 'add',
});

/** Provides blend states for pipeline `targets`. */
export const blend = Object.freeze({
  /** Adds fragment values to the target, for sparks, glow or bloom. */
  add: Object.freeze({ color: ADD, alpha: ADD }),
  /** Composites premultiplied colour over the target. The fragment returns `vec4f(rgb * a, a)`. */
  over: Object.freeze({ color: OVER, alpha: OVER }),
  /**
   * Composites straight colour over the target, keeping its alpha premultiplied. The fragment
   * returns `vec4f(rgb, a)`.
   */
  straight: Object.freeze({ color: STRAIGHT, alpha: OVER }),
}) satisfies Record<string, GPUBlendState>;

/** Compiles a TypeGPU pipeline or a module's own pipelines. */
export interface Warmable {
  initAsync(): Promise<unknown>;
}

export interface CompileOptions {
  /**
   * Receives the completed fraction whenever an item finishes compiling. Once it throws, it is not
   * called again, and `compile` rejects with that error after every item finishes. Default none.
   * @cpu
   */
  readonly onProgress?: (fraction: number) => void;
  /**
   * Limits time spent starting compilations at a stretch, in milliseconds; starting an item
   * resolves its WGSL on the calling thread. `compile` then yields to the event loop. Default 8.
   */
  readonly budget?: number;
}

function tally(
  items: readonly Warmable[],
  onProgress: CompileOptions['onProgress'],
  done: () => void,
) {
  const errors: unknown[] = [];
  // Stop calling an observer that threw, and rethrow its first error at the end
  let observer = onProgress;
  let observed: { readonly error: unknown } | undefined;
  let finished = 0;

  const finish = () => {
    finished += 1;
    try {
      observer?.(finished / items.length);
    } catch (error) {
      observer = undefined;
      observed = { error };
    }
    if (finished === items.length) done();
  };

  const fail = (error: unknown) => {
    errors.push(error);
    finish();
  };

  return {
    start(item: Warmable) {
      try {
        item.initAsync().then(finish, (error: unknown) => fail(explained(item, error)));
      } catch (error) {
        fail(error); // resolution errors can surface synchronously
      }
    },

    check() {
      if (observed) {
        if (errors.length === 0) throw observed.error;
        throw new AggregateError(
          [...errors, observed.error],
          'Pipeline compilation or progress reporting failed.',
        );
      }
      if (errors.length === 1) throw errors[0];
      if (errors.length > 1) {
        const reasons = new Set(errors.map((error) => String((error as Error)?.message ?? error)));
        throw new AggregateError(
          errors,
          `${errors.length} of ${items.length} items failed to compile: ${[...reasons].join('; ')}`,
        );
      }
    },
  };
}

/**
 * Starts compiling every item the list holds at the call, yielding to the event loop past the
 * time budget so the page's other work, such as a loading animation, keeps running. Use before the
 * frame loop starts. Waits for every item to finish, then rejects if any failed.
 *
 * @param from a list of pipelines and modules that compile their own, or an owner that took them
 * @returns a promise that resolves once every item has compiled
 */
export async function compile(
  from: readonly Warmable[] | Owner,
  options: CompileOptions = {},
): Promise<void> {
  const { onProgress, budget: ms = 8 } = options;
  // Snapshot the list, so an item added after the call can't keep it waiting
  const items = [...('warmables' in from ? from.warmables : from)];
  if (items.length === 0) return;
  let counted!: ReturnType<typeof tally>;
  const finished = new Promise<void>((resolve) => {
    counted = tally(items, onProgress, resolve);
  });
  let since = performance.now();
  for (const [i, item] of items.entries()) {
    counted.start(item);
    if (i < items.length - 1 && performance.now() - since >= ms) {
      await new Promise((resume) => setTimeout(resume));
      since = performance.now();
    }
  }
  await finished;
  counted.check();
}

/** Names a shader stage that counts bindings against its own limits. */
export type { ShaderStage };

/**
 * Names a kind of binding. Each counts against its own per-stage limit, except an external
 * texture, which counts as 4 textures, a sampler and a uniform buffer.
 */
export type BindingKind =
  | 'uniform'
  | 'storage'
  | 'texture'
  | 'storageTexture'
  | 'sampler'
  | 'externalTexture';

/** Describes one entry of a bind group layout a pipeline binds. */
export interface Binding {
  readonly group: number;
  readonly binding: number;
  /**
   * Names the entry: its WGSL variable where the pipeline's code declares it, otherwise its key in
   * the layout. TypeGPU names a fixed resource, such as a module's uniform, after its label.
   */
  readonly name: string;
  readonly kind: BindingKind;
  /** Lists the pipeline's stages that count the binding, whether or not their code reads it. */
  readonly stages: readonly ShaderStage[];
}

/** Reports a stage that binds more of one kind than its limit. */
export interface Overflow {
  readonly stage: ShaderStage;
  readonly kind: Exclude<BindingKind, 'externalTexture'>;
  readonly count: number;
  readonly limit: number;
}

export interface BindingsOptions {
  /**
   * Sets the per-stage limits to compare with, such as `root.device.limits`. Default the minimums
   * WebGPU guarantees: 8 storage buffers, 12 uniform buffers, 16 textures, 16 samplers and 4
   * storage textures.
   */
  readonly limits?: Partial<Record<(typeof limitNames)[Overflow['kind']], number>>;
}

/** Lists a pipeline's bindings and the stages over their limits. */
export interface Bindings {
  /** Names the pipeline by its label. */
  readonly label: string;
  /**
   * Lists every entry of every bind group layout the pipeline binds, in group and binding order.
   */
  readonly bindings: readonly Binding[];
  /** Lists each stage and kind over its limit. Empty when the pipeline fits. */
  readonly over: readonly Overflow[];
}

const STAGE_BITS = { vertex: 1, fragment: 2, compute: 4 } as const;

/** Reads a TypeGPU layout entry's kind and the stages that see it, by TypeGPU's defaults. */
function entryOf(entry: TgpuLayoutEntry): [BindingKind, number] | undefined {
  const kind: BindingKind | undefined =
    'uniform' in entry
      ? 'uniform'
      : 'storage' in entry
        ? 'storage'
        : 'storageTexture' in entry
          ? 'storageTexture'
          : 'texture' in entry
            ? 'texture'
            : 'sampler' in entry
              ? 'sampler'
              : 'externalTexture' in entry
                ? 'externalTexture'
                : undefined;
  if (!kind) return undefined;
  const writable =
    ('storage' in entry && entry.access === 'mutable') ||
    ('storageTexture' in entry && entry.storageTexture.access !== 'read-only');
  const visibility =
    entry.visibility ?? (writable ? ['compute', 'fragment'] : ['compute', 'vertex', 'fragment']);
  let bits = 0;
  for (const stage of visibility) bits |= STAGE_BITS[stage];
  return [kind, bits];
}

/**
 * Lists every entry of every bind group layout a pipeline binds, with the stages that count each
 * one, and the stages over their limits. WebGPU counts an entry in each stage its layout makes it
 * visible to, whether or not the stage's code reads it, so a shared layout's unread entries count
 * too. Use it to find what fills a stage; `compile` includes it in the error of a pipeline that
 * fails over the guaranteed limits.
 *
 * @param pipeline a TypeGPU render or compute pipeline
 * @returns its bindings and the stages over their limits
 */
export function bindingsOf(
  pipeline: TgpuRenderPipeline<never> | TgpuComputePipeline,
  options: BindingsOptions = {},
): Bindings {
  const { code, usedBindGroupLayouts } = tgpu.resolveWithContext([pipeline]);
  const pipelineStages: ShaderStage[] =
    pipeline.resourceType === 'compute-pipeline'
      ? ['compute']
      : /@fragment\b/.test(code)
        ? ['vertex', 'fragment']
        : ['vertex'];
  const names = declaredNames(code);
  const bindings: Binding[] = [];
  for (const [group, layout] of usedBindGroupLayouts.entries()) {
    if (!layout) continue;
    // A layout's binding numbers count its null entries too
    for (const [binding, [key, entry]] of Object.entries(layout.entries).entries()) {
      const found = entry && entryOf(entry);
      if (!found) continue;
      bindings.push({
        group,
        binding,
        name: names.get(`${group}:${binding}`) ?? key,
        kind: found[0],
        stages: stagesIn(found[1], pipelineStages),
      });
    }
  }
  return {
    label: String(pipeline).replace(/^\w+:/, ''),
    bindings,
    over: overflowing(bindings, pipelineStages, options.limits),
  };
}

/** Lists what fills each overflowing stage, for a failure message. */
function describeOverflow({ label, bindings, over }: Bindings): string {
  const lines = [
    `The pipeline '${label}' binds more than WebGPU guarantees per stage:`,
    ...over.map(
      ({ stage, kind, count, limit }) => `- ${stage}: ${count} ${nouns[kind]}, over ${limit}`,
    ),
  ];
  for (const kind of new Set(over.map((o) => o.kind))) {
    lines.push(`Its ${nouns[kind]}:`);
    for (const b of bindings.filter((each) => weight(each, kind) > 0)) {
      lines.push(`  ${b.name} (group ${b.group}, binding ${b.binding}; ${b.stages.join(', ')})`);
    }
  }
  lines.push(
    'A binding counts in every stage its layout makes it visible to. To make room, fill a hook on the renderers that read it instead of the whole lighting, pack uniforms into one struct, or keep values in a buffer the pipeline already binds.',
  );
  return lines.join('\n');
}

/** Names the bindings of a pipeline that failed over its stages' limits. */
function explained(item: Warmable, error: unknown): unknown {
  if (!isPipeline(item)) return error;
  let report: Bindings;
  try {
    report = bindingsOf(item);
  } catch {
    return error;
  }
  if (report.over.length === 0) return error;
  const reason = error instanceof Error ? error.message : String(error);
  return new Error(`${describeOverflow(report)}\nThe device reported: ${reason}`, {
    cause: error,
  });
}
