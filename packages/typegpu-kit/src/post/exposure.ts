import * as pipelines from '../pipelines.ts';
import type * as profiler from '../profiler.ts';
import { unfilled, type Inject } from '../inject.ts';
import { memo } from '../memo.ts';
import * as workgroup from '../workgroup.ts';
import { type SizeSource } from '../targets.ts';
import { d, std, tgpu, type TgpuCommandEncoder, type TgpuRoot, type TgpuUniform } from 'typegpu';
import { dtOf, encoderOf, type Frame } from '../frames.ts';
import { fitRegion, within, type FloatTexture, type Source } from './post.ts';
import { sourceOf } from './source.ts';
import { luminance } from './tone.ts';

export const Settings = d.struct({
  /**
   * Sets the average luminance the exposed picture aims for, as a geometric mean over the picture.
   * Must be above 0.
   */
  key: d.f32,
  /** Sets the lowest exposure, for the brightest pictures. Must be above 0. */
  lowest: d.f32,
  /** Sets the highest exposure, for the darkest pictures. Must be at least `lowest`. */
  highest: d.f32,
  /**
   * Sets how fast exposure rises after the picture darkens, per second. Each second closes
   * `1 - e^-rise` of the remaining stops.
   */
  rise: d.f32,
  /** Sets how fast exposure falls after the picture brightens, per second, as `rise` does. */
  fall: d.f32,
});

export const defaults: d.Infer<typeof Settings> = {
  key: 0.18,
  lowest: 1 / 16,
  highest: 16,
  rise: 1,
  fall: 3,
};

/**
 * Provides the adapted exposure in shaders created from `root.pipe(eye.inject())`, the multiplier
 * a final pass with `exposure: eye` applies before its tone curve.
 */
export const current = tgpu.accessor(
  d.f32,
  unfilled('`exposure.current`', 'eye', 'an `exposure.create`'),
);

export interface Options {
  /**
   * Provides the linear picture to measure, usually the final pass's source: a rendered picture,
   * such as `aa`, or a function returning a texture. `record` reads it on every call, so the
   * texture can be rebuilt.
   * @cpu
   */
  readonly source: Source;
  /**
   * Selects the source region from its top-left corner, such as `regions.hdr` when drawing below
   * full resolution. `record` reads it on every call. Default the region of a rendered `source`,
   * else the whole texture.
   */
  readonly region?: SizeSource;
  /**
   * Sets initial values over `defaults`. Default `defaults`, with key 0.18, lowest 1/16, highest
   * 16, rise 1 and fall 3.
   */
  readonly settings?: Partial<d.InferInput<typeof Settings>>;
}

export interface Exposure extends Disposable {
  /**
   * Holds `defaults` with overrides from `settings`. Use `patch` to upload only the fields you
   * supply.
   */
  readonly settings: TgpuUniform<typeof Settings>;
  /** Fills `exposure.current` with the adapted exposure. */
  inject(): Inject;
  /** Compiles the measuring pipeline. Calling it is optional. */
  initAsync(): Promise<void>;
  /**
   * Measures the source and moves the exposure towards its target by `dt` seconds of adaptation,
   * the frame's `dt` by default. Record it after whatever wrote the source and before the final
   * pass. The first `record`, the first after `reset`, and one given an encoder without `dt`
   * take the target at once. Optional timer writes measure the pass.
   */
  record(target: Frame | TgpuCommandEncoder, options?: RecordOptions): void;
  /** Takes the target at once at the next `record`, for example after a camera cut. */
  reset(): void;
  /** Destroys the exposure buffer. */
  destroy(): void;
}

export interface RecordOptions extends profiler.Timed {
  /** Sets the seconds of adaptation. Default the frame's `dt`, or the target at once. */
  readonly dt?: number;
}

/** Samples on a GRID × GRID lattice over the region, shared by one workgroup's threads. */
const GRID = 64;
const THREADS = 256;
/** Counts darker pixels as this luminance, so black areas do not drive exposure to infinity. */
const DARKEST = 1e-4;

const sourceLayout = tgpu.bindGroupLayout({ source: { texture: d.texture2d(d.f32) } });
const Step = d.struct({ region: d.vec2f, dt: d.f32 });

/**
 * Creates an exposure that adapts to the picture on the GPU. Each `record`
 * averages the log luminance of 4096 filtered samples over the source in one workgroup. It then
 * eases the exposure towards `key` divided by that average, within `lowest` and `highest`. The
 * exposure stays in a GPU buffer, and nothing is read back.
 *
 * @allocates on resize: a bind group for a new source
 * @owns 3 buffers, a sampler, a pipeline and a bind group: the settings, the frame's region and
 * time, the exposure, and a group for the source
 * @injects member `exposure.current`, for the final pass or shaders of your own
 * @binds 1 storage: `exposure.current`, read-only
 * @records `record` after whatever wrote the source, before the final pass that reads it
 * @param options.source the linear picture, read at every `record`
 * @param options.region the part of the source that holds the picture, read at every `record`
 * @returns an exposure you record every frame and destroy
 */
export function create(root: TgpuRoot, options: Options): Exposure {
  const source = sourceOf(options);
  const initial = { ...defaults, ...options.settings };
  if (!(initial.key > 0 && initial.lowest > 0 && initial.highest >= initial.lowest)) {
    throw new RangeError(
      `Exposure requires \`settings.key\` and \`settings.lowest\` above 0, and \`settings.highest\` at least as high as \`settings.lowest\`. Got key ${initial.key}, lowest ${initial.lowest} and highest ${initial.highest}.`,
    );
  }

  const settings = root.createUniform(Settings, initial);
  const pace = root.createUniform(Step);
  // 0 until the first record, which then takes the target at once
  const state = root.createBuffer(d.f32, 0).$usage('storage');
  const adapted = state.as('mutable');
  const readable = state.as('readonly');
  const sampler = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const summed = workgroup.reduce(workgroup.sum(d.f32), THREADS);

  const meter = tgpu.computeFn({
    workgroupSize: [THREADS],
    in: { lid: d.builtin.localInvocationIndex },
  })(({ lid }) => {
    'use gpu';
    const size = d.vec2f(std.textureDimensions(sourceLayout.$.source));
    const spacing = pace.$.region / size / GRID;

    // Each sample sits between texels, so linear filtering averages four of them. The last ones
    // stay on the centres of the region's edge texels, so filtering reads nothing beyond it
    let sum = d.f32(0);
    for (let i = d.u32(lid); i < GRID * GRID; i += THREADS) {
      const cell = d.vec2f(d.f32(i % GRID), d.f32(std.intdiv(i, d.u32(GRID))));
      const at = within((cell + 0.5) * spacing, pace.$.region, size);
      const color = std.textureSampleLevel(sourceLayout.$.source, sampler.$, at, 0).xyz;
      sum += std.log2(std.max(luminance(color), DARKEST));
    }
    const total = summed(sum, lid);

    if (lid === 0) {
      // Work in stops, so rising and falling ease evenly at any brightness
      const average = total / (GRID * GRID);
      const target = std.clamp(
        std.log2(settings.$.key) - average,
        std.log2(settings.$.lowest),
        std.log2(settings.$.highest),
      );

      let next = target;
      const was = adapted.$;
      if (was > 0) {
        const from = std.log2(was);
        const rate = std.select(settings.$.fall, settings.$.rise, target > from);
        next = from + (target - from) * (1 - std.exp(-pace.$.dt * rate));
      }
      adapted.$ = std.exp2(next);
    }
  });

  const pipeline = root.createComputePipeline({ compute: meter }).$name('exposure');
  const sourceGroup = memo((texture: FloatTexture) =>
    root.createBindGroup(sourceLayout, { source: texture }),
  );

  const stepped = { region: [0, 0] as [number, number], dt: 0 };
  const described: { timestampWrites?: profiler.Timed['timestampWrites'] } = {};

  const destroy = () => state.destroy();

  return {
    settings,

    inject: () => (cfg) => cfg.with(current, readable),

    initAsync: () => pipelines.compile([pipeline]),

    record(target, recording) {
      const texture = source.output;
      fitRegion(stepped.region, texture, source.region);
      stepped.dt = dtOf(target, recording?.dt);
      pace.write(stepped);

      described.timestampWrites = recording?.timestampWrites;
      const pass = encoderOf(target).beginComputePass(described);
      pass.setPipeline(pipeline);
      pass.setBindGroup(sourceGroup(texture));
      pass.dispatchWorkgroups(1);
      pass.end();
    },

    reset: () => state.write(0),

    destroy,
    [Symbol.dispose]: destroy,
  };
}
