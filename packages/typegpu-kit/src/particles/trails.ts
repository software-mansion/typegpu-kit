import { encoderOf, type Frame } from '../frames.ts';
import { byStage, type Inject } from '../inject.ts';
import * as lines from '../render/lines.ts';
import {
  d,
  std,
  tgpu,
  type StorageFlag,
  type TgpuBuffer,
  type TgpuCommandEncoder,
  type TgpuRoot,
  type TgpuUniform,
} from 'typegpu';

/** This schema describes a trail point passed to `width` and `shade`. */
export const Sample = d.struct({
  /** Holds a point on the trail centre line in world space. */
  position: d.vec3f,
  /** Identifies the trail from 0 to `capacity - 1`. */
  trail: d.u32,
  /**
   * Measures distance along stored points from 0 at the newest to 1 at the oldest point of a
   * full trail, `length - 1` points behind.
   */
  along: d.f32,
});

/** This type describes a trail point in shaders. */
export type Sample = d.InferGPU<typeof Sample>;

/** This schema holds the next point slot and the number of stored points in a trail ring. */
export const Ring = d.struct({ next: d.u32, count: d.u32 });

export const Settings = d.struct({
  /** Sets linear colour and opacity for the default `shade` function. */
  color: d.vec4f,
  /**
   * Sets width at the newest point in world units for the default `width` function. It tapers to
   * zero at the oldest point.
   */
  width: d.f32,
});

/** These defaults set white colour and a width of 0.1 world units. */
export const defaults: d.Infer<typeof Settings> = {
  color: d.vec4f(1, 1, 1, 1),
  width: 0.1,
};

/**
 * This type accepts a join or cap from `@typegpu/kit/geometry`, or a custom function with the same
 * signature.
 */
export type Join = lines.Join;

export interface Options {
  /** Sets the number of trails to an integer of at least 1. */
  readonly capacity: number;
  /**
   * Sets the number of points per trail to an integer of at least 2. Pushing beyond this count
   * drops the oldest point.
   */
  readonly length: number;
  /** Sets the drawing pipeline label. Default `trails`. */
  readonly label?: string;
  /**
   * Selects `add` for light, `over` for smoke or ink, `translucent` for the passes' translucency
   * targets, or a custom blend state. Default `add`.
   */
  readonly blend?: Exclude<lines.Blend, 'opaque'>;
  /**
   * Returns the trail width in world units at a point. Default `settings.width` at the newest
   * point, tapering with `along` to 0 at the oldest point of a full trail.
   * @gpu
   */
  readonly width?: (point: Sample) => number;
  /**
   * Returns premultiplied fragment colour. `across` ranges from -1 to 1 across the width. The
   * drawing multiplies it by how much of the pixel the trail covers. Default `settings.color`,
   * with soft edges and a fade towards the oldest point.
   * @gpu
   */
  readonly shade?: (point: Sample, across: number) => d.v4f;
  /**
   * Returns where a trail's newest end is drawn, in place of its newest point, such as where a
   * particle system draws the particle that pushes it: `sparks.drawnAt`. The trail then stays on an object drawn
   * between the steps that push it. Default the newest point.
   * @gpu
   */
  readonly head?: (trail: number) => d.v3f;
  /** Selects the join at a point, such as `lines.joins.miter`. Default round. @gpu */
  readonly join?: Join;
  /** Selects the cap at the newest end, such as `lines.caps.arrow`. Default round. @gpu */
  readonly startCap?: Join;
  /** Selects the cap at the oldest end. Default round. @gpu */
  readonly endCap?: Join;
  /**
   * Sets initial colour and width for the default callbacks. Default `defaults`, with white
   * colour and width 0.1.
   */
  readonly settings?: Partial<d.InferInput<typeof Settings>>;
}

export interface Trails extends Disposable {
  /** Reports the number of trails, from the `capacity` option. */
  readonly capacity: number;
  /** Reports the points each trail holds, from the `length` option. */
  readonly length: number;
  /** Fills the trail accessors with this set's buffers, as their defaults do. */
  inject(): Inject;
  /**
   * Pushes a point to the newest end from a shader. Once the trail holds `length` points, it drops the oldest. Only one thread may push a given
   * trail per dispatch; concurrent pushes race.
   */
  readonly push: (trail: number, position: d.v3f) => void;
  /**
   * Empties a trail in a shader, as `write(trail, [])` does on the CPU. Use it before pushing a respawned object at its new position. It counts as
   * a push for dispatch concurrency; a later push in the same thread follows it.
   */
  readonly reset: (trail: number) => void;
  /** Returns point `k`, counting back from the newest. `k` must be below `held(trail)`. */
  readonly pointAt: (trail: number, k: number) => d.v3f;
  /** Returns the stored point count, up to `length`. */
  readonly held: (trail: number) => number;
  /**
   * Holds settings for the default `width` and `shade` functions. Use `patch` to upload only the
   * fields you supply.
   */
  readonly settings: TgpuUniform<typeof Settings>;
  /** Holds each trail ring in `length` slots starting at `trail * length`. */
  readonly points: TgpuBuffer<d.WgslArray<d.Vec3f>> & StorageFlag;
  /** Holds the next slot and point count for each trail. */
  readonly rings: TgpuBuffer<d.WgslArray<typeof Ring>> & StorageFlag;
  /**
   * Writes x, y and z coordinates from oldest to newest, keeping only the newest `length`
   * points. Use `write(trail, [])` to empty a trail. Queue writes run before frame commands, so
   * kernel pushes in that frame follow these points.
   */
  write(trail: number, positions: ArrayLike<number>): void;
  /**
   * Draws trails with at least two points, newest first, as `lines` when added to a pass
   * description. It tests depth without writing it.
   */
  readonly drawing: lines.Lines['drawing'];
  /** Empties every trail, recorded into `target`, a frame or an encoder. */
  clear(target: Frame | TgpuCommandEncoder): void;
  /** Destroys the point and ring buffers and the lines that draw them. */
  destroy(): void;
}

const checked = (name: string, value: number, least: number) => {
  if (!Number.isInteger(value) || value < least) {
    throw new RangeError(
      `trails.create requires \`${name}\` to be an integer of at least ${least}. Received ${value}.`,
    );
  }
};

/**
 * Creates trails as rings of recent points, pushed by kernels or written on the CPU, and draws
 * them as `lines`. Throws unless `capacity` is an integer of at least 1 and `length` is an integer
 * of at least 2.
 *
 * @injects member `push`, `reset`, `pointAt` and `held`, which read its own buffers in any
 * pipeline
 * @binds 2 storage: `push` or `pointAt`, the points and the rings
 * @binds 1 storage: `held` or `reset` alone, the rings
 * @binds 2 storage, 2 uniform: `drawing`, the points and the rings, the settings and
 * `passes.viewport`
 * @owns 3 buffers, a lines: the points, the rings and the settings, and the lines that draw them
 * @records `drawing`, added to passes, after the passes that push; `clear` whenever you like.
 * @param options.width a `'use gpu'` function; it and `shade` carry the fills of what they read,
 * such as `tgpu.fn(swell).pipe(noise.inject())`. The ribbons face the camera of the pass
 * description they are drawn in, which `width` and `shade` read too
 * @returns a set of trails you destroy
 */
export function create(root: TgpuRoot, options: Options): Trails {
  const { capacity, length, blend = 'add' } = options;
  checked('capacity', capacity, 1);
  checked('length', length, 2);

  const segments = length - 1;

  const pointsBuffer = root.createBuffer(d.arrayOf(d.vec3f, capacity * length)).$usage('storage');
  const ringsBuffer = root.createBuffer(d.arrayOf(Ring, capacity)).$usage('storage');
  const settings = root.createUniform(Settings, { ...defaults, ...options.settings });

  // Per-set accessors filled with its own buffers, so one kernel can push to several trail sets
  const pointRows = byStage(pointsBuffer);
  const ringRows = byStage(ringsBuffer);
  const points = tgpu.accessor(d.arrayOf(d.vec3f), pointRows);
  const rings = tgpu.accessor(d.arrayOf(Ring), ringRows);
  const inject: Inject = (cfg) => cfg.with(points, pointRows).with(rings, ringRows);

  const held = (trail: number) => {
    'use gpu';
    return rings.$[trail].count;
  };

  const pointAt = (trail: number, k: number) => {
    'use gpu';
    const next = rings.$[trail].next;
    return d.vec3f(points.$[d.u32(trail) * length + ((next + length - 1 - d.u32(k)) % length)]);
  };

  const push = (trail: number, position: d.v3f) => {
    'use gpu';
    if (trail >= capacity) return;

    const ring = Ring(rings.$[trail]);
    points.$[d.u32(trail) * length + ring.next] = d.vec3f(position);
    rings.$[trail] = Ring({
      next: (ring.next + 1) % length,
      count: std.min(ring.count + 1, d.u32(length)),
    });
  };

  const reset = (trail: number) => {
    'use gpu';
    if (trail >= capacity) return;
    rings.$[trail] = Ring();
  };

  const taper = (point: Sample) => {
    'use gpu';
    return settings.$.width * (1 - point.along);
  };

  const glow = (point: Sample, across: number) => {
    'use gpu';
    const alpha = settings.$.color.w * (1 - across * across) * (1 - point.along);
    return d.vec4f(settings.$.color.xyz * alpha, alpha);
  };

  const width = options.width ?? taper;
  const shade = options.shade ?? glow;

  const { head } = options;
  const sampleAt = (trail: number, k: number) => {
    'use gpu';
    let position = pointAt(trail, k);
    if (head !== undefined && k === 0) position = head(trail);
    return Sample({ position, trail: d.u32(trail), along: d.f32(k) / segments });
  };

  const pointOf = (trail: number, k: number) => {
    'use gpu';
    const sample = sampleAt(trail, k);
    return lines.ControlPoint({
      position: sample.position,
      width: width(sample),
      color: d.vec4f(1),
      along: sample.along,
    });
  };

  const shown = (fragment: lines.Fragment) => {
    'use gpu';
    const point = Sample({
      position: fragment.position,
      trail: fragment.line,
      along: fragment.along,
    });
    return shade(point, fragment.across) * fragment.coverage;
  };

  const { join, startCap, endCap } = options;
  const paths = lines.create(root, {
    capacity,
    length,
    point: tgpu.fn(pointOf).pipe(inject),
    count: tgpu.fn(held).pipe(inject),
    shade: tgpu.fn(shown).pipe(inject),
    blend,
    label: options.label ?? 'trails',
    ...(join && { join }),
    ...(startCap && { startCap }),
    ...(endCap && { endCap }),
  });

  const queue = root.device.queue;
  const pointsGpu = root.unwrap(pointsBuffer);
  const ringsGpu = root.unwrap(ringsBuffer);
  const staging = new Float32Array(length * 4);
  const ring = new Uint32Array(2);

  const destroy = () => {
    pointsBuffer.destroy();
    ringsBuffer.destroy();
    paths.destroy();
  };

  return {
    capacity,
    length,
    inject: () => inject,
    push,
    reset,
    pointAt,
    held,
    settings,
    points: pointsBuffer,
    rings: ringsBuffer,

    drawing: paths.drawing,

    write(trail, positions) {
      if (!Number.isInteger(trail) || trail < 0 || trail >= capacity) {
        throw new RangeError(`The trails have no trail ${trail}; there are ${capacity}.`);
      }

      const given = Math.floor(positions.length / 3);
      const kept = Math.min(given, length);
      const first = given - kept;

      for (let i = 0; i < kept; i++) {
        const from = (first + i) * 3;
        staging[i * 4] = positions[from];
        staging[i * 4 + 1] = positions[from + 1];
        staging[i * 4 + 2] = positions[from + 2];
      }

      if (kept > 0) queue.writeBuffer(pointsGpu, trail * length * 16, staging.buffer, 0, kept * 16);

      ring[0] = kept % length;
      ring[1] = kept;
      queue.writeBuffer(ringsGpu, trail * 8, ring.buffer, 0, 8);
    },

    clear: (target) => ringsBuffer.clear(encoderOf(target)),

    destroy,
    [Symbol.dispose]: destroy,
  };
}
