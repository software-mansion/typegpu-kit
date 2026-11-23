import { d, std, tgpu, type TgpuComputePass, type TgpuRoot, type TgpuUniform } from 'typegpu';
import * as cameras from './cameras.ts';
import { checkDispatch } from './dispatchLimit.ts';
import type { Inject } from './inject.ts';
import * as pipelines from './pipelines.ts';
import { create as createReadback, type Readback } from './readback.ts';
import type { SizeSource } from './targets.ts';

/** Marks a pick that found no candidate, as returned by `found`. */
export const none = 0xffffffff;

export const Settings = d.struct({
  /** Extends the candidate edge by this many screen points. */
  reach: d.f32,
});

/** Sets the default picker reach to 24 screen points for touch input. */
export const defaults: d.Infer<typeof Settings> = { reach: 24 };

/** Describes a candidate within reach for the `rank` function. */
export const Candidate = d.struct({
  /** Identifies the candidate by its index from `0` to `capacity - 1`. */
  index: d.u32,
  /**
   * Measures the screen distance from the pointer to the candidate center as a fraction of the
   * screen diagonal.
   */
  gap: d.f32,
  /**
   * Gives the center depth, from 0 at the near plane to 1 at the far plane, in equal steps of
   * distance along the viewing direction. A rank keeps about 2^(32 - ceil(log2(capacity))) levels
   * of it, the bits of a `u32` that the index leaves, so a very large capacity ranks depth
   * coarsely.
   */
  depth: d.f32,
  /** Gives the candidate center in world space, as `position` returned it. */
  position: d.vec3f,
  /** Gives the ray through the pointer in world space, as `cameras.rayOf` computes it. */
  ray: cameras.Ray,
});
export type Candidate = d.Infer<typeof Candidate>;

/** Ranks candidates by the screen distance from their center to the pointer, nearest first. */
export const byPointer = (candidate: Candidate) => {
  'use gpu';
  return candidate.gap;
};

/** Ranks candidates by depth, nearest the eye first. */
export const byEye = (candidate: Candidate) => {
  'use gpu';
  return candidate.depth;
};

export interface Options {
  /**
   * Sets the number of candidate slots, including dead slots. Indices range from `0` to `capacity
   * - 1`. Must be an integer from 1 to 2^24.
   */
  readonly capacity: number;
  /**
   * Supplies the view size at each pick, such as the canvas surface. `pixelRatio` scales reach from
   * screen points to pixels. Without it, one point equals one pixel.
   */
  readonly size: SizeSource;
  /** Returns the candidate center in world space. @gpu */
  readonly position: (i: number) => d.v3f;
  /** Returns the candidate radius in world units. Default 0, a point. @gpu */
  readonly radius?: (i: number) => number;
  /** Tests whether a candidate can be picked. Default every candidate can be picked. @gpu */
  readonly alive?: (i: number) => boolean;
  /**
   * Returns the highest live slot plus one, read when `pick` records, such as a table's from
   * `live()`. Skips the slots above it. Default `capacity`.
   * @cpu
   */
  readonly span?: () => number;
  /**
   * Ranks candidates within reach, from 0 to 1. Keeps the lowest rank, then the lowest index among
   * ties. A negative rank turns the candidate down, so a rank can be the distance at which
   * `candidate.ray` meets the candidate's shape, negative for a miss. Can read a uniform to switch
   * rankings, such as between `byPointer` and `byEye`, without recreating the picker. Default
   * `byPointer`.
   * @gpu
   */
  readonly rank?: (candidate: Candidate) => number;
  /**
   * Reads results this many times per second for hovering, which picks every frame. Without a
   * rate, captures copy the result in each frame that picked, for tap input. Default none.
   */
  readonly hz?: number;
  /** Sets the initial reach. Change it with `settings.patch`. Default `defaults`, 24 points. */
  readonly settings?: Partial<d.InferInput<typeof Settings>>;
}

export interface Picker extends Disposable {
  /** Reports the number of candidate slots, from the `capacity` option. */
  readonly capacity: number;
  /** Fills `found` with this picker's key, as its default does. */
  inject(): Inject;
  /**
   * Returns the candidate index from the last pick, or `none`, in any shader. Available in later
   * passes of the same frame.
   */
  readonly found: () => number;
  /**
   * Returns the ray through the point of the last pick in world space, as `cameras.rayOf` computes
   * it, in a shader of a pipeline piped with the camera. `clear` keeps it.
   */
  readonly ray: () => d.Infer<typeof cameras.Ray>;
  /**
   * Reads the picked candidate index to the CPU, or null if nothing was found. Capture after the
   * picker pass, then call `take()` or await `next()`. Without `hz`, each pick requests a copy for the
   * next capture.
   */
  readonly result: Readback<number | null>;
  /** Holds the reach setting. `patch` uploads only the fields you give it. */
  readonly settings: TgpuUniform<typeof Settings>;
  /** Compiles the pick pipeline. Calling it is optional. */
  initAsync(): Promise<void>;
  /**
   * Picks at `x`, `y` from 0 to 1 across the view, from the top-left corner, as `view.rayAt` and
   * `pointing` take them. Uses the camera given to `create`. Records one pick per frame: a second
   * call into the same pass replaces the point and records no second dispatch.
   */
  pick(pass: TgpuComputePass, x: number, y: number): void;
  /**
   * Forgets the last pick without a dispatch, such as when the pointer leaves: `found` returns
   * `none` and the next capture reads null, until the next `pick`. A pick recorded in the same
   * frame still runs.
   */
  clear(): void;
  /** Destroys its key buffer and readback. */
  destroy(): void;
}

const WORKGROUP = 64;

const Pointer = d.struct({
  /** Gives pointer coordinates from 0 to 1 across the view, from the top-left corner. */
  at: d.vec2f,
  /** Gives the view size in pixels. */
  size: d.vec2f,
  /** Gives the number of pixels per screen point. */
  pixelRatio: d.f32,
});

/**
 * Finds a candidate under a screen point on the GPU. Projects every candidate with the camera and
 * keeps the one within reach with the best `rank`. Throws unless capacity is an integer from 1 to
 * 2^24 whose threads, 64 to a workgroup, fit the device's `maxComputeWorkgroupsPerDimension`.
 *
 * @injects member `found` and `ray`, for shaders that act on the pick in the same frame; `ray`
 * also reads `cameras.current`
 * @binds 1 storage: `found`, the pick's key
 * @binds 1 uniform: `ray`, the pick's point
 * @owns 3 buffers, a pipeline, a readback: the pick's key, the pointer and the settings; the
 * readback brings the key to the CPU
 * @records `pick` once in a frame that picks, in a compute pass after whatever moves the
 * candidates; `result.capture(frame)` after that pass, which copies what a pick found.
 * @param camera the camera the candidates are drawn with
 * @param options.position where each candidate is; it, `radius`, `alive` and `rank` carry the fills
 * of what they read, such as `Star.position.pipe(stars.inject())`
 * @returns a picker you destroy
 */
export function create(root: TgpuRoot, camera: cameras.CameraBase, options: Options): Picker {
  const { capacity, size, position, radius, alive, span, rank = byPointer, hz } = options;

  if (!Number.isInteger(capacity) || capacity < 1 || capacity > 2 ** 24) {
    throw new RangeError(
      `picker.create: \`capacity\` takes a whole number from 1 to 2^24, got ${capacity}.`,
    );
  }
  checkDispatch(root, 'picker.create', 'capacity', capacity, WORKGROUP);

  // Store rank above the candidate index so the smallest key wins, with index breaking ties. Limit
  // rank to 24 bits for exact f32 values and exclude all ones so no candidate key equals `none`
  const indexBits = Math.max(1, Math.ceil(Math.log2(capacity)));
  const rankBits = Math.min(24, 32 - indexBits);
  const highest = 2 ** rankBits - 2;
  const indexMask = 2 ** indexBits - 1;

  const keyBuffer = root.createBuffer(d.u32, none).$usage('storage');
  const keyAtomic = root.createBuffer(d.atomic(d.u32), root.unwrap(keyBuffer)).$usage('storage');
  const pointer = root.createUniform(Pointer);
  const settings = root.createUniform(Settings, { ...defaults, ...options.settings });

  // An accessor per picker, filled with its own key, so one shader can read several pickers
  const keyRows = keyBuffer.as('readonly');
  const key = tgpu.accessor(d.u32, keyRows);
  const inject: Inject = (cfg) => cfg.with(key, keyRows);

  const found = () => {
    'use gpu';
    const k = key.$;
    return std.select(k & indexMask, d.u32(none), k === none);
  };

  const ray = () => {
    'use gpu';
    const at = pointer.$.at;
    return cameras.rayOf(cameras.current.$, d.vec2f(at.x * 2 - 1, 1 - at.y * 2));
  };

  const keys = keyAtomic.as('mutable');

  const pickKernel = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (i >= capacity) return;
    if (alive !== undefined && !alive(i)) return;

    const view = cameras.current.$;
    const center = position(i);
    const clip = view.viewProjection * d.vec4f(center, 1);
    if (clip.w <= 0) return;

    const ratio = pointer.$.pixelRatio;
    const pixels = pointer.$.size;
    const at = pointer.$.at * pixels;
    const ndc = clip.xy / clip.w;
    const gap = std.distance(d.vec2f(ndc.x + 1, 1 - ndc.y) * 0.5 * pixels, at);
    const across =
      radius === undefined
        ? d.f32(0)
        : ((radius(i) * cameras.lensOf(view).y) / clip.w) * 0.5 * pixels.y;
    if (gap > settings.$.reach * ratio + across) return;

    // Clip depth crowds towards the far plane and loses its low bits there, where the levels of a
    // rank would tie, so depth comes from the distance along the viewing direction
    const near = cameras.distanceOf(view, d.f32(1 - cameras.farDepth()));
    const far = cameras.distanceOf(view, d.f32(cameras.farDepth()));
    const along = std.dot(center - view.position, cameras.forwardOf(view));
    const ranked = rank(
      Candidate({
        index: i,
        gap: gap / std.length(pixels),
        depth: std.saturate((along - near) / (far - near)),
        position: center,
        ray: ray(),
      }),
    );
    if (ranked < 0) return;
    std.atomicMin(keys.$, (d.u32(std.min(ranked, 1) * highest) << d.u32(indexBits)) | i);
  });

  const pickPipeline = root.pipe(camera.inject()).createComputePipeline({ compute: pickKernel });

  const result = createReadback(root, keyBuffer, {
    ...(hz === undefined ? {} : { hz }),
    decode: (k): number | null => (k === none ? null : k & indexMask),
  });
  const destroy = () => {
    keyBuffer.destroy();
    result.destroy();
  };

  const queue = root.device.queue;
  const keyGpu = root.unwrap(keyBuffer);
  const pointerGpu = root.unwrap(pointer.buffer);
  const reset = Uint32Array.of(none);
  const written = new Float32Array(d.sizeOf(Pointer) / 4);
  // The pass holding the pending pick. Queue writes land before the submit, so a second pick into
  // it moves the point of the one dispatch
  let picking: TgpuComputePass | undefined;

  return {
    capacity,
    inject: () => inject,
    found,
    ray,
    result,
    settings,

    initAsync: () => pipelines.compile([pickPipeline]),

    pick(pass, x, y) {
      written[0] = x;
      written[1] = y;
      written[2] = size.width;
      written[3] = size.height;
      written[4] = size.pixelRatio ?? 1;
      queue.writeBuffer(pointerGpu, 0, written);
      queue.writeBuffer(keyGpu, 0, reset);
      if (pass === picking) return;
      picking = pass;
      pass.setPipeline(pickPipeline);
      pass.dispatchWorkgroups(Math.ceil((span ? span() : capacity) / WORKGROUP));

      // Tap picks request one copy. Hover picks use the readback rate
      if (hz === undefined) result.request();
    },

    clear() {
      queue.writeBuffer(keyGpu, 0, reset);
      result.request();
    },

    destroy,
    [Symbol.dispose]: destroy,
  };
}
