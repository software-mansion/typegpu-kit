import * as backends from '../backends.ts';
import * as cameras from '../cameras.ts';
import { encoderOf, type Frame } from '../frames.ts';
import * as liveLists from '../liveLists.ts';
import * as history from '../history.ts';
import * as passes from '../passes.ts';
import * as pipelines from '../pipelines.ts';
import { byStage, type Inject } from '../inject.ts';
import { type CameraBase } from '../cameras.ts';
import { type Live } from '../table.ts';
import { randf } from '@typegpu/noise';
import {
  d,
  std,
  tgpu,
  type IndirectFlag,
  type StorageFlag,
  type TgpuBuffer,
  type TgpuCommandEncoder,
  type TgpuComputePass,
  type TgpuRoot,
  type TgpuUniform,
} from 'typegpu';

const fields = {
  position: d.vec3f,
  /** Holds particle age in seconds. */
  age: d.f32,
  velocity: d.vec3f,
  /** Sets lifetime in seconds. A particle is alive while `age < life`. */
  life: d.f32,
  /** Holds linear colour with opacity in `w`. */
  color: d.vec4f,
  /** Sets the particle width in world units. */
  size: d.f32,
  /** Holds a random value in [0, 1), fixed at birth, for variation in `update` and `shade`. */
  seed: d.f32,
};

type Fields = typeof fields;

export const Particle = d.struct(fields);

/** This type describes a particle in shaders. */
export type Particle = d.InferGPU<typeof Particle>;

/** This type accepts `Particle` or a particle struct with custom fields from `schema`. */
// oxlint-disable-next-line typescript/no-explicit-any -- matches a struct with any fields beside a particle's
export type Schema = d.WgslStruct<Fields & Record<string, any>>;

type Own<S extends Schema> = Omit<S['propTypes'], keyof Fields>;

/** A particle of the schema in shaders. */
type Of<S extends Schema> = d.InferGPU<S>;

/**
 * Defines a particle struct with custom fields after the built-in fields. Supply it as
 * `particle` to `create`; callbacks, `emit` and `particle(i)` then use this struct. Throws if a
 * custom field duplicates a built-in field.
 *
 * @param name a label for the struct in WGSL
 * @param own the fields of its own, as `d.struct` takes them
 * @returns the particle's struct, for `create` and for shaders that build one to `emit`
 */
export function schema<F extends Record<string, d.AnyWgslData>>(
  name: string,
  own: F & { readonly [K in keyof Fields]?: 'is a field every particle has' },
): d.WgslStruct<Fields & F> {
  for (const key of Object.keys(own)) {
    if (Object.hasOwn(fields, key)) {
      throw new Error(`'${key}' is a built-in particle field. Choose another name.`);
    }
  }
  return d.struct({ ...fields, ...(own as F) }).$name(name);
}

export const Settings = d.struct({
  /** Sets acceleration in world units per second squared. */
  gravity: d.vec3f,
  /** Sets velocity decay per second. The fraction remaining after one second is e^-drag. */
  drag: d.f32,
});

/**
 * These defaults set gravity to `[0, -9.8, 0]` metres per second squared and drag to 1. Drag
 * retains about 37% of speed after one second.
 */
export const defaults: d.Infer<typeof Settings> = {
  gravity: d.vec3f(0, -9.8, 0),
  drag: 1,
};

/** This burst queues particles on the CPU for the next GPU `spawn` call. */
export interface Burst<S extends Schema = typeof Particle> {
  /**
   * Sets the burst count. All bursts queued before the same `spawn` share the system capacity.
   */
  readonly count: number;
  /** Sets the centre of the starting box. */
  readonly position: readonly [number, number, number];
  /** Sets half the starting box size along each axis. Default none, which starts at a point. */
  readonly area?: readonly [number, number, number];
  /** Sets initial velocity before `scatter`. Default none, which starts at rest. */
  readonly velocity?: readonly [number, number, number];
  /** Limits random velocity to this speed, in a sphere or a disc for 2D. Default 0. */
  readonly scatter?: number;
  /** Sets linear colour and opacity. Default white. */
  readonly color?: readonly [number, number, number, number];
  /** Sets quad width in world units. Default 0.1. */
  readonly size?: number;
  /** Sets particle lifetime in seconds. Default 1. */
  readonly life?: number;
  /** Sets random variation in lifetime and size as a fraction. Default 0.25. */
  readonly variance?: number;
  /**
   * Starts particles at random ages within their lifetimes so a burst does not fade and die at
   * once. Default false.
   */
  readonly aged?: boolean;
  /** Sets custom particle fields to the same values across the burst. Default zero. */
  readonly fields?: { readonly [K in keyof Own<S>]?: d.InferInput<Own<S>[K]> };
}

/** This stream emits a steady number of particles a second, from `system.stream`. */
export interface Stream {
  /**
   * Sets particles a second, 0 or more. Each `step` owes `rate` times the step's length, and each
   * `spawn` queues the whole particles owed, carrying the fraction. Set it to 0 to pause the
   * stream.
   */
  rate: number;
}

/** This schema provides world position and colour to pass outputs. */
export const Seen = d.struct({
  /** Holds the point on the particle quad in world space. */
  world: d.vec3f,
  /** Holds the premultiplied colour returned by `shade`. */
  color: d.vec4f,
});
/**
 * This schema describes a particle quad corner with clip position, world position, local
 * coordinates and pool slot.
 */
export const Corner = d.struct({
  /** Holds the corner position in clip space. */
  position: d.vec4f,
  /** Holds the world position returned by `place`. */
  world: d.vec3f,
  /** Holds local coordinates from -1 to 1 across the quad. */
  local: d.vec2f,
  slot: d.u32,
});

export interface Options<S extends Schema = typeof Particle> {
  /**
   * Limits the number of particles alive at once. Each emitted particle takes the next ring
   * slot.
   */
  readonly capacity: number;
  /** Selects a particle struct with custom fields from `schema`. Default `Particle`. */
  readonly particle?: S;
  /** Sets the drawing pipeline label. Default `particles`. */
  readonly label?: string;
  /**
   * Sets the length of a step in seconds, such as a fixed-step clock's `sim.step`. Each `step`
   * advances by it.
   */
  readonly step: number;
  /**
   * Selects `add` for sparks and glow, `over` for dust, `translucent` for the passes'
   * translucency targets, or a custom blend state. Default `add`.
   */
  readonly blend?: 'add' | 'over' | 'translucent' | GPUBlendState;
  /** Selects 2D or 3D burst scatter. In 2D, velocities scatter in the xy plane. Default 3. */
  readonly dimensions?: 2 | 3;
  /** Limits bursts queued for one `spawn`. Default 64. */
  readonly bursts?: number;
  /**
   * Updates every live particle at the start of each step, including its first, before gravity,
   * drag and motion. `dt` is the step duration in seconds. Use the slot for related data, such as
   * pushing a trail point. Return the particle to keep, or set `life` to 0 to end it. The step
   * kernel binds 1 storage buffer, the pool, so `update` and `died` share the other 7 of a stage.
   * Default none, which leaves it unchanged.
   * @gpu
   */
  readonly update?: (particle: NoInfer<Of<S>>, dt: number, slot: number) => NoInfer<Of<S>>;
  /**
   * Initializes a particle and its slot when a burst or `emit` creates it. Return the particle
   * to keep. Use the slot to initialize related data, such as resetting a trail. The spawn kernel
   * binds 3 storage buffers, the pool, its ring counter and the burst queue, so `born` has 5, and
   * it runs in every kernel that calls `emit` too. Default none, which keeps the emitted values.
   * @gpu
   */
  readonly born?: (particle: NoInfer<Of<S>>, slot: number) => NoInfer<Of<S>>;
  /**
   * Returns a quad corner in world space from `corner` coordinates in [-1, 1]. The particle
   * position includes the drawing offset along velocity. The drawing binds 2 storage buffers in
   * both stages, the pool and the live slots, so `place` and `shade` share 6. Default `facing`,
   * which makes a camera-facing square of width `size`.
   * @gpu
   */
  readonly place?: (particle: NoInfer<Of<S>>, corner: d.v2f) => d.v3f;
  /**
   * Returns premultiplied colour for a quad fragment at `local` coordinates in [-1, 1]. It shares
   * 6 storage buffers with `place`, as `place` says. Default `softDisc`.
   * @gpu
   */
  readonly shade?: (particle: NoInfer<Of<S>>, local: d.v2f) => d.v4f;
  /**
   * Runs after a step ends a particle lifetime, with the particle and its slot. It can emit into
   * another system with that system's `emit`. It does not
   * run for live particles overwritten by the ring, particles born dead or particles ended by
   * `clear`. A system it emits into binds 2 storage buffers in the step, its pool and ring
   * counter. Default none.
   * @gpu
   */
  readonly died?: (particle: NoInfer<Of<S>>, slot: number) => void;
  /**
   * Sets initial gravity and drag over `defaults`. Set both to zero when `update` controls
   * motion. Default `defaults`.
   */
  readonly settings?: Partial<d.InferInput<typeof Settings>>;
}

/**
 * The members every particle system has, whatever its schema: its slots, recording and drawing.
 * A parameter typed `particles.System` takes a system of any schema.
 */
export interface System extends Disposable {
  /** Reports the number of ring slots, from the `capacity` option. */
  readonly capacity: number;
  /**
   * Gives the dispatch range for kernels over the pool. It includes every slot because any slot
   * can hold a particle and the CPU does not track live slots.
   */
  readonly span: number;
  /** Returns whether slot `i` is alive, in any shader. */
  readonly alive: (i: number) => boolean;
  /**
   * Returns the capacity and live test, filled, as a table's `live()` does. Spread it into a
   * live lists, a grid, a picker or state machines: `grid.create(root, { ...flashes.live(), position })`.
   */
  live(): Live;
  /**
   * Returns where the drawing places slot `i` this frame: its position moved along its velocity
   * by the `ahead` of the last `compact`.
   */
  readonly drawnAt: (i: number) => d.v3f;
  /** Fills `emit` and the pool's accessors with this system's buffers, as their defaults do. */
  inject(): Inject;
  /**
   * Holds gravity and drag from `defaults` and the `settings` option. Use `patch` to upload only
   * the fields you supply.
   */
  readonly settings: TgpuUniform<typeof Settings>;
  /**
   * Holds the 5-word indirect draw record written by `compact` and read by `draw`. Word 1 is the
   * live count from the last `compact`; copy it with a readback to display it.
   */
  readonly args: TgpuBuffer<d.WgslArray<d.U32>> & IndirectFlag & StorageFlag;
  /** Compiles the spawn, step and compaction pipelines. Calling it is optional. */
  initAsync(): Promise<void>;
  /**
   * Spawns queued bursts before `compact`. Record it before steps to move those particles this
   * frame, or after steps to show bursts queued during the steps. Each encoder takes one spawn
   * with bursts, submitted before the next: the bursts upload through the queue before the submit
   * runs, so a second spawn before then, in any pass or encoder, replaces the first one's bursts.
   * Throws when one pass spawns bursts twice.
   */
  spawn(pass: TgpuComputePass): void;
  /** Advances every live particle by one fixed step. */
  step(pass: TgpuComputePass): void;
  /**
   * Lists live particles for drawing. Record this once per frame after simulation steps. Drawing
   * moves each particle `ahead` seconds along its velocity, such as a fixed-step clock's
   * `sim.ahead`, so motion follows the display rate between steps.
   */
  compact(pass: TgpuComputePass, ahead: number): void;
  /**
   * Provides the particle drawing for pass descriptions, as in `scene.add(system.drawing)`. It
   * supplies colour, world position and corner data to outputs, tests depth without writing it
   * and draws the list from the last `compact`. The description `initAsync` also compiles system
   * kernels.
   */
  readonly drawing: passes.Drawing<typeof Corner, d.Infer<typeof Seen>>;
  /** Ends every particle, recorded into `target`, a frame or an encoder. */
  clear(target: Frame | TgpuCommandEncoder): void;
  /** Destroys the pool, its counter, the burst queue and the live lists. */
  destroy(): void;
}

/** A particle system of schema `S`, with the members that read and write its particles. */
export interface Particles<S extends Schema = typeof Particle> extends System {
  /**
   * Reads the particle in slot `i`, which must be below `span`.
   */
  readonly particle: (i: number) => Of<S>;
  /**
   * Emits a particle from any compute shader. It takes the next ring slot, overwriting the particle emitted `capacity` emissions earlier even if
   * that particle is alive.
   */
  readonly emit: (particle: Of<S>) => void;
  /** Stores every particle, one per ring slot. */
  readonly pool: TgpuBuffer<d.WgslArray<S>> & StorageFlag;
  /**
   * Queues a burst for the next `spawn`, trimming it to the remaining capacity. Returns false if
   * the queue is full or queued bursts fill capacity. Throws for a
   * `NaN` count.
   */
  burst(burst: Burst<S>): boolean;
  /**
   * Emits `rate` particles a second from `burst`, which the system keeps and reads at each `spawn`,
   * so changes to its fields between frames take effect. The steps owe the particles and the next
   * `spawn` queues them, so those a frame's steps owe spawn in the next frame. Throws unless `rate`
   * is 0 or more.
   *
   * @returns the stream, whose `rate` you set
   */
  stream(rate: number, burst: Omit<Burst<S>, 'count'>): Stream;
}

const WORKGROUP = 64;

/** Shades a round blob that fades towards its edge and over its lifetime. */
export const softDisc = (particle: Particle, local: d.v2f) => {
  'use gpu';
  const edge = std.saturate(1 - std.length(local));
  const alpha = particle.color.w * edge * edge * (1 - particle.age / particle.life);

  return d.vec4f(particle.color.xyz * alpha, alpha);
};

/**
 * Shades a glowing particle, such as an ember or a spark: a bright core with a soft halo, fading
 * with its age. It adds light with an alpha of 0, so it hides nothing behind it under `over` as
 * under `add`.
 */
export const glowingDisc = (particle: Particle, local: d.v2f) => {
  'use gpu';
  const radius = std.length(local);
  const core = 1 - std.smoothstep(0.3, 0.7, radius);
  const halo = std.saturate(1 - radius) * 0.25;
  const left = std.saturate(1 - particle.age / particle.life);
  return d.vec4f(particle.color.xyz * (particle.color.w * (core + halo) * left * left), 0);
};

/**
 * Places a corner on a camera-facing square of width `size`. This is the default `place`
 * function. The calling pipeline must fill `cameras.current`.
 */
export const facing = (particle: Particle, corner: d.v2f) => {
  'use gpu';
  const across = cameras.current.$.right * corner.x + cameras.current.$.up * corner.y;
  return particle.position + across * (particle.size * 0.5);
};

/** This type describes a solid that particles collide with, as a signed distance. */
export interface Solid {
  /** Returns the signed distance from `p` to the solid's surface, negative inside. */
  readonly distance: (p: d.v3f) => number;
  /**
   * Returns the direction in which `distance` grows fastest at `p`, of any length. Default an
   * estimate from 4 more calls of `distance`.
   */
  readonly gradient?: (p: d.v3f) => d.v3f;
}

export interface CollideOptions {
  /** Sets the share of a particle's speed into the surface that bounces back out. Default 0.5. */
  readonly bounce?: number;
  /**
   * Sets the share of a particle's speed along the surface that it loses each step it touches.
   * Default 0.
   */
  readonly friction?: number;
}

/** Offsets of a tetrahedron's corners, which estimate a gradient from 4 distances. */
const TAPS = [
  [1, -1, -1],
  [-1, -1, 1],
  [-1, 1, -1],
  [1, 1, 1],
] as const;
const REACH = 1e-3;

/**
 * Makes an `update` that keeps particles out of `solid`. A particle closer to its surface than
 * half its `size` moves out along the gradient, and its speed into the surface bounces back.
 * Pass it as `update`, or call it from your own: `return ground(tumble(pebble, dt))`.
 *
 * @param solid the signed distance of what particles bounce off, and its gradient if known
 * @returns a `'use gpu'` function that takes a particle and returns it collided
 */
export function collide(solid: Solid, options: CollideOptions = {}) {
  const { distance, gradient } = solid;
  const { bounce = 0.5, friction = 0 } = options;

  const gradientAt = (p: d.v3f) => {
    'use gpu';
    if (gradient !== undefined) return gradient(p);
    let sum = d.vec3f();
    for (const corner of tgpu.unroll(TAPS)) {
      const tap = d.vec3f(corner[0], corner[1], corner[2]);
      sum += tap * distance(p + tap * REACH);
    }
    return sum;
  };

  const collided = <P extends Particle>(particle: P): P => {
    'use gpu';
    let next = std.copy(particle);
    const depth = particle.size * 0.5 - distance(particle.position);
    if (depth > 0) {
      const outward = gradientAt(particle.position);
      // A zero gradient, as at a sphere's centre, has no direction to move along
      const steepness = std.dot(outward, outward);
      if (steepness === 0) return next;
      const normal = outward * std.inverseSqrt(steepness);
      next.position = particle.position + normal * depth;
      const into = std.dot(particle.velocity, normal);
      if (into < 0) {
        const along = particle.velocity - normal * into;
        next.velocity = along * (1 - friction) - normal * (into * bounce);
      }
    }
    return next;
  };
  return collided;
}

const BurstData = d.struct({
  position: d.vec3f,
  count: d.u32,
  area: d.vec3f,
  scatter: d.f32,
  velocity: d.vec3f,
  size: d.f32,
  color: d.vec4f,
  life: d.f32,
  variance: d.f32,
  seed: d.f32,
  aged: d.u32,
});

type BurstGpu = d.InferGPU<typeof BurstData> & { own: Particle };

type BurstRecord = d.InferInput<typeof BurstData> & {
  position: [number, number, number];
  area: [number, number, number];
  velocity: [number, number, number];
  color: [number, number, number, number];
  own?: Record<string, unknown>;
};

const ZERO = [0, 0, 0] as const;
const WHITE = [1, 1, 1, 1] as const;

const copy = (target: number[], source: readonly number[]) => {
  for (let i = 0; i < target.length; i++) target[i] = source[i];
};

const rateOf = (rate: number) => {
  if (!(rate >= 0) || rate === Infinity) {
    throw new RangeError(`particles: a stream takes 0 or more particles a second, got ${rate}.`);
  }
  return rate;
};

/**
 * Creates a GPU particle pool. CPU bursts and kernel `emit` calls allocate ring slots. Fixed
 * steps move particles, and live lists list only live particles for drawing.
 *
 * @injects member `emit` and the pool, which read its own buffers in any pipeline
 * @binds 2 storage: `emit`, the pool and its ring counter
 * @binds 1 storage: `alive` or `particle`, the pool
 * @binds 1 storage, 1 uniform: `drawnAt`, the pool and the last `compact`'s `ahead`
 * @owns 6 buffers, 2 pipelines, `liveLists`: the pool, its ring counter, the burst queue and three
 * uniforms; the live lists list the live particles. Where compute runs on the CPU, two buffers and
 * two pipelines that count the live particles take the live lists' place, and the draw covers
 * every slot. Each description `drawing` is added to builds
 * its pipeline.
 * @records `spawn` once a submission before `compact`, `step` once per fixed step, `compact` once
 * after the steps, then the draw a description added `drawing` as; `clear` whenever you like.
 * @param camera the camera the quads face unless `place` says otherwise; the callbacks read it too
 * @param options.update a `'use gpu'` function; it and the other callbacks carry the fills of what
 * they read beside the camera, such as `tgpu.fn(rise).pipe(noise.inject())`
 * @returns a system you destroy
 */
export function create<S extends Schema>(
  root: TgpuRoot,
  camera: CameraBase,
  options: Options<S> & { readonly particle: S },
): Particles<S>;
export function create(
  root: TgpuRoot,
  camera: CameraBase,
  options: Options & { readonly particle?: typeof Particle },
): Particles;
export function create<S extends Schema>(
  root: TgpuRoot,
  camera: CameraBase,
  options: Options<S>,
): Particles<S> {
  const { step, blend = 'add', dimensions = 3, bursts: queueSize = 64 } = options;
  const size = options.capacity;

  const Row = (options.particle ?? Particle) as unknown as typeof Particle;
  const ownKeys = Object.keys(Row.propTypes).filter((key) => !Object.hasOwn(fields, key));
  const owned = ownKeys.length > 0;
  const { update, born, died } = options as unknown as {
    readonly update?: (particle: Particle, dt: number, slot: number) => Particle;
    readonly born?: (particle: Particle, slot: number) => Particle;
    readonly died?: (particle: Particle, slot: number) => void;
  };
  const place = (options.place ?? facing) as (particle: Particle, corner: d.v2f) => d.v3f;
  const shade = (options.shade ?? softDisc) as (particle: Particle, local: d.v2f) => d.v4f;
  const flat = dimensions === 2;

  const Queued = owned
    ? d.struct({ ...BurstData.propTypes, own: Row }).$name('BurstData')
    : BurstData;

  const poolBuffer = root.createBuffer(d.arrayOf(Row, size)).$usage('storage');
  const headBuffer = root.createBuffer(d.atomic(d.u32)).$usage('storage');
  const burstBuffer = root.createBuffer(d.arrayOf(Queued, queueSize)).$usage('storage');
  const queued = root.createUniform(d.u32);
  // Create the buffers only spawns use now, so a first burst mid-run allocates nothing
  for (const buffer of [headBuffer, burstBuffer, queued.buffer]) root.unwrap(buffer);
  // Track both current and preceding drawing offsets along velocity for motion output
  const ahead = root.createUniform(d.vec2f);
  const aheadNow = () => {
    'use gpu';
    return ahead.$.x;
  };
  const aheadBefore = () => {
    'use gpu';
    return ahead.$.y;
  };
  const drawnAhead = tgpu.accessor(d.f32, history.temporal(aheadNow, aheadBefore));
  const settings = root.createUniform(Settings, { ...defaults, ...options.settings });

  const burstData = burstBuffer.as('readonly');

  // Per-system accessors filled with its own buffers, so one kernel can emit into several systems
  const rows = byStage(poolBuffer);
  const counter = headBuffer.as('mutable');
  const pool = tgpu.accessor(d.arrayOf(Row), rows);
  const head = tgpu.mutableAccessor(d.atomic(d.u32), counter);
  const inject: Inject = (cfg) =>
    cfg
      .with(pool, rows)
      .with(head, counter)
      .with(drawnAhead, history.temporal(aheadNow, aheadBefore));

  const alive = (i: number) => {
    'use gpu';
    return i < size && pool.$[i].age < pool.$[i].life;
  };

  const particleAt = (i: number) => {
    'use gpu';
    return Row(pool.$[i]);
  };

  const drawnAt = (i: number) => {
    'use gpu';
    const particle = pool.$[i];
    return particle.position + particle.velocity * drawnAhead.$;
  };

  const emit = (particle: Particle) => {
    'use gpu';
    const slot = std.atomicAdd(head.$, 1) % size;
    if (born !== undefined) {
      pool.$[slot] = born(Row(particle), slot);
    } else {
      pool.$[slot] = Row(particle);
    }
  };

  const filledLive = { capacity: size, alive };
  // Where compute runs on the CPU, listing live slots would read the pool back every frame. The
  // draw covers every slot instead and drops the dead ones, and a sum counts the live ones
  const alone = backends.of(root).compute === 'cpu';
  const live = alone
    ? undefined
    : liveLists.create(root, { ...filledLive, draws: [{ vertexCount: 4 }] });
  const tallyBuffer = alone
    ? root.createBuffer(d.arrayOf(d.atomic(d.u32), 1)).$usage('storage')
    : undefined;
  // Laid out as live lists' records: four vertices, the live count, then no offsets
  const countBuffer = alone
    ? root.createBuffer(d.arrayOf(d.u32, 5), [4, 0, 0, 0, 0]).$usage('storage', 'indirect')
    : undefined;
  const tallied = tallyBuffer?.as('mutable');
  const counted = countBuffer?.as('mutable');

  const birth = (burst: BurstGpu, index: number) => {
    'use gpu';
    randf.seed2(d.vec2f(d.f32(index), burst.seed));

    const life = burst.life * (1 + (randf.sample() * 2 - 1) * burst.variance);
    const scale = 1 + (randf.sample() * 2 - 1) * burst.variance;
    const push = flat ? d.vec3f(randf.inUnitCircle(), 0) : randf.inUnitSphere();

    let age = d.f32(0);
    if (burst.aged === 1) age = randf.sample() * life;

    let particle = Row();
    if (owned) particle = Row(burst.own);
    particle.position = burst.position + (randf.inUnitCube() * 2 - 1) * burst.area;
    particle.age = age;
    particle.velocity = burst.velocity + push * burst.scatter;
    particle.life = life;
    particle.color = d.vec4f(burst.color);
    particle.size = burst.size * scale;
    particle.seed = randf.sample();
    emit(particle);
  };

  const spawnKernel = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    let first = d.u32(0);
    for (let i = d.u32(0); i < queued.$; i++) {
      const burst = (burstData.$ as BurstGpu[])[i];
      if (gid.x < first + burst.count) {
        birth(burst, gid.x);
        return;
      }
      first += burst.count;
    }
  });

  /** The particle `birth` emits, for spawning where compute runs on the CPU. */
  const made = (burst: BurstGpu, index: number) => {
    'use gpu';
    randf.seed2(d.vec2f(d.f32(index), burst.seed));

    const life = burst.life * (1 + (randf.sample() * 2 - 1) * burst.variance);
    const scale = 1 + (randf.sample() * 2 - 1) * burst.variance;
    const push = flat ? d.vec3f(randf.inUnitCircle(), 0) : randf.inUnitSphere();

    let age = d.f32(0);
    if (burst.aged === 1) age = randf.sample() * life;

    let particle = Row();
    if (owned) particle = Row(burst.own);
    particle.position = burst.position + (randf.inUnitCube() * 2 - 1) * burst.area;
    particle.age = age;
    particle.velocity = burst.velocity + push * burst.scatter;
    particle.life = life;
    particle.color = d.vec4f(burst.color);
    particle.size = burst.size * scale;
    particle.seed = randf.sample();
    return particle;
  };

  // Where compute runs on the CPU, taking ring slots with an atomic keeps spawning there. Instead
  // each slot finds the burst particle the ring places in it, counting from the head, and one
  // thread then moves the head past them. Bursts fill the ring in order; an atomic fills it in any

  const spawnEach = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const slot = gid.x;
    if (slot >= size) return;
    const index = (slot + size - (std.atomicLoad(head.$) % size)) % size;
    let first = d.u32(0);
    let found = d.u32(queueSize);
    for (let i = d.u32(0); i < queued.$; i++) {
      const count = (burstData.$ as BurstGpu[])[i].count;
      if (index < first + count) {
        found = i;
        break;
      }
      first += count;
    }
    if (found === queueSize) return;
    const particle = made((burstData.$ as BurstGpu[])[found], index);
    if (born !== undefined) {
      pool.$[slot] = born(Row(particle), slot);
    } else {
      pool.$[slot] = Row(particle);
    }
  });

  const advanceKernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    let total = d.u32(0);
    for (let i = d.u32(0); i < queued.$; i++) total += (burstData.$ as BurstGpu[])[i].count;
    std.atomicStore(head.$, std.atomicLoad(head.$) + total);
  });

  const stepKernel = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (!alive(i)) return;

    let next = Row(pool.$[i]);
    if (update !== undefined) next = Row(update(next, d.f32(step), i));
    const velocity = (next.velocity + settings.$.gravity * step) * std.exp(-settings.$.drag * step);
    next.position += velocity * step;
    next.age += step;
    next.velocity = d.vec3f(velocity);

    pool.$[i] = Row(next);
    if (died !== undefined && next.age >= next.life) died(next, i);
  });

  const at = (vertexIndex: number, instanceIndex: number) => {
    'use gpu';
    const slot = live!.slot(0, instanceIndex);
    // A strip of two clockwise triangles split from corner (-1, 1) to (1, -1)
    const corner = d.vec2f(d.f32(vertexIndex >>> 1), d.f32(vertexIndex & 1)) * 2 - 1;
    const drawn = Row(pool.$[slot]);
    drawn.position = drawnAt(slot);
    const world = place(drawn, corner);

    return Corner({
      position: cameras.toClip(world),
      world,
      local: corner,
      slot,
    });
  };

  // Drawn over every slot where compute runs on the CPU: a dead one's corners all fall outside
  // the clip volume
  const covered = (vertexIndex: number, instanceIndex: number) => {
    'use gpu';
    const slot = d.u32(instanceIndex);
    if (!alive(slot)) {
      return Corner({ position: d.vec4f(2, 2, 2, 1), world: d.vec3f(), local: d.vec2f(), slot });
    }
    const corner = d.vec2f(d.f32(vertexIndex >>> 1), d.f32(vertexIndex & 1)) * 2 - 1;
    const drawn = Row(pool.$[slot]);
    drawn.position = drawnAt(slot);
    const world = place(drawn, corner);

    return Corner({
      position: cameras.toClip(world),
      world,
      local: corner,
      slot,
    });
  };

  const pixelOf = (corner: d.Infer<typeof Corner>) => {
    'use gpu';
    return Seen({ world: corner.world, color: shade(Row(pool.$[corner.slot]), corner.local) });
  };

  const filled: Inject = (cfg) => cfg.pipe(inject).pipe(camera.inject());
  const injected = root.pipe(filled);
  const spawnPipeline = (
    alone
      ? injected.createComputePipeline({ compute: spawnEach })
      : injected.createComputePipeline({ compute: spawnKernel })
  ).$name('spawnPipeline');
  const advancing = alone
    ? injected.createComputePipeline({ compute: advanceKernel }).$name('advancing')
    : undefined;
  const stepPipeline = injected.createComputePipeline({ compute: stepKernel });
  const drawn: Inject = (cfg) => (live ? cfg.pipe(filled).pipe(live.inject()) : cfg.pipe(filled));

  const tally = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x < size && alive(gid.x)) std.atomicAdd(tallied!.$[0], 1);
  });
  const recount = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    counted!.$[1] = std.atomicExchange(tallied!.$[0], 0);
  });
  const tallyPipeline = alone
    ? injected.createComputePipeline({ compute: tally }).$name('tally')
    : undefined;
  const countPipeline = alone
    ? root.createComputePipeline({ compute: recount }).$name('counting')
    : undefined;

  // Reuse burst records until spawn consumes them. Initialize custom fields from a zero particle
  const zero = Row() as unknown as Record<string, unknown>;
  const records: BurstRecord[] = Array.from({ length: queueSize }, () => ({
    position: [0, 0, 0],
    count: 0,
    area: [0, 0, 0],
    scatter: 0,
    velocity: [0, 0, 0],
    size: 0,
    color: [0, 0, 0, 0],
    life: 0,
    variance: 0,
    seed: 0,
    aged: 0,
    ...(owned ? { own: Row() as unknown as Record<string, unknown> } : {}),
  }));
  const pending: BurstRecord[] = [];
  let total = 0;
  const streams: { stream: Stream; burst: Omit<Burst, 'count'>; owed: number }[] = [];

  const queue = (burst: Omit<Burst, 'count'>, count: number) => {
    const record = records[pending.length];
    const kept = Math.min(Math.floor(count), size - total);
    if (!record || kept <= 0) return false;

    copy(record.position, burst.position);
    copy(record.area, burst.area ?? ZERO);
    copy(record.velocity, burst.velocity ?? ZERO);
    copy(record.color, burst.color ?? WHITE);
    record.count = kept;
    record.scatter = burst.scatter ?? 0;
    record.size = burst.size ?? 0.1;
    record.life = burst.life ?? 1;
    record.variance = burst.variance ?? 0.25;
    record.seed = Math.random();
    record.aged = burst.aged ? 1 : 0;
    if (record.own) {
      const given = (burst.fields ?? {}) as Record<string, unknown>;
      for (const key of ownKeys) record.own[key] = given[key] ?? zero[key];
    }

    pending.push(record);
    total += kept;
    return true;
  };

  const lastDrawn: [number, number] = [0, 0];

  let stepped = 0;
  let spawnedIn: TgpuComputePass | undefined;

  const initAsync = () =>
    pipelines.compile([
      ...(live ? [live] : []),
      ...(tallyPipeline && countPipeline ? [tallyPipeline, countPipeline] : []),
      spawnPipeline,
      stepPipeline,
      ...(advancing ? [advancing] : []),
    ]);

  const destroy = () => {
    poolBuffer.destroy();
    headBuffer.destroy();
    burstBuffer.destroy();
    live?.destroy();
    tallyBuffer?.destroy();
    countBuffer?.destroy();
  };

  const system: Particles<typeof Particle> = {
    capacity: size,
    span: size,
    alive,
    live: () => filledLive,
    particle: particleAt,
    drawnAt,

    inject: () => inject,

    emit,
    settings,
    pool: poolBuffer,
    args: live ? live.args : (countBuffer as NonNullable<typeof countBuffer>),

    initAsync,

    burst(burst) {
      if (Number.isNaN(burst.count)) throw new RangeError('A burst needs a count, got NaN.');
      return queue(burst, burst.count);
    },

    stream(rate, burst) {
      let current = rateOf(rate);
      const stream: Stream = {
        get rate() {
          return current;
        },
        set rate(next) {
          current = rateOf(next);
        },
      };
      streams.push({ stream, burst, owed: 0 });
      return stream;
    },

    spawn(pass) {
      for (const flowing of streams) {
        const due = Math.floor(flowing.owed);
        if (due <= 0) continue;
        flowing.owed -= due;
        queue(flowing.burst, due);
      }
      if (total === 0) return;
      if (pass === spawnedIn) {
        throw new Error(
          'particles: a second `spawn` in one submission would replace the first bursts. Submit between them.',
        );
      }
      spawnedIn = pass;

      burstBuffer.write(pending as d.InferInput<typeof BurstData>[]);
      queued.write(pending.length);
      pass.setPipeline(spawnPipeline);
      if (advancing) {
        pass.dispatchWorkgroups(Math.ceil(size / WORKGROUP));
        pass.setPipeline(advancing);
        pass.dispatchWorkgroups(1);
      } else {
        pass.dispatchWorkgroups(Math.ceil(total / WORKGROUP));
      }

      pending.length = 0;
      total = 0;
    },

    step(pass) {
      pass.setPipeline(stepPipeline);
      pass.dispatchWorkgroups(Math.ceil(size / WORKGROUP));
      stepped += 1;
      // A stream owes at most a pool's worth, however long spawns stop
      for (const flowing of streams) {
        flowing.owed = Math.min(flowing.owed + flowing.stream.rate * step, size);
      }
    },

    compact(pass, by) {
      if (live) live.compact(pass);
      else if (tallyPipeline && countPipeline) {
        pass.setPipeline(tallyPipeline);
        pass.dispatchWorkgroups(Math.ceil(size / WORKGROUP));
        pass.setPipeline(countPipeline);
        pass.dispatchWorkgroups(1);
      }

      // Update once per frame so all pass descriptions draw the same particle state
      const was = lastDrawn[0] - stepped * step;
      stepped = 0;
      if (by !== lastDrawn[0] || was !== lastDrawn[1]) {
        lastDrawn[0] = by;
        lastDrawn[1] = was;
        ahead.write(lastDrawn);
      }
    },

    drawing: {
      label: options.label ?? 'particles',
      Vertex: Corner,
      at: live ? at : covered,
      pixelOf,
      color: passes.shaded,
      ...(blend === 'translucent'
        ? { translucent: true }
        : { blend: typeof blend === 'string' ? pipelines.blend[blend] : blend }),
      depthStencil: { depthWriteEnabled: false },
      primitive: { topology: 'triangle-strip' },
      inject: () => drawn,
      draw: (pass) => (live ? live.draw(pass, 0) : pass.draw(4, size)),
      initAsync,
    },

    clear: (target) => poolBuffer.clear(encoderOf(target)),

    destroy,
    [Symbol.dispose]: destroy,
  };
  return system as unknown as Particles<S>;
}
