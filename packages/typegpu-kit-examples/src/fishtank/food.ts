import {
  cameras,
  liveLists,
  Particle,
  particles,
  passes,
  pipelines,
  readback,
  table,
  type CameraBase,
  type RowId,
} from '@typegpu/kit';
import type { Injectable } from '@typegpu/kit/inject';
import { srgbToLinear } from '@typegpu/color';
import { meshes } from '@typegpu/kit/geometry';
import { randf } from '@typegpu/noise';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot, type TgpuUniform } from 'typegpu';
import { caustics, murk, TANK } from './water.ts';

const cube = meshes.unindexed(meshes.box());

export const Pellet = table.define('Pellet', {
  host: {
    /** Drop position below the water surface. */
    drop: d.vec3f,
    droppedAt: d.f32,
  },
});

/** Bites needed to consume one pellet. */
export const MEAL = 120;

/** World units under the surface a pellet is dropped at. */
const DROPPED = 0.3;
/** World units a second a pellet sinks. */
const SINK = 0.45;
/** World units along a pellet's sides. */
const SIZE = 0.08;
/** Radians a second a pellet turns. */
const SPIN = 1.7;
const COLOR = srgbToLinear(d.vec3f(0.85, 0.45, 0.18));
/** Lighting weights for overhead light, ambient light and caustics. */
const LIT = { above: 0.6, around: 0.4, caustics: 0.4 };

/**
 * Crumb colour, width in world units, maximum launch speed in world units per second, and
 * minimum and maximum lifetime in seconds.
 */
const CRUMB = {
  color: d.vec4f(srgbToLinear(d.vec3f(0.6, 0.32, 0.13)), 1),
  size: 0.04,
  speed: 0.8,
  life: [3, 5],
};
/** Chance of emitting a crumb per bite and maximum crumbs per pellet per step. */
const SHED = { odds: 0.3, most: 4 };

/** Reads of the bite counters per second. */
const READS_HZ = 10;
const WORKGROUP = 64;

/** Rests a crumb that has sunk to the sand on it. */
const settle = particles.collide(
  {
    distance: (p) => {
      'use gpu';
      return p.y;
    },
  },
  { bounce: 0, friction: 1 },
);

const turnY = (p: d.v3f, angle: number) => {
  'use gpu';
  const c = std.cos(angle);
  const s = std.sin(angle);

  return d.vec3f(p.x * c + p.z * s, p.y, p.z * c - p.x * s);
};

const Corner = d.struct({ position: d.vec4f, world: d.vec3f, normal: d.vec3f });
const Seen = d.struct({ color: d.vec4f });

export interface FoodOptions {
  readonly capacity: number;
  /** The length of a step in seconds; see `particles.Options.step`. */
  readonly step: number;
  readonly time: TgpuUniform<d.F32>;
  readonly camera: CameraBase;
  readonly water: Injectable;
}

/** Stores sinking pellets in a table and reads back bite counters to remove eaten pellets. */
export function createFood(root: TgpuRoot, options: FoodOptions) {
  const { capacity, time } = options;

  const rows = table.create(root, Pellet, { capacity });
  const live = liveLists.create(root, { ...rows.live(), draws: [cube] });

  const crumbs = particles.create(root, options.camera, {
    capacity: 2048,
    step: options.step,
    blend: 'over',
    update: settle,
    settings: { gravity: [0, -1.5, 0], drag: 3 },
  });

  const counters = root.createBuffer(d.arrayOf(d.atomic(d.u32), capacity)).$usage('storage');
  const counted = readback.create(root, counters, { hz: READS_HZ });
  const seenBuffer = root.createBuffer(d.arrayOf(d.u32, capacity)).$usage('storage');
  const seen = seenBuffer.as('mutable');
  const bites = counters.as('mutable');
  const uneaten: RowId<typeof Pellet>[] = [];
  const pellet = rows.gpu;

  /** Pellet position at `now`. Sinks from its drop position, then rests on the sand. */
  const pelletPosition = (i: number, now: number) => {
    'use gpu';
    const drop = pellet.drop(i);
    const fallen = (now - pellet.droppedAt(i)) * SINK;

    return d.vec3f(drop.x, std.max(drop.y - fallen, SIZE / 2), drop.z);
  };

  /** Counts a bite on pellet `i`. The CPU removes pellets after enough bites. */
  const bite = (i: number) => {
    'use gpu';
    std.atomicAdd(bites.$[i], 1);
  };

  // Emit crumbs here because the boids kernel already uses the 8 storage buffers WebGPU guarantees
  const crumbleKernel = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (!pellet.alive(i)) return;

    const bitten = std.atomicLoad(bites.$[i]);
    const fresh = std.min(bitten - seen.$[i], SHED.most);
    seen.$[i] = bitten;

    // Use bite counts to vary the seed between steps sharing one frame time
    randf.seed3(d.vec3f(i, bitten, time.$));
    const at = pelletPosition(i, time.$);
    for (let k = d.u32(0); k < fresh; k++) {
      if (randf.sample() > SHED.odds) continue;

      crumbs.emit(
        Particle({
          position: at,
          age: 0,
          velocity: randf.inUnitSphere() * CRUMB.speed,
          life: std.mix(CRUMB.life[0], CRUMB.life[1], randf.sample()),
          color: CRUMB.color,
          size: CRUMB.size,
          seed: randf.sample(),
        }),
      );
    }
  });

  const crumblePipeline = root.createComputePipeline({ compute: crumbleKernel });

  const at = (vertexIndex: number, instanceIndex: number) => {
    'use gpu';
    const slot = live.slot(0, instanceIndex);
    const corner = cube.vertexAt(vertexIndex);

    const spin = time.$ * SPIN + d.f32(slot);
    const world = pelletPosition(slot, time.$) + turnY(corner.position, spin) * SIZE;

    return Corner({
      position: cameras.toClip(world),
      world,
      normal: turnY(corner.normal, spin),
    });
  };

  const pixelOf = (corner: d.Infer<typeof Corner>) => {
    'use gpu';
    const above = std.max(corner.normal.y, 0) * LIT.above;
    const light = above + LIT.around + caustics(corner.world, time.$) * LIT.caustics;
    const air = murk(std.length(cameras.current.$.position - corner.world), corner.world.y);

    return Seen({ color: d.vec4f(COLOR * light * air.through + air.added, 1) });
  };

  const drawing: passes.Drawing<typeof Corner, d.Infer<typeof Seen>> = {
    Vertex: Corner,
    at,
    pixelOf,
    color: passes.shaded,
    primitive: { cullMode: 'back' },
    inject: () => options.water.inject(),
    draw: (pass) => live.draw(pass, 0),
  };

  return {
    table: rows,

    pelletPosition,
    bite,

    crumbs: crumbs.drawing,
    capture: counted.capture,
    initAsync: () => pipelines.compile([live, crumbs, crumblePipeline]),

    pellets: drawing,

    /** Drops a pellet below the surface. Does nothing at capacity. */
    drop(x: number, z: number, now: number) {
      if (rows.count === capacity) return;

      const id = rows.add({ drop: [x, TANK.y - DROPPED, z], droppedAt: now });
      const zeroed = { [rows.slotOf(id)]: 0 };
      counters.patch(zeroed);
      seenBuffer.patch(zeroed);

      // Earlier counts belong to the previous pellet in this slot
      counted.reset();
      uneaten.push(id);
    },

    /** Removes pellets eaten to a meal by the newest bite counts. Call once a frame. */
    eat() {
      const eaten = counted.take();
      if (!eaten) return;

      for (let i = uneaten.length - 1; i >= 0; i--) {
        const id = uneaten[i];
        if (eaten[rows.slotOf(id)] < MEAL) continue;

        rows.remove(id);
        uneaten.splice(i, 1);
      }
    },

    /** Emits crumbs for this step's bites, then steps them. Record after the fish. */
    step(pass: TgpuComputePass) {
      const workgroups = Math.ceil(rows.span / WORKGROUP);
      if (workgroups > 0) {
        pass.setPipeline(crumblePipeline);
        pass.dispatchWorkgroups(workgroups);
      }
      crumbs.step(pass);
    },

    /** Lists pellets and crumbs for drawing. Record after simulation steps. */
    compact(pass: TgpuComputePass, ahead: number) {
      live.compact(pass);
      crumbs.compact(pass, ahead);
    },

    destroy() {
      rows.destroy();
      live.destroy();
      crumbs.destroy();
      counters.destroy();
      seenBuffer.destroy();
      counted.destroy();
    },
  };
}

export type Food = ReturnType<typeof createFood>;
