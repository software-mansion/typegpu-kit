import {
  cameras,
  counters,
  liveLists,
  Particle,
  passes,
  pipelines,
  table,
  type Particles,
} from '@typegpu/kit';
import type { Injectable } from '@typegpu/kit/inject';
import { randf } from '@typegpu/noise';
import { sdDisk, sdRoundedBox2d } from '@typegpu/sdf';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';
import { avatar } from './avatar.ts';
import { DUST } from './effects.ts';
import { HALF } from './hero.ts';
import { BAT, WALKER, type Level } from './level.ts';
import { eyes, quad, snapped, sprite, TEXELS, type ArtOptions, type Seen } from './sprites.ts';

const Critter = table.define('Critter', {
  host: {
    kind: d.u32,
    home: d.vec2f,
    /** Horizontal turning bounds in tiles. */
    patrol: d.vec2f,
    /** Loop phase offset in seconds. */
    phase: d.f32,
  },
  device: {
    position: d.vec2f,
    heading: d.f32,
    /** Stomp time in seconds, or -1 before a stomp. */
    squashedAt: d.f32,
  },
});

const WORKGROUP = 64;

/** Seconds a squashed critter stays flat before it goes. */
const VANISH = 0.5;
/** A squashed critter's width and height, as shares of its own. */
const FLAT = d.vec2f(1.3, 0.3);

/** Tiles a second a walker walks. */
const WALK = 1.6;
/** Bat loop frequencies in radians per second and horizontal and vertical amplitudes in tiles. */
const FLIGHT = { pace: 0.9, reach: 2.5, bob: 2.3, rise: 0.6 };
/** Radians a second a bat's wings beat. */
const WINGBEAT = 18;
/** Loop phase offset per critter in seconds. Keeps bats from flying in step. */
const STAGGER = 1.7;

/** Half a critter's size, in tiles. */
const SIZE = 0.4;
/** Overlap margin in tiles. A graze within this margin does not count as contact. */
const GRAZE = 0.08;
/** Contact distance between hero and critter centres. */
const TOUCH = d.vec2f(HALF[0] + SIZE - GRAZE, HALF[1] + SIZE - GRAZE);

/**
 * Stomp dust count, horizontal and vertical velocity spread and lift in tiles per second,
 * minimum and maximum lifetime in seconds, and size in tiles.
 */
const KICK = { motes: 8, spread: d.vec2f(5, 2), lift: 1.5, life: [0.3, 0.5], size: 0.3 };
const MOTE = d.vec4f(...DUST);

const SHELL = d.vec3f(0.42, 0.25, 0.6);
const WING = d.vec3f(0.22, 0.16, 0.32);
/** Eye size and centre offset forward and above the critter centre. */
const eyesAt = eyes({ apart: 0.2, radius: 0.16, pupil: 0.08, look: 0.06 });
const EYES = { ahead: 0.2, up: 0.15 };

const OUTLINE = d.vec3f(0.16, 0.09, 0.22);

/** A walker's shell distance at `local`. */
const shellAt = (local: d.v2f) => {
  'use gpu';
  return sdRoundedBox2d(local + d.vec2f(0, 0.15), d.vec2f(0.85, 0.7), 0.55);
};

/** What a critter's pixels read from its quad: the corner in quad units, its kind and facing. */
const Corner = d.struct({ position: d.vec4f, local: d.vec2f, kind: d.u32, facing: d.f32 });

/** The part of [0, 1] where `origin + path * t` lies strictly between `low` and `high`. */
const slab = (origin: number, path: number, low: number, high: number) => {
  'use gpu';
  if (std.abs(path) < 1e-6) {
    if (origin > low && origin < high) return d.vec2f(0, 1);
    return d.vec2f(1, 0);
  }
  const a = (low - origin) / path;
  const b = (high - origin) / path;
  return d.vec2f(std.max(std.min(a, b), 0), std.min(std.max(a, b), 1));
};

/**
 * First contact along the hero's frame path, from 0 to 1, or -1 for no contact. Every step
 * reads the whole path to prevent fast motion skipping critters.
 */
const touchAt = (centre: d.v2f) => {
  'use gpu';
  const origin = avatar.$.before;
  const path = avatar.$.after - origin;
  const x = slab(origin.x, path.x, centre.x - TOUCH.x, centre.x + TOUCH.x);
  const y = slab(origin.y, path.y, centre.y - TOUCH.y, centre.y + TOUCH.y);

  const enter = std.max(x.x, y.x);
  if (enter >= std.min(x.y, y.y)) return d.f32(-1);
  return enter;
};

export interface CrittersOptions extends ArtOptions {
  readonly capacity: number;
  /** Seconds per `step`. */
  readonly step: number;
  /** Fills `avatar` for contact detection. */
  readonly avatar: Injectable;
  /** Takes the dust a stomp kicks up. */
  readonly dust: Pick<Particles<typeof Particle>, 'emit'>;
}

/** Moves walkers and bats on the GPU and detects contacts with the hero. */
export function createCritters(root: TgpuRoot, options: CrittersOptions) {
  const { capacity, step, time, dust } = options;

  const critters = table.create(root, Critter, {
    capacity,
    defaults: { heading: 1, squashedAt: -1 },
  });
  const live = liveLists.create(root, { ...critters.live(), draws: [quad] });
  const critter = critters.gpu;

  // Read contacts every frame to minimise stomp-bounce latency
  const contacts = counters.create(root, ['stomps', 'hurts'], { hz: 60 });

  /** Emits stomp dust at `at`, seeded by the critter's index. */
  const kickUp = (at: d.v2f, index: number) => {
    'use gpu';
    randf.seed2(d.vec2f(d.f32(index), time.$));
    for (let k = 0; k < KICK.motes; k++) {
      const kick = randf.inUnitCircle() * KICK.spread + d.vec2f(0, KICK.lift);
      dust.emit(
        Particle({
          position: d.vec3f(at, 0),
          age: 0,
          velocity: d.vec3f(kick, 0),
          life: std.mix(KICK.life[0], KICK.life[1], randf.sample()),
          color: MOTE,
          size: KICK.size,
          seed: randf.sample(),
        }),
      );
    }
  };

  const stepKernel = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (!critter.alive(i)) return;

    const row = critter.device.$[i];
    if (row.squashedAt >= 0) return;

    if (critter.kind(i) === WALKER) {
      const patrol = critter.patrol(i);
      const x = row.position.x + row.heading * WALK * step;
      if (x < patrol.x || x > patrol.y) row.heading = -row.heading;
      row.position.x = std.clamp(x, patrol.x, patrol.y);
    } else {
      const t = time.$ + critter.phase(i);
      const across = std.sin(t * FLIGHT.pace) * FLIGHT.reach;
      row.position = critter.home(i) + d.vec2f(across, std.sin(t * FLIGHT.bob) * FLIGHT.rise);
      row.heading = std.sign(std.cos(t * FLIGHT.pace));
    }

    const enter = touchAt(row.position);
    if (enter < 0) return;

    const offset = std.mix(avatar.$.before, avatar.$.after, enter) - row.position;
    if (avatar.$.velocity.y < 0 && offset.y > SIZE) {
      row.squashedAt = time.$;
      contacts.add.stomps(1);
      kickUp(row.position, i);
    } else {
      contacts.add.hurts(1);
    }
  });

  /** A bat's wing distance at `local`, beating by the clock. */
  const wingAt = (local: d.v2f) => {
    'use gpu';
    const beat = std.sin(time.$ * WINGBEAT) * 0.3;
    const wing = d.vec2f(std.abs(local.x) - 0.6, local.y - beat * (std.abs(local.x) - 0.3));
    return sdRoundedBox2d(wing, d.vec2f(0.38, 0.12), 0.1);
  };

  const drawing: passes.Drawing<typeof Corner, Seen> = {
    label: 'critters',
    Vertex: Corner,
    flat: ['facing'],

    at: (vertexIndex, instanceIndex) => {
      'use gpu';
      const slot = live.slot(0, instanceIndex);
      const corner = quad.vertexAt(vertexIndex).position.xz;

      // Keep squashed critters on their feet while they shrink
      const squashedAt = critter.squashedAt(slot);
      const flat = squashedAt >= 0;
      const fade = std.saturate(1 - (time.$ - squashedAt) / VANISH);
      const scale = std.select(d.vec2f(1), FLAT * fade, flat);

      const local = corner * scale * SIZE + d.vec2f(0, (scale.y - 1) * SIZE);
      const world = snapped(critter.position(slot)) + local;

      return Corner({
        position: cameras.toClip(d.vec3f(world, 0)),
        local: corner,
        kind: critter.kind(slot),
        facing: critter.heading(slot),
      });
    },

    cutout: (corner) => {
      'use gpu';
      if (corner.kind === BAT) return wingAt(corner.local) > 0 && sdDisk(corner.local, 0.5) > 0;
      return shellAt(corner.local) > 0;
    },

    pixelOf: (corner) => {
      'use gpu';
      const local = corner.local;
      if (corner.kind === BAT && wingAt(local) < 0) return sprite(WING);

      const facing = corner.facing;
      const eye = eyesAt(local - d.vec2f(facing * EYES.ahead, EYES.up), facing);
      if (eye.w > 0) return sprite(eye.xyz);
      // Walkers wear a texel of outline, which bats' wings already give them
      if (corner.kind !== BAT && shellAt(local) > -1 / (TEXELS * SIZE)) return sprite(OUTLINE);
      return sprite(SHELL * (0.75 + 0.25 * std.step(-0.3, local.y)));
    },

    color: passes.shaded,
    draw: (pass) => live.draw(pass, 0),
  };

  const stepPipeline = root
    .pipe(options.avatar.inject())
    .createComputePipeline({ compute: stepKernel });

  return {
    table: critters,
    compact: live.compact,
    /** Stomps and hurts the steps count. Capture them after the steps each frame. */
    contacts,

    drawing,
    initAsync: () => pipelines.compile([live, stepPipeline]),

    reset(level: Level) {
      critters.clear();
      level.critters.forEach(({ kind, at, patrol }, i) => {
        critters.add({ kind, home: at, patrol, phase: i * STAGGER, position: at });
      });
    },

    /** Records one step after `table.flush()`. */
    step(pass: TgpuComputePass) {
      const workgroups = Math.ceil(critters.span / WORKGROUP);
      if (workgroups > 0) {
        pass.setPipeline(stepPipeline);
        pass.dispatchWorkgroups(workgroups);
      }
    },

    destroy() {
      critters.destroy();
      live.destroy();
      contacts.destroy();
    },
  };
}

export type Critters = ReturnType<typeof createCritters>;
