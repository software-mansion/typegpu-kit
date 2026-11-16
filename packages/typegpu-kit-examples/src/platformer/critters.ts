import {
  cameras,
  counters,
  images,
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
import type { Picture } from '../contract.ts';
import { avatar } from './avatar.ts';
import { blown, DUST } from './effects.ts';
import { HALF } from './hero.ts';
import { BAT, FORK, WALKER, type Level } from './level.ts';
import {
  eyes,
  pixelated,
  quad,
  snapped,
  sprite,
  TEXELS,
  type ArtOptions,
  type Seen,
} from './sprites.ts';

const Critter = table.define('Critter', {
  host: {
    kind: d.u32,
    home: d.vec2f,
    /** Horizontal turning bounds in tiles. */
    patrol: d.vec2f,
    /** Phase offset in seconds of a bat's loop, a sponge's wiggles and a fork's hops. */
    phase: d.f32,
  },
  device: {
    position: d.vec2f,
    heading: d.f32,
    /** Stomp time in seconds, or -1 before a stomp. */
    squashedAt: d.f32,
    /** When a sponge next blows a bubble, in seconds of the clock. */
    bubbleAt: d.f32,
  },
});

const WORKGROUP = 64;

/** Seconds a squashed critter stays flat before it goes. */
const VANISH = 0.5;
/** A squashed critter's width and height, as shares of its own. */
const FLAT = d.vec2f(1.3, 0.3);

/** Bubbles a second a sponge blows, on average. */
const BUBBLES = 2;
/** Tiles a second a sponge crawls while it wiggles. */
const WALK = 1.6;
/** Bat loop frequencies in radians per second and horizontal and vertical amplitudes in tiles. */
const FLIGHT = { pace: 0.9, reach: 2.5, bob: 2.3, rise: 0.6 };
/** Radians a second a bat's wings beat. */
const WINGBEAT = 18;
/** Loop phase offset per critter in seconds. Keeps bats from flying in step. */
const STAGGER = 1.7;

/** Half a bat's size, in tiles. */
const SIZE = 0.4;
/** Overlap margin in tiles. A graze within this margin does not count as contact. */
const GRAZE = 0.08;
/** The hero's share of the contact distance between hero and critter centres. */
const HERO = d.vec2f(HALF[0] - GRAZE, HALF[1] - GRAZE);

/** Texels along each side of a frame of the sponge's sheet, and the column its centre lies on. */
const FRAME = 32;
const MIDDLE = FRAME / 2;
/** Quad half-size in tiles, a frame across, and the quad's centre above the sponge's tile centre. */
const REACH = FRAME / TEXELS / 2;
const LIFT = REACH - 0.5;
/**
 * A wiggle plays the whole sheet, `frames` long at `rate` frames a second. Its second frame shuts
 * the eyes with chance `blink`, and otherwise holds the first.
 */
const WIGGLE = { frames: 6, rate: 8, blink: 0.2 };
/** The frames the sponge lies still in, open-eyed and blinking, before it wiggles. */
const STILL = 2;
/** The frame a squashed sponge shows, its eyes shut. */
const SHUT = 1;
/** Half a sponge's body lying flat, in tiles, with its feet on the floor below its tile centre. */
const BODY = { wide: 11 / TEXELS, tall: 5 / TEXELS };
const SPONGE = d.vec2f(BODY.wide, BODY.tall);
const SPONGE_AT = d.vec2f(0, BODY.tall - 0.5);

/**
 * A fork's hop, over and over: it stands for `still` seconds, winds up shaking for `wind`, flies
 * `far` tiles forward and `high` tiles up for `flight`, and lands crouched for `land`.
 */
const HOP = { still: 0.5, wind: 0.3, flight: 0.4, land: 0.08, far: 1.5, high: 0.5 };
const CYCLE = HOP.still + HOP.wind + HOP.flight + HOP.land;
/** The fork's frames: standing, crouched to jump or from landing, and in flight. */
const POSE = { still: 0, crouch: 1, flight: 2 };
/** Times a second a winding-up fork shakes from side to side, a texel each way. */
const SHAKE = 12;
/** Half a fork's hitbox standing, in tiles, with its foot on the floor below its tile centre. */
const PRONGS = { wide: 5 / TEXELS, tall: 13 / TEXELS };
const FORK_BOX = d.vec2f(PRONGS.wide, PRONGS.tall);
const FORK_AT = d.vec2f(0, PRONGS.tall - 0.5);

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

/**
 * What a critter's pixels read from its quad: the corner in quad units and in texels from the
 * critter's snapped position, its kind, facing and a sponge's or fork's frame.
 */
const Corner = d.struct({
  position: d.vec4f,
  local: d.vec2f,
  texel: d.vec2f,
  kind: d.u32,
  facing: d.f32,
  frame: d.i32,
});

/** Half a critter's hitbox in tiles, and its centre from the critter's position. */
const boxOf = (kind: number) => {
  'use gpu';
  if (kind === BAT) return d.vec2f(SIZE);
  if (kind === FORK) return FORK_BOX;
  return SPONGE;
};
const boxAt = (kind: number) => {
  'use gpu';
  if (kind === BAT) return d.vec2f();
  if (kind === FORK) return FORK_AT;
  return SPONGE_AT;
};

/** How far a ducking hero's contact box shrinks down from the top, in tiles. */
const CROUCH = HALF[1] / 2;

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
const touchAt = (centre: d.v2f, half: d.v2f) => {
  'use gpu';
  // A ducking hero is half its height, on its feet
  const crouch = d.vec2f(0, avatar.$.ducked * CROUCH);
  const touch = HERO - crouch + half;
  const origin = avatar.$.before - crouch;
  const path = avatar.$.after - origin;
  const x = slab(origin.x, path.x, centre.x - touch.x, centre.x + touch.x);
  const y = slab(origin.y, path.y, centre.y - touch.y, centre.y + touch.y);

  const enter = std.max(x.x, y.x);
  if (enter >= std.min(x.y, y.y)) return d.f32(-1);
  return enter;
};

export interface CrittersOptions extends ArtOptions {
  readonly capacity: number;
  /** The sponge's sheet, frames of `FRAME` texels square in a row, facing left. */
  readonly sponge: Picture;
  /** The fork's sheet, frames of `FRAME` texels square in a row, facing right. */
  readonly fork: Picture;
  /** Seconds per `step`. */
  readonly step: number;
  /** Fills `avatar` for contact detection. */
  readonly avatar: Injectable;
  /** Takes the dust a stomp kicks up. */
  readonly dust: Pick<Particles<typeof Particle>, 'emit'>;
  /** Takes the bubbles sponges blow. */
  readonly bubbles: Pick<Particles<typeof Particle>, 'emit'>;
}

/**
 * Moves sponges, forks and bats on the GPU and detects contacts with the hero. A slide defeats
 * any critter but a sponge, and a stomp any but a fork.
 */
export function createCritters(root: TgpuRoot, options: CrittersOptions) {
  const { capacity, step, time, dust, bubbles } = options;
  // The sheets hold display colours, as the print reads them, so they load unconverted
  const textures = [options.sponge, options.fork].map((picture) =>
    images.create(root, picture, { color: 'linear', mips: false }),
  );
  const spongeSheet = textures[0]!.createView(d.texture2d(d.f32));
  const forkSheet = textures[1]!.createView(d.texture2d(d.f32));
  /** Seconds since the last step, which walkers are drawn ahead by. */
  const ahead = root.createUniform(d.f32);
  let drawnAhead = 0;

  const critters = table.create(root, Critter, {
    capacity,
    defaults: { heading: 1, squashedAt: -1 },
  });
  const live = liveLists.create(root, { ...critters.live(), draws: [quad] });
  const critter = critters.gpu;

  /** Where bat `i` flies at `t` seconds, and which way. */
  const flightAt = (i: number, t: number) => {
    'use gpu';
    const at = t + critter.phase(i);
    const across = std.sin(at * FLIGHT.pace) * FLIGHT.reach;
    return critter.home(i) + d.vec2f(across, std.sin(at * FLIGHT.bob) * FLIGHT.rise);
  };

  /**
   * The frame sponge `i` shows at the frame's time. Each wiggle blinks with chance
   * `WIGGLE.blink`, and otherwise keeps its eyes open while it lies still.
   */
  const frameOf = (i: number) => {
    'use gpu';
    const beat = (time.$ + critter.phase(i)) * WIGGLE.rate;
    const wiggle = std.floor(beat / WIGGLE.frames);
    const frame = std.min(d.i32(beat - wiggle * WIGGLE.frames), WIGGLE.frames - 1);
    if (frame !== SHUT) return frame;
    randf.seed2(d.vec2f(d.f32(i), wiggle));
    return std.select(d.i32(0), d.i32(SHUT), randf.sample() < WIGGLE.blink);
  };

  /** Seconds fork `i` is into its hop at the frame's time, from 0 up to `CYCLE`. */
  const hopOf = (i: number) => {
    'use gpu';
    const at = time.$ + critter.phase(i);
    return at - std.floor(at / CYCLE) * CYCLE;
  };

  /** The frame fork `i` shows at the frame's time. */
  const poseOf = (i: number) => {
    'use gpu';
    const hop = hopOf(i);
    if (hop < HOP.still) return d.i32(POSE.still);
    if (hop >= HOP.still + HOP.wind && hop < CYCLE - HOP.land) return d.i32(POSE.flight);
    return d.i32(POSE.crouch);
  };

  /** Tiles fork `i` stands above the floor at the frame's time, along its hop's arc. */
  const liftOf = (i: number) => {
    'use gpu';
    const flown = (hopOf(i) - HOP.still - HOP.wind) / HOP.flight;
    if (flown <= 0 || flown >= 1) return d.f32(0);
    return 4 * HOP.high * flown * (1 - flown);
  };

  /** Tiles fork `i` is shaken sideways by at the frame's time: a texel each way as it winds up. */
  const shakeOf = (i: number) => {
    'use gpu';
    const hop = hopOf(i);
    if (hop < HOP.still || hop >= HOP.still + HOP.wind) return d.f32(0);
    return std.select(-1, d.f32(1), std.fract(time.$ * SHAKE) < 0.5) / TEXELS;
  };

  /**
   * Tiles a second walker `i` moves at the frame's time: a sponge only while it wiggles, and a
   * fork only in flight.
   */
  const paceOf = (i: number) => {
    'use gpu';
    if (critter.kind(i) === FORK) return std.select(0, HOP.far / HOP.flight, liftOf(i) > 0);
    return std.select(0, WALK, frameOf(i) >= STILL);
  };

  /**
   * Where critter `i` is drawn at the frame's time, between steps: a bat on its loop, a sponge or
   * fork ahead along its patrol, a fork up its hop and a squashed critter where it was stomped.
   */
  const shownAt = (i: number) => {
    'use gpu';
    const position = critter.position(i);
    if (critter.squashedAt(i) >= 0) return position;
    const kind = critter.kind(i);
    if (kind === BAT) return flightAt(i, time.$);
    const patrol = critter.patrol(i);
    const x = position.x + critter.heading(i) * paceOf(i) * ahead.$;
    const lift = std.select(0, liftOf(i), kind === FORK);
    return d.vec2f(std.clamp(x, patrol.x, patrol.y), position.y + lift);
  };

  // Read contacts every frame to minimise stomp-bounce latency
  const contacts = counters.create(root, ['stomps', 'hurts', 'slides'], { hz: 60 });

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

  /**
   * Blows a bubble from somewhere on sponge `i`'s body at `at`, and picks when the next one
   * comes, `BUBBLES` a second on average.
   */
  const blow = (i: number, at: d.v2f) => {
    'use gpu';
    randf.seed2(d.vec2f(d.f32(i), time.$ + 0.5));
    const spot = d.vec2f(randf.sample(), randf.sample()) * 2 - 1;
    bubbles.emit(blown(at + SPONGE_AT + spot * SPONGE));
    return time.$ + (0.5 + randf.sample()) / BUBBLES;
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

    const kind = critter.kind(i);
    let lift = d.f32(0);
    if (kind === FORK) {
      // A fork stops at the end of its patrol in flight, and turns once it lands
      const patrol = critter.patrol(i);
      row.position.x = std.clamp(
        row.position.x + row.heading * paceOf(i) * step,
        patrol.x,
        patrol.y,
      );
      lift = liftOf(i);
      const end = std.select(patrol.x, patrol.y, row.heading > 0);
      if (lift === 0 && row.position.x === end) row.heading = -row.heading;
    } else if (kind === WALKER) {
      const patrol = critter.patrol(i);
      const x = row.position.x + row.heading * paceOf(i) * step;
      if (x < patrol.x || x > patrol.y) row.heading = -row.heading;
      row.position.x = std.clamp(x, patrol.x, patrol.y);
      if (time.$ >= row.bubbleAt) row.bubbleAt = blow(i, row.position);
    } else {
      row.position = flightAt(i, time.$);
      row.heading = std.sign(std.cos((time.$ + critter.phase(i)) * FLIGHT.pace));
    }

    const centre = row.position + boxAt(kind) + d.vec2f(0, lift);
    const half = boxOf(kind);
    const enter = touchAt(centre, half);
    if (enter < 0) return;

    const offset = std.mix(avatar.$.before, avatar.$.after, enter) - centre;
    if (avatar.$.sliding > 0 && kind !== WALKER) {
      row.squashedAt = time.$;
      contacts.add.slides(1);
      kickUp(centre, i);
    } else if (kind !== FORK && avatar.$.velocity.y < 0 && offset.y > half.y) {
      row.squashedAt = time.$;
      contacts.add.stomps(1);
      kickUp(centre, i);
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

  /**
   * A sponge's or fork's colour at `local` in quad units, or zero alpha around it: the texel of
   * `frame` of its sheet, facing `facing`, with the frame's bottom row on the floor.
   */
  const sheetAt = (local: d.v2f, kind: number, frame: number, facing: number) => {
    'use gpu';
    // The sponge's sheet faces left, and the fork's right
    const fork = kind === FORK;
    const mirror = std.select(-facing, facing, fork);
    const t = d.vec2i(std.floor(d.vec2f(local.x * mirror, local.y) * MIDDLE));
    const column = t.x + MIDDLE;
    const row = MIDDLE - 1 - t.y;
    if (column < 0 || column >= FRAME || row < 0 || row >= FRAME) return d.vec4f();
    const texel = d.vec2i(frame * FRAME + column, row);
    if (fork) return std.textureLoad(forkSheet.$, texel, 0);
    return std.textureLoad(spongeSheet.$, texel, 0);
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

      // A walker's quad is a tile and a half across, standing on the floor
      const kind = critter.kind(slot);
      const bat = kind === BAT;
      const reach = std.select(REACH, SIZE, bat);
      const lift = std.select(LIFT, 0, bat);
      const local = corner * scale * reach + d.vec2f(0, (scale.y - 1) * reach);
      const fork = kind === FORK;
      // A winding-up fork shakes by whole texels, so its texels stay on the grid
      const shake = std.select(0, shakeOf(slot), fork && !flat);
      const world = snapped(shownAt(slot)) + d.vec2f(shake, lift) + local;
      const shown = std.select(frameOf(slot), poseOf(slot), fork);
      const still = std.select(d.i32(SHUT), d.i32(POSE.still), fork);

      return Corner({
        position: cameras.toClip(d.vec3f(world, 0)),
        local: corner,
        texel: local * TEXELS,
        kind,
        facing: critter.heading(slot),
        frame: std.select(shown, still, flat),
      });
    },

    cutout: (corner) => {
      'use gpu';
      const local = pixelated(corner.local, corner.texel);
      if (corner.kind === BAT) return wingAt(local) > 0 && sdDisk(local, 0.5) > 0;
      return sheetAt(local, corner.kind, corner.frame, corner.facing).w === 0;
    },

    pixelOf: (corner) => {
      'use gpu';
      const local = pixelated(corner.local, corner.texel);
      const facing = corner.facing;
      if (corner.kind !== BAT) {
        return sprite(sheetAt(local, corner.kind, corner.frame, facing).xyz);
      }
      if (wingAt(local) < 0) return sprite(WING);

      const eye = eyesAt(local - d.vec2f(facing * EYES.ahead, EYES.up), facing);
      if (eye.w > 0) return sprite(eye.xyz);
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
        // Keep a walker's body within its patrol's tiles
        const wide = kind === WALKER ? BODY.wide : kind === FORK ? PRONGS.wide : 0.5;
        const inset = Math.max(wide - 0.5, 0);
        const middle = (patrol[0] + patrol[1]) / 2;
        const ends = [Math.min(patrol[0] + inset, middle), Math.max(patrol[1] - inset, middle)];
        critters.add({ kind, home: at, patrol: ends, phase: i * STAGGER, position: at });
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

    /** Draws sponges `seconds` ahead of the last step, such as a fixed-step clock's `ahead`. */
    lead(seconds: number) {
      if (seconds === drawnAhead) return;
      drawnAhead = seconds;
      ahead.write(seconds);
    },

    destroy() {
      for (const texture of textures) texture.destroy();
      critters.destroy();
      live.destroy();
      contacts.destroy();
    },
  };
}

export type Critters = ReturnType<typeof createCritters>;
