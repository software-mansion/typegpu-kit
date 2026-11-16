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
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';
import type { Picture } from '../contract.ts';
import { avatar } from './avatar.ts';
import { blown, DUST, FALL, grain } from './effects.ts';
import { HALF } from './hero.ts';
import { FORK, GRINDER, solidAt, WALKER, type Level } from './level.ts';
import {
  pixelated,
  quad,
  shakenAt,
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
    /** Phase offset in seconds of a grinder's rounds, a sponge's wiggles and a fork's hops. */
    phase: d.f32,
  },
  device: {
    position: d.vec2f,
    heading: d.f32,
    /** Stomp time in seconds, or -1 before a stomp. */
    squashedAt: d.f32,
    /** When a sponge next blows a bubble or a grinding grinder drops pepper, in seconds of the clock. */
    emitAt: d.f32,
    /** 1 while a sponge crawls in a wiggle, a fork flies or a grinder grinds, and 0 otherwise. */
    busy: d.u32,
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
/** Grinder loop frequencies in radians per second and horizontal and vertical amplitudes in tiles. */
const FLIGHT = { pace: 0.9, reach: 2.5, bob: 2.3, rise: 0.6 };
/** Loop phase offset per critter in seconds. Keeps grinders from flying in step. */
const STAGGER = 1.7;
/**
 * A grinder's round, over and over: it flies its loop for `wander` seconds, then stops where it is
 * and grinds pepper for `grind`.
 */
export const ROUND = { wander: 3, grind: 2 };
const ROUNDS = ROUND.wander + ROUND.grind;
/** Frames a second a grinding grinder shows, and times a second a flying one bobs a texel up. */
const GRINDING = 10;
const WOBBLE = 4;
/** The grinder's frames: two grinding, then flying. */
const MILL = { frames: 2, flying: 2 };
/** Half a grinder's hitbox in tiles, centred on its quad. */
const GRINDER_BOX = d.vec2f(7 / TEXELS, 11 / TEXELS);
/** Where pepper leaves a grinder facing right, from its centre in tiles. */
const MOUTH = d.vec2f(6.5 / TEXELS, -12 / TEXELS);
/**
 * Pepper a grinder drops: grains a second on average, and tiles either side of its mouth they
 * fall within, which is half the width of the column that hurts.
 */
const PEPPER = { rate: 30, spread: 3 / TEXELS };
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

/**
 * What a critter's pixels read from its quad: the corner in quad units and in texels from the
 * critter's snapped position, its kind, facing and frame.
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
  if (kind === GRINDER) return GRINDER_BOX;
  if (kind === FORK) return FORK_BOX;
  return SPONGE;
};
const boxAt = (kind: number) => {
  'use gpu';
  if (kind === GRINDER) return d.vec2f();
  if (kind === FORK) return FORK_AT;
  return SPONGE_AT;
};

/**
 * How far in tiles from the hero, across and up or down, critters' sounds are heard: farther than
 * any view reaches, so a sound starts before its critter comes into view, and the scene's mix
 * fades it by its distance from the view.
 */
const HEARD = d.vec2f(24, 12);

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

/** Which way a grinder faces after flying its loop for `flown` seconds: 1 right, -1 left. */
const headingAt = (flown: number) => {
  'use gpu';
  return std.select(d.f32(-1), 1, std.cos(flown * FLIGHT.pace) >= 0);
};

/** Where pepper leaves a grinder at `at` facing `facing`. */
const mouthOf = (at: d.v2f, facing: number) => {
  'use gpu';
  return at + d.vec2f(facing * MOUTH.x, MOUTH.y);
};

/** Whether the hero hears what happens at `at`. */
const heard = (at: d.v2f) => {
  'use gpu';
  return std.all(std.lt(std.abs(at - avatar.$.after), HEARD));
};
/** `at` in texels, for counters to sum. */
const texelsOf = (at: d.v2f) => {
  'use gpu';
  return d.vec2u(std.round(std.max(at, d.vec2f()) * TEXELS));
};

export interface CrittersOptions extends ArtOptions {
  readonly capacity: number;
  /** The level's width in tiles. */
  readonly columns: number;
  /** The sponge's sheet, frames of `FRAME` texels square in a row, facing left. */
  readonly sponge: Picture;
  /** The fork's sheet, frames of `FRAME` texels square in a row, facing right. */
  readonly fork: Picture;
  /** The grinder's sheet, frames of `FRAME` texels square in a row, facing left. */
  readonly grinder: Picture;
  /** Seconds per `step`. */
  readonly step: number;
  /** Fills `avatar` for contact detection. */
  readonly avatar: Injectable;
  /** Takes the dust a stomp kicks up. */
  readonly dust: Pick<Particles<typeof Particle>, 'emit'>;
  /** Takes the bubbles sponges blow. */
  readonly bubbles: Pick<Particles<typeof Particle>, 'emit'>;
  /** Takes the pepper grinders grind, which falls at the speed `FALL`. */
  readonly pepper: Pick<Particles<typeof Particle>, 'emit'>;
}

/**
 * Moves sponges, forks and pepper grinders on the GPU and detects contacts with the hero, and with
 * the pepper the grinders grind. A slide defeats any critter but a sponge, and a stomp any but a
 * fork.
 */
export function createCritters(root: TgpuRoot, options: CrittersOptions) {
  const { capacity, step, time, dust, bubbles, pepper } = options;
  // The sheets hold display colours, as the print reads them, so they load unconverted
  const textures = [options.sponge, options.fork, options.grinder].map((picture) =>
    images.create(root, picture, { color: 'linear', mips: false }),
  );
  const spongeSheet = textures[0]!.createView(d.texture2d(d.f32));
  const forkSheet = textures[1]!.createView(d.texture2d(d.f32));
  const grinderSheet = textures[2]!.createView(d.texture2d(d.f32));
  // The top of each column's highest solid tile, where pepper lands, or 0 over a pit
  const tops = root.createReadonly(d.arrayOf(d.f32, options.columns));
  /** The height in tiles of the ground pepper lands on at `x`. */
  const groundAt = (x: number) => {
    'use gpu';
    return tops.$[std.clamp(d.i32(std.floor(x)), 0, options.columns - 1)];
  };
  /** Seconds since the last step, which walkers are drawn ahead by. */
  const ahead = root.createUniform(d.f32);
  let drawnAhead = 0;

  const critters = table.create(root, Critter, {
    capacity,
    defaults: { heading: 1, squashedAt: -1 },
  });
  const live = liveLists.create(root, { ...critters.live(), draws: [quad] });
  const critter = critters.gpu;

  /**
   * Grinder `i`'s rounds at `t` seconds: how many seconds it has flown, and how many since its
   * last grind began, counted on through the flight after it.
   */
  const roundAt = (i: number, t: number) => {
    'use gpu';
    const at = t + critter.phase(i);
    const rounds = std.floor(at / ROUNDS);
    const into = at - rounds * ROUNDS;
    const flown = rounds * ROUND.wander + std.min(into, ROUND.wander);
    const ground = std.select(into + ROUND.grind, into - ROUND.wander, into >= ROUND.wander);
    return d.vec2f(flown, ground);
  };

  /** Where grinder `i` is after flying its loop for `flown` seconds. */
  const flightAt = (i: number, flown: number) => {
    'use gpu';
    const across = std.sin(flown * FLIGHT.pace) * FLIGHT.reach;
    return critter.home(i) + d.vec2f(across, std.sin(flown * FLIGHT.bob) * FLIGHT.rise);
  };

  /**
   * The column of pepper grinder `i` has ground still falling at the frame's time, as its centre
   * and half size, or a zero half size for none. Pepper leaves the mouth from the grind's start to
   * its end, and each grain falls at `FALL` to the ground under the mouth.
   */
  const pepperOf = (i: number) => {
    'use gpu';
    const round = roundAt(i, time.$);
    const since = round.y;
    const after = std.max(since - ROUND.grind, 0);
    // The grinder has flown on since the grind ended, so find where it stood
    const flown = round.x - after;
    const mouth = mouthOf(flightAt(i, flown), headingAt(flown));
    const top = mouth.y - after * FALL;
    const bottom = std.max(groundAt(mouth.x), mouth.y - since * FALL);
    // No grind began before the clock started
    if (top <= bottom || since > time.$) return d.vec4f();
    return d.vec4f(mouth.x, (top + bottom) / 2, PEPPER.spread, (top - bottom) / 2);
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
    return shakenAt(time.$);
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

  /** The frame grinder `i` shows at the frame's time: grinding in turn, or flying. */
  const millOf = (i: number) => {
    'use gpu';
    if (roundAt(i, time.$).y >= ROUND.grind) return d.i32(MILL.flying);
    return d.i32(time.$ * GRINDING) % MILL.frames;
  };

  /**
   * Where critter `i` is drawn at the frame's time, between steps: a grinder on its loop, a sponge
   * or fork ahead along its patrol, a fork up its hop and a squashed critter where it was stomped.
   */
  const shownAt = (i: number) => {
    'use gpu';
    const position = critter.position(i);
    if (critter.squashedAt(i) >= 0) return position;
    const kind = critter.kind(i);
    if (kind === GRINDER) return flightAt(i, roundAt(i, time.$).x);
    const patrol = critter.patrol(i);
    const x = position.x + critter.heading(i) * paceOf(i) * ahead.$;
    const lift = std.select(0, liftOf(i), kind === FORK);
    return d.vec2f(std.clamp(x, patrol.x, patrol.y), position.y + lift);
  };

  // Read contacts every frame to minimise stomp-bounce latency
  // Of the hurts, `rightward` counts those from a critter on the hero's left, which knock it right.
  // Of the stomps, `sponges` counts sponges. `walks` counts crawls begun near the hero, and
  // `walkedX` and `walkedY` sum where they began, in texels. `grinds`, `groundX` and `groundY` do
  // the same for grinds, `hops` for forks' jumps and `lands` for their landings. `quiets` counts
  // grinders defeated mid-grind, and of the slides, `forks` counts forks. `grindLeft` sums the
  // milliseconds left of each grind heard
  const contacts = counters.create(
    root,
    [
      'stomps',
      'hurts',
      'rightward',
      'slides',
      'sponges',
      'walks',
      'walkedX',
      'walkedY',
      'grinds',
      'groundX',
      'groundY',
      'quiets',
      'hops',
      'hoppedX',
      'hoppedY',
      'lands',
      'landedX',
      'landedY',
      'forks',
      'grindLeft',
    ],
    { hz: 60 },
  );

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

  /**
   * Drops a grain of pepper from the mouth of grinder `i` at `at` facing `facing`, to the ground,
   * and picks when the next one comes, `PEPPER.rate` a second on average.
   */
  const grind = (i: number, at: d.v2f, facing: number) => {
    'use gpu';
    randf.seed2(d.vec2f(d.f32(i), time.$ + 0.25));
    const mouth = mouthOf(at, facing);
    const spot = mouth + d.vec2f((randf.sample() * 2 - 1) * PEPPER.spread, 0);
    pepper.emit(grain(spot, std.max(mouth.y - groundAt(spot.x), 0)));
    return time.$ + (0.5 + randf.sample()) / PEPPER.rate;
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
      const flying = d.u32(lift > 0);
      if (flying !== row.busy && heard(row.position)) {
        const at = texelsOf(row.position);
        if (flying === 1) {
          contacts.add.hops(1);
          contacts.add.hoppedX(at.x);
          contacts.add.hoppedY(at.y);
        } else {
          contacts.add.lands(1);
          contacts.add.landedX(at.x);
          contacts.add.landedY(at.y);
        }
      }
      row.busy = flying;
      const end = std.select(patrol.x, patrol.y, row.heading > 0);
      if (lift === 0 && row.position.x === end) row.heading = -row.heading;
    } else if (kind === WALKER) {
      const patrol = critter.patrol(i);
      const pace = paceOf(i);
      const walking = d.u32(pace > 0);
      if (walking > row.busy && heard(row.position)) {
        const at = texelsOf(row.position);
        contacts.add.walks(1);
        contacts.add.walkedX(at.x);
        contacts.add.walkedY(at.y);
      }
      row.busy = walking;
      const x = row.position.x + row.heading * pace * step;
      if (x < patrol.x || x > patrol.y) row.heading = -row.heading;
      row.position.x = std.clamp(x, patrol.x, patrol.y);
    } else {
      const round = roundAt(i, time.$);
      row.position = flightAt(i, round.x);
      row.heading = headingAt(round.x);
      // A grind is heard from when the hero first comes within earshot of it, until it ends
      const grinding = round.y < ROUND.grind;
      if (grinding && row.busy === 0 && heard(row.position)) {
        const at = texelsOf(row.position);
        contacts.add.grinds(1);
        contacts.add.groundX(at.x);
        contacts.add.groundY(at.y);
        contacts.add.grindLeft(d.u32((ROUND.grind - round.y) * 1000));
        row.busy = 1;
      }
      if (!grinding) row.busy = 0;
    }

    const centre = row.position + boxAt(kind) + d.vec2f(0, lift);
    const half = boxOf(kind);
    const enter = touchAt(centre, half);
    if (enter < 0) {
      // Falling pepper only hurts
      if (kind !== GRINDER) return;
      const column = pepperOf(i);
      if (column.w === 0 || touchAt(column.xy, column.zw) < 0) return;
      contacts.add.hurts(1);
      if (avatar.$.after.x > column.x) contacts.add.rightward(1);
      return;
    }

    const offset = std.mix(avatar.$.before, avatar.$.after, enter) - centre;
    if (avatar.$.sliding > 0 && kind !== WALKER) {
      row.squashedAt = time.$;
      contacts.add.slides(1);
      if (kind === FORK) contacts.add.forks(1);
      if (kind === GRINDER && row.busy === 1) contacts.add.quiets(1);
      kickUp(centre, i);
    } else if (kind !== FORK && avatar.$.velocity.y < 0 && offset.y > half.y) {
      row.squashedAt = time.$;
      contacts.add.stomps(1);
      if (kind === WALKER) contacts.add.sponges(1);
      if (kind === GRINDER && row.busy === 1) contacts.add.quiets(1);
      kickUp(centre, i);
    } else {
      contacts.add.hurts(1);
      if (offset.x > 0) contacts.add.rightward(1);
    }
  });

  // Emitting binds two storage buffers per system, more than the step has room for beside the dust
  const emitKernel = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (!critter.alive(i)) return;

    const row = critter.device.$[i];
    if (row.squashedAt >= 0 || time.$ < row.emitAt) return;

    const kind = critter.kind(i);
    if (kind === WALKER) {
      row.emitAt = blow(i, row.position);
    } else if (kind === GRINDER && roundAt(i, time.$).y < ROUND.grind) {
      row.emitAt = grind(i, row.position, row.heading);
    }
  });

  /**
   * A critter's colour at `local` in quad units, or zero alpha around it: the texel of `frame` of
   * its sheet, facing `facing`.
   */
  const sheetAt = (local: d.v2f, kind: number, frame: number, facing: number) => {
    'use gpu';
    // The sponge's and the grinder's sheets face left, and the fork's right
    const fork = kind === FORK;
    const mirror = std.select(-facing, facing, fork);
    const t = d.vec2i(std.floor(d.vec2f(local.x * mirror, local.y) * MIDDLE));
    const column = t.x + MIDDLE;
    const row = MIDDLE - 1 - t.y;
    if (column < 0 || column >= FRAME || row < 0 || row >= FRAME) return d.vec4f();
    const texel = d.vec2i(frame * FRAME + column, row);
    if (fork) return std.textureLoad(forkSheet.$, texel, 0);
    if (kind === GRINDER) return std.textureLoad(grinderSheet.$, texel, 0);
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

      // A walker's quad stands on the floor, and a grinder's centres on it
      const kind = critter.kind(slot);
      const grinder = kind === GRINDER;
      const local = corner * scale * REACH + d.vec2f(0, (scale.y - 1) * REACH);
      const fork = kind === FORK;
      // A winding-up fork shakes and a flying grinder bobs by whole texels, so its texels stay on
      // the grid
      const shake = std.select(0, shakeOf(slot), fork && !flat);
      let frame = frameOf(slot);
      let still = d.i32(SHUT);
      let lift = d.f32(LIFT);
      if (fork) {
        frame = poseOf(slot);
        still = d.i32(POSE.still);
      } else if (grinder) {
        frame = millOf(slot);
        still = d.i32(MILL.flying);
        const bob = std.select(0, 1 / TEXELS, std.fract(time.$ * WOBBLE) < 0.5);
        lift = std.select(0, bob, frame === MILL.flying && !flat);
      }
      const world = snapped(shownAt(slot)) + d.vec2f(shake, lift) + local;

      return Corner({
        position: cameras.toClip(d.vec3f(world, 0)),
        local: corner,
        texel: local * TEXELS,
        kind,
        facing: critter.heading(slot),
        frame: std.select(frame, still, flat),
      });
    },

    cutout: (corner) => {
      'use gpu';
      const local = pixelated(corner.local, corner.texel);
      return sheetAt(local, corner.kind, corner.frame, corner.facing).w === 0;
    },

    pixelOf: (corner) => {
      'use gpu';
      const local = pixelated(corner.local, corner.texel);
      return sprite(sheetAt(local, corner.kind, corner.frame, corner.facing).xyz);
    },

    color: passes.shaded,
    draw: (pass) => live.draw(pass, 0),
  };

  const stepPipeline = root
    .pipe(options.avatar.inject())
    .createComputePipeline({ compute: stepKernel });
  const emitPipeline = root.createComputePipeline({ compute: emitKernel });

  return {
    table: critters,
    compact: live.compact,
    /**
     * Stomps, slides and hurts the steps count, how many of the hurts came from the hero's left,
     * how many of the stomps were sponges', how many sponges began a crawl and grinders a grind
     * near the hero, forks jumped and landed near it, and the sums of where, in texels, how many
     * grinders were defeated mid-grind, how many of the slides were forks', and the milliseconds
     * left of the grinds heard.
     * Capture them after the steps each frame.
     */
    contacts,

    drawing,
    initAsync: () => pipelines.compile([live, stepPipeline, emitPipeline]),

    reset(level: Level) {
      critters.clear();
      tops.write(
        Array.from({ length: options.columns }, (_, x) => {
          let top = level.height;
          while (top > 0 && !solidAt(level, x, top - 1)) top--;
          return top;
        }),
      );
      level.critters.forEach(({ kind, at, patrol }, i) => {
        // Keep a walker's body within its patrol's tiles
        const wide = kind === WALKER ? BODY.wide : kind === FORK ? PRONGS.wide : 0.5;
        const inset = Math.max(wide - 0.5, 0);
        const middle = (patrol[0] + patrol[1]) / 2;
        const ends = [Math.min(patrol[0] + inset, middle), Math.max(patrol[1] - inset, middle)];
        critters.add({ kind, home: at, patrol: ends, phase: i * STAGGER, position: at });
      });
    },

    /** Records one step after `table.flush()`, then the bubbles and pepper the critters emit. */
    step(pass: TgpuComputePass) {
      const workgroups = Math.ceil(critters.span / WORKGROUP);
      if (workgroups > 0) {
        pass.setPipeline(stepPipeline);
        pass.dispatchWorkgroups(workgroups);
        pass.setPipeline(emitPipeline);
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
      tops.buffer.destroy();
    },
  };
}

export type Critters = ReturnType<typeof createCritters>;
