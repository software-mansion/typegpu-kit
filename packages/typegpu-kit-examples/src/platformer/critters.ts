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
    /** Phase offset in seconds of a bat's loop and a sponge's wiggles. */
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
 * A fork's parts in tiles from its quad's centre, a tile above the floor: the handle it stands on,
 * the bridge its eyes sit on and four tines, two texels wide with gaps of three.
 */
const HANDLE = { at: d.vec2f(0, -0.72), half: d.vec2f(1 / TEXELS, 0.28) };
const BRIDGE = { at: d.vec2f(0, -0.36), half: d.vec2f(8.5 / TEXELS, 0.14), round: 0.12 };
const TINES = {
  inner: 2.5 / TEXELS,
  outer: 7.5 / TEXELS,
  y: -0.03,
  half: d.vec2f(1 / TEXELS, 0.28),
};
/** Half a fork's hitbox in tiles, with its foot on the floor below its tile centre. */
const PRONGS = { wide: 0.5, tall: 0.6 };
const FORK_BOX = d.vec2f(PRONGS.wide, PRONGS.tall);
const FORK_AT = d.vec2f(0, PRONGS.tall - 0.5);
const STEEL = d.vec3f(0.608, 0.678, 0.718);
const GLINT = d.vec3f(0.796, 0.867, 0.918);
const INK = d.vec3f(0.133, 0.125, 0.204);
const forkEyesAt = eyes({ apart: 0.18, radius: 0.09, pupil: 0.05, look: 0.03 });
/** A fork's eyes, above its bridge's centre. */
const FORK_EYES = -0.36;

/** A fork's distance at `local`, in tiles. */
const forkAt = (local: d.v2f) => {
  'use gpu';
  const handle = sdRoundedBox2d(local - HANDLE.at, HANDLE.half, 0);
  const bridge = sdRoundedBox2d(local - BRIDGE.at, BRIDGE.half, BRIDGE.round);
  const across = std.abs(local.x);
  const inner = sdRoundedBox2d(
    d.vec2f(across - TINES.inner, local.y - TINES.y),
    TINES.half,
    1 / TEXELS,
  );
  const outer = sdRoundedBox2d(
    d.vec2f(across - TINES.outer, local.y - TINES.y),
    TINES.half,
    1 / TEXELS,
  );
  return std.min(std.min(handle, bridge), std.min(inner, outer));
};

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
 * critter's snapped position, its kind, facing and a sponge's frame.
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
  readonly sheet: Picture;
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
  // The sheet holds display colours, as the print reads them, so it loads unconverted
  const texture = images.create(root, options.sheet, { color: 'linear', mips: false });
  const sheet = texture.createView(d.texture2d(d.f32));
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

  /**
   * Tiles a second walker `i` moves at the frame's time: a sponge only while it wiggles, and a
   * fork steadily.
   */
  const paceOf = (i: number) => {
    'use gpu';
    if (critter.kind(i) === FORK) return d.f32(WALK);
    return std.select(0, WALK, frameOf(i) >= STILL);
  };

  /**
   * Where critter `i` is drawn at the frame's time, between steps: a bat on its loop, a sponge
   * ahead along its patrol and a squashed critter where it was stomped.
   */
  const shownAt = (i: number) => {
    'use gpu';
    const position = critter.position(i);
    if (critter.squashedAt(i) >= 0) return position;
    if (critter.kind(i) === BAT) return flightAt(i, time.$);
    const patrol = critter.patrol(i);
    const x = position.x + critter.heading(i) * paceOf(i) * ahead.$;
    return d.vec2f(std.clamp(x, patrol.x, patrol.y), position.y);
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
    if (kind !== BAT) {
      const patrol = critter.patrol(i);
      const x = row.position.x + row.heading * paceOf(i) * step;
      if (x < patrol.x || x > patrol.y) row.heading = -row.heading;
      row.position.x = std.clamp(x, patrol.x, patrol.y);
      if (kind === WALKER && time.$ >= row.bubbleAt) row.bubbleAt = blow(i, row.position);
    } else {
      row.position = flightAt(i, time.$);
      row.heading = std.sign(std.cos((time.$ + critter.phase(i)) * FLIGHT.pace));
    }

    const centre = row.position + boxAt(kind);
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
   * A sponge's colour at `local` in quad units, or zero alpha around it: the texel of `frame`,
   * facing `facing`, with the frame's bottom row on the floor.
   */
  const spongeAt = (local: d.v2f, frame: number, facing: number) => {
    'use gpu';
    // The sheet faces left
    const t = d.vec2i(std.floor(d.vec2f(-local.x * facing, local.y) * MIDDLE));
    const column = t.x + MIDDLE;
    const row = MIDDLE - 1 - t.y;
    if (column < 0 || column >= FRAME || row < 0 || row >= FRAME) return d.vec4f();
    return std.textureLoad(sheet.$, d.vec2i(frame * FRAME + column, row), 0);
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
      const world = snapped(shownAt(slot)) + d.vec2f(0, lift) + local;

      return Corner({
        position: cameras.toClip(d.vec3f(world, 0)),
        local: corner,
        texel: local * TEXELS,
        kind,
        facing: critter.heading(slot),
        frame: std.select(frameOf(slot), d.i32(SHUT), flat),
      });
    },

    cutout: (corner) => {
      'use gpu';
      const local = pixelated(corner.local, corner.texel);
      if (corner.kind === BAT) return wingAt(local) > 0 && sdDisk(local, 0.5) > 0;
      // A fork wears a texel of outline around it
      if (corner.kind === FORK) return forkAt(local) > 1 / TEXELS;
      return spongeAt(local, corner.frame, corner.facing).w === 0;
    },

    pixelOf: (corner) => {
      'use gpu';
      const local = pixelated(corner.local, corner.texel);
      const facing = corner.facing;
      if (corner.kind === WALKER) return sprite(spongeAt(local, corner.frame, facing).xyz);
      if (corner.kind === FORK) {
        if (forkAt(local) > 0) return sprite(INK);
        const eye = forkEyesAt(local - d.vec2f(0, FORK_EYES), facing);
        if (eye.w > 0) return sprite(eye.xyz);
        // Light falls from the left, so each part's left texel glints
        return sprite(std.select(STEEL, GLINT, forkAt(local - d.vec2f(1 / TEXELS, 0)) > 0));
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
      texture.destroy();
      critters.destroy();
      live.destroy();
      contacts.destroy();
    },
  };
}

export type Critters = ReturnType<typeof createCritters>;
