import { cameras, images, passes } from '@typegpu/kit';
import { unfilled, type Inject } from '@typegpu/kit/inject';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import type { Picture } from '../contract.ts';
import { HALF, HEEL, SKID, type Hero } from './hero.ts';
import {
  pixelated,
  quad,
  snapped,
  sprite as front,
  TEXELS,
  type ArtOptions,
  type Seen,
} from './sprites.ts';

const Avatar = d.struct({
  /** Interpolated draw position between fixed steps. */
  position: d.vec2f,
  velocity: d.vec2f,
  /** Hero positions before and after this frame's steps, used for swept contact tests. */
  before: d.vec2f,
  after: d.vec2f,
  /** 1 while the hero ducks, which halves its height on its feet, and 0 otherwise. */
  ducked: d.f32,
  /** 1 while the hero slides, ducked, which defeats forks and bats it touches, and 0 otherwise. */
  sliding: d.f32,
});

/** The hero's state for shaders, such as the critters' contact test. */
export const avatar = tgpu.accessor(Avatar, unfilled("the hero's state", 'avatar', 'createAvatar'));

/** Changes less often than `Avatar`, so it has a uniform of its own. */
const Sprite = d.struct({
  facing: d.f32,
  /** Invulnerability flag, 1 while the hero cannot be hurt and 0 otherwise. */
  blink: d.f32,
  /** When the hero last landed, in seconds of the clock. */
  landedAt: d.f32,
  /** 1 while the hero stands on the ground and 0 in the air. */
  grounded: d.f32,
  /** When the hero last turned on a heel, in seconds of the clock, or -1 once it can't. */
  turnedAt: d.f32,
  /** When the hero last backflipped, in seconds of the clock. Its spin ends when it lands. */
  flippedAt: d.f32,
  /** 1 while a jump would backflip, crouched in its turn with no direction held, and 0 otherwise. */
  poised: d.f32,
});

/** Texels along each side of a frame of the sheet. */
const FRAME = 32;
/** The sheet's frames in order: their first frame, how many there are and frames a second. */
const IDLE = { first: 0, count: 6, rate: 8 };
const RUN = { first: 6, count: 6, rate: 14 };
/**
 * Rising, around the top of a jump while slower than `apex` tiles a second up or down, and
 * falling.
 */
const JUMP = { up: 12, top: 13, down: 14, apex: 4 };
/** The squash after landing, two frames each shown this many seconds. */
const LANDING = { first: 15, count: 2, hold: 0.07 };
/**
 * Turning on a heel, from when the hero starts to skid, for `hold` seconds or while it skids,
 * unless it crouches for a backflip.
 */
const TURN = { frame: 17, hold: HEEL };
/** The frame the hero shows while it slides, feet first. */
const SLIDING = 18;
/**
 * Ducking, still on its first frame, or sneaking through its frames 0, 1, 0, 2, steps a second.
 * The first frame also shows the hero crouched for a backflip.
 */
const DUCKING = { first: 19, rate: 7 };
/** Seconds a backflip takes to spin once around. */
const SPIN = 0.6;
/** The speed in tiles a second that counts as running. */
const RUNNING = 0.5;
/** The texel row of the hero's feet below its centre, which the sheet's bottom row lands on. */
const FEET = Math.round(-HALF[1] * TEXELS);
/** The column of a frame the hero's centre lies on. */
const MIDDLE = FRAME / 2;
/** Quad half-size in tiles, a frame across. */
const REACH = FRAME / TEXELS / 2;
/** Blinks a second while the hero can't be hurt. */
const BLINKS = 8;

/** A corner of the hero's quad, in quad units and in texels from the hero's snapped position. */
const Corner = d.struct({ position: d.vec4f, local: d.vec2f, texel: d.vec2f });

/** Sets `pair` to `x, y`. Returns whether it changed. */
const set = (pair: [number, number], x: number, y: number) => {
  if (pair[0] === x && pair[1] === y) return false;
  pair[0] = x;
  pair[1] = y;
  return true;
};

export interface AvatarOptions extends ArtOptions {
  /** The hero's sheet, frames of `FRAME` texels square in a row, facing right. */
  readonly sheet: Picture;
}

/** Writes hero state to GPU buffers only when it changes and draws its sprite from a sheet. */
export function createAvatar(root: TgpuRoot, options: AvatarOptions) {
  const { time } = options;
  // The sheet holds display colours, as the print reads them, so it loads unconverted
  const texture = images.create(root, options.sheet, { color: 'linear', mips: false });
  const sheet = texture.createView(d.texture2d(d.f32));

  const state = root.createUniform(Avatar);
  const written = {
    position: [0, 0] as [number, number],
    velocity: [0, 0] as [number, number],
    before: [0, 0] as [number, number],
    after: [0, 0] as [number, number],
    ducked: 0,
    sliding: 0,
  };

  const shown = {
    facing: 1,
    blink: 0,
    landedAt: -1,
    grounded: 1,
    turnedAt: -1,
    flippedAt: -1,
    poised: 0,
  };
  let flipping = false;
  const sprite = root.createUniform(Sprite, shown);

  /** The frame of the sheet the hero shows now. */
  const frameAt = () => {
    'use gpu';
    const since = time.$ - sprite.$.landedAt;
    const velocity = avatar.$.velocity;
    if (avatar.$.sliding > 0) return d.i32(SLIDING);
    if (avatar.$.ducked > 0) {
      if (std.abs(velocity.x) > RUNNING) {
        // Odd steps put a foot forward, the first and then the second
        const step = d.i32(time.$ * DUCKING.rate) % 4;
        return DUCKING.first + std.select(0, d.i32((step + 1) / 2), step % 2 === 1);
      }
      return d.i32(DUCKING.first);
    }
    if (sprite.$.poised > 0) return d.i32(DUCKING.first);
    if (sprite.$.grounded === 0) {
      if (std.abs(velocity.y) < JUMP.apex) return d.i32(JUMP.top);
      return std.select(JUMP.down, JUMP.up, velocity.y > 0);
    }
    if (since >= 0 && since < LANDING.hold * LANDING.count) {
      return LANDING.first + d.i32(since / LANDING.hold);
    }
    const turning = time.$ - sprite.$.turnedAt < TURN.hold;
    if (turning || velocity.x * sprite.$.facing < -SKID) return d.i32(TURN.frame);
    if (std.abs(velocity.x) > RUNNING) {
      return RUN.first + (d.i32(time.$ * RUN.rate) % RUN.count);
    }
    return IDLE.first + (d.i32(time.$ * IDLE.rate) % IDLE.count);
  };

  /**
   * The hero's colour at `local`, or zero alpha around it: the texel of the current frame, facing
   * the hero's way, with the frame's bottom row on the hero's feet. A backflipping hero spins once
   * around its centre, head first away from the way it faces, until it lands.
   */
  const heroAt = (at: d.v2f) => {
    'use gpu';
    const spin = std.saturate((time.$ - sprite.$.flippedAt) / SPIN);
    const spinning = sprite.$.grounded === 0 && sprite.$.flippedAt >= 0 && spin < 1;
    const angle = std.select(0, sprite.$.facing * spin * Math.PI * 2, spinning);
    // Reads the unspun sprite where the spin carried this texel from
    const cos = std.cos(angle);
    const sin = std.sin(angle);
    const local = d.vec2f(cos * at.x + sin * at.y, cos * at.y - sin * at.x);

    const t = d.vec2i(std.floor(d.vec2f(local.x * sprite.$.facing, local.y) * TEXELS));
    const column = t.x + MIDDLE;
    const row = FRAME - 1 - (t.y - FEET);
    if (column < 0 || column >= FRAME || row < 0 || row >= FRAME) return d.vec4f();
    return std.textureLoad(sheet.$, d.vec2i(frameAt() * FRAME + column, row), 0);
  };

  const inject = (): Inject => (cfg) => cfg.with(avatar, state);

  const drawing: passes.Drawing<typeof Corner, Seen> = {
    label: 'hero',
    Vertex: Corner,

    at: (vertexIndex) => {
      'use gpu';
      const local = quad.vertexAt(vertexIndex).position.xz * REACH;
      const world = snapped(avatar.$.position) + local;
      return Corner({
        position: cameras.toClip(d.vec3f(world, 0)),
        local,
        texel: local * TEXELS,
      });
    },

    cutout: (corner) => {
      'use gpu';
      const local = pixelated(corner.local, corner.texel);
      if (sprite.$.blink > 0 && std.fract(time.$ * BLINKS) < 0.5) return true;
      return heroAt(local).w === 0;
    },

    pixelOf: (corner) => {
      'use gpu';
      return front(heroAt(pixelated(corner.local, corner.texel)).xyz);
    },

    color: passes.shaded,
    inject,
    draw: (pass) => pass.draw(quad.vertexCount),
  };

  return {
    inject,
    drawing,

    /**
     * Interpolates the hero by `alpha` between fixed steps and records its swept path, and a
     * landing at `now` seconds when the hero touches down, a turn when it turns against its
     * slide, or a backflip when it starts one. Writes position, facing, blink, whether it stands,
     * its turn, its crouch for a backflip and its backflip only when they change.
     */
    update(hero: Hero, alpha: number, blink: boolean, now: number) {
      const { position, previous, start, velocity } = hero;
      const x = previous[0] + (position[0] - previous[0]) * alpha;
      const y = previous[1] + (position[1] - previous[1]) * alpha;

      let moved = set(written.position, x, y);
      moved = set(written.velocity, velocity[0], velocity[1]) || moved;
      moved = set(written.before, start[0], start[1]) || moved;
      moved = set(written.after, position[0], position[1]) || moved;
      const ducked = hero.ducked ? 1 : 0;
      const sliding = hero.sliding ? 1 : 0;
      moved = moved || ducked !== written.ducked || sliding !== written.sliding;
      written.ducked = ducked;
      written.sliding = sliding;
      if (moved) state.write(written);

      const grounded = hero.grounded ? 1 : 0;
      const landed = grounded === 1 && shown.grounded === 0;
      if (landed) shown.landedAt = now;
      const flicker = blink ? 1 : 0;
      const poised = hero.poised ? 1 : 0;
      const flipped = hero.flipping && !flipping;
      flipping = hero.flipping;
      if (flipped) shown.flippedAt = now;
      if (
        flipped ||
        poised !== shown.poised ||
        grounded !== shown.grounded ||
        hero.facing !== shown.facing ||
        flicker !== shown.blink
      ) {
        // A turn starts when the hero starts to skid, and ends when it leaves the ground
        if (grounded === 0) shown.turnedAt = -1;
        else if (hero.facing !== shown.facing) shown.turnedAt = hero.skidding ? now : -1;
        shown.grounded = grounded;
        shown.facing = hero.facing;
        shown.blink = flicker;
        shown.poised = poised;
        sprite.write(shown);
      }
    },

    destroy() {
      texture.destroy();
    },
  };
}
