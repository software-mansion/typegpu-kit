import { cameras, passes } from '@typegpu/kit';
import { unfilled, type Inject } from '@typegpu/kit/inject';
import { sdRoundedBox2d } from '@typegpu/sdf';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { HALF, type Hero } from './hero.ts';
import {
  eyes,
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
});

/** The hero's state for shaders, such as the critters' contact test. */
export const avatar = tgpu.accessor(Avatar, unfilled("the hero's state", 'avatar', 'createAvatar'));

/** Changes less often than `Avatar`, so it has a uniform of its own. */
const Sprite = d.struct({
  facing: d.f32,
  /** Invulnerability flag, 1 while the hero cannot be hurt and 0 otherwise. */
  blink: d.f32,
});

const BODY = d.vec3f(0.93, 0.36, 0.24);
const LIGHT = d.vec3f(1, 0.62, 0.45);
const OUTLINE = d.vec3f(0.42, 0.12, 0.12);
/** Quad half-size in tiles, including room for stretching. */
const REACH = 0.6;
/**
 * Stretch per tile per second of upward velocity, and maximum stretch as a fraction of height.
 * Downward velocity squashes the body.
 */
const STRETCH = { rate: 0.012, most: 0.15 };
/** Blinks a second while the hero can't be hurt. */
const BLINKS = 8;
/** Eye size and centre offset forward and above the body centre. */
const eyesAt = eyes({ apart: 0.09, radius: 0.07, pupil: 0.035, look: 0.025 });
const EYES = { ahead: 0.1, up: 0.12 };

/** How much the body stretches as it rises, or squashes as it falls. */
const stretchAt = () => {
  'use gpu';
  return std.clamp(avatar.$.velocity.y * STRETCH.rate, -STRETCH.most, STRETCH.most);
};

/** The body's distance at `local`, its feet kept grounded while it stretches or squashes. */
const bodyAt = (local: d.v2f) => {
  'use gpu';
  const stretch = stretchAt();
  const half = d.vec2f(HALF[0] * (1 - stretch), HALF[1] * (1 + stretch));
  return sdRoundedBox2d(local - d.vec2f(0, half.y - HALF[1]), half, 0.14);
};

/** A corner of the hero's quad, in quad units and in texels from the hero's snapped position. */
const Corner = d.struct({ position: d.vec4f, local: d.vec2f, texel: d.vec2f });

/** Sets `pair` to `x, y`. Returns whether it changed. */
const set = (pair: [number, number], x: number, y: number) => {
  if (pair[0] === x && pair[1] === y) return false;
  pair[0] = x;
  pair[1] = y;
  return true;
};

/** Writes hero state to GPU buffers only when it changes and draws its sprite. */
export function createAvatar(root: TgpuRoot, options: ArtOptions) {
  const { time } = options;

  const state = root.createUniform(Avatar);
  const written: Record<keyof d.Infer<typeof Avatar>, [number, number]> = {
    position: [0, 0],
    velocity: [0, 0],
    before: [0, 0],
    after: [0, 0],
  };

  const sprite = root.createUniform(Sprite, { facing: 1, blink: 0 });
  const shown = { facing: 1, blink: 0 };

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
      return bodyAt(local) > 0;
    },

    pixelOf: (corner) => {
      'use gpu';
      const local = pixelated(corner.local, corner.texel);
      const facing = sprite.$.facing;
      const stretch = stretchAt();
      const eye = eyesAt(local - d.vec2f(facing * EYES.ahead, EYES.up + stretch * 0.3), facing);
      if (eye.w > 0) return front(eye.xyz);

      // A texel of outline, light along the top left and a darker band at the feet
      const inside = -bodyAt(local) * TEXELS;
      if (inside < 1) return front(OUTLINE);
      if (inside < 2.2 && local.y - local.x > 0.05) return front(LIGHT);
      return front(BODY * (0.8 + 0.2 * std.step(-0.2, local.y)));
    },

    color: passes.shaded,
    inject,
    draw: (pass) => pass.draw(quad.vertexCount),
  };

  return {
    inject,
    drawing,

    /**
     * Interpolates the hero by `alpha` between fixed steps and records its swept path. Writes
     * position, facing and blink state only when they change.
     */
    update(hero: Hero, alpha: number, blink: boolean) {
      const { position, previous, start, velocity } = hero;
      const x = previous[0] + (position[0] - previous[0]) * alpha;
      const y = previous[1] + (position[1] - previous[1]) * alpha;

      let moved = set(written.position, x, y);
      moved = set(written.velocity, velocity[0], velocity[1]) || moved;
      moved = set(written.before, start[0], start[1]) || moved;
      moved = set(written.after, position[0], position[1]) || moved;
      if (moved) state.write(written);

      const flicker = blink ? 1 : 0;
      if (hero.facing !== shown.facing || flicker !== shown.blink) {
        shown.facing = hero.facing;
        shown.blink = flicker;
        sprite.write(shown);
      }
    },
  };
}
