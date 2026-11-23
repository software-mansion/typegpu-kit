import { cameras } from '@typegpu/kit';
import { perlin3d } from '@typegpu/noise';
import { d, std, tgpu } from 'typegpu';

// Colours are in linear light
/** The dark below. */
export const ABYSS = d.vec3f(0.0006, 0.0018, 0.005);
/** The water level with the eye, a deep navy. */
export const NAVY = d.vec3f(0.003, 0.013, 0.029);
/** The water just under the surface, lit through it. */
export const SHALLOW = d.vec3f(0.009, 0.038, 0.065);
/** The light coming through the surface. */
export const SUNLIGHT = d.vec3f(0.78, 0.92, 1);
/** The cold blue-green light a disturbed jelly gives off. */
export const FLASH = d.vec3f(0.2, 0.85, 1);

/** Height of the water's surface above the tank's middle. */
export const SURFACE = 2.4;
/** Direction the light travels through the water, down and a little aside. */
export const LIGHT = std.normalize(d.vec3f(0.18, -1, 0.1));

export const Look = d.struct({
  /** Brightness of the jellies' own glow. */
  glow: d.f32,
  /** How strongly dragging stirs the water. */
  current: d.f32,
});

export const Stir = d.struct({
  /** Where the stir is, in world space. */
  at: d.vec3f,
  /** Its push, in world units a second squared. */
  push: d.vec3f,
});

/** What the tank's drawings give their colour output, premultiplied. */
export const Seen = d.struct({ color: d.vec4f });
export type Seen = d.Infer<typeof Seen>;

/** Seconds since the scene started. */
export const timeOf = tgpu.accessor(d.f32);
export const lookOf = tgpu.accessor(Look);
export const stirOf = tgpu.accessor(Stir);

/** Caustic web on a level plane at `xz`, bright where the waves above focus the light. */
export const caustic = (xz: d.v2f, time: number) => {
  'use gpu';
  const a = 1 - std.abs(perlin3d.sample(d.vec3f(xz * 1.1, time * 0.22)));
  const b = 1 - std.abs(perlin3d.sample(d.vec3f(xz * 1.9 + 7.3, time * 0.29 + 13)));
  const a2 = a * a;
  const b2 = b * b;
  return a2 * a2 * a2 * 0.7 + b2 * b2 * b2 * b2 * 0.5;
};

/** Where light reaching `world` crossed the surface. */
export const throughSurface = (world: d.v3f) => {
  'use gpu';
  return world.xz + LIGHT.xz * ((SURFACE - world.y) / -LIGHT.y);
};

/** Light reaching `world` from above, through the surface's caustics, dimming with depth. */
export const lightAt = (world: d.v3f, time: number) => {
  'use gpu';
  const below = std.max(SURFACE - world.y, 0);
  const web = caustic(throughSurface(world) * 0.8, time);
  return std.exp(-below * 0.38) * (0.6 + web * 1.4);
};

/** The water's colour seen along `direction`, without shafts. */
export const waterAlong = (direction: d.v3f) => {
  'use gpu';
  const up = direction.y;
  const low = std.mix(NAVY, ABYSS, std.smoothstep(0.05, -0.45, up));
  return std.mix(low, SHALLOW, std.smoothstep(0.1, 0.55, up));
};

/** How much of a surface `distance` away the water hides, from 0 to 1. */
export const fogOf = (distance: number) => {
  'use gpu';
  return 1 - std.exp(-std.max(distance - 2.2, 0) * 0.42);
};

/** Premultiplied `lit` of opacity `alpha` at `world`, hazed toward the water behind it. */
export const hazed = (world: d.v3f, lit: d.v3f, alpha: number) => {
  'use gpu';
  const toEye = cameras.current.$.position - world;
  const distance = std.length(toEye);
  const haze = fogOf(distance);
  const water = waterAlong(toEye / -distance);
  return d.vec4f(lit * (1 - haze) + water * (alpha * haze), alpha);
};

/** The push a drag gives the water at `at`. */
export const stirAt = (at: d.v3f) => {
  'use gpu';
  const away = at - stirOf.$.at;
  return stirOf.$.push * (std.exp(-std.dot(away, away) * 1.6) * lookOf.$.current);
};
