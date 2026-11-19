import { d, std } from 'typegpu';
import { mirrored as mirror } from './backdrop.ts';

/**
 * Ray-march steps through each prop and critter, the distance that counts as a hit, and how much
 * that distance grows per unit along the ray.
 */
export const MARCH = { steps: 56, hit: 0.0003, cone: 0.0005 };
/** Sunlight, the sky's cool fill and the warm bounce off the lawn, in display values. */
const LIGHT = {
  sun: d.vec3f(1.2, 1.04, 0.86),
  fill: d.vec3f(0.7, 0.86, 1.05),
  bounce: d.vec3f(0.85, 0.9, 0.6),
};
/** A soft window light high behind the view, for the broad highlight on round candy and the jelly. */
export const SOFTBOX = d.vec3f(-0.3, 0.85, 0.43);
/** How fast jelly deepens to its own colour with thickness, and how much sun shows through. */
const GUMMY = { density: 2.4, through: 1.6 };

/** Candy colours for icing, flowers and gumdrops. */
export const CANDY = {
  white: d.vec3f(1, 0.97, 0.94),
  pink: d.vec3f(1, 0.5, 0.68),
  lilac: d.vec3f(0.75, 0.6, 1),
  butter: d.vec3f(1, 0.86, 0.35),
  sky: d.vec3f(0.45, 0.78, 1),
  middle: d.vec3f(1, 0.75, 0.2),
  cherry: d.vec3f(1, 0.25, 0.38),
};
export const CREAM = d.vec3f(1, 0.93, 0.82);
export const INK = d.vec3f(0.03, 0.02, 0.05);

/** What a point of a prop or critter is made of. */
export const BODY = 0;
export const TRUNK = 1;
export const LINE = 2;
export const EYE = 3;
export const ROOF = 4;
/** An antenna's ball and a sprout's leaf. */
export const TIP = 10;
export const LEAF = 11;

export const Surface = d.struct({ distance: d.f32, material: d.i32 });

/** Where a ray enters and leaves a sphere in xyz with radius w, or a leave at or below 0. */
export const spanOf = (eye: d.v3f, direction: d.v3f, sphere: d.v4f) => {
  'use gpu';
  const from = eye - sphere.xyz;
  const b = std.dot(from, direction);
  const disc = b * b - std.dot(from, from) + sphere.w * sphere.w;
  if (disc <= 0) return d.vec2f(0, -1);
  const half = std.sqrt(disc);
  return d.vec2f(std.max(-b - half, 0), -b + half);
};

/** The normal of shape `i` of an indexed distance field, from four samples `step` apart. */
export const shapeNormal = (distance: (p: d.v3f, i: number) => number, step: number) => {
  const fourTapNormal = (p: d.v3f, i: number) => {
    'use gpu';
    const k = d.vec3f(1, -1, 0) * step;
    return std.normalize(
      k.xyy * distance(p + k.xyy, i) +
        k.yyx * distance(p + k.yyx, i) +
        k.yxy * distance(p + k.yxy, i) +
        k.xxx * distance(p + k.xxx, i),
    );
  };
  return fourTapNormal;
};

/**
 * Pastel jelly lit by `light` travelling that way and seen along `direction`. It deepens to a
 * saturated core where `thick` nears 1, glows where the sun shines through thin parts, and
 * `sunlit` tints the sun where something shades it. `finish` scales its gloss in x and the sun
 * shining through it in y.
 */
export const jelly = (
  base: d.v3f,
  normal: d.v3f,
  direction: d.v3f,
  light: d.v3f,
  thick: number,
  sunlit: d.v3f,
  finish: d.v2f,
) => {
  'use gpu';
  const toLight = light * -1;
  const facing = std.saturate(-std.dot(normal, direction));
  const deep = base * std.exp((base - 1) * (GUMMY.density * thick));
  const wrap = std.saturate(std.dot(normal, toLight) * 0.55 + 0.45);
  const ambient = std.mix(LIGHT.bounce, LIGHT.fill, normal.y * 0.5 + 0.5);
  const scattered = std.saturate(std.dot(direction * -1, std.normalize(light - normal * 0.4))) ** 2;
  const sun = LIGHT.sun * sunlit;
  const through = base * base * (scattered * (1.2 - thick) * GUMMY.through * finish.y);
  const body = deep * (ambient * 0.62 + sun * (wrap * 0.6)) + sun * through;
  // Thin edges glow with the light scattered inside them
  const edge = base * base * ((1 - facing) ** 2 * (1 - thick) * 0.8 * finish.y);

  const mirrored = std.reflect(direction, normal);
  const fresnel = (0.04 + 0.96 * (1 - facing) ** 5) * finish.x;
  const sharp = std.saturate(std.dot(mirrored, toLight)) ** 80 * 2;
  const soft = std.smoothstep(0.8, 0.95, std.dot(mirrored, std.normalize(SOFTBOX))) * 0.45;
  return (
    std.mix(body + edge, mirror(mirrored.y), fresnel * 0.5) +
    (sun * sharp + d.vec3f(soft)) * finish.x
  );
};

/** Jelly whose thickness follows how squarely it faces the view, as a round body's does. */
export const jellyLit = (base: d.v3f, normal: d.v3f, direction: d.v3f, light: d.v3f) => {
  'use gpu';
  const facing = std.saturate(-std.dot(normal, direction));
  return jelly(base, normal, direction, light, facing * 0.6, d.vec3f(1), d.vec2f(1));
};
