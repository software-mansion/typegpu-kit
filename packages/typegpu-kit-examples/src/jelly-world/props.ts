import { table } from '@typegpu/kit';
import * as sdf from '@typegpu/sdf';
import { d, std } from 'typegpu';
import { BODY, CANDY, CREAM, jelly, MARCH, ROOF, shapeNormal, Surface, TRUNK } from './candy.ts';
import { GROUND } from './tray.ts';

export const TREE = 0;
export const HOUSE = 1;
export const LEVER = 2;
/** What a tree grows into. */
export const PUFF = 0;
export const LOLLY = 1;
export const GUMDROP = 2;
export const MUSHROOM = 3;
export const BUSH = 4;

/** Height over the ground of the lever's pivot, and its arm's length to the knob's centre. */
export const PIVOT = 0.075;
export const ARM = 0.17;

/** The jelly world's trees, houses and lever, which the CPU places, grows and swings. */
export const Prop = table.define('Prop', {
  host: {
    /** Where it stands on the ground in x and z. */
    at: d.vec2f,
    /** Its heading in radians, and its size over its modelled size. */
    heading: d.f32,
    size: d.f32,
    kind: d.u32,
    /** What a tree grows into. */
    variant: d.u32,
    /** How far it has grown from 0 to 1, and its squash. */
    grown: d.f32,
    squash: d.f32,
    /** The lever's arm, in radians from upright towards +z. */
    angle: d.f32,
    /** How far a tree's top sways along x. */
    sway: d.f32,
    color: d.vec3f,
  },
});

/**
 * Each prop's bounding sphere before scaling, as its centre's height and its radius, and how wide
 * its shadow falls over that radius. Trees go by what they grow into.
 */
const EXTENTS = {
  puff: d.vec3f(0.17, 0.2, 0.6),
  lolly: d.vec3f(0.19, 0.2, 0.38),
  gumdrop: d.vec3f(0.15, 0.19, 0.55),
  mushroom: d.vec3f(0.09, 0.14, 0.75),
  bush: d.vec3f(0.05, 0.14, 0.9),
  house: d.vec3f(0.12, 0.17, 0.75),
  lever: d.vec3f(0.12, 0.24, 0.35),
};
/** Props cast shadows from a light this much steeper than the scene's, so they stay short. */
const SHADOW = { steep: 0.9 };
/** The houses' lit windows, past white. */
const WINDOW = d.vec3f(1.5, 1.05, 0.5);

const ellipsoid = (q: d.v3f, radii: d.v3f) => {
  'use gpu';
  const k0 = std.length(q / radii);
  const k1 = std.length(q / (radii * radii));
  return (k0 * (k0 - 1)) / k1;
};

const house = (q: d.v3f) => {
  'use gpu';
  const walls = sdf.sdRoundedBox3d(q - d.vec3f(0, 0.07, 0), d.vec3f(0.08, 0.07, 0.07), 0.018);
  const raised = q - d.vec3f(0, 0.14, 0);
  const turned = d.vec3f(
    (raised.x - raised.y) * Math.SQRT1_2,
    (raised.x + raised.y) * Math.SQRT1_2,
    raised.z,
  );
  const ridge = sdf.sdRoundedBox3d(turned, d.vec3f(0.075, 0.075, 0.088), 0.014);
  const chimney = sdf.sdRoundedBox3d(
    q - d.vec3f(0.035, 0.2, -0.03),
    d.vec3f(0.015, 0.035, 0.015),
    0.006,
  );
  const roof = std.min(std.max(ridge, 0.13 - q.y), chimney);
  if (roof < walls) return Surface({ distance: roof, material: ROOF });
  return Surface({ distance: walls, material: BODY });
};

/** A slot machine's arm on a jelly block, swung `angle` radians from upright towards +z. */
const lever = (q: d.v3f, angle: number) => {
  'use gpu';
  const block = sdf.opSmoothUnion(
    sdf.sdRoundedBox3d(q - d.vec3f(0, 0.035, 0), d.vec3f(0.05, 0.035, 0.06), 0.03),
    sdf.sdCapsule(q, d.vec3f(-0.05, PIVOT, 0), d.vec3f(0.05, PIVOT, 0), 0.022),
    0.02,
  );
  const tip = d.vec3f(0, PIVOT + std.cos(angle) * ARM, std.sin(angle) * ARM);
  const arm = sdf.sdCapsule(q, d.vec3f(0, PIVOT, 0), tip, 0.011);
  const knob = sdf.sdSphere(q - tip, 0.032);
  if (knob < block && knob < arm) return Surface({ distance: knob, material: ROOF });
  if (arm < block) return Surface({ distance: arm, material: TRUNK });
  return Surface({ distance: block, material: BODY });
};

const puff = (q: d.v3f) => {
  'use gpu';
  const crown = sdf.opSmoothUnion(
    sdf.opSmoothUnion(
      sdf.sdSphere(q - d.vec3f(0, 0.25, 0), 0.1),
      sdf.sdSphere(q - d.vec3f(0.07, 0.2, 0.03), 0.07),
      0.04,
    ),
    sdf.sdSphere(q - d.vec3f(-0.06, 0.21, -0.04), 0.075),
    0.04,
  );
  const trunk = sdf.sdCapsule(q, d.vec3f(), d.vec3f(0, 0.2, 0), 0.022);
  if (crown < trunk) return Surface({ distance: crown, material: BODY });
  return Surface({ distance: trunk, material: TRUNK });
};

/** A round candy facing local +z on a stick. */
const lolly = (q: d.v3f) => {
  'use gpu';
  const c = q - d.vec3f(0, 0.25, 0);
  const candy = sdf.opExtrudeZ(c, sdf.sdDisk(c.xy, 0.08), 0.008) - 0.02;
  const stick = sdf.sdCapsule(q, d.vec3f(), d.vec3f(0, 0.24, 0), 0.011);
  if (candy < stick) return Surface({ distance: candy, material: BODY });
  return Surface({ distance: stick, material: TRUNK });
};

/** Three gumdrops stacked on a stub, with a bead on top. */
const gumdrop = (q: d.v3f) => {
  'use gpu';
  const tiers = sdf.opSmoothUnion(
    sdf.opSmoothUnion(
      ellipsoid(q - d.vec3f(0, 0.1, 0), d.vec3f(0.1, 0.06, 0.1)),
      ellipsoid(q - d.vec3f(0, 0.175, 0), d.vec3f(0.078, 0.05, 0.078)),
      0.012,
    ),
    ellipsoid(q - d.vec3f(0, 0.24, 0), d.vec3f(0.055, 0.042, 0.055)),
    0.012,
  );
  const bead = sdf.sdSphere(q - d.vec3f(0, 0.292, 0), 0.02);
  const trunk = sdf.sdCapsule(q, d.vec3f(), d.vec3f(0, 0.06, 0), 0.022);
  if (bead < tiers && bead < trunk) return Surface({ distance: bead, material: ROOF });
  if (tiers < trunk) return Surface({ distance: tiers, material: BODY });
  return Surface({ distance: trunk, material: TRUNK });
};

const mushroom = (q: d.v3f) => {
  'use gpu';
  const cap = std.max(sdf.sdSphere(q - d.vec3f(0, 0.06, 0), 0.088), 0.1 - q.y) - 0.012;
  const stem = sdf.sdCapsule(q, d.vec3f(), d.vec3f(0, 0.085, 0), 0.03);
  if (cap < stem) return Surface({ distance: cap, material: BODY });
  return Surface({ distance: stem, material: TRUNK });
};

const bush = (q: d.v3f) => {
  'use gpu';
  const left = sdf.opSmoothUnion(
    sdf.sdSphere(q - d.vec3f(0, 0.06, 0), 0.075),
    sdf.sdSphere(q - d.vec3f(0.075, 0.04, 0.02), 0.055),
    0.03,
  );
  const right = sdf.opSmoothUnion(
    sdf.sdSphere(q - d.vec3f(-0.07, 0.045, -0.015), 0.06),
    sdf.sdSphere(q - d.vec3f(0.01, 0.04, 0.07), 0.05),
    0.03,
  );
  return Surface({ distance: sdf.opSmoothUnion(left, right, 0.03), material: BODY });
};

/**
 * A prop's shape at `q` in its own frame, by its kind and what a tree grows into, how far a tree
 * sways and the lever's angle.
 */
const shapeAt = (q: d.v3f, kind: number, variant: number, sway: number, angle: number) => {
  'use gpu';
  if (kind === HOUSE) return house(q);
  if (kind === LEVER) return lever(q, angle);
  const swayed = q + d.vec3f(sway * q.y * q.y, 0, 0);
  if (variant === LOLLY) return lolly(swayed);
  if (variant === GUMDROP) return gumdrop(swayed);
  if (variant === MUSHROOM) return mushroom(q);
  if (variant === BUSH) return bush(q);
  return puff(swayed);
};

/** A prop's size across and up, grown and stretched by its squash. */
const scaleOf = (k: number) => {
  'use gpu';
  const size = Prop.size(k) * Prop.grown(k);
  const squash = Prop.squash(k);
  return d.vec2f(size * (1 - squash * 0.5), size * (1 + squash));
};

/** Turns a world direction, or a point from the prop's foot, by the prop's heading. */
const unturn = (v: d.v3f, k: number) => {
  'use gpu';
  const c = std.cos(Prop.heading(k));
  const s = std.sin(Prop.heading(k));
  return d.vec3f(c * v.x - s * v.z, v.y, s * v.x + c * v.z);
};

/** Into a prop's frame: heading turned away and size divided out, y stretched by its squash. */
const toLocal = (p: d.v3f, k: number) => {
  'use gpu';
  const at = Prop.at(k);
  return unturn(p - d.vec3f(at.x, GROUND, at.y), k) / scaleOf(k).xyx;
};

const pixelOf = (p: d.v3f, k: number) => {
  'use gpu';
  const scale = scaleOf(k);
  const local = toLocal(p, k);
  const found = shapeAt(local, Prop.kind(k), Prop.variant(k), Prop.sway(k), Prop.angle(k));
  return Surface({
    distance: found.distance * std.min(scale.x, scale.y),
    material: found.material,
  });
};

const distanceOf = (p: d.v3f, k: number) => {
  'use gpu';
  return pixelOf(p, k).distance;
};

const normalOf = shapeNormal(distanceOf, 0.0008);

/** Prop `k`'s entry in `EXTENTS`. */
const extentOf = (k: number) => {
  'use gpu';
  const kind = Prop.kind(k);
  const variant = Prop.variant(k);
  if (kind === HOUSE) return EXTENTS.house;
  if (kind === LEVER) return EXTENTS.lever;
  if (variant === LOLLY) return EXTENTS.lolly;
  if (variant === GUMDROP) return EXTENTS.gumdrop;
  if (variant === MUSHROOM) return EXTENTS.mushroom;
  if (variant === BUSH) return EXTENTS.bush;
  return EXTENTS.puff;
};

/** Whether prop `k` has grown enough to show. */
export const shownProp = (k: number) => {
  'use gpu';
  return Prop.grown(k) >= 0.01;
};

/** A sphere around prop `k`. */
export const boundOf = (k: number) => {
  'use gpu';
  const scale = scaleOf(k);
  const bound = extentOf(k);
  const at = Prop.at(k);
  return d.vec4f(at.x, GROUND + bound.x * scale.y, at.y, bound.y * std.max(scale.x, scale.y));
};

/** Where a ray first meets prop `k` between `start` and `stop`, or -1. */
export const marchProp = (eye: d.v3f, direction: d.v3f, k: number, start: number, stop: number) => {
  'use gpu';
  // The ray moves into the prop's frame once, rather than each point along it
  const scale = scaleOf(k);
  const at = Prop.at(k);
  const origin = unturn(eye - d.vec3f(at.x, GROUND, at.y), k) / scale.xyx;
  const along = unturn(direction, k) / scale.xyx;
  const unit = std.min(scale.x, scale.y);
  const kind = Prop.kind(k);
  const variant = Prop.variant(k);
  const sway = Prop.sway(k);
  const angle = Prop.angle(k);
  let t = start;
  for (let i = 0; i < MARCH.steps; i++) {
    const q = origin + along * t;
    const distance = shapeAt(q, kind, variant, sway, angle).distance * unit;
    if (distance < MARCH.hit + t * MARCH.cone) return t;
    t += distance;
    if (t > stop) break;
  }
  return d.f32(-1);
};

/** The rest of a ray's way through jelly, from 0 at its surface to 1 deep inside. */
const depthOf = (inside: number, probe: number) => {
  'use gpu';
  return std.saturate(-inside / probe);
};

/**
 * A tree's body colour at `q` in its frame, with the swirl of a lolly, the cherries of a puff or
 * the spots of a mushroom.
 */
const decorated = (base: d.v3f, q: d.v3f, variant: number) => {
  'use gpu';
  if (variant === LOLLY) {
    const c = q.xy - d.vec2f(0, 0.25);
    const turns = (std.atan2(c.y, c.x) / Math.PI) * 1.5 + std.length(c) * 16;
    const stripe = std.fract(turns);
    const swirl = std.smoothstep(0.25, 0.32, stripe) * std.smoothstep(0.75, 0.68, stripe);
    return std.mix(base, CANDY.white, swirl);
  }
  if (variant === PUFF) {
    const v = std.normalize(q - d.vec3f(0, 0.22, 0));
    const fruit = std.max(
      std.max(
        std.max(std.dot(v, d.vec3f(0.81, 0.3, 0.51)), std.dot(v, d.vec3f(-0.7, 0.2, 0.68))),
        std.dot(v, d.vec3f(0.2, 0.35, 0.92)),
      ),
      std.max(
        std.max(std.dot(v, d.vec3f(-0.3, 0.4, -0.86)), std.dot(v, d.vec3f(0.84, 0.2, -0.5))),
        std.dot(v, d.vec3f(-0.91, 0.35, -0.2)),
      ),
    );
    return std.mix(base, CANDY.cherry, std.smoothstep(0.984, 0.99, fruit));
  }
  if (variant === MUSHROOM) {
    const v = std.normalize(q - d.vec3f(0, 0.06, 0));
    const spots = std.max(
      std.max(std.dot(v, d.vec3f(0, 1, 0)), std.dot(v, d.vec3f(0.72, 0.6, 0.35))),
      std.max(
        std.max(std.dot(v, d.vec3f(-0.66, 0.55, 0.5)), std.dot(v, d.vec3f(0.1, 0.55, -0.83))),
        std.dot(v, d.vec3f(-0.6, 0.62, -0.5)),
      ),
    );
    return std.mix(base, CANDY.white, std.smoothstep(0.945, 0.96, spots));
  }
  return d.vec3f(base);
};

/** A prop's colour where a ray found it. */
export const shadeProp = (p: d.v3f, direction: d.v3f, k: number, light: d.v3f) => {
  'use gpu';
  const color = Prop.color(k);
  const material = pixelOf(p, k).material;
  const normal = normalOf(p, k);
  const q = toLocal(p, k);
  const kind = Prop.kind(k);
  const variant = Prop.variant(k);
  // How deep the jelly runs on behind this point, from two samples along the ray
  const thick =
    (depthOf(distanceOf(p + direction * 0.03, k), 0.03) +
      depthOf(distanceOf(p + direction * 0.09, k), 0.09)) *
    0.5;

  let base = d.vec3f(color);
  let shine = d.vec3f();
  if (material === TRUNK) base = std.mix(color, CREAM, 0.75);
  if (material === ROOF) base = std.mix(color, d.vec3f(1, 0.35, 0.45), 0.55);
  if (kind === TREE && material === BODY) base = decorated(base, q, variant);
  if (kind === HOUSE && material === BODY) {
    const front = q.z > 0.06;
    const side = std.abs(q.x) > 0.07;
    const pane = std.select(q.zy - d.vec2f(0, 0.08), q.xy - d.vec2f(0.035, 0.08), front);
    const round = std.length(pane);
    if ((front || side) && round < 0.028) {
      base = d.vec3f(CREAM);
      const bar = std.min(std.abs(pane.x), std.abs(pane.y));
      if (round < 0.021 && bar > 0.0035) {
        base = WINDOW * 0.3;
        shine = WINDOW * (0.6 + 0.4 * std.smoothstep(0.021, 0, round));
      }
    }
    const door = d.vec2f(q.x + 0.03, q.y);
    const arch = std.length(door - d.vec2f(0, 0.068));
    if (front && ((std.abs(door.x) < 0.02 && door.y < 0.068) || arch < 0.02)) {
      base = color * color * 0.6;
      if (std.length(door - d.vec2f(0.011, 0.04)) < 0.004) base = d.vec3f(CANDY.butter);
    }
  }
  // Icing along the eaves, dripping a little
  if (kind === HOUSE && material === ROOF && q.y < 0.147 + 0.007 * std.cos((q.x + q.z) * 70)) {
    base = d.vec3f(CANDY.white);
  }
  // Darker where it meets the lawn, so it sits on it
  const grounded = std.mix(0.6, 1, std.smoothstep(0, 0.06, p.y - GROUND));
  return jelly(base, normal, direction, light, thick, d.vec3f(1), d.vec2f(1)) * grounded + shine;
};

/**
 * Short coloured shadows the props cast on a floor point along the light, in rgb, and a soft
 * contact under each in w.
 */
export const castAt = (p: d.v3f, sun: d.v3f) => {
  'use gpu';
  let tint = d.vec3f(1);
  let open = d.f32(1);
  const light = std.normalize(sun + d.vec3f(0, -SHADOW.steep, 0));
  const count = d.i32(Prop.span());
  for (let k = 0; k < count; k++) {
    if (!shownProp(k)) continue;
    const bound = boundOf(k);
    // From the prop's foot out to where its middle falls, widening along the way
    const base = bound.xz;
    const tip = bound.xz + light.xz * ((bound.y - p.y) / -light.y);
    const span = tip - base;
    const share = std.saturate(std.dot(p.xz - base, span) / std.max(std.dot(span, span), 1e-6));
    const gap = std.length(p.xz - base - span * share);
    const spread = bound.w * extentOf(k).z;
    const radius = spread * std.mix(0.6, 1, share);
    const cast = std.smoothstep(radius, radius * 0.5, gap);
    const contact = std.smoothstep(spread, 0, std.length(p.xz - bound.xz)) * 0.4;
    // Jelly passes on some of the sun, in its own colour
    const color = Prop.color(k);
    tint *= std.mix(d.vec3f(1), color * color * 0.6 + 0.15, cast * 0.8);
    open *= 1 - contact;
  }
  return d.vec4f(tint, open);
};
