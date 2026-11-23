import { table } from '@typegpu/kit';
import { unfilled } from '@typegpu/kit/inject';
import * as sdf from '@typegpu/sdf';
import { d, std, tgpu } from 'typegpu';
import { ANTENNA, CRITTERS, SPROUT } from './body.ts';
import { BODY, EYE, INK, jellyLit, LEAF, LINE, MARCH, shapeNormal, Surface, TIP } from './candy.ts';

/** One critter as the march draws it, in world units, which the CPU poses each frame. */
export const Critter = table.define('Critter', {
  host: {
    /** Body centre and radius. */
    body: d.vec4f,
    /** How far back the body was when the last frame was drawn, in xyz. */
    moved: d.vec4f,
    /** The body's up axis, and how much it stretches along it. */
    up: d.vec4f,
    /** Which way it faces, and how far its eyelids are shut. */
    ahead: d.vec4f,
    /** What it looks at, and how wide its mouth is open. */
    gaze: d.vec4f,
    /** Hip, knee and foot of each leg, then shoulder and hand of each arm. */
    joints: d.arrayOf(d.vec4f, 10),
    /** Colour, and how far it has grown in. */
    color: d.vec4f,
    /** A sphere around all of it. */
    bound: d.vec4f,
    /** How far the top of the body leans over its bottom, across its up axis. */
    bend: d.vec4f,
    /** How happy, surprised, sleepy and dizzy it looks, from 0 to 1. */
    face: d.vec4f,
    /** The tip of what grows on its head, and which: an antenna, a sprout or ears. */
    top: d.vec4f,
  },
});

/** Bit `c` is set for each critter the jelly shows in or through it. */
export const nearJelly = tgpu.accessor(
  d.u32,
  unfilled('the critters near the jelly', 'critters', 'the `createCritters` they belong to'),
);

/** Eye placement and size, and limb thickness, over a critter's body radius. */
const FACE = { ahead: 0.72, up: 0.3, apart: 0.4, eye: 0.36 };
const LIMB = { leg: 0.085, foot: 0.17, arm: 0.075, hand: 0.12 };
/** A critter's sphere in the jelly's reflections over its body radius, and its soft edge. */
const GLIMPSE = { size: 1.15, edge: 0.3 };

/** A critter's side, up and ahead axes as columns. */
const frameOf = (c: number) => {
  'use gpu';
  const up = Critter.up(c).xyz;
  const ahead = Critter.ahead(c).xyz;
  return d.mat3x3f(std.normalize(std.cross(up, ahead)), up, ahead);
};

/** Where a critter's eyes sit, between them. */
const faceOf = (c: number, frame: d.m3x3f) => {
  'use gpu';
  const body = Critter.body(c);
  return (
    body.xyz + frame.columns[2] * (body.w * FACE.ahead) + frame.columns[1] * (body.w * FACE.up)
  );
};

/** A point of the bent body moved back to where it would be upright, as poses.ts bends it. */
const unbent = (p: d.v3f, c: number) => {
  'use gpu';
  const body = Critter.body(c);
  const along = std.dot(p - body.xyz, Critter.up(c).xyz);
  const share = std.saturate((along / body.w + 1) * 0.5);
  return p - Critter.bend(c).xyz * (share * share);
};

/** A leg from the hip through the knee to its foot, for a critter of radius `r`. */
const leg = (p: d.v3f, hip: d.v3f, knee: d.v3f, foot: d.v3f, r: number) => {
  'use gpu';
  const bones = std.min(
    sdf.sdCapsule(p, hip, knee, r * LIMB.leg),
    sdf.sdCapsule(p, knee, foot, r * LIMB.leg),
  );
  return std.min(bones, sdf.sdSphere(p - foot, r * LIMB.foot));
};

/** An arm from the shoulder to its hand, for a critter of radius `r`. */
const arm = (p: d.v3f, shoulder: d.v3f, hand: d.v3f, r: number) => {
  'use gpu';
  return std.min(
    sdf.sdCapsule(p, shoulder, hand, r * LIMB.arm),
    sdf.sdSphere(p - hand, r * LIMB.hand),
  );
};

/** Critter `c`'s distance and material at `p`: its body, eyes, limbs and what grows on its head. */
const critterOf = (p: d.v3f, c: number) => {
  'use gpu';
  const critter = Critter.host.$[c];
  const r = critter.body.w;
  const frame = frameOf(c);
  const q = unbent(p, c);
  const rel = q - critter.body.xyz;
  const along = std.dot(rel, frame.columns[1]);
  const wide = 1 - critter.up.w * 0.5;
  const tall = 1 + critter.up.w;
  const across = std.length(rel - frame.columns[1] * along);
  const body = (std.length(d.vec2f(across / wide, along / tall)) - r) * std.min(wide, tall) * 0.85;

  const face = faceOf(c, frame);
  const apart = frame.columns[0] * (r * FACE.apart);
  const eye = r * FACE.eye * (1 + critter.face.y * 0.25);
  const eyes = std.min(sdf.sdSphere(q - face - apart, eye), sdf.sdSphere(q - face + apart, eye));

  const j = critter.joints;
  const legs = std.min(
    leg(p, j[0].xyz, j[1].xyz, j[2].xyz, r),
    leg(p, j[3].xyz, j[4].xyz, j[5].xyz, r),
  );
  const arms = std.min(arm(p, j[6].xyz, j[7].xyz, r), arm(p, j[8].xyz, j[9].xyz, r));
  const lines = std.min(legs, arms);
  let found = Surface({ distance: body, material: BODY });
  if (eyes < found.distance) found = Surface({ distance: eyes, material: EYE });
  if (lines < found.distance) found = Surface({ distance: lines, material: LINE });

  // What grows on its head, from the top of the bent body to the sprung tip
  const tip = critter.top.xyz;
  const crown = critter.body.xyz + frame.columns[1] * (r * 0.95 * tall) + critter.bend.xyz;
  const kind = d.i32(critter.top.w);
  if (kind === ANTENNA) {
    const stalk = sdf.sdCapsule(p, crown, tip, r * 0.06);
    const ball = sdf.sdSphere(p - tip, r * 0.2);
    if (stalk < found.distance) found = Surface({ distance: stalk, material: LINE });
    if (ball < found.distance) found = Surface({ distance: ball, material: TIP });
  } else if (kind === SPROUT) {
    const sprout = std.min(
      sdf.sdCapsule(p, crown, std.mix(crown, tip, 0.5), r * 0.07),
      sdf.sdCapsule(p, std.mix(crown, tip, 0.4), tip, r * 0.17),
    );
    if (sprout < found.distance) found = Surface({ distance: sprout, material: LEAF });
  } else {
    // Ears splay a little and round off at the tips
    const reach = tip - crown;
    const base = crown - frame.columns[1] * (r * 0.25);
    const out = frame.columns[0] * (r * 0.45);
    const ears = std.min(
      sdf.sdCapsule(p, base + out, crown + out * 1.6 + reach, r * 0.2),
      sdf.sdCapsule(p, base - out, crown - out * 1.6 + reach, r * 0.2),
    );
    if (ears < found.distance) found = Surface({ distance: ears, material: BODY });
  }
  return found;
};

const critterDistance = (p: d.v3f, c: number) => {
  'use gpu';
  return critterOf(p, c).distance;
};

/** Where a ray first meets critter `c` between `start` and `stop`, or -1. */
export const marchCritter = (
  eye: d.v3f,
  direction: d.v3f,
  c: number,
  start: number,
  stop: number,
) => {
  'use gpu';
  let t = start;
  for (let i = 0; i < MARCH.steps; i++) {
    const distance = critterDistance(eye + direction * t, c);
    if (distance < MARCH.hit + t * MARCH.cone) return t;
    t += distance;
    if (t > stop) break;
  }
  return d.f32(-1);
};

const critterNormal = shapeNormal(critterDistance, 0.0004);

/**
 * A critter's colour where a ray found it: jelly body, googly eyes, face and ink limbs. Dizzy
 * pupils spiral round with `time`.
 */
export const shadeCritter = (p: d.v3f, direction: d.v3f, c: number, light: d.v3f, time: number) => {
  'use gpu';
  const critter = Critter.host.$[c];
  const color = critter.color.xyz * 0.82;
  const material = critterOf(p, c).material;
  const normal = critterNormal(p, c);
  const frame = frameOf(c);
  const r = critter.body.w;
  const q = unbent(p, c);
  const mood = critter.face;
  if (material === LINE) return jellyLit(INK, normal, direction, light) * 0.6;
  if (material === TIP) {
    return jellyLit(std.mix(color, d.vec3f(1, 0.95, 0.75), 0.35), normal, direction, light);
  }
  if (material === LEAF) return jellyLit(d.vec3f(0.42, 0.85, 0.38), normal, direction, light);

  if (material === EYE) {
    const face = faceOf(c, frame);
    const side = std.sign(std.dot(q - face, frame.columns[0]));
    const eye = face + frame.columns[0] * (r * FACE.apart * side);
    const out = std.normalize(q - eye);
    const look = std.normalize(critter.gaze.xyz - eye);
    // Surprise shrinks the pupils; dizziness turns them into spirals
    const small = std.mix(0.72, 0.86, mood.y);
    let pupil = std.smoothstep(small, small + 0.08, std.dot(out, look));
    const off = out - look * std.dot(out, look);
    const turn = std.atan2(std.dot(off, frame.columns[1]), std.dot(off, frame.columns[0]));
    const coil = std.fract(turn / (Math.PI * 2) + std.length(off) * 2.5 - time * 1.5);
    const spiral =
      std.smoothstep(0.3, 0.2, std.abs(coil - 0.5)) * std.step(0.2, std.dot(out, look));
    pupil = std.mix(pupil, spiral, mood.w);
    const shine = std.normalize(look + frame.columns[1] * 0.6);
    const glint = std.smoothstep(0.93, 0.96, std.dot(out, shine));
    const white = std.mix(d.vec3f(1), INK, pupil * (1 - glint * (1 - mood.w)));
    // Lids close from above in the body's colour, blinking or sleepy, and shut into a line
    const u = std.dot(out, frame.columns[0]);
    const v = std.dot(out, frame.columns[1]);
    const shut = std.max(critter.ahead.w, mood.z * 0.95);
    const lid = std.smoothstep(0.02, -0.02, 1 - shut * 2.2 - v);
    const lash =
      std.smoothstep(0.07, 0.04, std.abs(v + 0.1 - 0.6 * u * u)) * std.smoothstep(0.7, 0.95, shut);
    let seen = std.mix(std.mix(white, color, lid), INK, lash);
    // Happy eyes squeeze into arcs
    const arc =
      std.smoothstep(0.08, 0.05, std.abs(v - 0.12 + 1.6 * u * u)) * std.step(std.abs(u), 0.5);
    seen = std.mix(seen, std.mix(color, INK, arc), mood.x);
    return jellyLit(seen, normal, direction, light);
  }

  const v = std.normalize(q - critter.body.xyz);
  const across = std.dot(v, frame.columns[0]);
  const up = std.dot(v, frame.columns[1]);
  const front = std.dot(v, frame.columns[2]);
  let base = d.vec3f(color);
  const blush = std.smoothstep(0.12, 0.06, std.length(d.vec2f(std.abs(across) - 0.52, up + 0.08)));
  base = std.mix(base, d.vec3f(1, 0.42, 0.5), blush * std.step(0, front) * (0.7 + mood.x * 0.3));
  if (front > 0.45) {
    // Wider when happy, a little round 'o' when surprised
    const open = std.max(critter.gaze.w, mood.y * 0.55);
    const curve = 1.8 + mood.x * 0.8;
    const smile =
      std.smoothstep(0.035, 0.018, std.abs(up + 0.2 - curve * across * across)) *
      std.step(std.abs(across), 0.26 + mood.x * 0.06);
    const wide = std.mix(0.12 + open * 0.05, 0.075, mood.y);
    const mouth = std.length(d.vec2f(across / wide, (up + 0.26) / (0.02 + open * 0.13)));
    const inside = std.smoothstep(1.05, 0.95, mouth);
    const tongue = std.smoothstep(0.2, 0.6, (up + 0.26) / -(0.02 + open * 0.13));
    const opened = std.mix(INK, d.vec3f(1, 0.4, 0.5), tongue * 0.8);
    if (open > 0.15) base = std.mix(base, opened, inside);
    else base = std.mix(base, INK, smile);
  }
  return jellyLit(base, normal, direction, light);
};

/**
 * The critters near the jelly a ray meets before `limit`, as lit spheres in their colours with
 * their eyes, and how much of the ray they cover, for what the jelly reflects and shows through.
 */
export const glimpse = (eye: d.v3f, direction: d.v3f, limit: number, light: d.v3f) => {
  'use gpu';
  let nearest = d.f32(limit);
  let found = d.i32(-1);
  let cover = d.f32(0);
  for (let c = 0; c < CRITTERS; c++) {
    if (((nearJelly.$ >>> d.u32(c)) & d.u32(1)) === 0) continue;
    const body = Critter.body(c);
    const from = eye - body.xyz;
    const b = std.dot(from, direction);
    const r = body.w * GLIMPSE.size;
    const miss = std.dot(from, from) - b * b;
    if (b > 0 || miss >= r * r) continue;
    const t = -b - std.sqrt(r * r - miss);
    if (t < 0 || t > nearest) continue;
    nearest = t;
    found = c;
    cover = std.saturate((r - std.sqrt(miss)) / (r * GLIMPSE.edge));
  }
  if (found < 0) return d.vec4f();
  const body = Critter.body(found);
  const r = body.w;
  const p = eye + direction * nearest;
  const normal = std.normalize(p - body.xyz);
  const frame = frameOf(found);
  const face = faceOf(found, frame);
  const apart = frame.columns[0] * (r * FACE.apart);
  const eyeAt =
    std.min(std.length(p - face - apart), std.length(p - face + apart)) / (r * FACE.eye);
  const white = std.mix(INK, d.vec3f(1), std.smoothstep(0.4, 0.55, eyeAt));
  const base = std.mix(Critter.color(found).xyz * 0.82, white, std.smoothstep(1.15, 1, eyeAt));
  return d.vec4f(jellyLit(base, normal, direction, light), cover);
};

/** Soft shadows under the critters on a floor point, by their height over it. */
export const underCritters = (p: d.v3f) => {
  'use gpu';
  let shaded = d.f32(1);
  for (let c = 0; c < CRITTERS; c++) {
    const body = Critter.body(c);
    if (body.w < 0.001) continue;
    const height = std.max(body.y - p.y, 0);
    const spread = body.w * (1.1 + height * 4);
    const under = std.smoothstep(spread, spread * 0.3, std.length(p.xz - body.xz));
    shaded *= 1 - under * 0.45 * std.exp(-height * 5);
  }
  return shaded;
};
