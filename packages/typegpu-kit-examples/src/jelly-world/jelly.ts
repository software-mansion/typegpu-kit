import { cameras, history, type CameraBase } from '@typegpu/kit';
import { perlin2d, randf } from '@typegpu/noise';
import * as sdf from '@typegpu/sdf';
import {
  common,
  d,
  std,
  tgpu,
  type TgpuReadonly,
  type TgpuRenderPass,
  type TgpuRoot,
  type TgpuTextureView,
  type TgpuUniform,
} from 'typegpu';
import { fog, sky } from './backdrop.ts';
import { SOFTBOX } from './candy.ts';
import { BOX, POINTS, SKIN } from './slider.ts';
import { GROUND, SLOT } from './tray.ts';

/** How the jelly and its light look, written when a setting changes. */
export const Look = d.struct({
  /** Jelly colour in display values. Its alpha is unused. */
  jelly: d.vec4f,
  /** Unit direction the light travels. */
  light: d.vec3f,
  /** 1 blurs the floor seen through the jelly. */
  blur: d.f32,
});

/** What changes every frame. */
export const Moment = d.struct({
  /** World x of the jelly world's front, and the time in seconds. */
  front: d.f32,
  time: d.f32,
  /** Changes every frame, so the blur's offsets differ between the frames TAA blends. */
  seed: d.vec2f,
});

/** The nearest thing besides the jelly a ray meets, how far along it, and which, or index -1. */
export const Found = d.struct({ t: d.f32, index: d.i32 });

/** A point of the floor as the tray shades it, for scenery to cover. */
export const Floor = d.struct({
  /** Where the ray meets the floor, and where it is shaded, shaken by the blur. */
  at: d.vec3f,
  shaken: d.vec3f,
  normal: d.vec3f,
  direction: d.vec3f,
  /** The tray's colour with the label printed on it. */
  color: d.vec3f,
  /** How much of the point the label covers. */
  ink: d.f32,
  /** The jelly's coloured shadow, and the share of open sky the tray leaves. */
  shadow: d.vec3f,
  open: d.f32,
  /** Light the jelly bounces onto the floor. */
  bounce: d.vec3f,
  /** A pixel's width on the floor. */
  soft: d.f32,
});

/** What the jelly's rays meet besides the jelly and its tray, as `'use gpu'` functions. */
export interface Scenery {
  /**
   * The nearest thing a ray through `pixel` of the march meets before `limit`. The jelly marches
   * only up to it.
   */
  readonly find: (
    eye: d.v3f,
    direction: d.v3f,
    limit: number,
    pixel: d.v2f,
  ) => d.Infer<typeof Found>;
  /** The colour at `p` of the thing `find` found. */
  readonly shade: (p: d.v3f, direction: d.v3f, index: number, light: d.v3f) => d.v3f;
  /** How far back the thing `find` found was when the last frame was drawn. */
  readonly moved: (index: number) => d.v3f;
  /**
   * What the jelly reflects and shows through before `limit`, and how much of the ray it covers in
   * w.
   */
  readonly glimpse: (eye: d.v3f, direction: d.v3f, limit: number, light: d.v3f) => d.v4f;
  /** What covers the floor at a point, from the tray's colour there. */
  readonly ground: (point: d.Infer<typeof Floor>, light: d.v3f) => d.v3f;
}

/**
 * What a ray of the march sees: its colour, how far along the surface is, or -1 for the sky, and
 * how far back the surface was when the last frame was drawn.
 */
const Seen = d.struct({ color: d.vec3f, distance: d.f32, moved: d.vec3f });
/** The motion target's format, and how far along a ray the sky's motion is taken. */
export const MOTION = 'rg16float';
const SKY_DISTANCE = 15;

const clipBefore = history.previously(cameras.toClip);
/** A clip position in UV units, y down, as `taa.motion` takes them. */
const uvOf = (clip: d.v4f) => {
  'use gpu';
  return (clip.xy / clip.w) * d.vec2f(0.5, -0.5);
};

/** Ray-march steps and reach, and the distance that counts as a hit, in world units. */
const MARCH = { steps: 64, far: 10, hit: 0.001 };
/** How far the jelly's reflections reach for critters. */
const REFLECTED = 3;
/** Steps through the jelly to the floor behind it. */
const THROUGH_STEPS = 6;
/** The floor's ambient light, and its highlight's sharpness and strength. */
const AMBIENT = 0.36;
const SPECULAR = { power: 10, intensity: 0.6 };
/** The jelly's index of refraction, and how much light it scatters along a refracted ray. */
const IOR = 1.42;
const SCATTER = 3;
/**
 * Absorption per unit of the jelly's missing colour, the depth light crosses, the least of it at
 * the jelly's start, and how much deeper it runs through the jelly at a glancing view.
 */
const ABSORPTION = { density: 20, depth: 0.08, least: 0.07, glancing: 1 };
/**
 * The room the jelly mirrors, its brightness below and above. Its highlights: the candy's softbox,
 * the cosines its edge fades between and its brightness, and the light's glint, its sharpness and
 * brightness.
 */
const ROOM = {
  low: 0.3,
  high: 0.55,
  softbox: SOFTBOX,
  edge: d.vec2f(0.7, 0.97),
  bright: 0.4,
  glint: d.vec2f(60, 0.9),
};
/** The most the light the jelly gathers brightens the floor, and how much it takes its colour. */
const CAUSTIC = { most: 0.8, tint: 0.6 };
/** Ambient occlusion on the floor: its steps, reach, strength and offset from the surface. */
const AO = { steps: 3, radius: 0.1, intensity: 0.5, bias: MARCH.hit * 5 };
/** The jelly's rounding radius and half its depth across the slider plane. */
const TUBE = { radius: SKIN, half: 0.17 };
/** The floor slab's half-thickness, its top at the ground, and its edges' rounding. */
const SLAB = { thickness: GROUND, rounding: 0.02 };
/** Where the progress along the spine hands over from the tube to the rounded end, and over how much. */
const CAP = { from: 0.94, blend: 0.02 };
/** Rays that would meet the tray farther than this see the sky instead. */
const HORIZON = 14;
/** A pixel's width on the floor over its distance, and at least, for soft-edged marks. */
const FOOTPRINT = { slope: 0.0016, least: 0.0005 };
/** Floor-label centre and size in world units. */
const LABEL = { center: d.vec3f(0.72, 0, 0), width: 0.38, depth: 0.33 };
/** Perlin cells cached per side, after which the lawn's and the sky's noise repeats. */
const NOISE_CELLS = 64;

/** The jelly's distance at a point, and the progress along the spine there. */
const Hit = d.struct({ distance: d.f32, t: d.f32 });
/** The baked field at a point: progress along the spine, distance to it and its gradient. */
const Spine = d.struct({ t: d.f32, distance: d.f32, normal: d.vec2f });

/** The slot the jelly lies in, seen from above. */
const slot = (p: d.v2f) => {
  'use gpu';
  return sdf.sdRoundedBox2d(p, d.vec2f(SLOT.x, SLOT.z), SLOT.z);
};

/** The jelly's body: the spine's distance, pushed out across the slider plane and rounded. */
const tube = (p: d.v3f, along: d.Infer<typeof Spine>) => {
  'use gpu';
  return sdf.opExtrudeZ(p, along.distance, TUBE.half) - TUBE.radius;
};

/** The normal of a distance field, from four samples `step` apart. */
const gradientOf = (distance: (p: d.v3f) => number, step: number) => {
  const fourTapGradient = (p: d.v3f) => {
    'use gpu';
    const k = d.vec3f(1, -1, 0) * step;
    return std.normalize(
      k.xyy * distance(p + k.xyy) +
        k.yyx * distance(p + k.yyx) +
        k.yxy * distance(p + k.yxy) +
        k.xxx * distance(p + k.xxx),
    );
  };
  return fourTapGradient;
};

/** The tray: a slab cut by the slot, over the slot's floor. */
const floor = (p: d.v3f) => {
  'use gpu';
  const slab = sdf.opExtrudeY(p, -slot(p.xz), SLAB.thickness - SLAB.rounding) - SLAB.rounding;
  return sdf.opUnion(sdf.sdPlane(p, d.vec3f(0, 1, 0), -SLOT.floor), slab);
};

const slabNormal = gradientOf(floor, 0.0001);

/** The tray's normal, flat away from the slot. */
const floorNormal = (p: d.v3f) => {
  'use gpu';
  if (std.abs(p.z) > SLOT.z || std.abs(p.x) > SLOT.x) return d.vec3f(0, 1, 0);
  return slabNormal(p);
};

/** The jelly, its slotted floor and the label under it, marched per pixel. */
export function createJelly(
  root: TgpuRoot,
  options: {
    readonly camera: CameraBase;
    readonly look: TgpuUniform<typeof Look>;
    /** The slider's baked field. */
    readonly field: TgpuTextureView<d.WgslTexture2d<d.F32>>;
    /** The slider's last segment, start in xy and end in zw. */
    readonly end: TgpuUniform<d.Vec4f>;
    /** How far back each point of the slider's spine was when the last frame was drawn. */
    readonly moved: TgpuReadonly<d.WgslArray<d.Vec2f>>;
    /** The floor label, white on clear. */
    readonly label: TgpuTextureView<d.WgslTexture2d<d.F32>>;
    /** Moves each frame's rays by a fraction of a pixel, as TAA's `offset`. */
    readonly offset: TgpuUniform<d.Vec2f>;
    readonly now: TgpuUniform<typeof Moment>;
    /** What else the rays meet, each carrying its own fills. */
    readonly scenery: Scenery;
  },
) {
  const { camera, look, field, end, moved, label, offset, now, scenery } = options;
  const sampler = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const noise = perlin2d.staticCache({ root, size: d.vec2u(NOISE_CELLS, NOISE_CELLS) });

  /** The baked field at `p` on the slider plane. */
  const spine = (p: d.v2f) => {
    'use gpu';
    const uv = d.vec2f(
      (p.x - BOX.left) / (BOX.right - BOX.left),
      (BOX.top - p.y) / (BOX.top - BOX.bottom),
    );
    const baked = std.textureSampleLevel(field.$, sampler.$, std.saturate(uv), 0);
    return Spine({ t: baked.y, distance: baked.x, normal: baked.zw });
  };

  /** How far back the jelly at progress `t` along its spine was when the last frame was drawn. */
  const spineMoved = (t: number) => {
    'use gpu';
    const at = std.saturate(t) * (POINTS - 1);
    const i = std.min(d.u32(at), POINTS - 2);
    return std.mix(moved.$[i], moved.$[i + 1], at - d.f32(i));
  };

  /** The jelly's rounded end, around its last segment. */
  const cap = (p: d.v3f) => {
    'use gpu';
    const from = end.$.xy;
    const along = std.normalize(end.$.zw - from);
    const turn = d.mat2x2f(along.x, -along.y, along.y, along.x);
    const local = p - d.vec3f(from, 0);
    const turned = d.vec3f(turn * local.xy, local.z);
    const pie = sdf.sdPie(turned.zx, d.vec2f(1, 0), TUBE.half);
    return sdf.opExtrudeY(turned, pie, 0.001) - TUBE.radius;
  };

  /** The jelly's distance at a point. The floor bounds the march instead of joining it. */
  const scene = (p: d.v3f) => {
    'use gpu';
    const along = spine(p.xy);
    if (along.t > CAP.from) return Hit({ distance: cap(p), t: along.t });
    return Hit({ distance: tube(p, along), t: along.t });
  };

  /** What occludes the floor: the tray and the jelly's tube within its box, skipping the cap. */
  const occluder = (p: d.v3f) => {
    'use gpu';
    let jelly = d.f32(1e9);
    if (p.x >= BOX.left && p.x <= BOX.right && p.y >= BOX.bottom && p.y <= BOX.top) {
      jelly = tube(p, spine(p.xy));
    }
    return std.min(floor(p), jelly);
  };

  const capNormal = gradientOf(cap, 0.01);

  /** Blends the spine's 2D gradient into the flat faces, rounding the tube without a gradient. */
  const tubeNormal = (p: d.v3f, t: number) => {
    'use gpu';
    const along = spine(p.xy);
    const threshold = TUBE.half * 0.85;
    const intoEdge = std.max(0, ((std.abs(p.z) - threshold) * TUBE.half) / (TUBE.half - threshold));
    const edge = 0.9;
    const blend = std.smoothstep(
      edge * TUBE.radius + (1 - edge) * TUBE.half,
      0,
      intoEdge * (1 - edge) + (TUBE.radius - along.distance) * edge,
    );
    const face = d.vec3f(0, 0, std.sign(p.z));
    let normal = std.normalize(std.mix(face, d.vec3f(along.normal, 0), blend * 0.5 + 0.5));
    if (t > CAP.from) {
      normal = std.normalize(std.mix(normal, capNormal(p), (t - CAP.from) / CAP.blend));
    }
    return normal;
  };

  const normalAt = (p: d.v3f, hit: d.Infer<typeof Hit>) => {
    'use gpu';
    if (hit.t < CAP.from + CAP.blend) return tubeNormal(p, hit.t);
    return capNormal(p);
  };

  /** The jelly's coloured shadow, from the baked field along the light, and the slot's lip. */
  const shadowAt = (p: d.v3f, toLight: d.v3f) => {
    'use gpu';
    if (p.y < -SLAB.thickness) {
      const lip = std.saturate(1 - (slot(p.xz) + 0.02) * 30);
      const gradient = std.saturate(-p.z * 4 * toLight.z + 1);
      return d.vec3f(lip * gradient * 0.5);
    }
    const uv = d.vec2f(
      (p.x - p.z * toLight.x * std.sign(toLight.z)) * 0.5 + 0.5,
      1 - (-p.z / toLight.z) * 0.5 - 0.2,
    );
    const baked = std.textureSampleLevel(field.$, sampler.$, uv, 0);
    // Fades into a position-based tint where a bunched-up jelly would switch it too suddenly
    const saturation = std.mix(0, baked.y, std.saturate(p.x * 1.5 + 1.1));
    const tint = std.mix(d.vec3f(), look.$.jelly.rgb, saturation);
    const contrast = 20 * std.saturate(uv.y) * (0.8 + end.$.x * 0.2);
    const feather =
      std.saturate(uv.x * 10) *
      std.saturate((1 - uv.x) * 10) *
      std.saturate((1 - uv.y) * 10) *
      std.saturate(uv.y);
    const reach = std.saturate((1 - toLight.y) * 2) * feather;
    const shade = std.mix(tint, d.vec3f(1), std.saturate(baked.x * contrast - 0.3));
    return std.mix(d.vec3f(1), shade, reach);
  };

  const occlusion = (p: d.v3f, normal: d.v3f) => {
    'use gpu';
    let total = d.f32(0);
    let weight = d.f32(1);
    for (let i = 1; i <= AO.steps; i++) {
      const height = (AO.radius / AO.steps) * d.f32(i);
      total += std.max(0, height - (occluder(p + normal * height) - AO.bias)) * weight;
      weight *= 0.5;
      if (total > AO.radius / AO.intensity) break;
    }
    return std.saturate(1 - (AO.intensity * total) / AO.radius);
  };

  const lit = (p: d.v3f, normal: d.v3f, eye: d.v3f, shadow: d.v3f) => {
    'use gpu';
    const toLight = look.$.light * -1;
    const diffuse = std.max(std.dot(normal, toLight), 0);
    const mirrored = std.reflect(look.$.light, normal);
    const specular =
      std.max(std.dot(std.normalize(eye - p), mirrored), 0) ** SPECULAR.power * SPECULAR.intensity;
    return std.saturate(d.vec3f(0.9) * (diffuse * shadow + AMBIENT) + shadow * specular);
  };

  const labelAt = (p: d.v3f) => {
    'use gpu';
    const local = p - LABEL.center;
    const uv = d.vec2f(local.x / LABEL.width, local.z / LABEL.depth) + 0.5;
    if (uv.x < 0 || uv.x > 1 || uv.y < 0 || uv.y > 1) return d.f32(0);
    return std.textureSampleLevel(label.$, sampler.$, uv, 0).w;
  };

  /** A bright streak where light gathers through the jelly onto the floor. */
  const caustic = (p: d.v3f) => {
    'use gpu';
    const light = look.$.light;
    const shift = d.vec2f(-light.x * 0.2, 0.05 + light.z * 0.2);
    const at = p.xz + shift;
    if (std.abs(at.x) >= 1 || std.abs(at.y) >= 0.2) return d.f32(0);

    const centered = d.vec2f((at.x + 2) * 0.5 - 0.5, (at.y + 0.4) / 0.4 - 0.5);
    const uv = d.vec2f(centered.x, 1 - (std.abs(centered.y - 0.5) * 2) ** 2 * 0.3);
    const baked = std.textureSampleLevel(field.$, sampler.$, uv, 0).x;
    const density = std.max(0, (baked - 0.25) * 8);
    const stretch = (end.$.x + 1) * 0.5;
    const fade =
      std.saturate(std.smoothstep(0, -0.2, p.x - end.$.x)) *
      std.saturate(1 - (std.abs(centered.y - 0.5) * 2) ** 3) *
      std.saturate(1 - stretch);
    return (density ** 3 * fade * 3 * (1 + light.z)) / 1.5;
  };

  const floorColor = (eye: d.v3f, direction: d.v3f, distance: number, blur: number) => {
    'use gpu';
    const p = eye + direction * distance;
    const height = std.saturate(eye.y + 0.01);
    const shaken = p + d.vec3f(0, blur * (height / (1 + height)) * (1 + randf.sample() / 2), 0);
    const normal = floorNormal(shaken);

    // Light the jelly bounces onto the floor around its end
    const near = p - d.vec3f(end.$.x, 0, 0);
    const near2 = std.dot(near, near);
    const bounce =
      look.$.jelly.rgb * (0.4 / (near2 * 15 + 1) + (0.3 / (near2 * 40 + 1)) * std.abs(normal.z));

    const shadow = shadowAt(shaken, look.$.light * -1);
    const open = occlusion(shaken, normal);
    const color = lit(shaken, normal, eye, shadow) * open + bounce;
    const ink = labelAt(p);
    // The gathered light rolls off below a bound and takes on the jelly's colour
    const gathered = caustic(p);
    const glow = (gathered / (1 + gathered)) * CAUSTIC.most;
    const printed =
      std.mix(color, std.saturate(color * 0.5), ink) *
      (1 + std.mix(d.vec3f(1), look.$.jelly.rgb, CAUSTIC.tint) * glow);
    const point = Floor({
      at: p,
      shaken,
      normal,
      direction,
      color: printed,
      ink,
      shadow,
      open,
      bounce,
      soft: distance * FOOTPRINT.slope + FOOTPRINT.least,
    });
    return scenery.ground(point, look.$.light);
  };

  const floorThrough = (eye: d.v3f, direction: d.v3f) => {
    'use gpu';
    let travelled = d.f32(0);
    for (let i = 0; i < THROUGH_STEPS; i++) {
      const step = floor(eye + direction * travelled);
      travelled += step;
      if (travelled > MARCH.far || step < MARCH.hit * 10) break;
    }
    const critter = scenery.glimpse(eye, direction, std.min(travelled, MARCH.far), look.$.light);
    if (critter.w > 0.99) return critter.xyz;
    let behind = d.vec3f();
    if (travelled >= MARCH.far) behind = sky(direction, look.$.light * -1, now.$.time);
    else behind = floorColor(eye, direction, travelled, d.f32(0.87) * look.$.blur);
    return std.mix(behind, critter.xyz, critter.w);
  };

  const jellyColor = (p: d.v3f, direction: d.v3f, hit: d.Infer<typeof Hit>) => {
    'use gpu';
    const normal = normalAt(p, hit);
    const cosine = std.saturate(std.dot(direction * -1, normal));
    const r0 = ((1 - IOR) / (1 + IOR)) ** 2;
    const fresnel = r0 + (1 - r0) * (1 - cosine) ** 5;
    const mirrored = std.reflect(direction, normal);
    const critter = scenery.glimpse(
      p + mirrored * (MARCH.hit * 4),
      mirrored,
      REFLECTED,
      look.$.light,
    );
    const room = std.mix(ROOM.low, ROOM.high, std.smoothstep(-0.2, 0.8, mirrored.y));
    const reflection = std.mix(d.vec3f(room), critter.xyz, critter.w);
    // Bounded highlights that roll off at their edges, over the colour the jelly lets through
    const softbox = std.smoothstep(
      ROOM.edge.x,
      ROOM.edge.y,
      std.dot(mirrored, std.normalize(ROOM.softbox)),
    );
    const glint = std.saturate(std.dot(mirrored, look.$.light * -1)) ** ROOM.glint.x;
    // Squared, the softbox fades in slowly, so the faceted bends of the spine don't show in it
    const shine = (softbox * softbox * ROOM.bright + glint * ROOM.glint.y) * (1 - critter.w);

    const eta = 1 / IOR;
    const k = 1 - eta * eta * (1 - cosine * cosine);
    let refracted = d.vec3f();
    if (k > 0) {
      const bent = std.normalize(direction * eta + normal * (eta * cosine - std.sqrt(k)));
      const behind = floorThrough(p + bent * (MARCH.hit * 4), bent);
      const tint = look.$.jelly.rgb;
      const depth =
        (ABSORPTION.least + hit.t ** 2) *
        ABSORPTION.depth *
        (1 + ABSORPTION.glancing * (1 - cosine));
      const absorbed = std.exp((tint - 1) * ABSORPTION.density * depth);
      const forward = std.max(0, std.dot(look.$.light * -1, bent));
      refracted = behind * absorbed + tint * (1.5 * SCATTER * forward * hit.t ** 3);
    }
    return std.mix(refracted, reflection, fresnel) + d.vec3f(shine);
  };

  /** The jelly's colour where a ray meets it before `limit` and how far along, or -1 in w. */
  const jellyAlong = (eye: d.v3f, direction: d.v3f, limit: number) => {
    'use gpu';
    // The jelly lives inside its baked box, so rays that miss the box skip it
    const inverse = 1 / direction;
    const low = (d.vec3f(BOX.left, BOX.bottom, -BOX.z) - eye) * inverse;
    const high = (d.vec3f(BOX.right, BOX.top, BOX.z) - eye) * inverse;
    const near = std.min(low, high);
    const far = std.max(low, high);
    const enter = std.max(near.x, near.y, near.z);
    const leave = std.min(far.x, far.y, far.z);
    // Past the box the field clamps to its edge, and past `limit` the jelly is hidden
    const stop = std.min(leave, limit);
    if (stop < enter || leave < 0)
      return Seen({ color: d.vec3f(), distance: -1, moved: d.vec3f() });

    let travelled = std.max(0, enter);
    for (let i = 0; i < MARCH.steps; i++) {
      const p = eye + direction * travelled;
      const hit = scene(p);
      travelled += hit.distance;
      if (travelled > stop) break;
      if (hit.distance < MARCH.hit) {
        return Seen({
          color: jellyColor(eye + direction * travelled, direction, hit),
          distance: travelled,
          moved: d.vec3f(spineMoved(hit.t), 0),
        });
      }
    }
    return Seen({ color: d.vec3f(), distance: -1, moved: d.vec3f() });
  };

  /**
   * The colour a ray sees, how far along it the surface is, or -1 for the sky, and how far back
   * the surface was when the last frame was drawn.
   */
  const march = (eye: d.v3f, direction: d.v3f, pixel: d.v2f) => {
    'use gpu';
    const open = direction.y > -1e-3 || (eye.y - SLAB.thickness) / -direction.y > HORIZON;
    let toFloor = d.f32(MARCH.far);
    if (!open) {
      // Nothing of the floor stands above its top
      toFloor = (eye.y - SLAB.thickness) / -direction.y;
      for (let i = 0; i < MARCH.steps; i++) {
        const step = floor(eye + direction * toFloor);
        toFloor += step;
        if (step < MARCH.hit) break;
      }
    }
    const found = scenery.find(eye, direction, toFloor, pixel);
    let limit = toFloor;
    if (found.index >= 0) limit = found.t;
    // The jelly goes first, so what it hides is never shaded
    const jelly = jellyAlong(eye, direction, limit);
    if (jelly.distance >= 0) return jelly;

    if (found.index >= 0) {
      const color = scenery.shade(eye + direction * found.t, direction, found.index, look.$.light);
      return Seen({
        color: fog(color, eye, direction, limit),
        distance: limit,
        moved: scenery.moved(found.index),
      });
    }
    if (open) {
      const color = sky(direction, look.$.light * -1, now.$.time);
      return Seen({ color, distance: -1, moved: d.vec3f() });
    }
    const ground = floorColor(eye, direction, toFloor, d.f32(0));
    return Seen({ color: fog(ground, eye, direction, limit), distance: limit, moved: d.vec3f() });
  };

  const fragment = tgpu.fragmentFn({
    in: { position: d.builtin.position, uv: d.vec2f },
    out: { color: d.vec4f, motion: d.vec4f, depth: d.builtin.fragDepth },
  })(({ position, uv }) => {
    'use gpu';
    randf.seed2(now.$.seed * uv);
    const ndc = d.vec2f(uv.x, 1 - uv.y) * 2 - 1 + offset.$;
    const ray = cameras.rayOf(cameras.current.$, ndc);
    const seen = march(ray.origin, ray.direction, position.xy);
    // Depth and motion let TAA follow the camera, the jelly and the critters
    let depth = d.f32(1);
    let distance = d.f32(SKY_DISTANCE);
    if (seen.distance >= 0) {
      distance = seen.distance;
      const clip = cameras.toClip(ray.origin + ray.direction * distance);
      depth = std.saturate(clip.z / clip.w);
    }
    const at = ray.origin + ray.direction * distance;
    const motion = uvOf(clipBefore(at + seen.moved)) - uvOf(cameras.toClip(at));
    return { color: d.vec4f(seen.color, 1), motion: d.vec4f(motion, 0, 0), depth };
  });

  const pipeline = root
    .pipe(camera.inject())
    .pipe(noise.inject())
    .createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment,
      targets: { color: { format: 'rgba16float' }, motion: { format: MOTION } },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'always' },
    });

  return {
    initAsync: () => pipeline.initAsync(),
    draw(pass: TgpuRenderPass) {
      pass.setPipeline(pipeline);
      pass.draw(3);
    },
    destroy() {
      noise.destroy();
    },
  };
}
