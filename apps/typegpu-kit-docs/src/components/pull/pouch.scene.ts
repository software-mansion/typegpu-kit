import { perlin3d } from '@typegpu/noise';
import { common, d, std, tgpu, type TgpuRoot } from 'typegpu';
import { bayerCell } from '../site/dither.ts';
import { bands, film, glint, print } from 'typegpu-kit-pull/foil.scene.ts';
import { BODY_BOX, ELEVATION, GRAINS, HALF, LIGHT, POUCH, STRIP_BOX, VIEW } from './pouch.ts';
import { CELL } from './print.ts';
import { Frame, fromLocal, toLocal } from 'typegpu-kit-pull/vessel.scene.ts';

/** One moment of the pouch, as `PouchInput` gives it, with the colours it prints in. */
const Pouch = d.struct({
  time: d.f32,
  anchor: d.vec2f,
  unit: d.f32,
  body: Frame,
  strip: Frame,
  tear: d.f32,
  hint: d.f32,
  mouth: d.f32,
  sink: d.f32,
  gone: d.f32,
  tint: d.vec4f,
  accent: d.vec4f,
  high: d.vec4f,
  grains: d.arrayOf(d.vec4f, GRAINS),
});

const layout = tgpu.bindGroupLayout({ pouch: { uniform: Pouch } });

const SIN = Math.sin(ELEVATION);
const COS = Math.cos(ELEVATION);
const MISS = 1e9;
const PI = Math.PI;
const TOP = POUCH.top;
const BODY = POUCH.body;
const SEAM = POUCH.seam;
const DEPTH = POUCH.depth;
const SEAL = POUCH.seal;
const CURL = POUCH.curl;
const TOWARD_LIGHT = d.vec3f(...LIGHT);
const TOWARD_VIEW = d.vec3f(...VIEW);
const HALFWAY = d.vec3f(...HALF);
const BODY_MIDDLE = d.vec3f(...BODY_BOX.middle);
const BODY_HALF = d.vec3f(...BODY_BOX.half);
const STRIP_MIDDLE = d.vec3f(...STRIP_BOX.middle);
const STRIP_HALF = d.vec3f(...STRIP_BOX.half);

const Hit = d.struct({ t: d.f32, point: d.vec3f });

const tri = (u: number) => {
  'use gpu';
  return std.abs(u - std.floor(u) - 0.5) * 2;
};

/** The crimped edge's height at `x`: the seal's teeth. */
const edge = (x: number) => {
  'use gpu';
  return TOP - 0.03 * tri(x * 11);
};

/** The torn line's height at `x`, ragged about the seam. */
const cut = (x: number) => {
  'use gpu';
  return SEAM + 0.012 * (tri(x * 13.7) + 0.7 * tri(x * 31.3 + 0.37)) - 0.01;
};

/** The flat strip above the seam, a crimped seal, at point `q` laid flat. */
const flatStrip = (q: d.v3f) => {
  'use gpu';
  return std.max(
    std.max(std.abs(q.x) - 1, q.y - edge(q.x)),
    std.max(cut(q.x) - q.y, std.abs(q.z) - SEAL),
  );
};

const Piece = d.struct({ flat: d.vec3f, turn: d.f32, distance: d.f32 });

/**
 * Strip point `p` as the tear leaves the strip: still sealed right of where the tear has reached,
 * curled up off the seam left of it about a point `CURL` above. Gives the point laid flat, how far
 * round the curl it is, and the distance to the strip.
 */
const stripPiece = (p: d.v3f) => {
  'use gpu';
  const reach = -1 + 2 * layout.$.pouch.tear;
  let best = Piece({ flat: p, turn: 0, distance: std.max(flatStrip(p), reach - p.x) });
  const w = p.xy - d.vec2f(reach, SEAM + CURL);
  const phi = std.atan2(-w.x, -w.y);
  if (reach > -1 && phi > 0) {
    const u = d.vec3f(reach - CURL * phi, SEAM + CURL - std.length(w), p.z);
    const curled = std.max(flatStrip(u), u.x - reach);
    if (curled < best.distance) {
      best = Piece({ flat: u, turn: phi, distance: curled });
    }
  }
  return best;
};

/** The body: a pillow thinning to its seals, puffing open at the mouth once it tears. */
const bodyDistance = (p: d.v3f) => {
  'use gpu';
  const s = layout.$.pouch;
  const across = std.saturate(1 - std.pow(std.abs(p.x), 4));
  const pillow = DEPTH * std.saturate(1 - (p.y / BODY) * (p.y / BODY)) * across;
  const puff = s.mouth * 0.09 * std.smoothstep(BODY - 0.35, SEAM, p.y) * across;
  const thick = std.max(SEAL, pillow + puff);
  return std.max(
    std.max(std.abs(p.x) - 1, p.y - cut(p.x)),
    std.max(-edge(p.x) - p.y, std.abs(p.z) - thick),
  );
};

/** The strip above the seam, peeled up where it is torn. */
const stripDistance = (p: d.v3f) => {
  'use gpu';
  return stripPiece(p).distance;
};

/** Where a ray enters and leaves a box, as x and y; y below x where it misses. */
const span = (o: d.v3f, dir: d.v3f, middle: d.v3f, half: d.v3f) => {
  'use gpu';
  const safe = std.select(dir, d.vec3f(1e-6), std.lt(std.abs(dir), d.vec3f(1e-6)));
  const inv = std.div(d.vec3f(1), safe);
  const a = (middle - half - o) * inv;
  const b = (middle + half - o) * inv;
  const lo = std.min(a, b);
  const hi = std.max(a, b);
  const near = std.max(std.max(lo.x, std.max(lo.y, lo.z)), 0);
  const far = std.min(hi.x, std.min(hi.y, hi.z));
  return d.vec2f(near, far);
};

/** The distance to the body, piece 0, or the strip, piece 1. */
const distanceTo = (p: d.v3f, piece: number) => {
  'use gpu';
  if (piece > 0) {
    return stripDistance(p);
  }
  return bodyDistance(p);
};

/** Marches the ray from `o` along `dir` to `piece`, within its bounds. */
const march = (o: d.v3f, dir: d.v3f, piece: number) => {
  'use gpu';
  const range = std.select(
    span(o, dir, BODY_MIDDLE, BODY_HALF),
    span(o, dir, STRIP_MIDDLE, STRIP_HALF),
    piece > 0,
  );
  const steps = std.select(48, 32, piece > 0);
  let t = range.x;
  let found = d.f32(MISS);
  for (let i = 0; i < steps; i++) {
    if (t > range.y) {
      break;
    }
    const distance = distanceTo(o + dir * t, piece);
    if (distance < 0.002) {
      found = t;
      break;
    }
    t += std.max(distance * 0.6, 0.002);
  }
  return Hit({ t: found, point: o + dir * t });
};

const normalOf = (p: d.v3f, piece: number) => {
  'use gpu';
  const e = 0.004;
  return std.normalize(
    d.vec3f(
      distanceTo(p + d.vec3f(e, 0, 0), piece) - distanceTo(p - d.vec3f(e, 0, 0), piece),
      distanceTo(p + d.vec3f(0, e, 0), piece) - distanceTo(p - d.vec3f(0, e, 0), piece),
      distanceTo(p + d.vec3f(0, 0, e), piece) - distanceTo(p - d.vec3f(0, 0, e), piece),
    ),
  );
};

/**
 * The stamped diamond in its ring on the front, 0 to 1, its strokes `stroke` units wide or more,
 * so that a small pouch keeps them a grain and a half across.
 */
const emblemAt = (x: number, y: number) => {
  'use gpu';
  const stroke = std.max(0.03, (1.5 * CELL) / layout.$.pouch.unit);
  const q = d.vec2f(x, (y - 0.08) * 0.95);
  const a = (std.abs(q.x) + std.abs(q.y) - 0.24) / stroke;
  const b = (std.length(q) - 0.42) / (stroke * 0.7);
  return std.max(std.exp(-a * a), 0.7 * std.exp(-b * b));
};

/** The seam's light: dashes and a notch to tear from, then the tear's bright run and its spark. */
const seamAt = (p: d.v3f, front: number) => {
  'use gpu';
  const s = layout.$.pouch;
  const off = (p.y - SEAM) / 0.02;
  const band = front * std.exp(-off * off);
  const reach = -1 + 2 * s.tear;
  const started = std.step(1e-4, s.tear);
  const torn = started * std.step(p.x, reach);
  const dash = std.step(0.45, std.fract(p.x * 7));
  const notch =
    front * (1 - started) * std.step(p.x, -0.9) * std.step(std.abs(p.y - SEAM), (-0.9 - p.x) * 0.7);
  const live = started * (1 - std.step(1, s.tear));
  const toSpark = d.vec2f(p.x - reach, p.y - SEAM);
  const spark = live * std.exp(-std.dot(toSpark, toSpark) / 0.004);
  const run = (p.x + 1 - 0.7 * std.sin(PI * 0.5 * s.hint)) / 0.1;
  const hinted = std.step(0, s.hint) * band * std.exp(-run * run) * (1 - s.hint);
  return (
    band * (dash * 0.4 * (1 - torn) + torn * 1.3 * (1 - s.gone)) + spark * 2 + notch + hinted * 1.4
  );
};

/** The foil pouch's colour at local point `p`, unprinted. */
const shade = (p: d.v3f, n: d.v3f, w: d.v3f, piece: number, cell: d.v2u) => {
  'use gpu';
  const s = layout.$.pouch;
  const lambert = std.max(std.dot(w, TOWARD_LIGHT), 0);
  const spec = std.pow(std.max(std.dot(w, HALFWAY), 0), 30);
  const rim = std.pow(1 - std.abs(std.dot(w, TOWARD_VIEW)), 2.5) * (0.35 + 0.65 * lambert);
  const seal = std.max(std.step(BODY, std.abs(p.y)), piece);
  const ridge = 0.5 + 0.5 * std.cos(p.x * PI * 22);
  const uv = d.vec2f(p.x * 0.5 + 0.5, 0.5 - p.y / (2 * TOP));
  // The foil's colours shift with the way it faces, so they slide as it turns
  const tilt = w.xy * 1.6;
  const colors = film(uv, tilt);
  const front = std.step(0.2, n.z);
  const emblem = front * (1 - seal) * emblemAt(p.x, p.y);
  const crimp = std.mix(1, 0.55 + 0.45 * ridge, seal);
  const foil =
    (0.03 + 0.55 * std.pow(bands(uv, tilt), 2) * lambert + 1.1 * spec + 0.3 * emblem) * crimp;
  const sparkle = glint(cell, tilt, d.f32(0)) * (0.3 + spec) * 0.6;
  const grey = (0.06 + 0.2 * lambert) * std.mix(1, 0.6 + 0.6 * ridge, seal);
  const metal = d.vec3f(grey * 0.92, grey * 0.98, grey * 1.1);
  const lit = std.mix(
    metal + colors * (foil * 0.45) + d.vec3f(spec * 0.35 + sparkle) + s.accent.xyz * (emblem * 0.3),
    s.accent.xyz,
    std.saturate(rim * 1.1),
  );
  const mouth = (1 - piece) * s.mouth * std.exp(-(SEAM - p.y) / 0.05) * 0.7;
  return lit + s.high.xyz * seamAt(p, front) + s.tint.xyz * (mouth * 1.3);
};

const fragment = tgpu.fragmentFn({
  in: { position: d.builtin.position },
  out: d.vec4f,
})(({ position }) => {
  'use gpu';
  const s = layout.$.pouch;
  const cell = d.vec2u(position.xy);
  const threshold = bayerCell(cell);
  const p = position.xy * CELL;
  const x = (p.x - s.anchor.x) / s.unit;
  const y = (p.y - s.anchor.y) / s.unit;
  const o = d.vec3f(x, -y * COS + 20 * SIN, y * SIN + 20 * COS);
  const forward = d.vec3f(0, -SIN, -COS);

  // The body and the strip, each falling into grain as it goes, in clumps of noise
  const body = march(toLocal(s.body, o - s.body.t.xyz), toLocal(s.body, forward), 0);
  const height = (body.point.y + TOP) / (2 * TOP);
  const clumps = threshold * 0.6 + 0.4 * (0.5 + 0.5 * perlin3d.sample(body.point * 2.5));
  const bodyCut = s.sink * 1.3 - 0.3 * (1 - height);
  let bodyT = d.f32(body.t);
  if (clumps < bodyCut || s.sink >= 1) {
    bodyT = MISS;
  }
  const strip = march(toLocal(s.strip, o - s.strip.t.xyz), toLocal(s.strip, forward), 1);
  const stripCut = s.gone * 1.3 - 0.3;
  let stripT = d.f32(strip.t);
  if (threshold < stripCut || s.gone >= 1) {
    stripT = MISS;
  }

  let color = d.vec3f();
  let alpha = d.f32(0);

  if (std.min(bodyT, stripT) < MISS) {
    let lit = d.vec3f();
    let edgeOfCut = d.f32(0);
    if (bodyT <= stripT) {
      const n = normalOf(body.point, 0);
      lit = shade(body.point, n, fromLocal(s.body, n), d.f32(0), cell);
      edgeOfCut = std.step(clumps, bodyCut + 0.045) * std.step(0.01, s.sink);
    } else {
      // The curl turns the foil's face, so its colours slide round it
      const piece = stripPiece(strip.point);
      const bent = normalOf(strip.point, 1);
      const n = std.normalize(bent + d.vec3f(std.sin(piece.turn * 1.3) * 0.8, 0, 0));
      lit = shade(piece.flat, bent, fromLocal(s.strip, n), d.f32(1), cell);
      edgeOfCut = std.step(threshold, stripCut + 0.07) * std.step(0.01, s.gone);
    }
    // Cells just above a dissolve catch the light as they go
    if (edgeOfCut > 0) {
      lit = std.mix(s.tint.xyz, d.vec3f(1), 0.3);
    }
    color = print(lit, threshold);
    alpha = 1;
  }

  // The grains the tear throws
  for (let k = 0; k < GRAINS; k++) {
    const grain = s.grains[k];
    const near = std.abs(p - grain.xy);
    if (grain.w > threshold * 0.9 && near.x < grain.z && near.y < grain.z) {
      color = std.mix(s.high.xyz, d.vec3f(1), 0.3);
      alpha = 1;
    }
  }

  return d.vec4f(color, alpha);
});

/** The pouch on its ground, its pipeline drawing three vertices over the whole canvas. */
export function createPouch(root: TgpuRoot, format: GPUTextureFormat) {
  const pouch = root.createUniform(Pouch);
  const pipeline = root
    .createRenderPipeline({ vertex: common.fullScreenTriangle, fragment, targets: { format } })
    .with(root.createBindGroup(layout, { pouch }));
  return { pouch, pipeline };
}
