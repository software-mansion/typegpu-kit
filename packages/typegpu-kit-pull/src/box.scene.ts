import { perlin2d, perlin3d } from '@typegpu/noise';
import { common, d, std, tgpu, type TgpuRoot } from 'typegpu';
import {
  BAND,
  BODY,
  ELEVATION,
  GLYPH,
  GLYPH_CELL,
  GLYPH_TOP,
  LID,
  LIGHT,
  SEAM,
  type BoxInput,
} from './box.ts';
import { bayerCell, CELL, toneOf, type Colors, type Rgb01 } from './dither.ts';
import { Frame, fromLocal, gpuFrame, toLocal } from './vessel.scene.ts';

/** One moment of the box, as `BoxInput` gives it, with the greys and the accent it prints in. */
export const Scene = d.struct({
  time: d.f32,
  /** Pixels of the canvas a cell: 1 where the canvas has a texel a cell. */
  texel: d.f32,
  /** The ground under the box's centre in points of the canvas, and points a unit. */
  anchor: d.vec2f,
  unit: d.f32,
  body: Frame,
  lid: Frame,
  leak: d.f32,
  open: d.f32,
  sink: d.f32,
  lidFade: d.f32,
  glow: d.f32,
  tint: d.vec4f,
  /** The five greys above the ground, darkest first. */
  greys: d.arrayOf(d.vec4f, 5),
  accent: d.vec4f,
});

const layout = tgpu.bindGroupLayout({ scene: { uniform: Scene } });

const SIN = Math.sin(ELEVATION);
const COS = Math.cos(ELEVATION);
const MISS = 1e9;
const BODY_HALF = d.vec3f(...BODY.half);
const BODY_Y = BODY.center[1];
const LID_HALF = d.vec3f(...LID.half);
const LID_Y = LID.half[1];
const TOWARD_LIGHT = d.vec3f(...LIGHT);

const Hit = d.struct({ t: d.f32, normal: d.vec3f, point: d.vec3f });

/** The ray from `o` along `dir` against a box of `half` extents at the origin. */
const slab = (o: d.v3f, dir: d.v3f, half: d.v3f) => {
  'use gpu';
  const safe = std.select(dir, d.vec3f(1e-6), std.lt(std.abs(dir), d.vec3f(1e-6)));
  const inv = d.vec3f(1) / safe;
  const a = (std.neg(half) - o) * inv;
  const b = (half - o) * inv;
  const lo = std.min(a, b);
  const hi = std.max(a, b);
  const near = std.max(lo.x, std.max(lo.y, lo.z));
  const far = std.min(hi.x, std.min(hi.y, hi.z));
  const axis = std.step(d.vec3f(near), lo);
  return Hit({
    t: std.select(MISS, near, far >= std.max(near, 0)),
    normal: std.neg(std.sign(dir)) * axis,
    point: o + dir * near,
  });
};

/** 1 where a side's point `a` across and `y` up lies in the stamped diamond, else 0. */
const glyphAt = (a: number, y: number) => {
  'use gpu';
  const i = std.floor(a / GLYPH_CELL + 2.5);
  const j = std.floor((GLYPH_TOP - y) / GLYPH_CELL);
  if (i < 0 || i > 4 || j < 0 || j > 5) {
    return d.f32(0);
  }
  return d.f32((d.u32(GLYPH) >>> d.u32(j * 5 + i)) & 1);
};

/** A lit grey from `value`, 0 to 1, among the five greys above the ground. */
const greyOf = (value: number, threshold: number) => {
  'use gpu';
  return layout.$.scene.greys[toneOf(value, threshold)].xyz;
};

/** The distance from ground point `g` to the box's footprint, in units; negative under it. */
const footprint = (g: d.v3f) => {
  'use gpu';
  const s = layout.$.scene;
  const l = toLocal(s.body, g - s.body.t.xyz);
  const q = std.abs(l.xz) - d.vec2f(1);
  return std.length(std.max(q, d.vec2f())) + std.min(std.max(q.x, q.y), 0);
};

const fragment = tgpu.fragmentFn({
  in: { position: d.builtin.position },
  out: d.vec4f,
})(({ position }) => {
  'use gpu';
  const s = layout.$.scene;
  const cell = std.floor(position.xy / s.texel);
  const threshold = bayerCell(d.vec2u(cell));
  const p = (cell + 0.5) * CELL;
  const x = (p.x - s.anchor.x) / s.unit;
  const y = (p.y - s.anchor.y) / s.unit;
  const o = d.vec3f(x, -y * COS + 20 * SIN, y * SIN + 20 * COS);
  const forward = d.vec3f(0, -SIN, -COS);
  const tint = s.tint.xyz;
  const white = d.vec3f(1);

  // The body, cut by the ground as it sinks and crumbling into grain just above it, and the lid
  const body = slab(
    toLocal(s.body, o - s.body.t.xyz) - d.vec3f(0, BODY_Y, 0),
    toLocal(s.body, forward),
    BODY_HALF,
  );
  const above = o.y - SIN * body.t;
  const crumble = std.saturate(1 - above / BAND) * std.step(0.001, s.sink);
  // The grain falls away in clumps rather than in the matrix's even order
  const clumps = threshold * 0.6 + 0.4 * (0.5 + 0.5 * perlin3d.sample(body.point * 2.5));
  let bodyT = d.f32(body.t);
  if (above < 0 || clumps < crumble * 0.9) {
    bodyT = MISS;
  }
  // The lid fades away in grain before the box sinks
  const lid = slab(toLocal(s.lid, o - s.lid.t.xyz), toLocal(s.lid, forward), LID_HALF);
  let lidT = d.f32(lid.t);
  if (threshold < s.lidFade * 1.3 - 0.3 || s.lidFade >= 1) {
    lidT = MISS;
  }

  let color = d.vec3f();
  let alpha = d.f32(0);

  if (std.min(bodyT, lidT) < MISS) {
    const onBody = bodyT <= lidT;
    let normal = d.vec3f(lid.normal);
    let point = d.vec3f(lid.point);
    let half = d.vec3f(LID_HALF);
    let world = fromLocal(s.lid, lid.normal);
    let rise = lid.point.y;
    if (onBody) {
      normal = d.vec3f(body.normal);
      point = d.vec3f(body.point);
      half = d.vec3f(BODY_HALF);
      world = fromLocal(s.body, body.normal);
      rise = body.point.y + BODY_Y;
    }
    const lambert = std.max(std.dot(world, TOWARD_LIGHT), 0);

    // How near the hit is to its face's edges: the face's own two axes, not the normal's
    const across = (std.abs(point) / half) * (d.vec3f(1) - std.abs(normal));
    const edge = std.max(across.x, std.max(across.y, across.z));
    // Rim light along the top and the upright edges, where the light catches them
    const upper = d.vec3f(
      across.x,
      (std.max(point.y, 0) / half.y) * (1 - std.abs(normal.y)),
      across.z,
    );
    const rim = std.smoothstep(0.93, 0.99, std.max(upper.x, std.max(upper.y, upper.z))) * lambert;
    const groove = 1 - std.smoothstep(0.012, 0.03, std.abs(edge - 0.8));
    const side = 1 - std.abs(normal.y);
    const isBody = std.select(d.f32(0), 1, onBody);
    const top = isBody * std.step(0.5, normal.y);
    const under = (1 - isBody) * std.step(0.5, -normal.y);
    const mark = isBody * side * glyphAt(normal.z * point.x - normal.x * point.z, rise);

    // The seam between body and lid, flickering along its length, lit from inside
    let seam = std.exp(-(rise + LID_Y) / 0.05);
    if (onBody) {
      seam = std.exp(-(SEAM - rise) / 0.05);
    }
    const flicker =
      0.75 + 0.45 * perlin2d.sample(d.vec2f((point.x + point.z) * 2.6, s.time * 1.7 + rise));
    const lit =
      s.leak * (seam * side * flicker * 1.4 + top * 0.5 * rim) +
      mark * 0.95 * s.leak +
      top * s.open * (1.15 - 0.6 * edge) +
      under * s.open * 0.7;

    // Cells about to crumble catch the light as they go
    const sparks = isBody * std.step(0.001, crumble) * std.step(clumps, crumble + 0.12);

    alpha = 1;
    if (lit > threshold || sparks > 0) {
      color = std.mix(tint, white, std.min(lit * 0.35, 0.6));
    } else if (rim > threshold) {
      color = std.mix(s.greys[3].xyz, s.accent.xyz, 0.6);
    } else {
      const shade = isBody * side * 0.14 * (1 - rise / SEAM);
      color = greyOf(0.1 + 0.75 * lambert - shade - groove * 0.22 - mark * 0.3, threshold);
    }
  } else {
    // The ground: a pool of light around the footprint
    const g = o + forward * (o.y / SIN);
    const outside = footprint(g);
    const spread = 0.3 + std.max(s.body.t.y, 0) * 1.2;
    const near = std.exp(-std.max(outside, 0) / spread) * std.step(-0.1, outside) * s.glow;
    // Where the sinking box meets the ground, a line of the tell's light
    const line = std.exp(-std.abs(outside) / 0.12) * std.sin(Math.PI * s.sink) * 0.8;
    const ground = std.max(near * (1 - s.sink), line);

    if (ground > threshold * 1.15) {
      const hue = std.mix(s.accent.xyz, tint, std.min(std.max(s.leak * 2, std.ceil(s.sink)), 1));
      alpha = 0.4 + 0.6 * ground;
      color = std.mix(s.greys[0].xyz, hue, 0.45 + 0.55 * ground) * alpha;
    }
  }

  return d.vec4f(color, alpha);
});

const rgba = (c: Rgb01) => d.vec4f(c[0], c[1], c[2], 1);

/** The box on its ground in `colors`, its pipeline drawing three vertices over the whole canvas. */
export function createBox(root: TgpuRoot, format: GPUTextureFormat, colors: Colors) {
  const scene = root.createUniform(Scene);
  const pipeline = root
    .createRenderPipeline({ vertex: common.fullScreenTriangle, fragment, targets: { format } })
    .with(root.createBindGroup(layout, { scene }));
  const value = Scene();
  value.greys = colors.grey.slice(1, 6).map(rgba);
  value.accent = rgba(colors.accent[0]);
  value.tint.w = 1;

  return {
    pipeline,
    /** Poses the box by `input` at `time` seconds, `texel` pixels a cell, allocating nothing. */
    write(input: BoxInput, time: number, texel: number) {
      value.time = time;
      value.texel = texel;
      value.anchor.x = input.anchor[0];
      value.anchor.y = input.anchor[1];
      value.unit = input.unit;
      gpuFrame(input.body, value.body);
      gpuFrame(input.lid, value.lid);
      value.leak = input.leak;
      value.open = input.open;
      value.sink = input.sink;
      value.lidFade = input.lidFade;
      value.glow = input.glow;
      value.tint.x = input.tint[0];
      value.tint.y = input.tint[1];
      value.tint.z = input.tint[2];
      scene.write(value);
    },
  };
}
