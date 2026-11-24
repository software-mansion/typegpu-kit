import { perlin3d } from '@typegpu/noise';
import { common, d, std, tgpu, type TgpuRenderCommands, type TgpuRoot } from 'typegpu';
import { bayerCell, plain } from './dither.ts';

/** How much of a cell `coverage` inks through `threshold`: all or none, or `coverage` with the dither off. */
const inked = plain
  ? (_threshold: number, coverage: number) => {
      'use gpu';
      return std.saturate(coverage);
    }
  : (threshold: number, coverage: number) => {
      'use gpu';
      return std.step(threshold, coverage);
    };

/** One list's light: where it was, where it goes, and since when, in CSS pixels of the list. */
export const Box = d.struct({
  /** sRGB, 0 to 1. */
  color: d.vec4f,
  /** The lit entry's left, top, right and bottom edges before the move and after it. */
  before: d.vec4f,
  after: d.vec4f,
  /** How lit the entry is before the move and after it, 0 to 1. */
  strength: d.vec2f,
  /** When the move started, in the glow's seconds. */
  start: d.f32,
  /** Device pixels per CSS pixel. */
  ratio: d.f32,
});

/** Most rules one list's light draws. */
const MAX_RULES = 16;

/** The rules down a list's sections, in CSS pixels of the list, which the light runs along. */
export const Rules = d.struct({
  /** sRGB, 0 to 1, where no light reaches. */
  color: d.vec4f,
  /** Each rule's x at its middle, its top and its bottom; the first `count` hold rules. */
  lines: d.arrayOf(d.vec4f, MAX_RULES),
  count: d.u32,
});

/** What `rules.write` takes for `lines`, padded to `MAX_RULES`; lines past it are dropped. */
export const rulesOf = (color: readonly number[], lines: readonly d.v4f[]) => ({
  color: d.vec4f(color[0] ?? 0, color[1] ?? 0, color[2] ?? 0, color[3] ?? 1),
  lines: Array.from({ length: MAX_RULES }, (_, i) => lines[i] ?? d.vec4f()),
  count: Math.min(lines.length, MAX_RULES),
});

const layout = tgpu.bindGroupLayout({ box: { uniform: Box }, rules: { uniform: Rules } });
const seconds = tgpu.accessor(d.f32);

/** CSS pixels over which the light falls to a third, whatever the entry's width. */
const REACH = 70;

/** Seconds the edge ahead takes to reach the next entry, and the edge behind. */
const LEAD = 0.24;
const TRAIL = 0.42;

/** CSS pixels over which the light spilling along a rule falls to a third. */
const SPILL = 32;

/** Seconds a move takes, until the light is still again. */
export const SETTLE = TRAIL;

const easeOut = (k: number) => {
  'use gpu';
  const rest = 1 - std.saturate(k);

  return 1 - rest * rest * rest;
};

const easeInOut = (k: number) => {
  'use gpu';
  const x = std.saturate(k);

  return x * x * (3 - 2 * x);
};

/**
 * The lit band `t` seconds into a move from `from` to `to`. The edge ahead leads and the edge
 * behind trails, so the light stretches toward the next entry and then gathers there.
 */
export const band = (from: d.v4f, to: d.v4f, t: number) => {
  'use gpu';
  const lead = easeOut(t / LEAD);
  const trail = easeInOut(t / TRAIL);
  const down = to.y > from.y;
  const top = std.mix(from.y, to.y, down ? trail : lead);
  const bottom = std.mix(from.w, to.w, down ? lead : trail);

  return d.vec4f(std.mix(from.x, to.x, trail), top, std.mix(from.z, to.z, trail), bottom);
};

/** How lit the band is `t` seconds into a move. */
export const strengthAt = (strength: d.v2f, t: number) => {
  'use gpu';
  return std.mix(strength.x, strength.y, easeOut(t / TRAIL));
};

/** Warped Perlin noise in [0, 1] for drifting smoke. */
const smoke = (p: d.v2f, t: number) => {
  'use gpu';
  const q = (p - d.vec2f(t * 9, t * 1.5)) * 0.045;
  const warp = perlin3d.sample(d.vec3f(q * 0.7, t * 0.11));
  const body = perlin3d.sample(d.vec3f(q + d.vec2f(warp, -warp) * 0.9, t * 0.18));
  const wisps = perlin3d.sample(d.vec3f(q * 2.3, t * 0.27 + 7));

  return std.smoothstep(0.3, 0.85, 0.5 + 0.6 * body + 0.3 * wisps);
};

/** Light fades from a source near the band's left edge. A stretched band uses a line source. */
const beam = (p: d.v2f, lit: d.v4f, height: number, t: number) => {
  'use gpu';
  const middle = (lit.y + lit.w) * 0.5;
  const low = std.min(lit.y + height * 0.5, middle);
  const high = std.max(lit.w - height * 0.5, middle);
  const source = d.vec2f(lit.x - height * 0.8, std.clamp(p.y, low, high));

  const sway = std.sin(t * 0.7) * 0.18;
  const toward = d.vec2f(std.cos(sway), std.sin(sway));
  const from = p - source;
  const along = std.max(std.dot(from, toward), 0);
  const aside = std.abs(from.x * toward.y - from.y * toward.x);
  const cone = 1 - std.smoothstep(along * 0.12, along * 0.4 + 4, aside);

  return cone * std.exp(-along / REACH);
};

/**
 * The rule through `cell`, premultiplied: dots in the rules' colour, which fill in and take the
 * light's colour where the light spills along its section's rule from the lit entry.
 */
const ruleAt = (cell: d.v2u, p: d.v2f, lit: d.v4f, strength: number) => {
  'use gpu';
  const box = layout.$.box;
  const rules = layout.$.rules;
  const middle = (lit.y + lit.w) * 0.5;
  let out = d.vec4f();

  for (let i = d.u32(0); i < rules.count; i++) {
    const line = rules.lines[i];
    if (cell.x !== d.u32(line.x * box.ratio) || p.y < line.y || p.y > line.z) {
      continue;
    }

    // Full beside the entry and fading along the rule; a light between sections fades from both
    const away = std.max(std.max(lit.y - p.y, p.y - lit.w), 0);
    const outside = std.max(std.max(line.y - middle, middle - line.z), 0);
    const spill = strength * std.exp(-away / SPILL) * std.exp(-outside / 8);
    const dots = inked(bayerCell(cell), 0.5 + 0.5 * spill);
    const alpha = dots * std.mix(rules.color.w, 1, spill);
    out = d.vec4f(std.mix(rules.color.xyz, box.color.xyz, spill) * alpha, alpha);
  }

  return out;
};

const fragment = tgpu.fragmentFn({
  in: { position: d.builtin.position },
  out: d.vec4f,
})(({ position }) => {
  'use gpu';
  const box = layout.$.box;
  const t = seconds.$;
  const since = t - box.start;
  const strength = strengthAt(box.strength, since);
  const lit = band(box.before, box.after, since);
  const p = position.xy / box.ratio;
  const rule = ruleAt(d.vec2u(position.xy), p, lit, strength);

  if (strength <= 0 || p.x < lit.x || p.x > lit.z || p.y < lit.y || p.y > lit.w) {
    return rule;
  }

  // Soft at the top and bottom, and where the light comes in
  const edges =
    std.smoothstep(lit.y, lit.y + 5, p.y) *
    (1 - std.smoothstep(lit.w - 5, lit.w, p.y)) *
    std.smoothstep(lit.x, lit.x + 3, p.x) *
    (1 - std.smoothstep(lit.z - 3, lit.z, p.x));
  const height = std.mix(
    box.before.w - box.before.y,
    box.after.w - box.after.y,
    easeInOut(since / TRAIL),
  );

  const ahead = box.after.y > box.before.y ? lit.w - height * 0.5 : lit.y + height * 0.5;
  const behind = std.max(std.abs(p.y - ahead) - height * 0.5, 0);
  const tail = std.exp(-behind / (height * 1.4));
  const light = beam(p, lit, height, t) * (0.08 + 0.92 * smoke(p, t)) * edges * strength * tail;

  // Lit smoke as grains, one per canvas pixel: on where the light passes its threshold, and
  // brighter nearer the source. The light never passes them all, so the brightest smoke keeps its
  // grain
  const grain = inked(bayerCell(d.vec2u(position.xy)), light * 0.8);
  const alpha = grain * (0.22 + 0.32 * light) + light * 0.07;

  return d.vec4f(box.color.xyz * alpha, alpha) + rule * (1 - alpha);
});

/** Draw dithered smoke and section rules behind the current entry. Update `box` on light moves or resizes, `rules` on layout changes, and `time` each frame. */
export function createGlow(root: TgpuRoot, format: GPUTextureFormat) {
  const time = root.createUniform(d.f32);
  const pipeline = root
    .with(seconds, time)
    .createRenderPipeline({ vertex: common.fullScreenTriangle, fragment, targets: { format } });

  const markers: { destroy(): void }[] = [];

  return {
    time,
    initAsync: () => pipeline.initAsync(),

    marker() {
      const box = root.createUniform(Box);
      const rules = root.createUniform(Rules);
      const drawing = pipeline.with(root.createBindGroup(layout, { box, rules }));
      const marker = {
        box,
        rules,
        /** Draws the whole canvas, which the pass clears. */
        draw(pass: TgpuRenderCommands) {
          pass.setPipeline(drawing);
          pass.draw(3);
        },
        destroy() {
          box.buffer.destroy();
          rules.buffer.destroy();
        },
      };

      markers.push(marker);
      return marker;
    },

    destroy() {
      for (const marker of markers) marker.destroy();
      time.buffer.destroy();
    },
  };
}

export type Glow = ReturnType<typeof createGlow>;
export type Marker = ReturnType<Glow['marker']>;
