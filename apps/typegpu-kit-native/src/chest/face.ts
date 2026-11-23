// The pulled card, drawn whole on a canvas over the overlay, so it stays sharp at every size and
// tilt of its flight: a transform of a React Native view would resample its text. Each pixel finds
// its point on the card through the pose's projection, the transforms the site's stage gives its
// card. The face is the poster printed in cells under the site's holo foil, with the caption's type
// in signed-distance fonts; the back is the seal stamped on a hatched ground, its edge lit in the
// rarity's colour as the box tells. The poster shrinks to cells on the GPU once, as the site's
// `print.ts` shrinks it on a 2D canvas.

import {
  baker,
  font,
  passes,
  pipelines,
  text,
  type Font,
  type Surface,
  type TextSet,
} from '@typegpu/kit';
import type { Scene } from '@typegpu/kit/native/host';
import { d, std, type TgpuRoot } from 'typegpu';
import { emblem, PERSPECTIVE, SEAL, type CardPose } from 'typegpu-kit-pull/card.ts';
import { bayerCell, CELL, rgbOf, type Rgb01 } from 'typegpu-kit-pull/dither.ts';
import { createFoil, faceAt, MAX_CELLS } from 'typegpu-kit-pull/foil.scene.ts';
import { colors } from '../theme.ts';
import { captionOf, type Label } from './caption.ts';
import type { Entry } from './cards.ts';

/** Taps across and down a cell when the poster shrinks to it. */
const TAPS = 4;
/** The back's hairlines run on the diagonal. */
const DIAGONAL = Math.SQRT1_2;
/** Segments of the box's lip that cut the card, as many as the lip has. */
const LIPS = 3;
/** The drop shadow under the card, in points of the card, as the gallery's cards cast it. */
const SHADOW = { drop: 20, blur: 30, alpha: 0.9 };
/** The card's corners' radius, in points. */
const CORNER = 2;
/** Glyphs a line of the caption holds at most. */
const LENGTH = 64;
const RAD = Math.PI / 180;

const rgb = (hex: string) => {
  const [r, g, b] = rgbOf(hex);
  return d.vec3f(r, g, b);
};
const GROUND = rgb(colors.cardGround);
const HATCH = rgb(colors.cardHatch);
const RULE = rgb(colors.rule);
const RULE_STRONG = rgb(colors.ruleStrong);
const ACCENT = rgb(colors.accent);
const SHADE = rgb(colors.cardShade);
const SHADOW_COLOR = rgb('#010204');

const Card = d.struct({
  /** Rows of the projection from the card's points to the canvas's, and back. */
  forward: d.arrayOf(d.vec4f, 3),
  inverse: d.arrayOf(d.vec4f, 3),
  /** The canvas's size in points, its pixels a point, and the card's size in points. */
  screen: d.vec2f,
  pixelRatio: d.f32,
  size: d.vec2f,
  /** 1 while the back faces the viewer. */
  back: d.u32,
  /** The rarity's tell, 0 to 1, and how far the side shown has turned into shade. */
  tell: d.f32,
  shade: d.f32,
  /** The lip's half-planes in canvas points, `dot(xy, p) + z >= 0` above it. */
  lips: d.arrayOf(d.vec4f, LIPS),
  /** The face's frame, the colour the back's frame turns to, and the rarity's colour. */
  frame: d.vec4f,
  told: d.vec4f,
  rarity: d.vec4f,
  /** A hairline, where the art window ends, the foot's rule, and the gem's centre and reach. */
  hair: d.f32,
  artBottom: d.f32,
  footTop: d.f32,
  rule: d.vec2f,
  gem: d.vec4f,
});

/** A font as baked, its JSON parsed and its atlas's bytes. */
export interface BakedFont {
  readonly json: unknown;
  readonly atlas: ArrayBuffer;
}

export interface FaceOptions {
  /** The example's poster, or none for a blank window. */
  readonly poster: ImageBitmap | undefined;
  /** The display face for the title and the mono face for the rest. */
  readonly fonts: { readonly display: BakedFont; readonly mono: BakedFont };
  /** The card's size in its slot, in points. */
  readonly size: readonly [number, number];
  readonly entry: Entry;
  /** The card at this moment, which the pull moves, or null before it starts. */
  readonly pose: { readonly value: CardPose | null };
}

const vec4 = (c: Rgb01) => d.vec4f(c[0], c[1], c[2], 1);

/** `a` mixed toward `b` by `k`, in whole bytes as the site's tokens mix. */
const mixed = (a: Rgb01, b: Rgb01, k: number) => {
  const at = (i: 0 | 1 | 2) => Math.round((a[i] + (b[i] - a[i]) * k) * 255) / 255;
  return d.vec4f(at(0), at(1), at(2), 1);
};

/** Distance from `p` to the card `half` each way from the origin, its corners rounded. */
const boxDistance = (p: d.v2f, half: d.v2f) => {
  'use gpu';
  const q = std.abs(p) - half + CORNER;
  return std.length(std.max(q, d.vec2f())) + std.min(std.max(q.x, q.y), 0) - CORNER;
};

/** The back at `p`, in points inside its frame `size` across, lit by `tell` in `rarity`. */
const backAt = (p: d.v2f, size: d.v2f, tell: number, rarity: d.v3f) => {
  'use gpu';
  // Hairlines on the diagonal, a point wide every seven
  const along = (p.x - p.y) * DIAGONAL;
  const stripe = along - std.floor(along / 7) * 7;
  let color = std.mix(GROUND, HATCH, 0.7 * std.step(stripe, 1));

  // The edge lit from inside as the box tells
  const edge = std.min(std.min(p.x, p.y), std.min(size.x - p.x, size.y - p.y));
  const reach = std.max((tell * 40 * size.x) / 340, 0.001);
  const glow = 0.34 * 0.6 * (1 - std.smoothstep(0, reach, edge)) * std.step(0.001, tell);
  color = std.mix(color, rarity, glow + 0.7 * tell * std.step(edge, 1));

  // The seal, 44 % of the width across, printed in its own cells
  const half = size.x * 0.22;
  const local = (p - size * 0.5) / half;
  if (std.abs(local.x) < 1 && std.abs(local.y) < 1) {
    const cell = std.floor((local * 0.5 + 0.5) * SEAL);
    const centre = ((cell + 0.5) / SEAL) * 2 - 1;
    if (emblem(centre.x, centre.y) > bayerCell(d.vec2u(cell))) {
      color = std.mix(ACCENT, rarity, tell);
    }
  }
  return color;
};

/**
 * Writes the inverse of the 3 × 3 row-major matrix `m` into `into`, or returns false when `m` is
 * singular, as a card seen edge-on is.
 */
function invert(into: Float64Array, m: Float64Array): boolean {
  const [a, b, c] = [m[0], m[1], m[2]];
  const [e, f, g] = [m[3], m[4], m[5]];
  const [h, i, j] = [m[6], m[7], m[8]];
  const det = a * (f * j - g * i) - b * (e * j - g * h) + c * (e * i - f * h);
  if (Math.abs(det) < 1e-9) return false;
  const k = 1 / det;
  into[0] = (f * j - g * i) * k;
  into[1] = (c * i - b * j) * k;
  into[2] = (b * g - c * f) * k;
  into[3] = (g * h - e * j) * k;
  into[4] = (a * j - c * h) * k;
  into[5] = (c * e - a * g) * k;
  into[6] = (e * i - f * h) * k;
  into[7] = (b * h - a * i) * k;
  into[8] = (a * f - b * e) * k;
  return true;
}

/** Copies the rows of `m` into the first three components of `rows`. */
function rowsOf(rows: d.v4f[], m: Float64Array) {
  for (let r = 0; r < 3; r++) {
    rows[r].x = m[r * 3];
    rows[r].y = m[r * 3 + 1];
    rows[r].z = m[r * 3 + 2];
  }
}

/**
 * Writes the projection of `now` from a card `width` by `height` points to the canvas: the turn
 * about the card's centre, flattened onto the card's plane as a view's sublayers are, then the
 * card's scale, lean and spin, its perspective, and its place.
 */
function project(
  into: Float64Array,
  l: Float64Array,
  now: CardPose,
  width: number,
  height: number,
) {
  const cy = Math.cos(now.ry * RAD);
  const sy = Math.sin(now.ry * RAD) / PERSPECTIVE;
  const cx = Math.cos(now.rx * RAD);
  const sx = Math.sin(now.rx * RAD) / PERSPECTIVE;
  const cz = Math.cos(now.rz * RAD);
  const sz = Math.sin(now.rz * RAD);
  const s = (now.scale * PERSPECTIVE) / (PERSPECTIVE - now.z);
  // The turn, of a point from the card's centre
  const t0 = (-cy * width) / 2;
  const t2 = 1 - (sy * width) / 2;
  const t5 = -height / 2;
  // The lean and spin, whose rows the place shifts by the perspective's row
  l[0] = s * cz;
  l[1] = -s * cx * sz - now.x * s * sx;
  l[2] = now.x;
  l[3] = s * sz;
  l[4] = s * cx * cz - now.y * s * sx;
  l[5] = now.y;
  l[6] = 0;
  l[7] = -s * sx;
  l[8] = 1;
  for (let r = 0; r < 3; r++) {
    const a = l[r * 3];
    const b = l[r * 3 + 1];
    const c = l[r * 3 + 2];
    into[r * 3] = a * cy + c * sy;
    into[r * 3 + 1] = b;
    into[r * 3 + 2] = a * t0 + b * t5 + c * t2;
  }
}

/** Creates the card's canvas scene over the whole overlay; host it through `worklets.ts`. */
export function create(root: TgpuRoot, surface: Surface, options: FaceOptions): Scene {
  const { poster, pose, entry } = options;
  const [width, height] = options.size;
  const ratio = surface.pixelRatio;
  const hair = 1 / ratio;
  const inner = d.vec2f(width - 2, height - 2);

  // The poster shrinks to its cells once, here, each cell the average of its taps, cropped as
  // `object-fit: cover` would. Copying an image into a texture needs it to be a render target too
  const foil = createFoil(root, surface.format);
  const art = d.vec4f(0, 0, inner.x, inner.x * 0.75);
  const across = Math.ceil(inner.x / CELL);
  const down = Math.ceil(art.w / CELL);
  const [posterWidth, posterHeight] = [poster?.width ?? 1, poster?.height ?? 1];
  const picture = root
    .createTexture({ size: [posterWidth, posterHeight], format: 'rgba8unorm' })
    .$usage('sampled', 'render');
  if (poster) picture.write(poster);
  else picture.write(new Uint8Array([13, 16, 22, 255]));
  const posterView = picture.createView(d.texture2d(d.f32));
  const smooth = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const scale = Math.max(across / posterWidth, down / posterHeight);
  const drawn = d.vec2f(posterWidth * scale, posterHeight * scale);
  const offset = d.vec2f((across - drawn.x) / 2, (down - drawn.y) / 2);
  const shrink = baker.create(root, foil.cells, {
    texel: (uv) => {
      'use gpu';
      const corner = std.floor(uv * MAX_CELLS);
      let sum = d.vec3f();
      for (let j = 0; j < TAPS; j++) {
        for (let i = 0; i < TAPS; i++) {
          const at = corner + (d.vec2f(d.f32(i), d.f32(j)) + 0.5) / TAPS;
          sum += std.textureSampleLevel(posterView.$, smooth.$, (at - offset) / drawn, 0).xyz;
        }
      }
      return d.vec4f(sum / (TAPS * TAPS), 1);
    },
  });
  const encoder = root.createCommandEncoder();
  shrink.record(encoder);
  encoder.submit();

  const face = {
    ground: vec4(rgbOf(colors.cardGround)),
    size: inner,
    cell: CELL,
    texel: CELL,
    origin: d.vec2f(),
    art,
    tilt: d.vec2f(),
    sheen: 0,
    flare: d.vec2f(),
    finish: entry.finish,
  };

  // The caption, laid out once in the card's points
  const { display: shown, mono: typed } = options.fonts;
  const display = font.create(root, font.read(shown.json, shown.atlas));
  const mono = font.create(root, font.read(typed.json, typed.atlas));
  const caption = captionOf(entry, width, height, hair, display, mono);
  const rule = rgbOf(colors.ruleStrong);

  const card = {
    forward: [d.vec4f(), d.vec4f(), d.vec4f()],
    inverse: [d.vec4f(), d.vec4f(), d.vec4f()],
    screen: d.vec2f(surface.width / ratio, surface.height / ratio),
    pixelRatio: ratio,
    size: d.vec2f(width, height),
    back: 0,
    tell: 0,
    shade: 0,
    lips: [d.vec4f(0, 0, 1, 0), d.vec4f(0, 0, 1, 0), d.vec4f(0, 0, 1, 0)],
    frame: entry.rarity === 'common' ? vec4(rule) : mixed(rule, entry.tint, 0.55),
    told: mixed(rule, entry.tint, 0.8),
    rarity: vec4(entry.tint),
    hair,
    artBottom: caption.artBottom,
    footTop: caption.footTop,
    rule: d.vec2f(caption.rule[0], caption.rule[1]),
    gem: d.vec4f(caption.gem[0], caption.gem[1], caption.gem[2], 0),
  };
  const cardBuffer = root.createUniform(Card, card);

  const Seen = d.struct({ color: d.vec4f });
  const drawing: passes.Drawing<typeof passes.ScreenVertex, d.Infer<typeof Seen>> = {
    ...passes.fullScreen,
    label: 'card',
    pixelOf: (corner) => {
      'use gpu';
      const c = cardBuffer.$;
      const p = corner.position.xy / c.pixelRatio;
      const h = d.vec3f(p, 1);
      const q = d.vec3f(
        std.dot(c.inverse[0].xyz, h),
        std.dot(c.inverse[1].xyz, h),
        std.dot(c.inverse[2].xyz, h),
      );
      const local = q.xy / q.z;
      // A pixel's width in points of the card, for edges a pixel soft at any tilt
      const pixel = std.max(std.length(std.fwidth(local)) * 0.7, 0.0001);

      // Above the box's lip, and in front of the viewer
      let kept = std.step(0, q.z);
      for (let i = 0; i < LIPS; i++) {
        kept *= std.step(0, std.dot(c.lips[i].xy, p) + c.lips[i].z);
      }

      const half = c.size * 0.5;
      const edge = boxDistance(local - half, half);
      const cover = std.saturate(0.5 - edge / pixel);
      const dropped = boxDistance(local - half - d.vec2f(0, SHADOW.drop), half);
      const shadow = SHADOW.alpha * (1 - std.smoothstep(-SHADOW.blur, SHADOW.blur, dropped));

      // Inside the frame, in points from the face's corner; the back reads it mirrored
      const size = c.size - 2;
      const at = local - 1;
      let color = d.vec3f();
      let frame = c.frame.xyz;
      if (c.back === 1) {
        color = backAt(d.vec2f(size.x - at.x, at.y), size, c.tell, c.rarity.xyz);
        frame = std.mix(RULE_STRONG, c.told.xyz, c.tell);
      } else {
        color = faceAt(d.vec2u(std.floor(std.max(at, d.vec2f()) / CELL))).xyz;
        // The art window's rule, the foot's, and the gem
        const thick = std.max(c.hair, pixel);
        const artRule = std.saturate(1 - std.abs(local.y - c.artBottom + c.hair * 0.5) / thick);
        const footRule =
          std.saturate(1 - std.abs(local.y - c.footTop - c.hair * 0.5) / thick) *
          std.step(c.rule.x, local.x) *
          std.step(local.x, c.rule.y);
        color = std.mix(color, RULE, std.saturate(artRule + footRule));
        const gem = std.abs(local.x - c.gem.x) + std.abs(local.y - c.gem.y) - c.gem.z;
        color = std.mix(color, c.rarity.xyz, std.saturate(0.5 - gem / pixel));
      }
      // The point-wide frame, then the shade the side has turned into
      color = std.mix(color, frame, std.saturate(0.5 - (-edge - 1) / pixel));
      color = std.mix(color, SHADE, c.shade);

      // The card over its shadow, premultiplied
      const alpha = cover + shadow * (1 - cover);
      const lit = color * cover + SHADOW_COLOR * (shadow * (1 - cover));
      return Seen({ color: d.vec4f(lit, alpha) * kept });
    },
    color: passes.shaded,
    blend: pipelines.blend.over,
    draw(pass) {
      pass.setBindGroup(foil.group);
      pass.draw(3);
    },
  };

  // Type, placed on the card through the same projection, and shaded with it
  const place = (position: d.v3f, pen: d.v2f) => {
    'use gpu';
    const c = cardBuffer.$;
    const h = d.vec3f(position.x + pen.x, position.y - pen.y, 1);
    const x = std.dot(c.forward[0].xyz, h);
    const y = std.dot(c.forward[1].xyz, h);
    const w = std.dot(c.forward[2].xyz, h);
    return d.vec4f((x * 2) / c.screen.x - w, w - (y * 2) / c.screen.y, 0, w);
  };
  const shade = (fragment: text.Fragment) => {
    'use gpu';
    const alpha = fragment.color.w * std.saturate(fragment.distance + 0.5);
    const color = std.mix(fragment.color.xyz, SHADE, cardBuffer.$.shade);
    return d.vec4f(color * alpha, alpha);
  };
  const setOf = (of: Font, labels: readonly Label[]): TextSet => {
    const set = text.create(root, of, {
      capacity: Math.max(labels.length, 1),
      length: LENGTH,
      place,
      shade,
      label: 'card type',
    });
    labels.forEach((label, i) => {
      const { em, tracking, align } = label;
      set.write(i, label.text, { em, tracking, align, baseline: 'alphabetic' });
    });
    set.labels.write(
      labels.map(({ x, y, color }) => ({
        position: d.vec3f(x, y, 0),
        color: d.vec4f(color[0], color[1], color[2], 1),
      })),
    );
    return set;
  };
  const titles = setOf(display, caption.display);
  const lines = setOf(mono, caption.mono);

  const paint = passes.create(root, surface);
  const drawnCard = paint.add(drawing);
  const faceUp = [drawnCard, paint.add(titles.drawing), paint.add(lines.drawing)];
  const backUp = [drawnCard];
  const none: readonly passes.Drawn[] = [];
  const forward = new Float64Array(9);
  const inverse = new Float64Array(9);
  const leaned = new Float64Array(9);

  // The card draws nothing until its pipelines have compiled
  let compiled = false;

  return {
    ready: paint.initAsync().then(() => {
      compiled = true;
    }),

    frame(frame) {
      const now = pose.value;
      const showing = compiled && now?.visible === true;
      if (showing) project(forward, leaned, now, width, height);
      if (!showing || !invert(inverse, forward)) {
        paint.record(frame, none);
        return;
      }
      card.screen.x = surface.width / ratio;
      card.screen.y = surface.height / ratio;
      rowsOf(card.forward, forward);
      rowsOf(card.inverse, inverse);
      const showsBack = Math.abs(now.ry) > 90;
      card.back = showsBack ? 1 : 0;
      card.tell = now.tell;
      card.shade = showsBack ? now.backShade : now.shade;

      // The lip, where the card's points meet the canvas while it rises out of the box
      const rim = now.rim;
      for (let i = 0; i < LIPS; i++) {
        const lip = card.lips[i];
        if (!rim || rim.length < 2) {
          lip.x = 0;
          lip.y = 0;
          lip.z = 1;
          continue;
        }
        const k = Math.min(i, rim.length - 2);
        const ax = now.x + (rim[k][0] - width / 2) * now.scale;
        const ay = now.y + (rim[k][1] - height / 2) * now.scale;
        const bx = now.x + (rim[k + 1][0] - width / 2) * now.scale;
        const by = now.y + (rim[k + 1][1] - height / 2) * now.scale;
        const length = Math.hypot(bx - ax, by - ay) || 1;
        lip.x = (by - ay) / length;
        lip.y = -(bx - ax) / length;
        lip.z = -(lip.x * ax + lip.y * ay);
      }
      cardBuffer.write(card);

      if (!showsBack) {
        face.tilt.x = now.tilt[0];
        face.tilt.y = now.tilt[1];
        face.sheen = now.sheen;
        face.flare.x = now.sweep;
        face.flare.y = now.flare;
        foil.face.write(face);
      }
      paint.record(frame, showsBack ? backUp : faceUp);
    },

    destroy() {
      paint.destroy();
      titles.destroy();
      lines.destroy();
      display.destroy();
      mono.destroy();
      picture.destroy();
      foil.cells.destroy();
    },
  };
}
