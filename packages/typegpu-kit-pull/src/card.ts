// The pulled card apart from how a host draws it: its finish by rarity, the seal on its back, its
// size on a stage, and where it is and how it is lit at a moment of the pull.

import { d, std } from 'typegpu';
import type { Rarity } from './rarity.ts';
import { clamp, clamp01, flightAt, LAUNCH, lerp, outAt, packAt, tellAt } from './timeline.ts';

/** How a card's face takes the foil, each from 0 for none to 1. */
export const Finish = d.struct({
  /** The poster. */
  art: d.f32,
  /** The rest of the face, under the text. */
  face: d.f32,
  /** How bright the glints are, 1 for a plain holo. */
  sparkle: d.f32,
  /** How far the foil turns gold. */
  warm: d.f32,
  /** The flip's flare, as a share of the brightest a flare gets. */
  flare: d.f32,
});
export type Finish = d.Infer<typeof Finish>;

/** The finish grows with rarity: matte, a faint foil poster, a foil poster, a foil face, then gold. */
export const FINISHES: Readonly<Record<Rarity, Finish>> = {
  common: { art: 0, face: 0, sparkle: 0, warm: 0, flare: 0 },
  uncommon: { art: 0.45, face: 0, sparkle: 0.5, warm: 0, flare: 0.2 },
  rare: { art: 1, face: 0, sparkle: 1, warm: 0, flare: 0.3 },
  epic: { art: 1, face: 0.8, sparkle: 1, warm: 0, flare: 0.4 },
  legendary: { art: 1, face: 1, sparkle: 1.6, warm: 1, flare: 0.5 },
};

/**
 * How much foil a point of the face takes under `finish`, in the poster or elsewhere on the face.
 */
export const reachOf = (finish: Finish, inArt: boolean) => {
  'use gpu';
  return std.select(finish.face, std.max(finish.art, finish.face), inArt);
};

/** The foil's colours, pulled toward gold as far as `warm`. */
export const warmed = (colors: d.v3f, warm: number) => {
  'use gpu';
  return std.mix(colors, d.vec3f(1, 0.8, 0.45), warm * 0.5);
};

/**
 * The seal on the card's back at `x`, `y`, each -1 to 1 across it: a glowing core, a diamond and a
 * ring, 0 to about 1, to print through a threshold. Runs on the CPU too.
 */
export const emblem = (x: number, y: number) => {
  'use gpu';
  const r = std.length(d.vec2f(x, y));
  const core = 0.92 * (1 - std.smoothstep(0, 0.6, r));
  const ringAt = (r - 0.84) / 0.04;
  const ring = std.exp(-(ringAt * ringAt));
  const diamondAt = (std.abs(x) + std.abs(y) - 0.62) / 0.035;
  const diamond = std.select(d.f32(0), std.exp(-(diamondAt * diamondAt)), r < 0.8);
  return std.max(core, std.max(ring * 1.1, diamond * 0.95));
};

/** How far the card peeks out of the torn pack, in its heights. */
const PEEK = 0.1;

/** Points from the viewer to the stage, as the site's stage sets its `perspective`. */
export const PERSPECTIVE = 1400;

/** Cells across the seal on the card's back. */
export const SEAL = 56;

/**
 * The card's width and height on a stage `width` by `height`, with `room` under it for its row, at
 * most `widest` wide.
 */
export function cardSize(width: number, height: number, room: number, widest: number) {
  const w = Math.floor(Math.min(widest, width - 40, ((height - room - 48) * 5) / 7));
  return [w, Math.round((w * 7) / 5)] as const;
}

type Point = readonly [number, number];

/** Where the card leaves the vessel this moment, as the vessel's own pose gives it. */
interface Exit {
  /** Where the card leaves it, and how wide it is there. */
  readonly mouth: Point;
  readonly span: number;
  /** The lip the card is hidden below, left to right. */
  readonly lip: readonly Point[];
}

/** The stage the card flies on, in points. */
export interface CardStage {
  readonly width: number;
  readonly height: number;
  /** Where the card rests once it has landed. */
  readonly slot: Point;
  /** Points a unit of the vessel. */
  readonly size: number;
}

/** The card at one moment, as a host places and lights it. */
export interface CardPose {
  /** Its centre on the stage and how near the viewer it is, in points. */
  readonly x: number;
  readonly y: number;
  readonly z: number;
  readonly scale: number;
  /** Its tip, its turn from face down to face up, and its roll, in degrees. */
  readonly rx: number;
  readonly ry: number;
  readonly rz: number;
  readonly visible: boolean;
  /** While it is in the vessel, the lip it hides below, in points of the card before scaling. */
  readonly rim: readonly Point[] | undefined;
  /** How dark the face and the back are from the light, 0 to 0.85. */
  readonly shade: number;
  readonly backShade: number;
  /** How far off the stage it is lifted, for its shadow. */
  readonly lift: number;
  /** The back's edge glowing in the rarity's colour, 0 to 1. */
  readonly tell: number;
  /** The row under it coming in, 0 to 1. */
  readonly row: number;
  /** Where its face points, -1 to 1 across and down, for the foil. */
  readonly tilt: Point;
  readonly sheen: number;
  /** The flare sweeping the face, -0.3 to 1.4, and its brightness. */
  readonly sweep: number;
  readonly flare: number;
}

const RAD = Math.PI / 180;

/**
 * The card `t` milliseconds into a pull from `vessel`, leaving it where `exit` says: rising face down
 * out of the box or standing in the pack, then tossed up on an arc and turned over into its slot,
 * the way `direction` (1 or -1) turns it, its flare `flare` of the brightest.
 */
export function cardAt(
  vessel: 'box' | 'pack',
  t: number,
  exit: Exit,
  stage: CardStage,
  direction: number,
  flare: number,
): CardPose {
  const { width: cardW, height: cardH, slot, size } = stage;
  const flying = t - LAUNCH[vessel];
  // Where the card starts its flight, and at what size: out of the box at its width, or peeking
  // out of the pack at its own
  const s0 = Math.min(exit.span / cardW, 1);
  const h0 = cardH * s0;
  const [mx, my] = exit.mouth;
  const start: Point =
    vessel === 'box' ? [mx, my - h0 / 2 - 0.2 * size] : [slot[0], slot[1] - PEEK * cardH];
  let x: number;
  let y: number;
  let scale = s0;
  let z = 0;
  let rx = 0;
  let ry = 180;
  let rz = 0;
  let sweep = -1;
  let shine = 0;
  let row = 0;
  let visible = true;
  let rim: Point[] | undefined;

  if (flying < 0) {
    // Face down, rising out of the box's mouth or standing in the pack as it falls away, hidden
    // below the lip
    if (vessel === 'box') {
      x = mx;
      y = my - lerp(-h0 / 2 - 4, h0 / 2 + 0.2 * size, outAt(t));
      visible = outAt(t) > 0;
    } else {
      x = slot[0];
      y = slot[1] - PEEK * cardH * packAt(t).peek;
      visible = t >= 0;
    }
    rim = exit.lip.map(([px, py]): Point => [
      (px - x) / scale + cardW / 2,
      (py - y) / scale + cardH / 2,
    ]);
  } else {
    // Tossed up on an arc, nearer the viewer in its middle, turning over as it comes
    const f = flightAt(flying);
    const bend = [lerp(start[0], slot[0], 0.5), Math.min(start[1], slot[1]) - 0.16 * cardH];
    const k = f.travel;
    x = (1 - k) ** 2 * start[0] + 2 * (1 - k) * k * bend[0] + k * k * slot[0];
    y = (1 - k) ** 2 * start[1] + 2 * (1 - k) * k * bend[1] + k * k * slot[1];
    scale = lerp(s0, 1, f.grow);
    z = f.near * cardH;
    ry = 180 * (1 - f.turn) * direction;
    rx = -16 * f.tip;
    rz = 6 * f.roll * direction;
    sweep = f.sweep;
    shine = f.flare * flare;
    row = f.row;
  }

  // Lit from the upper left: the face darkens as it turns away, and its foil follows its normal
  const a = ry * RAD;
  const b = rx * RAD;
  const normal = [Math.sin(a) * Math.cos(b), -Math.sin(b), Math.cos(a) * Math.cos(b)] as const;
  const light = 0.3 * -normal[0] - 0.4 * normal[1] + 0.87 * normal[2];
  const tilt: Point = [clamp(normal[0] * 1.6, -1.4, 1.4), clamp(normal[1] * 1.6, -1.4, 1.4)];
  const sheen = 0.22 + 0.78 * Math.min(1, Math.hypot(tilt[0], tilt[1]));
  return {
    x,
    y,
    z,
    scale,
    rx,
    ry,
    rz,
    visible,
    rim,
    shade: clamp(0.62 * (1 - light), 0, 0.85),
    backShade: clamp(0.62 * (1 + light), 0, 0.85),
    lift: z / cardH + 0.15 * (1 - clamp01(flying / 600)),
    tell: tellAt(t, vessel),
    row,
    tilt,
    sheen: Math.max(sheen, 0.3 * Math.sign(shine) + 0.7 * shine),
    sweep,
    flare: shine,
  };
}
