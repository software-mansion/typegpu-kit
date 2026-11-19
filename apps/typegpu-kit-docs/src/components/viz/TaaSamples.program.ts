import { frames, taa } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';
import { clamp } from '../home/geometry.ts';
import { floatsOf } from './program.ts';

/** The loupe's output pixels. */
export const WIDTH = 9;
export const HEIGHT = 6;
/** The share of each side a scene resolved up to `size` is drawn at here, the least TAA takes. */
const SCALE = 2 / 3;
/**
 * Shades the clamp moves a pixel's history by before the loupe says it threw old samples out. At
 * rest it moves them by less, and the pixels a moved edge crosses by more.
 */
export const THROWN = 0.4;

/** The shape's brightness over a black ground, in linear light. */
const BRIGHT = 1;
/** The neighbourhood the resolve reads around a pixel, as in taa.ts. */
const NEAR = [-1, 0, 1];
/** The least share of a pixel a sample keeps to be drawn as a dot. */
const LEAST = 0.004;

/**
 * What the loupe prints of a linear value, from 0 to 1 at the shape's brightness. The resolve's
 * brightness weights make its blend a plain mix of `v / (1 + v)`, so the loupe prints that.
 */
export const shade = (value: number) => ((value / (1 + value)) * (1 + BRIGHT)) / BRIGHT;

/** Lanczos-2 at `x` output pixels, from -2 to 2, the polynomial of taa.ts. */
const lanczos = (x: number) => {
  const v = x * x;
  return (-((v - 4) * (v - 4)) * (v - 1) * (1 - 0.354377 * v)) / 16;
};

export interface Size {
  readonly width: number;
  readonly height: number;
}

export interface Edge {
  /** Where the edge crosses the middle row, in output pixels from the left. */
  at: number;
  /** Output pixels it leans right by for each row down. */
  readonly slope: number;
}

export interface Dot {
  /** Where the sample was taken, in output pixels. */
  readonly x: number;
  readonly y: number;
  /** Whether it landed on the shape. */
  readonly hit: boolean;
  /** How much of the output pixel under it comes from the sample's frame, from 0 to 1. */
  readonly share: number;
  /** Whether this frame took it. */
  readonly fresh: boolean;
}

export interface Options {
  /** Draws the scene at `SCALE` of each side and resolves it up to the loupe's size. */
  readonly scaled?: boolean;
  readonly edge?: Partial<Edge>;
}

/**
 * A slanted edge sampled once a drawn pixel through a `taa.create`'s own jitter, and resolved on
 * the CPU as its fragment resolves a picture without motion or depth.
 */
export function createProgram({ scaled = false, edge: placed }: Options = {}) {
  const recording = recordingRoot();
  const { root } = recording;
  const { blend, spread } = taa.defaults;
  const edge: Edge = { at: 3.75, slope: 0.25, ...placed };

  const drawn: Size = scaled
    ? { width: WIDTH * SCALE, height: HEIGHT * SCALE }
    : { width: WIDTH, height: HEIGHT };
  const ratio = { x: WIDTH / drawn.width, y: HEIGHT / drawn.height };

  const picture = root
    .createTexture({ size: [drawn.width, drawn.height], format: 'rgba16float' })
    .$usage('sampled', 'render');
  const aa = taa.create(root, {
    source: () => picture,
    depth: null,
    ...(scaled && { size: { width: WIDTH, height: HEIGHT } }),
  });
  const loop = frames.create(root, (frame) => aa.record(frame));
  let time = 0;
  const record = () => loop.run((time += 1 / 60));

  /** How far this frame's jitter moves the picture, in drawn pixels: `shift` in taa.ts. */
  const shift = () => {
    const [x, y] = floatsOf(recording.bytes(aa.offset.buffer));
    return { x: x * 0.5 * drawn.width, y: y * -0.5 * drawn.height };
  };

  // The kit's cycle: its first frame draws unmoved, then its offsets come round in turn
  const cycle: { x: number; y: number }[] = [];
  for (;;) {
    record();
    const next = shift();
    if (cycle[0]?.x === next.x && cycle[0].y === next.y) break;
    cycle.push(next);
    if (cycle.length > 64) throw new Error('The jitter never came round.');
  }
  aa.reset();

  const phases = cycle.length;
  const source = new Float32Array(drawn.width * drawn.height);
  /** The resolved picture, which the next frame reads as its history. */
  const resolved = new Float32Array(WIDTH * HEIGHT);
  const rejected = new Float32Array(WIDTH * HEIGHT);
  /** Each drawn pixel's sample of each phase: -1 none yet, 0 off the shape, 1 on it. */
  const hits = new Int8Array(source.length * phases).fill(-1);
  /** How much of each output pixel comes from each phase's frames. */
  const shares = new Float32Array(WIDTH * HEIGHT * phases);
  let valid = false;
  let phase = 0;

  /** Where drawn pixel `(x, y)` takes its sample under `moved`, in output pixels. */
  const sampleAt = (x: number, y: number, moved: { x: number; y: number }) => ({
    x: (x + 0.5 - moved.x) * ratio.x,
    y: (y + 0.5 - moved.y) * ratio.y,
  });
  const covered = (x: number, y: number) => x < edge.at + edge.slope * (y - HEIGHT / 2);
  const read = (x: number, y: number) =>
    source[clamp(y, 0, drawn.height - 1) * drawn.width + clamp(x, 0, drawn.width - 1)];

  /**
   * The fragment of taa.ts for output pixel `(ox, oy)`, on luminance alone: the picture is grey,
   * so its two chroma channels stay 0. History is read at the pixel, as without motion or depth,
   * where `historyAt`'s cubic weights are 1 at the centre tap and 0 elsewhere. Returns the
   * colour, the share of the result that is history the clamp kept, the share that is this
   * frame, and how far the clamp moved the history, in shades.
   */
  const resolve = (ox: number, oy: number, moved: { x: number; y: number }) => {
    const here = { x: (ox + 0.5) / ratio.x, y: (oy + 0.5) / ratio.y };
    const pixel = scaled ? { x: Math.trunc(here.x), y: Math.trunc(here.y) } : { x: ox, y: oy };

    let sum = 0;
    let squares = 0;
    const gathered = { color: 0, weight: 0, low: Infinity, high: -Infinity };
    for (const y of NEAR) {
      for (const x of NEAR) {
        const at = {
          x: clamp(pixel.x + x, 0, drawn.width - 1),
          y: clamp(pixel.y + y, 0, drawn.height - 1),
        };
        const color = read(at.x, at.y);
        sum += color;
        squares += color * color;
        if (scaled) {
          // `gather`: weighted by how far the pixel's jittered sample lies, in output pixels
          const w =
            lanczos(clamp((at.x + 0.5 - moved.x - here.x) * ratio.x, -2, 2)) *
            lanczos(clamp((at.y + 0.5 - moved.y - here.y) * ratio.y, -2, 2));
          gathered.color += color * w;
          gathered.weight += w;
          gathered.low = Math.min(gathered.low, color);
          gathered.high = Math.max(gathered.high, color);
        }
      }
    }

    let current = read(pixel.x, pixel.y);
    if (scaled) {
      const filtered = gathered.color / Math.max(gathered.weight, 0.0001);
      current = clamp(filtered, gathered.low, gathered.high);
    }
    if (!valid) return { color: current, kept: 0, now: 1, cut: 0 };

    const mean = sum / 9;
    let extent = Math.max(Math.sqrt(Math.max(squares / 9 - mean * mean, 0)) * spread, 0.0001);
    if (scaled) extent *= Math.max(ratio.x, ratio.y);
    const before = resolved[oy * WIDTH + ox];
    const away = before - mean;
    const furthest = Math.abs(away) / extent;
    const previous = mean + away / Math.max(furthest, 1);

    let now = blend / (1 + Math.max(current, 0));
    const then = (1 - blend) / (1 + Math.max(previous, 0));
    if (scaled) now *= clamp(gathered.weight, 0, 1);
    return {
      color: (current * now + previous * then) / (now + then),
      kept: then / (now + then) / Math.max(furthest, 1),
      now: now / (now + then),
      cut: Math.abs(shade(before) - shade(previous)),
    };
  };

  return {
    drawn,
    /** Frames in a jitter cycle. */
    phases,
    edge,
    /** This frame's picture, a value a drawn pixel. */
    source,
    /** The resolved picture, a value an output pixel. */
    resolved,
    /** How far the clamp moved each output pixel's history this frame, in shades. */
    rejected,

    /** Where the last frame sits in the cycle, from 0. */
    get phase() {
      return phase;
    },

    /** Where the last frame's sample sits in every drawn pixel, from 0 to 1 across it. */
    get point() {
      return { x: 0.5 - cycle[phase].x, y: 0.5 - cycle[phase].y };
    },

    /** The samples that still count towards the picture, oldest first. */
    dots(): Dot[] {
      const dots: Dot[] = [];
      for (let back = phases - 1; back >= 0; back--) {
        const k = (phase - back + phases) % phases;
        for (let y = 0; y < drawn.height; y++) {
          for (let x = 0; x < drawn.width; x++) {
            const hit = hits[(y * drawn.width + x) * phases + k];
            if (hit < 0) continue;
            const at = sampleAt(x, y, cycle[k]);
            const under = Math.floor(at.y) * WIDTH + Math.floor(at.x);
            const share = shares[under * phases + k];
            if (share >= LEAST) dots.push({ ...at, hit: hit === 1, share, fresh: back === 0 });
          }
        }
      }
      return dots;
    },

    /** Draws a frame through the kit's jitter, resolves it, and lets the kit move the jitter on. */
    step() {
      const moved = shift();
      phase = cycle.findIndex(({ x, y }) => x === moved.x && y === moved.y);
      if (phase < 0) throw new Error('The jitter left its cycle.');

      for (let y = 0; y < drawn.height; y++) {
        for (let x = 0; x < drawn.width; x++) {
          const at = sampleAt(x, y, moved);
          const hit = covered(at.x, at.y);
          source[y * drawn.width + x] = hit ? BRIGHT : 0;
          hits[(y * drawn.width + x) * phases + phase] = hit ? 1 : 0;
        }
      }

      for (let y = 0; y < HEIGHT; y++) {
        for (let x = 0; x < WIDTH; x++) {
          const i = y * WIDTH + x;
          const { color, kept, now, cut } = resolve(x, y, moved);
          // Without motion a pixel reads history only at itself, so one picture holds both
          resolved[i] = color;
          rejected[i] = cut;
          for (let k = 0; k < phases; k++) shares[i * phases + k] *= kept;
          shares[i * phases + phase] += now;
        }
      }

      record();
      valid = true;
    },

    destroy() {
      loop.destroy();
      aa.destroy();
      picture.destroy();
    },
  };
}

export type Program = ReturnType<typeof createProgram>;
