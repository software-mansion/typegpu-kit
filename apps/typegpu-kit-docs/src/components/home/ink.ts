import { table, type Frame } from '@typegpu/kit';
import { perlin3d } from '@typegpu/noise';
import { d, std, tgpu, type TgpuRenderPass, type TgpuRoot } from 'typegpu';
import { bayer, type Rgb } from '../site/dither.ts';
import { ROWS } from '../site/lockup.ts';

/**
 * Ink cells at most: the lockup is 72 cells by 31, and the badge, a cell a CSS pixel, about 82
 * by 44.
 */
const CAPACITY = 8192;

/**
 * The pointer's reach and a press's, in lockup cells, and their pushes, cells/s², so a small
 * lockup, as on a phone, stirs as much of itself as a large one.
 */
const POINTER = { reach: 15, burst: 36, push: 2000, blast: 3600, wake: 5 };
const MAX_SPEED = 1000;
/** The pointer's lens, in cells. */
const LENS = 12;
/** The swirl's scale: its eddies are a few hundred pixels across. */
const SWIRL = 0.0045;
const WORKGROUP = 64;

export const Ink = table.define('Ink', {
  host: {
    home: d.vec2f,
    /** Its own colour in the lockup, sRGB, and its dither threshold. */
    own: d.vec4f,
    /** How the light ramps across it. */
    tilt: d.f32,
    /** Its cell's middle in the header's mark, CSS pixels from the mark's corner. */
    mark: d.vec2f,
    /** When it leaves for the mark in the handoff, from 0, first, to 1, last. */
    order: d.f32,
    /** Its cell's side, in CSS pixels. */
    side: d.f32,
    /** 1 for the badge's cells, which stay out of the handoff and fade with the title. */
    badge: d.f32,
  },
  device: { position: d.vec2f, velocity: d.vec2f, heat: d.f32 },
});

const Params = d.struct({
  /** The pointer's x and y in CSS pixels of the canvas, 1 while it is near, and a press's burst. */
  pointer: d.vec4f,
  /** The pointer's velocity, seconds this step, and seconds since the start. */
  motion: d.vec4f,
  /** How much of the lockup shows, device pixels a CSS pixel, and how much of the badge. */
  look: d.vec3f,
  /** The canvas's size in device pixels, and a lockup cell's side and the pointer's lens in CSS pixels. */
  size: d.vec4f,
  /** The header mark's corner in CSS pixels of the canvas, and how far the handoff to it has gone. */
  hand: d.vec4f,
});

const Palette = d.struct({ rule: d.vec4f, accent: d.vec4f, bright: d.vec4f });

const toColor = (c: Rgb) => d.vec4f(c[0] / 255, c[1] / 255, c[2] / 255, 1);

const same = (a: readonly number[], b: readonly number[]) => a.every((x, k) => x === b[k]);

/** A slow swirl, the curl of the noise, about unit length. */
const curlAt = (p: d.v2f, t: number) => {
  'use gpu';
  const q = d.vec3f(p * SWIRL, t * 0.05);
  const n = perlin3d.sample(q);
  const nx = perlin3d.sample(q + d.vec3f(SWIRL * 4, 0, 0));
  const ny = perlin3d.sample(q + d.vec3f(0, SWIRL * 4, 0));
  return d.vec2f(ny - n, n - nx) * 37;
};

/**
 * The lockup's pixels at `columns` by `rows`, RGBA from 0 to 255 a row at a time: the page draws its
 * SVG at that size.
 */
export type ReadLockup = (columns: number, rows: number) => Uint8ClampedArray;

export interface InkOptions {
  readonly colors: { readonly rule: Rgb; readonly accent: Rgb; readonly bright: Rgb };
  readonly format: GPUTextureFormat;
  readonly read: ReadLockup;
  /** The badge beside the lockup, the same way. */
  readonly readBadge?: ReadLockup | undefined;
}

export function createInk(root: TgpuRoot, { colors, format, read, readBadge }: InkOptions) {
  const cells = table.create(root, Ink, { capacity: CAPACITY });
  const params = root.createUniform(Params);
  const palette = root.createUniform(Palette, {
    rule: toColor(colors.rule),
    accent: toColor(colors.accent),
    bright: toColor(colors.bright),
  });

  const kernel = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (!Ink.alive(i)) return;
    const s = params.$;
    const home = Ink.home(i);
    const me = Ink.device.$[i];
    let p = d.vec2f(me.position);
    let v = d.vec2f(me.velocity);
    const dt = s.motion.z;

    const ptr = s.pointer;
    const toPointer = p - ptr.xy;
    const gap = std.length(toPointer) + 1e-3;
    const cell = s.size.z;
    const reach = std.max(POINTER.reach * ptr.z, POINTER.burst * ptr.w) * cell;
    const near = std.saturate(1 - gap / reach);
    const stirred = near * (ptr.z * 0.9 + ptr.w);

    // A cell at rest at home stays put until the pointer comes
    if (me.heat < 0.002 && stirred < 0.002 && std.length(home - p) < 0.05 && std.length(v) < 1) {
      return;
    }

    // Loose, a cell drifts on a slow swirl, held back by the air; as it cools the lockup holds
    // it again and it springs home
    const heat = std.max(me.heat * std.exp(-dt * 1.4), stirred);
    const hold = 1 - heat;
    let force = std.mix(curlAt(p, s.motion.w) * 30 - v * 1.8, (home - p) * 64 - v * 15, hold);
    const strength = (POINTER.push * ptr.z + POINTER.blast * ptr.w) * cell;
    force += (toPointer / gap) * (near * near * strength);
    force += s.motion.xy * (near * POINTER.wake * ptr.z);

    v += force * dt;
    const speed = std.length(v);
    if (speed > MAX_SPEED) v = v * (MAX_SPEED / speed);
    p += v * dt;
    if (heat < 0.002 && std.length(home - p) < 0.05 && std.length(v) < 1) {
      p = d.vec2f(home);
      v = d.vec2f();
    }
    me.position = d.vec2f(p);
    me.velocity = d.vec2f(v);
    me.heat = heat;
  });

  const Out = { position: d.builtin.position, color: d.interpolate('flat', d.vec4f) };

  /** Dither the ramp using the cell's tilt and pointer light. Hidden cells have zero alpha. */
  const colorOf = (i: number) => {
    'use gpu';
    const s = params.$;
    const p = palette.$;
    const own = Ink.own(i);
    const threshold = own.w;
    if (s.look.x < threshold) return d.vec4f();
    if (Ink.badge(i) > 0.5 && s.look.z <= threshold) return d.vec4f();

    const toPointer = (Ink.home(i) - s.pointer.xy) / s.size.w;
    const light = std.exp(-std.dot(toPointer, toPointer)) * 0.9 * s.pointer.z;
    const at = std.clamp(0.5 + light * 0.55 + Ink.tilt(i), 0, 0.9999) * 4;
    const low = std.floor(at);
    const step = d.u32(std.select(low, low + 1, at - low > threshold));
    if (step === 0) return d.vec4f();
    let color = std.mix(p.rule.xyz, own.xyz, 0.45);
    if (step === 2) color = d.vec3f(own.xyz);
    if (step === 3) color = std.mix(own.xyz, p.accent.xyz, 0.6);
    if (step >= 4) color = p.bright.xyz;
    return d.vec4f(color, 1);
  };

  const vertex = tgpu.vertexFn({
    in: { vertexIndex: d.builtin.vertexIndex, instanceIndex: d.builtin.instanceIndex },
    out: Out,
  })(({ vertexIndex, instanceIndex }) => {
    'use gpu';
    const i = instanceIndex;
    if (!Ink.alive(i)) return { position: d.vec4f(), color: d.vec4f() };
    const color = colorOf(i);
    if (color.w < 0.5) return { position: d.vec4f(), color };
    const s = params.$;
    const corner = d.vec2f(
      d.f32(vertexIndex === 1 || vertexIndex >= 4),
      d.f32(vertexIndex === 2 || vertexIndex === 3 || vertexIndex === 5),
    );
    let at = d.vec2f(Ink.position(i));
    let side = Ink.side(i);
    const hand = s.hand;
    if (hand.z > 0 && Ink.badge(i) < 0.5) {
      const t = std.saturate((hand.z - Ink.order(i) * 0.45) / 0.55);
      const e = t * t * (3 - 2 * t);
      const to = hand.xy + Ink.mark(i);
      const bend = d.vec2f(to.x, at.y);
      const u = 1 - e;
      at = at * (u * u) + bend * (2 * u * e) + to * (e * e);
      side = std.mix(side, 1, e);
    }
    const pixel = (at + (corner - 0.5) * side) * s.look.y;
    const clip = (pixel / s.size.xy) * 2 - 1;
    return { position: d.vec4f(clip.x, -clip.y, 0, 1), color };
  });

  const fragment = tgpu.fragmentFn({
    in: { color: d.interpolate('flat', d.vec4f) },
    out: d.vec4f,
  })(({ color }) => {
    'use gpu';
    return d.vec4f(color);
  });

  const stepping = root.pipe(cells.inject()).createComputePipeline({ compute: kernel });
  const drawing = root.pipe(cells.inject()).createRenderPipeline({
    vertex,
    fragment,
    targets: { format },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'always' },
  });

  let box = [0, 0, 0, 0];
  let badgeBox = [0, 0, 0, 0];
  let badgeShown = 1;
  let cell = 1;
  const handoff = { x: 0, y: 0, amount: 0 };
  const motion = { vx: 0, vy: 0, burst: 0, over: 0, x: -1e4, y: -1e4, time: 0 };

  /**
   * Lays the cells out at their homes in `rect`, the lockup, and `badgeRect`, CSS pixels of the
   * canvas, when either moves.
   */
  const layout = (
    rect: readonly [number, number, number, number],
    badgeRect: readonly [number, number, number, number] = [0, 0, 0, 0],
  ) => {
    const lockupAt = rect.map((x) => Math.round(x));
    const badgeAt = badgeRect.map((x) => Math.round(x));
    if (same(lockupAt, box) && same(badgeAt, badgeBox) && cells.count > 0) return;
    box = lockupAt;
    badgeBox = badgeAt;
    cells.clear();
    const [left, top, width, height] = lockupAt as [number, number, number, number];
    if (!(width > 0 && height > 0)) return;
    cell = height / ROWS;
    const rows = ROWS;
    const columns = Math.max(1, Math.ceil(width / cell));
    const data = read(columns, rows);
    for (let j = 0; j < rows; j++) {
      for (let i = 0; i < columns; i++) {
        const at = (j * columns + i) * 4;
        if ((data[at + 3] ?? 0) < 128 || cells.count >= CAPACITY) continue;
        const r = data[at];
        const g = data[at + 1];
        const b = data[at + 2];
        const home = [left + (i + 0.5) * cell, top + (j + 0.5) * cell] as [number, number];
        cells.add({
          home,
          own: [r / 255, g / 255, b / 255, bayer(i, j)],
          tilt: 0,
          mark: [i + 0.5, j + 0.5],
          order: (i / columns) * 0.7 + bayer(i, j) * 0.3,
          side: cell,
          badge: 0,
          position: home,
          velocity: [0, 0],
          heat: 0,
        });
      }
    }

    // The badge, a cell a CSS pixel so its wordmark reads
    const [bx, by, bw, bh] = badgeAt as [number, number, number, number];
    if (!readBadge || !(bw > 0 && bh > 0)) return;
    const badge = readBadge(bw, bh);
    for (let j = 0; j < bh; j++) {
      for (let i = 0; i < bw; i++) {
        const at = (j * bw + i) * 4;
        if ((badge[at + 3] ?? 0) < 128 || cells.count >= CAPACITY) continue;
        const r = badge[at];
        const g = badge[at + 1];
        const b = badge[at + 2];
        const lum = (0.45 * r + 0.25 * g + 0.3 * b) / 255;
        const home = [bx + i + 0.5, by + j + 0.5] as [number, number];
        cells.add({
          home,
          own: [r / 255, g / 255, b / 255, bayer(i, j)],
          tilt: (lum - 0.5) * 0.3,
          mark: [0, 0],
          order: 0,
          side: 1,
          badge: 1,
          position: home,
          velocity: [0, 0],
          heat: 0,
        });
      }
    }
  };

  return {
    cells,
    initAsync: () => Promise.all([stepping.initAsync(), drawing.initAsync()]),
    layout,

    /** The pointer in CSS pixels of the canvas, or off the stage. */
    point(x: number, y: number, dt: number) {
      if (motion.over > 0 && dt > 0) {
        motion.vx = motion.vx * 0.5 + ((x - motion.x) / dt) * 0.5;
        motion.vy = motion.vy * 0.5 + ((y - motion.y) / dt) * 0.5;
      }
      motion.x = x;
      motion.y = y;
      motion.over = 1;
    },
    leave() {
      motion.over = 0;
      motion.vx = 0;
      motion.vy = 0;
    },
    /** A press at (x, y): the cells near it burst, by `amount` of a full burst. */
    burst(x: number, y: number, amount = 1) {
      motion.x = x;
      motion.y = y;
      motion.burst = amount;
    },

    /** How much of the badge shows, from 0 to 1, as the title fades: it dissolves cell by cell. */
    fade(shown: number) {
      badgeShown = shown;
    },

    /** Header transition progress from 0 to 1. The mark starts at canvas coordinates (x, y) in CSS pixels. */
    hand(x: number, y: number, amount: number) {
      handoff.x = x;
      handoff.y = y;
      handoff.amount = amount;
    },

    /** Whether (x, y), CSS pixels of the canvas, is within `margin` of the lockup. */
    near(x: number, y: number, margin: number) {
      return (
        x >= box[0] - margin &&
        x <= box[0] + box[2] + margin &&
        y >= box[1] - margin &&
        y <= box[1] + box[3] + margin
      );
    },

    /** Advance by `dt` seconds and upload visibility and canvas size in device pixels. */
    step(
      frame: Frame,
      dt: number,
      shown: number,
      surface: { width: number; height: number; pixelRatio: number },
      still: boolean,
    ) {
      const step = still ? 0 : Math.min(dt, 1 / 30);
      motion.time += step;
      params.write({
        pointer: d.vec4f(motion.x, motion.y, motion.over, motion.burst),
        motion: d.vec4f(motion.vx, motion.vy, step, motion.time),
        look: d.vec3f(shown, surface.pixelRatio, badgeShown),
        size: d.vec4f(surface.width, surface.height, cell, LENS * cell),
        hand: d.vec4f(handoff.x, handoff.y, handoff.amount, 0),
      });
      motion.burst *= Math.exp(-step * 3);
      motion.vx *= Math.exp(-step * 10);
      motion.vy *= Math.exp(-step * 10);
      if (motion.burst < 0.01) motion.burst = 0;
      cells.flush();
      if (step > 0 && cells.span > 0) {
        const pass = frame.encoder.beginComputePass({ label: 'ink' });
        pass.setPipeline(stepping);
        pass.dispatchWorkgroups(Math.ceil(cells.span / WORKGROUP));
        pass.end();
      }
    },

    /** Draws the cells into the stack's last pass. */
    draw(pass: TgpuRenderPass) {
      if (cells.span > 0) {
        pass.setPipeline(drawing);
        pass.draw(6, cells.span);
      }
    },

    destroy() {
      cells.destroy();
      params.buffer.destroy();
      palette.buffer.destroy();
    },
  };
}

export type LooseLockup = ReturnType<typeof createInk>;
