import { cameras, passes, pipelines, text, type CameraBase, type Font } from '@typegpu/kit';
import type { Inject } from '@typegpu/kit/inject';
import { sdRoundedBox2d } from '@typegpu/sdf';
import { d, std, type TgpuRoot } from 'typegpu';

/** The labels: the star's name and its distance from the core. */
const [NAME, DISTANCE] = [0, 1];
/** Glyphs a label holds; spaces take none. */
const LENGTH = 24;
/** Points per em of the name and the distance. */
const EM = [16, 12.5];
/** Em between the name's characters. */
const TRACK = 0.07;

// The callout, in points from the star with y up, for a label on the star's upper right
/** Radius of the ring round the star, and the width of its line and of the leader. */
const RING = 9;
const STROKE = 1.25;
/** Where the leader turns from the ring's diagonal to run level into the plate. */
const ELBOW = 21;
/** Where the plate starts, and its padding, corner radius and border width. */
const PLATE_X = 31;
const PAD = d.vec2f(11, 9);
const CORNER = 7;
const BORDER = 1;
/** Points from the name's baseline to the distance's capitals. */
const LINE_GAP = 7;
/** Angles in radians: where the leader leaves the ring, and a full turn; and cos 45°. */
const QUARTER = Math.PI / 4;
const TAU = Math.PI * 2;
const DIAGONAL = Math.SQRT1_2;
/** Points of slack round the callout's quad, for antialiasing. */
const SLACK = 2;
/** Clip-space x and y past which the callout flips to the star's left or below it. */
const FLIP = d.vec2f(0.35, 0.55);

/** Linear colours: name, distance, ring and leader, plate fill and plate border. */
const NAME_COLOR = d.vec4f(0.95, 0.96, 1, 1);
const DISTANCE_COLOR = d.vec4f(0.62, 0.7, 0.88, 1);
const ACCENT = d.vec3f(0.5, 0.66, 1);
const PLATE = d.vec4f(0.012, 0.016, 0.04, 0.86);
const EDGE = d.vec4f(0.5, 0.66, 1, 0.38);

/** Seconds the ring takes to draw round, the leader to run out and the plate to open. */
const RING_TIME = 0.2;
const LEAD_TIME = 0.12;
const PLATE_TIME = 0.16;
/** Glyphs a second that type on once the plate is open. */
const TYPING = 48;
/** Points a glyph rises through as it types on. */
const RISE = 3;
/** Seconds from a choice until every glyph has typed on. */
const SETTLED = RING_TIME + LEAD_TIME + PLATE_TIME + (2 * LENGTH) / TYPING;

/**
 * The label's state: the chosen star, when it was chosen, now in seconds, the widest line in
 * points, and 1 while a star is chosen.
 */
const Callout = d.struct({
  position: d.vec3f,
  since: d.f32,
  now: d.f32,
  width: d.f32,
  open: d.u32,
});

const Corner = d.struct({ position: d.vec4f, local: d.vec2f });
const Seen = d.struct({ color: d.vec4f });

const ease = (t: number) => {
  'use gpu';
  const s = std.saturate(t);
  return s * s * (3 - 2 * s);
};

/**
 * Distance from `p` to the segment from `a` to `b`. Stays finite for a zero-length segment, where
 * `sdLine` divides by zero.
 */
const segment = (p: d.v2f, a: d.v2f, b: d.v2f) => {
  'use gpu';
  const along = b - a;
  const t = std.saturate(std.dot(p - a, along) / std.max(std.dot(along, along), 1e-6));
  return std.length(p - a - along * t);
};

/** Coverage of a pixel at signed distance `distance` in points, at `pixels` pixels a point. */
const cover = (distance: number, pixels: number) => {
  'use gpu';
  return std.saturate(0.5 - distance * pixels);
};

/** `color` over `under`, both premultiplied. */
const over = (under: d.v4f, color: d.v4f) => {
  'use gpu';
  return color + under * (1 - color.w);
};

/**
 * The chosen star's callout, pinned beside it at a constant size: a ring round the star, a leader
 * out to a dark plate, and on the plate its name over its distance from the core. The ring draws
 * round first, the leader runs out, the plate opens, and the glyphs type on.
 */
export function createLabel(root: TgpuRoot, font: Font, camera: CameraBase) {
  const callout = root.createUniform(Callout);

  // The plate holds the name's capitals and the distance's. Each line is anchored at the top of
  // its capitals, in points from the plate's middle: the name's `PAD.y` below the plate's top,
  // the distance's `LINE_GAP` below the name's baseline
  const capitals = font.data.capHeight;
  const plateHeight = PAD.y * 2 + capitals * (EM[NAME] + EM[DISTANCE]) + LINE_GAP;
  const nameTop = plateHeight / 2 - PAD.y;
  const distanceTop = nameTop - capitals * EM[NAME] - LINE_GAP;

  /** Seconds since the star was chosen. */
  const since = () => {
    'use gpu';
    return callout.$.now - callout.$.since;
  };

  /** Glyphs typed on so far, once the plate has opened. */
  const typedGlyphs = () => {
    'use gpu';
    return (since() - RING_TIME - LEAD_TIME - PLATE_TIME) * TYPING;
  };

  /** The open plate's size in points. */
  const plateSize = () => {
    'use gpu';
    return d.vec2f(callout.$.width + PAD.x * 2, plateHeight);
  };

  /** -1 or 1 per axis: which way from the star the callout points, to stay on screen. */
  const sides = () => {
    'use gpu';
    const clip = cameras.toClip(callout.$.position);
    const x = std.select(d.f32(1), d.f32(-1), clip.x > clip.w * FLIP.x);
    const y = std.select(d.f32(1), d.f32(-1), clip.y > clip.w * FLIP.y);
    return d.vec2f(x, y);
  };

  /** Sets each glyph on the plate, which mirrors with the callout but reads left to right. */
  const place = (position: d.v3f, offset: d.v2f, glyph: text.Placed) => {
    'use gpu';
    const shown = std.saturate(typedGlyphs() - d.f32(glyph.glyph));
    const local = d.vec2f(offset.x, offset.y - (1 - shown) * RISE);

    const plate = plateSize();
    const centre = d.vec2f(PLATE_X + plate.x / 2, ELBOW) * sides();
    const start = d.vec2f(
      centre.x - plate.x / 2 + PAD.x,
      centre.y + std.select(nameTop, distanceTop, glyph.label !== NAME),
    );
    return text.pinned(position, local + start);
  };

  const shade = (fragment: text.Fragment) => {
    'use gpu';
    const shown = std.saturate((typedGlyphs() - d.f32(fragment.glyph)) * 2);
    const alpha = fragment.color.w * shown * std.saturate(fragment.distance + 0.5);
    return d.vec4f(fragment.color.xyz * alpha, alpha);
  };

  const set = text.create(root, font, {
    capacity: 2,
    length: LENGTH,
    camera,
    place,
    shade,
    label: 'star label',
  });
  set.labels.patch({ [NAME]: { color: NAME_COLOR }, [DISTANCE]: { color: DISTANCE_COLOR } });

  const fill: Inject = (cfg) => cfg.pipe(camera.inject());

  /** One quad round the callout, shaded with distances to its ring, leader and plate. */
  const shapes: passes.Drawing<typeof Corner, d.Infer<typeof Seen>> = {
    label: 'star callout',
    Vertex: Corner,
    at: (vertexIndex: number, _instanceIndex: number) => {
      'use gpu';
      const plate = plateSize();
      const low = d.vec2f(-RING - SLACK);
      const high = d.vec2f(PLATE_X + plate.x, ELBOW + plate.y / 2) + SLACK;
      const corner = d.vec2f(d.f32(vertexIndex & 1), d.f32(vertexIndex >>> 1));
      const local = std.mix(low, high, corner);
      const position = text.pinned(callout.$.position, local * sides());
      // A cleared label collapses beyond the far plane
      const hidden = d.vec4f(0, 0, 2, 1);
      return Corner({ position: std.select(hidden, position, callout.$.open === 1), local });
    },
    pixelOf: (corner) => {
      'use gpu';
      const p = corner.local;
      const t = since();
      const pixels = passes.viewport.$.pixelRatio;

      // The ring draws round from where the leader leaves it
      const turned = std.fract((std.atan2(p.y, p.x) - QUARTER) / TAU + 1);
      const ringDrawn = turned <= ease(t / RING_TIME);
      const ring = std.select(d.f32(1e3), std.abs(std.length(p) - RING) - STROKE / 2, ringDrawn);

      // The leader runs out along the diagonal, then level into the plate
      const leaving = d.vec2f(RING * DIAGONAL);
      const elbow = d.vec2f(ELBOW);
      const end = d.vec2f(PLATE_X, ELBOW);
      const first = std.length(elbow - leaving);
      const total = first + (PLATE_X - ELBOW);
      const run = ease((t - RING_TIME) / LEAD_TIME) * total;
      const diagonal = leaving + (elbow - leaving) * std.saturate(run / first);
      const level = elbow + (end - elbow) * std.saturate((run - first) / (total - first));
      const lead = std.min(segment(p, leaving, diagonal), segment(p, elbow, level));
      const leader = std.select(d.f32(1e3), lead - STROKE / 2, run > 0);

      // The plate opens from the leader's end
      const opening = ease((t - RING_TIME - LEAD_TIME) / PLATE_TIME);
      const plate = plateSize() * d.vec2f(opening, 1);
      const centre = d.vec2f(PLATE_X + plate.x / 2, ELBOW);
      const box = sdRoundedBox2d(p - centre, plate / 2, std.min(CORNER, plate.x / 2));

      let color = PLATE * cover(box, pixels) * opening;
      color = over(color, EDGE * cover(std.abs(box + BORDER / 2) - BORDER / 2, pixels) * opening);
      const line = std.max(cover(ring, pixels), cover(leader, pixels));
      color = over(color, d.vec4f(ACCENT, 1) * line);
      return Seen({ color });
    },
    color: passes.shaded,
    blend: pipelines.blend.over,
    depthStencil: { depthWriteEnabled: false, depthCompare: 'always' },
    primitive: { topology: 'triangle-strip' },
    inject: () => fill,
    draw: (pass) => pass.draw(4),
  };

  /** What the callout uniform holds. */
  const state = { position: d.vec3f(), since: 0, now: 0, width: 0, open: 0 };
  let animating = false;
  const write = (now: number) => {
    state.now = now;
    callout.write(state);
  };

  return {
    /** Draws the ring, leader and plate; draw it before `text`. */
    shapes,
    /** Draws the name and distance over the plate. */
    text: set.drawing,

    /** Labels the star at `position` from `time`, or clears the label for null. */
    show(
      star: { readonly index: number; readonly position: readonly number[] } | null,
      time: number,
    ) {
      if (!star) {
        for (const label of [NAME, DISTANCE]) set.write(label, '', { em: EM[label] });
        state.open = 0;
        animating = false;
        write(time);
        return;
      }
      const [x, y, z] = star.position;
      const position = d.vec3f(x, y, z);
      const lines = [`STAR ${star.index}`, `${Math.hypot(x, y, z).toFixed(2)} from the core`];
      state.width = 0;
      for (const [label, line] of lines.entries()) {
        const tracking = label === NAME ? TRACK : 0;
        const style = { em: EM[label], align: 'left', baseline: 'cap', tracking } as const;
        state.width = Math.max(state.width, set.write(label, line, style).width);
      }
      set.labels.patch({ [NAME]: { position }, [DISTANCE]: { position } });
      state.position = position;
      state.since = time;
      state.open = 1;
      animating = true;
      write(time);
    },

    /** Advances the animation; writes nothing once it has finished. */
    tick(time: number) {
      if (!animating) return;
      write(time);
      animating = time - state.since < SETTLED;
    },

    destroy() {
      set.destroy();
    },
  };
}
