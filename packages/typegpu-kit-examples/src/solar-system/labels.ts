import { cameras, type Font, instances, text, typeset } from '@typegpu/kit';
import type { Inject } from '@typegpu/kit/inject';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { PLANETS, SUN } from './data.ts';
import {
  aboveness,
  bodies,
  DETAILS_ABOVE,
  inverse,
  NAME_LIFT,
  onPicture,
  SIZE_DISTANCE,
} from './planets.ts';

/**
 * The kinds of a planet's labels, and `PER`, the labels each planet has. Its details are written
 * twice, below its band and above its name, and show where the bodies leave them clear. A chosen
 * planet's name is written again at the head of the details below, for when the sun is behind it.
 */
const [NAME, BAND, DETAILS, ABOVE, UNDER, PER] = [0, 1, 2, 3, 4, 5];
/** Glyphs a label holds; spaces take none. */
const LENGTH = 100;
/** World units per em of names, bands and details. */
const EM = [0.5, 0.3, 0.3, 0.3, 0.5];
const BAND_EM = EM[BAND];
/** The colour of bands and details. Names take their planet's colour, lightened. */
const FACTS = d.vec4f(0.8, 0.84, 0.95, 1);
const COLORS = [FACTS, FACTS, FACTS];
const ANCHORS = [
  { baseline: 'alphabetic' },
  { align: 'left' },
  { baseline: 'top' },
  { baseline: 'bottom' },
  { baseline: 'top' },
] as const;
/** World units at `SIZE_DISTANCE` the details below move down to make room for a name above them. */
const NAME_LINE = 0.62;
/** Glyphs a second that type on. */
const TYPING = 28;
/** Seconds before the first planet's labels type on. */
const INTRO = 0.3;
/** Seconds between one planet's labels typing on and the next's. */
const STAGGER = 0.12;
/** Radians a second a band turns. */
const BAND_TURN = 0.25;
/** Radians a band leans about `LEAN_AXIS`, one way on odd planets and the other on even ones. */
const BAND_LEAN = 0.35;
const LEAN_AXIS = d.vec3f(0.6, 0, 0.8);
/** World units from the eye over which a planet's labels shrink away as it nears them. */
const NEAR = d.vec2f(4, 8);
/**
 * Boxes round a name and round the details, in world units at `SIZE_DISTANCE`: half the height and
 * how far the middle sits above the baseline. A name's covers the arch's rise at its ends, and the
 * details' their four lines below their top, or above their bottom when above the name.
 */
const NAME_BOX = d.vec2f(0.34, 0.3);
const [DETAILS_BOX, ABOVE_BOX] = [d.vec2f(0.75, -0.75), d.vec2f(0.75, 0.75)];
/** Picture heights of clear space a name needs from other names and bodies before it shows fully. */
const CLEARANCE = 0.012;

/** Seconds of simulated and of real time, and how many times their size labels are drawn. */
const Clock = d.struct({ sim: d.f32, real: d.f32, grow: d.f32 });
/** When a planet was last chosen or let go, and 1 while it is chosen. */
const Choice = d.struct({ since: d.f32, open: d.u32 });
/** A circle a label wraps round, its axis tilted, its baseline lifted up the axis. */
const Arc = d.struct({
  centre: d.vec3f,
  tilt: d.vec4f,
  radius: d.f32,
  /** How many times its `EM` the label's glyphs are drawn. */
  size: d.f32,
  lift: d.f32,
  start: d.f32,
  /** Radians round the circle to its point nearest the eye. */
  front: d.f32,
});

/** How far two boxes on the picture stand apart, negative where they overlap. */
const apart = (a: d.v4f, b: d.v4f) => {
  'use gpu';
  const gap = std.abs(a.xy - b.xy) - (a.zw + b.zw);
  return std.max(gap.x, gap.y);
};

/** The planet a label belongs to. */
const planetOf = (label: number) => {
  'use gpu';
  return std.intdiv(label, d.u32(PER));
};

/**
 * Writes each planet's name in the world, arched over it at one size on screen. While a planet is
 * chosen, a band of its facts types on round it on a tilted ring that turns, and its details type
 * on below the band.
 */
export function createLabels(root: TgpuRoot, font: Font, planets: { inject: () => Inject }) {
  const clock = root.createUniform(Clock);
  const choices = root.createReadonly(
    d.arrayOf(Choice, PLANETS.length),
    PLANETS.map(() => ({ since: -1e3, open: 0 })),
  );

  const measure = typeset.create(LENGTH);
  const widthOf = (line: string, kind: number) => measure.set(font, line, { em: EM[kind] }).width;
  // A band repeats its facts as often as they fit round it
  const lines = PLANETS.map((planet) => {
    const fits = Math.floor((Math.PI * 2 * planet.bandRadius) / widthOf(planet.band, BAND));
    const { name, band, details } = planet;
    const named = name.toUpperCase();
    return [named, band.repeat(Math.max(1, fits)), details, details, named];
  });
  /** The radius, at least `least`, at which a line spans about 1.8 radians facing the eye. */
  const arc = (line: string, kind: number, least: number) =>
    Math.max(least, widthOf(line, kind) / 1.8);
  /** Each label's radius and lift at `SIZE_DISTANCE`, and the least radius it keeps nearer. */
  const wraps = root.createReadonly(
    d.arrayOf(d.vec3f, PLANETS.length * PER),
    PLANETS.flatMap(({ radius, bandRadius: band }, i) => {
      const [name, , details] = lines[i] as [string, string, string];
      return [
        d.vec3f(arc(name, NAME, 0), radius + NAME_LIFT, radius + 0.3),
        d.vec3f(band, -BAND_EM * 0.35, band),
        d.vec3f(arc(details, DETAILS, band), -band - 0.15, band),
        d.vec3f(arc(details, DETAILS, band), radius + NAME_LIFT, band),
        d.vec3f(arc(name, UNDER, band), -band - 0.15, band),
      ];
    }),
  );

  /** The sun's radius, then each planet's extent, rings included, in world units. */
  const extents = root.createReadonly(d.arrayOf(d.f32, PLANETS.length + 1), [
    SUN.radius,
    ...PLANETS.map(({ extent }) => extent),
  ]);

  const arcOf = (label: number) => {
    'use gpu';
    const planet = planetOf(label);
    const band = label % PER === BAND;
    const name = label % PER === NAME;
    const wrap = wraps.$[label];
    const centre = d.vec3f(bodies.$[planet + 1].position);
    // Firefox on Windows fails to compile `sin` of a `select` between literals
    const lean = std.sin((d.f32(planet % 2) - 0.5) * BAND_LEAN);
    const leaning = d.vec4f(LEAN_AXIS * lean, std.sqrt(1 - lean * lean));
    const tilt = std.select(d.vec4f(0, 0, 0, 1), leaning, band);
    const eye = instances.rotate(inverse(tilt), cameras.current.$.position - centre);
    const front = std.atan2(-eye.z, eye.x);
    // Names keep clear of the planet as they shrink, and facts of the name as they grow
    const size = std.length(eye) / SIZE_DISTANCE;
    const radius = std.max(wrap.z, wrap.x * size);
    // Text facing the eye lifts so its nearest point keeps above or below the planet on screen
    const rise = std.clamp(eye.y / std.length(eye), -0.9, 0.9);
    // Details above a name lift as far as it does, then over its upright text, which the eye
    // sees foreshortened
    const above = label % PER === ABOVE;
    let lift = wrap.y * std.select(std.max(size, 1), d.f32(1), name || above);
    if (above) lift += DETAILS_ABOVE * size * std.sqrt(1 - rise * rise);
    const facing = (lift + radius * rise) / std.sqrt(1 - rise * rise);
    const turned = clock.$.sim * BAND_TURN * (1.2 - d.f32(planet) * 0.05);
    return Arc({
      centre,
      tilt,
      radius,
      size,
      lift: std.select(facing, lift, band),
      start: std.select(front, turned, band),
      front,
    });
  };

  /**
   * A name or details label as a box on the picture, from its `box`: its middle, then its half
   * width and height.
   */
  const boxOf = (label: number, box: d.v2f) => {
    'use gpu';
    const circle = arcOf(label);
    const grow = clock.$.grow;
    const front = d.vec3f(std.cos(circle.front), 0, -std.sin(circle.front)) * circle.radius;
    const baseline = onPicture(circle.centre + (front + d.vec3f(0, circle.lift, 0)) * grow);
    // Labels keep one size on the picture, so their boxes scale with the lens alone
    const scale = (cameras.lensOf(cameras.current.$).y / (2 * SIZE_DISTANCE)) * grow;
    const half = d.vec2f(wraps.$[label].x * 0.9, box.x) * scale;
    return d.vec4f(baseline + d.vec2f(0, box.y * scale), half);
  };

  /** How far planet `planet`'s details have moved above its name, from 0 to 1. */
  const abovenessOf = (planet: number) => {
    'use gpu';
    return aboveness(planet, wraps.$[planet * PER + DETAILS].x * 1.8, clock.$.grow);
  };

  /**
   * How far a chosen planet's name has moved to the head of its details below, from 0 to 1. It
   * moves while the sun stands behind it and the details keep below.
   */
  const nameUnder = (planet: number) => {
    'use gpu';
    if (choices.$[planet].open === 0) return d.f32(0);
    const eye = cameras.current.$;
    const sun = bodies.$[0].position;
    const disc = (SUN.radius * 1.2 * cameras.lensOf(eye).y) / (2 * std.distance(eye.position, sun));
    const box = boxOf(planet * PER + NAME, NAME_BOX);
    const outside = std.abs(onPicture(sun) - box.xy) - box.zw;
    const gap = std.length(std.max(outside, d.vec2f())) + std.min(std.max(outside.x, outside.y), 0);
    return (1 - std.smoothstep(0, CLEARANCE * 2, gap - disc)) * (1 - abovenessOf(planet));
  };

  /**
   * Fades a name in as it clears every other body, the chosen planet's band and details, and every
   * name that wins over it: the chosen planet's, or a nearer one. A chosen planet's name always
   * shows.
   */
  const clearance = (planet: number) => {
    'use gpu';
    if (choices.$[planet].open === 1) return d.f32(1);
    const eye = cameras.current.$;
    const lens = cameras.lensOf(eye);
    const box = boxOf(planet * PER + NAME, NAME_BOX);
    const away = std.distance(eye.position, bodies.$[planet + 1].position);
    let clear = d.f32(1);
    for (let body = d.u32(0); body <= PLANETS.length; body++) {
      if (body !== planet + 1) {
        const at = bodies.$[body].position;
        const distance = std.distance(eye.position, at);
        const other = body - 1;
        const chosen = body > 0 && choices.$[other].open === 1;
        let reach = extents.$[body];
        if (chosen) reach = wraps.$[other * PER + BAND].x * std.max(distance / SIZE_DISTANCE, 1);
        const disc = (reach * lens.y) / (2 * distance);
        const outside = std.abs(onPicture(at) - box.xy) - box.zw;
        const gap =
          std.length(std.max(outside, d.vec2f())) + std.min(std.max(outside.x, outside.y), 0);
        clear = std.min(clear, std.smoothstep(0, CLEARANCE, gap - disc));

        if (chosen || (body > 0 && distance < away)) {
          const rival = boxOf(other * PER + NAME, NAME_BOX);
          clear = std.min(clear, std.smoothstep(0, CLEARANCE, apart(rival, box)));
        }
        if (chosen) {
          const above = abovenessOf(other);
          const under = std.smoothstep(
            0,
            CLEARANCE,
            apart(boxOf(other * PER + DETAILS, DETAILS_BOX), box),
          );
          const over = std.smoothstep(
            0,
            CLEARANCE,
            apart(boxOf(other * PER + ABOVE, ABOVE_BOX), box),
          );
          clear = std.min(clear, std.min(std.mix(1, under, 1 - above), std.mix(1, over, above)));
        }
      }
    }
    return clear;
  };

  const place = (_position: d.v3f, offset: d.v2f, glyph: text.Placed) => {
    'use gpu';
    const kind = glyph.label % PER;
    const planet = planetOf(glyph.label);
    const choice = choices.$[planet];
    // Names always show; a planet's facts, and its name below, only while it is chosen
    const on = kind === NAME || choice.open === 1;

    // Glyphs grow in one after another, rising, and shrink back all at once. Names type on once.
    const real = clock.$.real;
    const intro = INTRO + d.f32(planet) * STAGGER;
    const since = std.select(std.max(choice.since, intro), intro, kind === NAME);
    const typed = (real - since) * TYPING;
    const opening = std.saturate(typed - d.f32(glyph.glyph));
    const closing = std.saturate(1 - (real - choice.since) * TYPING * 0.2);
    let shown = std.select(closing, opening, on);
    const middle = glyph.pen + d.vec2f(0.3, 0.35) * glyph.em;
    const circle = arcOf(glyph.label);
    // A band's glyphs grow in as they turn towards the eye and shrink away as they turn past
    if (kind === BAND) {
      const facing = std.cos(
        circle.start + (middle.x * circle.size) / circle.radius - circle.front,
      );
      shown *= std.smoothstep(0.45, 0.6, facing);
    }
    const local = middle + (offset - middle) * shown - d.vec2f(0, (1 - shown) * 0.5 * glyph.em);
    const eye = cameras.current.$;
    const grow = clock.$.grow;

    const near = std.smoothstep(NEAR.x, NEAR.y, std.distance(eye.position, circle.centre));
    const shrunk = (middle + (local - middle) * near) * circle.size;
    const angle = circle.start + shrunk.x / circle.radius;
    const around = d.vec3f(std.cos(angle), 0, -std.sin(angle)) * circle.radius;
    let lift = circle.lift;
    if (kind === DETAILS) lift -= nameUnder(planet) * NAME_LINE * circle.size;
    const point = (around + d.vec3f(0, lift + shrunk.y, 0)) * grow;
    const world = circle.centre + instances.rotate(circle.tilt, point);
    const clip = cameras.toClip(world);
    // The chosen planet's labels draw over every body, so nothing the viewer chose to read hides
    return std.select(clip, d.vec4f(clip.xy, 0, clip.w), kind !== NAME || choice.open === 1);
  };

  /**
   * A soft dark halo keeps text readable over orbit paths and stars, and a darker, wider one keeps
   * facts readable over the bodies they cross. Names fade where they would crowd another, and
   * details to whichever side of the planet the bodies leave clear.
   */
  const shade = (fragment: text.Fragment) => {
    'use gpu';
    let alpha = fragment.color.w;
    const planet = planetOf(fragment.label);
    if (fragment.label % PER === NAME) alpha *= clearance(planet) * (1 - nameUnder(planet));
    if (fragment.label % PER === UNDER) alpha *= nameUnder(planet);
    if (fragment.label % PER === DETAILS) alpha *= 1 - abovenessOf(planet);
    if (fragment.label % PER === ABOVE) alpha *= abovenessOf(planet);
    const fill = std.saturate(fragment.distance + 0.5) * alpha;
    const facts = fragment.label % PER !== NAME || choices.$[planet].open === 1;
    const width = std.select(d.f32(2), fragment.reach, facts);
    const dark = std.select(d.f32(0.5), d.f32(0.85), facts);
    const halo = std.saturate((fragment.round + width) / (width + 0.5)) * dark * alpha;
    return d.vec4f(fragment.color.xyz * fill, fill + (1 - fill) * halo);
  };

  const texts = text.create(root, font, {
    capacity: PLANETS.length * PER,
    length: LENGTH,
    place: tgpu.fn(place).pipe(planets.inject()),
    shade: tgpu.fn(shade).pipe(planets.inject()),
    label: 'labels',
  });

  for (const [i, planet] of PLANETS.entries()) {
    for (const [kind, line] of lines[i].entries()) {
      texts.write(i * PER + kind, line, { em: EM[kind], ...ANCHORS[kind] });
    }
    const name = d.vec4f(std.mix(planet.color, d.vec3f(1), 0.4), 1);
    const colors = [name, ...COLORS, name];
    texts.labels.patch(
      Object.fromEntries(colors.map((color, kind) => [i * PER + kind, { color }])),
    );
  }

  let chosen: number | undefined;
  const now = { sim: 0, real: 0, grow: 1 };

  return {
    drawing: texts.drawing,
    /** Each planet's details' width in world units at `SIZE_DISTANCE`. */
    widths: PLANETS.map((planet, i) => arc(lines[i][DETAILS], DETAILS, planet.bandRadius) * 1.8),

    /**
     * Turns the bands and types labels on to these seconds of simulated and real time, and draws
     * labels `grow` times their size.
     */
    tick(sim: number, real: number, grow: number) {
      now.sim = sim;
      now.real = real;
      now.grow = grow;
      clock.write(now);
    },

    /** Unravels planet `next`'s band and types its details on, and the reverse for the last. */
    choose(next: number | undefined, real: number) {
      if (next === chosen) return;
      const changes: Record<number, d.Infer<typeof Choice>> = {};
      if (chosen !== undefined) changes[chosen] = { since: real, open: 0 };
      if (next !== undefined) changes[next] = { since: real, open: 1 };
      choices.buffer.patch(changes);
      chosen = next;
    },

    destroy() {
      texts.destroy();
      choices.buffer.destroy();
      wraps.buffer.destroy();
      extents.buffer.destroy();
    },
  };
}
