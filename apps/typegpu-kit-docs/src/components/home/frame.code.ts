// The frame the home page builds, as it shows it: each region is a block of code beside its plate.
// In a region, `knob.name` and `tuned.$.name` show as numbers to drag, a line ending in
// `// [module]` strikes through while that module is out, and a line ending in `// hide` is not
// shown.

import {
  bloom,
  finish,
  lighting,
  passes,
  table,
  tone,
  type CameraBase,
  type Targets,
} from '@typegpu/kit';
import * as grid from '@typegpu/kit/grid';
import { d, std, type TgpuRoot } from 'typegpu';
import { bayerCell } from '../site/dither.ts';
import {
  AIR,
  Bird,
  birdsDrawing,
  bornAt,
  CAPACITY,
  createLamps,
  createSteering,
  degreesUp,
  groundDrawing,
  LOW,
  Seen,
  seenOf,
} from './birds.ts';
import { knob } from './knobs.ts';
import { flock, huddle, orbit } from './rules.ts';

/** Texels a side of the frame and every target drawn at its size. */
export const SIZE = 512;

/** What the frame draws into: lit colour, normals and depth, and the graded frame. */
export type Screen = Targets<{
  hdr: { format: 'rgba16float'; sampled: true };
  normals: { format: 'rgba8unorm'; sampled: true };
  depth: { format: 'depth24plus' };
  final: { format: 'rgba8unorm'; sampled: true };
}>;

/** The sun and the sky with the lighting in. */
export const LIT = {
  sunIntensity: 2.4,
  sunColor: d.vec3f(1, 0.93, 0.84),
  skyIntensity: 0.1,
  skyTop: d.vec3f(0.32, 0.36, 0.42),
  skyBottom: d.vec3f(0.05, 0.05, 0.05),
};

const Normal = d.struct({ normal: d.vec3f });

/** The normals each renderer saw, as 0 to 1 per axis, for the stack. */
const normals = passes.output(d.vec4f, {
  varyings: Normal,
  vary: (corner: { normal: d.v3f }) => {
    'use gpu';
    return Normal({ normal: corner.normal });
  },
  write: (_pixel: unknown, carried) => {
    'use gpu';
    return d.vec4f(std.normalize(carried.normal) * 0.5 + 0.5, 1);
  },
});

/** Bloom's bright part, of blue light alone: the lamps and their pools glow, the grey doesn't. */
const lampGlow = (color: d.v3f) => {
  'use gpu';
  const lean = (color.z - std.max(color.x, color.y)) / std.max(color.z, 1e-4);
  return bloom.threshold(color) * std.saturate(lean * 2 - 0.4);
};

/** The grade's numbers: texels a cell, the levels of light, and 0 while the grade is out. */
const Grade = d.struct({ cell: d.f32, levels: d.f32, on: d.f32 });

/**
 * A grade for `finish`: the picture in square cells, its light in a few levels, dithered between
 * them, with its hue kept.
 */
function dither(root: TgpuRoot, initial: { cell: number; levels: number }) {
  const settings = root.createUniform(Grade, { ...initial, on: 1 });
  const grade = (color: d.v3f, uv: d.v2f) => {
    'use gpu';
    const g = settings.$;
    if (g.on < 0.5) return d.vec3f(color);
    const cell = std.floor((uv * SIZE) / g.cell);
    const picture = finish.picture(((cell + 0.5) * g.cell) / SIZE);
    const value = tone.luminance(picture);
    const last = g.levels - 1;
    const at = std.saturate(value) * last;
    const low = std.floor(at);
    const stepped = std.select(low, low + 1, at - low > bayerCell(d.vec2u(cell))) / last;
    return (picture / std.max(value, 1e-4)) * stepped;
  };
  return { settings, grade };
}

/**
 * Builds the frame into `screen`, seen by `view`. `has` says whether a module is in the frame, as
 * the reader takes them out.
 */
// oxfmt-ignore
export function createFrame(root: TgpuRoot, screen: Screen, view: CameraBase, has: (module: 'shadow') => boolean) {
  // #region birds
  const birds = table.create(root, Bird, { capacity: CAPACITY });
  const rows = Array.from({ length: knob.count }, () => birds.add(bornAt()));
  // #endregion

  // #region grid
  const cells = grid.create(root, { // [grid]
    ...birds.live(), low: LOW, size: AIR, sight: 1.4, item: Seen, // [grid]
    position: Bird.position.pipe(birds.inject()), // [grid]
    itemOf: seenOf.pipe(birds.inject()), // [grid]
  }); // [grid]
  const steering = createSteering(root, birds, cells, { // [grid]
    sight: knob.sight, rules: { flock, orbit, huddle }, // [grid]
  }); // [grid]
  // #endregion

  // #region shadow
  const shadow = { // [shadow]
    size: 1024, extent: 20, filter: 4, enabled: () => has('shadow'), // [shadow]
  } as const; // [shadow]
  const sunDirection = degreesUp(knob.elevation);
  // #endregion

  // #region lit
  const lamps = createLamps(root, birds, { // [lit]
    range: knob.range, intensity: knob.intensity, // [lit]
  }); // [lit]
  const light = lighting.create(root, { // [lit]
    settings: { ...LIT, sunDirection }, // [lit]
    shadow, // [shadow]
    points: { light: lamps.light, candidates: cells.candidates() }, // [grid,lit]
  }); // [lit]
  // #endregion

  // #region normals
  const scene = passes
    .create(root, screen, { hdr: passes.color, normals, depth: passes.depth }, {
      clear: [0.01, 0.011, 0.013, 1], label: 'scene',
    })
    .pipe(view.inject())
    .pipe(light.inject());
  const chevrons = birdsDrawing(root, birds, lamps, { length: knob.length });
  const shown = [scene.add(groundDrawing(30)), scene.add(chevrons)];
  const cast = [light.shadow.add(chevrons)]; // [shadow]
  // #endregion

  // #region bloom
  const glow = bloom.create(root, { // [bloom]
    source: () => screen.textures.hdr, levels: 5, bright: lampGlow, // [bloom]
    settings: { threshold: knob.threshold, knee: 0.4 }, // [bloom]
  }); // [bloom]
  // #endregion

  // #region final
  const print = dither(root, { cell: knob.cell, levels: knob.levels });
  const final = finish.create(root, {
    format: 'rgba8unorm', source: () => screen.textures.hdr,
    glow, // [bloom]
    grade: print.grade,
    settings: { glowIntensity: knob.strength, vignette: 0.25, dither: 0 },
  });
  // #endregion

  return {
    birds, rows, cells, steering, lamps, light, scene, chevrons, shown, cast, glow, print, final,
  };
}
