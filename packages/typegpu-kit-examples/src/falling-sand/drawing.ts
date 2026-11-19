import { passes, tone } from '@typegpu/kit';
import { perlin2d } from '@typegpu/noise';
import { d, std, tgpu, type TgpuRoot, type TgpuUniform } from 'typegpu';
import {
  type Cells,
  GRID,
  Kind,
  lifeOf,
  Matter,
  MATERIALS,
  matterOf,
  OPAQUE,
  RELIEF_CELLS,
  shadeOf,
} from './cells.ts';
import type { Densities } from './densities.ts';
import { glowOf, type Light } from './light.ts';

/** Blue-violet ambient light outside the radiance field. */
const AMBIENT = d.vec3f(0.03, 0.028, 0.06);
/** Ambient rock colour preserves visible texture in darkness. */
const ROCK_AMBIENT = d.vec3f(0.035, 0.032, 0.08);
/** Far-wall colour, darker and bluer than foreground rock. */
const BACK = d.vec3f(0.028, 0.026, 0.042);
/** Air-light scattering strength for dust and vapour. */
const HAZE = 0.012;

/** How quickly light fades into a solid, per cell past its surface. */
const RIM_FALLOFF = 0.45;
/** Light multiplier for exposed solid edges relative to the cut face. */
const EDGE = 1.35;
const STONE_STEPS = 4;
const OBSIDIAN_GLINT = { shade: 0.86, gain: 1.6 };
/** Crystal brightness at base and tip, and outline highlight strength. */
const CRYSTAL_SHADE = { base: 0.35, tip: 1.2, outline: 1.3 };
/** Ambient grain colour and lighting desaturation strength. */
const GRAINS = { ambient: d.vec3f(0.07, 0.055, 0.04), wash: 0.4 };
/** How much more light grains catch while they fall through the air, spread out. */
const FALLING_GRAINS = 1.3;

const WATER = {
  /** RGB tint for the far wall seen through water. Depth adds exponential attenuation. */
  through: d.vec3f(0.25, 0.7, 0.78),
  /** Water scattering strength and depth scaling. */
  scatter: 0.55,
  deep: 0.06,
  /** Water-surface glint strength. */
  glint: 0.7,
  /** Foam colour and lighting multipliers for falling or splashing water. */
  foam: d.vec3f(0.6, 0.8, 0.85),
  foamLight: 3,
  foamAmbient: 8,
  /** Maximum water depth in rows. */
  depth: 8,
};
/** Opacity of an isolated falling drop against the background. */
const DROP = 0.06;

/** Vapour density, opacity and light. `peak` keeps steam over lava amber rather than white. */
const VAPOUR = {
  full: 0.32,
  steam: 0.85,
  smoke: 0.55,
  light: 3.2,
  peak: 3,
  ambient: 0.6,
  solids: 0.45,
};
const albedoOf = (matter: number) =>
  d.vec3f(...(MATERIALS[matter].albedo as [number, number, number]));
const STEAM = albedoOf(Matter.steam);
const SMOKE = albedoOf(Matter.smoke);
const SAND = albedoOf(Matter.sand);
const WATER_TINT = albedoOf(Matter.water);

/** Density for a full falling ribbon, and water and lava streak speeds in cells per second. */
const RIBBON = { full: 0.3, waterSpeed: 30, lavaSpeed: 18 };
/** Minimum water, lava and sand stream densities for drawing ribbons. */
const RIBBON_LEAST = d.vec3f(0.1, 0.04, 0.04);
/** Falling-water colour between foam and transparent water. */
const FALL = d.vec3f(0.22, 0.5, 0.56);
const LAVA_FALL = { edge: d.vec3f(1.6, 0.16, 0.01), core: d.vec3f(5, 2.2, 0.4) };
/**
 * Sand-stream speed in cells per second, full-density threshold, grain coverage at edges and
 * centre, and glint, light and dust strengths.
 */
const SAND_FALL = {
  speed: 26,
  full: 0.05,
  edge: 0.35,
  middle: -0.05,
  glint: 0.4,
  light: 2.2,
  dust: 0.45,
};

/** Lava drift in cells per second, noise frequencies per cell, and body, crust and vein colours. */
const LAVA = {
  drift: d.vec2f(1.2, -0.6),
  swirl: 0.05,
  patches: 0.09,
  veins: 0.11,
  spread: 0.035,
  body: d.vec3f(2.2, 0.28, 0.02),
  crust: d.vec3f(0.5, 0.04, 0.01),
  vein: d.vec3f(4.5, 1.9, 0.35),
};

const EXTINCTION_TINT = d.vec3f(0.11, 0.13, 0.17);
const DIAGNOSTIC_EMISSION_GAIN = 0.25;

export type View = 'cave' | 'radiance' | 'medium';
const VIEWS: Record<View, number> = { cave: 0, radiance: 1, medium: 2 };
const Seen = d.struct({ color: d.vec4f });

/** Quantised stone shade around 1, with strata, grain and cracks. */
const stoneAt = (p: d.v2f, beds: number) => {
  'use gpu';
  const grain = perlin2d.sample(p * 0.55) * 0.5 + 0.5;
  const crack = std.smoothstep(0.06, 0.01, std.abs(perlin2d.sample(p * 0.13 + 19)));
  const step = std.floor((beds * 0.75 + grain * 0.25) * STONE_STEPS) / (STONE_STEPS - 1);
  return (0.3 + step * 1.4) * (1 - crack * 0.55);
};

/** Far-wall shade around 1. Uses softer strata and grain than foreground rock. */
const wallAt = (p: d.v2f) => {
  'use gpu';
  const beds = perlin2d.sample(p * d.vec2f(0.02, 0.06) + 3) * 0.5 + 0.5;
  const grain = perlin2d.sample(p * 0.35) * 0.5 + 0.5;
  return 0.6 + beds * 0.6 + grain * 0.25;
};

/** Material-specific grain shade. */
const grainOf = (matter: number, shade: number, at: d.v2i) => {
  'use gpu';
  if (matter === Matter.wood) {
    const seam = d.f32((at.x * 3 + at.y * 5) % 7 === 0) * 0.25;
    return 0.75 + shade * 0.35 - seam;
  }
  if (matter === Matter.sand) return 0.55 + shade * 0.8;
  return 0.8 + shade * 0.4;
};

/** Foam of colour `foam` lit by `air`. */
const foamIn = (foam: d.v3f, air: d.v3f) => {
  'use gpu';
  return foam * (AMBIENT * WATER.foamAmbient + air * WATER.foamLight);
};

/** Lava colour sampled at cell coordinates. */
const lavaAt = (p: d.v2f, seconds: number) => {
  'use gpu';
  const q = p + LAVA.drift * seconds;
  const swirl = perlin2d.sample(q * LAVA.swirl) * 0.5 + 0.5;
  const crust = std.smoothstep(-0.05, 0.35, perlin2d.sample(q * LAVA.patches + 13));
  // Use a slower noise to vary vein width and brightness along the zero contour
  const seam = std.abs(perlin2d.sample(q * LAVA.veins + 29));
  const width = 0.03 + (perlin2d.sample(q * LAVA.spread + 41) * 0.5 + 0.5) * 0.08;
  const strength = std.smoothstep(-0.1, 0.4, perlin2d.sample(q * LAVA.spread + 53));
  const vein = std.smoothstep(width, width * 0.3, seam) * strength;
  const molten = std.mix(LAVA.body * (0.55 + swirl * 0.9), LAVA.crust, crust * 0.7);
  return std.mix(molten, LAVA.vein, vein * 0.5);
};

/** Falling streak intensity from 0 to 1 at `speed` cells per second. */
const streaksAt = tgpu.fn(
  [d.vec2f, d.f32, d.f32],
  d.f32,
)((p, seconds, speed) => {
  'use gpu';
  const flow = perlin2d.sample(d.vec2f(p.x * 0.45, (p.y - seconds * speed) * 0.06));
  return std.smoothstep(-0.2, 0.45, flow);
});

/** Sand-grain pattern from -1 to 1, with features about one cell wide. */
const grainsAt = (p: d.v2f, seconds: number) => {
  'use gpu';
  return perlin2d.sample(d.vec2f(p.x * 1.3, (p.y - seconds * SAND_FALL.speed) * 0.75));
};

/**
 * Draws one cell per pixel with lit solid faces, a far wall, scattered light, tinted water and
 * emissive materials.
 */
export function createDrawing(
  root: TgpuRoot,
  options: {
    readonly cells: Cells;
    readonly light: Light;
    readonly densities: Densities;
    readonly seconds: TgpuUniform<d.F32>;
  },
) {
  const { cells, light, densities, seconds } = options;
  const view = root.createUniform(d.u32);

  const isAir = (at: d.v2i) => {
    'use gpu';
    return cells.matterAt(at) === Matter.empty;
  };

  /** Surface light fading into a solid cell. */
  const faceLight = (at: d.v2i) => {
    'use gpu';
    const relief = cells.reliefAt(at);
    if (relief.z > RELIEF_CELLS || std.dot(relief.xy, relief.xy) === 0) return d.vec3f();
    let face = light.faceAt(at, relief.xy, relief.z) * std.exp((1 - relief.z) * RIM_FALLOFF);
    if (relief.z === 1) face = face * EDGE;
    return face;
  };

  /** The stream drawn as a ribbon over a cell: lava, water, sand, or empty for none. */
  const ribbonOf = (at: d.v2i, kind: number, streams: d.v3f) => {
    'use gpu';
    const falling = kind !== Kind.fixed && cells.kindAt(at + d.vec2i(0, 1)) === Kind.gas;
    if (kind !== Kind.gas && !falling) return d.u32(Matter.empty);
    if (streams.y > RIBBON_LEAST.y) return d.u32(Matter.lava);
    if (streams.x > RIBBON_LEAST.x) return d.u32(Matter.water);
    if (streams.z > RIBBON_LEAST.z) return d.u32(Matter.sand);
    return d.u32(Matter.empty);
  };

  /** A falling stream of lava, water or sand, drawn over the cell at `at`. */
  const fallColor = (ribbon: number, cell: number, at: d.v2i, back: d.v3f, streams: d.v3f) => {
    'use gpu';
    const center = d.vec2f(at) + 0.5;
    const air = light.lightAt(at);
    if (ribbon === Matter.lava) {
      const full = std.smoothstep(RIBBON_LEAST.y, RIBBON.full, streams.y);
      const streak = streaksAt(center, seconds.$, RIBBON.lavaSpeed);
      return std.mix(LAVA_FALL.edge, LAVA_FALL.core, full * (0.4 + streak * 0.6));
    }
    if (ribbon === Matter.water) {
      const full = std.smoothstep(RIBBON_LEAST.x, RIBBON.full, streams.x);
      const streak = streaksAt(center, seconds.$, RIBBON.waterSpeed);
      return foamIn(std.mix(FALL, WATER.foam, streak * (0.4 + full * 0.6)), air);
    }

    const full = std.smoothstep(RIBBON_LEAST.z, SAND_FALL.full, streams.z);
    const grains = grainsAt(center, seconds.$);
    const washed = std.mix(air, d.vec3f(tone.luminance(air)), GRAINS.wash);
    const sand = SAND * (GRAINS.ambient + washed * SAND_FALL.light);
    const covered = std.mix(SAND_FALL.edge, SAND_FALL.middle, full);
    if (matterOf(cell) !== Matter.sand && grains <= covered) {
      return std.mix(back, sand * 0.5, full * SAND_FALL.dust);
    }
    const grain = grainOf(d.u32(Matter.sand), shadeOf(cell), at);
    return sand * (1 + std.abs(grains) * SAND_FALL.glint) * grain;
  };

  /** A crystal, brighter towards its tip and along its outline. */
  const crystalColor = (cell: number, at: d.v2i, glow: d.v3f) => {
    'use gpu';
    const along = d.f32(lifeOf(cell)) / 255;
    let bright = shadeOf(cell) * std.mix(CRYSTAL_SHADE.base, CRYSTAL_SHADE.tip, along);
    if (
      cells.matterAt(at + d.vec2i(1, 0)) !== Matter.crystal ||
      cells.matterAt(at + d.vec2i(-1, 0)) !== Matter.crystal ||
      cells.matterAt(at + d.vec2i(0, 1)) !== Matter.crystal ||
      cells.matterAt(at + d.vec2i(0, -1)) !== Matter.crystal
    ) {
      bright = std.max(bright, CRYSTAL_SHADE.outline * (0.5 + along * 0.6));
    }
    return glow * bright * light.shineAt(at);
  };

  /** Water seen through to the far wall, darker in depth and foaming where it is open. */
  const waterColor = (at: d.v2i, back: d.v3f, streams: d.v3f) => {
    'use gpu';
    const air = light.lightAt(at);
    if (cells.kindAt(at + d.vec2i(0, 1)) === Kind.gas) {
      return std.mix(back, foamIn(WATER.foam, air), DROP);
    }

    let depth = d.f32(0);
    for (let k = 1; k <= WATER.depth; k++) {
      if (cells.matterAt(at - d.vec2i(0, k)) !== Matter.water) break;
      depth += 1;
    }
    const through = back * WATER.through * std.exp(-depth * 0.2);
    const scattered = WATER_TINT * (AMBIENT * 2 + air * 1.4);
    let color = std.mix(through, scattered, WATER.scatter + depth * WATER.deep);
    const center = d.vec2f(at) + 0.5;
    const shimmer = perlin2d.sample(center * d.vec2f(0.2, 0.35) + d.vec2f(0, seconds.$ * 1.3));
    color *= 0.92 + shimmer * 0.16;
    if (depth === 0 && isAir(at - d.vec2i(0, 1))) color += air * WATER.glint + WATER_TINT * 0.02;

    const open = d.f32(isAir(at + d.vec2i(-1, 0))) + d.f32(isAir(at + d.vec2i(1, 0)));
    const froth = std.saturate(open * 0.3 + std.smoothstep(0, RIBBON.full, streams.x));
    return std.mix(color, foamIn(WATER.foam, air), froth * 0.7);
  };

  /** Rock, wood, grains and other solids lit at their faces. Loose grains also catch light in air. */
  const solidColor = (cell: number, at: d.v2i, glow: d.v3f) => {
    'use gpu';
    const matter = matterOf(cell);
    const traits = cells.traitsOf(cell);
    const shade = shadeOf(cell);
    let face = faceLight(at);
    let ambient = d.vec3f(ROCK_AMBIENT);
    if (traits.kind === Kind.powder) {
      face = std.max(face, light.lightAt(at) * FALLING_GRAINS);
      face = std.mix(face, d.vec3f(tone.luminance(face)), GRAINS.wash);
      ambient = d.vec3f(GRAINS.ambient);
    }
    let texture = grainOf(matter, shade, at);
    if (traits.kind === Kind.fixed && matter !== Matter.wood && matter !== Matter.burning) {
      texture = stoneAt(d.vec2f(at) + 0.5, shade);
    }
    let color = traits.albedo * texture * (ambient + face);
    if (matter === Matter.burning) color += glow;
    if (matter === Matter.obsidian && shade > OBSIDIAN_GLINT.shade) {
      color += face * OBSIDIAN_GLINT.gain * traits.albedo * 4;
    }
    return color;
  };

  /** The cave at a cell under its vapour: a falling stream, or the cell's own matter. */
  const caveColor = (cell: number, at: d.v2i) => {
    'use gpu';
    const center = d.vec2f(at) + 0.5;
    const air = light.lightAt(at);
    const traits = cells.traitsOf(cell);
    const back = BACK * wallAt(center) * (AMBIENT + air) + air * HAZE;
    const glow = glowOf(traits.glow, cell, center, seconds.$);
    const streams = densities.streamsAt(at);
    const ribbon = ribbonOf(at, traits.kind, streams);
    if (ribbon !== Matter.empty) return fallColor(ribbon, cell, at, back, streams);

    switch (matterOf(cell)) {
      case Matter.lava:
        return lavaAt(center, seconds.$) * (1 + streams.y);
      case Matter.crystal:
        return crystalColor(cell, at, glow);
      case Matter.fire:
        return back + glow;
      case Matter.water:
        return waterColor(at, back, streams);
    }
    if (traits.kind === Kind.gas) return back;
    return solidColor(cell, at, glow);
  };

  /** Steam and smoke over `color`, thinner over solids and grains. */
  const veiled = (color: d.v3f, pixel: d.v2f, kind: number) => {
    'use gpu';
    const mist = densities.vapourAt(pixel);
    const lit = light.radianceAt(pixel);
    const shine = lit * VAPOUR.light;
    const glowing = shine / (1 + std.max(shine.x, std.max(shine.y, shine.z)) / VAPOUR.peak);
    let veil = d.f32(1);
    if (kind === Kind.fixed || kind === Kind.powder) veil = VAPOUR.solids;
    const steam = std.smoothstep(0, VAPOUR.full, mist.x) * VAPOUR.steam * veil;
    const smoke = std.smoothstep(0, VAPOUR.full, mist.y) * VAPOUR.smoke * veil;
    const steamy = std.mix(color, STEAM * (AMBIENT * VAPOUR.ambient + glowing), steam);
    return std.mix(steamy, SMOKE * (AMBIENT + lit), smoke);
  };

  const pixelOf = (corner: d.Infer<typeof passes.fullScreen.Vertex>) => {
    'use gpu';
    const pixel = corner.uv * GRID;
    const at = d.vec2i(std.floor(pixel));
    if (view.$ === VIEWS.radiance) return Seen({ color: d.vec4f(light.radianceAt(pixel), 1) });
    if (view.$ === VIEWS.medium) {
      const medium = light.mediumAt(at);
      const color =
        medium.rgb * DIAGNOSTIC_EMISSION_GAIN + EXTINCTION_TINT * std.min(medium.a, OPAQUE);
      return Seen({ color: d.vec4f(color, 1) });
    }

    const cell = cells.cellAt(at);
    const color = veiled(caveColor(cell, at), pixel, cells.traitsOf(cell).kind);
    return Seen({ color: d.vec4f(color, 1) });
  };

  const drawing: passes.Drawing<typeof passes.fullScreen.Vertex, d.Infer<typeof Seen>> = {
    ...passes.fullScreen,
    label: 'cave',
    pixelOf,
    color: passes.shaded,
    draw: (pass) => pass.draw(3),
  };

  return {
    drawing,
    view: (name: View) => view.write(VIEWS[name]),
  };
}
