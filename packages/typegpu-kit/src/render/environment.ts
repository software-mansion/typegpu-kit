/**
 * Lights surfaces with the light around them: an HDR panorama, a cube map or a sky you write.
 * `bake` turns a source into a cube of reflections at up to six roughness levels and nine spherical
 * harmonics of diffuse light, in the caller's encoder.
 *
 * @module
 */

import * as cameras from '../cameras.ts';
import { encoderOf, type Frame } from '../frames.ts';
import * as passes from '../passes.ts';
import * as pipelines from '../pipelines.ts';
import type * as profiler from '../profiler.ts';

import { unfilled, type Inject } from '../inject.ts';
import { randf } from '@typegpu/noise';
import * as workgroup from '../workgroup.ts';
import type { FloatTexture } from '../targets.ts';
import type { HdrImage } from '../images.ts';
import {
  d,
  isTexture,
  std,
  tgpu,
  type TgpuBindGroup,
  type TgpuCommandEncoder,
  type TgpuComputePipeline,
  type TgpuRoot,
  type TgpuTexture,
  type TgpuUniform,
} from 'typegpu';
import { Material, Point, reflectionOf, settings as lightSettings, surrounded } from './shading.ts';

/** This schema holds what changes how the environment looks without a new bake. */
export const Settings = d.struct({
  /** Multiplies every light the environment gives: diffuse, reflections and background. */
  intensity: d.f32,
  /** Turns the environment about the vertical axis, in radians, carrying +X towards +Z. */
  rotation: d.f32,
  /**
   * Sets the background's sharpness, from 0 for the sharpest reflection level to 1 for the
   * roughest.
   */
  blur: d.f32,
});

/** The defaults give intensity 1, no rotation and a sharp background. */
export const defaults: d.Infer<typeof Settings> = { intensity: 1, rotation: 0, blur: 0 };

/** This is a `'use gpu'` function from a unit direction to the light arriving from it. */
export type LightFrom = (direction: d.v3f) => d.v3f;

/**
 * This is what `add` takes: a texture or a `'use gpu'` function with its fills piped in. A texture
 * holds a 2D panorama in equirectangular projection, or a cube with six layers in WebGPU's order.
 * A panorama has +Y up, its centre facing +X and its left quarter facing −Z, as Blender writes it.
 * Its format must be filterable, such as `rgba16float`.
 */
export type Source = FloatTexture | LightFrom;

/** This is a source made ready to bake by `add`. */
export interface Prepared {
  readonly '~prepared': true;
}

export interface Options {
  /**
   * Sets the side of each cube face in texels, a power of two from 16 to 2048. Default 256.
   */
  readonly size?: number;
  /** Sets the samples per texel for rough reflections, from 1. Default 64. */
  readonly samples?: number;
  /** Sets initial values over `defaults`. Default `defaults`, with intensity 1, rotation 0 and blur 0. */
  readonly settings?: Partial<d.InferInput<typeof Settings>>;
}

export interface Environment extends Disposable {
  /**
   * Holds `defaults` with overrides from `settings`. Shaders read it at every lookup, so a new
   * value needs no bake.
   */
  readonly settings: TgpuUniform<typeof Settings>;
  /** Provides the cube of radiance from the last bake, with every mip level. */
  readonly radiance: TgpuTexture<{ size: [number, number, 6]; format: 'rgba16float' }>;
  /**
   * Provides the cube of reflections, mirror sharp at level 0 and rougher at each level down to
   * roughness 1 at the last.
   */
  readonly reflections: TgpuTexture<{ size: [number, number, 6]; format: 'rgba16float' }>;
  /**
   * Fills `lighting`'s `ambient` hook with diffuse light and reflections from this environment:
   * `lighting.create(root, { shading: { ambient: sky.ambient } })`.
   */
  readonly ambient: (material: Material, point: Point) => d.v3f;
  /**
   * Makes a source ready to bake. A 2D texture is read as a panorama in equirectangular projection
   * and a 6-layer texture as a cube, each through a bind group; a function gets a kernel of its
   * own. Throws for other layer counts.
   */
  add(source: Source): Prepared;
  /**
   * Records the bake of a source added to this environment into one compute pass of `target`, a
   * frame or an encoder. Pipelines that read the environment see it in later passes.
   */
  bake(target: Frame | TgpuCommandEncoder, source: Prepared, options?: profiler.Timed): void;
  /**
   * Draws the environment at `settings.blur` where nothing else drew, as `environment.background`
   * does. Add it after the opaque drawings.
   */
  background(options?: BackgroundOptions): passes.Compound;
  /**
   * Fills `environment.reflection`, `irradiance`, `occluded`, `backdrop` and `ambient` for
   * pipelines created from the configuration it is piped into.
   */
  inject(): Inject;
  /** Compiles the bake kernels and those of every added function. */
  initAsync(): Promise<void>;
  /** Destroys the cubes and the harmonics buffer. */
  destroy(): void;
}

const missing = unfilled('an environment', 'sky', 'an `environment.create`');
/** Holds nine spherical harmonics of light, the colour in `xyz`. */
const Harmonics = d.arrayOf(d.vec4f, 9);
const sky = tgpu.accessor(Settings, missing);
const harmonics = tgpu.accessor(Harmonics, missing);
const cube = tgpu.accessor(d.textureCube(d.f32), missing);
const smooth = tgpu.accessor(d.sampler(), missing);

const PI = Math.PI;

/**
 * Turns `direction` by `rotation` radians about the vertical, carrying +X towards +Z, as the
 * `rotation` setting turns the environment. Turn a direction in the panorama's frame, such as
 * `clip`'s sun, to find it in the world. Also runs on the CPU.
 */
export const turn = (direction: d.v3f, rotation: number) => {
  'use gpu';
  const c = std.cos(rotation);
  const s = std.sin(rotation);
  return d.vec3f(c * direction.x - s * direction.z, direction.y, s * direction.x + c * direction.z);
};

/** Turns a direction into the environment's frame, against `rotation`. */
const turned = (direction: d.v3f) => {
  'use gpu';
  return turn(direction, -sky.$.rotation);
};

/**
 * Returns the light arriving from `direction` blurred to `roughness`, from 0 for a mirror to 1,
 * times `intensity`. Pipelines read it through `sky.inject()`.
 */
export const reflection = (direction: d.v3f, roughness: number) => {
  'use gpu';
  // The last level has roughness 1
  const level = std.saturate(roughness) * d.f32(std.textureNumLevels(cube.$) - 1);
  return std.textureSampleLevel(cube.$, smooth.$, turned(direction), level).rgb * sky.$.intensity;
};

const NINE = Array.from({ length: 9 }, (_, k) => k);

/** Returns the nine spherical harmonic basis functions at the unit direction `n`. */
const basisOf = (n: d.v3f) => {
  'use gpu';
  return d.arrayOf(
    d.f32,
    9,
  )([
    0.282095,
    0.488603 * n.y,
    0.488603 * n.z,
    0.488603 * n.x,
    1.092548 * n.x * n.y,
    1.092548 * n.y * n.z,
    0.315392 * (3 * n.z * n.z - 1),
    1.092548 * n.x * n.z,
    0.546274 * (n.x * n.x - n.y * n.y),
  ]);
};

/**
 * Returns the light a white matte surface facing `normal` reflects, times `intensity`, from nine
 * spherical harmonics. A uniform environment of radiance L gives L.
 */
export const irradiance = (normal: d.v3f) => {
  'use gpu';
  const basis = basisOf(turned(normal));
  let light = d.vec3f();
  for (const k of tgpu.unroll(NINE)) light += harmonics.$[k].xyz * basis[k];
  return std.max(light, d.vec3f()) * sky.$.intensity;
};

/**
 * Computes diffuse light and reflections from the environment, with `visibility` from 0 to 1,
 * such as `ao.at(point.position)`, darkening both on top of the material's `occlusion`.
 * Reflections look along `shading.reflectionOf`, scaled by the lighting's `skyReflection`, and
 * both shares go through `shading.surrounded`, as `shading.hemisphere`'s do.
 */
export const occluded = (material: Material, point: Point, visibility: number) => {
  'use gpu';
  const roughness = material.roughness;
  const open = Material(material);
  open.occlusion = visibility * material.occlusion;
  const toward = reflectionOf(point, roughness);
  return surrounded(
    open,
    point,
    irradiance(point.normal) * open.occlusion,
    reflection(toward, roughness) * lightSettings().skyReflection,
  );
};

/** Computes diffuse light and reflections from the environment. This is `sky.ambient`'s hook. */
export const ambient = (material: Material, point: Point) => {
  'use gpu';
  return occluded(material, point, d.f32(1));
};

/** Returns the environment in `direction` at `settings.blur`, as `sky.background()` draws it. */
export const backdrop = (direction: d.v3f) => {
  'use gpu';
  return reflection(direction, sky.$.blur);
};

/**
 * Returns a radiance in the lighting's sky gradient, which `shading.hemisphere` blends from
 * `skyBottom` to `skyTop`. Baked, it lights a matte surface as `hemisphere` does while
 * `skyBottom` is at least a fifth of `skyTop`; below that, surfaces facing down get more. Pipe
 * `light.inject()` into it.
 */
export const gradient = (direction: d.v3f) => {
  'use gpu';
  const env = lightSettings();
  const middle = (env.skyTop + env.skyBottom) * 0.5;
  const half = (env.skyTop - env.skyBottom) * 0.5;
  // A linear sky convolves to two thirds of its slope
  return std.max(middle + half * (1.5 * direction.y), d.vec3f()) * env.skyIntensity;
};

// The atmosphere in kilometres, after Hillaire's 2020 Earth model
const EARTH = 6360;
const TOP = 6460;
const EYE = 0.1;
const RAYLEIGH = d.vec3f(5.802e-3, 13.558e-3, 33.1e-3);
const RAYLEIGH_HEIGHT = 8;
const MIE = 3.996e-3;
const MIE_EXTINCTION = 4.44e-3;
const MIE_HEIGHT = 1.2;
const MIE_G = 0.8;
const OZONE = d.vec3f(0.65e-3, 1.881e-3, 0.085e-3);
const VIEW_STEPS = 16;
const SUN_STEPS = 6;
const GROUND = d.vec3f(0.1, 0.09, 0.08);

/** Returns the extinction per kilometre at height `h` above the ground. */
const extinctionAt = (h: number) => {
  'use gpu';
  const ozone = std.max(1 - std.abs(h - 25) / 15, 0);
  return (
    RAYLEIGH * std.exp(-h / RAYLEIGH_HEIGHT) +
    d.vec3f(MIE_EXTINCTION * std.exp(-h / MIE_HEIGHT)) +
    OZONE * ozone
  );
};

/** Returns the distance from a point at radius `r` along `mu`, the cosine from up, to the top. */
const toTop = (r: number, mu: number) => {
  'use gpu';
  const below = (r - TOP) * (r + TOP);
  return -r * mu + std.sqrt(std.max(r * r * mu * mu - below, 0));
};

/** Returns the distance to the ground, or -1 where the ray misses it. */
const toGround = (r: number, mu: number) => {
  'use gpu';
  const disc = r * r * (mu * mu - 1) + EARTH * EARTH;
  if (mu >= 0 || disc < 0) return d.f32(-1);
  return -r * mu - std.sqrt(disc);
};

/** Returns the share of sunlight that reaches radius `r` with the sun at cosine `mu` from up. */
const sunThrough = (r: number, mu: number) => {
  'use gpu';
  if (toGround(r, mu) >= 0) return d.vec3f();
  const length = toTop(r, mu);
  const step = length / SUN_STEPS;
  let depth = d.vec3f();
  for (let i = 0; i < SUN_STEPS; i++) {
    const t = (d.f32(i) + 0.5) * step;
    const height = std.sqrt(r * r + t * t + 2 * r * mu * t) - EARTH;
    depth += extinctionAt(height) * step;
  }
  return std.exp(depth * -1);
};

/**
 * Returns the colour of sunlight at the ground with the sun in `direction`, for a sun of
 * intensity 1 above the atmosphere: white overhead, orange low and black below the horizon. Set
 * the lighting's `sunColor` to it with `daylight`. It runs on the CPU and in shaders.
 */
export const sunColor = (direction: d.v3f) => {
  'use gpu';
  const towards = std.normalize(direction);
  return sunThrough(EARTH + EYE, towards.y);
};

/**
 * Returns the light of a clear sky from Rayleigh and Mie scattering with ozone, lit by the
 * lighting's sun: `sunDirection`, and `sunIntensity` above the atmosphere. Below the horizon it
 * shows dark ground through haze. The lighting's sky gradient adds beneath, so it alone remains
 * at night. It draws no sun disc. Pipe `light.inject()` into it.
 */
export const daylight = (direction: d.v3f) => {
  'use gpu';
  const env = lightSettings();
  const sun = std.normalize(env.sunDirection);
  const view = std.normalize(direction);
  const r = EARTH + EYE;
  const ground = toGround(r, view.y);
  const length = std.select(toTop(r, view.y), ground, ground >= 0);

  const mu = std.dot(view, sun);
  const rayleighPhase = (3 / (16 * PI)) * (1 + mu * mu);
  const g2 = MIE_G * MIE_G;
  const miePhase =
    ((3 / (8 * PI)) * ((1 - g2) * (1 + mu * mu))) /
    ((2 + g2) * std.pow(std.max(1 + g2 - 2 * MIE_G * mu, 1e-4), 1.5));

  let light = d.vec3f();
  let depth = d.vec3f();
  for (let i = 0; i < VIEW_STEPS; i++) {
    // Square spacing puts more steps near the eye, where the air is dense
    const near = d.f32(i) / VIEW_STEPS;
    const far = d.f32(i + 1) / VIEW_STEPS;
    const t = length * (near + far) * (near + far) * 0.25;
    const step = length * (far * far - near * near);
    const position = d.vec3f(0, r, 0) + view * t;
    const radius = std.length(position);
    const height = radius - EARTH;
    const extinction = extinctionAt(height);
    const scattering =
      RAYLEIGH * (std.exp(-height / RAYLEIGH_HEIGHT) * rayleighPhase) +
      d.vec3f(MIE * std.exp(-height / MIE_HEIGHT) * miePhase);
    const reached = sunThrough(radius, std.dot(position, sun) / radius);
    light += std.exp((depth + extinction * (step * 0.5)) * -1) * scattering * reached * step;
    depth += extinction * step;
  }

  // The ground reflects the sun that reaches it, seen through the haze
  const lit = GROUND * sunThrough(d.f32(EARTH), sun.y) * std.max(sun.y, 0) * std.exp(depth * -1);
  const seen = std.select(d.vec3f(), lit, ground >= 0);
  // Sunlight of intensity 1 lights a white matte surface to 1, which is π times its irradiance
  return (light * PI + seen) * env.sunIntensity + gradient(direction);
};

/**
 * Returns the sun's disc in `direction`, 0.5° across with a soft edge, at the lighting's sun
 * colour and intensity times `brightness`. Add it to a background's function. Pipe
 * `light.inject()` into it.
 */
export const sunDisc = (direction: d.v3f, brightness: number) => {
  'use gpu';
  const env = lightSettings();
  const cosine = std.dot(std.normalize(direction), std.normalize(env.sunDirection));
  const inside = std.smoothstep(0.99996, 0.99999, cosine);
  return env.sunColor * (env.sunIntensity * d.f32(brightness) * inside);
};

/**
 * Returns a unit direction towards the sun at `hour`, from 0 to 24 in local solar time, at
 * `latitude` and with the sun's `declination`, both in radians: about 0.41 at midsummer in the
 * north, 0 at the equinoxes and −0.41 at midwinter. +X points east, +Y up and +Z south, so the sun
 * rises in +X and, north of the tropics, stands in +Z at noon. Turn the compass with
 * `environment.turn`. Below the horizon the direction points down. It runs on the CPU and in
 * shaders.
 */
export const sunAt = (hour: number, latitude: number, declination: number) => {
  'use gpu';
  const angle = (hour - 12) * (PI / 12);
  const sinLatitude = std.sin(latitude);
  const cosLatitude = std.cos(latitude);
  const sinDeclination = std.sin(declination);
  const cosDeclination = std.cos(declination);
  return d.vec3f(
    -cosDeclination * std.sin(angle),
    sinLatitude * sinDeclination + cosLatitude * cosDeclination * std.cos(angle),
    sinLatitude * cosDeclination * std.cos(angle) - cosLatitude * sinDeclination,
  );
};

/** This schema holds what `stars` finds in a direction: a star's disc there and its randoms. */
export const Star = d.struct({
  /** Rises from 0 to 1 at the centre of a star's disc, 0 where no star is. */
  disc: d.f32,
  /** Holds three random values from 0 to 1, fixed per star, for its brightness, tint and phase. */
  seed: d.vec3f,
});

export interface StarsOptions {
  /** Sets the cells along each edge of a cube around the eye; each holds a star or none. Default 96. */
  readonly cells?: number;
  /** Sets the share of cells that hold a star. Default 0.2. */
  readonly chance?: number;
  /** Sets the radius of a star's disc in pixels, at any resolution. Default 0.8. */
  readonly size?: number;
}

/**
 * Returns a `'use gpu'` function that finds the star in a direction: at most one per cell of a cube
 * around the eye, with a Gaussian disc of radius `size` pixels wherever the direction is drawn.
 * Shape the light from its `seed`, as in `tint * pow(star.seed.x, 6) * star.disc`. Call it only in
 * a fragment shader, since the disc's size reads derivatives.
 *
 * @returns a `'use gpu'` function from a unit direction to the `Star` there
 */
export function stars(options: StarsOptions = {}): (direction: d.v3f) => d.Infer<typeof Star> {
  const { cells = 96, chance = 0.2, size = 0.8 } = options;
  const star = (direction: d.v3f) => {
    'use gpu';
    // The grid jumps where neighbouring pixels fall on different faces, so its footprint comes
    // from the direction's derivatives, projected onto this pixel's face
    const dx = std.dpdx(direction);
    const dy = std.dpdy(direction);
    const away = std.abs(direction);
    let side = d.vec3f(direction.x, direction.z, direction.y);
    let sideX = d.vec3f(dx.x, dx.z, dx.y);
    let sideY = d.vec3f(dy.x, dy.z, dy.y);
    let face = std.select(d.f32(4), d.f32(5), direction.y < 0);
    if (away.x >= away.y && away.x >= away.z) {
      side = d.vec3f(direction.y, direction.z, direction.x);
      sideX = d.vec3f(dx.y, dx.z, dx.x);
      sideY = d.vec3f(dy.y, dy.z, dy.x);
      face = std.select(d.f32(0), d.f32(1), direction.x < 0);
    } else if (away.z >= away.y) {
      side = d.vec3f(direction.x, direction.y, direction.z);
      sideX = d.vec3f(dx.x, dx.y, dx.z);
      sideY = d.vec3f(dy.x, dy.y, dy.z);
      face = std.select(d.f32(2), d.f32(3), direction.z < 0);
    }
    // `uv` is the side's first two components over the absolute third
    const sign = std.sign(side.z);
    const major = std.abs(side.z);
    const uv = side.xy / major;
    const uvX = (sideX.xy - uv * (sideX.z * sign)) / major;
    const uvY = (sideY.xy - uv * (sideY.z * sign)) / major;

    const grid = (uv * 0.5 + 0.5) * cells;
    const cell = std.floor(grid);
    randf.seed3(d.vec3f(cell, face));
    const present = randf.sample() < chance;
    const at = d.vec2f(randf.sample(), randf.sample()) * 0.8 + 0.1;
    const seed = d.vec3f(randf.sample(), randf.sample(), randf.sample());

    const footprint = (std.abs(uvX) + std.abs(uvY)) * (0.5 * cells);
    const pixel = std.max(footprint.x, footprint.y);
    const off = std.length(grid - cell - at) / (pixel * size);
    const disc = std.select(d.f32(0), std.exp(-off * off), present);
    return Star({ disc, seed });
  };
  return star;
}

const Behind = d.struct({ color: d.vec4f });
type Behind = d.Infer<typeof Behind>;
type Screen = d.Infer<typeof passes.ScreenVertex>;

/** Configures a background. */
export interface BackgroundOptions {
  /** Labels its pipelines in GPU captures. Default `'environment background'`. */
  readonly label?: string;
}

/** A full-screen triangle on the far plane, at `far`, the depth the target is cleared to. */
const farPlane = (far: number) => {
  const farAt = (vertexIndex: number, _instanceIndex: number) => {
    'use gpu';
    const x = std.select(d.f32(-1), d.f32(3), vertexIndex === 1);
    const y = std.select(d.f32(-1), d.f32(3), vertexIndex === 2);
    return passes.ScreenVertex({
      position: d.vec4f(x, y, far, 1),
      uv: d.vec2f(x * 0.5 + 0.5, 0.5 - y * 0.5),
    });
  };
  return farAt;
};

/**
 * Makes a drawing that fills every pixel left uncovered with the light from its direction through
 * the scene's camera. It lies on the far plane and tests depth without writing it, so add it after
 * the opaque drawings, where early depth testing skips covered pixels, and before translucent ones.
 * It writes only colour: every other output, such as TAA's motion, keeps its clear value, as where
 * nothing drew. Depth must clear to 1, as it does by default, or to 0 in a description whose
 * `depthCompare` is `'greater'` or `'greater-equal'`. Pipe the function's fills in, such as
 * `tgpu.fn(f).pipe(sky.inject())`.
 *
 * @param lightFrom a `'use gpu'` function from a direction to its light, such as `environment.backdrop`
 * @returns a drawing for `scene.add`
 */
export function background(lightFrom: LightFrom, options: BackgroundOptions = {}): passes.Compound {
  const pixelOf = (screen: Screen): Behind => {
    'use gpu';
    const ndc = d.vec2f(screen.uv.x * 2 - 1, 1 - screen.uv.y * 2);
    const ray = cameras.rayOf(cameras.current.$, ndc);
    return Behind({ color: d.vec4f(lightFrom(ray.direction), 1) });
  };
  const drawingFor = (reversed: boolean): passes.Drawing<typeof passes.ScreenVertex, Behind> => ({
    label: options.label ?? 'environment background',
    Vertex: passes.ScreenVertex,
    at: farPlane(reversed ? 0 : 1),
    pixelOf,
    color: passes.shaded,
    depthStencil: {
      depthWriteEnabled: false,
      depthCompare: reversed ? 'greater-equal' : 'less-equal',
    },
    draw: (pass) => pass.draw(3),
  });
  const [standard, reversed] = [false, true].map(drawingFor);
  return {
    build(into) {
      const drawing = into.depthCompare.startsWith('greater') ? reversed : standard;
      const outputs = Object.keys(into.writes).filter((key) => {
        const write = into.writes[key];
        return (
          typeof write === 'object' && write !== passes.accumulation && write !== passes.revealage
        );
      });
      const bare = outputs.length > 0 ? into.without(...(outputs as never[])) : into;
      const drawn = bare.add(drawing);
      return {
        draw: (pass) => drawn.draw(pass),
        initAsync: () => drawn.initAsync(),
        destroy: () => drawn.destroy(),
      };
    },
  };
}

/** What `clip` returns: the clipped panorama and the strongest source it took, as a sun. */
export interface Clipped {
  /** Holds a copy of the panorama with no channel above the ceiling. */
  readonly image: HdrImage;
  /** Points towards the source in the panorama's frame, before `rotation`. */
  readonly direction: [number, number, number];
  /** Holds the source's linear colour, with its largest channel 1. */
  readonly color: [number, number, number];
  /** Holds the source's light as the lighting's `sunIntensity`, 0 when nothing was clipped. */
  readonly intensity: number;
}

const HALF_FLOAT_MAX = 65504;
const HALF_INFINITY = 0x7c00;
/** Clipped texels within this angle of a brighter one belong to its source. */
const SOURCE_COS = Math.cos((20 * Math.PI) / 180);

const fromHalf = (bits: number) => {
  const exponent = bits >> 10;
  const fraction = bits & 0x3ff;
  return exponent === 0 ? fraction * 2 ** -24 : (1 + fraction / 1024) * 2 ** (exponent - 15);
};

/** Returns the nearest half float's bits, for a value in the normal half-float range. */
const toHalf = (value: number) => {
  const exponent = Math.floor(Math.log2(value));
  // A fraction that rounds up to 1024 carries into the exponent
  return ((exponent + 15) << 10) + Math.round((value / 2 ** exponent - 1) * 1024);
};

/**
 * Lowers every channel of a panorama above `ceiling` to it, and returns the light it took from the
 * strongest source, such as a photographed sun, in the lighting's units. Give that back as the
 * lighting's sun: it then shadows, and the baked diffuse light doesn't ring. Clipped texels within
 * 20° of a brighter one count as one source. It runs on the CPU, in time proportional to the
 * pixels. Throws for a `ceiling` outside 2^-14 to 65 504.
 *
 * @param image a panorama in equirectangular projection, as `environment.add` reads it
 * @param ceiling the brightest a channel stays, in the panorama's linear units
 * @returns the clipped copy, and the strongest source's direction, colour and intensity
 */
export function clip(image: HdrImage, ceiling = 30): Clipped {
  if (!(ceiling >= 2 ** -14 && ceiling <= HALF_FLOAT_MAX)) {
    throw new RangeError(
      `environment.clip requires \`ceiling\` from 2^-14 to 65504. Received ${ceiling}.`,
    );
  }
  const { width, height } = image;
  const pixels = image.pixels.slice();
  const top = toHalf(ceiling);
  // Positive half floats order as their bits do
  const over = (bits: number) => bits > top && bits < 0x8000;
  // Each clipped texel's direction, and the light above the ceiling times its solid angle
  const taken: { direction: [number, number, number]; light: number[]; weight: number }[] = [];
  for (let y = 0; y < height; y++) {
    const latitude = (0.5 - (y + 0.5) / height) * PI;
    const angle = ((2 * PI) / width) * (PI / height) * Math.cos(latitude);
    for (let x = 0; x < width; x++) {
      const at = (y * width + x) * 4;
      const rgb = [pixels[at], pixels[at + 1], pixels[at + 2]];
      if (!over(rgb[0]) && !over(rgb[1]) && !over(rgb[2])) continue;
      const light = rgb.map((bits) =>
        over(bits) && bits < HALF_INFINITY ? (fromHalf(bits) - ceiling) * angle : 0,
      );
      for (let c = 0; c < 3; c++) if (over(rgb[c])) pixels[at + c] = top;
      const weight = light[0] + light[1] + light[2];
      if (weight === 0) continue;
      const around = ((x + 0.5) / width - 0.5) * 2 * PI;
      const ring = Math.cos(latitude);
      taken.push({
        direction: [ring * Math.cos(around), Math.sin(latitude), ring * Math.sin(around)],
        light,
        weight,
      });
    }
  }

  // Gather each texel into the source of the brightest texel near it
  taken.sort((a, b) => b.weight - a.weight);
  const sources: { seed: readonly number[]; members: typeof taken }[] = [];
  for (const texel of taken) {
    const [x, y, z] = texel.direction;
    const near = sources.find(({ seed }) => seed[0] * x + seed[1] * y + seed[2] * z > SOURCE_COS);
    if (near) near.members.push(texel);
    else sources.push({ seed: texel.direction, members: [texel] });
  }

  let best: Omit<Clipped, 'image'> = { direction: [0, 1, 0], color: [1, 1, 1], intensity: 0 };
  let most = 0;
  for (const { members } of sources) {
    const sum = [0, 0, 0];
    for (const { direction, weight } of members) {
      for (let i = 0; i < 3; i++) sum[i] += direction[i] * weight;
    }
    const length = Math.hypot(sum[0], sum[1], sum[2]);
    const direction: [number, number, number] = [sum[0] / length, sum[1] / length, sum[2] / length];
    // Irradiance facing the source; the lighting's units divide it by π
    const given = [0, 0, 0];
    for (const texel of members) {
      const [x, y, z] = texel.direction;
      const facing = Math.max(x * direction[0] + y * direction[1] + z * direction[2], 0);
      for (let c = 0; c < 3; c++) given[c] += (texel.light[c] * facing) / PI;
    }
    const total = given[0] + given[1] + given[2];
    if (total <= most) continue;
    most = total;
    const strongest = Math.max(given[0], given[1], given[2]);
    best = {
      direction,
      color: [given[0] / strongest, given[1] / strongest, given[2] / strongest],
      intensity: strongest,
    };
  }
  return { image: { width, height, pixels }, ...best };
}

/** Returns the unit direction through texel `uv` of cube face `face`, in WebGPU's layout. */
const faceDirection = (face: number, uv: d.v2f) => {
  'use gpu';
  const a = uv.x * 2 - 1;
  const b = uv.y * 2 - 1;
  let direction = d.vec3f(-a, -b, -1);
  if (face === 0) direction = d.vec3f(1, -b, -a);
  if (face === 1) direction = d.vec3f(-1, -b, a);
  if (face === 2) direction = d.vec3f(a, 1, b);
  if (face === 3) direction = d.vec3f(a, -1, -b);
  if (face === 4) direction = d.vec3f(a, -b, 1);
  return std.normalize(direction);
};

const SH_THREADS = 64;
/** Convolves each harmonic's band with the cosine lobe over π. */
const BANDS = [1, 2 / 3, 2 / 3, 2 / 3, 0.25, 0.25, 0.25, 0.25, 0.25];
const SH_SIDE = 32;

const halving = tgpu.bindGroupLayout({
  from: { texture: d.texture2dArray(d.f32) },
  into: { storageTexture: d.textureStorage2dArray('rgba16float', 'write-only') },
});
const prefiltering = tgpu.bindGroupLayout({
  into: { storageTexture: d.textureStorage2dArray('rgba16float', 'write-only') },
});

/** Returns a texel of a face `side` texels across times its solid angle, which alpha holds. */
const weighed = (texel: d.v2u, face: number, side: number) => {
  'use gpu';
  const ab = ((d.vec2f(texel) + 0.5) / side) * 2 - 1;
  const angle = 1 / std.pow(1 + std.dot(ab, ab), 1.5);
  return d.vec4f(std.textureLoad(halving.$.from, texel, face, 0).rgb * angle, angle);
};

// Texels near a face's centre span more of the sphere than those at its corners
const halve = /* @__PURE__ */ tgpu.computeFn({
  workgroupSize: [8, 8, 1],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  const side = std.textureDimensions(halving.$.into);
  if (gid.x >= side.x || gid.y >= side.y) return;
  const at = gid.xy * 2;
  const from = d.f32(side.x * 2);
  const sum =
    weighed(at, gid.z, from) +
    weighed(at + d.vec2u(1, 0), gid.z, from) +
    weighed(at + d.vec2u(0, 1), gid.z, from) +
    weighed(at + d.vec2u(1, 1), gid.z, from);
  std.textureStore(halving.$.into, gid.xy, gid.z, d.vec4f(sum.xyz / sum.w, 1));
});

const panoramic = tgpu.bindGroupLayout({ source: { texture: d.texture2d(d.f32) } });
const cubic = tgpu.bindGroupLayout({ source: { texture: d.textureCube(d.f32) } });

const level = (l: number) => ({ baseMipLevel: l, mipLevelCount: 1 });
/** Workgroups of 8 along a side. */
const groups = (side: number) => Math.ceil(side / 8);
const UNTIMED = {};

/**
 * Creates an environment: a cube of radiance, a cube of reflections and nine spherical harmonics
 * of diffuse light, filled by `bake` from any source `add` prepared. Lit pipelines read it through
 * `sky.ambient` as the lighting's `ambient` hook; `sky.background()` draws it behind the scene.
 * Throws for a `size` that is not a power of two from 16 to 2048, or `samples` below 1.
 *
 * @injects definition the cube, the harmonics and the settings that `environment.reflection`,
 * `irradiance`, `occluded`, `backdrop` and `ambient` read
 * @binds 2 uniform: `sky.ambient` and the environment's functions, the settings and the harmonics,
 * beside the reflection cube and a filtering sampler
 * @owns 2 textures, 2 buffers, a sampler, 5 pipelines and one per added function, bind groups: the
 * radiance and reflection cubes, the settings and the harmonics; the kernels that read a panorama,
 * read a cube, halve the radiance, blur the reflections and sum the harmonics; a group per mip
 * level and per added texture. Both cubes hold the mirror level, since a dispatch can't sample a
 * texture it writes
 * @records `bake(frame, source)` when the source changes, before the passes that shade: one
 * compute pass with a dispatch per mip level of both cubes and one for the harmonics
 * @param options.size the side of each cube face, which sets memory and bake time
 * @returns an environment you bake and destroy
 */
export function create(root: TgpuRoot, options: Options = {}): Environment {
  const { size = 256, samples = 64 } = options;
  const levels = Math.log2(size);
  if (!Number.isInteger(levels) || size < 16 || size > 2048) {
    throw new RangeError(
      `environment.create requires \`size\` to be a power of two from 16 to 2048. Received ${size}.`,
    );
  }
  if (!Number.isInteger(samples) || samples < 1) {
    throw new RangeError(
      `environment.create requires \`samples\` to be an integer of at least 1. Received ${samples}.`,
    );
  }
  const radianceLevels = levels + 1;
  // The roughest level keeps 8 texels a side, or its reflections turn faceted
  const reflectionLevels = Math.min(6, levels - 2);
  const shLevel = Math.max(levels - Math.log2(SH_SIDE), 0);
  const shSide = size >> shLevel;

  const settings = root.createUniform(Settings, { ...defaults, ...options.settings });
  const cubeOf = (mipLevelCount: number) =>
    root
      .createTexture({ size: [size, size, 6], format: 'rgba16float', mipLevelCount })
      .$usage('sampled', 'storage');
  const radianceCube = cubeOf(radianceLevels);
  const reflectionCube = cubeOf(reflectionLevels);
  const shBuffer = root.createBuffer(Harmonics).$usage('uniform', 'storage');
  const sampler = root.createSampler({
    addressModeU: 'repeat',
    addressModeV: 'clamp-to-edge',
    addressModeW: 'clamp-to-edge',
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
  });

  const storageAt = (texture: typeof radianceCube, l: number) =>
    texture.createView(d.textureStorage2dArray('rgba16float', 'write-only'), level(l));
  const radianceTop = storageAt(radianceCube, 0);
  const mirror = storageAt(reflectionCube, 0);
  const radianceView = radianceCube.createView(d.textureCube(d.f32));
  const reflectionView = reflectionCube.createView(d.textureCube(d.f32));
  const shView = radianceCube.createView(d.texture2dArray(d.f32), level(shLevel));
  const shOut = shBuffer.as('mutable');
  const shIn = shBuffer.as('uniform');

  const halvings = Array.from({ length: radianceLevels - 1 }, (_, i) => ({
    side: size >> (i + 1),
    group: root.createBindGroup(halving, {
      from: radianceCube.createView(d.texture2dArray(d.f32), level(i)),
      into: storageAt(radianceCube, i + 1),
    }),
  }));
  const blurs = Array.from({ length: reflectionLevels - 1 }, (_, i) => ({
    side: size >> (i + 1),
    group: root.createBindGroup(prefiltering, { into: storageAt(reflectionCube, i + 1) }),
  }));

  // The solid angle of one texel of the top radiance level, for filtered importance sampling
  const texelAngle = (4 * PI) / (6 * size * size);
  const lastLevel = reflectionLevels - 1;
  const prefilter = tgpu.computeFn({
    workgroupSize: [8, 8, 1],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const side = std.textureDimensions(prefiltering.$.into);
    if (gid.x >= side.x || gid.y >= side.y) return;
    const n = faceDirection(gid.z, (d.vec2f(gid.xy) + 0.5) / d.f32(side.x));
    const roughness = std.log2(size / d.f32(side.x)) / lastLevel;
    const alpha = roughness * roughness;
    const alpha2 = alpha * alpha;
    const pole = std.select(d.vec3f(0, 1, 0), d.vec3f(1, 0, 0), std.abs(n.y) > 0.999);
    const tangent = std.normalize(std.cross(pole, n));
    const bitangent = std.cross(n, tangent);

    let sum = d.vec3f();
    let weight = d.f32(0);
    for (let i = d.u32(0); i < samples; i++) {
      // Hammersley points: even steps around, the bit-reversed index up
      const around = (d.f32(i) / samples) * (2 * PI);
      const up = d.f32(std.reverseBits(i)) * 2.3283064365386963e-10;
      const cosTheta = std.sqrt((1 - up) / (1 + (alpha2 - 1) * up));
      const sinTheta = std.sqrt(1 - cosTheta * cosTheta);
      const h =
        tangent * (sinTheta * std.cos(around)) +
        bitangent * (sinTheta * std.sin(around)) +
        n * cosTheta;
      const l = h * (2 * cosTheta) - n;
      const nl = std.dot(n, l);
      if (nl > 0) {
        const spread = (cosTheta * alpha2 - cosTheta) * cosTheta + 1;
        const density = alpha2 / (PI * spread * spread);
        // Read a level whose texels cover the sample's share of the lobe
        const share = 4 / (samples * density);
        const lod = std.max(0.5 * std.log2(share / texelAngle) + 1, 0);
        sum += std.textureSampleLevel(radianceView.$, sampler.$, l, lod).rgb * nl;
        weight += nl;
      }
    }
    std.textureStore(prefiltering.$.into, gid.xy, gid.z, d.vec4f(sum / std.max(weight, 1e-4), 1));
  });

  const summed = workgroup.reduce(workgroup.sum(Harmonics), SH_THREADS);
  const texels = 6 * shSide * shSide;
  const project = tgpu.computeFn({
    workgroupSize: [SH_THREADS],
    in: { lid: d.builtin.localInvocationIndex },
  })(({ lid }) => {
    'use gpu';
    // The solid angle each texel covers adds up in the first harmonic's alpha
    const sums = Harmonics();
    for (let t = d.u32(lid); t < texels; t += SH_THREADS) {
      const face = std.intdiv(t, d.u32(shSide * shSide));
      const x = t % shSide;
      const y = std.intdiv(t, d.u32(shSide)) % shSide;
      const uv = (d.vec2f(d.f32(x), d.f32(y)) + 0.5) / shSide;
      const ab = uv * 2 - 1;
      // A texel's solid angle on a cube face of half-width 1
      const angle = 4 / (shSide * shSide * std.pow(1 + std.dot(ab, ab), 1.5));
      const n = faceDirection(face, uv);
      const c = std.textureLoad(shView.$, d.vec2u(x, y), face, 0).rgb * angle;
      const basis = basisOf(n);
      for (const k of tgpu.unroll(NINE)) sums[k] = sums[k] + d.vec4f(c * basis[k], 0);
      sums[0].w += angle;
    }
    const total = summed(sums, lid);
    if (lid === 0) {
      // Scale the summed solid angle to the sphere's, then convolve with the cosine lobe over π
      const whole = (4 * PI) / total[0].w;
      for (const k of tgpu.unroll(NINE)) {
        shOut.$[k] = d.vec4f(total[k].xyz * (whole * BANDS[k]), 0);
      }
    }
  });

  const halvingPipeline = root.createComputePipeline({ compute: halve });
  const prefilterPipeline = root.createComputePipeline({ compute: prefilter });
  const projectPipeline = root.createComputePipeline({ compute: project });

  const projected = (lightFrom: LightFrom) =>
    root.createComputePipeline({
      compute: tgpu.computeFn({
        workgroupSize: [8, 8, 1],
        in: { gid: d.builtin.globalInvocationId },
      })(({ gid }) => {
        'use gpu';
        if (gid.x >= size || gid.y >= size) return;
        const direction = faceDirection(gid.z, (d.vec2f(gid.xy) + 0.5) / size);
        const light = std.clamp(lightFrom(direction), d.vec3f(), d.vec3f(HALF_FLOAT_MAX));
        std.textureStore(radianceTop.$, gid.xy, gid.z, d.vec4f(light, 1));
        std.textureStore(mirror.$, gid.xy, gid.z, d.vec4f(light, 1));
      }),
    });

  // A panorama texel spans 2π / width and a face texel about π / (2 size)
  const fromPanorama = (direction: d.v3f) => {
    'use gpu';
    const width = d.f32(std.textureDimensions(panoramic.$.source).x);
    const lod = std.max(std.log2(width / (4 * size)), 0);
    const u = std.atan2(direction.z, direction.x) / (2 * PI) + 0.5;
    const v = 0.5 - std.asin(std.clamp(direction.y, -1, 1)) / PI;
    return std.textureSampleLevel(panoramic.$.source, sampler.$, d.vec2f(u, v), lod).rgb;
  };
  const fromCube = (direction: d.v3f) => {
    'use gpu';
    const width = d.f32(std.textureDimensions(cubic.$.source).x);
    const lod = std.max(std.log2(width / size), 0);
    return std.textureSampleLevel(cubic.$.source, sampler.$, direction, lod).rgb;
  };
  const panoramaPipeline = projected(fromPanorama);
  const cubePipeline = projected(fromCube);

  const prepared = new WeakMap<
    Prepared,
    { pipeline: TgpuComputePipeline; group?: TgpuBindGroup }
  >();
  const kernels: TgpuComputePipeline[] = [
    panoramaPipeline,
    cubePipeline,
    halvingPipeline,
    prefilterPipeline,
    projectPipeline,
  ];

  const inject: Inject = (cfg) =>
    cfg
      .with(sky, () => settings.$)
      .with(harmonics, () => shIn.$)
      .with(cube, () => reflectionView.$)
      .with(smooth, () => sampler.$);

  const destroy = () => {
    radianceCube.destroy();
    reflectionCube.destroy();
    shBuffer.destroy();
  };

  return {
    settings,
    radiance: radianceCube,
    reflections: reflectionCube,
    ambient: tgpu.fn(ambient).pipe(inject),

    add(source) {
      const handle: Prepared = { '~prepared': true };
      if (!isTexture(source)) {
        const pipeline = projected(source);
        kernels.push(pipeline);
        prepared.set(handle, { pipeline });
        return handle;
      }
      const [, , layers = 1] = source.props.size;
      if (layers === 6) {
        const faces = source.createView(d.textureCube(d.f32));
        prepared.set(handle, {
          pipeline: cubePipeline,
          group: root.createBindGroup(cubic, { source: faces }),
        });
        return handle;
      }
      if (layers !== 1) {
        throw new RangeError(
          `An environment reads a 2D panorama or a cube of 6 layers. Received ${layers} layers.`,
        );
      }
      const panorama = source.createView(d.texture2d(d.f32));
      prepared.set(handle, {
        pipeline: panoramaPipeline,
        group: root.createBindGroup(panoramic, { source: panorama }),
      });
      return handle;
    },

    bake(target, source, timed) {
      const found = prepared.get(source);
      if (!found) {
        throw new Error('Call `add` on this environment before passing its source to `bake`.');
      }
      const pass = encoderOf(target).beginComputePass(timed ?? UNTIMED);
      pass.setPipeline(found.pipeline);
      if (found.group) pass.setBindGroup(found.group);
      pass.dispatchWorkgroups(groups(size), groups(size), 6);
      pass.setPipeline(halvingPipeline);
      for (const { side, group } of halvings) {
        pass.setBindGroup(group);
        pass.dispatchWorkgroups(groups(side), groups(side), 6);
      }
      pass.setPipeline(prefilterPipeline);
      for (const { side, group } of blurs) {
        pass.setBindGroup(group);
        pass.dispatchWorkgroups(groups(side), groups(side), 6);
      }
      pass.setPipeline(projectPipeline);
      pass.dispatchWorkgroups(1);
      pass.end();
    },

    background: (given) => background(tgpu.fn(backdrop).pipe(inject), given),

    inject: () => inject,

    initAsync: () => pipelines.compile(kernels),

    destroy,
    [Symbol.dispose]: destroy,
  };
}
