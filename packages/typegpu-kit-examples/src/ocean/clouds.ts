import {
  cameras,
  history,
  pipelines,
  targets,
  tone,
  type CameraBase,
  type SizeSource,
} from '@typegpu/kit';
import { memo } from '@typegpu/kit/memo';
import { perlin3d, randf } from '@typegpu/noise';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';
import type { Atmosphere } from './atmosphere.ts';

const NOISE_SIZE = 96;
const COVERAGE = 1;
/** The volumes' texels, and the box they fill: metres across, base height and depth. */
const VOLUME = d.vec3u(192, 48, 192);
const EXTENT = 40_000;
const BASE = 800;
const HEIGHT = 1600;
/** Extinction per metre at density 1. */
const SIGMA = 0.005;
const SHADOW_SIZE = VOLUME.x * 3;
const VIEW_STEPS = 48;
const BAKE_STEPS = 40;
/** Below this transmittance the march is behind the surface the eye sees. */
const SURFACE = 0.3;
/** The share of each new march the view keeps, so the steps' offsets average out over frames. */
const BLEND = 0.1;
const CHECKER = 2;
const GOLDEN = 0.618034;
/** View heights over which the march fades into the atmosphere below the deck. */
const HORIZON = { from: 0.008, to: 0.02 };
const BILLOWS = { metres: 1600, octave: 2.7, erosion: 0.45 };
/** A low sun's horizon glow on the deck's base; `cone` is a cosine, about 14° round the sun. */
const UNDERLIGHT = {
  gain: 2.2,
  reach: 4000,
  cone: 0.97,
  fadeFrom: 0.1,
  fadeTo: 0.3,
  /** Sampled this far round the horizon from the sun, clear of its aureole. */
  cosine: Math.cos(0.6),
  sine: Math.sin(0.6),
};
/** The glow averages the bright band under the deck's edge at these angles from the sun. */
const BAND = [0.6, -0.6, 1.4, -1.4].map((angle) => ({
  cosine: Math.cos(angle),
  sine: Math.sin(angle),
}));
/** The horizon bank and the overcast beyond the box; `bank` and `veils` are view heights. */
const FAR = {
  bank: 0.0055,
  veils: 0.15,
  veil: 0.97,
  layer: BASE + HEIGHT / 2,
  levels: [0.1, 0.35, 0.6],
  hazed: 150_000,
  resolved: [20_000, 50_000] as const,
  pile: 0.02,
  forward: 0.85,
  lit: 0.4,
  under: 0.25,
  beneath: [0.02, 0.15] as const,
  metres: 25_000,
};
/** Zenith light on the deck per unit of escaping light, and the share of its colour kept. */
const SKY_FILL = { gain: 0.8, tint: 0.5 };
/** The deck's weather. Lengths are noise periods in metres that divide the box, so it tiles. */
const WEATHER = {
  large: EXTENT / 3,
  small: EXTENT / 6,
  strength: EXTENT,
  edge: 0.3,
  spread: 0.5,
  cell: 0.7,
  overcast: [0.6, 1] as const,
  floor: 0.6,
};
/** The clouds' shape. `metres` and `detail` are noise periods that divide the box. */
const SHAPE = {
  metres: EXTENT / 8,
  warp: 0.15,
  range: [0.5, 0.9] as const,
  detail: EXTENT / 13,
  erosion: 0.4,
};
/** The cells' vertical profile, in shares of the box's height. */
const PROFILE = { base: 0.03, rise: 0.06, rolls: 0.15, lowest: 0.3 };
/** Rays climbing to the deck leave the low haze early: they cross this share of its depth. */
const DECK_HAZE = 0.01;
/** How far a point looks for the billows that shade it, and how dense their shade is. */
const SELF_SHADOW = { metres: 180, depth: 3 };
/** Hillaire's scattering octaves, a colour channel each; the third holds diffused light. */
const ORDERS = [
  { channel: 0, attenuation: 1, weight: 1, anisotropy: 0.65 },
  { channel: 1, attenuation: 0.5, weight: 0.55, anisotropy: 0.3 },
];
const DIFFUSE = { channel: 2 };

const Shape = d.struct({
  coverage: d.f32,
  drift: d.vec2f,
  /** The frame count, and the share of the new march kept: 1 when the whole view is marched. */
  turn: d.f32,
  blend: d.f32,
});

const viewWriteLayout = tgpu.bindGroupLayout({
  target: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
});
const cloudsLayout = tgpu.bindGroupLayout({ seen: { texture: d.texture2d(d.f32) } });
const resolveLayout = tgpu.bindGroupLayout({
  fresh: { texture: d.texture2d(d.f32) },
  kept: { texture: d.texture2d(d.f32) },
  target: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
});

/** Where a direction meets the screen, as clip x, y and w. */
const vanishing = (direction: d.v3f) => {
  'use gpu';
  const clip = cameras.current.$.viewProjection * d.vec4f(direction, 0);
  return d.vec3f(clip.x, clip.y, clip.w);
};
const vanishingBefore = history.previously(vanishing);

const henyeyGreenstein = (cosTheta: number, g: number) => {
  'use gpu';
  const denom = 1 + g * g - 2 * g * cosTheta;
  return (1 - g * g) / (4 * Math.PI * denom * std.sqrt(denom));
};

const segmentTransmission = (density: number, distance: number) => {
  'use gpu';
  return std.exp(-std.max(density, 0) * SIGMA * std.max(distance, 0));
};

/** Diffuse transmission through a slab in the diffusion limit: one over its depth. */
const diffuseTransmission = (opticalDepth: number) => {
  'use gpu';
  return 1 / (1 + 0.75 * (1 - ORDERS[0].anisotropy) * std.max(opticalDepth, 0));
};

const worley = (point: d.v3f, cells: number) => {
  'use gpu';
  const p = point * cells;
  const base = std.floor(p);
  let nearest = d.f32(2);
  for (const z of std.range(-1, 2)) {
    for (const y of std.range(-1, 2)) {
      for (const x of std.range(-1, 2)) {
        const cell = base + d.vec3f(d.f32(x), d.f32(y), d.f32(z));
        randf.seed3(std.mod(std.mod(cell, d.f32(cells)) + cells, d.f32(cells)));
        const feature = cell + d.vec3f(randf.sample(), randf.sample(), randf.sample());
        nearest = std.min(nearest, std.distance(p, feature));
      }
    }
  }
  return std.saturate(1 - nearest);
};
const dome = (cone: number) => {
  'use gpu';
  return 1 - (1 - cone) * (1 - cone);
};

const volumeUv = (point: d.v3f) => {
  'use gpu';
  return d.vec3f(point.x / EXTENT + 0.5, (point.y - BASE) / HEIGHT, point.z / EXTENT + 0.5);
};

/** How far the deck has thinned out towards the box's sides, from 0 to 1. */
const rimOf = (uv: d.v2f) => {
  'use gpu';
  return std.smoothstep(0.38, 0.5, std.max(std.abs(uv.x - 0.5), std.abs(uv.y - 0.5)));
};

export type Clouds = ReturnType<typeof createClouds>;

/** A stratocumulus deck over a 40 km box round the eye, tiled so it drifts without a new bake. */
export function createClouds(
  root: TgpuRoot,
  atmosphere: Atmosphere,
  view: CameraBase,
  screen: SizeSource,
) {
  const shape = root.createUniform(Shape, {
    coverage: COVERAGE,
    drift: d.vec2f(),
    turn: 0,
    blend: 1,
  });
  const repeat = root.createSampler({
    minFilter: 'linear',
    magFilter: 'linear',
    addressModeU: 'repeat',
    addressModeV: 'repeat',
    addressModeW: 'repeat',
  });
  const volumeSampler = root.createSampler({
    minFilter: 'linear',
    magFilter: 'linear',
  });
  /** The deck tiles its box across but not up. */
  const deckSampler = root.createSampler({
    minFilter: 'linear',
    magFilter: 'linear',
    addressModeU: 'repeat',
    addressModeW: 'repeat',
  });

  const volume = () =>
    root
      .createTexture({
        size: [VOLUME.x, VOLUME.y, VOLUME.z],
        dimension: '3d',
        format: 'rgba16float',
      })
      .$usage('sampled', 'storage');
  const noise = root
    .createTexture({
      size: [NOISE_SIZE, NOISE_SIZE, NOISE_SIZE],
      dimension: '3d',
      format: 'rgba8unorm',
    })
    .$usage('sampled', 'storage');
  const noiseWrite = noise.createView(d.textureStorage3d('rgba8unorm', 'write-only'));
  const noiseRead = noise.createView(d.texture3d(d.f32));
  const density = volume();
  const densityRead = density.createView(d.texture3d(d.f32));
  const densityWrite = density.createView(d.textureStorage3d('rgba16float', 'write-only'));
  const lighting = volume();
  const lightRead = lighting.createView(d.texture3d(d.f32));
  const lightWrite = lighting.createView(d.textureStorage3d('rgba16float', 'write-only'));
  const shadowMap = root
    .createTexture({ size: [SHADOW_SIZE, SHADOW_SIZE], format: 'rgba8unorm' })
    .$usage('sampled', 'storage');
  const shadowRead = shadowMap.createView(d.texture2d(d.f32));
  const shadowWrite = shadowMap.createView(d.textureStorage2d('rgba8unorm', 'write-only'));
  const half = targets.create(root, screen, {
    fresh: { format: 'rgba16float', sampled: true, storage: true, scale: 0.5 },
    even: { format: 'rgba16float', sampled: true, storage: true, scale: 0.5 },
    odd: { format: 'rgba16float', sampled: true, storage: true, scale: 0.5 },
  });
  type Half = typeof half.textures.fresh;
  const resolveGroupOf = memo((fresh: Half, kept: Half, target: Half) =>
    root.createBindGroup(resolveLayout, {
      fresh: fresh.createView(d.texture2d(d.f32)),
      kept: kept.createView(d.texture2d(d.f32)),
      target: target.createView(d.textureStorage2d('rgba16float', 'write-only')),
    }),
  );
  const writeGroupOf = memo((texture: Half) =>
    root.createBindGroup(viewWriteLayout, {
      target: texture.createView(d.textureStorage2d('rgba16float', 'write-only')),
    }),
  );
  const readGroupOf = memo((texture: Half) =>
    root.createBindGroup(cloudsLayout, { seen: texture.createView(d.texture2d(d.f32)) }),
  );

  // Billowy Perlin-Worley in red, fine Worley in green, Perlin in blue, coarse Worley in alpha
  const gradients = perlin3d.staticCache({ root, size: d.vec3u(16) });
  const bakeNoise = root.pipe(gradients.inject()).createComputePipeline({
    compute: tgpu.computeFn({
      workgroupSize: [4, 4, 4],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      if (std.any(std.ge(gid, d.vec3u(NOISE_SIZE)))) return;
      const uv = (d.vec3f(gid) + 0.5) / NOISE_SIZE;
      let perlin = d.f32(0);
      for (const octave of tgpu.unroll(std.range(3))) {
        perlin += perlin3d.sample(uv * (16 * 2 ** octave)) * 0.5 ** (octave + 1);
      }
      const base = std.saturate(perlin * 0.5 + 0.5);
      const lumps = worley(uv, d.f32(4));
      const billows = std.saturate((base + 1 - lumps) / (2 - lumps));
      std.textureStore(noiseWrite.$, gid, d.vec4f(billows, worley(uv, d.f32(8)), base, lumps));
    }),
  });

  const slice = (ground: d.v2f, metres: number, at: number) => {
    'use gpu';
    return std.textureSampleLevel(
      noiseRead.$,
      repeat.$,
      d.vec3f(ground.x / metres, at, ground.y / metres),
      0,
    );
  };

  /** The weather over a ground point: its cells' share of the air, their strength, overcast. */
  const weatherAt = (ground: d.v2f) => {
    'use gpu';
    const large = slice(ground, WEATHER.large, 0.31).w;
    const small = slice(ground, WEATHER.small, 0.67).y;
    const strength = slice(ground, WEATHER.strength, 0.13).z;
    const cells = std.max(large, small * 0.9) * std.mix(0.7, 1.25, strength);
    const cover = shape.$.coverage;
    const overcast = std.smoothstep(WEATHER.overcast[0], WEATHER.overcast[1], cover);
    const scattered =
      std.saturate((cells - (1 - std.pow(cover, WEATHER.spread))) / WEATHER.edge) * WEATHER.cell;
    const local =
      std.mix(
        scattered,
        std.max(scattered, WEATHER.floor + (1 - WEATHER.floor) * std.smoothstep(0.2, 0.8, small)),
        overcast,
      ) * std.smoothstep(0, 0.05, cover);
    return d.vec3f(local, large, overcast);
  };

  const bakeDensity = root.createComputePipeline({
    compute: tgpu.computeFn({
      workgroupSize: [4, 4, 4],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      if (std.any(std.ge(gid, VOLUME))) return;
      const uv = (d.vec3f(gid) + 0.5) / d.vec3f(VOLUME);
      // The volume holds the deck in its own frame, which drifts
      const ground = (uv.xz - 0.5) * EXTENT;
      const weather = weatherAt(ground);
      const local = weather.x;
      const large = weather.y;
      const overcast = weather.z;

      const base = PROFILE.base + PROFILE.rolls * overcast * slice(ground, SHAPE.metres, 0.83).z;
      const ceiling = std.mix(PROFILE.lowest, 1, local * std.mix(0.5, 1, large));
      const profile =
        std.smoothstep(base, base + PROFILE.rise, uv.y) *
        (1 - std.smoothstep(ceiling * 0.5, ceiling, uv.y));

      const point = d.vec3f(ground.x, uv.y * HEIGHT, ground.y) / SHAPE.metres;
      const twist = std.textureSampleLevel(noiseRead.$, repeat.$, point * 0.5 + 0.21, 0);
      const warp = (twist.yz - 0.5) * SHAPE.warp;
      const billows = std.textureSampleLevel(
        noiseRead.$,
        repeat.$,
        point + d.vec3f(warp.x, 0, warp.y),
        0,
      ).x;
      // An overcast deck keeps its billows as thicker and thinner parts rather than holes
      const form =
        std.mix(1, 0.5, overcast) * std.smoothstep(SHAPE.range[0], SHAPE.range[1], billows) +
        0.5 * overcast;
      // Cells keep only the shape's strongest parts where the weather is weak
      const value = std.saturate((form * profile - (1 - local)) / std.max(local, 0.15));
      const detail = std.textureSampleLevel(
        noiseRead.$,
        repeat.$,
        (point * SHAPE.metres) / SHAPE.detail,
        0,
      ).y;
      const carved = std.saturate(value - detail * SHAPE.erosion * (1 - value));
      std.textureStore(densityWrite.$, gid, d.vec4f(carved, 0, 0, 1));
    }),
  });

  const densityAt = (uv: d.v3f) => {
    'use gpu';
    if (uv.y <= 0 || uv.y >= 1) return d.f32(0);
    return std.textureSampleLevel(densityRead.$, deckSampler.$, uv, 0).x;
  };

  /** Sunlight through the deck towards the sun, and sky light escaping upwards. */
  const bakeLighting = root.createComputePipeline({
    compute: tgpu.computeFn({
      workgroupSize: [4, 4, 4],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      if (std.any(std.ge(gid, VOLUME))) return;
      const uv = (d.vec3f(gid) + 0.5) / d.vec3f(VOLUME);
      const amount = std.textureLoad(densityRead.$, gid, 0).x;
      let orders = d.vec3f();
      let skyEscape = d.f32(0);
      if (amount > 0) {
        const sun = atmosphere.sunDirection();
        const length = std.min(((1 - uv.y) * HEIGHT) / std.max(sun.y, 0.08), 8000);
        let opticalDepth = d.f32(0);
        for (const tap of std.range(16)) {
          const start = length * (d.f32(tap) / 16) ** 2;
          const end = length * ((d.f32(tap) + 1) / 16) ** 2;
          const at = uv + (sun * ((start + end) * 0.5)) / d.vec3f(EXTENT, HEIGHT, EXTENT);
          opticalDepth += densityAt(at) * (end - start) * SIGMA;
        }
        const path = (1 - uv.y) * HEIGHT;
        let skyDepth = d.f32(0);
        for (const tap of std.range(8)) {
          const start = path * (d.f32(tap) / 8) ** 2;
          const end = path * ((d.f32(tap) + 1) / 8) ** 2;
          const at = uv + d.vec3f(0, (start + end) * 0.5, 0) / HEIGHT;
          skyDepth += densityAt(at) * (end - start) * SIGMA;
        }
        skyEscape = diffuseTransmission(skyDepth);
        for (const order of tgpu.unroll(ORDERS)) {
          orders[order.channel] = order.weight * std.exp(-opticalDepth * order.attenuation);
        }
        // Scattered isotropically, at 1/4π, it leaves as μ T / π, a diffuser's radiance
        orders[DIFFUSE.channel] = 4 * std.max(sun.y, 0.005) * diffuseTransmission(opticalDepth);
      }
      std.textureStore(lightWrite.$, gid, d.vec4f(orders * amount, skyEscape * amount));
    }),
  });

  const bakeShadow = root.createComputePipeline({
    compute: tgpu.computeFn({ workgroupSize: [8, 8], in: { gid: d.builtin.globalInvocationId } })(
      ({ gid }) => {
        'use gpu';
        if (gid.x >= SHADOW_SIZE || gid.y >= SHADOW_SIZE) return;
        const xz = (d.vec2f(gid.xy) + 0.5) / SHADOW_SIZE;
        const sun = atmosphere.sunDirection();
        const step = HEIGHT / std.max(sun.y, 0.08) / 32;
        let transmittance = d.f32(1);
        for (const tap of std.range(32)) {
          const at =
            d.vec3f(xz.x, 0, xz.y) +
            (sun * ((d.f32(tap) + 0.5) * step)) / d.vec3f(EXTENT, HEIGHT, EXTENT);
          transmittance *= segmentTransmission(densityAt(at), step);
          if (transmittance < 0.005) break;
        }
        std.textureStore(shadowWrite.$, gid.xy, d.vec4f(transmittance, 0, 0, 1));
      },
    ),
  });

  /** Erosion of the deck's edges, fading where it would alias; `detailed` adds a finer octave. */
  const billowsOf = (detailed: boolean) => (point: d.v3f, footprint: number) => {
    'use gpu';
    const at = point / BILLOWS.metres;
    const cells = std.textureSampleLevel(noiseRead.$, repeat.$, at, 0);
    const fine = 1 - std.smoothstep(1, 2.5, (footprint * NOISE_SIZE) / BILLOWS.metres);
    const smaller =
      1 - std.smoothstep(1, 2.5, (footprint * NOISE_SIZE * BILLOWS.octave) / BILLOWS.metres);
    let small = d.f32(0.5);
    if (detailed) {
      const finer = std.textureSampleLevel(noiseRead.$, repeat.$, at * BILLOWS.octave + 0.37, 0).y;
      small = std.mix(0.5, dome(finer), smaller);
    }
    const billow = 0.5 * dome(cells.y) + 0.25 * (1 - cells.z) + 0.25 * small;
    return billow * BILLOWS.erosion * fine;
  };

  const erodedOf = (detailed: boolean) => {
    const billowsAt = billowsOf(detailed);
    return (amount: number, height: number, point: d.v3f, footprint: number) => {
      'use gpu';
      const billow = billowsAt(point, footprint);
      // Billows bite deepest into the tops; the bases stay flat where the air condenses
      const bite = billow * std.mix(0.6, 1.3, std.smoothstep(0.05, 0.45, height));
      const eaten = std.saturate((amount * (1 - 0.5 * bite) - bite) / std.max(1 - bite, 0.05));
      // Cumulus ends at a rim rather than fading out, so faint density rises quickly to a body
      return std.max(eaten, 0.3 * std.smoothstep(0.02, 0.1, eaten));
    };
  };
  const eroded = erodedOf(true);
  const shading = erodedOf(false);

  const erodedAt = (point: d.v3f, footprint: number) => {
    'use gpu';
    const uv = volumeUv(point);
    return shading(densityAt(uv), uv.y, point, footprint);
  };

  /** The far bank and veils behind the deck: their light, and the share of sky showing through. */
  const farAt = (ray: d.v3f, sunlight: d.v3f, skylight: d.v3f, glow: d.v3f) => {
    'use gpu';
    const height = ray.y / std.length(ray);
    if (height > FAR.veils || height < -0.01) return d.vec4f(0, 0, 0, 1);
    const around = std.atan2(ray.x, -ray.z) / (2 * Math.PI);
    const outline = std.textureSampleLevel(
      noiseRead.$,
      repeat.$,
      d.vec3f(around * 3, 0.5, 0.21),
      0,
    );
    const heavy = std.smoothstep(0, 0.6, shape.$.coverage);
    const crest = FAR.bank * (0.35 + outline.z * 0.9 + outline.w * 0.4) * heavy;
    const bank = 1 - std.smoothstep(crest - 0.0008, crest + 0.0008, height);
    const sun = atmosphere.sunDirection();
    const lowest = std.max(height, FAR.layer / FAR.metres / 8);
    const distance = FAR.layer / lowest;
    const heading = ray.xz / std.length(ray.xz);
    // Only an overcast goes on past the box: scattered cells that far are smaller than a pixel
    const overcast = std.smoothstep(WEATHER.overcast[0], WEATHER.overcast[1], shape.$.coverage);
    let cells = d.f32(0);
    let thickness = d.f32(0);
    if (overcast > 0) {
      // The ray crosses the deck at several heights, and only strong cells reach the upper ones
      for (const level of tgpu.unroll(FAR.levels)) {
        const reach = (BASE + level * HEIGHT) / lowest;
        const across = heading * reach;
        const beyond = rimOf(across / EXTENT + 0.5);
        const weather = weatherAt(across - shape.$.drift);
        const averaged = std.mix(
          std.saturate(weather.x / WEATHER.cell),
          shape.$.coverage,
          std.smoothstep(FAR.resolved[0], FAR.resolved[1], reach),
        );
        const strong = std.saturate((averaged - level * 0.5) / (1 - level * 0.5));
        cells = std.max(cells, strong * beyond);
        thickness +=
          std.saturate((weather.x - WEATHER.floor) / (1 - WEATHER.floor)) / FAR.levels.length;
      }
    }
    const piled = 1 - std.pow(1 - cells, 1 + FAR.pile / std.max(height, 1e-3));
    const veil = FAR.veil * overcast * piled * std.smoothstep(FAR.bank * 0.3, FAR.bank, height);
    const through = (1 - bank) * (1 - veil);
    // A low sun lights an overcast's base from beyond the deck's edge; a high one diffuses through
    const forward = henyeyGreenstein(std.dot(ray, sun) / std.length(ray), FAR.forward);
    const high = std.smoothstep(FAR.beneath[0], FAR.beneath[1], sun.y);
    const base = std.mix(1, FAR.under * std.mix(1.4, 0.6, thickness), overcast);
    const ambient = sunlight * (FAR.lit * std.max(sun.y, 0)) + skylight * 0.4;
    const diffuse = ambient * base;
    const lit = (sunlight * forward + glow * 0.15) * (1 - overcast * high) + diffuse;
    // Under an overcast the haze takes only the light the deck lets through
    const hazed = std.mix(DECK_HAZE, 1, std.smoothstep(EXTENT / 2, FAR.hazed, distance));
    const dimmed = std.mix(1, FAR.under, overcast);
    const hazy = std.mix(1, dimmed, high);
    const veils = atmosphere.veiled(lit * veil, veil * hazy, ray, distance * hazed);
    // Under a high sun the bank is the same deck as the layer, further out
    const shade = std.mix(skylight * 0.35, ambient * dimmed, high * overcast);
    const banked = atmosphere.veiled(shade * bank, bank * hazy, ray, FAR.metres);
    return d.vec4f(veils * (1 - bank) + banked, through);
  };

  /**
   * Marches a ray from `eye` through the deck: the light per unit of sunlight and the transmittance
   * left. Billows finer than `pixel` fade; only a `billowed` march erodes and self-shades.
   */
  const marcher =
    (steps: number, billowed: boolean) =>
    (eye: d.v3f, ray: d.v3f, pixel: number, jitter: number) => {
      'use gpu';
      const up = std.max(ray.y, 0.001);
      const near = std.max((BASE - eye.y) / up, 0);
      const side =
        (d.vec2f(std.sign(ray.xz)) * (EXTENT / 2) - eye.xz) /
        std.max(std.abs(ray.xz), d.vec2f(1e-4));
      const far = std.min((BASE + HEIGHT - eye.y) / up, std.min(std.abs(side.x), std.abs(side.y)));
      const range = std.max(far - near, 0);
      const sun = atmosphere.sunDirection();
      const mu = std.clamp(std.dot(ray, sun), -1, 1);
      let phases = d.vec3f();
      for (const order of tgpu.unroll(ORDERS)) {
        phases[order.channel] = henyeyGreenstein(mu, order.anisotropy);
      }
      phases[DIFFUSE.channel] = 1 / (4 * Math.PI);
      // The deck's own altitude keeps the sun a little after it sets at the eye
      const sunlight = atmosphere.sunTransmittanceAt(BASE + HEIGHT * 0.25);
      const zenith = atmosphere.skyRadiance(d.vec3f(0, 1, 0));
      const skylight =
        std.mix(d.vec3f(tone.luminance(zenith)), zenith, SKY_FILL.tint) * SKY_FILL.gain;
      const aside = d.vec2f(
        sun.x * UNDERLIGHT.cosine - sun.z * UNDERLIGHT.sine,
        sun.x * UNDERLIGHT.sine + sun.z * UNDERLIGHT.cosine,
      );
      let band = d.vec3f();
      for (const turn of tgpu.unroll(BAND)) {
        const along = d.vec2f(
          sun.x * turn.cosine - sun.z * turn.sine,
          sun.x * turn.sine + sun.z * turn.cosine,
        );
        band += atmosphere.skyRadiance(std.normalize(d.vec3f(along.x, 0.03, along.y)));
      }
      const glow =
        band *
        ((UNDERLIGHT.gain / BAND.length) *
          (1 - std.smoothstep(UNDERLIGHT.fadeFrom, UNDERLIGHT.fadeTo, sun.y)));
      const toward = std.normalize(d.vec2f(sun.x, sun.z) + d.vec2f(1e-5, 0));
      const reach = (EXTENT * 0.44) / std.max(std.abs(toward.x), std.abs(toward.y));
      const below = std.normalize(d.vec3f(aside.x, -1, aside.y)) * SELF_SHADOW.metres;
      const overhead = std.normalize(d.vec3f(sun.x, 1, sun.z)) * SELF_SHADOW.metres;
      const drift = d.vec3f(shape.$.drift.x, 0, shape.$.drift.y);
      const ember = std.smoothstep(UNDERLIGHT.cone, 0.999, mu);
      let throughput = d.f32(1);
      let radiance = d.vec3f();
      let hazeDepth = d.f32(0);
      if (far > near && ray.y > HORIZON.from) {
        for (const tap of std.range(steps)) {
          // Steps start short where the ray enters the deck, the part the eye sees
          const start = (d.f32(tap) / steps) ** 2;
          const end = ((d.f32(tap) + 1) / steps) ** 2;
          const step = range * (end - start);
          const distance = near + range * std.mix(start, end, jitter);
          const world = eye + ray * distance;
          const point = world - drift;
          const uv = volumeUv(point);
          const footprint = distance * pixel;
          const amount = std.textureSampleLevel(densityRead.$, deckSampler.$, uv, 0).x;
          if (amount < 1e-3) continue;
          const light = std.textureSampleLevel(lightRead.$, deckSampler.$, uv, 0);
          let local = amount * (1 - rimOf(volumeUv(world).xz));
          if (billowed) {
            // The finer billows show only on the surface the eye sees, not behind it
            if (throughput > SURFACE) {
              local = eroded(local, uv.y, point, footprint);
            } else {
              local = shading(local, uv.y, point, footprint);
            }
          }
          const segment = segmentTransmission(local, step);
          const inverse = 1 / std.max(amount, 1e-6);
          // Folds above a point dim the light from above, folds below it the horizon's glow
          const above = billowed ? erodedAt(point + overhead, footprint) : d.f32(0);
          const shade = segmentTransmission(above, d.f32(SELF_SHADOW.metres) * SELF_SHADOW.depth);
          let underglow = d.vec3f();
          if (ember > 0 && uv.y < 0.35) {
            const under = billowed
              ? erodedAt(point + below, footprint)
              : densityAt(volumeUv(point + below));
            const inward = reach - std.dot(world.xz, toward);
            const gap = std.exp(-std.max(inward, 0) / UNDERLIGHT.reach);
            underglow =
              glow *
              (segmentTransmission(under, d.f32(SELF_SHADOW.metres) * SELF_SHADOW.depth) *
                (1 - std.smoothstep(0.05, 0.35, uv.y)) *
                gap *
                ember);
          }
          const lit =
            (sunlight * (std.dot(light.xyz, phases) * inverse) +
              skylight * std.saturate(light.w * inverse)) *
              shade +
            underglow;
          radiance += lit * (throughput * (1 - segment));
          hazeDepth += distance * throughput * (1 - segment);
          throughput *= segment;
          if (throughput < 0.01) break;
        }
      }
      const fade = std.smoothstep(HORIZON.from, HORIZON.to, ray.y);
      const cover = (1 - throughput) * fade;
      const deck = atmosphere.veiled(
        radiance * fade,
        cover,
        ray,
        (hazeDepth / std.max(1 - throughput, 1e-4)) * DECK_HAZE,
      );
      const beyond = farAt(ray, sunlight, skylight, glow);
      const through = 1 - cover;
      return d.vec4f(deck + beyond.xyz * through, through * beyond.w);
    };
  const viewMarch = marcher(VIEW_STEPS, true);
  const bakeMarch = marcher(BAKE_STEPS, false);

  const viewKernel = root.pipe(view.inject()).createComputePipeline({
    compute: tgpu.computeFn({ workgroupSize: [8, 8], in: { gid: d.builtin.globalInvocationId } })(
      ({ gid }) => {
        'use gpu';
        const size = std.textureDimensions(viewWriteLayout.$.target);
        let at = gid.xy;
        if (shape.$.blend < 1) {
          at = d.vec2u(gid.x * CHECKER + ((gid.y + d.u32(shape.$.turn)) % CHECKER), gid.y);
        }
        if (at.x >= size.x || at.y >= size.y) return;
        const camera = cameras.current.$;
        const lens = cameras.lensOf(camera);
        const pixel = std.max(2 / (lens.x * d.f32(size.x)), 2 / (lens.y * d.f32(size.y)));
        const uv = (d.vec2f(at) + 0.5) / d.vec2f(size);
        const ray = cameras.rayOf(camera, d.vec2f(uv.x * 2 - 1, 1 - uv.y * 2)).direction;
        // Each pixel's step offset moves on each frame, so the slices average out in the history
        randf.seed2(d.vec2f(at));
        const jitter = std.fract(randf.sample() + shape.$.turn * GOLDEN);
        std.textureStore(
          viewWriteLayout.$.target,
          at,
          viewMarch(camera.position, ray, pixel, jitter),
        );
      },
    ),
  });

  const resolve = root.pipe(view.inject()).createComputePipeline({
    compute: tgpu.computeFn({ workgroupSize: [8, 8], in: { gid: d.builtin.globalInvocationId } })(
      ({ gid }) => {
        'use gpu';
        const size = std.textureDimensions(resolveLayout.$.target);
        if (gid.x >= size.x || gid.y >= size.y) return;
        const fresh = std.textureLoad(resolveLayout.$.fresh, gid.xy, 0);
        // History outside the new march's neighbourhood is stale: the sun, cover or view moved
        let low = d.vec4f(fresh);
        let high = d.vec4f(fresh);
        for (const y of std.range(-1, 2)) {
          for (const x of std.range(-1, 2)) {
            const at = std.clamp(d.vec2i(gid.xy) + d.vec2i(x, y), d.vec2i(0), d.vec2i(size) - 1);
            const near = std.textureLoad(resolveLayout.$.fresh, at, 0);
            low = std.min(low, near);
            high = std.max(high, near);
          }
        }
        const uv = (d.vec2f(gid.xy) + 0.5) / d.vec2f(size);
        const ray = cameras.rayOf(cameras.current.$, d.vec2f(uv.x * 2 - 1, 1 - uv.y * 2)).direction;
        // The deck is far enough that last frame saw each direction where its camera points it
        const before = vanishingBefore(ray);
        let result = d.vec4f(fresh);
        if (before.z > 0) {
          const ndc = before.xy / before.z;
          const then = d.vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
          if (std.all(std.ge(then, d.vec2f(0))) && std.all(std.le(then, d.vec2f(1)))) {
            const kept = std.textureSampleLevel(resolveLayout.$.kept, volumeSampler.$, then, 0);
            // A pixel the march skipped this frame holds last frame's march, already in the history
            const marched =
              shape.$.blend >= 1 || gid.x % CHECKER === (gid.y + d.u32(shape.$.turn)) % CHECKER;
            const share = std.select(0, shape.$.blend, marched);
            result = std.mix(std.clamp(kept, low, high), fresh, share);
          }
        }
        std.textureStore(resolveLayout.$.target, gid.xy, result);
      },
    ),
  });

  let turn = 0;
  let resolved = 0;
  let started = false;
  let noiseBaked = false;

  return {
    initAsync: () =>
      pipelines.compile([bakeNoise, bakeDensity, bakeLighting, bakeShadow, viewKernel, resolve]),
    /** The light and transmittance along any ray, for an environment's bake; see `marcher`. */
    march: (eye: d.v3f, ray: d.v3f, pixel: number) => {
      'use gpu';
      return bakeMarch(eye, ray, pixel, 0.5);
    },
    /** The eye's view at screen `uv`, top left first, from the half-size march. Bind `group`. */
    seen: (uv: d.v2f) => {
      'use gpu';
      return std.textureSampleLevel(cloudsLayout.$.seen, volumeSampler.$, uv, 0);
    },
    /** The share of sunlight the deck lets through to a point on the sea. */
    shadow: (xz: d.v2f) => {
      'use gpu';
      const sun = atmosphere.sunDirection();
      // Where sunlight enters the deck's base; under the deck's thinned edge it reaches the sea
      const entry = (xz + sun.xz * (BASE / std.max(sun.y, 0.01))) / EXTENT + 0.5;
      const drifted = entry - shape.$.drift / EXTENT;
      const shade = std.textureSampleLevel(shadowRead.$, repeat.$, drifted, 0).x;
      return std.mix(shade, 1, rimOf(std.clamp(entry, d.vec2f(0), d.vec2f(1))));
    },
    /** The read group of the eye's view, for drawings that call `seen`. */
    get group() {
      return readGroupOf(resolved === 0 ? half.textures.even : half.textures.odd);
    },
    /** Moves the deck to `drift` metres from where it started. */
    move(drift: readonly [number, number]) {
      shape.patch({ drift: d.vec2f(drift[0], drift[1]) });
    },
    /** Sets the share of the sky the deck covers, from 0 to 1. Record `bake` after it. */
    cover(coverage: number) {
      shape.patch({ coverage });
    },
    /** Records the volume's bake: after creation, and after the sun or the cover changes. */
    bake(pass: TgpuComputePass) {
      if (!noiseBaked) {
        const n = NOISE_SIZE / 4;
        bakeNoise.with(pass).dispatchWorkgroups(n, n, n);
        noiseBaked = true;
      }
      const [x, y, z] = [VOLUME.x / 4, VOLUME.y / 4, VOLUME.z / 4];
      bakeDensity.with(pass).dispatchWorkgroups(x, y, z);
      bakeLighting.with(pass).dispatchWorkgroups(x, y, z);
      bakeShadow.with(pass).dispatchWorkgroups(SHADOW_SIZE / 8, SHADOW_SIZE / 8);
    },
    /** Records the march of half the eye's view, at half size, and its blend into the history. */
    record(pass: TgpuComputePass) {
      const resized = half.fit();
      const whole = !started || resized;
      shape.patch({ turn, blend: whole ? 1 : BLEND });
      turn = (turn + 1) % 4096;
      started = true;
      const { fresh, even, odd } = half.textures;
      const [width, height] = fresh.props.size;
      const groups = [Math.ceil(width / 8), Math.ceil(height / 8)] as const;
      const across = whole ? width : Math.ceil(width / CHECKER);
      viewKernel
        .with(pass)
        .with(writeGroupOf(fresh))
        .dispatchWorkgroups(Math.ceil(across / 8), groups[1]);
      resolved = 1 - resolved;
      const [kept, target] = resolved === 0 ? [odd, even] : [even, odd];
      resolve
        .with(pass)
        .with(resolveGroupOf(fresh, kept, target))
        .dispatchWorkgroups(...groups);
    },
    destroy() {
      for (const resource of [noise, density, lighting, shadowMap, half, gradients]) {
        resource.destroy();
      }
    },
  };
}
