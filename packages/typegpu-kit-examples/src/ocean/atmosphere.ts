import { pipelines } from '@typegpu/kit';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';

/** Planet and atmosphere radii in kilometres. */
const GROUND_RADIUS = 6360;
const TOP_RADIUS = 6460;
const ATMOSPHERE_HEIGHT = TOP_RADIUS - GROUND_RADIUS;
/** Scattering and extinction per kilometre at sea level, and scale heights in kilometres. */
const RAYLEIGH_SCATTERING = d.vec3f(5.802e-3, 13.558e-3, 33.1e-3);
const RAYLEIGH_HEIGHT = 8;
const MIE_SCATTERING = 3.996e-3;
const MIE_EXTINCTION = 4.44e-3;
const MIE_HEIGHT = 1.2;
const MIE_G = 0.8;
const HAZE = 3.5;
const OZONE_EXTINCTION = d.vec3f(0.65e-3, 1.881e-3, 0.085e-3);
const OZONE_CENTER = 25;
const OZONE_WIDTH = 15;
const TRANSMITTANCE_SIZE = d.vec2f(256, 64);
const MULTI_SCATTER_SIZE = 32;
const SKY_VIEW_SIZE = d.vec2f(192, 108);
const TRANSMITTANCE_STEPS = 40;
const MULTI_SCATTER_STEPS = 20;
const SPHERE_SAMPLES = 8;
const SKY_VIEW_STEPS = 32;
const TILE = 8;
const SUN_ANGULAR_RADIUS = 0.004675;
const SUN_SOLID_ANGLE = Math.PI * SUN_ANGULAR_RADIUS ** 2;
const SUN_COS_EDGE = Math.cos(SUN_ANGULAR_RADIUS);

const Settings = d.struct({
  sunDirection: d.vec3f,
  haze: d.f32,
  /** The eye's distance from the planet's centre in kilometres. */
  eyeRadius: d.f32,
});

const Medium = d.struct({ rayleigh: d.vec3f, mie: d.vec3f, extinction: d.vec3f });

const mediumAt = (r: number, haze: number) => {
  'use gpu';
  const height = std.max(r - GROUND_RADIUS, 0);
  const rayleigh = RAYLEIGH_SCATTERING * std.exp(-height / RAYLEIGH_HEIGHT);
  const mieDensity = std.exp(-height / MIE_HEIGHT) * haze;
  const ozone = OZONE_EXTINCTION * std.max(1 - std.abs(height - OZONE_CENTER) / OZONE_WIDTH, 0);
  return Medium({
    rayleigh,
    mie: d.vec3f(MIE_SCATTERING * mieDensity),
    extinction: rayleigh + MIE_EXTINCTION * mieDensity + ozone,
  });
};

/** Distance to the nearest intersection in front of the origin, negative when the ray misses. */
const intersectSphere = (origin: d.v3f, dir: d.v3f, radius: number) => {
  'use gpu';
  const b = std.dot(origin, dir);
  const c = std.dot(origin, origin) - d.f32(radius * radius);
  const disc = b * b - c;
  if (disc < 0) {
    return d.f32(-1);
  }
  const near = -b - std.sqrt(disc);
  if (near > 0) {
    return near;
  }
  return -b + std.sqrt(disc);
};

const rayLength = (origin: d.v3f, dir: d.v3f) => {
  'use gpu';
  const ground = intersectSphere(origin, dir, GROUND_RADIUS);
  if (ground > 0) {
    return ground;
  }
  return std.max(intersectSphere(origin, dir, TOP_RADIUS), 0);
};

const subUv = (uv: d.v2f, size: d.v2f) => {
  'use gpu';
  return (uv + 0.5 / size) * (size / (size + 1));
};

const fromSubUv = (uv: d.v2f, size: d.v2f) => {
  'use gpu';
  return (uv - 0.5 / size) * (size / (size - 1));
};

const transmittanceUv = (r: number, mu: number) => {
  'use gpu';
  const h = std.sqrt(TOP_RADIUS * TOP_RADIUS - GROUND_RADIUS * GROUND_RADIUS);
  const rho = std.sqrt(std.max(r * r - GROUND_RADIUS * GROUND_RADIUS, 0));
  const disc = r * r * (mu * mu - 1) + TOP_RADIUS * TOP_RADIUS;
  const dist = std.max(-r * mu + std.sqrt(std.max(disc, 0)), 0);
  const dMin = TOP_RADIUS - r;
  const dMax = rho + h;
  return d.vec2f((dist - dMin) / (dMax - dMin), rho / h);
};

/** Angle between the nadir and the horizon seen from radius r. */
const horizonBeta = (r: number) => {
  'use gpu';
  return std.acos(std.sqrt(std.max(r * r - GROUND_RADIUS * GROUND_RADIUS, 0)) / r);
};

const rayleighPhase = (cosTheta: number) => {
  'use gpu';
  return (3 / (16 * Math.PI)) * (1 + cosTheta * cosTheta);
};

const miePhase = (cosTheta: number) => {
  'use gpu';
  const g2 = MIE_G * MIE_G;
  return (
    ((3 / (8 * Math.PI)) * ((1 - g2) * (1 + cosTheta * cosTheta))) /
    ((2 + g2) * std.pow(1 + g2 - 2 * MIE_G * cosTheta, 1.5))
  );
};

export type Atmosphere = ReturnType<typeof createAtmosphere>;

/**
 * A planet's atmosphere after Hillaire's 2020 lookup tables: transmittance, multiple scattering
 * and the sky seen from the eye. Radiance is per unit of sunlight above the air.
 */
export function createAtmosphere(root: TgpuRoot) {
  const settings = root.createUniform(Settings, {
    sunDirection: d.vec3f(0, 1, 0),
    haze: HAZE,
    eyeRadius: GROUND_RADIUS,
  });
  const lutSampler = root.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });
  const createLut = (width: number, height: number) => {
    const texture = root
      .createTexture({ size: [width, height], format: 'rgba16float' })
      .$usage('sampled', 'storage');
    return {
      texture,
      target: texture.createView(d.textureStorage2d('rgba16float', 'write-only')),
      view: texture.createView(d.texture2d(d.f32)),
    };
  };
  const transmittanceLut = createLut(TRANSMITTANCE_SIZE.x, TRANSMITTANCE_SIZE.y);
  const multiScatterLut = createLut(MULTI_SCATTER_SIZE, MULTI_SCATTER_SIZE);
  const skyViewLut = createLut(SKY_VIEW_SIZE.x, SKY_VIEW_SIZE.y);
  const sunlight = root.createMutable(d.vec3f);

  const transmittance = (r: number, mu: number) => {
    'use gpu';
    const uv = transmittanceUv(r, mu);
    return std.textureSampleLevel(transmittanceLut.view.$, lutSampler.$, uv, 0).xyz;
  };

  const sunlightAt = (pos: d.v3f, sun: d.v3f) => {
    'use gpu';
    let light = d.vec3f();
    if (intersectSphere(pos, sun, GROUND_RADIUS) < 0) {
      light = transmittance(std.length(pos), std.dot(pos, sun) / std.length(pos));
    }
    return light;
  };

  const bakeTransmittance = tgpu.computeFn({
    workgroupSize: [TILE, TILE],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= TRANSMITTANCE_SIZE.x || gid.y >= TRANSMITTANCE_SIZE.y) return;
    const uv = (d.vec2f(gid.xy) + 0.5) / TRANSMITTANCE_SIZE;
    const h = std.sqrt(TOP_RADIUS * TOP_RADIUS - GROUND_RADIUS * GROUND_RADIUS);
    const rho = h * uv.y;
    const r = std.sqrt(rho * rho + GROUND_RADIUS * GROUND_RADIUS);
    const dMin = TOP_RADIUS - r;
    const dist = dMin + uv.x * (rho + h - dMin);
    let mu = d.f32(1);
    if (dist > 0) {
      mu = std.clamp((h * h - rho * rho - dist * dist) / (2 * r * dist), -1, 1);
    }
    const origin = d.vec3f(0, r, 0);
    const dir = d.vec3f(std.sqrt(1 - mu * mu), mu, 0);
    const step = rayLength(origin, dir) / TRANSMITTANCE_STEPS;
    let depth = d.vec3f();
    for (let i = 0; i < TRANSMITTANCE_STEPS; i++) {
      const pos = origin + dir * ((d.f32(i) + 0.5) * step);
      depth += mediumAt(std.length(pos), settings.$.haze).extinction * step;
    }
    std.textureStore(transmittanceLut.target.$, gid.xy, d.vec4f(std.exp(depth * -1), 1));
  });

  /** Second and higher scattering orders summed as a geometric series. */
  const bakeMultiScatter = tgpu.computeFn({
    workgroupSize: [TILE, TILE],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= MULTI_SCATTER_SIZE || gid.y >= MULTI_SCATTER_SIZE) return;
    const uv = (d.vec2f(gid.xy) + 0.5) / MULTI_SCATTER_SIZE;
    const sunCos = uv.x * 2 - 1;
    const sun = d.vec3f(std.sqrt(1 - sunCos * sunCos), sunCos, 0);
    const origin = d.vec3f(0, GROUND_RADIUS + uv.y * ATMOSPHERE_HEIGHT, 0);
    let luminance = d.vec3f();
    let scattered = d.vec3f();
    for (let a = 0; a < SPHERE_SAMPLES; a++) {
      for (let b = 0; b < SPHERE_SAMPLES; b++) {
        const theta = (Math.PI * (d.f32(a) + 0.5)) / SPHERE_SAMPLES;
        const phi = (2 * Math.PI * (d.f32(b) + 0.5)) / SPHERE_SAMPLES;
        const dir = d.vec3f(
          std.sin(theta) * std.cos(phi),
          std.cos(theta),
          std.sin(theta) * std.sin(phi),
        );
        const step = rayLength(origin, dir) / MULTI_SCATTER_STEPS;
        let throughput = d.vec3f(1);
        for (let j = 0; j < MULTI_SCATTER_STEPS; j++) {
          const pos = origin + dir * ((d.f32(j) + 0.5) * step);
          const medium = mediumAt(std.length(pos), settings.$.haze);
          const segment = std.exp(medium.extinction * -step);
          const scattering = medium.rayleigh + medium.mie;
          const integrated = (1 - segment) / std.max(medium.extinction, d.vec3f(1e-7));
          luminance +=
            throughput * scattering * sunlightAt(pos, sun) * (integrated / (4 * Math.PI));
          scattered += throughput * scattering * integrated;
          throughput *= segment;
        }
      }
    }
    const samples = SPHERE_SAMPLES * SPHERE_SAMPLES;
    const psi = luminance / samples / (1 - scattered / samples);
    std.textureStore(multiScatterLut.target.$, gid.xy, d.vec4f(psi, 1));
  });

  /** Sky radiance seen from the eye, by view zenith and azimuth from the sun. */
  const bakeSkyView = tgpu.computeFn({
    workgroupSize: [TILE, TILE],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= SKY_VIEW_SIZE.x || gid.y >= SKY_VIEW_SIZE.y) return;
    const uv = fromSubUv((d.vec2f(gid.xy) + 0.5) / SKY_VIEW_SIZE, SKY_VIEW_SIZE);
    const r = settings.$.eyeRadius;
    const beta = horizonBeta(r);
    const horizonAngle = Math.PI - beta;
    let zenithAngle = d.f32(0);
    if (uv.y < 0.5) {
      const coord = 1 - 2 * uv.y;
      zenithAngle = horizonAngle * (1 - coord * coord);
    } else {
      const coord = uv.y * 2 - 1;
      zenithAngle = horizonAngle + beta * coord * coord;
    }
    const azimuthCos = -(uv.x * uv.x * 2 - 1);
    const azimuthSin = std.sqrt(std.max(1 - azimuthCos * azimuthCos, 0));
    const dir = d.vec3f(
      std.sin(zenithAngle) * azimuthCos,
      std.cos(zenithAngle),
      std.sin(zenithAngle) * azimuthSin,
    );
    const sunCos = settings.$.sunDirection.y;
    const sun = d.vec3f(std.sqrt(std.max(1 - sunCos * sunCos, 0)), sunCos, 0);
    const origin = d.vec3f(0, r, 0);
    const cosTheta = std.dot(dir, sun);
    const phaseR = rayleighPhase(cosTheta);
    const phaseM = miePhase(cosTheta);
    const step = rayLength(origin, dir) / SKY_VIEW_STEPS;
    let luminance = d.vec3f();
    let throughput = d.vec3f(1);
    for (let i = 0; i < SKY_VIEW_STEPS; i++) {
      const pos = origin + dir * ((d.f32(i) + 0.5) * step);
      const height = std.length(pos);
      const medium = mediumAt(height, settings.$.haze);
      const segment = std.exp(medium.extinction * -step);
      const multiUv = d.vec2f(
        (std.dot(pos, sun) / height) * 0.5 + 0.5,
        (height - GROUND_RADIUS) / ATMOSPHERE_HEIGHT,
      );
      const psi = std.textureSampleLevel(multiScatterLut.view.$, lutSampler.$, multiUv, 0).xyz;
      const source =
        sunlightAt(pos, sun) * (medium.rayleigh * phaseR + medium.mie * phaseM) +
        psi * (medium.rayleigh + medium.mie);
      const integrated = (1 - segment) / std.max(medium.extinction, d.vec3f(1e-7));
      luminance += throughput * source * integrated;
      throughput *= segment;
    }
    std.textureStore(skyViewLut.target.$, gid.xy, d.vec4f(luminance, 1));
  });

  /** Sky radiance per unit of sunlight in a world direction, from the eye. */
  const skyRadiance = (dir: d.v3f) => {
    'use gpu';
    const r = settings.$.eyeRadius;
    const beta = horizonBeta(r);
    const horizonAngle = Math.PI - beta;
    const zenithAngle = std.acos(std.clamp(dir.y, -1, 1));
    let v = d.f32(0);
    if (zenithAngle < horizonAngle) {
      v = (1 - std.sqrt(std.saturate(1 - zenithAngle / horizonAngle))) * 0.5;
    } else {
      v = std.sqrt(std.saturate((zenithAngle - horizonAngle) / beta)) * 0.5 + 0.5;
    }
    const flat = dir.xz / std.max(std.length(dir.xz), 1e-4);
    const sun = settings.$.sunDirection;
    const sunFlat = sun.xz / std.max(std.length(sun.xz), 1e-4);
    const u = std.sqrt(std.saturate(-std.dot(flat, sunFlat) * 0.5 + 0.5));
    const uv = subUv(d.vec2f(u, v), SKY_VIEW_SIZE);
    return std.textureSampleLevel(skyViewLut.view.$, lutSampler.$, uv, 0).xyz;
  };

  const eyeTransmittance = (dir: d.v3f) => {
    'use gpu';
    const origin = d.vec3f(0, settings.$.eyeRadius, 0);
    if (intersectSphere(origin, dir, GROUND_RADIUS) > 0) {
      return d.vec3f();
    }
    return transmittance(settings.$.eyeRadius, dir.y);
  };

  const bakeLighting = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    sunlight.$ = eyeTransmittance(settings.$.sunDirection);
  });

  const transmittancePipeline = root.createComputePipeline({ compute: bakeTransmittance });
  const multiScatterPipeline = root.createComputePipeline({ compute: bakeMultiScatter });
  const skyViewPipeline = root.createComputePipeline({ compute: bakeSkyView });
  const lightingPipeline = root.createComputePipeline({ compute: bakeLighting });

  /** Like `aerialPerspective`, for cloud hiding `opacity` of the sky: haze adds only there. */
  const veiled = (color: d.v3f, opacity: number, dir: d.v3f, distance: number) => {
    'use gpu';
    const extinction = mediumAt(d.f32(GROUND_RADIUS), settings.$.haze).extinction;
    const fog = std.exp(extinction * (-distance / 1000));
    const horizon = skyRadiance(std.normalize(d.vec3f(dir.x, 0.002, dir.z)));
    return color * fog + horizon * ((1 - fog) * opacity);
  };

  return {
    initAsync: () =>
      pipelines.compile([
        transmittancePipeline,
        multiScatterPipeline,
        skyViewPipeline,
        lightingPipeline,
      ]),
    skyRadiance,
    /** A unit vector towards the sun. */
    sunDirection: () => {
      'use gpu';
      return d.vec3f(settings.$.sunDirection);
    },

    /** The share of sunlight reaching the eye's altitude. */
    sunTransmittance: () => {
      'use gpu';
      return d.vec3f(sunlight.$);
    },
    /** The share of sunlight reaching `metres` above the sea, lit after the sun sets at the eye. */
    sunTransmittanceAt: (metres: number) => {
      'use gpu';
      const position = d.vec3f(0, GROUND_RADIUS + metres / 1000, 0);
      return sunlightAt(position, settings.$.sunDirection);
    },
    /** The sun's disc, per unit of sunlight. */
    sunRadiance: (dir: d.v3f) => {
      'use gpu';
      const cosine = std.dot(dir, settings.$.sunDirection);
      const disc = std.smoothstep(SUN_COS_EDGE - 3e-6, SUN_COS_EDGE + 3e-6, cosine);
      return eyeTransmittance(dir) * (disc / SUN_SOLID_ANGLE);
    },
    /** Extinction and in-scattering along a near-level ray of `distance` metres. */
    aerialPerspective: (color: d.v3f, dir: d.v3f, distance: number) => {
      'use gpu';
      return veiled(color, d.f32(1), dir, distance);
    },
    veiled,
    /** Moves the sun and the eye, in metres above the sea. Record `bake` after it. */
    place(sunDirection: d.v3f, eyeHeight: number) {
      settings.patch({
        sunDirection,
        eyeRadius: GROUND_RADIUS + Math.max(eyeHeight, 0.5) / 1000,
      });
    },
    /** Records the four bakes; record them again after `place`. */
    bake(pass: TgpuComputePass) {
      transmittancePipeline
        .with(pass)
        .dispatchWorkgroups(TRANSMITTANCE_SIZE.x / TILE, TRANSMITTANCE_SIZE.y / TILE);
      multiScatterPipeline
        .with(pass)
        .dispatchWorkgroups(MULTI_SCATTER_SIZE / TILE, MULTI_SCATTER_SIZE / TILE);
      skyViewPipeline
        .with(pass)
        .dispatchWorkgroups(Math.ceil(SKY_VIEW_SIZE.x / TILE), Math.ceil(SKY_VIEW_SIZE.y / TILE));
      lightingPipeline.with(pass).dispatchWorkgroups(1);
    },
    destroy() {
      for (const lut of [transmittanceLut, multiScatterLut, skyViewLut]) lut.texture.destroy();
      sunlight.buffer.destroy();
    },
  };
}
