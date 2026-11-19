import { cameras, environment, pipelines, type CameraBase, type Environment } from '@typegpu/kit';
import { d, std, tgpu, type TgpuRenderCommands, type TgpuRoot } from 'typegpu';
import { fresnelWater, meanFresnel, roughFresnel, smithLambda, sunGlint } from './shading.ts';
import type { Waves } from './waves.ts';
import type { Atmosphere } from './atmosphere.ts';
import type { Clouds } from './clouds.ts';
import type { Foam } from './foam.ts';
import {
  CASCADES,
  N,
  patchToWorld,
  RIPPLES,
  surfaceCovariance,
  surfaceJacobian,
  surfaceSlope,
  tensorToWorld,
  WIND,
} from './spectrum.ts';

const GRID = 128;
const LEVELS = 11;
const BASE_CELL = 0.25;
const HOLE = GRID / 2 - 2;
const MORPH = { start: GRID / 2 - 16, end: GRID / 2 - 4 };

/**
 * After GodotOceanWaves (MIT, Ethan Truong) and the water BSDF of Atlas (GDC 2019): water and foam
 * albedos, and the green a wave scatters forward.
 */
const WATER = d.vec3f(0.0100228, 0.0196066, 0.0272118);
const FOAM = d.vec3f(0.75, 0.73, 0.7);
const SCATTER = d.vec3f(0.9, 1.15, 0.85);
/** The slopes' spread the body's Fresnel and masking see. */
const SLOPES = 0.65;
/** The body takes the sun's light by what enters the water, in full from a sun at 10°. */
const SUN_FULL = Math.sin((10 * Math.PI) / 180);
const ENTERING_FULL = SUN_FULL * (1 - fresnelWater(SUN_FULL));
/** A wave lights through from this many metres below the still sea. */
const SCATTER_DEPTH = 2.5;
const FOAM_FADE = 0.0075;
/** The share of the water foam hides, at most `most`, so the water shows through the lace. */
const LACE = {
  held: 0.45,
  gathered: 0.4,
  converging: 1,
  gather: 3,
  whitecap: 0.65,
  fresh: 4,
  most: 0.8,
};
/** Foam streaks upwind of a crest: `taps` samples `step` metres apart, each fading by `fade`. */
const TRAIL = { taps: 4, step: 2, fade: 0.7 };
/** Mist from `begin` to `end` metres, the light on the sea blended `aerial` to the sky. */
const MIST = {
  begin: 300,
  end: 3000,
  curve: 0.5,
  density: 0.7,
  tint: d.vec3f(0.029, 0.069, 0.092),
  aerial: 0.626,
  scatter: 0.05,
};
/** The share of the light's colour the eye adapts away; little at dusk, so dusk stays warm. */
const ADAPTATION = { dusk: 0.7, day: 1, from: 0.035, to: 0.2 };
/** The disc's peak per unit of sunlight, dim enough that the stratus over it shows. */
const SUN_PEAK = 20;
/** The deck hides the disc below this share of sky let through, and shows it from `clear`. */
const DISC = { hidden: 0.35, clear: 0.9 };
const LUMA = d.vec3f(0.2126, 0.7152, 0.0722);
/** The ripples' share of the visible slope; the rest is roughness, so they don't shimmer. */
const SURFACE_DETAIL = 0.7;
const GLINT_CAP = 4;
/** The share of the slopes' width the sky's reflection blurs over. */
const REFLECTION_BLUR = 0.08;
const RIPPLE_BIAS = 2;
/** Seconds of swell motion the ripples' footprint spans: two frames, past which they alias. */
const CARRIED = 2 / 60;

/** Beyond this distance, wave detail filters into roughness. */
const DETAIL_RANGE = 250;
const ROUGHNESS = { near: 0.03, far: 0.1, ramp: 400 };
/** Below the horizon the next wave, tilted skywards, mirrors `share` of the sky from `rise` up. */
const NEXT_WAVE = { share: 0.25, rise: 0.35 };

const Level = d.struct({ origin: d.vec2f, cellSize: d.f32 });
const Slopes = d.struct({
  slope: d.vec2f,
  deformation: d.vec3f,
  covariance: d.vec3f,
  height: d.f32,
});
const gridLayout = tgpu.vertexLayout(d.arrayOf(d.vec2f));

function clipmapMesh() {
  const vertices: d.v2f[] = [];
  for (let j = 0; j <= GRID; j++) {
    for (let i = 0; i <= GRID; i++) vertices.push(d.vec2f(i - GRID / 2, j - GRID / 2));
  }
  const plate: number[] = [];
  const ring: number[] = [];
  const holeStart = (GRID - HOLE) / 2;
  const holeEnd = (GRID + HOLE) / 2;
  for (let j = 0; j < GRID; j++) {
    for (let i = 0; i < GRID; i++) {
      const a = j * (GRID + 1) + i;
      const quad = [a, a + GRID + 1, a + 1, a + 1, a + GRID + 1, a + GRID + 2];
      plate.push(...quad);
      if (i < holeStart || i >= holeEnd || j < holeStart || j >= holeEnd) ring.push(...quad);
    }
  }
  return { vertices, plate, ring };
}

/** Blend towards the next coarser level near a ring's outer edge. */
const morphFactor = (grid: d.v2f) => {
  'use gpu';
  return std.smoothstep(MORPH.start, MORPH.end, std.max(std.abs(grid.x), std.abs(grid.y)));
};

/** Snaps odd vertices onto the coarser lattice so neighbouring levels share edges. */
const morphGrid = (grid: d.v2f, morph: number) => {
  'use gpu';
  return grid - std.fract(grid * 0.5) * 2 * morph;
};

const roughnessFloor = (distance: number) => {
  'use gpu';
  const alpha = std.mix(ROUGHNESS.near, ROUGHNESS.far, std.smoothstep(0, ROUGHNESS.ramp, distance));
  return 0.5 * alpha * alpha;
};

const skyAlong = (direction: d.v3f, roughness: number) => {
  'use gpu';
  const below = std.smoothstep(0, 0.05, -direction.y);
  const height = std.mix(std.max(direction.y, 0.01), std.max(-direction.y, NEXT_WAVE.rise), below);
  const seen = std.normalize(d.vec3f(direction.x, height, direction.z));
  return environment.reflection(seen, roughness) * std.mix(1, NEXT_WAVE.share, below);
};

/**
 * The sea's surface: a clipmap displaced by the cascades, shaded with filtered slope covariance,
 * the water body, sun glint, foam and mist, then the sky where the sea leaves the depth clear.
 */
export function createSurface(
  root: TgpuRoot,
  options: {
    view: CameraBase;
    waves: Waves;
    atmosphere: Atmosphere;
    clouds: Clouds;
    foam: Foam;
    light: Environment;
    format: GPUTextureFormat;
    depth: GPUTextureFormat;
  },
) {
  const { view, waves, atmosphere, clouds, foam: lace, light, format, depth } = options;
  const mesh = clipmapMesh();
  const grid = root
    .createBuffer(gridLayout.schemaForCount(mesh.vertices.length), mesh.vertices)
    .$usage('vertex');
  const plate = root.createBuffer(d.arrayOf(d.u32, mesh.plate.length), mesh.plate).$usage('index');
  const ring = root.createBuffer(d.arrayOf(d.u32, mesh.ring.length), mesh.ring).$usage('index');
  const levels = root.createUniform(d.arrayOf(Level, LEVELS));
  const placed = Array.from({ length: LEVELS }, (_, level) => ({
    origin: d.vec2f(),
    cellSize: BASE_CELL * 2 ** level,
  }));

  const vertex = tgpu.vertexFn({
    in: { grid: d.vec2f, instance: d.builtin.instanceIndex },
    out: { position: d.builtin.position, world: d.vec3f, rest: d.vec2f },
  })((input) => {
    'use gpu';
    const level = levels.$[input.instance];
    const morph = morphFactor(input.grid);
    const rest = level.origin + morphGrid(input.grid, morph) * level.cellSize;
    const footprint = level.cellSize * (1 + morph);
    let offset = d.vec3f();
    for (const c of tgpu.unroll(std.range(RIPPLES))) {
      const cascade = waves.cascades.$[c];
      // Cascades whose waves the cells cannot resolve fade out instead of aliasing
      const weight =
        1 - std.smoothstep(cascade.maxWavelength * 0.5, cascade.maxWavelength * 2, footprint);
      const lod = std.max(std.log2((footprint * N) / cascade.patchSize), 0);
      const sample = waves.displacementAt(rest, c, lod);
      const horizontal = patchToWorld(sample.xz, cascade.rotation);
      offset += d.vec3f(horizontal.x, sample.y, horizontal.y) * weight;
    }
    const world = d.vec3f(rest.x, 0, rest.y) + offset;
    return {
      position: cameras.toClip(world),
      world,
      rest,
    };
  });

  /** Partly adapts the eye to the light falling on the sea from the sun and the sky above it. */
  const balance = () => {
    'use gpu';
    const sun = atmosphere.sunDirection();
    const falling =
      atmosphere.sunTransmittance() * std.max(sun.y, 0.02) +
      environment.irradiance(d.vec3f(0, 1, 0)) +
      1e-6;
    const adaptation = std.mix(
      ADAPTATION.dusk,
      ADAPTATION.day,
      std.smoothstep(ADAPTATION.from, ADAPTATION.to, sun.y),
    );
    return std.pow(d.vec3f(std.dot(falling, LUMA)) / falling, d.vec3f(adaptation));
  };

  /**
   * The cascades at a rest point, filtered over the pixel's world footprint, `dx` by `dy` and
   * `wide` metres across: the resolved slope, the horizontal deformation, the slopes' covariance
   * and the height of the waves the mesh resolves.
   */
  const slopesAt = (rest: d.v2f, dx: d.v2f, dy: d.v2f, wide: number, distance: number) => {
    'use gpu';
    let gradient = d.vec2f();
    let deformation = d.vec3f();
    let cov = d.vec3f();
    let height = d.f32(0);
    const carried = waves.spectrum.$.peakFrequency * waves.spectrum.$.waveHeight * (0.5 * CARRIED);
    for (const c of tgpu.unroll(std.range(CASCADES))) {
      const rotation = waves.cascades.$[c].rotation;
      // Ripples finer than twice the pixel, or than the distance the swell carries them in a frame,
      // move into the slope variance: the glint lobe widens steadily instead of flickering
      const bias =
        c === RIPPLES ? std.max(d.f32(RIPPLE_BIAS), carried / std.max(wide, 1e-6)) : d.f32(1);
      const slopes =
        c > WIND
          ? waves.smoothDerivatives(rest, c, dx * bias, dy * bias)
          : waves.derivatives(rest, c, dx, dy);
      const displacement = waves.displacement(rest, c, dx, dy);
      const moment =
        c > WIND
          ? waves.smoothMoments(rest, c, dx * bias, dy * bias)
          : waves.moments(rest, c, dx, dy);

      const local = d.vec3f(slopes.zw, displacement.w);
      const detail = c === RIPPLES ? d.f32(SURFACE_DETAIL) : d.f32(1);
      const visible = slopes.xy * detail;
      gradient += patchToWorld(visible, rotation);
      deformation += tensorToWorld(local, rotation) * detail;
      cov += tensorToWorld(
        d.vec3f(
          moment.x - visible.x * visible.x,
          moment.y - visible.y * visible.y,
          moment.z - visible.x * visible.y,
        ),
        rotation,
      );
      if (c < RIPPLES) {
        height += displacement.y;
      }
    }
    // Filter linear gradients and moments first, then apply the nonlinear deformation
    cov = surfaceCovariance(cov, deformation);
    const unresolved = roughnessFloor(distance);
    const covX = std.max(cov.x, 0) + unresolved;
    const covY = std.max(cov.y, 0) + unresolved;
    const limit = 0.99 * std.sqrt(covX * covY);
    return Slopes({
      slope: surfaceSlope(gradient, deformation),
      deformation,
      covariance: d.vec3f(covX, covY, std.clamp(cov.z, -limit, limit)),
      height,
    });
  };

  /**
   * The foam's share of the pixel and its bubbles' brightness. The wind waves' coarse foam history
   * marks broken crests. Within it the sea's sharpest compression makes fresh caps, and the rest
   * gathers where the short waves converge.
   */
  const foamAt = (rest: d.v2f, dx: d.v2f, dy: d.v2f, footprint: number, deformation: d.v3f) => {
    'use gpu';
    const windAngle = waves.spectrum.$.windDirection;
    const downwind = d.vec2f(std.cos(windAngle), std.sin(windAngle)) * TRAIL.step;
    let breaking = waves.smoothMoments(rest, WIND, dx, dy).w;
    for (let tap = 1; tap <= TRAIL.taps; tap++) {
      const behind = waves.smoothMoments(rest + downwind * d.f32(tap), WIND, dx, dy).w;
      breaking = std.max(breaking, behind * std.pow(TRAIL.fade, d.f32(tap)));
    }
    const jacobian = surfaceJacobian(deformation);
    const fresh = std.saturate((LACE.whitecap - jacobian) * LACE.fresh);
    const gathered = std.saturate((LACE.converging - jacobian) * LACE.gather);
    const share =
      std.saturate(breaking * (LACE.held + LACE.gathered * gathered) + fresh) * LACE.most;
    return lace.layer(rest, share, 1 - breaking, footprint);
  };

  const fragment = tgpu.fragmentFn({
    in: { world: d.vec3f, rest: d.vec2f },
    out: d.vec4f,
  })((input) => {
    'use gpu';
    const eye = cameras.current.$.position;
    const distance = std.length(input.world - eye);
    const ray = (input.world - eye) / distance;
    const lodScale = std.exp2(std.max(std.log2(distance / DETAIL_RANGE), 0));
    const dx = std.dpdx(input.rest);
    const dy = std.dpdy(input.rest);
    const footprint = std.max(std.length(dx), std.length(dy));
    const lodDx = dx * lodScale;
    const lodDy = dy * lodScale;
    const sea = slopesAt(input.rest, lodDx, lodDy, footprint * lodScale, distance);
    const cov = sea.covariance;
    let normal = std.normalize(d.vec3f(-sea.slope.x, 1, -sea.slope.y));
    const v = ray * -1;
    const facing = std.dot(normal, v);
    if (facing < 0.02) {
      normal = std.normalize(normal - ray * (facing - 0.02));
    }

    const sun = atmosphere.sunDirection();
    const sunColor = atmosphere.sunTransmittance() * clouds.shadow(input.world.xz);
    const layer = foamAt(input.rest, lodDx, lodDy, footprint, sea.deformation);
    const foam = layer.x * std.exp(-distance * FOAM_FADE);
    const nv = std.max(std.dot(normal, v), 2e-5);
    const nl = std.max(std.dot(normal, sun), 2e-5);
    const fresnel = roughFresnel(nv, SLOPES);

    // The sun scattered forward through raised waves, back towards the eye, and onto lit faces
    const viewMask = smithLambda(SLOPES, nv);
    const through =
      std.max(sea.height + SCATTER_DEPTH, 0) *
      std.pow(std.max(std.dot(sun, ray), 0), 4) *
      std.pow(0.5 - 0.5 * std.dot(sun, normal), 3);
    const back = 0.5 * nv * nv;
    const lit = SCATTER * ((through + back) / (1 + viewMask)) + d.vec3f(0.5 * nl);
    const sunIn = std.max(sun.y, 0);
    const entering = std.saturate((sunIn * (1 - fresnelWater(sunIn))) / ENTERING_FULL);
    // Sky low on the horizon glances off the surface, so the body sees mostly the sky above it
    const ambient = environment.irradiance(std.normalize(normal + d.vec3f(0, 1, 0))) / Math.PI;
    const water = sunColor * (WATER * lit * ((1 - fresnel) * entering)) + ambient * WATER;
    // Foam lies on the surface, a rough heap of bubbles that shade each other from a glancing sun
    const bubbles = FOAM * std.mix(0.55, 1.05, layer.y);
    const froth = (sunColor * std.saturate(std.dot(normal, sun)) + ambient) * bubbles;
    const body = std.mix(water, froth, foam);

    const bitangent = std.normalize(std.cross(d.vec3f(1, 0, 0), normal));
    const tangent = std.cross(normal, bitangent);
    const vT = d.vec3f(std.dot(v, tangent), nv, std.dot(v, bitangent));
    const lT = d.vec3f(std.dot(sun, tangent), std.dot(sun, normal), std.dot(sun, bitangent));
    const width = std.sqrt(cov.x + cov.y);
    // Sea slopes are Gaussian, and the cube's blur has long tails that would gather the bright
    // band into every rough reflection: a narrower blur keeps the water mirroring what is above it
    const reflection = skyAlong(
      std.reflect(ray, normal),
      std.sqrt(std.min(width * REFLECTION_BLUR, 1)),
    );
    const surface =
      (reflection * meanFresnel(vT, cov) +
        std.min(sunGlint(vT, lT, cov, sunColor), sunColor * GLINT_CAP)) *
      (1 - foam);

    const falling =
      atmosphere.sunTransmittance() * std.max(sun.y, 0) + environment.irradiance(d.vec3f(0, 1, 0));
    const mistColor =
      std.mix(MIST.tint * falling, skyAlong(ray, 0.9), MIST.aerial) +
      sunColor * (std.pow(std.max(std.dot(ray, sun), 0), 8) * MIST.scatter);
    const mist = std.pow(std.smoothstep(MIST.begin, MIST.end, distance), MIST.curve) * MIST.density;
    const color = std.mix(body + surface, mistColor, mist);
    return d.vec4f(atmosphere.aerialPerspective(color, ray, distance) * balance(), 1);
  });

  // At the far plane, unlike `common.fullScreenTriangle`, so the sky fills only what the sea left
  const fullscreen = tgpu.vertexFn({
    in: { index: d.builtin.vertexIndex },
    out: { position: d.builtin.position, ndc: d.vec2f },
  })(({ index }) => {
    'use gpu';
    const ndc = d.vec2f(d.f32((index << 1) & 2), d.f32(index & 2)) * 2 - 1;
    return { position: d.vec4f(ndc, 1, 1), ndc };
  });

  const background = tgpu.fragmentFn({ in: { ndc: d.vec2f }, out: d.vec4f })(({ ndc }) => {
    'use gpu';
    const ray = cameras.rayOf(cameras.current.$, ndc).direction;
    const deck = clouds.seen(d.vec2f(ndc.x, -ndc.y) * 0.5 + 0.5);
    // The disc is capped so its bloom keeps one size as the sun climbs and brightens
    const disc = atmosphere.sunRadiance(ray);
    const peak = std.max(disc.x, std.max(disc.y, disc.z));
    const shown = std.smoothstep(DISC.hidden, DISC.clear, deck.w);
    const sky =
      atmosphere.skyRadiance(ray) + disc * (std.min(1, SUN_PEAK / std.max(peak, 1e-6)) * shown);
    return d.vec4f((sky * deck.w + deck.xyz) * balance(), 1);
  });

  const configured = root.pipe(view.inject()).pipe(light.inject());
  const ocean = configured.createRenderPipeline({
    attribs: { grid: gridLayout.attrib },
    vertex,
    fragment,
    targets: { format },
    primitive: { cullMode: 'none' },
    depthStencil: { format: depth, depthWriteEnabled: true, depthCompare: 'less' },
  });
  const behind = configured.createRenderPipeline({
    vertex: fullscreen,
    fragment: background,
    targets: { format },
    depthStencil: { format: depth, depthWriteEnabled: false, depthCompare: 'less-equal' },
  });

  return {
    initAsync: () => pipelines.compile([ocean, behind]),
    /** Centres the clipmap's levels under the eye; each snaps to twice its cell. */
    place(x: number, z: number) {
      for (const level of placed) {
        const snap = 2 * level.cellSize;
        level.origin = d.vec2f(
          Math.floor(x / snap + 0.5) * snap,
          Math.floor(z / snap + 0.5) * snap,
        );
      }
      levels.write(placed);
    },
    /** Draws the sea, then the sky where the sea left the depth clear. */
    draw(pass: TgpuRenderCommands) {
      const drawn = ocean.with(pass).with(gridLayout, grid);
      drawn.withIndexBuffer(plate).drawIndexed(mesh.plate.length, 1);
      drawn.withIndexBuffer(ring).drawIndexed(mesh.ring.length, LEVELS - 1, 0, 0, 1);
      behind.with(pass).with(clouds.group).draw(3);
    },
    destroy() {
      grid.destroy();
      plate.destroy();
      ring.destroy();
    },
  };
}
