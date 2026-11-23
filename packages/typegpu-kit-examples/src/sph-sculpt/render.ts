import {
  cameras,
  environment,
  lighting,
  materials,
  pipelines,
  shading,
  targets,
  type Surface,
} from '@typegpu/kit';
import { perlin3d } from '@typegpu/noise';
import {
  d,
  std,
  tgpu,
  type TgpuCommandEncoder,
  type TgpuComputePass,
  type TgpuRoot,
} from 'typegpu';
import { BAND, tetrahedralNormal, VOXEL, WORLD, type Clay, type Formats } from './clay.ts';
import { SPRING } from './island.ts';
import { CELL, type Water } from './water.ts';

/** Pixels per side of a cone pre-pass tile. */
const TILE = 8;
/** The coarsest mip rays start at, and the levels they drop or climb at a time. */
const TOP = 4;
const STRIDE = 2;
/** Noise period in noise units, and texels along each side of the baked volume. */
const NOISE_PERIOD = 32;
const NOISE_SIZE = 128;
/** Depth written where a ray meets nothing. */
const FAR = 1e4;
const STEPS = 128;
/** Voxels across the tetrahedron the shading normal samples. */
const NORMAL_SPAN = 2.5;
const FLUID_STEPS = 64;

/** What changes each frame: the screen, the cursor's pixel and the brush ring. */
export const View = d.struct({
  size: d.vec2u,
  cursor: d.vec2i,
  /** 0 no brush ring, 1 dig, 2 build, 3 hover. */
  mode: d.u32,
});

/** What the settings choose. */
export const Choices = d.struct({
  /** The brush ring's radius in metres. */
  radius: d.f32,
  shadows: d.u32,
  ao: d.u32,
});

/**
 * A golden-hour sun 22° up on the camera's right: high enough that the clay's gentle slopes read
 * as smooth light, low enough that the rock walls throw long shadows across the valley.
 */
const SUN = std.normalize(d.vec3f(0.886, 0.4, 0.463));
/**
 * The sun through the evening air, the daylight sky it scatters, and haze from 30 m out. The sky
 * gradient stays dark: the daylight adds to it.
 */
const LIGHT = {
  sunDirection: [SUN.x, SUN.y, SUN.z],
  sunColor: [...environment.sunColor(SUN)] as [number, number, number],
  sunIntensity: 6,
  skyTop: [0.01, 0.012, 0.02],
  skyBottom: [0.01, 0.008, 0.006],
  skyIntensity: 1,
  fogDensity: 0.004,
  fogStart: 30,
} satisfies Partial<d.InferInput<typeof lighting.Settings>>;
/** Height of the sea around the island, and its distance from the island's centre at the shore. */
const SEA = 1.1;
const SHORE = 10.6;
/** The sea's colour offshore and over the shallows by the island. */
const SEA_DEEP = d.vec3f(0.006, 0.03, 0.045);
const SEA_SHALLOW = d.vec3f(0.03, 0.16, 0.15);
/** The warm haze along the horizon. */
const HAZE = d.vec3f(1.1, 0.55, 0.25);
/** How bright the sun's disc is where a ray meets nothing, over its light on a surface. */
const SUN_DISC = 4;
/** Roughness of topsoil and dug clay, rock and built clay, and water, whose glint stays sharp. */
const ROUGHNESS = { clay: 0.55, rock: 0.75, built: 0.5, water: 0.22 };
/** Absorption per metre of water, for its colour by thickness. */
const ABSORB = d.vec3f(0.9, 0.22, 0.12);
/** Clay strata from low to high, like a cut through layered plasticine. */
const STRATA = [
  d.vec3f(0.56, 0.42, 0.28),
  d.vec3f(0.22, 0.27, 0.13),
  d.vec3f(0.55, 0.33, 0.09),
  d.vec3f(0.27, 0.1, 0.08),
  d.vec3f(0.56, 0.42, 0.28),
];
const ROCK = d.vec3f(0.122, 0.136, 0.202);
/** Terracotta plasticine, a shade deeper low down. */
const TOPSOIL_LOW = d.vec3f(0.44, 0.17, 0.09);
const TOPSOIL_HIGH = d.vec3f(0.5, 0.21, 0.11);
/** Clay built above the island as generated. */
const BUILT = d.vec3f(0.863, 0.768, 0.624);
const WATER = d.vec3f(0.13, 0.686, 0.631);
const DEEP = d.vec3f(0.017, 0.262, 0.397);
const FOAM = d.vec3f(0.807, 0.955, 0.823);
const SPRING_AT = d.vec3f(...SPRING.at);
/**
 * Metres from the island as generated where dug strata and built clay begin, the width of the
 * blend into them and into rock, and the distance from rock where the rock's colour is half in.
 */
const MATERIAL_EDGE = 0.06;
const MATERIAL_BLEND = 0.1;
const ROCK_EDGE = 0.12;
/** Coarse mips store distance less √3 of their voxels; samples add it back. */
const DILATION_2 = Math.sqrt(3) * VOXEL * 4;
const DILATION_4 = Math.sqrt(3) * VOXEL * 16;
/** How sharply the sun's shadows fall off, and the nearest distance that sharpens them. */
const PENUMBRA = 9;
const PENUMBRA_FLOOR = 0.4;

const pixelKernel = tgpu.computeFn({
  workgroupSize: [8, 8],
  in: { gid: d.builtin.globalInvocationId },
});

const boxHit = (o: d.v3f, dir: d.v3f) => {
  'use gpu';
  const inv = 1 / dir;
  const a = (d.vec3f() - o) * inv;
  const b = (WORLD - o) * inv;
  const lo = std.min(a, b);
  const hi = std.max(a, b);
  return d.vec2f(
    std.max(std.max(lo.x, lo.y), std.max(lo.z, 0)),
    std.min(hi.x, std.min(hi.y, hi.z)),
  );
};

const packNormal = (n: d.v3f) => {
  'use gpu';
  return d.vec4f(n * 0.5 + 0.5, 0);
};

const unpackNormal = (v: d.v4f) => {
  'use gpu';
  return std.normalize(v.xyz * 2 - 1);
};

/**
 * The baked sky with a band of warm haze along the horizon, warmest towards the sun. Fog takes
 * its colour from the same band, so the sea melts into the sky.
 */
const hazeOf = (direction: d.v3f) => {
  'use gpu';
  const flat = std.normalize(d.vec3f(direction.x, std.max(direction.y, 0), direction.z));
  const near = std.pow(1 - flat.y, 10);
  const sunward = std.saturate(std.dot(std.normalize(flat.xz), std.normalize(SUN.xz)) * 0.5 + 0.5);
  return environment.backdrop(flat) + HAZE * (near * (0.35 + 0.65 * sunward * sunward));
};

/** The sky and the sun where a ray meets nothing. */
const skyOf = (direction: d.v3f) => {
  'use gpu';
  return hazeOf(direction) + environment.sunDisc(direction, SUN_DISC);
};

/** Where a ray meets the sea, or a negative distance where it misses or the island stands. */
const seaHit = (o: d.v3f, dir: d.v3f) => {
  'use gpu';
  const t = (SEA - o.y) / dir.y;
  const at = o + dir * t;
  const offshore = std.length(at.xz - WORLD.xz * 0.5) > SHORE;
  return std.select(d.f32(-1), t, dir.y < 0 && t > 0 && offshore);
};

/** Sunlight at the pixel being painted, which the shadow pass found. */
const sunLit = tgpu.privateVar(d.f32, 1);

const fromShadowPass = (_point: shading.Point) => {
  'use gpu';
  return sunLit.$;
};

/** How far past the terminator light wraps round the clay, as through a waxy surface. */
const WRAP = 0.3;

/**
 * Plasticine: diffuse light that wraps a little past the terminator, and `pbr`'s broad highlight
 * on top. Surfaces without colour, such as the water's film, keep only the highlight.
 */
const waxy = (
  material: shading.Material,
  point: shading.Point,
  direction: d.v3f,
  radiance: d.v3f,
) => {
  'use gpu';
  const wrapped = std.saturate((std.dot(point.normal, direction) + WRAP) / (1 + WRAP));
  const glossy = shading.Material(material);
  glossy.color = d.vec3f();
  return (
    material.color * radiance * (wrapped * wrapped) +
    shading.pbr(glossy, point, direction, radiance)
  );
};

/** The sky's diffuse light on the clay over what the environment gives, so shadows stay soft. */
const FILL = 0.8;

/** The environment's light, with `FILL` more of its diffuse share. */
const filled = (material: shading.Material, point: shading.Point) => {
  'use gpu';
  const diffuse = material.color * (1 - material.metallic) * material.occlusion;
  return (
    environment.ambient(material, point) + diffuse * environment.irradiance(point.normal) * FILL
  );
};

/** Fog in the colour of the sky at the horizon behind the point, so the sea melts into it. */
const skyward = (point: shading.Point) => {
  'use gpu';
  const away = point.view * -1;
  const hidden = shading.haze(point);
  const added = hazeOf(d.vec3f(away.x, 0, away.z)) * hidden;
  return shading.Air({ added, through: d.vec3f(1 - hidden) });
};

export const CHOICES = { radius: 0.7, shadows: true, occlusion: true };

/**
 * Ray traces the clay and the water in compute: an 8 × 8 cone pre-pass, a trace that writes depth
 * and normals while the SDF is hot in cache, a half-resolution pass for sun shadow and ambient
 * occlusion together, and a material pass that never touches the SDF.
 */
export function createRender(
  root: TgpuRoot,
  surface: Surface,
  { clay, water, formats }: { clay: Clay; water: Water; formats: Formats },
) {
  const { distance, normal, pair } = formats;
  const camera = cameras.orbit(root, {
    size: surface,
    target: [WORLD.x / 2, 2, WORLD.z / 2],
    distance: 27,
    yaw: -0.3,
    pitch: 0.36,
    fov: 0.85,
    pitchRange: [0.15, 1.45],
    distanceRange: [6, 60],
    near: 0.1,
    far: 400,
  });
  const view = root.createUniform(View);
  const choices = root.createUniform(Choices, {
    radius: CHOICES.radius,
    shadows: CHOICES.shadows ? 1 : 0,
    ao: CHOICES.occlusion ? 1 : 0,
  });
  const sky = environment.create(root, { size: 256 });
  const light = lighting.create(root, {
    settings: LIGHT,
    shading: { sunShadow: fromShadowPass, ambient: filled, reflect: waxy, fog: skyward },
  });
  const daylight = sky.add(tgpu.fn(environment.daylight).pipe(light.inject()));

  // Perlin noise baked into a tiling volume: one filtered tap instead of eight gradients
  const noise = root
    .createTexture({
      size: [NOISE_SIZE, NOISE_SIZE, NOISE_SIZE],
      format: distance,
      dimension: '3d',
    })
    .$usage('sampled', 'storage');
  const noiseOut = noise.createView(d.textureStorage3d(distance, 'write-only'));
  const noiseIn = noise.createView(d.texture3d(d.f32));
  const noiseSampler = root.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'repeat',
    addressModeV: 'repeat',
    addressModeW: 'repeat',
  });
  const noiseCache = perlin3d.staticCache({
    root,
    size: d.vec3u(NOISE_PERIOD, NOISE_PERIOD, NOISE_PERIOD),
  });
  const bakeNoise = tgpu.computeFn({
    workgroupSize: [4, 4, 4],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const u = (d.vec3f(gid) + 0.5) * (NOISE_PERIOD / NOISE_SIZE);
    std.textureStore(noiseOut.$, gid, d.vec4f(perlin3d.sample(u) * 0.5 + 0.5, 0, 0, 1));
  });
  const bakingNoise = root.pipe(noiseCache.inject()).createComputePipeline({ compute: bakeNoise });

  const screen = targets.create(root, surface, {
    /** Where the clay cone stopped in r, and the water cone in g. */
    cone: { format: 'rg32float', scale: 1 / TILE, sampled: true, storage: true },
    depth: { format: 'r32float', sampled: true, storage: true },
    normal: { format: normal, sampled: true, storage: true },
    fluidDepth: { format: 'r32float', sampled: true, storage: true },
    fluidNormal: { format: normal, sampled: true, storage: true },
    /** Sun visibility in r, ambient openness in g. */
    shadow: { format: pair, scale: 0.5, sampled: true, storage: true },
    hdr: { format: 'rgba16float', sampled: true, storage: true },
  });

  const filtered = { texture: d.texture2d(d.f32) };
  const exact = { texture: d.texture2d(d.f32), sampleType: 'unfilterable-float' } as const;
  const coneLayout = tgpu.bindGroupLayout({
    cone: { storageTexture: d.textureStorage2d('rg32float', 'write-only') },
  });
  const worldLayout = tgpu.bindGroupLayout({
    cone: exact,
    depth: { storageTexture: d.textureStorage2d('r32float', 'write-only') },
    normal: { storageTexture: d.textureStorage2d(normal, 'write-only') },
  });
  const fluidLayout = tgpu.bindGroupLayout({
    cone: exact,
    depth: exact,
    fluidDepth: { storageTexture: d.textureStorage2d('r32float', 'write-only') },
    fluidNormal: { storageTexture: d.textureStorage2d(normal, 'write-only') },
  });
  const shadowLayout = tgpu.bindGroupLayout({
    depth: exact,
    normal: filtered,
    shadow: { storageTexture: d.textureStorage2d(pair, 'write-only') },
  });
  const paintLayout = tgpu.bindGroupLayout({
    depth: exact,
    normal: filtered,
    fluidDepth: exact,
    fluidNormal: filtered,
    shadow: filtered,
    hdr: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
  });

  const groupsFor = () => {
    const t = screen.textures;
    const read = d.texture2d(d.f32);
    return {
      cone: root.createBindGroup(coneLayout, {
        cone: t.cone.createView(d.textureStorage2d('rg32float', 'write-only')),
      }),
      world: root.createBindGroup(worldLayout, {
        cone: t.cone.createView(read),
        depth: t.depth.createView(d.textureStorage2d('r32float', 'write-only')),
        normal: t.normal.createView(d.textureStorage2d(normal, 'write-only')),
      }),
      fluid: root.createBindGroup(fluidLayout, {
        cone: t.cone.createView(read),
        depth: t.depth.createView(read),
        fluidDepth: t.fluidDepth.createView(d.textureStorage2d('r32float', 'write-only')),
        fluidNormal: t.fluidNormal.createView(d.textureStorage2d(normal, 'write-only')),
      }),
      shadow: root.createBindGroup(shadowLayout, {
        depth: t.depth.createView(read),
        normal: t.normal.createView(read),
        shadow: t.shadow.createView(d.textureStorage2d(pair, 'write-only')),
      }),
      paint: root.createBindGroup(paintLayout, {
        depth: t.depth.createView(read),
        normal: t.normal.createView(read),
        fluidDepth: t.fluidDepth.createView(read),
        fluidNormal: t.fluidNormal.createView(read),
        shadow: t.shadow.createView(read),
        hdr: t.hdr.createView(d.textureStorage2d('rgba16float', 'write-only')),
      }),
    };
  };

  /**
   * The clay's normal for shading, over a wider tetrahedron than the water's collisions use, so
   * the 8-bit field's steps don't show in grazing light.
   */
  const smoothNormalAt = tetrahedralNormal(clay.fine, VOXEL * NORMAL_SPAN);

  const rayAt = (pixel: d.v2f) => {
    'use gpu';
    const uv = pixel / d.vec2f(view.$.size);
    return cameras.rayOf(cameras.current.$, d.vec2f(uv.x * 2 - 1, 1 - uv.y * 2));
  };

  /** Tangent of one pixel's angle, for cone radii. */
  const pixelSpread = () => {
    'use gpu';
    return 2 / (d.f32(view.$.size.y) * cameras.lensOf(cameras.current.$).y);
  };

  /**
   * The tracing loop over the mip chain. Starts coarse, drops two levels when a sample
   * is within half a voxel of a surface, climbs back when the band clamps, and steps the sampled
   * distance less the cone radius. A zero `spread` traces a ray; the pre-pass traces a cone.
   */
  const march = (o: d.v3f, dir: d.v3f, start: number, stop: number, spread: number) => {
    'use gpu';
    const widen = d.f32(spread);
    let t = d.f32(start);
    let level = d.i32(TOP);
    let steps = d.u32(0);
    let hit = false;
    for (let i = 0; i < STEPS; i++) {
      if (t > stop) break;
      steps += 1;
      const p = o + dir * t;
      const s = VOXEL * std.exp2(d.f32(level));
      const dist = clay.sdf(p, d.f32(level)) - t * widen;
      if (level > 0) {
        if (dist < s * 0.5) {
          level -= STRIDE;
          continue;
        }
      } else if (dist < 0.0006 * t + VOXEL * 0.02) {
        hit = true;
        break;
      }
      t += dist;
      if (level < TOP && dist > s * (BAND - 0.5)) level += STRIDE;
    }
    return d.vec3f(t, d.f32(hit), d.f32(steps));
  };

  const cone = pixelKernel(function cone({ gid }) {
    'use gpu';
    const tiles = (view.$.size + TILE - 1) / TILE;
    if (gid.x >= tiles.x || gid.y >= tiles.y) return;
    const ray = rayAt((d.vec2f(gid.xy) + 0.5) * TILE);
    const span = boxHit(ray.origin, ray.direction);
    let start = d.f32(0);
    let wet = d.f32(0);
    if (span.x < span.y) {
      const spread = pixelSpread() * TILE * 0.75;
      start = march(ray.origin, ray.direction, span.x, span.y, spread).x;
      // The same cone through the water's tile gaps, so water rays skip the dry tiles too
      wet = span.x;
      for (let i = 0; i < 48; i++) {
        if (wet > span.y) break;
        const gap = water.fluidGapAt(ray.origin + ray.direction * wet) - wet * spread;
        if (gap < CELL * 0.5) break;
        wet += gap;
      }
    }
    std.textureStore(coneLayout.$.cone, gid.xy, d.vec4f(start, wet, 0, 0));
  });

  const traceWorld = pixelKernel(function traceWorld({ gid }) {
    'use gpu';
    if (gid.x >= view.$.size.x || gid.y >= view.$.size.y) return;
    const ray = rayAt(d.vec2f(gid.xy) + 0.5);
    const span = boxHit(ray.origin, ray.direction);
    let depth = d.f32(FAR);
    let packed = d.vec4f(0.5, 1, 0.5, 0);
    if (span.x < span.y) {
      const start = std.max(span.x, std.textureLoad(worldLayout.$.cone, gid.xy / TILE, 0).x);
      const traced = march(ray.origin, ray.direction, start, span.y, 0);
      if (traced.y > 0) {
        depth = traced.x;
        // The SDF is hot in cache right here, so the normal is cheapest now
        packed = packNormal(smoothNormalAt(ray.origin + ray.direction * depth));
      }
    }
    // The cursor's pixel leaves its hit on the clay for next frame's brush
    if (gid.x === d.u32(view.$.cursor.x) && gid.y === d.u32(view.$.cursor.y)) {
      const hit = depth < FAR;
      clay.pick.$.hit = d.u32(hit);
      clay.pick.$.eye = d.vec3f(ray.origin);
      clay.pick.$.direction = d.vec3f(ray.direction);
      if (hit) clay.pick.$.position = ray.origin + ray.direction * depth;
    }

    // The sea, where it lies nearer than the clay; the normal's alpha marks it
    const sea = seaHit(ray.origin, ray.direction);
    if (sea > 0 && sea < depth) {
      depth = sea;
      packed = d.vec4f(0.5, 1, 0.5, 1);
    }
    std.textureStore(worldLayout.$.depth, gid.xy, d.vec4f(depth, 0, 0, 0));
    std.textureStore(worldLayout.$.normal, gid.xy, packed);
  });

  /** Traces the water: its tiles skip empty space, its cells find the surface. */
  const traceFluid = pixelKernel(function traceFluid({ gid }) {
    'use gpu';
    if (gid.x >= view.$.size.x || gid.y >= view.$.size.y) return;
    const ray = rayAt(d.vec2f(gid.xy) + 0.5);
    const span = boxHit(ray.origin, ray.direction);
    const stop = std.min(span.y, std.textureLoad(fluidLayout.$.depth, gid.xy, 0).x);
    let depth = d.f32(FAR);
    let packed = d.vec4f(0.5, 1, 0.5, 0);
    if (span.x < stop) {
      let t = std.max(span.x, std.textureLoad(fluidLayout.$.cone, gid.xy / TILE, 0).y);
      for (let i = 0; i < FLUID_STEPS; i++) {
        if (t > stop) break;
        const p = ray.origin + ray.direction * t;
        // Tiles away from water skip by their gap. The fine field is only rebuilt near water, so
        // it is only sampled where the gap says water is close
        const gap = water.fluidGapAt(p);
        if (gap >= CELL * 0.5) {
          t += gap;
          continue;
        }
        const dist = water.fluidAt(p);
        if (dist < 0.002 * t + 0.004) {
          depth = t;
          break;
        }
        t += dist;
      }
      if (depth < FAR) packed = packNormal(water.fluidNormalAt(ray.origin + ray.direction * depth));
    }
    std.textureStore(fluidLayout.$.fluidDepth, gid.xy, d.vec4f(depth, 0, 0, 0));
    std.textureStore(fluidLayout.$.fluidNormal, gid.xy, packed);
  });

  /**
   * Soft sun shadow: a cone towards the sun through coarser mips as it widens. Its width never
   * narrows below `PENUMBRA_FLOOR`, so the 8-bit field's steps along a slope the ray grazes
   * don't darken the slope itself.
   */
  const sunThrough = (o: d.v3f) => {
    'use gpu';
    const sun = shading.towardsSun();
    const span = boxHit(o, sun);
    const exit = std.select(d.f32(-1), span.y, span.x < span.y);
    let lit = d.f32(1);
    let t = std.max(d.f32(VOXEL), span.x);
    for (let i = 0; i < 40; i++) {
      if (t > exit) break;
      const p = o + sun * t;
      let dist = d.f32(0);
      if (t < 1.2) dist = clay.sdf(p, 0);
      else if (t < 8) dist = clay.sdf(p, 2) + DILATION_2;
      else dist = clay.sdf(p, 4) + DILATION_4;
      lit = std.min(lit, (dist * PENUMBRA) / std.max(t, PENUMBRA_FLOOR));
      if (lit < 0.01) break;
      t += std.max(dist, 0.04 + t * 0.04);
    }
    return std.smoothstep(0, 1, lit);
  };

  /** Ambient openness from four taps along the normal, wider taps on coarser mips. */
  const openAt = (p: d.v3f, n: d.v3f) => {
    'use gpu';
    let shut = std.max(0.12 - clay.sdf(p + n * 0.12, 0), 0) / 0.12;
    shut += (std.max(0.3 - clay.sdf(p + n * 0.3, 0), 0) / 0.3) * 0.8;
    shut += (std.max(0.8 - clay.sdf(p + n * 0.8, 2) - DILATION_2, 0) / 0.8) * 0.6;
    shut += (std.max(2 - clay.sdf(p + n * 2, 4) - DILATION_4, 0) / 2) * 0.45;
    return std.saturate(1 - shut * 0.35);
  };

  /**
   * Shadow and occlusion in one half-resolution pass, the only one after the trace that reads the
   * SDF. They share the depth and normal loads and the ray rebuild.
   */
  const shadowKernel = pixelKernel(function shadowKernel({ gid }) {
    'use gpu';
    const half = std.textureDimensions(shadowLayout.$.shadow);
    if (gid.x >= half.x || gid.y >= half.y) return;
    // Each texel lights the top-left pixel of its 2 × 2 block
    const pixel = std.min(gid.xy * 2, view.$.size - 1);
    const depth = std.textureLoad(shadowLayout.$.depth, pixel, 0).x;
    let reached = d.vec2f(1, 1);
    if (depth < FAR) {
      const ray = rayAt(d.vec2f(pixel) + 0.5);
      const n = unpackNormal(std.textureLoad(shadowLayout.$.normal, pixel, 0));
      const p = ray.origin + ray.direction * depth;
      if (choices.$.shadows === 1) reached.x = sunThrough(p + n * (VOXEL * 2));
      if (choices.$.ao === 1) reached.y = openAt(p, n);
    }
    std.textureStore(shadowLayout.$.shadow, gid.xy, d.vec4f(reached, 0, 1));
  });

  const noiseAt = (p: d.v3f) => {
    'use gpu';
    return std.textureSampleLevel(noiseIn.$, noiseSampler.$, p / NOISE_PERIOD, 0).x * 2 - 1;
  };

  /** Bands of clay colour by height, wobbled by noise, that a dig cuts through. */
  const strataOf = (p: d.v3f) => {
    'use gpu';
    const wobble = noiseAt(p * 0.35) * 0.9;
    const band = std.fract((p.y + wobble) * 0.45) * 5;
    const i = d.u32(band);
    const f = std.smoothstep(0.82, 1, std.fract(band));
    let a = d.vec3f(STRATA[0]);
    let b = d.vec3f(STRATA[1]);
    if (i === 1) {
      a = d.vec3f(STRATA[1]);
      b = d.vec3f(STRATA[2]);
    } else if (i === 2) {
      a = d.vec3f(STRATA[2]);
      b = d.vec3f(STRATA[3]);
    } else if (i === 3) {
      a = d.vec3f(STRATA[3]);
      b = d.vec3f(STRATA[4]);
    } else if (i >= 4) {
      a = d.vec3f(STRATA[4]);
      b = d.vec3f(STRATA[0]);
    }
    return std.mix(a, b, f);
  };

  /**
   * What the clay is at a hit: topsoil, rock, dug strata or built clay, blended across
   * `MATERIAL_BLEND` at each boundary so the edges between them stay smooth. The spring's mouth
   * glows.
   */
  const clayAt = (p: d.v3f, occlusion: number) => {
    'use gpu';
    const origin = clay.originAt(p);
    const drift = noiseAt(p * 0.4) * 0.5 + 0.5;
    let albedo = std.mix(
      TOPSOIL_LOW,
      TOPSOIL_HIGH,
      std.saturate((p.y - 3) * 0.5 + drift * 0.5 - 0.25),
    );
    let roughness = d.f32(ROUGHNESS.clay);
    const dug = std.smoothstep(-MATERIAL_EDGE, -MATERIAL_EDGE - MATERIAL_BLEND, origin.x);
    if (dug > 0) albedo = std.mix(albedo, strataOf(p), dug);
    const built = std.smoothstep(MATERIAL_EDGE, MATERIAL_EDGE + MATERIAL_BLEND, origin.x);
    albedo = std.mix(albedo, BUILT, built);
    roughness = std.mix(roughness, ROUGHNESS.built, built);
    const rock = std.smoothstep(
      ROCK_EDGE + MATERIAL_BLEND / 2,
      ROCK_EDGE - MATERIAL_BLEND / 2,
      origin.y,
    );
    albedo = std.mix(albedo, ROCK * (0.92 + noiseAt(p * 0.5) * 0.12), rock);
    roughness = std.mix(roughness, ROUGHNESS.rock, rock);
    const glow = std.smoothstep(0.9, 0.3, std.length(p - SPRING_AT)) * 0.6;
    const made = materials.glossy(albedo, roughness);
    made.emission = d.vec3f(0.3, 0.9, 1.4) * glow;
    made.occlusion = occlusion;
    return made;
  };

  /**
   * The brush ring around the cursor's hit: faint while hovering, warm while digging, blue while
   * building, around the ground the clay rises from.
   */
  const ringOver = (color: d.v3f, p: d.v3f) => {
    'use gpu';
    const mode = view.$.mode;
    const building = mode === 2 && clay.aim.$.valid === 1;
    let ringed = d.vec3f(color);
    if (building || (mode > 0 && clay.pick.$.hit === 1)) {
      let centre = d.vec3f(clay.pick.$.position);
      if (building) centre = d.vec3f(clay.aim.$.ground);
      const edge = std.smoothstep(0.06, 0, std.abs(std.length(p - centre) - choices.$.radius));
      let tint = d.vec3f(1.4, 1.2, 1.1);
      let strength = d.f32(0.8);
      if (mode === 2) tint = d.vec3f(0.6, 1.1, 1.6);
      if (mode === 3) strength = 0.35;
      ringed = std.mix(color, tint, edge * strength);
    }
    return ringed;
  };

  /**
   * The half-resolution light at a full-resolution pixel: bilinear over the four nearest texels,
   * each weighted down when the pixel it lit sits at a different depth, so shadow and occlusion
   * don't bleed across silhouettes.
   */
  const shadowAt = (pixel: d.v2u, depth: number) => {
    'use gpu';
    const half = d.vec2i(std.textureDimensions(paintLayout.$.shadow));
    const at = (d.vec2f(pixel) + 0.5) * 0.5 - 0.5;
    const base = d.vec2i(std.floor(at));
    const f = at - std.floor(at);
    const tolerance = 0.05 + depth * 0.01;
    let sum = d.vec2f();
    let weight = d.f32(0);
    for (let j = 0; j < 2; j++) {
      for (let i = 0; i < 2; i++) {
        const q = std.clamp(base + d.vec2i(i, j), d.vec2i(), half - 1);
        const lit = std.min(d.vec2u(q) * 2, view.$.size - 1);
        const apart = std.abs(std.textureLoad(paintLayout.$.depth, lit, 0).x - depth);
        const wx = std.select(1 - f.x, f.x, i === 1);
        const wy = std.select(1 - f.y, f.y, j === 1);
        const w = wx * wy * std.exp2((-apart / tolerance) * 4) + 1e-5;
        sum += std.textureLoad(paintLayout.$.shadow, q, 0).xy * w;
        weight += w;
      }
    }
    return sum / weight;
  };

  /** The swell's height at `p`, from the baked noise. */
  const swellAt = (p: d.v3f) => {
    'use gpu';
    const at = d.vec3f(p.x, 0, p.z) * 0.14;
    return noiseAt(at) + noiseAt(at * 3.1 + 5) * 0.4;
  };

  /**
   * The sea at `p`: its body green over the shallows by the island and deep offshore, foam at the
   * shore, the sky and the sun's glint on its swell, and the island's shadow across it.
   */
  const seaAt = (p: d.v3f, sun: number) => {
    'use gpu';
    const e = 0.15;
    const h = swellAt(p);
    // The swell flattens with distance, where its waves are finer than a pixel
    const far = std.length(p - cameras.current.$.position);
    const calm = e * (4 + far * 0.25);
    const n = std.normalize(
      d.vec3f(h - swellAt(p + d.vec3f(e, 0, 0)), calm, h - swellAt(p + d.vec3f(0, 0, e))),
    );
    const offshore = std.length(p.xz - WORLD.xz * 0.5) - SHORE;
    const body = std.mix(SEA_SHALLOW, SEA_DEEP, std.saturate(offshore / 6));
    const foam = std.smoothstep(0.45, 0, offshore) * (0.4 + h * 0.25);
    sunLit.$ = sun;
    const waves = materials.glossy(
      std.mix(body, d.vec3f(0.7), foam * 0.7),
      std.mix(0.08, 0.6, foam),
    );
    return shading.shade(waves, shading.pointAt(p, n));
  };

  const paintKernel = pixelKernel(function paintKernel({ gid }) {
    'use gpu';
    if (gid.x >= view.$.size.x || gid.y >= view.$.size.y) return;
    const ray = rayAt(d.vec2f(gid.xy) + 0.5);
    const dir = ray.direction;
    const depth = std.textureLoad(paintLayout.$.depth, gid.xy, 0).x;

    let color = skyOf(dir);
    const packedNormal = std.textureLoad(paintLayout.$.normal, gid.xy, 0);
    if (depth < FAR && packedNormal.w > 0.5) {
      color = seaAt(ray.origin + dir * depth, shadowAt(gid.xy, depth).x);
    } else if (depth < FAR) {
      const p = ray.origin + dir * depth;
      const n = unpackNormal(packedNormal);
      const lit = shadowAt(gid.xy, depth);
      sunLit.$ = lit.x;
      color = ringOver(shading.shade(clayAt(p, lit.y), shading.pointAt(p, n)), p);
    }

    const top = std.textureLoad(paintLayout.$.fluidDepth, gid.xy, 0).x;
    if (top < depth) {
      const n = unpackNormal(std.textureLoad(paintLayout.$.fluidNormal, gid.xy, 0));
      const thick = std.min(depth - top, 4);
      const through = std.exp(ABSORB * -thick);
      const point = shading.pointAt(ray.origin + dir * top, n);
      // A body colour that holds even in a thin film, deeper where thicker
      const lit = 0.55 + 0.45 * std.saturate(std.dot(n, shading.towardsSun()));
      const opacity = std.saturate(0.45 + (1 - through.y) * 0.6);
      const tint = std.mix(WATER, DEEP, std.saturate(thick * 0.8));
      let body = std.mix(color * through, tint * lit * 1.15, opacity);
      // Foam where the water thins over the clay
      body = std.mix(body, FOAM * lit, std.smoothstep(0.1, 0.02, thick) * 0.6);
      // The surface reflects the sky and the sun's glint, and lets through what it doesn't reflect
      const facing = std.saturate(std.dot(n, point.view));
      const reflected = shading.reflectance(d.vec3f(0.04), ROUGHNESS.water, facing);
      sunLit.$ = 1;
      const film = materials.glossy(d.vec3f(), ROUGHNESS.water);
      film.emission = body * (1 - reflected);
      color = shading.shade(film, point);
    }
    std.textureStore(paintLayout.$.hdr, gid.xy, d.vec4f(color, 1));
  });

  const piped = root.pipe(camera.inject()).pipe(light.inject()).pipe(sky.inject());
  const coning = piped.createComputePipeline({ compute: cone });
  const tracing = piped.createComputePipeline({ compute: traceWorld });
  const watering = piped.createComputePipeline({ compute: traceFluid });
  const shadowing = piped.createComputePipeline({ compute: shadowKernel });
  const painting = piped.createComputePipeline({ compute: paintKernel });

  let groups = groupsFor();

  return {
    camera,
    view,
    choices,
    hdr: () => screen.textures.hdr,

    /** Bakes the material noise and morning sky. Call once, before the first `record`. */
    bake(encoder: TgpuCommandEncoder) {
      bakingNoise.with(encoder).dispatchWorkgroups(NOISE_SIZE / 4, NOISE_SIZE / 4, NOISE_SIZE / 4);
      sky.bake(encoder, daylight);
    },

    /** Rebuilds bind groups when the targets were replaced. Call once a frame before `record`. */
    fit() {
      if (screen.fit()) groups = groupsFor();
      return screen;
    },

    /** The cone pre-pass, the two traces, the shadow pass and the material pass. */
    record(pass: TgpuComputePass) {
      const pixels = [Math.ceil(screen.width / 8), Math.ceil(screen.height / 8)] as const;
      const tiles = [Math.ceil(pixels[0] / TILE), Math.ceil(pixels[1] / TILE)] as const;
      pass.setPipeline(coning);
      pass.setBindGroup(groups.cone);
      pass.dispatchWorkgroups(...tiles);
      pass.setPipeline(tracing);
      pass.setBindGroup(groups.world);
      pass.dispatchWorkgroups(...pixels);
      pass.setPipeline(watering);
      pass.setBindGroup(groups.fluid);
      pass.dispatchWorkgroups(...pixels);
      pass.setPipeline(shadowing);
      pass.setBindGroup(groups.shadow);
      pass.dispatchWorkgroups(Math.ceil(screen.width / 16), Math.ceil(screen.height / 16));
      pass.setPipeline(painting);
      pass.setBindGroup(groups.paint);
      pass.dispatchWorkgroups(...pixels);
    },

    initAsync: () =>
      pipelines.compile([bakingNoise, sky, coning, tracing, watering, shadowing, painting]),
    destroy() {
      for (const made of [sky, light, noise, noiseCache, screen]) made.destroy();
    },
  };
}
