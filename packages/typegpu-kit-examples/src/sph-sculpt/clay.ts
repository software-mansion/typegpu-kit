import { pipelines } from '@typegpu/kit';
import { perlin2d } from '@typegpu/noise';
import { opSmoothDifference, opSmoothUnion, sdRoundedBox3d } from '@typegpu/sdf';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';
import { MAX_PRIMS, Prim, PRIMS, Terrain, TERRAIN } from './island.ts';

/** Voxel edge in metres at mip 0. */
export const VOXEL = 0.1;
/** Volume size in voxels at mip 0. */
export const DIMS = [256, 128, 256] as const;
/** World box size in metres; the box starts at the origin. */
export const WORLD = d.vec3f(DIMS[0] * VOXEL, DIMS[1] * VOXEL, DIMS[2] * VOXEL);
/** Mip levels. Level 6 is 4 × 2 × 4 voxels. */
export const MIPS = 7;
/**
 * Each mip stores distance clamped to ±BAND of its own voxels, as Claybook's 8-bit volumes do. The
 * clamp is what makes coarse mips useful: level 4 skips up to 64 level-0 voxels in one step.
 */
export const BAND = 4;

/** Encodes a distance in metres for a mip with voxel edge `s`. */
export const encode = (dist: number, s: number) => {
  'use gpu';
  return dist / (2 * BAND * s) + 0.5;
};

/** Decodes a stored value back to metres for a mip with voxel edge `s`. */
export const decode = (value: number, s: number) => {
  'use gpu';
  return (value - 0.5) * (2 * BAND * s);
};

/**
 * Storage formats. Claybook's narrow ones need `texture-formats-tier1`; without it each falls back
 * to `rgba8unorm`, four times the bytes for the same values.
 */
export interface Formats {
  readonly distance: 'r8unorm' | 'rgba8unorm';
  readonly pair: 'rg8unorm' | 'rgba8unorm';
  readonly normal: 'rgb10a2unorm' | 'rgba8unorm';
}

export const formatsOf = (root: TgpuRoot): Formats =>
  root.device.features.has('texture-formats-tier1')
    ? { distance: 'r8unorm', pair: 'rg8unorm', normal: 'rgb10a2unorm' }
    : { distance: 'rgba8unorm', pair: 'rgba8unorm', normal: 'rgba8unorm' };

/** Height map resolution of the ground a build stroke aims at. */
const MAP = 128;
const HEIGHTS = DIMS[0] * DIMS[2];
/** The material volume is one mip down: 0.2 m voxels. */
const MATERIAL_VOXEL = VOXEL * 2;
const MATERIAL_SIZE = [DIMS[0] / 2, DIMS[1] / 2, DIMS[2] / 2] as const;

const voxelOf = (level: number) => VOXEL * 2 ** level;
/**
 * Coarse levels store distance minus √3 voxels, so trilinear filtering between voxel centres never
 * reads more than the true distance. Level 0 is exact; the tracers hit surfaces there.
 */
const dilationOf = (level: number) => (level === 0 ? 0 : Math.sqrt(3) * voxelOf(level));
const sizeOf = (level: number) =>
  DIMS.map((n) => Math.max(1, n >> level)) as [number, number, number];

const LevelInfo = d.struct({
  size: d.vec3u,
  /** Voxel edge in metres. */
  s: d.f32,
  dilation: d.f32,
  level: d.u32,
});

export const Brush = d.struct({
  radius: d.f32,
  /** Smooth blend distance in metres. */
  blend: d.f32,
  /** 0 off, 1 dig, 2 build. */
  mode: d.u32,
  /**
   * How far off the surface the brush centre sits, in radii. Digging moves it towards the eye;
   * building raises it above the ground as it was when the stroke started.
   */
  lift: d.f32,
  /** How far the surface may move this frame, in metres, so a held brush carves at a speed. */
  reach: d.f32,
  /** 1 on a stroke's first frame, which starts a new swept segment. */
  fresh: d.u32,
});

/** What the cursor ray hit in the last traced frame. */
export const Pick = d.struct({
  position: d.vec3f,
  hit: d.u32,
  eye: d.vec3f,
  /** The cursor ray's direction, written even when it misses. */
  direction: d.vec3f,
});

/** This frame's brush: a capsule from last frame's centre to this one's, so strokes leave no gaps. */
export const Aim = d.struct({
  previous: d.vec3f,
  valid: d.u32,
  center: d.vec3f,
  /** Where the cursor ray met the frozen ground, for the brush ring while building. */
  ground: d.vec3f,
});

/** Largest brush radius plus blend, which sizes the edit dispatch. */
export const BRUSH_REACH = 2;
/** Longest segment a brush sweeps in one frame, in metres. A faster cursor leaves gaps. */
const SEGMENT = 0.5;
/** The edit box's half size around the segment's middle. */
const EDIT_REACH = BRUSH_REACH + SEGMENT / 2;
const EDIT_GROUP = 4;
const editExtent = (level: number) => 2 * Math.ceil(EDIT_REACH / voxelOf(level) + 2) + 1;
const SCRATCH = editExtent(0) ** 3;

// `@typegpu/sdf`'s capsule divides by its squared length, and the brush's is 0 at a stroke's start
const sdCapsule = (p: d.v3f, a: d.v3f, b: d.v3f, r: number) => {
  'use gpu';
  const pa = p - a;
  const ba = b - a;
  const h = std.saturate(std.dot(pa, ba) / std.max(std.dot(ba, ba), 1e-8));
  return std.length(pa - ba * h) - r;
};

const primDistance = (p: d.v3f, prim: d.Infer<typeof Prim>) => {
  'use gpu';
  if (prim.shape === 0) return std.length(p - prim.a.xyz) - prim.a.w;
  if (prim.shape === 1) return sdRoundedBox3d(p - prim.a.xyz, prim.b.xyz, prim.a.w);
  return sdCapsule(p, prim.a.xyz, prim.b.xyz, prim.a.w);
};

/**
 * Returns a `'use gpu'` function giving the unit normal of `field` at a point, from four taps on
 * the corners of a tetrahedron `step` metres across each axis.
 */
export const tetrahedralNormal = (field: (p: d.v3f) => number, step: number) => {
  const normalAt = (p: d.v3f) => {
    'use gpu';
    const e = d.vec2f(1, -1) * step;
    return std.normalize(
      e.xyy * field(p + e.xyy) +
        e.yyx * field(p + e.yyx) +
        e.yxy * field(p + e.yxy) +
        e.xxx * field(p + e.xxx),
    );
  };
  return normalAt;
};

/**
 * Clay as an 8-bit narrow-band SDF volume with a mip chain, after Claybook. Generation evaluates
 * the terrain and its CSG list once per mip; edits apply the brush to the few voxels it covers at
 * every mip, so no mip is ever rebuilt from another.
 */
export function createClay(root: TgpuRoot, formats: Formats) {
  const { distance, pair } = formats;

  const volume = root
    .createTexture({ size: [...DIMS], format: distance, dimension: '3d', mipLevelCount: MIPS })
    .$usage('sampled', 'storage');
  const sampled = volume.createView(d.texture3d(d.f32));
  const sampler = root.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'nearest',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
    addressModeW: 'clamp-to-edge',
  });

  /**
   * The island as generated, beside the live SDF, like Claybook's material volume: distance to the
   * original surface in r and to rock in g. The material pass reads one tap of it to tell topsoil
   * from dug strata, built clay and rock, and never touches the SDF.
   */
  const origin = root
    .createTexture({ size: [...MATERIAL_SIZE], format: pair, dimension: '3d' })
    .$usage('sampled', 'storage');
  const originOut = origin.createView(d.textureStorage3d(pair, 'write-only'));
  const originIn = origin.createView(d.texture3d(d.f32));

  const terrain = root.createUniform(Terrain, { ...TERRAIN, prims: PRIMS.length });
  const prims = root.createReadonly(d.arrayOf(Prim, MAX_PRIMS), PRIMS);
  const heights = root.createBuffer(d.arrayOf(d.f32, HEIGHTS)).$usage('storage');
  const heightsOut = heights.as('mutable');
  const heightsIn = heights.as('readonly');
  const brush = root.createUniform(Brush);
  const pick = root.createMutable(Pick);
  const aim = root.createMutable(Aim);
  const scratch = root.createBuffer(d.arrayOf(d.f32, SCRATCH)).$usage('storage');
  const scratchOut = scratch.as('mutable');
  const scratchIn = scratch.as('readonly');

  const writeLayout = tgpu.bindGroupLayout({
    out: { storageTexture: d.textureStorage3d(distance, 'write-only') },
    info: { uniform: LevelInfo },
  });
  const readLayout = tgpu.bindGroupLayout({
    src: { texture: d.texture3d(d.f32) },
    info: { uniform: LevelInfo },
  });
  const levels = Array.from({ length: MIPS }, (_, level) => {
    const info = root
      .createBuffer(LevelInfo, {
        size: d.vec3u(...sizeOf(level)),
        s: voxelOf(level),
        dilation: dilationOf(level),
        level,
      })
      .$usage('uniform');
    const view = { baseMipLevel: level, mipLevelCount: 1 };
    return {
      size: sizeOf(level),
      extent: editExtent(level),
      write: root.createBindGroup(writeLayout, {
        out: volume.createView(d.textureStorage3d(distance, 'write-only'), view),
        info,
      }),
      read: root.createBindGroup(readLayout, {
        src: volume.createView(d.texture3d(d.f32), view),
        info,
      }),
    };
  });

  /** Terrain height at a world column, from the noise. */
  const heightAt = (xz: d.v2f) => {
    'use gpu';
    const t = terrain.$;
    let h = t.base;
    let amp = t.amp;
    let at = xz * t.freq + t.seed;
    // Three soft octaves: rolling ground with no dimples for a low sun to pick out
    for (let i = 0; i < 3; i++) {
      h += perlin2d.sample(at) * amp;
      amp *= 0.45;
      at = std.fma(at, d.vec2f(2.03), d.vec2f(17.1));
    }
    // The island falls away near its rim, so water can be lost over the edge
    const rim = std.length(xz - WORLD.xz * 0.5) / t.island;
    return std.mix(h, -4, std.smoothstep(0.9, 1.05, rim));
  };

  const heightKernel = tgpu.computeFn({
    workgroupSize: [8, 8],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= DIMS[0] || gid.y >= DIMS[2]) return;
    const xz = (d.vec2f(gid.xy) + 0.5) * VOXEL;
    heightsOut.$[gid.y * DIMS[0] + gid.x] = heightAt(xz);
  });

  const heightTap = (x: number, z: number) => {
    'use gpu';
    const ix = d.u32(std.clamp(x, 0, DIMS[0] - 1));
    const iz = d.u32(std.clamp(z, 0, DIMS[2] - 1));
    return heightsIn.$[iz * DIMS[0] + ix];
  };

  /** Bilinear terrain height from the baked columns. */
  const groundAt = (xz: d.v2f) => {
    'use gpu';
    const g = xz / VOXEL - 0.5;
    const base = std.floor(g);
    const f = g - base;
    const h00 = heightTap(base.x, base.y);
    const h10 = heightTap(base.x + 1, base.y);
    const h01 = heightTap(base.x, base.y + 1);
    const h11 = heightTap(base.x + 1, base.y + 1);
    return std.mix(std.mix(h00, h10, f.x), std.mix(h01, h11, f.x), f.y);
  };

  /** Distance to rock, which no brush removes. */
  const rockAt = (p: d.v3f) => {
    'use gpu';
    // The plinth: a slab under the island, tucked inside its rim
    const t = terrain.$;
    const out = std.length(p.xz - WORLD.xz * 0.5) - t.island * 0.86;
    let rock = std.max(p.y - t.bedrock, std.max(out, 0.3 - p.y));
    for (let i = d.u32(0); i < t.prims; i++) {
      const prim = Prim(prims.$[i]);
      if (prim.rock === 1) rock = std.min(rock, primDistance(p, prim));
    }
    return rock;
  };

  /**
   * Signed distance to the terrain, corrected up to `reach` metres. The vertical gap shrunk by the
   * local slope is close on smooth ground, but next to a cliff it reports the drop below, not the
   * wall beside: outside the island's rim that was metres of free space a ray could jump. Every
   * baked column within reach is a point on the surface, so the nearest one corrects the slope
   * estimate. Columns are visited `pitch` metres apart.
   */
  const groundDistance = (p: d.v3f, reach: number, pitch: number) => {
    'use gpu';
    const e = VOXEL;
    const h = groundAt(p.xz);
    const gx = (groundAt(p.xz + d.vec2f(e, 0)) - groundAt(p.xz - d.vec2f(e, 0))) / (2 * e);
    const gz = (groundAt(p.xz + d.vec2f(0, e)) - groundAt(p.xz - d.vec2f(0, e))) / (2 * e);
    const estimate = (p.y - h) / std.sqrt(1 + gx * gx + gz * gz);
    // Outside, the nearest solid column; inside, the nearest column with air above p
    const side = std.select(d.f32(-1), d.f32(1), estimate >= 0);
    let best = std.abs(estimate);
    const step = std.max(d.i32(std.round(pitch / VOXEL)), 1);
    const centre = d.vec2i(std.round((p.xz / VOXEL - 0.5) / d.f32(step))) * step;
    const n = d.i32(std.ceil(std.min(best, reach) / (d.f32(step) * VOXEL)));
    for (let dz = -n; dz <= n; dz++) {
      for (let dx = -n; dx <= n; dx++) {
        const column = centre + d.vec2i(dx, dz) * step;
        const across = std.length((d.vec2f(column) + 0.5) * VOXEL - p.xz);
        if (across >= best) continue;
        const gap = std.max(side * (p.y - heightTap(column.x, column.y)), 0);
        best = std.min(best, std.sqrt(across * across + gap * gap));
      }
    }
    return side * best;
  };

  /** The island's distance field: terrain, its CSG list in order, then rock on top. */
  const fieldAt = (p: d.v3f, reach: number, pitch: number) => {
    'use gpu';
    // A floor slab under everything keeps the island's underside closed
    let dist = std.max(groundDistance(p, reach, pitch), 0.3 - p.y);
    for (let i = d.u32(0); i < terrain.$.prims; i++) {
      const prim = Prim(prims.$[i]);
      if (prim.rock === 1) continue;
      const shape = primDistance(p, prim);
      if (prim.op === 0) dist = opSmoothUnion(dist, shape, prim.b.w);
      else dist = opSmoothDifference(dist, shape, prim.b.w);
    }
    return std.min(dist, rockAt(p));
  };

  const generateKernel = tgpu.computeFn({
    workgroupSize: [4, 4, 4],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const info = writeLayout.$.info;
    if (gid.x >= info.size.x || gid.y >= info.size.y || gid.z >= info.size.z) return;
    const p = (d.vec3f(gid) + 0.5) * info.s;
    // Coarse mips search coarser columns; their dilation covers the gaps between them
    const dist = fieldAt(p, BAND * info.s + info.dilation, info.s * 0.25) - info.dilation;
    std.textureStore(writeLayout.$.out, gid, d.vec4f(encode(dist, info.s), 0, 0, 1));
  });

  const originKernel = tgpu.computeFn({
    workgroupSize: [4, 4, 4],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= MATERIAL_SIZE[0] || gid.y >= MATERIAL_SIZE[1] || gid.z >= MATERIAL_SIZE[2]) return;
    const p = (d.vec3f(gid) + 0.5) * MATERIAL_VOXEL;
    const value = d.vec2f(
      encode(fieldAt(p, BAND * MATERIAL_VOXEL, VOXEL), MATERIAL_VOXEL),
      encode(rockAt(p), MATERIAL_VOXEL),
    );
    std.textureStore(originOut.$, gid, d.vec4f(value, 0, 1));
  });

  /** Edits run over a fixed box of voxels around the segment's middle, the same every frame. */
  const editOrigin = (s: number, extent: number) => {
    'use gpu';
    const middle = (aim.$.previous + aim.$.center) * 0.5;
    return d.vec3i(std.floor(middle / s)) - d.i32(extent / 2);
  };

  const editKernel = tgpu.computeFn({
    workgroupSize: [EDIT_GROUP, EDIT_GROUP, EDIT_GROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const info = readLayout.$.info;
    const extent = d.u32(2 * std.ceil(EDIT_REACH / info.s + 2) + 1);
    if (gid.x >= extent || gid.y >= extent || gid.z >= extent) return;
    const voxel = editOrigin(info.s, d.f32(extent)) + d.vec3i(gid);
    const slot = (gid.z * extent + gid.y) * extent + gid.x;
    if (
      std.any(std.lt(voxel, d.vec3i())) ||
      std.any(std.ge(d.vec3u(voxel), info.size)) ||
      aim.$.valid === 0
    ) {
      scratchOut.$[slot] = -1;
      return;
    }

    const p = (d.vec3f(voxel) + 0.5) * info.s;
    const old = decode(std.textureLoad(readLayout.$.src, voxel, 0).x, info.s);
    const reach = sdCapsule(p, aim.$.previous, aim.$.center, brush.$.radius) - info.dilation;
    let next = d.f32(old);
    if (brush.$.mode === 1) {
      next = std.min(
        opSmoothDifference(old, reach + 2 * info.dilation, brush.$.blend),
        old + brush.$.reach,
      );
    } else {
      next = std.max(opSmoothUnion(old, reach, brush.$.blend), old - brush.$.reach);
    }
    next = std.min(next, rockAt(p) - info.dilation);
    scratchOut.$[slot] = std.saturate(encode(next, info.s));
  });

  const storeKernel = tgpu.computeFn({
    workgroupSize: [EDIT_GROUP, EDIT_GROUP, EDIT_GROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const info = writeLayout.$.info;
    const extent = d.u32(2 * std.ceil(EDIT_REACH / info.s + 2) + 1);
    if (gid.x >= extent || gid.y >= extent || gid.z >= extent) return;
    const value = scratchIn.$[(gid.z * extent + gid.y) * extent + gid.x];
    if (value < 0) return;
    const voxel = editOrigin(info.s, d.f32(extent)) + d.vec3i(gid);
    std.textureStore(writeLayout.$.out, d.vec3u(voxel), d.vec4f(value, 0, 0, 1));
  });

  /**
   * Samples the distance in metres at a mip. Pass a literal level, so the voxel size folds into a
   * constant.
   */
  const sdf = (p: d.v3f, level: number) => {
    'use gpu';
    const lod = d.f32(level);
    const value = std.textureSampleLevel(sampled.$, sampler.$, p / WORLD, lod).x;
    return decode(value, VOXEL * std.exp2(lod));
  };

  /** Distance in metres at level 0. */
  const fine = (p: d.v3f) => {
    'use gpu';
    return sdf(p, 0);
  };
  /** Unit surface normal from level 0. */
  const normalAt = tetrahedralNormal(fine, VOXEL * 0.75);

  /**
   * The island as generated, at a surface point: the distance to its surface in x, negative where
   * the point was dug below it and positive where clay was built above, and the distance to rock
   * in y, both in metres. One filtered tap of the material volume.
   */
  const originAt = (p: d.v3f) => {
    'use gpu';
    const tap = std.textureSampleLevel(originIn.$, sampler.$, p / WORLD, 0).xy;
    return d.vec2f(decode(tap.x, MATERIAL_VOXEL), decode(tap.y, MATERIAL_VOXEL));
  };

  /** A top-down height map of the surface, frozen at a build stroke's start for it to aim at. */
  const map = root.createBuffer(d.arrayOf(d.f32, MAP * MAP)).$usage('storage');
  const mapOut = map.as('mutable');
  const mapIn = map.as('readonly');
  const mapKernel = tgpu.computeFn({
    workgroupSize: [8, 8],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= MAP || gid.y >= MAP) return;
    const xz = (d.vec2f(gid.xy) + 0.5) * (WORLD.x / MAP);
    let y = WORLD.y - VOXEL;
    for (let i = 0; i < 256; i++) {
      const dist = sdf(d.vec3f(xz.x, y, xz.y), 0);
      if (dist < VOXEL * 0.2 || y < 0) break;
      y -= std.max(dist, VOXEL * 0.2);
    }
    mapOut.$[gid.y * MAP + gid.x] = y;
  });

  /** Bilinear height of the ground as the last stroke start froze it. */
  const frozenAt = (xz: d.v2f) => {
    'use gpu';
    const cell = WORLD.x / MAP;
    const g = std.clamp(xz / cell - 0.5, d.vec2f(), d.vec2f(MAP - 1));
    const base = std.min(d.vec2u(g), d.vec2u(MAP - 2));
    const f = g - d.vec2f(base);
    const at = base.y * MAP + base.x;
    const top = std.mix(mapIn.$[at], mapIn.$[at + 1], f.x);
    const bottom = std.mix(mapIn.$[at + MAP], mapIn.$[at + MAP + 1], f.x);
    return std.mix(top, bottom, f.y);
  };

  /**
   * Places this frame's brush segment. Digging centres on the cursor's hit, so a held brush keeps
   * cutting down. Building aims at the ground as it was when the stroke started, so clay rises to
   * a set height instead of growing towards the eye along the cursor ray.
   */
  const aimKernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    const b = brush.$;
    let valid = pick.$.hit === 1;
    const toward = std.normalize(pick.$.eye - pick.$.position);
    let center = pick.$.position + toward * (b.radius * b.lift);
    if (b.mode === 2) {
      // Step along the cursor ray until it dips under the frozen ground, then halve the last step
      const eye = pick.$.eye;
      const dir = pick.$.direction;
      let near = d.f32(0);
      let far = d.f32(0);
      valid = false;
      for (let i = 0; i < 640; i++) {
        const p = eye + dir * far;
        const above = p.y - frozenAt(p.xz);
        if (above < 0) {
          valid = true;
          break;
        }
        near = far;
        far += std.clamp(above * 0.5, 0.02, 0.15);
      }
      for (let i = 0; i < 8; i++) {
        const mid = (near + far) * 0.5;
        const p = eye + dir * mid;
        if (p.y < frozenAt(p.xz)) far = mid;
        else near = mid;
      }
      const ground = eye + dir * far;
      aim.$.ground = d.vec3f(ground);
      center = ground + d.vec3f(0, b.radius * b.lift, 0);
    }

    // Continue from last frame's centre, at most a segment behind
    let start = d.vec3f(center);
    if (b.fresh === 0 && aim.$.valid === 1) {
      const back = aim.$.center - center;
      const length = std.length(back);
      start = center + back * (std.min(length, SEGMENT) / std.max(length, 1e-6));
    }
    aim.$.previous = d.vec3f(start);
    aim.$.center = d.vec3f(center);
    aim.$.valid = d.u32(valid);
  });

  const heighting = root.createComputePipeline({ compute: heightKernel });
  const originating = root.createComputePipeline({ compute: originKernel });
  const generating = root.createComputePipeline({ compute: generateKernel });
  const editing = root.createComputePipeline({ compute: editKernel });
  const storing = root.createComputePipeline({ compute: storeKernel });
  const mapping = root.createComputePipeline({ compute: mapKernel });
  const aiming = root.createComputePipeline({ compute: aimKernel });

  return {
    sampler,
    brush,
    pick,
    aim,
    sdf,
    fine,
    normalAt,
    originAt,

    /** Generates the island at every mip, and the material volume beside it. */
    generate(pass: TgpuComputePass) {
      pass.setPipeline(heighting);
      pass.dispatchWorkgroups(DIMS[0] / 8, DIMS[2] / 8);
      pass.setPipeline(originating);
      pass.dispatchWorkgroups(MATERIAL_SIZE[0] / 4, MATERIAL_SIZE[1] / 4, MATERIAL_SIZE[2] / 4);
      for (const level of levels) {
        const [x, y, z] = level.size;
        pass.setPipeline(generating);
        pass.setBindGroup(level.write);
        pass.dispatchWorkgroups(Math.ceil(x / 4), Math.ceil(y / 4), Math.ceil(z / 4));
      }
    },

    /**
     * Applies this frame's brush to the voxels it covers at every mip. `freeze` keeps the ground as
     * it is now for a build stroke to aim at; pass it on the stroke's first frame.
     */
    edit(pass: TgpuComputePass, freeze: boolean) {
      if (freeze) {
        pass.setPipeline(mapping);
        pass.dispatchWorkgroups(MAP / 8, MAP / 8);
      }
      pass.setPipeline(aiming);
      pass.dispatchWorkgroups(1);
      for (const level of levels) {
        const groups = Math.ceil(level.extent / EDIT_GROUP);
        pass.setPipeline(editing);
        pass.setBindGroup(level.read);
        pass.dispatchWorkgroups(groups, groups, groups);
        pass.setPipeline(storing);
        pass.setBindGroup(level.write);
        pass.dispatchWorkgroups(groups, groups, groups);
      }
    },

    initAsync: () =>
      pipelines.compile([heighting, originating, generating, editing, storing, mapping, aiming]),
    destroy() {
      for (const made of [
        volume,
        origin,
        prims.buffer,
        heights,
        pick.buffer,
        aim.buffer,
        scratch,
        map,
      ]) {
        made.destroy();
      }
    },
  };
}

export type Clay = ReturnType<typeof createClay>;
