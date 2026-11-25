import { backends, pipelines } from '@typegpu/kit';
import * as grid from '@typegpu/kit/grid';
import { randf } from '@typegpu/noise';
import {
  d,
  std,
  tgpu,
  type TgpuCommandEncoder,
  type TgpuComputePass,
  type TgpuRoot,
} from 'typegpu';
import { decode, DIMS, encode, tetrahedralNormal, WORLD, type Clay, type Formats } from './clay.ts';
import { SPRING } from './island.ts';

/** SPH cell edge, which is also the kernel radius and the fluid surface voxel. */
export const CELL = 0.2;
/** Particle pool size. */
export const POOL = 1 << 16;
/** Seconds per simulation step. */
export const STEP = 1 / 120;

/** Where new particles appear. */
const SPRING_AT = d.vec3f(...SPRING.at);

const GRID = [DIMS[0] / 2, DIMS[1] / 2, DIMS[2] / 2] as const;
const GRAVITY = 9.8;
/** Rest density in kernel units, and the stiffness of far and near pressure. */
const REST = 5;
const STIFFNESS = 120;
const NEAR = 260;
/** Linear and quadratic viscosity. */
const SIGMA = 0.3;
const BETA = 0.05;
/** Collision radius against the clay, and the tangential friction there. */
const RADIUS = 0.05;
const FRICTION = 0.02;
/** Fluid surface: blob radius and smooth-union blend, in metres. */
const BLOB = 0.11;
const BLEND = 0.12;
/** Largest pressure correction a particle takes in one step, in metres. */
const MAX_SHIFT = CELL * 0.25;
/** Speed limit in metres per second, about a fall from three metres. */
const MAX_SPEED = 8;
/**
 * Particle spacing at rest density: REST counts (1 - r / CELL)² over neighbours, which a uniform
 * density n integrates to n · 4π CELL³ / 30.
 */
const SPACING = Math.cbrt((4 * Math.PI * CELL ** 3) / 30 / REST);
/** Places on the spout, one per spacing² of its disc. */
const SPOUT = Math.round((Math.PI * SPRING.radius ** 2) / SPACING ** 2);
/** Pouring speed in metres per second that moves the stream one spacing per round of the spout. */
const POUR = (SPACING * SPRING.rate) / SPOUT;
const GOLDEN = Math.PI * (3 - Math.sqrt(5));
/**
 * Steps a new particle takes to reach full pressure weight. One that appears inside the water fades
 * in, so it eases its neighbours apart instead of throwing them as droplets in one step.
 */
const RIPEN = 12;

const Particle = d.struct({
  /** Position in metres; w is the pressure weight, which grows from 0 to 1 after spawning. */
  pos: d.vec4f,
  /** Velocity in metres per second; w is 1 alive, -1 dead. */
  vel: d.vec4f,
});

/** Particles the spring adds this frame, in the frame's first step. */
export const Spawn = d.struct({
  count: d.u32,
  first: d.u32,
  /** Particles spawned before this frame, which numbers each new one's place on the spout. */
  total: d.u32,
});

/** The live count and steps so far. */
export const Sim = d.struct({
  alive: d.u32,
  step: d.u32,
});

const WG = grid.WORKGROUP;
/** Cells per tile edge: a tile is one of the grid's blocks and one coarse fluid voxel. */
const T = grid.BLOCK;
const TILES = [GRID[0] / T, GRID[1] / T, GRID[2] / T] as const;
/** Coarse fluid voxel: one per tile. */
const COARSE = CELL * T;
/** Tiles the coarse fluid distance searches in each direction. */
const SEARCH = 3;

const inGrid = (c: d.v3i) => {
  'use gpu';
  return std.all(std.ge(c, d.vec3i())) && c.x < GRID[0] && c.y < GRID[1] && c.z < GRID[2];
};

const cellOf = (p: d.v3f) => {
  'use gpu';
  return d.vec3i(std.floor(p / CELL));
};

const kernelWeight = (r: d.v3f) => {
  'use gpu';
  return std.max(1 - std.length(r) / CELL, 0);
};

/**
 * Water as particles kept in grid order. Each step sorts the particles into `@typegpu/kit/grid`,
 * then solves pressure with double density relaxation (Clavet et al. 2005) over sorted indices,
 * where neighbours sit next to each other in memory. The grid lists the tiles holding water, so no
 * pass touches the whole box.
 */
export function createWater(root: TgpuRoot, clay: Clay, formats: Formats) {
  const spawn = root.createUniform(Spawn);

  /** Particles in slot order, predicted by `integrate`; `relax` writes them back in sorted order. */
  const state = root.createBuffer(d.arrayOf(Particle, POOL)).$usage('storage');
  /** Each particle's far and near pressure as an f16 pair, by sorted index. */
  const pressure = root.createBuffer(d.arrayOf(d.u32, POOL)).$usage('storage');
  const sim = root.createMutable(Sim);

  const stateRW = state.as('mutable');
  const stateIn = state.as('readonly');
  const pressureRW = pressure.as('mutable');
  const pressureIn = pressure.as('readonly');

  // The fluid surface, ray traced like the clay: a fine level per cell, a coarse one per tile
  const { distance } = formats;
  const surface = root
    .createTexture({ size: [...GRID], format: distance, dimension: '3d', mipLevelCount: 3 })
    .$usage('sampled', 'storage');
  const surfaceFine = surface.createView(d.textureStorage3d(distance, 'write-only'), {
    baseMipLevel: 0,
    mipLevelCount: 1,
  });
  const surfaceCoarse = surface.createView(d.textureStorage3d(distance, 'write-only'), {
    baseMipLevel: 2,
    mipLevelCount: 1,
  });
  const surfaceSampled = surface.createView(d.texture3d(d.f32));

  /** Spawns only in the frame's first step; every step reads the same uniform. */
  const end = () => {
    'use gpu';
    let born = d.u32(0);
    if (sim.$.step === spawn.$.first) born = spawn.$.count;
    return std.min(sim.$.alive + born, POOL);
  };

  const cells = grid.create(root, {
    capacity: POOL,
    low: d.vec3f(),
    size: WORLD,
    sight: CELL,
    // The surface is rebuilt in tiles within two of water. Rays only take fine samples in tiles
    // within one of water, and their filtered taps reach one tile further
    margin: 2,
    item: Particle,
    alive: (i) => {
      'use gpu';
      return i < end() && stateIn.$[i].vel.w > 0;
    },
    position: (i) => {
      'use gpu';
      return d.vec3f(stateIn.$[i].pos.xyz);
    },
    itemOf: (i) => {
      'use gpu';
      return Particle(stateIn.$[i]);
    },
  });

  const particleKernel = tgpu.computeFn({
    workgroupSize: [WG],
    in: { gid: d.builtin.globalInvocationId },
  });

  /** Spawns, integrates gravity and predicts positions, in slot order. */
  const integrate = particleKernel(function integrate({ gid }) {
    'use gpu';
    const i = gid.x;
    if (i >= end()) return;
    let p = d.vec3f();
    let v = d.vec3f();
    let weight = d.f32(0);
    if (i < sim.$.alive) {
      const q = Particle(stateRW.$[i]);
      if (q.vel.w < 0) return;
      p = d.vec3f(q.pos.xyz);
      v = d.vec3f(q.vel.xyz);
      weight = std.min(q.pos.w + 1 / RIPEN, 1);
    } else {
      // Places on a sunflower spiral across the spout, at rest spacing, taken in turn. The stream
      // leaves fast enough that a place is clear by the time it comes round again
      const place = d.f32((spawn.$.total + i - sim.$.alive) % SPOUT);
      const out = SPRING.radius * std.sqrt((place + 0.5) / SPOUT);
      const angle = place * GOLDEN;
      randf.seed2(d.vec2f(d.f32(i), d.f32(sim.$.step)));
      p = SPRING_AT + d.vec3f(std.cos(angle) * out, 0, std.sin(angle) * out);
      p += randf.inUnitSphere() * (SPACING * 0.1);
      v = d.vec3f(0, -POUR, 0);
    }

    v.y -= GRAVITY * STEP;
    p += v * STEP;
    const alive = std.select(d.f32(-1), d.f32(1), inGrid(cellOf(p)));
    stateRW.$[i] = Particle({ pos: d.vec4f(p, weight), vel: d.vec4f(v, alive) });
  });

  /** Counts the particles the sort kept alive, and the step. */
  const recount = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    sim.$.alive = cells.count();
    sim.$.step += 1;
  });

  /** Whether tile `t` holds water. None does after a reset, before a step sorts again. */
  const held = (t: d.v3i) => {
    'use gpu';
    return sim.$.alive > 0 && cells.holds(t);
  };

  /** Tiles holding water in and around the surface kernel's tile. */
  const tally = tgpu.workgroupVar(d.atomic(d.u32));

  /** Density and near density from the nine rows of cells around each sorted particle. */
  const density = particleKernel(function density({ gid }) {
    'use gpu';
    const k = gid.x;
    if (k >= cells.count()) return;
    const p = cells.neighbourAt(k).pos.xyz;
    let rho = d.f32(0);
    let near = d.f32(0);
    const from = cells.around(p);
    for (let r = d.u32(0); r < grid.RANGES; r++) {
      const range = cells.range(from, r);
      for (let j = d.u32(0); j < range.count; j++) {
        const other = cells.neighbour(range, j).pos;
        const q = kernelWeight(other.xyz - p);
        rho += q * q * other.w;
        near += q * q * q * other.w;
      }
    }
    pressureRW.$[k] = std.pack2x16float(d.vec2f(STIFFNESS * (rho - REST), NEAR * near));
  });

  /** Relaxes positions apart by pressure, adds viscosity and collides with the clay SDF. */
  const relax = particleKernel(function relax({ gid }) {
    'use gpu';
    const k = gid.x;
    if (k >= cells.count()) return;
    const me = cells.neighbourAt(k);
    const p = me.pos.xyz;
    let v = d.vec3f(me.vel.xyz);
    const mine = std.unpack2x16float(pressureIn.$[k]);
    // One accumulator for both terms. Pressure displaces by dt² / 2; viscosity changes velocity by
    // dt / 2, which moves the particle that much times dt, as in Clavet's prediction step. The
    // velocity update below divides by dt, so it gets both back
    const half2 = STEP * STEP * 0.5;
    let shift = d.vec3f();
    const from = cells.around(p);
    for (let r = d.u32(0); r < grid.RANGES; r++) {
      const range = cells.range(from, r);
      for (let j = d.u32(0); j < range.count; j++) {
        const index = cells.index(range, j);
        if (index === k) continue;
        const other = cells.neighbourAt(index);
        const offset = other.pos.xyz - p;
        const dist = std.length(offset);
        if (dist >= CELL || dist < 1e-5) continue;
        const q = 1 - dist / CELL;
        const dir = offset / dist;
        const theirs = std.unpack2x16float(pressureIn.$[index]);
        const ripe = std.min(me.pos.w, other.pos.w);
        const push = ((mine.x + theirs.x) * q + (mine.y + theirs.y) * q * q) * ripe;
        const u = std.max(std.dot(v - other.vel.xyz, dir), 0);
        shift -= dir * ((push + q * (SIGMA * u + BETA * u * u)) * half2);
      }
    }

    // Clay pressed into the water can still overlap particles; capping the correction and the
    // speed keeps that from flinging droplets across the island
    const moved = std.length(shift);
    if (moved > MAX_SHIFT) shift *= MAX_SHIFT / moved;
    let next = p + shift;
    v += shift / STEP;
    const speed = std.length(v);
    if (speed > MAX_SPEED) v *= MAX_SPEED / speed;

    const reach = clay.sdf(next, 0);
    if (reach < RADIUS) {
      const normal = clay.normalAt(next);
      next += normal * (RADIUS - reach);
      const into = std.dot(v, normal);
      if (into < 0) v -= normal * into;
      v -= (v - normal * std.dot(v, normal)) * FRICTION;
    }

    // Water that falls off the island dies under it
    let alive = d.f32(1);
    if (next.y < 0.45 || std.any(std.lt(next, d.vec3f())) || std.any(std.ge(next, WORLD))) {
      alive = -1;
    }
    stateRW.$[k] = Particle({ pos: d.vec4f(next, me.pos.w), vel: d.vec4f(v, alive) });
  });

  /**
   * The fluid surface at each cell centre of a listed tile: a smooth union of particle blobs.
   * Voxels of unlisted tiles keep stale values, which no ray reads.
   */
  const surfaceKernel = tgpu.computeFn({
    workgroupSize: [WG],
    in: { wid: d.builtin.workgroupId, lid: d.builtin.localInvocationIndex },
  })(({ wid, lid }) => {
    'use gpu';
    const t = d.vec3u(cells.block(wid.x));
    const voxel = t * T + d.vec3u(lid & 3, (lid >>> 2) & 3, lid >>> 4);
    const p = (d.vec3f(voxel) + 0.5) * CELL;
    // Particles outside the 27 cells are at least 1.5 cells from the centre
    let dist = d.f32(1.5 * CELL - BLOB);

    // With no water in this tile or its neighbours, there is nothing to gather. 27 threads each
    // test one neighbouring tile, so the check costs one load a thread
    if (lid < 27) {
      const o = d.vec3i(t) + d.vec3i(d.i32(lid % 3), d.i32((lid / 3) % 3), d.i32(lid / 9)) - 1;
      if (held(o)) std.atomicAdd(tally.$, 1);
    }
    std.workgroupBarrier();
    if (std.atomicLoad(tally.$) === 0) {
      std.textureStore(surfaceFine.$, voxel, d.vec4f(encode(dist, CELL), 0, 0, 1));
      return;
    }

    const from = cells.around(p);
    for (let r = d.u32(0); r < grid.RANGES; r++) {
      const range = cells.range(from, r);
      for (let j = d.u32(0); j < range.count; j++) {
        const blob = std.length(stateIn.$[cells.index(range, j)].pos.xyz - p) - BLOB;
        const h = std.max(BLEND - std.abs(dist - blob), 0) / BLEND;
        dist = std.min(dist, blob) - h * h * BLEND * 0.25;
      }
    }
    std.textureStore(surfaceFine.$, voxel, d.vec4f(encode(dist, CELL), 0, 0, 1));
  });

  /** Tile occupancy for a 4 × 4 × 4 block of tiles and its search margin, staged once. */
  const BLOCK = 4 + 2 * SEARCH;
  const occupied = tgpu.workgroupVar(d.arrayOf(d.u32, BLOCK * BLOCK * BLOCK));

  /**
   * One coarse voxel per tile: the gap from this tile's box to the nearest tile holding water, less
   * a cell for blobs that reach over a tile's edge. Read with textureLoad, not filtered, so the gap
   * holds anywhere inside the tile. The workgroup stages its block's occupancy in shared memory, so
   * the 343-tile search reads no buffers.
   */
  const coarseKernel = tgpu.computeFn({
    workgroupSize: [4, 4, 4],
    in: {
      gid: d.builtin.globalInvocationId,
      wid: d.builtin.workgroupId,
      lid: d.builtin.localInvocationIndex,
      local: d.builtin.localInvocationId,
    },
  })(({ gid, wid, lid, local }) => {
    'use gpu';
    const corner = d.vec3i(wid * 4) - SEARCH;
    for (let i = lid; i < BLOCK * BLOCK * BLOCK; i += WG) {
      const b = d.vec3i(d.i32(i % BLOCK), d.i32((i / BLOCK) % BLOCK), d.i32(i / (BLOCK * BLOCK)));
      occupied.$[i] = std.select(d.u32(0), d.u32(1), held(corner + b));
    }
    std.workgroupBarrier();

    const at = d.vec3i(local) + SEARCH;
    let gap = d.f32(SEARCH * COARSE);
    for (let z = -SEARCH; z <= SEARCH; z++) {
      for (let y = -SEARCH; y <= SEARCH; y++) {
        for (let x = -SEARCH; x <= SEARCH; x++) {
          const b = at + d.vec3i(x, y, z);
          if (occupied.$[(b.z * BLOCK + b.y) * BLOCK + b.x] === 0) continue;
          const apart = std.max(std.abs(d.vec3f(x, y, z)) - 1, d.vec3f()) * COARSE;
          gap = std.min(gap, std.length(apart));
        }
      }
    }
    gap = std.max(gap - CELL, 0);
    std.textureStore(surfaceCoarse.$, gid, d.vec4f(encode(gap, COARSE), 0, 0, 1));
  });

  /** Whether tile `t` or one of its 26 neighbours holds water. */
  const wet = (t: d.v3i) => {
    'use gpu';
    for (let z = -1; z <= 1; z++) {
      for (let y = -1; y <= 1; y++) {
        for (let x = -1; x <= 1; x++) {
          if (held(t + d.vec3i(x, y, z))) return true;
        }
      }
    }
    return false;
  };

  /**
   * `surfaceKernel` for backends where workgroups share nothing: one thread per voxel of the whole
   * box, each testing the tiles around its own. Voxels away from water get the far distance.
   */
  const surfaceEach = tgpu.computeFn({
    workgroupSize: [T, T, T],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const p = (d.vec3f(gid) + 0.5) * CELL;
    let dist = d.f32(1.5 * CELL - BLOB);
    if (wet(d.vec3i(gid / T))) {
      const from = cells.around(p);
      for (let r = d.u32(0); r < grid.RANGES; r++) {
        const range = cells.range(from, r);
        for (let j = d.u32(0); j < range.count; j++) {
          const blob = std.length(stateIn.$[cells.index(range, j)].pos.xyz - p) - BLOB;
          const h = std.max(BLEND - std.abs(dist - blob), 0) / BLEND;
          dist = std.min(dist, blob) - h * h * BLEND * 0.25;
        }
      }
    }
    std.textureStore(surfaceFine.$, gid, d.vec4f(encode(dist, CELL), 0, 0, 1));
  });

  /** `coarseKernel` for backends where workgroups share nothing: each thread searches alone. */
  const coarseEach = tgpu.computeFn({
    workgroupSize: [4, 4, 4],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const at = d.vec3i(gid);
    let gap = d.f32(SEARCH * COARSE);
    for (let z = -SEARCH; z <= SEARCH; z++) {
      for (let y = -SEARCH; y <= SEARCH; y++) {
        for (let x = -SEARCH; x <= SEARCH; x++) {
          if (!held(at + d.vec3i(x, y, z))) continue;
          const apart = std.max(std.abs(d.vec3f(x, y, z)) - 1, d.vec3f()) * COARSE;
          gap = std.min(gap, std.length(apart));
        }
      }
    }
    gap = std.max(gap - CELL, 0);
    std.textureStore(surfaceCoarse.$, gid, d.vec4f(encode(gap, COARSE), 0, 0, 1));
  });

  // Where compute runs on the CPU, the shared tally and staged occupancy keep the surface there;
  // a thread per voxel that reads what it needs runs on the GPU
  const alone = backends.of(root).compute === 'cpu';

  const integrating = root.createComputePipeline({ compute: integrate });
  const recounting = root.createComputePipeline({ compute: recount });
  const densities = root.createComputePipeline({ compute: density });
  const relaxing = root.createComputePipeline({ compute: relax });
  const surfacing = alone
    ? root.createComputePipeline({ compute: surfaceEach })
    : root.createComputePipeline({ compute: surfaceKernel });
  const coarsening = alone
    ? root.createComputePipeline({ compute: coarseEach })
    : root.createComputePipeline({ compute: coarseKernel });

  /** Fluid distance in metres at a fine sample point, filtered. */
  const fluidAt = (p: d.v3f) => {
    'use gpu';
    const value = std.textureSampleLevel(surfaceSampled.$, clay.sampler.$, p / WORLD, 0).x;
    return decode(value, CELL);
  };

  /** The conservative gap to water from anywhere in the tile around `p`. */
  const fluidGapAt = (p: d.v3f) => {
    'use gpu';
    const t = d.vec3i(
      std.clamp(p / COARSE, d.vec3f(), d.vec3f(TILES[0] - 1, TILES[1] - 1, TILES[2] - 1)),
    );
    return decode(std.textureLoad(surfaceSampled.$, t, 2).x, COARSE);
  };

  const fluidNormalAt = tetrahedralNormal(fluidAt, CELL * 0.5);

  const particleGroups = POOL / WG;

  return {
    spawn,
    sim,
    fluidAt,
    fluidGapAt,
    fluidNormalAt,

    /** One simulation step: integrate, sort, density, relax. */
    step(pass: TgpuComputePass) {
      pass.setPipeline(integrating);
      pass.dispatchWorkgroups(particleGroups);
      cells.sort(pass);
      pass.setPipeline(recounting);
      pass.dispatchWorkgroups(1);
      pass.setPipeline(densities);
      cells.dispatch(pass);
      pass.setPipeline(relaxing);
      cells.dispatch(pass);
    },

    /** Rebuilds the fluid surface from the last step's tiles. */
    surface(pass: TgpuComputePass) {
      pass.setPipeline(surfacing);
      if (alone) pass.dispatchWorkgroups(TILES[0], TILES[1], TILES[2]);
      else cells.dispatchBlocks(pass);
      pass.setPipeline(coarsening);
      pass.dispatchWorkgroups(TILES[0] / 4, TILES[1] / 4, TILES[2] / 4);
    },

    /**
     * Drains every particle and builds the grid, so `surface` rebuilds over no water. It skips the
     * recount: the step count must stay at zero, where the spring's first spawn waits for it.
     */
    reset(encoder: TgpuCommandEncoder) {
      sim.buffer.clear(encoder);
      const pass = encoder.beginComputePass();
      cells.sort(pass);
      pass.end();
    },

    initAsync: () =>
      pipelines.compile([
        cells,
        integrating,
        recounting,
        densities,
        relaxing,
        surfacing,
        coarsening,
      ]),
    destroy() {
      for (const made of [state, pressure, sim.buffer, surface, cells]) made.destroy();
    },
  };
}

export type Water = ReturnType<typeof createWater>;
