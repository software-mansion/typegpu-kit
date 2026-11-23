import { pipelines } from '@typegpu/kit';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';
import { dispatch, groups, inside, kernel, volume, WORKGROUP } from './cells.ts';
import { ALL_SOLID, createPressure, STEPS } from './pressure.ts';

type Vec3 = readonly [number, number, number];

/** How fast the sponge at the box's faces takes smoke and motion away, per second, at the face. */
const SPONGE_RATE = 6;

/** How the smoke moves. Accelerations are in metres per second squared, decays per second. */
const SmokeSettings = d.struct({
  /** The step's length in seconds. */
  dt: d.f32,
  /** Upward acceleration per unit of heat. */
  buoyancy: d.f32,
  /** Downward acceleration per unit of smoke. */
  weight: d.f32,
  /** How strongly vorticity confinement feeds back the swirls the grid smooths away. */
  vorticity: d.f32,
  smokeDecay: d.f32,
  heatDecay: d.f32,
  drag: d.f32,
});

/** Smoke and heat fainter than this are dropped as they move: too faint to show or lift the air. */
const FAINT = 1e-4;
/** The most steps a path is traced back in, one per cell it crosses. */
const MOST_TRACE = 8;
/**
 * How many times finer than the air's cells the smoke and heat are carried, along each axis, so
 * they keep edges sharper than the motion that carries them. Each workgroup covers whole cells of
 * the air's grid, which it averages its smoke into.
 */
const DETAIL = 2;

/** A thread's place in its workgroup's shared array. */
const slotOf = (local: d.v3u) => {
  'use gpu';
  return local.x + WORKGROUP[0] * (local.y + WORKGROUP[1] * local.z);
};

/** A cell of a field, or the nearest one inside it. */
const load = (texture: d.texture3d<d.F32>, cell: d.v3i) => {
  'use gpu';
  const edge = d.vec3i(std.textureDimensions(texture)) - 1;
  return std.textureLoad(texture, std.clamp(cell, d.vec3i(), edge), 0);
};

/** Whether the neighbour a bit of the mask names is solid. */
const isSolid = (mask: number, bit: number) => {
  'use gpu';
  return (mask & d.u32(bit)) !== 0;
};

interface SmokeOptions {
  /** Cells along x, y and z, each a multiple of 4. */
  readonly cells: Vec3;
  /** The box's lowest corner, in metres. */
  readonly origin: Vec3;
  /** A cell's edge in metres. */
  readonly cell: number;
  /**
   * What fills a point, a `'use gpu'` function of its position: the velocity of the solid there in
   * `xyz`, and 1 in `w` inside a solid, 0 in the air.
   */
  readonly solid: (position: d.v3f) => d.v4f;
  /** Smoke in `x` and heat in `y` added per second at a point, a `'use gpu'` function of it. */
  readonly source: (position: d.v3f) => d.v2f;
  /** The acceleration a point's air takes, in metres per second squared, such as a jet's. */
  readonly force: (position: d.v3f) => d.v3f;
  /**
   * The velocity a point's air is carried toward in `xyz`, and how fast in `w`, per second, such as
   * air a moving solid drags along beside it.
   */
  readonly carry: (position: d.v3f) => d.v4f;
  /**
   * Cells inside each face over which a sponge takes smoke, heat and motion away, more strongly
   * toward the face, so nothing reaches the walls.
   */
  readonly sponge: number;
  readonly settings: d.InferInput<typeof SmokeSettings>;
}

/**
 * A box of smoke on a grid of cells: the air's velocity, advected by MacCormack's method, lifted by
 * heat, kept swirling by vorticity confinement and made incompressible by a multigrid pressure
 * solve, carrying smoke and heat on a grid twice as fine. Solids slip freely and push the air
 * with their velocity. The box lets smoke out of every face, through a sponge that takes it away
 * before it reaches one.
 */
export function createSmoke(root: TgpuRoot, options: SmokeOptions) {
  const [nx, ny, nz] = options.cells;
  const { cell: h, solid, source, force, carry, sponge } = options;
  const [fx, fy, fz] = [nx * DETAIL, ny * DETAIL, nz * DETAIL];
  const origin = d.vec3f(...options.origin);
  const size = d.vec3f(nx, ny, nz);
  const box = d.vec3i(nx, ny, nz);
  const last = d.vec3i(nx - 1, ny - 1, nz - 1);
  const settings = root.createUniform(SmokeSettings, options.settings).$name('smoke settings');

  // Smoke and heat are a pair, which `texture-formats-tier1` stores in half the bytes
  const pair = root.device.features.has('texture-formats-tier1') ? 'rg16float' : 'rgba16float';
  const field = (
    name: string,
    format: 'rgba16float' | 'rg16float' = 'rgba16float',
    cells: Vec3 = options.cells,
  ) => volume(root, cells, format, `smoke ${name}`);
  // The velocity holds each step's result and the advected one holds it before the projection, with
  // 1 in `w` in solid cells
  const velocity = field('velocity');
  const advected = field('advected');
  const solids = field('solids');
  const curls = field('curls');
  const averaged = field('averaged', pair);
  const solver = createPressure(root, options.cells);
  const written = d.textureStorage3d('r32float', 'write-only');
  const spread = solver.rhs.createView(written);
  const masks = {
    write: solver.mask.createView(written),
    read: solver.mask.createView(d.texture3d(d.f32), { sampleType: 'unfilterable-float' }),
  };
  const pressure = solver.pressure.createView(d.textureStorage3d('r32float', 'read-write'));
  const smooth = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });

  // Smoke in x and heat in y, on the finer grid: each step carries one of the pair into the other
  const matter = [
    field('matter a', pair, [fx, fy, fz]),
    field('matter b', pair, [fx, fy, fz]),
  ] as const;
  const matterLayout = tgpu.bindGroupLayout({ matter: { texture: d.texture3d(d.f32) } });
  const stepLayout = tgpu.bindGroupLayout({
    from: { texture: d.texture3d(d.f32) },
    to: { storageTexture: d.textureStorage3d(pair, 'write-only') },
  });
  const reading = matter.map(({ texture }) =>
    root.createBindGroup(matterLayout, { matter: texture }),
  );
  const stepping = [0, 1].map((i) =>
    root.createBindGroup(stepLayout, { from: matter[i].texture, to: matter[1 - i].texture }),
  );
  let current = 0;

  /** Samples a field at a point in cells, with cell centres at half steps. */
  const sampled = (texture: d.texture3d<d.F32>, at: d.v3f) => {
    'use gpu';
    return std.textureSampleLevel(texture, smooth.$, at / size, 0);
  };
  const positionOf = (cell: d.v3i) => {
    'use gpu';
    return origin + (d.vec3f(cell) + 0.5) * h;
  };

  /**
   * Where the air at a point in cells was `dt` seconds before, or will be with a negative `dt`,
   * traced in steps of at most a cell so fast air follows its curve rather than cutting across it.
   */
  const trace = (at: d.v3f, dt: number) => {
    'use gpu';
    const scale = dt / h;
    const first = sampled(velocity.read.$, at).xyz * scale;
    const steps = std.clamp(std.ceil(std.length(first)), 1, MOST_TRACE);
    let point = at - first / steps;
    for (let i = d.i32(1); i < d.i32(steps); i++) {
      point = point - (sampled(velocity.read.$, point).xyz * scale) / steps;
    }
    return point;
  };

  /**
   * Advects a field from where the air came from, corrects it by tracing that forwards and back
   * again, and clamps it to the cells the path started between, so the correction adds no new
   * extremes. `predicted` is the field where the air came from. Points are in the air's cells; the
   * field's cells are `scale` times finer.
   */
  const macCormackOf = (scale: number) => {
    return (
      texture: d.texture3d<d.F32>,
      predicted: d.v4f,
      at: d.v3f,
      back: d.v3f,
      restored: d.v3f,
    ) => {
      'use gpu';
      const corrected = predicted + (sampled(texture, at) - sampled(texture, restored)) * 0.5;
      const base = d.vec3i(std.floor(back * scale - 0.5));
      let low = load(texture, base);
      let high = d.vec4f(low);
      for (const z of tgpu.unroll([0, 1])) {
        for (const y of tgpu.unroll([0, 1])) {
          for (const x of tgpu.unroll([0, 1])) {
            const corner = load(texture, base + d.vec3i(x, y, z));
            low = std.min(low, corner);
            high = std.max(high, corner);
          }
        }
      }
      return std.clamp(corrected, low, high);
    };
  };
  const macCormack = macCormackOf(1);
  const macCormackFine = macCormackOf(DETAIL);

  /** What the sponge leaves of anything at a point in cells over a step. */
  const spongeAt = (at: d.v3f) => {
    'use gpu';
    const wall = std.min(
      std.min(at.x, at.y),
      std.min(at.z, std.min(std.min(size.x - at.x, size.y - at.y), size.z - at.z)),
    );
    const deep = 1 - std.saturate(wall / sponge);
    return std.exp(-SPONGE_RATE * settings.$.dt * deep * deep);
  };

  const findSolids = kernel(({ id }) => {
    'use gpu';
    if (id.x >= nx || id.y >= ny || id.z >= nz) return;
    std.textureStore(solids.write.$, id, solid(positionOf(d.vec3i(id))));
  });

  const findCurl = kernel(({ id }) => {
    'use gpu';
    if (id.x >= nx || id.y >= ny || id.z >= nz) return;
    const cell = d.vec3i(id);
    const dx =
      load(velocity.read.$, cell + d.vec3i(1, 0, 0)) -
      load(velocity.read.$, cell - d.vec3i(1, 0, 0));
    const dy =
      load(velocity.read.$, cell + d.vec3i(0, 1, 0)) -
      load(velocity.read.$, cell - d.vec3i(0, 1, 0));
    const dz =
      load(velocity.read.$, cell + d.vec3i(0, 0, 1)) -
      load(velocity.read.$, cell - d.vec3i(0, 0, 1));
    const curl = d.vec3f(dy.z - dz.y, dz.x - dx.z, dx.y - dy.x) * (0.5 / h);
    std.textureStore(curls.write.$, id, d.vec4f(curl, std.length(curl)));
  });

  const advect = kernel(({ id }) => {
    'use gpu';
    if (id.x >= nx || id.y >= ny || id.z >= nz) return;
    const cell = d.vec3i(id);
    const here = std.textureLoad(solids.read.$, cell, 0);
    if (here.w > 0.5) {
      std.textureStore(advected.write.$, id, here);
      return;
    }
    const dt = settings.$.dt;
    const at = d.vec3f(cell) + 0.5;
    const back = trace(at, dt);
    const restored = trace(trace(at, -dt), dt);
    let moved = macCormack(velocity.read.$, sampled(velocity.read.$, back), at, back, restored).xyz;
    const held = sampled(matterLayout.$.matter, at).xy;

    // Vorticity confinement turns the air around the strongest swirls nearby
    const swirl = std.textureLoad(curls.read.$, cell, 0).xyz;
    const rise = d.vec3f(
      load(curls.read.$, cell + d.vec3i(1, 0, 0)).w - load(curls.read.$, cell - d.vec3i(1, 0, 0)).w,
      load(curls.read.$, cell + d.vec3i(0, 1, 0)).w - load(curls.read.$, cell - d.vec3i(0, 1, 0)).w,
      load(curls.read.$, cell + d.vec3i(0, 0, 1)).w - load(curls.read.$, cell - d.vec3i(0, 0, 1)).w,
    );
    const toward = rise / (std.length(rise) + 1e-5);
    const confinement = std.cross(toward, swirl) * (settings.$.vorticity * h);
    const position = positionOf(cell);
    const lift = d.vec3f(0, settings.$.buoyancy * held.y - settings.$.weight * held.x, 0);
    const pushed = confinement + lift + force(position);
    moved = (moved + pushed * dt) * std.exp(-settings.$.drag * dt);
    const dragged = carry(position);
    moved = std.mix(moved, dragged.xyz, 1 - std.exp(-dragged.w * dt));

    std.textureStore(advected.write.$, id, d.vec4f(moved * spongeAt(at), 0));
  });

  const carried = tgpu.workgroupVar(d.arrayOf(d.vec2f, WORKGROUP[0] * WORKGROUP[1] * WORKGROUP[2]));

  /**
   * Smoke and heat, carried on the finer grid by the air's velocity at the start of the step, and
   * averaged over each of the air's cells.
   */
  const advectMatter = tgpu.computeFn({
    workgroupSize: [...WORKGROUP],
    in: { id: d.builtin.globalInvocationId, local: d.builtin.localInvocationId },
  })(({ id, local }) => {
    'use gpu';
    const inBox = id.x < fx && id.y < fy && id.z < fz;
    const at = (d.vec3f(id) + 0.5) / DETAIL;
    let stored = d.vec4f();
    if (inBox && std.textureLoad(solids.read.$, d.vec3i(at), 0).w <= 0.5) {
      const dt = settings.$.dt;
      const back = trace(at, dt);
      const predicted = sampled(stepLayout.$.from, back);
      const added = source(origin + at * h) * dt;
      // Where the air brings in only traces, they are dropped rather than corrected
      if (std.max(predicted.x, predicted.y) < FAINT) {
        stored = d.vec4f(added * spongeAt(at), 0, 0);
      } else {
        const restored = trace(trace(at, -dt), dt);
        let moved = macCormackFine(stepLayout.$.from, predicted, at, back, restored).xy;
        moved = moved + added;
        moved =
          moved *
          d.vec2f(std.exp(-settings.$.smokeDecay * dt), std.exp(-settings.$.heatDecay * dt));
        stored = d.vec4f(std.max(moved * spongeAt(at), d.vec2f()), 0, 0);
      }
    }
    if (inBox) std.textureStore(stepLayout.$.to, id, stored);
    // As the texture holds it, in half precision
    carried.$[slotOf(local)] = std.quantizeToF16(stored.xy);
    std.workgroupBarrier();
    const corner = d.vec3u(
      std.intdiv(id.x, DETAIL),
      std.intdiv(id.y, DETAIL),
      std.intdiv(id.z, DETAIL),
    );
    const first = local.x % DETAIL === 0 && local.y % DETAIL === 0 && local.z % DETAIL === 0;
    if (first && corner.x < nx && corner.y < ny && corner.z < nz) {
      let sum = d.vec2f();
      for (const z of tgpu.unroll([0, 1])) {
        for (const y of tgpu.unroll([0, 1])) {
          for (const x of tgpu.unroll([0, 1])) sum += carried.$[slotOf(local + d.vec3u(x, y, z))];
        }
      }
      std.textureStore(averaged.write.$, corner, d.vec4f(sum / DETAIL ** 3, 0, 0));
    }
  });

  /** A neighbour's velocity, and whether it is solid. Outside the box is open. */
  const neighbour = (cell: d.v3i, offset: d.v3i, own: d.v3f) => {
    'use gpu';
    const next = cell + offset;
    if (!inside(next, box)) return d.vec4f(own, 0);
    return std.textureLoad(advected.read.$, next, 0);
  };

  const divergence = kernel(({ id }) => {
    'use gpu';
    if (id.x >= nx || id.y >= ny || id.z >= nz) return;
    const cell = d.vec3i(id);
    const here = std.textureLoad(advected.read.$, cell, 0);
    if (here.w > 0.5) {
      std.textureStore(spread.$, id, d.vec4f());
      std.textureStore(masks.write.$, id, d.vec4f(ALL_SOLID, 0, 0, 0));
      return;
    }
    const own = here.xyz;
    const left = neighbour(cell, d.vec3i(-1, 0, 0), own);
    const right = neighbour(cell, d.vec3i(1, 0, 0), own);
    const below = neighbour(cell, d.vec3i(0, -1, 0), own);
    const above = neighbour(cell, d.vec3i(0, 1, 0), own);
    const back = neighbour(cell, d.vec3i(0, 0, -1), own);
    const front = neighbour(cell, d.vec3i(0, 0, 1), own);
    const outflow = (right.x - left.x + above.y - below.y + front.z - back.z) * 0.5;
    const mask = left.w + right.w * 2 + below.w * 4 + above.w * 8 + back.w * 16 + front.w * 32;
    std.textureStore(spread.$, id, d.vec4f(outflow, 0, 0, 0));
    std.textureStore(masks.write.$, id, d.vec4f(mask, 0, 0, 0));
  });

  /** Pressure at a neighbour, 0 outside the box, or the cell's own where the neighbour is solid. */
  const pressureNear = (cell: d.v3i, offset: d.v3i, solidNeighbour: boolean, own: number) => {
    'use gpu';
    if (solidNeighbour) return own;
    const next = cell + offset;
    if (!inside(next, box)) return d.f32(0);
    return std.textureLoad(pressure.$, next).x;
  };

  /** The velocity of the solid at a neighbour. */
  const wall = (cell: d.v3i, offset: d.v3i) => {
    'use gpu';
    return std.textureLoad(solids.read.$, std.clamp(cell + offset, d.vec3i(), last), 0).xyz;
  };

  const project = kernel(({ id }) => {
    'use gpu';
    if (id.x >= nx || id.y >= ny || id.z >= nz) return;
    const cell = d.vec3i(id);
    const mask = d.u32(std.textureLoad(masks.read.$, cell, 0).x);
    if (mask === ALL_SOLID) {
      std.textureStore(
        velocity.write.$,
        id,
        d.vec4f(std.textureLoad(advected.read.$, cell, 0).xyz, 0),
      );
      return;
    }
    const own = std.textureLoad(pressure.$, cell).x;
    const moved = std.textureLoad(advected.read.$, cell, 0).xyz;
    // Air beside a solid moves with it across the face between them and slips along it
    for (const axis of tgpu.unroll([0, 1, 2])) {
      const before = isSolid(mask, 1 << (axis * 2));
      const after = isSolid(mask, 2 << (axis * 2));
      const down = STEPS[axis * 2];
      const up = STEPS[axis * 2 + 1];
      const near = pressureNear(cell, up, after, own) - pressureNear(cell, down, before, own);
      moved[axis] = moved[axis] - near * 0.5;
      if (before) moved[axis] = wall(cell, down)[axis];
      if (after) moved[axis] = wall(cell, up)[axis];
    }
    std.textureStore(velocity.write.$, id, d.vec4f(moved, 0));
  });

  const kernels = {
    solids: root.createComputePipeline({ compute: findSolids }),
    curl: root.createComputePipeline({ compute: findCurl }),
    advect: root.createComputePipeline({ compute: advect }),
    matter: root.createComputePipeline({ compute: advectMatter }),
    divergence: root.createComputePipeline({ compute: divergence }),
    project: root.createComputePipeline({ compute: project }),
  };
  const cellGroups = groups(options.cells);
  const fineGroups = groups([fx, fy, fz]);
  const advecting = reading.map((group) => kernels.advect.with(group));
  const carrying = stepping.map((group) => kernels.matter.with(group));
  const low: Vec3 = options.origin;
  const high: Vec3 = [low[0] + nx * h, low[1] + ny * h, low[2] + nz * h];
  const extent = d.vec3f(nx * h, ny * h, nz * h);

  return {
    /** Cells along x, y and z, and a cell's edge in metres. */
    cells: options.cells,
    cell: h,
    /** The box's lowest and highest corners, in metres. */
    low,
    high,
    /** Binds the smoke as it stands for pipelines that read `smokeAt`: changes with each step. */
    get matter() {
      return reading[current];
    },
    /** The smoke and heat at a point in metres, 0 outside the box, read where `matter` is bound. */
    smokeAt: (position: d.v3f) => {
      'use gpu';
      const at = (position - origin) / extent;
      if (std.any(std.lt(at, d.vec3f())) || std.any(std.gt(at, d.vec3f(1)))) return d.vec2f();
      return std.textureSampleLevel(matterLayout.$.matter, smooth.$, at, 0).xy;
    },
    /** The smoke and heat averaged over each of the air's cells, as the last step left them. */
    averaged: averaged.read,
    initAsync: () => pipelines.compile([...Object.values(kernels), solver]),
    /**
     * Records the first half of a step into a compute pass: finds the solids, then moves the air
     * and the smoke and measures how much the moved air spreads. `matter` changes here.
     */
    advect(pass: TgpuComputePass) {
      pass.setPipeline(kernels.solids);
      dispatch(pass, cellGroups);
      pass.setPipeline(kernels.curl);
      dispatch(pass, cellGroups);
      pass.setPipeline(advecting[current]);
      dispatch(pass, cellGroups);
      pass.setPipeline(carrying[current]);
      dispatch(pass, fineGroups);
      current = 1 - current;
      pass.setPipeline(kernels.divergence);
      dispatch(pass, cellGroups);
    },
    /**
     * Records the second half of a step, after `advect`: solves the pressure and makes the air
     * incompressible. The two halves cost about the same, so a program may record them in
     * different frames.
     */
    project(pass: TgpuComputePass) {
      solver.solve(pass);
      pass.setPipeline(kernels.project);
      dispatch(pass, cellGroups);
    },
    destroy() {
      for (const each of [velocity, advected, solids, curls, averaged, ...matter])
        each.texture.destroy();
      solver.destroy();
    },
  };
}

export type Smoke = ReturnType<typeof createSmoke>;
