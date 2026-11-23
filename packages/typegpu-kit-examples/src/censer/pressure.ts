import { pipelines } from '@typegpu/kit';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';
import { dispatch, groups, inside, kernel } from './cells.ts';

type Vec3 = readonly [number, number, number];

/** A cell's solid neighbours as bits: -x, +x, -y, +y, -z, +z. All six mark a solid cell. */
export const ALL_SOLID = 63;
/** The six neighbours of a cell, in the order of their bits in a mask. */
export const STEPS = [
  d.vec3i(-1, 0, 0),
  d.vec3i(1, 0, 0),
  d.vec3i(0, -1, 0),
  d.vec3i(0, 1, 0),
  d.vec3i(0, 0, -1),
  d.vec3i(0, 0, 1),
];

const exact = { texture: d.texture3d(d.f32), sampleType: 'unfilterable-float' } as const;
const solved = d.textureStorage3d('r32float', 'read-write');
const written = d.textureStorage3d('r32float', 'write-only');

/** One grid of the hierarchy: its unknowns, their right-hand side and each cell's solid neighbours. */
const Grid = tgpu.bindGroupLayout({
  value: { storageTexture: solved },
  rhs: exact,
  mask: exact,
});
/** The next coarser grid, as the finer one hands it the residual and takes back its correction. */
const Coarser = tgpu.bindGroupLayout({
  value: { storageTexture: solved },
  rhs: { storageTexture: written },
});
/** A coarser grid's solid cells, which the coarsening marks in its right-hand side, and its mask. */
const Marking = tgpu.bindGroupLayout({
  solid: exact,
  mask: { storageTexture: written },
});

const maskAt = (cell: d.v3i) => {
  'use gpu';
  return d.u32(std.textureLoad(Grid.$.mask, cell, 0).x);
};

/**
 * One V-cycle over four grids, each half the last along every axis. Red and black sweep once on
 * the finest grid each way, the dearest, twice on the others and 8 times on the coarsest. Each
 * sweep moves a value 1.2 times as far as Gauss–Seidel's.
 */
const LEVELS = 4;
const FINEST = 1;
const SMOOTHING = 2;
const COARSEST = 8;
const RELAXATION = 1.2;

/**
 * Solves the pressure's Poisson equation on a box of cells by geometric multigrid: a V-cycle of
 * red–black Gauss–Seidel sweeps over grids halved down to a coarse one, so a push reaches across
 * the whole box in a step. The equation is the 7-point Laplacian in cells: the sum of a cell's open
 * neighbours less their count times its own value equals the right-hand side. A solid neighbour
 * drops out of both, holding the pressure's gradient across it at zero, and a neighbour outside the
 * box holds 0, so air leaves freely. Each solve starts from the last.
 */
export function createPressure(root: TgpuRoot, cells: Vec3) {
  const sizes: Vec3[] = [cells];
  for (let k = 1; k < LEVELS; k++) {
    const [x, y, z] = sizes[k - 1];
    sizes.push([Math.ceil(x / 2), Math.ceil(y / 2), Math.ceil(z / 2)]);
  }
  const scalar = (size: Vec3, name: string) =>
    root
      .createTexture({ size: [...size], format: 'r32float', dimension: '3d' })
      .$usage('storage', 'sampled')
      .$name(name);
  const grids = sizes.map((size, k) => ({
    size,
    value: scalar(size, `pressure ${k}`),
    rhs: scalar(size, `pressure rhs ${k}`),
    mask: scalar(size, `pressure mask ${k}`),
  }));

  /**
   * The pressure is 0 half a finest cell past the box on every grid, so a cell just beyond a grid
   * holds its neighbour inside's value times this, on the line through that 0.
   */
  const beyondOf = (size: d.v3i) => {
    'use gpu';
    // A cell of this grid spans `span` finest ones
    const span = d.f32(cells[0]) / d.f32(size.x);
    return (1 - span) / (1 + span);
  };
  /**
   * The sum of a cell's open neighbours' values in the box, and the weight of its own value: one
   * per open neighbour, less what each beyond the box holds of it.
   */
  const neighbours = (cell: d.v3i, mask: number, size: d.v3i) => {
    'use gpu';
    let sum = d.f32(0);
    let beyond = d.f32(0);
    for (const k of tgpu.unroll([0, 1, 2, 3, 4, 5])) {
      const next = cell + STEPS[k];
      if ((mask & d.u32(1 << k)) === 0) {
        if (inside(next, size)) sum += std.textureLoad(Grid.$.value, next).x;
        else beyond += 1;
      }
    }
    return d.vec2f(sum, d.f32(6 - std.countOneBits(mask)) - beyond * beyondOf(size));
  };
  /** What the cell's value leaves of its right-hand side. */
  const residualAt = (cell: d.v3i, size: d.v3i) => {
    'use gpu';
    const mask = maskAt(cell);
    if (mask === ALL_SOLID) return d.f32(0);
    const around = neighbours(cell, mask, size);
    const own = std.textureLoad(Grid.$.value, cell).x;
    return std.textureLoad(Grid.$.rhs, cell, 0).x - (around.x - around.y * own);
  };

  // Red and black cells alternate, so each half reads only the other half's latest values
  const sweep = (parity: number) =>
    kernel(({ id }) => {
      'use gpu';
      const size = d.vec3i(std.textureDimensions(Grid.$.value));
      const cell = d.vec3i(
        d.i32(id.x * 2 + ((id.y + id.z + parity) & 1)),
        d.i32(id.y),
        d.i32(id.z),
      );
      if (!inside(cell, size)) return;
      const mask = maskAt(cell);
      if (mask === ALL_SOLID) {
        std.textureStore(Grid.$.value, cell, d.vec4f());
        return;
      }
      const around = neighbours(cell, mask, size);
      const rhs = std.textureLoad(Grid.$.rhs, cell, 0).x;
      const was = std.textureLoad(Grid.$.value, cell).x;
      const next = std.mix(was, (around.x - rhs) / around.y, RELAXATION);
      std.textureStore(Grid.$.value, cell, d.vec4f(next, 0, 0, 0));
    });

  /**
   * Hands the coarser grid the residual its eight cells leave, scaled to its doubled spacing, and
   * clears its correction.
   */
  const restrict = kernel(({ id }) => {
    'use gpu';
    const coarse = d.vec3i(id);
    if (!inside(coarse, d.vec3i(std.textureDimensions(Coarser.$.value)))) return;
    const size = d.vec3i(std.textureDimensions(Grid.$.value));
    let total = d.f32(0);
    for (const z of tgpu.unroll([0, 1])) {
      for (const y of tgpu.unroll([0, 1])) {
        for (const x of tgpu.unroll([0, 1])) {
          const fine = coarse * 2 + d.vec3i(x, y, z);
          if (inside(fine, size)) total += residualAt(fine, size);
        }
      }
    }
    // The average of eight, times four for a Laplacian over twice the spacing
    std.textureStore(Coarser.$.rhs, coarse, d.vec4f(total * 0.5, 0, 0, 0));
    std.textureStore(Coarser.$.value, coarse, d.vec4f());
  });

  /**
   * Adds the coarser grid's correction to each cell, interpolated between its eight nearest, with
   * those beyond the box continuing the line to its 0.
   */
  const prolong = kernel(({ id }) => {
    'use gpu';
    const fine = d.vec3i(id);
    if (!inside(fine, d.vec3i(std.textureDimensions(Grid.$.value)))) return;
    if (maskAt(fine) === ALL_SOLID) return;
    const coarse = d.vec3i(std.textureDimensions(Coarser.$.value));
    const last = coarse - 1;
    const beyond = beyondOf(coarse);
    const at = d.vec3f(fine) * 0.5 - 0.25;
    const base = d.vec3i(std.floor(at));
    const t = at - d.vec3f(base);
    let correction = d.f32(0);
    for (const z of tgpu.unroll([0, 1])) {
      for (const y of tgpu.unroll([0, 1])) {
        for (const x of tgpu.unroll([0, 1])) {
          const wanted = base + d.vec3i(x, y, z);
          const corner = std.clamp(wanted, d.vec3i(), last);
          const reach = std.select(d.vec3f(1), d.vec3f(beyond), std.ne(wanted, corner));
          const weight = std.mix(1 - t, t, d.vec3f(x, y, z)) * reach;
          correction += std.textureLoad(Coarser.$.value, corner).x * weight.x * weight.y * weight.z;
        }
      }
    }
    const own = std.textureLoad(Grid.$.value, fine).x;
    std.textureStore(Grid.$.value, fine, d.vec4f(own + correction, 0, 0, 0));
  });

  /** Marks a coarser cell solid when all its finer cells are, in its right-hand side for now. */
  const coarsen = kernel(({ id }) => {
    'use gpu';
    const coarse = d.vec3i(id);
    if (!inside(coarse, d.vec3i(std.textureDimensions(Coarser.$.value)))) return;
    const size = d.vec3i(std.textureDimensions(Grid.$.value));
    let solid = true;
    for (const z of tgpu.unroll([0, 1])) {
      for (const y of tgpu.unroll([0, 1])) {
        for (const x of tgpu.unroll([0, 1])) {
          const fine = coarse * 2 + d.vec3i(x, y, z);
          if (inside(fine, size) && maskAt(fine) !== ALL_SOLID) solid = false;
        }
      }
    }
    std.textureStore(Coarser.$.rhs, coarse, d.vec4f(std.select(0, 1, solid), 0, 0, 0));
  });

  /** Gathers each coarser cell's solid neighbours from the marks into its mask. */
  const mark = kernel(({ id }) => {
    'use gpu';
    const cell = d.vec3i(id);
    const size = d.vec3i(std.textureDimensions(Marking.$.solid));
    if (!inside(cell, size)) return;
    let mask = d.u32(ALL_SOLID);
    if (std.textureLoad(Marking.$.solid, cell, 0).x < 0.5) {
      mask = 0;
      for (const k of tgpu.unroll([0, 1, 2, 3, 4, 5])) {
        const next = cell + STEPS[k];
        if (inside(next, size) && std.textureLoad(Marking.$.solid, next, 0).x > 0.5) {
          mask |= d.u32(1 << k);
        }
      }
    }
    std.textureStore(Marking.$.mask, cell, d.vec4f(d.f32(mask), 0, 0, 0));
  });

  const kernels = {
    red: root.createComputePipeline({ compute: sweep(0) }),
    black: root.createComputePipeline({ compute: sweep(1) }),
    restrict: root.createComputePipeline({ compute: restrict }),
    prolong: root.createComputePipeline({ compute: prolong }),
    coarsen: root.createComputePipeline({ compute: coarsen }),
    mark: root.createComputePipeline({ compute: mark }),
  };

  const read = d.texture3d(d.f32);
  const bound = grids.map(({ value, rhs, mask }, k) => {
    const coarser = grids[k + 1];
    return {
      grid: root.createBindGroup(Grid, {
        value: value.createView(solved),
        rhs: rhs.createView(read, { sampleType: 'unfilterable-float' }),
        mask: mask.createView(read, { sampleType: 'unfilterable-float' }),
      }),
      coarser:
        coarser &&
        root.createBindGroup(Coarser, {
          value: coarser.value.createView(solved),
          rhs: coarser.rhs.createView(written),
        }),
      marking:
        coarser &&
        root.createBindGroup(Marking, {
          solid: coarser.rhs.createView(read, { sampleType: 'unfilterable-float' }),
          mask: coarser.mask.createView(written),
        }),
    };
  });
  const whole = sizes.map((size) => groups(size));
  const half = sizes.map((size) => groups(size, true));

  const smooth = (pass: TgpuComputePass, k: number, count: number) => {
    pass.setBindGroup(Grid, bound[k].grid);
    for (let i = 0; i < count; i++) {
      pass.setPipeline(kernels.red);
      dispatch(pass, half[k]);
      pass.setPipeline(kernels.black);
      dispatch(pass, half[k]);
    }
  };
  /** Runs a pipeline that reads grid `k` and writes the next coarser one, over grid `over`. */
  const between = (pass: TgpuComputePass, name: keyof typeof kernels, k: number, over: number) => {
    pass.setPipeline(kernels[name]);
    pass.setBindGroup(Grid, bound[k].grid);
    pass.setBindGroup(Coarser, bound[k].coarser!);
    dispatch(pass, whole[over]);
  };

  return {
    /** The finest grid's pressure, which each solve refines, as `read-write` r32float. */
    pressure: grids[0].value,
    /** The finest grid's right-hand side, which the caller writes before each solve. */
    rhs: grids[0].rhs,
    /** The finest grid's cells' solid neighbours as bits, which the caller writes before each solve. */
    mask: grids[0].mask,
    initAsync: () => pipelines.compile(Object.values(kernels)),
    /** Records a solve into a compute pass, after the right-hand side and mask are written. */
    solve(pass: TgpuComputePass) {
      for (let k = 0; k + 1 < LEVELS; k++) {
        between(pass, 'coarsen', k, k + 1);
        pass.setPipeline(kernels.mark);
        pass.setBindGroup(Marking, bound[k].marking!);
        dispatch(pass, whole[k + 1]);
      }
      for (let k = 0; k + 1 < LEVELS; k++) {
        smooth(pass, k, k === 0 ? FINEST : SMOOTHING);
        between(pass, 'restrict', k, k + 1);
      }
      smooth(pass, LEVELS - 1, COARSEST);
      for (let k = LEVELS - 2; k >= 0; k--) {
        between(pass, 'prolong', k, k);
        smooth(pass, k, k === 0 ? FINEST : SMOOTHING);
      }
    },
    destroy() {
      for (const each of grids) {
        each.value.destroy();
        each.rhs.destroy();
        each.mask.destroy();
      }
    },
  };
}
