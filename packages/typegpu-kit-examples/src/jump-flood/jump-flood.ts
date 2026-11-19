import { pipelines, type SizeSource } from '@typegpu/kit';
import { randf } from '@typegpu/noise';
import {
  common,
  d,
  std,
  tgpu,
  type TgpuComputePass,
  type TgpuRenderCommands,
  type TgpuRoot,
  type TgpuUniform,
} from 'typegpu';

const WORKGROUP = 8;
const SEED_WORKGROUP = 64;
const TAU = Math.PI * 2;

/**
 * Seed circles have centres anywhere on the grid, radii in `CIRCLES` and speeds up to `DRIFT`
 * radians per second in either direction. Radii use fractions of grid size.
 */
const CIRCLES = [0.02, 0.1];
const DRIFT = 0.12;

/** Right, down and both diagonals below, which meet each touching pair of cells once. */
const TOUCHING = tgpu.const(d.arrayOf(d.vec2i, 4), [
  d.vec2i(1, 0),
  d.vec2i(-1, 1),
  d.vec2i(0, 1),
  d.vec2i(1, 1),
]);

/** Sentinel for an unreached cell. */
export const NONE = 0xffffffff;

/** Dispatch step distance and input-buffer half. */
const Pass = d.struct({ step: d.u32, source: d.u32 });

const Seed = d.struct({ position: d.vec2f });

/** What the draw knows of a pixel. Distances and positions are in cells. */
export const Cell = d.struct({
  /** The pixel's position on the grid. */
  point: d.vec2f,
  /** The nearest seed's index, or `NONE` where no seed has reached yet. */
  seed: d.u32,
  /** The nearest seed's position on the grid. */
  site: d.vec2f,
  /** Distance to the nearest seed. */
  toSite: d.f32,
  /** The second-nearest seed's index, or `NONE`. */
  second: d.u32,
  /** Distance to the second-nearest seed. */
  toSecond: d.f32,
  /** Distance to the nearest edge of the seed's region. */
  toEdge: d.f32,
  /** Cells one pixel spans, for anti-aliasing. */
  footprint: d.f32,
});

/** Colours a pixel of the diagram. Runs on the GPU. */
export type Paint = (cell: d.Infer<typeof Cell>) => d.v3f;

const View = d.struct({ viewport: d.vec2f, result: d.u32 });

export interface JumpFloodOptions {
  /** Cells along each side of the square grid. A power of two. */
  readonly size: number;
  readonly seeds: number;
  readonly format: GPUTextureFormat;
  /** Frame-clock time in seconds for seed drift. */
  readonly time: TgpuUniform<d.F32>;
  /** What the diagram is drawn into, such as the canvas surface. */
  readonly target: SizeSource;
  readonly paint: Paint;
  /** Uses immediates for dispatch values. Defaults to device support. */
  readonly immediates?: boolean | undefined;
}

/**
 * Finds nearest seeds by jump flooding. Each dispatch checks eight cells at the current step
 * distance, then the distance halves. Owns buffers and records into caller passes.
 */
export function createJumpFlood(
  root: TgpuRoot,
  {
    size,
    seeds: count,
    format,
    time,
    target,
    paint,
    immediates = root.enabledWgslLanguageFeatures.has('immediate_address_space'),
  }: JumpFloodOptions,
) {
  // Repeat step 1 after halving to correct most remaining errors
  const steps: number[] = [];
  for (let step = size / 2; step >= 1; step /= 2) steps.push(step);
  steps.push(1);

  const cellCount = size * size;
  const cellGroups = Math.ceil(size / WORKGROUP);
  const words = Math.ceil(count / 32);

  // Read and write separate halves so dispatches cannot race with their own reads
  const cells = root.createBuffer(d.arrayOf(d.u32, cellCount * 2)).$usage('storage');
  const seeds = root.createBuffer(d.arrayOf(Seed, count)).$usage('storage');
  const [writableCells, readableCells] = [cells.as('mutable'), cells.as('readonly')];
  const [writableSeeds, readableSeeds] = [seeds.as('mutable'), seeds.as('readonly')];
  // A row of bits per seed. The flood's result sets the bits of seeds whose regions touch, so
  // the draw measures exact distances to a region's edges. Fragments read it atomically too
  const touching = root.createBuffer(d.arrayOf(d.atomic(d.u32), count * words)).$usage('storage');
  const neighbours = touching.as('mutable');
  const view = root.createUniform(View, { viewport: [target.width, target.height], result: 0 });

  // Queue writes cannot change uniforms between dispatches in one compute pass. Set an
  // immediate on the pass, or bind a separate uniform per dispatch. `current` reads either
  // source and defaults to the bind-group entry
  const layout = tgpu.bindGroupLayout({ pass: { uniform: Pass } });
  const current = tgpu.accessor(Pass, () => {
    'use gpu';
    return Pass(layout.$.pass);
  });
  const immediate = immediates ? tgpu['~unstable'].immediateVar(Pass) : undefined;
  const stepValues = steps.map((step, index) => new Uint32Array([step, index % 2]));
  const uniforms = immediate
    ? []
    : steps.map((step, index) =>
        root.createBuffer(Pass, { step, source: index % 2 }).$usage('uniform'),
      );
  const groups = uniforms.map((uniform) => root.createBindGroup(layout, { pass: uniform }));

  const clearKernel = tgpu.computeFn({
    workgroupSize: [WORKGROUP, WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= size || gid.y >= size) return;

    const index = gid.y * size + gid.x;
    writableCells.$[index] = NONE;
    if (index < count * words) std.atomicStore(neighbours.$[index], 0);
  });

  const placeKernel = tgpu.computeFn({
    workgroupSize: [SEED_WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= count) return;

    randf.seed2(d.vec2f(d.f32(gid.x), 0.5));
    const center = d.vec2f(randf.sample(), randf.sample());
    const radius = std.mix(CIRCLES[0], CIRCLES[1], randf.sample());
    const speed = (randf.sample() * 2 - 1) * DRIFT;
    const angle = randf.sample() * TAU + time.$ * speed;
    const circling = center + d.vec2f(std.cos(angle), std.sin(angle)) * radius;
    const position = std.clamp(circling, d.vec2f(0), d.vec2f(1 - 0.5 / size));
    writableSeeds.$[gid.x] = Seed({ position });

    const cell = d.vec2u(position * size);
    writableCells.$[cell.y * size + cell.x] = gid.x;
  });

  const floodKernel = tgpu.computeFn({
    workgroupSize: [WORKGROUP, WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= size || gid.y >= size) return;
    const pass = current.$;

    const here = d.vec2f(gid.xy) + 0.5;
    let nearest = d.u32(NONE);
    let shortest = d.f32(size * 2);

    for (let y = -1; y <= 1; y++) {
      for (let x = -1; x <= 1; x++) {
        const at = d.vec2i(gid.xy) + d.vec2i(x, y) * d.i32(pass.step);
        if (at.x >= 0 && at.y >= 0 && at.x < size && at.y < size) {
          // A dispatch may not bind one buffer both read-only and read-write, so the source half
          // is read through the mutable binding too
          const seed = writableCells.$[d.u32(at.y) * size + d.u32(at.x) + pass.source * cellCount];
          if (seed !== NONE) {
            const distance = std.distance(readableSeeds.$[seed].position * size, here);
            if (distance < shortest) {
              shortest = distance;
              nearest = seed;
            }
          }
        }
      }
    }

    writableCells.$[gid.y * size + gid.x + (1 - pass.source) * cellCount] = nearest;
  });

  const link = (from: number, to: number) => {
    'use gpu';
    const word = from * words + (to >>> 5);
    const bit = d.u32(1) << (to & 31);
    if ((std.atomicLoad(neighbours.$[word]) & bit) === 0) std.atomicOr(neighbours.$[word], bit);
  };

  const neighboursKernel = tgpu.computeFn({
    workgroupSize: [WORKGROUP, WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= size || gid.y >= size) return;
    const half = view.$.result * cellCount;
    const seed = readableCells.$[gid.y * size + gid.x + half];
    if (seed === NONE) return;

    for (let i = 0; i < 4; i++) {
      const at = d.vec2i(gid.xy) + TOUCHING.$[i];
      if (at.x >= 0 && at.x < size && at.y < size) {
        const other = readableCells.$[d.u32(at.y) * size + d.u32(at.x) + half];
        if (other !== NONE && other !== seed) {
          link(seed, other);
          link(other, seed);
        }
      }
    }
  });

  const fragment = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({ uv }) => {
    'use gpu';
    // Fit the grid across the target's longer side
    const scale = view.$.viewport / std.max(view.$.viewport.x, view.$.viewport.y);
    const point = ((uv - 0.5) * scale + 0.5) * size;
    const cell = d.vec2u(std.clamp(point, d.vec2f(0), d.vec2f(size - 1)));
    const footprint = size / std.max(view.$.viewport.x, view.$.viewport.y);

    const half = view.$.result * cellCount;

    let seed = readableCells.$[cell.y * size + cell.x + half];
    if (seed === NONE) {
      const unreached = Cell({
        point,
        seed,
        site: d.vec2f(),
        toSite: 0,
        second: seed,
        toSecond: 0,
        toEdge: 0,
        footprint,
      });
      return d.vec4f(paint(unreached), 1);
    }

    // Each neighbour bounds the distance to the edge by its bisector. Near an edge a neighbour
    // may be nearer to the pixel than the cell's seed; the scan then repeats from that neighbour
    let site = d.vec2f(readableSeeds.$[seed].position * size);
    let toSite = std.distance(site, point);
    let toEdge = d.f32(size);
    let second = d.u32(NONE);
    let toSecond = d.f32(size * 2);
    for (let scan = 0; scan < 2; scan++) {
      toEdge = d.f32(size);
      second = d.u32(NONE);
      toSecond = d.f32(size * 2);
      let secondSite = d.vec2f();
      for (let word = d.u32(0); word < words; word++) {
        let bits = std.atomicLoad(neighbours.$[seed * words + word]);
        while (bits !== 0) {
          const other = word * 32 + std.countTrailingZeros(bits);
          bits &= bits - 1;
          const there = readableSeeds.$[other].position * size;
          const across = std.normalize(there - site);
          toEdge = std.min(toEdge, std.dot((site + there) * 0.5 - point, across));
          const toThere = std.distance(there, point);
          if (toThere < toSecond) {
            second = other;
            toSecond = toThere;
            secondSite = d.vec2f(there);
          }
        }
      }
      if (scan > 0 || toSecond >= toSite) break;
      seed = second;
      site = d.vec2f(secondSite);
      toSite = toSecond;
    }

    return d.vec4f(
      paint(Cell({ point, seed, site, toSite, second, toSecond, toEdge, footprint })),
      1,
    );
  });

  const clearPipeline = root.createComputePipeline({ compute: clearKernel });
  const placePipeline = root.createComputePipeline({ compute: placeKernel });
  const neighboursPipeline = root.createComputePipeline({ compute: neighboursKernel });
  const floodPipeline = (immediate ? root.with(current, immediate) : root).createComputePipeline({
    compute: floodKernel,
  });
  const drawPipeline = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment,
    targets: { format },
  });

  let width = target.width;
  let height = target.height;
  let result = 0;

  return {
    /** Nearest seed per cell, or `NONE`. Each half contains `size * size` cells. */
    cells,
    seeds,
    /** Bit rows of touching regions, `ceil(seeds / 32)` words per seed. */
    neighbours: touching,
    /** Whether dispatch values use immediates. */
    immediates: immediate !== undefined,
    /** Maximum flood-dispatch count. */
    passes: steps.length,

    initAsync: () =>
      pipelines.compile([
        clearPipeline,
        placePipeline,
        floodPipeline,
        neighboursPipeline,
        drawPipeline,
      ]),

    /** Places seeds, records the first `passes` flood dispatches and finds touching regions. */
    flood(pass: TgpuComputePass, passes = steps.length) {
      pass.setPipeline(clearPipeline);
      pass.dispatchWorkgroups(cellGroups, cellGroups);
      pass.setPipeline(placePipeline);
      pass.dispatchWorkgroups(Math.ceil(count / SEED_WORKGROUP));

      pass.setPipeline(floodPipeline);
      for (let index = 0; index < passes; index++) {
        if (immediate) pass.setImmediates(immediate, stepValues[index]);
        else pass.setBindGroup(layout, groups[index]);
        pass.dispatchWorkgroups(cellGroups, cellGroups);
      }

      // The draw reads the half the last dispatch wrote
      if (passes % 2 !== result) {
        result = passes % 2;
        view.patch({ result });
      }

      pass.setPipeline(neighboursPipeline);
      pass.dispatchWorkgroups(cellGroups, cellGroups);
    },

    draw(pass: TgpuRenderCommands) {
      if (target.width !== width || target.height !== height) {
        width = target.width;
        height = target.height;
        view.patch({ viewport: d.vec2f(width, height) });
      }

      pass.setPipeline(drawPipeline);
      pass.draw(3);
    },

    destroy() {
      cells.destroy();
      seeds.destroy();
      touching.destroy();
    },
  };
}
