import { srgbToLinear } from '@typegpu/color';
import { instances, materials, shading, type meshPool } from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import { perlin2d } from '@typegpu/noise';
import { clamp } from 'math';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';

/** Square ground width in world units and grid cells per side. */
const SIZE = 80;
const CELLS = 200;
/** Grid points per side, and world units across a cell. */
const SIDE = CELLS + 1;
const CELL = SIZE / CELLS;
/**
 * Fractal Perlin frequency in cells per world unit and octave count. Each octave multiplies
 * frequency by `DETAIL` and amplitude by `GAIN`. Initial amplitude is in world units.
 */
const FREQUENCY = 0.11;
const OCTAVES = 4;
const DETAIL = 2;
const GAIN = 0.45;
const HEIGHT = 1.8;
const WORKGROUP = 8;

const SAND = srgbToLinear(d.vec3f(0.92, 0.76, 0.5));
const STRAW = srgbToLinear(d.vec3f(0.86, 0.62, 0.34));
const KHAKI = srgbToLinear(d.vec3f(0.7, 0.6, 0.4));
const EARTH = srgbToLinear(d.vec3f(0.66, 0.4, 0.24));
const SCRUB = srgbToLinear(d.vec3f(0.36, 0.3, 0.18));
/** Wind ripples per world unit and scrub tufts per world unit. */
const RIPPLES = 2.2;
const TUFTS = 5;
/** Radii from the centre in world units over which the ground fades into the fog. */
const FADE = [24, 38] as const;

/**
 * Fades detail of `scale` cells per world unit out where a cell spans more than about two pixels,
 * so it does not shimmer at distance.
 */
const resolved = (at: d.v2f, scale: number) => {
  'use gpu';
  return std.saturate(1.5 - std.length(std.fwidth(at)) * scale * 4);
};

/** Applies height fog, then fades the ground into the fog colour before its edge. */
const fog = (point: shading.Point) => {
  'use gpu';
  const edge = std.smoothstep(FADE[0], FADE[1], std.length(point.position.xz));
  const air = shading.heightFog(point);
  const added = std.mix(air.added, shading.settings().fogColor, edge);
  return shading.Air({ added, through: air.through * (1 - edge) });
};

/**
 * Dry grass on the slopes, sand on the crests, khaki in the hollows and bare earth where steep,
 * with wind ripples and scattered scrub up close.
 */
const Ground = instances.define('Ground', {
  material: (point) => {
    'use gpu';
    const at = point.position.xz;
    const patches = perlin2d.sample(at * 0.13) * 0.5 + 0.5;
    const scrub = perlin2d.sample(at * 1.7 + 31);
    const hollow = std.smoothstep(0.05, -0.7, point.position.y + scrub * 0.15);
    const steep = std.smoothstep(0.1, 0.32, 1 - point.normal.y);
    let color = std.mix(STRAW, SAND, std.saturate(patches * 1.4 - 0.2));
    color = std.mix(color, KHAKI, hollow * 0.6);
    color = std.mix(color, EARTH, steep * 0.6);

    const bend = perlin2d.sample(at * 0.4 + 7) * 4;
    const ripple = std.sin((at.x * 0.8 + at.y * 0.6) * RIPPLES * Math.PI * 2 + bend);
    color *= 1 + ripple * 0.1 * resolved(at, RIPPLES) * (1 - hollow);
    const tuft = std.smoothstep(0.38, 0.5, perlin2d.sample(at * TUFTS + 53));
    color = std.mix(color, SCRUB, tuft * 0.35 * resolved(at, d.f32(TUFTS)) * (1 - steep));
    return materials.matte(color * (1 + scrub * 0.1));
  },
});

/**
 * Generates terrain heights and bakes the mesh on the GPU. Reads heights back once so CPU foot
 * placement matches the drawn triangles.
 */
export async function createTerrain(root: TgpuRoot, pool: meshPool.MeshPool) {
  const grid = root.createBuffer(d.arrayOf(d.f32, SIDE * SIDE)).$usage('storage');
  const writing = grid.as('mutable');
  const reading = grid.as('readonly');

  /** Fills the grid with fractal Perlin heights, its columns and rows along the plane's u and v. */
  const fill = tgpu.computeFn({
    workgroupSize: [WORKGROUP, WORKGROUP],
    in: { id: d.builtin.globalInvocationId },
  })(({ id }) => {
    'use gpu';
    if (id.x >= SIDE || id.y >= SIDE) return;
    const at = (d.vec2f(id.xy) / CELLS - 0.5) * d.vec2f(1, -1) * SIZE * FREQUENCY;

    let height = d.f32(0);
    let amplitude = d.f32(1);
    let scale = d.f32(1);
    for (let octave = 0; octave < OCTAVES; octave++) {
      height += perlin2d.sample(at * scale) * amplitude;
      amplitude *= GAIN;
      scale *= DETAIL;
    }
    writing.$[id.y * SIDE + id.x] = height * HEIGHT;
  });

  const heightOf = (column: number, row: number) => {
    'use gpu';
    const kept = std.clamp(d.vec2i(column, row), d.vec2i(0), d.vec2i(CELLS));
    return reading.$[kept.y * SIDE + kept.x];
  };

  const shape = meshes.parametric(
    {
      at: (u, v) => {
        'use gpu';
        const cell = d.vec2i(std.round(d.vec2f(u, v) * CELLS));
        return d.vec3f((u - 0.5) * SIZE, heightOf(cell.x, cell.y), (0.5 - v) * SIZE);
      },
      // Rows run toward -z
      normalAt: (u, v) => {
        'use gpu';
        const cell = d.vec2i(std.round(d.vec2f(u, v) * CELLS));
        const dx = heightOf(cell.x + 1, cell.y) - heightOf(cell.x - 1, cell.y);
        const dz = heightOf(cell.x, cell.y - 1) - heightOf(cell.x, cell.y + 1);
        return std.normalize(d.vec3f(-dx, 2 * CELL, -dz));
      },
    },
    { cols: CELLS, rows: CELLS },
  );

  const encoder = root.createCommandEncoder();
  const groups = Math.ceil(SIDE / WORKGROUP);
  const pass = encoder.beginComputePass();
  pass.setPipeline(root.createComputePipeline({ compute: fill }));
  pass.dispatchWorkgroups(groups, groups);
  pass.end();
  const mesh = pool.add(shape, encoder);
  encoder.submit();

  const heights = Float32Array.from(await grid.read());
  grid.destroy();

  const field = instances.create(root, pool, { kind: Ground, capacity: 1 });
  field.add(mesh);
  field.flush();

  return {
    drawing: field.renderer().pipe((cfg) => cfg.with(shading.fog, fog)),
    destroy() {
      field.destroy();
    },

    /** Ground height on the rendered triangle at a position. */
    heightAt(x: number, z: number) {
      const u = clamp((x / SIZE + 0.5) * CELLS, 0, CELLS);
      const v = clamp((0.5 - z / SIZE) * CELLS, 0, CELLS);
      const column = Math.min(CELLS - 1, Math.floor(u));
      const row = Math.min(CELLS - 1, Math.floor(v));
      const across = u - column;
      const along = v - row;

      const corner = row * SIDE + column;
      const h00 = heights[corner];
      const h11 = heights[corner + SIDE + 1];
      // Match the mesh diagonal from (0, 0) to (1, 1) in each cell
      if (across >= along) {
        const h10 = heights[corner + 1];
        return h00 + across * (h10 - h00) + along * (h11 - h10);
      }
      const h01 = heights[corner + SIDE];
      return h00 + along * (h01 - h00) + across * (h11 - h01);
    },
  };
}
