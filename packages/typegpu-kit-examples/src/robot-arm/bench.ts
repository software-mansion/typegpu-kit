import { srgbToLinear } from '@typegpu/color';
import { instances, materials, type MeshPool } from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import { perlin2d } from '@typegpu/noise';
import { quat, vec3, type Vec3Arg } from 'wgpu-matrix';
import { d, std, type TgpuRoot } from 'typegpu';

type Triple = [number, number, number];
type Look = ReturnType<typeof materials.matte>;

const paint = (r: number, g: number, b: number) => srgbToLinear(d.vec3f(r, g, b));
const OAK = paint(0.62, 0.43, 0.26);
const WALNUT = paint(0.42, 0.27, 0.15);
const SEAM = paint(0.2, 0.12, 0.07);
const HARDBOARD = paint(0.36, 0.28, 0.2);
const HOLE = paint(0.12, 0.09, 0.06);
const IRON = materials.metal(paint(0.55, 0.56, 0.58), 0.35);
const RED_HANDLE = materials.glossy(paint(0.7, 0.12, 0.08), 0.4);
const YELLOW_HANDLE = materials.glossy(paint(0.92, 0.7, 0.12), 0.4);
const ASH = materials.matte(paint(0.72, 0.58, 0.4));
const ENAMEL = materials.glossy(paint(0.2, 0.46, 0.38), 0.28);
const BRASS = materials.metal(paint(0.85, 0.66, 0.36), 0.3);
const TIN = materials.metal(paint(0.6, 0.6, 0.62), 0.5);
const LABEL = materials.matte(paint(0.75, 0.2, 0.12));
const SHELF = materials.matte(paint(0.38, 0.26, 0.16));
const BULB = materials.glow(d.vec3f(1, 0.82, 0.55), 6);

/**
 * The bench top's half width along x, half depth along z and thickness. A unit is a toy's
 * decimetre: the arm stands about 35 cm.
 */
const BENCH = { x: 7, z: 4.6, thick: 0.6 };
/** Plank width along z, and growth rings across a plank. */
const PLANK = 1.15;
const RINGS = 22;
/** The pegboard's hole spacing. */
const PEGS = 0.3;
/** Where the desk lamp's base, elbow and shade sit, and the bench point its shade aims at. */
const LAMP = {
  base: [-3.6, 0, -3.2] as Triple,
  elbow: [-4.0, 2.7, -3.0] as Triple,
  head: [-2.7, 3.0, -1.8] as Triple,
  aim: [-0.6, 0, -0.1] as Triple,
};
/** How far the bulb sits inside the shade, from its top towards its mouth. */
const BULB_DEPTH = 0.3;
/** The shade's scale. */
const SHADE = 0.75;
/** Tins on the shelf: across the bench and height. */
const TINS: [x: number, height: number, look: Look][] = [
  [-5.7, 1.2, TIN],
  [-4.6, 1.6, LABEL],
  [-3.4, 1.1, TIN],
  [-2.3, 1.4, TIN],
  [-1.2, 0.9, LABEL],
];

/** Fades detail of `scale` cells per unit where a cell spans less than about two pixels. */
const resolved = (at: d.v2f, scale: number) => {
  'use gpu';
  return std.saturate(1.5 - std.length(std.fwidth(at)) * scale * 4);
};

/** Planks of oak along x, with wavy grain of uneven weight and thin seams. */
const Bench = instances.define('Bench', {
  material: (point) => {
    'use gpu';
    const p = point.position;
    const plank = std.floor(p.z / PLANK);
    const across = p.z / PLANK - plank;
    // Growth rings run along the plank and wander across it
    const wander =
      perlin2d.sample(d.vec2f(p.x * 0.35, across * 2.5 + plank * 7.3)) * 2.5 +
      perlin2d.sample(d.vec2f(p.x * 1.3, across * 6 + plank * 3.1)) * 0.5;
    // Rings crowd and spread across the plank as the tree grew faster or slower
    const grown = across + perlin2d.sample(d.vec2f(across * 3, plank * 5.1)) * 0.06;
    const rings = grown * RINGS + wander;
    const weight =
      0.4 + 0.6 * std.saturate(perlin2d.sample(d.vec2f(std.floor(rings) * 0.9, plank)) + 0.5);
    const late = std.pow(0.5 + 0.5 * std.cos(rings * Math.PI * 2), 4) * weight;
    const fine = std.saturate(1.2 - std.fwidth(rings) * 2.5);
    const tone =
      perlin2d.sample(d.vec2f(plank * 3.3, 0.5)) * 0.18 + perlin2d.sample(p.xz * 0.35) * 0.05;
    let color = std.mix(OAK, WALNUT, std.saturate(0.35 + tone));
    color *= 1 - late * 0.12 * fine;
    const seam = std.smoothstep(0.012, 0.004, std.min(across, 1 - across) * PLANK);
    color = std.mix(color, SEAM, seam * 0.7 * resolved(p.xz, 1 / PLANK));
    return materials.glossy(color, std.mix(0.68 + late * 0.06, 1, seam));
  },
});

/** Painted hardboard pierced in a square grid, facing +z. */
const Pegboard = instances.define('Pegboard', {
  material: (point) => {
    'use gpu';
    const at = point.position.xy / PEGS;
    const off = (std.fract(at) - 0.5) * PEGS;
    const hole = std.smoothstep(0.035, 0.022, std.length(off)) * resolved(at, d.f32(1));
    const tone = 1 + perlin2d.sample(point.position.xy * 0.3) * 0.08;
    return materials.matte(std.mix(HARDBOARD * tone, HOLE, hole * 0.8));
  },
});

/** Painted beech, a faint grain showing under the paint. */
export const Painted = instances.define('Painted', {
  material: (point, _instance, base) => {
    'use gpu';
    const p = point.position;
    const grain = perlin2d.sample(d.vec2f(p.x * 2 + p.z * 2, p.y * 40)) * 0.06;
    return materials.glossy(base.color.xyz * (1 + grain), 0.42);
  },
});

/** The desk lamp's shade: down the outside to the rim, then back up the inside. */
const shade = meshes.parametric(
  {
    at: (u, v) => {
      'use gpu';
      const angle = u * Math.PI * 2;
      const inside = v > 0.5;
      const t = std.select(v * 2, 2 - v * 2, inside);
      const radius = std.mix(0.35, 1.25, t * t) * std.select(d.f32(1), 0.95, inside);
      const y = -t * 1.3 + std.select(d.f32(0), 0.03, inside);
      return d.vec3f(std.cos(angle) * radius, y, std.sin(angle) * radius);
    },
  },
  { cols: 32, rows: 16 },
);

const UP: Triple = [0, 1, 0];

/**
 * A workbench of oak planks with a pegboard of tools and a shelf of tins behind it, and a desk
 * lamp whose shade points at the arm. Its `lamp` gives the spot light the bulb casts.
 */
export function createBench(root: TgpuRoot, pool: MeshPool) {
  const block = pool.add(meshes.box({ width: 1, height: 1, depth: 1 }));
  const rod = pool.add(meshes.cylinder({ radius: 1, height: 1, radialSegments: 20 }));
  const ball = pool.add(meshes.sphere({ radius: 1 }));
  const loop = pool.add(meshes.torus({ radius: 0.32, tube: 0.09 }));
  const hood = pool.add(shade);

  const top = instances.create(root, pool, { kind: Bench, capacity: 1 });
  top.add(block, {
    position: [0, -BENCH.thick / 2, 0],
    scale: [BENCH.x * 2, BENCH.thick, BENCH.z * 2],
  });
  const board = instances.create(root, pool, { kind: Pegboard, capacity: 1 });
  board.add(block, { position: [0, 6, -BENCH.z - 0.1], scale: [BENCH.x * 2 + 12, 16, 0.2] });

  const props = instances.create(root, pool, { capacity: 32 });
  const along = vec3.create();
  const turn = quat.create();
  /** Adds a rod of `radius` from `from` to `to`. */
  const between = (from: Vec3Arg, to: Vec3Arg, radius: number, look: Look) => {
    vec3.sub(to, from, along);
    const length = vec3.length(along);
    quat.rotationTo(UP, vec3.normalize(along, along), turn);
    props.add(rod, {
      position: vec3.lerp(from, to, 0.5),
      rotation: turn,
      scale: [radius, length, radius],
      ...look,
    });
  };

  // The desk lamp, its shade's mouth towards the aim and the bulb inside it
  const { base, elbow, head, aim } = LAMP;
  props.add(rod, { position: [base[0], 0.12, base[2]], scale: [0.85, 0.24, 0.85], ...ENAMEL });
  between([base[0], 0.24, base[2]], elbow, 0.07, BRASS);
  between(elbow, head, 0.07, BRASS);
  props.add(ball, { position: elbow, scale: 0.15, ...BRASS });
  const facing = vec3.normalize(vec3.sub(aim, head));
  quat.rotationTo(UP, vec3.negate(facing), turn);
  props.add(hood, { position: head, rotation: turn, scale: SHADE, ...ENAMEL });
  props.add(ball, { position: head, scale: 0.3, ...ENAMEL });
  const bulb = vec3.addScaled(head, facing, BULB_DEPTH);
  props.add(ball, { position: bulb, scale: 0.2, ...BULB });

  // Tools on the pegboard: two screwdrivers, a hammer and a wrench
  const wall = -BENCH.z + 0.15;
  between([1.6, 0.7, wall], [1.6, 1.9, wall], 0.035, IRON);
  between([1.6, 1.9, wall], [1.6, 2.7, wall], 0.12, RED_HANDLE);
  between([2.2, 0.9, wall], [2.2, 1.9, wall], 0.03, IRON);
  between([2.2, 1.9, wall], [2.2, 2.6, wall], 0.11, YELLOW_HANDLE);
  between([3.2, 0.7, wall], [3.2, 2.5, wall], 0.08, ASH);
  props.add(block, { position: [3.2, 2.65, wall + 0.08], scale: [0.8, 0.26, 0.26], ...IRON });
  props.add(block, { position: [4.2, 1.6, wall], scale: [0.16, 1.5, 0.06], ...IRON });
  props.add(loop, {
    position: [4.2, 2.5, wall],
    rotation: [Math.SQRT1_2, 0, 0, Math.SQRT1_2],
    scale: 0.65,
    ...IRON,
  });

  // A shelf of tins over the bench
  props.add(block, { position: [-3.5, 6.8, wall + 0.6], scale: [6, 0.2, 1.4], ...SHELF });
  for (const [x, height, look] of TINS) {
    props.add(rod, {
      position: [x, 6.9 + height / 2, wall + 0.6],
      scale: [0.42, height, 0.42],
      ...look,
    });
  }

  top.flush();
  board.flush();
  props.flush();

  return {
    drawings: [top.renderer(), board.renderer(), props.renderer()],
    destroy() {
      top.destroy();
      board.destroy();
      props.destroy();
    },
    /** Where the lamp's light starts and the way it shines, for a spot light. */
    lamp: {
      position: [bulb[0], bulb[1], bulb[2]] as Triple,
      aim: [facing[0], facing[1], facing[2]] as Triple,
    },
  };
}
