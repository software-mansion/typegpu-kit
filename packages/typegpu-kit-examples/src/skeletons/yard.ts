import { srgbToLinear } from '@typegpu/color';
import { instances, lighting, materials, type MeshPool } from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import { perlin2d } from '@typegpu/noise';
import { mulberry32 } from 'math/random';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';

type Vec3 = [number, number, number];

const paint = (r: number, g: number, b: number) => srgbToLinear(d.vec3f(r, g, b));
const WARM = paint(0.4, 0.37, 0.33);
const COOL = paint(0.33, 0.34, 0.35);
const JOINT = paint(0.14, 0.13, 0.12);
const GROUND = paint(0.12, 0.115, 0.11);
const IRON = materials.metal(paint(0.2, 0.19, 0.18), 0.55);
const COALS = materials.glow(d.vec3f(1, 0.22, 0.03), 3);
const DRESSED = materials.matte(paint(0.4, 0.38, 0.35));

/** Flagstone width and depth in metres; each row runs along x, half a stone on from the last. */
const FLAG = d.vec2f(1.6, 1.1);
/** Half the width of a joint between flagstones, in metres. */
const JOINT_HALF = 0.02;
/** Height of a course of the wall's stones and length of a stone, in metres. */
const COURSE = 0.6;
const BLOCK = 1.3;

/** The first gate's angle about +y from +x, and each gate's width in metres. */
export const GATE_ANGLE = Math.PI / 4;
const GATE_WIDTH = 6;
/** Metres beyond the skeletons' edge to the wall, and in from the wall to the braziers. */
const WALL_OUT = 3.5;
const BRAZIERS_IN = 2.2;
/** Height of the wall in metres, how much a stretch of it rises or sinks, and its thickness. */
const WALL = 3.4;
const WALL_VARY = 0.2;
const WALL_THICK = 1;
/** Mean length of a stretch of wall in metres. */
const STRETCH = 4;
/** Height of a brazier's rim over the ground, and of its flame's light over the rim, in metres. */
export const RIM = 1.25;
const FLAME_ABOVE_RIM = 0.35;
const FLAME = d.vec3f(1, 0.5, 0.2);
/** A brazier's light: a lantern's intensity, reaching a pool of the yard round it. */
const BRAZIER = { intensity: 100, range: 22 };

/** Fades a pattern of `scale` cells per metre where a cell spans less than about two pixels. */
const resolved = (at: d.v2f, scale: number) => {
  'use gpu';
  return std.saturate(1.5 - std.length(std.fwidth(at)) * scale * 4);
};

/**
 * How much of a fine joint `distance` metres from the point covers its pixel: full where the
 * joint is wider than the pixel, thinner and fainter where it is not.
 */
const jointAt = (distance: number, footprint: number) => {
  'use gpu';
  const width = std.max(JOINT_HALF, footprint * 0.5);
  return std.smoothstep(width * 1.5, width * 0.5, distance) * std.saturate(JOINT_HALF / width);
};

/**
 * Flagstones in running bond, each a little warmer or cooler, lighter or darker than the next,
 * with fine joints that hold less of the sky's light, out to `wall` metres from the middle, beyond
 * which lies bare ground.
 */
const yardOf = (wall: number) =>
  instances.define('Yard', {
    material: (point) => {
      'use gpu';
      const at = point.position.xz;
      const row = std.floor(at.y / FLAG.y);
      const bond = d.vec2f(at.x / FLAG.x + row * 0.5, at.y / FLAG.y);
      const cell = std.floor(bond);
      const inside = bond - cell;
      const edge = std.min(
        std.min(inside.x, 1 - inside.x) * FLAG.x,
        std.min(inside.y, 1 - inside.y) * FLAG.y,
      );

      // Per stone a tint and a shade, and over the yard a slow drift between warmer and cooler stone
      const tint = perlin2d.sample(cell * 0.61 + 0.37) * 0.5 + 0.5;
      const shade = 1 + perlin2d.sample(cell * 0.83 + 5.3) * 0.2;
      const drift = std.saturate(perlin2d.sample(at * 0.035 + 19) + 0.5);
      let stone = std.mix(COOL, WARM, std.mix(tint, drift, 0.5)) * shade;

      const joint = jointAt(edge, std.length(std.fwidth(at))) * resolved(at, 1 / FLAG.y);
      // Worn arrises: each stone darkens a little towards its joints
      const worn = std.smoothstep(0.1, 0, edge) * 0.15 * resolved(at, 4 / FLAG.y);
      stone = std.mix(stone * (1 - worn), JOINT, joint * 0.8);
      stone = std.mix(stone, GROUND, std.smoothstep(wall, wall + 1.5, std.length(at)));

      const paving = materials.glossy(stone, std.mix(0.68 + tint * 0.14, 0.95, joint));
      paving.occlusion = 1 - joint * 0.6 - worn;
      return paving;
    },
  });

/** Dressed stone in courses round the yard, each course's joints half a stone on from the last. */
const Masonry = instances.define('Masonry', {
  material: (point, _instance, base) => {
    'use gpu';
    const p = point.position;
    const along = std.atan2(p.z, p.x) * std.length(p.xz);
    const course = std.floor(p.y / COURSE);
    const laid = along / BLOCK + course * 0.5;
    const block = std.floor(laid);
    const height = p.y / COURSE - course;
    const across = laid - block;
    const footprint = std.length(std.fwidth(d.vec2f(along, p.y)));
    // Tops show no courses
    const side = std.select(d.f32(1), 0, std.abs(point.normal.y) > 0.5);
    const edge = std.min(std.min(height, 1 - height) * COURSE, std.min(across, 1 - across) * BLOCK);
    const joint = jointAt(edge, footprint) * side * std.saturate(1.5 - footprint * (4 / COURSE));
    const shade = 1 + perlin2d.sample(d.vec2f(block, course) * 0.83 + 0.21) * 0.16;
    const color = std.mix(base.color.xyz * shade, JOINT, joint * 0.75);
    const made = materials.glossy(color, 0.85);
    made.occlusion = 1 - joint * 0.5;
    return made;
  },
});

/** How brightly brazier `k` burns at `seconds`, around 0.85 of full. */
const flicker = (seconds: number, k: number) => {
  'use gpu';
  return (
    0.85 +
    0.08 * std.sin(seconds * 13 + k * 1.7) +
    0.05 * std.sin(seconds * 29 + k * 2.3) +
    0.04 * std.sin(seconds * 7.3 + k)
  );
};

/** Turns about +y so a box's x axis points out from the middle at `angle`. */
const facingOut = (angle: number) => [0, Math.sin(-angle / 2), 0, Math.cos(-angle / 2)] as const;

/**
 * A walled courtyard: flagstones, a wall round them with a gate on each side, and a brazier on a
 * stone plinth either side of each gate, against the wall, whose flames light the yard.
 */
export function createYard(
  root: TgpuRoot,
  pool: MeshPool,
  options: { radius: number; gates: number },
) {
  const { radius, gates } = options;
  const gateAt = (gate: number) => (gate / gates) * Math.PI * 2 + GATE_ANGLE;

  const wallAt = radius + WALL_OUT;
  const ground = instances.create(root, pool, { kind: yardOf(wallAt + 0.5), capacity: 1 });
  ground.add(pool.add(meshes.plane({ width: radius * 16, depth: radius * 16 })), {});

  const block = pool.add(meshes.box({ width: 1, height: 1, depth: 1 }));
  const stem = pool.add(meshes.cylinder({ radius: 0.07, height: 1, radialSegments: 12 }));
  const bowl = pool.add(meshes.cylinder({ radius: 0.5, height: 0.32 }));
  const lip = pool.add(meshes.torus({ radius: 0.5, tube: 0.045 }));
  const embers = pool.add(meshes.sphere({ radius: 1 }));

  // A wall of stretches round the yard, each a little higher or lower, open at the gates
  const stretches = Math.round((Math.PI * 2 * wallAt) / STRETCH);
  const stretch = (Math.PI * 2 * wallAt) / stretches;
  const gateHalf = GATE_WIDTH / 2 / wallAt;
  const between = (Math.PI * 2) / gates;
  const state = mulberry32.create(7);
  const roll = () => mulberry32.sample(state);
  const stone = instances.create(root, pool, { kind: Masonry, capacity: stretches + gates * 3 });
  for (let i = 0; i < stretches; i++) {
    const angle = ((i + 0.5) / stretches) * Math.PI * 2;
    const past = (((angle - GATE_ANGLE) % between) + between) % between;
    if (Math.min(past, between - past) < gateHalf) continue;
    const height = WALL + (roll() - 0.5) * 2 * WALL_VARY;
    stone.add(block, {
      position: [Math.cos(angle) * wallAt, height / 2, Math.sin(angle) * wallAt],
      rotation: facingOut(angle),
      scale: [WALL_THICK, height, stretch + 0.08],
      ...DRESSED,
    });
  }

  const brazierAt = wallAt - BRAZIERS_IN;
  const flank = (GATE_WIDTH / 2 + 1) / brazierAt;
  /** Each brazier's angle: either side of each gate, then one halfway between gates. */
  const angles = Array.from({ length: gates * 3 }, (_, k) =>
    k < gates * 2
      ? gateAt(Math.floor(k / 2)) + (k % 2 ? flank : -flank)
      : gateAt(k - gates * 2) + between / 2,
  );
  /** Where each brazier's rim is. */
  const rims: Vec3[] = angles.map((angle) => [
    Math.cos(angle) * brazierAt,
    RIM,
    Math.sin(angle) * brazierAt,
  ]);
  const flames = tgpu.const(
    d.arrayOf(d.vec3f, rims.length),
    rims.map(([x, y, z]) => d.vec3f(x, y + FLAME_ABOVE_RIM, z)),
  );
  const iron = instances.create(root, pool, { capacity: rims.length * 4 });
  for (const [k, [x, , z]] of rims.entries()) {
    const angle = angles[k];
    stone.add(block, {
      position: [x, 0.3, z],
      rotation: facingOut(angle),
      scale: [0.75, 0.6, 0.75],
      ...DRESSED,
    });
    iron.add(stem, { position: [x, 0.85, z], scale: [1, 0.5, 1], ...IRON });
    iron.add(bowl, { position: [x, RIM - 0.16, z], ...IRON });
    iron.add(lip, { position: [x, RIM, z], ...IRON });
    iron.add(embers, { position: [x, RIM - 0.06, z], scale: [0.42, 0.08, 0.42], ...COALS });
  }

  const time = root.createUniform(d.f32);
  /** Brazier `slot`'s flame as a point light, flickering on its own. */
  const flameLight = (slot: number) => {
    'use gpu';
    return lighting.Light({
      position: d.vec3f(flames.$[slot]),
      color: FLAME,
      intensity: BRAZIER.intensity * flicker(time.$, d.f32(slot)),
      range: BRAZIER.range,
    });
  };

  ground.flush();
  stone.flush();
  iron.flush();

  return {
    ground,
    stone,
    iron,
    burn: (now: number) => time.write(now),
    rims: rims as readonly Readonly<Vec3>[],
    /** Metres from the middle to just beyond the wall, which a shadow covering the yard reaches. */
    reach: wallAt + 2,
    /** The braziers' flames, for the lighting's `points`. */
    points: { light: flameLight, count: rims.length },

    destroy() {
      ground.destroy();
      stone.destroy();
      iron.destroy();
    },
  };
}
