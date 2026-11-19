import { instances, materials, meshPool } from '@typegpu/kit';
import { perlin3d } from '@typegpu/noise';
import { d, std, type TgpuRoot } from 'typegpu';

/** The named nodes of `models/arcade.glb`: the hall, and the pendulum about its pivot. */
export const PARTS = ['arcade', 'pendulum'] as const;
export type Part = (typeof PARTS)[number];

/** The glTF materials, in the order the shader numbers them. */
const FINISHES = ['concrete', 'floor', 'ground', 'stone', 'bronze', 'brass', 'steel'];
const [, FLOOR, GROUND, STONE, BRONZE, BRASS, STEEL] = FINISHES.map((_, i) => i);

/** Where the pendulum hangs, in the model's metres. */
const PIVOT: [number, number, number] = [0, 9.7, 0];

/**
 * The free air inside the hall, in metres: between the piers and the east wall's pilasters, above
 * the floor, below the transverse beams and between the end walls.
 */
export const ROOM = { x: [-4.95, 4.7], y: [0.3, 7.9], z: [-20, 15] } as const;

/**
 * The sun's light bounced around the hall, baked into each vertex's colour as a share of its
 * brightest value. Blender baked it as 0.0514 of a unit sun, whose diffuse light carries a 1/π
 * the kit's leaves out; the kit's sun has intensity 3.4.
 */
const BOUNCE = 0.0514 * Math.PI * 3.4 * 0.35;
/** The sky the unbaked pendulum sees in the dark hall, as an occlusion. */
const UNBAKED_SKY = 0.4;
/** The warm light the bronze's edge catches from the sunlit hall around it. */
const RIM = d.vec3f(1, 0.52, 0.2);

/**
 * A line every `spacing` metres along `t`, 0 on the line and 1 off it, blurred over `blur`
 * metres, `t`'s change across a pixel, and fading where lines crowd closer than pixels.
 */
const groove = (t: number, blur: number, spacing: number, width: number) => {
  'use gpu';
  const offset = std.abs(std.fract(t / spacing + 0.5) - 0.5) * spacing;
  const fade = std.saturate(1.5 - (blur * 6) / spacing);
  return 1 - (1 - std.smoothstep(width, width + blur, offset)) * fade;
};

export function createArcade(root: TgpuRoot, model: Readonly<Record<Part, meshPool.ModelData>>) {
  const pool = meshPool.create(root, {
    ...meshPool.roomFor([model.arcade, model.pendulum]),
    extra: { color: d.vec4f },
  });

  /**
   * Board-formed concrete, a polished floor, garden gravel, dark stone and bronze. The baked
   * colour carries the sun's bounce in rgb and the occlusion of the sky in alpha.
   */
  const Hall = instances.define('Hall', {
    host: { finish: d.u32, baked: d.f32 },
    carry: (vertexIndex) => {
      'use gpu';
      return pool.extra.color(vertexIndex);
    },
    material: (point, hall, _base, carried) => {
      'use gpu';
      const p = point.position;
      const n = point.normal;
      // Derivatives run first, in uniform control flow
      const blur = std.max(std.fwidth(p), d.vec3f(1e-4));
      const finish = hall.finish;
      const light = std.mix(d.vec4f(0, 0, 0, UNBAKED_SKY), carried, hall.baked);
      const grazing = 1 - std.saturate(std.dot(n, point.view));
      const grain = perlin3d.sample(p * 7.5);
      const stain = perlin3d.sample(p * 0.43);
      const upright = 1 - std.smoothstep(0.5, 0.8, std.abs(n.y));
      const along = std.select(p.z, p.x, std.abs(n.x) < std.abs(n.z));

      // Concrete: boards cast 30 cm high, each poured a shade apart, with lift lines and streaks
      const board = std.floor(p.y / 0.3);
      const pour = perlin3d.sample(d.vec3f(board * 1.7, along * 0.35, 0.5));
      const streak = perlin3d.sample(d.vec3f(along * 2.3, p.y * 0.12, 0.25));
      let tone = 0.9 + grain * 0.05 + stain * 0.1 + (pour * 0.05 + streak * 0.07) * upright;
      tone *= std.mix(1, groove(p.y, blur.y, 0.3, 0.008) * 0.06 + 0.94, upright);
      tone *= std.mix(1, groove(p.y + 0.05, blur.y, 1.35, 0.012) * 0.2 + 0.8, upright);
      let color = d.vec3f(0.43, 0.405, 0.35) * tone;
      let roughness = d.f32(0.88);
      let metallic = d.f32(0);

      if (finish === FLOOR) {
        const joint = std.min(groove(p.x, blur.x, 2.5, 0.012), groove(p.z, blur.z, 2.5, 0.012));
        color =
          d.vec3f(0.3, 0.28, 0.235) * (0.84 + grain * 0.05 + stain * 0.12) * (0.55 + joint * 0.45);
        roughness = 0.42 + stain * 0.2;
      }
      if (finish === GROUND) {
        color = d.vec3f(0.22, 0.18, 0.13) * (0.75 + perlin3d.sample(p * 3.1) * 0.3 + stain * 0.2);
        roughness = 1;
      }
      if (finish === STONE) {
        color = d.vec3f(0.05, 0.047, 0.042) * (1 + grain * 0.15);
        roughness = 0.22;
      }
      let rim = d.f32(0);
      // Dark bronze with a little patina, and polished brass, both lit hot at their edges
      if (finish === BRONZE || finish === BRASS) {
        const patina =
          std.smoothstep(0.05, 0.45, stain + grain * 0.18) * std.select(1, 0.2, finish === BRASS);
        color = std.mix(d.vec3f(0.4, 0.2, 0.07), d.vec3f(0.08, 0.12, 0.09), patina * 0.6);
        roughness = 0.2 + patina * 0.3;
        metallic = 1;
        if (finish === BRASS) color = std.mix(d.vec3f(0.9, 0.6, 0.24), color, patina);
        rim = std.pow(grazing, 4) * (1 - patina) * 0.7;
      }
      if (finish === STEEL) {
        color = d.vec3f(0.065, 0.06, 0.05);
        metallic = 0.7;
        roughness = 0.45;
      }

      const made = materials.glossy(color, roughness);
      made.metallic = metallic;
      made.occlusion = light.w;
      made.emission = color * light.xyz * (BOUNCE * (1 - metallic)) + RIM * rim;
      return made;
    },
  });

  const set = instances.create(root, pool, { kind: Hall, capacity: 16 });
  const hall = pool.add(model.arcade);
  const swinging = pool.add(model.pendulum);
  const finishOf = (part: Part, i: number) =>
    FINISHES.indexOf(model[part].meshes[i].material.name ?? '');
  set.add(hall, (_, i) => ({ finish: finishOf('arcade', i), baked: 1 }));
  const pendulum = set.add(swinging, (_, i) => ({
    finish: finishOf('pendulum', i),
    baked: 0,
    position: PIVOT,
  }));
  const turn: [number, number, number, number] = [0, 0, 0, 1];

  return {
    set,
    ready: Promise.all([hall.ready, swinging.ready]),
    /** Turns the pendulum `radians` about its pivot, across the hall. */
    swing(radians: number) {
      turn[2] = Math.sin(radians / 2);
      turn[3] = Math.cos(radians / 2);
      set.write(pendulum, { rotation: turn });
    },

    destroy() {
      set.destroy();
      pool.destroy();
    },
  };
}
