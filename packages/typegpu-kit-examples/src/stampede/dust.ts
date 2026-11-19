import {
  cameras,
  materials,
  Particle,
  particles,
  pipelines,
  shading,
  type CameraBase,
  type Lighting,
} from '@typegpu/kit';
import { srgbToLinear } from '@typegpu/color';
import { randf } from '@typegpu/noise';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';
import { STEP_SECONDS, type Herd } from './herd.ts';

const WORKGROUP = 64;
/** Horse speeds where hoof dust begins and reaches maximum emission. */
const KICK_FROM = 5;
const KICK_FULL = 14;
/** Puffs a galloping horse kicks up a second at full speed. */
const PUFFS_PER_SECOND = 0.45;
const LIFE_SECONDS = [2.5, 4.5] as const;
const SIZE_METRES = 2.6;
const GROWTH_METRES_PER_SECOND = 0.9;
const OPACITY = 0.18;
const LIFT_METRES_PER_SECOND = 0.7;
const SCATTER_METRES_PER_SECOND = 1.2;
const DUST = srgbToLinear(d.vec3f(0.78, 0.64, 0.46));
/** Share of the sunlight a puff scatters, and how forward scattering brightens it toward the sun. */
const SUN_SHARE = 0.28;
const FORWARD_POWER = 5;
const FORWARD_GLOW = 3.2;
const CLAP_PUFFS = 160;
const CLAP_SPEED_METRES_PER_SECOND = 7;

const Settings = d.struct({ amount: d.f32 });

/** Sunlight a puff scatters toward the eye, most when the eye looks toward the sun. */
const scatter: shading.ReflectHook = (material, point, direction, radiance) => {
  'use gpu';
  const forward = std.pow(std.saturate(-std.dot(point.view, direction)), FORWARD_POWER);
  return material.color * radiance * (SUN_SHARE * (0.35 + forward * FORWARD_GLOW));
};

/** A puff takes the light at its centre, so the herd's shadows would flicker across it whole. */
const unshadowed: shading.SunShadowHook = () => {
  'use gpu';
  return d.f32(1);
};

interface DustOptions {
  readonly camera: CameraBase;
  readonly light: Lighting;
  readonly step: number;
  readonly herd: Herd;
  readonly capacity: number;
  /** Scales how much dust the herd kicks up and how thick it looks. */
  readonly amount: number;
}

/** Draws drifting, spreading hoof dust lit by the scene's sun and sky. */
export function createDust(root: TgpuRoot, options: DustOptions) {
  const { camera: view, light, step, herd, capacity, amount } = options;
  const settings = root.createUniform(Settings, { amount });

  const shadePuff = (puff: Particle, local: d.v2f) => {
    'use gpu';
    const r2 = std.dot(local, local);
    const body = std.saturate(1 - r2) * std.saturate(1 - r2);
    const t = puff.age / puff.life;
    const life = std.saturate(t * 6) * (1 - t);
    const alpha = std.saturate(body * life * puff.color.w * settings.$.amount);
    const facing = cameras.towards(cameras.current.$, puff.position);
    const lit = shading.shade(materials.matte(DUST), shading.pointAt(puff.position, facing));
    return d.vec4f(lit * alpha, alpha);
  };

  const dust = particles.create(root, view, {
    label: 'dust',
    capacity,
    step,
    blend: 'over',
    settings: { gravity: [0, 0, 0], drag: 0.8 },
    update: (puff: Particle, dt: number) => {
      'use gpu';
      const next = Particle(puff);
      next.size = puff.size + GROWTH_METRES_PER_SECOND * dt;
      return next;
    },
    shade: tgpu
      .fn(shadePuff)
      .pipe(light.inject())
      .pipe((cfg) => cfg.with(shading.reflect, scatter).with(shading.sunShadow, unshadowed)),
  });

  const kick = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (i >= herd.controls.$.count) return;
    const horse = herd.horses.$[i];
    const speed = std.length(horse.velocity);
    const kicking = std.saturate((speed - KICK_FROM) / (KICK_FULL - KICK_FROM));
    randf.seed3(d.vec3f(d.f32(i), herd.controls.$.time + 0.5, horse.phase));
    if (randf.sample() >= kicking * PUFFS_PER_SECOND * STEP_SECONDS * settings.$.amount) return;
    dust.emit(
      Particle({
        position: horse.position + d.vec3f(randf.sample() - 0.5, 0.3, randf.sample() - 0.5),
        age: 0,
        velocity:
          d.vec3f(horse.velocity.x * 0.12, LIFT_METRES_PER_SECOND, horse.velocity.y * 0.12) +
          randf.inUnitSphere() * SCATTER_METRES_PER_SECOND,
        life: std.mix(LIFE_SECONDS[0], LIFE_SECONDS[1], randf.sample()),
        color: d.vec4f(0, 0, 0, OPACITY),
        size: SIZE_METRES * (0.7 + randf.sample() * 0.6),
        seed: randf.sample(),
      }),
    );
  });
  const kicking = root.createComputePipeline({ compute: kick });

  return {
    settings,
    drawing: dust.drawing,
    spawn: dust.spawn,
    compact: dust.compact,
    destroy() {
      dust.destroy();
    },
    initAsync: () => pipelines.compile([dust, kicking]),
    /** Emits a dust ring at a thunderclap. */
    clap(at: readonly [number, number, number]) {
      dust.burst({
        count: CLAP_PUFFS,
        position: [at[0], at[1] + 0.5, at[2]],
        area: [3, 0.3, 3],
        velocity: [0, 1.2, 0],
        scatter: CLAP_SPEED_METRES_PER_SECOND,
        color: [0, 0, 0, OPACITY * 2.2],
        size: SIZE_METRES * 1.6,
        life: LIFE_SECONDS[1],
      });
    },
    /** Steps dust and emits hoof puffs. Record after the herd step. */
    step(pass: TgpuComputePass) {
      dust.step(pass);
      pass.setPipeline(kicking);
      pass.dispatchWorkgroups(Math.ceil(herd.count / WORKGROUP));
    },
  };
}
