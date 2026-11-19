import { environment, lighting, particles, table, type CameraBase } from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import { srgbToLinear } from '@typegpu/color';
import { d, std, type TgpuRoot } from 'typegpu';

type Vec3 = [number, number, number];

export const cube = meshes.box();

export const STAGE = {
  settings: {
    // A cool fill from high behind the well, its rim light and shadow
    sunDirection: [0.35, 0.9, -0.75] as Vec3,
    sunIntensity: 1.2,
    sunColor: [0.55, 0.65, 1] as Vec3,
    skyTop: [0.05, 0.065, 0.14] as Vec3,
    skyBottom: [0.03, 0.03, 0.045] as Vec3,
    skyIntensity: 1,
    skyReflection: 1,
    fogDensity: 0,
  },
  plinth: srgbToLinear(d.vec3f(0.09, 0.085, 0.11)),
  finish: { exposure: 1.45, vignette: 0.3 },
  bloom: { threshold: 1.4 },
};

/** Gradient colours, from looking straight down to above the horizon. */
const DEEP = srgbToLinear(d.vec3f(0.05, 0.06, 0.08));
const HIGH = srgbToLinear(d.vec3f(0.13, 0.15, 0.2));

/** The dark gradient the well floats in. */
const gradient = (direction: d.v3f) => {
  'use gpu';
  return std.mix(DEEP, HIGH, std.smoothstep(-0.9, 0.5, direction.y));
};

export const backdrop = environment.background(gradient);

/** The warm key, from high at the front left, out of view, and where it aims, low in the well. */
const KEY = {
  position: [-9, 15, 14] as Vec3,
  aim: [3, 3.5, 3] as Vec3,
  color: [1, 0.68, 0.42] as Vec3,
  intensity: 320,
  range: 45,
  cone: lighting.cone(0.55, 0.85),
};

/** Embers drifting up through the space around the well, from fires out of view below. */
const EMBERS = {
  position: [3, -3, 3],
  area: [7, 4, 7],
  velocity: [0, 1.1, 0],
  scatter: 0.35,
  color: [3, 1.1, 0.25, 1],
  size: 0.05,
  life: 14,
  variance: 0.5,
} as const;

/**
 * The well's light, a warm spot out of view, and the embers drifting past it, stepped every `step`
 * seconds.
 */
export function createStage(root: TgpuRoot, camera: CameraBase, options: { step: number }) {
  const { step } = options;
  const lamps = table.create(root, lighting.SpotLight, {
    capacity: 1,
    defaults: { ...lighting.WHITE_SPOT },
  });
  const toward = KEY.aim.map((at, axis) => at - KEY.position[axis]);
  const length = Math.hypot(...toward);
  const key = lamps.add({
    position: KEY.position,
    aim: d.vec3f(...(toward.map((v) => v / length) as Vec3)),
    cone: KEY.cone,
    color: KEY.color,
    intensity: KEY.intensity,
    range: KEY.range,
  });
  lamps.flush();

  const embers = particles.create(root, camera, {
    capacity: 128,
    step,
    shade: particles.glowingDisc,
    label: 'embers',
    settings: { gravity: [0, 0, 0], drag: 0 },
  });
  // The space starts with embers already risen through it, then more drift up from below
  embers.burst({ ...EMBERS, count: 48, position: [3, 5, 3], area: [7, 8, 7], aged: true });
  embers.stream(4, { ...EMBERS });

  return {
    lamps,
    key,
    embers,

    destroy() {
      lamps.destroy();
      embers.destroy();
    },
  };
}
