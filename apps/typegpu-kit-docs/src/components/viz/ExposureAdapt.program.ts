import { clock, exposure, tone } from '@typegpu/kit';
import { linearToSrgb } from '@typegpu/color';
import { d } from 'typegpu';
import { clamp } from '../home/geometry.ts';

export type Settings = d.Infer<typeof exposure.Settings>;

/** The average luminance of the picture under each light the figure switches between. */
export const LIGHTS = { dark: 0.045, bright: 0.72 } as const;

export type Light = keyof typeof LIGHTS;

/** What the final pass shows of grey `light`, from 0 to 1: its default tone curve, encoded. */
export const screen = (light: number) => clamp(linearToSrgb(tone.neutral(d.vec3f(light))).x);

/**
 * The exposure of an `exposure.create`, moved on the CPU as its meter moves it on the GPU, for a
 * picture whose average luminance the caller gives.
 */
export function createProgram() {
  /** Patched as `eye.settings` is. */
  const settings: Settings = { ...exposure.defaults };
  // 0 until the first record, which then takes the target at once
  let adapted = 0;

  /** `key` over the average, within `lowest` and `highest`, in stops. */
  const goal = (average: number) =>
    clamp(
      Math.log2(settings.key) - Math.log2(average),
      Math.log2(settings.lowest),
      Math.log2(settings.highest),
    );

  return {
    settings,

    /** The adapted exposure, 0 before the first `record`. */
    get exposure() {
      return adapted;
    },

    /** The exposure a picture of `average` luminance settles at. */
    target: (average: number) => 2 ** goal(average),

    /** Moves the exposure towards its target by `dt` seconds of adaptation, and returns it. */
    record(average: number, dt: number) {
      const target = goal(average);
      let next = target;
      if (adapted > 0) {
        const from = Math.log2(adapted);
        const rate = target > from ? settings.rise : settings.fall;
        next = from + (target - from) * clock.ease(dt, rate);
      }
      adapted = 2 ** next;
      return adapted;
    },
  };
}
