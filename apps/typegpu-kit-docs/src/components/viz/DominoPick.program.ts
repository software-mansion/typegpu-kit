import { cameras } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';
import { LAYOUTS } from 'typegpu-kit-examples/dominoes/dominoes.ts';
import { HEIGHT } from 'typegpu-kit-examples/dominoes/layout.ts';
import { VIEWS } from 'typegpu-kit-examples/dominoes/scene.ts';

/** Where a domino's middle lands, in points from the view's top left, and its radius there. */
export interface Landed {
  readonly x: number;
  readonly y: number;
  readonly radius: number;
}

export interface Pick {
  /** The dominoes within reach of the pointer, by index. */
  readonly within: readonly number[];
  /** The one the picker keeps, or undefined. */
  readonly nearest: number | undefined;
  /** How far its middle lands from the pointer, in points. */
  readonly gap: number;
}

export interface Options {
  /** Radians the camera looks down. Default the example's, 0.38. */
  readonly pitch?: number;
  /** The view's size in points. Default 640 by 400. */
  readonly width?: number;
  readonly height?: number;
}

/** The world units around a domino's middle that count as the domino: half its height. */
const RADIUS = HEIGHT / 2;

export function createProgram({
  pitch = VIEWS.run.camera.pitch,
  width = 640,
  height = 400,
}: Options = {}) {
  const { root } = recordingRoot();
  const { camera } = VIEWS.run;
  const view = cameras.orbit(root, {
    size: { width, height },
    ...camera,
    target: [...camera.target],
    pitch,
    fov: 0.6,
    near: 2,
    far: 4000,
  });

  const onScreen = [0, 0, 0];
  /** Where `point` lands, in points; undefined behind the eye. */
  const land = (point: readonly [number, number, number]) => {
    if (!view.projectAt(point, onScreen)) return undefined;
    return { x: onScreen[0] * width, y: onScreen[1] * height };
  };
  const right = [Math.cos(camera.yaw), 0, -Math.sin(camera.yaw)] as const;

  const middles: Landed[] = [];
  for (const each of LAYOUTS.run) {
    const middle = land([each.x, HEIGHT / 2, each.z]);
    const side = land([each.x + right[0] * RADIUS, HEIGHT / 2, each.z + right[2] * RADIUS]);
    if (!middle || !side) continue;
    middles.push({ ...middle, radius: Math.hypot(side.x - middle.x, side.y - middle.y) });
  }

  return {
    middles,

    /** What a pick at `x, y`, in points, finds within `reach` points of each domino's circle. */
    pick(x: number, y: number, reach: number): Pick {
      const within: number[] = [];
      let nearest: number | undefined;
      let gap = Infinity;
      middles.forEach((middle, i) => {
        const away = Math.hypot(middle.x - x, middle.y - y);
        if (away > reach + middle.radius) return;
        within.push(i);
        if (away < gap) {
          gap = away;
          nearest = i;
        }
      });
      return { within, nearest, gap };
    },

    destroy() {},
  };
}

export type Program = ReturnType<typeof createProgram>;
