import { cameras, lighting } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';

/** A point across the sun's rays: along light space's `right` and `up`, in world units. */
export type Point = readonly [number, number];

export interface Cascade {
  /** Where its slice of the view starts and ends, in world units from the eye. */
  readonly near: number;
  readonly farther: number;
  /** The slice's eight corners. */
  readonly corners: readonly Point[];
  /** The centre of the square its layer covers, and the distance from there to each edge. */
  readonly x: number;
  readonly y: number;
  readonly radius: number;
  /** How many texels of its layer cross one world unit. */
  readonly density: number;
}

export interface Fit {
  readonly eye: Point;
  /** The width of each layer, in texels. */
  readonly size: number;
  readonly far: number;
  readonly cascades: readonly Cascade[];
}

export type Count = 1 | 2 | 3 | 4;

export interface Shape {
  readonly cascades: Count;
  /** Splits the view at equal steps instead of the shadow's own `(k / cascades)²`. */
  readonly even: boolean;
}

/** The screen's corners, as `ray` takes them. */
const CORNERS = [
  [-1, -1],
  [1, -1],
  [1, 1],
  [-1, 1],
] as const;

type Vec3 = ArrayLike<number>;
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** A level view's cascades as `lighting.fitSlices` places them, seen from a sun overhead. */
export function createProgram({ far = 120 } = {}) {
  const { root } = recordingRoot();
  const view = cameras.orbit(root, {
    size: { width: 16, height: 9 },
    target: [0, 1, 0],
    distance: 12,
    // Level, looking up the page
    yaw: Math.PI,
    pitch: 0,
  });
  view.update(0);

  let fitted: Fit | undefined;
  const ray = { origin: [0, 0, 0], direction: [0, 0, 0] } satisfies cameras.CpuRay;
  const axis = { origin: [0, 0, 0], direction: [0, 0, 0] } satisfies cameras.CpuRay;

  /** The shadow's own fit, kept as it placed the cascades. */
  const fit: lighting.CascadeFit = (input, places) => {
    if (!lighting.fitSlices(input, places)) return false;
    const { eye, right, up, splits, size } = input;
    eye.ray(0, 0, axis);

    /** The point `depth` from the eye along the view's axis, on the ray through a corner. */
    const corner = ([x, y]: Point, depth: number): Point => {
      eye.ray(x, y, ray);
      const along = depth / dot(ray.direction, axis.direction);
      const point = ray.origin.map((from, i) => from + ray.direction[i] * along);
      return [dot(right, point), dot(up, point)];
    };

    fitted = {
      eye: [dot(right, axis.origin), dot(up, axis.origin)],
      size,
      far: input.far,
      cascades: places.map(({ x, y, radius }, k) => {
        const near = input.far * splits[k];
        const farther = input.far * splits[k + 1];
        return {
          near,
          farther,
          corners: [near, farther].flatMap((depth) => CORNERS.map((at) => corner(at, depth))),
          x,
          y,
          radius,
          density: size / (radius * 2),
        };
      }),
    };
    return true;
  };

  const create = (shape: Shape, reach: number) =>
    lighting.create(root, {
      settings: { sunDirection: [0, 1, 0] },
      shadow: {
        eye: view,
        cascades: shape.cascades,
        far: reach,
        fit,
        ...(shape.even && {
          splits: Array.from({ length: shape.cascades - 1 }, (_, k) => (k + 1) / shape.cascades),
        }),
      },
    });

  let light = create({ cascades: 3, even: false }, far);

  return {
    /** The cascades as the shadow last placed them. */
    get fit() {
      if (!fitted) throw new Error('The shadow placed no cascades.');
      return fitted;
    },

    /** `cascades` and `splits` are options, so a new shape takes a new lighting. */
    shape(next: Shape) {
      const { far: reach } = light.shadow;
      light.destroy();
      light = create(next, reach);
    },

    /** `setFar` fits the cascades again. */
    far(next: number) {
      light.shadow.setFar(next);
    },

    destroy() {
      light.destroy();
    },
  };
}
