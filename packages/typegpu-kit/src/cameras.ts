/**
 * Provides orbit, 2D, custom and layered cameras through the `cameras.current` accessor. Orbit and 2D
 * cameras write their uniforms only when the view or size changes, and once more to update the
 * previous view. Custom and layered cameras take matrices you write. Orbit, 2D and custom cameras
 * keep the previous frame's view for code compiled with `history.previously`.
 *
 * @module
 * @injects hook `cameras.current`, for every shader that projects, and `cameras.reversed` from a
 * camera created with `reversed: true`
 * @binds 1 uniform: `cameras.current.$`, the camera's matrices
 * @binds 1 uniform: `cameras.current.$` reads the previous matrices in code compiled with `history.previously`
 * @binds 1 uniform: `cameras.layered`, every layer's view, beside the layer its description binds
 * @owns 2 buffers: the current and previous camera uniforms. Writes when the view or size
 * changes and once more in the next frame. `cameras.layered` owns 1 buffer, writing a layer's view
 * when it changes
 * @records `update(dt)` once a frame, to ease the view towards its goal, or `jump()` to go there.
 */

import { d, std, tgpu, type TgpuRoot, type TgpuUniform } from 'typegpu';
import { mat4, type Mat4Arg } from 'wgpu-matrix';
import * as clock from './clock.ts';
import { temporal } from './history.ts';
import { unfilled, type Inject } from './inject.ts';
import { layer as drawn, layerCount } from './passes.ts';
import type { SizeSource } from './targets.ts';

export const Camera = d.struct({
  /** Transforms world coordinates to clip space. */
  viewProjection: d.mat4x4f,
  /** Gives the eye position in world space. */
  position: d.vec3f,
  /**
   * Gives the unit x and y axes of the view in world space, for quads that face the screen. With
   * the viewing direction from `forwardOf`, they form a right-handed basis.
   */
  right: d.vec3f,
  up: d.vec3f,
});

/** Provides the camera for a pipeline. A camera's `inject()` fills it with the camera uniform. */
export const current = tgpu.accessor(
  Camera,
  unfilled(
    '`cameras.current`',
    'view',
    'a `cameras.orbit`, `cameras.flat`, `cameras.custom` or `cameras.layered`',
  ),
);

/** The camera a shader reads, as `cameras.current.$` gives it. */
export type Camera = d.InferGPU<typeof Camera>;

/**
 * Returns a world point in clip space, as a vertex shader's position. It reads `cameras.current`;
 * code that projects with a view of its own multiplies by that view's `viewProjection`.
 */
export const toClip = (point: d.v3f) => {
  'use gpu';
  return current.$.viewProjection * d.vec4f(point, 1);
};

/**
 * Holds whether the camera's projection is reversed, with depth 1 at the near plane and 0 at the
 * far plane. The `inject()` of a camera created with `reversed: true` fills it, so code that reads
 * the depth the camera drew, through `farDepth` and `nearerDepth`, follows the camera. False
 * elsewhere.
 */
export const reversed = tgpu.slot(false);

/** Returns the depth of the far plane, where nothing drew: 1, or 0 under a reversed camera. */
export const farDepth = tgpu.comptime((): number => (reversed.$ ? 0 : 1));

/**
 * Returns the function that keeps the nearer of two depths: `std.min`, or `std.max` under a
 * reversed camera. Call it as `cameras.nearerDepth()(a, b)`.
 */
export const nearerDepth = tgpu.comptime((): typeof std.min => (reversed.$ ? std.max : std.min));

/** Returns the unit viewing direction in world space. Pass `cameras.current.$` as `view`. */
export const forwardOf = (view: Camera) => {
  'use gpu';
  return std.cross(view.up, view.right);
};

/** Returns the x and y projection scales from the lengths of the view-projection matrix rows. */
export const lensOf = (view: Camera) => {
  'use gpu';
  const m = view.viewProjection;
  return d.vec2f(
    std.length(d.vec3f(m.columns[0].x, m.columns[1].x, m.columns[2].x)),
    std.length(d.vec3f(m.columns[0].y, m.columns[1].y, m.columns[2].y)),
  );
};

/**
 * Returns the distance along the viewing direction from the eye to the point a depth value holds,
 * for perspective and orthographic views. Read the depth from the target the view drew, such as
 * with `textureLoad`, and pass `cameras.current.$` as `view`.
 */
export const distanceOf = (view: Camera, depth: number) => {
  'use gpu';
  const eye = view.viewProjection * d.vec4f(view.position, 1);
  const ahead = view.viewProjection * d.vec4f(forwardOf(view), 0);
  return (eye.z - depth * eye.w) / (depth * ahead.w - ahead.z);
};

export const Ray = d.struct({
  origin: d.vec3f,
  /** Gives the unit ray direction. */
  direction: d.vec3f,
});

const isPerspective = (view: Camera) => {
  'use gpu';
  const m = view.viewProjection;
  const w = d.vec3f(m.columns[0].w, m.columns[1].w, m.columns[2].w);
  return std.dot(w, w) > 0;
};

/**
 * Returns the ray through `ndc`, with coordinates from -1 to 1 and +y up. Perspective rays start
 * at the eye. Orthographic rays, such as those from `cameras.flat`, are parallel and start at
 * their screen points at the depth of the eye. To go back to the screen, multiply a point on the
 * ray by `viewProjection` and divide by `w`.
 */
export const rayOf = (view: Camera, ndc: d.v2f) => {
  'use gpu';
  const at = ndc / lensOf(view);
  const across = view.right * at.x + view.up * at.y;
  const forward = forwardOf(view);
  const perspective = isPerspective(view);

  return Ray({
    origin: std.select(view.position + across, view.position, perspective),
    direction: std.select(forward, std.normalize(forward + across), perspective),
  });
};

/**
 * Returns the unit direction towards the viewer from a world point, for shading. Under perspective
 * projection, it points towards the eye. Under orthographic projection, such as a flat camera or
 * the sun, it points against the viewing direction.
 */
export const towards = (view: Camera, point: d.v3f) => {
  'use gpu';
  if (isPerspective(view)) return std.normalize(view.position - point);
  // Reversing the cross product gives the direction opposite to the view
  return std.cross(view.right, view.up);
};

const within = (plane: d.v4f, point: d.v4f, radius: number) => {
  'use gpu';
  return std.dot(plane, point) >= -radius * std.length(plane.xyz);
};

/**
 * Tests whether a world-space sphere may be visible. Returns false only when the whole sphere lies
 * outside one of the six view planes. A sphere beyond a corner can still return true. Use it for
 * culling, such as in live lists' `alive` function with the camera piped in: `sees(cameras.current.$,
 * center, radius)`.
 */
export const sees = (view: Camera, center: d.v3f, radius: number) => {
  'use gpu';
  // Clip bounds are -w ≤ x ≤ w, -w ≤ y ≤ w and 0 ≤ z ≤ w. Adding or subtracting view-projection
  // rows gives the corresponding world-space planes
  const rows = std.transpose(view.viewProjection);
  const x = rows.columns[0];
  const y = rows.columns[1];
  const z = rows.columns[2];
  const w = rows.columns[3];
  const point = d.vec4f(center, 1);

  return (
    within(w + x, point, radius) &&
    within(w - x, point, radius) &&
    within(w + y, point, radius) &&
    within(w - y, point, radius) &&
    within(z, point, radius) &&
    within(w - z, point, radius)
  );
};

/** Receives three numbers from a CPU call, such as a point or direction. */
export type Out3 = Float32Array | number[];

/** A ray on the CPU, as a view's `rayAt` writes it and `hitHeight` reads it. */
export interface CpuRay<V extends ArrayLike<number> = Out3> {
  readonly origin: V;
  /** Gives the unit ray direction. */
  readonly direction: V;
}

/**
 * Provides CPU access to the last view written to a camera uniform. Use it for work that must
 * match the rendered frame, such as placing a label over a point or testing a ray against a board.
 */
export interface CpuView {
  /**
   * Projects a world point to screen coordinates. Writes x and y from -1 to 1 with +y up, as
   * `rayOf` takes them, and depth from 0 to 1 in z. Returns false without changing `out` for a
   * point behind the eye or before the first write.
   */
  project(point: ArrayLike<number>, out: Out3): boolean;
  /**
   * Projects a world point to a point of the surface, as `rayAt` takes it. Writes x and y from 0 to
   * 1 across the surface with +y down, as pointer handlers receive them, and depth from 0 to 1 in
   * z. Returns false without changing `out` for a point behind the eye or before the first write.
   */
  projectAt(point: ArrayLike<number>, out: Out3): boolean;
  /**
   * Writes the ray through the screen point `x`, `y` into `out`, as `rayOf` computes it on the GPU.
   * Coordinates range from -1 to 1 with +y up.
   */
  ray(x: number, y: number, out: CpuRay): void;
  /**
   * Writes the ray through a point of the surface into `out`, as `ray` does. `x` and `y` range from
   * 0 to 1 across the surface with +y down, as pointer handlers receive them.
   */
  rayAt(x: number, y: number, out: CpuRay): void;
}

/** Provides the uniform and fill that a module needs to draw with a camera. */
export interface CameraBase {
  /** Holds the camera view from creation onwards. */
  readonly uniform: TgpuUniform<typeof Camera>;
  /** Fills `cameras.current` for the pipelines created from the configuration it is piped into. */
  inject(): Inject;
}

/**
 * Keeps the previous view that `inject` supplies to `cameras.current` in code compiled with
 * `history.previously`.
 */
interface LastFrame {
  /** Holds the previous frame's view for motion vectors and reprojection. */
  readonly previous: TgpuUniform<typeof Camera>;
}

export interface CustomCamera extends CameraBase, CpuView, LastFrame {
  /**
   * Writes the camera from a view matrix and a projection matrix. The view matrix, such as one
   * from `mat4.lookAt`, supplies the eye position and screen axes. Moves the current view to
   * `previous`. Call once per frame, even when the view is still. Repeated writes of the same view
   * write nothing once `previous` matches it. Throws for a view that mirrors the world. Mirror the
   * image too, with `mat4.multiply(mat4.scaling([-1, 1, 1]), view)`. Write a reversed projection,
   * such as one from `mat4.perspectiveReverseZ`, into a camera created with `reversed: true`.
   */
  write(view: Mat4Arg, projection: Mat4Arg): void;
}

/** Configures a camera's depth. */
export interface DepthOptions {
  /**
   * Projects reversed depth, 1 at the near plane and 0 at the far plane, for a description created
   * with `depthClear: 0` and `depthCompare: 'greater'`. `inject()` fills `cameras.reversed` with
   * it. Default false.
   */
  readonly reversed?: boolean;
}

export interface OrbitOptions extends DepthOptions {
  /** Supplies the size used for the aspect ratio, such as the canvas surface. */
  readonly size: SizeSource;
  /**
   * Sets the initial point that the camera orbits and looks at. Change it through `goal.target`.
   */
  readonly target: readonly [number, number, number];
  /**
   * Sets the initial distance from the target to the eye, in world units. Change it through
   * `goal.distance`.
   */
  readonly distance: number;
  /**
   * Sets the initial yaw in radians around +y. At 0, the camera is on +z looking towards -z.
   * Default 0.
   */
  readonly yaw?: number;
  /** Sets the initial pitch in radians above the horizon. Default 0.6. */
  readonly pitch?: number;
  /**
   * Sets the initial vertical field of view in radians. Change it through `goal.fov`. Default 0.8.
   */
  readonly fov?: number;
  /**
   * Sets the initial distance from the eye to the near clipping plane, in world units. Change it
   * through `clip.near`. Default 0.1.
   */
  readonly near?: number;
  /**
   * Sets the initial distance from the eye to the far clipping plane, in world units. Change it
   * through `clip.far`. Default 500.
   */
  readonly far?: number;
  /** Limits the goal pitch to this range, in radians. Default -1.5 to 1.5. */
  readonly pitchRange?: readonly [number, number];
  /**
   * Limits the goal distance to this range, in world units. Default `near * 2` to `far / 2`, using
   * the option values.
   */
  readonly distanceRange?: readonly [number, number];
  /**
   * Returns the lowest height the eye may reach at a point x, z, in world units, such as a
   * terrain's height plus a margin. `update` and `jump` raise the goal pitch, within `pitchRange`,
   * until the eye is at or above it. Default no floor.
   * @cpu
   */
  readonly floor?: (x: number, z: number) => number;
  /**
   * Returns how far the eye may go from `target` along a yaw and pitch, such as the distance to a
   * room's wall less a margin. `update` and `jump` place the eye no farther, leaving
   * `goal.distance` as the viewer zoomed. Default no limit.
   * @cpu
   */
  readonly reach?: (target: ArrayLike<number>, yaw: number, pitch: number) => number;
  /**
   * Controls how fast the view approaches the goal. The fraction remaining after one second is
   * e^-damping. Default 14.
   */
  readonly damping?: number;
}

export interface Orbit {
  yaw: number;
  pitch: number;
  distance: number;
  /** Sets the vertical field of view in radians. */
  fov: number;
  /** Sets the point that the camera orbits and looks at. */
  target: [number, number, number];
}

export interface OrbitCamera extends CameraBase, CpuView, LastFrame {
  /** Sets the desired camera pose. `update` eases the view towards it. */
  readonly goal: Orbit;
  /** Reports the eased pose, the view as drawn while it approaches the goal. */
  readonly pose: {
    readonly yaw: number;
    readonly pitch: number;
    readonly distance: number;
    readonly fov: number;
    readonly target: readonly [number, number, number];
  };
  /**
   * Sets the distances from the eye to the clipping planes, in world units. The next `update` or
   * `jump` applies them without easing.
   */
  readonly clip: { near: number; far: number };
  /** Reports the width divided by the height of the current view, from `size` at the last move. */
  readonly aspect: number;
  /**
   * Eases towards the goal by `dt` seconds and writes the uniform if the view changed. Call once
   * per frame.
   */
  update(dt: number): void;
  /** Moves the view to the goal at once and writes the uniform if the view changed. Call it instead of `update`. */
  jump(): void;
}

export interface FlatOptions extends DepthOptions {
  /**
   * Supplies the size used for the aspect ratio, such as the canvas surface or a low-resolution
   * target.
   */
  readonly size: SizeSource;
  /**
   * Sets the initial view height in world units, required without `fit`. Change it through
   * `goal.height`. Default the height `fit` gives.
   */
  readonly height?: number;
  /**
   * Fits a `width × height` world rectangle centred on the view: `cover` fills the view with it,
   * cropping its longer side, and `contain` shows all of it, leaving bands. Sets `goal.height` at
   * once at creation and whenever the aspect changes. Default none.
   */
  readonly fit?: {
    readonly width: number;
    readonly height: number;
    readonly mode: 'cover' | 'contain';
  };
  /** Sets the initial view center in world units with +y up. Default the origin. */
  readonly center?: readonly [number, number];
  /**
   * Rounds the view center to multiples of this distance in world units, such as one texel of a
   * pixel-art target. Default 0, which disables rounding.
   */
  readonly snap?: number;
  /**
   * Controls how fast the view approaches the goal. The fraction remaining after one second is
   * e^-damping. Default 8.
   */
  readonly damping?: number;
}

export interface FlatCamera extends CameraBase, CpuView, LastFrame {
  /**
   * Sets the desired center in world units with +y up, and the desired view height. A smaller
   * `height` zooms in.
   */
  readonly goal: { x: number; y: number; height: number };
  /**
   * Reports the eased center and height, the view as drawn while it approaches the goal. The center
   * is the value before `snap` rounds it.
   */
  readonly pose: { readonly x: number; readonly y: number; readonly height: number };
  /** Reports the width divided by the height of the current view, from `size` at the last move. */
  readonly aspect: number;
  /**
   * Eases towards the goal by `dt` seconds and writes the uniform if the view changed. Call once
   * per frame.
   */
  update(dt: number): void;
  /** Moves the view to the goal at once and writes the uniform if the view changed. Call it instead of `update`. */
  jump(): void;
}

const ORBIT_KEYS = ['yaw', 'pitch', 'distance', 'fov'] as const;
const ANGLE_KEYS = ['yaw', 'pitch', 'fov'] as const;
const DISTANCE_KEY = ['distance'] as const;
const TARGET_KEYS = [0, 1, 2] as const;
const TURN = Math.PI * 2;
/** Halvings of the pitch range when finding the lowest pitch that clears a floor. */
const FLOOR_STEPS = 16;
const FLAT_KEYS = ['x', 'y', 'height'] as const;
const WORLD_UP = [0, 1, 0] as const;

const clamp = (value: number, [low, high]: readonly [number, number]) =>
  Math.min(high, Math.max(low, value));

function ease<K extends PropertyKey>(
  state: Record<K, number>,
  goal: Readonly<Record<K, number>>,
  keys: readonly K[],
  amount: number,
) {
  let moving = false;

  for (const key of keys) {
    const gap = goal[key] - state[key];
    state[key] = Math.abs(gap) < 1e-5 ? goal[key] : state[key] + gap * amount;
    moving ||= gap !== 0;
  }

  return moving;
}

const clipRow = (m: Float32Array, row: number, x: number, y: number, z: number) =>
  m[row] * x + m[4 + row] * y + m[8 + row] * z + m[12 + row];

const scaleOf = (m: Float32Array, row: number) =>
  Math.sqrt(m[row] * m[row] + m[4 + row] * m[4 + row] + m[8 + row] * m[8 + row]);

/**
 * Creates a camera from view and projection matrices that you write. Use it for a follow camera, a
 * scripted camera or another rig. It fills `cameras.current` and holds zeros until the first `write`.
 *
 * @returns a camera you write
 */
export function custom(root: TgpuRoot, options: DepthOptions = {}): CustomCamera {
  return lens(root, options.reversed ?? false);
}

/** Tells whether the rotation of a column-major view matrix has a negative determinant. */
const mirrors = (m: Mat4Arg) =>
  m[0] * (m[5] * m[10] - m[6] * m[9]) -
    m[4] * (m[1] * m[10] - m[2] * m[9]) +
    m[8] * (m[1] * m[6] - m[2] * m[5]) <
  0;

/** Writes the `Camera` fields of a view and projection into `out`, throwing for a mirrored view. */
const viewed = (view: Mat4Arg, projection: Mat4Arg, out: Float32Array) => {
  if (mirrors(view)) {
    throw new Error(
      'The view matrix mirrors the world, which turns `forwardOf`, `rayOf` and `towards` around. Mirror the image too: write `mat4.multiply(mat4.scaling([-1, 1, 1]), view)` and read the result with u mirrored.',
    );
  }
  mat4.multiply(projection, view, out.subarray(0, 16));

  // The view matrix is column-major. Its rows give the screen axes, and its transposed rotation
  // maps the negative translation back to the eye
  for (let i = 0; i < 3; i++) {
    out[20 + i] = view[i * 4];
    out[24 + i] = view[i * 4 + 1];
    out[16 + i] = -(
      view[12] * view[i * 4] +
      view[13] * view[i * 4 + 1] +
      view[14] * view[i * 4 + 2]
    );
  }
};

const equal = (a: Float32Array, b: Float32Array) => {
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
};

function lens(root: TgpuRoot, reversedDepth: boolean): CustomCamera & { settle(): void } {
  const uniform = root.createUniform(Camera);
  const previous = root.createUniform(Camera);
  const bytes = new Float32Array(d.sizeOf(Camera) / 4);
  const viewProjection = bytes.subarray(0, 16);
  const position = bytes.subarray(16, 19);
  const right = bytes.subarray(20, 23);
  const up = bytes.subarray(24, 27);
  const forward = new Float32Array(3);
  const next = new Float32Array(bytes.length);

  let written = false;
  // Avoid copying a still view once both uniforms match
  let settled = false;

  const filled: Inject = (cfg) => cfg.with(current, temporal(uniform, previous));
  const inject: Inject = reversedDepth ? (cfg) => filled(cfg).with(reversed, true) : filled;
  const settle = () => {
    if (settled) return;
    previous.buffer.write(bytes.buffer);
    settled = true;
  };

  const ray: CpuView['ray'] = (x, y, { origin, direction }) => {
    const atX = x / scaleOf(viewProjection, 0);
    const atY = y / scaleOf(viewProjection, 1);
    const perspective =
      viewProjection[3] !== 0 || viewProjection[7] !== 0 || viewProjection[11] !== 0;

    forward[0] = up[1] * right[2] - up[2] * right[1];
    forward[1] = up[2] * right[0] - up[0] * right[2];
    forward[2] = up[0] * right[1] - up[1] * right[0];

    let length = 0;
    for (let i = 0; i < 3; i++) {
      const across = right[i] * atX + up[i] * atY;
      origin[i] = perspective ? position[i] : position[i] + across;
      direction[i] = perspective ? forward[i] + across : forward[i];
      length += direction[i] * direction[i];
    }

    length = Math.sqrt(length);
    for (let i = 0; i < 3; i++) direction[i] = direction[i] / length;
  };

  const project: CpuView['project'] = (point, out) => {
    const x = point[0] ?? 0;
    const y = point[1] ?? 0;
    const z = point[2] ?? 0;

    const w = clipRow(viewProjection, 3, x, y, z);
    if (!(w > 0)) return false;

    out[0] = clipRow(viewProjection, 0, x, y, z) / w;
    out[1] = clipRow(viewProjection, 1, x, y, z) / w;
    out[2] = clipRow(viewProjection, 2, x, y, z) / w;
    return true;
  };

  return {
    uniform,
    previous,

    inject: () => inject,

    write(view, projection) {
      viewed(view, projection, next);

      if (written) {
        if (equal(next, bytes)) return settle();
        previous.buffer.write(bytes.buffer);
      }
      bytes.set(next);
      uniform.buffer.write(bytes.buffer);
      settled = false;
      // Matching both views on the first frame prevents false motion
      if (!written) settle();
      written = true;
    },

    settle,

    project,

    projectAt(point, out) {
      if (!project(point, out)) return false;
      out[0] = (out[0] + 1) / 2;
      out[1] = (1 - out[1]) / 2;
      return true;
    },

    rayAt: (x, y, out) => ray(x * 2 - 1, 1 - y * 2, out),

    ray,
  };
}

/** Holds a view per layer of a description drawn into layers, such as a shadow cascade each. */
export interface LayeredCamera {
  /** Reports the number of views. */
  readonly layers: number;
  /** Holds the views, one `Camera` per layer, zeros until written. */
  readonly uniform: TgpuUniform<d.WgslArray<typeof Camera>>;
  /**
   * Fills `cameras.current` with the view of the layer drawn, read through `passes.layer`. Pipe it into a
   * description of targets with as many `layers`; a pipeline of a description drawn into another
   * number of layers throws as it builds. In other pipelines `cameras.current.$` is view 0.
   */
  inject(): Inject;
  /**
   * Writes the view of `layer` from a view matrix and a projection matrix, as a custom camera's
   * `write` does. Writes nothing when the layer's view is unchanged. Throws for a view that mirrors
   * the world and for a layer out of range.
   */
  write(layer: number, view: Mat4Arg, projection: Mat4Arg): void;
}

/**
 * Creates a view for each layer of a description drawn into layers, in one uniform array. It
 * keeps no previous views, so `cameras.current.$` reads the current view in code compiled with
 * `history.previously`.
 *
 * @param root the root that owns the uniform
 * @param count the number of layers, the `layers` of the targets the description draws into
 * @returns views you write
 */
export function layered(root: TgpuRoot, count: number): LayeredCamera {
  if (!Number.isInteger(count) || count < 1) {
    throw new RangeError(`cameras.layered needs a whole number of layers from 1, got ${count}.`);
  }
  const uniform = root.createUniform(d.arrayOf(Camera, count));
  const stride = d.sizeOf(Camera);
  const bytes = new Float32Array((stride / 4) * count);
  const next = new Float32Array(stride / 4);

  const view = tgpu
    .fn(
      [],
      Camera,
    )(() => Camera(uniform.$[drawn.$]))
    .$name('layer view');

  // Runs while each pipeline resolves, inside the fills of the description it belongs to
  const checked = () => {
    const described = layerCount.$;
    if (described > 1 && described !== count) {
      throw new Error(
        `cameras.layered holds ${count} views, but the description it was piped into draws ${described} layers.`,
      );
    }
    return view;
  };
  // A plain function filling an accessor runs at resolution, as `byStage` does
  const inject: Inject = (cfg) => cfg.with(current, checked as unknown as typeof view);

  return {
    layers: count,
    uniform,

    inject: () => inject,

    write(layer, matrix, projection) {
      if (!Number.isInteger(layer) || layer < 0 || layer >= count) {
        throw new RangeError(`cameras.layered holds ${count} views, so it has no layer ${layer}.`);
      }
      viewed(matrix, projection, next);
      const kept = bytes.subarray((layer * stride) / 4, ((layer + 1) * stride) / 4);
      if (equal(next, kept)) return;
      kept.set(next);
      root.device.queue.writeBuffer(root.unwrap(uniform.buffer), layer * stride, kept);
    },
  };
}

export interface TurnOptions {
  /** Sets the radians a dragged pixel or point turns. Default 0.008. */
  readonly sensitivity?: number | undefined;
}

/**
 * Turns an orbit's goal by a drag: a drag to the right turns the eye left around the target, and a
 * drag down raises it. `web.orbitControls` turns by it.
 *
 * @param goal an orbit camera's goal, whose yaw and pitch it writes
 * @param dx the drag to the right, in pixels or points
 * @param dy the drag down, in pixels or points
 * @returns `goal`
 */
export function turn<G extends Pick<Orbit, 'yaw' | 'pitch'>>(
  goal: G,
  dx: number,
  dy: number,
  options: TurnOptions = {},
): G {
  const { sensitivity = 0.008 } = options;
  goal.yaw -= dx * sensitivity;
  goal.pitch += dy * sensitivity;
  return goal;
}

export interface PanOptions {
  /** Sets the view's height in the units of the drag, one camera distance of panning. */
  readonly height: number;
}

/**
 * Pans an orbit's goal by a drag: the target moves across the view against the drag, one camera
 * distance per `height` dragged, so the scene follows the pointer. `web.orbitControls` pans by it.
 *
 * @param goal an orbit camera's goal, whose target it writes
 * @param dx the drag to the right, in pixels or points
 * @param dy the drag down, in pixels or points
 * @returns `goal`
 */
export function pan<G extends Pick<Orbit, 'yaw' | 'pitch' | 'distance' | 'target'>>(
  goal: G,
  dx: number,
  dy: number,
  options: PanOptions,
): G {
  const { yaw, pitch, target } = goal;
  const scale = goal.distance / options.height;
  const across = -dx * scale;
  const up = dy * scale;
  target[0] += across * Math.cos(yaw) - up * Math.sin(pitch) * Math.sin(yaw);
  target[1] += up * Math.cos(pitch);
  target[2] -= across * Math.sin(yaw) + up * Math.sin(pitch) * Math.cos(yaw);
  return goal;
}

/**
 * Writes the eye position of an orbit pose into `out`. Pass an orbit camera's `goal` or `pose`,
 * for example to place a sound at the eye.
 *
 * @param pose an orbit's yaw, pitch, distance and target
 * @param out where the eye goes
 * @returns `out`
 */
export function eyeOf(
  pose: Pick<Orbit, 'yaw' | 'pitch' | 'distance'> & { readonly target: ArrayLike<number> },
  out: Out3,
): Out3 {
  const { yaw, pitch, distance, target } = pose;
  out[0] = target[0] + distance * Math.cos(pitch) * Math.sin(yaw);
  out[1] = target[1] + distance * Math.sin(pitch);
  out[2] = target[2] + distance * Math.cos(pitch) * Math.cos(yaw);
  return out;
}

/**
 * Creates a perspective camera that orbits a target and eases towards a goal set by input.
 *
 * @returns a camera you update every frame
 */
export function orbit(root: TgpuRoot, options: OrbitOptions): OrbitCamera {
  const { size, near = 0.1, far = 500, damping = 14 } = options;
  const { pitchRange = [-1.5, 1.5], distanceRange = [near * 2, far / 2], floor, reach } = options;

  const eyes = lens(root, options.reversed ?? false);
  const perspective = options.reversed ? mat4.perspectiveReverseZ : mat4.perspective;
  const eye = new Float32Array(3);
  const view = mat4.create();
  const projection = mat4.create();

  const [x, y, z] = options.target;
  const goal: Orbit = {
    yaw: options.yaw ?? 0,
    pitch: options.pitch ?? 0.6,
    distance: options.distance,
    fov: options.fov ?? 0.8,
    target: [x, y, z],
  };
  const lifted: Out3 = [0, 0, 0];
  const clears = (pitch: number) => {
    goal.pitch = pitch;
    eyeOf(goal, lifted);
    return lifted[1] >= floor!(lifted[0], lifted[2]);
  };
  const clampGoal = () => {
    goal.pitch = clamp(goal.pitch, pitchRange);
    goal.distance = clamp(goal.distance, distanceRange);
    if (!floor) return;
    let [low, high] = [goal.pitch, pitchRange[1]];
    if (clears(low) || !clears(high)) return;
    for (let step = 0; step < FLOOR_STEPS; step++) {
      const middle = (low + high) / 2;
      if (clears(middle)) high = middle;
      else low = middle;
    }
    clears(high);
  };
  clampGoal();
  const pose: Orbit = { ...goal, target: [x, y, z] };
  const holdReach = () => {
    if (reach) pose.distance = Math.min(pose.distance, reach(pose.target, pose.yaw, pose.pitch));
  };
  holdReach();
  const clip = { near, far };
  const placed = { near, far };
  let aspect = size.width / size.height;

  const place = () => {
    mat4.lookAt(eyeOf(pose, eye), pose.target, WORLD_UP, view);
    perspective(pose.fov, aspect, clip.near, clip.far, projection);
    eyes.write(view, projection);
    placed.near = clip.near;
    placed.far = clip.far;
  };

  place();

  return {
    uniform: eyes.uniform,
    previous: eyes.previous,
    inject: eyes.inject,
    project: eyes.project,
    projectAt: eyes.projectAt,
    ray: eyes.ray,
    rayAt: eyes.rayAt,
    goal,
    pose,
    clip,
    get aspect() {
      return aspect;
    },

    update(dt) {
      clampGoal();
      // Equivalent whole turns let yaw ease along the shorter arc
      pose.yaw += TURN * Math.round((goal.yaw - pose.yaw) / TURN);

      const amount = clock.ease(dt, damping);
      const from = pose.distance;
      const turning = ease(pose, goal, ANGLE_KEYS, amount);
      ease(pose, goal, DISTANCE_KEY, amount);
      const moving = ease(pose.target, goal.target, TARGET_KEYS, amount);
      holdReach();
      // Held at its reach, the eye rests though the goal lies farther
      const zooming = pose.distance !== from;
      const nextAspect = size.width / size.height;
      const clipped = clip.near !== placed.near || clip.far !== placed.far;
      if (!turning && !zooming && !moving && !clipped && nextAspect === aspect) {
        return eyes.settle();
      }

      aspect = nextAspect;
      place();
    },

    jump() {
      clampGoal();
      for (const key of ORBIT_KEYS) pose[key] = goal[key];
      for (const i of TARGET_KEYS) pose.target[i] = goal.target[i];
      holdReach();
      aspect = size.width / size.height;
      place();
    },
  };
}

/**
 * Creates an orthographic camera over the xy plane that eases its center towards a goal.
 *
 * @returns a camera you update every frame
 */
export function flat(root: TgpuRoot, options: FlatOptions): FlatCamera {
  const { size, snap = 0, damping = 8, fit } = options;
  if (options.height === undefined && !fit) {
    throw new TypeError('cameras.flat: give a `height` or a `fit`.');
  }

  const eyes = lens(root, options.reversed ?? false);
  // Swapping the planes reverses depth
  const [front, back] = options.reversed ? [1, -1] : [-1, 1];
  const view = mat4.identity();
  const projection = mat4.create();

  const fitted = (aspect: number) =>
    fit!.mode === 'cover'
      ? Math.min(fit!.height, fit!.width / aspect)
      : Math.max(fit!.height, fit!.width / aspect);

  const [x = 0, y = 0] = options.center ?? [];
  let aspect = size.width / size.height;
  const goal = { x, y, height: fit ? fitted(aspect) : options.height! };
  const pose = { ...goal };
  /** Takes the aspect of `size`, and refits the height at once when it changed. */
  const measure = () => {
    const next = size.width / size.height;
    if (fit && next !== aspect) goal.height = pose.height = fitted(next);
    aspect = next;
  };

  const place = () => {
    view[12] = -(snap > 0 ? Math.round(pose.x / snap) * snap : pose.x);
    view[13] = -(snap > 0 ? Math.round(pose.y / snap) * snap : pose.y);

    const halfHeight = pose.height / 2;
    const halfWidth = halfHeight * aspect;
    mat4.ortho(-halfWidth, halfWidth, -halfHeight, halfHeight, front, back, projection);
    eyes.write(view, projection);
  };

  place();

  return {
    uniform: eyes.uniform,
    previous: eyes.previous,
    inject: eyes.inject,
    project: eyes.project,
    projectAt: eyes.projectAt,
    ray: eyes.ray,
    rayAt: eyes.rayAt,
    goal,
    pose,
    get aspect() {
      return aspect;
    },

    update(dt) {
      const moving = ease(pose, goal, FLAT_KEYS, clock.ease(dt, damping));
      if (!moving && size.width / size.height === aspect) return eyes.settle();

      measure();
      place();
    },

    jump() {
      for (const key of FLAT_KEYS) pose[key] = goal[key];
      measure();
      place();
    },
  };
}

export interface HitOptions {
  /**
   * Misses rays whose direction's y is nearer 0 than this, so a ray near the horizon finds no hit
   * far away. Default 0, which misses only rays parallel to a level plane.
   */
  readonly grazing?: number;
  /** With a height function, the farthest distance along the ray to search. Default 1000. */
  readonly reach?: number;
  /**
   * With a height function, the first step of the search along the ray. Each later step is 4%
   * longer. Default `reach / 1000`.
   */
  readonly step?: number;
}

const STEP_GROWTH = 1.04;
const BISECTIONS = 16;

const below = (
  ray: CpuRay<ArrayLike<number>>,
  height: (x: number, z: number) => number,
  along: number,
) => {
  const { origin, direction } = ray;
  const x = origin[0] + direction[0] * along;
  const z = origin[2] + direction[2] * along;
  return origin[1] + direction[1] * along < height(x, z);
};

/**
 * Finds where a ray, such as one from `view.rayAt`, meets the ground at `height`, such as the
 * ground under a pointer. A height function, such as a terrain's, is searched in growing steps
 * up to `reach`, then refined by halving the last step. Writes the point into `out` when given.
 * Allocates nothing.
 *
 * @param ray an origin and a unit direction
 * @param height the y of a level plane, or of the ground at each x and z
 * @param out where the point goes
 * @returns the distance along the ray, or undefined for a ray that misses
 */
export function hitHeight(
  ray: CpuRay<ArrayLike<number>>,
  height: number | ((x: number, z: number) => number),
  out?: Out3,
  options?: HitOptions,
): number | undefined {
  const { origin, direction } = ray;
  const fall = direction[1];
  const grazing = options?.grazing ?? 0;
  let along: number | undefined;

  if (typeof height === 'number') {
    if (!(Math.abs(fall) > grazing)) return undefined;
    along = (height - origin[1]) / fall;
    if (along < 0) return undefined;
  } else {
    if (Math.abs(fall) < grazing) return undefined;
    const reach = options?.reach ?? 1000;
    let step = options?.step ?? reach / 1000;
    let before = 0;
    for (let t = step; t < reach; t += step) {
      if (below(ray, height, t)) {
        let low = before;
        let high = t;
        for (let i = 0; i < BISECTIONS; i++) {
          const middle = (low + high) / 2;
          if (below(ray, height, middle)) high = middle;
          else low = middle;
        }
        along = high;
        break;
      }
      before = t;
      step *= STEP_GROWTH;
    }
    if (along === undefined) return undefined;
  }

  if (out) {
    out[0] = origin[0] + direction[0] * along;
    out[1] = typeof height === 'number' ? height : origin[1] + fall * along;
    out[2] = origin[2] + direction[2] * along;
  }
  return along;
}
