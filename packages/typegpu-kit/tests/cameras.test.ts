import { d, tgpu } from 'typegpu';
import { describe, expect, it } from 'vitest';
import { mat4, vec3, vec4 } from 'wgpu-matrix';
import { Camera, cameras, history } from '../src/index.ts';
import { recordingRoot } from '../src/testing.ts';

const size = { width: 200, height: 100 };

/** The camera's view-projection, eye and screen axes, as last written. */
function written(recording: ReturnType<typeof recordingRoot>, buffer: GPUBuffer) {
  const floats = new Float32Array(recording.bytes(buffer).slice().buffer);
  return {
    viewProjection: floats.subarray(0, 16),
    position: [...floats.subarray(16, 19)],
    right: [...floats.subarray(20, 23)],
    up: [...floats.subarray(24, 27)],
  };
}

/** Ground that rises 3 units past x = `at`. */
const wall = (at: number) => (x: number) => (x > at ? 3 : 0);
const near = (actual: number[], expected: number[]) =>
  actual.forEach((value, i) => expect(value).toBeCloseTo(expected[i], 5));

const vector = ([x, y, z]: number[]) => d.vec3f(x!, y!, z!);

/** The uniform as a shader reads it, so the camera's `'use gpu'` functions run on the CPU. */
function viewOf(recording: ReturnType<typeof recordingRoot>, buffer: GPUBuffer) {
  const { viewProjection, position, right, up } = written(recording, buffer);
  const matrix = d.mat4x4f();
  viewProjection.forEach((value, i) => (matrix[i] = value));

  return Camera({
    viewProjection: matrix,
    position: vector(position),
    right: vector(right),
    up: vector(up),
  });
}

/** Where a point in the world lands on the screen, in normalized device coordinates. */
function screenOf(view: d.Infer<typeof Camera>, point: Float32Array) {
  const clip = vec4.transformMat4([point[0], point[1], point[2], 1], [...view.viewProjection]);
  return [clip[0] / clip[3], clip[1] / clip[3]];
}

/** A slope that rises along +z, where an orbit camera's eye sits at yaw 0. */
const slope = (_x: number, z: number) => z * 0.5;

describe('orbit camera', () => {
  it('writes its uniform at creation, then when the view or the aspect changes', () => {
    const recording = recordingRoot();
    const orbit = cameras.orbit(recording.root, {
      size,
      target: [0, 0, 0],
      distance: 10,
      pitch: 0,
    });
    const buffer = recording.root.unwrap(orbit.uniform.buffer);
    const writes = () => recording.log.writes.filter((each) => each.buffer === buffer).length;

    const first = writes();
    expect(first).toBe(1);
    expect(written(recording, buffer).position).toEqual([0, 0, 10]);

    orbit.update(1 / 60);
    expect(writes()).toBe(first);

    orbit.goal.yaw += 1;
    orbit.update(1 / 60);
    expect(writes()).toBe(first + 1);

    const resized = { width: 100, height: 100 };
    const square = cameras.orbit(recording.root, {
      size: resized,
      target: [0, 0, 0],
      distance: 10,
    });
    const squareBuffer = recording.root.unwrap(square.uniform.buffer);
    const squareWrites = () =>
      recording.log.writes.filter((each) => each.buffer === squareBuffer).length;
    square.update(0);
    const before = squareWrites();
    resized.width = 300;
    square.update(0);
    expect(squareWrites()).toBe(before + 1);
  });

  it('keeps last frame’s view: the one before a move, then this one the frame it holds still', () => {
    const recording = recordingRoot();
    const orbit = cameras.orbit(recording.root, {
      size,
      target: [0, 0, 0],
      distance: 10,
      pitch: 0,
    });
    const now = () => written(recording, recording.root.unwrap(orbit.uniform.buffer)).position;
    const then = () => written(recording, recording.root.unwrap(orbit.previous.buffer)).position;
    const previousWrites = () =>
      recording.log.writes.filter(
        (each) => each.buffer === recording.root.unwrap(orbit.previous.buffer),
      ).length;

    // At creation the view is last frame's too.
    near(then(), now());

    const start = now();
    // A long step reaches the goal at once.
    orbit.goal.yaw = 1;
    orbit.update(10);
    near(then(), start);
    expect(now()[0]).toBeGreaterThan(1);

    const moved = now();
    orbit.update(1 / 60);
    near(then(), moved);

    // Holding still writes nothing more.
    const settled = previousWrites();
    orbit.update(1 / 60);
    expect(previousWrites()).toBe(settled);
  });

  it('writes the screen axes in world space', () => {
    const recording = recordingRoot();
    const orbit = cameras.orbit(recording.root, {
      size,
      target: [0, 0, 0],
      distance: 10,
      pitch: 0,
    });
    const buffer = recording.root.unwrap(orbit.uniform.buffer);

    orbit.update(0);
    near(written(recording, buffer).right, [1, 0, 0]);
    near(written(recording, buffer).up, [0, 1, 0]);

    // From +x, looking down at the target.
    orbit.goal.yaw = Math.PI / 2;
    orbit.goal.pitch = Math.PI / 4;
    for (let i = 0; i < 300; i++) orbit.update(1 / 60);
    near(written(recording, buffer).right, [0, 0, -1]);
    near(written(recording, buffer).up, [-Math.SQRT1_2, Math.SQRT1_2, 0]);
  });

  it('pans its target against a drag along the screen axes, one distance per height', () => {
    const recording = recordingRoot();
    const orbit = cameras.orbit(recording.root, {
      size,
      target: [1, 2, 3],
      distance: 4,
      yaw: Math.PI / 2,
      pitch: Math.PI / 4,
    });
    orbit.update(0);
    const { right, up } = written(recording, recording.root.unwrap(orbit.uniform.buffer));

    // 25 and 50 of a 100 tall view move a quarter and half of the distance 4
    expect(cameras.pan(orbit.goal, 25, 50, { height: 100 })).toBe(orbit.goal);
    near(
      orbit.goal.target,
      [1, 2, 3].map((value, axis) => value - right[axis]! + 2 * up[axis]!),
    );
  });

  it('projects its target to the middle of the view', () => {
    const recording = recordingRoot();
    const orbit = cameras.orbit(recording.root, { size, target: [4, 0, -2], distance: 5 });
    orbit.update(0);

    const { viewProjection } = written(recording, recording.root.unwrap(orbit.uniform.buffer));
    const clip = vec4.transformMat4([4, 0, -2, 1], viewProjection);

    expect(clip[0] / clip[3]).toBeCloseTo(0, 5);
    expect(clip[1] / clip[3]).toBeCloseTo(0, 5);
  });

  it('clamps the goal it is given', () => {
    const recording = recordingRoot();
    const orbit = cameras.orbit(recording.root, {
      size,
      target: [0, 0, 0],
      distance: 10,
      pitchRange: [0, 1],
      distanceRange: [2, 20],
    });

    orbit.goal.pitch = 3;
    orbit.goal.distance = 100;
    orbit.update(0);

    expect(orbit.goal).toMatchObject({ pitch: 1, distance: 20 });
  });

  it('raises the goal pitch until the eye clears its floor, within the pitch range', () => {
    const recording = recordingRoot();
    const orbit = cameras.orbit(recording.root, {
      size,
      target: [0, 1, 0],
      distance: 10,
      pitch: -0.3,
      pitchRange: [-1, 1],
      floor: slope,
    });
    const eye = cameras.eyeOf(orbit.goal, [0, 0, 0]);

    expect(eye[1]).toBeGreaterThanOrEqual(slope(eye[0], eye[2]));
    expect(eye[1]).toBeCloseTo(slope(eye[0], eye[2]), 2);
    expect(orbit.pose.pitch).toBe(orbit.goal.pitch);

    orbit.goal.pitch = 0.9;
    orbit.jump();
    expect(orbit.goal.pitch).toBe(0.9);

    const steep = cameras.orbit(recording.root, {
      size,
      target: [0, 0, 0],
      distance: 10,
      pitch: 0,
      pitchRange: [-1, 0.2],
      floor: () => 5,
    });
    expect(steep.goal.pitch).toBe(0.2);
  });

  it('holds the eye at its reach and returns it to the goal once the reach grows', () => {
    let room = 4;
    const orbit = cameras.orbit(recordingRoot().root, {
      size,
      target: [0, 0, 0],
      distance: 10,
      reach: () => room,
    });
    expect(orbit.pose.distance).toBe(4);

    for (let i = 0; i < 60; i++) orbit.update(1 / 60);
    expect([orbit.pose.distance, orbit.goal.distance]).toEqual([4, 10]);

    room = Infinity;
    for (let i = 0; i < 120; i++) orbit.update(1 / 60);
    expect(orbit.pose.distance).toBe(10);
  });
});

describe('2d camera', () => {
  it('centres its goal, snapped, from creation', () => {
    const recording = recordingRoot();
    const view = cameras.flat(recording.root, {
      size,
      height: 10,
      center: [3.3, 1.2],
      snap: 0.5,
    });

    const { viewProjection, position, right, up } = written(
      recording,
      recording.root.unwrap(view.uniform.buffer),
    );
    expect(position.slice(0, 2)).toEqual([3.5, 1]);
    expect([right, up]).toEqual([
      [1, 0, 0],
      [0, 1, 0],
    ]);

    const center = vec4.transformMat4([3.5, 1, 0, 1], viewProjection);
    const top = vec4.transformMat4([3.5, 6, 0, 1], viewProjection);
    const edge = vec4.transformMat4([13.5, 1, 0, 1], viewProjection);
    expect(center[0]).toBeCloseTo(0, 6);
    expect(center[1]).toBeCloseTo(0, 6);
    expect(top[1]).toBeCloseTo(1, 6);
    expect(edge[0]).toBeCloseTo(1, 6);
  });

  it('fits a rectangle by cover or contain, and refits it when the aspect changes', () => {
    const recording = recordingRoot();
    const resized = { width: 200, height: 100 };
    const rectangle = { width: 16, height: 12 };
    const cover = cameras.flat(recording.root, {
      size: resized,
      fit: { ...rectangle, mode: 'cover' },
    });
    const contain = cameras.flat(recording.root, {
      size: resized,
      fit: { ...rectangle, mode: 'contain' },
    });
    expect([cover.goal.height, contain.goal.height]).toEqual([8, 12]);

    // A zoom eases as usual until the aspect changes
    cover.goal.height = 4;
    cover.update(1);
    expect(cover.goal.height).toBe(4);

    resized.width = 50;
    cover.update(0);
    contain.jump();
    expect([cover.pose.height, contain.pose.height]).toEqual([12, 32]);
    expect(() => cameras.flat(recording.root, { size })).toThrow(
      'cameras.flat: give a `height` or a `fit`.',
    );
  });
});

describe('rays through the screen', () => {
  const ndc = [0.3, -0.6];

  it('look from an orbit camera at its target, through its projection', () => {
    const recording = recordingRoot();
    const target = [1, 2, -3] as const;
    const fov = 0.9;

    for (const aspect of [0.5, 1, 2.4]) {
      for (const yaw of [0, 1, -2.5]) {
        for (const pitch of [-1.2, 0, 0.7]) {
          const surface = { width: 120 * aspect, height: 120 };
          const orbit = cameras.orbit(recording.root, {
            size: surface,
            target,
            distance: 7,
            yaw,
            pitch,
            fov,
          });
          const view = viewOf(recording, recording.root.unwrap(orbit.uniform.buffer));

          const forward = cameras.forwardOf(view);
          near([...forward], [...vec3.normalize(vec3.sub(target, view.position))]);

          const lens = cameras.lensOf(view);
          const projection = mat4.perspective(fov, aspect, 0.1, 500);
          near([...lens], [projection[0], projection[5]]);

          const ray = cameras.rayOf(view, d.vec2f(ndc[0], ndc[1]));
          near([...ray.origin], [...view.position]);
          expect(vec3.length(ray.direction)).toBeCloseTo(1, 5);
          for (const t of [1, 4])
            near(screenOf(view, vec3.addScaled(ray.origin, ray.direction, t)), ndc);
        }
      }
    }
  });

  it('are parallel from a flat camera, each from its point on the screen', () => {
    const recording = recordingRoot();
    const flat = cameras.flat(recording.root, { size, height: 10, center: [3, -1] });
    const view = viewOf(recording, recording.root.unwrap(flat.uniform.buffer));

    const forward = cameras.forwardOf(view);
    near([...forward], [0, 0, -1]);

    // 10 high and 20 wide, around (3, -1).
    const lens = cameras.lensOf(view);
    const projection = mat4.ortho(-7, 13, -6, 4, -1, 1);
    near([...lens], [projection[0], projection[5]]);

    const ray = cameras.rayOf(view, d.vec2f(ndc[0], ndc[1]));
    near([...ray.direction], [0, 0, -1]);
    for (const t of [0, 0.5])
      near(screenOf(view, vec3.addScaled(ray.origin, ray.direction, t)), ndc);
  });

  it('are the same on the CPU as on the GPU, and project back to their point', () => {
    const recording = recordingRoot();
    const orbit = cameras.orbit(recording.root, {
      size,
      target: [1, 2, -3],
      distance: 7,
      yaw: 1,
      pitch: 0.7,
    });
    const flat = cameras.flat(recording.root, { size, height: 10, center: [3, -1] });
    const origin = [0, 0, 0];
    const direction = new Float32Array(3);
    const out = [0, 0, 0];

    for (const lens of [orbit, flat]) {
      const view = viewOf(recording, recording.root.unwrap(lens.uniform.buffer));
      const gpu = cameras.rayOf(view, d.vec2f(ndc[0], ndc[1]));

      lens.ray(ndc[0], ndc[1], { origin, direction });
      near(origin, [...gpu.origin]);
      near([...direction], [...gpu.direction]);

      expect(lens.project(vec3.addScaled(origin, direction, 0.5), out)).toBe(true);
      near(out.slice(0, 2), ndc);
      expect(out[2]).toBeGreaterThan(0);
      expect(out[2]).toBeLessThan(1);
    }
  });

  it('start from a point of the surface that projectAt gives, on any aspect', () => {
    const recording = recordingRoot();
    const tall = { width: 300, height: 500 };
    const orbit = cameras.orbit(recording.root, { size: tall, target: [1, 2, -3], distance: 7 });
    const flat = cameras.flat(recording.root, { size: tall, height: 10, center: [3, -1] });
    const point = [1.6, 2.4, -2.5];
    const at = [0, 0, 0];
    const ray = { origin: [0, 0, 0], direction: [0, 0, 0] };

    for (const lens of [orbit, flat]) {
      expect(lens.aspect).toBe(0.6);
      expect(lens.projectAt(point, at)).toBe(true);
      lens.rayAt(at[0], at[1], ray);
      const along = vec3.dot(vec3.subtract(point, ray.origin), ray.direction);
      near([...vec3.addScaled(ray.origin, ray.direction, along)], point);
    }
  });

  it('meet a level plane ahead of them, and miss one level with them or behind', () => {
    const recording = recordingRoot();
    const orbit = cameras.orbit(recording.root, {
      size,
      target: [1, 0, -3],
      distance: 7,
      pitch: 0.7,
    });
    const ray = { origin: [0, 0, 0], direction: [0, 0, 0] };
    const out = [0, 0, 0];

    orbit.rayAt(0.5, 0.5, ray);
    const along = cameras.hitHeight(ray, 0, out);
    near(out, [1, 0, -3]);
    near([along!], [7]);
    near([cameras.hitHeight(ray, 1)!], [7 - 1 / Math.sin(0.7)]);

    expect(cameras.hitHeight(ray, 10, out)).toBeUndefined();
    const level = { origin: [0, 1, 0], direction: [1, 0, 0] };
    expect(cameras.hitHeight(level, 0)).toBeUndefined();
    const low = { origin: [0, 1, 0], direction: [Math.sqrt(1 - 0.005 ** 2), -0.005, 0] };
    expect(cameras.hitHeight(low, 0)).toBeCloseTo(200);
    expect(cameras.hitHeight(low, 0, undefined, { grazing: 0.01 })).toBeUndefined();

    // A height function, refined to within a 65 536th of a step: level ground meets the plane,
    // and a level ray finds a wall within reach
    expect(cameras.hitHeight(ray, () => 0)).toBeCloseTo(7, 4);
    expect(cameras.hitHeight(level, wall(5), out)).toBeCloseTo(5, 4);
    expect(out[1]).toBe(1);
    expect(cameras.hitHeight(level, wall(5), out, { grazing: 0.01 })).toBeUndefined();
    expect(cameras.hitHeight(level, wall(2000))).toBeUndefined();
    const far = cameras.hitHeight(level, wall(2000), undefined, { reach: 2500, step: 1 });
    expect(far).toBeCloseTo(2000, 2);
  });

  it('project nothing behind the eye, nor before the first write', () => {
    const recording = recordingRoot();
    const orbit = cameras.orbit(recording.root, { size, target: [0, 0, 0], distance: 5, pitch: 0 });
    const out = [7, 7, 7];

    expect(orbit.project([0, 0, 6], out)).toBe(false);
    expect(cameras.custom(recording.root).project([0, 0, 0], out)).toBe(false);
    expect(out).toEqual([7, 7, 7]);
  });
});

describe('what a camera sees', () => {
  it('is a sphere within each plane of a perspective view, or reaching into it', () => {
    const recording = recordingRoot();
    // The eye at +z 10, looking towards -z, half a view 4.23 high and 8.46 wide at the origin.
    const orbit = cameras.orbit(recording.root, {
      size,
      target: [0, 0, 0],
      distance: 10,
      pitch: 0,
      near: 0.5,
      far: 50,
    });
    const view = viewOf(recording, recording.root.unwrap(orbit.uniform.buffer));
    const sees = (center: number[], radius: number) => cameras.sees(view, vector(center), radius);

    expect(sees([0, 0, 0], 0.1)).toBe(true);
    // 0.8 beyond the right plane, 0.71 above the top, 0.3 before the near and 1 past the far.
    for (const [center, out] of [
      [[9.5, 0, 0], 0.8],
      [[0, 5, 0], 0.71],
      [[0, 0, 9.8], 0.3],
      [[0, 0, -41], 1],
    ] as const) {
      expect(sees([...center], out - 0.05)).toBe(false);
      expect(sees([...center], out + 0.05)).toBe(true);
    }
    expect(sees([0, 0, 12], 1)).toBe(false);
  });

  it('is a sphere over the rectangle of a flat view', () => {
    const recording = recordingRoot();
    const flat = cameras.flat(recording.root, { size, height: 10, center: [3, -1] });
    const view = viewOf(recording, recording.root.unwrap(flat.uniform.buffer));
    const sees = (center: number[], radius: number) => cameras.sees(view, vector(center), radius);

    // 10 high and 20 wide, around (3, -1).
    expect(sees([3, -1, 0], 0.1)).toBe(true);
    expect(sees([14, 0, 0], 0.9)).toBe(false);
    expect(sees([14, 0, 0], 1.1)).toBe(true);
    expect(sees([0, -7, 0], 0.9)).toBe(false);
    expect(sees([0, -7, 0], 1.1)).toBe(true);
  });
});

describe('the way back to the viewer', () => {
  it('points at the eye of a perspective view, and against the direction of a flat one', () => {
    const recording = recordingRoot();
    const orbit = cameras.orbit(recording.root, {
      size,
      target: [0, 0, 0],
      distance: 10,
      pitch: 0,
    });
    const flat = cameras.flat(recording.root, { size, height: 10, center: [3, -1] });
    const seen = viewOf(recording, recording.root.unwrap(orbit.uniform.buffer));
    const above = viewOf(recording, recording.root.unwrap(flat.uniform.buffer));

    // The eye is at +z 10.
    const back = cameras.towards(seen, vector([6, 0, 2]));
    expect([back.x, back.y, back.z].map((v) => v * 10)).toEqual(
      [-6, 0, 8].map((v) => expect.closeTo(v, 4)),
    );
    // A flat camera looks towards -z from wherever the point is.
    for (const point of [
      [3, -1, 0],
      [20, 5, -4],
    ]) {
      const way = cameras.towards(above, vector(point));
      expect([way.x, way.y, way.z]).toEqual([0, 0, 1].map((v) => expect.closeTo(v, 5)));
    }
  });
});

describe('distance from depth', () => {
  it('gives the distance along the view to a point from its depth, perspective and flat', () => {
    const recording = recordingRoot();
    const orbit = cameras.orbit(recording.root, {
      size,
      target: [1, 2, -3],
      distance: 7,
      yaw: 1,
      pitch: 0.7,
    });
    const flat = cameras.flat(recording.root, { size, height: 10, center: [3, -1] });

    for (const lens of [orbit, flat]) {
      const view = viewOf(recording, recording.root.unwrap(lens.uniform.buffer));
      const forward = cameras.forwardOf(view);
      for (const [ahead, across] of [
        [0.5, 0],
        [3, 1.5],
        [6, -2],
      ] as const) {
        const point = vec3.add(
          vec3.addScaled(view.position, forward, ahead),
          vec3.scale(view.right, across),
        );
        const clip = vec4.transformMat4([...point, 1], [...view.viewProjection]);
        expect(cameras.distanceOf(view, clip[2] / clip[3])).toBeCloseTo(ahead, 3);
      }
    }
  });
});

/** A view from x along the x axis. */
const from = (x: number) => mat4.translation([-x, 0, 0]);

describe('custom camera', () => {
  it('writes the eye and the screen axes from a view matrix, as the orbit camera does', () => {
    const recording = recordingRoot();
    const orbit = cameras.orbit(recording.root, {
      size,
      target: [1, 2, -3],
      distance: 6,
      yaw: 0.7,
      pitch: 0.4,
    });
    const lens = cameras.custom(recording.root);

    const eye = [...orbit.pose.target];
    eye[0] += 6 * Math.cos(0.4) * Math.sin(0.7);
    eye[1] += 6 * Math.sin(0.4);
    eye[2] += 6 * Math.cos(0.4) * Math.cos(0.7);
    lens.write(mat4.lookAt(eye, [1, 2, -3], [0, 1, 0]), mat4.perspective(0.8, 2, 0.1, 500));

    const fromOrbit = written(recording, recording.root.unwrap(orbit.uniform.buffer));
    const fromLens = written(recording, recording.root.unwrap(lens.uniform.buffer));
    near(fromLens.position, eye);
    near(fromLens.position, fromOrbit.position);
    near(fromLens.right, fromOrbit.right);
    near(fromLens.up, fromOrbit.up);
    near([...fromLens.viewProjection], [...fromOrbit.viewProjection]);
  });

  it('fills last frame’s view at the first write, then moves each view there at the next write', () => {
    const recording = recordingRoot();
    const lens = cameras.custom(recording.root);
    const now = () => written(recording, recording.root.unwrap(lens.uniform.buffer)).position;
    const then = () => written(recording, recording.root.unwrap(lens.previous.buffer)).position;

    lens.write(from(1), mat4.identity());
    near(now(), [1, 0, 0]);
    near(then(), [1, 0, 0]);

    lens.write(from(2), mat4.identity());
    near(now(), [2, 0, 0]);
    near(then(), [1, 0, 0]);

    lens.write(from(2), mat4.identity());
    near(then(), [2, 0, 0]);

    // Once last frame's view has caught up, writing the view it holds writes nothing.
    const writes = recording.log.writes.length;
    lens.write(from(2), mat4.identity());
    expect(recording.log.writes.length).toBe(writes);
  });
  it('throws for a view that mirrors the world, and takes it with the image mirrored too', () => {
    const recording = recordingRoot();
    const lens = cameras.custom(recording.root);
    const projection = mat4.perspective(0.8, 2, 0.1, 500);
    // A view of the world reflected in the ground, y = 0
    const mirrored = mat4.multiply(
      mat4.lookAt([2, 3, 6], [0, 1, 0], [0, 1, 0]),
      mat4.scaling([1, -1, 1]),
    );

    expect(() => lens.write(mirrored, projection)).toThrow('mirrors the world');
    expect(() => cameras.layered(recording.root, 1).write(0, mirrored, projection)).toThrow(
      'mirrors the world',
    );

    lens.write(mat4.multiply(mat4.scaling([-1, 1, 1]), mirrored), projection);
    const view = viewOf(recording, recording.root.unwrap(lens.uniform.buffer));
    near([...view.position], [2, -3, 6]);
    near([...cameras.forwardOf(view)], [...vec3.normalize([-2, 2, -6])]);

    const ray = cameras.rayOf(view, d.vec2f(0.3, -0.6));
    near(screenOf(view, vec3.add(ray.origin, ray.direction)), [0.3, -0.6]);
  });
});

describe('layered camera', () => {
  it('writes the view of one layer where that layer sits, and only when it changes', () => {
    const recording = recordingRoot();
    const views = cameras.layered(recording.root, 3);
    const gpu = recording.root.unwrap(views.uniform.buffer);
    const stride = 112;
    const lens = cameras.custom(recording.root);
    const projection = mat4.perspective(0.8, 2, 0.1, 500);
    const view = mat4.lookAt([2, 3, 6], [0, 1, 0], [0, 1, 0]);
    lens.write(view, projection);

    const writes = recording.log.writes.length;
    views.write(1, view, projection);
    expect(recording.log.writes.slice(writes)).toEqual([
      { buffer: gpu, offset: stride, size: stride },
    ]);
    views.write(1, view, projection);
    expect(recording.log.writes).toHaveLength(writes + 1);

    const all = new Float32Array(recording.bytes(gpu).slice().buffer);
    const own = written(recording, recording.root.unwrap(lens.uniform.buffer));
    near([...all.subarray(stride / 4, stride / 4 + 16)], [...own.viewProjection]);
    near([...all.subarray(stride / 4 + 16, stride / 4 + 19)], own.position);
    expect(all.subarray(0, stride / 4).every((value) => value === 0)).toBe(true);
  });

  it('throws for a layer it lacks', () => {
    const { root } = recordingRoot();
    const views = cameras.layered(root, 2);
    const projection = mat4.perspective(0.8, 2, 0.1, 500);
    const view = mat4.lookAt([2, 3, 6], [0, 1, 0], [0, 1, 0]);

    expect(() => views.write(2, view, projection)).toThrow(RangeError);
    expect(() => cameras.layered(root, 0)).toThrow(RangeError);
  });

  it('holds every view in one uniform array', async () => {
    const recording = recordingRoot();
    const views = cameras.layered(recording.root, 2);
    const vertex = tgpu.vertexFn({ out: { position: d.builtin.position } })(() => {
      'use gpu';
      return { position: cameras.current.$.viewProjection * d.vec4f(1) };
    });
    const fragment = tgpu.fragmentFn({ out: d.vec4f })(() => {
      'use gpu';
      return d.vec4f(1);
    });

    await recording.root
      .pipe(views.inject())
      .createRenderPipeline({ vertex, fragment, targets: { format: 'rgba8unorm' } })
      .initAsync();
    expect(recording.log.shaders.at(-1)).toMatch(/var<uniform> \w+: array<Camera, 2>/);
  });
});

const clipOf = () => {
  'use gpu';
  return cameras.current.$.viewProjection * d.vec4f(cameras.current.$.position, 1);
};
const clipBefore = history.previously(clipOf);

describe('camera accessor', () => {
  const vertex = tgpu.vertexFn({ out: { position: d.builtin.position } })(() => {
    'use gpu';
    return { position: cameras.current.$.viewProjection * d.vec4f(cameras.current.$.position, 1) };
  });
  const fragment = tgpu.fragmentFn({ out: d.vec4f })(() => {
    'use gpu';
    return d.vec4f(1);
  });

  it('reads the uniform of the camera injected into the pipeline', async () => {
    const recording = recordingRoot();
    const orbit = cameras.orbit(recording.root, { size, target: [0, 0, 0], distance: 1 });

    await recording.root
      .pipe(orbit.inject())
      .createRenderPipeline({ vertex, fragment, targets: { format: 'rgba8unorm' } })
      .initAsync();

    expect(recording.log.shaders.some((code) => code.includes('var<uniform>'))).toBe(true);
  });

  it('reads last frame’s uniform in code `history.previously` compiled', () => {
    const recording = recordingRoot();
    const orbit = cameras.orbit(recording.root, { size, target: [0, 0, 0], distance: 1 });
    const moving = tgpu.vertexFn({ out: { position: d.builtin.position } })(() => {
      'use gpu';
      return { position: clipOf() - clipBefore() };
    });

    const wgsl = tgpu.resolve([
      recording.root
        .pipe(orbit.inject())
        .createRenderPipeline({ vertex: moving, fragment, targets: { format: 'rgba8unorm' } }),
    ]);
    expect(wgsl.match(/var<uniform>/g)).toHaveLength(2);
  });

  it('projects a point with toClip through the camera injected, and last frame’s', () => {
    const recording = recordingRoot();
    const orbit = cameras.orbit(recording.root, { size, target: [0, 0, 0], distance: 1 });
    const toClipBefore = history.previously(cameras.toClip);
    const moving = tgpu.vertexFn({ out: { position: d.builtin.position } })(() => {
      'use gpu';
      const point = d.vec3f(1, 2, 3);
      return { position: cameras.toClip(point) - toClipBefore(point) };
    });

    const wgsl = tgpu.resolve([
      recording.root
        .pipe(orbit.inject())
        .createRenderPipeline({ vertex: moving, fragment, targets: { format: 'rgba8unorm' } }),
    ]);
    expect(wgsl.match(/var<uniform>/g)).toHaveLength(2);
    expect(wgsl.match(/\.viewProjection \* vec4f\(point, 1f\)/g)).toHaveLength(2);
  });

  it('names the inject() to pipe in when no camera filled it', () => {
    const { root } = recordingRoot();
    const pipeline = root.createRenderPipeline({
      vertex,
      fragment,
      targets: { format: 'rgba8unorm' },
    });

    expect(() => pipeline.initSync()).toThrow(
      'vertexFn:vertex: Nothing filled `cameras.current`: create the pipeline from `root.pipe(view.inject())`, where `view` is a `cameras.orbit`, `cameras.flat`, `cameras.custom` or `cameras.layered`.',
    );
  });
});

describe('changing the lens', () => {
  it('takes a new field of view and new clipping planes, writing once each', () => {
    const recording = recordingRoot();
    const orbit = cameras.orbit(recording.root, { size, target: [0, 0, 0], distance: 10 });
    const buffer = recording.root.unwrap(orbit.uniform.buffer);
    const writes = () => recording.log.writes.filter((each) => each.buffer === buffer).length;
    const projected = () => [...written(recording, buffer).viewProjection];
    const eye = cameras.eyeOf(orbit.goal, [0, 0, 0]);
    const lookAt = mat4.lookAt(eye, [0, 0, 0], [0, 1, 0]);

    orbit.goal.fov = 0.4;
    orbit.update(10);
    near(projected(), [...mat4.multiply(mat4.perspective(0.4, 2, 0.1, 500), lookAt)]);

    const before = writes();
    orbit.update(1 / 60);
    expect(writes()).toBe(before);

    orbit.clip.far = 50;
    orbit.update(1 / 60);
    expect(writes()).toBe(before + 1);
    near(projected(), [...mat4.multiply(mat4.perspective(0.4, 2, 0.1, 50), lookAt)]);
  });
});

describe('easing', () => {
  type Recording = ReturnType<typeof recordingRoot>;
  /** Moves part of a camera's goal, then checks it eases there, settles and writes nothing more. */
  const eases =
    <C extends cameras.CameraBase & { update(dt: number): void }>(
      make: (root: Recording['root']) => C,
      move: (camera: C) => void,
      read: (camera: C, recording: Recording) => number,
      goal: number,
    ) =>
    () => {
      const recording = recordingRoot();
      const camera = make(recording.root);
      const start = read(camera, recording);
      move(camera);

      camera.update(1 / 60);
      const step = read(camera, recording);
      expect((step - start) * (goal - step)).toBeGreaterThan(0);

      for (let i = 0; i < 300; i++) camera.update(1 / 60);
      expect(read(camera, recording)).toBeCloseTo(goal, 4);
      const writes = recording.log.writes.length;
      camera.update(1 / 60);
      expect(recording.log.writes).toHaveLength(writes);
    };
  const orbitAt = (yaw: number) => (root: Recording['root']) =>
    cameras.orbit(root, { size, target: [0, 0, 0], distance: 10, yaw, pitch: 0 });
  /** The eye's yaw from where the uniform puts it, from 0 to a full turn. */
  const yawOf = (camera: cameras.OrbitCamera, recording: Recording) => {
    const [x, , z] = written(recording, recording.root.unwrap(camera.uniform.buffer)).position;
    const yaw = Math.atan2(x!, z!);
    return yaw < 0 ? yaw + Math.PI * 2 : yaw;
  };

  it.each([
    ['yaw', eases(orbitAt(0), (orbit) => (orbit.goal.yaw = 1), yawOf, 1)],
    [
      'yaw the short way round',
      eases(orbitAt(3.1), (orbit) => (orbit.goal.yaw = -3.1), yawOf, Math.PI * 2 - 3.1),
    ],
    [
      'the target',
      eases(
        orbitAt(0),
        (orbit) => (orbit.goal.target[0] = 4),
        (orbit) => orbit.pose.target[0],
        4,
      ),
    ],
    [
      'the field of view',
      eases(
        orbitAt(0),
        (orbit) => (orbit.goal.fov = 0.4),
        (orbit) => orbit.pose.fov,
        0.4,
      ),
    ],
    [
      'a flat camera’s height',
      eases(
        (root) => cameras.flat(root, { size, height: 10 }),
        (flat) => (flat.goal.height = 4),
        (flat) => flat.pose.height,
        4,
      ),
    ],
  ])('eases %s towards its goal, settles on it, then writes nothing', (_, run) => run());
});

describe('jump', () => {
  it('moves an orbit or flat view to the goal at once, keeping last frame’s, then writes nothing', () => {
    const recording = recordingRoot();
    const orbit = cameras.orbit(recording.root, { size, target: [0, 0, 0], distance: 10 });
    const buffer = recording.root.unwrap(orbit.uniform.buffer);
    const previous = recording.root.unwrap(orbit.previous.buffer);
    const writes = () =>
      recording.log.writes.filter((each) => each.buffer === buffer || each.buffer === previous)
        .length;
    const start = written(recording, buffer).position;

    orbit.goal.yaw = 2;
    orbit.goal.distance = 6;
    orbit.goal.fov = 1;
    orbit.goal.target[1] = 3;
    orbit.jump();

    expect(orbit.pose).toEqual(orbit.goal);
    near(written(recording, buffer).position, [...cameras.eyeOf(orbit.goal, [0, 0, 0])]);
    near(written(recording, previous).position, start);

    orbit.jump();
    const settled = writes();
    orbit.jump();
    expect(writes()).toBe(settled);

    const flat = cameras.flat(recording.root, { size, height: 10 });
    flat.goal.x = 4;
    flat.goal.height = 2;
    flat.jump();

    expect(flat.pose).toEqual({ x: 4, y: 0, height: 2 });
    expect(written(recording, recording.root.unwrap(flat.uniform.buffer)).position[0]).toBe(4);
  });
});

describe('eyeOf', () => {
  it('places the eye of an orbit where the orbit camera puts it', () => {
    const recording = recordingRoot();
    const orbit = cameras.orbit(recording.root, {
      size,
      target: [1, 2, -3],
      distance: 7,
      yaw: 1,
      pitch: 0.3,
    });
    const out = [0, 0, 0];

    expect(cameras.eyeOf(orbit.pose, out)).toBe(out);
    near(written(recording, recording.root.unwrap(orbit.uniform.buffer)).position, out);
  });
});
