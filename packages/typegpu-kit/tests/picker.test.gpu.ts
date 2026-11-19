import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { expect, it } from 'vitest';
import { cameras, frames, picker } from '../src/index.ts';

const COUNT = 300;
const size = { width: 200, height: 100 };

/** Candidates far behind the eye, but for the ones `places` puts on the screen. */
async function setup(
  places: Record<number, readonly [number, number, number]>,
  options: Partial<picker.Options> | ((root: TgpuRoot) => Partial<picker.Options>) = {},
  reversed = false,
) {
  const root = await gpuRoot();
  const view = cameras.orbit(root, { size, target: [0, 0, 0], distance: 10, pitch: 0, reversed });

  const positions = Array.from({ length: COUNT }, (_, i) => places[i] ?? [0, 0, 50 + i]);
  const where = root.createReadonly(
    d.arrayOf(d.vec3f, COUNT),
    positions.map((p) => d.vec3f(...p)),
  );
  const sizes = root.createReadonly(
    d.arrayOf(d.f32, COUNT),
    positions.map((_, i) => i / 100),
  );

  const hits = picker.create(root, view, {
    capacity: COUNT,
    size,
    position: (i) => {
      'use gpu';
      return d.vec3f(where.$[i]);
    },
    radius: (i) => {
      'use gpu';
      return sizes.$[i];
    },
    alive: (i) => {
      'use gpu';
      return i !== 7;
    },
    ...(typeof options === 'function' ? options(root) : options),
  });

  const seen = root.createMutable(d.u32);
  const mark = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    seen.$ = hits.found();
  });
  const marking = root.pipe(hits.inject()).createComputePipeline({ compute: mark });

  /** Picks at a pixel of the camera's view, and what a kernel later in the pass reads of it. */
  const pickAt = async (x: number, y: number) => {
    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    hits.pick(pass, x / size.width, y / size.height);
    pass.setPipeline(marking);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.submit();

    return seen.read();
  };

  /** Where a point lands, from 0 to 1 across the view from the top-left corner. */
  const fractionOf = (point: readonly [number, number, number]) => {
    const ndc = [0, 0, 0];
    expect(view.project(point, ndc)).toBe(true);
    return [(ndc[0] + 1) / 2, (1 - ndc[1]) / 2] as const;
  };

  /** Where a point lands, in pixels of the camera's view from the top-left corner. */
  const pixelOf = (point: readonly [number, number, number]) => {
    const [x, y] = fractionOf(point);
    return [x * size.width, y * size.height] as const;
  };

  return { root, view, hits, pickAt, fractionOf, pixelOf };
}

it('finds the candidate nearest the point within reach, and nothing out of reach', async () => {
  const a = [3, 0, 0] as const;
  const b = [5, 0, 0] as const;
  const { root, pickAt, pixelOf } = await setup({ 3: a, 150: b, 7: [0, 0, 0] });

  const [ax, ay] = pixelOf(a);
  const [bx] = pixelOf(b);
  expect(await pickAt(ax + 3, ay)).toBe(3);
  expect(await pickAt(bx - 2, ay)).toBe(150);

  // The dead candidate 7 is under the point, and `a` is beyond the reach of 24 pixels.
  const [cx, cy] = pixelOf([0, 0, 0]);
  expect(cx + 24).toBeLessThan(ax);
  expect(await pickAt(cx - 8, cy)).toBe(picker.none);
  root.destroy();
});

it('takes a reach in points on a size of two pixels per point', async () => {
  const a = [3, 0, 0] as const;

  // The camera's view is 200 by 100, so a pixel of it is a point of this size.
  const { root, pickAt, pixelOf } = await setup(
    { 3: a },
    { size: { width: 400, height: 200, pixelRatio: 2 } },
  );
  const [x, y] = pixelOf(a);
  expect(await pickAt(x, y + 20)).toBe(3); // 40 pixels away, within the reach of 24 points
  expect(await pickAt(x, y + 30)).toBe(picker.none);
  root.destroy();
});

it('counts a candidate’s radius as part of it, and keeps the nearest the eye when asked', async () => {
  const front = [0, 0, 4] as const;
  const back = [0, 0, -4] as const;

  // Both lie on the line of sight through the centre. Candidate 250's radius is 2.5 units, about
  // 49 pixels at its distance, so a point 40 pixels from its centre is on it.
  const { root, pickAt, pixelOf } = await setup({ 250: front, 20: back }, { rank: picker.byEye });
  const [x, y] = pixelOf(front);
  expect(await pickAt(x, y)).toBe(250);
  expect(await pickAt(x + 40, y)).toBe(250);
  root.destroy();
});

it.each([false, true])(
  'measures depth in equal steps of distance, reversed %s',
  async (reversed) => {
    // Halfway from the near plane, 0.1 from the eye, to the far plane, 500 from it.
    const halfway = [0, 0, -240] as const;
    const { root, pickAt, pixelOf } = await setup(
      { 5: halfway },
      {
        rank: (candidate) => {
          'use gpu';
          return std.select(d.f32(-1), d.f32(0), std.abs(candidate.depth - 0.5) < 0.01);
        },
      },
      reversed,
    );
    expect(await pickAt(...pixelOf(halfway))).toBe(5);
    root.destroy();
  },
);

it('ranks by where the ray meets a shape, and turns down the candidates it misses', async () => {
  const front = [0.3, 0, 4] as const;
  const back = [0, 0, -4] as const;
  const { root, pickAt, pixelOf } = await setup(
    { 250: front, 20: back },
    {
      // The distance along the ray to a ball of half a unit around the centre.
      rank: (candidate) => {
        'use gpu';
        const to = candidate.position - candidate.ray.origin;
        const along = std.dot(to, candidate.ray.direction);
        const off = std.length(to - candidate.ray.direction * along);
        return std.select(d.f32(-1), along / 100, off < 0.5);
      },
    },
  );

  // The point is on the centre of the ball at the back, and its ray meets the one in front first.
  const [x, y] = pixelOf(back);
  expect(await pickAt(x, y)).toBe(250);

  // 15 pixels to the left, both centres are within reach and the ray passes beside both balls.
  expect(await pickAt(x - 15, y)).toBe(picker.none);
  root.destroy();
});

it('switches between rankings through a uniform its rank reads, without a new picker', async () => {
  let byEye: { write(value: number): void } | undefined;
  const { root, pickAt, pixelOf } = await setup({ 250: [0, 0, 4], 20: [0, 0, -4] }, (gpu) => {
    const eye = gpu.createUniform(d.u32);
    byEye = eye;
    return {
      rank: (candidate) => {
        'use gpu';
        return std.select(picker.byPointer(candidate), picker.byEye(candidate), eye.$ === 1);
      },
    };
  });

  // Both centres are under the point, so on the screen the lower index wins.
  const [x, y] = pixelOf([0, 0, 4]);
  expect(await pickAt(x, y)).toBe(20);
  byEye!.write(1);
  expect(await pickAt(x, y)).toBe(250);
  root.destroy();
});

it('brings what it found to the CPU through its readback', async () => {
  const { root, hits, fractionOf } = await setup({ 42: [-1, 0.5, 0] }, { hz: 120 });
  const [x, y] = fractionOf([-1, 0.5, 0]);

  const loop = frames.create(root, (frame) => {
    const pass = frame.encoder.beginComputePass();
    hits.pick(pass, x, y);
    pass.end();
    hits.result.capture(frame);
  });

  for (let i = 0; i < 20 && hits.result.latest === undefined; i++) {
    loop.run(i / 60);
    await root.device.queue.onSubmittedWorkDone();
    await new Promise((resolve) => setTimeout(resolve, 5));
  }

  expect(hits.result.latest).toBe(42);
  root.destroy();
});

it('brings a tap’s pick to the CPU without hz, once', async () => {
  const { root, hits, fractionOf } = await setup({ 42: [-1, 0.5, 0] });
  let tap: readonly [number, number] | undefined;

  const loop = frames.create(root, (frame) => {
    const pass = frame.encoder.beginComputePass();
    if (tap) hits.pick(pass, ...tap);
    pass.end();
    hits.result.capture(frame);
    tap = undefined;
  });

  const taken: (number | null)[] = [];
  for (let i = 0; i < 20; i++) {
    if (i === 2) tap = fractionOf([-1, 0.5, 0]);
    loop.run(i / 60);
    await root.device.queue.onSubmittedWorkDone();
    await new Promise((resolve) => setTimeout(resolve, 5));

    const found = hits.result.take();
    if (found !== undefined) taken.push(found);
  }

  expect(taken).toEqual([42]);
  expect(hits.result.latestFrame).toBe(2);
  root.destroy();
});
