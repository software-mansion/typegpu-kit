import { afterEach, expect, it, vi } from 'vitest';
import { frames, profiler } from '../src/index.ts';
import { recordingRoot } from '../src/testing.ts';

afterEach(() => {
  vi.unstubAllGlobals();
});

it('is inert without timestamp queries', () => {
  const { root, log } = recordingRoot();
  const timer = profiler.create(root);
  const scene = timer.timer('scene');
  const loop = frames.create(root, (frame) => timer.capture(frame));

  loop.run(0);

  expect(timer.available).toBe(false);
  expect(scene.writes()).toBeUndefined();
  expect(scene.ms).toBeUndefined();
  expect(timer.frame.ms).toBeUndefined();
  expect([scene.count, scene.last, timer.frame.count]).toEqual([0, undefined, 0]);
  expect(log.events).toEqual(['submit']);
});

it('makes timers up to its capacity, told apart by identity, not name', () => {
  for (const features of [[], ['timestamp-query']]) {
    const { root } = recordingRoot({ features });
    const timer = profiler.create(root, { capacity: 2 });
    const first = timer.timer('pass');
    const second = timer.timer('pass');

    expect(first).not.toBe(second);
    expect(first.name).toBe('pass');
    expect(timer.timers).toEqual([first, second]);
    expect(() => timer.timer('third')).toThrow(
      'This profiler has reached its limit of 2 timers. Increase `capacity` in its options.',
    );
  }
});

it('rejects a capacity, samples or slots that is no whole number above zero', () => {
  const { root } = recordingRoot({ features: ['timestamp-query'] });

  expect(() => profiler.create(root, { capacity: 0 })).toThrow('capacity must be');
  expect(() => profiler.create(root, { samples: 1.5 })).toThrow('samples must be');
  expect(() => profiler.create(root, { staging: 0 })).toThrow('staging must be');
});

it('gives each timer two queries and resolves only frames that recorded a pass', () => {
  const { root, log } = recordingRoot({ features: ['timestamp-query'] });
  const timer = profiler.create(root);
  timer.timer('scene');
  const post = timer.timer('post');
  let timed = true;
  const loop = frames.create(root, (frame) => {
    if (timed) {
      const pass = frame.encoder.beginComputePass({ timestampWrites: post.writes() });
      pass.end();
    }

    timer.capture(frame);
  });

  expect(post.writes()).toMatchObject({
    beginningOfPassWriteIndex: 2,
    endOfPassWriteIndex: 3,
  });

  loop.run(0);
  timed = false;
  loop.run(1);

  expect(log.events).toEqual([
    'begin-compute',
    'end-compute',
    'resolve-queries',
    'copy-buffer',
    'submit',
    'submit',
  ]);
});

it('records once a frame, however often capture is called', () => {
  const { root, log } = recordingRoot({ features: ['timestamp-query'] });
  const timer = profiler.create(root);
  const scene = timer.timer('scene');
  const post = timer.timer('post');

  const loop = frames.create(root, (frame) => {
    frame.encoder.beginComputePass({ timestampWrites: scene.writes() }).end();
    timer.capture(frame);
    frame.encoder.beginComputePass({ timestampWrites: post.writes() }).end();
    timer.capture(frame);
  });

  loop.run(0);

  expect(log.events.filter((e) => e === 'copy-buffer')).toHaveLength(1);
});

it('copies into as many staging buffers as it has slots before a frame goes untimed', () => {
  const { root, log } = recordingRoot({ features: ['timestamp-query'] });
  const timer = profiler.create(root, { staging: 2 });
  const scene = timer.timer('scene');

  const loop = frames.create(root, (frame) => {
    frame.encoder.beginComputePass({ timestampWrites: scene.writes() }).end();
    timer.capture(frame);
  });

  // The maps land after a task, so every slot is still mapping by the third frame.
  for (let i = 0; i < 3; i++) loop.run(i);

  expect(log.events.filter((e) => e === 'copy-buffer')).toHaveLength(2);
});

it('times passes from their timestamps without BigInts, and hands every sample on', async () => {
  vi.stubGlobal('BigUint64Array', undefined);
  const { root, bytes } = recordingRoot({ features: ['timestamp-query'] });
  const created = vi.spyOn(root.device, 'createBuffer');
  const samples: [string, number][] = [];
  const timer = profiler.create(root, {
    skip: 0,
    samples: 1,
    onSample: (timing, ms) => samples.push([timing.name, ms]),
  });
  const scene = timer.timer('scene');
  const post = timer.timer('post');

  const resolved = created.mock.results.find(
    ({ value }) => value.usage & GPUBufferUsage.QUERY_RESOLVE,
  )!.value;

  // A low word, then a high word, per timestamp. The scene's pass crosses into the next high word.
  const words = new Uint32Array(bytes(resolved).buffer);
  words.set([2 ** 32 - 1_000_000, 5, 1_500_000, 6]);
  words.set([2_000_000, 6, 3_000_000, 6], 4);

  const loop = frames.create(root, (frame) => {
    for (const each of [scene, post]) {
      frame.encoder.beginComputePass({ timestampWrites: each.writes() }).end();
    }
    timer.capture(frame);
  });

  loop.run(0);
  await new Promise((resolve) => setTimeout(resolve, 0));

  expect(scene.ms).toBe(2.5);
  expect(post.ms).toBe(1);
  expect(timer.frame.ms).toBe(4);
  expect([scene.count, scene.last, timer.frame.count, timer.frame.last]).toEqual([1, 2.5, 1, 4]);
  expect(samples).toEqual([
    ['scene', 2.5],
    ['post', 1],
    ['frame', 4],
  ]);
});

it('counts samples after the skipped ones, and reports the newest beside the median', async () => {
  const { root, bytes } = recordingRoot({ features: ['timestamp-query'] });
  const created = vi.spyOn(root.device, 'createBuffer');
  const timer = profiler.create(root, { skip: 1, samples: 3 });
  const scene = timer.timer('scene');
  const resolved = created.mock.results.find(
    ({ value }) => value.usage & GPUBufferUsage.QUERY_RESOLVE,
  )!.value;
  const words = new Uint32Array(bytes(resolved).buffer);

  const loop = frames.create(root, (frame) => {
    frame.encoder.beginComputePass({ timestampWrites: scene.writes() }).end();
    timer.capture(frame);
  });
  // A beginning at 0 and an end this many milliseconds later, each frame read back before the next
  for (const [i, ms] of [1, 2, 3, 6].entries()) {
    words.set([0, 0, ms * 1_000_000, 0]);
    loop.run(i);
    await new Promise((resolve) => setTimeout(resolve, 0));
  }

  // The first sample is skipped; the median of 2, 3 and 6 is 3, and the newest is 6
  expect([scene.count, scene.ms, scene.last]).toEqual([3, 3, 6]);
  expect([timer.frame.count, timer.frame.ms, timer.frame.last]).toEqual([3, 3, 6]);
});

it('shares the frame out to the earliest recorded pass running at each moment', async () => {
  const { root, bytes } = recordingRoot({ features: ['timestamp-query'] });
  const created = vi.spyOn(root.device, 'createBuffer');
  const timer = profiler.create(root, { skip: 0, samples: 1 });
  const passes = [timer.timer('scene'), timer.timer('light'), timer.timer('canvas')];
  const resolved = created.mock.results.find(
    ({ value }) => value.usage & GPUBufferUsage.QUERY_RESOLVE,
  )!.value;

  // In milliseconds: the scene runs 0 to 3, the light 2 to 2.5 inside it, and the canvas pass
  // writes its start at 1, before the light it was recorded after, and ends at 5
  const words = new Uint32Array(bytes(resolved).buffer);
  for (const [i, [begin, end]] of [
    [0, 3],
    [2, 2.5],
    [1, 5],
  ].entries()) {
    words.set([begin * 1_000_000, 0, end * 1_000_000, 0], i * 4);
  }

  const loop = frames.create(root, (frame) => {
    for (const each of passes) {
      frame.encoder.beginComputePass({ timestampWrites: each.writes() }).end();
    }
    timer.capture(frame);
  });
  loop.run(0);
  await new Promise((resolve) => setTimeout(resolve, 0));

  // The canvas pass starts with the light, at 2
  expect(passes.map((each) => each.ms)).toEqual([3, 0.5, 3]);
  // The scene owns 0 to 3, the light runs inside it, and the canvas pass adds its tail
  expect(passes.map((each) => each.share)).toEqual([3, 0, 2]);
  expect(timer.frame.ms).toBe(5);
});

it('starts a timing no earlier than the pass timed before it, which also bounds the frame', async () => {
  const { root, bytes } = recordingRoot({ features: ['timestamp-query'] });
  const created = vi.spyOn(root.device, 'createBuffer');
  const timer = profiler.create(root, { skip: 0, samples: 1 });
  // Created in one order, recorded in the other
  const canvas = timer.timer('canvas');
  const scene = timer.timer('scene');
  const resolved = created.mock.results.find(
    ({ value }) => value.usage & GPUBufferUsage.QUERY_RESOLVE,
  )!.value;

  // The canvas pass's start is written 20 ms before the scene's, as Dawn on Metal does without vsync
  const words = new Uint32Array(bytes(resolved).buffer);
  words.set([0, 0, 23_000_000, 0, 20_000_000, 0, 22_000_000, 0]);

  const loop = frames.create(root, (frame) => {
    frame.encoder.beginComputePass({ timestampWrites: scene.writes() }).end();
    frame.encoder.beginComputePass({ timestampWrites: canvas.writes() }).end();
    timer.capture(frame);
  });
  loop.run(0);
  await new Promise((resolve) => setTimeout(resolve, 0));

  expect([canvas.ms, canvas.share, scene.ms, scene.share, timer.frame.ms]).toEqual([3, 1, 2, 2, 3]);
});

it('skips warm-up frames, not samples, so a pass timed now and then still reports', async () => {
  const { root, bytes } = recordingRoot({ features: ['timestamp-query'] });
  const created = vi.spyOn(root.device, 'createBuffer');
  const timer = profiler.create(root, { skip: 2 });
  const scene = timer.timer('scene');
  const load = timer.timer('load');
  const resolved = created.mock.results.find(
    ({ value }) => value.usage & GPUBufferUsage.QUERY_RESOLVE,
  )!.value;
  const words = new Uint32Array(bytes(resolved).buffer);
  words.set([0, 0, 1_000_000, 0, 1_000_000, 0, 4_000_000, 0]);

  let loading = false;
  const loop = frames.create(root, (frame) => {
    frame.encoder.beginComputePass({ timestampWrites: scene.writes() }).end();
    if (loading) frame.encoder.beginComputePass({ timestampWrites: load.writes() }).end();
    timer.capture(frame);
  });
  for (let i = 0; i < 4; i++) {
    loading = i === 3;
    loop.run(i);
    await new Promise((resolve) => setTimeout(resolve, 0));
  }

  // Two frames warm up; the load ran once, after them
  expect([scene.count, load.count, load.ms, load.share]).toEqual([2, 1, 3, 3]);
});

it('splits writes between a first and a last pass, the same objects for the same writes', () => {
  const writes = {
    querySet: {} as GPUQuerySet,
    beginningOfPassWriteIndex: 2,
    endOfPassWriteIndex: 3,
  };
  const { opening, closing } = profiler.split(writes);
  expect(opening).toEqual({ querySet: writes.querySet, beginningOfPassWriteIndex: 2 });
  expect(closing).toEqual({ querySet: writes.querySet, endOfPassWriteIndex: 3 });
  expect(profiler.split(writes).opening).toBe(opening);
  expect(profiler.split(undefined)).toEqual({ opening: undefined, closing: undefined });
});
