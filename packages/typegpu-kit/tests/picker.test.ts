import { d, std, tgpu } from 'typegpu';
import { describe, expect, it } from 'vitest';
import { cameras, frames, picker } from '../src/index.ts';
import { recordingRoot } from '../src/testing.ts';

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

const position = () => {
  'use gpu';
  return d.vec3f();
};

function setup(options: Partial<picker.Options> = {}) {
  const recording = recordingRoot();
  const { root } = recording;
  const size = { width: 200, height: 100 };
  const view = cameras.orbit(root, { size, target: [0, 0, 0], distance: 5 });
  const places = root.createReadonly(d.arrayOf(d.vec3f, 100));
  const hits = picker.create(root, view, {
    capacity: 100,
    size,
    position: (i) => {
      'use gpu';
      return d.vec3f(places.$[i]);
    },
    ...options,
  });

  return { recording, root, view, hits, size };
}

describe('picker', () => {
  it('picks in one dispatch over every candidate, after writing the point and clearing the key', () => {
    const { recording, root, hits } = setup();
    const writes = recording.log.writes.length;

    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    hits.pick(pass, 0.25, 0.25);
    pass.end();
    encoder.submit();

    expect(recording.log.pipelines).toBe(1);
    expect(recording.log.events.filter((event) => event === 'dispatch')).toHaveLength(1);
    expect(recording.log.writes.slice(writes).map(({ size }) => size)).toEqual([24, 4]);

    const pointer = recording.log.writes.at(-2)!;
    expect([...new Float32Array(recording.bytes(pointer.buffer).slice().buffer)]).toEqual([
      0.25, 0.25, 200, 100, 1, 0,
    ]);
  });

  it('reads the pixel ratio of its size at each pick, and scales the reach by it', () => {
    const { recording, root, hits, size } = setup();
    const surface = Object.assign(size, { pixelRatio: 2 });
    surface.width = 400;
    surface.height = 200;

    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    hits.pick(pass, 0.25, 0.25);
    pass.end();
    encoder.submit();

    const pointer = recording.log.writes.at(-2)!;
    expect([...new Float32Array(recording.bytes(pointer.buffer).slice().buffer)]).toEqual([
      0.25, 0.25, 400, 200, 2, 0,
    ]);

    const kernel = recording.log.shaders.at(-1)!;
    expect(kernel).toMatch(/let at = \(\w+\.at \* \(\*pixels\)\);/);
    expect(kernel).toMatch(/if \(\(gap > \(\(\w+\.reach \* ratio\) \+ across\)\)\)/);
  });

  it('creates nothing in a steady frame that picks and reads', () => {
    const { recording, hits } = setup({ hz: 30 });
    const loop = frames.create(recording.root, (frame) => {
      const pass = frame.encoder.beginComputePass();
      hits.pick(pass, 0.05, 0.1);
      pass.end();
      hits.result.capture(frame);
    });

    loop.run(0);
    loop.run(0.1);
    const before = recording.log.counts();

    loop.run(0.2);
    loop.run(0.3);
    expect(recording.log.counts()).toEqual(before);
  });

  it('copies a tap without hz in the frame that picked, and in no other', async () => {
    const { recording, hits } = setup();
    let tap: [number, number] | undefined;
    const loop = frames.create(recording.root, (frame) => {
      const pass = frame.encoder.beginComputePass();
      if (tap) hits.pick(pass, ...tap);
      pass.end();
      hits.result.capture(frame);
      tap = undefined;
    });
    const copies = () => recording.log.events.filter((event) => event === 'copy-buffer').length;

    loop.run(0);
    tap = [0.5, 0.5];
    loop.run(0.1);
    expect(copies()).toBe(1);

    await settle();
    loop.run(0.2);
    loop.run(0.3);
    expect(copies()).toBe(1);
    expect(hits.result.take()).toBeNull(); // the recording root's key reads back as none
    expect(hits.result.latestFrame).toBe(1);
  });

  it('copies at the rate with hz, though it picks every frame', () => {
    const { recording, hits } = setup({ hz: 5 });
    const loop = frames.create(recording.root, (frame) => {
      const pass = frame.encoder.beginComputePass();
      hits.pick(pass, 0.05, 0.1);
      pass.end();
      hits.result.capture(frame);
    });

    for (let i = 0; i < 6; i++) loop.run(i / 30);
    expect(recording.log.events.filter((event) => event === 'copy-buffer')).toHaveLength(1);
  });

  it('clears the key without a dispatch, and settles the readback to null', async () => {
    const { recording, root, hits } = setup({ hz: 1 });
    const loop = frames.create(root, (frame) => hits.result.capture(frame));
    loop.run(0);
    await settle();
    const before = recording.log.events.length;

    hits.clear();
    loop.run(0.1);
    await settle();

    const cleared = recording.log.writes.at(-1)!;
    expect([...new Uint32Array(recording.bytes(cleared.buffer).slice().buffer)]).toEqual([
      picker.none,
    ]);
    expect(recording.log.events.slice(before)).not.toContain('dispatch');
    expect(recording.log.events.slice(before)).toContain('copy-buffer');
    expect(hits.result.latestFrame).toBe(1);
    expect(hits.result.latest).toBeNull();
  });

  it('lets another kernel act on the pick through its inject', async () => {
    const { recording, root, hits } = setup();
    const marked = root.createMutable(d.u32);
    const mark = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      if (hits.found() !== picker.none) marked.$ = hits.found();
    });

    await root.pipe(hits.inject()).createComputePipeline({ compute: mark }).initAsync();
    expect(recording.log.shaders.at(-1)).toMatch(/var<storage, read> \w+: u32/);
  });

  it('keeps the candidate its rank puts first, in the rank above the index', async () => {
    const { recording, hits } = setup({ rank: picker.byEye });
    await hits.initAsync();

    const wgsl = recording.log.shaders.at(-1)!;
    expect(wgsl).toContain('atomicMin');
    expect(wgsl).toContain('fn byEye');
    expect(wgsl).not.toContain('fn byPointer');
  });

  it('switches rankings through a uniform its rank reads, without a new pipeline', async () => {
    const recording = recordingRoot();
    const { root } = recording;
    const size = { width: 200, height: 100 };
    const view = cameras.orbit(root, { size, target: [0, 0, 0], distance: 5 });
    const eye = root.createUniform(d.u32);
    const hits = picker.create(root, view, {
      capacity: 100,
      size,
      position,
      rank: (candidate) => {
        'use gpu';
        return std.select(picker.byPointer(candidate), picker.byEye(candidate), eye.$ === 1);
      },
    });
    await hits.initAsync();
    const pipelines = recording.log.pipelines;

    eye.write(1);
    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    hits.pick(pass, 0.25, 0.25);
    pass.end();
    encoder.submit();

    expect(recording.log.pipelines).toBe(pipelines);
    const wgsl = recording.log.shaders.at(-1)!;
    expect(wgsl).toContain('fn byPointer');
    expect(wgsl).toContain('fn byEye');
  });

  it('refuses a capacity it cannot index', () => {
    const { root, view, size } = setup();

    expect(() => picker.create(root, view, { capacity: 0, size, position })).toThrow(RangeError);
    expect(() => picker.create(root, view, { capacity: 2 ** 24 + 1, size, position })).toThrow(
      'picker.create: `capacity` takes a whole number from 1 to 2^24, got 16777217.',
    );
  });
});
