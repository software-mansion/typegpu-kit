import { tgpu } from 'typegpu';
import { describe, expect, expectTypeOf, it } from 'vitest';
import { counters, frames } from '../src/index.ts';
import { recordingRoot } from '../src/testing.ts';

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

function setup() {
  const recording = recordingRoot();
  const { root } = recording;
  const contacts = counters.create(root, ['stomps', 'hurts'], { hz: 60 });
  const loop = frames.create(root, (frame) => contacts.capture(frame));

  // The GPU never runs here, so tests write the counters it would add to.
  const stored = recording.bytes(root.unwrap(contacts.buffer));
  const count = (stomps: number, hurts: number) =>
    new Uint32Array(stored.buffer, stored.byteOffset, 2).set([stomps, hurts]);

  let time = 0;
  const run = async () => {
    loop.run(time);
    time += 1 / 60;
    await settle();
  };

  return { recording, root, contacts, loop, count, run };
}

describe('counters', () => {
  it('reports what each counter gained since the last take, once', async () => {
    const { contacts, count, run } = setup();
    expectTypeOf(contacts.take()).toEqualTypeOf<counters.Counts<'stomps' | 'hurts'> | undefined>();
    expect(contacts.take()).toBeUndefined();

    count(3, 1);
    await run();
    expect(contacts.take()).toEqual({ stomps: 3, hurts: 1 });
    expect(contacts.take()).toBeUndefined();

    count(5, 1);
    await run();
    expect(contacts.take()).toEqual({ stomps: 2, hurts: 0 });
    expect(contacts.latest).toEqual({ stomps: 5, hurts: 1 });
  });

  it('takes the gain across the wrap of a u32', async () => {
    const { contacts, count, run } = setup();
    count(0xfffffffe, 0);
    await run();
    contacts.take();

    count(3, 0);
    await run();
    expect(contacts.take()).toEqual({ stomps: 5, hurts: 0 });
  });

  it('starts again from 0 at a clear, and drops what was read before it', async () => {
    const { root, contacts, count, run } = setup();
    count(9, 9);
    await run();
    expect(contacts.take()).toEqual({ stomps: 9, hurts: 9 });

    const encoder = root.createCommandEncoder();
    contacts.clear(encoder);
    encoder.submit();
    expect(contacts.latest).toBeUndefined();

    count(1, 0);
    await run();
    expect(contacts.take()).toEqual({ stomps: 1, hurts: 0 });
  });

  it('copies at its rate, and creates nothing in a steady frame', async () => {
    const { recording, run } = setup();
    await run();
    await run();
    const before = recording.log.counts();

    await run();
    await run();
    expect(recording.log.counts()).toEqual(before);
    expect(recording.log.events.filter((event) => event === 'copy-buffer')).toHaveLength(4);
  });

  it('lets a kernel add to a counter by name through its inject', async () => {
    const { recording, root, contacts } = setup();
    const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      contacts.add.hurts(2);
    });

    await root.pipe(contacts.inject()).createComputePipeline({ compute: kernel }).initAsync();

    const wgsl = recording.log.shaders.at(-1)!;
    expect(wgsl).toMatch(/var<storage, read_write> \w+: array<atomic<u32>, 2>/);
    expect(wgsl).toMatch(/fn add\(amount: i32\) {\s+atomicAdd\(&\w+\[1i\], u32\(amount\)\);/);
  });

  it('refuses no names, or a name twice', () => {
    const { root } = recordingRoot();

    expect(() => counters.create(root, [])).toThrow(RangeError);
    expect(() => counters.create(root, ['hit', 'hit'])).toThrow(
      'counters.create: the names must be one or more, each once, got hit, hit.',
    );
  });
});
