import { d } from 'typegpu';
import { expect, expectTypeOf, it } from 'vitest';
import { type Frame } from '../src/frames.ts';
import { frames, readback, type Readback } from '../src/index.ts';
import { recordingRoot } from '../src/testing.ts';

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

function setup(options: readback.Options = {}, after?: (frame: Frame) => void) {
  const { root, log } = recordingRoot();
  const counts = root.createBuffer(d.arrayOf(d.u32, 4)).$usage('storage');
  counts.write([1, 2, 3, 4]);

  const probe = readback.create(root, counts, options);

  const loop = frames.create(root, (frame) => {
    probe.capture(frame);
    after?.(frame);
  });

  const copies = () => log.events.filter((e) => e === 'copy-buffer').length;

  return { counts, probe, loop, copies };
}

it('keeps the latest value at a rate when asked to', async () => {
  const { counts, probe, loop, copies } = setup({ hz: 10 });

  for (const time of [0, 0.05, 0.1, 0.15, 0.2]) {
    if (time === 0.2) counts.write([9, 9, 9, 9]);
    loop.run(time);
    await settle();
  }

  expect(copies()).toBe(3);
  expect(probe.latest).toEqual([9, 9, 9, 9]);
  expect(probe.latestFrame).toBe(4);
});

it('keeps its rate when frame times jitter', async () => {
  const { loop, copies } = setup({ hz: 30 });

  let time = 0;
  for (let i = 0; i < 60; i++) {
    loop.run(time);
    time += i % 2 ? 0.0168 : 0.0165;
    await settle();
  }

  expect(copies()).toBe(30);
});

it('copies at the next capture after a request, once, with no read waiting', async () => {
  const { probe, loop, copies } = setup();

  loop.run(0);
  probe.request();
  loop.run(1);
  loop.run(2);
  await settle();

  expect(copies()).toBe(1);
  expect(probe.take()).toEqual([1, 2, 3, 4]);
  expect(probe.latestFrame).toBe(1);
});

it('skips a full staging ring and keeps a request pending until a slot frees', async () => {
  const { probe, loop, copies } = setup({ hz: 1000, staging: 2 });

  loop.run(0);
  loop.run(0.001); // both staging buffers are now mapping
  probe.request();
  loop.run(0.001); // skipped: nothing is free
  expect(copies()).toBe(2);

  await settle();
  loop.run(0.001); // the same time, so the rate isn't due; the request is
  expect(copies()).toBe(3);
  loop.run(0.002); // the rate alone fills the ring again
  expect(copies()).toBe(4);
  loop.run(0.003); // skipped with no request waiting
  expect(copies()).toBe(4);
});

it('drops results of copies recorded before a reset', async () => {
  const { counts, probe, loop } = setup();

  const value = probe.next();
  loop.run(0); // copies [1, 2, 3, 4]
  probe.reset();
  counts.write([5, 6, 7, 8]);
  loop.run(1); // the stale result is dropped and the promise waits for this capture

  await expect(value).resolves.toEqual([5, 6, 7, 8]);
});

it('recaptures when the frame that copied is not submitted', async () => {
  let fail = false;
  const { probe, loop, copies } = setup({}, () => {
    if (fail) throw new Error('recording failed');
  });

  const value = probe.next();
  fail = true;
  expect(() => loop.run(0)).toThrow('recording failed'); // copied, never submitted
  fail = false;
  loop.run(0);

  expect(copies()).toBe(2);
  await expect(value).resolves.toEqual([1, 2, 3, 4]);
});

it('reads a copy that an early submit sent before its frame threw', async () => {
  const { probe, loop, copies } = setup({}, (frame) => {
    frame.submit();
    throw new Error('recording failed');
  });

  probe.request();
  expect(() => loop.run(0)).toThrow('recording failed');
  await settle();

  expect(probe.latest).toEqual([1, 2, 3, 4]);
  expect(copies()).toBe(1);
});

it('keeps its rate when the dropped copy was recorded before a reset', () => {
  let fail = false;
  const { probe, loop, copies } = setup({ hz: 10 }, () => {
    if (fail) throw new Error('recording failed');
  });

  fail = true;
  expect(() => loop.run(0)).toThrow('recording failed'); // copied, never submitted
  fail = false;
  probe.reset();
  loop.run(0.05);

  expect(copies()).toBe(1);
});

it('captures for a promise from next, then stops', async () => {
  const { probe, loop, copies } = setup();

  const value = probe.next();
  loop.run(0);
  expect(copies()).toBe(1);
  await expect(value).resolves.toEqual([1, 2, 3, 4]);
  expect(probe.latestFrame).toBe(0);

  loop.run(1);
  expect(copies()).toBe(1); // nothing waits any more
});

it('rejects pending and future reads on destroy and stops capturing', async () => {
  const { probe, loop, copies } = setup({ hz: 10 });

  const captured = probe.next();
  loop.run(0);
  const waiting = probe.next();
  probe.destroy();

  await expect(captured).rejects.toThrow('The readback was destroyed.');
  await expect(waiting).rejects.toThrow('The readback was destroyed.');
  await expect(probe.next()).rejects.toThrow('The readback was destroyed.');
  loop.run(1);
  expect(copies()).toBe(1);
});

it('records one copy a frame, however often capture is called', () => {
  const { root, log } = recordingRoot();
  const counts = root.createBuffer(d.u32).$usage('storage');
  const probe = readback.create(root, counts, { hz: 1000 });

  const loop = frames.create(root, (frame) => {
    probe.capture(frame);
    probe.capture(frame);
  });

  loop.run(0);
  loop.run(1);

  expect(log.events.filter((e) => e === 'copy-buffer')).toHaveLength(2);
});

it('captures into the frame’s current encoder after an early submit', async () => {
  const { root, log } = recordingRoot();
  const counts = root.createBuffer(d.arrayOf(d.u32, 4)).$usage('storage');
  const results = root.createBuffer(d.arrayOf(d.u32, 4));
  counts.write([1, 2, 3, 4]);
  results.write([9, 9, 9, 9]);

  const probe = readback.create(root, counts, { hz: 1000 });
  const loop = frames.create(root, (frame) => {
    counts.copyFrom(results, frame.encoder);
    frame.submit();
    probe.capture(frame);
  });

  loop.run(0);
  await settle();

  expect(log.events).toEqual(['copy-buffer', 'submit', 'copy-buffer', 'submit']);
  expect(probe.latest).toEqual([9, 9, 9, 9]);
});

it('fails the reads of a copy whose decode throws, and frees its staging buffer', async () => {
  const { root } = recordingRoot();
  const counts = root.createBuffer(d.u32).$usage('storage');
  let broken = true;

  const probe = readback.create(root, counts, {
    staging: 1,
    decode: (count) => {
      if (broken) throw new Error('bad count');
      return count;
    },
  });

  const loop = frames.create(root, (frame) => probe.capture(frame));

  const first = probe.next();
  loop.run(0);
  await expect(first).rejects.toThrow('bad count');

  broken = false;
  const second = probe.next();
  loop.run(1);
  await expect(second).resolves.toBe(0);
});

it('hands each value over once and forgets the latest value on reset', async () => {
  const { counts, probe, loop } = setup({ hz: 10 });

  expect(probe.take()).toBeUndefined();
  loop.run(0);
  await settle();
  expect(probe.take()).toEqual([1, 2, 3, 4]);
  expect(probe.take()).toBeUndefined();

  loop.run(0.1);
  await settle();
  probe.reset(); // with a value not yet taken
  expect([probe.latest, probe.latestFrame, probe.take()]).toEqual([undefined, -1, undefined]);
  counts.write([0, 0, 0, 0]);
  loop.run(0.2);
  await settle();
  expect(probe.take()).toEqual([0, 0, 0, 0]);
  expect(probe.take()).toBeUndefined();
});

it('types a read as the buffer’s value, or as what decode returns', () => {
  const { root } = recordingRoot();
  const counts = root.createBuffer(d.arrayOf(d.u32, 4)).$usage('storage');

  expectTypeOf(readback.create(root, counts)).toEqualTypeOf<Readback<number[]>>();
  expectTypeOf(readback.create(root, counts, { decode: (values) => values.length })).toEqualTypeOf<
    Readback<number>
  >();

  // @ts-expect-error without decode, a read is the buffer's value
  const wrong: Readback<string> = readback.create(root, counts);
  void wrong;

  expectTypeOf(readback.create(root, counts, { bytes: true })).toEqualTypeOf<
    Readback<ArrayBuffer>
  >();
  expectTypeOf(
    readback.create(root, counts, { bytes: true, decode: (bytes) => bytes.byteLength }),
  ).toEqualTypeOf<Readback<number>>();
  // @ts-expect-error without decode, a read of bytes is an ArrayBuffer
  readback.create<typeof counts.dataType, string>(root, counts, { bytes: true });
  // @ts-expect-error the same, whatever the result is assigned to
  const bytes: Readback<string> = readback.create(root, counts, { bytes: true });
  void bytes;

  // Options declared with their own type, as a wrapper forwards them
  const declared: readback.BytesOptions = { bytes: true };
  expectTypeOf(readback.create(root, counts, declared)).toEqualTypeOf<Readback<ArrayBuffer>>();
  const unsure: readback.BytesOptions<string> = { bytes: true, decode: () => '' };
  // @ts-expect-error a decode the type leaves optional can't promise a string
  readback.create(root, counts, unsure);
});

it('rejects a ring without staging buffers, and a rate of zero', () => {
  const { root } = recordingRoot();
  const counts = root.createBuffer(d.u32).$usage('storage');

  expect(() => readback.create(root, counts, { staging: 0 })).toThrow(RangeError);
  expect(() => readback.create(root, counts, { hz: 0 })).toThrow(RangeError);
  expect(() => readback.create(root, counts, { bytes: true, offset: 2 })).toThrow(RangeError);
  expect(() => readback.create(root, counts, { bytes: true, size: 8 })).toThrow(RangeError);
});

it('reads bytes without decoding the schema, through a decode of its own, or a range', async () => {
  const { root } = recordingRoot();
  const counts = root.createBuffer(d.arrayOf(d.u32, 4)).$usage('storage');
  counts.write([1, 2, 3, 4]);

  const raw = readback.create(root, counts, { bytes: true });
  const words = readback.create(root, counts, {
    bytes: true,
    staging: 1,
    decode: (bytes) => new Uint32Array(bytes).slice(2),
  });
  const middle = readback.create(root, counts, { bytes: true, staging: 1, offset: 4, size: 8 });
  expectTypeOf(raw.latest).toEqualTypeOf<ArrayBuffer | undefined>();
  expectTypeOf(words.latest).toEqualTypeOf<Uint32Array<ArrayBuffer> | undefined>();

  const loop = frames.create(root, (frame) => {
    raw.capture(frame);
    words.capture(frame);
    middle.capture(frame);
  });
  raw.request();
  words.request();
  middle.request();
  loop.run(0);
  await settle();
  counts.write([5, 6, 7, 8]);
  words.request(); // its one staging buffer is free again once decode returns
  loop.run(1);
  await settle();

  expect([...new Uint32Array(raw.latest!)]).toEqual([1, 2, 3, 4]);
  expect([...words.latest!]).toEqual([7, 8]);
  // Its staging buffer holds only the range
  expect([...new Uint32Array(middle.latest!)]).toEqual([2, 3]);
});
