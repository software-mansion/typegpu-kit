import { describe, expect, it } from 'vitest';
import { type Frame } from '../src/frames.ts';
import { frames } from '../src/index.ts';
import { recordingRoot } from '../src/testing.ts';

describe('frames.create', () => {
  it('submits one encoder per frame and advances the clock only on submission', () => {
    const { root, log } = recordingRoot();
    const seen: Pick<Frame, 'time' | 'dt' | 'index'>[] = [];
    const loop = frames.create(root, ({ time, dt, index }) => {
      seen.push({ time, dt, index });
    });

    expect(loop.run(10)).toBe(true);
    expect(loop.run(10.5)).toBe(true);
    expect(loop.run(11.25)).toBe(true);

    expect(log.submissions).toBe(3);
    expect(seen).toEqual([
      { time: 0, dt: 0, index: 0 },
      { time: 0.5, dt: 0.5, index: 1 },
      { time: 1.25, dt: 0.75, index: 2 },
    ]);
  });

  it('drops a frame whose recorder throws', () => {
    const { root, log } = recordingRoot();
    let fail = true;
    const indices: number[] = [];
    const loop = frames.create(root, ({ index }) => {
      if (fail) throw new Error('recording failed');
      indices.push(index);
    });

    expect(() => loop.run(1)).toThrow('recording failed');
    fail = false;
    loop.run(2);

    expect(log.submissions).toBe(1);
    expect(indices).toEqual([0]);
  });

  it('rejects asynchronous recorders, nested frames and time running backwards', () => {
    const { root, log } = recordingRoot();

    const asyncFrames = frames.create(root, (async () => {}) as () => void);
    expect(() => asyncFrames.run(0)).toThrow('synchronous');

    const nested = frames.create(root, () => {
      nested.run(1);
    });
    expect(() => nested.run(0)).toThrow('inside another frame');

    const ordered = frames.create(root, () => {});
    ordered.run(2);
    expect(() => ordered.run(1)).toThrow(RangeError);
    expect(() => ordered.run(Number.NaN)).toThrow(RangeError);
    expect(log.submissions).toBe(1);
  });

  it('stops submitting after destroy, including from inside a recorder', () => {
    const { root, log } = recordingRoot();
    const loop = frames.create(root, ({ index }) => {
      if (index === 1) loop.destroy();
    });

    expect(loop.run(0)).toBe(true);
    expect(loop.run(1)).toBe(false);
    expect(loop.run(2)).toBe(false);
    expect(log.submissions).toBe(1);
  });

  it('runs after-submit callbacks once, after the submit, in order', () => {
    const { root, log } = recordingRoot();
    const calls: string[] = [];
    const first = () => calls.push(`first ${log.submissions}`);
    const second = () => calls.push(`second ${log.submissions}`);

    const loop = frames.create(root, (frame) => {
      frame.afterSubmit(first);
      frame.afterSubmit(second);
    });

    loop.run(0);
    loop.run(1);

    expect(calls).toEqual(['first 1', 'second 1', 'first 2', 'second 2']);
  });

  it('drops after-submit callbacks of frames that are not submitted', () => {
    const { root } = recordingRoot();
    let calls = 0;
    const callback = () => calls++;
    let mode: 'throw' | 'destroy' = 'throw';

    const loop = frames.create(root, (frame) => {
      frame.afterSubmit(callback);
      if (mode === 'throw') throw new Error('recording failed');
      loop.destroy();
    });

    expect(() => loop.run(0)).toThrow('recording failed');
    mode = 'destroy';
    expect(loop.run(1)).toBe(false);
    expect(calls).toBe(0);
  });

  it('makes a new frame per run, and a kept frame can’t add callbacks to a later one', () => {
    const { root } = recordingRoot();
    const seen: Frame[] = [];
    let calls = 0;
    const callback = () => calls++;

    const loop = frames.create(root, (frame) => {
      if (seen.length > 0) {
        expect(() => seen[0].afterSubmit(callback)).toThrow('while its frame is recording');
      }
      seen.push(frame);
    });

    loop.run(0);
    loop.run(1);

    expect(seen[0]).not.toBe(seen[1]);
    expect(calls).toBe(0);
  });

  it('runs every after-submit callback before rethrowing, and still counts the frame', () => {
    const { root, log } = recordingRoot();
    let reached = false;
    const indices: number[] = [];

    const loop = frames.create(root, (frame) => {
      indices.push(frame.index);
      if (frame.index > 0) return;

      frame.afterSubmit(() => {
        throw new Error('first');
      });
      frame.afterSubmit(() => {
        throw new Error('second');
      });
      frame.afterSubmit(() => {
        reached = true;
      });
    });

    expect(() => loop.run(0)).toThrow(AggregateError);
    expect(reached).toBe(true);
    expect(loop.run(1)).toBe(true);
    expect(indices).toEqual([0, 1]);
    expect(log.submissions).toBe(2);
  });
});

describe('frame.submit', () => {
  it('submits what is recorded so far into a fresh encoder, and the loop submits the rest', () => {
    const { root, log } = recordingRoot();
    const encoders: unknown[] = [];

    const loop = frames.create(root, (frame) => {
      encoders.push(frame.encoder);
      frame.encoder.beginComputePass().end();
      frame.submit();

      encoders.push(frame.encoder);
      frame.encoder.beginComputePass().end();
    });

    loop.run(0);

    expect(log.events).toEqual([
      'begin-compute',
      'end-compute',
      'submit',
      'begin-compute',
      'end-compute',
      'submit',
    ]);
    expect(encoders[1]).not.toBe(encoders[0]);
  });

  it('counts one submit per call, sends no empty command buffer, and keeps one clock tick', () => {
    const { root, log } = recordingRoot();
    const indices: number[] = [];

    const loop = frames.create(root, (frame) => {
      indices.push(frame.index);
      for (let step = 0; step < 3; step++) {
        frame.encoder.beginComputePass().end();
        frame.submit();
      }
    });

    loop.run(0);
    loop.run(1);

    expect(log.submissions).toBe(6);
    expect(indices).toEqual([0, 1]);
  });

  it('runs after-submit callbacks after the frame’s last submit', () => {
    const { root, log } = recordingRoot();
    const calls: number[] = [];
    const callback = () => calls.push(log.submissions);

    const loop = frames.create(root, (frame) => {
      frame.afterSubmit(callback);
      frame.submit();
      frame.encoder.beginComputePass().end();
    });

    loop.run(0);

    expect(calls).toEqual([2]);
  });

  it('keeps what it sent when the recorder throws later, but the frame doesn’t count', () => {
    const { root, log } = recordingRoot();
    let fail = true;
    let calls = 0;
    const callback = () => calls++;
    const indices: number[] = [];

    const loop = frames.create(root, (frame) => {
      indices.push(frame.index);
      frame.afterSubmit(callback);
      frame.submit();
      if (fail) throw new Error('recording failed');
    });

    expect(() => loop.run(0)).toThrow('recording failed');
    expect(log.submissions).toBe(1);
    expect(calls).toBe(0);

    fail = false;
    loop.run(1);
    expect(indices).toEqual([0, 0]);
    expect(calls).toBe(1);
  });

  it('throws outside its frame’s recording, and sends nothing once the loop is destroyed', () => {
    const { root, log } = recordingRoot();
    let stale: Frame | undefined;

    const kept = frames.create(root, (frame) => {
      stale = frame;
    });
    kept.run(0);
    expect(() => stale!.submit()).toThrow('while its frame is recording');

    const ended = frames.create(root, (frame) => {
      frame.encoder.beginComputePass().end();
      ended.destroy();
      frame.submit();
    });

    expect(ended.run(0)).toBe(false);
    expect(log.submissions).toBe(1);
  });
});
