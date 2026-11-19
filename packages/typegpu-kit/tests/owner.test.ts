import { d } from 'typegpu';
import { describe, expect, it } from 'vitest';
import { owner, pipelines, targets } from '../src/index.ts';
import { recordingRoot } from '../src/testing.ts';

describe('owner', () => {
  it('frees what it took, modules and fixed resources alike, back to what was alive before', () => {
    const { root, log } = recordingRoot();
    const before = log.alive();

    const size = { width: 8, height: 8 };
    const own = owner();
    own(targets.create(root, size, { hdr: { format: 'rgba16float' } }));
    const time = own(root.createUniform(d.f32));
    own(root.createBuffer(d.arrayOf(d.u32, 4)).$usage('storage'));
    time.write(1);
    expect(own(undefined)).toBeUndefined();
    expect(log.alive().length).toBeGreaterThan(before.length);

    own.destroy();
    expect(own.destroyed).toBe(true);
    expect(log.alive()).toEqual(before);
  });

  it('frees the last taken first, once', () => {
    const freed: string[] = [];
    const thing = (name: string) => ({ destroy: () => freed.push(name) });
    const own = owner();
    own(thing('screen'));
    own(thing('glow'));
    own({ buffer: thing('time') });
    own({ [Symbol.dispose]: () => freed.push('listener') });

    own.destroy();
    own[Symbol.dispose]();
    expect(freed).toEqual(['listener', 'time', 'glow', 'screen']);
  });

  it('lists what it took that compiles, which `pipelines.compile` takes in place of a list', async () => {
    const started: string[] = [];
    const part = (name: string) => ({ initAsync: async () => void started.push(name) });
    const own = owner();
    own({ ...part('scene'), destroy() {} });
    own({ destroy() {} });
    // A pipeline has nothing to free
    own(part('kernel'));

    const progress: number[] = [];
    await pipelines.compile(own, { onProgress: (fraction) => progress.push(fraction) });
    expect(started).toEqual(['scene', 'kernel']);
    expect(progress).toEqual([0.5, 1]);

    own.destroy();
    expect(own.warmables).toEqual([]);
  });

  it('runs cleanups among what it took, and destroys an owner it took rather than calling it', () => {
    const freed: string[] = [];
    const inner = owner();
    inner({ destroy: () => freed.push('inner') });
    const own = owner();
    own(() => freed.push('cleanup'));
    own(inner);
    own({ destroy: () => freed.push('last') });

    own.destroy();
    expect(freed).toEqual(['last', 'inner', 'cleanup']);
    expect(inner.destroyed).toBe(true);
  });

  it('throws its message from usable once destroyed', () => {
    const own = owner('The swarm was destroyed.');
    const { usable } = own;
    expect(usable).not.toThrow();

    own.destroy();
    expect(usable).toThrow('The swarm was destroyed.');

    const plain = owner();
    plain.destroy();
    expect(plain.usable).toThrow('It was destroyed.');
  });

  it('frees at once what it takes after destroy', () => {
    const freed: string[] = [];
    const own = owner();
    own.destroy();

    const late = { destroy: () => freed.push('late') };
    expect(own(late)).toBe(late);
    expect(freed).toEqual(['late']);
  });

  it('frees the rest when one throws, then rethrows', () => {
    const freed: string[] = [];
    const own = owner();
    own({ destroy: () => freed.push('first') });
    own({
      destroy() {
        throw new Error('stuck');
      },
    });
    own({ destroy: () => freed.push('last') });

    expect(() => own.destroy()).toThrow('stuck');
    expect(freed).toEqual(['last', 'first']);
    expect(() => own.destroy()).not.toThrow();
  });
});
