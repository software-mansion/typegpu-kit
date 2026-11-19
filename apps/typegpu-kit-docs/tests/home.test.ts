import { frames } from '@typegpu/kit';
import { offscreenSurface, recordingRoot } from '@typegpu/kit/testing';
import { expect, it } from 'vitest';
import { createHome, type Drawn } from '../src/components/home/home.ts';
import { KNOBS, type Knobs } from '../src/components/home/knobs.ts';
import { KIND, type Kind, type SheetState } from '../src/components/home/stack.ts';
import { createSubject } from '../src/components/home/subject.ts';

/** A lockup whose every fourth pixel is ink, for the loose cells to come from. */
const read = (columns: number, rows: number) => {
  const data = new Uint8ClampedArray(columns * rows * 4);
  for (let i = 0; i < columns * rows; i += 4) data.set([120, 90, 220, 255], i * 4);
  return data;
};

/** Every kind of sheet, a sheet apart. */
const sheets: SheetState[] = (Object.keys(KIND) as Kind[]).map((kind, i) => ({
  kind,
  x: 0,
  y: -i * 0.7,
  z: 0,
  half: 2,
  hot: i === 1 ? 1 : 0,
  shown: 1,
  keeps: 1,
}));

it('records the frame, both stacks and the lockup in one submit, creates nothing in a steady frame and frees what it made', () => {
  const { root, log } = recordingRoot({ features: ['indirect-first-instance'] });
  const before = log.alive({ uniforms: false });
  const home = createHome(root, { format: 'bgra8unorm', read });
  const surfaces = [0, 1].map(() =>
    offscreenSurface(root, { width: 64, height: 40, format: 'bgra8unorm' }),
  );
  const camera = { yaw: 0.8, pitch: 0.8, scale: 6, target: [0, 0, 0], anchor: [0.5, 0.5] } as const;
  const { ink } = home;
  ink.layout([4, 4, 40, 10]);

  const drawn: Drawn[] = surfaces.map((surface, i) => {
    const view = home.view(surface, () => 2);
    const count = view.write(
      { ...camera, target: [...camera.target], anchor: [...camera.anchor] },
      sheets,
      { base: 0.1, line: 0.6, thick: 0.035 },
    );
    return i === 0
      ? {
          view,
          count,
          before: (frame, dt) => ink.step(frame, dt, 1, surface, false),
          after: (pass) => ink.draw(pass),
        }
      : { view, count };
  });
  const loop = frames.create(root, (frame) => home.record(frame, 1 / 60, drawn));
  log.events.length = 0;

  loop.run(0);
  loop.run(1 / 60);
  const steady = log.counts();
  loop.run(2 / 60);
  expect(log.counts()).toEqual(steady);
  expect(log.events.filter((event) => event === 'submit')).toHaveLength(3);

  loop.destroy();
  home.destroy();
  for (const surface of surfaces) surface.destroy();
  expect(log.alive({ uniforms: false })).toEqual(before);
});

it('writes each number the reader drags into what the next frame reads', () => {
  const { root, log, bytes } = recordingRoot({ features: ['indirect-first-instance'] });
  const subject = createSubject(root);
  const loop = frames.create(root, (frame) => subject.record(frame, 0));
  loop.run(0);
  const written = () =>
    new Map(log.writes.map(({ buffer }) => [buffer, bytes(buffer).join()] as const));

  for (const key of Object.keys(KNOBS) as (keyof Knobs)[]) {
    const before = written();
    const { min, max, value } = KNOBS[key];
    subject.knob(key, value === max ? min : max);
    loop.run(0);
    const changed = [...written()].some(([buffer, now]) => before.get(buffer) !== now);
    expect(changed, key).toBe(true);
  }

  loop.destroy();
  subject.destroy();
});
