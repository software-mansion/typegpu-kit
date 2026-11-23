import type { RecordingRoot } from '@typegpu/kit/testing';
import { expect, it } from 'vitest';
import { MEAL } from '../src/fishtank/food.ts';
import { createScene } from '../src/fishtank/scene.ts';
import { BLOOM, describeExample, pass, settle, SORTING, recorded } from './utils/examples.ts';

const times = (count: number, event: string) => Array<string>(count).fill(event);

/**
 * A frame's events: its dispatches in one compute pass, the copy of the bite counters when the eat
 * loop reads them, the shadow of each species, then the sand, each species, the pellets, the
 * crumbs and the bubbles, the bloom's six levels down and five up, and the final pass.
 */
const frame = (steps: number, read: boolean) => [
  'begin-compute',
  'dispatch',
  ...Array.from({ length: steps }, () => STEP).flat(),
  ...times(COMPACT, 'dispatch'),
  'end-compute',
  ...(read ? ['copy-buffer'] : []),
  ...pass(...times(3, 'draw-indexed')),
  ...pass(...times(4, 'draw-indexed'), ...times(3, 'draw-indirect')),
  ...BLOOM,
  ...pass('draw'),
  'submit',
];

/** A step's kernels: the grid's sorting the school, its steering, the crumbs, the bubbles. */
const STEP = [...SORTING, ...times(3, 'dispatch')];

/** LiveLists: two kernels each for the crumbs and bubbles, and the counts of no pellets yet. */
const COMPACT = 5;

/** The bite counters, as the queue writes and the copies left them, from the first drop's writes. */
function counters(recording: RecordingRoot, buffer: GPUBuffer) {
  const bytes = recording.bytes(buffer);
  return new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

describeExample(
  'fishtank',
  createScene,
  (build) => {
    it('compiles before it draws', async () => {
      const { log, next, warmUp } = await build();
      const warming = await recorded(log, warmUp);

      expect(log.shaders.filter((code) => code.includes('@vertex'))).toHaveLength(10);

      // Frames record nothing until the pipelines compile. The first frame after lets out bubbles,
      // steps once and starts reading the bites.
      expect(warming).not.toContain('begin-compute');
      expect(recorded(log, next)).toEqual(frame(1, true));

      // A second apart, a frame takes its three steps at most, lets out more bubbles and reads.
      const second = await recorded(log, async () => {
        next(1);
        await settle();
        next(1);
      });
      expect(second).toEqual([...frame(3, true), ...frame(3, true)]);
    });

    it('knocks crumbs off bitten pellets from its own kernel, not the boids kernel', async () => {
      const { log, warmUp } = await build();
      await warmUp();

      const kernels = log.shaders.filter((code) => code.includes('@compute'));
      const steer = kernels.find((code) => code.includes('fn bite('));
      const crumble = kernels.find(
        (code) => code.includes('fn pelletPosition(') && !code.includes('fn bite('),
      );
      expect(steer).not.toContain('fn emit(');
      expect(crumble).toContain('fn emit(');
    });

    it('drops twelve pellets for its feed button, and removes one once the fish have had their fill', async () => {
      const recording = await build();
      const { log, scene, run, warmUp } = recording;
      await warmUp();

      // Each of the twelve drops zeroes its pellet's bite counter and the bites seen of it, and
      // nothing else.
      const before = log.writes.length;
      scene.ui.controls.feed.press();
      const resets = log.writes.slice(before);
      expect(resets.map(({ offset, size }) => [offset, size])).toEqual(
        Array.from({ length: 12 }, (_, slot) => [
          [slot * 4, 4],
          [slot * 4, 4],
        ]).flat(),
      );

      // The fish bite the first pellet to its fill. The eat loop reads ten times a second.
      counters(recording, resets[0].buffer)[0] = MEAL;
      await run(8);
      expect(scene.pellets).toBe(11);
    });

    // The eaten pellet's count stays in its slot until a drop zeroes it, so a count copied before
    // the drop would have the new pellet eaten at once.
    it.each([
      ['while the copy is mapping', false],
      ['after the copy has landed', true],
    ])('keeps an old count from eating a pellet dropped into its slot %s', async (_, landed) => {
      const recording = await build();
      const { log, scene, next, run, warmUp } = recording;
      await warmUp();

      const before = log.writes.length;
      scene.ui.controls.feed.press();
      const bites = counters(recording, log.writes[before].buffer);
      bites[0] = MEAL;
      await run(8);
      expect(scene.pellets).toBe(11);

      // The next read copies the counts with the eaten one's still in slot 0.
      log.events.length = 0;
      while (!log.events.includes('copy-buffer')) {
        await settle();
        next();
      }
      if (landed) await settle();

      // The first of these takes slot 0.
      scene.ui.controls.feed.press();
      expect(bites[0]).toBe(0);
      await run(8);
      expect(scene.pellets).toBe(23);
    });
  },
  {
    // 40 frames a second, so the first frame after the warm-up steps once.
    gap: 0.025,
    // Pellets, crumbs and bubbles each compact with two kernels. The grid's seven kernels and the
    // steering, the crumbling, two particle kernels each for crumbs and bubbles, a colour and a
    // shadow pipeline for the school, pellets, crumbs, bubbles, sand, three bloom passes and the
    // final pass. At creation, a kernel filled the noise cache and four baked the meshes.
    pipelines: 34,
  },
);
