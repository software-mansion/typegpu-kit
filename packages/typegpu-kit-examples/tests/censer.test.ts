import { offscreenSurface, recordingRoot } from '@typegpu/kit/testing';
import { expect, it } from 'vitest';
import { createScene } from '../src/censer/scene.ts';
import { compute, describeExample, pass, type Built } from './utils/examples.ts';

/** Floats of the censer's pose as it last wrote them, at byte offsets: the vents or the middle. */
function poseAt({ log, bytes }: Pick<Built<unknown>, 'log' | 'bytes'>, ...offsets: number[]) {
  const write = log.writes.findLast(({ buffer }) => buffer.label === 'censer pose')!;
  const data = bytes(write.buffer);
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  return offsets.map((at) => view.getFloat32(at, true));
}
const VENTS = [96, 100];
const MIDDLE = [48, 52, 56];

/** A smoke step's first half: solids, curl, the air's advection, the finer smoke's and divergence. */
const ADVECT = 5;
/**
 * Its second half, the pressure: the solid marks of three coarser grids, one V-cycle of red and black
 * sweeps, one on the finest grid and two on the next two each way, a restriction or prolongation on
 * each of the finer grids and 8 sweeps on the coarsest, and the projection.
 */
const PROJECT = 6 + (13 + 16 + 13) + 1;
/** Finding the bricks the smoke fills and lighting it, after a first half. */
const LIGHT = 2;
/** The censer's draw, the march, the smoke laid on and the final pass. */
const DRAWN = [...pass('draw', 'draw'), ...pass('draw'), ...pass('draw'), ...pass('draw')];

describeExample(
  'censer',
  createScene,
  (build) => {
    it('steps the smoke, lights it, draws the censer, marches and lays the smoke and finishes', async () => {
      const { log, next } = await build();
      next();
      log.events.length = 0;
      next();
      // Two smoke steps, as a frame 0.04 s long records them
      expect(log.events).toEqual([...compute(2 * (ADVECT + PROJECT) + LIGHT), ...DRAWN, 'submit']);
    });

    it('records half a smoke step a frame at 120 Hz, even after one that misses the display', async () => {
      const { log, next } = await build();
      next();
      next();
      const recorded = [1, 1, 1, 1, 2].map((frames) => {
        log.events.length = 0;
        next(frames / 120);
        return [...log.events];
      });
      const moved = [...compute(ADVECT + LIGHT), ...DRAWN, 'submit'];
      const projected = [...compute(PROJECT), ...DRAWN, 'submit'];
      expect(recorded).toEqual([moved, projected, moved, projected, moved]);
    });

    it('scales the smoke and heat the vents breathe out, creating nothing', async () => {
      const { log, scene, bytes, next } = await build();
      next();
      next();
      expect(poseAt({ log, bytes }, ...VENTS)).toEqual([Math.fround(1.3), 17]);
      const counts = log.counts();
      scene.settings.values.emission = 4;
      await scene.settings.write!('emission');
      next();
      expect(poseAt({ log, bytes }, ...VENTS)).toEqual([Math.fround(1.3 * 4), 68]);
      expect(log.counts()).toEqual(counts);
    });

    it('takes hold of the censer under a drag and orbits from anywhere else, creating nothing', async () => {
      const built = await build();
      const { log, scene, next } = built;
      const middle = () => poseAt(built, ...MIDDLE);
      next();
      next();
      // A frame of each half at 120 Hz, so the march has bound the smoke both ways
      next(1 / 120);
      next(1 / 120);
      const counts = log.counts();
      const seen = [0, 0, 0];
      expect(scene.view.projectAt(middle(), seen)).toBe(true);
      const yaw = scene.view.goal.yaw;
      scene.pointing.drag(seen[0], seen[1], true);
      for (let i = 1; i <= 10; i++) {
        scene.pointing.drag(seen[0] + i * 0.03, seen[1], false);
        next(1 / 60);
      }
      scene.pointing.end();
      expect(scene.view.goal.yaw).toBe(yaw);
      expect(middle()[0]).toBeGreaterThan(0.5);
      for (let i = 0; i < 4; i++) next(1 / 120);
      expect(log.counts()).toEqual(counts);

      scene.pointing.drag(0.02, 0.02, true);
      scene.pointing.drag(0.12, 0.02, false);
      scene.pointing.end();
      expect(scene.view.goal.yaw).toBeLessThan(yaw);
    });
  },
  { pipelines: 20, features: ['texture-formats-tier1'] },
);

/** The WGSL of a scene built on a device with `features`. */
const shadersWith = async (features: GPUFeatureName[]) => {
  const { root, log } = recordingRoot({ features });
  const surface = offscreenSurface(root, { width: 64, height: 36, format: 'bgra8unorm' });
  (await createScene(root, surface)).destroy();
  return log.shaders.join('\n');
};

it('stores the smoke and heat in two halves with texture-formats-tier1, and rgba16float without', async () => {
  expect(await shadersWith(['texture-formats-tier1'])).toContain(
    'texture_storage_3d<rg16float, write>',
  );
  expect(await shadersWith([])).not.toContain('rg16float');
});
