import { expect, it } from 'vitest';
import { createScene } from '../src/sph-sculpt/scene.ts';
import {
  compute,
  describeExample,
  pass,
  SORTING,
  recorded,
  shadersWith,
} from './utils/examples.ts';

const times = (count: number, event: string) => Array<string>(count).fill(event);

/** The island's height columns and material volume, then each of the seven mips. */
const GENERATE = times(2 + 7, 'dispatch');
/** The brush's aim, then a compute and a store at each mip. */
const EDIT = times(1 + 2 * 7, 'dispatch');
/**
 * Integrate, sort into the grid and list the tiles near water, count, then density and relax over
 * the sorted particles.
 */
const STEP = [
  'dispatch',
  ...SORTING,
  'dispatch-indirect',
  'dispatch',
  'dispatch-indirect',
  'dispatch-indirect',
];
/** Clear the pool, then sort it empty and list its tiles in a pass of its own, without a count. */
const DRAIN = ['clear-buffer', 'begin-compute', ...SORTING, 'dispatch-indirect', 'end-compute'];
/** Rebuild the tiles near water, then the coarse tile gaps. */
const SURFACE = ['dispatch-indirect', 'dispatch'];
/** The material noise and the morning sky's bake into the environment. */
const BAKE = [...compute(), ...compute(15)];
/** The cone pre-pass, the clay and water traces, the shadow pass and the material pass. */
const RENDER = times(5, 'dispatch');
/** The final pass into a display-range target, then FXAA onto the canvas. */
const FINISH = [...pass('draw'), ...pass('draw')];

/** A frame's events: every kernel in one compute pass, then the final and FXAA passes. */
const frame = (...dispatches: string[]) => [
  'begin-compute',
  ...dispatches,
  ...SURFACE,
  ...RENDER,
  'end-compute',
  ...FINISH,
];

describeExample(
  'sph-sculpt',
  createScene,
  (build) => {
    it('generates the island, then steps the water twice a frame', async () => {
      const { log, next } = await build();

      // At time zero, the frame bakes the noise and sky, drains the water and generates the island.
      // Nothing steps, and the surface is built over the drained pool.
      expect(recorded(log, next)).toEqual([
        ...BAKE,
        ...DRAIN,
        ...frame(...GENERATE),
        'copy-buffer',
        'submit',
      ]);

      // 40 ms later the fixed 1/120 s step catches up two steps at most
      expect(recorded(log, next)).toEqual([...frame(...STEP, ...STEP), 'submit']);
    });

    it('digs where the cursor is dragged, and freezes the ground to build on', async () => {
      const { log, next, scene } = await build();
      next();
      scene.pointing.drag(0.5, 0.5, true);
      next();
      expect(recorded(log, next)).toEqual([...frame(...EDIT, ...STEP, ...STEP), 'submit']);

      // Holding build starts a new stroke, which first maps the ground as it is
      scene.ui.controls.build.press();
      expect(recorded(log, next)).toEqual([
        ...frame('dispatch', ...EDIT, ...STEP, ...STEP),
        'submit',
      ]);

      scene.pointing.end();
      expect(recorded(log, next)).toEqual([...frame(...STEP, ...STEP), 'submit']);
    });

    it('digs a new stroke where the finger lands, once a trace has found it', async () => {
      const { log, next, scene } = await build();
      next();
      scene.pointing.drag(0.2, 0.3, true);
      next();
      next();
      scene.pointing.end();
      next();

      // The finger lands elsewhere: this frame's trace finds it, and the next frame digs there
      scene.pointing.drag(0.8, 0.6, true);
      expect(recorded(log, next)).toEqual([...frame(...STEP, ...STEP), 'submit']);
      expect(recorded(log, next)).toEqual([...frame(...EDIT, ...STEP, ...STEP), 'submit']);
    });

    it('regenerates the island and drains the water on reset', async () => {
      const { log, next, scene } = await build();
      next();
      next();
      scene.settings.actions?.reset?.();
      expect(recorded(log, next)).toEqual([
        ...DRAIN,
        ...frame(...GENERATE, ...STEP, ...STEP),
        'submit',
      ]);
    });
  },
  {
    features: ['texture-formats-tier1'],
    // Seven clay kernels, six water kernels, eight grid kernels, five render kernels, the noise
    // bake and its cache, the sky's six bake kernels and the final pass
    pipelines: 36,
  },
);

it('keeps 8-bit distances with texture-formats-tier1, and rgba8unorm without', async () => {
  const narrow = await shadersWith(createScene, ['texture-formats-tier1']);
  expect(narrow).toContain('texture_storage_3d<r8unorm, write>');
  expect(narrow).toContain('texture_storage_2d<rgb10a2unorm, write>');
  const wide = await shadersWith(createScene, []);
  expect(wide).toContain('texture_storage_3d<rgba8unorm, write>');
  expect(wide).not.toMatch(/\b(r8|rg8|rgb10a2)unorm/);
});
