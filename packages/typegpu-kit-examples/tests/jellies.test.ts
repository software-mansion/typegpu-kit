import { expect, it } from 'vitest';
import { createScene } from '../src/jellies/scene.ts';
import { compute, describeExample, pass, recorded } from './utils/examples.ts';

const render = pass('draw');
/** The water, marine snow, tentacles, oral arms, bells and bubbles, in one pass. */
const tank = pass('draw', 'draw', 'draw-indexed', 'draw', 'draw', 'draw-indirect');

/** The render passes of a frame, after its kernels. */
const passesOf = (events: readonly string[]) => events.slice(events.indexOf('end-compute') + 1);

describeExample(
  'jellies',
  createScene,
  (build) => {
    it('draws the tank translucent, then lays it over the water in a pass of its own', async () => {
      const { log, scene, next } = await build();
      next();
      next();

      const layered = passesOf(recorded(log, next));
      expect(layered.slice(0, tank.length + render.length)).toEqual([...tank, ...render]);

      // Without order-independence the same drawings blend over in order, with no composite
      scene.settings.values.oit = false;
      await scene.settings.write!('oit');
      expect(log.pipelines, 'the six drawings without the targets').toBe(26);
      const ordered = passesOf(recorded(log, next));
      expect(ordered.slice(0, tank.length)).toEqual(tank);
      expect(ordered.slice(tank.length)).toEqual(layered.slice(tank.length + render.length));
    });

    it('picks once for a tap and startles what it found', async () => {
      const { log, scene, next } = await build();
      next();
      next();
      /** The kernels of a frame that takes no step: the bubbles' alone. */
      const kernels = () => {
        const events = recorded(log, () => next(0));
        return events.slice(0, events.indexOf('end-compute') + 1);
      };
      expect(kernels()).toEqual(compute(2));

      scene.pointing!.press!(0.5, 0.5);
      expect(kernels()).toEqual(compute(4));
      expect(kernels()).toEqual(compute(2));
    });

    it('sways about the yaw an orbit drag leaves', async () => {
      const { scene, next } = await build();
      next();
      const start = scene.view.goal.yaw;
      scene.view.goal.yaw += 1;
      for (let i = 0; i < 10; i++) next();
      expect(scene.view.goal.yaw - start).toBeCloseTo(1, 1);
    });
  },
  {
    // Six drawings, the composite, the swim, trail, pick and startle kernels, the bubbles'
    // kernels, bloom and the final pass
    pipelines: 20,
  },
);
