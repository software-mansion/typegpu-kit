import { expect, it } from 'vitest';
import { createScene } from '../src/lattice/scene.ts';
import { compute, describeExample, pass, recorded } from './utils/examples.ts';

/** The scene's own events in a frame of `steps` steps, up to and including the TAA's resolve. */
const drawn = (steps: number) => [
  // The steps and the live lists.
  ...compute(steps + 2),
  // The lattice's turn: this frame's kept as last frame's, then the new one.
  'copy-buffer',
  'copy-buffer',
  // The lattice, the sky and the embers, into colour, motion and depth.
  ...pass('draw', 'draw', 'draw-indirect'),
  // The resolve, then the next jitter copied in.
  ...pass('draw'),
  'copy-buffer',
];

/** The pipeline that drew each of the scene pass's draws, by what its WGSL holds. */
const shaderWith = (shaders: string[], text: string) =>
  shaders.filter((code) => code.includes(text));

describeExample(
  'lattice',
  createScene,
  (build) => {
    it('draws the lattice, the sky and the embers in one pass, then resolves, blooms and draws the picture', async () => {
      const { log, next } = await build();
      next();
      next();
      const events = recorded(log, next);
      expect(events.slice(0, drawn(3).length)).toEqual(drawn(3));
      expect(events.at(-1)).toBe('submit');
    });

    it('writes motion from the lattice by its last turn and the embers by their velocity, and leaves the sky’s to TAA', async () => {
      const { log, next } = await build();
      next();

      // Both lattices and the embers in the scene read this frame's camera and last frame's.
      const drawing = log.shaders.filter(
        (code) => code.match(/var<uniform> \w+: Camera/g)?.length === 2,
      );
      expect(drawing).toHaveLength(3);
      // Each carries its clip positions now and last frame.
      const carrying = shaderWith(drawing, 'motion_0');
      expect(carrying).toHaveLength(3);
      // The sky writes motion's clear value, so TAA reconstructs its motion from the far depth.
      const sky = shaderWith(log.shaders, 'fn farAt');
      expect(sky).toHaveLength(1);
      expect(sky[0]).not.toContain('motion_0');
      // The turn now and last frame's from its history; the plain uniform's is one turn for both.
      const turns = shaderWith(carrying, 'fn placed').map(
        (code) => code.match(/var<uniform> \w+: f32/g)?.length,
      );
      expect(turns.toSorted()).toEqual([1, 2]);
      // The embers left out of motion carry nothing for it.
      const embers = shaderWith(log.shaders, 'fn glowing');
      expect(embers).toHaveLength(2);
      expect(shaderWith(embers, 'motion_0')).toHaveLength(1);
    });

    it('draws the drawings turned on, the embers into either description, and shows the depth', async () => {
      const { log, scene, next } = await build();
      const { values } = scene.settings;
      const write = scene.settings.write!;
      next();

      /** The draws of the scene's pass in the next frame. */
      const drawsOfNext = () => {
        const events = recorded(log, next);
        const begin = events.indexOf('begin-render');
        return events.slice(begin + 1, events.indexOf('end-render', begin));
      };

      values.sky = false;
      values.embers = false;
      expect(drawsOfNext()).toEqual(['draw']);

      values.beams = false;
      values.embers = true;
      values.into = "scene.without('motion')";
      expect(drawsOfNext()).toEqual(['draw-indirect']);

      values.view = 'depth';
      write('view');
      expect(recorded(log, next).slice(-4)).toEqual([...pass('draw'), 'submit']);
      // The two resolves' and the depth view's.
      expect(shaderWith(log.shaders, 'texture_depth_2d')).toHaveLength(3);
    });

    it('draws two thirds of each side and resolves them up to the canvas, creating nothing in a steady frame', async () => {
      const { log, scene, next } = await build();
      const { values } = scene.settings;
      next();

      values.scale = '67%';
      scene.settings.write!('scale');
      next();
      next();
      const counts = log.counts();
      const events = recorded(log, next);
      expect(log.counts()).toEqual(counts);
      expect(events.slice(0, drawn(3).length)).toEqual(drawn(3));
      // The resolve up to the canvas gathers the smaller picture around each pixel.
      expect(shaderWith(log.shaders, 'Gathered')).toHaveLength(1);
    });

    it('shows the motion in place of the picture, skipping the bloom, and the resolve with TAA off', async () => {
      const { log, scene, next } = await build();
      const { values } = scene.settings;
      const write = scene.settings.write!;

      next();
      next();

      values.view = 'motion';
      write('view');
      expect(recorded(log, next).slice(drawn(3).length)).toEqual([...pass('draw'), 'submit']);

      values.taa = false;
      write('taa');
      const resolve = [...pass('draw'), 'copy-buffer'];
      expect(recorded(log, next).slice(drawn(3).length - resolve.length)).toEqual([
        ...pass('draw'),
        'submit',
      ]);
    });
  },
  {
    // A frame every 0.05 s takes three steps of the embers.
    gap: 0.05,
    // The sky, the lattice turned by its history and by a plain uniform, the embers' spawn, step,
    // liveLists (2) and draw into the scene and into it without motion, the resolves at the drawn
    // size and up to the canvas's, the bloom's 3, the final pass, and the motion and depth views.
    pipelines: 17,
  },
);
