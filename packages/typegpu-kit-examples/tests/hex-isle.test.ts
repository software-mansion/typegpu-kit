import { Camera } from '@typegpu/kit';
import { d } from 'typegpu';
import { expect, it, vi } from 'vitest';
import { resolve } from '../src/assets.ts';
import { assets } from '../src/hex-isle/assets.ts';
import { layout, MAP } from '../src/hex-isle/island.ts';
import { createScene } from '../src/hex-isle/scene.ts';
import { daylight } from '../src/hex-isle/sky.ts';
import { BLOOM, describeExample, fromDisk, pass, white } from './utils/examples.ts';

/** A draw for each of the 22 models the island uses, a mesh each: all but the fourth beach. */
const meshes = Array<string>(22).fill('draw-indexed');

/**
 * A frame's events: the shadow, the half-resolution depth and the scene, the depth of field's
 * four passes, the bloom's six levels down and five up, and the final pass. The pieces are listed
 * on the CPU, with no pass.
 */
const frame = () => [
  ...pass(...meshes),
  ...pass(...meshes),
  ...pass(...meshes),
  ...Array.from({ length: 4 }, () => pass('draw')).flat(),
  ...BLOOM,
  ...pass('draw'),
  'submit',
];

/**
 * What a new hour writes: the sun, then the sky's top and bottom and the air's colour, into the
 * lighting's settings, each run of fields a write. The shadow's record then aims it: the sun's view for its casters and
 * its light space for shading.
 */
const sky = [28, 12, 12, 12];
const aim = [d.sizeOf(Camera), 64 + 4 * 32];

/** Each beach's edges facing the sea as its model has them, +x first, turning towards +z. */
const SHORES: Record<string, number[]> = {
  'coast-a': [1],
  'coast-b': [1, 2],
  'coast-c': [0, 1, 2],
  'coast-d': [5, 0, 1, 2],
};

const key = (x: number, z: number) => `${Math.round(x)},${Math.round(z / Math.sqrt(3))}`;

describeExample(
  'hex-isle',
  createScene,
  (build) => {
    it('draws the shadow and the scene', async () => {
      const { log, next } = await build();
      next();
      next();
      expect(log.events).toEqual([...frame(), ...frame()]);
    });

    it('lights the isle and aims the shadow only when the hour moves', async () => {
      const { log, scene, next } = await build();
      const sizes = (from: number) => log.writes.slice(from).map(({ size }) => size);
      scene.settings.values.paused = true;
      next();
      next();

      // Paused, a frame writes the time and nothing else.
      let from = log.writes.length;
      next();
      next();
      expect(sizes(from)).toEqual([4, 4]);

      // Moving the hour, as the slider does, writes the light once. The lamps stay dark at noon.
      from = log.writes.length;
      scene.settings.values.hour = 12;
      scene.settings.write!('hour');
      next();
      expect(sizes(from).slice(0, sky.length)).toEqual(sky);
      expect(sizes(from).slice(-aim.length)).toEqual(aim);

      from = log.writes.length;
      scene.settings.values.paused = false;
      next();
      expect(sizes(from)).toEqual([4, ...sky, ...aim]);
      expect(scene.settings.values.hour).toBeCloseTo(12 + 0.04 * scene.settings.values.speed);
    });

    it('moves the clouds and the fan in the shadow pass too', async () => {
      const { log } = await build();

      const isle = log.shaders.filter((code) => code.includes('@vertex') && code.includes('Piece'));
      // The shadow and the depth for the depth of field draw depth alone, with no fragment stage.
      expect(isle.filter((code) => !code.includes('@fragment'))).toHaveLength(2);
      for (const code of isle) expect(code).toContain('fract(');
    });

    it('decodes the atlas its models share once, so the pool uploads it once', async () => {
      const decode = vi.fn(white);
      const { models } = await resolve('hex-isle', assets, { ...fromDisk, decode });
      const images = new Set(Object.values(models).flatMap((model) => model.images));

      expect(decode).toHaveBeenCalledOnce();
      expect(images.size).toBe(1);
    });

    it('lays a beach along every shore, turned to face the sea', () => {
      const { pieces } = layout();
      const land = new Set(
        pieces
          .filter(({ model, position }) => model !== 'water' && position[1] === 0)
          .map(({ position }) => key(position[0], position[2])),
      );

      const beaches = pieces.filter(({ model }) => model in SHORES);
      expect(beaches.length).toBeGreaterThan(10);

      for (const { model, position, yaw } of beaches) {
        const wet = SHORES[model].map((edge) => {
          // The model's edge, turned by the piece's yaw about +y.
          const angle = (edge * Math.PI) / 3 - yaw;
          return !land.has(
            key(position[0] + 2 * Math.cos(angle), position[2] + 2 * Math.sin(angle)),
          );
        });
        expect(wet).toEqual(wet.map(() => true));
      }
    });

    it('refuses a house on a beach, and a shore that no beach fits', () => {
      expect(() => layout(['  . . .  ', ' h . . . ', '  . . .  '])).toThrow(
        'The home-a at column 1, row 1 stands on a beach.',
      );
      expect(() => layout(['. . .'])).toThrow(
        'The tile at column 0, row 0 has no beach that fits its coast.',
      );
      expect(() => layout(['. .', '. .'])).toThrow('The tile at column 0, row 1 is off the grid.');
    });

    it('hangs a lamp in each house, and the fan on the windmill', () => {
      const { pieces, lamps } = layout();
      const houses = pieces.filter(({ glow = 0 }) => glow > 0);
      expect(lamps).toHaveLength(houses.length);
      expect(houses.length).toBe(MAP.join('').replace(/[^hHctmb]/g, '').length);

      const mill = pieces.find(({ model }) => model === 'windmill')!;
      const fan = pieces.find(({ model }) => model === 'windmill-fan')!;
      expect(fan.yaw).toBe(mill.yaw);
      expect(
        Math.hypot(fan.position[0] - mill.position[0], fan.position[2] - mill.position[2]),
      ).toBeCloseTo(0.332);
      expect(fan.spin).toBeGreaterThan(0);
    });

    it('turns day into night', () => {
      // The sun stands high in the south at noon, and the moon at midnight.
      const noon = daylight(12);
      expect(noon.sunDirection[1]).toBeGreaterThan(0.9);
      expect(noon.sunDirection[2]).toBeGreaterThan(0);
      expect(noon.sunIntensity).toBeGreaterThan(1);
      expect(noon.lamps).toBe(0);

      // The sun fades out on the horizon, where the moon takes over the light.
      for (const hour of [6, 18]) expect(daylight(hour).sunIntensity).toBeCloseTo(0);

      const midnight = daylight(0);
      midnight.sunDirection.forEach((value, i) => expect(value).toBeCloseTo(noon.sunDirection[i]));
      expect(midnight.sunIntensity).toBeCloseTo(0.25);
      expect(midnight.lamps).toBe(1);
      expect(midnight.horizon[2]).toBeLessThan(noon.horizon[2]);
    });
  },
  {
    // The island is a colour, a shadow and a depth pipeline for all of its 23 meshes, with no device
    // feature. Then the depth of field's four passes and its focus kernel, three bloom passes and
    // the final pass.
    pipelines: 3 + 5 + 3 + 1,
  },
);
