import { cameras, passes, font, text } from '@typegpu/kit';
import { fontFromDisk } from 'typegpu-kit-examples/testing';
import type { Fixtures } from './harness.ts';

const size = { width: 64, height: 36 };

const inter = async () => {
  const { json, atlas } = await fontFromDisk('inter');
  return font.read(json, atlas);
};

export const fixtures: Fixtures = {
  'kit/font': {
    setup: async (root) => {
      const baked = await inter();
      return {
        create: () => {
          const sans = font.create(root, baked);
          return {
            // What a text set's pipeline binds of it.
            frame: () => void root.unwrap(sans.glyphs),
            destroy: () => sans.destroy(),
          };
        },
      };
    },
  },

  'kit/text': {
    setup: async (root, surface) => {
      const sans = font.create(root, await inter());
      root.unwrap(sans.glyphs);
      const view = cameras.orbit(root, { size, target: [0, 0, 0], distance: 5 });
      return {
        create: async (others) => {
          // The description holds the viewport the screen text reads.
          const canvas = others(() =>
            passes.create(root, surface, { clear: [0, 0, 0, 1] }).pipe(view.inject()),
          );
          const labels = text.create(root, sans, {
            camera: view,
            capacity: 2,
            length: 8,
            place: text.screen,
          });
          const drawn = canvas.add(labels.drawing);
          await others(() => drawn.initAsync());
          return {
            frame: (frame) => {
              labels.write(0, `t ${frame.index % 10}`, { em: 0.2 });
              const pass = canvas.begin(frame);
              drawn.draw(pass);
              pass.end();
            },
            destroy: () => {
              labels.destroy();
              canvas.destroy();
            },
          };
        },
      };
    },
  },
};
