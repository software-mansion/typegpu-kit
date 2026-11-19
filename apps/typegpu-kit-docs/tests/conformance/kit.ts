import {
  cameras,
  liveLists,
  counters,
  baker,
  history,
  machines,
  mipmaps,
  picker,
  profiler,
  readback,
  sorter,
  table,
  video,
  targets,
  images,
} from '@typegpu/kit';
import * as grid from '@typegpu/kit/grid';
import { d } from 'typegpu';
import type { Fixtures } from './harness.ts';

const size = { width: 64, height: 36 };
const Block = table.define('Block', { host: { pos: d.vec3f } });
const Item = d.struct({ position: d.vec3f });
const Herd = machines.define('Herd', {
  events: ['startle'],
  states: {
    graze: { next: [{ to: 'flee', on: 'startle' }] },
    flee: { next: [{ to: 'graze', after: 1 }] },
  },
});

const origin = () => {
  'use gpu';
  return d.vec3f();
};

export const kit: Fixtures = {
  'kit/bindings': {
    skip: 'its one bind group comes from `group()` on a pass; bindings.test counts it',
  },

  'kit/cameras': {
    setup: (root) => ({
      create: () => {
        const view = cameras.orbit(root, { size, target: [0, 0, 0], distance: 5 });
        return { frame: (frame) => view.update(frame.dt) };
      },
    }),
  },

  'kit/liveLists': {
    setup: (root) => {
      const rows = table.create(root, Block, { capacity: 64 });
      rows.add(); // a live row, so compact has slots to test
      return {
        create: () => {
          const live = liveLists.create(root, { ...rows.live(), draws: [{ vertexCount: 6 }] });
          return {
            frame: ({ encoder }) => {
              rows.flush();
              const pass = encoder.beginComputePass();
              live.compact(pass);
              pass.end();
            },
            destroy: () => live.destroy(),
          };
        },
      };
    },
  },

  'kit/counters': {
    setup: (root) => ({
      create: () => {
        const counts = counters.create(root, ['hit', 'miss']);
        return {
          frame: (frame) => {
            counts.capture(frame);
            counts.take();
          },
          destroy: () => counts.destroy(),
        };
      },
    }),
  },

  'kit/frames': { skip: "the checks' own loop; frames.test counts its submits" },

  'kit/grid': {
    setup: (root) => ({
      create: () => {
        const cells = grid.create(root, {
          capacity: 64,
          low: d.vec3f(-1),
          size: d.vec3f(2),
          sight: 1,
          item: Item,
          alive: () => {
            'use gpu';
            return true;
          },
          position: origin,
          itemOf: () => {
            'use gpu';
            return Item();
          },
        });
        return {
          frame: ({ encoder }) => {
            const pass = encoder.beginComputePass();
            cells.sort(pass);
            pass.end();
          },
          destroy: () => cells.destroy(),
        };
      },
    }),
  },

  'kit/sorter': {
    setup: (root) => {
      const keys = root.createBuffer(d.arrayOf(d.u32, 64)).$usage('storage');
      const values = root.createBuffer(d.arrayOf(d.u32, 64)).$usage('storage');
      // TypeGPU makes a buffer on first use, which would count as the sorter's
      root.unwrap(keys);
      root.unwrap(values);
      return {
        create: () => {
          const order = sorter.create(root, keys, { values, keyBits: 8 });
          return {
            frame: ({ encoder }) => {
              const pass = encoder.beginComputePass();
              order.sort(pass);
              pass.end();
            },
            destroy: () => order.destroy(),
          };
        },
      };
    },
  },

  'kit/baker': {
    setup: (root) => {
      const texture = root
        .createTexture({ size: [16, 16], format: 'rgba16float' })
        .$usage('sampled', 'storage');
      root.unwrap(texture);
      return {
        create: () => {
          const baking = baker.create(root, texture, {
            texel: (uv) => {
              'use gpu';
              return d.vec4f(uv, 0, 1);
            },
          });
          return { frame: ({ encoder }) => baking.record(encoder), destroy: () => {} };
        },
      };
    },
  },

  'kit/history': {
    setup: (root) => ({
      create: () => {
        const state = history.create(root, d.vec4f, { initial: d.vec4f(1) });
        return {
          frame: (frame) => {
            state.write(d.vec4f(2));
            state.record(frame);
          },
          destroy: () => state.destroy(),
        };
      },
    }),
  },

  'kit/machines': {
    setup: (root) => ({
      create: () => {
        const herd = machines.create(root, Herd, {
          capacity: 64,
          step: 1 / 60,
          lists: [Herd.flee],
        });
        herd.start(3, Herd.graze);
        return {
          frame: ({ encoder }) => {
            herd.send.startle(3);
            const pass = encoder.beginComputePass();
            herd.step(pass);
            pass.end();
          },
          destroy: () => herd.destroy(),
        };
      },
    }),
  },

  'kit/passes': {
    skip: 'owns only the pipelines `add` builds; passes.test checks them with each renderer',
  },

  'kit/mipmaps': {
    setup: (root) => {
      const texture = root
        .createTexture({ size: [16, 16], format: 'rgba16float', mipLevelCount: 5 })
        .$usage('sampled', 'storage');
      // TypeGPU makes a texture on first use, which would count as the mipmaps'
      root.unwrap(texture);
      return {
        create: () => {
          const smaller = mipmaps.create(root, texture, { combine: mipmaps.mean });
          return { frame: ({ encoder }) => smaller.record(encoder), destroy: () => {} };
        },
      };
    },
  },

  'kit/picker': {
    setup: (root) => {
      const view = cameras.orbit(root, { size, target: [0, 0, 0], distance: 5 });
      return {
        create: () => {
          const hits = picker.create(root, view, { capacity: 64, size, position: origin });
          return {
            frame: (frame) => {
              const pass = frame.encoder.beginComputePass();
              hits.pick(pass, 10, 10);
              pass.end();
              hits.result.capture(frame);
            },
            destroy: () => hits.destroy(),
          };
        },
      };
    },
  },

  'kit/profiler': {
    features: ['timestamp-query'],
    setup: (root) => ({
      create: () => {
        const timer = profiler.create(root);
        const scene = timer.timer('scene');
        return {
          frame: (frame) => {
            const pass = frame.encoder.beginComputePass({ timestampWrites: scene.writes() });
            pass.end();
            timer.capture(frame);
          },
          destroy: () => timer.destroy(),
        };
      },
    }),
  },

  'kit/readback': {
    setup: (root) => {
      const source = root.createBuffer(d.u32).$usage('storage');
      return {
        create: () => {
          const probe = readback.create(root, source);
          return { frame: (frame) => probe.capture(frame), destroy: () => probe.destroy() };
        },
      };
    },
  },

  'kit/images': {
    setup: (root) => ({
      create: () => {
        // Mips would add TypeGPU's blit pipeline, which it keeps for the device
        const image = { width: 4, height: 4, pixels: new Uint8Array(64) };
        const texture = images.create(root, image, { mips: false });
        images.sampler(root);
        return { frame: () => {}, destroy: () => texture.destroy() };
      },
    }),
  },

  'kit/video': {
    setup: (root) => ({
      create: () => {
        const feed = video.create(root, { size: [16, 9] });
        const frame = { texture: {} as GPUExternalTexture, width: 32, height: 24 };
        // A camera with a new frame every frame
        let serial = 0;
        feed.attach({ latest: () => ({ ...frame, serial: ++serial }) });
        return { frame: (each) => feed.update(each), destroy: () => feed.destroy() };
      },
    }),
  },

  'kit/table': {
    setup: (root) => ({
      create: () => {
        const rows = table.create(root, Block, { capacity: 64 });
        const id = rows.add({ pos: d.vec3f(1) });
        return {
          frame: () => {
            rows.write(id, { pos: d.vec3f(2) });
            rows.flush();
          },
          destroy: () => rows.destroy(),
        };
      },
    }),
  },

  'kit/targets': {
    setup: (root, surface) => ({
      create: () => {
        const screen = targets.create(root, surface, {
          hdr: { format: 'rgba16float' },
          depth: { format: 'depth24plus' },
        });
        return { frame: () => void screen.fit(), destroy: () => screen.destroy() };
      },
    }),
  },
};
