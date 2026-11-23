import { cameras, targets, bloom, dof, exposure, fxaa, finish, taa } from '@typegpu/kit';
import type { Fixtures } from './harness.ts';

const size = { width: 64, height: 36 };

/** The picture a scene drew, sampled, as the finish passes read it. */
const screenOf = (root: Parameters<typeof targets.create>[0]) =>
  targets.create(root, size, {
    hdr: { format: 'rgba16float', sampled: true },
    ldr: { format: 'rgba8unorm', sampled: true },
    motion: { format: 'rg16float', sampled: true },
    depth: { format: 'depth24plus', sampled: true },
  });

export const fixtures: Fixtures = {
  'kit/bloom': {
    setup: (root) => {
      const screen = screenOf(root);
      return {
        create: () => {
          const glow = bloom.create(root, { source: () => screen.textures.hdr, levels: 3 });
          return {
            frame: ({ encoder }) => {
              screen.fit();
              glow.record(encoder);
            },
            destroy: () => glow.destroy(),
          };
        },
      };
    },
  },

  'kit/dof': {
    setup: (root) => {
      const screen = screenOf(root);
      const view = cameras.orbit(root, { size, target: [0, 0, 0], distance: 5 });
      return {
        create: () => {
          const blur = dof.create(root, view, {
            source: () => screen.textures.hdr,
            depth: () => screen.textures.depth,
          });
          return {
            frame: ({ encoder }) => {
              screen.fit();
              blur.record(encoder);
            },
            destroy: () => blur.destroy(),
          };
        },
      };
    },
  },

  'kit/exposure': {
    setup: (root) => {
      const screen = screenOf(root);
      return {
        create: () => {
          const eye = exposure.create(root, { source: () => screen.textures.hdr });
          return {
            frame: ({ encoder, dt }) => {
              screen.fit();
              eye.record(encoder, { dt });
            },
            destroy: () => eye.destroy(),
          };
        },
      };
    },
  },

  'kit/fxaa': {
    setup: (root, surface) => {
      const screen = screenOf(root);
      return {
        create: () => {
          const smooth = fxaa.create(root, {
            format: surface.format,
            source: () => screen.textures.ldr,
          });
          return {
            frame: (frame) => {
              screen.fit();
              const pass = frame.encoder.beginRenderPass({
                colorAttachments: [
                  { view: surface.view(frame), loadOp: 'clear', storeOp: 'store' },
                ],
              });
              smooth.draw(pass);
              pass.end();
            },
            destroy: () => {},
          };
        },
      };
    },
  },

  'kit/finish': {
    setup: (root, surface) => {
      const screen = screenOf(root);
      const glow = bloom.create(root, { source: () => screen.textures.hdr, levels: 3 });
      // The bloom's groups and pipelines are made at its first `record`, which is setup's.
      const encoder = root.createCommandEncoder();
      glow.record(encoder);
      encoder.submit();
      return {
        create: () => {
          const final = finish.create(root, {
            format: surface.format,
            source: () => screen.textures.hdr,
            glow,
          });
          return {
            frame: (frame) => {
              screen.fit();
              glow.record(frame);
              const pass = frame.encoder.beginRenderPass({
                colorAttachments: [
                  { view: surface.view(frame), loadOp: 'clear', storeOp: 'store' },
                ],
              });
              final.draw(pass);
              pass.end();
            },
            destroy: () => {},
          };
        },
      };
    },
  },

  'kit/taa': {
    setup: (root) => {
      const screen = screenOf(root);
      const view = cameras.orbit(root, { size, target: [0, 0, 0], distance: 5 });
      return {
        create: () => {
          const aa = taa.create(root, {
            source: () => screen.textures.hdr,
            motion: () => screen.textures.motion,
            depth: () => screen.textures.depth,
            camera: view,
          });
          return {
            frame: ({ encoder }) => {
              screen.fit();
              aa.record(encoder);
            },
            destroy: () => aa.destroy(),
          };
        },
      };
    },
  },
};
