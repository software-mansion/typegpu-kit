import {
  passes,
  pipelines,
  targets,
  type CameraBase,
  type Frame,
  type SizeSource,
} from '@typegpu/kit';
import { d, type TgpuRoot } from 'typegpu';
import { pictureGroups, sample } from './picture.ts';

const Seen = d.struct({ color: d.vec4f });
const CLEARED = { clear: ['picture'] } as const;

/**
 * Draws translucent matter such as smoke at reduced resolution to lower fragment cost. Renders
 * foreground depth first to hide occluded matter, then composites the picture over the scene.
 */
export function createLowRes(
  root: TgpuRoot,
  view: CameraBase,
  options: { size: SizeSource; scale: number },
) {
  const { size, scale } = options;
  const low = targets.create(root, size, {
    picture: { format: 'rgba16float', sampled: true, scale },
    depth: { format: 'depth24plus', scale },
  });
  const occluders = passes
    .create(root, low, { depth: passes.depth }, { label: 'smoke depth' })
    .pipe(view.inject());
  const matter = passes
    .create(root, low, { picture: passes.color, depth: passes.depth }, { label: 'smoke' })
    .pipe(view.inject());

  const groupOf = pictureGroups(root);
  const over: passes.Drawing<typeof passes.ScreenVertex, d.Infer<typeof Seen>> = {
    ...passes.fullScreen,
    label: 'smoke over',
    pixelOf: (corner) => {
      'use gpu';
      return Seen({ color: sample(corner.uv) });
    },
    color: passes.shaded,
    blend: pipelines.blend.over,
    depthStencil: { depthWriteEnabled: false, depthCompare: 'always' },
    draw(pass) {
      pass.setBindGroup(groupOf(low.textures.picture));
      pass.draw(3);
    },
  };

  return {
    /** Add foreground occluders. Draws only their depth. */
    occluders,
    /** Add matter with premultiplied blending. */
    matter,
    /** Add this composite drawing at the matter's position in scene draw order. */
    over,
    /** Begins the foreground depth pass. End it before calling `begin`. */
    beginDepth(frame: Frame) {
      low.fit();
      return occluders.begin(frame);
    },
    /** Begins the matter pass using the foreground depth. */
    begin: (frame: Frame) => matter.begin(frame, CLEARED),
    initAsync: () => pipelines.compile([occluders, matter]),
    destroy() {
      low.destroy();
    },
  };
}
