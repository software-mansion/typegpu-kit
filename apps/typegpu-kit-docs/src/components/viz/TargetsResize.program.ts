import { memo, targets } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';
import { d, tgpu } from 'typegpu';

/** The canvas surface's default cap on the pixel ratio. */
export const MAX_PIXEL_RATIO = 2;

const layout = tgpu.bindGroupLayout({ hdr: { texture: d.texture2d() } });

export interface Settings {
  /** The canvas, in CSS pixels. */
  width: number;
  height: number;
  ratio: number;
  resolution: number;
}

export function createProgram({ width = 640, height = 360, ratio = 2, resolution = 1 } = {}) {
  const gpu = recordingRoot().root;
  const settings: Settings = { width, height, ratio, resolution };
  const pixelRatio = () => Math.min(settings.ratio, MAX_PIXEL_RATIO);

  // Stands in for a canvas surface, whose drawing buffer is its CSS size times the capped ratio
  const surface = {
    get width() {
      return settings.width * pixelRatio();
    },

    get height() {
      return settings.height * pixelRatio();
    },

    get pixelRatio() {
      return pixelRatio();
    },
  };

  const screen = targets.create(
    gpu,
    surface,
    {
      scene: { format: 'rgba16float', samples: 4 },
      depth: { format: 'depth24plus', samples: 4 },
      hdr: { format: 'rgba16float', sampled: true },
      bloom: { format: 'rgba16float', scale: 1 / 2, mips: 5, sampled: true },
      blur: { format: 'rgba16float', scale: 1 / 4, sampled: true },
    },
    { resolution: () => settings.resolution },
  );

  let groups = 0;
  const hdrGroup = memo((hdr: typeof screen.textures.hdr) => {
    groups += 1;
    return gpu.createBindGroup(layout, { hdr });
  });

  const once = screen.textures.hdr;
  gpu.createBindGroup(layout, { hdr: once });

  let frames = 0;
  let rebuilds = 0;

  return {
    /** What the figure changes; the next frame fits the set to them. */
    settings,
    screen,
    /** The texture the group made once holds, the set's first `hdr`. */
    once,

    /** Bind groups `memo` made: one per rebuild. */
    get groups() {
      return groups;
    },

    /** Frames, each of which called `memo` once. */
    get frames() {
      return frames;
    },

    get rebuilds() {
      return rebuilds;
    },

    /** Whether the group made once holds a destroyed texture. */
    get stale() {
      return screen.textures.hdr !== once;
    },

    /** A frame: fits the set, then asks `memo` for the bind group. Returns whether it rebuilt. */
    frame() {
      const rebuilt = screen.fit();
      if (rebuilt) rebuilds += 1;
      hdrGroup(screen.textures.hdr);
      frames += 1;
      return rebuilt;
    },

    destroy() {
      screen.destroy();
    },
  };
}
