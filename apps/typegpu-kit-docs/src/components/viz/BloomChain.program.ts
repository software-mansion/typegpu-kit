import { targets, bloom } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';
import { boundBy, onBindGroups } from './program.ts';

interface Level {
  readonly texture: GPUTexture;
  readonly level: number;
}

interface Pass {
  readonly writes: Level | undefined;
  readonly clears: boolean;
  readonly reads: Level[];
}

/**
 * The recording device keeps the descriptors it is given, so wrapping two of its methods shows
 * which level each pass reads and writes.
 */
function watch(device: GPUDevice) {
  const levels = new WeakMap<object, Level>();
  const sizes = new Map<GPUTexture, { width: number; height: number; mips: number }>();
  const passes: Pass[] = [];

  const createTexture = device.createTexture.bind(device);
  device.createTexture = (descriptor) => {
    const texture = createTexture(descriptor);
    const [width = 1, height = 1] = descriptor.size as number[];
    sizes.set(texture, { width, height, mips: descriptor.mipLevelCount ?? 1 });

    const createView = texture.createView.bind(texture);
    texture.createView = (view) => {
      const created = createView(view);
      levels.set(created, { texture, level: view?.baseMipLevel ?? 0 });
      return created;
    };

    return texture;
  };

  const createCommandEncoder = device.createCommandEncoder.bind(device);
  device.createCommandEncoder = (descriptor) => {
    const encoder = createCommandEncoder(descriptor);
    const beginRenderPass = encoder.beginRenderPass.bind(encoder);

    encoder.beginRenderPass = (passDescriptor) => {
      const pass = beginRenderPass(passDescriptor);
      const [attachment] = [...passDescriptor.colorAttachments];
      const record: Pass = {
        writes: attachment ? levels.get(attachment.view as object) : undefined,
        clears: attachment?.loadOp === 'clear',
        reads: [],
      };
      passes.push(record);

      return onBindGroups(pass, (group) => {
        for (const { object } of boundBy(group)) {
          const read = levels.get(object);
          if (read) record.reads.push(read);
        }
      });
    };

    return encoder;
  };

  return { passes, sizes };
}

export interface Size {
  width: number;
  height: number;
}

/** A pass that reads one row and writes another; row 0 is the source, row 1 + n is level n. */
export interface Arrow {
  /** Its place among the passes, from 1. */
  readonly pass: number;
  readonly from: number;
  readonly to: number;
  /** It clears what it writes, on the way down, rather than adding to it, on the way up. */
  readonly clears: boolean;
}

export interface Chain {
  /** The size of each row: the source, then the chain's levels. */
  readonly rows: readonly Size[];
  readonly arrows: readonly Arrow[];
  /** Every pass record() recorded. */
  readonly passes: number;
}

export interface Options {
  /** The source's size, 1920 by 1080 by default. */
  readonly width?: number;
  readonly height?: number;
  /** As bloom.create takes it. Default 6. */
  readonly levels?: number;
}

export function createProgram({ width = 1920, height = 1080, levels = 6 }: Options = {}) {
  const { root } = recordingRoot();
  const gpu = watch(root.device);
  const size: Size = { width, height };
  const screen = targets.create(root, size, {
    hdr: { format: 'rgba16float', sampled: true },
  });
  const glow = bloom.create(root, { source: () => screen.textures.hdr, levels });

  return {
    /** The source's size; the next frame fits the chain to it. */
    size,

    /** Records a frame, and what its passes read and wrote. */
    frame(): Chain {
      gpu.passes.length = 0;
      screen.fit();

      const encoder = root.createCommandEncoder();
      glow.record(encoder);
      encoder.submit();

      const source = root.unwrap(screen.textures.hdr);
      const passes = [...gpu.passes];
      const chain = passes[0]?.writes?.texture;
      const sourceSize = gpu.sizes.get(source)!;
      const glowSize = chain && gpu.sizes.get(chain);
      const count = glowSize?.mips ?? 0;

      const rowOf = (read: Level) => (read.texture === source ? 0 : read.level + 1);
      const rows = Array.from({ length: count + 1 }, (_, row) =>
        row === 0 || !glowSize
          ? { width: sourceSize.width, height: sourceSize.height }
          : {
              width: Math.max(1, glowSize.width >> (row - 1)),
              height: Math.max(1, glowSize.height >> (row - 1)),
            },
      );

      const arrows = passes.flatMap((pass, index) => {
        const read = pass.reads[0];
        if (!pass.writes || !read) return [];
        return [
          { pass: index + 1, from: rowOf(read), to: rowOf(pass.writes), clears: pass.clears },
        ];
      });

      return { rows, arrows, passes: passes.length };
    },

    destroy() {
      glow.destroy();
      screen.destroy();
    },
  };
}
