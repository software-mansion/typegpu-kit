import {
  d,
  std,
  tgpu,
  type StorageFlag,
  type TgpuBindGroup,
  type TgpuCommandEncoder,
  type TgpuRenderPassDescriptor,
  type TgpuRoot,
  type TgpuTexture,
} from 'typegpu';
import { adopt, type AnyComputePass } from './passes.ts';

type Target = TgpuCommandEncoder | AnyComputePass;

const isEncoder = (target: Target): target is TgpuCommandEncoder =>
  'resourceType' in target && target.resourceType === 'command-encoder';

/** This type writes a texture's texels each time it records. */
export interface Filled {
  record(target: Target): void;
  initAsync(): Promise<void>;
}

/** One mip level `fill` writes, with the bind group its texels read. */
interface Step {
  readonly level: number;
  readonly group?: TgpuBindGroup;
}

const WORKGROUP = 8;

const covering = tgpu.vertexFn({
  in: { vertexIndex: d.builtin.vertexIndex, instanceIndex: d.builtin.instanceIndex },
  out: { position: d.builtin.position, layer: d.interpolate('flat', d.u32) },
})(({ vertexIndex, instanceIndex }) => {
  'use gpu';
  const corner = d.vec2f(d.f32((vertexIndex << 1) & 2), d.f32(vertexIndex & 2));
  return {
    position: d.vec4f(corner * d.vec2f(2, -2) + d.vec2f(-1, 1), 0, 1),
    layer: instanceIndex,
  };
});

export const one = (baseMipLevel: number) => ({ baseMipLevel, mipLevelCount: 1 });

/**
 * Throws, naming the property, unless `texture` holds single-sampled 2D layers of float texels, as
 * a `texture_2d_array<f32>` reads and `vec4f` writes. `what` names the texture in the message.
 *
 * @returns the texture's array layers
 */
export function floatLayers(texture: TgpuTexture, what: string): number {
  const { format, dimension = '2d', sampleCount = 1 } = texture.props;
  if (dimension !== '2d') {
    throw new Error(`${what} requires dimension '2d', not '${dimension}'.`);
  }
  if (sampleCount !== 1) {
    throw new Error(`${what} requires one sample, not ${sampleCount}.`);
  }
  if (/[us]int|^stencil8$/.test(format)) {
    throw new Error(`${what} requires a format with float texels, not '${format}'.`);
  }
  const [, , layers = 1] = texture.props.size as readonly number[];
  return layers;
}

/**
 * Builds the pipeline that writes `texel(at, layer)` for `texture`, in a compute pass for a
 * texture with `storage` usage and in render passes otherwise. The function it returns checks
 * `filled`, that texture or another of its format and usage, then fills each mip level that
 * `stepsFor` lists for its array layers, every layer of each.
 */
export function fill(
  root: TgpuRoot,
  texture: TgpuTexture,
  texel: (at: d.v2u, layer: number) => d.v4f,
): (filled: TgpuTexture, stepsFor: (layers: number) => readonly Step[]) => Filled {
  const { format } = texture.props;
  const storage = texture.usableAsStorage;
  const writable = (other: TgpuTexture) => {
    const layers = floatLayers(other, 'Filling a texture');
    if (other.props.format.startsWith('depth')) {
      throw new Error(`Filling a texture requires a colour format, not '${other.props.format}'.`);
    }
    if (!other.usableAsStorage && !other.usableAsRender) {
      throw new Error("Filling a texture requires 'storage' or 'render' usage.");
    }
    if (other.props.format !== format || other.usableAsStorage !== storage) {
      throw new Error(
        `The destination texture must match the pipeline's format '${format}' and must be created with${storage ? '' : 'out'} 'storage' usage.`,
      );
    }
    const [width, height = 1] = other.props.size as readonly number[];
    return { width, height, layers };
  };
  const size = writable(texture);
  const sized = (filled: TgpuTexture) => (filled === texture ? size : writable(filled));

  if (storage) {
    const write = tgpu.bindGroupLayout({
      below: { storageTexture: d.textureStorage2dArray(format, 'write-only') },
    });
    const kernel = tgpu.computeFn({
      workgroupSize: [WORKGROUP, WORKGROUP],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      const below = std.textureDimensions(write.$.below);
      if (gid.x >= below.x || gid.y >= below.y) return;
      std.textureStore(write.$.below, gid.xy, gid.z, texel(gid.xy, gid.z));
    });
    const pipeline = root.createComputePipeline({ compute: kernel }).$name('fill');

    return (filled, stepsFor) => {
      const { width, height, layers } = sized(filled);
      const dispatches = stepsFor(layers).map(({ level, group }) => {
        // `sized` checked the storage usage and the layout's format
        const below = (filled as TgpuTexture & StorageFlag).createView(
          write.entries.below.storageTexture,
          one(level),
        );
        const writing = root.createBindGroup(write, { below });
        // Create the views and groups now, so the first record allocates nothing
        root.unwrap(writing);
        if (group) root.unwrap(group);
        const bound = pipeline.with(writing);
        return {
          pipeline: group ? bound.with(group) : bound,
          x: Math.ceil(Math.max(1, width >> level) / WORKGROUP),
          y: Math.ceil(Math.max(1, height >> level) / WORKGROUP),
        };
      });

      return {
        record(target) {
          if (dispatches.length === 0) return;
          const opened = isEncoder(target) ? target.beginComputePass() : undefined;
          const into = opened ?? adopt(target as AnyComputePass);
          for (const dispatch of dispatches) {
            into.setPipeline(dispatch.pipeline);
            into.dispatchWorkgroups(dispatch.x, dispatch.y, layers);
          }
          opened?.end();
        },
        initAsync: () => pipeline.initAsync(),
      };
    };
  }

  const fragment = tgpu.fragmentFn({
    in: { position: d.builtin.position, layer: d.interpolate('flat', d.u32) },
    out: d.vec4f,
  })(({ position, layer }) => {
    'use gpu';
    return texel(d.vec2u(position.xy), layer);
  });
  const pipeline = root
    .createRenderPipeline({ vertex: covering, fragment, targets: { format } })
    .$name('fill');

  return (filled, stepsFor) => {
    const { layers } = sized(filled);
    const passes = stepsFor(layers).flatMap(({ level, group }) =>
      Array.from({ length: layers }, (_, layer) => {
        const view = filled.createView('render', {
          ...one(level),
          baseArrayLayer: layer,
          arrayLayerCount: 1,
        });
        // Create the views and groups now, so the first record allocates nothing
        root.unwrap(view);
        if (group) root.unwrap(group);
        return {
          group,
          layer,
          descriptor: {
            colorAttachments: [{ view, loadOp: 'clear', storeOp: 'store' }],
          } satisfies TgpuRenderPassDescriptor,
        };
      }),
    );

    return {
      record(target) {
        if (!isEncoder(target)) {
          throw new Error(
            'Pass a command encoder to `record` when the texture lacks `storage` usage. The fill needs to open render passes.',
          );
        }
        for (const { group, layer, descriptor } of passes) {
          const pass = target.beginRenderPass(descriptor);
          pass.setPipeline(pipeline);
          if (group) pass.setBindGroup(group);
          pass.draw(3, 1, 0, layer);
          pass.end();
        }
      },
      initAsync: () => pipeline.initAsync(),
    };
  };
}
