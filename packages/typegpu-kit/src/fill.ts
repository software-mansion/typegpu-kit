import {
  d,
  std,
  tgpu,
  type SampledFlag,
  type StorageFlag,
  type TgpuBindGroup,
  type TgpuCommandEncoder,
  type TgpuComputePass,
  type TgpuRenderPassDescriptor,
  type TgpuRoot,
  type TgpuTexture,
} from 'typegpu';

/** This type writes a texture's texels each time it records. */
export interface Filled {
  record(target: TgpuCommandEncoder | TgpuComputePass): void;
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
 * Builds the pipeline that writes `texel(at, layer)` for `texture`, in a compute pass for a
 * texture with `storage` usage and in render passes otherwise. The function it returns fills each
 * step's mip level, every array layer, of that texture or of `filled`, another of its format and
 * usage.
 */
export function fill(
  root: TgpuRoot,
  texture: TgpuTexture,
  texel: (at: d.v2u, layer: number) => d.v4f,
): (steps: readonly Step[], filled?: TgpuTexture) => Filled {
  const { format } = texture.props;
  const storage = texture.usableAsStorage;
  if (!storage && !texture.usableAsRender) {
    throw new Error('Filling a texture requires `storage` or `render` usage.');
  }
  const sized = (other: TgpuTexture) => {
    if (other.props.format !== format || other.usableAsStorage !== storage) {
      throw new Error(
        'Another texture fills through this pipeline only with the same format and usage.',
      );
    }
    const [width, height = 1, layers = 1] = other.props.size as number[];
    return { width, height, layers };
  };

  if (storage) {
    const write = tgpu.bindGroupLayout({
      below: { storageTexture: d.textureStorage2dArray(format as 'rgba8unorm', 'write-only') },
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

    return (steps, filled = texture) => {
      const { width, height, layers } = sized(filled);
      const writable = filled as unknown as TgpuTexture<{
        size: number[];
        format: 'rgba8unorm';
      }> &
        StorageFlag;
      const dispatches = steps.map(({ level, group }) => {
        const below = writable.createView(write.entries.below.storageTexture, one(level));
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
          const pass =
            target.resourceType === 'command-encoder' ? target.beginComputePass() : target;
          for (const dispatch of dispatches) {
            pass.setPipeline(dispatch.pipeline);
            pass.dispatchWorkgroups(dispatch.x, dispatch.y, layers);
          }
          if (pass !== target) pass.end();
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

  return (steps, filled = texture) => {
    const { layers } = sized(filled);
    const drawn = filled as TgpuTexture & SampledFlag;
    const passes = steps.flatMap(({ level, group }) =>
      Array.from({ length: layers }, (_, layer) => {
        const view = drawn.createView('render', {
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
        if (target.resourceType !== 'command-encoder') {
          throw new Error(
            'A texture without `storage` usage fills in render passes on an encoder.',
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
