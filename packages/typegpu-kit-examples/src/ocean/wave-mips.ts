import {
  d,
  std,
  tgpu,
  type SampledFlag,
  type StorageFlag,
  type TgpuComputePass,
  type TgpuRoot,
  type TgpuTexture,
} from 'typegpu';

const GROUP = 16;
const storage = d.textureStorage2dArray('rgba16float', 'write-only');
const layout = tgpu.bindGroupLayout({
  src: { texture: d.texture2dArray(d.f32) },
  first: { storageTexture: storage },
  second: { storageTexture: storage },
  third: { storageTexture: storage },
});
const held = tgpu.workgroupVar(d.arrayOf(d.vec4f, GROUP * GROUP));

/** Rounds half floats towards zero, below the smallest normal in steps of 2⁻²⁴. */
const towardZero = (value: number) => {
  'use gpu';
  if (std.abs(value) < 2 ** -14) {
    return std.trunc(value * 2 ** 24) / 2 ** 24;
  }
  return std.bitcastU32toF32(std.bitcastF32toU32(value) & 0xffffe000);
};

/** Rounds as `rgba16float` stores do on Apple GPUs, so held and stored averages match. */
const asStored = (value: d.v4f) => {
  'use gpu';
  return d.vec4f(
    towardZero(value.x),
    towardZero(value.y),
    towardZero(value.z),
    towardZero(value.w),
  );
};

const averageHeld = (index: number, stride: number) => {
  'use gpu';
  const sum =
    held.$[index] +
    held.$[index + stride] +
    held.$[index + stride * GROUP] +
    held.$[index + stride * GROUP + stride];
  return asStored(sum * 0.25);
};

/** Each thread averages a texel of the first level, and the workgroup those into two more. */
const fillThree = tgpu.computeFn({
  workgroupSize: [GROUP, GROUP],
  in: { gid: d.builtin.globalInvocationId, lid: d.builtin.localInvocationId },
})(({ gid, lid }) => {
  'use gpu';
  const index = lid.y * GROUP + lid.x;
  let value = d.vec4f();
  if (std.all(std.lt(gid.xy, std.textureDimensions(layout.$.first)))) {
    const at = gid.xy * 2;
    const sum =
      std.textureLoad(layout.$.src, at, gid.z, 0) +
      std.textureLoad(layout.$.src, at + d.vec2u(1, 0), gid.z, 0) +
      std.textureLoad(layout.$.src, at + d.vec2u(0, 1), gid.z, 0) +
      std.textureLoad(layout.$.src, at + d.vec2u(1, 1), gid.z, 0);
    value = asStored(sum * 0.25);
    std.textureStore(layout.$.first, gid.xy, gid.z, value);
  }
  held.$[index] = d.vec4f(value);
  std.workgroupBarrier();
  const half = d.vec2u(gid.x >>> 1, gid.y >>> 1);
  const second = (lid.x & 1) === 0 && (lid.y & 1) === 0;
  if (second && std.all(std.lt(half, std.textureDimensions(layout.$.second)))) {
    const average = averageHeld(index, d.u32(1));
    held.$[index] = d.vec4f(average);
    std.textureStore(layout.$.second, half, gid.z, average);
  }
  std.workgroupBarrier();
  const quarter = d.vec2u(gid.x >>> 2, gid.y >>> 2);
  const third = (lid.x & 3) === 0 && (lid.y & 3) === 0;
  if (third && std.all(std.lt(quarter, std.textureDimensions(layout.$.third)))) {
    std.textureStore(layout.$.third, quarter, gid.z, averageHeld(index, d.u32(2)));
  }
});

type WaveMap = TgpuTexture<{
  size: readonly number[];
  format: 'rgba16float';
  mipLevelCount?: number | undefined;
}> &
  SampledFlag &
  StorageFlag;

const only = (baseMipLevel: number) => ({ baseMipLevel, mipLevelCount: 1 });

/**
 * Box-filtered mips of the wave maps, three levels a dispatch. The maps refill every frame, and the
 * kit's `mipmaps`, a level a dispatch, waits between three times as many.
 */
export function createWaveMips(root: TgpuRoot, maps: readonly WaveMap[]) {
  const pipeline = root.createComputePipeline({ compute: fillThree });
  const dispatches = maps.flatMap((map) => {
    const { size, mipLevelCount = 1 } = map.props;
    return Array.from({ length: Math.floor((mipLevelCount - 1) / 3) }, (_, batch) => {
      const top = batch * 3;
      const group = root.createBindGroup(layout, {
        src: map.createView(d.texture2dArray(d.f32), only(top)),
        first: map.createView(storage, only(top + 1)),
        second: map.createView(storage, only(top + 2)),
        third: map.createView(storage, only(top + 3)),
      });
      const x = Math.ceil(Math.max(size[0] >> (top + 1), 1) / GROUP);
      const y = Math.ceil(Math.max(size[1] >> (top + 1), 1) / GROUP);
      return (pass: TgpuComputePass) =>
        pipeline.with(pass).with(group).dispatchWorkgroups(x, y, size[2]);
    });
  });
  return {
    pipeline,
    record(pass: TgpuComputePass) {
      for (const dispatch of dispatches) dispatch(pass);
    },
  };
}
