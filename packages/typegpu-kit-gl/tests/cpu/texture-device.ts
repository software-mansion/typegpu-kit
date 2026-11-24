import type { CpuTexture } from '../../src/cpu/types.ts';

const extent = (size: GPUExtent3D) =>
  Symbol.iterator in size
    ? Array.from(size as Iterable<number>)
    : [
        (size as GPUExtent3DDict).width,
        (size as GPUExtent3DDict).height ?? 1,
        (size as GPUExtent3DDict).depthOrArrayLayers ?? 1,
      ];

export function textureDevice(original: GPUDevice) {
  const textures = new WeakMap<object, CpuTexture>();
  const samplers = new WeakMap<object, GPUSamplerDescriptor>();

  return {
    textures,
    samplers,
    createSampler(descriptor: GPUSamplerDescriptor = {}) {
      const sampler = original.createSampler(descriptor);
      samplers.set(sampler, descriptor);
      return sampler;
    },
    createTexture(descriptor: GPUTextureDescriptor) {
      const size = extent(descriptor.size);
      const channels =
        descriptor.format.startsWith('rgba') || descriptor.format.startsWith('bgra')
          ? 4
          : descriptor.format.startsWith('rg')
            ? 2
            : 1;
      const bits = Number(/(8|16|32)/.exec(descriptor.format)?.[1] ?? 32);
      const ctor =
        bits === 8
          ? Uint8Array
          : bits === 16
            ? Uint16Array
            : descriptor.format.endsWith('float')
              ? Float32Array
              : descriptor.format.endsWith('sint')
                ? Int32Array
                : Uint32Array;
      const cpu: CpuTexture = {
        format: descriptor.format,
        dimension: descriptor.dimension === '3d' ? '3d' : '2d',
        levels: Array.from({ length: descriptor.mipLevelCount ?? 1 }, (_, level) => {
          const width = Math.max(1, size[0]! >> level),
            height = Math.max(1, (size[1] ?? 1) >> level),
            depthOrArrayLayers =
              descriptor.dimension === '3d' ? Math.max(1, (size[2] ?? 1) >> level) : (size[2] ?? 1);
          return {
            width,
            height,
            depthOrArrayLayers,
            data: new ctor(width * height * depthOrArrayLayers * channels),
          };
        }),
      };
      const texture = original.createTexture(descriptor);
      const wrapped = new Proxy(texture, {
        get(target, key) {
          if (key === 'createView')
            return (view: GPUTextureViewDescriptor = {}) => {
              const handle = target.createView(view);
              const levels = cpu.levels.slice(
                view.baseMipLevel ?? 0,
                (view.baseMipLevel ?? 0) + (view.mipLevelCount ?? cpu.levels.length),
              );
              const selected = levels.map((level) => {
                const layer = view.baseArrayLayer ?? 0,
                  count = view.arrayLayerCount ?? level.depthOrArrayLayers - layer;
                const data = level.data as Uint8Array;
                return {
                  ...level,
                  depthOrArrayLayers: count,
                  data: data.subarray(
                    layer * level.width * level.height * channels,
                    (layer + count) * level.width * level.height * channels,
                  ),
                };
              });
              textures.set(handle, {
                format: view.format ?? cpu.format,
                dimension: view.dimension ?? cpu.dimension,
                levels: selected,
              });
              return handle;
            };
          const value = Reflect.get(target, key);
          return typeof value === 'function' ? value.bind(target) : value;
        },
      });
      textures.set(wrapped, cpu);
      return wrapped;
    },
    writeTexture(
      destination: GPUTexelCopyTextureInfo,
      data: GPUAllowSharedBufferSource,
      layout: GPUTexelCopyBufferLayout,
      writeSize: GPUExtent3D,
    ) {
      const texture = textures.get(destination.texture)!;
      const level = texture.levels[destination.mipLevel ?? 0]!;
      const size = extent(writeSize);
      const origin = destination.origin
        ? Symbol.iterator in destination.origin
          ? Array.from(destination.origin as Iterable<number>)
          : [
              (destination.origin as GPUOrigin3DDict).x ?? 0,
              (destination.origin as GPUOrigin3DDict).y ?? 0,
              (destination.origin as GPUOrigin3DDict).z ?? 0,
            ]
        : [0, 0, 0];
      const bytesPerPixel =
        level.data.byteLength / (level.width * level.height * level.depthOrArrayLayers);
      const source = ArrayBuffer.isView(data)
        ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
        : new Uint8Array(data);
      const target = new Uint8Array(
        level.data.buffer,
        level.data.byteOffset,
        level.data.byteLength,
      );
      const rowBytes = size[0]! * bytesPerPixel;
      for (let z = 0; z < (size[2] ?? 1); z++)
        for (let y = 0; y < (size[1] ?? 1); y++) {
          const from =
            (layout.offset ?? 0) +
            z * (layout.rowsPerImage ?? size[1] ?? 1) * (layout.bytesPerRow ?? rowBytes) +
            y * (layout.bytesPerRow ?? rowBytes);
          const to =
            (((z + (origin[2] ?? 0)) * level.height + y + (origin[1] ?? 0)) * level.width +
              (origin[0] ?? 0)) *
            bytesPerPixel;
          target.set(source.subarray(from, from + rowBytes), to);
        }
    },
  };
}
