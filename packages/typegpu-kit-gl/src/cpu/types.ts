/**
 * The contract between the CPU compute backend and the device. The device depends only on these
 * types and on `compile` and `isCpuModule` from `./index.ts`.
 */

/** A bound range of a buffer's bytes. Kernels read and write it in place. */
export interface CpuBuffer {
  /** The whole buffer's bytes. Its length is a multiple of 4. */
  readonly bytes: ArrayBuffer;
  /** The first byte of the bound range, a multiple of 4. */
  readonly offset: number;
  /** The bound range's size in bytes, which `arrayLength` derives from. */
  readonly size: number;
}

/** One mip level of a texture, rows top to bottom, layers or slices in order, no row padding. */
export interface CpuTextureLevel {
  readonly width: number;
  readonly height: number;
  readonly depthOrArrayLayers: number;
  /**
   * The texels in the format's own layout: `Uint8Array` for 8-bit formats, `Uint16Array` for 16-bit
   * ones, half floats included, `Float32Array`, `Uint32Array` or `Int32Array` for 32-bit ones.
   */
  readonly data: ArrayBufferView;
}

export interface CpuTexture {
  readonly format: GPUTextureFormat;
  readonly dimension: GPUTextureViewDimension;
  /** The view's levels, from its base mip level. */
  readonly levels: readonly CpuTextureLevel[];
}

/** How a kernel uses a binding, so the device knows what to read back before and upload after. */
export interface CpuBindingUse {
  readonly group: number;
  readonly binding: number;
  readonly kind: 'buffer' | 'texture' | 'storageTexture' | 'sampler';
  readonly reads: boolean;
  readonly writes: boolean;
}

/** The resources a dispatch runs over, looked up by `@group` and `@binding`. */
export interface CpuResources {
  buffer(group: number, binding: number): CpuBuffer;
  texture(group: number, binding: number): CpuTexture;
  sampler(group: number, binding: number): GPUSamplerDescriptor;
}

export interface CpuModule {
  readonly kind: 'compute';
  readonly workgroupSize: readonly [number, number, number];
  readonly bindings: readonly CpuBindingUse[];
  /** Runs every invocation of `x * y * z` workgroups, in order, over `resources`. */
  dispatch(x: number, y: number, z: number, resources: CpuResources): void;
  /**
   * The words of a written buffer binding the last dispatch stored to, from the lowest to one past
   * the highest, empty when it stored none; undefined when the module cannot tell which. A module
   * without it wrote every written binding whole.
   */
  writtenWords?(group: number, binding: number): readonly [number, number] | undefined;
}
