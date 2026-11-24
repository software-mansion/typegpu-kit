/**
 * WebGPU's constant namespaces. A browser without WebGPU lacks them, and TypeGPU and the kit read
 * them wherever they create a resource.
 */
const constants = {
  GPUBufferUsage: {
    MAP_READ: 0x0001,
    MAP_WRITE: 0x0002,
    COPY_SRC: 0x0004,
    COPY_DST: 0x0008,
    INDEX: 0x0010,
    VERTEX: 0x0020,
    UNIFORM: 0x0040,
    STORAGE: 0x0080,
    INDIRECT: 0x0100,
    QUERY_RESOLVE: 0x0200,
  },
  GPUTextureUsage: {
    COPY_SRC: 0x01,
    COPY_DST: 0x02,
    TEXTURE_BINDING: 0x04,
    STORAGE_BINDING: 0x08,
    RENDER_ATTACHMENT: 0x10,
  },
  GPUShaderStage: { VERTEX: 0x1, FRAGMENT: 0x2, COMPUTE: 0x4 },
  GPUMapMode: { READ: 0x1, WRITE: 0x2 },
  GPUColorWrite: { RED: 0x1, GREEN: 0x2, BLUE: 0x4, ALPHA: 0x8, ALL: 0xf },
} as const;

/**
 * Defines each of WebGPU's constant namespaces the environment lacks, and leaves the ones it has.
 * They are fixed by the WebGPU specification, so a definition here equals the browser's own.
 */
export function defineConstants(): void {
  for (const [name, value] of Object.entries(constants)) {
    if (!(name in globalThis)) Object.assign(globalThis, { [name]: Object.freeze({ ...value }) });
  }
}
