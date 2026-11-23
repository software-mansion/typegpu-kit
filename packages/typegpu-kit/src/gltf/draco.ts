/** This type holds the values of an attribute a Draco decoder returns, in its accessor's type. */
export type DracoArray =
  | Int8Array
  | Uint8Array
  | Int16Array
  | Uint16Array
  | Uint32Array
  | Float32Array;

/** This type asks a Draco decoder for one attribute of a compressed primitive. */
export interface DracoAttribute {
  /** Gives the attribute's unique id in the compressed data. */
  readonly id: number;
  /** Gives the glTF component type of the attribute's accessor, such as 5126 for floats. */
  readonly componentType: number;
}

/** This type holds a decoded primitive. */
export interface DracoMesh {
  /** Lists three vertex indices per triangle. */
  readonly indices: Uint32Array;
  /**
   * Holds the requested attributes in request order, each with every vertex's components in the
   * accessor's component type: a `Float32Array` for 5126, a `Uint16Array` for 5123.
   */
  readonly attributes: readonly DracoArray[];
}

/**
 * Decodes the bytes of one primitive compressed with `KHR_draco_mesh_compression`. Make one with
 * `draco` from a Draco decoder module, or write one for a platform without WebAssembly.
 */
export type Draco = (bytes: Uint8Array, attributes: readonly DracoAttribute[]) => DracoMesh;

/** This type describes what `draco` calls on a Draco decoder module, such as `draco3d`'s. */
export interface DracoModule {
  readonly Decoder: new () => {
    DecodeBufferToMesh(buffer: object, mesh: object): { ok(): boolean; error_msg(): string };
    GetAttributeByUniqueId(mesh: object, id: number): { num_components(): number };
    GetTrianglesUInt32Array(mesh: object, byteLength: number, at: number): void;
    GetAttributeDataArrayForAllPoints(
      mesh: object,
      attribute: object,
      type: number,
      byteLength: number,
      at: number,
    ): void;
  };
  readonly DecoderBuffer: new () => { Init(bytes: Int8Array, byteLength: number): void };
  readonly Mesh: new () => { num_points(): number; num_faces(): number };
  readonly DT_INT8: number;
  readonly DT_UINT8: number;
  readonly DT_INT16: number;
  readonly DT_UINT16: number;
  readonly DT_UINT32: number;
  readonly DT_FLOAT32: number;
  readonly HEAPU8: Uint8Array;
  _malloc(byteLength: number): number;
  _free(at: number): void;
  destroy(object: object): void;
}

type ArrayType<T extends DracoArray> = {
  readonly BYTES_PER_ELEMENT: number;
  new (buffer: ArrayBufferLike, byteOffset: number, length: number): { slice(): T };
};

/**
 * Makes a Draco decoder for the `draco` option of `read` and `load` from a decoder module you
 * create, such as `await createDecoderModule({ wasmBinary })` from the `draco3d` package. The kit
 * bundles no decoder, so a program without Draco models pays nothing for one.
 *
 * @param module a Draco decoder module, made once and shared by every read
 * @returns a decoder for the `draco` option
 */
export function draco(module: DracoModule): Draco {
  const types = new Map<number, readonly [type: number, Array: ArrayType<DracoArray>]>([
    [5120, [module.DT_INT8, Int8Array]],
    [5121, [module.DT_UINT8, Uint8Array]],
    [5122, [module.DT_INT16, Int16Array]],
    [5123, [module.DT_UINT16, Uint16Array]],
    [5125, [module.DT_UINT32, Uint32Array]],
    [5126, [module.DT_FLOAT32, Float32Array]],
  ]);

  // Copy out of the module's heap after each call, which may grow it and detach older views
  const copied = <T extends DracoArray>(
    Array: ArrayType<T>,
    length: number,
    fill: (byteLength: number, at: number) => void,
  ) => {
    const byteLength = length * Array.BYTES_PER_ELEMENT;
    // oxlint-disable-next-line no-underscore-dangle -- Emscripten's name for the module's malloc
    const at = module._malloc(byteLength);
    try {
      fill(byteLength, at);
      return new Array(module.HEAPU8.buffer, at, length).slice();
    } finally {
      // oxlint-disable-next-line no-underscore-dangle -- Emscripten's name for the module's free
      module._free(at);
    }
  };

  return (bytes, attributes) => {
    const decoder = new module.Decoder();
    const buffer = new module.DecoderBuffer();
    const mesh = new module.Mesh();
    try {
      buffer.Init(
        new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength),
        bytes.byteLength,
      );
      const status = decoder.DecodeBufferToMesh(buffer, mesh);
      if (!status.ok()) throw new Error(`Draco couldn't decode a primitive: ${status.error_msg()}`);

      const indices = copied(Uint32Array, mesh.num_faces() * 3, (byteLength, at) =>
        decoder.GetTrianglesUInt32Array(mesh, byteLength, at),
      );

      return {
        indices,
        attributes: attributes.map(({ id, componentType }) => {
          const type = types.get(componentType);
          if (!type) throw new Error(`Draco can't decode component type ${componentType}.`);
          const attribute = decoder.GetAttributeByUniqueId(mesh, id);
          // Emscripten wraps a null pointer for an id the data doesn't hold
          if ('ptr' in attribute && attribute.ptr === 0) {
            throw new Error(`The Draco data has no attribute ${id}.`);
          }
          const length = mesh.num_points() * attribute.num_components();
          return copied(type[1], length, (byteLength, at) =>
            decoder.GetAttributeDataArrayForAllPoints(mesh, attribute, type[0], byteLength, at),
          );
        }),
      };
    } finally {
      module.destroy(mesh);
      module.destroy(buffer);
      module.destroy(decoder);
    }
  };
}
