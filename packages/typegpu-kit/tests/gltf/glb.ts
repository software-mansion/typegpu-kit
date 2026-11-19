/** Builds small binary glTFs for tests: views into one binary chunk, and the JSON beside them. */
export function builder() {
  const parts: Uint8Array[] = [];
  let length = 0;

  const json = {
    asset: { version: '2.0' },
    accessors: [] as object[],
    bufferViews: [] as object[],
    buffers: [] as object[],
  } as Record<string, unknown> & { accessors: object[]; bufferViews: object[] };

  /** Appends bytes as a buffer view, 4-byte aligned, and returns its index. */
  const view = (data: ArrayBufferView, byteStride?: number) => {
    const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    const padding = (4 - (length % 4)) % 4;
    parts.push(new Uint8Array(padding), bytes);
    length += padding;

    json.bufferViews.push({
      buffer: 0,
      byteOffset: length,
      byteLength: bytes.byteLength,
      byteStride,
    });
    length += bytes.byteLength;
    return json.bufferViews.length - 1;
  };

  const TYPES = new Map<Function, number>([
    [Int8Array, 5120],
    [Uint8Array, 5121],
    [Int16Array, 5122],
    [Uint16Array, 5123],
    [Uint32Array, 5125],
    [Float32Array, 5126],
  ]);
  const WIDTHS = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };

  /** Appends a tightly packed accessor and returns its index. */
  const accessor = (
    data: Float32Array | Uint32Array | Uint16Array | Int16Array | Uint8Array | Int8Array,
    type: keyof typeof WIDTHS,
    extra: object = {},
  ) => {
    json.accessors.push({
      bufferView: view(data),
      componentType: TYPES.get(data.constructor),
      count: data.length / WIDTHS[type],
      type,
      ...extra,
    });
    return json.accessors.length - 1;
  };

  const build = () => {
    const bin = new Uint8Array(Math.ceil(length / 4) * 4);
    let at = 0;
    for (const part of parts) {
      bin.set(part, at);
      at += part.byteLength;
    }

    json.buffers = [{ byteLength: bin.byteLength }];
    return glb(json, bin);
  };

  return { json, view, accessor, build };
}

/** A binary glTF of a JSON chunk and an optional binary chunk. */
export function glb(json: object, bin?: Uint8Array) {
  const text = new TextEncoder().encode(JSON.stringify(json));
  const jsonLength = Math.ceil(text.byteLength / 4) * 4;
  const binLength = bin ? 8 + bin.byteLength : 0;

  const out = new Uint8Array(12 + 8 + jsonLength + binLength);
  const view = new DataView(out.buffer);
  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, out.byteLength, true);

  view.setUint32(12, jsonLength, true);
  view.setUint32(16, 0x4e4f534a, true);
  out.fill(0x20, 20, 20 + jsonLength);
  out.set(text, 20);

  if (bin) {
    view.setUint32(20 + jsonLength, bin.byteLength, true);
    view.setUint32(24 + jsonLength, 0x004e4942, true);
    out.set(bin, 28 + jsonLength);
  }
  return out;
}
