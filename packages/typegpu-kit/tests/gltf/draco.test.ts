import { readFileSync } from 'node:fs';
import { createDecoderModule } from 'draco3d';
import { describe, expect, it, vi } from 'vitest';
import { gltf } from '../../src/index.ts';
import { glb } from './glb.ts';

// hex_grass.glb compressed by `gltf-transform draco` at its default quantization: 14-bit
// positions, 10-bit normals and 12-bit uvs. ribbon.glb is a generated two-joint ribbon (CC0) with
// byte joints, short indices and a clip, and ribbon_draco.glb the same at 16-bit positions.
const fixture = (name: string) => readFileSync(new URL(`fixtures/${name}`, import.meta.url));
const draco = gltf.draco(await createDecoderModule());

/** Whether every vertex of `a` has one in `b` within `tolerance`, in any order. */
function within(a: Float32Array, b: Float32Array, width: number, tolerance: number) {
  const vertices = (values: Float32Array) =>
    Array.from({ length: values.length / width }, (_, i) =>
      values.subarray(i * width, (i + 1) * width),
    );
  const others = vertices(b);
  return vertices(a).every((p) =>
    others.some((q) => p.every((value, c) => Math.abs(value - q[c]) <= tolerance)),
  );
}

/** A decoder that drops the first three values of every attribute. */
const short: gltf.Draco = (bytes, attributes) => {
  const mesh = draco(bytes, attributes);
  return { ...mesh, attributes: mesh.attributes.map((values) => values.subarray(3)) };
};

describe('gltf.read with draco', () => {
  it('decodes a compressed tile into the arrays of the uncompressed one', () => {
    const floats = gltf.read(fixture('hex_grass.glb')).meshes[0];
    const decoded = gltf.read(fixture('hex_grass_draco.glb'), { draco }).meshes[0];

    // Draco splits vertices along its traversal, so the tile keeps its triangles, not its count
    expect(decoded.indices).toHaveLength(floats.indices.length);
    expect(within(decoded.positions, floats.positions, 3, 2e-4)).toBe(true);
    expect(within(decoded.uvs, floats.uvs, 2, 5e-4)).toBe(true);
    expect(within(decoded.normals, floats.normals, 3, 0.01)).toBe(true);
    expect(decoded.material).toEqual(floats.material);
  });

  it('decodes skinned vertices, byte joints and short indices', () => {
    const plain = gltf.read(fixture('ribbon.glb'));
    const decoded = gltf.read(fixture('ribbon_draco.glb'), { draco });
    const [want, got] = [plain.meshes[0], decoded.meshes[0]];

    expect(decoded.skin!.joints.map(({ name }) => name)).toEqual(
      plain.skin!.joints.map(({ name }) => name),
    );
    expect(decoded.clips).toEqual(plain.clips);
    expect(got.indices).toHaveLength(want.indices.length);
    expect(within(got.positions, want.positions, 3, 2e-5)).toBe(true);
    // Each decoded vertex keeps its joints and weights together
    const bound = (mesh: typeof got) =>
      new Float32Array(
        Array.from({ length: mesh.positions.length / 3 }, (_, i) => [
          ...mesh.positions.subarray(i * 3, i * 3 + 3),
          ...mesh.joints!.subarray(i * 4, i * 4 + 4),
          ...mesh.weights!.subarray(i * 4, i * 4 + 4),
        ]).flat(),
      );
    expect(within(bound(got), bound(want), 11, 2e-5)).toBe(true);
  });

  it('decodes a primitive once however many nodes place it', () => {
    // Split the fixture into its chunks and place its one mesh by a second node
    const bytes = fixture('hex_grass_draco.glb');
    const view = new DataView(bytes.buffer, bytes.byteOffset);
    const length = view.getUint32(12, true);
    const json = JSON.parse(new TextDecoder().decode(bytes.subarray(20, 20 + length)));
    const bin = bytes.subarray(28 + length, 28 + length + view.getUint32(20 + length, true));
    json.nodes.push({ mesh: json.nodes[0].mesh, translation: [3, 0, 0] });
    json.scenes[0].nodes.push(json.nodes.length - 1);

    const counted = vi.fn(draco);
    const [mesh] = gltf.read(glb(json, bin), { draco: counted }).meshes;
    expect(counted).toHaveBeenCalledOnce();
    expect(mesh!.indices).toHaveLength(2 * 108);
  });

  it('decodes TEXCOORD_1 only where a texture samples it or the read asks for it', () => {
    // Give the fixture's primitive a second set, compressed as its first
    const bytes = fixture('hex_grass_draco.glb');
    const view = new DataView(bytes.buffer, bytes.byteOffset);
    const length = view.getUint32(12, true);
    const json = JSON.parse(new TextDecoder().decode(bytes.subarray(20, 20 + length)));
    const bin = bytes.subarray(28 + length, 28 + length + view.getUint32(20 + length, true));
    const primitive = json.meshes[0].primitives[0];
    primitive.attributes.TEXCOORD_1 = primitive.attributes.TEXCOORD_0;
    const ids = primitive.extensions.KHR_draco_mesh_compression.attributes;
    ids.TEXCOORD_1 = ids.TEXCOORD_0;
    const file = glb(json, bin);

    const asked: number[] = [];
    const counting: gltf.Draco = (data, attributes) => {
      asked.push(attributes.length);
      return draco(data, attributes);
    };
    expect(gltf.read(file, { draco: counting }).meshes[0]).not.toHaveProperty('uvs1');
    const kept = gltf.read(file, { draco: counting, uvs1: true }).meshes[0];
    expect(asked[1]).toBe(asked[0] + 1);
    expect(kept.uvs1).toEqual(kept.uvs);
  });

  it('asks for a decoder, naming the conversion too', () => {
    expect(() => gltf.read(fixture('hex_grass_draco.glb'))).toThrow(
      'Pass a decoder made with `gltf.draco` as the draco option, or convert it with `gltf-transform copy in.glb out.glb`.',
    );
  });

  it('checks what a decoder returns against the accessors', () => {
    expect(() => gltf.read(fixture('hex_grass_draco.glb'), { draco: short })).toThrow(
      /Accessor \d+ holds \d+ values, and Draco decoded \d+\./,
    );
  });

  it('reads a compressed strip as the list of triangles Draco decodes', () => {
    const bytes = fixture('hex_grass_draco.glb');
    const view = new DataView(bytes.buffer, bytes.byteOffset);
    const length = view.getUint32(12, true);
    const json = JSON.parse(new TextDecoder().decode(bytes.subarray(20, 20 + length)));
    const bin = bytes.subarray(28 + length, 28 + length + view.getUint32(20 + length, true));
    const listed = gltf.read(glb(json, bin), { draco }).meshes[0];
    // A strip of the same triangles lists two indices more than it has triangles
    const primitive = json.meshes[0].primitives[0];
    primitive.mode = 5;
    json.accessors[primitive.indices].count = listed.indices.length / 3 + 2;

    expect(gltf.read(glb(json, bin), { draco }).meshes[0].indices).toEqual(listed.indices);
  });
});
