import { recordingRoot } from '@typegpu/kit/testing';
import { meshes } from '@typegpu/kit/geometry';
import { d, tgpu } from 'typegpu';
import { describe, expect, it, onTestFinished, vi } from 'vitest';
import { images, materials, meshPool } from '../../src/index.ts';

/** A 2×2 image as pixels, which the pool takes without decoding. */
const CHECKER = {
  width: 2,
  height: 2,
  pixels: new Uint8Array([255, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 0, 0, 255]),
};

/** A quad facing +z, with `material`'s factors and textures. */
function quad(material: Partial<meshPool.MaterialData> = {}): meshPool.MeshData {
  return {
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, -1, 1, 0, 0, 0, 0, 1]),
    uvs: new Float32Array([0, 1, 1, 1, 1, 0, 0.5, 0.25]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    material: {
      color: [1, 1, 1, 1],
      roughness: 1,
      metallic: 1,
      emission: [0, 0, 0],
      textures: {},
      ...material,
    },
  };
}

/** A float from the bits a mesh's box holds it in. */
function fromOrdered(bits: number) {
  const word = new Uint32Array([bits >= 0x80000000 ? bits ^ 0x80000000 : ~bits]);
  return new Float32Array(word.buffer)[0];
}

/** Two quads, the second changed by `mesh`. */
const quads = (mesh: Partial<meshPool.MeshData>) => ({
  meshes: [quad(), { ...quad(), ...mesh }],
  images: [],
});

describe('meshPool', () => {
  it('makes exactly the room its models, meshes, shapes and counts take', () => {
    const room = meshPool.roomFor([
      { meshes: [quad(), quad()], images: [] },
      quad(),
      meshes.box(),
      { meshes: 1, vertices: 4, indices: 6 },
    ]);
    expect(room).toEqual({ capacity: 5, vertices: 4 * 3 + 24 + 4, indices: 6 * 3 + 36 + 6 });

    const { root } = recordingRoot();
    const pool = meshPool.create(root, room);
    pool.add({ meshes: [quad(), quad(), quad()], images: [] });
    pool.add(meshes.box());
    pool.add({ meshes: [quad()], images: [] });
    expect(() => pool.add(meshes.box())).toThrow();
  });

  it('keeps a second uv set when one of the meshes it makes room for carries one', () => {
    const lightmapped = { ...quad(), uvs1: new Float32Array(8) };
    expect(meshPool.roomFor([quad(), lightmapped])).toMatchObject({ uvs1: true });
    expect(meshPool.roomFor([quad()])).not.toHaveProperty('uvs1');
  });

  it('bakes each shape into the next ranges, each in a pass submitted at once', () => {
    const { root, log } = recordingRoot();
    const pool = meshPool.create(root, { vertices: 100, indices: 200, capacity: 4 });
    const box = meshes.box();

    const indexed = pool.add(box);
    const flat = pool.add(meshes.unindexed(box));

    expect(indexed).toMatchObject({
      index: 0,
      firstIndex: 0,
      indexCount: 36,
      baseVertex: 0,
      vertexCount: 24,
    });
    // An unindexed shape gets one index per vertex.
    expect(flat).toMatchObject({
      index: 1,
      firstIndex: 36,
      indexCount: 36,
      baseVertex: 24,
      vertexCount: 36,
    });
    expect(pool.meshes).toEqual([indexed, flat]);
    // An instance of a shape starts as white chalk.
    expect(flat.material).toEqual({
      color: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      emission: [0, 0, 0],
      opacity: 1,
    });
    expect(log.pipelines).toBe(2);
    expect(log.events).toEqual([
      'begin-compute',
      'dispatch',
      'end-compute',
      'submit',
      'begin-compute',
      'dispatch',
      'end-compute',
      'submit',
    ]);
  });

  it('compiles one bake for the functions of a shape, ahead without blocking for the shapes it names', async () => {
    const { root, log } = recordingRoot();
    const box = meshes.box();
    const ball = meshes.sphere({ segments: 8, rings: 4 });
    const pool = meshPool.create(root, {
      vertices: 200,
      indices: 400,
      capacity: 5,
      shapes: [ball],
    });

    pool.add(box);
    expect([log.pipelines, log.blocking]).toEqual([1, 1]);
    // The same shape again, and one that shares its functions, reuse its bake.
    pool.add(box);
    pool.add({ ...box });
    expect([log.pipelines, log.blocking]).toEqual([1, 1]);

    await pool.initAsync();
    pool.add(ball);
    expect([log.pipelines, log.blocking]).toEqual([2, 1]);
  });

  it('bakes shapes into an encoder it is given, and leaves the submit to it', () => {
    const { root, log } = recordingRoot();
    const pool = meshPool.create(root, { vertices: 100, indices: 200, capacity: 4 });
    const encoder = root.createCommandEncoder();

    pool.add(meshes.box(), encoder);
    pool.add(meshes.unindexed(meshes.box()), encoder);
    expect(log.submissions).toBe(0);

    encoder.submit();
    expect(log.events).toEqual([
      'begin-compute',
      'dispatch',
      'end-compute',
      'begin-compute',
      'dispatch',
      'end-compute',
      'submit',
    ]);
  });

  it('binds the positions for positionAt, and the normals and uvs as well for vertexAt', async () => {
    const { root, log } = recordingRoot();
    const pool = meshPool.create(root, { vertices: 16, indices: 16, capacity: 2 });
    const out = root.createBuffer(d.vec3f).$usage('storage').as('mutable');

    const position = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      out.$ = meshPool.positionAt(d.u32(0));
    });
    const vertex = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      out.$ = meshPool.vertexAt(d.u32(0)).normal;
    });
    for (const compute of [position, vertex]) {
      await root.pipe(pool.inject()).createComputePipeline({ compute }).initAsync();
    }

    // Each kernel also binds its output.
    expect(log.shaders.slice(-2).map((wgsl) => wgsl.match(/var<storage/g)!.length - 1)).toEqual([
      1, 2,
    ]);
  });

  it('binds a second uv set for uv1At only in a pool made with uvs1, and reads zero without, even piped after one with', async () => {
    const { root, log } = recordingRoot();
    const out = root.createBuffer(d.vec2f).$usage('storage').as('mutable');
    const compute = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      const uvs = meshPool.textureLayout.$.reading.uvs.color;
      out.$ = materials.uvOf(uvs, d.vec2f(), meshPool.uv1At(d.u32(0)));
    });
    const [plain, lit] = [false, true].map((uvs1) =>
      meshPool.create(root, { vertices: 16, indices: 16, capacity: 2, uvs1 }),
    );
    for (const piped of [
      root.pipe(plain!.inject()),
      root.pipe(lit!.inject()),
      root.pipe(lit!.inject()).pipe(plain!.inject()),
    ]) {
      await piped.createComputePipeline({ compute }).initAsync();
    }

    // Each kernel also binds its output.
    const shaders = log.shaders.slice(-3);
    expect(shaders.map((wgsl) => wgsl.match(/var<storage/g)!.length - 1)).toEqual([0, 1, 0]);
    // Without the set, textures read the first one with no choice between them
    expect(shaders.map((wgsl) => wgsl.includes('select('))).toEqual([false, true, false]);
  });

  it('names its inject() when a shader reads vertexAt without it', () => {
    const { root } = recordingRoot();

    const vertex = tgpu.vertexFn({
      in: { i: d.builtin.vertexIndex },
      out: { position: d.builtin.position },
    })(({ i }) => {
      'use gpu';
      return { position: d.vec4f(meshPool.vertexAt(i).position, 1) };
    });
    const fragment = tgpu.fragmentFn({ out: d.vec4f })(() => {
      'use gpu';
      return d.vec4f(1);
    });
    const pipeline = root.createRenderPipeline({
      vertex,
      fragment,
      targets: { format: 'rgba8unorm' },
    });

    expect(() => pipeline.initSync()).toThrow(
      "fn*:vertexAt(u32): Nothing filled the mesh pool's vertices: create the pipeline from `root.pipe(pool.inject())`, where `pool` is a `meshPool.create`.",
    );
  });

  it('throws when a shape does not fit, and leaves the pool as it was', () => {
    const { root } = recordingRoot();
    const pool = meshPool.create(root, { vertices: 30, indices: 200, capacity: 2 });

    expect(() => pool.add(meshes.box())).not.toThrow();
    expect(() => pool.add(meshes.box())).toThrow('room for 6 vertices');
    expect(pool.meshes).toHaveLength(1);

    pool.add(meshes.plane());
    expect(() => pool.add(meshes.plane())).toThrow('full (2 meshes)');
  });

  it('takes triangle lists only, and 1 to 256 meshes', () => {
    const { root } = recordingRoot();
    const pool = meshPool.create(root, { capacity: 64 });
    const lines = { ...meshes.box(), topology: 'line-list' as const };

    expect(() => pool.add(lines)).toThrow('triangle lists, got line-list');
    expect(() => meshPool.create(root, { capacity: 0 })).toThrow(RangeError);
    expect(() => meshPool.create(root, { capacity: 257 })).toThrow(RangeError);
  });

  it('copies a model in with a write per buffer, and starts its instances at its factors', () => {
    const { root, log, bytes } = recordingRoot();
    const pool = meshPool.create(root, { vertices: 100, indices: 200, capacity: 4 });
    pool.add(meshes.box());

    const model = pool.add({
      meshes: [quad(), quad({ color: [1, 0.5, 0, 0.25], roughness: 0.5, metallic: 0 })],
      images: [],
    });
    expect(model.meshes.map((mesh) => [mesh.index, mesh.baseVertex, mesh.firstIndex])).toEqual([
      [1, 24, 36],
      [2, 28, 42],
    ]);
    expect(model.meshes[1].material).toEqual({
      color: [1, 0.5, 0],
      roughness: 0.5,
      metallic: 0,
      emission: [0, 0, 0],
      opacity: 0.25,
    });
    // No pipeline: the arrays are written as they are, with normals and uvs packed on the CPU.
    expect(log.pipelines).toBe(1);

    const [positions, attributes, indices, box] = log.writes.slice(-4);
    expect([positions!.offset, attributes!.offset, indices!.offset, box!.offset]).toEqual([
      28 * 12,
      28 * 8,
      42 * 4,
      2 * 24,
    ]);
    // The quad's box, from (0, 0, 0) to (1, 1, 0), in bits whose unsigned order is the floats'.
    const bits = new Uint32Array(bytes(box!.buffer).buffer, 2 * 24, 6);
    expect([...bits].map(fromOrdered)).toEqual([0, 0, 0, 1, 1, 0]);
    expect([...new Uint32Array(bytes(indices!.buffer).buffer, 42 * 4, 6)]).toEqual([
      0, 1, 2, 0, 2, 3,
    ]);

    // +z folds to the centre, -z to a corner and +x to an edge; uvs are half floats.
    const words = new Uint32Array(bytes(attributes!.buffer).buffer, 28 * 8, 8);
    expect([...words].map((word) => word.toString(16))).toEqual([
      '0',
      '3c000000',
      '7fff7fff',
      '3c003c00',
      '7fff',
      '3c00',
      '0',
      '34003800',
    ]);
  });

  it('packs joints into bytes and weights into 255ths adding up to 255, once a skinned model comes', () => {
    const { root, log, bytes } = recordingRoot();
    const pool = meshPool.create(root, { vertices: 100, indices: 200, capacity: 4 });
    const [plain] = pool.add({ meshes: [quad()], images: [] }).meshes;
    const buffers = log.counts().buffers;
    expect(plain!.skinned).toBe(false);

    const joints = new Uint16Array([0, 0, 0, 0, 3, 2, 1, 0, 255, 0, 0, 0, 1, 2, 0, 0]);
    const weights = new Float32Array([
      1, 0, 0, 0, 0.75, 0.125, 0.125, 0, 1, 0, 0, 0, 0.5, 0.5, 0, 0,
    ]);
    const [skinned] = pool.add({ meshes: [{ ...quad(), joints, weights }], images: [] }).meshes;
    expect(skinned!.skinned).toBe(true);
    // The joints and weights of every vertex the pool holds.
    expect(log.counts().buffers).toBe(buffers + 1);

    const written = log.writes.at(-1)!;
    expect([written.offset, written.size]).toEqual([4 * 8, 4 * 8]);
    const words = new Uint8Array(bytes(written.buffer).buffer, 4 * 8, 4 * 8);
    // Per vertex: four joints, then four weights.
    expect(Array.from(words)).toEqual([
      0, 0, 0, 0, 255, 0, 0, 0, 3, 2, 1, 0, 191, 32, 32, 0, 255, 0, 0, 0, 255, 0, 0, 0, 1, 2, 0, 0,
      128, 127, 0, 0,
    ]);
  });

  it('refuses joints past 255, and weights that are missing or do not add up to 1', () => {
    const { root } = recordingRoot();
    const pool = meshPool.create(root, { vertices: 100, indices: 200, capacity: 4 });
    const joints = new Uint16Array(16);
    const weights = new Float32Array(16).fill(0.25);

    expect(() => pool.add(quads({ joints }))).toThrow(
      'Mesh 1 of the model has joints without weights, or weights without joints.',
    );
    expect(() => pool.add(quads({ joints: new Uint16Array(12), weights }))).toThrow(
      'has 12 joints and 16 weights for 4 vertices',
    );
    expect(() => pool.add(quads({ joints: joints.with(5, 256), weights }))).toThrow(
      'is bound to joint 256, and a pool keeps 256 joints at most',
    );
    expect(() => pool.add(quads({ joints, weights: weights.with(6, 0.5) }))).toThrow(
      "Mesh 1 of the model's vertex 1 has weights 0.25, 0.25, 0.5, 0.25, which don't add up to 1.",
    );
    expect(() =>
      pool.add(quads({ joints, weights: weights.with(0, -0.25).with(1, 0.75) })),
    ).toThrow('vertex 0 has weights -0.25, 0.75, 0.25, 0.25');
    expect(pool.meshes).toHaveLength(0);
  });

  it('names a skinned model when a shader reads joints before the pool holds one', () => {
    const { root } = recordingRoot();
    const pool = meshPool.create(root, { vertices: 16, indices: 16, capacity: 2 });
    const out = root.createBuffer(d.vec4f).$usage('storage').as('mutable');
    const compute = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      out.$ = meshPool.influencesAt(d.u32(0)).weights;
    });

    const pipeline = root.pipe(pool.inject()).createComputePipeline({ compute });
    expect(() => pipeline.initSync()).toThrow(
      'The mesh pool has no skinned model. Add one before compiling pipelines that read joints.',
    );
  });

  it('drops the extras it was not created with, warning once for each colour that varies', () => {
    const { root, log } = recordingRoot();
    const pool = meshPool.create(root, { vertices: 100, indices: 200, capacity: 6 });
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    pool.add(quads({ extra: { color: new Float32Array(16).fill(1 - 0.5 / 255) } }));
    expect(warn).not.toHaveBeenCalled();
    const before = log.writes.length;

    const colored = quads({ extra: { color: new Float32Array(16) } });
    pool.add(colored);
    pool.add(colored);

    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0][0]).toBe(
      "The mesh pool discards extra 'color'. Create it with `extra: { color: d.vec4f }` to keep the attribute.",
    );
    // Four writes a mesh, and none for its colours.
    expect(log.writes.length - before).toBe(16);
    expect(pool.extra).toEqual({});
    warn.mockRestore();
  });

  it('keeps the extras it was created with, as their struct lays them out, zero where none', () => {
    const { root, log, bytes } = recordingRoot();
    const pool = meshPool.create(root, {
      vertices: 16,
      indices: 64,
      capacity: 4,
      extra: { tint: d.vec3f, weight: d.f32, layer: d.u32 },
    });
    const before = log.writes.length;
    const tint = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1]);
    const layer = new Uint32Array([7, 8, 9, 10]);
    const [plain, extra] = pool.add(quads({ extra: { tint, layer } })).meshes;

    // The plain quad gives none, so only the second writes them.
    expect(log.writes.length - before).toBe(9);
    const written = log.writes.at(-1)!;
    // A vec3f, an f32 beside it and a u32 take 20 bytes, laid 32 apart.
    expect(written.offset).toBe(extra!.baseVertex * 32);
    const held = bytes(written.buffer);
    const words = new Uint32Array(held.buffer, held.byteOffset, held.byteLength / 4);
    const floats = new Float32Array(held.buffer, held.byteOffset, held.byteLength / 4);
    const at = (v: number) => (extra!.baseVertex + v) * 8;
    expect([...floats.subarray(at(1), at(1) + 3)]).toEqual([0, 1, 0]);
    expect(floats[at(1) + 3]).toBe(0);
    expect(words[at(1) + 4]).toBe(8);
    expect([...words.subarray(at(-plain!.vertexCount), at(0))].every((word) => word === 0)).toBe(
      true,
    );
    expect(Object.keys(pool.extra)).toEqual(['tint', 'weight', 'layer']);
  });

  it('reads an extra in a shader created from its inject', () => {
    const { root, log } = recordingRoot();
    const pool = meshPool.create(root, { capacity: 1, extra: { tint: d.vec4f } });
    const out = root.createBuffer(d.vec4f).$usage('storage').as('mutable');
    const compute = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      out.$ = pool.extra.tint(d.u32(2));
    });

    root.pipe(pool.inject()).createComputePipeline({ compute }).dispatchWorkgroups(1);
    expect(log.shaders.at(-1)).toMatch(/var<storage, read> \w+: array<\w+, 65536>;/);
    expect(log.shaders.at(-1)).toMatch(/return \w+\[i\]\.tint;/);
  });

  it('throws for an extra of another type or length, naming what it reads', () => {
    const { root } = recordingRoot();
    const pool = meshPool.create(root, { capacity: 4, extra: { tint: d.vec4f } });
    const ints = { extra: { tint: new Int32Array(16) } } as Partial<meshPool.MeshData>;

    expect(() => pool.add(quads(ints))).toThrow(
      "Mesh 1 of the model supplies extra 'tint' as Int32Array. Use Float32Array for the pool type vec4f.",
    );
    expect(() => pool.add(quads({ extra: { tint: new Float32Array(12) } }))).toThrow(
      "Mesh 1 of the model has 12 values for extra 'tint' and 4 vertices. Supply 4 per vertex.",
    );
  });

  it('throws when a model does not fit, and leaves the pool as it was', () => {
    const { root } = recordingRoot();
    const pool = meshPool.create(root, { vertices: 6, indices: 200, capacity: 4 });

    expect(() => pool.add({ meshes: [quad(), quad()], images: [] })).toThrow(
      'room for 6 vertices and 200 indices; the model needs 8 and 12',
    );
    expect(pool.meshes).toHaveLength(0);
  });

  it('refuses a model with no meshes, or whose arrays disagree, and leaves the pool as it was', () => {
    const { root } = recordingRoot();
    const pool = meshPool.create(root, { vertices: 100, indices: 200, capacity: 4 });

    expect(() => pool.add({ meshes: [], images: [] })).toThrow('no meshes');
    expect(() => pool.add(quads({ positions: new Float32Array(11) }))).toThrow(
      'Mesh 1 of the model has 11 position floats',
    );
    expect(() => pool.add(quads({ normals: new Float32Array(9) }))).toThrow(
      '9 normal floats for 4 vertices',
    );
    expect(() => pool.add(quads({ uvs: new Float32Array(10) }))).toThrow(
      '10 UV floats for 4 vertices',
    );
    expect(() => pool.add(quads({ indices: new Uint32Array([0, 1, 4]) }))).toThrow(
      'index of 4, past its 4 vertices',
    );
    expect(() => pool.add(quads({ uvs1: new Float32Array(6) }))).toThrow(
      '6 uvs1 floats for 4 vertices',
    );
    const lightmap = quad({ textures: { emission: { image: 0, uvSet: 1 } } });
    expect(() => pool.add({ meshes: [lightmap], images: [CHECKER] })).toThrow(
      'Mesh 0 of the model reads uv set 1 in its emission texture, and has no uvs1.',
    );
    expect(() =>
      pool.add({ meshes: [{ ...lightmap, uvs1: new Float32Array(8) }], images: [CHECKER] }),
    ).toThrow('Create the pool with `uvs1: true` to keep it.');
    expect(pool.meshes).toHaveLength(0);
  });

  it('uploads nothing for a model whose pool is destroyed while its images decode', async () => {
    const { root } = recordingRoot();
    const pool = meshPool.create(root, { capacity: 64 });
    const created = vi.spyOn(root.device, 'createTexture');

    const model = pool.add({
      meshes: [quad({ textures: { color: { image: 0 } } })],
      images: [CHECKER],
    });
    // A model whose material names no texture binds its own group and is ready at once.
    const plain = pool.add({ meshes: [quad({ alpha: 'mask' })], images: [] });
    const masked = pool.texturesOf(plain.meshes[0].index);
    expect(masked).not.toBe(pool.texturesOf(model.meshes[0].index));
    pool.destroy();

    await expect(model.ready).rejects.toThrow('destroyed before its images were uploaded');
    await expect(plain.ready).resolves.toBeUndefined();
    expect(created).not.toHaveBeenCalled();
  });

  it('decodes an image again for a later model when it failed', async () => {
    const decode = vi.fn(async () => {
      throw new Error('Not a PNG.');
    });
    vi.stubGlobal('createImageBitmap', decode);

    const { root } = recordingRoot();
    const pool = meshPool.create(root, { capacity: 64 });
    const broken = { bytes: new Uint8Array([1, 2, 3]), mimeType: 'image/png' };
    const textured = quad({ textures: { color: { image: 0 } } });

    for (let i = 0; i < 2; i++) {
      await expect(pool.add({ meshes: [textured], images: [broken] }).ready).rejects.toThrow(
        'Not a PNG.',
      );
    }
    expect(decode).toHaveBeenCalledTimes(2);

    vi.unstubAllGlobals();
  });

  it('hands createImageBitmap the bytes where a Blob of them throws, as on React Native', async () => {
    const decode = vi.fn(async (_source: unknown, _options?: ImageBitmapOptions) => {
      throw new Error('Not a PNG.');
    });
    vi.stubGlobal('createImageBitmap', decode);
    onTestFinished(() => {
      vi.unstubAllGlobals();
    });

    const { root } = recordingRoot();
    const pool = meshPool.create(root, { capacity: 64 });
    const png = { bytes: new Uint8Array([1, 2, 3]), mimeType: 'image/png' };
    const textured = quad({ textures: { color: { image: 0 } } });
    const options = { colorSpaceConversion: 'none', premultiplyAlpha: 'none' };

    await expect(pool.add({ meshes: [textured], images: [png] }).ready).rejects.toThrow(
      'Not a PNG.',
    );
    expect(decode.mock.calls[0][0]).toBeInstanceOf(Blob);

    // React Native's Blob, which throws on bytes.
    vi.stubGlobal('Blob', function Blob() {
      throw new Error("Creating blobs from 'ArrayBuffer' and 'ArrayBufferView' are not supported");
    });
    await expect(pool.add({ meshes: [textured], images: [png] }).ready).rejects.toThrow(
      'Not a PNG.',
    );
    expect(decode.mock.calls[1]).toEqual([png.bytes, options]);
  });

  it('decodes encoded images with its decode, and refuses them where it has no way to', async () => {
    const { root } = recordingRoot();
    const textured = quad({ textures: { color: { image: 0 } } });
    const png = { bytes: new Uint8Array([1, 2, 3]), mimeType: 'image/png' };

    // Node has no createImageBitmap, as React Native doesn't.
    const without = meshPool.create(root, { capacity: 64 });
    expect(() => without.add({ meshes: [textured], images: [png] })).toThrow('a `decode`');
    expect(without.meshes).toEqual([]);

    const seen: unknown[] = [];
    const pool = meshPool.create(root, {
      capacity: 64,
      decode: async (image) => {
        seen.push(image);
        return CHECKER;
      },
    });
    const model = pool.add({ meshes: [textured], images: [png] });
    const index = model.meshes[0].index;
    const before = pool.texturesOf(index);

    await model.ready;
    expect(seen).toEqual([png]);
    expect(pool.texturesOf(index)).not.toBe(before);
  });

  it('takes bitmaps decoded elsewhere, as a worker does from the JS thread', async () => {
    const decode = vi.fn(async (_source: unknown, _options?: ImageBitmapOptions) => {
      return { width: 4, height: 2, close() {} } as ImageBitmap;
    });
    vi.stubGlobal('createImageBitmap', decode);
    const png = { bytes: new Uint8Array([1, 2, 3]), mimeType: 'image/png' };
    const bitmap = await images.decodeBitmap(png);
    expect(decode).toHaveBeenCalledWith(expect.any(Blob), {
      colorSpaceConversion: 'none',
      premultiplyAlpha: 'none',
    });
    // The worker has no createImageBitmap.
    vi.unstubAllGlobals();

    const { root } = recordingRoot();
    const copied = vi.spyOn(root.device.queue, 'copyExternalImageToTexture');
    const created = vi.spyOn(root.device, 'createTexture');
    const pool = meshPool.create(root, { capacity: 64 });
    const textured = quad({ textures: { color: { image: 0 } } });

    const first = pool.add({ meshes: [textured], images: [bitmap] });
    const second = pool.add({ meshes: [textured], images: [bitmap] });
    await Promise.all([first.ready, second.ready]);

    // One texture of the bitmap's size, beside the 1×1 defaults.
    const sizes = created.mock.calls.map(([descriptor]) => descriptor.size);
    expect(sizes.filter((size) => size.toString() !== '1,1')).toEqual([[4, 2]]);
    expect(copied).toHaveBeenCalledTimes(1);
    expect(copied.mock.calls[0][0].source).toBe(bitmap);
  });

  it('filters textures anisotropically when asked, except through a nearest sampler', async () => {
    const { root } = recordingRoot();
    const pool = meshPool.create(root, { capacity: 64, anisotropy: 8 });
    const created = vi.spyOn(root.device, 'createSampler');
    const nearest = { image: 0, sampler: { magFilter: 'nearest' } } as const;

    const model = pool.add({
      meshes: [quad({ textures: { color: { image: 0 } } }), quad({ textures: { color: nearest } })],
      images: [CHECKER],
    });
    await model.ready;
    for (const mesh of model.meshes) root.unwrap(pool.texturesOf(mesh.index));

    const anisotropy = created.mock.calls.map(([descriptor]) => descriptor?.maxAnisotropy);
    expect(anisotropy).toContain(8);
    expect(anisotropy).toContain(1);
  });

  it('decodes each image once, however many meshes and models use it', async () => {
    const { root } = recordingRoot();
    const pool = meshPool.create(root, { capacity: 64 });
    const created = vi.spyOn(root.device, 'createTexture');
    const textured = quad({ textures: { color: { image: 0 } } });

    const first = pool.add({ meshes: [textured], images: [CHECKER] });
    const second = pool.add({
      meshes: [textured, quad({ textures: { emission: { image: 0 } } })],
      images: [{ ...CHECKER, pixels: CHECKER.pixels.slice() }],
    });
    // Until its textures are uploaded, a mesh binds the defaults.
    expect(pool.texturesOf(first.meshes[0].index)).toBe(pool.texturesOf(0));

    await Promise.all([first.ready, second.ready]);
    const formats = created.mock.calls.map(([descriptor]) => descriptor.format);
    // The same pixels as colour and as emission, both sRGB: one texture, with a mip.
    expect(formats).toEqual(['rgba8unorm-srgb']);
    expect(created.mock.calls[0][0].mipLevelCount).toBe(2);

    const [a, b, c] = [first.meshes[0], ...second.meshes].map((mesh) =>
      pool.texturesOf(mesh.index),
    );
    expect(a).toBe(b);
    expect(c).not.toBe(a);
  });
});
