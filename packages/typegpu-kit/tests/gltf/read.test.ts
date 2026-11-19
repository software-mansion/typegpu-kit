import { readFileSync } from 'node:fs';
import { describe, expect, expectTypeOf, it } from 'vitest';
import { gltf } from '../../src/index.ts';
import { builder, glb } from './glb.ts';

// KayKit's hex_grass tile (CC0, Kay Lousberg), packed by gltf-transform as floats, and again
// quantized: normalized shorts and bytes, posed by its node.
const fixture = (name: string) => readFileSync(new URL(`fixtures/${name}`, import.meta.url));

/** The values with -0 as 0, which `toEqual` tells apart. */
const plain = (values: ArrayLike<number>) => Array.from(values, (v) => v + 0);

/** Whether every vertex of `a` has one in `b` within `tolerance`, so files compare in any order. */
function same(a: Float32Array, b: Float32Array, width: number, tolerance: number) {
  const vertex = (values: Float32Array, i: number) => values.subarray(i * width, (i + 1) * width);
  const near = (p: Float32Array, q: Float32Array) =>
    p.every((value, c) => Math.abs(value - q[c]) <= tolerance);

  const count = b.length / width;
  return (
    a.length === b.length &&
    Array.from({ length: count }, (_, i) => vertex(a, i)).every((p) =>
      Array.from({ length: count }, (_, j) => vertex(b, j)).some((q) => near(p, q)),
    )
  );
}

/** One triangle in the xy plane, counter-clockwise seen from +z. */
const TRIANGLE = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);

/** A file of `TRIANGLE` under one node, its POSITION accessor 0 and its view 0, to break. */
function triangle() {
  const file = builder();
  const position = file.accessor(TRIANGLE, 'VEC3');
  Object.assign(file.json, {
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: position } }] }],
  });
  return file;
}

/**
 * `triangle()` in `material`, with texture 0 of image 0 of its view 0 beside it, then `json`.
 * Returns a function that reads it.
 */
function painted(material: object, json: object = {}) {
  const file = triangle();
  Object.assign(file.json, {
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }],
    materials: [material],
    textures: [{ source: 0 }],
    images: [{ bufferView: 0, mimeType: 'image/png' }],
    ...json,
  });
  return () => gltf.read(file.build());
}

/** Rotates `v` by a unit quaternion `[x, y, z, w]`. */
function rotate([x, y, z, w]: readonly number[], v: readonly number[]) {
  const t = [2 * (y! * v[2] - z! * v[1]), 2 * (z! * v[0] - x! * v[2]), 2 * (x! * v[1] - y! * v[0])];
  return [
    v[0] + w! * t[0] + (y! * t[2] - z! * t[1]),
    v[1] + w! * t[1] + (z! * t[0] - x! * t[2]),
    v[2] + w! * t[2] + (x! * t[1] - y! * t[0]),
  ];
}

/** A base at x = 1, with a barrel above it, turned a quarter about y and twice as big. */
function turret(barrel: object = {}) {
  const file = builder();
  const position = file.accessor(TRIANGLE, 'VEC3');
  const primitive = { attributes: { POSITION: position }, material: 0 };
  const quarter = Math.SQRT1_2;
  Object.assign(file.json, {
    scenes: [{ nodes: [0] }],
    nodes: [
      { name: 'base', mesh: 0, translation: [1, 0, 0], children: [1] },
      {
        name: 'barrel',
        mesh: 0,
        translation: [0, 0.5, 0],
        rotation: [0, quarter, 0, quarter],
        scale: [2, 2, 2],
        ...barrel,
      },
    ],
    meshes: [{ primitives: [primitive] }],
    materials: [{ name: 'colormap' }],
  });
  return file.build();
}

/** A mesh of two materials under three nodes, one of them in the part 'sign'. */
function town() {
  const file = builder();
  const position = file.accessor(TRIANGLE, 'VEC3');
  const quarter = Math.SQRT1_2;
  Object.assign(file.json, {
    scenes: [{ nodes: [0, 1, 2, 3] }],
    nodes: [
      { mesh: 0, translation: [4, 0, 0] },
      { mesh: 0, rotation: [0, quarter, 0, quarter], scale: [2, 2, 2] },
      { name: 'sign', mesh: 0, translation: [0, 3, 0] },
      { mesh: 1, matrix: [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 6, 7, 1] },
    ],
    meshes: [
      {
        primitives: [
          { attributes: { POSITION: position }, material: 0 },
          { attributes: { POSITION: position }, material: 1 },
        ],
      },
      { primitives: [{ attributes: { POSITION: position }, material: 0 }] },
    ],
    materials: [{ name: 'wood' }, { name: 'stone' }],
  });
  return file.build();
}

describe('gltf.read', () => {
  it('reads a KayKit tile: interleaved floats, its material and its atlas', () => {
    const model = gltf.read(fixture('hex_grass.glb'));
    expect(model.meshes).toHaveLength(1);

    const [mesh] = model.meshes;
    expect(mesh!.positions).toHaveLength(70 * 3);
    expect(mesh!.indices).toHaveLength(108);
    expect(Math.max(...mesh!.positions.filter((_, i) => i % 3 === 2))).toBeCloseTo(1.1547, 4);

    for (let i = 0; i < 70; i++) {
      const [x, y, z] = mesh!.normals.subarray(i * 3, i * 3 + 3);
      expect(Math.hypot(x!, y!, z!)).toBeCloseTo(1, 5);
    }

    expect(mesh!.material).toMatchObject({
      name: 'hexagons_medieval',
      color: [1, 1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      alpha: 'opaque',
      textures: {
        color: {
          image: 0,
          sampler: { magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear' },
        },
      },
    });
    expect(model.images[0].mimeType).toBe('image/png');
    expect([...model.images[0].bytes.subarray(1, 4)]).toEqual([0x50, 0x4e, 0x47]);
  });

  it('dequantizes normalized integers and poses them by their node', () => {
    const floats = gltf.read(fixture('hex_grass.glb')).meshes[0];
    const quantized = gltf.read(fixture('hex_grass_quantized.glb')).meshes[0];

    // Positions and uvs are 16-bit there, and normals 8-bit.
    expect(same(quantized.positions, floats.positions, 3, 2e-4)).toBe(true);
    expect(same(quantized.uvs, floats.uvs, 2, 2e-4)).toBe(true);
    expect(same(quantized.normals, floats.normals, 3, 0.01)).toBe(true);
  });

  it('poses meshes through the node tree, normals by the inverse transpose', () => {
    const file = builder();
    const position = file.accessor(TRIANGLE, 'VEC3');
    const normal = file.accessor(new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0]), 'VEC3');
    Object.assign(file.json, {
      scenes: [{ nodes: [0] }],
      nodes: [
        { translation: [10, 0, 0], children: [1] },
        // Twice as wide along x, and turned a quarter about z: x goes to y.
        { rotation: [0, 0, Math.SQRT1_2, Math.SQRT1_2], scale: [2, 1, 1], mesh: 0 },
      ],
      meshes: [{ primitives: [{ attributes: { POSITION: position, NORMAL: normal } }] }],
    });

    const [mesh] = gltf.read(file.build()).meshes;
    expect(plain(mesh!.positions.map((v) => +v.toFixed(5)))).toEqual([10, 0, 0, 10, 2, 0, 9, 0, 0]);
    expect(plain(mesh!.normals.subarray(0, 3).map((v) => +v.toFixed(5)))).toEqual([-1, 0, 0]);
    expect([...mesh!.indices]).toEqual([0, 1, 2]);
  });

  it('keeps triangles facing out under a mirroring scale', () => {
    const file = builder();
    const position = file.accessor(TRIANGLE, 'VEC3');
    const normal = file.accessor(new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]), 'VEC3');
    Object.assign(file.json, {
      nodes: [{ scale: [-1, 1, 1], mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: position, NORMAL: normal } }] }],
    });

    const [mesh] = gltf.read(file.build()).meshes;
    expect(plain(mesh!.positions)).toEqual([0, 0, 0, -1, 0, 0, 0, 1, 0]);
    expect([...mesh!.indices]).toEqual([0, 2, 1]);
    expect(plain(mesh!.normals.subarray(0, 3))).toEqual([0, 0, 1]);
  });

  it('turns strips and fans into lists, every triangle wound like the first', () => {
    const file = builder();
    const square = file.accessor(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]), 'VEC3');
    const strip = file.accessor(new Uint16Array([0, 1, 2, 3]), 'SCALAR');
    const fan = file.accessor(new Uint16Array([0, 1, 3, 2]), 'SCALAR');
    Object.assign(file.json, {
      nodes: [{ mesh: 0 }],
      meshes: [
        {
          primitives: [
            { attributes: { POSITION: square }, indices: strip, mode: 5, material: 0 },
            { attributes: { POSITION: square }, indices: fan, mode: 6, material: 1 },
          ],
        },
      ],
      materials: [{}, {}],
    });

    const [strips, fans] = gltf.read(file.build()).meshes;
    expect([...strips!.indices]).toEqual([0, 1, 2, 3, 4, 5]);
    // Unwelded for flat normals: every triangle faces +z.
    expect(plain(strips!.normals)).toEqual(Array.from({ length: 6 }, () => [0, 0, 1]).flat());
    expect(plain(fans!.normals)).toEqual(Array.from({ length: 6 }, () => [0, 0, 1]).flat());
  });

  it('reads strided, normalized and sparse accessors', () => {
    const file = builder();
    // Positions and normalized byte uvs interleaved, 16 bytes a vertex.
    const interleaved = new DataView(new ArrayBuffer(48));
    TRIANGLE.forEach((value, i) => {
      interleaved.setFloat32(Math.floor(i / 3) * 16 + (i % 3) * 4, value, true);
    });
    [0, 0, 255, 0, 0, 255].forEach((value, i) => {
      interleaved.setUint8(Math.floor(i / 2) * 16 + 12 + (i % 2), value);
    });
    const vertices = file.view(new Uint8Array(interleaved.buffer), 16);

    const slots = file.view(new Uint8Array([2]));
    const moved = file.view(new Float32Array([0, 3, 0]));
    file.json.accessors.push(
      { bufferView: vertices, componentType: 5126, count: 3, type: 'VEC3' },
      {
        bufferView: vertices,
        byteOffset: 12,
        componentType: 5121,
        normalized: true,
        count: 3,
        type: 'VEC2',
      },
      // The same positions, with the third vertex moved up.
      {
        bufferView: vertices,
        componentType: 5126,
        count: 3,
        type: 'VEC3',
        sparse: {
          count: 1,
          indices: { bufferView: slots, componentType: 5121 },
          values: { bufferView: moved },
        },
      },
    );
    Object.assign(file.json, {
      nodes: [{ mesh: 0 }, { mesh: 1 }],
      meshes: [
        { primitives: [{ attributes: { POSITION: 0, TEXCOORD_0: 1 }, material: 0 }] },
        { primitives: [{ attributes: { POSITION: 2 }, material: 1 }] },
      ],
      materials: [{}, {}],
    });

    const [first, sparse] = gltf.read(file.build()).meshes;
    expect([...first!.positions]).toEqual([...TRIANGLE]);
    expect([...first!.uvs]).toEqual([0, 0, 1, 0, 0, 1]);
    expect([...sparse!.positions]).toEqual([0, 0, 0, 1, 0, 0, 0, 3, 0]);
    expect([...sparse!.uvs]).toEqual([0, 0, 0, 0, 0, 0]);
  });

  it('joins every primitive of a material, and gives the rest glTF defaults', () => {
    const file = builder();
    const position = file.accessor(TRIANGLE, 'VEC3');
    const primitive = { attributes: { POSITION: position } };
    Object.assign(file.json, {
      nodes: [{ mesh: 0 }, { mesh: 0, translation: [0, 0, 5] }],
      meshes: [{ primitives: [primitive] }],
    });

    const { meshes } = gltf.read(file.build());
    expect(meshes).toHaveLength(1);
    expect([...meshes[0].indices]).toEqual([0, 1, 2, 3, 4, 5]);
    expect(meshes[0].material).toMatchObject({ color: [1, 1, 1, 1], roughness: 1, metallic: 1 });
  });

  it('reads vertex colours, white where a joined primitive has none', () => {
    const file = builder();
    const position = file.accessor(TRIANGLE, 'VEC3');
    const normal = file.accessor(new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]), 'VEC3');
    const rgb = file.accessor(new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), 'VEC3');
    const bytes = file.accessor(
      new Uint8Array([255, 0, 0, 255, 0, 255, 0, 0, 0, 0, 255, 51]),
      'VEC4',
      {
        normalized: true,
      },
    );
    Object.assign(file.json, {
      nodes: [{ mesh: 0 }, { mesh: 1 }],
      meshes: [
        {
          primitives: [
            { attributes: { POSITION: position, NORMAL: normal, COLOR_0: rgb } },
            { attributes: { POSITION: position, NORMAL: normal } },
          ],
        },
        { primitives: [{ attributes: { POSITION: position, COLOR_0: bytes }, material: 0 }] },
      ],
      materials: [{}],
    });

    const [joined, flat] = gltf.read(file.build()).meshes;
    const white = Array.from({ length: 12 }, () => 1);
    expect([...joined!.extra!.color!]).toEqual([1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, ...white]);
    // Without normals each corner gets a vertex of its own, and keeps its colour.
    expect(plain(flat!.extra!.color!.map((value) => Math.round(value * 255)))).toEqual([
      255, 0, 0, 255, 0, 255, 0, 0, 0, 0, 255, 51,
    ]);

    const uncoloured = triangle();
    expect(gltf.read(uncoloured.build()).meshes[0]).not.toHaveProperty('extra');

    const integers = triangle();
    integers.accessor(new Uint8Array(12), 'VEC4');
    Object.assign(integers.json, {
      meshes: [{ primitives: [{ attributes: { POSITION: 0, COLOR_0: 1 } }] }],
    });
    expect(() => gltf.read(integers.build())).toThrow(
      "Accessor 1 holds integers that aren't normalized, which COLOR_0 can't.",
    );
  });

  describe('parts', () => {
    it('keeps a named node out of the meshes, posed about its own origin', () => {
      const model = gltf.read(turret(), { parts: ['barrel'] });
      const { barrel } = model.parts;

      expect(model.meshes).toHaveLength(1);
      expect([...model.meshes[0].positions]).toEqual([1, 0, 0, 2, 0, 0, 1, 1, 0]);

      expect(barrel.meshes).toHaveLength(1);
      expect(barrel.images).toBe(model.images);
      expect(barrel.meshes[0].material.name).toBe('colormap');
      expect(plain(barrel.meshes[0].positions)).toEqual([0, 0, 0, 2, 0, 0, 0, 2, 0]);
      expect(barrel.position).toEqual([1, 0.5, 0]);
      for (const [axis, value] of [0, Math.SQRT1_2, 0, Math.SQRT1_2].entries()) {
        expect(barrel.rotation[axis]).toBeCloseTo(value, 6);
      }
    });

    it('puts a part back where the whole model has it', () => {
      const whole = gltf.read(turret({ scale: [2, 1, 3] }));
      const { barrel } = gltf.read(turret({ scale: [2, 1, 3] }), { parts: ['barrel'] }).parts;

      const local = barrel.meshes[0].positions;
      const placed = new Float32Array(local.length);
      for (let i = 0; i < local.length; i += 3) {
        const turned = rotate(barrel.rotation, [...local.subarray(i, i + 3)]);
        placed.set(
          turned.map((value, axis) => value + barrel.position[axis]),
          i,
        );
      }

      expect(same(placed, whole.meshes[0].positions.subarray(9), 3, 1e-6)).toBe(true);
    });

    it('keeps a mirror in the part, facing the way the whole model does', () => {
      const whole = gltf.read(turret({ scale: [-1, 1, 1] })).meshes[0];
      const { barrel } = gltf.read(turret({ scale: [-1, 1, 1] }), { parts: ['barrel'] }).parts;
      const mesh = barrel.meshes[0];

      // Flat normals, turned back into the model, point where the whole model's do.
      const normal = rotate(barrel.rotation, [...mesh.normals.subarray(0, 3)]);
      const expected = [...whole.normals.subarray(9, 12)];
      for (const [axis, value] of expected.entries()) expect(normal[axis]).toBeCloseTo(value, 6);

      // And the winding agrees with them.
      const [a, b, c] = [0, 1, 2].map((v) => [...mesh.positions.subarray(v * 3, v * 3 + 3)]);
      const facing = [
        (b![1] - a![1]) * (c![2] - a![2]) - (b![2] - a![2]) * (c![1] - a![1]),
        (b![2] - a![2]) * (c![0] - a![0]) - (b![0] - a![0]) * (c![2] - a![2]),
        (b![0] - a![0]) * (c![1] - a![1]) - (b![1] - a![1]) * (c![0] - a![0]),
      ];
      expect(
        facing.reduce((sum, value, axis) => sum + value * mesh.normals[axis], 0),
      ).toBeGreaterThan(0);
    });

    it('turns the whole model and its parts by the rotation option', () => {
      // A quarter turn about y, which takes +x to -z.
      const quarter = [0, Math.SQRT1_2, 0, Math.SQRT1_2] as const;
      const model = gltf.read(turret(), { parts: ['barrel'], rotation: quarter });
      const { barrel } = model.parts;

      const base = [...model.meshes[0].positions].map((value) => Number(value.toFixed(6)) + 0);
      expect(base).toEqual([0, 0, -1, 0, 0, -2, 0, 1, -1]);
      expect(barrel.position.map((value) => Number(value.toFixed(6)) + 0)).toEqual([0, 0.5, -1]);
      for (const [axis, value] of [0, 1, 0, 0].entries()) {
        expect(barrel.rotation[axis]).toBeCloseTo(value, 6);
      }

      expect(() => gltf.read(turret(), { rotation: [0, 1, 0] as never })).toThrow(
        "The rotation option isn't 4 numbers.",
      );
    });

    it('makes a named node below another one a part of its own', () => {
      const model = gltf.read(turret(), { parts: ['base', 'barrel'] });

      expect(model.meshes).toEqual([]);
      expect(Object.keys(model.parts)).toEqual(['base', 'barrel']);
      expect(plain(model.parts.base.meshes[0].positions)).toEqual(Array.from(TRIANGLE));
      expect(model.parts.barrel.position).toEqual([1, 0.5, 0]);
    });

    it('names the nodes there are when a name matches none, or two', () => {
      expect(() => gltf.read(turret(), { parts: ['Barrel'] })).toThrow(
        "No node of the default scene is named 'Barrel'. Its nodes are named base, barrel.",
      );
      expect(() => gltf.read(turret({ name: 'base' }), { parts: ['base'] })).toThrow(
        "Two nodes are named 'base'",
      );
    });

    it('types each part by its name', () => {
      const model = gltf.read(turret(), { parts: ['barrel'] });

      expectTypeOf(model.parts.barrel).toEqualTypeOf<gltf.PartData>();
      expectTypeOf(gltf.read(turret()).parts).toEqualTypeOf<{}>();
      // @ts-expect-error -- only the parts asked for are there.
      void model.parts.base;
    });
  });

  it('reads emission and occlusion strength, the textures and their samplers', () => {
    const file = builder();
    const position = file.accessor(TRIANGLE, 'VEC3');
    const png = file.view(new Uint8Array([0x89, 0x50, 0x4e, 0x47]));
    const webp = file.view(new Uint8Array([0x52, 0x49, 0x46, 0x46]));
    Object.assign(file.json, {
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: position }, material: 0 }] }],
      materials: [
        {
          pbrMetallicRoughness: { metallicRoughnessTexture: { index: 0 } },
          normalTexture: { index: 1, scale: 0.5 },
          occlusionTexture: { index: 0, strength: 0.25 },
          emissiveFactor: [1, 0.5, 0],
          extensions: { KHR_materials_emissive_strength: { emissiveStrength: 4 } },
          alphaMode: 'MASK',
          doubleSided: true,
        },
      ],
      textures: [{ source: 0, sampler: 0 }, { extensions: { EXT_texture_webp: { source: 1 } } }],
      samplers: [{ magFilter: 9728, minFilter: 9984, wrapS: 33071, wrapT: 33648 }],
      images: [
        { bufferView: png, mimeType: 'image/png' },
        { bufferView: webp, mimeType: 'image/webp' },
      ],
    });

    const { meshes, images } = gltf.read(file.build());
    expect(meshes[0].material).toMatchObject({
      emission: [4, 2, 0],
      normalScale: 0.5,
      occlusionStrength: 0.25,
      alpha: 'mask',
      alphaCutoff: 0.5,
      doubleSided: true,
      textures: {
        metallicRoughness: {
          image: 0,
          sampler: {
            addressModeU: 'clamp-to-edge',
            addressModeV: 'mirror-repeat',
            magFilter: 'nearest',
            minFilter: 'nearest',
            mipmapFilter: 'nearest',
          },
        },
        normal: { image: 1, sampler: { addressModeU: 'repeat', minFilter: 'linear' } },
        occlusion: { image: 0 },
      },
    });
    expect(images.map((image) => image.mimeType)).toEqual(['image/png', 'image/webp']);

    // Copies, so a texture cache that keeps them doesn't keep the file.
    expect(images.map((image) => image.bytes.buffer.byteLength)).toEqual([4, 4]);
    expect([...images[1].bytes]).toEqual([0x52, 0x49, 0x46, 0x46]);
  });

  it('reads nothing outside the file, its chunks or their buffer views', () => {
    const file = triangle().build();
    expect(() => gltf.read(file.subarray(0, -4))).toThrow('The binary glTF is incomplete');

    const long = file.slice();
    new DataView(long.buffer).setUint32(12, 1000, true);
    expect(() => gltf.read(long)).toThrow('A chunk of the binary glTF runs past its end.');

    // Moved 4 bytes on, the view ends past the binary chunk, at the end of a larger buffer.
    const past = triangle();
    Object.assign(past.json.bufferViews[0], { byteOffset: 4 });
    const bytes = past.build();
    const larger = new Uint8Array(bytes.byteLength + 64);
    larger.set(bytes);

    expect(() => gltf.read(larger.subarray(0, bytes.byteLength))).toThrow(
      'Buffer view 0 exceeds buffer 0. It ends at byte 40, but the buffer has 36 bytes.',
    );

    // A view into a second buffer, which a .glb keeps in a file beside it.
    const outside = triangle();
    Object.assign(outside.json.bufferViews[0], { buffer: 1 });
    expect(() => gltf.read(outside.build())).toThrow(
      'The model keeps data outside its binary chunk; load it with `gltf.load`',
    );

    const broken = glb({ asset: { version: '2.0' }, nodes: [{ mesh: 0 }] }).slice();
    broken[20] = 0x5b;
    expect(() => gltf.read(broken)).toThrow('The binary glTF contains invalid JSON');
  });

  it('checks an accessor against its buffer view before it allocates', () => {
    // A million positions would take 12 MB; the range check comes before that.
    const huge = triangle();
    Object.assign(huge.json.accessors[0], { count: 1e6 });
    expect(() => gltf.read(huge.build())).toThrow(
      'Accessor 0 exceeds its buffer view. It needs 12000000 bytes, but the view has 36.',
    );

    const zeros = triangle();
    Object.assign(zeros.json.accessors[0], { bufferView: undefined, count: 1e6 });
    expect(() => gltf.read(zeros.build())).toThrow('Accessor 0 has no buffer view');

    const fraction = triangle();
    Object.assign(fraction.json.accessors[0], { count: 1.5 });
    expect(() => gltf.read(fraction.build())).toThrow(
      "Accessor 0's count is 1.5, not a positive whole number.",
    );

    const sparse = triangle();
    const slots = sparse.view(new Uint8Array([7]));
    const values = sparse.view(new Float32Array([0, 3, 0]));
    Object.assign(sparse.json.accessors[0], {
      sparse: {
        count: 1,
        indices: { bufferView: slots, componentType: 5121 },
        values: { bufferView: values },
      },
    });
    expect(() => gltf.read(sparse.build())).toThrow(
      "Accessor 0's sparse indices reach element 7, and it has 3.",
    );
  });

  it('checks attributes and indices against what glTF allows', () => {
    const plane = triangle();
    Object.assign(plane.json.accessors[0], { type: 'VEC2', count: 4 });
    expect(() => gltf.read(plane.build())).toThrow(
      'Accessor 0 is VEC2, and POSITION must be VEC3.',
    );

    const normals = triangle();
    normals.accessor(new Float32Array(12), 'VEC3');
    Object.assign(normals.json, {
      meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 1 } }] }],
    });
    expect(() => gltf.read(normals.build())).toThrow('A primitive has 3 positions and 4 normals.');

    /** The triangle, its corners listed by `indices`. */
    const listed = (indices: Uint16Array | Float32Array) => {
      const file = triangle();
      file.accessor(indices, 'SCALAR');
      Object.assign(file.json, {
        meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
      });
      return () => gltf.read(file.build());
    };
    expect(listed(new Float32Array([0, 1, 2]))).toThrow(
      "Accessor 1 has component type 5126, which indices can't have.",
    );
    expect(listed(new Uint16Array([0, 1, 3]))).toThrow(
      "A primitive's indices reach vertex 3, and it has 3.",
    );
    expect(listed(new Uint16Array([0, 1, 2, 0]))).toThrow(
      "A primitive of triangles lists 4 vertices, which isn't a multiple of three.",
    );
  });

  it('walks the node tree once, refusing cycles and second parents', () => {
    const loop = triangle();
    Object.assign(loop.json, {
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0, children: [1] }, { children: [0] }],
    });
    expect(() => gltf.read(loop.build())).toThrow('Node 0 is its own ancestor.');

    // Each node lists the next twice, which would make 2^39 copies of the last.
    const doubled = triangle();
    Object.assign(doubled.json, {
      nodes: Array.from({ length: 40 }, (_, i) => (i < 39 ? { children: [i + 1, i + 1] } : {})),
    });
    expect(() => gltf.read(doubled.build())).toThrow('Node 39 appears twice in the scene');

    // A chain deeper than the call stack, each node a step along x.
    const deep = triangle();
    Object.assign(deep.json, {
      nodes: Array.from({ length: 100_000 }, (_, i) => ({
        translation: [1, 0, 0],
        ...(i < 99_999 ? { children: [i + 1] } : { mesh: 0 }),
      })),
    });
    expect(gltf.read(deep.build()).meshes[0].positions[0]).toBe(100_000);
  });

  it('refuses a model whose nodes place more than maxBytes of meshes, before reading them', () => {
    // Ten nodes place a mesh whose positions claim a million vertices, from a file of a few hundred
    // bytes: each placement is 36 MB of vertices and indices.
    const crowd = triangle();
    Object.assign(crowd.json.accessors[0], { count: 1e6 });
    Object.assign(crowd.json, { nodes: Array.from({ length: 10 }, () => ({ mesh: 0 })) });

    expect(() => gltf.read(crowd.build())).toThrow(
      "The model's nodes place 343.3 MiB of meshes, past the 256.0 MiB of maxBytes. Read it with placements: true to read each mesh once.",
    );
    // Placed, the mesh counts once; maxBytes moves the limit either way.
    expect(() => gltf.read(crowd.build(), { placements: true, maxBytes: 2 ** 20 })).toThrow(
      "The model's meshes take 34.3 MiB, past the 1.0 MiB of maxBytes.",
    );
    expect(() => gltf.read(crowd.build(), { maxBytes: 0 })).toThrow(
      'gltf.read requires maxBytes to be greater than 0. Received 0.',
    );
  });

  describe('placements', () => {
    it('reads each mesh once per material, in its own space, and where each node places it', () => {
      const model = gltf.read(town(), { placements: true, parts: ['sign'] });
      expect(model.meshes.map((mesh) => mesh.material.name)).toEqual(['wood', 'stone', 'wood']);
      expect(plain(model.meshes[0].positions)).toEqual(plain(TRIANGLE));

      const placed = model.placements.map(({ mesh, position, rotation, scale }) => ({
        mesh,
        position: plain(position),
        rotation: plain(rotation.map((v) => +v.toFixed(5))),
        scale: plain(scale.map((v) => +v.toFixed(5))),
      }));
      const half = +Math.SQRT1_2.toFixed(5);
      expect(placed).toEqual([
        { mesh: 0, position: [4, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
        { mesh: 1, position: [4, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
        { mesh: 0, position: [0, 0, 0], rotation: [0, half, 0, half], scale: [2, 2, 2] },
        { mesh: 1, position: [0, 0, 0], rotation: [0, half, 0, half], scale: [2, 2, 2] },
        { mesh: 2, position: [5, 6, 7], rotation: [0, 0, 0, 1], scale: [-1, 1, 1] },
      ]);
      // A part is read as without placements.
      expect(model.parts.sign.position).toEqual([0, 3, 0]);
      expect(model.parts.sign.meshes).toHaveLength(2);
    });

    it('places by the rotation option, and refuses a skinned model', () => {
      const quarter = Math.SQRT1_2;
      const turned = gltf.read(town(), { placements: true, rotation: [0, quarter, 0, quarter] });
      expect(plain(turned.placements[0].position.map((v) => +v.toFixed(5)))).toEqual([0, 0, -4]);

      const skinned = triangle();
      Object.assign(skinned.json, {
        nodes: [{ mesh: 0, skin: 0 }, { name: 'root' }],
        skins: [{ joints: [1] }],
      });
      expect(() => gltf.read(skinned.build(), { placements: true })).toThrow(
        'Skinned meshes require the bind pose. Read this model without placements.',
      );
    });

    it('types placements only when asked for', () => {
      expectTypeOf(gltf.read(town(), { placements: true }).placements).toEqualTypeOf<
        readonly gltf.PlacementData[]
      >();
      expectTypeOf(gltf.read(town())).not.toHaveProperty('placements');
    });
  });

  it('refuses indices past the lists they point into, and values of the wrong shape', () => {
    const elsewhere = triangle();
    Object.assign(elsewhere.json, { nodes: [{ mesh: 1 }] });
    expect(() => gltf.read(elsewhere.build())).toThrow('The model has no mesh 1.');

    const turned = triangle();
    Object.assign(turned.json, { nodes: [{ mesh: 0, rotation: [0, 0, 1] }] });
    expect(() => gltf.read(turned.build())).toThrow("Node 0's rotation isn't 4 numbers.");

    const normal = { normalTexture: { index: 0 } };
    expect(painted({}, { materials: [] })).toThrow('The model has no material 0.');
    expect(painted(normal, { textures: [] })).toThrow('The model has no texture 0.');
    expect(painted(normal, { textures: [{ source: 0, sampler: 0 }] })).toThrow(
      'The model has no sampler 0.',
    );
    expect(painted(normal, { images: [] })).toThrow('The model has no image 0.');
    expect(painted({ pbrMetallicRoughness: { baseColorFactor: 'red' } })).toThrow(
      "Material 0's baseColorFactor isn't 4 numbers.",
    );
  });

  it('reads TEXCOORD_1 as uvs1 where a texture samples it or the read asks for it', () => {
    const file = triangle();
    const second = file.accessor(new Float32Array([0.5, 0, 1, 0.5, 0, 1]), 'VEC2');
    const read = (material: object, options: { uvs1?: boolean } = {}) => {
      Object.assign(file.json, {
        meshes: [
          { primitives: [{ attributes: { POSITION: 0, TEXCOORD_1: second }, material: 0 }] },
        ],
        materials: [material],
        textures: [{ source: 0 }],
        images: [{ bufferView: 0, mimeType: 'image/png' }],
      });
      return gltf.read(file.build(), options).meshes[0];
    };

    const lightmapped = read({ emissiveTexture: { index: 0, texCoord: 1 } });
    expect([...lightmapped.uvs1!]).toEqual([0.5, 0, 1, 0.5, 0, 1]);
    expect(lightmapped.material.textures.emission!.uvSet).toBe(1);
    // Exporters write every uv set, and one no texture samples stays out of the mesh
    expect(read({ emissiveTexture: { index: 0 } })).not.toHaveProperty('uvs1');
    // A lightmap the program samples itself asks for it
    const asked = read({ emissiveTexture: { index: 0 } }, { uvs1: true });
    expect([...asked.uvs1!]).toEqual([0.5, 0, 1, 0.5, 0, 1]);
  });

  it('refuses a uv set the primitive lacks, and sets past TEXCOORD_1', () => {
    const second = { pbrMetallicRoughness: { baseColorTexture: { index: 0, texCoord: 1 } } };
    expect(painted(second)).toThrow(
      'Material 0 reads TEXCOORD_1, and a primitive drawn with it has none.',
    );

    const third = { KHR_texture_transform: { texCoord: 2 } };
    expect(painted({ normalTexture: { index: 0, extensions: third } })).toThrow(
      "Material 0's normal texture reads TEXCOORD_2, and gltf.read reads TEXCOORD_0 and 1.",
    );
  });

  it("keeps each texture's KHR_texture_transform, and leaves out one that doesn't move", () => {
    const moved = { KHR_texture_transform: { scale: [2, 2], rotation: 0.5 } };
    const still = { KHR_texture_transform: { offset: [0, 0], rotation: 0 } };
    const read = painted({
      normalTexture: { index: 0, extensions: moved },
      emissiveTexture: { index: 0, extensions: still },
    });

    const { textures } = read().meshes[0].material;
    expect(textures.normal!.transform).toEqual({ offset: [0, 0], rotation: 0.5, scale: [2, 2] });
    expect(textures.emission).not.toHaveProperty('transform');
  });

  it("doesn't find names from Object's prototype in its lookups", () => {
    const file = triangle();
    const png = file.view(new Uint8Array([0x89, 0x50, 0x4e, 0x47]));
    Object.assign(file.json, {
      meshes: [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }],
      materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }],
      textures: [{ source: 0, sampler: 0 }],
      samplers: [{ wrapS: 'constructor', wrapT: '__proto__' }],
      images: [{ bufferView: png, mimeType: 'image/png' }],
    });

    const [mesh] = gltf.read(file.build()).meshes;
    expect(mesh!.material.textures.color!.sampler).toMatchObject({
      addressModeU: 'repeat',
      addressModeV: 'repeat',
    });

    const odd = glb({ asset: { version: '2.0' }, extensionsRequired: ['constructor'] });
    expect(() => gltf.read(odd)).toThrow(
      "The model needs constructor, which gltf.read doesn't support.",
    );
  });

  it('refuses what it cannot read, naming the command that converts it', () => {
    const json = new TextEncoder().encode('{"asset":{"version":"2.0"}}');
    expect(() => gltf.read(json)).toThrow('gltf-transform copy in.gltf out.glb');

    const draco = glb({
      asset: { version: '2.0' },
      extensionsRequired: ['KHR_draco_mesh_compression'],
    });
    expect(() => gltf.read(draco)).toThrow('convert it with `gltf-transform copy in.glb out.glb`');

    const lines = builder();
    const position = lines.accessor(TRIANGLE, 'VEC3');
    Object.assign(lines.json, {
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: position }, mode: 1 }] }],
    });
    expect(() => gltf.read(lines.build())).toThrow('reads triangles');

    const basis = builder();
    const corners = basis.accessor(TRIANGLE, 'VEC3');
    Object.assign(basis.json, {
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: corners }, material: 0 }] }],
      materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }],
      textures: [{ extensions: { KHR_texture_basisu: { source: 0 } } }],
    });
    expect(() => gltf.read(basis.build())).toThrow('gltf-transform ktxdecompress');

    const beside = painted(
      { pbrMetallicRoughness: { baseColorTexture: { index: 0 } } },
      { images: [{ uri: '../Textures/variation-a.png' }] },
    );
    expect(beside).toThrow(
      'Image 0 of the model is a separate file, ../Textures/variation-a.png; load the model with `gltf.load`',
    );
  });
});
