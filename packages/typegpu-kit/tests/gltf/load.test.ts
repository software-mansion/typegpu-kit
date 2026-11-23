import { describe, expect, it } from 'vitest';
import { gltf } from '../../src/index.ts';
import { glb } from './glb.ts';

/** One triangle in the xy plane, its bytes the model's only buffer. */
const TRIANGLE = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
const PNG = new Uint8Array([0x89, 0x50, 0x4e, 0x47]);
const JPEG = new Uint8Array([0xff, 0xd8, 0xff, 0xe0]);

/** The triangle textured by image 0, with `buffer` and `image` overriding how the files are kept. */
function model(buffer: object, image: object) {
  return {
    asset: { version: '2.0' },
    buffers: [{ byteLength: TRIANGLE.byteLength, ...buffer }],
    bufferViews: [{ buffer: 0, byteLength: TRIANGLE.byteLength }],
    accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }],
    materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }],
    textures: [{ source: 0 }],
    images: [image],
  };
}

/** A fetch serving `files` by URL, recording what it was asked for. */
function server(files: Record<string, Uint8Array>) {
  const asked: string[] = [];
  const fetch: gltf.Fetch = async (url) => {
    asked.push(url);
    const bytes = files[url];
    return {
      ok: bytes !== undefined,
      status: bytes ? 200 : 404,
      arrayBuffer: async () => bytes!.slice().buffer,
    };
  };
  return { fetch, asked };
}

const json = (value: object) => new TextEncoder().encode(JSON.stringify(value));
const bin = new Uint8Array(TRIANGLE.buffer);

describe('gltf.load', () => {
  it('fetches the buffers and images a model keeps beside it', async () => {
    const { fetch, asked } = server({
      'https://example.com/pack/Models/tri.gltf': json(
        model({ uri: 'buffers/tri.bin' }, { uri: '../Textures/colormap.png' }),
      ),
      'https://example.com/pack/Models/buffers/tri.bin': bin,
      'https://example.com/pack/Textures/colormap.png': PNG,
      '/pack/tri.glb': glb(model({}, { uri: 'Textures/colormap.jpg' }), bin),
      '/pack/Textures/colormap.jpg': JPEG,
    });

    const separate = await gltf.load('https://example.com/pack/Models/tri.gltf', { fetch });
    expect([...separate.meshes[0].positions]).toEqual([...TRIANGLE]);
    expect(separate.meshes[0].material.textures.color?.image).toBe(0);
    expect(separate.images).toEqual([{ bytes: PNG, mimeType: 'image/png' }]);

    const packed = await gltf.load('/pack/tri.glb', { fetch });
    expect([...packed.meshes[0].positions]).toEqual([...TRIANGLE]);
    expect(packed.images).toEqual([{ bytes: JPEG, mimeType: 'image/jpeg' }]);

    expect(asked).toHaveLength(5);
  });

  it('names the file it could not fetch', async () => {
    const { fetch } = server({
      '/tri.glb': glb(model({}, { uri: 'Textures/colormap.png' }), bin),
    });
    await expect(gltf.load('/tri.glb', { fetch })).rejects.toThrow(
      "gltf.load couldn't fetch image 0 of the model, /Textures/colormap.png: HTTP 404.",
    );
  });

  it('reports an image failure even while a separate buffer is pending', async () => {
    const { promise: buffer, resolve } = Promise.withResolvers<Awaited<ReturnType<gltf.Fetch>>>();
    const { fetch } = server({
      '/tri.gltf': json(model({ uri: 'tri.bin' }, { uri: 'tri.png' })),
    });
    try {
      await expect(
        gltf.load('/tri.gltf', { fetch: (url) => (url === '/tri.bin' ? buffer : fetch(url)) }),
      ).rejects.toThrow("gltf.load couldn't fetch image 0 of the model, /tri.png: HTTP 404.");
    } finally {
      resolve({ ok: true, status: 200, arrayBuffer: async () => bin.slice().buffer });
    }
  });

  it('checks the lists of files before fetching any', async () => {
    const { fetch, asked } = server({
      '/tri.gltf': json({ ...model({ uri: 'tri.bin' }, {}), images: 5 }),
    });
    await expect(gltf.load('/tri.gltf', { fetch })).rejects.toThrow(
      "The model's images must be a list.",
    );
    expect(asked).toEqual(['/tri.gltf']);
  });
});
