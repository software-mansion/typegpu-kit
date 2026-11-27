import { cameras, passes, targets, type CameraBase } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';
import { meshes } from '@typegpu/kit/geometry';
import { d, type TgpuRoot } from 'typegpu';
import { describe, expect, it } from 'vitest';
import { leg, rest, shin } from './leg.ts';
import { impostors, instances, lighting, materials, meshPool, poses } from '../../src/index.ts';

const FIRST_INSTANCE = ['indirect-first-instance'];

/** Passes into an HDR target and depth, of `samples`, seen through `view`. */
const hdr = (root: TgpuRoot, view: CameraBase, samples = 1) =>
  passes
    .create(
      root,
      targets.create(
        root,
        { width: 1, height: 1 },
        {
          hdr: { format: 'rgba16float', samples },
          depth: { format: 'depth24plus', samples },
        },
      ),
      { hdr: passes.color, depth: passes.depth },
    )
    .pipe(view.inject());

/** Adds `renderer` to passes seen through `view`, and to the lighting's shadow when it has one. */
function compiled(
  root: TgpuRoot,
  view: CameraBase,
  light: lighting.Lighting,
  renderer: instances.Renderer,
  samples = 1,
) {
  const lit = hdr(root, view, samples);
  lit.add(renderer);
  light.shadow?.add(renderer);
  return Promise.all([lit.initAsync(), light.shadow?.initAsync()]).then(() => undefined);
}

function scene() {
  const { root, log } = recordingRoot({ features: FIRST_INSTANCE });
  const pool = meshPool.create(root, { vertices: 200, indices: 400, capacity: 8 });
  const box = pool.add(meshes.box());
  const plane = pool.add(meshes.plane());
  const card = pool.add(impostors.CARD).meshes[0];
  return { root, log, pool, box, plane, card };
}

/** A mesh of one triangle, of `material`. */
const triangle = (material: Partial<meshPool.MaterialData>): meshPool.MeshData => ({
  positions: new Float32Array(9),
  normals: new Float32Array(9),
  uvs: new Float32Array(6),
  indices: new Uint32Array([0, 1, 2]),
  material: {
    color: [1, 1, 1, 1],
    roughness: 1,
    metallic: 0,
    emission: [0, 0, 0],
    textures: {},
    ...material,
  },
});

describe('impostors', () => {
  it('capture the sphere around the parts, each view in a pass drawing every part, then the mipmaps, into the encoder given', () => {
    const { root, log, pool, box, plane, card } = scene();
    const start = log.events.length;
    const atlas = impostors.create(
      root,
      pool,
      [{ mesh: box }, { mesh: plane, position: [0, 1, 0] }],
      { card, views: 3, size: 16 },
    );
    expect(log.events.slice(start)).toEqual([]);

    const encoder = root.createCommandEncoder();
    atlas.record(encoder);
    expect(log.events.slice(start)).not.toContain('submit');
    encoder.submit();

    const capture = log.events.slice(start);
    // The sphere's extents emptied, its box and its radius fitted, then the sphere written.
    const fit = ['begin-compute', 'dispatch', 'dispatch', 'dispatch', 'dispatch', 'end-compute'];
    expect(capture.slice(0, fit.length)).toEqual(fit);
    const view = ['begin-render', 'draw-indexed', 'draw-indexed', 'end-render'];
    const views = capture.slice(fit.length, fit.length + 9 * 4);
    expect(views).toEqual(Array.from({ length: 9 }, () => view).flat());
    // For each texture, kernels fill the four levels below the first of 16 texels, every layer
    // at once, and copies move them in. A pass per layer and level would reach Firefox's 2048
    // passes per submit.
    const mips = [
      'begin-compute',
      ...Array.from({ length: 4 }, () => 'dispatch'),
      'end-compute',
      ...Array.from({ length: 4 }, () => 'copy-texture'),
    ];
    expect(capture.slice(fit.length + 9 * 4, -1)).toEqual([...mips, ...mips]);
    expect(capture.at(-1)).toBe('submit');
  });

  it('capture each look of mesh through a pipeline of its own: two-sided, masked or neither', () => {
    const { root, log, pool, box, card } = scene();
    const model = pool.add({
      meshes: [
        triangle({ doubleSided: true }),
        triangle({ alpha: 'mask' }),
        triangle({ doubleSided: true }),
      ],
      images: [],
    });

    const culls: (GPUCullMode | undefined)[] = [];
    const create = root.device.createRenderPipeline.bind(root.device);
    root.device.createRenderPipeline = (descriptor) => {
      culls.push(descriptor.primitive?.cullMode);
      return create(descriptor);
    };
    const pipelines = log.pipelines;
    const shaders = log.shaders.length;
    const encoder = root.createCommandEncoder();
    impostors
      .create(root, pool, [{ mesh: box }, ...model.meshes.map((mesh) => ({ mesh }))], {
        card,
        views: 2,
        size: 16,
      })
      .record(encoder);
    encoder.submit();

    // The sphere's four kernels, a pipeline for each of the three looks, and three that make
    // mipmaps.
    expect(log.pipelines - pipelines).toBe(10);
    expect(culls.toSorted()).toEqual(['back', 'back', 'none']);
    const fragments = log.shaders.slice(shaders).filter((code) => code.includes('@fragment'));
    expect(fragments.filter((code) => code.includes('discard'))).toHaveLength(1);
    expect(fragments.filter((code) => code.includes('front_facing'))).toHaveLength(1);
  });

  it('share the card, adding nothing to the pool, and free what they own when destroyed', () => {
    const { root, log, pool, box, card } = scene();
    const before = log.alive();
    const count = pool.meshes.length;
    const atlas = impostors.create(root, pool, box, { card, size: 16 });
    impostors.create(root, pool, box, { card, size: 16 }).destroy();
    expect(pool.meshes).toHaveLength(count);
    const encoder = root.createCommandEncoder();
    atlas.record(encoder);
    encoder.submit();
    // The atlas's two textures, its three multisampled targets, the two that fill its mips, the
    // sphere, its extents and the parts.
    expect(log.alive()).toHaveLength(before.length + 10);
    atlas.destroy();
    expect(log.alive()).toEqual(before);
  });

  it('refuse views, sizes, meshes and cards they cannot capture, and samplings they cannot read', () => {
    const { root, pool, box, card } = scene();
    const other = meshPool.create(root, { capacity: 64, vertices: 100, indices: 200 });
    const stranger = other.add(meshes.box());
    const strangerCard = other.add(impostors.CARD).meshes[0];

    expect(() => impostors.create(root, pool, box, { card, views: 1 })).toThrow(/2 to 16 views/);
    expect(() => impostors.create(root, pool, box, { card, views: 17 })).toThrow(/2 to 16 views/);
    expect(() => impostors.create(root, pool, box, { card, size: 100 })).toThrow(/power of two/);
    expect(() => impostors.create(root, pool, box, { card, size: 1024 })).toThrow(/power of two/);
    expect(() => impostors.create(root, pool, box, { card, size: 16.5 })).toThrow(
      "An atlas's `size` is a power of two from 16 to 512 texels, got 16.5.",
    );
    expect(() => impostors.create(root, pool, [], { card })).toThrow(/at least one mesh/);
    expect(() => impostors.create(root, pool, stranger, { card })).toThrow(
      /must belong to the pool/,
    );
    expect(() => impostors.create(root, pool, box, { card: strangerCard })).toThrow(
      /added to the pool supplied/,
    );
    const atlas = impostors.create(root, pool, box, { card, size: 16 });
    expect(() => atlas.inject('bilinear' as never)).toThrow(/'nearest', 'blended' or 'parallax'/);
  });

  it('rejects missing poses, unequal phase counts and atlases beyond the device layer limit before allocating', () => {
    const { root, log, pool, card } = scene();
    const body = pool.add({ meshes: [shin], images: [] });
    const mesh = body.meshes[0];
    const figures = poses.create(root, leg, { rest }, { capacity: 256 });
    const first = figures.add('rest', 0);
    const second = figures.add('rest', 0);
    const before = log.counts();
    expect(() =>
      impostors.create(root, pool, [{ mesh, pose: [] }], { card, poses: figures }),
    ).toThrow(/nonzero phase count/);
    expect(() =>
      impostors.create(
        root,
        pool,
        [
          { mesh, pose: [first] },
          { mesh, pose: [first, second] },
        ],
        { card, poses: figures },
      ),
    ).toThrow(/same nonzero phase count/);
    expect(() => impostors.create(root, pool, [{ mesh, pose: first }], { card })).toThrow(
      /live in options.poses/,
    );
    expect(() =>
      impostors.create(root, pool, [{ mesh, pose: [first, second] }], {
        card,
        poses: figures,
        views: 16,
      }),
    ).toThrow(/512 layers/);
    expect(log.counts()).toEqual(before);
    root.destroy();
  });

  it('cut out what they cover too little of in every pass, shading once after a prepass if asked', async () => {
    const { root, log, pool, box, card } = scene();
    const atlas = impostors.create(root, pool, box, { card, size: 16 });
    const view = cameras.orbit(root, {
      size: { width: 1, height: 1 },
      target: [0, 0, 0],
      distance: 5,
    });
    const light = lighting.create(root, { shadow: {} });
    const cards = instances.create(root, pool, { kind: impostors.Impostor, capacity: 1024 });
    cards.add(card, materials.matte(d.vec3f(1)));

    const fragmentsOf = async (prepass: boolean) => {
      const shaders = log.shaders.length;
      const pipelines = log.pipelines;
      const drawn = cards.renderer({ prepass }).pipe(light.inject()).pipe(atlas.inject());
      await compiled(root, view, light, drawn, 4);
      const made = log.shaders.slice(shaders).filter((code) => code.includes('@fragment'));
      expect(made.every((code) => code.includes('@invariant'))).toBe(true);
      return { made, pipelines: log.pipelines - pipelines };
    };

    // By default the colour pass cuts out what it shades; the shadow writes the surface's depth
    const once = await fragmentsOf(false);
    expect(once.pipelines).toBe(2);
    const [shadow, color] = once.made.toSorted(
      (a, b) => Number(b.includes('frag_depth')) - Number(a.includes('frag_depth')),
    );
    expect(shadow).toContain('@builtin(frag_depth)');
    expect(shadow).toContain('discard');
    expect(color).toContain('discard');

    // A prepass cuts out, and the colour pass shades what shows
    const twice = await fragmentsOf(true);
    expect(twice.pipelines).toBe(3);
    const [cover, shaded] = twice.made
      .filter((code) => !code.includes('frag_depth'))
      .toSorted((a, b) => Number(b.includes('discard')) - Number(a.includes('discard')));
    expect(cover).toContain('discard');
    expect(shaded).not.toContain('discard');
  });

  it('read one atlas each way a renderer asks, through pipelines of its own', async () => {
    const { root, log, pool, box, card } = scene();
    const atlas = impostors.create(root, pool, box, { card, size: 16 });
    const view = cameras.orbit(root, {
      size: { width: 1, height: 1 },
      target: [0, 0, 0],
      distance: 5,
    });
    const light = lighting.create(root);
    const cards = instances.create(root, pool, { kind: impostors.Impostor, capacity: 1024 });
    cards.add(card);

    const shaders = log.shaders.length;
    for (const sampling of ['nearest', 'blended', 'parallax'] as const) {
      await compiled(
        root,
        view,
        light,
        cards.renderer().pipe(light.inject()).pipe(atlas.inject(sampling)),
      );
    }
    const fragments = log.shaders.slice(shaders).filter((code) => code.includes('@fragment'));
    expect(fragments).toHaveLength(3);
    expect(new Set(fragments).size).toBe(3);
  });

  it('need their atlas, naming what fills it', async () => {
    const { root, pool, card } = scene();
    const view = cameras.orbit(root, {
      size: { width: 1, height: 1 },
      target: [0, 0, 0],
      distance: 5,
    });
    const cards = instances.create(root, pool, { kind: impostors.Impostor, capacity: 1024 });
    cards.add(card);

    const light = lighting.create(root);
    await expect(
      compiled(root, view, light, cards.renderer().pipe(light.inject())),
    ).rejects.toThrow(/impostors\.create/);
  });
});
