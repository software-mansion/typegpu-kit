import type { SampledFlag, StorageFlag } from 'typegpu';
import { afterEach, expect, expectTypeOf, it, vi } from 'vitest';
import { targets } from '../src/index.ts';
import { recordingRoot } from '../src/testing.ts';

const specs = {
  scene: { format: 'rgba16float', samples: 4 },
  hdr: { format: 'rgba16float', sampled: true },
  depth: { format: 'depth24plus', samples: 4 },
  bloom: { format: 'rgba16float', scale: 1 / 2, mips: 3, sampled: true },
  kept: { format: 'rgba16float', samples: 4, transient: false },
  read: { format: 'rgba16float', samples: 4, sampled: true },
  tile: { format: 'rgba16float', transient: true },
} as const;

afterEach(() => {
  vi.unstubAllGlobals();
});

function setup(transientUsage = true) {
  const { root } = recordingRoot();
  if (!transientUsage) {
    const { TRANSIENT_ATTACHMENT: _, ...usages } = GPUTextureUsage;
    vi.stubGlobal('GPUTextureUsage', usages);
  }

  const size = { width: 101, height: 50 };
  const screen = targets.create(root, size, specs);

  const raw = (name: keyof typeof specs) =>
    root.unwrap(screen.textures[name]) as GPUTexture & { destroyed: boolean };

  return { root, size, screen, raw };
}

it('sizes each target from the source and its scale', () => {
  const { screen, raw } = setup();

  expect([screen.width, screen.height]).toEqual([101, 50]);
  expect([raw('hdr').width, raw('hdr').height]).toEqual([101, 50]);
  expect([raw('bloom').width, raw('bloom').height, raw('bloom').mipLevelCount]).toEqual([
    51, 25, 3,
  ]);
  expect(raw('scene').sampleCount).toBe(4);
});

it('makes multisampled targets transient unless sampled or kept, and others only when asked', () => {
  const { screen } = setup();

  expect(screen.transient).toEqual({
    scene: true,
    hdr: false,
    depth: true,
    bloom: false,
    kept: false,
    read: false,
    tile: true,
  });
});

it('makes transient targets memoryless where the browser has the usage', () => {
  const memoryless = setup();

  expect(memoryless.screen.memoryless).toBe(true);
  expect(memoryless.raw('depth').usage).toBe(
    GPUTextureUsage.TRANSIENT_ATTACHMENT | GPUTextureUsage.RENDER_ATTACHMENT,
  );
  expect(memoryless.raw('hdr').usage & GPUTextureUsage.TEXTURE_BINDING).not.toBe(0);
  expect(memoryless.raw('kept').usage & GPUTextureUsage.TRANSIENT_ATTACHMENT).toBe(0);
});

it('falls back to plain attachments without the usage', () => {
  const plain = setup(false);

  expect(plain.screen.memoryless).toBe(false);
  const { RENDER_ATTACHMENT, COPY_SRC, COPY_DST } = GPUTextureUsage;
  expect(plain.raw('scene').usage).toBe(RENDER_ATTACHMENT | COPY_SRC | COPY_DST);
});

it('creates attachment views once per rebuild, one per mip', () => {
  const { screen } = setup();
  const views = { ...screen.views };

  expect(screen.levels.bloom).toHaveLength(3);
  expect(screen.levels.bloom[2]).toEqual({ baseMipLevel: 2, mipLevelCount: 1 });
  expect(screen.views.hdr).toBe(views.hdr);
});

it('makes no more mips than the size has', () => {
  const { root } = recordingRoot();
  const size = { width: 16, height: 9 };
  const screen = targets.create(root, size, { chain: { format: 'rgba16float', mips: 8 } });
  const raw = () => root.unwrap(screen.textures.chain);

  expect(raw().mipLevelCount).toBe(5);
  expect(screen.levels.chain).toHaveLength(5);

  size.width = 1024;
  screen.fit();
  expect(raw().mipLevelCount).toBe(8);
  expect(screen.levels.chain).toHaveLength(8);
});

it('rebuilds the whole set when the size changes, and only then', () => {
  const { size, screen, raw } = setup();
  const old = raw('hdr');

  expect(screen.fit()).toBe(false);

  size.width = 200;
  expect(screen.fit()).toBe(true);
  expect(old.destroyed).toBe(true);
  expect(raw('hdr').width).toBe(200);
  expect(raw('hdr').destroyed).toBe(false);
});

it('refuses transient targets that would be read', () => {
  const { root } = recordingRoot();
  const size = { width: 1, height: 1 };

  expect(() =>
    // @ts-expect-error a transient target can't be sampled
    targets.create(root, size, { bad: { format: 'rgba8unorm', transient: true, sampled: true } }),
  ).toThrow("Target 'bad' is transient");
});

it('refuses multisampled targets that would be mipped or stored, at any size', () => {
  const { root } = recordingRoot();
  const size = { width: 1, height: 1 };

  expect(() =>
    targets.create(root, size, { bad: { format: 'rgba8unorm', samples: 4, mips: 3 } }),
  ).toThrow("Target 'bad' is multisampled");
  expect(() =>
    targets.create(root, size, { bad: { format: 'rgba8unorm', samples: 4, storage: true } }),
  ).toThrow("Target 'bad' is multisampled");
});

it('binds storage only in formats that TypeGPU binds as storage', () => {
  const { root } = recordingRoot();
  const size = { width: 1, height: 1 };
  const screen = targets.create(root, size, { field: { format: 'rgba16float', storage: true } });

  expectTypeOf(screen.textures.field).toExtend<StorageFlag>();
  // A parameter typed by the bare name takes any set
  expectTypeOf(screen).toExtend<targets.Targets>();
  expect(root.unwrap(screen.textures.field).usage & GPUTextureUsage.STORAGE_BINDING).not.toBe(0);

  const rejected = () => {
    // @ts-expect-error a depth format has no storage binding
    targets.create(root, size, { depth: { format: 'depth24plus', storage: true } });
    // @ts-expect-error a transient target can't be stored
    targets.create(root, size, { scene: { format: 'rgba8unorm', transient: true, storage: true } });
  };
  void rejected;
});

it('reports a mistake in a spec at the spec, and keeps the other targets typed', () => {
  const { root } = recordingRoot();
  const size = { width: 1, height: 1 };

  const rejected = () => {
    const misspelt = targets.create(root, size, {
      hdr: { format: 'rgba16float', sampled: true },
      // @ts-expect-error not a field of a spec
      depth: { format: 'depth24plus', smapled: true },
    });
    expectTypeOf(misspelt.textures.hdr).toExtend<SampledFlag>();

    const mistyped = targets.create(root, size, {
      hdr: { format: 'rgba16float', sampled: true },
      // @ts-expect-error mips is a number
      bloom: { format: 'rgba16float', mips: 'three' },
    });
    expectTypeOf(mistyped.textures.hdr).toExtend<SampledFlag>();
  };
  void rejected;
});

it('follows a scaled source when its scale changes', () => {
  const { root } = recordingRoot();
  const surface = { width: 200, height: 100 };
  let resolution = 1;

  const screen = targets.create(
    root,
    targets.scaled(surface, () => resolution),
    {
      color: { format: 'rgba8unorm' },
      half: { format: 'rgba8unorm', scale: 1 / 2 },
    },
  );

  expect(screen.fit()).toBe(false);

  resolution = 0.5;
  expect(screen.fit()).toBe(true);
  expect([screen.width, screen.height]).toEqual([100, 50]);
  expect(screen.textures.half.props.size).toEqual([50, 25]);

  surface.width = 300;
  expect(screen.fit()).toBe(true);
  expect(screen.width).toBe(150);
});

it('draws a region at a new resolution without making textures', () => {
  const { root } = recordingRoot();
  const surface = { width: 200, height: 101 };
  let resolution = 1;

  const screen = targets.create(
    root,
    surface,
    {
      color: { format: 'rgba8unorm', sampled: true },
      half: { format: 'rgba8unorm', scale: 1 / 2 },
    },
    { resolution: () => resolution },
  );
  const { color, half } = screen.regions;
  const texture = screen.textures.color;

  expect([color.width, color.height]).toEqual([200, 101]);
  expect([half.width, half.height]).toEqual([100, 51]);

  resolution = 0.5;
  expect([color.width, color.height]).toEqual([200, 101]);
  expect(screen.fit()).toBe(false);
  expect(screen.textures.color).toBe(texture);
  expect([color.width, color.height]).toEqual([100, 51]);
  expect([half.width, half.height]).toEqual([50, 26]);

  resolution = 2;
  screen.fit();
  expect([color.width, color.height]).toEqual([200, 101]);

  resolution = 0.5;
  surface.width = 300;
  expect(screen.fit()).toBe(true);
  expect(screen.regions.color).toBe(color);
  expect([color.width, color.height]).toEqual([150, 51]);
});

it('measures nothing in a fit that keeps the size and the resolution', () => {
  const { root } = recordingRoot();
  let resolution = 0.5;

  const screen = targets.create(
    root,
    { width: 200, height: 100 },
    { color: { format: 'rgba8unorm' } },
    { resolution: () => resolution },
  );
  const measured = vi.spyOn(screen.textures.color, 'props', 'get');

  screen.fit();
  expect(measured).not.toHaveBeenCalled();

  resolution = 0.25;
  screen.fit();
  expect(measured).toHaveBeenCalled();
  expect([screen.regions.color.width, screen.regions.color.height]).toEqual([50, 25]);
});

it('rounds a scaled set up, as a spec scale does', () => {
  const { root } = recordingRoot();
  const size = { width: 101, height: 49 };
  const color = { format: 'rgba8unorm' } as const;

  const scaled = targets.create(root, targets.scaled(size, 1 / 4), { color });
  const spec = targets.create(root, size, { color: { ...color, scale: 1 / 4 } });

  expect(scaled.textures.color.props.size).toEqual([26, 13]);
  expect(spec.textures.color.props.size).toEqual([26, 13]);

  const floats = targets.create(root, targets.scaled({ width: 100, height: 50 }, 1.1), { color });
  expect([floats.width, floats.height]).toEqual([110, 55]);
});

it('scales by a constant', () => {
  const size = targets.scaled({ width: 101, height: 50 }, 0.25);

  expect([size.width, size.height]).toEqual([25.25, 12.5]);
});

it('makes a layered target one 2D array, with an attachment view per layer', () => {
  const { root } = recordingRoot();
  const screen = targets.create(
    root,
    { width: 8, height: 4 },
    {
      cascades: { format: 'depth16unorm', layers: 3, sampled: true },
      flat: { format: 'depth16unorm' },
    },
  );

  expect(screen.textures.cascades.props.size).toEqual([8, 4, 3]);
  expect(screen.layers.cascades).toHaveLength(3);
  expect(screen.layers.cascades[2]).toEqual({
    dimension: '2d',
    arrayLayerCount: 1,
    baseArrayLayer: 2,
  });
  expect(screen.layers.cascades[0]).toBe(screen.views.cascades);
  expect(screen.layers.flat).toEqual([screen.views.flat]);
});

it('rejects layers it cannot make', () => {
  const { root } = recordingRoot();
  const size = { width: 8, height: 4 };
  const make = (spec: object) => () =>
    targets.create(root, size, { each: { format: 'depth16unorm', ...spec } });

  expect(make({ layers: 0 })).toThrow('needs 1 to 256 layers, got 0');
  expect(make({ layers: 257 })).toThrow(RangeError);
  expect(make({ layers: 2, samples: 4 })).toThrow(
    'Omit `layers` to set `samples` or `mips` above 1.',
  );
  expect(make({ layers: 2, mips: 2 })).toThrow('Omit `layers` to set `samples` or `mips` above 1.');
});

it('leaves a set whose rebuild threw unfitted, so a retry at the same size rebuilds it', () => {
  const { root, size, screen } = setup();
  const createTexture = root.createTexture.bind(root);
  let made = 0;
  root.createTexture = ((descriptor: never) => {
    if (++made === 2) throw new Error('out of memory');
    return createTexture(descriptor);
  }) as never;

  size.width = 200;
  expect(() => screen.fit()).toThrow('out of memory');
  expect(screen.fitted).toBe(false);
  expect(screen.fit()).toBe(true);
  expect(screen.fitted).toBe(true);
  expect(screen.textures.hdr.props.size).toEqual([200, 50]);
});

it('keeps a target named like a property of every object', () => {
  const { root } = recordingRoot();
  const named = targets.create(root, { width: 4, height: 4 }, {
    ['__proto__']: { format: 'rgba8unorm' },
  } as Record<string, targets.Spec>);

  expect(Object.keys(named.textures)).toEqual(['__proto__']);
  expect(named.regions.__proto__).toEqual({ width: 4, height: 4 });
});
