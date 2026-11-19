import { bindings, Camera, cameras, frames, table } from '@typegpu/kit';
import * as grid from '@typegpu/kit/grid';
import { recordingRoot } from '@typegpu/kit/testing';
import { d, std, tgpu, type Configurable, type TgpuRoot } from 'typegpu';
import { describe, expect, expectTypeOf, it } from 'vitest';
import { lighting, materials, shading } from '../../src/index.ts';

const floatsOf = (bytes: Uint8Array) =>
  new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);

/** Where a column-major `matrix` takes `point`. */
const transform = (matrix: Float32Array, [x, y, z]: readonly number[]) =>
  [0, 1, 2].map(
    (row) => matrix[row] * x! + matrix[4 + row] * y! + matrix[8 + row] * z! + matrix[12 + row],
  );

/** A kernel that shades one point, so its pipeline holds the whole light loop. */
function shadeOnce(
  root: TgpuRoot,
  light: lighting.Lighting,
  around: {
    before?: (cfg: Configurable) => Configurable;
    after?: (cfg: Configurable) => Configurable;
  } = {},
) {
  const out = root.createBuffer(d.vec3f).$usage('storage').as('mutable');
  const eye = root.createUniform(Camera);

  const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    const surface = shading.pointAt(d.vec3f(), d.vec3f(0, 1, 0));
    out.$ = shading.shade(materials.matte(d.vec3f(1)), surface);
  });

  const pipeline = root
    .with(cameras.current, eye)
    .pipe(around.before ?? ((cfg) => cfg))
    .pipe(light.inject())
    .pipe(around.after ?? ((cfg) => cfg))
    .createComputePipeline({ compute: kernel });
  pipeline.initSync();

  return pipeline;
}

const Lamp = table.define('Lamp', { host: lighting.Light.propTypes });

/** Point lights from a table of lamps, and the neighbour grid that holds them, cells of `sight`. */
function lampsInCells(root: TgpuRoot, sight: number) {
  const lamps = table.create(root, Lamp, { capacity: 32 });
  const rows = lamps.inject();
  const cells = grid.create(root, {
    capacity: 32,
    low: d.vec3f(-8, 0, -8),
    size: d.vec3f(16, 4, 16),
    sight,
    item: d.u32,
    alive: Lamp.alive.pipe(rows),
    position: Lamp.position.pipe(rows),
    itemOf: (i) => {
      'use gpu';
      return d.u32(i);
    },
  });

  const lampAt = (slot: number) => {
    'use gpu';
    return lighting.Light({
      position: Lamp.position(slot),
      range: Lamp.range(slot),
      color: Lamp.color(slot),
      intensity: Lamp.intensity(slot),
    });
  };
  const candidates: grid.Candidates = cells.candidates();

  return { lamps, cells, light: tgpu.fn(lampAt).pipe(rows), candidates };
}

/** Malformed candidates' parts: a range that is a vector, and a slot that is a boolean. */
const vector = (_position: d.v3f, _r: number) => {
  'use gpu';
  return d.vec3f();
};
const yes = (_index: number) => {
  'use gpu';
  return true;
};

/** Fills `reflect` with `lambert` for one pipeline. */
const flat = (cfg: Configurable) => cfg.with(shading.reflect, shading.lambert);

/** A fog that reads the lighting's settings. */
const hazy: shading.FogHook = (point) => {
  'use gpu';
  const settings = shading.settings();
  const hidden = std.saturate(point.position.y * settings.fogDensity);
  return shading.Air({ added: settings.fogColor * hidden, through: d.vec3f(1 - hidden) });
};

/** A table of white point lights, or spot lights aimed down, for `points.table`. */
const pointsIn = (root: TgpuRoot, capacity = 16) =>
  table.create(root, lighting.PointLight, { capacity, defaults: { ...lighting.WHITE } });
const spotsIn = (root: TgpuRoot, capacity = 16) =>
  table.create(root, lighting.SpotLight, { capacity, defaults: { ...lighting.WHITE_SPOT } });

const SUN = 64 + 4 * 32;
const fitted = (root: TgpuRoot, shadow: Partial<lighting.FittedShadowOptions> = {}) => {
  const eye = cameras.orbit(root, {
    size: { width: 16, height: 9 },
    target: [0, 1, 0],
    distance: 12,
    pitch: 0.3,
    far: 400,
  });
  eye.jump();
  const light = lighting.create(root, {
    settings: { sunDirection: [0.6, 0.35, 0.3] },
    shadow: { eye, cascades: 3, far: 120, size: 512, ...shadow },
  });
  return { eye, light };
};
const passes = (events: readonly string[]) =>
  events.filter((event) => event === 'begin-render').length;

const cascadesOf = (floats: Float32Array) =>
  [0, 1, 2, 3].map((k) => {
    const at = 16 + k * 8;
    return { scale: floats[at], texel: floats[at + 3], x: floats[at + 4] };
  });

const lightOf = (_slot: number) => ({}) as lighting.Light;
const spotOf = (_slot: number) => ({}) as lighting.Spot;

describe('lighting', () => {
  it('creates only the environment buffer for shading alone or a function giving the point lights', () => {
    const { root, log } = recordingRoot();
    const shaded = lighting.create(root, { shading: { sunShadow: () => 1 } });
    const light = lighting.create(root, {
      points: {
        count: 64,
        light: (_slot) => {
          'use gpu';
          return lighting.Light({ position: d.vec3f(), range: 0, color: d.vec3f(1), intensity: 1 });
        },
      },
    });
    root.unwrap(shaded.settings.buffer);
    root.unwrap(light.settings.buffer);

    expect('points' in light).toBe(false);
    expect([log.buffers, log.textures, log.samplers]).toEqual([2, 0, 0]);
  });

  it('holds the sun, sky and fog as given over the defaults, which have no fog and nothing infinite', () => {
    const { root, bytes } = recordingRoot();
    const light = lighting.create(root, {
      settings: {
        sunDirection: [0, 2, 0],
        sunColor: [1, 0.5, 0],
        sunIntensity: 2,
        skyTop: [0.1, 0.2, 0.3],
        skyIntensity: 0.5,
        fogDensity: 0.05,
        fogFalloff: 0.25,
        skyReflection: 3,
      },
    });

    const values = [...floatsOf(bytes(root.unwrap(light.settings.buffer)))];

    expect(values.slice(0, 4)).toEqual([0, 2, 0, 2]);
    expect(values.slice(4, 8)).toEqual([...Float32Array.of(1, 0.5, 0, 0.5)]);
    expect(values.slice(8, 12)).toEqual([...Float32Array.of(0.1, 0.2, 0.3, 0.05)]);
    expect(values.slice(12, 16)).toEqual([...Float32Array.of(0.06, 0.05, 0.045, 0.25)]);
    expect(values.slice(16, 20)).toEqual([...Float32Array.of(0.55, 0.62, 0.7, 3)]);

    const plain = floatsOf(bytes(root.unwrap(lighting.create(root).settings.buffer)));
    expect([plain[11], plain[15], plain[19]]).toEqual([0, 0, 1]);
    expect(plain.every(Number.isFinite)).toBe(true);
  });

  it('loops over point lights only in a lighting that has them', () => {
    const { root, log } = recordingRoot();
    shadeOnce(root, lighting.create(root));
    shadeOnce(root, lighting.create(root, { points: { table: pointsIn(root) } }));

    const [plain, withPoints] = log.shaders;
    expect(plain).not.toContain('for (');
    expect(withPoints).toContain('for (');
  });

  it('binds a uniform for the sun, sky and fog, two storage buffers and a uniform for point lights, a third storage buffer for their positions on the device, and a uniform for a shadow', () => {
    const { root, log } = recordingRoot();
    const moving = table.create(root, lighting.MovingPointLight, { capacity: 16 });
    shadeOnce(root, lighting.create(root));
    shadeOnce(root, lighting.create(root, { points: { table: pointsIn(root) } }));
    shadeOnce(root, lighting.create(root, { points: { table: moving } }));
    shadeOnce(root, lighting.create(root, { shadow: {} }));

    // Each kernel also binds its output, a storage buffer, and the camera, a uniform.
    const bound = log.shaders.map((wgsl) => [
      wgsl.match(/var<storage/g)!.length - 1,
      wgsl.match(/var<uniform/g)!.length - 1,
    ]);
    expect(bound).toEqual([
      [0, 1],
      [2, 2],
      [3, 2],
      [0, 2],
    ]);
  });

  it("reads a light's colour and intensity only once its position and range say it reaches", () => {
    const { root, log } = recordingRoot();
    shadeOnce(root, lighting.create(root, { points: { table: pointsIn(root) } }));

    const wgsl = log.shaders[0];
    expect(wgsl).toMatch(/fn lightReach\(.*\n.*PointLight_position\(\w+\), PointLight_range/);
    const loop = wgsl.slice(wgsl.indexOf('for (var r'));
    const test = loop.indexOf('if (');
    expect(loop.indexOf('lightReach(')).toBeLessThan(test);
    expect(loop.indexOf('PointLight_alive(')).toBeGreaterThan(test);
    expect(loop.indexOf('lightGlow(')).toBeGreaterThan(loop.indexOf('PointLight_alive('));
    expect(wgsl).toMatch(/position: vec3f,\s*range: f32,\s*color: vec3f,\s*intensity: f32/);
  });

  it('visits the slots its candidates name, binding the two storage buffers a grid reads', () => {
    const { root, log } = recordingRoot();
    const { candidates, light: lampAt } = lampsInCells(root, 3);
    shadeOnce(root, lighting.create(root, { points: { table: pointsIn(root) } }));
    shadeOnce(root, lighting.create(root, { points: { table: pointsIn(root), candidates } }));
    shadeOnce(root, lighting.create(root, { points: { light: lampAt, candidates } }));

    const [all, fromTable, fromFunction] = log.shaders;
    expect(all).toContain('fn allSlots(');
    for (const wgsl of [fromTable!, fromFunction!]) {
      expect(wgsl).not.toContain('fn allSlots(');
      expect(wgsl).toContain(`r < ${grid.RANGES}`);
      expect(wgsl).toContain('fn slotAt(');
    }

    // Beside the kernel's output: the table's two or the lamps' rows, and the grid's two.
    const storage = log.shaders.map((wgsl) => wgsl.match(/var<storage/g)!.length - 1);
    expect(storage).toEqual([2, 4, 3]);
  });

  it("takes cones from the table's definition, in 64-byte records read only once a light is in range, binding nothing new", () => {
    const { root, log } = recordingRoot();
    shadeOnce(root, lighting.create(root, { points: { table: pointsIn(root) } }));
    shadeOnce(root, lighting.create(root, { points: { table: spotsIn(root) } }));

    expect(d.sizeOf(lighting.Spot)).toBe(64);
    const [plain, coned] = log.shaders;
    expect(plain).not.toContain('SpotLight_aim');
    expect(plain).not.toContain('fn beam(');
    const loop = coned!.slice(coned!.indexOf('for (var r'));
    expect(loop.indexOf('reaching(')).toBeGreaterThan(loop.indexOf('if ('));
    expect(coned).toMatch(/fn \w+\([^)]*\) -> f32 \{[^}]*SpotLight_aim\(/);
    const storage = log.shaders.map((wgsl) => wgsl.match(/var<storage/g)!.length);
    expect(storage[0]).toBe(storage[1]);
  });

  it('reads point lights from exactly one source, a function counted or given candidates, and types a spot where it has a cone', () => {
    const { root } = recordingRoot();
    const lamps = pointsIn(root, 4);
    const { candidates } = lampsInCells(root, 3);
    const loose = lighting.create as (root: TgpuRoot, options: object) => unknown;
    expect(() => loose(root, { points: {} })).toThrow(/one of `points.table`/);
    expect(() => loose(root, { points: { table: lamps, light: lightOf, count: 1 } })).toThrow(
      /one of `points.table`/,
    );
    expect(() => loose(root, { points: { light: lightOf } })).toThrow(/`points.count` or/);
    expect(() => loose(root, { points: { table: lamps, count: 4 } })).toThrow(/goes with a light/);
    expect(() => loose(root, { points: { light: lightOf, count: 1.5 } })).toThrow(RangeError);

    void (() => {
      lighting.create(root, { points: { light: lightOf, count: 4 } });
      lighting.create(root, { points: { spot: spotOf, candidates } });
      // @ts-expect-error a spot function returns a spot, with an aim and a cone
      lighting.create(root, { points: { spot: lightOf, count: 4 } });
      // @ts-expect-error a table's live slots set the loop, so it takes no count
      lighting.create(root, { points: { table: lamps, count: 4 } });
      // @ts-expect-error one source of lights
      lighting.create(root, { points: { light: lightOf, spot: spotOf, count: 4 } });
      // @ts-expect-error a lighting reads lights from a table you own, so it has none of its own
      lighting.create(root, { points: { table: lamps } }).points.add();
    });
  });

  it('gives tables white lights of range 10, aimed down where they have cones', () => {
    const { root } = recordingRoot();
    const lamps = table.create(root, lighting.PointLight, {
      capacity: 4,
      defaults: { ...lighting.WHITE, range: 8 },
    });
    const lamp = lamps.add();
    expect(lamps.fields.range.get(lamp)).toBe(8);
    expect(lamps.fields.intensity.get(lamp)).toBe(1);
    expect([...lamps.fields.color.get(lamp, d.vec3f())]).toEqual([1, 1, 1]);

    const spots = table.create(root, lighting.MovingSpotLight, {
      capacity: 4,
      defaults: lighting.WHITE_SPOT,
    });
    expect([...spots.fields.aim.get(spots.add(), d.vec3f())]).toEqual([0, -1, 0]);
  });

  it('rejects candidates without a whole number of ranges, and types a malformed one as an error', () => {
    const { root } = recordingRoot();
    const { candidates } = lampsInCells(root, 3);
    const lamps = pointsIn(root, 4);
    expect(() =>
      lighting.create(root, { points: { table: lamps, candidates: { ...candidates, ranges: 0 } } }),
    ).toThrow(RangeError);
    expect(() =>
      lighting.create(root, {
        points: { table: lamps, candidates: { ...candidates, ranges: 1.5 } },
      }),
    ).toThrow(RangeError);

    void (() => {
      const ranged = { ...candidates, range: vector };
      // @ts-expect-error a range is a first index and a count
      lighting.create(root, { points: { table: lamps, candidates: ranged } });
      // @ts-expect-error a candidate names a slot
      lighting.create(root, { points: { table: lamps, candidates: { ...candidates, slot: yes } } });
      const { ranges: _, ...uncounted } = candidates;
      // @ts-expect-error the count of ranges is known when shaders compile
      lighting.create(root, { points: { table: lamps, candidates: uncounted } });
    });
  });

  it('adds and removes point lights without a new pipeline', () => {
    const { root, log } = recordingRoot();
    const lamps = pointsIn(root, 4);
    shadeOnce(root, lighting.create(root, { points: { table: lamps } }));
    const pipelines = log.pipelines;

    const loop = frames.create(root, () => lamps.flush());
    const lamp = lamps.add({ position: [0, 2, 0], color: [1, 0.6, 0.3], range: 6 });
    loop.run(0);
    lamps.remove(lamp);
    loop.run(1);

    expect(lamps.count).toBe(0);
    expect(log.pipelines).toBe(pipelines);
  });

  it('samples a shadow only in a lighting that has one, as many times as its filter says', () => {
    const { root, log } = recordingRoot();
    const light = lighting.create(root, { shadow: { filter: 9 } });
    shadeOnce(root, lighting.create(root));
    shadeOnce(root, light);

    const [plain, shadowed] = log.shaders;
    expect(plain).not.toContain('textureSampleCompareLevel');
    expect(shadowed!.split('textureSampleCompareLevel')).toHaveLength(10);
    expect(lighting.create(root).shadow).toBeUndefined();
    expect(light.shadow.size).toBe(2048);
  });

  it('aims at the sun it starts with, again when aimed, and not at a zero direction', () => {
    const { root, log, bytes } = recordingRoot();
    const light = lighting.create(root, { settings: { sunDirection: [1, 2, 0.5] }, shadow: {} });
    const view = () => [...bytes(log.writes.at(-1)!.buffer)];
    const first = view();

    light.shadow.aim([0, 1, 0]);
    expect(view()).not.toEqual(first);

    light.shadow.aim([2, 4, 1]);
    expect(view()).toEqual(first);

    const writes = log.writes.length;
    light.shadow.aim([0, 0, 0]);
    expect(log.writes).toHaveLength(writes);
  });

  it("aims at its next record when the settings change the sun's direction, and only then", () => {
    const { root, log, bytes } = recordingRoot();
    const light = lighting.create(root, { settings: { sunDirection: [1, 2, 0.5] }, shadow: {} });
    const loop = frames.create(root, (frame) => light.shadow.record(frame, []));
    const view = () => [...bytes(log.writes.findLast(({ size }) => size === SUN)!.buffer)];
    const first = view();

    light.settings.patch({ sunDirection: [0, 1, 0] });
    expect(view()).toEqual(first);
    loop.run(0);
    expect(view()).not.toEqual(first);
    light.settings.write({ ...lighting.defaults, sunDirection: [2, 4, 1] });
    loop.run(1);
    expect(view()).toEqual(first);

    // A sun aimed by hand stays until the settings name a new direction
    light.shadow.aim([0, 1, 0]);
    const writes = log.writes.length;
    light.settings.patch({ fogDensity: 0.1 });
    loop.run(2);
    expect(log.writes).toHaveLength(writes + 1);
    expect(view()).not.toEqual(first);
  });

  it('looks from the sun at its center, then at the point it follows, on whole texels', () => {
    const { root, log, bytes } = recordingRoot();
    const light = lighting.create(root, {
      settings: { sunDirection: [1, 2, 0.5] },
      shadow: { size: 100, extent: 10, center: [4, 1, -2] },
    });
    // The caster's view, beside the shading's light space and cascade
    const cameraSize = d.sizeOf(Camera);
    const view = () =>
      floatsOf(bytes(log.writes.findLast(({ size }) => size === cameraSize)!.buffer)).subarray(
        0,
        16,
      );

    for (const point of [
      [4, 1, -2],
      [0, 0, 0],
      [0.53, 2, -0.26],
      [-7.1, 0.4, 3.33],
    ] as const) {
      if (point[0] !== 4) light.shadow.follow(point);
      const [x, y, depth] = transform(view(), point);

      // Within a texel of the middle, at mid depth, and with the world's origin on a texel's edge.
      expect(Math.abs(x!)).toBeLessThan(2 / 100);
      expect(Math.abs(y!)).toBeLessThan(2 / 100);
      expect(depth).toBeCloseTo(0.5, 5);
      for (const clip of transform(view(), [0, 0, 0]).slice(0, 2)) {
        const texels = ((clip + 1) * 100) / 2;
        expect(Math.abs(texels - Math.round(texels))).toBeLessThan(1e-3);
      }
    }
  });

  it('resizes its map in place once an app owns it, without new pipelines', () => {
    const { root, log, bytes } = recordingRoot();
    const light = lighting.create(root, { shadow: { size: 256, extent: 10 } });
    const shared = bindings.share(root, { shadow: light.shadow.part });
    shadeOnce(root, light);
    const compiled = log.pipelines;
    const group = shared.group();

    light.shadow.resize(1024);

    expect(light.shadow.size).toBe(1024);
    expect(log.pipelines).toBe(compiled);
    expect(shared.group()).not.toBe(group);
    // The shading's cascade holds the new texel width, 10 world units over 1024 texels.
    const sun = floatsOf(bytes(log.writes.at(-1)!.buffer));
    expect(sun[19]).toBeCloseTo(10 / 1024, 7);
    expect(() => light.shadow.resize(0.5)).toThrow(RangeError);
  });

  it('compiles the casters of its shadow from initAsync, so a frame waits for none', async () => {
    const { root, log } = recordingRoot();
    const light = lighting.create(root, { shadow: { size: 256 } });
    const Corner = d.struct({ position: d.vec4f });
    const cast = light.shadow.add({
      Vertex: Corner,
      at: (_vertex: number, _instance: number) => {
        'use gpu';
        return Corner({ position: cameras.current.$.viewProjection * d.vec4f(0, 0, 0, 1) });
      },
      draw: (pipeline: { draw(count: number): void }) => pipeline.draw(3),
    });

    await light.initAsync();
    frames.create(root, (frame) => light.shadow.record(frame, cast)).run(0);

    expect(log.pipelines).toBeGreaterThan(0);
    expect(log.blocking).toBe(0);
  });

  it('renders the map through one view, made again only when it resizes', () => {
    const { root, log } = recordingRoot();
    let views = 0;

    const createTexture = root.device.createTexture.bind(root.device);
    root.device.createTexture = (descriptor) => {
      const texture = createTexture(descriptor);
      const createView = texture.createView.bind(texture);
      texture.createView = (view) => {
        views += 1;
        return createView(view);
      };
      return texture;
    };

    const light = lighting.create(root, { shadow: { size: 256 } });
    bindings.share(root, { shadow: light.shadow.part });
    const loop = frames.create(root, (frame) => light.shadow.record(frame.encoder, []));

    loop.run(0);
    const made = views;
    loop.run(1);
    loop.run(2);
    expect(views).toBe(made);

    light.shadow.resize(512);
    const resized = views;
    loop.run(3);
    loop.run(4);
    expect(views).toBe(resized);
    expect(log.events.filter((event) => event === 'begin-render')).toHaveLength(5);
  });

  it('refuses to resize a map that pipelines hold as a fixed binding', () => {
    const { root } = recordingRoot();
    const light = lighting.create(root, { shadow: { size: 256 } });
    shadeOnce(root, light);

    expect(() => light.shadow.resize(512)).toThrow(/needs its map shared/);
    expect(light.shadow.size).toBe(256);
  });

  it('turns the shadow off and on with no new textures or pipelines, and no depth pass while off', () => {
    const { root, log } = recordingRoot();
    let on = true;
    const light = lighting.create(root, { shadow: { size: 256, enabled: () => on } });
    shadeOnce(root, light);
    const compiled = log.pipelines;
    const { textures } = log;
    const loop = frames.create(root, (frame) => light.shadow.record(frame.encoder, []));

    const writes = log.writes.length;
    on = false;
    loop.run(0);
    loop.run(1);

    expect(light.shadow.enabled).toBe(false);
    expect(log.events).not.toContain('begin-render');

    on = true;
    loop.run(2);

    expect(light.shadow.enabled).toBe(true);
    expect(log.events.filter((event) => event === 'begin-render')).toHaveLength(1);
    expect([log.pipelines, log.textures]).toEqual([compiled, textures]);

    // Each change writes the shading's light space, which covers nothing while it is off, and
    // nothing else.
    const sun = log.writes[writes].size;
    expect(sun).toBe(64 + 4 * 32);
    expect(log.writes.slice(writes).map(({ size }) => size)).toEqual([sun, sun]);
  });

  describe('cascades', () => {
    it('draws a pass into each cascade through the pipelines a single map builds', () => {
      const { root, log } = recordingRoot();
      const Corner = d.struct({ position: d.vec4f });
      const caster = {
        Vertex: Corner,
        at: (_vertex: number, _instance: number) => {
          'use gpu';
          return Corner({ position: cameras.current.$.viewProjection * d.vec4f(0, 0, 0, 1) });
        },
        draw: (pipeline: { draw(count: number): void }) => pipeline.draw(3),
      };
      const built = (light: lighting.Shadowed) => {
        const from = log.pipelines;
        const cast = light.shadow.add(caster);
        const loop = frames.create(root, (frame) => light.shadow.record(frame, cast));
        loop.run(0);
        log.events.length = 0;
        loop.run(1);
        return log.pipelines - from;
      };

      const single = built(lighting.create(root, { shadow: {} }));
      expect(log.events.filter((event) => event === 'begin-render')).toHaveLength(1);
      const { light } = fitted(root);
      expect(built(light)).toBe(single);
      expect(light.shadow.cascades).toBe(3);
      expect(light.shadow.layers).toBe(3);
      expect(log.events.filter((event) => event === 'begin-render')).toHaveLength(3);

      shadeOnce(root, light);
      const wgsl = log.shaders.at(-1)!;
      expect(wgsl).toContain('texture_depth_2d_array');
      expect(wgsl.split('textureSampleCompareLevel')).toHaveLength(5);
    });

    it('opens a timer on the first cascade, closes it on the last, and times an empty pass when none draws', () => {
      const { root } = recordingRoot();
      const { light } = fitted(root);
      const querySet = root.device.createQuerySet({ type: 'timestamp', count: 2 });
      const timestampWrites = { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 };
      const begun: (GPURenderPassDescriptor | GPUComputePassDescriptor | undefined)[] = [];
      const frame = (redraw: boolean) => {
        const encoder = root.createCommandEncoder();
        const raw = root.unwrap(encoder);
        const [render, compute] = [raw.beginRenderPass.bind(raw), raw.beginComputePass.bind(raw)];
        raw.beginRenderPass = (descriptor) => (begun.push({ ...descriptor }), render(descriptor));
        raw.beginComputePass = (descriptor) => (begun.push({ ...descriptor }), compute(descriptor));
        light.shadow.record(encoder, [], { timestampWrites, redraw: () => redraw });
        encoder.submit();
      };

      frame(true);
      frame(false);
      expect(begun.map((descriptor) => descriptor?.timestampWrites)).toEqual([
        { querySet, beginningOfPassWriteIndex: 0 },
        undefined,
        { querySet, endOfPassWriteIndex: 1 },
        timestampWrites,
      ]);
    });

    it('writes nothing for a still eye, fits again when it moves, and keeps each cascade the size it was as it turns', () => {
      const { root, log, bytes } = recordingRoot();
      const { eye, light } = fitted(root);
      const loop = frames.create(root, (frame) => light.shadow.record(frame, []));
      const sun = () => floatsOf(bytes(log.writes.findLast(({ size }) => size === SUN)!.buffer));
      loop.run(0);
      const first = cascadesOf(sun());

      let from = log.writes.length;
      loop.run(1);
      loop.run(2);
      expect(log.writes.length).toBe(from);

      eye.goal.yaw += 0.7;
      eye.jump();
      from = log.writes.length;
      loop.run(3);
      const sizes = log.writes.slice(from).map(({ size }) => size);
      expect(sizes.filter((size) => size === SUN)).toHaveLength(1);
      expect(sizes.filter((size) => size === d.sizeOf(Camera))).toHaveLength(3);

      const turned = cascadesOf(sun());
      for (let k = 0; k < 3; k++) {
        expect(turned[k].texel).toBe(first[k].texel);
        expect(turned[k].x).not.toBe(first[k].x);
      }
      // Each cascade is wider than the one before it, and the fourth is unused.
      expect(first[0].texel).toBeLessThan(first[1].texel);
      expect(first[1].texel).toBeLessThan(first[2].texel);
    });

    it('changes how far the cascades reach without creating anything', () => {
      const { root, log, bytes } = recordingRoot();
      const { light } = fitted(root);
      const loop = frames.create(root, (frame) => light.shadow.record(frame, []));
      loop.run(0);
      const counts = log.counts();
      const sun = () => floatsOf(bytes(log.writes.findLast(({ size }) => size === SUN)!.buffer));
      const before = cascadesOf(sun())[2].texel;

      light.shadow.setFar(240);
      loop.run(1);
      expect(light.shadow.far).toBe(240);
      expect(light.shadow.reach).toBe(240);
      expect(cascadesOf(sun())[2].texel).toBeGreaterThan(before * 1.5);
      expect(log.counts()).toEqual(counts);
    });

    it('splits where the options say, and rejects what needs an eye without one', () => {
      const { root, bytes, log } = recordingRoot();
      const even = fitted(root, { splits: [1 / 3, 2 / 3] });
      frames.create(root, (frame) => even.light.shadow.record(frame, [])).run(0);
      const sun = floatsOf(bytes(log.writes.findLast(({ size }) => size === SUN)!.buffer));
      const [a, b, c] = cascadesOf(sun);
      // Even slices of a widening view: each wider than the last, by less than the default's.
      expect(b!.texel / a!.texel).toBeLessThan(3);
      expect(c!.texel / b!.texel).toBeLessThan(2);

      const eye = cameras.orbit(root, {
        size: { width: 1, height: 1 },
        target: [0, 0, 0],
        distance: 4,
      });
      expect(() => fitted(root, { splits: [0.5] })).toThrow(RangeError);
      expect(() => fitted(root, { splits: [0.6, 0.2] })).toThrow(RangeError);
      expect(() => fitted(root, { cascades: 5 as 4 })).toThrow(RangeError);
      // JavaScript callers get past the types, so `create` checks the same combinations
      const loose = lighting.create as (root: TgpuRoot, options: object) => unknown;
      expect(() => loose(root, { shadow: { cascades: 3 } })).toThrow(/Give it an `eye`/);
      expect(() => loose(root, { shadow: { fit: lighting.fitSlices } })).toThrow(
        /Give it an `eye`/,
      );
      expect(() => loose(root, { shadow: { eye, extent: 10 } })).toThrow(/no `extent`/);
    });

    it('type a fixed map and a fitted shadow apart, in their options and their methods', () => {
      const { root } = recordingRoot();
      const eye = cameras.custom(root);
      void (() => {
        // @ts-expect-error cascades fit a shadow to an eye
        lighting.create(root, { shadow: { cascades: 3 } });
        // @ts-expect-error `far` is how far from an eye the cascades reach
        lighting.create(root, { shadow: { far: 80 } });
        // @ts-expect-error splits divide the cascades of an eye
        lighting.create(root, { shadow: { splits: [0.5] } });
        // @ts-expect-error a fit places the cascades of an eye
        lighting.create(root, { shadow: { fit: lighting.fitSlices } });
        // @ts-expect-error a shadow fitted to an eye covers no fixed square
        lighting.create(root, { shadow: { eye, extent: 10 } });
        // @ts-expect-error cascades fit themselves to the eye, so nothing centres them
        lighting.create(root, { shadow: { eye, center: [0, 0, 0] } });
        // @ts-expect-error a fitted shadow needs its eye
        const eyeless: lighting.FittedShadowOptions = { cascades: 2 };

        const fixed = lighting.create(root, { shadow: { extent: 30, reach: 60 } });
        expectTypeOf(fixed.shadow).toExtend<lighting.Shadow>();
        // @ts-expect-error a fixed square has no far end
        fixed.shadow.setFar(10);

        const cascaded = lighting.create(root, { shadow: { eye, cascades: 2, far: 80 } });
        expectTypeOf(cascaded.shadow).toExtend<lighting.Cascades>();
        // @ts-expect-error cascades fit themselves to the eye, so nothing centres them
        cascaded.shadow.follow([0, 0, 0]);
        // @ts-expect-error cascades take their size from the view
        cascaded.shadow.setExtent(10);

        const options: lighting.ShadowOptions = { eye, cascades: 3 };
        expectTypeOf(lighting.create(root, { shadow: options }).shadow).toExtend<
          lighting.Shadow | lighting.Cascades
        >();
        // A shadow's passes open through `record`, and a depth map has no translucent layer.
        for (const shadow of [fixed.shadow, cascaded.shadow]) {
          expectTypeOf(shadow).not.toHaveProperty('begin');
          expectTypeOf(shadow).not.toHaveProperty('composite');
        }
        return eyeless;
      });
    });

    it('lights every point until the eye has a view to fit', () => {
      const { root, log, bytes } = recordingRoot();
      const eye = cameras.custom(root);
      const light = lighting.create(root, { shadow: { eye, cascades: 2 } });
      frames.create(root, (frame) => light.shadow.record(frame, [])).run(0);
      const sun = floatsOf(bytes(log.writes.findLast(({ size }) => size === SUN)!.buffer));
      expect([sun[16 + 6], sun[24 + 6]]).toEqual([2, 2]);
    });

    it('redraws only the cascades `redraw` picks, and any whose place moved or that come back from a frame without the shadow', () => {
      const { root, log } = recordingRoot();
      let on = true;
      const { eye, light } = fitted(root, { enabled: () => on });
      let picked = -1;
      const loop = frames.create(root, (frame) =>
        light.shadow.record(frame, [], { redraw: (cascade) => cascade === picked }),
      );
      const drawn = (t: number) => {
        log.events.length = 0;
        loop.run(t);
        return passes(log.events);
      };

      expect(drawn(0)).toBe(3); // nothing drawn yet
      expect(drawn(1)).toBe(0);
      picked = 2;
      expect(drawn(2)).toBe(1);
      picked = -1;
      eye.goal.yaw += 0.7;
      eye.jump();
      expect(drawn(3)).toBe(3); // the fit moved every cascade
      light.shadow.aim([0, 1, 0]);
      expect(drawn(4)).toBe(3);
      on = false;
      expect(drawn(5)).toBe(0);
      on = true;
      expect(drawn(6)).toBe(3); // casters may have moved while it was off
    });
  });

  describe('point shadows', () => {
    /** Bytes per shadow in the uniform: six view-projections, then slot, first layer, faces, texel and spread. */
    const SHADOW = 416;
    const Corner = d.struct({ position: d.vec4f });
    const caster = {
      Vertex: Corner,
      at: (_vertex: number, _instance: number) => {
        'use gpu';
        return Corner({ position: cameras.current.$.viewProjection * d.vec4f(0, 0, 0, 1) });
      },
      draw: (pipeline: { draw(count: number): void }) => pipeline.draw(3),
    };
    const market = (root: TgpuRoot, count = 2, enabled?: () => boolean) => {
      const lamps = spotsIn(root, 8);
      const light = lighting.create(root, {
        points: { table: lamps },
        pointShadow: { count, size: 256, enabled: enabled ?? (() => true) },
      });
      const fire = lamps.add({ position: [0, 1, 0], range: 10, cone: [0, 1] });
      const sign = lamps.add({
        position: [3, 3, 0],
        range: 8,
        aim: [0, -1, 0],
        cone: [...lighting.cone(0.3, 0.6)] as [number, number],
      });
      const wide = lamps.add({
        position: [-3, 3, 0],
        range: 8,
        cone: [...lighting.cone(0.9, 1.2)] as [number, number],
      });
      const cast = light.pointShadow.add(caster);
      const loop = frames.create(root, (frame) => {
        lamps.flush();
        light.pointShadow.record(frame, cast);
      });
      return { light, lamps, fire, sign, wide, cast, loop };
    };

    it('draws six faces for a point light or a wide spot and one layer for a narrow spot, through pipelines that do not grow with count', () => {
      const { root, log } = recordingRoot();
      const built = (count: number) => {
        const from = log.pipelines;
        const scene = market(root, count);
        scene.light.pointShadow.cast([scene.fire, scene.sign]);
        scene.loop.run(0);
        log.events.length = 0;
        scene.loop.run(1);
        return { scene, pipelines: log.pipelines - from };
      };

      const two = built(2);
      expect(passes(log.events)).toBe(7);
      expect(two.scene.light.pointShadow.layers).toBe(12);
      expect(built(4).pipelines).toBe(two.pipelines);

      two.scene.light.pointShadow.cast([two.scene.fire, two.scene.wide]);
      log.events.length = 0;
      two.scene.loop.run(2);
      expect(passes(log.events)).toBe(12);

      shadeOnce(root, two.scene.light);
      const wgsl = log.shaders.at(-1)!;
      expect(wgsl).toContain('texture_depth_2d_array');
      expect(wgsl).toContain('textureSampleCompareLevel');
      expect(wgsl).toContain('sampler_comparison');
    });

    it('redraws only the lights `redraw` picks, and any light given a shadow, moved or back from a frame without shadows', () => {
      const { root, log } = recordingRoot();
      let on = true;
      const { light, lamps, fire, sign, wide, cast } = market(root, 2, () => on);
      let picked: typeof fire | undefined;
      const loop = frames.create(root, (frame) => {
        lamps.flush();
        light.pointShadow.record(frame, cast, { redraw: (id) => id === picked });
      });
      const drawn = (t: number) => {
        log.events.length = 0;
        loop.run(t);
        return passes(log.events);
      };
      light.pointShadow.cast([fire, sign]);

      expect(drawn(0)).toBe(7); // both are new
      expect(drawn(1)).toBe(0);
      picked = sign;
      expect(drawn(2)).toBe(1);
      picked = undefined;
      lamps.fields.position.set(fire, 1, 1, 0);
      expect(drawn(3)).toBe(6); // the fire moved
      light.pointShadow.cast([fire, wide]);
      expect(drawn(4)).toBe(6); // the wide spot takes the sign's shadow
      on = false;
      expect(drawn(5)).toBe(0);
      on = true;
      expect(drawn(6)).toBe(12); // casters may have moved while it was off
    });

    it('drops the passes of a light left out and creates nothing', () => {
      const { root, log } = recordingRoot();
      const { light, fire, sign, loop } = market(root);
      light.pointShadow.cast([fire, sign]);
      loop.run(0);
      const counts = log.counts();

      light.pointShadow.cast([fire]);
      log.events.length = 0;
      loop.run(1);
      expect(passes(log.events)).toBe(6);
      expect(log.counts()).toEqual(counts);

      // A light listed twice casts once
      light.pointShadow.cast([fire, fire]);
      log.events.length = 0;
      loop.run(2);
      expect(passes(log.events)).toBe(6);

      light.pointShadow.cast([]);
      log.events.length = 0;
      loop.run(3);
      expect(passes(log.events)).toBe(0);
    });

    it('keeps the shadow of a light that still casts, so casting the same lights each frame writes and creates nothing', () => {
      const { root, log, bytes } = recordingRoot();
      const { light, lamps, fire, sign, wide, loop } = market(root);
      const size = 2 * SHADOW;
      const slots = () => {
        const words = new Uint32Array(
          bytes(log.writes.findLast((write) => write.size === size)!.buffer).slice().buffer,
        );
        return [words[96], words[SHADOW / 4 + 96]];
      };
      const lit = (id: typeof fire) => lamps.slotOf(id);
      const nearest = [fire, sign];
      light.pointShadow.cast(nearest);
      loop.run(0);
      expect(slots()).toEqual([lit(fire), lit(sign)]);

      const counts = log.counts();
      const from = log.writes.length;
      for (let i = 1; i < 4; i++) {
        light.pointShadow.cast(nearest);
        loop.run(i);
      }
      expect(log.writes.length).toBe(from);
      expect(log.counts()).toEqual(counts);

      // The sign gives way to the wide spot, and the fire keeps shadow 0
      light.pointShadow.cast([wide, fire]);
      loop.run(4);
      expect(slots()).toEqual([lit(fire), lit(wide)]);
    });

    it("places each shadow at its light's row, writes nothing while lights hold still, and frees a removed light's shadow", () => {
      const { root, log, bytes } = recordingRoot();
      const { light, lamps, fire, loop } = market(root);
      const size = 2 * SHADOW;
      const last = () => bytes(log.writes.findLast((write) => write.size === size)!.buffer);
      const words = () => new Uint32Array(last().slice().buffer);

      loop.run(0);
      expect([words()[96], words()[SHADOW / 4 + 96]]).toEqual([0xffffffff, 0xffffffff]);

      light.pointShadow.cast([fire]);
      loop.run(1);
      const placed = words();
      expect(Array.from(placed.subarray(96, 99))).toEqual([0, 0, 6]);
      // The +x face sees a point 2 m along x from the fire in the middle of its view
      const face = new Float32Array(last().slice().buffer, 0, 16);
      const clip = [0, 1, 2, 3].map((row) => face[row] * 2 + face[4 + row] + face[12 + row]);
      expect(clip[0] / clip[3]).toBeCloseTo(0);
      expect(clip[1] / clip[3]).toBeCloseTo(0);

      let from = log.writes.length;
      loop.run(2);
      expect(log.writes.filter((write) => write.size === size).length).toBe(
        log.writes.slice(0, from).filter((write) => write.size === size).length,
      );

      lamps.fields.position.set(fire, 1, 1, 0);
      from = log.writes.length;
      loop.run(3);
      expect(log.writes.slice(from).filter((write) => write.size === size)).toHaveLength(1);

      lamps.remove(fire);
      log.events.length = 0;
      loop.run(4);
      expect(words()[96]).toBe(0xffffffff);
      expect(passes(log.events)).toBe(0);
    });

    it('casts only from lights the CPU places', () => {
      const { root } = recordingRoot();
      const movers = table.create(root, lighting.MovingPointLight, { capacity: 4 });
      expect(() => lighting.create(root, { points: { table: movers }, pointShadow: {} })).toThrow(
        /move on the device/,
      );
      expect(() =>
        lighting.create(root, { points: { light: lightOf, count: 1 }, pointShadow: {} }),
      ).toThrow(/Give it `points.table`/);

      const { light, lamps, fire, sign, wide } = market(root);
      expect(() => light.pointShadow.cast([fire, sign, wide])).toThrow(
        "The lighting has 2 point shadows, so 3 lights can't cast at once.",
      );
      lamps.remove(fire);
      expect(() => light.pointShadow.cast([sign, fire])).toThrow();
      light.pointShadow.cast([sign]);
      expect(() => lighting.create(root, { pointShadow: {} })).toThrow(/Give it `points.table`/);
      expect(() =>
        lighting.create(root, { points: { table: lamps }, pointShadow: { count: 0 } }),
      ).toThrow(RangeError);

      void (() => {
        // @ts-expect-error a lighting without point shadows has none to cast
        lighting.create(root, { points: { table: lamps } }).pointShadow.cast([]);
      });
    });

    it('frees the map and its views on destroy', () => {
      const { root, log } = recordingRoot();
      const { light, lamps, fire, loop } = market(root);
      light.pointShadow.cast([fire]);
      loop.run(0);
      light.destroy();
      lamps.destroy();
      expect(log.alive({ uniforms: false })).toEqual([]);
    });
  });

  it('rejects a shadow filter other than 1, 4 or 9', () => {
    const { root } = recordingRoot();
    expect(() => lighting.create(root, { shadow: { filter: 2 as 4 } })).toThrow(RangeError);
  });

  it('types an option it does not have as an error, and has a shadow only when given one', () => {
    const { root } = recordingRoot();

    // @ts-expect-error the fog thins by fogFalloff
    lighting.create(root, { settings: { fogDensity: 0.1, fogHeight: 4 } });
    // @ts-expect-error the sun is a setting
    lighting.create(root, { sun: { intensity: 2 } });
    // @ts-expect-error whether a shadow draws is read each frame
    void (() => lighting.create(root, { shadow: { size: 512, enabled: false } }));

    const shadowed = (on: boolean) => lighting.create(root, on ? { shadow: { size: 512 } } : {});
    // @ts-expect-error a shadow given only sometimes may be absent
    shadowed(true).shadow.aim([0, 1, 0]);

    // A hook alone still types its parameters from Options.
    lighting.create(root, {
      shading: { ambient: (material, point) => shading.hemisphere(material, point) },
    });

    const lit = lighting.create(root, { shadow: { size: 512 } });
    expect(lit.shadow.size).toBe(512);

    // The named type takes one map or cascades.
    const named: lighting.Shadowed = lighting.create(root, { shadow: { size: 256 } });
    expect(named.shadow.size).toBe(256);
  });

  it('puts the shading functions the options name into the shader, and the defaults otherwise', () => {
    const { root, log } = recordingRoot();
    shadeOnce(root, lighting.create(root, { shading: { reflect: shading.lambert } }));

    expect(log.shaders[0]).toContain('lambert');
    expect(log.shaders[0]).not.toContain('pbr');
    expect(log.shaders[0]).toContain('hemisphere');
    expect(log.shaders[0]).toContain('heightFog');
  });

  it('keeps the branch in lambert that lets the point-light loop skip lights out of range', () => {
    const { root, log } = recordingRoot();
    shadeOnce(root, lighting.create(root, { shading: { reflect: shading.lambert } }));

    expect(log.shaders[0]).toMatch(
      /fn lambert\([^)]*\) -> vec3f \{[^}]*if \(\(cosine <= 0f\)\) \{/,
    );
  });

  it('lets a pipeline choose its own function after the lighting, or before one that names none', () => {
    const { root, log } = recordingRoot();
    const named = lighting.create(root, { shading: { reflect: shading.pbr } });

    shadeOnce(root, named, { after: flat });
    shadeOnce(root, lighting.create(root), { before: flat });
    shadeOnce(root, named, { before: flat });

    const [after, beforeDefault, beforeNamed] = log.shaders;
    expect(after).toContain('lambert');
    expect(after).not.toContain('pbr');
    expect(beforeDefault).toContain('lambert');
    expect(beforeNamed).toContain('pbr');
    expect(beforeNamed).not.toContain('lambert');
  });

  it('leaves out the shadows of a lighting piped before one without them', () => {
    const { root, log } = recordingRoot();
    const lamps = pointsIn(root);
    const { light: sunlit } = fitted(root);
    const lamplit = lighting.create(root, { points: { table: lamps }, pointShadow: {} });
    const plain = lighting.create(root, { points: { table: lamps } });

    shadeOnce(root, plain);
    shadeOnce(root, plain, { before: (cfg) => cfg.pipe(sunlit.inject()).pipe(lamplit.inject()) });

    expect(log.shaders[1]).toBe(log.shaders[0]);
  });

  it('lets your own functions read the settings', () => {
    const { root, log } = recordingRoot();
    shadeOnce(root, lighting.create(root, { shading: { fog: hazy } }));

    expect(log.shaders[0]).toMatch(/fn settings\w*\(\) -> LightSettings\w*/);
  });
});

describe('shading on the CPU', () => {
  const up = d.vec3f(0, 1, 0);
  const zero = d.vec2f();
  const facing = shading.Point({ position: d.vec3f(), normal: up, view: up, uv: zero, uv1: zero });

  it('lights a white matte surface facing a light of radiance 1 to about 1', () => {
    const white = materials.matte(d.vec3f(1));

    expect(shading.lambert(white, facing, up, d.vec3f(1)).x).toBe(1);
    expect(shading.pbr(white, facing, up, d.vec3f(1)).x).toBeCloseTo(1, 1);
    expect(shading.pbr(white, facing, d.vec3f(0, -1, 0), d.vec3f(1)).x).toBe(0);
  });

  it('gives a smooth metal a highlight and no diffuse light', () => {
    const gold = materials.metal(d.vec3f(1, 0.8, 0.3), 0.2);
    const aside = d.vec3f(0.6, 0.8, 0);
    const mirror = shading.Point({
      position: d.vec3f(),
      normal: up,
      view: d.vec3f(-0.6, 0.8, 0),
      uv: d.vec2f(),
      uv1: d.vec2f(),
    });

    expect(shading.pbr(gold, mirror, aside, d.vec3f(1)).x).toBeGreaterThan(1);
    expect(shading.pbr(gold, facing, aside, d.vec3f(1)).x).toBeLessThan(0.1);
  });

  it('lights a metal in its colour under lambert, and nothing from behind', () => {
    const gold = materials.metal(d.vec3f(1, 0.8, 0.3), 0.2);

    expect([...shading.lambert(gold, facing, up, d.vec3f(1))]).toEqual([...gold.color]);
    expect(
      shading.lambert(materials.matte(d.vec3f(1)), facing, d.vec3f(0, -1, 0), d.vec3f(1)).x,
    ).toBe(0);
  });

  it('eases point lights to zero at their range', () => {
    expect(shading.falloff(1, 10)).toBeCloseTo(1, 3);
    expect(shading.falloff(2, 10)).toBeCloseTo(0.25, 2);
    expect(shading.falloff(10, 10)).toBe(0);
    expect(shading.falloff(12, 10)).toBe(0);
  });

  it('makes materials as plain values', () => {
    const lamp = materials.glow(d.vec3f(1, 0.5, 0), 4);

    expect([...lamp.emission]).toEqual([4, 2, 0]);
    expect([...lamp.color]).toEqual([0, 0, 0]);
    expect(materials.glossy(d.vec3f(1), 0.3).roughness).toBeCloseTo(0.3);
    expect(materials.metal(d.vec3f(1), 0.3).metallic).toBe(1);
  });
});
