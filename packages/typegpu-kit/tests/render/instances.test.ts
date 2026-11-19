import { cameras, frames, passes, targets, type CameraBase, type Frame } from '@typegpu/kit';
import { census } from '@typegpu/kit/debug';
import { recordingRoot } from '@typegpu/kit/testing';
import { meshes } from '@typegpu/kit/geometry';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { describe, expect, expectTypeOf, it, vi } from 'vitest';
import { instances, lighting, materials, meshPool, shading, taa } from '../../src/index.ts';

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

/** A range, measured from the camera the passes see through. */
type Range = Pick<instances.RendererOptions, 'from' | 'to' | 'fade'>;

/** A pool with a box and a plane, a camera and a lighting. */
function scene(root: TgpuRoot) {
  const pool = meshPool.create(root, { vertices: 100, indices: 200, capacity: 4 });
  const box = pool.add(meshes.box());
  const plane = pool.add(meshes.plane());
  const view = cameras.orbit(root, {
    size: { width: 1, height: 1 },
    target: [0, 0, 0],
    distance: 5,
  });

  return { pool, box, plane, view, light: lighting.create(root) };
}

/** The depth-only shader of a shadow's pass. */
const casting = (shaders: string[]) =>
  shaders.find((code) => code.includes('@vertex') && !code.includes('@fragment'));

/** The shadow's shader for a set of `kind`. */
async function castBy(kind: instances.AnyKind) {
  const { root, log } = recordingRoot({ features: FIRST_INSTANCE });
  const { pool, view } = scene(root);
  const light = lighting.create(root, { shadow: {} });
  const renderer = instances
    .create(root, pool, { capacity: 1024, kind })
    .renderer()
    .pipe(light.inject());
  const lit = hdr(root, view);
  lit.add(renderer);
  light.shadow.add(renderer);
  await Promise.all([lit.initAsync(), light.shadow.initAsync()]);

  return casting(log.shaders);
}

/** A deform that reads the uv, so the shadow's pass needs the whole vertex. */
const wave = (vertex: instances.Vertex) => {
  'use gpu';
  return vertex.position + d.vec3f(vertex.uv.x, 0, 0);
};

/** `v` turned by the unit quaternion `q`, on the CPU. */
function turn(q: d.v4f, [x, y, z]: readonly number[]) {
  const [cx, cy, cz] = [q.y * z! - q.z * y!, q.z * x! - q.x * z!, q.x * y! - q.y * x!];
  const twice = [cx * 2, cy * 2, cz * 2] as const;

  return [
    x! + twice[0] * q.w + (q.y * twice[2] - q.z * twice[1]),
    y! + twice[1] * q.w + (q.z * twice[0] - q.x * twice[2]),
    z! + twice[2] * q.w + (q.x * twice[1] - q.y * twice[0]),
  ];
}

/** A triangle of a model, of `color`. */
const part = (color: [number, number, number, number]) => ({
  positions: new Float32Array(9),
  normals: new Float32Array(9),
  uvs: new Float32Array(6),
  indices: new Uint32Array([0, 1, 2]),
  material: { color, roughness: 0.5, metallic: 0, emission: [0, 0, 0] as const, textures: {} },
});

const nothing = () => {};

const PASS_CALLS = [
  'setPipeline',
  'setBindGroup',
  'setIndexBuffer',
  'drawIndexed',
  'drawIndexedIndirect',
] as const;

/**
 * Records what each render pass begun from now on sets and draws, as WebGPU sees it: each call's
 * name and arguments, in order. Counts the bind groups and texture views made, too.
 */
function watch(root: TgpuRoot) {
  const device = root.device;
  const calls: unknown[][] = [];
  const made = { bindGroups: 0, views: 0 };

  const createEncoder = device.createCommandEncoder.bind(device);
  device.createCommandEncoder = (descriptor) => {
    const encoder = createEncoder(descriptor);
    const begin = encoder.beginRenderPass.bind(encoder);

    encoder.beginRenderPass = (passDescriptor) => {
      const pass = begin(passDescriptor);
      for (const name of PASS_CALLS) {
        const method = (pass[name] as (...args: unknown[]) => void).bind(pass);
        pass[name] = ((...args: unknown[]) => {
          calls.push([name, ...args]);
          method(...args);
        }) as never;
      }
      return pass;
    };
    return encoder;
  };

  const createBindGroup = device.createBindGroup.bind(device);
  device.createBindGroup = (descriptor) => {
    made.bindGroups += 1;
    return createBindGroup(descriptor);
  };

  const createTexture = device.createTexture.bind(device);
  device.createTexture = (descriptor) => {
    const texture = createTexture(descriptor);
    const createView = texture.createView.bind(texture);
    texture.createView = (view) => {
      made.views += 1;
      return createView(view);
    };
    return texture;
  };

  return { calls, made };
}

/** The shaders a culling renderer over `range` compiles, with a shadow in its lighting. */
async function shadersOver(range?: Range) {
  const { root, log } = recordingRoot({ features: FIRST_INSTANCE });
  const { pool, view } = scene(root);
  const light = lighting.create(root, { shadow: {} });
  const options = { cull: true, ...(range && { ...range, eye: view }) } as const;
  const renderer = instances
    .create(root, pool, { capacity: 1024 })
    .renderer(options)
    .pipe(light.inject());
  const into = hdr(root, view);
  into.add(renderer);
  light.shadow.add(renderer);
  await Promise.all([into.initAsync(), light.shadow.initAsync()]);
  return log.shaders;
}

const firstFragment = (shaders: string[]) => shaders.find((code) => code.includes('@fragment'))!;

/** The bind groups that the calls `watch` recorded set. */
const groupsIn = (calls: unknown[][]) =>
  calls.filter(([name]) => name === 'setBindGroup').map(([, , group]) => group);

/** The byte offset of a mesh's draw record in layer `k`, in a pool of 4 meshes. */
const recordOf = (k: number, mesh: { index: number }) => (k * 4 + mesh.index) * 5 * 4;

/** Runs a frame of whatever it is given, and returns the events that frame recorded. */
function framesOf(root: TgpuRoot, events: string[]) {
  let record: (frame: Frame) => void = nothing;
  const loop = frames.create(root, (frame) => record(frame));
  let time = 0;

  return (next: (frame: Frame) => void) => {
    record = next;
    const before = events.length;
    loop.run(time++);

    return events.slice(before);
  };
}

describe('instances', () => {
  it('writes a row per instance, starting from the identity and white chalk', () => {
    const { root, log, bytes } = recordingRoot();
    const { pool, box } = scene(root);
    const things = instances.create(root, pool, { capacity: 4 });

    things.add(box);
    const lifted: number[] = [1, 2, 3];
    things.add(box, {
      position: lifted,
      rotation: Float32Array.of(0, 0, 0, 1),
      scale: 2,
      ...materials.metal(d.vec3f(1, 0.5, 0), 0.25),
    });
    frames.create(root, () => things.flush()).run(0);

    const rows = log.writes.find((write) => write.size === 2 * 80)!;
    const floats = new Float32Array(bytes(rows.buffer).buffer, 0, 40);
    const words = new Uint32Array(floats.buffer, 0, 40);

    // position, mesh, rotation, scale, roughness, color, metallic, emission, opacity
    expect([...floats.subarray(0, 3), words[3], ...floats.subarray(4, 20)]).toEqual([
      0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1,
    ]);
    expect([...floats.subarray(20, 23), ...floats.subarray(28, 40)]).toEqual([
      1, 2, 3, 2, 2, 2, 0.25, 1, 0.5, 0, 1, 0, 0, 0, 1,
    ]);
  });

  it('changes only the fields given', () => {
    const { root } = recordingRoot();
    const { pool, box } = scene(root);
    const things = instances.create(root, pool, { capacity: 4 });

    const id = things.add(box, { position: [1, 2, 3], color: [1, 0, 0] });
    things.write(id, { scale: [1, 2, 3], opacity: 0.5 });
    expect(things.count).toBe(1);

    things.remove(id);
    expect(things.has(id)).toBe(false);
    expect(() => things.write(id, { scale: 1 })).toThrow('stale');
  });

  it('lists the live instances only in frames after some come or go, with no pass', () => {
    const { root, log } = recordingRoot();
    const { pool, box } = scene(root);
    const things = instances.create(root, pool, { capacity: 4 });
    const frame = framesOf(root, log.events);
    const listed = () => {
      const before = log.writes.length;
      const events = frame(() => things.flush());
      expect(events).toEqual(['submit']);

      return log.writes.slice(before).some((write) => write.buffer.label.endsWith('_listing'));
    };

    expect(listed()).toBe(false);

    const id = things.add(box);
    expect(listed()).toBe(true);

    things.write(id, { position: [0, 1, 0] });
    expect(listed()).toBe(false);

    things.remove(id);
    expect(listed()).toBe(true);
  });

  it('draws each mesh with instances directly, through one pipeline, with no device feature', () => {
    const { root, log } = recordingRoot();
    const { pool, box, plane, view, light } = scene(root);
    const things = instances.create(root, pool, { capacity: 16 });
    const drawn = hdr(root, view).add(things.renderer().pipe(light.inject()));
    const before = log.pipelines;
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    for (let i = 0; i < 3; i++) things.add(box, { position: [i, 0, 0] });
    const flat = things.add(plane);

    const { calls } = watch(root);
    const frame = framesOf(root, log.events);
    const record = () => {
      const from = calls.length;
      frame((each) => {
        things.flush();
        const pass = each.encoder.beginRenderPass({
          colorAttachments: { view: {} as GPUTextureView },
        });
        drawn.draw(pass);
        pass.end();
      });
      return calls.slice(from).filter(([name]) => String(name).startsWith('draw'));
    };

    // Each mesh's rows follow the pool's 4 starts, the box's first.
    expect(record()).toEqual([
      ['drawIndexed', box.indexCount, 3, box.firstIndex, box.baseVertex, 4],
      ['drawIndexed', plane.indexCount, 1, plane.firstIndex, plane.baseVertex, 7],
    ]);
    expect(log.pipelines - before).toBe(1);

    things.remove(flat);
    expect(record()).toEqual([
      ['drawIndexed', box.indexCount, 3, box.firstIndex, box.baseVertex, 4],
    ]);
    expect(warn).not.toHaveBeenCalled();
    warn.mockRestore();
  });

  it('draws in a bundle the meshes that had instances when it was recorded', () => {
    const { root, log } = recordingRoot({ features: FIRST_INSTANCE });
    const { pool, box, plane, view, light } = scene(root);
    const things = instances.create(root, pool, { capacity: 16 });
    const drawn = hdr(root, view).add(things.renderer().pipe(light.inject()));
    const draws = (bundle: GPURenderBundle) => {
      const encoder = root.createCommandEncoder();
      const pass = encoder.beginRenderPass({ colorAttachments: { view: {} as GPUTextureView } });
      const from = log.events.length;
      pass.executeBundles([bundle]);
      pass.end();
      encoder.submit();
      return log.events.slice(from).filter((event) => event.startsWith('draw')).length;
    };
    const record = () => {
      const bundler = root['~unstable'].createRenderBundleEncoder({
        colorFormats: ['rgba16float'],
        depthStencilFormat: 'depth24plus',
      });
      drawn.draw(bundler);
      return bundler.finish();
    };

    things.add(box);
    const early = record();
    things.add(plane);
    things.flush();

    expect([draws(early), draws(record())]).toEqual([1, 2]);
  });

  it('sets the pipeline, its groups and the indices once for a run of meshes that share them', () => {
    const { root, log } = recordingRoot({ features: FIRST_INSTANCE });
    const { pool, box, plane, view, light } = scene(root);
    const things = instances.create(root, pool, { capacity: 1024 });
    const drawn = hdr(root, view).add(things.renderer().pipe(light.inject()));
    things.add(box);
    things.add(plane);

    const { calls } = watch(root);
    framesOf(
      root,
      log.events,
    )((frame) => {
      things.flush();

      const pass = frame.encoder.beginRenderPass({
        colorAttachments: { view: {} as GPUTextureView },
      });
      drawn.draw(pass);
      pass.end();
    });

    const counts = Object.fromEntries(PASS_CALLS.map((name) => [name, 0]));
    for (const [name] of calls) counts[name as string] += 1;

    // The shapes share a look and read no textures, so both draw through one bound pipeline.
    expect(counts).toEqual({
      setPipeline: 1,
      setBindGroup: 1,
      setIndexBuffer: 1,
      drawIndexed: 2,
      drawIndexedIndirect: 0,
    });
  });

  it("binds the same objects every frame, making none, until a mesh's textures arrive", async () => {
    const { root, log } = recordingRoot({ features: FIRST_INSTANCE });
    const { pool, box, view } = scene(root);
    const light = lighting.create(root, { shadow: {} });
    const things = instances.create(root, pool, { capacity: 1024 });
    const renderer = things.renderer().pipe(light.inject());
    const drawn = hdr(root, view).add(renderer);
    const cast = light.shadow.add(renderer);

    const triangle = part([1, 1, 1, 1]);
    const model = pool.add({
      meshes: [
        { ...triangle, material: { ...triangle.material, textures: { color: { image: 0 } } } },
      ],
      images: [{ width: 1, height: 1, pixels: new Uint8Array([255, 0, 0, 255]) }],
    });
    things.add(box);
    things.add(model);

    const { calls, made } = watch(root);
    const frame = framesOf(root, log.events);

    /** A frame of the shadow's pass and a colour pass, and what WebGPU was told in them. */
    const record = () => {
      const before = calls.length;
      frame((each) => {
        things.flush();
        light.shadow.record(each.encoder, [cast]);

        const pass = each.encoder.beginRenderPass({
          colorAttachments: { view: {} as GPUTextureView },
        });
        drawn.draw(pass);
        pass.end();
      });

      return calls.slice(before);
    };

    const first = record();
    const counts = { ...made, pipelines: log.pipelines };
    const second = record();

    expect({ ...made, pipelines: log.pipelines }).toEqual(counts);
    expect(second).toHaveLength(first.length);
    second.forEach((call, i) => call.forEach((argument, j) => expect(argument).toBe(first[i][j])));

    const index = model.meshes[0].index;
    const fallback = root.unwrap(pool.texturesOf(index));
    await model.ready;
    const textures = root.unwrap(pool.texturesOf(index));

    expect(textures).not.toBe(fallback);
    expect(groupsIn(first)).not.toContain(textures);
    expect(groupsIn(record())).toContain(textures);
  });

  it("lights from the description's beneath, under a lighting of its own, and names both when neither", async () => {
    const { root, log } = recordingRoot({ features: FIRST_INSTANCE });
    const { pool, view } = scene(root);
    /** A lighting whose ambient light is `level`, to tell it in the WGSL. */
    const glowing = (level: number) =>
      lighting.create(root, {
        shading: {
          ambient: () => {
            'use gpu';
            return d.vec3f(level);
          },
        },
      });
    const things = instances.create(root, pool, { capacity: 8 });
    const fragmentsOf = async (into: passes.Description, renderer: instances.Renderer) => {
      const before = log.shaders.length;
      await into.add(renderer).initAsync();
      return log.shaders.slice(before).filter((code) => code.includes('@fragment'));
    };

    const lit = hdr(root, view).beneath(glowing(0.25).inject());
    expect((await fragmentsOf(lit, things.renderer())).join()).toContain('vec3f(0.25)');
    const own = await fragmentsOf(lit, things.renderer().pipe(glowing(0.75).inject()));
    expect(own.join()).toContain('vec3f(0.75)');
    expect(own.join()).not.toContain('vec3f(0.25)');

    await expect(fragmentsOf(hdr(root, view), things.renderer())).rejects.toThrow(
      'give the pass description `beneath(light.inject())`',
    );
  });

  it('shades shapes with the material alone and binds no textures', async () => {
    const { root, log } = recordingRoot({ features: FIRST_INSTANCE });
    const { pool, box, view, light } = scene(root);
    const things = instances.create(root, pool, { capacity: 1024 });
    const into = hdr(root, view);
    const shapes = into.add(things.renderer().pipe(light.inject()));
    things.add(box);
    await into.initAsync();

    const fragment = log.shaders.find((code) => code.includes('@fragment'));
    expect(fragment).toBeDefined();
    expect(fragment).not.toContain('textureSample');

    const { calls } = watch(root);
    framesOf(
      root,
      log.events,
    )((frame) => {
      things.flush();
      const pass = frame.encoder.beginRenderPass({
        colorAttachments: { view: {} as GPUTextureView },
      });
      shapes.draw(pass);
      pass.end();
    });

    expect(calls.some(([name]) => name === 'drawIndexed')).toBe(true);
    expect(groupsIn(calls)).not.toContain(root.unwrap(pool.texturesOf(box.index)));
  });

  it('widens roughness with antialiased unless specularAntialiasing is false', async () => {
    const { root, log } = recordingRoot({ features: FIRST_INSTANCE });
    const { pool, view, light } = scene(root);
    const into = hdr(root, view);
    into.add(instances.create(root, pool, { capacity: 8 }).renderer().pipe(light.inject()));
    into.add(
      instances
        .create(root, pool, { capacity: 8 })
        .renderer({ specularAntialiasing: false })
        .pipe(light.inject()),
    );
    await into.initAsync();

    const [smooth, plain] = log.shaders.filter((code) => code.includes('@fragment'));
    expect(smooth).toContain('fn antialiased(');
    expect(plain).not.toContain('fn antialiased(');
  });

  it('carries a value from each vertex only for a kind with a carry, to a material of three arguments too', async () => {
    const { root, log } = recordingRoot({ features: FIRST_INSTANCE });
    const { pool, view, light } = scene(root);
    const Carried = instances.define('Carried', {
      carry: (vertexIndex) => {
        'use gpu';
        return d.vec4f(d.f32(vertexIndex));
      },
      material: (_point, _thing, base) => {
        'use gpu';
        return shading.Material(base);
      },
    });
    const into = hdr(root, view);
    into.add(instances.create(root, pool, { capacity: 8 }).renderer().pipe(light.inject()));
    const things = instances.create(root, pool, { kind: Carried, capacity: 8 });
    // A parameter typed by the bare name takes a set of any kind
    expectTypeOf(things).toExtend<instances.Instances>();
    into.add(things.renderer().pipe(light.inject()));
    await into.initAsync();

    const [plain, carried] = log.shaders.filter((code) => code.includes('@fragment'));
    expect(plain).not.toContain('struct Carrying');
    expect(carried).toMatch(/struct Carrying \{[^}]*carried: vec4f,\n\}/);
  });

  it("leaves the pass's own state alone, so a pipeline drawn after it keeps its index buffer", () => {
    const { root, log } = recordingRoot({ features: FIRST_INSTANCE });
    const { pool, box, view, light } = scene(root);
    const things = instances.create(root, pool, { capacity: 1024 });
    const drawn = hdr(root, view).add(things.renderer().pipe(light.inject()));
    things.add(box);

    const indices = root.createBuffer(d.arrayOf(d.u32, 3)).$usage('index');
    const own = root
      .createRenderPipeline({
        vertex: tgpu.vertexFn({ out: { position: d.builtin.position } })(() => {
          'use gpu';
          return { position: d.vec4f() };
        }),
        fragment: tgpu.fragmentFn({ out: d.vec4f })(() => {
          'use gpu';
          return d.vec4f();
        }),
        targets: { format: 'rgba16float' },
      })
      .withIndexBuffer(indices);

    const { calls } = watch(root);
    framesOf(
      root,
      log.events,
    )((frame) => {
      things.flush();

      const pass = frame.encoder.beginRenderPass({
        colorAttachments: { view: {} as GPUTextureView },
      });
      drawn.draw(pass);
      pass.setPipeline(own);
      pass.drawIndexed(3);
      pass.end();
    });

    const buffers = calls.filter(([name]) => name === 'setIndexBuffer').map(([, buffer]) => buffer);
    expect(buffers[0]).toBe(root.unwrap(pool.indices));
    expect(buffers.at(-1)).toBe(root.unwrap(indices));
  });

  // One group leaves none for the mesh's, as layers and a shared group do for a textured mesh.
  it.each([
    { through: 'one pipeline, binding each mesh', maxBindGroups: 4, colored: 1, meshGroups: 2 },
    {
      through: 'a pipeline per mesh with no group left',
      maxBindGroups: 1,
      colored: 2,
      meshGroups: 0,
    },
  ])('culls without indirect-first-instance through $through', async (each) => {
    const { root, log } = recordingRoot({ limits: { maxBindGroups: each.maxBindGroups } });
    const { pool, box, plane, view, light } = scene(root);
    const things = instances.create(root, pool, { capacity: 1024 });
    const renderer = things.renderer({ blend: 'translucent', cull: true }).pipe(light.inject());
    const ghosts = hdr(root, view).add(renderer);
    things.add(box);
    things.add(plane);

    const before = log.pipelines;
    await ghosts.initAsync();
    // The colour pipelines and the culling's two kernels.
    expect(log.pipelines - before).toBe(each.colored + 2);

    const { calls } = watch(root);
    framesOf(
      root,
      log.events,
    )((frame) => {
      things.flush();
      const culling = frame.encoder.beginComputePass();
      renderer.cull(culling);
      culling.end();
      const pass = frame.encoder.beginRenderPass({
        colorAttachments: { view: {} as GPUTextureView },
      });
      ghosts.draw(pass);
      pass.end();
    });
    const draws = calls.filter(([name]) => name === 'drawIndexedIndirect');
    expect(draws).toHaveLength(2);
    expect(log.pipelines - before).toBe(each.colored + 2);
    // Group 0 is the pipeline's own; each draw binds its mesh's group after it.
    const meshGroups = calls.filter(([name, index]) => name === 'setBindGroup' && index === 1);
    expect(new Set(meshGroups.map((call) => call[2])).size).toBe(each.meshGroups);
  });

  it("binds an indirect draw's mesh to the vertex stage alone, and textures to the fragment stage alone", async () => {
    const { root } = recordingRoot();
    const counted = census.create(root);
    const { view, light } = scene(root);
    const pool = meshPool.create(root, { vertices: 100, indices: 200, capacity: 4 });
    const { material } = part([1, 1, 1, 1]);
    const model = pool.add({
      meshes: [
        { ...part([1, 1, 1, 1]), material: { ...material, textures: { color: { image: 0 } } } },
      ],
      images: [{ width: 1, height: 1, pixels: new Uint8Array([255, 255, 255, 255]) }],
    });
    const things = instances.create(root, pool, { capacity: 4 });
    things.add(model);
    await hdr(root, view)
      .add(things.renderer({ cull: true }).pipe(light.inject()))
      .initAsync();

    const stagesOf = (name: string) =>
      counted.pipelines.flatMap((each) => each.bindings).find((each) => each.name === name)?.stages;
    expect(stagesOf('mesh')).toEqual(['vertex']);
    for (const name of ['reading', 'color', 'filtering'])
      expect(stagesOf(name)).toEqual(['fragment']);
    counted.destroy();
  });

  it('casts with a depth pipeline that reads positions only, once added to the shadow', async () => {
    const { root, log } = recordingRoot({ features: FIRST_INSTANCE });
    const { pool, box, view } = scene(root);
    const light = lighting.create(root, { shadow: {} });
    const things = instances.create(root, pool, { capacity: 1024 });
    const renderer = things.renderer().pipe(light.inject());
    const lit = hdr(root, view);
    lit.add(renderer);
    things.add(box);

    const before = log.pipelines;
    await Promise.all([lit.initAsync(), light.shadow.initAsync()]);
    expect(log.pipelines - before).toBe(1);
    const cast = light.shadow.add(renderer);
    await light.shadow.initAsync();
    expect(log.pipelines - before).toBe(2);

    const events = framesOf(
      root,
      log.events,
    )((frame) => {
      things.flush();
      light.shadow.record(frame.encoder, [cast]);
    });
    expect(events.slice(events.indexOf('begin-render'))).toEqual([
      'begin-render',
      'draw-indexed',
      'end-render',
      'submit',
    ]);
    expect(casting(log.shaders)).not.toContain('unpack2x16snorm');

    const ghosts = things.renderer({ blend: 'translucent' }).pipe(light.inject());
    expect(() => light.shadow.add(ghosts)).toThrow("blend: 'translucent' leaves depth alone");
  });

  it("shades a clear renderer through shadeClear, premultiplied, and keeps it out of depth alone as 'translucent' is", async () => {
    const { root, log } = recordingRoot({ features: FIRST_INSTANCE });
    const { pool, view } = scene(root);
    const light = lighting.create(root, { shadow: {} });
    const things = instances.create(root, pool, { capacity: 4 });

    const fragmentOf = async (blend: 'translucent' | 'clear') => {
      const into = hdr(root, view);
      into.add(things.renderer({ blend }).pipe(light.inject()));
      const from = log.shaders.length;
      await into.initAsync();
      return firstFragment(log.shaders.slice(from));
    };

    const blended = await fragmentOf('translucent');
    const clear = await fragmentOf('clear');
    expect(blended).not.toContain('shadeClear');
    expect(clear).toMatch(/fn shadeClear\w*\(/);

    const glass = things.renderer({ blend: 'clear' }).pipe(light.inject());
    expect(() => light.shadow.add(glass)).toThrow("blend: 'clear' leaves depth alone");
  });

  it('takes a transmit only for a clear renderer, in shares from 0 to 1', () => {
    const { root } = recordingRoot({ features: FIRST_INSTANCE });
    const { pool } = scene(root);
    const light = lighting.create(root);
    const things = instances.create(root, pool, { capacity: 4 });

    expect(() =>
      things.renderer({ blend: 'translucent', transmit: [1, 1, 1] }).pipe(light.inject()),
    ).toThrow("requires blend: 'clear'");
    expect(() =>
      things.renderer({ blend: 'clear', transmit: [1, 1.5, 1] }).pipe(light.inject()),
    ).toThrow(RangeError);
    expect(() =>
      things.renderer({ blend: 'clear', transmit: [0.9, 0.95, 1] }).pipe(light.inject()),
    ).not.toThrow();
  });

  it('draws each look of mesh through a pipeline of its own: two-sided, masked, textured or none', async () => {
    const { root, log } = recordingRoot({ features: FIRST_INSTANCE });
    const { view } = scene(root);
    const pool = meshPool.create(root, { vertices: 100, indices: 200, capacity: 4 });
    const box = pool.add(meshes.box());
    const { material } = part([1, 1, 1, 1]);
    const model = pool.add({
      meshes: [
        { ...part([1, 1, 1, 1]), material: { ...material, doubleSided: true } },
        { ...part([1, 1, 1, 1]), material: { ...material, alpha: 'mask' } },
        { ...part([1, 1, 1, 1]), material: { ...material, textures: { color: { image: 0 } } } },
      ],
      images: [{ width: 1, height: 1, pixels: new Uint8Array([255, 255, 255, 255]) }],
    });
    const looks = [box, ...model.meshes].map((mesh) => [
      mesh.doubleSided,
      mesh.masked,
      mesh.textured,
    ]);
    expect(looks).toEqual([
      [false, false, false],
      [true, false, false],
      [false, true, true],
      [false, false, true],
    ]);

    const culls: (GPUCullMode | undefined)[] = [];
    const create = root.device.createRenderPipelineAsync.bind(root.device);
    root.device.createRenderPipelineAsync = (descriptor) => {
      culls.push(descriptor.primitive?.cullMode);
      return create(descriptor);
    };

    const light = lighting.create(root, { shadow: {} });
    const things = instances.create(root, pool, { capacity: 1024 });
    const renderer = things.renderer().pipe(light.inject());
    const lit = hdr(root, view);
    lit.add(renderer);
    light.shadow.add(renderer);
    things.add(box);
    things.add(model);

    const before = log.pipelines;
    const shaders = log.shaders.length;
    await Promise.all([lit.initAsync(), light.shadow.initAsync()]);

    // Colour for each look; shadow for the box, the two-sided mesh and the masked one, since
    // textures alone don't change how a mesh casts.
    expect(log.pipelines - before).toBe(7);
    expect(culls.toSorted()).toEqual(['back', 'back', 'back', 'back', 'back', 'none', 'none']);
    const made = log.shaders.slice(shaders);
    expect(made.filter((code) => code.includes('discard'))).toHaveLength(2);
    // The masked mesh casts through a fragment that reads its colour's alpha, at its own depth.
    const masking = made.filter((code) => /alphaCutoff/.test(code));
    expect(masking.filter((code) => /@fragment fn [^\n]*\) \{\n/.test(code))).toHaveLength(1);
    expect(made.filter((code) => code.includes('frag_depth'))).toHaveLength(0);

    // With textures ignored, nothing reads a mask and the textured mesh shares the box's pipeline.
    const plain = hdr(root, view);
    plain.add(things.renderer({ textured: false }).pipe(light.inject()));
    const untextured = log.shaders.length;
    await plain.initAsync();
    const ignoring = log.shaders.slice(untextured);
    expect(ignoring.some((code) => code.includes('discard'))).toBe(false);
    expect(ignoring.some((code) => /textureSample\(/.test(code))).toBe(false);
  });

  it('culls in the pass it is given, for each description it was added to while it is enabled', async () => {
    const { root, log } = recordingRoot({ features: FIRST_INSTANCE });
    const { pool, box, view } = scene(root);
    let on = true;
    const light = lighting.create(root, { shadow: { enabled: () => on } });
    const things = instances.create(root, pool, { capacity: 600 });
    const renderer = things.renderer({ cull: true }).pipe(light.inject());
    const lit = hdr(root, view);
    lit.add(renderer);
    const inShadow = light.shadow.add(renderer);
    for (let i = 0; i < 300; i++) things.add(box, { position: [i, 0, 0] });

    const before = log.pipelines;
    await Promise.all([lit.initAsync(), light.shadow.initAsync()]);
    // The colour pipeline and the shadow's, and two kernels for each view.
    expect(log.pipelines - before).toBe(2 + 2 * 2);

    const run = framesOf(root, log.events);
    const record = (frame: Frame) => {
      things.flush();
      const pass = frame.encoder.beginComputePass();
      renderer.cull(pass);
      pass.end();
      light.shadow.record(frame.encoder, [inShadow]);
    };
    const cast = ['begin-render', 'draw-indexed-indirect', 'end-render'];
    expect(run(record)).toEqual([
      'begin-compute',
      ...Array<string>(4).fill('dispatch'),
      'end-compute',
      ...cast,
      'submit',
    ]);

    const steady = log.counts();
    run(record);
    expect(log.counts()).toEqual(steady);

    on = false;
    expect(run(record)).toEqual(['begin-compute', 'dispatch', 'dispatch', 'end-compute', 'submit']);
  });

  it('culls every layer of a layered description in one pair of dispatches, drawing each its own list', async () => {
    const { root, log } = recordingRoot({ features: FIRST_INSTANCE });
    const { pool, box, plane, light } = scene(root);
    const things = instances.create(root, pool, { capacity: 600 });
    const renderer = things.renderer({ cull: true }).pipe(light.inject());
    const views = cameras.layered(root, 3);
    const casters = passes
      .create(
        root,
        targets.create(
          root,
          { width: 4, height: 4 },
          { depth: { format: 'depth16unorm', layers: 3 } },
        ),
        { depth: passes.depth },
      )
      .pipe(views.inject());
    const drawn = casters.add(renderer);
    for (let i = 0; i < 300; i++) things.add(i % 2 ? box : plane, { position: [i, 0, 0] });

    const before = log.pipelines;
    await casters.initAsync();
    // One cast pipeline and the culling's two kernels, whatever the number of layers.
    expect(log.pipelines - before).toBe(1 + 2);

    const device = root.device;
    const dispatched: number[][] = [];
    const drawnAt: number[] = [];
    const createEncoder = device.createCommandEncoder.bind(device);
    device.createCommandEncoder = (descriptor) => {
      const encoder = createEncoder(descriptor);
      const compute = encoder.beginComputePass.bind(encoder);
      encoder.beginComputePass = (passDescriptor) => {
        const pass = compute(passDescriptor);
        const dispatch = pass.dispatchWorkgroups.bind(pass);
        pass.dispatchWorkgroups = (...size: number[]) => {
          dispatched.push(size.slice(0, 2));
          dispatch(...(size as [number]));
        };
        return pass;
      };
      const render = encoder.beginRenderPass.bind(encoder);
      encoder.beginRenderPass = (passDescriptor) => {
        const pass = render(passDescriptor);
        const draw = pass.drawIndexedIndirect.bind(pass);
        pass.drawIndexedIndirect = (buffer, offset) => {
          drawnAt.push(offset);
          draw(buffer, offset);
        };
        return pass;
      };
      return encoder;
    };

    const run = framesOf(root, log.events);
    const record = (frame: Frame) => {
      things.flush();
      const pass = frame.encoder.beginComputePass();
      renderer.cull(pass);
      pass.end();
      casters.record(frame, drawn);
    };
    const cast = ['begin-render', 'draw-indexed-indirect', 'draw-indexed-indirect', 'end-render'];
    expect(run(record)).toEqual([
      'begin-compute',
      'dispatch',
      'dispatch',
      'end-compute',
      ...cast,
      ...cast,
      ...cast,
      'submit',
    ]);
    expect(dispatched).toEqual([
      [2, 3],
      [2, 3],
    ]);
    // Each layer's records, one per mesh the pool holds, follow those of the layers before it.
    expect(drawnAt).toEqual([0, 1, 2].flatMap((k) => [recordOf(k, plane), recordOf(k, box)]));

    const steady = log.counts();
    run(record);
    expect(log.counts()).toEqual(steady);
  });

  it('cull only when made to, and a kind that deforms only within bounds it gives', () => {
    const { root } = recordingRoot({ features: FIRST_INSTANCE });
    const { pool, light } = scene(root);
    const pass = {} as Parameters<instances.Renderer['cull']>[0];

    const plain = instances.create(root, pool, { capacity: 1024 }).renderer().pipe(light.inject());
    expect(() => plain.cull(pass)).toThrow('without `cull: true`');

    const waving = instances.define('Waving', { deform: wave });
    const options = { cull: true } as const;
    expect(() =>
      instances
        .create(root, pool, { kind: waving, capacity: 1024 })
        .renderer(options)
        .pipe(light.inject()),
    ).toThrow("'Waving' uses deform");

    const bounded = instances.define('Bounded', {
      deform: wave,
      bounds: (_, base) => {
        'use gpu';
        return instances.Sphere({ center: base.center, radius: base.radius + 1 });
      },
    });
    expect(() =>
      instances
        .create(root, pool, { kind: bounded, capacity: 1024 })
        .renderer(options)
        .pipe(light.inject()),
    ).not.toThrow();
  });

  it('draw a range only when culling, from 0 or more to further on', () => {
    const { root } = recordingRoot({ features: FIRST_INSTANCE });
    const { pool, view, light } = scene(root);
    const things = instances.create(root, pool, { capacity: 1024 });
    const make =
      (range: Range, cull = true) =>
      () =>
        things.renderer({ cull, ...range, eye: view }).pipe(light.inject());

    expect(make({ to: 10 }, false)).toThrow('requires cull: true');
    expect(make({ from: -1 })).toThrow(RangeError);
    expect(make({ from: 10, to: 10 })).toThrow(RangeError);
    expect(make({ to: 10, fade: -1 })).toThrow(RangeError);
    expect(make({ from: 5, to: 10, fade: 2 })).not.toThrow();
    expect(() => things.renderer({ cull: true, to: 10 }).pipe(light.inject())).toThrow(
      'requires eye',
    );
  });

  it('leave pixels out only across the fade, and measure from their own camera in the shadow', async () => {
    const fades = await shadersOver({ from: 5, to: 10, fade: 2 });
    expect(firstFragment(fades)).toContain('discard');
    expect(firstFragment(await shadersOver({ to: 10 }))).not.toContain('discard');
    expect(firstFragment(await shadersOver())).not.toContain('discard');
    // The camera's list and the shadow's both measure from the range's eye.
    const kernels = fades.filter((code) => code.includes('@compute') && code.includes('distance('));
    expect(kernels).toHaveLength(2);
    for (const code of fades) expect(code).not.toMatch(/\binf\b/);
  });

  it("adds a model as a row per mesh under one id, from each mesh's factors", () => {
    const { root, log, bytes } = recordingRoot();
    const { pool } = scene(root);
    const model = pool.add({ meshes: [part([1, 0, 0, 1]), part([0, 0, 1, 1])], images: [] });
    const things = instances.create(root, pool, { capacity: 3 });
    const frame = framesOf(root, log.events);

    const id = things.add(model, { position: [1, 2, 3], roughness: 0.25 });
    expect([things.count, things.span]).toEqual([1, 2]);
    things.write(id, { position: [4, 5, 6] });
    frame(() => things.flush());

    const rows = log.writes.find((write) => write.size === 2 * 80)!;
    const floats = new Float32Array(bytes(rows.buffer).buffer, 0, 40);
    // Each row: position, roughness, color.
    expect(
      [0, 20].map((at) => [
        ...floats.subarray(at, at + 3),
        floats[at + 11],
        ...floats.subarray(at + 12, at + 15),
      ]),
    ).toEqual([
      [4, 5, 6, 0.25, 1, 0, 0],
      [4, 5, 6, 0.25, 0, 0, 1],
    ]);

    expect(() => things.add(model)).toThrow('full (3 rows; a model takes a row per mesh)');
    expect(things.count).toBe(1);
    things.remove(id);
    expect(things.count).toBe(0);
    expect(things.has(id)).toBe(false);
  });

  it('are a table: field handles reach every row of a model, and any of its slots names it', () => {
    const { root, bytes } = recordingRoot();
    const { pool, box } = scene(root);
    const model = pool.add({ meshes: [part([1, 0, 0, 1]), part([0, 0, 1, 1])], images: [] });
    const things = instances.create(root, pool, { capacity: 4 });
    const crate = things.add(box);
    const figure = things.add(model);
    const position = things.fields.position;

    position.set(figure, 4, 5, 6);
    things.fields.roughness.set(crate, 0.5);
    things.flush();

    const floats = new Float32Array(bytes(things.buffers.host!).buffer, 0, 60);
    // Each row: position, roughness
    expect([0, 20, 40].map((at) => [...floats.subarray(at, at + 3), floats[at + 11]])).toEqual([
      [0, 0, 0, 0.5],
      [4, 5, 6, 0.5],
      [4, 5, 6, 0.5],
    ]);
    expect(position.get(figure, d.vec3f())).toEqual(d.vec3f(4, 5, 6));
    expect('mesh' in things.fields).toBe(false);

    expect([0, 1, 2].map((slot) => things.idAt(slot))).toEqual([crate, figure, figure]);
    expect(things.slotOf(figure)).toBe(1);
    expect(things.live().capacity).toBe(4);
    expect(() => things.idAt(3)).toThrow('Slot 3 holds no row.');

    things.clear();
    expect([things.count, things.span, things.has(figure)]).toEqual([0, 0, false]);
  });

  it("gives a model's meshes values of their own, and defaults in place of the mesh's", () => {
    const { root, bytes } = recordingRoot();
    const { pool, box } = scene(root);
    const model = pool.add({ meshes: [part([1, 0, 0, 1]), part([0, 0, 1, 1])], images: [] });
    const things = instances.create(root, pool, {
      capacity: 4,
      defaults: { roughness: 0.25, scale: 2 },
    });

    things.add(model, (mesh, index) => ({ emission: [index, 0, 0], color: mesh.material.color }));
    things.add(box, { roughness: 0.75 });
    things.flush();

    const floats = new Float32Array(bytes(things.buffers.host!).buffer, 0, 60);
    // Each row: scale.x, roughness, color, emission.x
    expect(
      [0, 20, 40].map((at) => [
        floats[at + 8],
        floats[at + 11],
        ...floats.subarray(at + 12, at + 15),
        floats[at + 16],
      ]),
    ).toEqual([
      [2, 0.25, 1, 0, 0, 0],
      [2, 0.25, 0, 0, 1, 1],
      [2, 0.75, 1, 1, 1, 0],
    ]);
  });

  it('keep last frame’s rows with history, so instances the CPU moves have motion', async () => {
    const { root, log } = recordingRoot();
    const { pool, box, view, light } = scene(root);
    const things = instances.create(root, pool, { capacity: 4, history: true });
    const moving = passes
      .create(
        root,
        targets.create(
          root,
          { width: 1, height: 1 },
          {
            hdr: { format: 'rgba16float' },
            motion: { format: 'rg16float' },
            depth: { format: 'depth24plus' },
          },
        ),
        { hdr: passes.color, motion: taa.motion, depth: passes.depth },
      )
      .pipe(view.inject());
    moving.add(things.renderer().pipe(light.inject()));
    await moving.initAsync();
    things.add(box);

    // Last frame's rows follow the current ones in the same buffer, so motion binds nothing more
    const vertex = log.shaders.find((code) => code.includes('@vertex'))!;
    expect(vertex).toContain('Instance_host[(i + 4u)]');
    expect(vertex).not.toContain('Instance_host_previous');
    // @ts-expect-error a set with history copies last frame's rows in the frame
    expect(() => things.flush()).toThrow('flush(frame)');
    frames.create(root, (frame) => things.flush(frame)).run(0);
  });

  it('take no fields of their own without a kind, and passes whose outputs they feed', () => {
    const { root } = recordingRoot();
    const { pool, box, view, light } = scene(root);
    const things = instances.create(root, pool, { capacity: 1024 });

    // @ts-expect-error -- a plain set has no fields of its own.
    things.add(box, { speed: 1 });

    const id = passes.output(d.u32, {
      write: (pixel: { readonly id: number }) => {
        'use gpu';
        return d.u32(pixel.id);
      },
    });
    const screen = targets.create(
      root,
      { width: 1, height: 1 },
      {
        hdr: { format: 'rgba16float' },
        id: { format: 'r32uint' },
        depth: { format: 'depth24plus' },
      },
    );
    const described = passes
      .create(root, screen, { hdr: passes.color, id, depth: passes.depth })
      .pipe(view.inject());
    const renderer = things.renderer().pipe(light.inject());
    // @ts-expect-error -- an instance knows no id at a pixel.
    described.add(renderer);
    described.without('id').add(renderer);
  });

  it('refuses a mesh from another pool, and a model with no meshes', () => {
    const { root } = recordingRoot();
    const { pool } = scene(root);
    const other = meshPool.create(root, { capacity: 64, vertices: 100, indices: 200 });
    const things = instances.create(root, pool, { capacity: 1024 });

    expect(() => things.add(other.add(meshes.box()))).toThrow('another pool');
    expect(() => things.add({ meshes: [] })).toThrow('no meshes');
    expect(things.count).toBe(0);
  });
});

/** Lit or dark, nothing between: a cel-shaded `reflect`. */
const toon = (
  material: shading.Material,
  point: shading.Point,
  direction: d.v3f,
  radiance: d.v3f,
) => {
  'use gpu';
  return material.color * radiance * std.step(0.5, std.dot(point.normal, direction));
};

const asIs = (_point: shading.Point, _row: unknown, base: shading.Material) => {
  'use gpu';
  return base;
};

const wholly = (point: shading.Point, _row: unknown, base: shading.Material) => {
  'use gpu';
  return instances.Shaded({ point, material: base });
};

const lace = (point: shading.Point) => {
  'use gpu';
  return std.step(0.5, std.fract(point.uv.x * 8));
};

describe('kinds', () => {
  it("lay their own fields after every instance's, and move the transform to the device part", () => {
    const Fish = instances.define('Fish', {
      host: { belly: d.vec3f, speed: d.f32 },
      device: { velocity: d.vec3f },
      transform: 'device',
    });
    expect(d.sizeOf(Fish['~parts'].host!)).toBe(64);
    expect(d.sizeOf(Fish['~parts'].device!)).toBe(64);

    const { root, log, bytes } = recordingRoot();
    const { pool, box } = scene(root);
    const school = instances.create(root, pool, { kind: Fish, capacity: 4 });
    school.add(box, {
      position: [1, 2, 3],
      color: [1, 0, 0],
      belly: [0, 0, 1],
      speed: 2,
      velocity: [4, 5, 6],
    });
    frames.create(root, () => school.flush()).run(0);

    const [host, device] = log.writes
      .filter((write) => write.size === 64)
      .map((write) => [...new Float32Array(bytes(write.buffer).buffer, 0, 16)]);

    // color, roughness, emission, metallic, mesh, opacity, belly, speed
    expect(host).toEqual([1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2]);
    // rotation, position, scale, velocity
    expect(device).toEqual([0, 0, 0, 1, 1, 2, 3, 0, 1, 1, 1, 0, 4, 5, 6, 0]);
  });

  it('brand their ids with their name, and a plain set’s with its own', () => {
    const { root } = recordingRoot();
    const { pool, box } = scene(root);
    const Fish = instances.define('Fish', { host: { belly: d.vec3f } });
    const Bird = instances.define('Bird', { host: { belly: d.vec3f } });
    const school = instances.create(root, pool, { kind: Fish, capacity: 1024 });
    const flock = instances.create(root, pool, { kind: Bird, capacity: 1024 });
    const things = instances.create(root, pool, { capacity: 1024 });

    const fish: instances.Id<typeof Fish> = school.add(box);
    const crate: instances.Id = things.add(box);

    // @ts-expect-error: a fish's id is not a bird's, though their fields are the same
    flock.has(fish);
    // @ts-expect-error: nor is a plain set's
    flock.has(crate);
    // @ts-expect-error: and a fish's is not a plain set's
    things.has(fish);
    expect([school.has(fish), things.has(crate)]).toEqual([true, true]);
  });

  it('name their structs, buffers and functions after the kind', async () => {
    const { root, log } = recordingRoot({ features: FIRST_INSTANCE });
    const { pool, view, light } = scene(root);
    const Fish = instances.define('Fish', { device: { velocity: d.vec3f }, transform: 'device' });

    const lit = hdr(root, view);
    lit.add(
      instances.create(root, pool, { kind: Fish, capacity: 8 }).renderer().pipe(light.inject()),
    );
    await lit.initAsync();
    const wgsl = log.shaders.join('\n');

    expect(wgsl).toContain('var<storage, read> Fish_device: array<FishDevice, 8>;');
    expect(wgsl).toContain(
      'fn Fish_position(i: u32) -> vec3f {\n  return Fish_device[(i + 0u)].position;\n}',
    );
    expect(wgsl).toContain('var<storage, read> Fish_listing: array<u32, 12>;');
  });

  it('refuse a field that every instance has, or a name the table keeps', () => {
    // @ts-expect-error -- `color` is a field of every instance.
    expect(() => instances.define('Fish', { host: { color: d.vec3f } })).toThrow(
      'built-in instance field',
    );
    // @ts-expect-error -- and so is `position`.
    expect(() => instances.define('Fish', { device: { position: d.vec3f } })).toThrow(
      'built-in instance field',
    );

    // @ts-expect-error -- `alive` would hide the table's own.
    expect(() => instances.define('Fish', { host: { alive: d.u32 } })).toThrow('reserved');
    // @ts-expect-error -- and so would `span`.
    expect(() => instances.define('Fish', { device: { span: d.u32 } })).toThrow('reserved');
    // @ts-expect-error -- and `row`.
    expect(() => instances.define('Fish', { host: { row: d.u32 } })).toThrow('reserved');
    // @ts-expect-error -- and `slotOf`.
    expect(() => instances.define('Fish', { host: { slotOf: d.u32 } })).toThrow('reserved');
  });

  it('give their material or their surface, not both', () => {
    expect(() => instances.define('Ghost', { material: asIs, surface: wholly })).toThrow(
      'either material or surface, not both',
    );
  });

  it('draw a surface once, cutting out only by its coverage, and twice when asked for a prepass', async () => {
    const Whole = instances.define('Whole', { surface: wholly });
    const Lace = instances.define('Lace', { surface: wholly, coverage: lace });

    const { root, log } = recordingRoot({ features: FIRST_INSTANCE });
    const { pool, box, view, light } = scene(root);
    const frame = framesOf(root, log.events);
    const drawn = async (kind: typeof Whole | typeof Lace, options = {}, samples = 1) => {
      const set = instances.create(root, pool, { kind: kind as typeof Whole, capacity: 4 });
      set.add(box);
      const lit = hdr(root, view, samples);
      const built = lit.add(set.renderer(options).pipe(light.inject()));
      const shaders = log.shaders.length;
      await lit.initAsync();
      const events = frame((each) => {
        set.flush();
        const pass = lit.begin(each.encoder);
        built.draw(pass);
        pass.end();
      });
      const fragments = log.shaders.slice(shaders).filter((code) => code.includes('@fragment'));
      return {
        draws: events.filter((event) => event.startsWith('draw')).length,
        discards: fragments.map((code) => code.includes('discard')),
      };
    };

    expect(await drawn(Whole)).toEqual({ draws: 1, discards: [false] });
    expect(await drawn(Lace)).toEqual({ draws: 1, discards: [true] });
    expect(await drawn(Lace, { prepass: true })).toEqual({ draws: 2, discards: [false, true] });
    // Multisampled, coverage selects samples, sharpened to a pixel
    const sampled = log.shaders.length;
    await drawn(Lace, {}, 4);
    expect(log.shaders.slice(sampled).some((code) => code.includes('fwidth'))).toBe(true);
  });

  it('cast with the whole vertex once they deform it, unless the shadow is left undeformed', async () => {
    const Waving = instances.define('Waving', { deform: wave });
    const Beating = instances.define('Beating', { deform: wave, deformShadow: false });

    expect(await castBy(Waving)).toContain('unpack2x16snorm');
    expect(await castBy(Beating)).not.toContain('unpack2x16snorm');
  });

  it('pipe a fill into its pipelines', async () => {
    const tint = tgpu.accessor(d.vec3f);
    const Tinted = instances.define('Tinted', {
      material: () => {
        'use gpu';
        return materials.matte(tint.$);
      },
    });

    const { root } = recordingRoot({ features: FIRST_INSTANCE });
    const { pool, view, light } = scene(root);
    const set = instances.create(root, pool, { kind: Tinted, capacity: 1024 });

    const tinted = hdr(root, view);
    tinted.add(
      set
        .renderer()
        .pipe(light.inject())
        .pipe((cfg) => cfg.with(tint, d.vec3f(1, 0, 0))),
    );
    await expect(tinted.initAsync()).resolves.toBeUndefined();
    const untinted = hdr(root, view);
    untinted.add(set.renderer().pipe(light.inject()));
    await expect(untinted.initAsync()).rejects.toThrow('Missing value');
  });

  it('pipe a fill after the lighting, so it can fill a hook the lighting names', async () => {
    const { root, log } = recordingRoot({ features: FIRST_INSTANCE });
    const { pool, view } = scene(root);
    const light = lighting.create(root, { shading: { reflect: shading.lambert } });
    const set = instances.create(root, pool, { capacity: 1024 });

    const lit = hdr(root, view);
    lit.add(
      set
        .renderer()
        .pipe(light.inject())
        .pipe((cfg) => cfg.with(shading.reflect, toon)),
    );
    await lit.initAsync();

    const fragment = log.shaders.find((code) => code.includes('@fragment'))!;
    expect(fragment).toContain('fn toon');
    expect(fragment).not.toContain('fn lambert');
  });

  it('turn +z towards a direction, keeping +x level and +y up', () => {
    const directions = [
      [1, 0, 0],
      [0, 0, -1],
      [1, 1, 0],
      [0.3, -0.5, 0.8],
      [0, 1, 0],
    ];

    for (const direction of directions) {
      const q = instances.toward(d.vec3f(...(direction as [number, number, number])));
      const length = Math.hypot(...direction);

      turn(q, [0, 0, 1]).forEach((value, axis) =>
        expect(value).toBeCloseTo(direction[axis] / length, 5),
      );
      expect(turn(q, [1, 0, 0])[1]).toBeCloseTo(0, 5);
      expect(turn(q, [0, 1, 0])[1]).toBeGreaterThanOrEqual(-1e-6);
    }
  });
});
