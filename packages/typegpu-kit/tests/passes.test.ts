import {
  d,
  std,
  tgpu,
  type TgpuCommandEncoder,
  type TgpuRenderCommands,
  type TgpuRoot,
} from 'typegpu';
import { describe, expect, it } from 'vitest';
import {
  bindings,
  cameras,
  font,
  particles,
  passes,
  pipelines,
  targets,
  text,
  trails,
  type Frame,
} from '../src/index.ts';
import { recordingRoot } from '../src/testing.ts';
import { stagesOf, type Stages } from './stages.ts';
import { inter } from './text/inter.ts';

const size = { width: 64, height: 32 };

const Clips = d.struct({ current: d.vec4f, previous: d.vec4f });

const uvOf = (clip: d.v4f) => {
  'use gpu';
  return (clip.xy / clip.w) * d.vec2f(0.5, -0.5);
};

const motion = passes.output(d.vec2f, {
  varyings: Clips,
  previous: true,
  vary: (current: { position: d.v4f }, previous: { position: d.v4f }) => {
    'use gpu';
    return Clips({ current: current.position, previous: previous.position });
  },
  write: (_pixel: unknown, clips) => {
    'use gpu';
    return uvOf(clips.previous) - uvOf(clips.current);
  },
});

const id = passes.output(d.u32, {
  clear: 0xffffffff,
  write: (pixel: { id: number }) => {
    'use gpu';
    return pixel.id;
  },
});

const Vertex = d.struct({ position: d.vec4f, world: d.vec3f, id: d.u32 });
const Surface = d.struct({ world: d.vec3f, id: d.u32, color: d.vec4f });
const Glow = d.struct({ color: d.vec4f });

const at = (vertexIndex: number, instanceIndex: number) => {
  'use gpu';
  const corners = [d.vec3f(-1, -1, 0), d.vec3f(1, -1, 0), d.vec3f(0, 1, 0)];
  const world = corners[vertexIndex] + d.vec3f(d.f32(instanceIndex), 0, 0);
  return Vertex({
    position: cameras.current.$.viewProjection * d.vec4f(world, 1),
    world,
    id: instanceIndex,
  });
};

const pixelOf = (vertex: d.Infer<typeof Vertex>) => {
  'use gpu';
  return Surface({ world: vertex.world, id: vertex.id, color: d.vec4f(1) });
};

const shaded = passes.output(d.vec4f, {
  write: (pixel: { color: d.v4f }) => {
    'use gpu';
    return d.vec4f(pixel.color);
  },
});

const Flat = d.struct({ position: d.vec4f });
const flatAt = (vertexIndex: number, _instanceIndex: number) => {
  'use gpu';
  const corners = [d.vec2f(-1, -1), d.vec2f(3, -1), d.vec2f(-1, 3)];
  return Flat({ position: d.vec4f(corners[vertexIndex], 0.5, 1) });
};
const flatSurface = (_vertex: d.Infer<typeof Flat>) => {
  'use gpu';
  return Surface({ world: d.vec3f(), id: 0, color: d.vec4f(0.5) });
};

function scene(root: TgpuRoot, options: { resolution?: number } = {}) {
  const screen = targets.create(
    root,
    size,
    {
      hdr: { format: 'rgba16float' },
      motion: { format: 'rg16float' },
      id: { format: 'r32uint' },
      depth: { format: 'depth24plus' },
    },
    options,
  );
  const pass = passes.create(root, screen, {
    hdr: passes.color,
    motion,
    id,
    depth: passes.depth,
  });
  return { screen, pass };
}

/** A pipeline from a renderer's stages, under an orbit camera, and its WGSL. */
function compiled(root: TgpuRoot, stages: Stages) {
  const view = cameras.orbit(root, { size, target: [0, 0, 0], distance: 5 });
  const pipeline = root.pipe(view.inject()).createRenderPipeline(stages);
  return tgpu.resolve([pipeline]);
}

/** An encoder that keeps the descriptor of the pass it begins and the viewport set on it. */
function watched() {
  const seen: { descriptor?: GPURenderPassDescriptor; viewport?: number[] } = {};
  const encoder = {
    beginRenderPass(descriptor: GPURenderPassDescriptor) {
      seen.descriptor = descriptor;
      return {
        setViewport: (...viewport: number[]) => (seen.viewport = viewport),
      };
    },
  } as unknown as TgpuCommandEncoder;
  return { encoder, seen };
}

describe('passes', () => {
  it('builds a renderer’s stages around its vertex and surface, running every output', () => {
    const { root } = recordingRoot();
    const { pass } = scene(root);
    const stages = stagesOf(pass, { Vertex, at, pixelOf, color: shaded });
    const wgsl = compiled(root, stages);

    expect(Object.keys(stages.targets)).toEqual(['hdr', 'motion', 'id']);
    expect(stages.targets.id).toEqual({ format: 'r32uint' });
    // Integers go between the stages flat: the vertex's id, written out by `id`.
    expect(wgsl).toMatch(/@location\(\d+\) @interpolate\(flat\) id: u32/);
    expect(wgsl).toContain('motion_current: vec4f');
    expect(wgsl).toContain('motion_previous: vec4f');
    expect(wgsl).toMatch(
      /@location\(0\) hdr: vec4f,\s*@location\(1\) motion: vec2f,\s*@location\(2\) id: u32/,
    );
    // Motion's `vary` takes last frame's vertex, so `at` compiles twice, the second reading last
    // frame's camera.
    expect(wgsl.match(/var<uniform>/g)).toHaveLength(2);
    expect(wgsl.match(/fn at(_\d+)?\(/g)).toHaveLength(2);
  });

  it('compiles no last frame when no output’s `vary` takes it', () => {
    const { root } = recordingRoot();
    const screen = targets.create(root, size, {
      hdr: { format: 'rgba16float' },
      depth: { format: 'depth24plus' },
      clip: { format: 'rgba16float' },
    });
    const Clip = d.struct({ clip: d.vec4f });
    const clip = passes.output(d.vec4f, {
      varyings: Clip,
      vary: (current: { position: d.v4f }) => {
        'use gpu';
        return Clip({ clip: current.position });
      },
      write: (_pixel: unknown, carried) => {
        'use gpu';
        return d.vec4f(carried.clip);
      },
    });
    const pass = passes.create(root, screen, { hdr: passes.color, clip, depth: passes.depth });
    const wgsl = compiled(root, stagesOf(pass, { Vertex, at, pixelOf, color: shaded }));

    expect(wgsl.match(/var<uniform>/g)).toHaveLength(1);
    expect(wgsl.match(/fn at(_\d+)?\(/g)).toHaveLength(1);
  });

  it('writes the clear of outputs an opaque renderer leaves out, as if nothing drew there', () => {
    const { root } = recordingRoot();
    const { pass } = scene(root);
    const stages = stagesOf(pass.without('motion', 'id'), {
      Vertex: Flat,
      at: flatAt,
      pixelOf: flatSurface,
      color: shaded,
    });
    const wgsl = compiled(root, stages);

    expect(stages.targets).toEqual({
      hdr: { format: 'rgba16float' },
      motion: { format: 'rg16float' },
      id: { format: 'r32uint' },
    });
    expect(wgsl).not.toContain('motion_current');
    expect(wgsl).toContain('4294967295u');
  });

  it('writes nothing into the targets of outputs a blended renderer leaves out, carrying nothing for them', () => {
    const { root } = recordingRoot();
    const { pass } = scene(root);
    const stages = stagesOf(pass.without('motion', 'id'), {
      Vertex: Flat,
      at: flatAt,
      pixelOf: flatSurface,
      color: shaded,
      blend: { color: { dstFactor: 'one' }, alpha: { dstFactor: 'one' } },
    });
    const wgsl = compiled(root, stages);

    expect(stages.targets).toEqual({
      hdr: {
        format: 'rgba16float',
        blend: { color: { dstFactor: 'one' }, alpha: { dstFactor: 'one' } },
      },
      motion: { format: 'rg16float', writeMask: 0 },
      id: { format: 'r32uint', writeMask: 0 },
    });
    expect(wgsl).not.toContain('motion_current');
    expect(wgsl.match(/var<uniform>/g)).toBeNull();
  });

  it('writes depth alone under a write mask of 0, still running its colour', () => {
    const { root } = recordingRoot();
    const { pass } = scene(root);
    const stages = stagesOf(pass, {
      Vertex: Flat,
      at: flatAt,
      pixelOf: flatSurface,
      color: shaded,
      writeMask: 0,
    });
    const wgsl = compiled(root, stages);

    expect(stages.targets).toEqual({
      hdr: { format: 'rgba16float', writeMask: 0 },
      motion: { format: 'rg16float', writeMask: 0 },
      id: { format: 'r32uint', writeMask: 0 },
    });
    expect(wgsl).not.toContain('motion_current');
    expect(stages.depthStencil).toMatchObject({ depthWriteEnabled: true });
  });

  it('writes a renderer’s own output in place of the description’s', () => {
    const { root } = recordingRoot();
    const { pass } = scene(root);
    const still = passes.output(d.vec2f, {
      write: () => {
        'use gpu';
        return d.vec2f(0.125, 0);
      },
    });
    const wgsl = compiled(
      root,
      stagesOf(pass.instead({ motion: still }), {
        Vertex: Flat,
        at: flatAt,
        pixelOf: flatSurface,
        color: shaded,
      }),
    );

    expect(wgsl).toContain('0.125');
    expect(wgsl).not.toContain('motion_current');
  });

  it('begins a pass into its targets in order, each cleared to its output’s clear or kept', () => {
    const { root } = recordingRoot();
    const { screen, pass } = scene(root);
    const { encoder, seen } = watched();

    pass.begin(encoder);
    const colors = [...(seen.descriptor!.colorAttachments as GPURenderPassColorAttachment[])];
    expect(colors.map((each) => each.view)).toEqual([
      screen.views.hdr,
      screen.views.motion,
      screen.views.id,
    ]);
    expect(colors.map((each) => [each.loadOp, each.clearValue])).toEqual([
      ['clear', [0, 0, 0, 0]],
      ['clear', [0, 0, 0, 0]],
      ['clear', [0xffffffff, 0, 0, 0]],
    ]);
    expect(seen.descriptor!.depthStencilAttachment).toMatchObject({
      view: screen.views.depth,
      depthLoadOp: 'clear',
      depthClearValue: 1,
    });
    expect(seen.viewport).toBeUndefined();

    pass.begin(encoder, { clear: false });
    const kept = [...(seen.descriptor!.colorAttachments as GPURenderPassColorAttachment[])];
    expect(kept.map((each) => each.loadOp)).toEqual(['load', 'load', 'load']);
    expect(seen.descriptor!.depthStencilAttachment?.depthLoadOp).toBe('load');

    pass.begin(encoder, { clear: ['hdr', 'id'] });
    const some = [...(seen.descriptor!.colorAttachments as GPURenderPassColorAttachment[])];
    expect(some.map((each) => each.loadOp)).toEqual(['clear', 'load', 'clear']);
    expect(seen.descriptor!.depthStencilAttachment?.depthLoadOp).toBe('load');
  });

  it('reads its clear colour at each begin, so writing into it changes the next pass', () => {
    const { root } = recordingRoot();
    const screen = targets.create(root, size, { hdr: { format: 'rgba16float' } });
    const sky = [0, 0, 0, 1];
    const pass = passes.create(root, screen, { hdr: passes.color }, { clear: sky });
    const { encoder, seen } = watched();

    sky[2] = 0.5;
    pass.begin(encoder);
    const [color] = seen.descriptor!.colorAttachments as GPURenderPassColorAttachment[];
    expect(color?.clearValue).toEqual([0, 0, 0.5, 1]);
  });

  it('clears and tests depth as its options say, and clears a stencil with it', () => {
    const { root } = recordingRoot();
    const screen = targets.create(root, size, {
      hdr: { format: 'rgba16float' },
      depth: { format: 'depth24plus-stencil8' },
    });
    const pass = passes.create(
      root,
      screen,
      { hdr: passes.color, depth: passes.depth },
      { depthClear: 0, depthCompare: 'greater', stencilClear: 3 },
    );
    const { encoder, seen } = watched();

    pass.begin(encoder);
    expect(seen.descriptor!.depthStencilAttachment).toMatchObject({
      depthLoadOp: 'clear',
      depthClearValue: 0,
      stencilLoadOp: 'clear',
      stencilStoreOp: 'store',
      stencilClearValue: 3,
    });
    expect(stagesOf(pass, { Vertex, at, pixelOf, color: shaded }).depthStencil).toMatchObject({
      depthCompare: 'greater',
    });

    pass.begin(encoder, { clear: false });
    expect(seen.descriptor!.depthStencilAttachment?.stencilLoadOp).toBe('load');
  });

  it('names its passes and the pipelines it builds after their labels', () => {
    const { root } = recordingRoot();
    const screen = targets.create(root, size, { hdr: { format: 'rgba16float' } });
    const pass = passes.create(root, screen, { hdr: passes.color }, { label: 'scene' });
    const { encoder, seen } = watched();

    pass.pipe((cfg) => cfg).begin(encoder);
    expect(seen.descriptor!.label).toBe('scene');

    const drawing = { Vertex: Flat, at: flatAt, pixelOf: flatSurface, color: shaded };
    expect(String(pass.pipeline({ ...drawing, label: 'flat' }))).toBe('renderPipeline:flat');
    expect(String(pass.pipeline(drawing))).not.toContain('flat');
  });

  it('draws into the region of targets below their full resolution', () => {
    const { root } = recordingRoot();
    const { pass } = scene(root, { resolution: 0.5 });
    const { encoder, seen } = watched();

    pass.begin(encoder);
    expect(seen.viewport).toEqual([0, 0, 32, 16, 0, 1]);
  });

  it('draws multisampled targets, resolving them where it is told', () => {
    const { root } = recordingRoot();
    const screen = targets.create(root, size, {
      depth: { format: 'depth24plus', samples: 4 },
      scene: { format: 'rgba16float', samples: 4 },
      hdr: { format: 'rgba16float', sampled: true },
    });
    const pass = passes.create(
      root,
      screen,
      { depth: passes.depth, scene: passes.color },
      { resolve: { scene: 'hdr' }, clear: [0, 0, 0, 1] },
    );
    const stages = stagesOf(pass, {
      Vertex: Flat,
      at: flatAt,
      pixelOf: flatSurface,
      color: shaded,
      depthStencil: { depthWriteEnabled: false, depthCompare: 'less' },
    });
    const { encoder, seen } = watched();
    pass.begin(encoder);

    expect(stages.multisample).toEqual({ count: 4 });
    expect(stages.depthStencil).toEqual({
      format: 'depth24plus',
      depthWriteEnabled: false,
      depthCompare: 'less',
    });
    expect(seen.descriptor!.colorAttachments).toEqual([
      {
        view: screen.views.scene,
        resolveTarget: screen.views.hdr,
        loadOp: 'clear',
        storeOp: 'discard',
        clearValue: [0, 0, 0, 1],
      },
    ]);
    expect(seen.descriptor!.depthStencilAttachment).toMatchObject({ depthStoreOp: 'discard' });
  });

  it('keeps a multisampled target for a later pass only when it is not transient', () => {
    const { root } = recordingRoot();
    const screen = targets.create(root, size, {
      depth: { format: 'depth24plus', samples: 4 },
      scene: { format: 'rgba16float', samples: 4, transient: false },
      hdr: { format: 'rgba16float', sampled: true },
    });
    const pass = passes.create(
      root,
      screen,
      { depth: passes.depth, scene: passes.color },
      { resolve: { scene: 'hdr' } },
    );
    const { encoder, seen } = watched();

    pass.begin(encoder, { clear: ['depth'] });
    const [kept] = seen.descriptor!.colorAttachments as GPURenderPassColorAttachment[];
    expect([kept!.loadOp, kept!.storeOp]).toEqual(['load', 'store']);
    expect(() => pass.begin(encoder, { clear: false })).toThrow(
      "begin keeps what 'depth' holds, but it is transient",
    );
  });

  it('draws into a surface alone, from the frame', () => {
    const { root } = recordingRoot();
    const view = {} as GPUTextureView;
    const canvas = passes.create(
      root,
      { format: 'bgra8unorm', width: 4, height: 4, view: () => view },
      { clear: [0, 0, 0.1, 1] },
    );
    const stages = stagesOf(canvas, {
      Vertex: Flat,
      at: flatAt,
      pixelOf: flatSurface,
      color: shaded,
    });
    const { encoder, seen } = watched();

    expect(stages.targets).toEqual({ color: { format: 'bgra8unorm' } });
    expect(stages.depthStencil).toBeUndefined();
    canvas.begin({ encoder } as never);
    expect(seen.descriptor!.colorAttachments).toEqual([
      { view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0.1, 1] },
    ]);
    expect(() => canvas.begin(encoder)).toThrow('begins from the frame');
  });

  it('refuses what it can’t draw', () => {
    const { root } = recordingRoot();
    const screen = targets.create(root, size, {
      a: { format: 'rgba8unorm' },
      b: { format: 'rgba8unorm' },
      many: { format: 'rgba8unorm', samples: 4 },
      wide: { format: 'rgba16float', samples: 4 },
    });

    expect(() => passes.create(root, screen, { a: passes.color, b: passes.color })).toThrow(
      'at most one target marked `passes.color`',
    );
    // @ts-expect-error `c` is not a target of the set
    expect(() => passes.create(root, screen, { c: passes.color })).toThrow(
      "the set has no target 'c'",
    );
    expect(() => passes.create(root, screen, { many: passes.color, a: shaded })).toThrow(
      'the same sample count for every target it draws into',
    );
    expect(() => passes.create(root, screen, { a: passes.color }, { resolve: { a: 'b' } })).toThrow(
      "cannot resolve 'a' because it is not multisampled",
    );
    expect(() =>
      passes.create(root, screen, { many: passes.color }, { resolve: { many: 'wide' } }),
    ).toThrow("resolve destination 'wide' for 'many' to be an existing target with one sample");
    expect(() =>
      passes.create(root, screen, { wide: passes.color }, { resolve: { wide: 'a' } }),
    ).toThrow("the resolve destination for 'wide' to have the same format");
    expect(() =>
      passes.output(d.vec2f, {
        vary: () => undefined,
        write: () => d.vec2f(),
      } as never),
    ).toThrow('requires the `varyings` struct returned by `vary`');
    expect(() =>
      passes.output(d.vec2f, {
        varyings: Clips,
        vary: (current: { position: d.v4f }, previous: { position: d.v4f }) =>
          Clips({ current: current.position, previous: previous.position }),
        write: () => d.vec2f(),
      }),
    ).toThrow("say `previous: true` to give it last frame's vertex");
  });

  it('asks a drawing for `color` only where a target is marked `passes.color`', () => {
    const { root } = recordingRoot();
    const screen = targets.create(root, size, {
      hdr: { format: 'rgba16float' },
      id: { format: 'r32uint' },
      depth: { format: 'depth24plus' },
    });
    const gbuffer = passes.create(root, screen, { id, depth: passes.depth });
    expect(Object.keys(stagesOf(gbuffer, { Vertex, at, pixelOf }).targets)).toEqual(['id']);
    expect(() => stagesOf(gbuffer, { Vertex, at })).toThrow('requires `pixelOf`');

    const lit = passes.create(root, screen, { hdr: passes.color, id, depth: passes.depth });
    expect(() => stagesOf(lit, { Vertex, at, pixelOf })).toThrow(
      "requires `color` for a drawing into 'hdr', the target marked `passes.color`",
    );
  });

  it('gives last frame’s vertex to a `vary` made with `tgpu.fn`, which hides its parameters', () => {
    const { root } = recordingRoot();
    const { screen } = scene(root);
    const vary = tgpu.fn((current: { position: d.v4f }, previous: { position: d.v4f }) => {
      'use gpu';
      return Clips({ current: current.position, previous: previous.position });
    });
    expect(vary.length).toBe(0);
    const wrapped = passes.output(d.vec2f, {
      varyings: Clips,
      previous: true,
      vary,
      write: (_pixel: unknown, clips) => {
        'use gpu';
        return uvOf(clips.previous) - uvOf(clips.current);
      },
    });
    const pass = passes.create(root, screen, { hdr: passes.color, motion: wrapped });
    const wgsl = compiled(root, stagesOf(pass, { Vertex, at, pixelOf, color: shaded }));

    expect(wgsl.match(/fn at(_\d+)?\(/g)).toHaveLength(2);
  });

  it('builds a drawing it adds through what is piped into it, after the drawing’s own', async () => {
    const { root, log } = recordingRoot();
    const view = cameras.orbit(root, { size, target: [0, 0, 0], distance: 5 });
    const screen = targets.create(root, size, {
      hdr: { format: 'rgba16float' },
      depth: { format: 'depth24plus' },
    });
    // The drawing reads the camera, which only the description fills.
    const pass = passes
      .create(root, screen, { hdr: passes.color, depth: passes.depth })
      .pipe(view.inject());
    let built = 0;
    const drawn = pass.add({
      Vertex,
      at,
      pixelOf,
      color: shaded,
      primitive: { cullMode: 'back' },
      draw: (begun) => begun.draw(3, 4),
      initAsync: async () => {
        built += 1;
      },
    });
    const plain = passes.create(root, screen, { hdr: passes.color, depth: passes.depth });
    plain.add({ Vertex, at, pixelOf, color: shaded, draw: () => {} });

    await pass.initAsync();
    expect(built).toBe(1);
    expect(log.pipelines).toBe(1);
    await expect(plain.initAsync()).rejects.toThrow('camera');

    const encoder = root.createCommandEncoder();
    const begun = pass.begin(encoder);
    drawn.draw(begun);
    begun.end();
    expect(log.events.filter((name) => name === 'draw')).toHaveLength(1);
  });

  it('sets the pipeline and its groups once for draws of one drawing into a pass', () => {
    const { root, log } = recordingRoot();
    const view = cameras.orbit(root, { size, target: [0, 0, 0], distance: 5 });
    const screen = targets.create(root, size, { hdr: { format: 'rgba16float' } });
    const colored = passes.create(root, screen, { hdr: passes.color }).pipe(view.inject());
    const drawn = colored.add({
      Vertex,
      at,
      pixelOf,
      color: shaded,
      draw: (into) => into.draw(3),
    });

    const begun = colored.begin(root.createCommandEncoder());
    drawn.draw(begun);
    const once = [...log.state];
    drawn.draw(begun);
    drawn.draw(begun);
    begun.end();
    expect(log.events.filter((name) => name === 'draw')).toHaveLength(3);
    expect(once.filter((name) => name === 'set-pipeline')).toHaveLength(1);
    expect(log.state).toEqual(once);
  });

  it('compiles a destroyed drawing no more, and frees what it built', async () => {
    const { root } = recordingRoot();
    const view = cameras.orbit(root, { size, target: [0, 0, 0], distance: 5 });
    const screen = targets.create(root, size, { hdr: { format: 'rgba16float' } });
    const pass = passes.create(root, screen, { hdr: passes.color }).pipe(view.inject());
    let built = 0;
    let freed = 0;
    const drawn = pass.add({
      Vertex,
      at,
      pixelOf,
      color: shaded,
      draw: (begun) => begun.draw(3),
      initAsync: async () => {
        built += 1;
      },
    });
    const compound = pass.add({
      build: () => ({
        draw: () => {},
        initAsync: async () => {
          built += 1;
        },
        destroy: () => {
          freed += 1;
        },
      }),
    });

    await pass.initAsync();
    expect(built).toBe(2);
    drawn.destroy();
    compound[Symbol.dispose]();
    await pass.initAsync();
    expect(built).toBe(2);
    expect(freed).toBe(1);
  });

  it('destroys every drawing added to it and to what it made', () => {
    const { root } = recordingRoot();
    const screen = targets.create(root, size, {
      hdr: { format: 'rgba16float' },
      motion: { format: 'rg16float' },
    });
    const pass = passes.create(root, screen, { hdr: passes.color, motion });
    let freed = 0;
    const compound = (): passes.Compound => ({
      build: () => ({
        draw: () => {},
        initAsync: async () => {},
        destroy: () => {
          freed += 1;
        },
      }),
    });
    pass.add(compound());
    pass.without('motion').add(compound());
    pass.add(compound()).destroy();
    expect(freed).toBe(1);

    pass.destroy();
    expect(freed).toBe(3);
  });

  it('builds depth alone without a fragment stage, unless a drawing writes its own depth', () => {
    const { root } = recordingRoot();
    const map = targets.create(root, size, { depth: { format: 'depth16unorm', sampled: true } });
    const shadow = passes.create(
      root,
      map,
      { depth: passes.depth },
      { bias: { constant: 2, slope: 2 }, enabled: () => false },
    );
    expect(shadow.depthOnly).toBe(true);
    expect(shadow.enabled).toBe(false);

    const wgsl = tgpu.resolve([shadow.pipeline({ Vertex: Flat, at: flatAt })]);
    expect(wgsl).not.toContain('@fragment');
    expect(stagesOf(shadow, { Vertex: Flat, at: flatAt }).fragment).toBeUndefined();

    const written = stagesOf(shadow, {
      Vertex: Flat,
      at: flatAt,
      pixelOf: flatSurface,
      depthOf: () => {
        'use gpu';
        return d.f32(0.25);
      },
    });
    expect(written.targets).toEqual({});
    expect(written.depthStencil).toEqual({
      format: 'depth16unorm',
      depthWriteEnabled: true,
      depthCompare: 'less',
      depthBias: 2,
      depthBiasSlopeScale: 2,
    });
    const writing = tgpu.resolve([root.createRenderPipeline(written)]);
    expect(writing).toContain('@builtin(frag_depth)');
    // Two steps of a 16-bit depth and twice the slope, as the rasteriser adds them.
    expect(writing).toContain('dpdx');
    expect(writing).toContain('(slope * 2f)) + 3.0517578125e-5f');
  });

  it('writes no depth of its own into a description without a depth target', () => {
    const { root } = recordingRoot();
    const plain = targets.create(root, size, { hdr: { format: 'rgba16float' } });
    const flat = passes.create(root, plain, { hdr: passes.color });
    const stages = stagesOf(flat, {
      Vertex: Flat,
      at: flatAt,
      pixelOf: flatSurface,
      color: shaded,
      depthOf: () => {
        'use gpu';
        return d.f32(0.25);
      },
    });

    expect(stages.depthStencil).toBeUndefined();
    expect(compiled(root, stages)).not.toContain('frag_depth');
  });

  it('writes a sample mask, but not beside alpha to coverage', () => {
    const { root } = recordingRoot();
    const many = targets.create(root, size, { hdr: { format: 'rgba16float', samples: 4 } });
    const flat = passes.create(root, many, { hdr: passes.color });
    const masked = {
      Vertex: Flat,
      at: flatAt,
      pixelOf: flatSurface,
      color: shaded,
      sampleMask: () => {
        'use gpu';
        return d.u32(0b0101);
      },
    };

    const stages = stagesOf(flat, masked);
    expect(stages.multisample).toEqual({ count: 4 });
    expect(compiled(root, stages)).toContain('@builtin(sample_mask)');
    expect(() => stagesOf(flat, { ...masked, alphaToCoverage: true })).toThrow(
      'takes `sampleMask` or `alphaToCoverage`, not both',
    );
  });

  it('cuts a shape out before its surface, and into depth alone without the surface', () => {
    const { root } = recordingRoot();
    const lit = tgpu.accessor(d.f32);
    const Sprite = d.struct({ position: d.vec4f, uv: d.vec2f });
    const sprite = {
      Vertex: Sprite,
      at: (vertexIndex: number, _instanceIndex: number) => {
        'use gpu';
        const corner = flatAt(vertexIndex, 0).position;
        return Sprite({ position: corner, uv: corner.xy });
      },
      cutout: (vertex: d.Infer<typeof Sprite>) => {
        'use gpu';
        return std.length(vertex.uv) > 1;
      },
      // Shading reads a fill that only colour descriptions give
      pixelOf: (_vertex: d.Infer<typeof Sprite>) => {
        'use gpu';
        return Surface({ world: d.vec3f(), id: 0, color: d.vec4f(lit.$) });
      },
      color: shaded,
    };

    const map = targets.create(root, size, { depth: { format: 'depth16unorm' } });
    const shadow = passes.create(root, map, { depth: passes.depth });
    const casting = compiled(root, stagesOf(shadow, sprite));
    expect(casting).toContain('discard;');
    expect(casting).not.toContain('frag_depth');

    const { pass } = scene(root);
    const view = cameras.orbit(root, { size, target: [0, 0, 0], distance: 5 });
    const pipeline = root
      .pipe(view.inject())
      .with(lit, 2)
      .createRenderPipeline(stagesOf(pass.without('motion', 'id'), sprite));
    const shading = tgpu.resolve([pipeline]);
    const fragment = shading.slice(shading.indexOf('@fragment'));
    expect(fragment.indexOf('discard;')).toBeGreaterThan(0);
    expect(fragment.indexOf('discard;')).toBeLessThan(fragment.indexOf('pixelOf('));
  });

  it('builds a compound drawing into each description it is added to', async () => {
    const { root, log } = recordingRoot();
    const { pass } = scene(root);
    const canvas = passes.create(root, {
      format: 'bgra8unorm',
      width: 4,
      height: 4,
      view: () => ({}) as GPUTextureView,
    });
    const seen: passes.Into[] = [];
    const compound: passes.Compound<{ color: d.v4f }, { position: d.v4f }> = {
      build(into) {
        seen.push(into);
        const drawing = { Vertex: Flat, at: flatAt, pixelOf: flatSurface, color: shaded };
        const pipeline = into.pipeline(drawing);
        return {
          draw: (begun) => {
            begun.setPipeline(pipeline);
            begun.draw(3);
          },
          initAsync: () => pipeline.initAsync(),
        };
      },
    };

    pass.without('id').add(compound);
    canvas.add(compound);
    expect(seen.map((each) => Object.keys(each.writes))).toEqual([
      ['hdr', 'motion', 'depth'],
      ['color'],
    ]);
    await Promise.all([pass.initAsync(), canvas.initAsync()]);
    expect(log.pipelines).toBe(2);
  });

  it('refuses a renderer that leaves out or replaces what the description doesn’t write', () => {
    const { root } = recordingRoot();
    const { pass } = scene(root);
    const drawing = { Vertex: Flat, at: flatAt, pixelOf: flatSurface, color: shaded };

    expect(() => pass.without('hdr' as never)).toThrow(
      "a renderer left out 'hdr', which isn't one of its outputs",
    );
    expect(() => pass.instead({ hdr: shaded } as never)).toThrow(
      "a renderer replaced 'hdr', which isn't one of its outputs",
    );
    const Clashing = d.struct({ position: d.vec4f, motion_current: d.vec4f });
    expect(() =>
      stagesOf(pass, {
        ...drawing,
        Vertex: Clashing,
        at: () => {
          'use gpu';
          return Clashing();
        },
        pixelOf: () => {
          'use gpu';
          return Surface();
        },
      }),
    ).toThrow("reserves vertex field 'motion_current' for stage outputs");
  });
});

/** A compound drawing that carries pixel `P` and vertex `X` and builds nothing. */
const compound = <P, X>(): passes.Compound<P, X> => ({
  build: () => ({ draw: () => {}, initAsync: async () => {} }),
});

it('type-checks a drawing against the outputs it can feed', () => {
  const { root } = recordingRoot();
  const { pass } = scene(root);

  pass.add(compound<{ id: number; color: d.v4f }, { position: d.v4f }>());

  const sparks = compound<{ color: d.v4f }, { position: d.v4f }>();
  // @ts-expect-error -- sparks have no id, and the error names the target to leave out.
  pass.add(sparks);
  pass.without('id').add(sparks);

  const sky = compound<{ color: d.v4f }, {}>();
  // @ts-expect-error -- a full-screen pass has no vertex for motion either.
  pass.without('id').add(sky);
  pass.without('id', 'motion').add(sky);
  const still = passes.output(d.vec2f, {
    write: () => {
      'use gpu';
      return d.vec2f();
    },
  });
  pass.without('id').instead({ motion: still }).add(sky);
  // A description typed without its targets takes any drawing.
  const any: passes.Description = pass;
  any.add(sky);
  // @ts-expect-error -- the colour isn't an output to leave out.
  const unwritable = () => pass.without('hdr');
  expect(unwritable).toThrow("left out 'hdr'");

  // The canvas alone feeds any renderer.
  const canvas = passes.create(root, {
    format: 'bgra8unorm',
    width: 4,
    height: 4,
    view: () => ({}) as GPUTextureView,
  });
  canvas.add(sky);

  // A drawing written as a value type-checks the same way.
  const embers = {
    Vertex: Flat,
    at: flatAt,
    pixelOf: (_vertex: d.Infer<typeof Flat>) => {
      'use gpu';
      return Glow({ color: d.vec4f(1) });
    },
    color: shaded,
    draw: () => {},
  };
  // @ts-expect-error -- embers have no id.
  pass.add(embers);
  pass.without('id').add(embers);
  canvas.add(embers);
  // `passes.drawing` infers the vertex and pixel, so the callbacks need no annotations.
  const inferred = passes.drawing({
    Vertex: Flat,
    at: flatAt,
    pixelOf: (vertex) => {
      'use gpu';
      return Glow({ color: d.vec4f(vertex.position.z) });
    },
    color: shaded,
    draw: (into) => into.draw(3),
  });
  pass.without('id').add(inferred);

  passes.output(d.vec2f, {
    // @ts-expect-error -- a clear value of another type.
    clear: 1,
    write: () => d.vec2f(),
  });
});

/** Targets for translucent drawings, beside colour and depth. */
const layers = (root: TgpuRoot, sampled = true) =>
  targets.create(root, size, {
    hdr: { format: 'rgba16float', sampled: true },
    accum: { format: 'rgba16float', sampled },
    reveal: { format: 'r16float', sampled },
    depth: { format: 'depth24plus' },
  });

describe('translucent drawings', () => {
  const marked = {
    hdr: passes.color,
    accum: passes.accumulation,
    reveal: passes.revealage,
    depth: passes.depth,
  } as const;
  const drawing = { Vertex: Flat, at: flatAt, pixelOf: flatSurface, color: shaded };

  it('adds into accumulation and multiplies revealage, writing no colour and testing depth without writing it', () => {
    const { root } = recordingRoot();
    const layered = passes.create(root, layers(root), marked);

    const glass = stagesOf(layered, { ...drawing, translucent: true });
    expect(glass.targets).toEqual({
      hdr: { format: 'rgba16float', writeMask: 0 },
      accum: {
        format: 'rgba16float',
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        },
      },
      reveal: {
        format: 'r16float',
        blend: {
          color: { srcFactor: 'zero', dstFactor: 'one-minus-src', operation: 'add' },
          alpha: { srcFactor: 'zero', dstFactor: 'one-minus-src', operation: 'add' },
        },
      },
    });
    expect(glass.depthStencil).toMatchObject({ depthWriteEnabled: false, depthCompare: 'less' });
    expect(compiled(root, glass)).toContain('accum_depth');

    // An opaque drawing in the same description leaves the translucency targets alone
    const solid = stagesOf(layered, drawing);
    expect(solid.targets).toEqual({
      hdr: { format: 'rgba16float' },
      accum: { format: 'rgba16float', writeMask: 0 },
      reveal: { format: 'r16float', writeMask: 0 },
    });
    expect(solid.depthStencil).toMatchObject({ depthWriteEnabled: true });
    expect(compiled(root, solid)).not.toContain('accum_depth');
  });

  it('blends over in draw order without the translucency targets', () => {
    const { root } = recordingRoot();
    const plain = passes.create(root, layers(root), { hdr: passes.color, depth: passes.depth });
    const glass = stagesOf(plain, { ...drawing, translucent: true });
    expect(glass.targets).toEqual({
      hdr: { format: 'rgba16float', blend: pipelines.blend.over },
    });
    expect(glass.depthStencil).toMatchObject({ depthWriteEnabled: false });
    expect(() => plain.composite(root.createCommandEncoder())).toThrow(
      'composite needs targets marked `passes.accumulation` and `passes.revealage`',
    );
  });

  it('composites in one pass of its own, over what the colour target holds', () => {
    const { root, log } = recordingRoot();
    const layered = passes.create(root, layers(root), marked);
    const encoder = root.createCommandEncoder();
    const from = log.events.length;
    layered.composite(encoder);
    expect(log.events.slice(from)).toEqual(['begin-render', 'draw', 'end-render']);
  });

  it('records and composites nothing while disabled, though begin still opens a pass', () => {
    const { root, log } = recordingRoot();
    let on = false;
    const layered = passes.create(root, layers(root), marked, { enabled: () => on });
    const encoder = root.createCommandEncoder();
    const from = log.events.length;
    layered.record(encoder, []);
    layered.composite(encoder);
    expect(log.events.slice(from)).toEqual([]);

    layered.begin(encoder).end();
    on = true;
    layered.record(encoder, []);
    expect(log.events.slice(from)).toEqual([
      'begin-render',
      'end-render',
      'begin-render',
      'end-render',
    ]);
  });

  it('refuses translucency it can’t draw', () => {
    const { root } = recordingRoot();
    const screen = layers(root);
    expect(() =>
      passes.create(root, screen, { hdr: passes.color, accum: passes.accumulation }),
    ).toThrow('`passes.accumulation` and `passes.revealage` together, beside `passes.color`');
    expect(() =>
      passes.create(root, screen, { accum: passes.accumulation, reveal: passes.revealage }),
    ).toThrow('beside `passes.color`');
    expect(() => passes.create(root, layers(root, false), marked)).toThrow(
      "requires 'accum' to be sampled: true",
    );

    const layered = passes.create(root, screen, marked);
    expect(() => layered.without('accum')).toThrow("'accum' holds translucent drawings");
    expect(() =>
      stagesOf(layered, { ...drawing, translucent: true, blend: pipelines.blend.over }),
    ).toThrow('leave out `blend`');
    const shadow = passes.create(root, screen, { depth: passes.depth });
    expect(() => stagesOf(shadow, { ...drawing, translucent: true })).toThrow(
      'a translucent drawing leaves depth alone',
    );
  });
});

const picture = (root: TgpuRoot) =>
  root.createTexture({ size: [1, 1], format: 'rgba8unorm' }).$usage('sampled');

describe('a shared layout', () => {
  const Tinted = bindings.layout({ tint: { texture: d.texture2d(d.f32) } });
  const tinted = (_vertex: d.Infer<typeof Flat>) => {
    'use gpu';
    return Surface({
      world: d.vec3f(),
      id: 0,
      color: std.textureLoad(Tinted.$.tint, d.vec2i(), 0),
    });
  };
  function shared(given: 'layout' | 'parts' | 'elsewhere') {
    const recording = recordingRoot();
    const { root } = recording;
    const part = Tinted.bind({ tint: picture(root) });
    let groups = 0;
    const counted = (): bindings.Shared => {
      const share = bindings.share(root, { tinted: part });
      return {
        ...share,
        group: () => {
          groups += 1;
          return share.group();
        },
      };
    };
    const options: passes.Options =
      given === 'parts' ? { shared: { tinted: part } } : { shared: counted() };
    const screen = targets.create(root, size, { hdr: { format: 'rgba16float' } });
    const pass = passes.create(
      root,
      screen,
      { hdr: passes.color },
      given === 'elsewhere' ? {} : options,
    );
    const drawn = pass.add({
      Vertex: Flat,
      at: flatAt,
      pixelOf: tinted,
      color: shaded,
      inject: () => part.inject(),
      draw: (begun) => begun.draw(3),
    });
    return { ...recording, part, pass, drawn, groups: () => groups };
  }

  it('sets its group in every pass the description begins, so record draws through it', () => {
    const { root, log, pass, drawn, groups } = shared('layout');
    const encoder = root.createCommandEncoder();
    pass.record(encoder, drawn);
    pass.record(encoder, [drawn]);
    pass.begin(encoder).end();

    expect(groups()).toBe(3);
    expect(log.events.filter((name) => name === 'draw')).toHaveLength(2);
  });

  it('binds a replaced resource with a new group and no new pipeline', () => {
    const { root, log, part, pass, drawn } = shared('layout');
    const encoder = root.createCommandEncoder();
    pass.record(encoder, drawn);
    const before = log.counts();

    pass.record(encoder, drawn);
    expect(log.counts()).toEqual(before);

    part.replace({ tint: picture(root) });
    pass.record(encoder, drawn);
    expect(log.counts()).toEqual({
      ...before,
      textures: before.textures + 1,
      views: before.views + 1,
      bindGroups: before.bindGroups + 1,
    });
  });

  it('shares parts given by key, in a layout of its own', () => {
    const { root, log, part, pass, drawn } = shared('parts');
    const encoder = root.createCommandEncoder();
    pass.record(encoder, drawn);
    const before = log.counts();

    part.replace({ tint: picture(root) });
    pass.record(encoder, drawn);
    expect(log.counts().bindGroups).toBe(before.bindGroups + 1);
    expect(log.counts().pipelines).toBe(before.pipelines);
    expect(log.events.filter((name) => name === 'draw')).toHaveLength(2);
    expect(() => bindings.share(root, { tinted: part })).toThrow('already has an owner');
  });

  it('throws at draw when the shared group was not set', async () => {
    const { root, pass, drawn } = shared('elsewhere');
    await pass.initAsync();
    expect(() => drawn.draw(pass.begin(root.createCommandEncoder()))).toThrow(
      'Missing bind groups for layouts',
    );
  });
});

/**
 * Records each render pass begun from now on, as WebGPU sees it: its descriptor and the layer
 * index in every bind group it sets that holds one u32.
 */
function watchLayers(recording: ReturnType<typeof recordingRoot>) {
  const { root, bytes } = recording;
  const device = root.device;
  const indexOf = new Map<object, number>();
  const begun: { descriptor: GPURenderPassDescriptor; layers: number[] }[] = [];

  const createBindGroup = device.createBindGroup.bind(device);
  device.createBindGroup = (descriptor) => {
    const group = createBindGroup(descriptor);
    const [entry, ...rest] = [...descriptor.entries];
    const resource = entry?.resource as { buffer?: GPUBuffer } | undefined;
    if (rest.length === 0 && resource?.buffer?.size === 4) {
      indexOf.set(group, new Uint32Array(bytes(resource.buffer).slice().buffer)[0]);
    }
    return group;
  };

  const createEncoder = device.createCommandEncoder.bind(device);
  device.createCommandEncoder = (descriptor) => {
    const encoder = createEncoder(descriptor);
    const begin = encoder.beginRenderPass.bind(encoder);
    encoder.beginRenderPass = (passDescriptor) => {
      const pass = begin(passDescriptor);
      const seen = { descriptor: passDescriptor, layers: [] as number[] };
      begun.push(seen);
      const set = pass.setBindGroup.bind(pass) as (...args: unknown[]) => void;
      pass.setBindGroup = ((index: number, group: GPUBindGroup, ...rest: unknown[]) => {
        const layer = indexOf.get(group);
        if (layer !== undefined) seen.layers.push(layer);
        set(index, group, ...rest);
      }) as never;
      return pass;
    };
    return encoder;
  };

  return begun;
}

function layeredScene(count: number) {
  const recording = recordingRoot();
  const { root } = recording;
  const begun = watchLayers(recording);
  const screen = targets.create(root, size, {
    depth: { format: 'depth16unorm', layers: count, sampled: true },
  });
  const views = cameras.layered(root, count);
  const casters = passes
    .create(root, screen, { depth: passes.depth }, { label: 'casters' })
    .pipe(views.inject());
  return { ...recording, begun, screen, views, scene: casters };
}

describe('layers', () => {
  const casterAt = (vertexIndex: number, _instanceIndex: number) => {
    'use gpu';
    const corners = [d.vec3f(-1, -1, 0), d.vec3f(1, -1, 0), d.vec3f(0, 1, 0)];
    return Flat({ position: cameras.current.$.viewProjection * d.vec4f(corners[vertexIndex], 1) });
  };
  const caster = {
    label: 'caster',
    Vertex: Flat,
    at: casterAt,
    pixelOf: flatSurface,
    draw: (pass: { draw(count: number): void }) => pass.draw(3),
  };

  it('builds pipelines once and records a pass into each layer, binding its index', () => {
    const one = layeredScene(1);
    one.scene.record(one.root.createCommandEncoder(), one.scene.add(caster));
    const three = layeredScene(3);
    const drawn = three.scene.add(caster);
    const encoder = three.root.createCommandEncoder();
    three.scene.record(encoder, drawn);
    expect(three.log.pipelines).toBe(one.log.pipelines);

    const steady = three.log.counts();
    three.scene.record(encoder, drawn);
    expect(three.log.counts()).toEqual(steady);

    expect(three.log.events.filter((name) => name === 'begin-render')).toHaveLength(6);
    expect(three.log.events.filter((name) => name === 'draw')).toHaveLength(6);
    expect(three.begun.map((each) => each.layers)).toEqual([[0], [1], [2], [0], [1], [2]]);
    expect(three.begun.map(({ descriptor }) => descriptor.label)).toEqual([
      'casters 0',
      'casters 1',
      'casters 2',
      'casters 0',
      'casters 1',
      'casters 2',
    ]);
    expect(three.begun.map(({ descriptor }) => descriptor.depthStencilAttachment?.view)).toEqual([
      ...three.screen.layers.depth,
      ...three.screen.layers.depth,
    ]);
    expect(three.scene.layers).toBe(3);
  });

  it("gives a compound drawing's draw the layer of each pass", () => {
    const { root, scene: casters } = layeredScene(3);
    const seen: number[] = [];
    const drawn = casters.add({
      build: () => ({
        draw: (_pass, layer) => seen.push(layer),
        initAsync: async () => {},
      }),
    });
    const encoder = root.createCommandEncoder();

    casters.record(encoder, drawn);
    const pass = casters.begin(encoder, { layer: 1 });
    drawn.draw(pass);
    pass.end();
    expect(seen).toEqual([0, 1, 2, 1]);
  });

  it('throws as a pipeline builds when its layered camera holds another number of views', () => {
    const { root } = recordingRoot();
    const screen = targets.create(root, size, { depth: { format: 'depth16unorm', layers: 3 } });
    const casters = passes
      .create(root, screen, { depth: passes.depth })
      .pipe(cameras.layered(root, 2).inject());
    const drawn = casters.add(caster);
    expect(() => casters.record(root.createCommandEncoder(), drawn)).toThrow(
      'holds 2 views, but the description',
    );
  });

  it('records or begins one layer when given, and begins none without it', () => {
    const { root, begun, scene: casters } = layeredScene(3);
    const drawn = casters.add(caster);
    const encoder = root.createCommandEncoder();

    casters.record(encoder, drawn, { layer: 1 });
    const pass = casters.begin(encoder, { layer: 2 });
    drawn.draw(pass);
    pass.end();
    expect(begun.map((each) => each.layers)).toEqual([[1], [2]]);

    expect(() => casters.begin(encoder)).toThrow('begins one layer at a time');
    expect(() => casters.begin(encoder, { layer: 3 })).toThrow(RangeError);
  });

  it('times a record over every layer from the start of the first pass to the end of the last', () => {
    const { root, begun, scene: casters } = layeredScene(3);
    const drawn = casters.add(caster);
    const encoder = root.createCommandEncoder();
    const querySet = root.device.createQuerySet({ type: 'timestamp', count: 2 });
    const timestampWrites = { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 };

    casters.record(encoder, drawn, { timestampWrites });
    casters.record(encoder, drawn, { timestampWrites, layer: 1 });
    expect(begun.map(({ descriptor }) => descriptor.timestampWrites)).toEqual([
      { querySet, beginningOfPassWriteIndex: 0 },
      undefined,
      { querySet, endOfPassWriteIndex: 1 },
      timestampWrites,
    ]);
  });

  it('draws particles, trails and text into every layer, through the view of each', () => {
    const recording = recordingRoot();
    const { root, log } = recording;
    const begun = watchLayers(recording);
    const screen = targets.create(root, size, {
      color: { format: 'rgba8unorm', layers: 2 },
      depth: { format: 'depth24plus', layers: 2 },
    });
    const own = cameras.custom(root);
    const views = cameras.layered(root, 2);
    const into = passes
      .create(root, screen, { color: passes.color, depth: passes.depth })
      .pipe(views.inject());
    const sparks = particles.create(root, own, {
      capacity: 64,
      step: 1 / 60,
    });
    const wakes = trails.create(root, { capacity: 4, length: 8 });
    const labels = text.create(root, font.create(root, inter), {
      camera: own,
      capacity: 2,
      length: 4,
    });
    const drawn = [into.add(sparks.drawing), into.add(wakes.drawing), into.add(labels.drawing)];

    const encoder = root.createCommandEncoder();
    const compute = encoder.beginComputePass();
    sparks.compact(compute, 0);
    compute.end();
    into.record(encoder, drawn);
    expect(begun.map((each) => each.layers[0])).toEqual([0, 1]);
    expect(log.shaders.filter((code) => code.includes('array<Camera, 2>'))).toHaveLength(3);
  });

  it('takes its layers from its targets, which must agree', () => {
    const { root } = recordingRoot();
    const deep = targets.create(root, size, { depth: { format: 'depth16unorm', layers: 2 } });
    expect(passes.create(root, deep, { depth: passes.depth }).layers).toBe(2);

    const mixed = targets.create(root, size, {
      color: { format: 'rgba8unorm', layers: 3 },
      flat: { format: 'rgba8unorm' },
      depth: { format: 'depth16unorm', layers: 2 },
    });
    expect(() => passes.create(root, mixed, { color: passes.color, depth: passes.depth })).toThrow(
      "requires the same `layers` in targets.create for every target it draws into, got 'color' 3, 'depth' 2.",
    );
    expect(() => passes.create(root, mixed, { flat: passes.color, depth: passes.depth })).toThrow(
      "got 'flat' none, 'depth' 2.",
    );
  });
});

/** What the last pass wrote into its viewport: width, height and pixels per point. */
const lastOf = (recording: ReturnType<typeof recordingRoot>) => {
  const write = recording.log.writes.findLast(({ buffer }) => buffer.label.includes('viewport'));
  return write && [...new Float32Array(recording.bytes(write.buffer).slice(0, 12).buffer)];
};

describe('viewport', () => {
  const Region = d.struct({ position: d.vec4f });

  const Shown = d.struct({ color: d.vec4f });
  const blank = () => {
    'use gpu';
    return Shown({ color: d.vec4f() });
  };

  /** A drawing whose vertex reads `passes.viewport`, so its pipeline binds it. */
  const sized = {
    Vertex: Region,
    at: () => {
      'use gpu';
      const view = passes.viewport.$;
      return Region({ position: d.vec4f(view.size * view.pixelRatio, 0, 1) });
    },
    draw: (pass: TgpuRenderCommands) => pass.draw(3),
  };

  it('holds the surface size and its pixel ratio, written only when either changes', () => {
    const recording = recordingRoot();
    const { root } = recording;
    const surface = {
      format: 'bgra8unorm' as const,
      width: 64,
      height: 32,
      pixelRatio: 2,
      view: () => ({}) as GPUTextureView,
    };
    const canvas = passes.create(root, surface);
    const drawn = canvas.add({ ...sized, pixelOf: blank, color: passes.shaded });
    const frame = () => canvas.record({ encoder: root.createCommandEncoder() } as Frame, drawn);

    frame();
    expect(lastOf(recording)).toEqual([64, 32, 2]);
    const writes = recording.log.writes.length;
    frame();
    expect(recording.log.writes.length).toBe(writes);

    surface.width = 128;
    surface.pixelRatio = 1;
    frame();
    expect(lastOf(recording)).toEqual([128, 32, 1]);
  });

  it('refuses to begin into targets that no longer fit what they follow, naming `fit()`', () => {
    const { root } = recordingRoot();
    const surface = { width: 64, height: 32 };
    let resolution = 1;
    const screen = targets.create(
      root,
      surface,
      { color: { format: 'rgba16float' } },
      { resolution: () => resolution },
    );
    const into = passes.create(root, screen, { color: passes.color });
    const begin = () => into.begin(root.createCommandEncoder()).end();

    begin();
    surface.width = 128;
    expect(screen.fitted).toBe(false);
    expect(begin).toThrow('Call `fit()` on the set once a frame');
    screen.fit();
    begin();

    resolution = 0.5;
    expect(begin).toThrow('no longer match the size or resolution');
    screen.fit();
    begin();
  });

  it("holds the drawn region of a low-resolution target, its points as large as the surface's", () => {
    const recording = recordingRoot();
    const { root } = recording;
    const surface = { width: 200, height: 100, pixelRatio: 2 };
    const low = targets.create(
      root,
      surface,
      { color: { format: 'rgba16float' } },
      { resolution: 0.5 },
    );
    const quarter = targets.create(root, targets.scaled(surface, 0.25), {
      color: { format: 'rgba16float' },
    });
    const drawing = { ...sized, pixelOf: blank, color: passes.shaded };

    for (const [screen, expected] of [
      [low, [100, 50, 1]],
      [quarter, [50, 25, 0.5]],
    ] as const) {
      const into = passes.create(root, screen, { color: passes.color });
      into.record(root.createCommandEncoder(), into.add(drawing));
      expect(lastOf(recording)).toEqual(expected);
    }
  });

  it('holds a shadow map its own size, one pixel to the point', () => {
    const recording = recordingRoot();
    const { root } = recording;
    const shadow = targets.create(
      root,
      { width: 1024, height: 1024 },
      {
        depth: { format: 'depth16unorm', sampled: true },
      },
    );
    const casters = passes.create(root, shadow, { depth: passes.depth });
    casters.record(root.createCommandEncoder(), casters.add(sized));
    expect(lastOf(recording)).toEqual([1024, 1024, 1]);
  });
});
