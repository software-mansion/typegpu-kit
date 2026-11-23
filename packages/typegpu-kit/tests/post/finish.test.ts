import { targets } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';
import { common, d, std, tgpu } from 'typegpu';
import { describe, expect, it, vi } from 'vitest';
import { bindings, bloom, fxaa, finish } from '../../src/index.ts';
import { aces, agx, luminance, neutral, reinhard } from '../../src/post/tone.ts';

function setup(size = { width: 64, height: 32 }) {
  const recording = recordingRoot();
  const { root } = recording;
  const screen = targets.create(root, size, { hdr: { format: 'rgba16float', sampled: true } });
  const source = () => screen.textures.hdr;

  return { recording, root, size, screen, source };
}

const passes = (events: string[]) => events.filter((event) => event === 'begin-render').length;

/** The view and timer writes of each render pass `root` begins, as they were when it began. */
function watchPasses(root: ReturnType<typeof recordingRoot>['root']) {
  const device = root.device;
  const begun: { view: unknown; timestampWrites: unknown }[] = [];
  const createEncoder = device.createCommandEncoder.bind(device);
  device.createCommandEncoder = (descriptor) => {
    const encoder = createEncoder(descriptor);
    const begin = encoder.beginRenderPass.bind(encoder);
    encoder.beginRenderPass = (pass) => {
      const [color] = [...pass.colorAttachments];
      begun.push({ view: color?.view, timestampWrites: pass.timestampWrites });
      return begin(pass);
    };
    return encoder;
  };
  return begun;
}

/** A grade that reads the picture beside its pixel, and the glow. */
const shifted = (_color: d.v3f, uv: d.v2f) => {
  'use gpu';
  return finish.picture(uv + 0.01) + finish.glow(uv);
};

describe('bloom', () => {
  it('downsamples through every level and adds them back up, in one encoder', async () => {
    const { recording, root, source } = setup();
    const glow = bloom.create(root, { source, levels: 4 });
    await glow.initAsync();
    expect(recording.log.pipelines).toBe(3);

    const encoder = root.createCommandEncoder();
    glow.record(encoder);
    encoder.submit();

    // Four levels down, three back up.
    expect(passes(recording.log.events)).toBe(7);
    expect(recording.log.submissions).toBe(1);
  });

  it('stops at the smallest level the source allows', () => {
    const { recording, root, source } = setup({ width: 16, height: 9 });
    const glow = bloom.create(root, { source, levels: 8 });

    const encoder = root.createCommandEncoder();
    glow.record(encoder);
    encoder.submit();

    // Half of 16×9 is 8×5, which has 4 levels.
    expect(passes(recording.log.events)).toBe(7);
  });

  it('follows a source that is rebuilt at a new size', () => {
    const { root, size, screen, source } = setup();
    const glow = bloom.create(root, { source });
    const before = glow.output;

    expect(glow.output).toBe(before);

    size.width = 128;
    screen.fit();
    expect(glow.output).not.toBe(before);
  });
});

describe('finish', () => {
  it('compiles one pipeline, choosing the tone curve and the filter when it is created', async () => {
    const { recording, root, source } = setup();
    const posts = [
      finish.create(root, { format: 'bgra8unorm', source, tone: aces, filter: 'sharp' }),
      finish.create(root, { format: 'bgra8unorm', source, tone: null }),
    ];
    await Promise.all(posts.map((final) => final.initAsync()));

    const [sharp, linear] = recording.log.shaders.filter((code) => code.includes('@fragment'));
    expect(recording.log.pipelines).toBe(2);
    expect(sharp).toContain('fwidth');
    expect(linear).not.toContain('fwidth');
  });

  it('starts its settings and its bloom’s at the settings options, over the defaults', () => {
    const { recording, root, source } = setup();
    const glow = bloom.create(root, { source, settings: { knee: 0.25 } });
    const final = finish.create(root, {
      format: 'bgra8unorm',
      source,
      settings: { dither: 0 },
      glow,
    });

    const floats = (buffer: GPUBuffer) => [
      ...new Float32Array(recording.bytes(buffer).slice().buffer),
    ];
    expect(floats(root.unwrap(final.settings.buffer))).toEqual([
      ...Float32Array.of(1, 0.35, 0, 0.12),
    ]);
    expect(floats(root.unwrap(glow.settings.buffer))).toEqual([1, 0.25, 1]);
  });

  it('patches one look into its settings and its bloom’s, each taking its own fields', () => {
    const { recording, root, source } = setup();
    const glow = bloom.create(root, { source });
    const final = finish.create(root, {
      format: 'bgra8unorm',
      source,
      settings: { vignette: 0.5 },
      glow,
    });

    const look = { exposure: 2, spread: 1.5 };
    final.settings.patch(look);
    glow.settings.patch(look);

    const floats = (buffer: GPUBuffer) => [
      ...new Float32Array(recording.bytes(buffer).slice().buffer),
    ];
    expect(floats(root.unwrap(final.settings.buffer))).toEqual([
      ...Float32Array.of(2, 0.5, 1, 0.12),
    ]);
    expect(floats(root.unwrap(glow.settings.buffer))).toEqual([1, 0.5, 1.5]);
  });

  it('takes only sources it can sample as floats', () => {
    const { root, size, source } = setup();
    const odd = targets.create(root, size, {
      ids: { format: 'r32uint', sampled: true },
      depth: { format: 'depth24plus', sampled: true },
      msaa: { format: 'rgba16float', samples: 4, sampled: true },
    });

    bloom.create(root, { source });
    // @ts-expect-error -- an integer texture samples as integers.
    bloom.create(root, { source: () => odd.textures.ids });
    // @ts-expect-error -- a depth texture samples as depth.
    finish.create(root, { format: 'bgra8unorm', source: () => odd.textures.depth });
    // @ts-expect-error -- a multisampled texture can't be sampled with a sampler.
    finish.create(root, { format: 'bgra8unorm', source: () => odd.textures.msaa });
  });

  it('takes a picture beside a glow, which may bloom another', () => {
    const { root, size, source } = setup();
    const emission = targets.create(root, size, {
      glowing: { format: 'rgba16float', sampled: true },
    });
    const glow = bloom.create(root, { source: () => emission.textures.glowing });

    finish.create(root, { format: 'bgra8unorm', source, glow });
  });

  it('grades through a lookup table its grade carries in a fill', async () => {
    const { recording, root, source } = setup();
    const lut = root
      .createTexture({ size: [16, 16, 16], format: 'rgba8unorm', dimension: '3d' })
      .$usage('sampled');
    const view = lut.createView(d.texture3d(d.f32));
    const linear = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    const table = tgpu.accessor(d.texture3d(d.f32));

    const looked = (color: d.v3f, _uv: d.v2f) => {
      'use gpu';
      return std.textureSampleLevel(table.$, linear.$, std.saturate(color), 0).xyz;
    };
    const grade = tgpu.fn(looked).pipe((cfg) => cfg.with(table, () => view.$));
    await finish.create(root, { format: 'bgra8unorm', source, grade }).initAsync();

    const [fragment] = recording.log.shaders.filter((code) => code.includes('@fragment'));
    expect(fragment).toContain('texture_3d<f32>');
  });

  it('shows another target through its grade, which follows the target through a shared part', () => {
    const recording = recordingRoot();
    const { root } = recording;
    const size = { width: 64, height: 32 };
    const screen = targets.create(root, size, {
      hdr: { format: 'rgba16float', sampled: true },
      depth: { format: 'depth24plus', sampled: true },
    });
    const shown = bindings.layout({ depth: { texture: d.textureDepth2d() } });
    const part = shown.bind({ depth: screen.textures.depth });
    const shared = bindings.share(root, { shown: part });

    const depthOf = (_color: d.v3f, uv: d.v2f) => {
      'use gpu';
      const texel = d.vec2i(uv * d.vec2f(std.textureDimensions(shown.$.depth)));
      return d.vec3f(1 - std.textureLoad(shown.$.depth, texel, 0));
    };
    const final = finish.create(root, {
      format: 'bgra8unorm',
      source: () => screen.textures.hdr,
      grade: tgpu.fn(depthOf).pipe(part.inject()),
    });

    const frame = () => {
      if (screen.fit()) part.replace({ depth: screen.textures.depth });
      const encoder = root.createCommandEncoder();
      const pass = encoder.beginRenderPass({ colorAttachments: { view: screen.views.hdr } });
      pass.setBindGroup(shared.layout, shared.group());
      final.draw(pass);
      pass.end();
      encoder.submit();
    };

    frame();
    const pipelines = recording.log.pipelines;
    const [fragment] = recording.log.shaders.filter((code) => code.includes('@fragment'));
    expect(fragment).toContain('texture_depth_2d');

    size.height = 64;
    frame();
    expect(recording.log.pipelines).toBe(pipelines);
    expect(part.resources.depth).toBe(screen.textures.depth);
  });

  it('lets only its grade read the picture elsewhere', async () => {
    const { recording, root, source } = setup();
    await finish.create(root, { format: 'bgra8unorm', source, grade: shifted }).initAsync();
    expect(recording.log.pipelines).toBe(1);

    const stray = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({ uv }) => {
      'use gpu';
      return d.vec4f(finish.picture(uv), 1);
    });
    const outside = root.createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment: stray,
      targets: { format: 'bgra8unorm' },
    });
    await expect(async () => outside.initAsync()).rejects.toThrow(
      'only from the `grade` callback of `finish.create`',
    );
  });

  it('takes a tone curve, not its name', () => {
    const { root, source } = setup();

    // @ts-expect-error -- `tone` is the curve itself.
    finish.create(root, { format: 'bgra8unorm', source, tone: 'neutral' });
  });

  it('binds a new bind group only when the source or the glow changes', () => {
    const { root, size, screen, source } = setup();
    const glow = bloom.create(root, { source });
    const final = finish.create(root, { format: 'bgra8unorm', source, glow });
    const bind = vi.spyOn(root, 'createBindGroup');

    const frame = () => {
      const encoder = root.createCommandEncoder();
      glow.record(encoder);
      const pass = encoder.beginRenderPass({ colorAttachments: { view: screen.views.hdr } });
      final.draw(pass);
      pass.end();
      encoder.submit();
    };

    frame();
    expect(bind).toHaveBeenCalledTimes(2);

    frame();
    expect(bind).toHaveBeenCalledTimes(2);

    // The source, every level of the chain, and the glow.
    size.height = 64;
    screen.fit();
    frame();
    expect(bind.mock.calls.length).toBeGreaterThan(2 + 2);
    const rebound = bind.mock.calls.length;

    frame();
    expect(bind).toHaveBeenCalledTimes(rebound);
  });

  it('binds each of three sources taking turns once', () => {
    const { recording, root, size } = setup();
    const shown = targets.create(root, size, {
      a: { format: 'rgba16float', sampled: true },
      b: { format: 'rgba16float', sampled: true },
      c: { format: 'rgba16float', sampled: true },
      out: { format: 'bgra8unorm' },
    });
    const turns = [shown.textures.a, shown.textures.b, shown.textures.c];
    let turn = 0;
    const final = finish.create(root, { format: 'bgra8unorm', source: () => turns[turn % 3] });
    const frame = () => {
      const encoder = root['~unstable'].createCommandEncoder();
      final.record(encoder, shown.views.out);
      encoder.submit();
      turn++;
    };

    for (const _ of turns) frame();
    const counts = recording.log.counts();
    for (let i = 0; i < 6; i++) frame();
    expect(recording.log.counts()).toEqual(counts);
  });

  it('records into a pass of its own on the view it is given, timed when asked', () => {
    const { recording, root, size, source } = setup();
    const begun = watchPasses(root);
    const shown = targets.create(root, size, {
      first: { format: 'bgra8unorm' },
      second: { format: 'bgra8unorm' },
    });
    const final = finish.create(root, { format: 'bgra8unorm', source });
    const querySet = root.device.createQuerySet({ type: 'timestamp', count: 2 });
    const timestampWrites = { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 };

    const encoder = root.createCommandEncoder();
    final.record(encoder, shown.views.first);
    final.record(encoder, shown.views.second, { timestampWrites });
    encoder.submit();

    const pass = ['begin-render', 'draw', 'end-render'];
    expect(recording.log.events).toEqual([...pass, ...pass, 'submit']);
    expect(begun).toEqual([
      { view: shown.views.first, timestampWrites: undefined },
      { view: shown.views.second, timestampWrites },
    ]);
  });

  it('makes no pipeline in a steady frame, each pipeline holding its groups', () => {
    const { root, screen, source } = setup();
    const glow = bloom.create(root, { source, levels: 4 });
    const final = finish.create(root, { format: 'bgra8unorm', source, glow });

    const frame = () => {
      const encoder = root.createCommandEncoder();
      glow.record(encoder);
      const pass = encoder.beginRenderPass({ colorAttachments: { view: screen.views.hdr } });
      final.draw(pass);
      pass.end();
      encoder.submit();
    };
    frame();

    // Every render pipeline's `with`, which makes a new pipeline whatever it binds.
    const probe = root.createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment: tgpu.fragmentFn({ out: d.vec4f })(() => {
        'use gpu';
        return d.vec4f();
      }),
      targets: { format: 'bgra8unorm' },
    });
    const bind = vi.spyOn(Object.getPrototypeOf(probe), 'with');

    frame();
    expect(bind).not.toHaveBeenCalled();
    bind.mockRestore();
  });
});

describe('fxaa', () => {
  it('compiles one pipeline and draws one triangle into the caller’s pass', async () => {
    const { recording, root, size } = setup();
    const screen = targets.create(root, size, { ldr: { format: 'rgba8unorm', sampled: true } });
    const smooth = fxaa.create(root, { format: 'bgra8unorm', source: () => screen.textures.ldr });
    await smooth.initAsync();
    expect(recording.log.pipelines).toBe(1);

    const encoder = root.createCommandEncoder();
    const pass = encoder.beginRenderPass({ colorAttachments: { view: screen.views.ldr } });
    smooth.draw(pass);
    pass.end();
    encoder.submit();

    expect(recording.log.events).toEqual(['begin-render', 'draw', 'end-render', 'submit']);
  });

  it('records into a pass of its own on the view it is given', () => {
    const { recording, root, size } = setup();
    const begun = watchPasses(root);
    const screen = targets.create(root, size, {
      ldr: { format: 'rgba8unorm', sampled: true },
      output: { format: 'bgra8unorm' },
    });
    const smooth = fxaa.create(root, { format: 'bgra8unorm', source: () => screen.textures.ldr });

    const encoder = root.createCommandEncoder();
    smooth.record(encoder, screen.views.output);
    encoder.submit();

    expect(recording.log.events).toEqual(['begin-render', 'draw', 'end-render', 'submit']);
    expect(begun).toEqual([{ view: screen.views.output, timestampWrites: undefined }]);
  });

  it('binds a new bind group only when the source is rebuilt', () => {
    const { root, size } = setup();
    const screen = targets.create(root, size, {
      ldr: { format: 'rgba8unorm', sampled: true },
      output: { format: 'bgra8unorm' },
    });
    const smooth = fxaa.create(root, { format: 'bgra8unorm', source: () => screen.textures.ldr });
    const bind = vi.spyOn(root, 'createBindGroup');

    const frame = () => {
      const encoder = root.createCommandEncoder();
      const pass = encoder.beginRenderPass({ colorAttachments: { view: screen.views.output } });
      smooth.draw(pass);
      pass.end();
      encoder.submit();
    };

    frame();
    frame();
    expect(bind).toHaveBeenCalledTimes(1);

    size.width = 128;
    screen.fit();
    frame();
    frame();
    expect(bind).toHaveBeenCalledTimes(2);
  });

  it('starts its settings at the settings option, over the defaults', () => {
    const { recording, root, size } = setup();
    const screen = targets.create(root, size, { ldr: { format: 'rgba8unorm', sampled: true } });
    const smooth = fxaa.create(root, {
      format: 'bgra8unorm',
      source: () => screen.textures.ldr,
      settings: { subpixel: 0.5 },
    });

    const floats = [
      ...new Float32Array(recording.bytes(root.unwrap(smooth.settings.buffer)).slice().buffer),
    ];
    expect(floats).toEqual([...Float32Array.of(0.166, 0.0833, 0.5)]);
  });

  it('throws for a source that decodes as it is read', () => {
    const { root, size } = setup();
    const screen = targets.create(root, size, {
      ldr: { format: 'rgba8unorm-srgb', sampled: true },
    });
    const smooth = fxaa.create(root, { format: 'bgra8unorm', source: () => screen.textures.ldr });

    const encoder = root.createCommandEncoder();
    const pass = encoder.beginRenderPass({ colorAttachments: { view: screen.views.ldr } });
    expect(() => smooth.draw(pass)).toThrow('Draw the picture into rgba8unorm instead');
  });
});

describe('regions', () => {
  it('follows a rendered picture’s region with writes, not textures or bind groups', () => {
    const recording = recordingRoot();
    const { root } = recording;
    let resolution = 1;
    const screen = targets.create(
      root,
      { width: 64, height: 32 },
      { hdr: { format: 'rgba16float', sampled: true } },
      { resolution: () => resolution },
    );
    // A rendered picture brings its region along
    const picture = {
      get output() {
        return screen.textures.hdr;
      },
      region: screen.regions.hdr,
    };

    const glow = bloom.create(root, { source: picture, levels: 3 });
    const final = finish.create(root, { format: 'bgra8unorm', source: picture, glow });

    const frame = () => {
      screen.fit();
      const encoder = root.createCommandEncoder();
      glow.record(encoder);
      const pass = encoder.beginRenderPass({ colorAttachments: { view: screen.views.hdr } });
      final.draw(pass);
      pass.end();
      encoder.submit();
    };

    frame();
    expect([glow.region.width, glow.region.height]).toEqual([32, 16]);
    const { textures, bindGroups } = recording.log;
    const writes = recording.log.writes.length;

    frame();
    expect(recording.log.writes.length).toBe(writes);

    resolution = 0.5;
    frame();
    expect([glow.region.width, glow.region.height]).toEqual([16, 8]);
    expect(recording.log.textures).toBe(textures);
    expect(recording.log.bindGroups).toBe(bindGroups);
    // The source's and three levels' parts for the bloom, and finish's regions.
    expect(recording.log.writes.length).toBe(writes + 5);

    // Finish reads its part of the source, and the glow's part of the output.
    const regions = recording.bytes(recording.log.writes.at(-1)!.buffer);
    expect([...new Float32Array(regions.slice().buffer)]).toEqual([32, 16, 16, 8]);
  });
});

describe('colour', () => {
  it('runs every tone curve on the CPU', () => {
    const bright = d.vec3f(8, 2, 0.5);

    for (const tone of [aces, agx, neutral, reinhard]) {
      const mapped = tone(bright);
      expect(luminance(mapped)).toBeLessThan(1);
      expect(mapped.x).toBeGreaterThan(mapped.z);
    }
  });
});
