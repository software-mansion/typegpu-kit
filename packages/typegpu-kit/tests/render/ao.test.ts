import { bindings, cameras, passes, targets, type profiler } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';
import { d, tgpu } from 'typegpu';
import { describe, expect, it } from 'vitest';
import { lighting, ao, shading } from '../../src/index.ts';

function setup(options: { temporal?: boolean; normal?: boolean } = {}) {
  const recording = recordingRoot();
  const { root } = recording;
  const size = { width: 64, height: 32 };
  const early = targets.create(root, size, {
    normal: { format: 'rg16float', scale: 0.5, sampled: true },
    depth: { format: 'depth32float', scale: 0.5, sampled: true },
  });
  const view = cameras.orbit(root, { size, target: [0, 0, 0], distance: 6 });
  const switched = { on: true };
  const occlusion = ao.create(root, view, {
    depth: () => early.textures.depth,
    enabled: () => switched.on,
    ...(options.normal && { normal: () => early.textures.normal }),
    ...(options.temporal && { temporal: true }),
  });
  const begun: (GPURenderPassDescriptor | GPUComputePassDescriptor | undefined)[] = [];
  const frame = (timing?: profiler.Timed) => {
    const encoder = root.createCommandEncoder();
    const raw = root.unwrap(encoder);
    const [render, compute] = [raw.beginRenderPass.bind(raw), raw.beginComputePass.bind(raw)];
    raw.beginRenderPass = (descriptor) => (begun.push({ ...descriptor }), render(descriptor));
    raw.beginComputePass = (descriptor) => (begun.push({ ...descriptor }), compute(descriptor));
    occlusion.record(encoder, timing);
    encoder.submit();
  };

  return { recording, root, size, early, view, occlusion, switched, begun, frame };
}

const Vertex = d.struct({ position: d.vec4f, normal: d.vec3f });
const corner = () => {
  'use gpu';
  return Vertex({ position: d.vec4f(0, 0, 0.5, 1), normal: d.vec3f(0, 1, 0) });
};
const pixelOf = () => {
  'use gpu';
  return Vertex();
};

const sky = (material: shading.Material, point: shading.Point) => {
  'use gpu';
  return shading.hemisphere(material, point) * ao.at(point.position);
};

describe('ao', () => {
  it('traces and denoises in two passes of two pipelines, writing nothing in a steady frame', async () => {
    const { recording, occlusion, frame } = setup();
    await occlusion.initAsync();
    expect(recording.log.pipelines).toBe(2);
    for (const wgsl of recording.log.shaders) expect(wgsl).not.toContain('var<storage');

    frame();
    recording.log.events.length = 0;
    const counts = recording.log.counts();
    const writes = recording.log.writes.length;
    frame();
    expect(recording.log.events).toEqual([
      'begin-render',
      'draw',
      'end-render',
      'begin-render',
      'draw',
      'end-render',
      'submit',
    ]);
    expect(recording.log.counts()).toEqual(counts);
    expect(recording.log.writes.length).toBe(writes);
  });

  it('copies the next frame in with temporal, and reads the normal it is given', async () => {
    const { recording, occlusion, frame } = setup({ temporal: true, normal: true });
    await occlusion.initAsync();
    frame();
    expect(recording.log.events.slice(-2)).toEqual(['copy-buffer', 'submit']);
    const trace = recording.log.shaders.find((wgsl) => wgsl.includes('texture_depth_2d'))!;
    expect(trace.match(/texture_2d<f32>/g)).toHaveLength(1);
  });

  it('clears to full visibility once when off, then records nothing', () => {
    const { recording, switched, frame } = setup({ temporal: true });
    frame();
    switched.on = false;
    recording.log.events.length = 0;
    frame();
    frame();
    expect(recording.log.events).toEqual(['begin-render', 'end-render', 'submit', 'submit']);

    switched.on = true;
    recording.log.events.length = 0;
    frame();
    expect(recording.log.events.filter((event) => event === 'draw')).toHaveLength(2);
  });

  it('times both passes while on, and an empty pass while off, as the shadows do', () => {
    const { root, switched, begun, frame } = setup();
    const querySet = root.device.createQuerySet({ type: 'timestamp', count: 2 });
    const timestampWrites = { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 };
    frame({ timestampWrites });
    switched.on = false;
    frame({ timestampWrites });
    frame({ timestampWrites });
    expect(begun.map((descriptor) => descriptor?.timestampWrites)).toEqual([
      { querySet, beginningOfPassWriteIndex: 0 },
      { querySet, endOfPassWriteIndex: 1 },
      undefined,
      timestampWrites,
      timestampWrites,
    ]);
  });

  it('clears again when off after the depth changes size', () => {
    const { recording, size, early, switched, frame } = setup();
    switched.on = false;
    frame();
    size.width = 80;
    early.fit();
    recording.log.events.length = 0;
    frame();
    expect(recording.log.events).toEqual(['begin-render', 'end-render', 'submit']);
  });

  it('follows the depth to a new size once, keeping the pipelines that share its part', () => {
    const { recording, root, size, early, view, occlusion, frame } = setup();
    const shared = bindings.share(root, { ao: occlusion.part });
    const light = lighting.create(root, {
      shading: { ambient: tgpu.fn(sky).pipe(occlusion.inject()) },
    });
    const screen = targets.create(root, size, {
      hdr: { format: 'rgba16float', sampled: true },
      depth: { format: 'depth24plus' },
    });
    const scene = passes
      .create(root, screen, { hdr: passes.color, depth: passes.depth }, { shared })
      .pipe(view.inject())
      .pipe(light.inject());
    const lit = passes.output(d.vec4f, {
      write: () => {
        'use gpu';
        const material = shading.Material({
          color: d.vec3f(1),
          roughness: 1,
          emission: d.vec3f(),
          metallic: 0,
          occlusion: 1,
        });
        return d.vec4f(shading.shade(material, shading.pointAt(d.vec3f(), d.vec3f(0, 1, 0))), 1);
      },
    });
    const drawn = scene.add({
      Vertex,
      at: corner,
      pixelOf,
      color: lit,
      draw: (pass) => pass.draw(3),
    });
    const record = () => {
      const encoder = root.createCommandEncoder();
      occlusion.record(encoder);
      scene.record(encoder, drawn);
      encoder.submit();
    };
    record();
    const wgsl = recording.log.shaders.find((code) => code.includes('ao_occlusion'))!;
    // The texture and the region, in the shared group
    expect(wgsl.match(/var ao_occlusion: texture_2d<f32>/g)).toHaveLength(1);
    expect(wgsl.match(/var<uniform> ao_state/g)).toHaveLength(1);

    const pipelines = recording.log.pipelines;
    const textures = recording.log.textures;
    const groups = recording.log.bindGroups;
    size.width = 128;
    early.fit();
    screen.fit();
    record();
    // The prepass's two and the scene's two, then the occlusion's two
    expect(recording.log.textures - textures).toBe(2 + 2 + 2);
    expect(recording.log.pipelines).toBe(pipelines);
    const made = recording.log.bindGroups - groups;
    record();
    frame();
    expect(recording.log.bindGroups - groups).toBe(made);
  });

  it('asks for its share as the first pipeline that reads it builds, before any resize', () => {
    const { root, view, occlusion } = setup();
    const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      ao.at(d.vec3f());
    });
    const reading = root.pipe(view.inject()).pipe(occlusion.inject());
    expect(() => reading.createComputePipeline({ compute: kernel }).initSync()).toThrow(
      'need it shared: give `passes.create` `{ shared: { ao: occlusion.part } }`, or `bindings.share(root, { ao: occlusion.part })` for passes of your own, before creating them.',
    );
  });

  it('refuses slices outside 1 to 3', () => {
    const { root, early, view } = setup();
    expect(() =>
      ao.create(root, view, { depth: () => early.textures.depth, slices: 4 as 3 }),
    ).toThrow('The occlusion takes 1 to 3 slices, got 4.');
  });
});
