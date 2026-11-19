import { passes, targets } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';
import { d, tgpu } from 'typegpu';
import { describe, expect, it } from 'vitest';
import { taa } from '../../src/index.ts';
import { stagesOf } from '../stages.ts';

function setup(options: { depth?: boolean; still?: boolean } = {}) {
  const recording = recordingRoot();
  const { root } = recording;
  const size = { width: 64, height: 32 };
  const screen = targets.create(root, size, {
    hdr: { format: 'rgba16float', sampled: true },
    motion: { format: 'rg16float', sampled: true },
    depth: { format: 'depth24plus', sampled: true },
  });
  const aa = taa.create(root, {
    source: () => screen.textures.hdr,
    ...(!options.still && { motion: () => screen.textures.motion }),
    depth: options.depth ? () => screen.textures.depth : null,
  });
  const frame = () => {
    const encoder = root.createCommandEncoder();
    aa.record(encoder);
    encoder.submit();
  };
  /** The offset the next frame draws with. */
  const offset = () => [
    ...new Float32Array(recording.bytes(root.unwrap(aa.offset.buffer)).slice().buffer),
  ];

  return { recording, root, size, screen, aa, frame, offset };
}

/** The resolve's state uniform: the region it reads, and whether it blends with last frame. */
function stateOf(recording: ReturnType<typeof recordingRoot>) {
  const write = recording.log.writes.findLast(({ size }) => size === 16)!;
  const bytes = recording.bytes(write.buffer).slice().buffer;
  return { region: [...new Float32Array(bytes, 0, 2)], valid: new Uint32Array(bytes, 8, 1)[0] };
}

describe('taa', () => {
  it('resolves in one pass and copies the next jitter in, writing nothing in a steady frame', async () => {
    const { recording, aa, frame } = setup({ depth: true });
    await aa.initAsync();
    expect(recording.log.pipelines).toBe(1);
    expect(recording.log.shaders.at(-1)).toContain('texture_depth_2d');

    // initAsync bound both histories; the first frame creates only the buffers, sampler and
    // automatic group TypeGPU makes at a pipeline's first draw. The second writes that it blends.
    frame();
    const counts = recording.log.counts();
    frame();
    recording.log.events.length = 0;
    const writes = recording.log.writes.length;
    frame();
    frame();

    expect(recording.log.events).toEqual([
      'begin-render',
      'draw',
      'end-render',
      'copy-buffer',
      'submit',
      'begin-render',
      'draw',
      'end-render',
      'copy-buffer',
      'submit',
    ]);
    expect(recording.log.counts()).toEqual(counts);
    expect(recording.log.writes.length).toBe(writes);
  });

  it('finishes a warmup it is destroyed during', async () => {
    const { aa } = setup();
    const ready = aa.initAsync();
    aa.destroy();
    await ready;
  });

  it('reads no depth without one', async () => {
    const { recording, aa } = setup();
    await aa.initAsync();
    expect(recording.log.shaders.at(-1)).not.toContain('texture_depth_2d');
  });

  it('reads no size or reactive texture without them', async () => {
    const { recording, aa } = setup({ depth: true });
    await aa.initAsync();
    const wgsl = recording.log.shaders.at(-1)!;
    // The source, the motion and the history, with nothing gathered for a larger output
    expect(wgsl.match(/texture_2d<f32>/g)).toHaveLength(3);
    expect(wgsl).not.toContain('Gathered');
  });

  it('resolves at `size` from a region of the source, keeping history as the region changes and taking more jitter phases', async () => {
    const recording = recordingRoot();
    const { root } = recording;
    const size = { width: 64, height: 32 };
    let resolution = 1;
    const screen = targets.create(
      root,
      size,
      {
        hdr: { format: 'rgba16float', sampled: true },
        reactive: { format: 'r8unorm', sampled: true },
      },
      { resolution: () => resolution },
    );
    const aa = taa.create(root, {
      source: () => screen.textures.hdr,
      region: screen.regions.hdr,
      depth: null,
      size: screen,
      reactive: () => screen.textures.reactive,
    });
    const frame = () => {
      screen.fit();
      const encoder = root.createCommandEncoder();
      aa.record(encoder);
      encoder.submit();
    };
    const offset = () => [
      ...new Float32Array(recording.bytes(root.unwrap(aa.offset.buffer)).slice().buffer),
    ];
    /** The distinct offsets of `count` frames, and whether the next comes round to the first. */
    const phases = (count: number) => {
      const seen: string[] = [];
      for (let i = 0; i <= count; i++) {
        frame();
        seen.push(String(offset()));
      }
      return { distinct: new Set(seen.slice(0, count)).size, round: seen[count] === seen[0] };
    };
    await aa.initAsync();
    expect(recording.log.shaders.at(-1)!.match(/texture_2d<f32>/g)).toHaveLength(3);
    expect(aa.output.props.size).toEqual([64, 32]);

    frame();
    frame();
    expect(phases(8)).toEqual({ distinct: 8, round: true });

    // Two thirds of each side: a region of 43 × 22, which the output enlarges by 2.17 in area
    resolution = 2 / 3;
    frame();
    expect(stateOf(recording)).toEqual({ region: [43, 22], valid: 1 });
    expect([aa.region.width, aa.region.height]).toEqual([64, 32]);
    expect(phases(18)).toEqual({ distinct: 18, round: true });

    const counts = recording.log.counts();
    const writes = recording.log.writes.length;
    recording.log.events.length = 0;
    frame();
    expect(recording.log.events).toEqual([
      'begin-render',
      'draw',
      'end-render',
      'copy-buffer',
      'submit',
    ]);
    expect(recording.log.counts()).toEqual(counts);
    expect(recording.log.writes.length).toBe(writes);

    // Off, it hands back the source in its own region
    aa.enable(false);
    expect([aa.output, aa.region]).toEqual([screen.textures.hdr, screen.regions.hdr]);
  });

  it('reads no motion without one, and no depth without motion or a camera', async () => {
    const { recording, aa } = setup({ depth: true, still: true });
    await aa.initAsync();
    const wgsl = recording.log.shaders.at(-1)!;
    // The source and the history only
    expect(wgsl.match(/texture_2d<f32>/g)).toHaveLength(2);
    expect(wgsl).not.toContain('texture_depth_2d');
  });

  it('blends from the second frame on, starts over after a reset, a resize or turning on, and records nothing while off', () => {
    const { recording, size, screen, aa, frame, offset } = setup();

    frame();
    expect(stateOf(recording)).toEqual({ region: [64, 32], valid: 0 });
    frame();
    expect(stateOf(recording).valid).toBe(1);

    aa.reset();
    frame();
    expect(stateOf(recording).valid).toBe(0);
    frame();
    expect(stateOf(recording).valid).toBe(1);

    const before = aa.output;
    size.width = 128;
    screen.fit();
    frame();
    expect(aa.output).not.toBe(before);
    expect(stateOf(recording)).toEqual({ region: [128, 32], valid: 0 });

    // Off, it records nothing, draws without jitter and hands back the source
    aa.enable(false);
    recording.log.events.length = 0;
    frame();
    expect([aa.enabled, aa.output, offset()]).toEqual([false, screen.textures.hdr, [0, 0]]);
    expect(recording.log.events).toEqual(['submit']);
    aa.enable(true);
    frame();
    expect(stateOf(recording).valid).toBe(0);
    frame();
    expect(stateOf(recording).valid).toBe(1);
  });

  it('moves the scene’s vertices by eight offsets under a pixel in turn, and by none while off', () => {
    const { recording, root, screen, aa, frame, offset } = setup();
    const scene = passes.create(
      root,
      screen,
      { hdr: passes.color, depth: passes.depth },
      { jitter: aa.jitter },
    );
    const Vertex = d.struct({ position: d.vec4f });
    const stages = stagesOf(scene, {
      Vertex,
      at: () => {
        'use gpu';
        return Vertex({ position: d.vec4f(0, 0, 0, 1) });
      },
      pixelOf: () => {
        'use gpu';
        return Vertex();
      },
      color: passes.output(d.vec4f, {
        write: () => {
          'use gpu';
          return d.vec4f(1);
        },
      }),
    });
    const wgsl = tgpu.resolve([
      root.createRenderPipeline({
        ...stages,
        depthStencil: { format: scene.depthFormat!, depthWriteEnabled: true, depthCompare: 'less' },
      }),
    ]);
    expect(wgsl).toMatch(/var<uniform> \w+: vec2f/);

    const offsets: number[][] = [];
    for (let i = 0; i < 9; i++) {
      frame();
      offsets.push(offset());
    }
    expect(recording.log.events.filter((event) => event === 'copy-buffer')).toHaveLength(9);
    // Within half a pixel each way, in clip space, and back to the first after eight.
    for (const [x, y] of offsets) {
      expect(Math.abs(x!)).toBeLessThanOrEqual(1 / 64);
      expect(Math.abs(y!)).toBeLessThanOrEqual(1 / 32);
    }
    expect(new Set(offsets.slice(0, 8).map(String)).size).toBe(8);
    expect(offsets[8]).toEqual(offsets[0]);

    aa.enable(false);
    frame();
    expect(offset()).toEqual([0, 0]);
  });
});
