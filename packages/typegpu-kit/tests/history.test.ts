import { d, tgpu } from 'typegpu';
import { describe, expect, it } from 'vitest';
import { frames, history } from '../src/index.ts';
import { recordingRoot } from '../src/testing.ts';

const floatsOf = (recording: ReturnType<typeof recordingRoot>, buffer: GPUBuffer) => [
  ...new Float32Array(recording.bytes(buffer).slice().buffer),
];

describe('history', () => {
  it('lands a write at the next record, setting the state before it aside', () => {
    const recording = recordingRoot();
    const { root } = recording;
    const state = history.create(root, d.arrayOf(d.f32, 2), { initial: [1, 2] });
    const now = () => floatsOf(recording, root.unwrap(state.current));
    const then = () => floatsOf(recording, root.unwrap(state.previous));
    const loop = frames.create(root, (frame) => state.record(frame));
    const frame = () => loop.run(0);

    expect([now(), then()]).toEqual([
      [1, 2],
      [1, 2],
    ]);

    state.write([3, 4]);
    expect(now()).toEqual([1, 2]);
    frame();
    expect([now(), then()]).toEqual([
      [3, 4],
      [1, 2],
    ]);

    // A frame without a write keeps the state, which is then last frame's too.
    frame();
    expect([now(), then()]).toEqual([
      [3, 4],
      [3, 4],
    ]);
  });

  it('copies last frame aside before landing the write, and only copies when nothing was written', () => {
    const recording = recordingRoot();
    const { root, log } = recording;
    const state = history.create(root, d.vec4f, { usage: 'uniform' });
    const copies = () => log.events.filter((event) => event === 'copy-buffer').length;

    frames
      .create(root, (frame) => {
        state.record(frame);
        expect(copies()).toBe(1);
        state.write(d.vec4f(1));
        state.record(frame);
        expect(copies()).toBe(3);
      })
      .run(0);
  });

  it('keeps a write for the next frame when the frame that copied it does not submit', () => {
    const recording = recordingRoot();
    const { root } = recording;
    const state = history.create(root, d.f32, { usage: 'uniform' });
    let failing = true;
    const loop = frames.create(root, (frame) => {
      state.record(frame);
      if (failing) throw new Error('frame');
    });

    state.write(5);
    expect(() => loop.run(0)).toThrow('frame');
    failing = false;
    loop.run(0);

    expect(floatsOf(recording, root.unwrap(state.current))).toEqual([5]);
  });

  it('fills an accessor with the state now, and with last frame’s in code `previously` compiled', () => {
    const { root } = recordingRoot();
    const state = history.create(root, d.arrayOf(d.f32, 4));
    const values = tgpu.accessor(d.arrayOf(d.f32, 4));
    const out = root.createMutable(d.f32);

    const first = () => {
      'use gpu';
      return values.$[0];
    };
    const firstBefore = history.previously(first);
    const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      out.$ = first() - firstBefore();
    });
    const vertex = tgpu.vertexFn({ out: { position: d.builtin.position } })(() => {
      'use gpu';
      return { position: d.vec4f(first(), firstBefore(), 0, 1) };
    });
    const fragment = tgpu.fragmentFn({ out: d.vec4f })(() => {
      'use gpu';
      return d.vec4f(1);
    });

    const cfg = root.with(values, state.reading);
    const compute = tgpu.resolve([cfg.createComputePipeline({ compute: kernel })]);
    const render = tgpu.resolve([
      cfg.createRenderPipeline({ vertex, fragment, targets: { format: 'rgba8unorm' } }),
    ]);

    expect(compute.match(/var<storage, read_write> \w+: array<f32, 4>/g)).toHaveLength(2);
    expect(render.match(/var<storage, read> \w+: array<f32, 4>/g)).toHaveLength(2);
  });

  it('resolves a temporal fill of plain values as each frame’s', () => {
    const { root } = recordingRoot();
    const speed = tgpu.accessor(d.f32);
    const out = root.createMutable(d.f32);
    const read = () => {
      'use gpu';
      return speed.$;
    };
    const readBefore = history.previously(read);
    const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      out.$ = read() * 10 + readBefore();
    });

    const wgsl = tgpu.resolve([
      root
        .with(speed, history.temporal(d.f32(0.25), d.f32(0.75)))
        .createComputePipeline({ compute: kernel }),
    ]);

    expect(wgsl).toContain('0.25');
    expect(wgsl).toContain('0.75');
  });

  it('leaves code without `previously` reading the state now only', () => {
    const { root } = recordingRoot();
    const state = history.create(root, d.f32, { usage: 'uniform' });
    const value = tgpu.accessor(d.f32);
    const out = root.createMutable(d.f32);
    const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      out.$ = value.$;
    });

    const wgsl = tgpu.resolve([
      root.with(value, state.reading).createComputePipeline({ compute: kernel }),
    ]);
    expect(wgsl.match(/var<uniform>/g)).toHaveLength(1);
  });
});
