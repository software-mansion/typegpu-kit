import { d, tgpu, type TgpuBuffer } from 'typegpu';
import { describe, expect, expectTypeOf, it } from 'vitest';
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

  it('keeps what an early submit sent when its frame then throws', () => {
    const recording = recordingRoot();
    const { root } = recording;
    const state = history.create(root, d.f32, { usage: 'uniform' });
    const moved = root.createBuffer(d.f32, 9);
    const values = () =>
      [state.current, state.previous].flatMap((buffer) => floatsOf(recording, root.unwrap(buffer)));
    let failing = true;
    const loop = frames.create(root, (frame) => {
      state.record(frame);
      if (!failing) return;
      // A kernel moves the state after its copy, then the frame sends both and throws
      root
        .unwrap(frame.encoder)
        .copyBufferToBuffer(root.unwrap(moved), 0, root.unwrap(state.current), 0, 4);
      frame.submit();
      throw new Error('frame');
    });

    state.write(5);
    expect(() => loop.run(0)).toThrow('frame');
    failing = false;
    loop.run(0);

    expect(values()).toEqual([9, 9]);
  });

  it('keeps a write made after record out of the recorded frame, and allocates nothing for it', () => {
    const recording = recordingRoot();
    const { root, log } = recording;
    const state = history.create(root, d.u32, { usage: 'uniform' });
    const u32 = (buffer: TgpuBuffer<d.U32>) => new Uint32Array(recording.bytes(buffer).buffer)[0];
    const values = () => [u32(state.current), u32(state.previous)];
    let late: number | undefined = 2;
    const loop = frames.create(root, (frame) => {
      state.record(frame);
      if (late !== undefined) state.write(late);
    });
    state.write(1);
    loop.run(0);
    expect(values()).toEqual([1, 0]);

    const before = log.counts();
    late = undefined;
    loop.run(0);
    expect(values()).toEqual([2, 1]);
    expect(log.counts()).toEqual(before);
  });

  it('uploads halves padded to the buffer, alone and in an array', () => {
    const recording = recordingRoot({ features: ['shader-f16'] });
    const { root } = recording;
    const half = history.create(root, d.f16);
    const halves = history.create(root, d.arrayOf(d.f16, 3));
    half.write(1);
    halves.write([1, 2, 3]);
    frames
      .create(root, (frame) => {
        half.record(frame);
        halves.record(frame);
      })
      .run(0);

    // 1, 2 and 3 as halves
    const now = (buffer: TgpuBuffer<d.AnyWgslData>) =>
      [...new Uint16Array(recording.bytes(buffer).slice().buffer)].slice(
        0,
        d.sizeOf(buffer.dataType) / 2,
      );
    expect([now(half.current), now(halves.current)]).toEqual([[0x3c00], [0x3c00, 0x4000, 0x4200]]);
  });

  it('types a uniform history only with the usage that makes one', () => {
    const { root } = recordingRoot();
    // @ts-expect-error a uniform history names its usage
    history.create<typeof d.u32, 'uniform'>(root, d.u32);
    const options: history.Options<typeof d.u32, 'uniform'> = {};
    // @ts-expect-error the options type cannot leave the usage out either
    history.create(root, d.u32, options);
    expectTypeOf(history.create(root, d.u32, { usage: 'uniform' })).toEqualTypeOf<
      history.History<typeof d.u32, 'uniform'>
    >();
    expectTypeOf(history.create(root, d.u32)).toEqualTypeOf<history.History<typeof d.u32>>();
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
