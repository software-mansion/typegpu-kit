import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { common, d, std, tgpu, type TgpuCommandEncoder } from 'typegpu';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { instrument } from '../../src/debug/instrument.ts';
import { measure } from '../../src/debug/measure.ts';
import { type CapturedCommand, createRecorder, type Recorder } from '../../src/debug/recorder.ts';
import { timestampsRead } from './timestamps.ts';

let cleanup: (() => void) | undefined;

afterEach(() => {
  cleanup?.();
  cleanup = undefined;
  vi.unstubAllGlobals();
});

/** A page's animation frames, run by hand 16 ms apart, with every callback requested before. */
function handDisplay() {
  const callbacks = new Map<number, FrameRequestCallback>();
  let requests = 0;
  let time = 0;
  vi.stubGlobal('window', globalThis);
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
    callbacks.set(++requests, callback);
    return requests;
  });
  vi.stubGlobal('cancelAnimationFrame', (request: number) => callbacks.delete(request));
  return {
    step() {
      time += 16;
      const due = [...callbacks.values()];
      callbacks.clear();
      for (const callback of due) callback(time);
    },
    get pending() {
      return callbacks.size;
    },
  };
}

/**
 * Installs the instrumentation idle before the device is requested, as the example pages do, and
 * builds a frame on that device. A first root installs Dawn's globals the instrumentation patches.
 */
async function setup() {
  await gpuRoot();
  const original = GPUCommandEncoder.prototype.beginComputePass;
  const instrumentation = instrument();
  const root = await gpuRoot();

  const values = root.createMutable(d.arrayOf(d.f32, 64));
  const doubled = root.createMutable(d.arrayOf(d.f32, 64));

  const fill = tgpu.computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })(({
    gid,
  }) => {
    'use gpu';
    values.$[gid.x] = d.f32(gid.x);
  });

  const double = tgpu.computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })(
    ({ gid }) => {
      'use gpu';
      doubled.$[gid.x] = values.$[gid.x] * 2;
    },
  );

  const fillPipeline = root.createComputePipeline({ compute: fill });
  const doublePipeline = root.createComputePipeline({ compute: double });
  // Timestamps tick in steps of up to 65 µs on Apple GPUs, and a pass that measures zero counts
  // as untimed. The draw shades a megapixel with a loop, so it always measures.
  const drawPipeline = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({ uv }) => {
      'use gpu';
      let value = uv.x;
      for (let i = 0; i < 256; i++) {
        value = std.sin(value + uv.y);
      }
      return d.vec4f(value, 0, 0, 1);
    }),
    targets: { format: 'rgba8unorm' },
  });
  const target = root.createTexture({ size: [1024, 1024], format: 'rgba8unorm' }).$usage('render');

  // Frames report to the recorder profiling started with, as the page's animation frames do.
  let recorder: Recorder | undefined;
  let time = 0;

  const start = (started = createRecorder()) => {
    recorder = started;
    instrumentation.start(started);
    return started;
  };

  const stop = () => {
    recorder = undefined;
    instrumentation.stop();
  };

  const tick = () => recorder?.frameStarted((time += 16));

  const record = (
    encoder: TgpuCommandEncoder,
    descriptor: GPUComputePassDescriptor = {},
    midPass?: () => void,
  ) => {
    const compute = encoder.beginComputePass(descriptor);
    midPass?.();
    compute.setPipeline(fillPipeline);
    compute.dispatchWorkgroups(1);
    compute.setPipeline(doublePipeline);
    compute.dispatchWorkgroups(1);
    compute.end();

    const render = encoder.beginRenderPass({ colorAttachments: { view: target } });
    render.setPipeline(drawPipeline);
    render.draw(3);
    render.end();
  };

  const frame = (descriptor?: GPUComputePassDescriptor) => {
    tick();
    const encoder = root.createCommandEncoder();
    record(encoder, descriptor);
    encoder.submit();
  };

  const capture = async (descriptor?: GPUComputePassDescriptor) => {
    const captured = recorder!.capture();
    frame(descriptor);
    frame(descriptor);
    tick();
    return captured;
  };

  const readBack = () => timestampsRead(root.device);

  let done = false;
  cleanup = () => {
    if (done) return;
    done = true;
    instrumentation[Symbol.dispose]();
    root.destroy();
  };

  return {
    original,
    instrumentation,
    root,
    start,
    stop,
    tick,
    record,
    frame,
    capture,
    readBack,
    done: cleanup,
  };
}

/**
 * Each pass's pipelines, and whether its timestamps came back. A dispatch this small can measure
 * zero, which the recorder counts as untimed.
 */
function timedPasses(commands: CapturedCommand[]) {
  return commands
    .filter(({ kind }) => kind !== 'upload')
    .map(({ pipelines, start, end }) => [pipelines, start !== undefined && end! >= start]);
}

const timedFrame = [
  [['fillPipeline'], true],
  [['doublePipeline'], true],
  [['drawPipeline'], true],
];

describe('instrument', () => {
  it('times each dispatch and the render pass, and links each read to its writer', async () => {
    const { root, start, capture, done } = await setup();
    const recorder = start();
    expect(recorder.timestampQuery()).toBe(true);
    expect(root.device.features.has('timestamp-query')).toBe(true);

    const { commands } = await capture();
    const passes = commands.filter(({ kind }) => kind !== 'upload');

    expect(passes.map(({ kind, pipelines, calls }) => [kind, pipelines, calls])).toEqual([
      ['compute', ['fillPipeline'], 1],
      ['compute', ['doublePipeline'], 1],
      ['render', ['drawPipeline'], 1],
    ]);

    expect(timedPasses(commands)).toEqual(timedFrame);

    const [fill, double, draw] = passes;
    const writes = (command: typeof fill) =>
      command?.uses.filter(({ access }) => access !== 'read').map(({ resource }) => resource.label);
    expect(writes(fill)).toContain('values');
    expect(double?.after).toContainEqual({ from: fill?.index, resource: fill?.uses[0]?.resource });
    expect(draw?.uses.map(({ resource }) => resource.kind)).toEqual(['texture']);

    // Timings reach the graph too.
    const names = recorder.bars().flatMap((bar) => bar.segments.gpu.map(({ name }) => name));
    expect(names).toContain('drawPipeline');

    done();
  });

  it('keeps a compute pass whole per pass, with its WGSL only in a capture', async () => {
    const { start, frame, capture, done } = await setup();
    const recorder = start();
    recorder.setDetail('pass');
    const submitted = vi.spyOn(recorder, 'submitted');
    frame();
    expect(submitted.mock.calls.flat(2).flatMap(({ wgsl }) => wgsl)).toEqual([]);

    const { commands } = await capture();
    const [compute] = commands.filter(({ kind }) => kind === 'compute');
    expect(compute?.pipelines).toEqual(['fillPipeline', 'doublePipeline']);
    expect(compute?.calls).toBe(2);
    // Each pipeline's own module
    expect(compute?.wgsl).toHaveLength(2);
    expect(compute?.wgsl.every((code) => code.includes('@compute'))).toBe(true);

    done();
  });

  it('leaves passes that write their own timestamps alone, and lists them untimed', async () => {
    const { root, start, capture, done } = await setup();
    const recorder = start();
    const querySet = root.device.createQuerySet({ type: 'timestamp', count: 2 });
    const timestampWrites = { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 };

    const { commands } = await capture({ timestampWrites });
    const computes = commands.filter(({ kind }) => kind === 'compute');
    expect(computes).toHaveLength(1);
    expect(computes[0]?.ms).toBeUndefined();
    expect(computes[0]?.timestamps).toBe('own');
    expect(recorder.summary()).toContain('gpu  untimed  fillPipeline + doublePipeline');

    done();
  });

  it('drops timings, rather than failing, when frames outrun their readbacks', async () => {
    const { start, frame, readBack, done } = await setup();
    const recorder = start();
    recorder.setDetail('pass');

    // No readback can finish between these frames, as none can without vsync. Each frame runs
    // in a task of its own and takes a query set, and Metal refuses a device its 33rd.
    for (let i = 0; i < 40; i++) {
      frame();
      await Promise.resolve();
    }
    await readBack();
    frame();
    await readBack();
    recorder.frameStarted(10_000);

    const counted = (processor: 'gpu' | 'cpu') =>
      recorder.bars().reduce((sum, bar) => sum + bar.frames[processor], 0);
    expect(counted('cpu')).toBe(41);
    expect(counted('gpu')).toBe(17);

    done();
  });

  it('takes back the query sets of encoders never finished', async () => {
    const { root, start, capture, readBack, done } = await setup();
    start();

    // Each encoder takes a query set in a task of its own, more than the 16 there are.
    for (let i = 0; i < 17; i++) {
      root.device.createCommandEncoder().beginComputePass().end();
      await Promise.resolve();
    }
    await readBack();

    expect(timedPasses((await capture()).commands)).toEqual(timedFrame);
    done();
  });

  it('drops the passes of an encoder submitted after its query set was taken back', async () => {
    const { root, start, tick, record, done } = await setup();
    const recorder = start();
    const captured = recorder.capture();
    tick();
    const late = root.createCommandEncoder();
    record(late);
    await Promise.resolve();

    // The 16th encoder after it finds no query set left and takes back the late encoder's.
    for (let i = 0; i < 16; i++) {
      root.device.createCommandEncoder().beginComputePass().end();
      await Promise.resolve();
    }
    late.submit();
    tick();

    expect((await captured).timings).toBe('dropped');
    done();
  });

  it('drops a frame whose timestamps fail to read, and gives its query set back', async () => {
    const { root, start, capture, done } = await setup();
    start();
    const createQuerySet = vi.spyOn(root.device, 'createQuerySet');
    const mapAsync = vi
      .spyOn(GPUBuffer.prototype, 'mapAsync')
      .mockRejectedValueOnce(new Error('The read failed.'));

    expect((await capture()).timings).toBe('dropped');
    expect(timedPasses((await capture()).commands)).toEqual(timedFrame);
    expect(createQuerySet).toHaveBeenCalledOnce();

    mapAsync.mockRestore();
    done();
  });

  it('times every pass of a frame that submits an encoder per dispatch', async () => {
    const { root, start, readBack, done } = await setup();
    const recorder = start();
    const values = root.createMutable(d.arrayOf(d.f32, 64));
    const step = root.createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })(
        ({ gid }) => {
          'use gpu';
          values.$[gid.x] += 1;
        },
      ),
    });

    // As TypeGPU's own pipelines do without an encoder: 40 submits, more than the 16 query sets.
    recorder.frameStarted(16);
    for (let i = 0; i < 40; i++) step.dispatchWorkgroups(1);
    recorder.frameStarted(32);
    await readBack();

    // The frame counts for the GPU only if none of its 40 passes went untimed.
    expect(recorder.bars()[0]?.frames).toEqual({ gpu: 1, cpu: 1 });

    done();
  });

  it('stays idle until started, and times frames again after a stop', async () => {
    const { root, start, stop, frame, capture, readBack, done } = await setup();
    const createQuerySet = vi.spyOn(root.device, 'createQuerySet');

    // Idle, the device still gets timestamp-query and pipelines get their names, but no pass is
    // timed.
    expect(root.device.features.has('timestamp-query')).toBe(true);
    frame();
    frame();
    await readBack();
    expect(createQuerySet).not.toHaveBeenCalled();

    expect(start().timestampQuery()).toBe(true);
    expect(timedPasses((await capture()).commands)).toEqual(timedFrame);

    stop();
    frame();
    await readBack();

    // The second session takes the query set the first one gave back.
    start();
    expect(timedPasses((await capture()).commands)).toEqual(timedFrame);
    expect(createQuerySet).toHaveBeenCalledOnce();

    done();
  });

  it('starts and stops in the middle of a frame', async () => {
    const { root, start, stop, tick, record, capture, readBack, done } = await setup();
    const createQuerySet = vi.spyOn(root.device, 'createQuerySet');

    // An encoder created while idle records nothing, even when profiling starts in its pass.
    const recorder = createRecorder();
    const early = recorder.capture();
    const idle = root.createCommandEncoder();
    record(idle, {}, () => {
      start(recorder);
      tick();
    });
    idle.submit();
    tick();
    expect((await early).commands).toEqual([]);

    // One created while profiling still resolves its timestamps when profiling stops in its pass,
    // which splits compute passes, and its submit gives the query set back.
    tick();
    const profiled = root.createCommandEncoder();
    record(profiled, {}, stop);
    profiled.submit();
    await readBack();

    start();
    expect(timedPasses((await capture()).commands)).toEqual(timedFrame);
    expect(createQuerySet).toHaveBeenCalledOnce();

    done();
  });

  it('instruments WebGPU once at a time, and a disposal takes back only its own patches', async () => {
    await gpuRoot();
    const original = GPUCommandEncoder.prototype.beginComputePass;
    const first = instrument();
    expect(() => instrument()).toThrow(
      'Dispose of the existing WebGPU instrumentation before installing another instance.',
    );

    first[Symbol.dispose]();
    expect(GPUCommandEncoder.prototype.beginComputePass).toBe(original);

    const second = instrument();
    const patched = GPUCommandEncoder.prototype.beginComputePass;
    first[Symbol.dispose]();
    expect(GPUCommandEncoder.prototype.beginComputePass).toBe(patched);
    second[Symbol.dispose]();
    expect(GPUCommandEncoder.prototype.beginComputePass).toBe(original);
  });

  it('keeps an empty compute pass when splitting, so split frames have the same passes', async () => {
    const { root, start, tick, record, done } = await setup();
    const recorder = start();
    const captured = recorder.capture();
    const frame = () => {
      tick();
      const encoder = root.createCommandEncoder();
      encoder.beginComputePass({ label: 'empty' }).end();
      record(encoder);
      encoder.submit();
    };
    frame();
    frame();
    tick();

    const { commands } = await captured;
    const compute = commands.filter(({ kind }) => kind === 'compute');
    expect(compute.map(({ group, pipelines }) => [group, pipelines])).toEqual([
      ['empty', []],
      [undefined, ['fillPipeline']],
      [undefined, ['doublePipeline']],
    ]);

    done();
  });

  it('drives animation frames back to back to measure them, then gives them back', async () => {
    const display = handDisplay();
    const { root, instrumentation, start, record, done } = await setup();
    start();

    // The page's loop, with its frame in a debug group
    const times: number[] = [];
    const tick = (now: number) => {
      times.push(now);
      const encoder = root.createCommandEncoder();
      root.unwrap(encoder).pushDebugGroup('sim');
      record(encoder);
      root.unwrap(encoder).popDebugGroup();
      encoder.submit();
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    display.step();

    const stepping = setInterval(() => display.step(), 4);
    const measured = await measure(instrumentation, {
      seconds: 0.1,
      dt: 0.005,
      detail: 'dispatch',
    }).finally(() => clearInterval(stepping));

    // The display's next frame runs as the measurement starts, and driven ones follow 5 ms apart
    expect(times.slice(0, 2)).toEqual([16, 32]);
    const driven = times.slice(
      2,
      times.findIndex((time, i) => i > 2 && time !== times[i - 1] + 5),
    );
    expect(driven.slice(0, 3)).toEqual([37, 42, 47]);
    expect(driven.length).toBeGreaterThan(measured.frames);
    expect(measured.parts.map(({ name }) => name)).toContain('sim › drawPipeline');
    expect(measured.gpu?.median).toBeGreaterThan(0);
    expect(measured.wall).toBeGreaterThan(0);

    // The loop is back on the display, at its time
    expect(display.pending).toBe(1);
    display.step();
    expect(times.at(-1)! % 16).toBe(0);

    done();
  });
});
