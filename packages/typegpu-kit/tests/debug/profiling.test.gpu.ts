import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { common, d, std, tgpu } from 'typegpu';
import { describe, expect, it } from 'vitest';
import { animate, type Schedule } from '../../src/animate.ts';
import { createGraphDrawing } from '../../src/debug/graphGpu.ts';
import { createGraphLayout, rectFloats } from '../../src/debug/graphLayout.ts';
import { withoutTracking } from '../../src/debug/instrument.ts';
import {
  captureProfile,
  measureProfile,
  profileSummary,
  profileTrace,
  readProfile,
  startProfiling,
  stopProfiling,
} from '../../src/debug/profiling.ts';
import { create as createFrames } from '../../src/frames.ts';

/** A schedule stepped by hand, as a worker's display relay steps it. */
function manualSchedule() {
  let pending: ((milliseconds: number) => void) | undefined;
  const schedule: Schedule = {
    request(callback) {
      pending = callback;
      return 1;
    },
    cancel() {
      pending = undefined;
    },
  };
  let time = 0;
  const step = (count = 1) => {
    for (let i = 0; i < count; i++) pending?.((time += 16));
  };
  return { schedule, step };
}

const settle = async (root: { device: GPUDevice }) => {
  await root.device.queue.onSubmittedWorkDone();
  await new Promise((resolve) => setTimeout(resolve, 100));
};

describe('profiling on the scene runtime', () => {
  it('records frames that animate runs, on a device requested before it starts', async () => {
    // As on React Native, the device comes first, with timestamp queries requested by the app
    const root = await gpuRoot({ optionalFeatures: ['timestamp-query'] });
    const target = root.createTexture({ size: [512, 512], format: 'rgba8unorm' }).$usage('render');
    // Timestamps tick in steps of up to 65 µs on Apple GPUs, so the draw shades with a loop to
    // measure above zero
    const shadePipeline = root.createRenderPipeline({
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
    let panelDraws = 0;
    const loop = createFrames(root, (frame) => {
      const pass = frame.encoder.beginRenderPass({ colorAttachments: { view: target } });
      pass.setPipeline(shadePipeline);
      pass.draw(3);
      pass.end();

      // A panel drawing on the same runtime stays out of the recording
      withoutTracking(() => {
        const encoder = root.createCommandEncoder();
        const panel = encoder.beginRenderPass({
          colorAttachments: { view: target },
          label: 'panel',
        });
        panel.end();
        encoder.submit();
        panelDraws++;
      });
    });
    const { schedule, step } = manualSchedule();

    startProfiling();
    const animation = animate(loop, { schedule });
    step(20);
    await settle(root);
    step(10);

    const profile = readProfile(50);
    expect(profile.timestampQuery).toBe(true);
    expect(profile.fps).toBeCloseTo(62.5, 1);
    expect(profile.bars.length).toBeGreaterThan(2);
    const cpu = profile.bars.flatMap((bar) => bar.segments.cpu.map(({ name }) => name));
    const gpu = profile.bars.flatMap((bar) => bar.segments.gpu.map(({ name }) => name));
    expect(new Set(cpu)).toEqual(new Set(['frame']));
    // The pipeline was created before profiling started, so its name comes from its label
    expect(new Set(gpu)).toEqual(new Set(['shadePipeline']));
    expect(readProfile(2).bars).toEqual(profile.bars.slice(-2));
    expect(readProfile(0).bars).toEqual([]);

    expect(profileSummary()).toMatch(/^63 fps · gpu .*\ngpu +[\d.]+ms {2}shadePipeline/);

    const captured = captureProfile();
    step(2);
    await settle(root);
    step();
    expect(await captured).toMatch(/^frame \d+ · timings complete\n#0 +[\d.]+ms {2}shadePipeline$/);
    expect(panelDraws).toBeGreaterThan(30);

    const trace = JSON.parse(profileTrace()) as { traceEvents: { name: string }[] };
    expect(trace.traceEvents.some(({ name }) => name === 'shadePipeline')).toBe(true);

    stopProfiling();
    expect(() => readProfile(1)).toThrow('Start profiling on this runtime before reading it.');

    // Frames keep running unrecorded, and profiling starts again with a fresh recording
    step(3);
    startProfiling();
    expect(readProfile(50).bars).toEqual([]);
    stopProfiling();
    animation.destroy();
  });
});

describe('measuring on the scene runtime', () => {
  it('runs the frames animate offers back to back, a fixed step apart, then gives them back', async () => {
    const root = await gpuRoot({ optionalFeatures: ['timestamp-query'] });
    const target = root.createTexture({ size: [256, 256], format: 'rgba8unorm' }).$usage('render');
    const times: number[] = [];
    const loop = createFrames(root, (frame) => {
      times.push(Math.round(frame.time * 1000));
      frame.encoder.beginRenderPass({ colorAttachments: { view: target } }).end();
    });
    const { schedule, step } = manualSchedule();

    startProfiling();
    const animation = animate(loop, { schedule });
    step(3);
    const measured = measureProfile({ seconds: 0.2, dt: 0.01, detail: 'pass' });
    // The display's next frame runs, and the measurement runs ahead from it
    step();
    const text = await measured;

    const count = Number(/^measured (\d+) frames back to back/.exec(text)?.[1]);
    expect(text).toMatch(/10\.00ms of scene time apart\n.*a frame every [\d.]+ms/);
    // Display frames come 16 ms apart; driven ones, warm-up and measured, 10 ms apart
    expect(times.slice(0, 4)).toEqual([0, 16, 32, 48]);
    const driven = times.slice(4);
    expect(driven.length).toBeGreaterThan(count);
    expect(driven.filter((time, i) => time !== 48 + (i + 1) * 10)).toEqual([]);

    // The display, behind the measurement, adds no time, and then its own
    step(2);
    expect(times.slice(-2)).toEqual([driven.at(-1), driven.at(-1)! + 16]);

    stopProfiling();
    animation.destroy();
  });
});

describe('graph drawing', () => {
  it('draws premultiplied shapes where the layout puts them', async () => {
    const root = await gpuRoot();
    const { device } = root;
    const texture = device.createTexture({
      size: [20, 10],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const view = texture.createView();
    const drawing = createGraphDrawing(root, 'rgba8unorm', 4);
    const rects = new Float32Array(4 * rectFloats);
    // Opaque red on the left half, and half-transparent green on the right, its right edge
    // moved down by 5 points
    rects.set([0, 0, 10, 10, 1, 0, 0, 1, 0, 0]);
    rects.set([10, 0, 10, 10, 0, 0.5, 0, 0.5, 5, 5], rectFloats);
    const loop = createFrames(root, (frame) => drawing.draw(frame, view, rects, 2, 20, 10));
    loop.run(0);

    const buffer = device.createBuffer({
      size: 256 * 10,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: 256 }, [20, 10]);
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const pixels = new Uint8Array(buffer.getMappedRange());
    const at = (x: number, y: number) => [...pixels.subarray(y * 256 + x * 4, y * 256 + x * 4 + 4)];

    expect(at(5, 5)).toEqual([255, 0, 0, 255]);
    expect(at(15, 5)).toEqual([0, 128, 0, 128]);
    expect(at(19, 1)).toEqual([0, 0, 0, 0]);
    buffer.unmap();
    drawing.destroy();
  });

  it('draws a full layout within its capacity', async () => {
    const root = await gpuRoot();
    const layout = createGraphLayout(4096);
    const bars = Array.from({ length: 200 }, (_, index) => ({
      time: index * 100,
      frames: { gpu: 6, cpu: 6 },
      segments: {
        gpu: Array.from({ length: 12 }, (__, part) => ({ name: `pass ${part}`, ms: 6, calls: 6 })),
        cpu: [{ name: 'frame', ms: 12, calls: 6 }],
      },
      peak: { gpu: 14, cpu: 3 },
    }));
    const { rects, count } = layout.layout(bars, 1000, 300, 1);
    expect(count).toBeLessThan(4096);

    const texture = root
      .createTexture({ size: [1000, 300], format: 'rgba8unorm' })
      .$usage('render');
    const drawing = createGraphDrawing(root, 'rgba8unorm', 4096);
    const loop = createFrames(root, (frame) =>
      drawing.draw(frame, root.unwrap(texture).createView(), rects, count, 1000, 300),
    );
    loop.run(0);
    expect(await root.takeErrors()).toEqual([]);
    drawing.destroy();
  });
});
