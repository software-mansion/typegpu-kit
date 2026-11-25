import { layoutOf } from '../../src/cpu/memory.ts';
import { expect, it } from 'vitest';
import { d, std, tgpu } from 'typegpu';
import { cpuRoot } from './device.ts';
import { SweepGenerator } from './scene-generator.ts';
import { observeScene } from './scene-observer.ts';
import {
  captureScene,
  compareCaptures,
  compareExampleCaptures,
  fieldAt,
  type SceneCapture,
} from './example-comparison.ts';

it('captures typed buffer writes and propagates render dependencies through compute', () => {
  const observer = observeScene();
  const root = cpuRoot({ generator: SweepGenerator, observeDevice: observer.device });
  try {
    const image = root
      .createTexture({ size: [2, 2], format: 'rgba32float' })
      .$usage('sampled', 'render');
    const input = image.createView(d.texture2d(d.f32));
    const output = root.createMutable(d.arrayOf(d.vec4f, 4)).$name('result');
    const run = root.createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [1] })(() => {
        'use gpu';
        output.$[0] = d.vec4f(stdLoad());
      }),
    });
    function stdLoad() {
      'use gpu';
      return std.textureLoad(input.$, d.vec2i(0), 0);
    }
    const encoder = root.createCommandEncoder();
    encoder
      .beginRenderPass({
        colorAttachments: {
          view: image,
          loadOp: 'clear',
          clearValue: [0, 0, 0, 0],
          storeOp: 'store',
        },
      })
      .end();
    run.with(encoder).dispatchWorkgroups(1);
    encoder.submit();
    const result = observer.resources.find((resource) => resource.label === 'result')!;
    expect(result.written).toBe(true);
    expect(result.reasons).toContain('written by a render pass');
    expect(fieldAt(result.layout!, 8)).toEqual({ path: '[0][2]', type: 'f32' });
  } finally {
    root.destroy();
  }
});

const capture = (words: number[]): SceneCapture => ({
  frames: 2,
  skipped: [],
  resources: [
    {
      key: 'buffer',
      label: 'buffer',
      kind: 'buffer',
      written: true,
      destroyed: false,
      reasons: [],
      order: [],
      approximate: false,
      quotient: false,
      layout: {
        type: 'struct',
        size: 16,
        fields: {
          count: { offset: 0, layout: { type: 'u32', size: 4 } },
          value: { offset: 4, layout: { type: 'f32', size: 4 } },
        },
      },
      levels: [{ width: 0, height: 0, depth: 0, bytes: Uint32Array.from(words).buffer }],
    },
  ],
});
it('compares integer words exactly and ignores only layout padding', () => {
  const result = compareCaptures(capture([1, 0x3f800000, 0, 0]), capture([2, 0x3f801000, 99, 99]));
  expect(result.differences[0]!.words).toBe(2);
  expect(result.differences[0]!.first[0]).toContain('.count');
  expect(result.differences[0]!.first[1]).toContain('.value');
});

const runScene = async () => {
  const observer = observeScene();
  const root = cpuRoot({
    generator: SweepGenerator,
    observeDevice: observer.device,
    onTexture: (texture, data) => observer.cpuTextures.set(texture, data),
  });
  try {
    return await captureScene(root, observer, 'forest');
  } finally {
    root.destroy();
  }
};
it('captures a whole seeded forest scene identically twice on the CPU', async () => {
  const result = compareCaptures(await runScene(), await runScene());
  expect(result.compared).toBeGreaterThan(0);
  expect(result.differences).toEqual([]);
  expect(result.skipped.length).toBeGreaterThan(0);
}, 30000);

const foamCapture = (values: Uint32Array): SceneCapture => ({
  ...capture([]),
  resources: [
    {
      ...capture([]).resources[0]!,
      key: 'buffer:foam:host#0',
      label: 'foam',
      layout: layoutOf(d.arrayOf(d.u32, values.length)),
      levels: [{ width: 0, height: 0, depth: 0, bytes: values.slice().buffer }],
    },
  ],
});

it('compares ocean foam history in cascade/x/y order without ignoring words', () => {
  const cpu = new Uint32Array(4 * 512 * 512),
    gpu = new Uint32Array(cpu.length);
  cpu[512 * 512 + 2 * 512 + 7] = 17;
  gpu[512 * 512 + 7 * 512 + 2] = 17;
  cpu[3 * 512 * 512 + 511 * 512 + 5] = 29;
  gpu[3 * 512 * 512 + 5 * 512 + 511] = 29;
  const matched = compareExampleCaptures('ocean', foamCapture(gpu), foamCapture(cpu));
  expect(matched.differences).toEqual([]);
  expect(matched.representations).toHaveLength(1);
  cpu[512 * 512 + 2 * 512 + 7] = 18;
  expect(
    compareExampleCaptures('ocean', foamCapture(gpu), foamCapture(cpu)).differences[0]!.words,
  ).toBe(1);
});
