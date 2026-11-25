import { expect, it } from 'vitest';
import { d, std, tgpu } from 'typegpu';
import { cpuRoot } from './device.ts';
import { SweepGenerator } from './scene-generator.ts';
import { observeScene } from './scene-observer.ts';
import { captureScene, compareCaptures, fieldAt, type SceneCapture } from './example-comparison.ts';

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
