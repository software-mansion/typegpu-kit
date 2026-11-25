import { gpuDebug } from '@typegpu/kit/debug';
import {
  common,
  d,
  std,
  tgpu,
  type TgpuComputePipeline,
  type TgpuRenderPipeline,
  type TgpuRoot,
} from 'typegpu';
import { expect, test } from 'vitest';
import type { GLDevice } from '../src/device/index.ts';
import { webgl } from '../src/index.ts';

/** A kernel and a full-screen draw that shades a quarter megapixel with a loop, so it measures. */
function kernels(root: TgpuRoot) {
  const values = root.createMutable(d.arrayOf(d.f32, 64));
  const fillValues = tgpu.computeFn({
    workgroupSize: [64],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    values.$[gid.x] = d.f32(gid.x);
  });
  const shadeWaves = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({ uv }) => {
    'use gpu';
    let value = uv.x;
    for (let i = 0; i < 256; i++) {
      value = std.sin(value + uv.y);
    }
    return d.vec4f(value, 0, 0, 1);
  });
  return { fillValues, shadeWaves };
}

/** Runs frames of one compute and one render pass, and captures one with the profiler. */
async function capture(
  debug: gpuDebug.GpuDebug,
  root: TgpuRoot,
  fill: TgpuComputePipeline,
  draw: TgpuRenderPipeline,
) {
  const target = root.createTexture({ size: [512, 512], format: 'rgba8unorm' }).$usage('render');
  let running = true;
  debug.start();
  const frame = () => {
    if (!running) return;
    const encoder = root.createCommandEncoder();
    const compute = encoder.beginComputePass();
    compute.setPipeline(fill);
    compute.dispatchWorkgroups(1);
    compute.end();
    const render = encoder.beginRenderPass({ colorAttachments: { view: target } });
    render.setPipeline(draw);
    render.draw(3);
    render.end();
    encoder.submit();
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
  try {
    const { commands } = await debug.capture();
    return commands.filter(({ kind }) => kind !== 'upload' && kind !== 'copy');
  } finally {
    running = false;
  }
}

test("the profiler times a fallback device's passes when given its interfaces", async () => {
  const debug = gpuDebug.install({ interfaces: webgl.interfaces });
  const root = await webgl.init();
  try {
    const { fillValues, shadeWaves } = kernels(root);
    const fill = root.createComputePipeline({ compute: fillValues });
    const draw = root.createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment: shadeWaves,
      targets: { format: 'rgba8unorm' },
    });
    const passes = await capture(debug, root, fill, draw);
    expect(passes.map(({ kind, pipelines }) => [kind, pipelines])).toEqual([
      ['compute', ['fill']],
      ['render', ['draw']],
    ]);
    const [, render] = passes;
    expect(render?.start).toBeTypeOf('number');
    expect(render!.end!).toBeGreaterThan(render!.start!);
    expect((root.device as GLDevice).gl.getError()).toBe(0);
  } finally {
    debug.destroy();
    root.destroy();
  }
});

test('the profiler names unnamed pipelines on the fallback by their entry functions', async () => {
  const debug = gpuDebug.install({ interfaces: webgl.interfaces });
  const root = await webgl.init();
  try {
    const { fillValues, shadeWaves } = kernels(root);
    // Made in a list, so TypeGPU names neither pipeline
    const [fill] = [fillValues].map((compute) => root.createComputePipeline({ compute }));
    const [draw] = [shadeWaves].map((fragment) =>
      root.createRenderPipeline({
        vertex: common.fullScreenTriangle,
        fragment,
        targets: { format: 'rgba8unorm' },
      }),
    );
    const passes = await capture(debug, root, fill!, draw!);
    expect(passes.map(({ kind, pipelines }) => [kind, pipelines])).toEqual([
      ['compute', ['fillValues']],
      ['render', ['shadeWaves']],
    ]);
  } finally {
    debug.destroy();
    root.destroy();
  }
});
