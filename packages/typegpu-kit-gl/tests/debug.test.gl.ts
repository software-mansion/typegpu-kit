import { gpuDebug } from '@typegpu/kit/debug';
import { common, d, std, tgpu } from 'typegpu';
import { expect, test } from 'vitest';
import type { GLDevice } from '../src/device/index.ts';
import { webgl } from '../src/index.ts';

test("the profiler times a fallback device's passes when given its interfaces", async () => {
  const debug = gpuDebug.install({ interfaces: webgl.interfaces });
  const root = await webgl.init();
  let running = true;
  try {
    const values = root.createMutable(d.arrayOf(d.f32, 64));
    const fill = root.createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })(
        ({ gid }) => {
          'use gpu';
          values.$[gid.x] = d.f32(gid.x);
        },
      ),
    });
    const draw = root.createRenderPipeline({
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
    const target = root.createTexture({ size: [512, 512], format: 'rgba8unorm' }).$usage('render');

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

    const { commands } = await debug.capture();
    const passes = commands.filter(({ kind }) => kind !== 'upload' && kind !== 'copy');
    expect(passes.map(({ kind, pipelines }) => [kind, pipelines])).toEqual([
      ['compute', ['fill']],
      ['render', ['draw']],
    ]);
    // The draw shades a quarter megapixel with a loop, so it always measures
    const [, render] = passes;
    expect(render?.start).toBeTypeOf('number');
    expect(render!.end!).toBeGreaterThan(render!.start!);
    expect((root.device as GLDevice).gl.getError()).toBe(0);
  } finally {
    running = false;
    debug.destroy();
    root.destroy();
  }
});
