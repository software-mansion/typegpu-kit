import { d, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { frames, profiler, dynamicResolution } from '../src/index.ts';
import { gpuRoot } from '@typegpu/kit/testing/dawn';

const settle = () => new Promise((resolve) => setTimeout(resolve, 20));

it('lowers the scale of work that runs over budget, from timings read back on the GPU', async () => {
  const root = await gpuRoot({ features: ['timestamp-query'] });
  const gpu = profiler.create(root, { skip: 2 });
  const busy = gpu.timer('busy');

  const SIDE = 512;
  const data = root.createMutable(d.arrayOf(d.f32, SIDE * SIDE));
  const size = root.createUniform(d.u32);
  const kernel = tgpu.computeFn({
    workgroupSize: [8, 8],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= size.$ || gid.y >= size.$) return;
    const i = gid.y * SIDE + gid.x;
    let x = data.$[i];
    for (let k = 0; k < 400; k++) x = x * 0.999 + 0.001;
    data.$[i] = x;
  });
  const pipeline = root.createComputePipeline({ compute: kernel });

  // Work over a square of `scale` times the side, as a target drawn at that scale
  let scale = 1;
  let written = 0;
  let detail: dynamicResolution.DynamicResolution | undefined;
  const loop = frames.create(root, (frame) => {
    if (detail) {
      detail.update(frame);
      scale = detail.scale;
    }
    const side = Math.ceil(SIDE * scale);
    if (side !== written) size.write((written = side));
    const pass = frame.encoder.beginComputePass({ timestampWrites: busy.writes() });
    pass.setPipeline(pipeline);
    pass.dispatchWorkgroups(Math.ceil(side / 8), Math.ceil(side / 8));
    pass.end();
    gpu.capture(frame);
  });

  for (let i = 0; i < 20; i++) {
    loop.run(i);
    await settle();
  }
  const full = busy.ms!;
  expect(full).toBeGreaterThan(0);

  detail = dynamicResolution.create(busy, { budget: full / 2, range: [0.25, 1] });
  for (let i = 0; i < 200 && detail.scale === 1; i++) {
    loop.run(20 + i);
    await settle();
  }

  expect(detail.scale).toBeLessThan(1);

  gpu.destroy();
  root.destroy();
});
