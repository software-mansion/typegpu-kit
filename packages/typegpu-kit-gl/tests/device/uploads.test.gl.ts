import { d, tgpu, type TgpuRoot } from 'typegpu';
import { afterEach, expect, test } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { glRoot } from './compare.ts';

const COUNT = 64;
let root: TgpuRoot | undefined;

afterEach(() => {
  root?.destroy();
  root = undefined;
});

/**
 * A CPU kernel that reads `data` through a mutable binding, and writes one word of it when
 * `writing`, then a draw whose fragment stage reads `data`: frame after frame, `data`'s uploads.
 */
async function frames(writing: boolean) {
  root = glRoot();
  const data = root
    .createBuffer(
      d.arrayOf(d.u32, COUNT),
      Array.from({ length: COUNT }, (_, i) => i),
    )
    .$usage('storage')
    .$name('data');
  const sums = root.createBuffer(d.arrayOf(d.u32, 4)).$usage('storage').$name('sums');
  const reading = data.as('mutable');
  const into = sums.as('mutable');
  const seen = data.as('readonly');
  // Writes to computed cells inside a loop keep the kernel on the CPU
  const kernel = tgpu.computeFn({ workgroupSize: [1], in: { gid: d.builtin.globalInvocationId } })(
    ({ gid }) => {
      'use gpu';
      for (let i = d.u32(0); i < 4; i++) {
        into.$[(gid.x + i) % 4] = reading.$[i * 16] + reading.$[gid.x];
      }
      if (writing) reading.$[0] = reading.$[0] + 1;
    },
  );
  const summing = root.createComputePipeline({ compute: kernel }).$name('summing');
  const vertex = tgpu.vertexFn({
    in: { index: d.builtin.vertexIndex },
    out: { position: d.builtin.position },
  })(({ index }) => {
    'use gpu';
    const corners = [d.vec2f(-1, -1), d.vec2f(3, -1), d.vec2f(-1, 3)];
    return { position: d.vec4f(corners[index]!, 0, 1) };
  });
  const fragment = tgpu.fragmentFn({ out: d.vec4f })(() => {
    'use gpu';
    return d.vec4f(d.f32(seen.$[0] % 256) / 255, d.f32(seen.$[17]) / 255, 0, 1);
  });
  const color = root['~unstable']
    .createTexture({ size: [4, 4], format: 'rgba8unorm' })
    .$usage('render')
    .createView('render');
  const drawing = root.createRenderPipeline({
    vertex,
    fragment,
    targets: { format: 'rgba8unorm' },
  });
  const { stats, gl } = root.device as unknown as GLDevice;
  const uploads: number[] = [];
  for (let frame = 0; frame < 5; frame++) {
    stats.reset();
    summing.dispatchWorkgroups(2);
    drawing.withColorAttachment({ view: color }).draw(3);
    await new Promise((resolve) => setTimeout(resolve, 0));
    const entry = [...stats.buffers].find(([label]) => label.startsWith('data '));
    uploads.push(entry?.[1].uploads ?? 0);
  }
  expect(gl.getError()).toBe(0);
  expect([...stats.kernels.keys()]).toEqual(['summing']);
  return { uploads, data: await data.read(), sums: await sums.read() };
}

test('a CPU kernel that only reads a buffer leaves GL its copy, through a mutable binding', async () => {
  const { uploads, data, sums } = await frames(false);
  expect(uploads.slice(1)).toEqual([0, 0, 0, 0]);
  expect(data[0]).toBe(0);
  // The kernel ran: invocation 1 runs last and overwrites what 0 wrote
  expect(sums.toSorted((a, b) => a - b)).toEqual([1, 17, 33, 49]);
});

test('a CPU kernel that writes one word uploads it for the draws after it', async () => {
  const { uploads, data } = await frames(true);
  expect(uploads.slice(1).every((each) => each >= 1)).toBe(true);
  expect(data[0]).toBe(5 * 2);
});
