import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { afterAll, afterEach, beforeAll, expect, it } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { roots } from '../device/compare.ts';

let gpu: TgpuRoot;
let gl: TgpuRoot;
beforeAll(async () => {
  ({ gpu, gl } = await roots());
});
afterAll(() => {
  gpu.destroy();
  gl.destroy();
});
afterEach(() => {
  expect((gl.device as unknown as GLDevice).gl.getError()).toBe(0);
});

const N = 200;
const START = [5, 0, 9, 1, 40, 2, 7, 3];

/**
 * Loads two counters, then adds to the counters on one of two paths, one of which returns. Every
 * load comes before every addition, so the loads may all read the counters from before the
 * dispatch, which is what the form does. The additions do not depend on what was loaded, so the
 * counters end the same in any order.
 */
async function loadsThenAdds(root: TgpuRoot) {
  const counts = root.createMutable(d.arrayOf(d.atomic(d.u32), 8), START);
  const out = root.createMutable(d.arrayOf(d.u32, N));
  const kernel = tgpu.computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })(
    ({ gid }) => {
      'use gpu';
      const i = gid.x;
      if (i >= N) return;
      const seen = std.atomicLoad(counts.$[i % 8]) + std.atomicLoad(counts.$[(i + 3) % 8]);
      if (i % 3 === 0) {
        std.atomicAdd(counts.$[(i * 5) % 8], 3);
        return;
      }
      std.atomicSub(counts.$[i % 8], 1);
      std.atomicAdd(counts.$[(i + 1) % 8], 2);
      out.$[i] = seen;
    },
  );
  const pipeline = root.createComputePipeline({ compute: kernel }).$name('ordered');
  const seen: number[][] = [];
  for (let k = 0; k < 2; k++) {
    pipeline.dispatchWorkgroups(Math.ceil(N / 64));
    seen.push(await out.read());
  }
  return { seen, counts: await counts.read() };
}

/** What the kernel gives when every invocation's loads run before any invocation's additions. */
function loadsFirst() {
  const counts = [...START];
  const out = Array<number>(N).fill(0);
  const seen: number[][] = [];
  for (let k = 0; k < 2; k++) {
    const before = [...counts];
    for (let i = 0; i < N; i++) {
      const loaded = ((before[i % 8] as number) + (before[(i + 3) % 8] as number)) >>> 0;
      if (i % 3 === 0) {
        counts[(i * 5) % 8] = ((counts[(i * 5) % 8] as number) + 3) >>> 0;
        continue;
      }
      counts[i % 8] = ((counts[i % 8] as number) - 1) >>> 0;
      counts[(i + 1) % 8] = ((counts[(i + 1) % 8] as number) + 2) >>> 0;
      out[i] = loaded;
    }
    seen.push([...out]);
  }
  return { seen, counts };
}

it('runs loads before additions on the GPU, as if every load came first', async () => {
  const actual = await loadsThenAdds(gl);
  const stats = (gl.device as unknown as GLDevice).stats;
  expect(stats.fallbacks.get('ordered')).toBeUndefined();
  expect([...stats.kernels.keys()]).toContain('ordered (gpu)');
  expect(actual).toEqual(loadsFirst());
  // WebGPU's loads race its additions, but its counters end the same
  expect((await loadsThenAdds(gpu)).counts).toEqual(actual.counts);
});
