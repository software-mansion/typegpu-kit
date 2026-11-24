import { d, tgpu } from 'typegpu';
import { afterEach, expect, test, vi } from 'vitest';
import { glRoot } from './compare.ts';

afterEach(() => vi.restoreAllMocks());

test('a compute pipeline binds 10 storage buffers without a limit warning', async () => {
  const warn = vi.spyOn(console, 'warn');
  const root = glRoot();
  try {
    const inputs = Array.from({ length: 9 }, (_, i) =>
      root.createReadonly(d.arrayOf(d.f32, 4), [i, i, i, i]),
    );
    const sums = root.createMutable(d.arrayOf(d.f32, 4));
    const [a, b, c, e, f, g, h, k, m] = inputs;
    const add = tgpu.computeFn({ workgroupSize: [4], in: { id: d.builtin.globalInvocationId } })(({
      id,
    }) => {
      'use gpu';
      const i = id.x;
      sums.$[i] =
        a!.$[i]! +
        b!.$[i]! +
        c!.$[i]! +
        e!.$[i]! +
        f!.$[i]! +
        g!.$[i]! +
        h!.$[i]! +
        k!.$[i]! +
        m!.$[i]!;
    });
    root.createComputePipeline({ compute: add }).dispatchWorkgroups(1);
    expect(await sums.read()).toEqual([36, 36, 36, 36]);
    const limits = warn.mock.calls.filter((call) => String(call[0]).includes('exceeds'));
    expect(limits).toEqual([]);
    expect(root.device.limits.maxStorageBuffersPerShaderStage).toBeGreaterThanOrEqual(10);
  } finally {
    root.destroy();
  }
});
