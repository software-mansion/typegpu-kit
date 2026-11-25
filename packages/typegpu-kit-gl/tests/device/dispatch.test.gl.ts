import { d, tgpu } from 'typegpu';
import { expect, test } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { glRoot } from './compare.ts';

const COUNT = 256;

test.each([
  ['its GPU form', true],
  ['its CPU module', false],
] as const)(
  'dispatch counts convert as WebIDL converts a GPUSize32, on %s',
  async (_, gpuCompute) => {
    const root = glRoot({ gpuCompute, placement: false });
    try {
      const errors: string[] = [];
      root.device.addEventListener('uncapturederror', (e) =>
        errors.push((e as GPUUncapturedErrorEvent).error.message),
      );
      const out = root.createMutable(d.arrayOf(d.u32, COUNT));
      const kernel = tgpu.computeFn({
        workgroupSize: [64],
        in: { id: d.builtin.globalInvocationId },
      })(({ id }) => {
        'use gpu';
        out.$[id.x] = out.$[id.x]! + 1;
      });
      const pipeline = root.createComputePipeline({ compute: kernel }).$name('counting');
      const dispatch = (...counts: unknown[]) =>
        (pipeline.dispatchWorkgroups as (...counts: unknown[]) => void)(...counts);
      const ran = (workgroups: number) =>
        Array.from({ length: COUNT }, (__, i) => (i < workgroups * 64 ? 1 : 0));

      dispatch('3');
      expect(await out.read()).toEqual(ran(3));
      dispatch(-0);
      dispatch(0.9);
      dispatch(2 ** 32 - 1, 1, 0);
      dispatch(4, '0');
      expect(await out.read()).toEqual(ran(3));
      dispatch(1.7, undefined, undefined);
      dispatch(-0.5);
      expect(await out.read()).toEqual(ran(3).map((each, i) => each + (i < 64 ? 1 : 0)));
      for (const count of [Number.NaN, Number.POSITIVE_INFINITY, -1, 2 ** 32, 'three', 1n]) {
        expect(() => dispatch(count), String(count)).toThrow(TypeError);
      }
      expect(() => dispatch(1, Number.NaN)).toThrow(TypeError);
      expect(await out.read()).toEqual(ran(3).map((each, i) => each + (i < 64 ? 1 : 0)));
      expect(errors).toEqual([]);
      expect((root.device as GLDevice).gl.getError()).toBe(0);
      if (gpuCompute)
        expect([...(root.device as GLDevice).stats.kernels.keys()]).toEqual(['counting (gpu)']);
    } finally {
      root.destroy();
    }
  },
);
