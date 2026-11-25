import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { examples } from '../../../typegpu-kit-examples/src/index.ts';
import { cpuRoot } from './device.ts';
import { SweepGenerator } from './scene-generator.ts';
import { observeScene } from './scene-observer.ts';
import { captureScene, compareCaptures } from './example-comparison.ts';
import { recordResults } from './results.ts';

for (const example of examples.filter(
  (item) => !process.env.CPU_EXAMPLES || process.env.CPU_EXAMPLES.split(',').includes(item.key),
)) {
  it.runIf(process.env.CPU_EXAMPLE_COMPARE === '1')(
    `compares seeded ${example.key} compute resources with dawn`,
    async () => {
      const dawn = await gpuRoot();
      const gpuObserver = observeScene(),
        cpuObserver = observeScene();
      const gpu = tgpu.initFromDevice({ device: gpuObserver.device(dawn.device) });
      const cpu = cpuRoot({
        generator: SweepGenerator,
        observeDevice: cpuObserver.device,
        onTexture: (texture, data) => cpuObserver.cpuTextures.set(texture, data),
      });
      try {
        const expected = await captureScene(gpu, gpuObserver, example.key);
        const actual = await captureScene(cpu, cpuObserver, example.key);
        const result = compareCaptures(expected, actual);
        recordResults(`comparison:${example.key}`, result);
        expect(
          result.differences.filter((difference) => difference.order.length === 0),
          `${example.key}: deterministic resource differences; atomic allocation differences and excluded resources are in results.json`,
        ).toEqual([]);
      } finally {
        gpu.destroy();
        cpu.destroy();
        dawn.destroy();
      }
    },
    600000,
  );
}
