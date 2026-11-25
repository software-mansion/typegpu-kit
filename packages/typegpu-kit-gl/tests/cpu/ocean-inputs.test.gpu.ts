import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';
import { SweepGenerator } from './scene-generator.ts';
import { observeScene } from './scene-observer.ts';
import { captureScene, compareExampleCaptures } from './example-comparison.ts';
import { classifyAmplification } from './amplification.ts';
import { recordResults } from './results.ts';

it.runIf(process.env.CPU_OCEAN_INPUTS === '1')(
  'separates ocean frequency quantization from independent resource differences',
  async () => {
    const dawn = await gpuRoot({
      optionalFeatures: ['indirect-first-instance', 'depth32float-stencil8'],
    });
    const observer = observeScene();
    const gpu = tgpu.initFromDevice({ device: observer.device(dawn.device) });
    try {
      const expected = await captureScene(gpu, observer, 'ocean');
      const frequencies = expected.resources.find((item) => item.label === 'omegas')!;
      expect(frequencies).toBeDefined();
      const omegaBytes = frequencies.levels[0]!.bytes;
      const runs = [];
      for (const equalize of [false, true]) {
        const cpuObserver = observeScene();
        const substitutions: { pipeline: string; changedWords: number }[] = [];
        const cpu = cpuRoot({
          generator: SweepGenerator,
          observeDevice: cpuObserver.device,
          onTexture: (texture, data) => cpuObserver.cpuTextures.set(texture, data),
          beforeDispatch(_code, pipeline, snapshot) {
            if (!equalize) return;
            for (const use of snapshot.module.bindings) {
              if (use.kind !== 'buffer' || !use.reads) continue;
              const entry = Array.from(snapshot.groups.get(use.group)!.entries).find(
                (item) => item.binding === use.binding,
              )!;
              const binding = entry.resource as GPUBufferBinding;
              if (cpuObserver.resourceOf(binding.buffer)?.label !== 'omegas') continue;
              const range = snapshot.resources.buffer(use.group, use.binding);
              expect(range.size).toBe(omegaBytes.byteLength);
              const target = new Uint32Array(range.bytes, range.offset, range.size / 4);
              const source = new Uint32Array(omegaBytes);
              let changedWords = 0;
              for (let i = 0; i < source.length; i++) if (target[i] !== source[i]) changedWords++;
              target.set(source);
              substitutions.push({ pipeline, changedWords });
            }
          },
        });
        try {
          const actual = await captureScene(cpu, cpuObserver, 'ocean');
          const result = classifyAmplification(
            'ocean',
            compareExampleCaptures('ocean', expected, actual),
          );
          runs.push({ equalize, substitutions, ...result });
          recordResults('oceanEqualFrequencies', runs);
          if (equalize) {
            expect(substitutions.length).toBeGreaterThan(0);
            expect(
              new Uint32Array(
                actual.resources.find((item) => item.label === 'omegas')!.levels[0]!.bytes,
              ),
            ).toEqual(new Uint32Array(omegaBytes));
          }
        } finally {
          cpu.destroy();
        }
      }
    } finally {
      gpu.destroy();
      dawn.destroy();
    }
  },
  600000,
);
