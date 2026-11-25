import { summarizeSamplingTrace } from './sampling-trace.ts';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, tgpu, type TgpuRoot } from 'typegpu';
import { expect, it } from 'vitest';
import { createAtmosphere } from '../../../typegpu-kit-examples/src/ocean/atmosphere.ts';
import { cpuRoot } from './device.ts';
import { SweepGenerator } from './scene-generator.ts';
import { observeScene } from './scene-observer.ts';
import { replay, snapshotDispatch, type ReplaySnapshot } from './replay.ts';
import { recordResults } from './results.ts';

async function bake(root: TgpuRoot) {
  const atmosphere = createAtmosphere(root);
  const elevation = (0.5 * Math.PI) / 180,
    azimuth = 0.26;
  atmosphere.place(
    d.vec3f(
      Math.cos(elevation) * Math.sin(azimuth),
      Math.sin(elevation),
      -Math.cos(elevation) * Math.cos(azimuth),
    ),
    4,
  );
  await atmosphere.initAsync();
  const encoder = root.createCommandEncoder(),
    pass = encoder.beginComputePass();
  atmosphere.bake(pass);
  pass.end();
  encoder.submit();
  await root.device.queue.onSubmittedWorkDone();
  return atmosphere;
}

it.runIf(process.env.CPU_OCEAN_ATMOSPHERE === '1')(
  'isolates ocean atmospheric lookup-table arithmetic',
  async () => {
    const dawn = await gpuRoot(),
      gpuObserver = observeScene(),
      cpuObserver = observeScene();
    const gpu = tgpu.initFromDevice({ device: gpuObserver.device(dawn.device) });
    const snapshots = new Map<string, ReplaySnapshot>();
    const cpu = cpuRoot({
      generator: SweepGenerator,
      observeDevice: cpuObserver.device,
      beforeDispatch(code, label, dispatch) {
        const entry = (JSON.parse(/^\/\/#entry (.*)$/m.exec(code)![1]!) as { name: string }).name;
        if (!snapshots.has(entry))
          snapshots.set(entry, snapshotDispatch(code, label, dispatch, cpuObserver));
      },
    });
    try {
      const gpuAtmosphere = await bake(gpu),
        cpuAtmosphere = await bake(cpu);
      try {
        const results = [];
        for (const [entry, snapshot] of snapshots) {
          if (!new RegExp(process.env.CPU_ATMOSPHERE_ENTRY ?? '.*').test(entry)) continue;
          const result = await replay(dawn.device, gpuObserver.dispatches, snapshot, false, {
            sampling: process.env.CPU_SAMPLE_TRACE === '1',
            geometry: process.env.CPU_ATMOSPHERE_GEOMETRY === '1' && entry !== 'bakeLighting',
          });
          const samplingSummary = result.samplingTrace
            ? summarizeSamplingTrace(result.samplingTrace)
            : undefined;
          if (samplingSummary) {
            expect(samplingSummary.matched).toBe(true);
            expect(samplingSummary.boundsMisses).toBe(0);
            expect(samplingSummary.radiusMaxUlps).toBeLessThanOrEqual(4);
            expect(samplingSummary.storeMaxUlps).toBeLessThanOrEqual(1);
          }
          const measured = { samplingSummary, ...result };
          results.push({ entry, ...measured });
          recordResults(`oceanAtmosphere:${entry}`, measured);
          recordResults('oceanAtmosphere', results);
        }
        expect(results.length).toBeGreaterThan(0);
      } finally {
        gpuAtmosphere.destroy();
        cpuAtmosphere.destroy();
      }
    } finally {
      gpu.destroy();
      cpu.destroy();
      dawn.destroy();
    }
  },
  300000,
);
