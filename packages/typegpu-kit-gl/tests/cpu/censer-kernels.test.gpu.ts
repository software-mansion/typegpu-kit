import { summarizeSamplingTrace } from './sampling-trace.ts';
import { backends } from '@typegpu/kit';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { cpuBackend, cpuRoot } from './device.ts';
import { SweepGenerator } from './scene-generator.ts';
import { observeScene } from './scene-observer.ts';
import { captureScene } from './example-comparison.ts';
import { replay, snapshotDispatch, type ReplaySnapshot } from './replay.ts';
import { recordResults } from './results.ts';

it.runIf(process.env.CPU_CENSER_KERNELS === '1')(
  'isolates censer pressure and velocity writers',
  async () => {
    const dawn = await gpuRoot();
    Object.defineProperty(dawn.device, backends.brand, { value: cpuBackend });
    const gpuObserver = observeScene(),
      cpuObserver = observeScene();
    const gpu = tgpu.initFromDevice({ device: gpuObserver.device(dawn.device) });
    Object.defineProperty(gpu, 'enabledWgslLanguageFeatures', { value: new Set<string>() });
    const snapshots = new Map<string, ReplaySnapshot>();
    let advections = 0;
    const cpu = cpuRoot({
      generator: SweepGenerator,
      observeDevice: cpuObserver.device,
      beforeDispatch(code, label, dispatch) {
        if (label === 'advect') advections++;
        if (advections < 2 && !(advections === 1 && ['solids', 'curl'].includes(label))) return;
        const targets = dispatch.module.bindings
          .filter((use) => use.writes && use.kind === 'storageTexture')
          .flatMap((use) => {
            const entry = Array.from(dispatch.groups.get(use.group)!.entries).find(
              (item) => item.binding === use.binding,
            )!;
            const resource = cpuObserver.resourceOf(entry.resource);
            if (!resource || !/^(smoke|pressure) /.test(resource.label)) return [];
            const level = dispatch.resources.texture(use.group, use.binding).levels[0]!;
            return [`${resource.label}:${level.width}x${level.height}x${level.depthOrArrayLayers}`];
          });
        if (!targets.length) return;
        const entry = (JSON.parse(/^\/\/#entry (.*)$/m.exec(code)![1]!) as { name: string }).name;
        const key = `${label}:${entry}:${targets.toSorted().join(',')}`;
        if (!snapshots.has(key))
          snapshots.set(key, snapshotDispatch(code, label, dispatch, cpuObserver));
      },
    });
    try {
      await captureScene(gpu, gpuObserver, 'censer', false);
      await captureScene(cpu, cpuObserver, 'censer', false);
      expect(snapshots.size).toBeGreaterThan(0);
      expect(snapshots.size).toBe(27);
      const results = [];
      for (const [key, snapshot] of snapshots) {
        const result = await replay(dawn.device, gpuObserver.dispatches, snapshot, false, {
          sampling: process.env.CPU_CENSER_TRACE === '1' && snapshot.label === 'matter',
        });
        const samplingSummary = result.samplingTrace
          ? summarizeSamplingTrace(result.samplingTrace)
          : undefined;
        results.push({ key, samplingSummary, ...result });
        recordResults('censerIndependentKernels', results);
        if (!['advect', 'matter'].includes(snapshot.label))
          expect(result.differences, key).toEqual([]);
        if (samplingSummary) {
          expect(samplingSummary.matched).toBe(true);
          expect(samplingSummary.boundsMisses).toBe(0);
          expect(samplingSummary.storeMaxUlps).toBeLessThanOrEqual(1);
        }
      }
    } finally {
      gpu.destroy();
      cpu.destroy();
      dawn.destroy();
    }
  },
  900000,
);
