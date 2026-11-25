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

it.runIf(process.env.CPU_OCEAN_KERNELS === '1')(
  'replays the ocean resources independent of wave frequencies',
  async () => {
    const dawn = await gpuRoot({
      optionalFeatures: ['indirect-first-instance', 'depth32float-stencil8'],
    });
    Object.defineProperty(dawn.device, backends.brand, { value: cpuBackend });
    const gpuObserver = observeScene(),
      cpuObserver = observeScene();
    const gpu = tgpu.initFromDevice({ device: gpuObserver.device(dawn.device) });
    Object.defineProperty(gpu, 'enabledWgslLanguageFeatures', { value: new Set<string>() });
    const snapshots = new Map<string, ReplaySnapshot>();
    const cpu = cpuRoot({
      generator: SweepGenerator,
      observeDevice: cpuObserver.device,
      beforeDispatch(code, label, dispatch) {
        if (dispatch.workgroups.some((count) => count === 0)) return;
        const targets = dispatch.module.bindings
          .filter((use) => use.writes)
          .flatMap((use) => {
            const entry = Array.from(dispatch.groups.get(use.group)!.entries).find(
              (item) => item.binding === use.binding,
            )!;
            const handle = (entry.resource as GPUBufferBinding).buffer ?? entry.resource;
            const resource = cpuObserver.resourceOf(handle);
            if (!resource) return [];
            if (resource.kind === 'buffer' && !['sunlight', 'shBuffer'].includes(resource.label))
              return [];
            if (
              resource.kind === 'texture' &&
              !/^texture:(?:tile:host#0|<unnamed>:typegpu-kit-examples\/src\/ocean\/clouds.ts#[012]|<unnamed>:host#[0156]|texture:host#[012])$/.test(
                resource.key,
              )
            )
              return [];
            const shape =
              use.kind === 'buffer'
                ? ''
                : dispatch.resources
                    .texture(use.group, use.binding)
                    .levels.map((level) =>
                      [level.width, level.height, level.depthOrArrayLayers].join('x'),
                    )
                    .join('/');
            return [`${resource.key}:${shape}`];
          });
        if (!targets.length) return;
        const entry = (JSON.parse(/^\/\/#entry (.*)$/m.exec(code)![1]!) as { name: string }).name;
        const key = `${entry}:${targets.toSorted().join(',')}`;
        if (!snapshots.has(key))
          snapshots.set(key, snapshotDispatch(code, label, dispatch, cpuObserver));
      },
    });
    try {
      await captureScene(gpu, gpuObserver, 'ocean', false);
      await captureScene(cpu, cpuObserver, 'ocean', false);
      expect(snapshots.size).toBeGreaterThan(0);
      const results = [];
      for (const [key, snapshot] of snapshots) {
        const result = await replay(dawn.device, gpuObserver.dispatches, snapshot, false, {
          sampling:
            !!process.env.CPU_OCEAN_TRACE_KEYS &&
            new RegExp(process.env.CPU_OCEAN_TRACE_KEYS).test(key),
          geometry:
            process.env.CPU_OCEAN_GEOMETRY === '1' && /^bake(?:Transmittance|SkyView):/.test(key),
          prefilter: process.env.CPU_OCEAN_PREFILTER === '1' && key.startsWith('prefilter:'),
        });
        const samplingSummary = result.samplingTrace
          ? summarizeSamplingTrace(result.samplingTrace)
          : undefined;
        results.push({ key, samplingSummary, ...result });
        recordResults('oceanIndependentKernels', results);
        if (result.prefilterTrace) {
          for (const row of result.prefilterTrace) {
            expect(
              row.cpu.filter((value) => value.tag === 4),
              key,
            ).toHaveLength(64);
            expect(
              row.dawn.filter((value) => value.tag === 4),
              key,
            ).toHaveLength(64);
            expect(row.sameSamplesMaxHalfUlps, key).toBeLessThanOrEqual(1);
            for (const flip of row.branchFlips) expect(flip.delta, key).toBeLessThanOrEqual(1e-6);
            for (const sample of row.samples)
              for (let c = 0; c < 4; c++) {
                expect(sample.dawn![c]!, key).toBeGreaterThanOrEqual(sample.envelope[c]![0]!);
                expect(sample.dawn![c]!, key).toBeLessThanOrEqual(sample.envelope[c]![1]!);
              }
          }
        }
        if (samplingSummary) {
          expect(samplingSummary.matched, key).toBe(true);
          expect(samplingSummary.boundsMisses, key).toBe(0);
          expect(samplingSummary.storeMaxUlps, key).toBeLessThanOrEqual(1);
          expect(samplingSummary.radiusMaxUlps, key).toBeLessThanOrEqual(4);
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
