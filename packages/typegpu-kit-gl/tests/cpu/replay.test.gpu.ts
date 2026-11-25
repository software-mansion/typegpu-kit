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

const batch = [
  { example: 'spider', label: '.*', resource: 'buffer:grid:', skip: 0 },
  { example: 'survey-model', label: '^tracing$', resource: 'buffer:contours_points:', skip: 0 },
  { example: 'stampede', label: '.*', resource: 'buffer:positionData:.*#0$', skip: 0 },
  { example: 'jellies', label: '^trailing$', resource: 'buffer:pointBuffer:', skip: 15 },
  { example: 'sph-sculpt', label: '^heighting$', resource: 'buffer:heights:', skip: 0 },
  { example: 'censer', label: '^advect$', resource: 'texture:smoke advected:', skip: 1 },
];
const probes =
  process.env.CPU_REPLAY_BATCH === '1'
    ? batch
    : [
        {
          example: process.env.CPU_EXAMPLES ?? 'flow-field',
          label: process.env.CPU_REPLAY_PIPELINE ?? 'step',
          resource: process.env.CPU_REPLAY_RESOURCE ?? '.*',
          skip: Number(process.env.CPU_REPLAY_SKIP ?? 0),
        },
      ];
for (const probe of probes)
  it.runIf(process.env.CPU_REPLAY === '1' || process.env.CPU_REPLAY_BATCH === '1')(
    `replays ${probe.example} ${probe.label} from identical buffer and texture bytes`,
    async () => {
      const { example, skip } = probe;
      const label = new RegExp(probe.label);
      const resource = new RegExp(probe.resource);
      const dawn = await gpuRoot({
        optionalFeatures: ['indirect-first-instance', 'depth32float-stencil8'],
      });
      Object.defineProperty(dawn.device, backends.brand, { value: cpuBackend });
      const gpuObserver = observeScene(),
        cpuObserver = observeScene();
      const gpu = tgpu.initFromDevice({ device: gpuObserver.device(dawn.device) });
      Object.defineProperty(gpu, 'enabledWgslLanguageFeatures', { value: new Set<string>() });
      let snapshot: ReplaySnapshot | undefined;
      const seen = new Set<string>();
      let matched = 0;
      const stop = new Error('CPU replay inputs captured');
      const cpu = cpuRoot({
        generator: SweepGenerator,
        observeDevice: cpuObserver.device,
        beforeDispatch(code, pipeline, dispatch) {
          seen.add(pipeline);
          if (!label.test(pipeline)) return;
          const keys = dispatch.module.bindings
            .filter((use) => use.writes)
            .flatMap((use) => {
              const entry = Array.from(dispatch.groups.get(use.group)!.entries).find(
                (item) => item.binding === use.binding,
              )!;
              const buffer = entry.resource as GPUBufferBinding;
              const item = cpuObserver.resourceOf(buffer.buffer ?? entry.resource);
              return item ? [item.key] : [];
            });
          if (!keys.some((key) => resource.test(key)) || matched++ < skip) return;
          snapshot = snapshotDispatch(code, pipeline, dispatch, cpuObserver);
          throw stop;
        },
      });
      try {
        await captureScene(gpu, gpuObserver, example, false);
        try {
          await captureScene(cpu, cpuObserver, example, false);
        } catch (error) {
          if (!snapshot) throw error;
        }
        expect(
          snapshot,
          `No ${label} / ${resource} dispatch; saw ${[...seen].join(', ')}`,
        ).toBeDefined();
        const result = await replay(dawn.device, gpuObserver.dispatches, snapshot!);
        recordResults(`replay:${example}:${snapshot!.label}:${skip}`, result);
        expect(result.differences).toEqual([]);
      } finally {
        gpu.destroy();
        cpu.destroy();
        dawn.destroy();
      }
    },
    600000,
  );
