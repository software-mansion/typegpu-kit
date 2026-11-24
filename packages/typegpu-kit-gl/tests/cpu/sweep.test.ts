import { cpuRoot } from './device.ts';
import { writeFile } from 'node:fs/promises';
import { writeFileSync } from 'node:fs';
import { Session } from 'node:inspector/promises';
import { frames } from '@typegpu/kit';
import { offscreenSurface, recordingRoot } from '@typegpu/kit/testing';
import { tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { examples } from '../../../typegpu-kit-examples/src/index.ts';
import { begin, filesOf } from '../../../typegpu-kit-examples/src/hosts.ts';
import { fromDisk, driven } from '../../../typegpu-kit-examples/tests/utils/examples.ts';
import { SweepGenerator } from './scene-generator.ts';
import { compile, isCpuModule } from '../../src/cpu/index.ts';

it.runIf(process.env.CPU_SWEEP === '1')(
  'records per-example CPU compilation coverage',
  async () => {
    const results: { example: string; status: string; modules: number; error?: string }[] = [];
    for (const example of examples.filter(
      (candidate) =>
        (!process.env.CPU_EXAMPLES ||
          process.env.CPU_EXAMPLES.split(',').includes(candidate.key)) &&
        !process.env.CPU_SKIP?.split(',').includes(candidate.key),
    )) {
      const recording = recordingRoot();
      const dispatched = new Map<
        string,
        { calls: number; milliseconds: number; invocations: number }
      >();
      let steady = false;
      const steadyPipelines = new Map<
        string,
        { pipeline: string; calls: number; milliseconds: number; invocations: number }
      >();
      let profileDispatches = 0;
      const steadyDispatches: { calls: number; milliseconds: number; invocations: number } = {
        calls: 0,
        milliseconds: 0,
        invocations: 0,
      };
      const session = process.env.CPU_PROFILE === '1' ? new Session() : undefined;
      let profiling = false;
      session?.connect();
      if (session) {
        await session.post('Profiler.enable');
        if (process.env.CPU_PROFILE_PHASE !== 'steady') {
          await session.post('Profiler.start');
          profiling = true;
        }
      }
      const root =
        process.env.CPU_EXECUTE === '1'
          ? cpuRoot({
              generator: SweepGenerator,
              onDispatch: (code, milliseconds, invocations, label) => {
                if (
                  process.env.CPU_PROFILE_DISPATCHES &&
                  ++profileDispatches >= Number(process.env.CPU_PROFILE_DISPATCHES)
                )
                  throw new Error('Profile dispatch limit reached');
                if (steady) {
                  const held = steadyPipelines.get(code) ?? {
                    pipeline: label,
                    calls: 0,
                    milliseconds: 0,
                    invocations: 0,
                  };
                  held.calls++;
                  held.milliseconds += milliseconds;
                  held.invocations += invocations;
                  steadyPipelines.set(code, held);
                  steadyDispatches.calls++;
                  steadyDispatches.milliseconds += milliseconds;
                  steadyDispatches.invocations += invocations;
                }
                const previous = dispatched.get(code) ?? {
                  calls: 0,
                  milliseconds: 0,
                  invocations: 0,
                };
                dispatched.set(code, {
                  calls: previous.calls + 1,
                  milliseconds: previous.milliseconds + milliseconds,
                  invocations: previous.invocations + invocations,
                });
              },
              onModule: (code) => {
                recording.log.shaders.push(code);
                if (process.env.CPU_DIAGNOSTICS === '1' && isCpuModule(code))
                  writeFileSync(
                    new URL(
                      `./module-${recording.log.shaders.length}.generated.txt`,
                      import.meta.url,
                    ),
                    code,
                  );
              },
            })
          : tgpu.initFromDevice({
              device: recording.root.device,
              unstable_shaderGeneratorClass: SweepGenerator,
            });
      const surface = offscreenSurface(root, { width: 64, height: 36, format: 'bgra8unorm' });
      let scene:
        | Awaited<ReturnType<Awaited<ReturnType<typeof example.scene>>['createScene']>>
        | undefined;
      let loop: ReturnType<typeof frames.create> | undefined;
      try {
        const { createScene } = await example.scene();
        const reduced: Record<string, object> = {
          'slime-mould': { grid: 64, count: 50_000 },
          stampede: { count: 256 },
          forest: { trees: 40 },
          'star-map': { count: 256 },
          'quiver-trees': { count: 60 },
          skeletons: { count: 8 },
        };
        scene = await begin(
          await createScene(root, surface, {
            ...(await filesOf(example.key, fromDisk)),
            ...(process.env.CPU_EXECUTE === '1' && process.env.CPU_DEFAULTS !== '1'
              ? reduced[example.key]
              : {}),
          }),
        );
        loop = frames.create(root, scene.frame);
        const driver = driven(loop, scene.ready);
        await driver.warmUp();
        await driver.run(2);
        steady = true;
        if (session && process.env.CPU_PROFILE_PHASE === 'steady') {
          await session.post('Profiler.start');
          profiling = true;
        }
        await driver.run(2);
        steady = false;
        const rawCompute = recording.log.shaders.find(
          (code) => !isCpuModule(code) && /@compute\b/.test(code),
        );
        if (rawCompute) throw new Error('Compute module bypassed CpuGenerator: raw WGSL');
        const modules = recording.log.shaders.filter(isCpuModule);
        for (const [index, code] of modules.entries()) {
          if (process.env.CPU_DIAGNOSTICS === '1')
            writeFileSync(
              new URL(`./${example.key}-${index}.generated.txt`, import.meta.url),
              code,
            );
          try {
            compile(code);
          } catch (error) {
            if (process.env.CPU_DIAGNOSTICS === '1')
              await writeFile(new URL('./failure.generated.txt', import.meta.url), code);
            throw error;
          }
        }
        results.push({
          example: example.key,
          status: 'pass',
          modules: modules.length,
          ...(process.env.CPU_EXECUTE === '1'
            ? {
                executedModules: dispatched.size,
                steadyPipelines: [...steadyPipelines]
                  .map(([code, timing]) => ({
                    entry: JSON.parse(/^\/\/#entry (.*)$/m.exec(code)![1]!).name,
                    pipeline: timing.pipeline,
                    calls: timing.calls / 2,
                    milliseconds: timing.milliseconds / 2,
                    invocations: timing.invocations / 2,
                  }))
                  .toSorted((a, b) => b.milliseconds - a.milliseconds),
                unexecutedEntries: modules
                  .filter((code) => !dispatched.has(code))
                  .map((code) => JSON.parse(/^\/\/#entry (.*)$/m.exec(code)![1]!).name),
                perFrame: {
                  dispatches: steadyDispatches.calls / 2,
                  invocations: steadyDispatches.invocations / 2,
                  milliseconds: steadyDispatches.milliseconds / 2,
                },
                dispatches: [...dispatched].map(([code, timing]) => ({
                  entry: JSON.parse(/^\/\/#entry (.*)$/m.exec(code)![1]!).name,
                  ...timing,
                })),
              }
            : {}),
        });
      } catch (error) {
        results.push({
          example: example.key,
          status: 'fail',
          modules: recording.log.shaders.filter(isCpuModule).length,
          error: String(error),
          ...(process.env.CPU_EXAMPLES
            ? {
                stack:
                  error instanceof AggregateError
                    ? error.errors.map((item) => ({ stack: item.stack, cause: item.cause?.stack }))
                    : error instanceof Error
                      ? error.stack
                      : undefined,
              }
            : {}),
        });
      } finally {
        if (session && profiling) {
          const { profile } = await session.post('Profiler.stop');
          await writeFile(
            new URL(`./${example.key}.cpuprofile`, import.meta.url),
            JSON.stringify(profile),
          );
        }
        session?.disconnect();
        loop?.destroy();
        scene?.destroy?.();
        surface.destroy();
        root.destroy();
        if (process.env.CPU_EXECUTE === '1')
          await writeFile(
            new URL(process.env.CPU_RESULTS ?? './execution-results.json', import.meta.url),
            JSON.stringify(results, null, 2),
          );
      }
    }
    await writeFile(
      new URL(
        process.env.CPU_EXECUTE === '1'
          ? (process.env.CPU_RESULTS ?? './execution-results.json')
          : './sweep-results.json',
        import.meta.url,
      ),
      `${JSON.stringify(results, null, 2)}\n`,
    );
    expect(results.filter((result) => result.status === 'fail')).toEqual([]);
  },
  600000,
);
