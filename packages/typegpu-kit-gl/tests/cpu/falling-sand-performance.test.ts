import { cameras, counters } from '@typegpu/kit';
import { expect, it } from 'vitest';
import {
  createCells,
  HEIGHT,
  WIDTH,
} from '../../../typegpu-kit-examples/src/falling-sand/cells.ts';
import { createCave } from '../../../typegpu-kit-examples/src/falling-sand/cave.ts';
import { createEmbers } from '../../../typegpu-kit-examples/src/falling-sand/embers.ts';
import { createRules } from '../../../typegpu-kit-examples/src/falling-sand/rules.ts';
import { cpuRoot, type DispatchSnapshot } from './device.ts';
import { recordResults } from './results.ts';
import { writeFileSync } from 'node:fs';

const median = (values: number[]) =>
  values.toSorted((a, b) => a - b)[Math.floor(values.length / 2)]!;

it.runIf(process.env.CPU_FALLING_SAND_PERF === '1')(
  'measures falling-sand simulation after its 450-step prewarm',
  async () => {
    const captures = new Map<
      string,
      {
        snapshot: DispatchSnapshot;
        code: string;
        entry: string;
        samples: number[];
        invocations: number;
      }
    >();
    let steady = false;
    let prewarming = false;
    const firstSteps: number[] = [];
    const root = cpuRoot({
      onDispatch(code, milliseconds, invocations, _label, snapshot) {
        const entry = JSON.parse(
          code
            .split('\n')
            .find((line) => line.startsWith('//#entry '))!
            .slice(9),
        ).name as string;
        if (prewarming && /^step(?:_\d+)?$/.test(entry) && firstSteps.length < 50)
          firstSteps.push(milliseconds);
        if (!steady || (!/^step(?:_\d+)?$/.test(entry) && entry !== 'flow')) return;
        const held = captures.get(code) ?? { snapshot, code, entry, samples: [], invocations };
        held.samples.push(milliseconds);
        captures.set(code, held);
      },
    });
    const cells = createCells(root);
    const cave = createCave(root, cells);
    const counts = counters.create(root, ['cooled', 'burnt']);
    const camera = cameras.flat(root, {
      size: { width: WIDTH * 4, height: HEIGHT * 4 },
      height: HEIGHT,
      center: [WIDTH / 2, HEIGHT / 2],
    });
    const { sparks, spark, sparked } = createEmbers(root, { camera, step: 1 / 150, cells, counts });
    const rules = createRules(root, { cells, counts, spark, sparked });
    try {
      await Promise.all([cave.initAsync(), rules.initAsync(), sparks.initAsync()]);
      const initial = root.createCommandEncoder();
      cave.record(initial);
      initial.submit();
      const advance = (steps: number) => {
        const encoder = root.createCommandEncoder();
        const pass = encoder.beginComputePass();
        for (let i = 0; i < steps; i++) {
          rules.step(pass);
          sparks.step(pass);
        }
        pass.end();
        encoder.submit();
      };
      const prewarmStart = performance.now();
      prewarming = true;
      for (let chunk = 0; chunk < 15; chunk++) advance(30);
      prewarming = false;
      const prewarmMilliseconds = performance.now() - prewarmStart;
      advance(5);
      steady = true;
      for (let sample = 0; sample < 9; sample++) advance(5);
      steady = false;
      const dispatches = [...captures.values()].map(
        ({ snapshot, code, entry, samples, invocations }, index) => {
          if (process.env.CPU_DIAGNOSTICS === '1')
            writeFileSync(new URL(`./falling-sand-${index}.generated.txt`, import.meta.url), code);
          const { module, resources } = snapshot;
          const setup = Array.from({ length: 9 }, () => {
            const start = performance.now();
            for (let repeat = 0; repeat < 100; repeat++) module.dispatch(0, 1, 1, resources);
            return (performance.now() - start) / 100;
          });
          const fixedMilliseconds = median(setup);
          const milliseconds = median(samples);
          return {
            entry,
            workgroups: snapshot.workgroups,
            invocations,
            milliseconds,
            fixedMilliseconds,
            nanosecondsPerInvocation: ((milliseconds - fixedMilliseconds) * 1e6) / invocations,
            measurements: samples.length,
          };
        },
      );
      recordResults('fallingSandSimulation', {
        prewarmSteps: 450,
        first50StepMilliseconds: firstSteps,
        firstStepMilliseconds: firstSteps[0],
        first50StepMedian: median(firstSteps),
        first5StepMedian: median(firstSteps.slice(0, 5)),
        last5OfFirst50StepMedian: median(firstSteps.slice(-5)),
        prewarmMilliseconds,
        steadySteps: 45,
        dispatches,
        note: 'Same cells/cave/rules/embers as the scene. Light/render passes omitted. Steady medians exclude prewarm. Fixed cost measured with zero workgroups over the same bindings.',
      });
      expect(dispatches.filter((dispatch) => dispatch.entry === 'flow')).toHaveLength(4);
      expect(dispatches.filter((dispatch) => /^step(?:_\d+)?$/.test(dispatch.entry))).toHaveLength(
        1,
      );
    } finally {
      rules.destroy();
      sparks.destroy();
      cave.destroy();
      cells.destroy();
      counts.destroy();
      root.destroy();
    }
  },
  120_000,
);
