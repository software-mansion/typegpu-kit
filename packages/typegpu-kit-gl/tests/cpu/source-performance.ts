import { cpus } from 'node:os';
import { d, std, tgpu } from 'typegpu';
import assert from 'node:assert/strict';
import { cpuRoot, type DispatchSnapshot } from './device.ts';
import { recordResults } from './results.ts';
import { compile } from '../../src/cpu/index.ts';

const count = 16_384;
const Particle = d.struct({ position: d.vec3f, velocity: d.vec3f, life: d.f32, mass: d.f32 });
const Transform = d.struct({ matrix: d.mat4x4f, position: d.vec4f, destination: d.vec3f });

const particleSource = (row: d.Infer<typeof Particle>) => {
  'use gpu';
  let position = d.vec3f(row.position);
  const velocity = d.vec3f(row.velocity);
  for (let i = 0; i < 10; i++) {
    position = std.add(std.mul(position, 0.999), std.mul(velocity, 0.01));
  }
  return d.vec4f(position, row.life - 0.01);
};
const hashSource = (seed: number) => {
  'use gpu';
  let hash = d.u32(seed);
  for (let i = 0; i < 8; i++) {
    hash = d.u32(hash * d.u32(1664525) + d.u32(1013904223));
    hash = d.u32(hash ^ (hash >>> d.u32(13)));
  }
  return hash;
};
const transformSource = (row: d.Infer<typeof Transform>) => {
  'use gpu';
  const transformed = std.mul(row.matrix, row.position);
  const direction = std.normalize(transformed.xyz);
  const wave = std.sin(direction);
  const bend = std.cos(transformed.xyz);
  const mixed = std.mix(std.add(wave, bend), row.destination, 0.375);
  return d.vec4f(std.normalize(mixed), transformed.w);
};
const particleStep = tgpu.fn([Particle], d.vec4f)(particleSource);
const hashStep = tgpu.fn([d.u32], d.u32)(hashSource);
const transformStep = tgpu.fn([Transform], d.vec4f)(transformSource);

interface BenchmarkOptions {
  generator?: NonNullable<Parameters<typeof cpuRoot>[0]>['generator'];
  previousGenerator?: NonNullable<Parameters<typeof cpuRoot>[0]>['generator'];
  kernels?: string[];
  section?: string;
}
let generator: BenchmarkOptions['generator'];
let previousGenerator: BenchmarkOptions['generator'];
const gridSource = (coordinate: number) => {
  'use gpu';
  return d.u32(coordinate * 0.125);
};
const gridStep = tgpu.fn([d.f32], d.u32)(gridSource);

interface Measurement {
  repeats: number;
  samplesMs: number[];
  medianMs: number;
  invocationsPerSecond: number;
}
function measureAll(
  entries: { path: string; run: (repeats: number) => void; repeats: number }[],
): Record<string, Measurement> {
  const states = entries.map((entry) => ({ ...entry, samplesMs: [] as number[] }));
  let round = 0;
  while (states.some((state) => state.samplesMs.length < 15)) {
    // Rotate the order to share thermal and other-process variation between paths.
    for (let i = 0; i < states.length; i++) {
      const state = states[(round + i) % states.length]!;
      if (state.samplesMs.length >= 15) continue;
      const started = performance.now();
      state.run(state.repeats);
      const elapsed = performance.now() - started;
      if (elapsed < 20) {
        state.repeats *= 2;
        state.samplesMs.length = 0;
      } else {
        state.samplesMs.push(elapsed);
      }
    }
    round++;
  }
  return Object.fromEntries(
    states.map(({ path, repeats, samplesMs }) => {
      const medianMs = samplesMs.toSorted((a, b) => a - b)[7]!;
      return [
        path,
        { repeats, samplesMs, medianMs, invocationsPerSecond: (count * repeats * 1000) / medianMs },
      ];
    }),
  );
}
function warm(run: (repeats: number) => void): number {
  let repeats = 1;
  for (let i = 0; i < 8; i++) {
    const started = performance.now();
    run(repeats);
    if (performance.now() - started < 30) repeats *= 2;
  }
  return repeats;
}

async function benchmark<T extends d.AnyWgslData, O extends d.AnyWgslData>(
  name: string,
  inputSchema: T,
  outputSchema: O,
  inputs: d.Infer<T>[],
  step: (value: d.Infer<T>) => d.Infer<O>,
  raw: (value: d.Infer<T>) => d.Infer<O>,
) {
  let snapshot: DispatchSnapshot | undefined;
  let code = '';
  const root = cpuRoot({
    ...(generator ? { generator } : {}),
    onModule: (value) => (code = value),
    beforeDispatch: (_code, _label, value) => (snapshot = value),
  });
  const input = root.createReadonly(d.arrayOf(inputSchema, count), inputs);
  const output = root.createMutable(d.arrayOf(outputSchema, count));
  const entrySource = ({ id }: { id: d.v3u }): undefined => {
    'use gpu';
    output.$[id.x] = step(input.$[id.x] as d.Infer<T>) as d.InferGPU<O>;
  };
  const entry = tgpu.computeFn({ workgroupSize: [64], in: { id: d.builtin.globalInvocationId } })(
    entrySource,
  );
  root.createComputePipeline({ compute: entry }).dispatchWorkgroups(count / 64);
  const captured = snapshot!;
  const generated = (repeats: number) => {
    for (let repeat = 0; repeat < repeats; repeat++) {
      captured.module.dispatch(...captured.workgroups, captured.resources);
    }
  };
  // This diagnostic preserves f32 literals and buffer stores, but removes intermediate rounding.
  // Vector wrappers are plain arrays; arithmetic is emitted directly for these fixtures.
  assert.doesNotMatch(code, /\$construct\(/);
  const relaxed = compile(code.replaceAll('Math.fround(', '('));
  const generatedWithoutIntermediateRounding = (repeats: number) => {
    for (let repeat = 0; repeat < repeats; repeat++) {
      relaxed.dispatch(...captured.workgroups, captured.resources);
    }
  };
  let sourceOutput: d.Infer<O>[] = [];
  const source = (repeats: number) => {
    for (let repeat = 0; repeat < repeats; repeat++) {
      for (let i = 0; i < count; i++) sourceOutput[i] = step(inputs[i]!);
    }
  };
  const rawSource = (repeats: number) => {
    for (let repeat = 0; repeat < repeats; repeat++) {
      for (let i = 0; i < count; i++) sourceOutput[i] = raw(inputs[i]!);
    }
  };
  let simulated: d.Infer<O>[] = [];
  const simulate = (repeats: number) => {
    const result = tgpu['~unstable'].simulate(() => {
      for (let repeat = 0; repeat < repeats; repeat++) {
        for (let i = 0; i < count; i++) entrySource({ id: d.vec3u(i, 0, 0) });
      }
    });
    simulated = result.buffers.get(output.buffer) as d.Infer<O>[];
  };
  source(1);
  simulate(1);
  const actual = await output.read();
  let maximumAbsoluteError = 0;
  const compare = (expected: unknown, received: unknown): void => {
    if (typeof expected === 'number') {
      assert.equal(typeof received, 'number');
      const error = Math.abs(expected - (received as number));
      maximumAbsoluteError = Math.max(maximumAbsoluteError, error);
      assert.ok(
        error <= (name === 'integerHash' ? 0 : 2e-6 * Math.max(1, Math.abs(expected))),
        `${name}: ${expected} vs ${String(received)}`,
      );
    } else {
      for (const key of Object.keys(expected as object)) {
        compare(
          (expected as Record<string, unknown>)[key],
          (received as Record<string, unknown>)[key],
        );
      }
    }
  };
  for (let i = 0; i < count; i += 127) {
    compare(actual[i], sourceOutput[i]);
    compare(actual[i], simulated[i]);
    compare(actual[i], raw(inputs[i]!));
  }
  generatedWithoutIntermediateRounding(1);
  const relaxedOutput = await output.read();
  for (let i = 0; i < count; i += 127) compare(actual[i], relaxedOutput[i]);
  let previous: { root: ReturnType<typeof cpuRoot>; run: (repeats: number) => void } | undefined;
  if (previousGenerator) {
    let previousSnapshot: DispatchSnapshot | undefined;
    const previousRoot = cpuRoot({
      generator: previousGenerator,
      beforeDispatch: (_code, _label, value) => (previousSnapshot = value),
    });
    const previousInput = previousRoot.createReadonly(d.arrayOf(inputSchema, count), inputs);
    const previousOutput = previousRoot.createMutable(d.arrayOf(outputSchema, count));
    previousRoot
      .createComputePipeline({
        compute: tgpu.computeFn({ workgroupSize: [64], in: { id: d.builtin.globalInvocationId } })(
          ({ id }) => {
            'use gpu';
            previousOutput.$[id.x] = step(previousInput.$[id.x] as d.Infer<T>) as d.InferGPU<O>;
          },
        ),
      })
      .dispatchWorkgroups(count / 64);
    assert.deepEqual(await previousOutput.read(), actual);
    const previousCapture = previousSnapshot!;
    previous = {
      root: previousRoot,
      run: (repeats) => {
        for (let repeat = 0; repeat < repeats; repeat++)
          previousCapture.module.dispatch(...previousCapture.workgroups, previousCapture.resources);
      },
    };
  }
  const runners = {
    generated,
    generatedWithoutIntermediateRounding,
    source,
    rawSource,
    simulate,
    ...(previous ? { previousGenerated: previous.run } : {}),
  };
  const warmed = Object.entries(runners).map(([path, run]) => ({ path, run, repeats: warm(run) }));
  const measurements = measureAll(warmed) as Record<keyof typeof runners, Measurement>;
  // Retain observable results after timing so the source loops cannot become dead work.
  assert.equal(sourceOutput.length, count);
  assert.equal(simulated.length, count);
  const finalOutput = await output.read();
  for (let i = 0; i < count; i += 127) {
    compare(actual[i], sourceOutput[i]);
    compare(actual[i], simulated[i]);
    compare(actual[i], finalOutput[i]);
  }
  sourceOutput = [];
  previous?.root.destroy();
  root.destroy();
  return {
    name,
    count,
    maximumAbsoluteError,
    intermediateRoundingCalls: code.match(/Math\.fround\(/g)?.length ?? 0,
    measurements,
    roundingTimeOverhead:
      measurements.generatedWithoutIntermediateRounding.invocationsPerSecond /
        measurements.generated.invocationsPerSecond -
      1,
    speedupVsSource:
      measurements.generated.invocationsPerSecond / measurements.source.invocationsPerSecond,
    speedupVsRawSource:
      measurements.generated.invocationsPerSecond / measurements.rawSource.invocationsPerSecond,
    speedupVsSimulate:
      measurements.generated.invocationsPerSecond / measurements.simulate.invocationsPerSecond,
  };
}

export async function runSourcePerformance(options: BenchmarkOptions = {}): Promise<void> {
  generator = options.generator;
  previousGenerator = options.previousGenerator;
  const particles = Array.from({ length: count }, (_, i) =>
    Particle({
      position: d.vec3f((i % 97) / 32, -0.5, 1.25),
      velocity: d.vec3f(0.125, -0.25, (i % 17) / 32),
      life: 10,
      mass: 1,
    }),
  );
  const transforms = Array.from({ length: count }, (_, i) =>
    Transform({
      matrix: d.mat4x4f(
        d.vec4f(1, 0.125, 0.25, 0),
        d.vec4f(-0.25, 1, 0.125, 0),
        d.vec4f(0.125, -0.25, 1, 0),
        d.vec4f((i % 13) / 32, 0.5, -0.25, 1),
      ),
      position: d.vec4f((i % 97) / 32, -0.5, 1.25, 1),
      destination: d.vec3f(0.5, 0.25, 1),
    }),
  );
  const runs = {
    particles: () =>
      benchmark('particles', Particle, d.vec4f, particles, particleStep, particleSource),
    integerHash: () =>
      benchmark(
        'integerHash',
        d.u32,
        d.u32,
        Array.from({ length: count }, (_, i) => Math.imul(i, 2654435761) >>> 0),
        hashStep,
        hashSource,
      ),
    transforms: () =>
      benchmark('transforms', Transform, d.vec4f, transforms, transformStep, transformSource),
    floatIndices: () =>
      benchmark(
        'floatIndices',
        d.f32,
        d.u32,
        Array.from({ length: count }, (_, i) => i + 0.5),
        gridStep,
        gridSource,
      ),
  };
  const results = [];
  for (const name of options.kernels ?? ['particles', 'integerHash', 'transforms']) {
    assert.ok(name in runs, `Unknown benchmark ${name}`);
    results.push(await runs[name as keyof typeof runs]());
  }
  recordResults(options.section ?? 'sourcePerformance', {
    node: process.version,
    cpu: cpus()[0]?.model,
    date: new Date().toISOString(),
    execution: 'native Node imports; all paths in one process; interleaved rotating sample order',
    validationRows: Math.ceil(count / 127),
    relaxedSemantics: 'remove generated Math.fround calls; retain f32 literals and buffer stores',
    samples: 15,
    minimumSampleMs: 20,
    results,
  });
}
