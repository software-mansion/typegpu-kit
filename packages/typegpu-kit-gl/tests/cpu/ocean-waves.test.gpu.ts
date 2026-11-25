import { backends } from '@typegpu/kit';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { tgpu, type TgpuRoot } from 'typegpu';
import { expect, it } from 'vitest';
import { createWaves } from '../../../typegpu-kit-examples/src/ocean/waves.ts';
import {
  CASCADES,
  N,
  foamingAt,
  spectrumAt,
  cascadeValues,
} from '../../../typegpu-kit-examples/src/ocean/spectrum.ts';
import { cpuBackend, cpuRoot } from './device.ts';
import { SweepGenerator } from './scene-generator.ts';
import { observeScene } from './scene-observer.ts';
import {
  readBuffer,
  readTexture,
  compareCaptures,
  type SceneCapture,
  type CapturedResource,
} from './example-comparison.ts';
import { encodeHalf } from '../../src/cpu/textures.ts';
import { replay, snapshotDispatch, type ReplaySnapshot } from './replay.ts';
import { recordResults } from './results.ts';

const ordered = (bits: number) => (bits & 0x8000 ? -(bits & 0x7fff) : bits);

async function wavesCapture(
  root: TgpuRoot,
  observer: ReturnType<typeof observeScene>,
  inputs?: Map<string, ArrayBuffer>,
  retainLattice = false,
) {
  const waves = createWaves(root);
  const historyTrace: {
    time: number;
    dt: number;
    rows: { cell: number; signals: number[]; foam: number }[];
  }[] = [];
  const traceHistory = async (time: number, dt: number) => {
    if (process.env.CPU_OCEAN_HISTORY_TRACE !== '1' || backends.of(root).compute !== 'cpu') return;
    const latticeResource = observer.resources.find((resource) => resource.label === 'lattice')!;
    const foamResource = observer.resources.find((resource) => resource.label === 'foam')!;
    const lattice = new Float32Array(
      await readBuffer(root.device, latticeResource.handle as GPUBuffer),
    );
    const foam = new Float32Array(await readBuffer(root.device, foamResource.handle as GPUBuffer));
    const cells = [
      262360, 263130, 265585, 266063, 267002, 269845, 271600, 274444, 274755, 274763,
    ].map((cell) => {
      const cascade = Math.floor(cell / (N * N)),
        x = Math.floor(cell / N) % N,
        y = cell % N;
      return (cascade * N + y) * N + x;
    });
    historyTrace.push({
      time,
      dt,
      rows: cells.map((cell) => {
        const cascade = Math.floor(cell / (N * N)),
          x = cell % N,
          y = Math.floor(cell / N) % N;
        const base = (cascade * 4 * N + y) * N + x;
        return {
          cell,
          signals: Array.from(
            { length: 8 },
            (_, component) =>
              lattice[(base + Math.floor(component / 2) * N * N) * 2 + (component % 2)]!,
          ),
          foam: foam[cell]!,
        };
      }),
    });
  };
  try {
    await waves.initAsync();
    waves.spectrum.write(spectrumAt(20, Math.PI / 2));
    waves.foaming.write(foamingAt(20));
    const initial = root.createCommandEncoder();
    const setup = initial.beginComputePass();
    waves.rebuild(setup);
    setup.end();
    initial.submit();
    await root.device.queue.onSubmittedWorkDone();
    for (const resource of observer.resources) {
      const bytes = inputs?.get(resource.label);
      if (bytes && resource.kind === 'buffer')
        root.device.queue.writeBuffer(resource.handle as GPUBuffer, 0, bytes);
    }
    if (process.env.CPU_OCEAN_HISTORY === '1')
      for (let history = 0; history < 24; history++) {
        waves.step.write({ time: 195.4 - (24 - history) * 0.125, dt: 0.125 });
        const encoder = root.createCommandEncoder(),
          pass = encoder.beginComputePass();
        waves.record(pass);
        pass.end();
        waves.recordMips(encoder);
        encoder.submit();
        await root.device.queue.onSubmittedWorkDone();
        await traceHistory(195.4 - (24 - history) * 0.125, 0.125);
      }
    for (let frame = 0; frame < 2; frame++) {
      waves.step.write({ time: 195.4 + frame * 0.04, dt: frame === 0 ? 0 : 0.04 });
      const encoder = root.createCommandEncoder();
      const pass = encoder.beginComputePass();
      waves.record(pass);
      pass.end();
      waves.recordMips(encoder);
      encoder.submit();
      await root.device.queue.onSubmittedWorkDone();
      await traceHistory(195.4 + frame * 0.04, frame === 0 ? 0 : 0.04);
    }
    const resources: CapturedResource[] = [];
    for (const resource of observer.resources) {
      if (
        !resource.written ||
        resource.destroyed ||
        (resource.kind === 'buffer' &&
          !['amplitudes', 'omegas', 'foam', ...(retainLattice ? ['lattice'] : [])].includes(
            resource.label,
          ))
      )
        continue;
      const levels =
        resource.kind === 'buffer'
          ? [
              {
                width: 0,
                height: 0,
                depth: 0,
                bytes: await readBuffer(root.device, resource.handle as GPUBuffer),
              },
            ]
          : (observer.cpuTextures.get(resource.handle)?.levels.map((level) => ({
              width: level.width,
              height: level.height,
              depth: level.depthOrArrayLayers,
              bytes: new Uint8Array(
                level.data.buffer,
                level.data.byteOffset,
                level.data.byteLength,
              ).slice().buffer,
            })) ?? (await readTexture(root.device, resource.handle as GPUTexture)));
      resources.push({ ...resource, reasons: [], order: [], levels });
    }
    const capture: SceneCapture = { resources, skipped: [], frames: 2 };
    return { ...capture, historyTrace };
  } finally {
    waves.destroy();
  }
}

function transposeFoam(capture: SceneCapture): SceneCapture {
  return {
    ...capture,
    resources: capture.resources.map((resource) => {
      if (resource.label !== 'foam') return resource;
      const original = new Uint32Array(resource.levels[0]!.bytes),
        transposed = new Uint32Array(original.length);
      for (let c = 0; c < CASCADES; c++)
        for (let y = 0; y < N; y++)
          for (let x = 0; x < N; x++)
            transposed[(c * N + x) * N + y] = original[(c * N + y) * N + x]!;
      return { ...resource, levels: [{ ...resource.levels[0]!, bytes: transposed.buffer }] };
    }),
  };
}

function foamArithmetic(signals: number[], cell: number, previous: number, dt: number) {
  const f = Math.fround,
    sign = ((cell % N) + (Math.floor(cell / N) % N)) & 1 ? -1 : 1;
  const lambda = f(-1.3),
    xx = f(lambda * f(signals[5]! * sign)),
    zz = f(lambda * f(signals[6]! * sign)),
    xz = f(lambda * f(signals[7]! * sign));
  const left = f(1 + xx),
    right = f(1 + zz),
    product = f(left * right),
    cross = f(xz * xz),
    jacobian = f(product - cross);
  const rule = foamingAt(20)[1]!,
    exponent = f(-f(rule.decay) * f(dt)),
    decay = f(Math.exp(exponent));
  const calculate = (j: number, rate: number, fused = false, deficit = f(f(rule.whitecap) - j)) => {
    const grown = f(f(Math.max(deficit, 0) * f(rule.grow)) * f(dt));
    return Math.max(0, Math.min(1, f((fused ? previous * rate : f(previous * rate)) + grown)));
  };
  const jacobianUlp = Math.max(2 ** -149, 2 ** (Math.floor(Math.log2(Math.abs(jacobian))) - 23));
  const encoded = new Uint32Array(new Float32Array([decay]).buffer)[0]!;
  const candidates = [-1, 0, 1].flatMap((offset) =>
    [-3, 0, 3].flatMap((delta) => {
      const rate = new Float32Array(new Uint32Array([encoded + delta]).buffer)[0]!;
      const deficit = f(f(rule.whitecap) - (jacobian + offset * jacobianUlp));
      return [calculate(jacobian, rate, false, deficit), calculate(jacobian, rate, true, deficit)];
    }),
  );
  return {
    jacobian,
    jacobianUlp,
    product,
    cross,
    decay,
    canonical: calculate(jacobian, decay),
    minimum: Math.min(...candidates),
    maximum: Math.max(...candidates),
  };
}

it.runIf(process.env.CPU_OCEAN_WAVES === '1')(
  'isolates ocean FFT branches with equal spectra and frequencies',
  async () => {
    const dawn = await gpuRoot();
    const observer = observeScene();
    const gpu = tgpu.initFromDevice({ device: observer.device(dawn.device) });
    try {
      const expected = await wavesCapture(gpu, observer);
      const inputs = new Map(
        expected.resources
          .filter((item) => ['amplitudes', 'omegas'].includes(item.label))
          .map((item) => [item.label, item.levels[0]!.bytes]),
      );
      expect(inputs.size).toBe(2);
      const cpuObserver = observeScene();
      const snapshots = new Map<string, ReplaySnapshot>();
      const counts = new Map<string, number>();
      const cpu = cpuRoot({
        generator: SweepGenerator,
        observeDevice: cpuObserver.device,
        onTexture: (texture, data) => cpuObserver.cpuTextures.set(texture, data),
        beforeDispatch(code, label, dispatch) {
          if (process.env.CPU_OCEAN_STAGES !== '1') return;
          const entry = (JSON.parse(/^\/\/#entry (.*)$/m.exec(code)![1]!) as { name: string }).name;
          const count = counts.get(entry) ?? 0;
          counts.set(entry, count + 1);
          const limit =
            entry === 'rowStages' || entry === 'columnStages'
              ? 3
              : entry === 'fillOne'
                ? 27
                : ['evolveSignal', 'packDense'].includes(entry)
                  ? 1
                  : 0;
          if (count < limit)
            snapshots.set(
              `${entry}:${count}`,
              snapshotDispatch(code, label, dispatch, cpuObserver),
            );
        },
      });
      try {
        const actual = await wavesCapture(cpu, cpuObserver, inputs);
        const results = {
          raw: compareCaptures(expected, actual),
          transposed: compareCaptures(expected, transposeFoam(actual)),
        };
        recordResults(
          process.env.CPU_OCEAN_HISTORY === '1' ? 'oceanEqualSpectraHistory' : 'oceanEqualSpectra',
          results,
        );
        Object.defineProperty(dawn.device, backends.brand, { value: cpuBackend });
        const stagedObserver = observeScene();
        const staged = tgpu.initFromDevice({ device: stagedObserver.device(dawn.device) });
        try {
          const stagedValues = await wavesCapture(staged, stagedObserver, inputs, true);
          if (process.env.CPU_OCEAN_HISTORY_TRACE === '1') {
            const comparisons = actual.historyTrace.map((stage, index) => ({
              time: stage.time,
              rows: stage.rows.map((row, i) => {
                const gpuRow = stagedValues.historyTrace[index]!.rows[i]!;
                const cpuMath = foamArithmetic(
                  row.signals,
                  row.cell,
                  actual.historyTrace[index - 1]?.rows[i]?.foam ?? 0,
                  stage.dt,
                );
                const gpuMath = foamArithmetic(
                  gpuRow.signals,
                  gpuRow.cell,
                  stagedValues.historyTrace[index - 1]?.rows[i]?.foam ?? 0,
                  stage.dt,
                );
                const prior = stagedValues.historyTrace[index - 1]?.rows[i]?.foam ?? 0;
                const growth = gpuRow.foam - prior * gpuMath.decay;
                const inferredJacobian =
                  stage.dt > 0 &&
                  growth > 1e-8 &&
                  gpuRow.foam < 1 &&
                  gpuMath.jacobian < Math.fround(foamingAt(20)[1]!.whitecap)
                    ? Math.fround(foamingAt(20)[1]!.whitecap) - growth / (67.5 * stage.dt)
                    : undefined;
                return {
                  inferredJacobian,
                  inferredJacobianUlps:
                    inferredJacobian === undefined
                      ? undefined
                      : Math.abs(inferredJacobian - gpuMath.jacobian) / gpuMath.jacobianUlp,
                  cell: row.cell,
                  cpu: row.foam,
                  dawn: gpuRow.foam,
                  cpuMath,
                  gpuMath,
                  sourceGap: Math.max(
                    ...row.signals.map((v, c) => Math.abs(v - gpuRow.signals[c]!)),
                  ),
                  jacobianGap: Math.abs(cpuMath.jacobian - gpuMath.jacobian),
                };
              }),
            }));
            recordResults('oceanFoamHistoryStages', {
              cpu: actual.historyTrace,
              dawn: stagedValues.historyTrace,
              comparisons,
            });
            expect(comparisons).toHaveLength(26);
            for (const stage of comparisons)
              for (const row of stage.rows) {
                if (row.inferredJacobianUlps !== undefined)
                  expect(row.inferredJacobianUlps).toBeLessThanOrEqual(1);
                const allowance = Math.max(
                  2 ** -149,
                  2 ** (Math.floor(Math.log2(Math.abs(row.dawn))) - 23),
                );
                expect(row.cpuMath.canonical, `${stage.time} cell ${row.cell}: CPU formula`).toBe(
                  row.cpu,
                );
                expect(
                  row.dawn,
                  `${stage.time} cell ${row.cell}: effective Jacobian lower bound`,
                ).toBeGreaterThanOrEqual(row.gpuMath.minimum - allowance);
                expect(
                  row.dawn,
                  `${stage.time} cell ${row.cell}: effective Jacobian upper bound`,
                ).toBeLessThanOrEqual(row.gpuMath.maximum + allowance);
              }
          }
          const stageResults = [];
          for (const [entry, snapshot] of snapshots) {
            const result = await replay(dawn.device, stagedObserver.dispatches, snapshot);
            stageResults.push({ entry, ...result });
            expect(result.differences, entry).toEqual([]);
            recordResults('oceanStageReplays', stageResults);
          }
          if (process.env.CPU_OCEAN_STAGES === '1') expect(stageResults).toHaveLength(35);
          recordResults(
            process.env.CPU_OCEAN_HISTORY === '1'
              ? 'oceanEqualSpectraHistory'
              : 'oceanEqualSpectra',
            {
              ...results,
              sameBranch: compareCaptures(stagedValues, actual),
            },
          );
          if (process.env.CPU_OCEAN_STORE_CONTROLS === '1') {
            const controls = [];
            for (const control of ['lattice', 'base'] as const) {
              const controlObserver = observeScene();
              let packCount = 0;
              const controlled = cpuRoot({
                generator: SweepGenerator,
                observeDevice: controlObserver.device,
                onTexture: (texture, data) => controlObserver.cpuTextures.set(texture, data),
                beforeDispatch(code, _label, dispatch) {
                  const entry = (
                    JSON.parse(/^\/\/#entry (.*)$/m.exec(code)![1]!) as { name: string }
                  ).name;
                  if (entry === 'packDense') packCount++;
                  if (packCount !== 2) return;
                  for (const use of dispatch.module.bindings) {
                    const item = Array.from(dispatch.groups.get(use.group)!.entries).find(
                      (binding) => binding.binding === use.binding,
                    )!;
                    const resource = controlObserver.resourceOf(
                      (item.resource as GPUBufferBinding).buffer ?? item.resource,
                    );
                    if (!resource) continue;
                    const expectedResource = stagedValues.resources.find(
                      (value) => value.key === resource.key,
                    );
                    if (!expectedResource) continue;
                    if (entry === 'packDense' && resource.label === 'lattice') {
                      const into = dispatch.resources.buffer(use.group, use.binding);
                      new Uint8Array(into.bytes, into.offset, into.size).set(
                        new Uint8Array(expectedResource.levels[0]!.bytes),
                      );
                    }
                    if (control === 'base' && entry === 'fillOne' && use.kind === 'texture') {
                      const level = dispatch.resources.texture(use.group, use.binding).levels[0]!;
                      if (level.width === N)
                        new Uint8Array(
                          level.data.buffer,
                          level.data.byteOffset,
                          level.data.byteLength,
                        ).set(new Uint8Array(expectedResource.levels[0]!.bytes));
                    }
                  }
                },
              });
              try {
                const value = await wavesCapture(controlled, controlObserver, inputs);
                const compared = compareCaptures(stagedValues, value);
                const baseOnly = compareCaptures(
                  {
                    ...stagedValues,
                    resources: stagedValues.resources.map((resource) => ({
                      ...resource,
                      levels: resource.levels.slice(0, 1),
                    })),
                  },
                  {
                    ...value,
                    resources: value.resources.map((resource) => ({
                      ...resource,
                      levels: resource.levels.slice(0, 1),
                    })),
                  },
                );
                const packOutliers = [];
                if (control === 'lattice') {
                  const lattice = new Float32Array(
                    stagedValues.resources.find((resource) => resource.label === 'lattice')!
                      .levels[0]!.bytes,
                  );
                  for (const resource of value.resources.filter(
                    (item) => item.kind === 'texture',
                  )) {
                    const expectedResource = stagedValues.resources.find(
                      (item) => item.key === resource.key,
                    )!;
                    const a = new Uint16Array(resource.levels[0]!.bytes),
                      b = new Uint16Array(expectedResource.levels[0]!.bytes);
                    for (let word = 0; word < a.length; word++) {
                      const gap = Math.abs(ordered(a[word]!) - ordered(b[word]!));
                      if (gap <= 1) continue;
                      const texel = Math.floor(word / 4),
                        x = texel % N,
                        y = Math.floor(texel / N) % N,
                        c = Math.floor(texel / (N * N));
                      const sign = (x + y) & 1 ? -1 : 1,
                        at = (c * 4 * N + y) * N + x;
                      const signals = Array.from({ length: 4 }, (_, signal) => [
                        lattice[(at + signal * N * N) * 2]! * sign,
                        lattice[(at + signal * N * N) * 2 + 1]! * sign,
                      ]);
                      const f = Math.fround,
                        lambda = f(f(-1.3) * f(cascadeValues[c]!.chop));
                      const xx = f(lambda * signals[2]![1]!),
                        zz = f(lambda * signals[3]![0]!),
                        xz = f(lambda * signals[3]![1]!);
                      const left = f(1 + xx),
                        right = f(1 + zz),
                        product = f(left * right),
                        cross = f(xz * xz);
                      const jacobians = [
                        f(product - cross),
                        f(left * right - cross),
                        f(product - xz * xz),
                      ];
                      const rule = foamingAt(20)[c]!;
                      const foam = jacobians.map((jacobian) =>
                        f(f(Math.max(f(f(rule.whitecap) - jacobian), 0) * f(rule.grow)) * f(0.04)),
                      );
                      packOutliers.push({
                        resource: resource.key,
                        word,
                        cell: [x, y, c],
                        component: word % 4,
                        gap,
                        cpuHalf: a[word],
                        dawnHalf: b[word],
                        signals,
                        deformation: [xx, zz, xz],
                        jacobians,
                        foam,
                        halfCandidates: foam.map(encodeHalf),
                      });
                    }
                  }
                }
                controls.push({ control, baseOnly, packOutliers, ...compared });
                recordResults('oceanStoreControls', controls);
                expect(baseOnly.differences).toEqual([]);
                for (const outlier of packOutliers) {
                  expect(outlier.component).toBe(3);
                  expect(outlier.dawnHalf!).toBeGreaterThanOrEqual(
                    Math.min(...outlier.halfCandidates) - 1,
                  );
                  expect(outlier.dawnHalf!).toBeLessThanOrEqual(
                    Math.max(...outlier.halfCandidates) + 1,
                  );
                }
                if (control === 'base') expect(compared.differences).toEqual([]);
              } finally {
                controlled.destroy();
              }
            }
          }
        } finally {
          staged.destroy();
        }
      } finally {
        cpu.destroy();
      }
    } finally {
      gpu.destroy();
      dawn.destroy();
    }
  },
  600000,
);
