import { mkdirSync, writeFileSync } from 'node:fs';
import { d, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';
import { SweepGenerator } from './scene-generator.ts';
import { observeScene } from './scene-observer.ts';
import { captureScene } from './example-comparison.ts';
import { snapshotDispatch, type ReplaySnapshot } from './replay.ts';

it('snapshots bound bytes before a dispatch and tracks its corresponding pipeline', async () => {
  const observer = observeScene();
  let snapshot: ReplaySnapshot | undefined;
  const root = cpuRoot({
    generator: SweepGenerator,
    observeDevice: observer.device,
    beforeDispatch: (code, label, dispatch) => {
      snapshot = snapshotDispatch(code, label, dispatch, observer);
    },
  });
  try {
    const output = root.createMutable(d.u32, 7);
    const pipeline = root.createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [1] })(() => {
        'use gpu';
        output.$ += 1;
      }),
    });
    pipeline.dispatchWorkgroups(1);
    expect(await output.read()).toBe(8);
    expect(new Uint32Array(snapshot!.bindings[0]!.buffer!.bytes)[0]).toBe(7);
    output.write(99);
    expect(new Uint32Array(snapshot!.bindings[0]!.buffer!.bytes)[0]).toBe(7);
    expect(observer.dispatches).toHaveLength(1);
    expect(observer.dispatches[0]!.bindings[0]!.resource.key).toBe(snapshot!.bindings[0]!.key);
  } finally {
    root.destroy();
  }
});

it.runIf(process.env.CPU_CAPTURE === '1')(
  'captures inputs for a scene numeric diagnostic',
  async () => {
    const observer = observeScene();
    const example = process.env.CPU_EXAMPLES ?? 'jelly-world';
    const label = new RegExp(process.env.CPU_REPLAY_PIPELINE ?? 'fill');
    const root = cpuRoot({
      generator: SweepGenerator,
      observeDevice: observer.device,
      beforeDispatch: (code, pipeline, dispatch) => {
        if (!label.test(pipeline)) return;
        const snapshot = snapshotDispatch(code, pipeline, dispatch, observer);
        if (
          process.env.CPU_REPLAY_RESOURCE &&
          !snapshot.bindings.some(
            (binding) =>
              binding.use.writes && new RegExp(process.env.CPU_REPLAY_RESOURCE!).test(binding.key),
          )
        )
          return;
        const directory = process.env.CPU_CAPTURE_DIRECTORY!;
        mkdirSync(directory, { recursive: true });
        writeFileSync(`${directory}/${example}.js`, code);
        writeFileSync(
          `${directory}/${example}.json`,
          JSON.stringify(
            snapshot.bindings.map((binding) => ({
              key: binding.key,
              use: binding.use,
              ...(binding.buffer
                ? {
                    offset: binding.buffer.offset,
                    size: binding.buffer.size,
                    words: Array.from(new Uint32Array(binding.buffer.bytes)),
                  }
                : {}),
            })),
          ),
        );
        throw new Error('CPU inputs saved');
      },
    });
    try {
      await expect(captureScene(root, observer, example, false)).rejects.toThrow(
        'CPU inputs saved',
      );
    } finally {
      root.destroy();
    }
  },
  600000,
);
