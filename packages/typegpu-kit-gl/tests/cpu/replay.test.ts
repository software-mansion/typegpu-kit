import { d, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';
import { SweepGenerator } from './scene-generator.ts';
import { observeScene } from './scene-observer.ts';
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
