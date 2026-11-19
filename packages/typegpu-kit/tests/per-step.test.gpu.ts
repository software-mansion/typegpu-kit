import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, tgpu } from 'typegpu';
import { afterEach, describe, expect, it } from 'vitest';
import { frames, type Frame } from '../src/index.ts';

// One frame of four fixed steps, each of which should see its own phase: written between steps
// recorded into one encoder, the steps all see the last; submitted after each step, each its own.
const phases = [10, 11, 12, 13];
const Params = d.struct({ phase: d.f32 });
const Log = d.struct({ count: d.u32, phases: d.arrayOf(d.f32, 8) });

let cleanup: (() => void) | undefined;

afterEach(() => {
  cleanup?.();
  cleanup = undefined;
});

async function setup() {
  const root = await gpuRoot();
  cleanup = () => root.destroy();
  const log = root.createMutable(Log);

  /** Appends the phase a step saw to the log. */
  const record = (phase: number) => {
    'use gpu';
    log.$.phases[log.$.count] = phase;
    log.$.count += 1;
  };

  const frame = (recorder: (frame: Frame) => void) => {
    const loop = frames.create(root, recorder);
    loop.run(0);
    loop.destroy();
  };

  const seen = async () => {
    const { count, phases: logged } = await log.read();
    return logged.slice(0, count);
  };

  return { root, record, frame, seen };
}

describe('a value per step', () => {
  it('is lost when a uniform is written between the steps', async () => {
    const { root, record, frame, seen } = await setup();
    const params = root.createUniform(Params);
    const step = root.createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [1] })(() => {
        'use gpu';
        record(params.$.phase);
      }),
    });

    frame(({ encoder }) => {
      const pass = encoder.beginComputePass();
      pass.setPipeline(step);
      for (const phase of phases) {
        params.write({ phase });
        pass.dispatchWorkgroups(1);
      }
      pass.end();
    });

    // Queue writes land before the frame's commands, so every step reads the last one.
    expect(await seen()).toEqual([13, 13, 13, 13]);
  });

  it('is written before each step’s own submit', async () => {
    const { root, record, frame, seen } = await setup();
    const params = root.createUniform(Params);
    const step = root.createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [1] })(() => {
        'use gpu';
        record(params.$.phase);
      }),
    });

    frame((current) => {
      for (const phase of phases) {
        params.write({ phase });
        const pass = current.encoder.beginComputePass();
        pass.setPipeline(step);
        pass.dispatchWorkgroups(1);
        pass.end();

        // Queues this step ahead of the next write, at one queue.submit per step.
        current.submit();
      }
    });

    expect(await seen()).toEqual(phases);
  });
});
