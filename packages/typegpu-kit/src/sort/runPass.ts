import type { TgpuComputePass, TgpuComputePipeline } from 'typegpu';
import type { RunOptions } from './types.ts';

export type RunPass = GPUComputePassEncoder | TgpuComputePass;

export interface RunRecording {
  pass: RunPass;
  finish(): void;
}

export interface DispatchStep {
  pipeline: TgpuComputePipeline;
  workgroups: [number, number, number];
}

/** Sets the pipeline on a TypeGPU pass, which binds nothing again for the same pipeline. */
export function dispatch(
  pipeline: TgpuComputePipeline,
  pass: RunPass,
  workgroups: readonly [number, number, number],
): void {
  if ('resourceType' in pass && pass.resourceType === 'compute-pass') {
    pass.setPipeline(pipeline);
    pass.dispatchWorkgroups(...workgroups);
  } else {
    pipeline.with(pass as TgpuComputePass).dispatchWorkgroups(...workgroups);
  }
}

const noop = () => {};

export function beginRunPass(device: GPUDevice, options?: RunOptions): RunRecording {
  if (options?.pass) {
    return { pass: options.pass, finish: noop };
  }

  const externalEncoder = options?.encoder;
  if (externalEncoder) {
    const pass = externalEncoder.beginComputePass();
    return {
      pass,
      finish() {
        pass.end();
      },
    };
  }

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  return {
    pass,
    finish() {
      pass.end();
      device.queue.submit([encoder.finish()]);
    },
  };
}

export function stepRunner(device: GPUDevice, steps: DispatchStep[]) {
  return {
    initSync(): void {
      for (const step of steps) {
        step.pipeline.initSync();
      }
    },

    async initAsync(): Promise<void> {
      await Promise.all(steps.map((step) => step.pipeline.initAsync()));
    },

    run(options?: RunOptions): void {
      const recording = beginRunPass(device, options);
      for (const step of steps) {
        dispatch(step.pipeline, recording.pass, step.workgroups);
      }
      recording.finish();
    },
  };
}
