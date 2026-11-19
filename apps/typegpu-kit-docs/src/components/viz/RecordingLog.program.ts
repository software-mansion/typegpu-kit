import { frames } from '@typegpu/kit';
import { offscreenSurface, recordingRoot } from '@typegpu/kit/testing';
import { createFlowField } from 'typegpu-kit-examples/flow-field/flow-field.ts';

export const COUNTERS = [
  'shaders',
  'pipelines',
  'buffers',
  'views',
  'bindGroups',
  'writes',
  'submissions',
] as const;
export type Counter = (typeof COUNTERS)[number];

export function createProgram() {
  const { root, log } = recordingRoot();
  const surface = offscreenSurface(root, { width: 640, height: 360, format: 'bgra8unorm' });
  const field = createFlowField(root, {
    count: 4096,
    step: 1 / 60,
    format: surface.format,
    size: surface,
  });
  let steps = 0;
  let time = 0;

  const loop = frames.create(root, (frame) => {
    const compute = frame.encoder.beginComputePass();
    for (let i = 0; i < steps; i++) field.step(compute);
    compute.end();

    const pass = frame.encoder.beginRenderPass({
      colorAttachments: { view: surface.view(frame) },
    });
    field.draw(pass);
    pass.end();
  });

  return {
    log,

    /** The log's count of `name`. */
    count: (name: Counter) =>
      name === 'shaders' || name === 'writes' ? log[name].length : log[name],

    initAsync: () => field.initAsync(),

    /** Records a frame that steps the field `count` times and draws it. */
    frame(count: number) {
      steps = count;
      loop.run((time += 1 / 60));
    },

    destroy() {
      loop.destroy();
      field.destroy();
    },
  };
}
