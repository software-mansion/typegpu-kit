import { frames } from '@typegpu/kit';
import { offscreenSurface, recordingRoot } from '@typegpu/kit/testing';
import { createFlowField } from 'typegpu-kit-examples/flow-field/flow-field.ts';
import { plural } from './figure.ts';

export interface Life {
  readonly label: string;
  /** Not the module's: the surface's texture, made before it and kept after. */
  readonly borrowed: boolean;
  /** Index of the step that created it; -1 before the first. */
  readonly from: number;
  /** Index of the step that destroyed it, or undefined. */
  readonly to: number | undefined;
}

export const STEPS = ['create', 'initAsync', 'write', 'frame 1', 'frame 2', 'destroy'] as const;

export async function createProgram(): Promise<Life[]> {
  const { root, log } = recordingRoot();
  const device = root.device;
  let step = -1;

  interface Made {
    label: string;
    borrowed: boolean;
    from: number;
    to?: number;
    object?: { destroyed: boolean };
  }
  const made: Made[] = [];

  const createBuffer = device.createBuffer.bind(device);
  device.createBuffer = (descriptor) => {
    const buffer = createBuffer(descriptor);
    made.push({
      label: descriptor.label ?? '',
      borrowed: false,
      from: step,
      object: buffer as never,
    });
    return buffer;
  };

  const surface = offscreenSurface(root, { width: 64, height: 36, format: 'bgra8unorm' });
  made.push({ label: 'surface texture', borrowed: true, from: -1 });

  // Counted objects that WebGPU has no destroy for: they go when nothing holds them
  const counted = (what: 'pipelines' | 'bindGroups', label: string) => {
    let before = log[what];
    return () => {
      const added = log[what] - before;
      before = log[what];
      if (added > 0)
        made.push({
          label: plural(added, label),
          borrowed: false,
          from: step,
        });
    };
  };
  const pipelines = counted('pipelines', 'pipeline');
  const bindGroups = counted('bindGroups', 'bind group');

  const steps: (() => unknown)[] = [];
  const field = createFlowField(root, {
    count: 4096,
    step: 1 / 60,
    format: surface.format,
    size: surface,
  });
  let seeding = true;
  const loop = frames.create(root, (frame) => {
    const compute = frame.encoder.beginComputePass();
    if (seeding) field.seed(compute);
    seeding = false;
    field.step(compute);
    compute.end();

    const pass = frame.encoder.beginRenderPass({ colorAttachments: { view: surface.view(frame) } });
    field.draw(pass);
    pass.end();
  });

  steps.push(
    () => field.initAsync(),
    () => field.settings.write({ speed: 0.2, scale: 1.5, pointSize: 2 }),
    () => loop.run(0),
    () => loop.run(1 / 60),
    () => field.destroy(),
  );

  // The factory ran above, before any step: what it made counts as step 0
  step = 0;
  for (const entry of made) if (!entry.borrowed) entry.from = 0;
  pipelines();
  bindGroups();

  for (const [i, run] of steps.entries()) {
    step = i + 1;
    await run();
    pipelines();
    bindGroups();
    for (const entry of made) {
      if (entry.to === undefined && entry.object?.destroyed) entry.to = step;
    }
  }

  return made.map(({ label, borrowed, from, to }) => ({ label, borrowed, from, to }));
}
