import type { Frame, Surface } from '@typegpu/kit';
import { d, type TgpuRoot } from 'typegpu';
import type { Settings } from '../settings.ts';
import { createJumpFlood } from './jump-flood.ts';
import { createMap } from './map.ts';

/** Cells along each side of the grid. */
const SIZE = 512;
const SEEDS = 48;

export interface SceneOptions {
  /** Uses immediates for dispatch values. Defaults to device support. */
  readonly immediates?: boolean;
}

/**
 * A territory map whose regions are the Voronoi cells of drifting towns, rebuilt each frame by
 * jump flooding in one compute pass.
 */
export async function createScene(root: TgpuRoot, surface: Surface, options: SceneOptions = {}) {
  const time = root.createUniform(d.f32);
  const map = createMap(root, { size: SIZE, seeds: SEEDS });

  const build = (immediates?: boolean) =>
    createJumpFlood(root, {
      size: SIZE,
      seeds: SEEDS,
      format: surface.format,
      time,
      target: surface,
      paint: map.paint,
      immediates,
    });

  let open = true;
  let flood = build(options.immediates);
  await flood.initAsync();
  const values = { passes: flood.passes, immediates: flood.immediates };
  const supported = root.enabledWgslLanguageFeatures.has('immediate_address_space');

  const settings: Settings<typeof values> = {
    values,
    knobs: {
      passes: { label: 'flood dispatches', min: 0, max: flood.passes, step: 1 },
      immediates: supported ? {} : { disabled: 'unsupported' },
    },

    /** Changing `immediates` rebuilds the flood. Frames read `passes` directly. */
    async write(key) {
      if (key === 'passes' || values.immediates === flood.immediates) return;

      const next = build(values.immediates);
      await next.initAsync();
      // A later write may have switched it back while this one compiled
      if (!open || next.immediates !== values.immediates) return next.destroy();
      flood.destroy();
      flood = next;
    },
  };

  return {
    settings,

    frame(frame: Frame) {
      time.write(frame.time);

      const compute = frame.encoder.beginComputePass();
      flood.flood(compute, values.passes);
      compute.end();

      const output = frame.encoder.beginRenderPass({
        colorAttachments: { view: surface.view(frame) },
      });
      flood.draw(output);
      output.end();
    },

    destroy() {
      open = false;
      for (const made of [flood, map]) made.destroy();
    },
  };
}
