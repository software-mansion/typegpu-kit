// The docs site's box on the UI runtime: resting on the gallery's banner, where a finger presses it,
// and rising out of its spot over the dimmed gallery to open, as `Pull.tsx` poses it. Both draw the
// site's shader at full resolution, a cell every three points.

import type { Frame, Surface } from '@typegpu/kit';
import type { Scene } from '@typegpu/kit/native/host';
import type { TgpuRoot } from 'typegpu';
import { restingBox, restOf, type BoxInput } from 'typegpu-kit-pull/box.ts';
import { createBox } from 'typegpu-kit-pull/box.scene.ts';
import { CELL, COLORS } from 'typegpu-kit-pull/dither.ts';

/** Seconds the press and its light take to close most of the way to where they are going. */
const EASE = 0.06;
/** The longest step in seconds an ease takes, so a stalled frame doesn't jump it. */
const MAX_STEP = 0.1;

/** Draws the box on `surface`: clears it each frame and, given an input, draws the box posed by it. */
function boxOn(root: TgpuRoot, surface: Surface) {
  const box = createBox(root, surface.format, COLORS);
  return (frame: Frame, input: BoxInput | null) => {
    if (input) box.write(input, frame.time, CELL * surface.pixelRatio);
    const pass = frame.encoder.beginRenderPass({
      colorAttachments: { view: surface.view(frame), loadOp: 'clear', clearValue: [0, 0, 0, 0] },
    });
    if (input) {
      pass.setPipeline(box.pipeline);
      pass.draw(3);
    }
    pass.end();
  };
}

/** Shared values from Reanimated, which the scenes read and write on the UI runtime. */
export interface RestingOptions {
  /** Whether a finger is down on the box, which pushes it down. */
  readonly pressed: { readonly value: boolean };
  /** How far a finger has lit the box, 0 to 1, which the scene keeps for the pull to start from. */
  readonly over: { value: number };
}

/** The box resting in its canvas, lit and pushed down by a finger; host it through `worklets.ts`. */
export function resting(root: TgpuRoot, surface: Surface, options: RestingOptions): Scene {
  const draw = boxOn(root, surface);
  let press = 0;
  let posed = { press: -1, width: 0, height: 0 };
  let input: BoxInput | null = null;

  return {
    frame(frame) {
      const width = surface.width / surface.pixelRatio;
      const height = surface.height / surface.pixelRatio;
      const aim = options.pressed.value ? 1 : 0;
      press += (aim - press) * (1 - Math.exp(-Math.min(frame.dt, MAX_STEP) / EASE));
      if (Math.abs(press - aim) < 1e-3) press = aim;
      // Posed again only when it moves, so a box at rest allocates nothing
      if (press !== posed.press || width !== posed.width || height !== posed.height) {
        const { anchor, unit } = restOf(width, height);
        input = restingBox(anchor, unit, press, press, COLORS.accent[0]);
        posed = { press, width, height };
        options.over.value = press;
      }
      draw(frame, input);
    },

    destroy() {},
  };
}

export interface PulledOptions {
  /** The box at this moment of the open pull, or null while none shows it. */
  readonly box: { readonly value: BoxInput | null };
}

/** The box of the open pull over the gallery; host it through `worklets.ts`. */
export function pulled(root: TgpuRoot, surface: Surface, options: PulledOptions): Scene {
  const draw = boxOn(root, surface);
  return {
    frame: (frame) => draw(frame, options.box.value),
    destroy() {},
  };
}
