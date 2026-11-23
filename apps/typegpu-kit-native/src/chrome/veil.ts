// The veil over a loading example: the poster prints in as dithered cells, a light crosses it
// while the example compiles, and once it is ready a front of accent cells lifts the veil off the
// scene

import type { Frame, Surface } from '@typegpu/kit';
import { perlin3d } from '@typegpu/noise';
import { common, d, std, tgpu, type TgpuRoot } from 'typegpu';
import { bayerCell, CELL } from 'typegpu-kit-pull/dither.ts';
import { INK, printed } from './dither.ts';

/** Seconds the print takes to come in over the plain poster. */
const INTRO = 0.5;
/** Seconds between the light's passes. */
const PASS = 2.4;
/** Seconds the front takes to cross when the veil lifts. */
export const LIFT = 0.7;
/** How bright the print is at rest, so the light reads over it. */
const DIM = 0.55;
/** How far the print leans to the accent, by its brightness. */
const TINT = 0.3;
/** Steps each colour channel prints in. */
const LEVELS = 5;

const State = d.struct({
  /** The drawing buffer in pixels, then pixels per cell and the poster's width over its height. */
  size: d.vec4f,
  /** Seconds since the veil started, and since it began to lift, negative before. */
  time: d.vec2f,
});

const layout = tgpu.bindGroupLayout({
  state: { uniform: State },
  poster: { texture: d.texture2d(d.f32) },
  smooth: { sampler: 'filtering' },
});

const easeOut = (k: number) => {
  'use gpu';
  const rest = 1 - std.saturate(k);
  return 1 - rest * rest * rest;
};

const fragment = tgpu.fragmentFn({
  in: { position: d.builtin.position },
  out: d.vec4f,
})(({ position }) => {
  'use gpu';
  const state = layout.$.state;
  const size = state.size.xy;
  const cellSize = state.size.z;
  const cell = d.vec2u(position.xy / cellSize);
  const threshold = bayerCell(cell);
  const uv = ((d.vec2f(cell) + 0.5) * cellSize) / size;

  // The poster covers the screen, as the gallery's card shows it
  const screenAspect = size.x / size.y;
  const posterAspect = state.size.w;
  let fit = d.vec2f(1, posterAspect / screenAspect);
  if (posterAspect > screenAspect) fit = d.vec2f(screenAspect / posterAspect, 1);
  const at = (uv - 0.5) * fit + 0.5;
  const source = std.textureSampleLevel(layout.$.poster, layout.$.smooth, at, 0).xyz;

  // A light crosses on the diagonal, its edge broken up by drifting smoke
  const t = state.time.x;
  const along = uv.x * screenAspect * 0.6 + uv.y * 0.8;
  const reach = screenAspect * 0.6 + 0.8;
  const light = -0.4 + (reach + 0.8) * std.fract(t / PASS);
  const wisp = perlin3d.sample(d.vec3f(uv * d.vec2f(screenAspect, 1) * 5, t * 0.35));
  const across = (along - light) * 3.2 + wisp * 0.9;
  const glow = std.exp(-across * across);

  const brightness = std.dot(source, d.vec3f(0.2126, 0.7152, 0.0722));
  const tinted = std.mix(source, std.mix(INK.ground3, INK.accent, brightness), TINT);
  const lit = tinted * (DIM + 0.55 * glow) + INK.accent * (glow * 0.16);
  const ink = d.vec3f(
    printed(lit.x, LEVELS, threshold),
    printed(lit.y, LEVELS, threshold),
    printed(lit.z, LEVELS, threshold),
  );
  let color = std.max(ink, INK.ground);
  // The brightest cells in the light catch the accent
  if (glow * (0.6 + 0.4 * brightness) > 0.55 + threshold * 0.45) color = INK.accentHigh;

  // The print comes in cell by cell over the plain poster below the canvas
  let alpha = std.step(threshold, easeOut(t / INTRO));

  // Lifting, a front crosses on the diagonal: accent cells, then the highlight, the scene behind it
  const since = state.time.y;
  if (since >= 0) {
    const front = -0.25 + 1.6 * easeOut(since / LIFT);
    const swept = (front - (0.75 * uv.x + 0.25 * uv.y)) * 9 + (threshold - 0.5) * 1.6;
    if (swept > 0) color = INK.accent;
    if (swept > 1) color = INK.accentHigh;
    if (swept > 2) alpha = 0;
  }

  return d.vec4f(color * alpha, alpha);
});

export interface VeilOptions {
  /** The example's poster, or none for a plain print. */
  readonly poster: ImageBitmap | undefined;
}

export interface Veil {
  frame(frame: Frame): void;
  /** Starts the front that lifts the veil off the scene. */
  lift(): void;
  destroy(): void;
}

/** Creates the veil scene; host it through `chrome/worklets.ts`. */
export function create(root: TgpuRoot, surface: Surface, options: VeilOptions): Veil {
  const { poster } = options;
  const picture = root
    .createTexture({ size: [poster?.width ?? 1, poster?.height ?? 1], format: 'rgba8unorm' })
    .$usage('sampled', 'render');
  if (poster) picture.write(poster);
  else picture.write(new Uint8Array([20, 24, 31, 255]));

  const state = root.createUniform(State);
  const group = root.createBindGroup(layout, {
    state,
    poster: picture,
    smooth: root.createSampler({ magFilter: 'linear', minFilter: 'linear' }),
  });
  const pipeline = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment,
    targets: { format: surface.format },
  });
  const drawing = pipeline.with(group);
  const aspect = (poster?.width ?? 16) / (poster?.height ?? 10);
  const values: { size: [number, number, number, number]; time: [number, number] } = {
    size: [0, 0, 0, aspect],
    time: [0, -1],
  };
  let lifting: number | undefined;
  let now = 0;

  return {
    frame(frame) {
      now = frame.time;
      values.size[0] = surface.width;
      values.size[1] = surface.height;
      values.size[2] = CELL * surface.pixelRatio;
      values.time[0] = now;
      values.time[1] = lifting === undefined ? -1 : now - lifting;
      state.write(values);
      const pass = frame.encoder.beginRenderPass({
        colorAttachments: { view: surface.view(frame), loadOp: 'clear', clearValue: [0, 0, 0, 0] },
      });
      pass.setPipeline(drawing);
      pass.draw(3);
      pass.end();
    },

    lift() {
      lifting ??= now;
    },

    destroy() {
      picture.destroy();
      state.buffer.destroy();
    },
  };
}
