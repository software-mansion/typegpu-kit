// The gallery's banner: smoke drifting through the theme's ramp in dithered cells, lit by a light
// that wanders on its own and follows a finger on the banner

import { clock, type Frame, type Surface } from '@typegpu/kit';
import { perlin3d } from '@typegpu/noise';
import { common, d, std, tgpu, type TgpuRoot } from 'typegpu';
import { bayerCell, CELL } from 'typegpu-kit-pull/dither.ts';
import { ramp } from './dither.ts';

/** Seconds the light takes to close most of the way to where it is going. */
const FOLLOW = 0.18;
/** Seconds a finger's light takes to fade in or out. */
const HOLD = 0.35;
/** The light's reach, as a share of the banner's height. */
const REACH = 0.42;
/** The longest step in seconds the light takes, so a stalled frame doesn't jump it. */
const MAX_STEP = 0.1;

const State = d.struct({
  /** The drawing buffer in pixels, pixels per cell and seconds. */
  size: d.vec4f,
  /** The light's centre in pixels and how bright it is, the finger's share of it last. */
  light: d.vec4f,
});

const layout = tgpu.bindGroupLayout({ state: { uniform: State } });

/** Warped Perlin noise from 0 to 1 for drifting smoke, `p` in banner heights. */
const smoke = (p: d.v2f, t: number) => {
  'use gpu';
  const q = (p - d.vec2f(t * 0.05, t * 0.012)) * 2.2;
  const warp = perlin3d.sample(d.vec3f(q * 0.7, t * 0.09));
  const body = perlin3d.sample(d.vec3f(q + d.vec2f(warp, -warp) * 0.9, t * 0.14));
  const wisps = perlin3d.sample(d.vec3f(q * 2.3, t * 0.21 + 7));
  return std.smoothstep(0.25, 0.9, 0.5 + 0.6 * body + 0.3 * wisps);
};

const fragment = tgpu.fragmentFn({
  in: { position: d.builtin.position },
  out: d.vec4f,
})(({ position }) => {
  'use gpu';
  const state = layout.$.state;
  const size = state.size.xy;
  const cellSize = state.size.z;
  const t = state.size.w;
  const cell = d.vec2u(position.xy / cellSize);
  const threshold = bayerCell(cell);
  const centre = (d.vec2f(cell) + 0.5) * cellSize;
  const p = centre / size.y;

  const fog = smoke(p, t);
  const away = std.length(centre - state.light.xy) / (size.y * REACH);
  const light = state.light.z * std.exp(-away * away);
  // The light shows in the smoke it passes through, a little more under a finger
  const lit = fog * (0.45 + 1.1 * light) + light * (0.12 + 0.18 * state.light.w);

  // Cells thin out toward the bottom so the banner meets the page
  const v = centre.y / size.y;
  const value = lit * (1 - std.smoothstep(0.45, 1, v));
  return d.vec4f(ramp(value, threshold), 1);
});

export interface Banner {
  frame(frame: Frame): void;
  /** Moves the light toward a finger at `x`, `y` points, or lets it wander again without one. */
  touch(x: number, y: number): void;
  release(): void;
  destroy(): void;
}

/** Creates the banner scene; host it through `chrome/worklets.ts`. */
export function create(root: TgpuRoot, surface: Surface): Banner {
  const state = root.createUniform(State);
  const group = root.createBindGroup(layout, { state });
  const pipeline = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment,
    targets: { format: surface.format },
  });
  const drawing = pipeline.with(group);
  const values: {
    size: [number, number, number, number];
    light: [number, number, number, number];
  } = { size: [0, 0, 0, 0], light: [0, 0, 0, 0] };
  const at = { x: 0, y: 0, held: 0 };
  let finger: { x: number; y: number } | undefined;

  return {
    frame(frame) {
      const t = frame.time;
      const step = Math.min(frame.dt, MAX_STEP);
      const width = surface.width;
      const height = surface.height;

      // On its own the light drifts slowly across the banner's upper half
      const goalX = finger ? finger.x / width : 0.5 + 0.38 * Math.sin(t * 0.21);
      const goalY = finger ? finger.y / height : 0.32 + 0.12 * Math.sin(t * 0.33 + 1.3);
      const follow = frame.index === 0 ? 1 : clock.ease(step, 1 / FOLLOW);
      at.x += (goalX - at.x) * follow;
      at.y += (goalY - at.y) * follow;
      at.held += ((finger ? 1 : 0) - at.held) * clock.ease(step, 1 / HOLD);

      values.size[0] = width;
      values.size[1] = height;
      values.size[2] = CELL * surface.pixelRatio;
      values.size[3] = t;
      values.light[0] = at.x * width;
      values.light[1] = at.y * height;
      values.light[2] = 0.7 + 0.3 * at.held;
      values.light[3] = at.held;
      state.write(values);

      const pass = frame.encoder.beginRenderPass({
        colorAttachments: { view: surface.view(frame), loadOp: 'clear', clearValue: [0, 0, 0, 1] },
      });
      pass.setPipeline(drawing);
      pass.draw(3);
      pass.end();
    },

    touch(x, y) {
      // Points to pixels
      finger = { x: x * surface.pixelRatio, y: y * surface.pixelRatio };
    },

    release() {
      finger = undefined;
    },

    destroy() {
      state.buffer.destroy();
    },
  };
}
