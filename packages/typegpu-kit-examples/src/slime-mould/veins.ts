import { pipelines } from '@typegpu/kit';
import {
  common,
  d,
  std,
  tgpu,
  type TgpuRenderCommands,
  type TgpuRoot,
  type TgpuUniform,
} from 'typegpu';
import { hex, sheetAt, World } from './agar.ts';
import type { Oats } from './oats.ts';
import type { Trail } from './trail.ts';

/** Colour ramp for trail density, from faint threads to dense hubs. */
const CRIMSON = hex(0x3a0610);
const RED = hex(0xb0201a);
const ORANGE = hex(0xff7a2a);
const GOLD = hex(0xffe0a0);
const FOOD = hex(0xffe8b8);
const HALO = hex(0xc8803a);
/**
 * Trail density at the top of the colour ramp. The tone curve compresses the density range while
 * preserving fine threads and bright hubs.
 */
export const FULL = 18;
/** Gain for faint trails in the tone curve. */
export const GAIN = 0.6;
/** Pulse period in seconds. */
const PERIOD = 3.5;
/** Waves between the centre and the edge at once. */
const WAVES = 1.6;

/** How bright the mould is and how strongly it pulses, each 0 to 2. */
export const Look = d.struct({ light: d.f32, pulse: d.f32 });

/** Maps tone from 0 to 1 to a colour. Values above 1 exceed white. */
export const ramp = (tone: number) => {
  'use gpu';
  let color = CRIMSON * std.smoothstep(0, 0.25, tone);
  color = std.mix(color, RED, std.smoothstep(0.2, 0.5, tone));
  color = std.mix(color, ORANGE, std.smoothstep(0.45, 0.8, tone));
  color = std.mix(color, GOLD, std.smoothstep(0.75, 1, tone));
  // Values above white let bloom pick up dense hubs and thick trails
  return color * (1 + 1.2 * std.smoothstep(0.85, 1.2, tone));
};

export const VEINS = { light: 1, pulse: 1 };

/**
 * Maps trail density through a tone curve and colour ramp, then adds it over the agar with
 * glowing food.
 */
export function createVeins(
  root: TgpuRoot,
  world: TgpuUniform<typeof World>,
  options: { trail: Trail; oats: Oats; format: GPUTextureFormat },
) {
  const { trail, oats, format } = options;
  const look = root.createUniform(Look, VEINS);

  const veins = tgpu.fragmentFn({ in: { pixel: d.builtin.position }, out: d.vec4f })(({
    pixel,
  }) => {
    'use gpu';
    const at = sheetAt(pixel.xy, world.$.size);
    const uv = at * 0.5 + 0.5;

    const glow = oats.glowAt(uv);
    const food = FOOD * (glow.x * 3) + HALO * (glow.y * 0.008);

    // Average nearby cells to hide grain from individual agents
    const reach = 0.75 / d.f32(trail.size);
    const density =
      (trail.at(uv) * 2 +
        trail.at(uv + d.vec2f(reach, 0)) +
        trail.at(uv - d.vec2f(reach, 0)) +
        trail.at(uv + d.vec2f(0, reach)) +
        trail.at(uv - d.vec2f(0, reach))) /
      6;
    const tone = std.log(1 + density * GAIN) / std.log(1 + FULL * GAIN);

    // Outward waves of brightness imitate shuttle streaming
    const swirl = 0.15 * std.sin(std.atan2(at.y, at.x) * 3 + std.length(at) * 5);
    const phase = world.$.time / PERIOD - std.length(at) * WAVES + swirl;
    const wave = std.sin(phase * Math.PI * 2) * std.smoothstep(0.55, 0.9, tone);
    const pulse = 1 + 0.4 * look.$.pulse * wave;

    return d.vec4f(ramp(tone) * (look.$.light * pulse) + food, 0);
  });

  const drawing = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: veins,
    targets: { format, blend: pipelines.blend.add },
  });

  return {
    look,
    initAsync: () => drawing.initAsync(),
    draw(pass: TgpuRenderCommands) {
      pass.setPipeline(drawing);
      pass.draw(3);
    },
  };
}
