import { clock, pipelines, targets, type Frame, type Surface, finish } from '@typegpu/kit';
import { common, d, std, tgpu, type TgpuRoot } from 'typegpu';
import { createAgar, hex, sheetAt, World } from 'typegpu-kit-examples/slime-mould/agar.ts';
import { createAgents } from 'typegpu-kit-examples/slime-mould/agents.ts';
import { createOats } from 'typegpu-kit-examples/slime-mould/oats.ts';
import { createTrail } from 'typegpu-kit-examples/slime-mould/trail.ts';

export const STAGES = ['specks', 'spread', 'sensing', 'food'] as const;
export type Stage = (typeof STAGES)[number];

export interface Options {
  /** Which step's picture. Default `sensing`. */
  readonly stage?: Stage;
  /** Default 100 000, as many to a cell as the example's 400 000 on its larger grid. */
  readonly count?: number;
  /** Cells across the trail. Default 1024. */
  readonly grid?: number;
}

const STEP = 1 / 60;
const HDR = 'rgba16float';
const DEGREES = Math.PI / 180;
const ORANGE = hex(0xff7a2a);
const FOOD = hex(0xffe8b8);
const HALO = hex(0xc8803a);
/** The example's grid, which its sensor distance is measured on. */
const EXAMPLE = 2048;

export async function createScene(root: TgpuRoot, surface: Surface, options: Options = {}) {
  const { stage = 'sensing', count = 100_000, grid = 1024 } = options;
  const steering = stage === 'sensing' || stage === 'food';
  const fed = stage === 'food';

  const world = root.createUniform(World, {
    size: d.vec2f(surface.width, surface.height),
    time: 0,
  });
  const screen = targets.create(root, surface, { hdr: { format: HDR, sampled: true } });
  const final = finish.create(root, {
    format: surface.format,
    source: () => screen.textures.hdr,
    tone: null,
    settings: { vignette: 0 },
  });
  const agar = createAgar(root, world, { format: HDR });
  const trail = createTrail(root, { size: grid });
  const oats = fed ? createOats(root, { size: grid }) : undefined;
  const agents = createAgents(root, trail, {
    capacity: count,
    ...(oats && { food: oats.foodAt }),
  });

  const glowAt =
    oats?.glowAt ??
    ((_uv: d.v2f) => {
      'use gpu';
      return d.vec2f();
    });

  const veins = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: tgpu.fragmentFn({ in: { pixel: d.builtin.position }, out: d.vec4f })(({ pixel }) => {
      'use gpu';
      const uv = sheetAt(pixel.xy, world.$.size) * 0.5 + 0.5;
      const glow = glowAt(uv);
      const food = FOOD * (glow.x * 3) + HALO * (glow.y * 0.008);
      return d.vec4f(ORANGE * std.saturate(trail.at(uv)) + food, 0);
    }),
    targets: { format: HDR, blend: pipelines.blend.add },
  });

  await Promise.all([
    final.initAsync(),
    agar.initAsync(),
    trail.initAsync(),
    oats?.initAsync(),
    agents.initAsync(),
    veins.initAsync(),
  ]);

  // The specks are each step's drops alone; later stages keep and blur the trail as the example does
  trail.settings.write(stage === 'specks' ? { keep: 0, blur: 0 } : { keep: 0.93, blur: 1 });
  oats?.set(100);
  agents.set({
    count,
    sensorAngle: 30 * DEGREES,
    sensorDistance: (35 * grid) / EXAMPLE,
    turn: steering ? 45 * DEGREES : 0,
    speed: 1.2,
    deposit: 0.5,
    hunger: fed ? STEP / 6 : 0,
  });

  const sim = clock.fixedStep({ step: STEP });
  /** Growth start time, in simulated seconds. */
  let born = 0;
  let scattering = true;

  return {
    /** Starts the mould again as a blob at the centre. */
    restart() {
      trail.clear();
      scattering = true;
      born = sim.time;
    },

    frame(frame: Frame) {
      if (screen.fit()) world.patch({ size: d.vec2f(screen.width, screen.height) });

      const steps = sim.update(frame.dt);
      if (steps > 0) {
        world.patch({ time: sim.time });
        // Until the fifth step, only the blob moves; then the rest grow, doubling every 2 seconds
        if (steering) agents.grow(sim.time - born, 2);
      }
      if (scattering) agents.scatter(frame.encoder);
      scattering = false;
      oats?.paint(frame.encoder);
      for (let i = 0; i < steps; i++) {
        const pass = frame.encoder.beginComputePass();
        agents.move(pass);
        trail.spread(pass);
        pass.end();
      }

      const pass = frame.encoder.beginRenderPass({ colorAttachments: { view: screen.views.hdr } });
      agar.draw(pass);
      pass.setPipeline(veins);
      pass.draw(3);
      pass.end();

      const output = frame.encoder.beginRenderPass({
        colorAttachments: { view: surface.view(frame) },
      });
      final.draw(output);
      output.end();
    },

    destroy() {
      agents.destroy();
      trail.destroy();
      oats?.destroy();
      screen.destroy();
    },
  };
}
