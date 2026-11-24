import { backends, pipelines } from '@typegpu/kit';
import { randf } from '@typegpu/noise';
import {
  d,
  std,
  tgpu,
  type StorageFlag,
  type TgpuBuffer,
  type TgpuUniform,
  type TgpuCommandEncoder,
  type TgpuComputePass,
  type TgpuRoot,
} from 'typegpu';
import type { Trail } from './trail.ts';

const WORKGROUP = 64;
/** Initial agent count. */
const SEED = 16384;
/** Fraction of the initial blob's cells occupied. */
const FILL = 0.2;
/** Agents a 3 × 3 neighbourhood holds before growth into it stops. */
const ROOM = 5;
/** Agents in a 3 × 3 neighbourhood that make an agent there leave. */
const CROWD = 30;
/** The chance each step that an agent checks whether it stands in a crowd. */
const THINNING = 0.05;
/** The golden angle prevents agents in the initial blob from lining up. */
const GOLDEN = Math.PI * (3 - Math.sqrt(5));
/**
 * Agents in a 3 × 3 neighbourhood per unit of trail density there, roughly, from an agent's
 * deposit, the decay and the blur. Estimates crowding where agents leave cells unclaimed.
 */
const AGENTS_PER_TRAIL = 0.85;
/** How much an agent's heading wanders each step on its own, in radians either way. */
const WANDER = 0.08;

/**
 * Agent position in trail cells and heading in radians. `awake` is 1 after growth and 0 before
 * it. Food level falls from 1 when fed to 0 when starved.
 */
const Agent = d.struct({ position: d.vec2f, heading: d.f32, awake: d.u32, fed: d.f32 });

/**
 * Maximum agent count, growth limit, sensing, steering and hunger. Angles are in radians and
 * distances in trail cells. Hunger is the food level lost per step.
 */
export const Physarum = d.struct({
  count: d.u32,
  alive: d.u32,
  sensorAngle: d.f32,
  sensorDistance: d.f32,
  turn: d.f32,
  speed: d.f32,
  deposit: d.f32,
  hunger: d.f32,
});

/** No food attraction or hunger. */
const enough = (_uv: d.v2f) => {
  'use gpu';
  return d.vec2f(0, 1);
};

const toward = (angle: number) => {
  'use gpu';
  return d.vec2f(std.cos(angle), std.sin(angle));
};

/**
 * Moves Physarum agents along a trail. Each agent senses ahead, turns toward the strongest
 * trail, moves and deposits more. Agents grow where there is room, forming a network of
 * threads.
 *
 * @param trail The grid for sensing and deposits. Its size sets the sheet size.
 * @param options.food Samples food at grid coordinates from 0 to 1. Returns attraction in `x`
 * and 1 in `y` where an agent can eat. Defaults to no attraction or hunger.
 */
export function createAgents(
  root: TgpuRoot,
  trail: Trail,
  options: { capacity: number; food?: (uv: d.v2f) => d.v2f },
) {
  const { capacity, food = enough } = options;
  const size = trail.size;
  const centre = d.vec2f(size / 2);
  const agentData = root.createBuffer(d.arrayOf(Agent, capacity)).$usage('storage');
  const agents = agentData.as('mutable');
  const physarum = root.createUniform(Physarum);
  // Where kernels run on the CPU, agents leave cells unclaimed, so each writes only itself and
  // runs as a GPU pass; growth then skips the room check and crowds never thin
  const claims = backends.of(root).compute === 'gpu';
  const taken = root.createMutable(d.arrayOf(d.atomic(d.u32), claims ? size * size : 1));

  const cellOf = (position: d.v2f) => {
    'use gpu';
    const cell = d.vec2u(position);
    return cell.y * size + cell.x;
  };

  /** Excludes a one-cell border around the sheet. */
  const onSheet = (position: d.v2f) => {
    'use gpu';
    return std.all(std.ge(position, d.vec2f(1))) && std.all(std.lt(position, d.vec2f(size - 1)));
  };

  const enter = (position: d.v2f) => {
    'use gpu';
    if (!onSheet(position)) return false;
    if (claims) {
      std.atomicAdd(taken.$[cellOf(position)], 1);
    }
    return true;
  };

  const neighbours = (position: d.v2f) => {
    'use gpu';
    let count = d.u32(0);
    if (!claims) {
      // Agents in the neighbourhood, estimated from the trail they keep up
      return d.u32(trail.at(position / size) * AGENTS_PER_TRAIL);
    }
    for (let y = -1; y <= 1; y++) {
      for (let x = -1; x <= 1; x++) {
        count += std.atomicLoad(taken.$[cellOf(position + d.vec2f(d.f32(x), d.f32(y)))]);
      }
    }
    return count;
  };

  /** Lays agents on a spiral, the first `SEED` awake and the rest asleep, reserved for growth. */
  const scatter = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= physarum.$.count) return;
    randf.seed(d.f32(gid.x));
    const out = d.f32(gid.x) * GOLDEN;
    const position = centre + toward(out) * std.sqrt(d.f32(gid.x) / (Math.PI * FILL));
    const heading = randf.sample() * Math.PI * 2;
    let awake = d.u32(0);
    if (gid.x < SEED && enter(position)) awake = 1;
    agents.$[gid.x] = Agent({ position, heading, awake, fed: 0.5 + 0.5 * randf.sample() });
  });

  /**
   * Samples attraction `distance` cells away at `angle` radians from the heading. Food begins
   * to attract a half-starved agent and attracts it more as hunger grows.
   */
  const sniff = (agent: d.Infer<typeof Agent>, angle: number) => {
    'use gpu';
    const uv = (agent.position + toward(agent.heading + angle) * physarum.$.sensorDistance) / size;
    const hunger = std.saturate(1 - agent.fed * 2);
    return trail.at(uv) + food(uv).x * hunger;
  };

  const move = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= physarum.$.count) return;
    const agent = agents.$[gid.x];

    // A sleeping agent grows ahead of a randomly chosen awake parent if there is room. Its
    // heading changes the random seed between steps while it sleeps
    if (agent.awake === 0) {
      if (gid.x >= physarum.$.alive) return;
      randf.seed2(d.vec2f(d.f32(gid.x), agent.heading));
      agent.heading = randf.sample() * Math.PI * 2;
      const parent = agents.$[d.u32(randf.sample() * d.f32(physarum.$.alive))];
      if (parent.awake === 0) return;
      const heading = parent.heading + (randf.sample() - 0.5) * 2;
      const position = parent.position + toward(heading) * 2;
      if (neighbours(position) < ROOM && enter(position)) {
        // Vary the inherited food level so agents do not all starve at once
        const fed = std.max(parent.fed, randf.sample());
        agents.$[gid.x] = Agent({ position, heading, awake: 1, fed });
      }
      return;
    }
    randf.seed3(d.vec3f(agent.position, d.f32(gid.x)));

    const eats = food(agent.position / size).y > 0.5;
    agent.fed = std.select(agent.fed - physarum.$.hunger, 1, eats);

    // Starved or crowded agents sleep until they can regrow where there is room
    const crowded = randf.sample() < THINNING && neighbours(agent.position) >= CROWD;
    if (agent.fed <= 0 || crowded) {
      if (claims) {
        std.atomicSub(taken.$[cellOf(agent.position)], 1);
      }
      agent.awake = 0;
      return;
    }

    const angle = physarum.$.sensorAngle;
    const ahead = sniff(agent, d.f32(0));
    const left = sniff(agent, -angle);
    const right = sniff(agent, angle);

    // When both sides beat the forward sample, choose a side at random
    const turn = physarum.$.turn * (0.5 + randf.sample());
    let heading = agent.heading + (randf.sample() - 0.5) * 2 * WANDER;
    if (ahead < left && ahead < right) {
      heading += std.select(-turn, turn, randf.sample() < 0.5);
    } else if (left > ahead && left > right) {
      heading -= turn;
    } else if (right > ahead && right > left) {
      heading += turn;
    }

    // At the sheet edge, turn back and deposit trail without moving
    const next = agent.position + toward(heading) * physarum.$.speed;
    const from = cellOf(agent.position);
    if (!onSheet(next)) {
      heading += Math.PI * (0.75 + 0.5 * randf.sample());
    } else {
      if (claims && cellOf(next) !== from) {
        std.atomicSub(taken.$[from], 1);
        std.atomicAdd(taken.$[cellOf(next)], 1);
      }
      agent.position = d.vec2f(next);
    }
    agent.heading = heading;
    // Fed agents deposit more, strengthening trails near food and along paths from it
    trail.deposit(
      d.vec2i(agent.position),
      physarum.$.deposit * (0.3 + 1.7 * agent.fed * agent.fed),
    );
  });

  const scattering = root.createComputePipeline({ compute: scatter });
  const moving = root.createComputePipeline({ compute: move });
  const stamping = claims ? undefined : createStamping(root, agentData, physarum, size);
  let count = 0;
  const grown = { alive: SEED };

  return {
    initAsync: () => pipelines.compile([scattering, moving, ...(stamping ? [stamping] : [])]),

    /** Whether `stamp` adds the deposits, as it does where kernels run on the CPU. */
    stamps: stamping !== undefined,

    /**
     * Draws this step's deposits into the trail's fresh deposits, additively, where kernels run on
     * the CPU and `move` cannot add them with atomics. Does nothing elsewhere.
     */
    stamp(encoder: TgpuCommandEncoder) {
      if (!stamping || !trail.fresh) return;
      stamping
        .with(encoder)
        .withColorAttachment({ view: trail.fresh, loadOp: 'clear', clearValue: [0, 0, 0, 0] })
        .draw(6, count);
    },

    /** Sets the growth limit, sensing, steering and hunger rate. Clamps the count to capacity. */
    set(values: Omit<d.InferInput<typeof Physarum>, 'alive'>) {
      count = Math.min(values.count, capacity);
      grown.alive = Math.min(grown.alive, count);
      physarum.write({ ...values, count, alive: grown.alive });
    },

    /**
     * Limits awake agents by growth age. The initial blob doubles every `doubling` seconds
     * until it reaches the requested count.
     */
    grow(seconds: number, doubling: number) {
      const alive = Math.min(count, Math.floor(SEED * 2 ** (seconds / doubling)));
      if (alive === grown.alive) return;
      grown.alive = alive;
      physarum.patch(grown);
    },

    scatter(encoder: TgpuCommandEncoder) {
      grown.alive = Math.min(SEED, count);
      physarum.patch(grown);
      taken.buffer.clear(encoder);
      const pass = encoder.beginComputePass();
      pass.setPipeline(scattering);
      pass.dispatchWorkgroups(Math.ceil(count / WORKGROUP));
      pass.end();
    },

    move(pass: TgpuComputePass) {
      pass.setPipeline(moving);
      pass.dispatchWorkgroups(Math.ceil(count / WORKGROUP));
    },

    destroy() {
      agentData.destroy();
      taken.buffer.destroy();
    },
  };
}

/** A cell's corners as two triangles, in cells. */
const CORNERS = [
  d.vec2f(0, 0),
  d.vec2f(1, 0),
  d.vec2f(0, 1),
  d.vec2f(0, 1),
  d.vec2f(1, 0),
  d.vec2f(1, 1),
];

/** Draws each awake agent's deposit as its cell, for an additive render target over the trail. */
function createStamping(
  root: TgpuRoot,
  agentData: TgpuBuffer<d.WgslArray<typeof Agent>> & StorageFlag,
  physarum: TgpuUniform<typeof Physarum>,
  size: number,
) {
  const agents = agentData.as('readonly');
  const corners = tgpu.const(d.arrayOf(d.vec2f, 6), CORNERS);
  const vertex = tgpu.vertexFn({
    in: { corner: d.builtin.vertexIndex, agent: d.builtin.instanceIndex },
    out: { position: d.builtin.position, amount: d.f32 },
  })(({ corner, agent }) => {
    'use gpu';
    const one = agents.$[agent];
    // Asleep agents and those beyond the count deposit nothing
    if (one.awake === 0 || agent >= physarum.$.count) {
      return { position: d.vec4f(2, 2, 0, 1), amount: 0 };
    }
    const at = (std.floor(one.position) + corners.$[corner]!) / size;
    const amount = physarum.$.deposit * (0.3 + 1.7 * one.fed * one.fed);
    return { position: d.vec4f(at.x * 2 - 1, 1 - at.y * 2, 0, 1), amount };
  });
  const fragment = tgpu.fragmentFn({ in: { amount: d.f32 }, out: d.vec4f })(({ amount }) => {
    'use gpu';
    return d.vec4f(amount, 0, 0, 1);
  });
  return root.createRenderPipeline({
    vertex,
    fragment,
    targets: {
      format: 'rgba16float',
      blend: {
        color: { srcFactor: 'one', dstFactor: 'one' },
        alpha: { srcFactor: 'one', dstFactor: 'one' },
      },
    },
  });
}
