import { picker, pipelines, type CameraBase, type SizeSource } from '@typegpu/kit';
import type { Inject } from '@typegpu/kit/inject';
import { perlin3d } from '@typegpu/noise';
import { mulberry32 } from 'math/random';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';
import {
  ARMS,
  bellPoint,
  contraction,
  depthOf,
  frameOf,
  Jelly,
  MAX,
  MOON,
  NETTLE,
  placed,
  POINTS,
  Strand,
  STRANDS,
  TENTACLES,
} from './jelly.ts';
import { createTissue } from './tissue.ts';
import { timeOf, stirAt } from './water.ts';

type Vec3 = [number, number, number];
type Cast = { home: Vec3; size: number; kind: number };

/**
 * The jellies the tank opens with: a sea nettle close to the eye, moon jellies behind it at
 * middle distance, and a nettle further back. The rest are small and far, where the haze takes
 * them.
 */
const CAST: Cast[] = [
  { home: [0.72, 0.5, 1.3], size: 0.32, kind: NETTLE },
  { home: [-0.55, 0.3, 0.55], size: 0.36, kind: MOON },
  { home: [0.3, -0.45, 0.35], size: 0.28, kind: MOON },
  { home: [1.2, 0.75, -0.3], size: 0.3, kind: MOON },
  { home: [-1.45, -0.5, -0.2], size: 0.22, kind: MOON },
  { home: [-0.15, 0.8, -1], size: 0.22, kind: MOON },
  { home: [-1.7, 0.55, -2.4], size: 0.32, kind: NETTLE },
];

/**
 * Each kind's look and rhythm, indexed by kind. Strand lengths are the least and the most added at
 * random, and widths are at the root, all per unit of bell radius.
 */
const KINDS = [
  {
    /** Moon jellies' milky blue-whites, taken in turn. */
    tints: [
      d.vec3f(0.84, 0.9, 1),
      d.vec3f(0.9, 0.88, 1),
      d.vec3f(0.8, 0.92, 1),
      d.vec3f(0.92, 0.9, 1),
    ],
    /** What a moon jelly's gonads and rim glow with. */
    glow: d.vec3f(0.6, 0.72, 1),
    size: 0.12,
    period: 2.6,
    tentacles: { reach: [0.25, 0.75], width: 0.004 },
    arms: { reach: [0.7, 0.35], width: 0.16 },
  },
  {
    /** The sea nettles' amber, and their warm rim. */
    tints: [d.vec3f(1, 0.55, 0.18)],
    glow: d.vec3f(1, 0.5, 0.2),
    size: 0.24,
    period: 3.4,
    tentacles: { reach: [2.2, 4], width: 0.0045 },
    arms: { reach: [3.4, 1.6], width: 0.36 },
  },
] as const;

/** Seconds a contraction takes, and the relaxation after it. */
const CLOSING_SECONDS = 0.35;
const OPENING_SECONDS = 1.6;
/** A pulse's push and a bell's weight per unit of its radius, and the pull of the water. */
const THRUST = 8;
const SINK = 0.5;
const DRAG = 1.8;
/** Seconds after a fright before a jelly can be startled again. */
const CALMING_SECONDS = 1.5;
/** Whether a jelly last startled at `startled` can be startled again at `time`. */
const canStartle = (startled: number, time: number) => {
  'use gpu';
  return time - startled > CALMING_SECONDS;
};
/** Threads a trail workgroup runs, one a strand. */
const TRAIL_WORKGROUP = 64;

/** The flow the jellies drift in, slow and wide. */
const drift = (at: d.v3f, time: number) => {
  'use gpu';
  const p = at * 0.35 + d.vec3f(0, 0, time * 0.03);
  return d.vec3f(
    perlin3d.sample(p),
    perlin3d.sample(p + d.vec3f(5.2, 1.3, 0)) * 0.4,
    perlin3d.sample(p + d.vec3f(9.7, 3.1, 0)),
  );
};

/** Places `MAX` jellies and their strands. */
function seeded() {
  const rng = mulberry32.create(11);
  const next = () => mulberry32.sample(rng);
  const jellies: Jelly[] = [];
  const strands: Strand[] = [];

  for (let i = 0; i < MAX; i++) {
    const cast = CAST[i];
    const kind = cast?.kind ?? (i % 5 === 3 ? NETTLE : MOON);
    const look = KINDS[kind];
    const home: Vec3 = cast?.home ?? [
      (next() * 2 - 1) * 3.2,
      (next() * 2 - 1) * 1.2,
      -2.4 - next() * 4,
    ];
    const size = cast?.size ?? look.size + next() * 0.08;
    const tilt = [(next() - 0.5) * 0.4, 1, (next() - 0.5) * 0.4];
    const length = Math.hypot(...tilt);
    jellies.push({
      position: d.vec3f(...home),
      phase: next(),
      velocity: d.vec3f(),
      size,
      axis: d.vec3f(tilt[0] / length, tilt[1] / length, tilt[2] / length),
      squeeze: 0,
      tint: look.tints[i % look.tints.length],
      kind,
      glow: look.glow,
      seed: next(),
      home: d.vec3f(...home),
      // The far jellies pulse more slowly
      period: look.period * (0.85 + next() * 0.35) * (cast ? 1 : 1.2),
      startled: -100,
    });

    for (let s = 0; s < STRANDS; s++) {
      const arm = s >= TENTACLES;
      const around = arm
        ? ((s - TENTACLES + 0.5) / ARMS) * Math.PI * 2
        : ((s + (next() - 0.5) * 0.6) / TENTACLES) * Math.PI * 2;
      const { reach, width } = arm ? look.arms : look.tentacles;
      strands.push({
        around,
        length: (reach[0] + next() * reach[1]) * size,
        width: width * size,
        arm: arm ? 1 : 0,
        curl: 0.3 + next() * next() * 2.4,
        pace: 0.6 + next() * 0.9,
      });
    }
  }
  return { jellies, strands };
}

/** Each strand's points, hanging straight down from under its jelly's bell. */
function hanging(jellies: readonly Jelly[], strands: readonly Strand[]) {
  const points = new Float32Array(MAX * STRANDS * POINTS * 4);
  for (let i = 0; i < MAX * STRANDS; i++) {
    const jelly = jellies[Math.floor(i / STRANDS)];
    const strand = strands[i];
    const out = strand.arm > 0.5 ? 0.1 : 1;
    for (let p = 0; p < POINTS; p++) {
      const at = (i * POINTS + p) * 4;
      const down = jelly.size * 0.3 + (strand.length * p) / (POINTS - 1);
      points[at] = jelly.position.x + Math.cos(strand.around) * jelly.size * out;
      points[at + 1] = jelly.position.y - down;
      points[at + 2] = jelly.position.z + Math.sin(strand.around) * jelly.size * out;
      points[at + 3] = 1;
    }
  }
  return points;
}

export interface SwarmOptions {
  /** The view the jellies are drawn in, for picking what a tap lands on. */
  readonly size: SizeSource;
  /** Fills the water's accessors. */
  readonly water: Inject;
  /** Seconds per `step`. */
  readonly step: number;
  /** How many jellies swim at first, up to `MAX`. */
  readonly count: number;
}

/** Creates the jellies, the kernels that move them and their strands, and their drawings. */
export function createSwarm(root: TgpuRoot, camera: CameraBase, options: SwarmOptions) {
  const { water, step: dt } = options;
  const start = seeded();
  const jellyBuffer = root.createBuffer(d.arrayOf(Jelly, MAX), start.jellies).$usage('storage');
  const strandBuffer = root
    .createBuffer(d.arrayOf(Strand, MAX * STRANDS), start.strands)
    .$usage('storage');
  // Each strand's points this step and last, for Verlet integration
  const Points = d.arrayOf(d.vec4f, MAX * STRANDS * POINTS);
  const hung = hanging(start.jellies, start.strands);
  const pointBuffer = root.createBuffer(Points).$usage('storage');
  const lastBuffer = root.createBuffer(Points).$usage('storage');
  pointBuffer.write(hung.buffer);
  lastBuffer.write(hung.buffer);

  const counted = root.createUniform(d.u32, options.count);

  const jellies = jellyBuffer.as('mutable');
  const strands = strandBuffer.as('readonly');
  const points = pointBuffer.as('mutable');
  const last = lastBuffer.as('mutable');

  // Each jelly's position and size as the step begins, so jellies keep apart from where the others
  // were rather than from wherever their threads have moved them
  const crowd = tgpu.workgroupVar(d.arrayOf(d.vec4f, MAX));

  // One workgroup holds every jelly
  const swim = tgpu.computeFn({
    workgroupSize: [MAX],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    const jelly = Jelly(jellies.$[i]);
    crowd.$[i] = d.vec4f(jelly.position, jelly.size);
    std.workgroupBarrier();
    if (i >= counted.$) return;
    const time = timeOf.$;
    const stirred = stirAt(jelly.position);

    // A hard stir through the bell startles the jelly, as a tap does
    let startled = jelly.startled;
    if (std.length(stirred) > 0.6 && canStartle(startled, time)) startled = time;
    const fright = std.exp(-std.max(time - startled, 0) * 1.2);

    // A quick contraction pushes the bell up; it relaxes, glides and sinks until the next
    const phase = jelly.phase + dt / jelly.period;
    const closing = CLOSING_SECONDS / jelly.period;
    const beat = phase - std.floor(phase);
    const opening = OPENING_SECONDS / jelly.period;
    const squeeze = contraction(beat, closing, opening) * (1 + fright * 0.25);
    const thrust = std.select(0, std.sin((beat / closing) * Math.PI), beat < closing);

    // Lean toward home, with the flow and the way it is going, always mostly up
    const home = jelly.home - jelly.position;
    const wander = drift(d.vec3f(jelly.seed * 40, 0, 0), time);
    // Each moon jelly leans its own way, most of them a little toward the eye
    const own = d.vec3f(std.fract(jelly.seed * 13) - 0.5, 0, std.fract(jelly.seed * 7) * 0.8 - 0.1);
    const leaning = own * std.select(d.f32(0.6), d.f32(0.2), jelly.kind === NETTLE);
    const going = d.vec3f(jelly.velocity.x, 0, jelly.velocity.z) / jelly.size;
    const lean =
      d.vec3f(home.x, 0, home.z) * 0.4 + wander * 0.25 + stirred * 0.5 + leaning + going * 0.25;
    // Never tipping further than about 30 degrees
    const sideways = std.length(lean.xz);
    const tilt = lean.xz * (std.min(sideways, 0.6) / std.max(sideways, 0.0001));
    const heading = std.normalize(d.vec3f(tilt.x, 1, tilt.y));
    const axis = std.normalize(jelly.axis + (heading - jelly.axis) * (dt * (0.8 + fright * 3)));

    // Keep apart from the jellies around
    let apart = d.vec3f();
    for (let j = d.u32(0); j < counted.$; j++) {
      if (j === i) continue;
      const other = crowd.$[j];
      const away = jelly.position - other.xyz;
      const near = (jelly.size + other.w) * 1.2;
      const distance = std.length(away);
      if (distance < near && distance > 0.0001) {
        apart += (away / distance) * (1 - distance / near);
      }
    }

    const accel =
      axis * (thrust * THRUST * jelly.size) +
      d.vec3f(0, -SINK * jelly.size, 0) +
      home * d.vec3f(0.12, 0.06, 0.12) +
      drift(jelly.position, time) * 0.03 +
      stirred +
      apart * 0.5;
    const velocity = jelly.velocity * std.exp(-DRAG * dt) + accel * dt;

    jellies.$[i] = Jelly({
      position: jelly.position + velocity * dt,
      phase,
      velocity,
      size: jelly.size,
      axis,
      squeeze,
      tint: jelly.tint,
      kind: jelly.kind,
      glow: jelly.glow,
      seed: jelly.seed,
      home: jelly.home,
      period: jelly.period,
      startled,
    });
  });

  const trail = tgpu.computeFn({
    workgroupSize: [TRAIL_WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    const jellyIndex = std.intdiv(i, d.u32(STRANDS));
    if (jellyIndex >= counted.$) return;
    const time = timeOf.$;
    const jelly = Jelly(jellies.$[jellyIndex]);
    const strand = strands.$[i];
    const arm = strand.arm > 0.5;
    const frame = frameOf(jelly.axis);

    // A tentacle hangs from the margin, an oral arm from round the middle under the bell
    let local = bellPoint(jelly, time, 0.99, strand.around);
    if (arm) {
      local = d.vec3f(
        std.cos(strand.around) * 0.2,
        -depthOf(jelly.kind) * 0.4,
        std.sin(strand.around) * 0.2,
      );
    }
    const anchor = placed(jelly, local);
    const base = i * POINTS;
    points.$[base] = d.vec4f(anchor, 1);
    last.$[base] = d.vec4f(anchor, 1);

    const segment = strand.length / (POINTS - 1);
    const keep = std.select(0.97, 0.94, arm);
    const sink = std.select(d.f32(0.3), d.f32(0.45), arm);
    // A curl that travels down each strand, and the water a pulse draws in and the glide lets out
    const curl = std.select(strand.curl * 1.6, d.f32(0.15), arm);
    const spread = std.select((0.3 - jelly.squeeze) * 0.45, d.f32(0.12), arm);
    // Strands resist bending, so a bunched strand curls instead of folding into corners
    const stiff = std.select(d.f32(0.04), d.f32(0.03), arm);
    let grand = anchor + jelly.axis * segment;
    let parent = d.vec3f(anchor);
    let held = d.vec3f(anchor);
    // Follow the leader, then take back from each point's speed the pull its follower got
    // (Müller's dynamic follow-the-leader), so a pulse doesn't fling the strand past the bell
    for (let p = d.u32(1); p < POINTS; p++) {
      const now = points.$[base + p].xyz;
      const before = last.$[base + p].xyz;
      const down = d.f32(p) / (POINTS - 1);
      const wave =
        d.f32(p) * (0.3 + strand.pace * 0.3) -
        time * 2.6 * strand.pace +
        strand.around * 7 +
        jelly.seed * 6.28;
      const across = frame * d.vec3f(std.cos(wave), 0, std.sin(wave));
      const out = now - jelly.position;
      const radial = out - jelly.axis * std.dot(out, jelly.axis);
      const outward = radial / std.max(std.length(radial), 0.0001);
      const force =
        d.vec3f(0, -sink, 0) +
        across * (curl * down) +
        outward * (spread * (1 - down)) +
        drift(now * 2.5, time * 3) * 0.15 +
        stirAt(now) * 1.4;
      const free = now + (now - before) * keep + force * (dt * dt);
      const straight = parent + std.normalize(parent - grand + 0.000001) * segment;
      const moved = std.mix(free, straight, stiff);
      const reach = moved - parent;
      const next = parent + reach * (segment / std.max(std.length(reach), 1e-6));
      if (p > 1) last.$[base + p - 1] = d.vec4f(held + (next - moved) * 0.9, 1);
      held = d.vec3f(now);
      points.$[base + p] = d.vec4f(next, 1);
      grand = d.vec3f(parent);
      parent = d.vec3f(next);
    }
    last.$[base + POINTS - 1] = d.vec4f(held, 1);
  });

  // The tapped jelly nearest the eye flashes, contracts hard and darts away from the tap
  const hits = picker.create(root, camera, {
    capacity: MAX,
    size: options.size,
    position: (i) => {
      'use gpu';
      return d.vec3f(jellies.$[i].position);
    },
    radius: (i) => {
      'use gpu';
      return jellies.$[i].size * 1.4;
    },
    alive: (i) => {
      'use gpu';
      return i < counted.$;
    },
    span: () => count,
    rank: picker.byEye,
  });

  const startle = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    const i = hits.found();
    if (i === picker.none) return;
    const time = timeOf.$;
    const jelly = jellies.$[i];
    const ray = hits.ray();
    const toward = jelly.position - ray.origin;
    const miss = toward - ray.direction * std.dot(toward, ray.direction);
    const dart = std.normalize(d.vec3f(miss.x, 0, miss.z) + ray.direction * 0.3 + 0.0001);
    if (canStartle(jelly.startled, time)) jelly.startled = time;
    // Contract at once, unless already contracting
    if (jelly.phase - std.floor(jelly.phase) > CLOSING_SECONDS / jelly.period) {
      jelly.phase = std.ceil(jelly.phase);
    }
    jelly.velocity = std.mix(jelly.velocity, dart * (jelly.size * 1.6), 0.4);
  });

  const swimming = root.pipe(water).createComputePipeline({ compute: swim });
  const trailing = root.pipe(water).createComputePipeline({ compute: trail });
  const startling = root
    .pipe(water)
    .pipe(camera.inject())
    .createComputePipeline({ compute: startle });
  let count = options.count;
  let tap: [x: number, y: number] | undefined;

  const tissue = createTissue(root, {
    jellies: jellyBuffer.as('readonly'),
    strands,
    points: pointBuffer.as('readonly'),
    count: options.count,
  });

  return {
    tissue,
    initAsync: () => pipelines.compile([hits, swimming, trailing, startling]),

    /** Moves the jellies one fixed step, then their strands. */
    step(pass: TgpuComputePass) {
      pass.setPipeline(swimming);
      pass.dispatchWorkgroups(1);
      pass.setPipeline(trailing);
      pass.dispatchWorkgroups(Math.ceil((count * STRANDS) / TRAIL_WORKGROUP));
    },

    /** Startles the jelly at `x`, `y`, from 0 to 1 across the view from its top left, in `startle`. */
    tap(x: number, y: number) {
      tap = [x, y];
    },

    /** Startles the jelly the latest tap found. Call once a frame, before `step`. */
    startle(pass: TgpuComputePass) {
      if (!tap) return;
      hits.pick(pass, ...tap);
      pass.setPipeline(startling);
      pass.dispatchWorkgroups(1);
      tap = undefined;
    },

    /** Sets how many jellies swim and draw. */
    show(next: number) {
      count = next;
      tissue.show(next);
      counted.write(next);
    },

    destroy() {
      tissue.destroy();
      hits.destroy();
      jellyBuffer.destroy();
      strandBuffer.destroy();
      pointBuffer.destroy();
      lastBuffer.destroy();
    },
  };
}
