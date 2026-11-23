import { instances, machines, pipelines, poses, type MeshPool, type meshPool } from '@typegpu/kit';
import * as grid from '@typegpu/kit/grid';
import { random } from 'math/random';
import { mat4 } from 'wgpu-matrix';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';
import type { Model } from '../models.ts';

export const MODELS = [
  'warrior',
  'minion',
  'rogue',
  'mage',
  'blade',
  'shield',
  'axe',
  'dagger',
  'staff',
  'skeleton-clips',
  'melee-clips',
  'hit-clips',
] as const;
export type ModelName = (typeof MODELS)[number];

/** Animation clip names in the model files. */
const CLIPS = {
  awaken: 'Skeletons_Awaken_Floor',
  walk: 'Skeletons_Walking',
  idle: 'Skeletons_Idle',
  taunt: 'Skeletons_Taunt',
  death: 'Skeletons_Death',
  resurrect: 'Skeletons_Death_Resurrect',
  chop: 'Melee_1H_Attack_Chop',
  stab: 'Melee_1H_Attack_Stab',
  smash: 'Melee_2H_Attack_Chop',
  hit: 'Hit_A',
};
type Clip = keyof typeof CLIPS;
type Attack = 'chop' | 'stab' | 'smash';

/** Seconds into each attack where the blow lands. */
const LANDS: Record<Attack, number> = { chop: 0.63, stab: 0.42, smash: 0.85 };

const CHARACTERS = {
  warrior: { right: 'blade', left: 'shield', attack: 'chop' },
  minion: { right: 'axe', attack: 'chop' },
  rogue: { right: 'dagger', attack: 'stab' },
  mage: { right: 'staff', attack: 'smash' },
} as const satisfies Record<string, { right: ModelName; left?: ModelName; attack: Attack }>;
type Character = keyof typeof CHARACTERS;
const CHARACTER_NAMES = Object.keys(CHARACTERS) as Character[];

/** Walking speed in metres per second, matching planted-foot motion in the clip. */
const PACE = 0.6;
/** Minimum and maximum walking distance in metres. The yard edge can shorten a walk. */
const WALK = [2, 8] as const;
/** Minimum walking duration in seconds, including walks shortened by the yard edge. */
const SHORTEST_WALK = 0.5;
/** Seconds a skeleton takes to turn to where it walks. */
const TURNING = 0.4;
/** Seconds a skeleton takes to blend from one clip into the next. */
const FADE = 0.3;
/** Seconds a flinch takes to blend in over the one before it. */
const FLINCH_FADE = 0.1;

/** Minimum and maximum delay before rising from the ground, in seconds. */
const WAKING = [0.5, 8] as const;
/** Minimum and maximum idle duration in seconds. */
const IDLING = [2, 6] as const;
/** Minimum and maximum delay before rising after a fall, in seconds. */
const LYING = [2, 5] as const;
/** Probability of walking after a stationary clip. Otherwise idles. */
const WANDERS = 0.7;
/** Action probabilities after walking. Unassigned probability means idle. */
const ARRIVED = { death: 0.05, taunt: 0.15, attack: 0.3 };

/** Metres within which an attacker turns on the nearest skeleton and steps in. */
const SIGHT = 2.4;
/** Metres between an attacker and the skeleton it steps in to. */
const CLOSE = 1.5;
/** Metres a blow reaches, in front of the attacker. */
const BLOW = 2;

/** Pose-bound centre height above the feet in metres, for standing and fallen figures. */
const MIDDLE = 1.2;
/** Pose-bound radius in metres. */
const REACH = 2.8;
const WORKGROUP = 64;
const NONE = 0xffffffff;

/**
 * A skeleton between the actions its machine starts. It moves from `start` to `goal` between
 * `departs` and `arrives`, turning about +y from `turn.x` to `turn.y` radians, and its action
 * lasts `lasts` seconds. `attack` is its attack clip, which lasts `swing` seconds and whose blow
 * lands `lands` seconds in.
 */
const Member = d.struct({
  start: d.vec2f,
  goal: d.vec2f,
  turn: d.vec2f,
  departs: d.f32,
  arrives: d.f32,
  lasts: d.f32,
  pose: d.u32,
  attack: d.u32,
  lands: d.f32,
  swing: d.f32,
});

/** One row per mesh, the body and each thing it holds, placed by the march from its member. */
export const Skeleton = instances.define('Skeleton', {
  transform: 'device',
  host: { pose: d.u32, member: d.u32 },
  pose: (skeleton) => {
    'use gpu';
    return skeleton.pose;
  },
  bounds: (skeleton) => {
    'use gpu';
    return instances.Sphere({ center: skeleton.position + d.vec3f(0, MIDDLE, 0), radius: REACH });
  },
});

/** What a walk through the grid reads of each skeleton. */
const Neighbour = d.struct({ at: d.vec2f, slot: d.u32 });

/** Wraps `angle` in radians to -π through π. */
const wrapped = (angle: number) => {
  'use gpu';
  return angle - Math.PI * 2 * std.floor(angle / (Math.PI * 2) + 0.5);
};

type Models = Readonly<Record<ModelName, Model>>;

/** Each clip `CLIPS` names, from the clip files. */
function clipsOf(models: Models) {
  const files = ['skeleton-clips', 'melee-clips', 'hit-clips'] as const;
  const clips = files.flatMap((file) => models[file].clips);
  const pairs = Object.entries(CLIPS).map(([key, name]) => [key, poses.clip({ clips }, name)]);
  return Object.fromEntries(pairs) as Record<Clip, poses.ClipData>;
}

/** Adds each character's meshes to `pool`, rebound to `skin`, with what it holds. */
function arm(pool: MeshPool, models: Models, skin: poses.SkinData) {
  // KayKit weapons face backwards in the hand slot, so they turn half a turn about their grip
  const grip = mat4.rotationY(Math.PI);
  const held = (name: ModelName, joint: string, placed?: { offset: Float32Array }) =>
    pool.add(poses.attach(models[name], skin, joint, placed)).meshes;
  return Object.fromEntries(
    CHARACTER_NAMES.map((name) => {
      const { right, left }: { right: ModelName; left?: ModelName } = CHARACTERS[name];
      const body = pool.add(poses.rebind(models[name], models[name].skin!, skin));
      const meshes = [...body.meshes, ...held(right, 'handslot.r', { offset: grip })];
      if (left) meshes.push(...held(left, 'handslot.l'));
      return [name, { meshes }];
    }),
  ) as Record<Character, { meshes: meshPool.Mesh[] }>;
}

export interface HordeOptions {
  readonly models: Models;
  readonly count: number;
  /** Metres from the middle of the yard to its edge. */
  readonly radius: number;
  /** Seconds in a step of the skeletons' machine. */
  readonly step: number;
  /** Draws where the skeletons lie and when they wake. Default `Math.random`. */
  readonly random?: () => number;
}

/**
 * Animates four armed characters on one rig. They rise, wander, idle, taunt, fight and fall. A
 * machine on the GPU starts each action and its clip, turns attackers on their nearest neighbour
 * and makes the one a blow lands on flinch on an additive layer. A kernel moves each skeleton.
 */
export function createHorde(root: TgpuRoot, pool: MeshPool, options: HordeOptions) {
  const { models, count, radius, step, random: dice = Math.random } = options;

  const skin = models.warrior.skin!;
  const clips = clipsOf(models);
  const figures = poses.create(root, skin, clips, {
    capacity: count,
    layers: { flinch: { joints: ['spine'], add: true } },
  });
  const { base, flinch } = figures.layers;
  const { idle, walk, taunt, death, resurrect, hit } = figures.clips;

  const armed = arm(pool, models, skin);
  const rows = Math.max(...CHARACTER_NAMES.map((name) => armed[name].meshes.length));
  const horde = instances.create(root, pool, { kind: Skeleton, capacity: count * rows });
  const now = root.createUniform(d.f32);

  const initial: d.InferInput<typeof Member>[] = [];
  for (let i = 0; i < count; i++) {
    const character = CHARACTER_NAMES[i % CHARACTER_NAMES.length];
    const distance = Math.sqrt(dice()) * radius;
    const angle = random.float(dice, 0, Math.PI * 2);
    const at = d.vec2f(Math.cos(angle) * distance, Math.sin(angle) * distance);
    const yaw = random.float(dice, -Math.PI, Math.PI);
    const wakes = random.float(dice, ...WAKING);
    const pose = figures.add('awaken', wakes, { loop: false });
    const { attack } = CHARACTERS[character];
    horde.add(armed[character], { pose, member: i });
    initial.push({
      start: at,
      goal: at,
      turn: d.vec2f(yaw, yaw),
      departs: 0,
      arrives: 1,
      lasts: wakes + clips.awaken.duration,
      pose,
      attack: figures.clips[attack],
      lands: LANDS[attack],
      swing: clips[attack].duration,
    });
  }
  const members = root.createBuffer(d.arrayOf(Member, count), initial).$usage('storage');
  const member = members.as('mutable');

  /** Where skeleton `k` stands now. */
  const placeOf = (k: number) => {
    'use gpu';
    const one = member.$[k];
    const along = std.saturate((now.$ - one.departs) / std.max(one.arrives - one.departs, 1e-3));
    return std.mix(one.start, one.goal, along);
  };

  // The grid sorts the skeletons once a frame, before the steps whose hooks walk it
  const cells = grid.create(root, {
    capacity: count,
    low: d.vec3f(-radius - SIGHT, -SIGHT / 2, -radius - SIGHT),
    size: d.vec3f((radius + SIGHT) * 2, SIGHT, (radius + SIGHT) * 2),
    sight: SIGHT,
    item: Neighbour,
    position: (k) => {
      'use gpu';
      const at = placeOf(k);
      return d.vec3f(at.x, 0, at.y);
    },
    itemOf: (k) => {
      'use gpu';
      return Neighbour({ at: placeOf(k), slot: k });
    },
  });

  /** A number from `low` to `high` drawn for skeleton `i`. */
  const roughly = tgpu.fn(
    [d.u32, d.f32, d.f32],
    d.f32,
  )((i, low, high) => {
    'use gpu';
    return low + Doing.roll(i) * (high - low);
  });

  /**
   * The nearest skeleton within `reach` of skeleton `i` as the grid sorted it, with slot `NONE`
   * for none. A blow finds one on its feet in the half ahead; otherwise it is one standing still,
   * which an attacker can step in to.
   */
  const nearest = (i: number, reach: number, blow: boolean) => {
    'use gpu';
    const here = member.$[i].goal;
    const yaw = member.$[i].turn.y;
    const facing = d.vec2f(std.sin(yaw), std.cos(yaw));
    let found = Neighbour({ at: d.vec2f(), slot: NONE });
    let best = d.f32(reach);
    const near = cells.around(d.vec3f(here.x, 0, here.y));
    for (let r = d.u32(0); r < grid.RANGES; r++) {
      const range = cells.range(near, r);
      for (let j = d.u32(0); j < range.count; j++) {
        const other = cells.neighbour(range, j);
        const away = other.at - here;
        const gap = std.length(away);
        // The states on their feet run from walking to striking
        const state = Doing.state(other.slot);
        const afoot = state >= Doing.walking && state <= Doing.striking;
        const fits = blow ? std.dot(away, facing) > 0 : state !== Doing.walking;
        if (other.slot !== i && gap < best && afoot && fits) {
          best = gap;
          found = Neighbour(other);
        }
      }
    }
    return found;
  };

  /**
   * Moves skeleton `i` from where it stands to `to` over `seconds`, turning to `yaw` by the
   * shorter arc, and makes its action last as long.
   */
  const go = (i: number, seconds: number, to: d.v2f, yaw: number) => {
    'use gpu';
    const one = member.$[i];
    const facing = wrapped(one.turn.y);
    one.start = d.vec2f(one.goal);
    one.goal = d.vec2f(to);
    one.turn = d.vec2f(facing, facing + wrapped(yaw - facing));
    one.departs = now.$;
    one.arrives = now.$ + seconds;
    one.lasts = seconds;
  };

  /** Keeps skeleton `i` where it stands, facing its way, for `seconds`. */
  const stand = (i: number, seconds: number) => {
    'use gpu';
    const one = member.$[i];
    go(i, seconds, one.goal, one.turn.y);
  };

  /** Plays `clip` once in place, for as long as it lasts. */
  const act = tgpu.fn([d.u32, d.u32, d.f32])((i, clip, seconds) => {
    'use gpu';
    poses.once(member.$[i].pose, clip, base, FADE);
    stand(i, seconds);
  });

  /** Chooses a nearby walking target inside the yard. */
  const wander = (i: number) => {
    'use gpu';
    const here = d.vec2f(member.$[i].goal);
    const angle = roughly(i, 0, Math.PI * 2);
    let to = here + d.vec2f(std.cos(angle), std.sin(angle)) * roughly(i, WALK[0], WALK[1]);
    const out = std.length(to) / radius;
    if (out > 1) to = to / out;
    const seconds = std.max(std.distance(to, here) / PACE, SHORTEST_WALK);

    poses.play(member.$[i].pose, walk, base, FADE, 0);
    // glTF models face +z
    go(i, seconds, to, std.atan2(to.x - here.x, to.y - here.y));
  };

  /** Swings at the nearest skeleton in sight, stepping in to it, or at the air. */
  const windUp = (i: number) => {
    'use gpu';
    const one = member.$[i];
    poses.once(one.pose, one.attack, base, FADE);
    const foe = nearest(i, SIGHT, false);
    if (foe.slot === NONE) {
      stand(i, one.lands);
      return;
    }
    const here = d.vec2f(one.goal);
    const away = foe.at - here;
    const gap = std.length(away);
    go(i, one.lands, here + away * (std.max(gap - CLOSE, 0) / gap), std.atan2(away.x, away.y));
  };

  /** Lands the blow: the nearest skeleton within reach ahead flinches over what it is doing. */
  const strike = (i: number) => {
    'use gpu';
    const one = member.$[i];
    one.lasts = one.swing - one.lands;
    const struck = nearest(i, BLOW, true).slot;
    if (struck !== NONE) poses.once(member.$[struck].pose, hit, flinch, FLINCH_FADE);
  };

  const lasted = (i: number): number => {
    'use gpu';
    return member.$[i].lasts;
  };
  const stood = [
    { to: 'walking', after: lasted, chance: WANDERS },
    { to: 'idling', after: lasted },
  ] as const;

  const Doing = machines.define('Doing', {
    states: {
      waking: { next: stood },
      walking: {
        enter: wander,
        next: [
          { to: 'falling', after: lasted, chance: ARRIVED.death },
          { to: 'taunting', after: lasted, chance: ARRIVED.taunt },
          { to: 'attacking', after: lasted, chance: ARRIVED.attack },
          { to: 'idling', after: lasted },
        ],
      },
      idling: {
        enter: (i) => {
          'use gpu';
          poses.play(member.$[i].pose, idle, base, FADE, 0);
          stand(i, roughly(i, IDLING[0], IDLING[1]));
        },
        next: stood,
      },
      taunting: {
        enter: (i) => {
          'use gpu';
          act(i, taunt, clips.taunt.duration);
        },
        next: stood,
      },
      attacking: { enter: windUp, next: [{ to: 'striking', after: lasted }] },
      striking: { enter: strike, next: stood },
      falling: {
        enter: (i) => {
          'use gpu';
          act(i, death, clips.death.duration);
        },
        next: [{ to: 'lying', after: lasted }],
      },
      // Keep the fallen pose until the rise clip starts
      lying: {
        enter: (i) => {
          'use gpu';
          member.$[i].lasts = roughly(i, LYING[0], LYING[1]);
        },
        next: [{ to: 'rising', after: lasted }],
      },
      rising: {
        enter: (i) => {
          'use gpu';
          act(i, resurrect, clips.resurrect.duration);
        },
        next: stood,
      },
    },
  });
  const doing = machines.create(root, Doing, { capacity: count, step }).pipe(figures.inject());

  const skeleton = horde.gpu;
  const march = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (!skeleton.alive(i)) return;

    const k = skeleton.member(i);
    const one = member.$[k];
    const yaw = std.mix(one.turn.x, one.turn.y, std.smoothstep(0, TURNING, now.$ - one.departs));
    const at = placeOf(k);

    const row = skeleton.device.$[i];
    row.position = d.vec3f(at.x, 0, at.y);
    row.rotation = d.vec4f(0, std.sin(yaw / 2), 0, std.cos(yaw / 2));
  });
  const marching = root.createComputePipeline({ compute: march });

  return {
    horde,
    poses: figures,
    /** What each skeleton is doing, a machine row each. */
    doing,

    initAsync: () => pipelines.compile([figures, cells, doing, marching]),

    /**
     * Takes `steps` fixed steps of the machine, then figures and moves the skeletons at `time`. The
     * steps start clips at `time` too.
     */
    animate(pass: TgpuComputePass, time: number, steps: number) {
      horde.flush();
      now.write(time);
      if (steps > 0) cells.sort(pass);
      for (let s = 0; s < steps; s++) doing.step(pass);
      figures.animate(pass, time);
      pass.setPipeline(marching);
      pass.dispatchWorkgroups(Math.ceil(horde.span / WORKGROUP));
    },

    destroy() {
      horde.destroy();
      figures.destroy();
      cells.destroy();
      doing.destroy();
      members.destroy();
    },
  };
}
