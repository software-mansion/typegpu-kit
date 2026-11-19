import { instances, machines, meshPool, poses } from '@typegpu/kit';
import { clamp } from 'math';
import { mat4n, quat, quatn, vec3n, type QuatArg } from 'wgpu-matrix';
import { d, type TgpuRoot } from 'typegpu';
import type { Model } from '../models.ts';
import { breathing, gripIn, opposite, raised } from './clips.ts';
import { isGlow } from './stalls.ts';

type Vec3 = [number, number, number];
type Part = Pick<Model, 'meshes' | 'images'>;
/** A ray from the eye, as `view.rayAt` writes it. */
interface Ray {
  readonly origin: Float32Array;
  readonly direction: Float32Array;
}

/** Clip names in the KayKit animation file. */
const CLIPS = {
  stand: 'Idle_A',
  idle: 'Idle_B',
  walk: 'Walking_A',
  run: 'Running_A',
  hold: 'Holding_A',
  wave: 'Waving',
};
/** Seconds into the run where its left foot lands, as the walk's does at its start. */
const RUN_LANDS = 0.08;
/** Ground speed in metres per second at mix 1, the walk, and mix 2, the run. */
const WALK_SPEED = 0.8;
const RUN_SPEED = 3;

/** How the customer hurries in and strolls back: top speeds in m/s, braking and speeding up in m/s². */
const HURRY = 2.4;
const BRAKE = 0.45;
const ACCEL = 1.6;
const STROLL = 0.95;
/** Metres a second the customer still walks as they reach the end of the street. */
const CREEP = 0.1;
/** Metres from either end of the route that count as there. */
const ARRIVED_METRES = 0.01;
/** How fast the customer and the seller turn to face their way, as a share a second. */
const TURNING = 5;
const SELLER_TURNING = 2;
/** Seconds the customer stays at the table, and waits down the street before coming back. */
const STAY = 9;
const AWAY = 3;
/** The customer comes up the street from the tavern, round the table's end to `customer`. */
const ROUTE: readonly (readonly [number, number])[] = [
  [-3, -18],
  [-3.2, -9.3],
  [-2.9, -4],
  [-2.1, 0.4],
  [-1.6, 2.2],
];

/** How the seller's arm moves: fades in seconds, and how far she turns to a customer. */
const RAISE_FADE = 0.6;
const LOWER_FADE = 0.8;
const LEAN_IN = 0.4;
/** Seconds a wave takes to lift the arm, and to lower it again. */
const WAVE_IN = 0.35;
const WAVE_OUT = 0.5;

/** Where a tap finds a person: a standing cylinder this wide and tall around each. */
const REACH = 0.5;
const HEIGHT = 2.3;

/** How much the lantern the seller holds is scaled, and how bright its glass is. */
const LANTERN_SCALE = 1.1;
const GLASS: Vec3 = [5, 4, 3];
/** KayKit items face backwards in the hand slots; half a turn about the grip faces them forward. */
const GRIP = mat4n.rotationY(Math.PI);
const UP = [0, 1, 0] as const;

const Person = instances.define('Person', {
  host: { pose: d.u32 },
  pose: (row) => {
    'use gpu';
    return row.pose;
  },
});

/** Wraps `angle` in radians to -π through π. */
const wrapped = (angle: number) => angle - Math.PI * 2 * Math.round(angle / (Math.PI * 2));

/** Ground speed to a mix of the line: standing at 0, walking at 1 and running at 2. */
const mixFor = (speed: number) =>
  speed <= WALK_SPEED ? speed / WALK_SPEED : 1 + (speed - WALK_SPEED) / (RUN_SPEED - WALK_SPEED);

/**
 * The offset that hangs `lantern` upright, its ring at the grip, from a hand slot that turns as
 * `grip` does: the `GRIP` half turn, the turn upright, the scale and the drop from its top.
 */
function hanging(lantern: Part, grip: QuatArg) {
  const up = vec3n.transformQuat(UP, quatn.conjugate(grip));
  // Up in the item's space, before the half turn about y that GRIP adds
  const upright = quatn.rotationTo(UP, [-up[0], up[1], -up[2]]);
  let top = -Infinity;
  for (const { positions } of lantern.meshes) {
    for (let i = 1; i < positions.length; i += 3) top = Math.max(top, positions[i]);
  }
  const offset = mat4n.multiply(GRIP, mat4n.fromQuat(upright));
  mat4n.uniformScale(offset, LANTERN_SCALE, offset);
  return mat4n.translate(offset, [0, -top, 0], offset);
}

export interface PeopleOptions {
  readonly mage: Model;
  readonly rogue: Model;
  readonly clips: { readonly clips: readonly poses.ClipData[] };
  /** The lantern the seller holds, from the market's file. */
  readonly lantern: Part;
  /** Where the seller stands behind her table. */
  readonly seller: Vec3;
  /** Where the customer stops at the table. */
  readonly customer: Vec3;
}

/**
 * A potion seller behind her table and a customer who hurries up the street out of the rain,
 * slows to a stop at the table and leaves again. Both breathe on an additive layer. The seller
 * holds a lantern on a layer masked to her right arm, and raises it while a customer is there.
 * A tap on either plays a wave on a layer masked to the spine, before the lantern's layer, so
 * the seller waves her free left hand and keeps the lantern.
 */
export function createPeople(root: TgpuRoot, options: PeopleOptions) {
  const skin = options.mage.skin!;
  const clip = (name: string) => poses.clip(options.clips, name);
  const hold = clip(CLIPS.hold);
  const stand = clip(CLIPS.stand);
  // KayKit's wave leans the body away from the hand from the hips; masked to the spine, the
  // spine takes that lean
  const wave = poses.lean(clip(CLIPS.wave), skin, 'spine', { from: stand });
  const figures = poses.create(
    root,
    skin,
    {
      moving: [stand, clip(CLIPS.walk), poses.shift(clip(CLIPS.run), RUN_LANDS)],
      idle: clip(CLIPS.idle),
      hold,
      raise: raised(hold, skin),
      wave,
      waveLeft: poses.mirror(wave, opposite),
      breathe: breathing(skin),
    },
    {
      capacity: 2,
      layers: {
        wave: { joints: ['spine'] },
        lantern: { joints: ['upperarm.r'] },
        breath: { joints: ['spine'], add: true },
      },
    },
  );

  // The bodies, and the lantern hanging from the seller's right grip
  const mage = poses.rebind(options.mage, skin, skin);
  const rogue = poses.rebind(options.rogue, options.rogue.skin!, skin);
  const { lantern } = options;
  const offset = hanging(lantern, gripIn(hold, skin));
  const carried = (meshes: Part['meshes']) =>
    poses.attach({ meshes, images: lantern.images }, skin, 'handslot.r', { offset });
  const iron = carried(lantern.meshes.filter((mesh) => !isGlow(mesh)));
  const glass = carried(lantern.meshes.filter(isGlow));
  const room = meshPool.roomFor([mage, rogue, iron, glass]);
  const pool = meshPool.create(root, room);
  const bodies = {
    mage: pool.add(mage),
    rogue: pool.add(rogue),
    iron: pool.add(iron),
    glass: pool.add(glass),
  };
  // Each body once, a row for each of its meshes
  // They walk, so they keep last frame's rows for TAA's motion
  const people = instances.create(root, pool, {
    kind: Person,
    capacity: room.capacity,
    history: true,
  });

  let now = 0;
  const seller = {
    pose: figures.add('idle', 0, { from: 0.7 }),
    at: options.seller,
    yaw: 0,
    id: undefined as unknown as instances.Id<typeof Person>,
    /** The clip a tap waves, with her free left hand. */
    waveClip: 'waveLeft' as const,
    waving: -Infinity,
  };
  figures.play(seller.pose, 'hold', 0, { layer: 'lantern' });
  figures.play(seller.pose, 'breathe', 0, { layer: 'breath', from: 1.3 });
  // One figure: the lantern's glass glows
  const dressed = [...bodies.mage.meshes, ...bodies.iron.meshes, ...bodies.glass.meshes];
  seller.id = people.add({ meshes: dressed }, (mesh) =>
    bodies.glass.meshes.includes(mesh)
      ? { pose: seller.pose, emission: GLASS }
      : { pose: seller.pose },
  );

  // The customer's route by length, so a distance along it gives a place and a heading
  const route = [...ROUTE, [options.customer[0], options.customer[2]] as const];
  const lengths = [0];
  for (let k = 1; k < route.length; k++) {
    const [ax, az] = route[k - 1];
    const [bx, bz] = route[k];
    lengths.push(lengths[k - 1] + Math.hypot(bx - ax, bz - az));
  }
  const length = lengths.at(-1)!;
  /** Where the customer is, and which way the route heads there, written by `along`. */
  const spot = { x: 0, z: 0, heading: 0 };
  const along = (metres: number) => {
    let k = 1;
    while (k < route.length - 1 && lengths[k] < metres) k++;
    const a = route[k - 1];
    const b = route[k];
    const t = clamp((metres - lengths[k - 1]) / (lengths[k] - lengths[k - 1]), 0, 1);
    spot.x = a[0] + (b[0] - a[0]) * t;
    spot.z = a[1] + (b[1] - a[1]) * t;
    spot.heading = Math.atan2(b[0] - a[0], b[1] - a[1]);
    return spot;
  };

  const customer = {
    pose: figures.add('moving', 0, { mix: 0 }),
    id: undefined as unknown as instances.Id<typeof Person>,
    metres: 0,
    speed: 0,
    yaw: 0,
    waveClip: 'wave' as const,
    waving: -Infinity,
  };
  figures.play(customer.pose, 'breathe', 0, { layer: 'breath', from: 2.1 });
  figures.weigh(customer.pose, 0.7, { layer: 'breath' });
  customer.id = people.add(bodies.rogue, { pose: customer.pose });

  const { position, rotation } = people.fields;
  const turned = quat.create();
  const place = (id: instances.Id<typeof Person>, x: number, y: number, z: number, yaw: number) => {
    position.set(id, x, y, z);
    rotation.write(id, quat.fromAxisAngle(UP, yaw, turned));
  };

  const halt = () => {
    customer.speed = 0;
  };
  /**
   * The customer's visit: waiting down the street, hurrying up it, staying at the table and
   * strolling back. The seller raises her lantern while they stay.
   */
  const Visit = machines.define('Visit', {
    states: {
      away: { enter: halt, next: [{ to: 'coming', after: AWAY }] },
      coming: { next: [{ to: 'staying', when: () => length - customer.metres < ARRIVED_METRES }] },
      staying: {
        enter: () => {
          halt();
          figures.play(seller.pose, 'raise', now, { layer: 'lantern', fade: RAISE_FADE });
        },
        exit: () => {
          figures.play(seller.pose, 'hold', now, { layer: 'lantern', fade: LOWER_FADE });
        },
        next: [{ to: 'going', after: STAY }],
      },
      going: { next: [{ to: 'away', when: () => customer.metres <= ARRIVED_METRES }] },
    },
  });
  const visit = machines.run(Visit);

  const waveOut = { layer: 'wave' as const, fade: WAVE_OUT };
  const endWave = (person: { pose: poses.Id; waving: number }, time: number) => {
    if (person.waving === -Infinity || time <= person.waving) return;
    figures.stop(person.pose, time, waveOut);
    person.waving = -Infinity;
  };

  /** Moves the customer through the visit, turns both to face their way and ends waves. */
  const step = (time: number, dt: number) => {
    now = time;
    visit.step(dt);
    const state = visit.state();
    const c = customer;
    if (state === Visit.coming || state === Visit.going) {
      const coming = state === Visit.coming;
      const goal = coming
        ? Math.min(HURRY, Math.sqrt(2 * BRAKE * (length - c.metres)))
        : Math.min(STROLL, Math.sqrt(2 * BRAKE * c.metres) + CREEP);
      c.speed += clamp(goal - c.speed, -BRAKE * 2 * dt, ACCEL * dt);
      c.metres = clamp(c.metres + c.speed * (coming ? dt : -dt), 0, length);
    }
    along(c.metres);
    const staying = state === Visit.staying;
    const facing = staying
      ? Math.atan2(seller.at[0] - spot.x, seller.at[2] - spot.z)
      : spot.heading + (state === Visit.going ? Math.PI : 0);
    c.yaw = wrapped(c.yaw + wrapped(facing - c.yaw) * Math.min(TURNING * dt, 1));
    place(c.id, spot.x, 0, spot.z, c.yaw);
    figures.mix(c.pose, mixFor(c.speed), time);

    // The seller turns a little towards a customer at the table
    const toward = Math.atan2(spot.x - seller.at[0], spot.z - seller.at[2]);
    const lean = staying ? wrapped(toward) * LEAN_IN : 0;
    seller.yaw += (lean - seller.yaw) * Math.min(SELLER_TURNING * dt, 1);
    place(seller.id, seller.at[0], seller.at[1], seller.at[2], seller.yaw);

    // A wave ends by handing the arm back to the base
    endWave(seller, time);
    endWave(c, time);
  };

  /** The nearest of the two along `ray` within a standing cylinder, if any. */
  const hit = (ray: Ray) => {
    along(customer.metres);
    const standing = [
      { who: 'seller' as const, x: seller.at[0], z: seller.at[2] },
      { who: 'customer' as const, x: spot.x, z: spot.z },
    ];
    let best: { who: 'seller' | 'customer'; t: number } | undefined;
    const [ox, oy, oz] = ray.origin;
    const [dx, dy, dz] = ray.direction;
    for (const { who, x, z } of standing) {
      // Solve |o + t·d - c| = REACH in the ground plane, then check the height there
      const px = ox! - x;
      const pz = oz! - z;
      const a = dx! * dx! + dz! * dz!;
      const b = px * dx! + pz * dz!;
      const c = px * px + pz * pz - REACH * REACH;
      const disc = b * b - a * c;
      if (disc < 0 || a === 0) continue;
      const t = (-b - Math.sqrt(disc)) / a;
      const y = oy! + t * dy!;
      if (t > 0 && y > 0 && y < HEIGHT && (!best || t < best.t)) best = { who, t };
    }
    return best?.who;
  };

  return {
    initAsync: () => figures.initAsync(),
    animate: figures.animate,
    flush: people.flush,
    ready: Promise.all(Object.values(bodies).map((body) => body.ready)),
    renderer: people.renderer().pipe(figures.inject()),
    step,

    /** Waves the arm of whoever `ray` passes through first. */
    tap(ray: Ray) {
      const who = hit(ray);
      if (!who) return;
      const person = who === 'seller' ? seller : customer;
      figures.play(person.pose, person.waveClip, now, {
        layer: 'wave',
        loop: false,
        fade: WAVE_IN,
      });
      person.waving = now + wave.duration;
    },

    destroy() {
      figures.destroy();
      people.destroy();
      pool.destroy();
    },
  };
}
