import { type CameraBase, type Counters, Particle, particles } from '@typegpu/kit';
import { randf } from '@typegpu/noise';
import { d, std, type TgpuRoot } from 'typegpu';
import { BURN_LIFE, type Cells, cellOf, HEIGHT, Kind, Matter, matterOf } from './cells.ts';

const CAPACITY = 2048;
const GRAVITY_CELLS = 45;
const DRAG = 0.5;
const SPEED = { least: 22, most: 52, across: 14 };
const LIFE_SECONDS = { least: 0.7, most: 2.2 };
const SIZE_CELLS = 1.8;

const HOT = d.vec3f(7, 3.4, 0.8);
const COOL = d.vec3f(3.2, 0.55, 0.05);
const CORE = 0.35;
const HALO = 0.2;

/** Seconds an ember ignores its source material after birth. */
const GRACE_SECONDS = 0.08;
const STEAM_LIFE = 120;
/** Chance an ember that lands on wood sets it alight. */
const IGNITES = 0.2;

const glowing = (ember: Particle, local: d.v2f) => {
  'use gpu';
  const radius = std.length(local);
  const core = 1 - std.smoothstep(CORE, CORE * 2, radius);
  const halo = std.saturate(1 - radius) * HALO;
  const left = std.saturate(1 - ember.age / ember.life);
  const brightness = (core + halo) * left;
  return d.vec4f(ember.color.rgb * brightness, 0);
};

/** Emits sparks from lava and burning wood. Sparks ignite wood and turn to steam on water. */
export function createEmbers(
  root: TgpuRoot,
  options: {
    readonly camera: CameraBase;
    readonly step: number;
    readonly cells: Cells;
    readonly counts: Pick<Counters<'burnt'>, 'add'>;
  },
) {
  const { camera, step, cells, counts } = options;

  const land = (ember: Particle) => {
    'use gpu';
    const next = Particle(ember);
    if (next.age >= next.life || next.age < GRACE_SECONDS) return next;
    const at = d.vec2i(
      d.i32(std.floor(next.position.x)),
      d.i32(std.floor(HEIGHT - next.position.y)),
    );
    const cell = cells.cellAt(at);
    if (cells.traitsOf(cell).kind === Kind.gas) return next;
    randf.seed2(next.position.xy + next.seed);
    const matter = matterOf(cell);
    if (matter === Matter.wood && randf.sample() < IGNITES) {
      cells.put(at, cellOf(Matter.burning, BURN_LIFE));
      counts.add.burnt(1);
    } else if (matter === Matter.water) {
      cells.put(at, cellOf(Matter.steam, STEAM_LIFE));
    }
    next.life = next.age;
    return next;
  };

  const sparks = particles.create(root, camera, {
    label: 'embers',
    capacity: CAPACITY,
    step,
    dimensions: 2,
    update: land,
    shade: glowing,
    settings: { gravity: [0, -GRAVITY_CELLS, 0], drag: DRAG },
  });

  /** Emits from the top of a cell using y-down cell coordinates and `heat` from 0 to 1. */
  const spark = (at: d.v2f, heat: number) => {
    'use gpu';
    const speed = SPEED.least + (SPEED.most - SPEED.least) * (randf.sample() * 0.6 + heat * 0.4);
    const across = (randf.sample() * 2 - 1) * SPEED.across;
    sparks.emit(
      Particle({
        position: d.vec3f(at.x, HEIGHT - at.y + 0.5, 0),
        age: 0,
        velocity: d.vec3f(across, speed, 0),
        life: LIFE_SECONDS.least + (LIFE_SECONDS.most - LIFE_SECONDS.least) * randf.sample(),
        color: d.vec4f(std.mix(COOL, HOT, heat), 1),
        size: SIZE_CELLS,
        seed: randf.sample(),
      }),
    );
  };

  return { sparks, spark };
}
