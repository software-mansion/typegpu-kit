import {
  counters,
  impostors,
  instances,
  poses,
  shading,
  type CameraBase,
  type MeshPool,
  type meshPool,
  type Picker,
} from '@typegpu/kit';
import { srgbToLinear } from '@typegpu/color';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';
import type { Model } from '../models.ts';
import { CAPACITY, HORSE_SCALE, MIXES, PHASES, POSES, RESTING_POSES, type Herd } from './herd.ts';

const FADE_METRES = 8;
const NEVER_HANDS_OVER_METRES = 1e6;
/** Body-bound centre height above the hooves and radius in model units. */
const BODY_HEIGHT = 1.2 / HORSE_SCALE;
const BODY_RADIUS = 2 / HORSE_SCALE;
const ATLAS_VIEWS = 5;
const ATLAS_TEXELS = 64;
/** Captured card phases, gallop first and walk second. */
const CARD_GALLOP = 6;
const CARD_WALK = 4;
const COUNTS_HZ = 15;
const WORKGROUP = 64;
const TINT = [0.25, 0.7, 1.4] as const;
const WARM = d.vec3f(1.15, 1, 0.8);
const GLOW = d.vec3f(0.35, 0.2, 0.06);
const GLOW_BASE = d.vec3f(0.02, 0.01, 0.002);
const RIM_POWER = 3;
const ROUGHNESS = 0.72;

type Triple = readonly [number, number, number];

/** Body and mane colours in sRGB, with coat frequency weights. */
const COATS = [
  { name: 'bay', body: [0.45, 0.25, 0.13], mane: [0.07, 0.05, 0.04], weight: 5 },
  { name: 'chestnut', body: [0.56, 0.3, 0.14], mane: [0.5, 0.27, 0.12], weight: 4 },
  { name: 'brown', body: [0.28, 0.17, 0.1], mane: [0.06, 0.05, 0.04], weight: 3 },
  { name: 'black', body: [0.11, 0.1, 0.095], mane: [0.05, 0.05, 0.05], weight: 2 },
  { name: 'grey', body: [0.62, 0.62, 0.6], mane: [0.36, 0.36, 0.36], weight: 2 },
  { name: 'white', body: [0.85, 0.83, 0.79], mane: [0.9, 0.88, 0.84], weight: 1 },
  { name: 'palomino', body: [0.8, 0.6, 0.31], mane: [0.94, 0.89, 0.76], weight: 2 },
  { name: 'dun', body: [0.7, 0.57, 0.4], mane: [0.14, 0.11, 0.09], weight: 2 },
] as const;
/** Model coat-material brightness multipliers. */
const SHADES: Readonly<Record<string, number>> = {
  Main: 1,
  Main_Dark: 0.6,
  Main_Light: 1.45,
  Muzzle: 0.35,
};
const MANE = 'Hair';
/** Neutral capture grey. Card tint multiplies it into the selected coat. */
const CAPTURED = 0.5;
const CAPTURED_MANE = 0.45;
const GOLDEN = (Math.sqrt(5) - 1) / 2;

const linear = (color: Triple): Triple => {
  const value = srgbToLinear(d.vec3f(...color));
  return [value.x, value.y, value.z];
};
const scaled = (color: Triple, by: number): Triple => [color[0] * by, color[1] * by, color[2] * by];
/** `color` times `by`, each channel at most 1. */
const brightened = (color: Triple, by: number): Triple => [
  Math.min(color[0] * by, 1),
  Math.min(color[1] * by, 1),
  Math.min(color[2] * by, 1),
];
const coats = COATS.map(({ body, mane }) => ({ body: linear(body), mane: linear(mane) }));
const totalWeight = COATS.reduce((sum, { weight }) => sum + weight, 0);

/** Assigns coats by weighted frequency across horse indices, with brightness variation. */
function coatOf(member: number) {
  let pick = ((member * GOLDEN) % 1) * totalWeight;
  let coat = 0;
  while (pick >= COATS[coat].weight) pick -= COATS[coat++].weight;
  const shade = 0.88 + 0.24 * ((member * GOLDEN * GOLDEN * 7) % 1);
  const { body, mane } = coats[coat];
  return { body: scaled(body, shade), mane };
}

/**
 * How a horse mesh takes its coat: the mane colour, the body colour at a brightness, or its own
 * material colour.
 */
type Coating = 'mane' | number | undefined;
const coatingOf = (material: string): Coating => (material === MANE ? 'mane' : SHADES[material]);
const coatColor = (coat: ReturnType<typeof coatOf>, coating: Coating, own: Triple) => {
  if (coating === 'mane') return coat.mane;
  return coating === undefined ? own : brightened(coat.body, coating);
};
/** Grey a mesh is captured in for cards, which the card tint multiplies into its coat. */
const capturedGrey = (coating: Coating) => {
  if (coating === 'mane') return CAPTURED_MANE;
  return coating === undefined ? undefined : CAPTURED * coating;
};

/** Animation clip names in the model file. */
const CLIPS = { gallop: 'Gallop', walk: 'Walk', idle: 'Idle', graze: 'Eating' };

/** Shared figures ordered to match herd pose ids. */
export function createPoses(root: TgpuRoot, model: Model) {
  const clip = (name: string) => poses.clip(model, name);
  const clips = {
    gallop: clip(CLIPS.gallop),
    walk: clip(CLIPS.walk),
    idle: clip(CLIPS.idle),
    graze: clip(CLIPS.graze),
  };
  const figures = poses.create(
    root,
    model.skin!,
    { moving: [clips.walk, clips.gallop], idle: clips.idle, graze: clips.graze },
    { capacity: POSES },
  );
  // Hold every phase of the walk-to-gallop cycle at each mix, so a horse changing gait keeps its
  // phase and steps through the mixes
  const moving = Array.from({ length: MIXES * PHASES }, (_, i) => {
    const mix = Math.floor(i / PHASES) / (MIXES - 1);
    const cycle = clips.walk.duration + (clips.gallop.duration - clips.walk.duration) * mix;
    return figures.add('moving', 0, { from: ((i % PHASES) / PHASES) * cycle, mix, speed: 0 });
  });
  const playing = (name: 'idle' | 'graze') =>
    Array.from({ length: RESTING_POSES }, (_, i) =>
      figures.add(name, 0, { from: (clips[name].duration * i) / RESTING_POSES }),
    );
  const gallop = moving.slice((MIXES - 1) * PHASES);
  const walk = moving.slice(0, PHASES);
  const ids = [...moving, ...playing('idle'), ...playing('graze')];
  return {
    poses: figures,
    ids: ids as readonly number[],
    /** Captures every fourth gallop phase and every sixth walk phase. */
    captured: [
      ...Array.from({ length: CARD_GALLOP }, (_, i) => gallop[(i * PHASES) / CARD_GALLOP]),
      ...Array.from({ length: CARD_WALK }, (_, i) => walk[(i * PHASES) / CARD_WALK]),
    ],
  };
}

export type Posed = ReturnType<typeof createPoses>;

interface LevelsOptions {
  readonly body: meshPool.Model;
  /** `impostors.CARD` in the pool. */
  readonly card: meshPool.Mesh;
  readonly posed: Posed;
  readonly herd: Herd;
  readonly picked: Picker;
  readonly view: CameraBase;
  /** Original model for material-name lookup. */
  readonly model: Model;
}

/** Draws nearby horses as skins and distant horses as cards, culled for scene and shadow views. */
export function createLevels(root: TgpuRoot, pool: MeshPool, options: LevelsOptions) {
  const { body, card, posed, herd, picked, view, model } = options;
  const coatings = model.meshes.map((mesh) => coatingOf(mesh.material.name ?? ''));
  /** The hovered and the followed horse, which stay skinned and glow. */
  const selectedOf = (member: number) => {
    'use gpu';
    return member === picked.found() || member === herd.controls.$.followed;
  };

  /** Where horse `member` stands and faces, at its row's `scale`, which `populate` sizes. */
  const placementOf = (member: number, scale: d.v3f) => {
    'use gpu';
    const horse = herd.horses.$[member];
    return instances.Placement({
      rotation: instances.toward(d.vec3f(horse.facing.x, 0, horse.facing.y)),
      position: herd.drawnAt(member),
      scale,
    });
  };

  const HorseBody = instances.define('Horse', {
    host: { member: d.u32 },
    place: (horse) => {
      'use gpu';
      return placementOf(horse.member, horse.scale);
    },
    pose: (horse) => {
      'use gpu';
      return herd.horses.$[horse.member].pose;
    },
    bounds: (horse) => {
      'use gpu';
      return instances.Sphere({
        center: herd.drawnAt(horse.member) + d.vec3f(0, BODY_HEIGHT, 0) * horse.scale,
        radius: BODY_RADIUS * horse.scale.x,
      });
    },
    material: (pixel, base) => {
      'use gpu';
      const point = pixel.point;
      const selected = selectedOf(pixel.instance.member);
      const rim = std.pow(1 - std.saturate(std.dot(point.normal, point.view)), RIM_POWER);
      const lit = shading.Material(base);
      lit.color = base.color * std.select(d.vec3f(1), WARM, selected);
      lit.emission = base.emission + std.select(d.vec3f(), GLOW * rim + GLOW_BASE, selected);
      return lit;
    },
  });
  const HorseCard = impostors.define('HorseCard', {
    host: { member: d.u32 },
    place: (horse) => {
      'use gpu';
      return placementOf(horse.member, horse.scale);
    },
  });
  const captured = body.meshes.map((mesh, i) => {
    const grey = capturedGrey(coatings[i]);
    return {
      mesh,
      pose: posed.captured,
      ...(grey === undefined ? {} : { color: [grey, grey, grey] as const }),
    };
  });
  const atlas = impostors.create(root, pool, captured, {
    card,
    views: ATLAS_VIEWS,
    size: ATLAS_TEXELS,
    poses: posed.poses,
  });
  // Evaluate capture figures earlier in the same encoder
  const capture = root.createCommandEncoder();
  const posing = capture.beginComputePass();
  posed.poses.animate(posing, 0);
  posing.end();
  atlas.record(capture);
  capture.submit();

  const bodies = instances.create(root, pool, {
    kind: HorseBody,
    capacity: CAPACITY * body.meshes.length,
  });
  const cards = instances.create(root, pool, { kind: HorseCard, capacity: CAPACITY });
  const cardIds: instances.Id<typeof HorseCard>[] = [];
  let tinted = false;
  const cardColorOf = (member: number) => {
    const { body: coat } = coatOf(member);
    const tint = tinted ? TINT : ([1, 1, 1] as const);
    return d.vec3f(
      (coat[0] / CAPTURED) * tint[0],
      (coat[1] / CAPTURED) * tint[1],
      (coat[2] / CAPTURED) * tint[2],
    );
  };
  const populate = () => {
    bodies.clear();
    cards.clear();
    cardIds.length = 0;
    for (let member = 0; member < herd.count; member++) {
      const coat = coatOf(member);
      const scale = herd.traits[member].size * HORSE_SCALE;
      bodies.add(body, (mesh, i) => {
        const color = coatColor(coat, coatings[i], mesh.material.color);
        return { member, scale, color, roughness: ROUGHNESS, metallic: 0 };
      });
      cardIds.push(
        cards.add(card, {
          member,
          scale,
          color: cardColorOf(member),
          roughness: ROUGHNESS,
          metallic: 0,
        }),
      );
    }
    picked.result.reset();
  };
  populate();

  const keptDistance = (selected: boolean) => {
    'use gpu';
    if (selected) return d.f32(NEVER_HANDS_OVER_METRES);
    return herd.controls.$.handOff;
  };
  const keptTo = (row: number) => {
    'use gpu';
    return keptDistance(selectedOf(bodies.gpu.member(row)));
  };
  const keptFrom = (row: number) => {
    'use gpu';
    return keptDistance(selectedOf(cards.gpu.member(row)));
  };
  const solid = bodies
    .renderer({ cull: true, to: keptTo, fade: FADE_METRES, eye: view })
    .pipe(posed.poses.inject());
  const distant = cards
    .renderer({
      cull: true,
      from: keptFrom,
      fade: FADE_METRES,
      eye: view,
    })
    .pipe(atlas.inject('nearest'));
  const tally = counters.create(root, ['skinned', 'impostors'], { hz: COUNTS_HZ, per: 'frame' });

  // Picks each card's captured phase and counts the horses drawn each way
  const phaseCards = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const row = gid.x;
    if (!cards.gpu.alive(row)) return;
    const member = cards.gpu.member(row);
    const horse = herd.horses.$[member];
    cards.gpu.device.$[row].phase = std.select(
      CARD_GALLOP + (d.u32(horse.phase * CARD_WALK) % CARD_WALK),
      d.u32(horse.phase * CARD_GALLOP) % CARD_GALLOP,
      herd.gaits.gpu.state(member) === herd.gaits.definition.galloping,
    );
    const kept = keptDistance(selectedOf(member));
    if (std.distance(view.uniform.$.position, horse.position) < kept) tally.add.skinned(1);
    else tally.add.impostors(1);
  });
  const phasing = root.createComputePipeline({ compute: phaseCards });

  return {
    flush() {
      bodies.flush();
      cards.flush();
    },
    capture: tally.capture,
    get counts() {
      return tally.latest;
    },
    solid,
    distant,
    populate,
    tint(show: boolean) {
      tinted = show;
      for (let i = 0; i < cardIds.length; i++) cards.write(cardIds[i], { color: cardColorOf(i) });
    },
    initAsync: () => phasing.initAsync(),
    arrange(pass: TgpuComputePass) {
      pass.setPipeline(phasing);
      pass.dispatchWorkgroups(Math.ceil(cards.span / WORKGROUP));
      solid.cull(pass);
      distant.cull(pass);
    },

    destroy() {
      tally.destroy();
      bodies.destroy();
      cards.destroy();
      atlas.destroy();
    },
  };
}
