// The examples, each a `<name>/` directory with `meta.json`, `scene.ts`, a poster and,
// when it ships files, an `assets.ts` manifest. `examples.test.ts` checks this list against the
// directories, and the required files and an individual test for each example

import type { Surface } from '@typegpu/kit';
import type { TgpuRoot } from 'typegpu';
import type { Loaded, Manifest } from './assets.ts';
import type { ExampleScene } from './contract.ts';
import type { Bound, Ui } from './ui.ts';
import afterimage from './afterimage/meta.json';
import fishtank from './fishtank/meta.json';
import { ui as fishtankUi } from './fishtank/ui.ts';
import platformer from './platformer/meta.json';
import { ui as platformerUi } from './platformer/ui.ts';
import blockfall from './blockfall/meta.json';
import censer from './censer/meta.json';
import { ui as blockfallUi } from './blockfall/ui.ts';
import cloister from './cloister/meta.json';
import classroom from './classroom/meta.json';
import benthos from './benthos/meta.json';
import dominoes from './dominoes/meta.json';
import { ui as dominoesUi } from './dominoes/ui.ts';
import fallingSand from './falling-sand/meta.json';
import fireworks from './fireworks/meta.json';
import emberMirror from './ember-mirror/meta.json';
import { ui as emberMirrorUi } from './ember-mirror/ui.ts';
import flowField from './flow-field/meta.json';
import forest from './forest/meta.json';
import hexIsle from './hex-isle/meta.json';
import jellies from './jellies/meta.json';
import jellyWorld from './jelly-world/meta.json';
import jumpFlood from './jump-flood/meta.json';
import kinetic from './kinetic/meta.json';
import lattice from './lattice/meta.json';
import nightMarket from './night-market/meta.json';
import ocean from './ocean/meta.json';
import quiverTrees from './quiver-trees/meta.json';
import raymarch from './raymarch/meta.json';
import robotArm from './robot-arm/meta.json';
import skeletons from './skeletons/meta.json';
import slimeMould from './slime-mould/meta.json';
import solarSystem from './solar-system/meta.json';
import sphSculpt from './sph-sculpt/meta.json';
import { ui as sphSculptUi } from './sph-sculpt/ui.ts';
import spider from './spider/meta.json';
import stampede from './stampede/meta.json';
import starMap from './star-map/meta.json';
import surveyModel from './survey-model/meta.json';

/** The gallery's sections, in order. */
export const categories = [
  { name: 'worlds', title: 'Worlds', line: 'Lit scenes of models, shadows and post-processing.' },
  {
    name: 'simulations',
    title: 'Simulations',
    line: 'Physics, growth and particles stepped on the GPU.',
  },
  { name: 'animation', title: 'Animation', line: 'Skinned crowds, solved limbs and soft bodies.' },
  { name: 'data', title: 'Data', line: 'Picking, labels and lines.' },
  { name: 'games', title: 'Games', line: 'Something to play.' },
  {
    name: 'techniques',
    title: 'Techniques',
    line: 'One technique each, small enough to read in one sitting.',
  },
] as const;

export interface Meta {
  readonly category: string;
  /** The card's tier in the pull, one of `RARITIES` in typegpu-kit-pull's `rarity.ts`. */
  readonly rarity: string;
  readonly title: string;
  /** Page description of the example and its implementation. */
  readonly description: string;
  /** Gallery line. */
  readonly line: string;
  /** Gallery order, lowest first. */
  readonly order: number;
  /** Scene option overrides for gallery cards, such as fewer particles. Default none. */
  readonly card?: Readonly<Record<string, unknown>>;
  /**
   * Scene option overrides where compute runs on the CPU, as on the WebGL fallback: fewer rows or a
   * smaller grid for a simulation whose kernels cannot move to fragment passes. Default none.
   */
  readonly fallback?: Readonly<Record<string, unknown>>;
  /** Device features the scene uses when the adapter has them, such as `timestamp-query`. Default none. */
  readonly features?: readonly string[];
  /** The page's line of keys and gestures, split at `·`, shown to fine pointers. Default none. */
  readonly hint?: string;
  /** How `pnpm posters` renders the poster. */
  readonly poster?: {
    /** Seconds the scene runs before the capture. Default 4. */
    readonly seconds?: number;
    /** WebP quality from 0 to 100. Default 70. */
    readonly quality?: number;
    /**
     * Pixels across the scene renders the poster at, for a scene drawn in pixels, such as points or
     * text, to match a live card. Default 1280.
     */
    readonly width?: number;
  };
}

/** Creates an example's scene from options its files and the host give. */
export type AnyCreateScene = (
  root: TgpuRoot,
  surface: Surface,
  options: object,
) => ExampleScene | Promise<ExampleScene>;

export interface Example extends Meta {
  /** `<name>`, the example's directory under `src`. */
  readonly key: string;
  /** The example's UI declaration, when it has one. */
  readonly ui?: Ui;
  /** Loads the scene file. */
  readonly scene: () => Promise<{ readonly createScene: AnyCreateScene }>;
  /** Loads the manifest, when the example ships files. */
  readonly assets?: () => Promise<Files>;
}

/** An `assets.ts`. */
interface Files {
  readonly assets: Manifest;
}

/** What an example's files give its scene. */
type Given<F> = F extends { readonly assets: infer A extends Manifest } ? Loaded<A> : object;

/** A scene that binds every element of `U`, or any scene without a UI declaration. */
type Returned<U extends Ui> = {} extends U
  ? ExampleScene
  : ExampleScene<U> & { readonly ui: Bound<U> };

type Created<O, U extends Ui> = (
  root: TgpuRoot,
  surface: Surface,
  options: O,
) => Returned<U> | Promise<Returned<U>>;

/**
 * One entry of the list. The scene must take what its files give, and bind every element its UI
 * declares.
 */
function example<F extends Files | undefined = undefined, U extends Ui = {}>(
  key: string,
  meta: Meta,
  scene: () => Promise<{ readonly createScene: Created<NoInfer<Given<F>>, NoInfer<U>> }>,
  files: { readonly assets?: () => Promise<F>; readonly ui?: U } = {},
): Example {
  const { ui, assets } = files;
  return {
    ...meta,
    key,
    ...(ui && { ui }),
    scene: scene as Example['scene'],
    ...(assets && { assets: assets as NonNullable<Example['assets']> }),
  };
}

const list: Example[] = [
  example('afterimage', afterimage, () => import('./afterimage/scene.ts'), {
    assets: () => import('./afterimage/assets.ts'),
  }),
  example('cloister', cloister, () => import('./cloister/scene.ts'), {
    assets: () => import('./cloister/assets.ts'),
  }),
  example('classroom', classroom, () => import('./classroom/scene.ts'), {
    assets: () => import('./classroom/assets.ts'),
  }),
  example('fishtank', fishtank, () => import('./fishtank/scene.ts'), {
    ui: fishtankUi,
  }),
  example('platformer', platformer, () => import('./platformer/scene.ts'), {
    ui: platformerUi,
  }),
  example('blockfall', blockfall, () => import('./blockfall/scene.ts'), {
    ui: blockfallUi,
  }),
  example('censer', censer, () => import('./censer/scene.ts')),
  example('benthos', benthos, () => import('./benthos/scene.ts')),
  example('dominoes', dominoes, () => import('./dominoes/scene.ts'), {
    ui: dominoesUi,
  }),
  example('falling-sand', fallingSand, () => import('./falling-sand/scene.ts')),
  example('fireworks', fireworks, () => import('./fireworks/scene.ts'), {
    assets: () => import('./fireworks/assets.ts'),
  }),
  example('ember-mirror', emberMirror, () => import('./ember-mirror/scene.ts'), {
    ui: emberMirrorUi,
  }),
  example('flow-field', flowField, () => import('./flow-field/scene.ts')),
  example('forest', forest, () => import('./forest/scene.ts')),
  example('hex-isle', hexIsle, () => import('./hex-isle/scene.ts'), {
    assets: () => import('./hex-isle/assets.ts'),
  }),
  example('jellies', jellies, () => import('./jellies/scene.ts')),
  example('jelly-world', jellyWorld, () => import('./jelly-world/scene.ts'), {
    assets: () => import('./jelly-world/assets.ts'),
  }),
  example('jump-flood', jumpFlood, () => import('./jump-flood/scene.ts')),
  example('kinetic', kinetic, () => import('./kinetic/scene.ts'), {
    assets: () => import('./kinetic/assets.ts'),
  }),
  example('lattice', lattice, () => import('./lattice/scene.ts')),
  example('night-market', nightMarket, () => import('./night-market/scene.ts'), {
    assets: () => import('./night-market/assets.ts'),
  }),
  example('ocean', ocean, () => import('./ocean/scene.ts')),
  example('quiver-trees', quiverTrees, () => import('./quiver-trees/scene.ts'), {
    assets: () => import('./quiver-trees/assets.ts'),
  }),
  example('raymarch', raymarch, () => import('./raymarch/scene.ts')),
  example('robot-arm', robotArm, () => import('./robot-arm/scene.ts')),
  example('skeletons', skeletons, () => import('./skeletons/scene.ts'), {
    assets: () => import('./skeletons/assets.ts'),
  }),
  example('slime-mould', slimeMould, () => import('./slime-mould/scene.ts')),
  example('solar-system', solarSystem, () => import('./solar-system/scene.ts'), {
    assets: () => import('./solar-system/assets.ts'),
  }),
  example('sph-sculpt', sphSculpt, () => import('./sph-sculpt/scene.ts'), {
    ui: sphSculptUi,
  }),
  example('spider', spider, () => import('./spider/scene.ts')),
  example('stampede', stampede, () => import('./stampede/scene.ts'), {
    assets: () => import('./stampede/assets.ts'),
  }),
  example('star-map', starMap, () => import('./star-map/scene.ts'), {
    assets: () => import('./star-map/assets.ts'),
  }),
  example('survey-model', surveyModel, () => import('./survey-model/scene.ts')),
];
// Hermes lacks `toSorted`
list.sort((a, b) => a.order - b.order);

export const examples: readonly Example[] = list;

/** The example `key` names. */
export function exampleOf(key: string): Example {
  const found = examples.find((each) => each.key === key);
  if (!found) throw new Error(`No example ${key}.`);
  return found;
}

/** The examples the galleries show first, the first of them largest. */
export const highlights: readonly Example[] = [
  'jelly-world',
  'classroom',
  'falling-sand',
  'ocean',
  'censer',
].map(exampleOf);
