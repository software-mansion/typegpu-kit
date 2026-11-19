// Where each card hides in the docs: one place an example, on a page about something it uses, in
// a sealed box or foil pack. The first ones, the box on the home page and the pack on the examples
// page, give a random common card. Imports nothing.

/** A card's hiding place, and the hints toward it, vaguest first. */
export interface Hiding {
  /** The page, written from the root, and its title. */
  readonly page: string;
  readonly title: string;
  /** The section it waits in, by its heading's id, if not the top of the page. */
  readonly section?: string;
  /** What the page is about, in a few words: the first hint. */
  readonly subject: string;
}

/** Each card's page, with the section after `#` if not its top, its title and its subject. */
const ROWS: Readonly<Record<string, readonly [string, string, string]>> = {
  afterimage: ['/kit/finish/', 'Final pass', 'the last pass to the screen'],
  benthos: ['/kit/clock/', 'Clock', 'keeping time'],
  blockfall: ['/kit/table/#removing-rows', 'Tables', 'rows that come and go'],
  censer: ['/kit/targets/', 'Render targets', 'textures that share a size'],
  classroom: ['/kit/history/#a-buffers-history', 'History', "last frame's state"],
  cloister: ['/kit/bindings/', 'Shared bindings', 'resources that modules share'],
  dominoes: [
    '/tutorials/domino-run/#going-further-with-a-portrait',
    'Domino run',
    'a long chain falling',
  ],
  'ember-mirror': ['/kit/video/', 'Camera and video', 'frames from a camera'],
  'falling-sand': ['/kit/counters/', 'Counters', 'counting what kernels saw'],
  fireworks: ['/kit/trails/', 'Trails', 'what moving things leave behind'],
  fishtank: ['/kit/grid/', 'Neighbour grid', 'finding neighbours'],
  'flow-field': ['/kit/pipelines/', 'Pipeline warm-up', 'compiling ahead of time'],
  forest: ['/kit/instances/', 'Instances', 'many copies of one mesh'],
  'hex-isle': ['/kit/environment/', 'Environment light', 'light from all around'],
  jellies: ['/kit/particles/', 'Particles', 'many small things at once'],
  'jelly-world': ['/kit/profiler/', 'Profiler', 'timing the GPU'],
  'jump-flood': ['/tutorials/write-a-module/', 'Write a module', 'writing your own module'],
  kinetic: ['/kit/images/', 'Textures from images', 'pictures made into textures'],
  lattice: ['/kit/taa/', 'Temporal antialiasing', 'smoothing over several frames'],
  'night-market': ['/kit/dof/', 'Depth of field', 'what the focus leaves soft'],
  ocean: ['/kit/cameras/', 'Cameras', 'where the eye looks from'],
  platformer: ['/kit/live-lists/', 'Live lists', 'who is still alive'],
  'quiver-trees': ['/kit/impostors/', 'Impostors', 'models as flat cards'],
  raymarch: ['/kit/fxaa/', 'FXAA', 'smoothing in one pass'],
  'robot-arm': ['/kit/lighting/#shadows', 'Lighting', 'where the light does not reach'],
  skeletons: ['/kit/gltf/', 'glTF models', 'models from files'],
  'slime-mould': ['/tutorials/slime-mould/', 'Slime mould', 'growing something from nothing'],
  'solar-system': ['/kit/text/', 'Text', 'words on the GPU'],
  'sph-sculpt': ['/kit/readback/', 'Readback', 'answers back from the GPU'],
  spider: ['/kit/lighting/', 'Lighting', 'lamps, the sun and the sky'],
  stampede: ['/kit/machines/', 'State machines', 'states and the ways between them'],
  'star-map': ['/kit/picker/', 'Picker', 'what is under the pointer'],
  'survey-model': ['/kit/lines/', 'Lines', 'strokes with joins and caps'],
};

/** Every card's hiding place, by example key. */
export const HIDDEN: Readonly<Record<string, Hiding>> = Object.fromEntries(
  Object.entries(ROWS).map(([key, [at, title, subject]]) => {
    const [page, section] = at.split('#') as [string, string?];
    return [key, section ? { page, title, section, subject } : { page, title, subject }];
  }),
);
