// The words the docs use in a sense of their own. The build marks each one where a page first uses
// it (build/terms.ts), and the Words section of the reading page lists them all (Words.astro).

import { READING } from './urls.ts';

export interface Term {
  /** The word as prose writes it, in the singular. */
  readonly term: string;
  /** Its plural and the other forms that count as a use. */
  readonly also?: readonly string[];
  /**
   * `false` for a word that prose also uses in its everyday sense, as it does `part`: the word
   * alone is then no use, and only the phrases in `also` are.
   */
  readonly alone?: false;
  /** What it means, in one sentence. */
  readonly means: string;
  /**
   * The section that teaches it, as a link from the root, or its row in the Words section when no
   * page does.
   */
  readonly home: string;
}

/** A term's name in an id: `pass-description`. */
const termSlug = ({ term }: Term) => term.toLowerCase().replace(/[^a-z0-9]+/g, '-');

/** The id of a term's popover on a page that uses it. */
export const termId = (term: Term) => `term-${termSlug(term)}`;

/** The id of a term's row in the Words section. */
export const wordId = (term: Term) => `word-${termSlug(term)}`;

/** Whether a term's home is its own row in the Words section, where no page teaches it. */
export const inWords = (term: Term) => term.home === `/${READING}/#${wordId(term)}`;

/** Every term, in alphabetical order. */
export const terms: readonly Term[] = [
  {
    term: 'accessor',
    also: ['accessors'],
    means:
      'An accessor is a typed value that shader code reads and that each pipeline can fill with a different resource.',
    home: '/concepts/how-the-kit-works/#modules-connect-through-typegpu',
  },
  {
    term: 'atlas',
    also: ['atlases'],
    means:
      'An atlas is one texture that holds many small pictures side by side, such as every glyph of a font or every view of a model.',
    home: '/how-to-read/#word-atlas',
  },
  {
    term: 'bind group',
    also: ['bind groups'],
    means:
      'A bind group is a set of buffers, textures and samplers that a pipeline reads, and a pipeline has room for four.',
    home: '/how-to-read/#word-bind-group',
  },
  {
    term: 'binding',
    also: ['bindings'],
    means:
      'A binding is one buffer, texture or sampler in a bind group, which a shader reads by its place there.',
    home: '/how-to-read/#word-binding',
  },
  {
    term: 'cadence',
    also: ['cadences'],
    means: 'A cadence tells you on which frames work that runs less often than every frame is due.',
    home: '/kit/clock/#cadences',
  },
  {
    term: 'capture',
    also: ['a capture', 'each capture'],
    alone: false,
    means:
      'A capture copies a buffer at one point in the frame, so the CPU can read it after the submit.',
    home: '/kit/readback/#placing-the-capture',
  },
  {
    term: 'compound',
    also: ['compounds', 'compound drawing', 'compound drawings'],
    means:
      'A compound is a drawing made of several pipelines, with resources of its own for each view it draws into.',
    home: '/kit/passes/#compound-drawings',
  },
  {
    term: 'definition',
    also: ['definitions'],
    means:
      "A definition names a table's fields before any table exists, so shader code written over it works with every table made from it.",
    home: '/concepts/connecting-modules/#members-and-placeholders',
  },
  {
    term: 'dispatch',
    also: ['dispatches'],
    means: 'A dispatch runs a kernel once over a number of threads, inside a compute pass.',
    home: '/how-to-read/#word-dispatch',
  },
  {
    term: 'drawing',
    also: ['drawings', 'a drawing', 'one drawing', 'translucent drawing', "drawing's"],
    alone: false,
    means:
      'A drawing is what a renderer hands to a pass description: what each vertex carries to its pixels, what a pixel looks like and the draw call.',
    home: '/concepts/scenes-and-passes/#every-renderer-is-a-drawing',
  },
  {
    term: 'factory',
    also: ['factories'],
    means:
      'A factory is the function that creates a module, which allocates what the module needs and returns its resources, functions and recording methods.',
    home: '/concepts/how-the-kit-works/#a-module-borrows',
  },
  {
    term: 'fill',
    also: [
      'a fill',
      'no fill',
      'its fill',
      'its fills',
      'that fill',
      'missing fills',
      'carry fills',
      'temporal fills',
    ],
    alone: false,
    means: 'A fill is the resource a pipeline binds for an accessor its shaders read.',
    home: '/concepts/connecting-modules/#members-and-placeholders',
  },
  {
    term: 'fixed step',
    also: ['fixed steps'],
    means:
      'A fixed step advances a simulation by one set length of time, and each frame runs a whole number of them, so the result is the same at any frame rate.',
    home: '/concepts/data-on-the-gpu/#a-fixed-step',
  },
  {
    term: 'flush',
    means: 'A flush uploads the rows the CPU changed since the last one.',
    home: '/concepts/data-on-the-gpu/#rows',
  },
  {
    term: 'frame',
    also: ['frames'],
    means:
      'A frame is one turn of the loop, and the object your function gets for it, which holds the encoder, the time and the time since the last frame.',
    home: '/concepts/how-the-kit-works/#you-own-the-frame',
  },
  {
    term: 'HDR',
    also: ['high dynamic range'],
    means:
      'HDR, high dynamic range, is colour stored with more range than a screen shows, so a light can be brighter than white.',
    home: '/concepts/how-the-kit-works/#a-module-borrows',
  },
  {
    term: 'hook',
    also: ['hooks'],
    means:
      'A hook is a placeholder that each pipeline fills with its own choice, such as the camera it draws through.',
    home: '/concepts/connecting-modules/#members-and-placeholders',
  },
  {
    term: 'host',
    also: ['hosts'],
    means:
      'A host runs a scene for you, which means it makes the device, the surface and the frame loop and frees them in order.',
    home: '/kit/hosting/',
  },
  {
    term: 'impostor',
    also: ['impostors'],
    means:
      'An impostor draws a distant model as one card that faces the camera, textured with pictures of the model taken in advance.',
    home: '/kit/impostors/',
  },
  {
    term: 'indirect draw',
    also: ['indirect draws'],
    means:
      'An indirect draw reads how much to draw from a buffer, so a kernel can decide the count and the CPU never reads it.',
    home: '/kit/passes/#indirect-draws',
  },
  {
    term: 'inject',
    means:
      "A module's inject() returns the fills a pipeline needs before its shaders can read the module's placeholders.",
    home: '/concepts/how-the-kit-works/#modules-connect-through-typegpu',
  },
  {
    term: 'instance',
    also: ['instances'],
    means:
      'An instance is one copy of a mesh with its own position, rotation, scale and material, kept as a row of a table.',
    home: '/kit/instances/',
  },
  {
    term: 'kernel',
    also: ['kernels'],
    means:
      'A kernel is a compute function that the GPU runs many times at once, each run on its own thread.',
    home: '/concepts/how-the-kit-works/#a-frame-in-plain-typegpu',
  },
  {
    term: 'kind',
    also: ['instance kind', 'instance kinds', 'its kind', "a kind's", "the kind's"],
    alone: false,
    means:
      'A kind adds fields to the rows of a set of instances and replaces functions in the shaders that draw them.',
    home: '/kit/instances/#kinds',
  },
  {
    term: 'linear light',
    also: ['linear colour', 'linear colours'],
    means:
      'Linear light is colour as amounts of light, the form in which lighting adds up correctly, before the final pass encodes it for the screen.',
    home: '/how-to-read/#word-linear-light',
  },
  {
    term: 'live list',
    also: ['live lists'],
    means:
      'A live list is a list of the live rows of one group, made on the GPU each frame so that a draw covers only them.',
    home: '/concepts/data-on-the-gpu/#only-what-is-alive',
  },
  {
    term: 'member',
    also: ['members'],
    means:
      "A member is a function or accessor on what a factory returns that reads that instance's own resources, so a pipeline that uses it needs no fill.",
    home: '/concepts/connecting-modules/#members-and-placeholders',
  },
  {
    term: 'mesh pool',
    also: ['mesh pools'],
    means:
      "A mesh pool stores every mesh's vertices in shared buffers, so one pipeline can draw any mesh.",
    home: '/kit/mesh-pool/',
  },
  {
    term: 'module',
    also: ['modules'],
    means:
      'A module is one piece of GPU work, made by a factory that borrows your root and returns resources, shader functions and methods that record into your frame.',
    home: '/concepts/how-the-kit-works/#a-module-borrows',
  },
  {
    term: 'output',
    also: ['outputs', 'motion output', 'every output', 'each output', 'other output'],
    alone: false,
    means:
      'An output is a value every drawing writes at each pixel beside its colour, such as motion or an id.',
    home: '/concepts/scenes-and-passes/#outputs-every-drawing-writes',
  },
  {
    term: 'part',
    also: ['shared part', 'shared parts', 'its part'],
    alone: false,
    means:
      'A part is a set of resources that can be replaced, such as a texture that follows the screen size, which pipelines read through a layout they share.',
    home: '/concepts/connecting-modules/#resources-that-are-replaced',
  },
  {
    term: 'pass description',
    also: [
      'pass descriptions',
      'description of passes',
      'descriptions of passes',
      'description',
      'descriptions',
    ],
    means:
      'A pass description says once which textures a pass draws into and what each holds, and builds the pipeline for every drawing added to it.',
    home: '/concepts/scenes-and-passes/',
  },
  {
    term: 'patch',
    alone: false,
    means: 'A patch uploads only the fields you pass and leaves the others as they are.',
    home: '/concepts/data-on-the-gpu/#settings-that-change',
  },
  {
    term: 'placeholder',
    also: ['placeholders'],
    means:
      'A placeholder is an accessor with no resource of its own, which each pipeline fills through the inject() of the module it reads.',
    home: '/concepts/connecting-modules/#members-and-placeholders',
  },
  {
    term: 'readback',
    also: ['readbacks'],
    means:
      'A readback copies a buffer during a frame and hands it to the CPU a frame or two later, without making the frame wait.',
    home: '/concepts/data-on-the-gpu/#results-back',
  },
  {
    term: 'recorder',
    also: ['recorders'],
    means:
      'The recorder is the function you give the frame loop, which records one frame into the encoder it is handed.',
    home: '/how-to-read/#word-recorder',
  },
  {
    term: 'recording method',
    also: ['recording methods'],
    means:
      "A recording method records a module's work into the pass, the encoder or the frame you hand it, and never submits.",
    home: '/concepts/modules/#recording-methods',
  },
  {
    term: 'renderer',
    also: ['renderers'],
    means:
      'A renderer draws one sort of thing, such as instances or particles, by handing a pass description a drawing.',
    home: '/concepts/scenes-and-passes/#every-renderer-is-a-drawing',
  },
  {
    term: 'row',
    also: ['rows'],
    means:
      "A row is one object's values in a buffer, such as one enemy, one particle or one light.",
    home: '/concepts/how-the-kit-works/#objects-are-rows-in-buffers',
  },
  {
    term: 'row id',
    also: ['row ids'],
    means:
      'A row id names one row of a table for as long as the row lives, and stops working once the row is removed.',
    home: '/kit/table/#row-ids',
  },
  {
    term: 'runtime',
    also: ['runtimes'],
    means:
      'A runtime is one JavaScript thread with its own memory, such as the page, a worker or a React Native worklet thread.',
    home: '/how-to-read/#word-runtime',
  },
  {
    term: 'sampler',
    also: ['samplers'],
    means:
      'A sampler tells the GPU how to read a texture between its texels and past its edges, such as smoothly or in blocks.',
    home: '/how-to-read/#word-sampler',
  },
  {
    term: 'scene',
    also: ['hosted scene', 'hosted scenes', 'scene file', 'scene files'],
    alone: false,
    means:
      'A scene is the function a scene file exports, which creates its modules and returns the function that records each frame.',
    home: '/kit/hosting/',
  },
  {
    term: 'slot',
    also: ['slots'],
    means:
      "A slot is a row's place in its buffers, which a new row can take once the first is removed.",
    home: '/kit/table/#row-ids',
  },
  {
    term: 'sRGB',
    means:
      'sRGB is the encoding screens and most image files use for colour, which a shader turns into linear light before it does maths on it.',
    home: '/how-to-read/#word-srgb',
  },
  {
    term: 'staging buffer',
    also: ['staging buffers'],
    means:
      'A staging buffer is a buffer the CPU can map, which a result is copied into so that it can be read after the frame.',
    home: '/how-to-read/#word-staging-buffer',
  },
  {
    term: 'steady frame',
    also: ['steady frames'],
    means:
      'A steady frame is any frame after setup, in which a module only records and creates no GPU objects.',
    home: '/concepts/how-the-kit-works/#nothing-is-hidden',
  },
  {
    term: 'storage buffer',
    also: ['storage buffers'],
    means:
      'A storage buffer is GPU memory that shaders read by index and kernels can write, which is where the kit keeps rows.',
    home: '/how-to-read/#word-storage-buffer',
  },
  {
    term: 'surface',
    also: ['canvas surface', "the surface's"],
    alone: false,
    means:
      'A surface is a canvas set up for your root, which keeps its drawing buffer the size the page shows it at.',
    home: '/kit/web/',
  },
  {
    term: 'table',
    also: ['tables'],
    means:
      'A table is a buffer of rows with named fields, some written by the CPU and the others by kernels.',
    home: '/concepts/data-on-the-gpu/#rows',
  },
  {
    term: 'target',
    also: ['targets', 'render target', 'render targets', 'target set', 'target sets', 'kit target'],
    alone: false,
    means: 'A target is a texture that a frame draws into.',
    home: '/concepts/scenes-and-passes/',
  },
  {
    term: 'texel',
    also: ['texels'],
    means: 'A texel is one cell of a texture, as a pixel is one cell of the screen.',
    home: '/how-to-read/#word-texel',
  },
  {
    term: 'thread',
    also: ['threads'],
    means:
      'A thread is one run of a kernel, and the GPU starts many at once, each with its own number.',
    home: '/tutorials/slime-mould/',
  },
  {
    term: 'tone curve',
    also: ['tone curves'],
    means:
      'A tone curve maps light of any brightness into the range a screen can show, keeping detail in the bright parts.',
    home: '/how-to-read/#word-tone-curve',
  },
  {
    term: 'workgroup',
    also: ['workgroups'],
    means:
      'A workgroup is a block of threads that the GPU starts together, and a dispatch runs as many workgroups as cover the work.',
    home: '/tutorials/slime-mould/',
  },
  {
    term: 'worklet',
    also: ['worklets'],
    means:
      'A worklet is a function that react-native-worklets can run on a thread other than the main JavaScript thread.',
    home: '/how-to-read/#word-worklet',
  },
];

/**
 * Phrases that hold a term's word in another sense, such as a JavaScript thread beside a kernel's.
 * They take no mark, since the longest phrase at a place is the one read.
 */
export const plain: readonly string[] = [
  'across threads',
  'at runtime',
  'calling thread',
  'camera frame',
  'camera frames',
  'camera thread',
  'Dawn instance',
  'decoder module',
  'holds the thread',
  'host fields',
  'host part',
  'host rows',
  "instance's members",
  'JS thread',
  'last row',
  'lookup table',
  'main thread',
  'members of instances',
  'morph targets',
  'on any thread',
  'rows from the top',
  'shader module',
  'table above',
  'thread that created',
  'two threads',
  'UI thread',
  'unpacks accessors',
];
