import { z } from 'astro/zod';

export const kinds = ['module', 'cpu', 'shader', 'platform', 'tool'] as const;
export const recordings = ['compute', 'render', 'encoder', 'frame'] as const;
export const injections = ['member', 'definition', 'hook'] as const;
export const modes = ['required', 'optional', 'option'] as const;
export const submissions = ['setup', 'always'] as const;
export const platforms = ['web', 'native'] as const;

/** GPU objects whose creation the code shows, as `@owns` names them. */
export const objects = [
  'buffer',
  'texture',
  'sampler',
  'pipeline',
  'query set',
  'bind group',
] as const;

/** The groups a module's value exports sort into, beside its factories, in the order cards list them. */
export const exportGroups = [
  'functions',
  'accessors',
  'outputs',
  'schemas',
  'constants',
  'helpers',
] as const;

export type Kind = (typeof kinds)[number];
export type Recording = (typeof recordings)[number];
export type Injection = (typeof injections)[number];
export type Mode = (typeof modes)[number];
export type Submission = (typeof submissions)[number];
export type Platform = (typeof platforms)[number];
export type GpuObject = (typeof objects)[number];
export type ExportGroup = (typeof exportGroups)[number];

const note = z.string().min(1).optional();
const names = z.array(z.string().min(1));

/** A field of an options object, as the signature shows it. */
const field = z.strictObject({
  name: z.string(),
  optional: z.boolean().optional(),
  /** The module whose instance it holds. */
  module: z.string().optional(),
  /** What the factory does with it, from `@param options.field`. */
  note,
});

/** A factory's parameter, as the signature labels it. */
const parameter = z.strictObject({
  name: z.string(),
  optional: z.boolean().optional(),
  /** The root, which the module borrows. */
  root: z.boolean().optional(),
  /** The module whose instance it is. */
  module: z.string().optional(),
  /**
   * An options object of the module's own: its required fields, and the ones that hold a module or
   * have a note, in order. Absent when none do.
   */
  fields: z.array(field).optional(),
  /** The options object has fields besides `fields`. */
  more: z.boolean().optional(),
  /** What the factory does with it, from its `@param`. */
  note,
});

const factory = z.strictObject({
  name: z.string(),
  /** A short signature, e.g. `create(root, options)`. */
  signature: z.string(),
  parameters: z.array(parameter),
  /** The returned type's name, when the packages declare it: `Particles`, `Promise<Model>`. */
  returns: z.string().optional(),
  /** What the caller gets, from the factory's `@returns`: "a system you destroy". */
  result: note,
});

/** One field of an options type, for its table. */
const option = z.strictObject({
  name: z.string(),
  optional: z.boolean(),
  /** Its type as the source writes it. */
  type: z.string(),
  /** What its TSDoc says it is when left out, from its closing "Default …." sentence. */
  default: note,
  /** The rest of its TSDoc's first sentence. */
  doc: note,
  /** It takes a `'use gpu'` function, which runs in a shader, from the field's `@gpu` tag. */
  gpu: z.literal(true).optional(),
});

/** An options type that factories or methods take, with every field. */
const options = z.strictObject({
  /** The type's name: `Options`, `SurfaceOptions`. */
  name: z.string(),
  /**
   * The factories that take it, or the methods as `Finish.record`, and the parameter: `create`,
   * `options`, or the field of another options type that holds it: `options.shadow`.
   */
  takenBy: names,
  parameter: z.string(),
  fields: z.array(option),
});

/** A member of a handle: a method by its short signature, or a value by its type. */
const member = z.strictObject({
  name: z.string(),
  optional: z.boolean().optional(),
  /** A method's short signature, as factories show theirs: `record(target, view, options?)`. */
  signature: z.string().optional(),
  /** A value's type as the source writes it. */
  type: z.string().optional(),
  /** The first sentence of its TSDoc. */
  doc: note,
});

/** A struct schema a module exports, with its fields in order, each as the code writes it. */
const struct = z.strictObject({
  name: z.string(),
  fields: z.array(z.strictObject({ name: z.string(), type: z.string(), doc: note })),
});

/** A type the packages declare that factories return, with its members in declaration order. */
const handle = z.strictObject({
  /** `Finish`, `Particles`. */
  name: z.string(),
  /** The factories that return it. */
  from: names,
  members: z.array(member),
});

export const moduleContract = z.strictObject({
  /** `<package short name>/<namespace>`: `kit/particles`, `kit/table`, `kit/bloom`. */
  id: z.string().regex(/^[\w-]+\/[\w-]+$/),
  /** `@typegpu/kit` */
  package: z.string(),
  /** `particles`, or the export name for flat exports such as `memo`. */
  namespace: z.string(),
  /** What code imports it from: `@typegpu/kit`, or a subpath such as `@typegpu/kit/web`. */
  specifier: z.string(),
  /** Its file in the package, which ships its source: `src/frames.ts`. */
  file: z.string(),
  /** One line, from the module block or the factory's JSDoc. */
  summary: z.string().min(1),
  /** Exported functions that aren't `'use gpu'`, with short signatures. */
  factories: z.array(factory),
  /** `'use gpu'` functions and TypeGPU functions, in export order. */
  functions: names,
  /** Accessors and slots that shaders read and a pipeline fills: hooks such as `cameras.current`. */
  accessors: names,
  /** What a description of passes writes, such as `passes.shaded`. */
  outputs: names,
  /** WGSL data schemas: `Settings`, `Camera`. */
  schemas: names,
  /** The schemas among them that are structs, with their fields. */
  structs: z.array(struct),
  /** Plain data: numbers, strings, symbols and objects of them, such as `defaults`. */
  constants: names,
  /** Every other value, and functions its overrides list as helpers rather than factories. */
  helpers: names,
  /** Exported types: options, data and callbacks a program names. */
  types: names,
  /** The options types its factories and their handles' methods take. */
  options: z.array(options),
  /** What its factories return. */
  handles: z.array(handle),
  /**
   * module: takes a root and owns GPU resources. cpu: no GPU (`clock`, `coroutines`, `gltf`).
   * shader: `'use gpu'` functions only. platform: web/native glue (`web`, `native`). tool: dev and
   * test tools (`testing`, `gpu-debug`).
   */
  kind: z.enum(kinds),
  /** What its recording methods take, from their parameter types. */
  records: z.array(z.strictObject({ method: z.string(), into: z.enum(recordings) })),
  /** How to call them, from `@records`: "In this order every frame; clear whenever you like." */
  order: note,
  /** setup: submits while setting up (`@submits setup`). always: submits on its own (frames). */
  submits: z.strictObject({ when: z.enum(submissions), note }).optional(),
  /** The kinds of accessors its `inject()` fills, each with what it fills. */
  injects: z.array(z.strictObject({ kind: z.enum(injections), note })),
  /**
   * The buffers a stage binds when its shader reads some of what the module injects or exposes,
   * from `@binds`: `2 storage` for `push`, the points and the rings.
   */
  binds: z.array(
    z.strictObject({
      storage: z.number().int().min(0),
      uniform: z.number().int().min(0),
      note: z.string().min(1),
    }),
  ),
  /** Ids of modules its factories take as arguments. */
  needs: names,
  /** Where its code takes each module it needs, by id: `camera`, `renderer(options.eye)`. */
  via: z.record(z.string(), z.string()),
  /** For a shader module: modules whose accessors its functions read, which the calling pipeline pipes in. */
  reads: names,
  requires: z.array(z.strictObject({ feature: z.string(), mode: z.enum(modes), note })),
  /** The one platform it runs on, from `@platform`; absent, it runs on both. */
  platform: z.enum(platforms).optional(),
  lifecycle: z.strictObject({ initAsync: z.boolean(), destroy: z.boolean(), dispose: z.boolean() }),
  /** What it creates on the GPU, from `@owns`: `5 buffers`, `live lists`, and what they are. */
  owns: z.strictObject({ items: z.array(z.string().min(1)), note }).optional(),
  /** Exceptions to "a steady frame allocates nothing": `on resize`, and what. */
  allocates: z.strictObject({ when: z.string().min(1), note }).optional(),
  /** Example keys (`blockfall`) whose code uses it. */
  usedBy: names,
  /** A copy of an unpublished TypeGPU package, which TypeGPU Kit doesn't edit. */
  vendored: z.boolean().optional(),
  /** May change in any way in a minor release, from `@unstable`; a vendored module always may. */
  unstable: z.boolean().optional(),
  /** Imported by name and called without a namespace: `memo(fn)`. */
  flat: z.boolean().optional(),
  /** A named namespace exported by a subpath entry, such as debug's `gpuDebug`. */
  namespaceExport: z.boolean().optional(),
});

export type ModuleContract = z.infer<typeof moduleContract>;

/** The module with an id; throws on an unknown one. */
export function moduleById<M extends { readonly id: string }>(modules: readonly M[], id: string) {
  const module = modules.find((each) => each.id === id);
  if (!module) throw new Error(`No module ${id}.`);
  return module;
}

export type Factory = ModuleContract['factories'][number];
export type Parameter = Factory['parameters'][number];
export type Field = NonNullable<Parameter['fields']>[number];
export type Options = ModuleContract['options'][number];
export type OptionField = Options['fields'][number];
export type Handle = ModuleContract['handles'][number];
export type Struct = ModuleContract['structs'][number];
export type Member = Handle['members'][number];

/** What a module's tags say, as the contract holds it. */
export type TagFacts = Pick<
  ModuleContract,
  'injects' | 'binds' | 'requires' | 'submits' | 'allocates' | 'owns' | 'order' | 'platform'
> & {
  kind?: Kind | undefined;
  flat: boolean;
  unstable: boolean;
};
