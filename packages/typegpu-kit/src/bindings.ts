/**
 * Declares module bindings that an app can combine in one shared layout. Each module can also use
 * its resources without a shared bind group.
 *
 * @module
 * @unstable
 * @injects definition each part's accessors, filled from the shared layout or its own resources
 * @binds 0 storage: parts declare entries without binding resources. Each entry uses one binding
 * in the group that holds it
 * @allocates on change: a bind group from `group()`, after a shared part's resources change
 * @owns a bind group: the shared layout recreates it when a part's resources change
 * @records Set `shared.group()` on each pass that reads a part, or give the layout to
 * `passes.create` as `shared`.
 */

import {
  d,
  tgpu,
  type Configurable,
  type LayoutEntryToInput,
  type TgpuBindGroup,
  type TgpuBindGroupLayout,
  type TgpuLayoutEntry,
  type TgpuRoot,
} from 'typegpu';
import { unfilled, type Inject } from './inject.ts';

export type Entries = Record<string, TgpuLayoutEntry>;

/** Maps each layout entry to its resource, such as a buffer, texture, texture view or sampler. */
export type Resources<E extends Entries> = { readonly [K in keyof E]: LayoutEntryToInput<E[K]> };

/** Provides shader access to a layout's `$` values while shaders resolve. */
export type Values<E extends Entries> = TgpuBindGroupLayout<E>['$'];

/**
 * Declares bindings that another layout can hold, such as one made by `share`. A bound
 * `bindings.layout` implements this interface. Other modules can implement it too, such as the
 * dynamic cache from `@typegpu/noise` with `{ entries: config.layout, resources: cache.bindings,
 * inject: config.inject }`.
 */
export interface Part<E extends Entries = Entries> {
  readonly entries: E;
  /** Provides the resources to bind. `group()` reads them and compares each entry by identity. */
  readonly resources: Resources<E>;
  /** Fills shader accessors from `values`, the `$` of a layout containing this part's entries. */
  inject(values?: Values<E>): Inject;
  /**
   * Checks whether `share` can take this part and throws if it cannot, such as when it already has
   * an owner. `share` checks every part before taking any. A failed check leaves all parts
   * unowned.
   */
  check?(): void;
  /**
   * Receives the layout values from `share`. A part with its own fill must read these values after
   * this call, whichever fill a pipeline receives last.
   */
  ownedBy?(values: Values<E>): void;
}

/** A part of any entries, as `share` takes it. */
// oxlint-disable-next-line typescript/no-explicit-any -- entries type a part's resources and its values, which vary opposite ways
export type AnyPart = Part<any>;

/** Binds one set of resources to the entries from `layout.bind`. */
export interface Bound<E extends Entries> extends Part<E> {
  /**
   * Fills shader accessors from the shared layout after `share` takes this part. Otherwise, it
   * binds the resources in each pipeline's automatic group.
   */
  inject(values?: Values<E>): Inject;
  /**
   * Replaces resources without creating pipelines. The next shared group uses the replacements.
   * Throws if a pipeline already binds the resources directly, because that pipeline would keep
   * the old resources.
   */
  replace(resources: Partial<Resources<E>>): void;
}

/** Configures a part made by `bind`. */
export interface BindOptions {
  /**
   * Makes a pipeline that reads the part throw this message as it builds, unless a
   * `bindings.share` holds the part. Give it to a part whose resources change while the program
   * runs, and name the call that shares it, so a missing share fails before the first change.
   * Default none: the pipeline binds the resources in its automatic group, and `replace` throws
   * after that.
   */
  readonly unshared?: string;
}

/** Declares bindings for module shaders without creating a bind group. */
export interface Layout<E extends Entries> {
  /** Holds the layout entries the layout was made from. */
  readonly entries: E;
  /** Provides shader access to the entries, as a `tgpu.bindGroupLayout`'s `$` does. */
  readonly $: Values<E>;
  /** Binds one set of resources, using the same inputs as `root.createBindGroup`. */
  bind(resources: Resources<E>, options?: BindOptions): Bound<E>;
}

// oxlint-disable-next-line typescript/no-explicit-any -- an accessor of whichever schema the entry has
type AnyAccessor = any;

function accessorOf(entry: TgpuLayoutEntry, key: string): AnyAccessor {
  const empty = unfilled(`\`${key}\``, 'part', "its `bindings.layout`'s `bind()`");

  if ('uniform' in entry) return tgpu.accessor(entry.uniform as d.AnyData, empty);

  if ('storage' in entry) {
    const schema = entry.storage as d.AnyData;
    return entry.access === 'mutable'
      ? tgpu.mutableAccessor(schema, empty)
      : tgpu.accessor(schema, empty);
  }

  if ('texture' in entry) return tgpu.accessor(entry.texture, empty);
  if ('storageTexture' in entry) return tgpu.accessor(entry.storageTexture, empty);
  if ('externalTexture' in entry) return tgpu.accessor(entry.externalTexture, empty);
  if ('sampler' in entry) {
    const schema = entry.sampler === 'comparison' ? d.comparisonSampler() : d.sampler();
    return tgpu.accessor(schema, empty);
  }

  throw new Error(
    'Bindings take uniforms, storage, textures, storage textures, external textures and samplers.',
  );
}

const kindOf = (value: unknown) =>
  typeof value === 'object' && value !== null && 'resourceType' in value
    ? value.resourceType
    : undefined;

/**
 * Declares bindings for module shaders from the entries that `tgpu.bindGroupLayout` takes. Define
 * the layout once beside the shaders. A part made with `bind` can bind its resources directly.
 * Call `share` before creating pipelines to put the part in a shared layout.
 *
 * @param entries the bindings, as `tgpu.bindGroupLayout` takes them
 * @returns a layout whose `bind` makes a part from resources
 */
export function layout<const E extends Entries>(entries: E): Layout<E> {
  const keys = Object.keys(entries) as (keyof E & string)[];
  const accessors: Record<string, AnyAccessor> = {};
  for (const key of keys) accessors[key] = accessorOf(entries[key], key).$name(key);

  const $ = {} as Values<E>;
  for (const key of keys) Object.defineProperty($, key, { get: () => accessors[key].$ });

  const fill = (cfg: Configurable, value: (key: keyof E & string) => unknown) =>
    keys.reduce((next, key) => next.with(accessors[key], value(key)), cfg);

  return {
    entries,
    $,

    bind(initial, { unshared } = {}) {
      let resources = initial;
      let owner: Values<E> | undefined;
      let fixed = false;
      const views = new WeakMap<object, unknown>();

      const fixedValue = (key: keyof E & string) => {
        const entry = entries[key] as TgpuLayoutEntry;
        const resource = resources[key] as object;
        const kind = kindOf(resource);

        if (kind === 'buffer') {
          const usage =
            'uniform' in entry ? 'uniform' : 'access' in entry ? entry.access : 'readonly';
          return (resource as { as(usage: string): unknown }).as(usage ?? 'readonly');
        }

        if (kind === 'texture') {
          if (!views.has(resource)) {
            const schema =
              'texture' in entry
                ? entry.texture
                : (entry as { storageTexture: unknown }).storageTexture;
            views.set(
              resource,
              (resource as { createView(schema: unknown): unknown }).createView(schema),
            );
          }
          return views.get(resource);
        }

        if (kind === undefined) {
          throw new Error(
            `'${key}' is a WebGPU object, which only a bind group takes. Bind a TypeGPU resource, or share the part.`,
          );
        }

        return (resource as { $: unknown }).$;
      };

      const check = () => {
        if (owner) {
          throw new Error(
            'This part already has an owner. Call `bindings.share` once, and give its result to every description that reads the part.',
          );
        }
        if (fixed) {
          throw new Error(
            'A pipeline already holds this part as fixed bindings. Share it before creating the pipelines that read it.',
          );
        }
      };

      return {
        entries,

        get resources() {
          return resources;
        },

        inject(values) {
          // Resolve the owner when the pipeline resolves, even if the fill was piped before
          // sharing
          return (cfg) =>
            fill(cfg, (key) => () => {
              const source = values ?? owner;
              if (source) return source[key];
              if (unshared !== undefined) throw new Error(unshared);

              fixed = true;
              return fixedValue(key);
            });
        },

        check,

        ownedBy(values) {
          check();
          owner = values;
        },

        replace(changed) {
          if (fixed) {
            throw new Error(
              'A pipeline holds this part as fixed bindings and would keep the old resources. Take it into a layout you own with `bindings.share` before creating pipelines.',
            );
          }
          resources = { ...resources, ...changed };
        },
      };
    },
  };
}

/** Combines parts in one layout that you bind on each pass. */
export interface Shared {
  /**
   * Contains every part's entries. Each entry is named `key_name` using the part's key in `share`.
   */
  readonly layout: TgpuBindGroupLayout;
  /**
   * Fills shader accessors for every part from the shared layout. Parts made with
   * `bindings.layout` use this layout even without this fill.
   */
  inject(): Inject;
  /**
   * Returns the group for the current resources, for `pass.setBindGroup(shared.layout, …)`.
   * Creates a group only when a resource has changed. Otherwise, it allocates nothing.
   */
  group(): TgpuBindGroup;
}

/**
 * Combines parts from modules or your own code in one layout. Call it before creating pipelines.
 * Set its group on each pass with `pass.setBindGroup(shared.layout, shared.group())`, or give it to
 * `passes.create` as `shared`, which sets it in every pass it begins. Every pipeline in that pass
 * that reads a part then uses the shared group. A dispatch or draw without the group throws
 * `Missing bind groups for layouts`.
 *
 * @param parts parts from any modules and your own, by key
 * @returns a layout you set on each pass, with the group for the parts' resources
 */
export function share(root: TgpuRoot, parts: Record<string, AnyPart>): Shared {
  const named = Object.entries(parts);
  const names = named.map(([, part]) => Object.keys(part.entries));

  if (new Set(Object.values(parts)).size < named.length) {
    throw new Error('A part is in one share twice. Give each part one key.');
  }
  for (const [, part] of named) part.check?.();

  const entries: Entries = {};
  named.forEach(([key, part], i) => {
    for (const name of names[i]) {
      const merged = `${key}_${name}`;
      if (merged in entries) {
        throw new Error(`Two parts would both be '${merged}' in the shared layout. Rename a key.`);
      }
      entries[merged] = part.entries[name];
    }
  });

  const owned = tgpu.bindGroupLayout(entries).$name('shared');

  // Parts keep their entry names while resolving through the shared layout
  const values = named.map(([key], i) => {
    const own = {};
    for (const name of names[i]) {
      Object.defineProperty(own, name, { get: () => owned.$[`${key}_${name}`] });
    }
    return own as Values<Entries>;
  });
  named.forEach(([, part], i) => part.ownedBy?.(values[i]));

  const inject: Inject = (cfg) =>
    named.reduce((next, [, part], i) => next.pipe(part.inject(values[i])), cfg);

  // Compare resources by identity in entry order without allocating each frame
  const last: unknown[] = [];
  let group: TgpuBindGroup | undefined;

  const changed = () => {
    let at = 0;
    for (let i = 0; i < named.length; i++) {
      const { resources } = named[i][1];
      for (const name of names[i]) {
        if (resources[name] !== last[at++]) return true;
      }
    }
    return false;
  };

  return {
    layout: owned,
    inject: () => inject,

    group() {
      if (group && !changed()) return group;

      // The input type of the entry union leaves out a mutable buffer
      const flat: Record<string, unknown> = {};
      let at = 0;
      named.forEach(([key, part], i) => {
        for (const name of names[i]) {
          flat[`${key}_${name}`] = part.resources[name];
          last[at++] = part.resources[name];
        }
      });

      group = root.createBindGroup(owned, flat as Resources<Entries>);
      return group;
    },
  };
}
