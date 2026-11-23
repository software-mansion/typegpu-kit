import { bindings } from '@typegpu/kit';
import { census } from '@typegpu/kit/debug';
import { recordingRoot } from '@typegpu/kit/testing';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';

export const OWNERS = ['shadow', 'fog', 'palette', 'tint'] as const;
export type Owner = (typeof OWNERS)[number];

/** The owners each `bindings.share` takes. An owner in none keeps fixed bindings. */
export type Shares = readonly (readonly Owner[])[];

/** A layout for each owner, which is one bind group more than a pipeline has. */
export const OPENING: Shares = OWNERS.map((owner) => [owner]);

/**
 * `shares` with `owner` moved into the group that holds `beside`: the pipeline's own group makes
 * it fixed, an empty group gives it a share of its own, and any other joins that group's share.
 */
export function move(shares: Shares, owner: Owner, beside: readonly (Owner | 'output')[]): Shares {
  const rest = shares.map((owners) => owners.filter((each) => each !== owner));
  if (beside.includes('output')) return rest.filter((owners) => owners.length > 0);
  const into = rest.findIndex((owners) => owners.includes(beside[0] as Owner));
  if (into < 0) return [...rest.filter((owners) => owners.length > 0), [owner]];
  return rest
    .map((owners, i) => (i === into ? [...owners, owner] : owners))
    .filter((owners) => owners.length > 0);
}

/** What the device created to bind a replacement for one of an owner's resources. */
export interface Swap {
  readonly pipelines: number;
  readonly bindGroups: number;
}

export interface Placement {
  /** The owners each of the pipeline's groups holds, by group index; `output` is the pipeline's own. */
  readonly groups: readonly (readonly (Owner | 'output')[])[];
  readonly count: number;
  readonly limit: number;
  /** Whether the census finds the pipeline within the device's bind groups. */
  readonly fits: boolean;
  readonly swaps: Readonly<Record<Owner | 'output', Swap>>;
}

const shadow = bindings.layout({ map: { texture: d.textureDepth2d() } });
const fog = bindings.layout({
  volume: { texture: d.texture3d(d.f32) },
  density: { uniform: d.f32 },
});
const palette = bindings.layout({ colors: { storage: d.arrayOf(d.vec4f), access: 'readonly' } });
const tint = bindings.layout({ tint: { uniform: d.vec4f } });
const output = tgpu.mutableAccessor(d.vec4f);

const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
  'use gpu';
  const lit = std.textureLoad(shadow.$.map, d.vec2i(), 0);
  const haze = std.textureLoad(fog.$.volume, d.vec3i(), 0) * fog.$.density;
  output.$ = haze * palette.$.colors[0] * tint.$.tint * lit;
});

/** A part of `layout` over what `resources` makes, which can make its resources again. */
function own<E extends bindings.Entries>(
  layout: bindings.Layout<E>,
  resources: () => bindings.Resources<E>,
) {
  const part = layout.bind(resources());
  return {
    part,
    /** Swaps in new resources, which throws once a pipeline holds the part as fixed bindings. */
    replace: () => part.replace(resources()),
    /** A part of new resources, for a pipeline of its own. */
    again: () => layout.bind(resources()),
  };
}

/** Each owner's part, its resources named after it so the WGSL says whose a binding is. */
function partsOf(root: TgpuRoot) {
  return {
    shadow: own(shadow, () => ({
      map: root
        .createTexture({ size: [4, 4], format: 'depth16unorm' })
        .$usage('sampled')
        .createView(d.textureDepth2d())
        .$name('shadow'),
    })),
    fog: own(fog, () => ({
      volume: root
        .createTexture({ size: [2, 2, 2], dimension: '3d', format: 'rgba8unorm' })
        .$usage('sampled')
        .createView(d.texture3d(d.f32))
        .$name('fog'),
      density: root.createBuffer(d.f32).$usage('uniform').$name('fog'),
    })),
    palette: own(palette, () => ({
      colors: root.createBuffer(d.arrayOf(d.vec4f, 2)).$usage('storage').$name('palette'),
    })),
    tint: own(tint, () => ({
      tint: root.createBuffer(d.vec4f).$usage('uniform').$name('tint'),
    })),
  };
}

/** The pipeline's bind groups with each owner's bindings where `shares` puts them. */
export function createProgram(shares: Shares = OPENING): Placement {
  const { root, log } = recordingRoot();
  const limit = root.device.limits.maxBindGroups;
  const counted = census.create(root, { limits: { maxBindGroups: limit } });

  const owned = partsOf(root);
  const holder = new Map<Owner, bindings.Shared>();
  for (const owners of shares) {
    const shared = bindings.share(
      root,
      Object.fromEntries(owners.map((owner) => [owner, owned[owner].part])),
    );
    for (const owner of owners) holder.set(owner, shared);
  }

  // Only the modules' own fills, as a renderer pipes them: a shared part follows its owner
  const written = () => root.createMutable(d.vec4f).$name('output');
  const first = written();
  const build = (replaced?: Owner | 'output') =>
    OWNERS.reduce(
      (cfg, owner) =>
        cfg.pipe((owner === replaced ? owned[owner].again() : owned[owner].part).inject()),
      root.with(output, replaced === 'output' ? written() : first),
    ).createComputePipeline({ compute: kernel });

  /** One frame's pass, with every shared group set. */
  const dispatch = (pipeline: ReturnType<typeof build>) =>
    [...new Set(holder.values())]
      .reduce((bound, shared) => bound.with(shared.group()), pipeline)
      .dispatchWorkgroups(1);

  const pipeline = build();
  dispatch(pipeline);

  const named = [...OWNERS, 'output' as const];
  const groups: (Owner | 'output')[][] = [];
  const declarations = log.shaders
    .join('\n')
    .matchAll(/@group\((\d+)\) @binding\(\d+\) var(?:<[^>]+>)? (\w+)/g);
  for (const [, group, name] of declarations) {
    const owner = named.find((each) => name!.startsWith(each))!;
    const holds = (groups[Number(group)] ??= []);
    if (!holds.includes(owner)) holds.push(owner);
  }
  const count = counted.pipelines[0].groups;
  const fits = counted.overBudget.length === 0;

  // What the next frame creates once an owner has a new resource to bind
  const swaps = {} as Record<Owner | 'output', Swap>;
  for (const owner of named) {
    const before = log.counts();
    try {
      // The pipeline's own uniform is fixed, as is a part that a pipeline holds unshared
      if (owner === 'output') throw new Error('A fixed resource has no `replace`.');
      owned[owner].replace();
      dispatch(pipeline);
    } catch {
      dispatch(build(owner));
    }
    const after = log.counts();
    swaps[owner] = {
      pipelines: after.pipelines - before.pipelines,
      bindGroups: after.bindGroups - before.bindGroups,
    };
  }

  return { groups, count, limit, fits, swaps };
}
