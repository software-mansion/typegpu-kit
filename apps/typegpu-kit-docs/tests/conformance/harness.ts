import type { Frame, Surface } from '@typegpu/kit';
import type { RecordingCounts } from '@typegpu/kit/testing';
import type { TgpuRoot } from 'typegpu';
import type { GpuObject, ModuleContract } from '../../src/contracts/contract.ts';
import { readOwns } from '../../src/contracts/owns.ts';

/** One instance of a module, made by its fixture, and a frame that uses it. */
export interface Instance {
  /** Records what a program would each frame, through its recording methods. */
  frame(frame: Frame): void;
  /** Frees the module, for one that has `destroy`. */
  destroy?(): void;
}

/**
 * Runs `make`, and leaves what it creates out of the module's counts: what another module owns,
 * such as the pipeline a description builds in `add`, which `drawn.initAsync()` compiles.
 */
export type Others = <T>(make: () => T) => T;

type Made = { create(others: Others): Instance | Promise<Instance> };

/**
 * How to make a module for its contract's checks. `setup` makes what the module needs, which the
 * counts leave out; the instance `create` returns is what they count.
 */
export interface Fixture {
  /** Device features the recording root reports. Default none. */
  readonly features?: readonly string[];
  setup(root: TgpuRoot, surface: Surface): Made | Promise<Made>;
}

/** A module the checks can't build, and why. */
export interface Skip {
  readonly skip: string;
}

export type Fixtures = Record<string, Fixture | Skip>;

/** The GPU objects `@owns` counts, as the recording log names them. */
export const counted = [
  'buffers',
  'textures',
  'samplers',
  'querySets',
  'pipelines',
  'bindGroups',
] as const;
export type Counted = (typeof counted)[number];

/** How many of an object a module may own: at least `min`, at most `max`, unbounded without. */
export interface Range {
  min: number;
  max?: number;
}

/** What `@owns` says a module creates, by object, and the items it can't count. */
export interface Owned {
  readonly ranges: Partial<Record<Counted, Range>>;
  readonly vague: string[];
}

/** The counts' key for each object `@owns` names. */
const keys: Record<GpuObject, Counted> = {
  buffer: 'buffers',
  texture: 'textures',
  sampler: 'samplers',
  'query set': 'querySets',
  pipeline: 'pipelines',
  'bind group': 'bindGroups',
};

const add = (ranges: Partial<Record<Counted, Range>>, key: Counted, range: Range) => {
  const was = ranges[key] ?? { min: 0, max: 0 };
  const max = was.max === undefined || range.max === undefined ? undefined : was.max + range.max;
  ranges[key] =
    max === undefined ? { min: was.min + range.min } : { min: was.min + range.min, max };
};

/**
 * Counts what `@owns` says, as `readOwns` reads it. A part that names a module, `a readback`, adds
 * that module's `@owns`, and a condition after it makes what that module owns optional.
 */
export function ownedBy(contract: ModuleContract, contracts: readonly ModuleContract[]): Owned {
  const ranges: Partial<Record<Counted, Range>> = {};
  const vague: string[] = [];

  for (const part of readOwns(contract.owns?.items ?? [])) {
    const inner = part.module && contracts.find((each) => each.namespace === part.module);
    if (part.object) {
      const { min, max } = part;
      add(ranges, keys[part.object], max === undefined ? { min } : { min, max });
    } else if (inner) {
      const owned = ownedBy(inner, contracts);
      for (const [key, range] of Object.entries(owned.ranges) as [Counted, Range][]) {
        add(ranges, key, part.optional ? { ...range, min: 0 } : range);
      }
      vague.push(...owned.vague);
    } else {
      vague.push(part.text);
    }
  }

  return { ranges, vague: [...new Set(vague)] };
}

/** Each object a module created beyond its claim, as `buffers: 4, owns 3`. */
export function overClaim(
  owned: Owned,
  made: Partial<RecordingCounts>,
  checked: readonly Counted[],
): string[] {
  return checked.flatMap((key) => {
    const n = made[key] ?? 0;
    const range = owned.ranges[key] ?? { min: 0, max: 0 };
    const fits = n >= range.min && (range.max === undefined || n <= range.max);
    if (fits) return [];
    const claim =
      range.max === undefined
        ? `${range.min}+`
        : range.min === range.max
          ? range.min
          : `${range.min}–${range.max}`;
    return [`${key}: made ${n}, owns ${claim}`];
  });
}

export const delta = (after: RecordingCounts, before: RecordingCounts): RecordingCounts =>
  Object.fromEntries(
    Object.entries(after).map(([key, n]) => [key, n - before[key as keyof RecordingCounts]]),
  ) as unknown as RecordingCounts;
