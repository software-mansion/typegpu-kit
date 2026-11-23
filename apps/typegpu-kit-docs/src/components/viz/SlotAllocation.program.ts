import { table, type RowId } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';
import { d } from 'typegpu';
import { seeded } from './svg.ts';

export const CAPACITY = 48;

const Row = table.define('Row', { host: { x: d.f32 } });

/** One way of handing out slots, as the figure draws it. */
export interface Slots {
  readonly live: ReadonlySet<number>;
  /** One past the highest live slot: what a kernel runs over. */
  readonly span: number;
  readonly count: number;
  /** The mean of `count / span` over the samples so far, or undefined before the first. */
  readonly mean: number | undefined;
}

/** The mean of sampled occupancies, from their sum and count. */
const mean = ([sum, n]: number[]) => (n! > 0 ? sum! / n! : undefined);

export function createProgram() {
  const random = seeded(7);
  const rows = table.create(recordingRoot().root, Row, { capacity: CAPACITY });
  const entities: { id: RowId<typeof Row>; lifo: number }[] = [];
  const lifoFree: number[] = [];
  const lifoLive = new Set<number>();
  const samples = { kit: [0, 0], lifo: [0, 0] };
  let lifoNext = 0;
  let tick = 0;

  const lifoSpan = () => {
    let span = 0;
    for (const slot of lifoLive) span = Math.max(span, slot + 1);
    return span;
  };

  const add = (n: number) => {
    for (let i = 0; i < n && entities.length < CAPACITY; i++) {
      const lifo = lifoFree.length > 0 ? lifoFree.pop()! : lifoNext++;
      lifoLive.add(lifo);
      entities.push({ id: rows.add(), lifo });
    }
  };

  /** Removes `n` rows at random, from both. */
  const remove = (n: number) => {
    for (let i = 0; i < n && entities.length > 0; i++) {
      const [entity] = entities.splice(Math.floor(random() * entities.length), 1);
      rows.remove(entity!.id);
      lifoLive.delete(entity!.lifo);
      lifoFree.push(entity!.lifo);
    }
  };

  /** Adds this moment's occupancy to the means. */
  const sample = () => {
    if (rows.span > 0) {
      samples.kit[0] += rows.count / rows.span;
      samples.kit[1] += 1;
    }

    const span = lifoSpan();
    if (span > 0) {
      samples.lifo[0] += lifoLive.size / span;
      samples.lifo[1] += 1;
    }
  };

  return {
    get kit(): Slots {
      return {
        live: new Set(entities.map((entity) => rows.slotOf(entity.id))),
        span: rows.span,
        count: rows.count,
        mean: mean(samples.kit),
      };
    },

    get lifo(): Slots {
      return { live: lifoLive, span: lifoSpan(), count: lifoLive.size, mean: mean(samples.lifo) };
    },

    add,
    remove,
    sample,

    /** A step of the waves: the population swells and shrinks, with deaths at random between. */
    wave() {
      tick += 1;
      const target = 24 + 18 * Math.sin(tick / 9);

      if (entities.length < target) add(1 + Math.floor(random() * 3));
      else remove(1 + Math.floor(random() * 3));

      if (random() < 0.5) {
        remove(1);
        add(1);
      }

      sample();
    },

    destroy() {
      rows.destroy();
    },
  };
}
