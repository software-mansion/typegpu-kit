import { table, type RowId } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';
import { d } from 'typegpu';
import { seeded } from './svg.ts';

export const SLOTS = 12;

const Enemy = table.define('Enemy', {
  host: { tint: d.vec4f },
  device: { position: d.vec3f, health: d.f32 },
});

export type Field = 'tint' | 'position' | 'health';
const ALL: readonly Field[] = ['tint', 'position', 'health'];

const HOST_STRIDE = d.sizeOf(d.struct({ tint: d.vec4f }));
const Device = d.struct({ position: d.vec3f, health: d.f32 });
const STRIDE = d.sizeOf(Device);
const FIELDS = (['position', 'health'] as const).map((field) => {
  const { offset } = d.memoryLayoutOf(Device, (row) => row[field]);
  return { field, start: offset, end: offset + d.sizeOf(Device.propTypes[field]) };
});

const range = (first: number, last: number) =>
  first === last ? `slot ${first}` : `slots ${first} to ${last}`;

/** One upload of the last flush: the part, the slots, the fields and the bytes. */
export type Upload = [string, string, string, string];

export function createProgram() {
  const random = seeded(11);
  const recording = recordingRoot();
  const gpu = recording.root;
  const enemies = table.create(gpu, Enemy, { capacity: SLOTS });
  const buffers = {
    host: gpu.unwrap(enemies.buffers.host!),
    device: gpu.unwrap(enemies.buffers.device!),
    live: gpu.unwrap(enemies.buffers.live),
    span: gpu.unwrap(enemies.buffers.index.buffer),
  };

  const ids: (RowId<typeof Enemy> | undefined)[] = Array.from({ length: SLOTS }, () => undefined);
  // `slot:field`, written since the last flush
  let changed = new Set<string>();
  let added = new Set<number>();
  // `slot:field`, and `live` or `span`, from the last flush
  let uploaded = new Set<string>();
  let uploads: Upload[] = [];

  const flush = () => {
    const from = recording.log.writes.length;
    enemies.flush();

    uploaded = new Set();
    uploads = [];

    for (const { buffer, offset, size } of recording.log.writes.slice(from)) {
      const bytes = `${size} B`;

      if (buffer === buffers.host) {
        const first = offset / HOST_STRIDE;
        const last = (offset + size) / HOST_STRIDE - 1;
        for (let slot = first; slot <= last; slot++) uploaded.add(`${slot}:tint`);
        uploads.push(['host', range(first, last), '', bytes]);
      }

      if (buffer === buffers.device) {
        const first = Math.floor(offset / STRIDE);
        const last = Math.ceil((offset + size) / STRIDE) - 1;
        const covered = new Set<string>();

        for (let slot = first; slot <= last; slot++) {
          for (const { field, start, end } of FIELDS) {
            const row = slot * STRIDE;
            if (offset < row + end && offset + size > row + start) {
              uploaded.add(`${slot}:${field}`);
              covered.add(field);
            }
          }
        }

        const fresh = added.has(first);
        uploads.push([
          'device',
          range(first, last),
          fresh ? 'new' : [...covered].join(', '),
          bytes,
        ]);
      }

      if (buffer === buffers.live) {
        uploaded.add('live');
        uploads.push(['live', `word ${offset / 4}`, '', bytes]);
      }

      if (buffer === buffers.span) {
        uploaded.add('span');
        uploads.push(['span', String(enemies.span), '', bytes]);
      }
    }

    changed = new Set();
    added = new Set();
  };

  return {
    get span() {
      return enemies.span;
    },

    alive: (slot: number) => ids[slot] !== undefined,
    /** Whether `slot:field` was written since the last flush. */
    changed: (key: string) => changed.has(key),
    /** Whether the last flush uploaded `slot:field`, `live` or `span`. */
    uploaded: (key: string) => uploaded.has(key),

    /** What the last flush uploaded, one line a write. */
    get uploads(): readonly Upload[] {
      return uploads;
    },

    /** Adds a row, unless every slot holds one. */
    add() {
      if (enemies.count === SLOTS) return;
      const id = enemies.add();
      const slot = enemies.slotOf(id);
      ids[slot] = id;
      added.add(slot);
      for (const field of ALL) changed.add(`${slot}:${field}`);
    },

    /** Removes the row in `slot`, or a row at random. */
    remove(slot?: number) {
      const live = ids.flatMap((id, at) => (id === undefined ? [] : [at]));
      const at = slot ?? live[Math.floor(random() * live.length)];
      if (at === undefined || ids[at] === undefined) return;

      enemies.remove(ids[at]);
      ids[at] = undefined;
      added.delete(at);
      for (const field of ALL) changed.delete(`${at}:${field}`);
    },

    /** Writes a field of the row in `slot`. */
    write(slot: number, field: Field) {
      const id = ids[slot]!;
      if (field === 'tint') enemies.fields.tint.set(id, random(), random(), random(), 1);
      if (field === 'position') enemies.fields.position.set(id, random() * 8, 0, random() * 8);
      if (field === 'health') enemies.fields.health.set(id, 10);
      changed.add(`${slot}:${field}`);
    },

    flush,

    destroy() {
      enemies.destroy();
    },
  };
}
