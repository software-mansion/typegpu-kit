import { frames, table, type Frames } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';
import { d } from 'typegpu';
import { floatsOf } from './program.ts';
import { seeded } from './svg.ts';

export const ROWS = 8;
/** Seconds an upload or an overwrite stays marked. */
export const FADE = 0.9;
/** Seconds between the hits that come on their own. */
const EVERY = 1.4;

const NORMAL = d.vec3f(0.5, 0.5, 0.5);
const FLASH = d.vec3f(1, 0.4, 0.2);

/** What a row holds besides its tint and `x`; nothing in the figure changes them. */
export const RADIUS = 0.5;
export const HEALTH = 10;

const Row = d.struct({ tint: d.vec3f, radius: d.f32, x: d.f32, health: d.f32 });
const HostRow = d.struct({ tint: d.vec3f, radius: d.f32 });
const DeviceRow = d.struct({ x: d.f32, health: d.f32 });

const OneStruct = table.define('OneStruct', { host: Row.propTypes });
const TwoParts = table.define('TwoParts', { host: HostRow.propTypes, device: DeviceRow.propTypes });

export interface Upload {
  readonly rows: readonly [number, number];
  readonly bytes: number;
  readonly at: number;
}

export interface Overwrite {
  readonly row: number;
  readonly from: number;
  readonly to: number;
  readonly at: number;
}

/** One table, the frames that flush it, and its GPU buffers as the recording root holds them. */
export interface Layout {
  readonly name: string;
  /** Whether `x` is in a buffer of its own part, apart from the host fields. */
  readonly split: boolean;
  readonly loop: Frames;
  /** `x` of every row in the GPU buffer that holds it. */
  readonly xs: Float32Array;
  readonly xStride: number;
  readonly hostBuffer: GPUBuffer;
  readonly hostStride: number;
  readonly log: { writes: { buffer: GPUBuffer; offset: number; size: number }[] };
  tint(row: number, value: d.v3f): void;
  flashed(row: number): boolean;
  uploads: Upload[];
  overwrites: Overwrite[];
  overwritten: number;
}

function oneStruct(spawn: readonly number[]): Layout {
  const { root, log, bytes } = recordingRoot();
  const rows = table.create(root, OneStruct, { capacity: ROWS });
  const loop = frames.create(root, () => rows.flush());
  const ids = spawn.map((x) => rows.add({ tint: NORMAL, radius: RADIUS, x, health: HEALTH }));
  loop.run(0);

  const host = root.unwrap(rows.buffers.host!);
  const tint = d.vec3f();

  return {
    name: 'one struct',
    split: false,
    loop,
    xs: floatsOf(bytes(host)).subarray(d.memoryLayoutOf(Row, (r) => r.x).offset / 4),
    xStride: d.sizeOf(Row) / 4,
    hostBuffer: host,
    hostStride: d.sizeOf(Row),
    log,
    tint: (i, value) => rows.fields.tint.write(ids[i], value),
    flashed: (i) => rows.fields.tint.get(ids[i], tint).x === 1,
    uploads: [],
    overwrites: [],
    overwritten: 0,
  };
}

function twoParts(spawn: readonly number[]): Layout {
  const { root, log, bytes } = recordingRoot();
  const rows = table.create(root, TwoParts, { capacity: ROWS });
  const loop = frames.create(root, () => rows.flush());
  const ids = spawn.map((x) => rows.add({ tint: NORMAL, radius: RADIUS, x, health: HEALTH }));
  loop.run(0);

  const host = root.unwrap(rows.buffers.host!);
  const tint = d.vec3f();

  return {
    name: 'a table',
    split: true,
    loop,
    xs: floatsOf(bytes(root.unwrap(rows.buffers.device!))).subarray(
      d.memoryLayoutOf(DeviceRow, (r) => r.x).offset / 4,
    ),
    xStride: d.sizeOf(DeviceRow) / 4,
    hostBuffer: host,
    hostStride: d.sizeOf(HostRow),
    log,
    tint: (i, value) => rows.fields.tint.write(ids[i], value),
    flashed: (i) => rows.fields.tint.get(ids[i], tint).x === 1,
    uploads: [],
    overwrites: [],
    overwritten: 0,
  };
}

export function createProgram() {
  const random = seeded(5);
  const spawn = Array.from({ length: ROWS }, () => random() * 0.2);
  const speeds = Array.from({ length: ROWS }, () => 0.06 + random() * 0.1);
  const layouts = [oneStruct(spawn), twoParts(spawn)];
  let time = 0;
  let nextHit = 1;
  let pending: number[] = [];

  /** One frame: the CPU's writes, the flush that uploads them, then the kernel. */
  const frame = (dt: number) => {
    time += dt;

    for (const layout of layouts) {
      for (const row of pending) layout.tint(row, layout.flashed(row) ? NORMAL : FLASH);

      const before = Array.from({ length: ROWS }, (_, row) => layout.xs[row * layout.xStride]);
      const writes = layout.log.writes.length;
      layout.loop.run(time);

      for (const write of layout.log.writes.slice(writes)) {
        if (write.buffer !== layout.hostBuffer) continue;
        const first = write.offset / layout.hostStride;
        const last = (write.offset + write.size) / layout.hostStride - 1;
        layout.uploads.push({ rows: [first, last], bytes: write.size, at: time });
      }

      for (let row = 0; row < ROWS; row++) {
        const index = row * layout.xStride;
        const x = layout.xs[index];

        if (x !== before[row]) {
          layout.overwrites.push({ row, from: before[row], to: x, at: time });
          layout.overwritten += 1;
        }

        layout.xs[index] = (x + speeds[row] * dt) % 1;
      }

      layout.uploads = layout.uploads.filter(
        (upload) => time - upload.at < FADE || upload === layout.uploads.at(-1),
      );
      layout.overwrites = layout.overwrites.filter((overwrite) => time - overwrite.at < FADE);
    }

    pending = [];
  };

  /** Writes rows `a` and `b`'s tint on the CPU in the next frame. */
  const hit = (a: number, b: number) => {
    pending = [a, b];
    nextHit = time + EVERY;
  };

  return {
    layouts: layouts as readonly Layout[],

    get time() {
      return time;
    },

    hit,
    frame,

    /** A frame as the figure plays it, hitting two random rows every 1.4 s. */
    step(dt: number) {
      if (time >= nextHit) {
        const a = Math.floor(random() * (ROWS - 3));
        hit(a, a + 3 + Math.floor(random() * (ROWS - a - 3)));
      }
      frame(dt);
    },

    destroy() {
      for (const layout of layouts) layout.loop.destroy();
    },
  };
}
