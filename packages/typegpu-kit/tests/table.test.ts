import { d, tgpu } from 'typegpu';
import { describe, expect, expectTypeOf, it } from 'vitest';
import { frames, history, table, type Frame, type RowId, type Table } from '../src/index.ts';
import { recordingRoot } from '../src/testing.ts';

// host: pos 0..12, kind 12, scale 16; stride 32. device: health 0, vel 8; stride 16.
const Enemy = table.define('Enemy', {
  host: { pos: d.vec3f, kind: d.u32, scale: d.f32 },
  device: { health: d.f32, vel: d.vec2f },
});

function setup(options: Partial<table.Options<typeof Enemy>> = {}) {
  const recording = recordingRoot();
  const { root, log, bytes } = recording;
  const enemies = table.create(root, Enemy, { capacity: 8, ...options });

  const gpu = {
    host: () => new Float32Array(bytes(enemies.buffers.host!).buffer),
    hostU32: () => new Uint32Array(bytes(enemies.buffers.host!).buffer),
    device: () => new Float32Array(bytes(enemies.buffers.device!).buffer),
    live: () => new Uint32Array(bytes(enemies.buffers.live).buffer),
    span: () => new Uint32Array(bytes(enemies.buffers.index.buffer).buffer)[0],
    mask: () => new Uint32Array(bytes(enemies.buffers.index.buffer).buffer)[1],
  };

  const buffers = {
    host: root.unwrap(enemies.buffers.host!),
    device: root.unwrap(enemies.buffers.device!),
    live: root.unwrap(enemies.buffers.live),
    span: root.unwrap(enemies.buffers.index.buffer),
  };

  const name = (buffer: GPUBuffer) =>
    Object.entries(buffers).find(([, b]) => b === buffer)?.[0] ?? 'other';

  /** Flushes, with no frame, and returns the uploads as `name offset+size`. */
  const flush = () => {
    const before = log.writes.length;
    enemies.flush();
    return log.writes.slice(before).map((w) => `${name(w.buffer)} ${w.offset}+${w.size}`);
  };

  return { root, enemies, gpu, flush };
}

describe('definition', () => {
  it('exposes each part as a typed array schema and rejects an empty part', () => {
    const host = table.rows(Enemy, 'host', 8);
    const device = table.rows(Enemy, 'device', 8);
    expectTypeOf(host).toEqualTypeOf<table.Rows<{ pos: d.Vec3f; kind: d.U32; scale: d.F32 }>>();
    expectTypeOf(device).toEqualTypeOf<table.Rows<{ health: d.F32; vel: d.Vec2f }>>();
    expect(host.elementType.propTypes).toEqual({ pos: d.vec3f, kind: d.u32, scale: d.f32 });
    expect(device.elementType.propTypes).toEqual({ health: d.f32, vel: d.vec2f });
    expect(d.sizeOf(host)).toBe(8 * 32);
    expect(d.sizeOf(device)).toBe(8 * 16);
    const Coin = table.define('Coin', { host: { pos: d.vec2f }, device: {} });
    const Spark = table.define('Spark', { device: { pos: d.vec2f } });
    expect(() => table.rows(Coin, 'device', 8)).toThrow("'Coin' has no device part");
    expect(() => table.rows(Spark, 'host', 8)).toThrow("'Spark' has no host part");
    expect(() => table.rows(Enemy, 'device', 0)).toThrow(RangeError);
  });

  it('rejects a WGSL keyword as a field name where the table is defined', () => {
    expect(() => table.define('Enemy', { host: { from: d.vec3f } })).toThrow(
      "Invalid property key 'from'",
    );
  });

  it('refuses f16 fields, which the CPU copy cannot write', () => {
    expect(() => table.define('Enemy', { device: { heat: d.f16 } })).toThrow("'heat' has type f16");
    expect(() => table.define('Enemy', { host: { tint: d.vec4h } })).toThrow(
      "'tint' has type vec4h",
    );
  });

  it('refuses a field in both parts, in the types too', () => {
    expect(() =>
      // @ts-expect-error: a field has one writer
      table.define('Enemy', { host: { pos: d.vec3f }, device: { pos: d.vec3f } }),
    ).toThrow("'pos' is both a host and a device field");
  });

  it('refuses a field named after one of its functions, in the types too', () => {
    // @ts-expect-error: `alive` would hide the definition's own
    expect(() => table.define('Enemy', { host: { alive: d.u32 } })).toThrow("'alive' is reserved");

    // As JavaScript calls it, where nothing checks the names before it runs.
    const define = table.define as (name: string, parts: { device: table.Props }) => unknown;
    for (const key of ['host', 'device', 'row', 'alive', 'span', 'slotOf']) {
      expect(() => define('Enemy', { device: { [key]: d.u32 } })).toThrow(`'${key}' is reserved`);
    }
  });

  it('treats an empty part as none', () => {
    const { root } = recordingRoot();
    const Coin = table.define('Coin', { host: { pos: d.vec2f }, device: {} });
    const coins = table.create(root, Coin, { capacity: 4 });

    expect(coins.buffers.device).toBeUndefined();
    expect(() => table.define('Enemy', { host: {} })).toThrow('A table needs a field');
  });

  it('names a table’s types from its definition', () => {
    const { root } = recordingRoot();
    const options: table.Options<typeof Enemy> = { capacity: 4, defaults: { health: 10 } };
    const enemies: table.Table<typeof Enemy> = table.create(root, Enemy, options);
    const scale: table.Fields<typeof Enemy>['scale'] = enemies.fields.scale;
    expect(scale.get(enemies.add())).toBe(0);

    // @ts-expect-error: an enemy has no speed
    table.create(root, Enemy, { capacity: 4, defaults: { speed: 1 } });
  });

  it('keeps each getter’s schema in its type, so a reader of u32 takes no f32', () => {
    const Light = table.define('Light', { host: { kind: d.f32 } });

    // @ts-expect-error: kind is an f32 here
    const kind: table.Getter<d.U32> = Light.kind;
    expect(kind).toBe(Light.kind);
  });
});

/** The id of a row at `slot` in a table of 8 rows. */
const idOf = (generation: number, slot: number) => generation * 8 + slot;

describe('slots and ids', () => {
  it('brands ids with the definition, and another table of it rejects them', () => {
    const { root } = recordingRoot();
    const Coin = table.define('Coin', { host: { position: d.vec2f } });
    const Gem = table.define('Gem', { host: { position: d.vec2f } });
    const Token = table.define('Coin', { host: { value: d.f32 } });
    const coins = table.create(root, Coin, { capacity: 4 });
    const gems = table.create(root, Gem, { capacity: 4 });
    const tokens = table.create(root, Token, { capacity: 4 });
    const coin: RowId<typeof Coin> = coins.add();

    // @ts-expect-error: a coin's id is not a gem's
    gems.has(coin);
    // @ts-expect-error: nor a row's of another definition with the same name
    tokens.has(coin);

    // The types accept an id from another table of coins, and that table rejects it
    const purse = table.create(root, Coin, { capacity: 4 });
    purse.add();
    expect(purse.has(coin)).toBe(false);
    expect(() => purse.remove(coin)).toThrow('another table');
  });

  it('takes the lowest free slot and bumps the generation on reuse', () => {
    const { enemies } = setup(); // 8 rows: 3 bits of slot
    const a = enemies.add();
    const b = enemies.add();
    const c = enemies.add();

    expect([a, b, c].map((id) => enemies.slotOf(id))).toEqual([0, 1, 2]);

    enemies.remove(b);
    const reused = enemies.add();

    expect(enemies.slotOf(reused)).toBe(1);
    const generation = a >>> 3;
    expect([a, b, c, reused]).toEqual([
      idOf(generation, 0),
      idOf(generation, 1),
      idOf(generation, 2),
      idOf(generation + 1, 1),
    ]);
    expect(enemies.count).toBe(3);
  });

  it('gives the slot the bits the capacity needs, and the generation the rest', () => {
    const { root } = recordingRoot();
    // The first table of a definition starts its generations at 1
    const first = (capacity: number) =>
      table.create(root, table.define('Bullet', { host: { speed: d.f32 } }), { capacity }).add();

    expect([1, 2, 5, 2 ** 20].map(first)).toEqual([0b10, 0b10, 0b1000, 2 ** 20]);
  });

  it('wraps the generation past its last value to 1, so no id is ever 0', () => {
    const { root } = recordingRoot();
    const Bullet = table.define('Bullet', { host: { speed: d.f32 } });
    const bullets = table.create(root, Bullet, { capacity: 2 ** 20 }); // 12 bits of generation

    const first = bullets.add();
    let id = first;
    let highest = 0;
    for (let i = 0; i < 4095; i++) {
      bullets.remove(id);
      id = bullets.add();
      highest = Math.max(highest, id);
    }

    expect(highest).toBe(0xfff00000);
    expect(id).toBe(first);
    expect(bullets.has(0 as typeof id)).toBe(false);
  });

  it('rejects stale ids', () => {
    const { enemies } = setup();
    const a = enemies.add();
    enemies.remove(a);
    enemies.add();

    expect(enemies.has(a)).toBe(false);
    expect(() => enemies.slotOf(a)).toThrow('stale');
    expect(() => enemies.remove(a)).toThrow('stale');
    expect(() => enemies.write(a, { scale: 1 })).toThrow('stale');
    expect(() => enemies.fields.scale.set(a, 1)).toThrow('stale');
  });

  it('shrinks the span to the highest live slot', () => {
    const { enemies } = setup();
    const [a, b, c] = [enemies.add(), enemies.add(), enemies.add()];

    expect(enemies.span).toBe(3);
    enemies.remove(c);
    expect(enemies.span).toBe(2);
    enemies.remove(a);
    expect(enemies.span).toBe(2);
    enemies.remove(b);
    expect(enemies.span).toBe(0);
  });

  it('fills every slot across bitset words, then throws', () => {
    const capacity = 32 * 32 + 5;
    const { enemies } = setup({ capacity });
    const slots = Array.from({ length: capacity }, () => enemies.slotOf(enemies.add()));

    expect(slots).toEqual(Array.from({ length: capacity }, (_, i) => i));
    expect(() => enemies.add()).toThrow('full');

    enemies.remove(enemies.idAt(1000));
    enemies.remove(enemies.idAt(40));
    expect(enemies.slotOf(enemies.add())).toBe(40);
    expect(enemies.slotOf(enemies.add())).toBe(1000);

    // A refilled slot of the last, partial word leaves the table full
    enemies.remove(enemies.idAt(capacity - 2));
    expect(enemies.slotOf(enemies.add())).toBe(capacity - 2);
    expect(() => enemies.add()).toThrow('The table is full');
    expect([enemies.count, enemies.span]).toEqual([capacity, capacity]);
  });

  it('frees the slot when a value fails to write', () => {
    const { enemies } = setup({ capacity: 1 });
    expect(() => enemies.add({ vel: null as never })).toThrow(TypeError);

    expect([enemies.count, enemies.span]).toEqual([0, 0]);
    expect(enemies.slotOf(enemies.add())).toBe(0);
  });

  it('maps a slot back to the id of its row', () => {
    const { enemies } = setup();
    const a = enemies.add();
    enemies.remove(a);
    const b = enemies.add();

    expect(enemies.idAt(0)).toBe(b);
    expect(() => enemies.idAt(1)).toThrow('holds no row');
  });

  it('validates the capacity', () => {
    const { root } = recordingRoot();

    expect(() => table.create(root, Enemy, { capacity: 0 })).toThrow(RangeError);
    expect(() => table.create(root, Enemy, { capacity: 2 ** 20 + 1 })).toThrow(RangeError);
    expect(() => table.create(root, Enemy, { capacity: 1.5 })).toThrow(RangeError);
  });

  it('clears every row', () => {
    const { enemies, gpu, flush } = setup();
    const ids = [enemies.add(), enemies.add(), enemies.add()];
    flush();

    enemies.clear();
    expect(enemies.count).toBe(0);
    expect(enemies.span).toBe(0);
    expect(ids.some((id) => enemies.has(id))).toBe(false);

    flush();
    expect(gpu.live()[0]).toBe(0);
    expect(gpu.span()).toBe(0);
  });
});

describe('values', () => {
  it('starts fields at zero or their default, and add sets the rest', () => {
    const { enemies, gpu, flush } = setup({ defaults: { scale: 1, health: 100 } });

    enemies.add({ scale: 3, kind: 2 }); // dirties the slot before reuse
    enemies.remove(enemies.idAt(0));
    enemies.add();
    enemies.add({ pos: [1, 2, 3], scale: 2, vel: [4, 5] });
    flush();

    expect([...gpu.host().subarray(0, 8)]).toEqual([0, 0, 0, 0, 1, 0, 0, 0]);
    expect([...gpu.host().subarray(8, 16)]).toEqual([1, 2, 3, 0, 2, 0, 0, 0]);
    expect([...gpu.device().subarray(0, 8)]).toEqual([100, 0, 0, 0, 100, 0, 4, 5]);
  });

  it('writes and reads fields through their handles', () => {
    const { enemies, gpu, flush } = setup();
    const id = enemies.add();
    const { pos, kind, vel } = enemies.fields;

    pos.set(id, 1, 2, 3);
    kind.set(id, 7);
    vel.write(id, d.vec2f(8, 9));

    expect(pos.get(id, d.vec3f())).toEqual(d.vec3f(1, 2, 3));
    expect(kind.get(id)).toBe(7);

    flush();
    expect([...gpu.host().subarray(0, 3)]).toEqual([1, 2, 3]);
    expect(gpu.hostU32()[3]).toBe(7);
    expect([...gpu.device().subarray(2, 4)]).toEqual([8, 9]);
  });

  it('writes vectors from any array of numbers and reads them into one', () => {
    const { enemies, gpu, flush } = setup();
    const id = enemies.add({ vel: new Float32Array([1, 2, 3]) });
    const pos: number[] = [4, 5, 6];
    enemies.fields.pos.write(id, pos);

    expect(enemies.fields.pos.get(id, new Float32Array(3))).toEqual(new Float32Array([4, 5, 6]));
    flush();
    expect([...gpu.host().subarray(0, 3)]).toEqual([4, 5, 6]);
    expect([...gpu.device().subarray(2, 4)]).toEqual([1, 2]);
  });

  it('writes several fields of both parts at once', () => {
    const { enemies, gpu, flush } = setup();
    const id = enemies.add();
    flush();

    enemies.write(id, { scale: 4, health: 5 });
    flush();

    expect(gpu.host()[4]).toBe(4);
    expect(gpu.device()[0]).toBe(5);
  });
});

describe('flush', () => {
  it('uploads new rows whole, then only what changed', () => {
    const { enemies, flush } = setup();
    const ids = [enemies.add(), enemies.add(), enemies.add()];

    expect(flush()).toEqual(['host 0+96', 'device 0+48', 'live 0+4', 'span 0+4']);
    expect(flush()).toEqual([]);

    enemies.fields.health.set(ids[1], 5);
    enemies.fields.vel.set(ids[2], 1, 1);
    enemies.fields.scale.set(ids[2], 2);

    expect(flush()).toEqual(['host 64+32', 'device 16+8', 'device 40+8']);
  });

  it('uploads each run of written device fields, across the padding between them', () => {
    const { enemies, flush } = setup();
    const [a, b] = [enemies.add(), enemies.add()];
    flush();

    enemies.fields.vel.set(a, 1, 1);
    enemies.fields.health.set(a, 5);
    enemies.fields.vel.set(a, 2, 2);
    enemies.write(b, { vel: [3, 3] });

    expect(flush()).toEqual(['device 0+16', 'device 24+8']);
    expect(flush()).toEqual([]);
  });

  it('gives each device field past the 32nd a bit of its own', () => {
    const { root, log } = recordingRoot();
    const fields = Object.fromEntries(Array.from({ length: 34 }, (_, i) => [`f${i}`, d.f32]));
    const wide = table.create(root, table.define('Wide', { device: fields }), { capacity: 2 });
    const id = wide.add();
    wide.flush();

    const before = log.writes.length;
    wide.fields.f31!.set(id, 1);
    wide.fields.f33!.set(id, 1);
    wide.flush();

    expect(log.writes.slice(before).map((w) => `${w.offset}+${w.size}`)).toEqual([
      '124+4',
      '132+4',
    ]);
  });

  it('uploads rows added since the last flush in runs, and skips their field writes', () => {
    const { enemies, flush } = setup();
    const ids = [enemies.add(), enemies.add(), enemies.add()];
    flush();

    enemies.remove(ids[1]);
    const b = enemies.add();
    enemies.add();
    enemies.add();
    enemies.fields.health.set(b, 1);

    expect(flush()).toEqual([
      'host 32+128',
      'device 16+16',
      'device 48+32',
      'live 0+4',
      'span 0+4',
    ]);
  });
});

describe('flush runs', () => {
  it('uploads adjacent rows written end to end as one run, and the rest one row each', () => {
    const { enemies, flush } = setup();
    const ids = Array.from({ length: 6 }, () => enemies.add());
    flush();

    for (const at of [1, 2, 3, 5]) enemies.write(ids[at], { health: at, vel: [at, at] });
    expect(flush()).toEqual(['device 16+48', 'device 80+16']);

    // Health sits between each row's velocity, so these can't merge
    for (const at of [1, 2]) enemies.fields.vel.set(ids[at], 0, 0);
    expect(flush()).toEqual(['device 24+8', 'device 40+8']);

    // A new row is written whole, so it joins the written row before it
    enemies.remove(ids[4]);
    enemies.add();
    enemies.write(ids[3], { health: 1, vel: [1, 1] });
    expect(flush()).toEqual(['host 128+32', 'device 48+32', 'live 0+4']);
  });

  it('uploads written host rows in runs that bridge gaps under 16 clean rows', () => {
    const { enemies, flush } = setup({ capacity: 64 });
    const ids = Array.from({ length: 64 }, () => enemies.add());
    flush();

    for (const at of [0, 16, 33, 63]) enemies.fields.scale.set(ids[at], 2);
    expect(flush()).toEqual(['host 0+544', 'host 1056+32', 'host 2016+32']);
  });

  it('keeps what a kernel wrote between two written fields of a row, with history too', () => {
    const Trio = table.define('Trio', { device: { a: d.u32, b: d.u32, c: d.u32 } });
    for (const keeps of [false, true]) {
      const { root, bytes } = recordingRoot();
      const trio = table.create(root, Trio, { capacity: 2, history: keeps });
      const frame = () => {
        const encoder = root.createCommandEncoder();
        trio.flush(encoder);
        encoder.submit();
      };
      const id = trio.add();
      frame();
      const device = new Uint32Array(bytes(trio.buffers.device!).buffer);

      device[1] = 99; // a kernel's b
      trio.write(id, { a: 11, c: 22 });
      frame();

      expect([...device.subarray(0, 3)]).toEqual([11, 99, 22]);
    }
  });
});

/** Resource declarations, sorted. */
const vars = (wgsl: string) => (wgsl.match(/var<[^>]+> \w+: [^;]+/g) ?? []).toSorted();

describe('shaders', () => {
  const step = tgpu.computeFn({
    workgroupSize: [64],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (!Enemy.alive(i)) return;

    Enemy.device.$[i].health = Enemy.health(i) - Enemy.scale(i);
  });

  const vertex = tgpu.vertexFn({
    in: { iid: d.builtin.instanceIndex },
    out: { position: d.builtin.position },
  })(({ iid }) => {
    'use gpu';
    const row = Enemy.row(iid);
    return { position: d.vec4f(row.pos, row.health) };
  });

  const fragment = tgpu.fragmentFn({ out: d.vec4f })(() => {
    'use gpu';
    return d.vec4f(1);
  });

  it('binds device rows read-write in compute and read-only in render', () => {
    const { root, enemies } = setup();
    const gpu = root.pipe(enemies.inject());

    const compute = vars(tgpu.resolve([gpu.createComputePipeline({ compute: step })]));
    const render = vars(
      tgpu.resolve([
        gpu.createRenderPipeline({ vertex, fragment, targets: { format: 'rgba8unorm' } }),
      ]),
    );

    expect(compute).toEqual([
      'var<storage, read> Enemy_host: array<EnemyHost, 8>',
      'var<storage, read> Enemy_live: array<u32, 1>',
      'var<storage, read_write> Enemy_device: array<EnemyDevice, 8>',
      'var<uniform> Enemy_index: index',
    ]);
    expect(render).toEqual([
      'var<storage, read> Enemy_device: array<EnemyDevice, 8>',
      'var<storage, read> Enemy_host: array<EnemyHost, 8>',
    ]);
  });

  it('reads and writes two tables of one definition in one kernel through gpu, unpiped', () => {
    const { root, enemies: reds } = setup();
    const blues = table.create(root, Enemy, { capacity: 8 });
    const red = reds.gpu;
    const blue = blues.gpu;
    const defect = tgpu.computeFn({
      workgroupSize: [64],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      const i = gid.x;
      if (!red.alive(i)) return;
      blue.device.$[i].health = red.health(i) + Enemy.health(i);
      red.device.$[i].health = 0;
    });

    // The definition's own members, filled from reds, share reds' bindings
    const wgsl = tgpu.resolve([
      root.pipe(reds.inject()).createComputePipeline({ compute: defect }),
    ]);
    expect(vars(wgsl)).toEqual([
      'var<storage, read> Enemy_live: array<u32, 1>',
      'var<storage, read_write> Enemy_device: array<EnemyDevice, 8>',
      'var<storage, read_write> Enemy_device_1: array<EnemyDevice, 8>',
      'var<uniform> Enemy_index: index',
    ]);
    expect(() => tgpu.resolve([root.createComputePipeline({ compute: defect })])).toThrow(
      /The pipeline is missing a fill for `Enemy_device`/,
    );
  });

  it('takes a slot from an id with the mask in the table’s uniform', () => {
    const { root, enemies, gpu } = setup();
    const revive = tgpu.computeFn({
      workgroupSize: [1],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      Enemy.device.$[Enemy.slotOf(gid.x)].health = 1;
    });

    const wgsl = tgpu.resolve([
      root.pipe(enemies.inject()).createComputePipeline({ compute: revive }),
    ]);

    expect(wgsl).toContain(
      'fn Enemy_slotOf(id: u32) -> u32 {\n  return (id & Enemy_index.mask);\n}',
    );
    expect(gpu.mask()).toBe(0b111);
  });

  it('names its structs and functions after the definition', () => {
    const { root, enemies } = setup();
    const wgsl = tgpu.resolve([
      root.pipe(enemies.inject()).createComputePipeline({ compute: step }),
      root.pipe(enemies.inject()).createRenderPipeline({
        vertex,
        fragment,
        targets: { format: 'rgba8unorm' },
      }),
    ]);

    expect(wgsl.match(/(?<=fn |struct )\w+/g)!.filter((n) => n.startsWith('Enemy'))).toEqual([
      'Enemy_alive',
      'EnemyDevice',
      'Enemy_health',
      'EnemyHost',
      'Enemy_scale',
      'EnemyRow',
      'Enemy_row',
    ]);
  });

  it('fails to resolve without the table, and names the inject() to pipe in', () => {
    const { root } = setup();
    const pipeline = root.createComputePipeline({ compute: step });
    expect(() => tgpu.resolve([pipeline])).toThrow(
      'fn*:Enemy_alive(u32): The pipeline is missing a fill for `Enemy_index`. Create it from `root.pipe(rows.inject())`, where `rows` is a `table.create` of `Enemy`.',
    );
  });
});

/** A table that keeps history, with its buffers found by label and a frame that flushes. */
function kept() {
  const recording = recordingRoot();
  const { root, bytes } = recording;
  const made: GPUBuffer[] = [];
  const create = root.device.createBuffer.bind(root.device);
  root.device.createBuffer = (descriptor) => {
    const buffer = create(descriptor);
    made.push(buffer);
    return buffer;
  };
  const enemies = table.create(root, Enemy, { capacity: 8, history: true });
  const floats = (label: string) => {
    const buffer = made.find((each) => each.label === label)!;
    return new Float32Array(bytes(buffer).buffer);
  };
  /**
   * Each slot's `pos.x` in host rows and `health` in device rows, now and before this frame, which
   * follows the current rows from slot 8.
   */
  const rows = (slots: number) => {
    const of = (label: string, stride: number, from: number) =>
      Array.from({ length: slots }, (_, slot) => floats(label)[(from + slot) * stride]);
    return {
      now: [of('Enemy_host', 8, 0), of('Enemy_device', 4, 0)],
      before: [of('Enemy_host', 8, 8), of('Enemy_device', 4, 8)],
    };
  };
  const frame = () => {
    const encoder = root.createCommandEncoder();
    enemies.flush(encoder);
    encoder.submit();
  };
  return { root, enemies, floats, rows, frame };
}

describe('history', () => {
  it('starts new rows last frame where they are, and moves written rows from where they were', () => {
    const { enemies, floats, rows, frame } = kept();
    const a = enemies.add({ pos: [1, 0, 0], health: 10 });
    frame();
    expect(rows(1)).toEqual({ now: [[1], [10]], before: [[1], [10]] });

    enemies.fields.pos.set(a, 2, 0, 0);
    const b = enemies.add({ pos: [5, 0, 0], health: 20 });
    frame();
    expect(rows(2)).toEqual({
      now: [
        [2, 5],
        [10, 20],
      ],
      before: [
        [1, 5],
        [10, 20],
      ],
    });

    // A kernel wrote after the last flush; the next frame uploads nothing and catches up
    floats('Enemy_device')[0] = 9;
    frame();
    expect(rows(2)).toEqual({
      now: [
        [2, 5],
        [9, 20],
      ],
      before: [
        [2, 5],
        [9, 20],
      ],
    });

    // A device write overwrites the kernel's, and moves from it
    enemies.fields.health.set(b, 30);
    frame();
    expect(rows(2)).toEqual({
      now: [
        [2, 5],
        [9, 30],
      ],
      before: [
        [2, 5],
        [9, 20],
      ],
    });
  });

  it('flushes again what a frame that threw recorded, and keeps last frame through two flushes', () => {
    const { root, enemies, rows } = kept();
    const a = enemies.add({ pos: [1, 0, 0], health: 10 });
    let record = (frame: Frame) => enemies.flush(frame);
    const loop = frames.create(root, (frame) => record(frame));
    loop.run(0);

    enemies.fields.pos.set(a, 2, 0, 0);
    record = (frame) => {
      enemies.flush(frame);
      throw new Error('lost');
    };
    expect(() => loop.run(1)).toThrow('lost');

    enemies.fields.health.set(a, 15);
    record = (frame) => {
      enemies.flush(frame);
      enemies.fields.health.set(a, 20);
      enemies.add({ pos: [5, 0, 0], health: 50 });
      enemies.flush(frame);
    };
    loop.run(2);
    expect(rows(2)).toEqual({
      now: [
        [2, 5],
        [20, 50],
      ],
      before: [
        [1, 5],
        [10, 50],
      ],
    });
  });

  it('keeps what an early submit sent when its frame then throws, and last frame through submits', () => {
    const { root, enemies, floats, rows } = kept();
    const a = enemies.add({ pos: [1, 0, 0], health: 10 });
    let record = (frame: Frame) => enemies.flush(frame);
    const loop = frames.create(root, (frame) => record(frame));
    loop.run(0);
    loop.run(1);

    enemies.fields.pos.set(a, 2, 0, 0);
    enemies.fields.health.set(a, 15);
    record = (frame) => {
      enemies.flush(frame);
      frame.submit();
      floats('Enemy_device')[0] = 9; // a kernel moves the sent health
      throw new Error('lost');
    };
    expect(() => loop.run(2)).toThrow('lost');

    enemies.fields.pos.set(a, 3, 0, 0);
    record = (frame) => {
      enemies.flush(frame);
      frame.submit();
      enemies.fields.pos.set(a, 4, 0, 0);
      enemies.flush(frame);
    };
    loop.run(3);
    expect(rows(1)).toEqual({ now: [[4], [9]], before: [[2], [9]] });
  });

  it('takes the frame, in its type and, held as a plain table, when it runs', () => {
    const { enemies } = kept();
    const message =
      "The table 'Enemy' keeps history and needs the frame when you flush it. Call `flush(frame)`.";
    // @ts-expect-error a table with history copies last frame's rows in the frame
    expect(() => enemies.flush()).toThrow(message);
    const plain: Table<typeof Enemy> = enemies;
    expect(() => plain.flush()).toThrow(message);
  });

  it('reads last frame’s rows from the same buffers in code compiled with previously', () => {
    const { root, enemies } = kept();
    const lastFrame = history.previously((i: number) => {
      'use gpu';
      return d.vec4f(Enemy.pos(i), Enemy.health(i));
    });
    const vertex = tgpu.vertexFn({
      in: { iid: d.builtin.instanceIndex },
      out: { position: d.builtin.position, before: d.vec4f },
    })(({ iid }) => {
      'use gpu';
      return { position: d.vec4f(Enemy.pos(iid), Enemy.health(iid)), before: lastFrame(iid) };
    });
    const fragment = tgpu.fragmentFn({ in: { before: d.vec4f }, out: d.vec4f })(({ before }) => {
      'use gpu';
      return before;
    });

    const pipeline = root
      .pipe(enemies.inject())
      .createRenderPipeline({ vertex, fragment, targets: { format: 'rgba8unorm' } });

    const wgsl = tgpu.resolve([pipeline]);
    expect(vars(wgsl)).toEqual([
      'var<storage, read> Enemy_device: array<EnemyDevice, 16>',
      'var<storage, read> Enemy_host: array<EnemyHost, 16>',
    ]);
    expect(wgsl).toContain('return Enemy_host[(i + 8u)].pos;');
  });
});
