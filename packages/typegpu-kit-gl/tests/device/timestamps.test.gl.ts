import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { afterEach, describe, expect, test } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { errorsOf, glDevice, glRoot } from './compare.ts';
import { module } from './glsl.ts';
import { install } from './minimum.ts';

const TIMER = 'EXT_disjoint_timer_query_webgl2';
const SIZE = 1024;

let device: GLDevice | undefined;
let root: TgpuRoot | undefined;

afterEach(() => {
  const gl = (root?.device as GLDevice | undefined)?.gl ?? device?.gl;
  const error = gl?.getError();
  root?.destroy();
  device?.destroy();
  root = device = undefined;
  expect(error ?? 0).toBe(0);
});

function target(on: GPUDevice, size = SIZE) {
  return on.createTexture({
    size: [size, size],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
}

/** A full-screen triangle whose fragments each loop `steps` times. */
function heavy(on: GLDevice, steps: number) {
  const shader = module(on, {
    vertex: '',
    vertexMain: `  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)) * 2.0 - 1.0;
  gl_Position = vec4(p, 0.5, 1.0);`,
    fragment: 'layout(location = 0) out vec4 color;',
    fragmentMain: `  float v = gl_FragCoord.x;
  for (int i = 0; i < ${steps}; i++) v = fract(sin(v) * 43758.5453 + float(i));
  color = vec4(v, 0.0, 0.0, 1.0);`,
  });
  return on.createRenderPipeline({
    layout: 'auto',
    vertex: { module: shader },
    fragment: { module: shader, targets: [{ format: 'rgba8unorm' }] },
  });
}

/** Draws with the pipeline once, so a timed pass does not wait for its program to link. */
function warm(on: GPUDevice, pipeline: GPURenderPipeline, texture: GPUTexture) {
  const encoder = on.createCommandEncoder();
  const pass = renderPass(encoder, texture);
  pass.setPipeline(pipeline);
  pass.draw(3);
  pass.end();
  on.queue.submit([encoder.finish()]);
}

function renderPass(
  encoder: GPUCommandEncoder,
  texture: GPUTexture,
  timestampWrites?: GPURenderPassTimestampWrites,
) {
  return encoder.beginRenderPass({
    colorAttachments: [{ view: texture.createView(), loadOp: 'clear', storeOp: 'store' }],
    ...(timestampWrites && { timestampWrites }),
  });
}

/** Resolves `count` timestamps into a resolve buffer, copies them to a mapped one and reads them. */
async function read(on: GPUDevice, set: GPUQuerySet, count: number, first = 0) {
  const resolved = on.createBuffer({
    size: count * 8,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const mapped = on.createBuffer({
    size: count * 8,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const encoder = on.createCommandEncoder();
  encoder.resolveQuerySet(set, first, count, resolved, 0);
  encoder.copyBufferToBuffer(resolved, 0, mapped, 0, count * 8);
  on.queue.submit([encoder.finish()]);
  const mapping = mapped.mapAsync(GPUMapMode.READ);
  // The results come back in a later task, so the map must still wait after this one's microtasks
  await Promise.resolve();
  const waited = mapped.mapState === 'pending';
  await mapping;
  const stamps = [...new BigUint64Array(mapped.getMappedRange().slice(0))];
  mapped.unmap();
  resolved.destroy();
  mapped.destroy();
  return { stamps, waited };
}

const ns = (stamps: bigint[], begin: number, end: number) => Number(stamps[end]! - stamps[begin]!);

describe('the feature', () => {
  test("a context with the timer extension gives 'timestamp-query'", () => {
    device = glDevice();
    expect(device.gl.getExtension(TIMER)).not.toBeNull();
    expect(device.features.has('timestamp-query')).toBe(true);
  });

  test("without the extension the device has no 'timestamp-query', and a timestamp query set is an error", async () => {
    const undo = install({ minimum: false, hide: [TIMER] });
    try {
      device = glDevice();
      const made = device;
      expect(made.features.has('timestamp-query')).toBe(false);
      const errors = await errorsOf(made, () =>
        made.createQuerySet({ type: 'timestamp', count: 2 }),
      );
      expect(errors).toHaveLength(1);
      expect(errors[0]).toMatch(/timestamp-query/);
    } finally {
      undo();
    }
  });
});

describe('render passes', () => {
  test('a heavy pass measures longer than a trivial one, end after beginning, without waiting', async () => {
    device = glDevice();
    const made = device;
    const pipeline = heavy(made, 400);
    const set = made.createQuerySet({ type: 'timestamp', count: 4 });
    const texture = target(made);
    warm(made, pipeline, texture);
    const errors = await errorsOf(made, async () => {
      const encoder = made.createCommandEncoder();
      renderPass(encoder, texture, {
        querySet: set,
        beginningOfPassWriteIndex: 0,
        endOfPassWriteIndex: 1,
      }).end();
      const pass = renderPass(encoder, texture, {
        querySet: set,
        beginningOfPassWriteIndex: 2,
        endOfPassWriteIndex: 3,
      });
      pass.setPipeline(pipeline);
      for (let i = 0; i < 8; i++) pass.draw(3);
      pass.end();
      made.queue.submit([encoder.finish()]);
      const { stamps, waited } = await read(made, set, 4);
      expect(waited).toBe(true);
      expect(stamps.every((stamp) => stamp > 0n)).toBe(true);
      const trivial = ns(stamps, 0, 1);
      const loaded = ns(stamps, 2, 3);
      expect(trivial).toBeGreaterThanOrEqual(0);
      // 8 million fragments of 400 steps each take milliseconds; a clear takes microseconds
      expect(loaded).toBeGreaterThan(1e6);
      expect(loaded).toBeGreaterThan(trivial * 10);
    });
    expect(errors).toEqual([]);
  });

  test('timestamps of several passes in a frame never go back', async () => {
    device = glDevice();
    const made = device;
    const pipeline = heavy(made, 20);
    const set = made.createQuerySet({ type: 'timestamp', count: 8 });
    const texture = target(made, 256);
    const encoder = made.createCommandEncoder();
    for (let i = 0; i < 4; i++) {
      const pass = renderPass(encoder, texture, {
        querySet: set,
        beginningOfPassWriteIndex: i * 2,
        endOfPassWriteIndex: i * 2 + 1,
      });
      pass.setPipeline(pipeline);
      pass.draw(3);
      pass.end();
    }
    made.queue.submit([encoder.finish()]);
    const { stamps } = await read(made, set, 8);
    for (let i = 1; i < 8; i++) expect(stamps[i]).toBeGreaterThanOrEqual(stamps[i - 1]!);
    expect(stamps[0]).toBeGreaterThan(0n);
  });

  test('a timer split over passes counts the untimed pass between them', async () => {
    device = glDevice();
    const made = device;
    const pipeline = heavy(made, 400);
    const set = made.createQuerySet({ type: 'timestamp', count: 4 });
    const texture = target(made);
    warm(made, pipeline, texture);
    // Each pass clears its own target: a clear of the same one lets the driver drop the draws
    const encoder = made.createCommandEncoder();
    renderPass(encoder, target(made, 16), { querySet: set, beginningOfPassWriteIndex: 0 }).end();
    const pass = renderPass(encoder, texture);
    pass.setPipeline(pipeline);
    for (let i = 0; i < 8; i++) pass.draw(3);
    pass.end();
    renderPass(encoder, target(made, 16), { querySet: set, endOfPassWriteIndex: 1 }).end();
    made.queue.submit([encoder.finish()]);
    const { stamps } = await read(made, set, 2);
    expect(ns(stamps, 0, 1)).toBeGreaterThan(1e6);
  });

  test('a query never written resolves to zero beside one that was', async () => {
    device = glDevice();
    const made = device;
    const set = made.createQuerySet({ type: 'timestamp', count: 4 });
    const encoder = made.createCommandEncoder();
    renderPass(encoder, target(made, 16), {
      querySet: set,
      beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1,
    }).end();
    made.queue.submit([encoder.finish()]);
    const { stamps } = await read(made, set, 4);
    expect(stamps[0]).toBeGreaterThan(0n);
    expect(stamps[1]).toBeGreaterThanOrEqual(stamps[0]!);
    expect(stamps.slice(2)).toEqual([0n, 0n]);
  });

  test('destroying a query set with results pending still reads them', async () => {
    device = glDevice();
    const made = device;
    const set = made.createQuerySet({ type: 'timestamp', count: 2 });
    const resolved = made.createBuffer({
      size: 16,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const mapped = made.createBuffer({
      size: 16,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const errors = await errorsOf(made, async () => {
      const encoder = made.createCommandEncoder();
      renderPass(encoder, target(made, 16), {
        querySet: set,
        beginningOfPassWriteIndex: 0,
        endOfPassWriteIndex: 1,
      }).end();
      encoder.resolveQuerySet(set, 0, 2, resolved, 0);
      encoder.copyBufferToBuffer(resolved, 0, mapped, 0, 16);
      made.queue.submit([encoder.finish()]);
      set.destroy();
      await mapped.mapAsync(GPUMapMode.READ);
      const stamps = new BigUint64Array(mapped.getMappedRange());
      expect(stamps[0]).toBeGreaterThan(0n);
      expect(stamps[1]).toBeGreaterThanOrEqual(stamps[0]!);
      mapped.unmap();
    });
    expect(errors).toEqual([]);
  });

  test('destroying the device with results pending deletes each query once', async () => {
    const made = glDevice();
    const { gl } = made;
    const live = new Set<WebGLQuery>();
    let twice = 0;
    const createQuery = gl.createQuery.bind(gl);
    const deleteQuery = gl.deleteQuery.bind(gl);
    gl.createQuery = () => {
      const query = createQuery();
      if (query) live.add(query);
      return query;
    };
    gl.deleteQuery = (query: WebGLQuery | null) => {
      if (query && !live.delete(query)) twice++;
      deleteQuery(query);
    };
    const set = made.createQuerySet({ type: 'timestamp', count: 4 });
    const resolved = made.createBuffer({
      size: 32,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const mapped = made.createBuffer({
      size: 32,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = made.createCommandEncoder();
    for (let i = 0; i < 2; i++) {
      renderPass(encoder, target(made, 16), {
        querySet: set,
        beginningOfPassWriteIndex: i * 2,
        endOfPassWriteIndex: i * 2 + 1,
      }).end();
    }
    encoder.resolveQuerySet(set, 0, 4, resolved, 0);
    encoder.copyBufferToBuffer(resolved, 0, mapped, 0, 32);
    made.queue.submit([encoder.finish()]);
    const mapping = mapped.mapAsync(GPUMapMode.READ).then(
      () => 'mapped',
      () => 'rejected',
    );
    expect(live.size).toBeGreaterThan(0);
    expect(gl.getError()).toBe(0);
    made.destroy();
    expect(live.size).toBe(0);
    expect(twice).toBe(0);
    // The map settles rather than waiting for results that never come
    expect(await mapping).toMatch(/mapped|rejected/);
  });

  test('a disjoint span zeroes the timestamps of its task', async () => {
    device = glDevice();
    const made = device;
    const set = made.createQuerySet({ type: 'timestamp', count: 4 });
    const encoder = made.createCommandEncoder();
    for (let i = 0; i < 2; i++) {
      renderPass(encoder, target(made, 16), {
        querySet: set,
        beginningOfPassWriteIndex: i * 2,
        endOfPassWriteIndex: i * 2 + 1,
      }).end();
    }
    made.queue.submit([encoder.finish()]);
    // GL reports a disjoint operation, such as a GPU clock change, at its next read
    const { gl } = made;
    const timer = gl.getExtension(TIMER)!;
    const getParameter = gl.getParameter.bind(gl);
    let reported = false;
    gl.getParameter = (name: number) => {
      if (name !== timer.GPU_DISJOINT_EXT || reported) return getParameter(name);
      reported = true;
      return true;
    };
    await new Promise((resolve) => setTimeout(resolve, 0));
    const { stamps } = await read(made, set, 4);
    expect(reported).toBe(true);
    expect(stamps).toEqual([0n, 0n, 0n, 0n]);
    // The next frame's timestamps are taken afresh
    const next = made.createCommandEncoder();
    renderPass(next, target(made, 16), {
      querySet: set,
      beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1,
    }).end();
    made.queue.submit([next.finish()]);
    const again = await read(made, set, 2);
    expect(again.stamps[0]).toBeGreaterThan(0n);
    expect(again.stamps[1]).toBeGreaterThanOrEqual(again.stamps[0]!);
  });
});

const CROWD = 100_000;

/** Adds into one word from `count` invocations, which piles every record onto one texel. */
function crowded(on: TgpuRoot, count: number, label: string) {
  const cells = on.createMutable(d.arrayOf(d.atomic(d.u32), 4));
  const kernel = tgpu.computeFn({ workgroupSize: [64], in: { id: d.builtin.globalInvocationId } })(
    ({ id }) => {
      'use gpu';
      if (id.x >= count) return;
      std.atomicAdd(cells.$[0], 1);
    },
  );
  return { cells, pipeline: on.createComputePipeline({ compute: kernel }).$name(label) };
}

describe('compute passes', () => {
  test("TypeGPU's timestamp writes time a crowded kernel longer than a small one", async () => {
    root = glRoot();
    const made = root;
    const big = crowded(made, CROWD, 'big');
    const small = crowded(made, 64, 'small');
    const set = made.createQuerySet('timestamp', 4);
    // ANGLE on Metal gives a pass right after a heavy one some of its time, so the small one goes
    // first
    const times = async () => {
      small.pipeline
        .withTimestampWrites({
          querySet: set,
          beginningOfPassWriteIndex: 0,
          endOfPassWriteIndex: 1,
        })
        .dispatchWorkgroups(1);
      big.pipeline
        .withTimestampWrites({
          querySet: set,
          beginningOfPassWriteIndex: 2,
          endOfPassWriteIndex: 3,
        })
        .dispatchWorkgroups(Math.ceil(CROWD / 64));
      set.resolve();
      return set.read();
    };
    // The first dispatches pay for linking
    await times();
    const stamps = await times();
    expect(stamps.every((stamp) => stamp > 0n)).toBe(true);
    expect(stamps[1]).toBeGreaterThanOrEqual(stamps[0]!);
    expect(stamps[3]).toBeGreaterThanOrEqual(stamps[2]!);
    expect(stamps[2]).toBeGreaterThanOrEqual(stamps[1]!);
    expect(ns(stamps, 0, 1) * 10).toBeLessThan(ns(stamps, 2, 3));
    const { stats } = made.device as GLDevice;
    expect([...stats.kernels.keys()]).toContain('big (gpu)');
  });

  test('placement still moves a crowded form to the CPU while its passes are timed', async () => {
    root = glRoot();
    const made = root;
    const { cells, pipeline } = crowded(made, CROWD, 'timed crowded');
    const set = made.createQuerySet('timestamp', 2);
    const timed = pipeline.withTimestampWrites({
      querySet: set,
      beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1,
    });
    const errors = await errorsOf(made.device, async () => {
      const words: number[] = [];
      const durations: number[] = [];
      for (let frame = 1; frame <= 24; frame++) {
        timed.dispatchWorkgroups(Math.ceil(CROWD / 64));
        set.resolve();
        const stamps = await set.read();
        durations.push(ns(stamps, 0, 1));
        await new Promise((resolve) => setTimeout(resolve, 20));
        if (frame % 3 === 0) words.push((await cells.read())[0] as number);
      }
      expect(words).toEqual(Array.from({ length: 8 }, (_, i) => ((i + 1) * 3 * CROWD) >>> 0));
      expect(durations.every((each) => each >= 0)).toBe(true);
    });
    expect(errors).toEqual([]);
    const { stats } = made.device as GLDevice;
    expect(stats.placement.get('timed crowded')).toMatch(
      /^cpu: its GPU form took [\d.]+ ms a dispatch on the GPU, and a CPU run/,
    );
  });
});

/** Times one empty render pass into slots 0 and 1 of a new query set. */
function timedPass(on: GLDevice) {
  const set = on.createQuerySet({ type: 'timestamp', count: 2 });
  const encoder = on.createCommandEncoder();
  renderPass(encoder, target(on, 16), {
    querySet: set,
    beginningOfPassWriteIndex: 0,
    endOfPassWriteIndex: 1,
  }).end();
  return { set, encoder };
}

/** Maps a `MAP_READ` buffer and returns its bytes. */
async function bytesOf(buffer: GPUBuffer) {
  await buffer.mapAsync(GPUMapMode.READ);
  const bytes = new Uint8Array(buffer.getMappedRange().slice(0));
  buffer.unmap();
  return bytes;
}

const stampsOf = (bytes: Uint8Array) => [...new BigUint64Array(bytes.buffer, 0, 2)];

describe('pending resolves', () => {
  test('a write elsewhere in the buffer keeps the timestamps waiting', async () => {
    device = glDevice();
    const made = device;
    const { set, encoder } = timedPass(made);
    const resolved = made.createBuffer({
      size: 32,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const mapped = made.createBuffer({
      size: 32,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    encoder.resolveQuerySet(set, 0, 2, resolved, 0);
    made.queue.submit([encoder.finish()]);
    made.queue.writeBuffer(resolved, 16, new Uint32Array([7]));
    const copy = made.createCommandEncoder();
    copy.copyBufferToBuffer(resolved, 0, mapped, 0, 32);
    made.queue.submit([copy.finish()]);
    const bytes = await bytesOf(mapped);
    const [begin, end] = stampsOf(bytes);
    expect(begin).toBeGreaterThan(0n);
    expect(end).toBeGreaterThanOrEqual(begin!);
    expect(new Uint32Array(bytes.buffer)[4]).toBe(7);
  });

  test('copies of four bytes carry the halves of a waiting timestamp', async () => {
    device = glDevice();
    const made = device;
    const { set, encoder } = timedPass(made);
    const resolved = made.createBuffer({
      size: 16,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const halves = made.createBuffer({
      size: 16,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const whole = made.createBuffer({
      size: 16,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    encoder.resolveQuerySet(set, 0, 2, resolved, 0);
    for (let at = 0; at < 16; at += 4) encoder.copyBufferToBuffer(resolved, at, halves, at, 4);
    encoder.copyBufferToBuffer(resolved, 0, whole, 0, 16);
    made.queue.submit([encoder.finish()]);
    const expected = stampsOf(await bytesOf(whole));
    expect(expected[0]).toBeGreaterThan(0n);
    expect(stampsOf(await bytesOf(halves))).toEqual(expected);
  });

  test('a later copy of what GL holds wins over the waiting timestamps it covers', async () => {
    root = glRoot();
    const made = root;
    const raw = made.device as GLDevice;
    const fresh = made.createMutable(d.arrayOf(d.u32, 4));
    const fill = made.createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [4], in: { id: d.builtin.globalInvocationId } })(
        ({ id }) => {
          'use gpu';
          fresh.$[id.x] = 0xbbbbbbbb;
        },
      ),
    });
    const resolved = raw.createBuffer({
      size: 16,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const mapped = raw.createBuffer({
      size: 32,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const { set, encoder } = timedPass(raw);
    encoder.resolveQuerySet(set, 0, 2, resolved, 0);
    encoder.copyBufferToBuffer(resolved, 0, mapped, 0, 16);
    raw.queue.submit([encoder.finish()]);
    fill.dispatchWorkgroups(1);
    const copy = raw.createCommandEncoder();
    copy.copyBufferToBuffer(made.unwrap(fresh.buffer), 0, mapped, 0, 16);
    // A write elsewhere finishes the copy and keeps what is left of the timestamps waiting
    copy.clearBuffer(mapped, 16, 4);
    raw.queue.submit([copy.finish()]);
    const words = new Uint32Array((await bytesOf(mapped)).buffer);
    expect([...words.subarray(0, 4)]).toEqual([0xbbbbbbbb, 0xbbbbbbbb, 0xbbbbbbbb, 0xbbbbbbbb]);
  });

  test('a kernel that reads a resolve before its results arrive reads zeros', async () => {
    root = glRoot();
    const made = root;
    const raw = made.device as GLDevice;
    const stamps = made
      .createBuffer(d.arrayOf(d.u32, 4), [99, 99, 99, 99])
      .$usage('storage')
      .$addFlags(GPUBufferUsage.QUERY_RESOLVE);
    const seen = made.createMutable(d.arrayOf(d.u32, 4));
    const reading = stamps.as('readonly');
    const look = made.createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [4], in: { id: d.builtin.globalInvocationId } })(
        ({ id }) => {
          'use gpu';
          seen.$[id.x] = reading.$[id.x]!;
        },
      ),
    });
    const { set, encoder } = timedPass(raw);
    encoder.resolveQuerySet(set, 0, 2, made.unwrap(stamps), 0);
    raw.queue.submit([encoder.finish()]);
    look.dispatchWorkgroups(1);
    expect(await seen.read()).toEqual([0, 0, 0, 0]);
  });

  test('a span whose result never arrives resolves to zero within seconds', async () => {
    device = glDevice();
    const made = device;
    const { gl } = made;
    const getQueryParameter = gl.getQueryParameter.bind(gl);
    gl.getQueryParameter = (query: WebGLQuery, name: number) =>
      name === gl.QUERY_RESULT_AVAILABLE ? false : getQueryParameter(query, name);
    const { set, encoder } = timedPass(made);
    made.queue.submit([encoder.finish()]);
    const started = performance.now();
    const { stamps } = await read(made, set, 2);
    gl.getQueryParameter = getQueryParameter;
    expect(stamps).toEqual([0n, 0n]);
    expect(performance.now() - started).toBeLessThan(5000);
  });
});

describe('query sets the device rejected', () => {
  test("a timestamp query set without 'timestamp-query' is refused by passes and resolves", async () => {
    const undo = install({ minimum: false, hide: [TIMER] });
    try {
      device = glDevice();
      const made = device;
      const errors = await errorsOf(made, () => {
        const set = made.createQuerySet({ type: 'timestamp', count: 2 });
        const resolved = made.createBuffer({
          size: 16,
          usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
        const encoder = made.createCommandEncoder();
        renderPass(encoder, target(made, 16), {
          querySet: set,
          beginningOfPassWriteIndex: 0,
          endOfPassWriteIndex: 1,
        }).end();
        encoder.resolveQuerySet(set, 0, 2, resolved, 0);
        made.queue.submit([encoder.finish()]);
      });
      expect(errors).toHaveLength(3);
      expect(errors[1]).toMatch(/invalid query set/);
      expect(errors[2]).toMatch(/invalid query set/);
    } finally {
      undo();
    }
  });
});
