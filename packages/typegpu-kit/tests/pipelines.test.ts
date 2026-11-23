import { d, tgpu } from 'typegpu';
import { describe, expect, it } from 'vitest';
import { pipelines } from '../src/index.ts';
import { recordingRoot } from '../src/testing.ts';

function item(started: number[], error?: Error): pipelines.Warmable {
  return {
    initAsync() {
      started.push(0);
      return error ? Promise.reject(error) : Promise.resolve();
    },
  };
}

describe('compile', () => {
  it('starts every item within its budget at once and reports progress', async () => {
    const started: number[] = [];
    const progress: number[] = [];
    const items = [0, 1, 2].map(() => item(started));

    await pipelines.compile(items, { onProgress: (p) => progress.push(p) });

    expect(started).toEqual([0, 0, 0]);
    expect(progress).toEqual([1 / 3, 2 / 3, 1]);
  });

  it('stops calling a progress callback that threw, and rejects with its error once every item finishes', async () => {
    const last = Promise.withResolvers<void>();
    const progress: number[] = [];
    const error = new Error('progress failed');
    const items = [{ initAsync: async () => {} }, { initAsync: () => last.promise }];
    const compiling = pipelines.compile(items, {
      onProgress(fraction) {
        progress.push(fraction);
        throw error;
      },
    });
    items.push({ initAsync: () => new Promise<void>(() => {}) });
    await Promise.resolve();
    expect(progress).toEqual([0.5]);
    last.resolve();
    await expect(compiling).rejects.toBe(error);
    expect(progress).toEqual([0.5]);
  });

  it('yields to the event loop between items past its budget', async () => {
    const events: string[] = [];
    const items = [0, 1].map((i) => ({
      initAsync() {
        events.push(`item ${i}`);
        return Promise.resolve();
      },
    }));
    setTimeout(() => events.push('task'));

    await pipelines.compile(items, { budget: 0 });

    expect(events).toEqual(['item 0', 'task', 'item 1']);
  });

  it('rejects after every item has finished, with all the errors', async () => {
    const started: number[] = [];
    const thrown: pipelines.Warmable = {
      initAsync() {
        throw new Error('c');
      },
    };
    const items = [item(started, new Error('a')), item(started), thrown];

    const failed = (await pipelines
      .compile(items)
      .catch((error: unknown) => error)) as AggregateError;
    expect(started).toEqual([0, 0]);
    expect(failed.errors.map((e: Error) => e.message)).toEqual(['c', 'a']);
  });
  it('names the bindings of a pipeline that fails over a stage limit', async () => {
    const { root } = recordingRoot();
    const pools = Array.from({ length: 9 }, (_, i) =>
      root.createReadonly(d.arrayOf(d.f32, 4)).$name(`pool ${i}`),
    );
    const counts = root.createMutable(d.arrayOf(d.u32, 4)).$name('counts');
    const vertex = tgpu.vertexFn({ out: { pos: d.builtin.position } })(() => {
      'use gpu';
      let sum = d.f32(0);
      for (const pool of tgpu.unroll(pools)) sum += pool.$[0];
      return { pos: d.vec4f(sum) };
    });
    const fragment = tgpu.fragmentFn({ out: d.vec4f })(() => {
      'use gpu';
      counts.$[0] = d.u32(1);
      return d.vec4f(1);
    });
    const figure = root
      .createRenderPipeline({ vertex, fragment, targets: { format: 'rgba8unorm' } })
      .$name('figure');
    // Dawn rejects with this when a layout is over a limit; recordingRoot has no limits
    figure.initAsync = () =>
      Promise.reject(new Error('[Invalid PipelineLayout] is invalid due to a previous error.'));

    await expect(pipelines.compile([figure])).rejects.toThrowErrorMatchingInlineSnapshot(`
      [Error: The pipeline 'figure' binds more than WebGPU guarantees per stage:
      - vertex: 9 storage buffers, over 8
      Its storage buffers:
        pool_0 (group 0, binding 0; vertex)
        pool_1 (group 0, binding 1; vertex)
        pool_2 (group 0, binding 2; vertex)
        pool_3 (group 0, binding 3; vertex)
        pool_4 (group 0, binding 4; vertex)
        pool_5 (group 0, binding 5; vertex)
        pool_6 (group 0, binding 6; vertex)
        pool_7 (group 0, binding 7; vertex)
        pool_8 (group 0, binding 8; vertex)
        counts (group 0, binding 9; fragment)
      A binding counts in every stage its layout makes it visible to. To make room, fill a hook on the renderers that read it instead of the whole lighting, pack uniforms into one struct, or keep values in a buffer the pipeline already binds.
      The device reported: [Invalid PipelineLayout] is invalid due to a previous error.]
    `);
  });
});

describe('bindingsOf', () => {
  it('counts every entry of a shared layout, also those the shader never reads', () => {
    const { root } = recordingRoot();
    const pools = tgpu.bindGroupLayout(
      Object.fromEntries(
        Array.from({ length: 9 }, (_, i) => [`pool${i}`, { storage: d.arrayOf(d.f32, 4) }]),
      ),
    );
    const sink = tgpu.privateVar(d.f32);
    const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      sink.$ = pools.$.pool0![0];
    });
    const reading = root.createComputePipeline({ compute: kernel });

    const { bindings, over } = pipelines.bindingsOf(reading);
    expect(bindings.map((b) => b.name)).toEqual(Array.from({ length: 9 }, (_, i) => `pool${i}`));
    expect(over).toEqual([{ stage: 'compute', kind: 'storage', count: 9, limit: 8 }]);
    expect(
      pipelines.bindingsOf(reading, { limits: { maxStorageBuffersPerShaderStage: 9 } }).over,
    ).toEqual([]);
  });

  it("counts a layout's entries in the stages TypeGPU shows them to by default", () => {
    const { root } = recordingRoot();
    const layout = tgpu.bindGroupLayout({
      readable: { storageTexture: d.textureStorage2d('rgba8unorm', 'read-only') },
      writable: { storageTexture: d.textureStorage2d('rgba8unorm', 'write-only') },
      counts: { storage: d.arrayOf(d.u32, 4), access: 'mutable' },
    });
    const vertex = tgpu.vertexFn({ out: { pos: d.builtin.position } })(() => {
      'use gpu';
      return { pos: d.vec4f() };
    });
    const fragment = tgpu.fragmentFn({ out: d.vec4f })(() => {
      'use gpu';
      layout.$.counts[0] = d.u32(1);
      return d.vec4f(1);
    });
    const drawing = root.createRenderPipeline({
      vertex,
      fragment,
      targets: { format: 'rgba8unorm' },
    });

    expect(pipelines.bindingsOf(drawing).bindings.map((b) => [b.name, b.stages])).toEqual([
      ['readable', ['vertex', 'fragment']],
      ['writable', ['fragment']],
      ['counts', ['fragment']],
    ]);
  });
});

describe('blend', () => {
  it('adds light, and lays premultiplied colour over', () => {
    expect(pipelines.blend.add.color).toEqual({
      srcFactor: 'one',
      dstFactor: 'one',
      operation: 'add',
    });
    expect(pipelines.blend.over.alpha).toEqual({
      srcFactor: 'one',
      dstFactor: 'one-minus-src-alpha',
      operation: 'add',
    });
  });
});
