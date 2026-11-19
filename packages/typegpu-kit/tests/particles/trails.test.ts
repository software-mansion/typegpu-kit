import { cameras, passes, targets } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';
import { caps } from '@typegpu/kit/geometry';
import { d, tgpu } from 'typegpu';
import { describe, expect, it, vi } from 'vitest';
import { trails } from '../../src/index.ts';

function setup() {
  const recording = recordingRoot();
  const { root } = recording;
  const view = cameras.orbit(root, {
    size: { width: 4, height: 3 },
    target: [0, 0, 0],
    distance: 5,
  });
  const screen = targets.create(
    root,
    { width: 4, height: 3 },
    { color: { format: 'rgba16float' }, depth: { format: 'depth24plus' } },
  );
  const into = passes.create(root, screen, { color: passes.color }).pipe(view.inject());
  const wakes = trails.create(root, { capacity: 3, length: 4 });
  const drawn = into.add(wakes.drawing);

  const frame = () => {
    const encoder = root.createCommandEncoder();
    const pass = into.begin(encoder);
    drawn.draw(pass);
    pass.end();
    encoder.submit();
  };

  return { recording, root, view, screen, into, wakes, frame };
}

const words = (bytes: Uint8Array) => [...new Uint32Array(bytes.slice().buffer)];

const floats = (bytes: Uint8Array) => [...new Float32Array(bytes.slice().buffer)];

describe('trails', () => {
  it('draws every segment of every trail in one indexed draw of one pipeline', () => {
    const { recording, frame } = setup();
    frame();

    expect(recording.log.pipelines).toBe(1);
    expect(recording.log.events.filter((event) => event.startsWith('draw'))).toEqual([
      'draw-indexed',
    ]);
  });

  it('names its drawing from the label option, or trails by default', () => {
    const { root, wakes } = setup();
    const tails = trails.create(root, { capacity: 1, length: 2, label: 'tails' });

    expect([wakes.drawing.label, tails.drawing.label]).toEqual(['trails', 'tails']);
  });

  it('creates nothing in a steady frame', () => {
    const { recording, wakes, frame } = setup();
    frame();
    const before = recording.log.counts();

    wakes.write(1, [0, 0, 0, 1, 0, 0]);
    frame();
    frame();
    expect(recording.log.counts()).toEqual(before);
  });

  it('writes the newest points it keeps, oldest first, and the ring that finds them', () => {
    const { recording, root, wakes } = setup();
    wakes.write(1, [0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0]);

    const points = floats(recording.bytes(root.unwrap(wakes.points)));
    const second = points.slice(4 * 4, 8 * 4);
    expect(second.filter((_, i) => i % 4 === 0)).toEqual([2, 3, 4, 5]);
    expect(words(recording.bytes(root.unwrap(wakes.rings)))).toEqual([0, 0, 0, 4, 0, 0]);

    wakes.write(2, [7, 8, 9]);
    expect(words(recording.bytes(root.unwrap(wakes.rings))).slice(4)).toEqual([1, 1]);

    wakes.write(1, []);
    expect(words(recording.bytes(root.unwrap(wakes.rings))).slice(2, 4)).toEqual([0, 0]);
  });

  it('refuses a trail it does not have', () => {
    const { wakes } = setup();

    expect(() => wakes.write(3, [0, 0, 0])).toThrow(RangeError);
    expect(() => wakes.write(-1, [0, 0, 0])).toThrow(RangeError);
  });

  it('refuses fewer than two points a trail', () => {
    const { root } = setup();

    expect(() => trails.create(root, { capacity: 1, length: 1 })).toThrow(
      'trails.create requires `length` to be an integer of at least 2. Received 1.',
    );
  });

  it('starts its colour and width at the defaults, under the settings option', () => {
    const { recording, root, wakes } = setup();
    const thick = trails.create(root, { capacity: 1, length: 2, settings: { width: 0.5 } });

    expect(floats(recording.bytes(root.unwrap(wakes.settings.buffer))).slice(0, 5)).toEqual(
      [1, 1, 1, 1, 0.1].map(Math.fround),
    );
    expect(floats(recording.bytes(root.unwrap(thick.settings.buffer)))[4]).toBe(0.5);
  });

  it('lets another module’s kernel push through its inject', async () => {
    const { recording, root, wakes } = setup();
    const kernel = tgpu.computeFn({
      workgroupSize: [64],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      wakes.push(gid.x, d.vec3f(1, 2, 3));
    });

    await root.pipe(wakes.inject()).createComputePipeline({ compute: kernel }).initAsync();

    const wgsl = recording.log.shaders.at(-1)!;
    expect(wgsl).toMatch(/var<storage, read_write> \w+: array<vec3f, 12>/);
    expect(wgsl).toMatch(/var<storage, read_write> \w+: array<Ring, 3>/);
  });

  it('resets a trail from a kernel through its rings alone', async () => {
    const { recording, root, wakes } = setup();
    const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      wakes.reset(2);
    });

    await root.pipe(wakes.inject()).createComputePipeline({ compute: kernel }).initAsync();

    const wgsl = recording.log.shaders.at(-1)!;
    expect(wgsl.match(/var<storage/g)).toHaveLength(1);
    expect(wgsl).toMatch(/var<storage, read_write> \w+: array<Ring, 3>/);
  });

  it('takes the caps and joins of @typegpu/kit/geometry in its options', async () => {
    const { recording, root, into } = setup();
    const squared = trails.create(root, { capacity: 1, length: 2, startCap: caps.square });
    into.add(squared.drawing);
    await into.initAsync();

    expect(recording.log.shaders.at(-1)).toMatch(/fn square/);
  });

  it('tests the depth of its passes without writing it, and blends over, when its options say so', async () => {
    const { root, view, screen } = setup();
    const created = vi.spyOn(root.device, 'createRenderPipelineAsync');
    const ink = trails.create(root, { capacity: 1, length: 2, blend: 'over' });
    const into = passes
      .create(root, screen, { color: passes.color, depth: passes.depth })
      .pipe(view.inject());
    into.add(ink.drawing);
    await into.initAsync();

    const [descriptor] = created.mock.calls.map(([each]) => each);
    expect(descriptor!.depthStencil).toEqual({
      format: 'depth24plus',
      depthWriteEnabled: false,
      depthCompare: 'less',
    });
    expect([...descriptor!.fragment!.targets][0]!.blend!.color).toEqual({
      srcFactor: 'one',
      dstFactor: 'one-minus-src-alpha',
      operation: 'add',
    });
  });
});
