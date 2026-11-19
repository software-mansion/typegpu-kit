import { cameras, lines, passes, targets } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';
import { caps, joins } from '@typegpu/kit/geometry';
import { d, std, tgpu } from 'typegpu';
import { describe, expect, it } from 'vitest';

function setup(options: Partial<lines.Options> = {}) {
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
  const into = passes
    .create(root, screen, { color: passes.color, depth: passes.depth })
    .pipe(view.inject());
  const paths = lines.create(root, { capacity: 3, length: 4, ...options });
  const drawn = into.add(paths.drawing);

  const frame = () => {
    const encoder = root.createCommandEncoder();
    const pass = into.begin(encoder);
    drawn.draw(pass);
    pass.end();
    encoder.submit();
  };

  /** Draws `set` once, which creates its buffers. */
  const drawOnce = (set: lines.Lines) => {
    const encoder = root.createCommandEncoder();
    const pass = into.begin(encoder);
    into.add(set.drawing).draw(pass);
    pass.end();
    encoder.submit();
  };

  return { recording, root, view, into, paths, frame, drawOnce };
}

const floats = (bytes: Uint8Array) => [...new Float32Array(bytes.slice().buffer)];
const words = (bytes: Uint8Array) => [...new Uint32Array(bytes.slice().buffer)];

describe('lines', () => {
  it('draws every segment of every line in one indexed draw of one pipeline', () => {
    const { recording, frame } = setup();
    frame();

    expect(recording.log.pipelines).toBe(1);
    expect(recording.log.events.filter((event) => event.startsWith('draw'))).toEqual([
      'draw-indexed',
    ]);
  });

  it('draws the segments of the shown lines alone, and refuses a count past its capacity', () => {
    const { paths } = setup({ closed: true });
    const draws: number[] = [];
    const pass = {
      drawIndexed: (_indices: number, instances: number) => draws.push(instances),
    } as unknown as Parameters<typeof paths.drawing.draw>[0];

    paths.drawing.draw(pass);
    paths.shown = 1;
    paths.drawing.draw(pass);
    paths.shown = 0;
    paths.drawing.draw(pass);
    expect(draws).toEqual([3 * 4, 4]);
    expect(() => (paths.shown = 4)).toThrow('`shown` takes an integer from 0 to the capacity, 3');
  });

  it('builds one pipeline per join, cap and blend', () => {
    for (const options of [
      { join: joins.miter },
      { startCap: caps.arrow, endCap: caps.butt },
      { blend: 'opaque' as const },
      { units: 'points' as const, closed: true },
    ]) {
      const { recording, frame } = setup(options);
      frame();
      expect(recording.log.pipelines).toBe(1);
    }
  });

  it('creates nothing in a steady frame', () => {
    const { recording, paths, frame } = setup();
    frame();
    const before = recording.log.counts();

    paths.write(1, [0, 0, 0, 1, 0, 0]);
    frame();
    frame();
    expect(recording.log.counts()).toEqual(before);
  });

  it('writes points with the distance along the line, and the count it keeps', () => {
    const { recording, root, paths } = setup();
    paths.write(1, [0, 0, 0, 3, 4, 0, 3, 4, 2, 9, 9, 9, 7, 7, 7], {
      width: [1, 2, 3, 4],
      color: [1, 0, 0, 1],
    });

    const points = floats(recording.bytes(root.unwrap(paths.points!))).slice(4 * 12, 8 * 12);
    expect(points.filter((_, i) => i % 12 === 8)).toEqual(
      [0, 5, 7, 7 + Math.hypot(6, 5, 7)].map(Math.fround),
    );
    expect(points.filter((_, i) => i % 12 === 3)).toEqual([1, 2, 3, 4]);
    expect(points.slice(16, 20)).toEqual([1, 0, 0, 1]);
    expect(words(recording.bytes(root.unwrap(paths.counts!)))).toEqual([0, 4, 0]);

    paths.write(1, []);
    expect(words(recording.bytes(root.unwrap(paths.counts!)))).toEqual([0, 0, 0]);
  });

  it('refuses a line it does not have, too few widths or colours, and fewer than two points a line', () => {
    const { root, paths } = setup();

    expect(() => paths.write(3, [0, 0, 0])).toThrow(RangeError);
    expect(() => paths.write(-1, [0, 0, 0])).toThrow(RangeError);
    expect(() => paths.write(0, [0, 0, 0, 1, 0, 0], { width: [1] })).toThrow(RangeError);
    expect(() =>
      paths.write(0, [0, 0, 0, 1, 0, 0, 2, 0, 0], { color: [1, 1, 1, 1, 1, 1, 1, 1] }),
    ).toThrow(RangeError);
    expect(() => lines.create(root, { capacity: 1, length: 1 })).toThrow(
      'lines.create requires `length` to be an integer of at least 2. Received 1.',
    );
  });

  it('owns only its indices when a function gives the points', () => {
    const { recording, root, frame, drawOnce } = setup();
    frame();
    const before = recording.log.alive().length;
    const ring = lines.create(root, {
      capacity: 1,
      length: 16,
      closed: true,
      point: (_line, i) => {
        'use gpu';
        const angle = (d.f32(i) / 16) * Math.PI * 2;
        return lines.ControlPoint({
          position: d.vec3f(std.cos(angle), std.sin(angle), 0),
          width: 0.1,
          color: d.vec4f(1),
          along: 0,
        });
      },
    });

    drawOnce(ring);
    expect([ring.points, ring.counts]).toEqual([undefined, undefined]);
    expect(recording.log.alive().length - before).toBe(1);
    expect(() => ring.write(0, [0, 0, 0, 1, 0, 0])).toThrow('no buffer to write');
  });

  it('lets a kernel write points and counts through its inject', async () => {
    const { recording, root, paths } = setup();
    const kernel = tgpu.computeFn({
      workgroupSize: [4],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      const point = paths.pointAt(0, gid.x);
      paths.setPoint(2, gid.x, point);
      if (gid.x === 0) paths.setCount(2, paths.held(0));
    });

    await root.pipe(paths.inject()).createComputePipeline({ compute: kernel }).initAsync();
    expect(recording.log.shaders.at(-1)!.match(/var<storage, read_write>/g)).toHaveLength(2);
  });

  it('frees what it made', () => {
    const { recording, root, frame, drawOnce } = setup();
    frame();
    const before = recording.log.alive().length;
    const paths = lines.create(root, { capacity: 2, length: 8 });
    drawOnce(paths);
    expect(recording.log.alive().length - before).toBe(3);

    paths.destroy();
    paths.destroy();
    expect(recording.log.alive().length).toBe(before);
  });
});
