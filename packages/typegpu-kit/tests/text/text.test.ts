import { readFileSync } from 'node:fs';
import { cameras, passes, targets } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';
import { d, std, tgpu } from 'typegpu';
import { describe, expect, it, vi } from 'vitest';
import { font, typeset, text } from '../../src/index.ts';
import { inter } from './inter.ts';

function setup() {
  const recording = recordingRoot();
  const { root } = recording;
  const size = { width: 64, height: 32 };
  const view = cameras.orbit(root, { size, target: [0, 0, 0], distance: 5 });
  const sans = font.create(root, inter);
  const screen = targets.create(root, size, { color: { format: 'rgba8unorm' } });
  const into = passes.create(root, screen, { color: passes.color });
  const labels = text.create(root, sans, {
    camera: view,
    capacity: 3,
    length: 8,
    place: text.screen,
  });
  const drawn = into.add(labels.drawing);

  const frame = () => {
    const encoder = root.createCommandEncoder();
    const pass = into.begin(encoder);
    drawn.draw(pass);
    pass.end();
    encoder.submit();
  };

  return { recording, root, size, view, sans, into, labels, frame };
}

/** Adds a text set's drawing to `into` and compiles it. */
const warmed = (
  into: { add(drawing: text.TextSet['drawing']): unknown; initAsync(): Promise<void> },
  labels: text.TextSet,
) => {
  into.add(labels.drawing);
  return into.initAsync();
};

const floats = (bytes: Uint8Array) => [...new Float32Array(bytes.slice().buffer)];

const words = (bytes: Uint8Array) => [...new Uint32Array(bytes.slice().buffer)];

/** Each quad of a label as its glyph index plus 1. */
const glyphsOf = (bytes: Uint8Array, label: number, length = 8) =>
  words(bytes)
    .slice(label * length * 4, (label + 1) * length * 4)
    .filter((_, i) => i % 4 === 2);

/** A height a custom `place` lifts its text by. */
const lift = tgpu.accessor(d.f32);

/** The docs' outline: two pixels of the rounded distance around the fill, or what it reaches. */
const outlined = (fragment: text.Fragment) => {
  'use gpu';
  const width = std.min(2, fragment.reach - 0.5);
  const fill = std.saturate(fragment.distance + 0.5);
  const outline = std.saturate(fragment.round + width + 0.5) * fragment.color.w;
  return d.vec4f(fragment.color.xyz * fill * outline, outline);
};

describe('font', () => {
  it('holds each glyph’s quad and its cell of the atlas, and the atlas’s texels', () => {
    const { recording, root, sans } = setup();
    const a = sans.glyphOf('A'.codePointAt(0)!);
    const [l, t, r, b] = inter.data.glyphs.A!.atlas!;
    const { width, height } = inter.data.atlas;

    const glyph = floats(recording.bytes(root.unwrap(sans.glyphs))).slice(a * 8, a * 8 + 8);
    expect(glyph).toEqual(
      [...inter.data.glyphs.A!.plane!, l / width, t / height, r / width, b / height].map(
        Math.fround,
      ),
    );
    expect(sans.atlas.props.size).toEqual([width, height]);
  });

  it('takes the missing glyph for a character it lacks, and kerns only the pairs the font does', () => {
    const { sans } = setup();
    const missing = Object.keys(inter.data.glyphs).length;

    expect(sans.glyphOf(0x4e00)).toBe(missing);
    expect(sans.inked(sans.glyphOf(' '.codePointAt(0)!))).toBe(false);
    expect(sans.kerning(sans.glyphOf(65), sans.glyphOf(86))).toBe(inter.data.kerning.AV);
    expect(sans.kerning(sans.glyphOf(65), missing)).toBe(0);
  });

  it('refuses a bake of another version, or an atlas of the wrong size', () => {
    const { root } = setup();

    expect(() => font.create(root, { ...inter, data: { ...inter.data, version: 2 as 1 } })).toThrow(
      'font.create requires bake version 1. Received version 2. Bake the font again.',
    );
    expect(() => font.create(root, { ...inter, atlas: inter.atlas.subarray(4) })).toThrow(
      RangeError,
    );
  });

  it('measures with metrics alone, as with the font', () => {
    const { sans } = setup();
    const measured = font.metrics(inter.data);
    const options = { em: 10, align: 'left' } as const;
    const withFont = typeset.create(8).set(sans, 'AVa', options);
    const withMetrics = typeset.create(8).set(measured, 'AVa', options);

    expect(measured.glyphOf(65)).toBe(sans.glyphOf(65));
    expect([...withMetrics.pens]).toEqual([...withFont.pens]);
    expect(withMetrics.width).toBe(withFont.width);
  });

  it('reads a bake from its text or its parsed JSON, and names what is wrong with one', () => {
    const json = readFileSync(new URL('../../fonts/inter.json', import.meta.url), 'utf8');
    const read = font.read(JSON.parse(json), inter.atlas.slice().buffer);

    expect(read.data).toEqual(inter.data);
    expect(read.atlas.byteLength).toBe(inter.atlas.byteLength);
    expect(() => font.read({ ...inter.data, version: 2 }, inter.atlas)).toThrow(
      'font.read requires bake version 1. Received version 2. Bake the font again.',
    );
    expect(() => font.read('{}', inter.atlas)).toThrow(
      new TypeError(
        'font.read requires 1 at `version`. Received undefined. Use JSON written by typegpu-kit-bake-font.',
      ),
    );
    const broken = { ...inter.data, glyphs: { ...inter.data.glyphs, A: { advance: '1' } } };
    expect(() => font.read(broken, inter.atlas)).toThrow(
      'font.read requires a number at `glyphs.A.advance`',
    );
    const flat = { ...inter.data, glyphs: { A: { advance: 1, plane: [0, 0, 1] } } };
    expect(() => font.read(flat, inter.atlas)).toThrow('font.read requires both');
    const plane: number[] = [];
    plane[3] = 1;
    const sparse = { advance: 1, plane, atlas: [0, 0, 1, 1] };
    expect(() => font.read({ ...inter.data, glyphs: { A: sparse } }, inter.atlas)).toThrow(
      'font.read requires a number at `glyphs.A.plane[0]`',
    );
    expect(() => font.read(json, inter.atlas.subarray(4))).toThrow(RangeError);
  });
});

describe('text', () => {
  it('draws every label in one draw of one pipeline', () => {
    const { recording, frame } = setup();
    frame();

    expect(recording.log.pipelines).toBe(1);
    expect(recording.log.events.filter((event) => event.startsWith('draw'))).toEqual(['draw']);
  });

  it('creates and uploads nothing in a steady frame', () => {
    const { recording, labels, frame } = setup();
    labels.write(0, 'Score', { em: 16 });
    frame();
    const before = recording.log.counts();
    const writes = recording.log.writes.length;

    frame();
    frame();
    expect(recording.log.counts()).toEqual(before);
    expect(recording.log.writes.length).toBe(writes);
  });

  it('uploads only the label it writes, and clears what longer text left', () => {
    const { recording, sans, labels } = setup();
    const writes = recording.log.writes.length;

    labels.write(1, 'Hi there', { em: 16 });
    const first = recording.log.writes.slice(writes);
    expect(first.map(({ offset, size }) => [offset, size])).toEqual([[8 * 16, 7 * 16]]);

    const buffer = first[0].buffer;
    const index = (character: string) => sans.glyphOf(character.codePointAt(0)!) + 1;
    expect(glyphsOf(recording.bytes(buffer), 1)).toEqual([...'Hithere'.split('').map(index), 0]);

    labels.write(1, 'Ok', { em: 16 });
    expect(recording.log.writes.at(-1)).toMatchObject({ offset: 8 * 16, size: 7 * 16 });
    expect(glyphsOf(recording.bytes(buffer), 1)).toEqual([
      index('O'),
      index('k'),
      0,
      0,
      0,
      0,
      0,
      0,
    ]);
    expect(glyphsOf(recording.bytes(buffer), 0)).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);

    labels.write(1, 'Ok', { em: 16 });
    expect(recording.log.writes.at(-1)).toMatchObject({ offset: 8 * 16, size: 2 * 16 });
  });

  it('writes each glyph’s pen and size from the layout it returns, as typeset lays it out', () => {
    const { recording, sans, labels } = setup();
    const laid = labels.write(2, 'AV', { em: 20 });

    const { buffer } = recording.log.writes.at(-1)!;
    const quads = floats(recording.bytes(buffer)).slice(2 * 8 * 4, 2 * 8 * 4 + 8);
    const measured = typeset.create(8).set(sans, 'AV', { em: 20 });
    expect([laid.count, laid.lines, laid.width, laid.height]).toEqual([
      2,
      1,
      measured.width,
      measured.height,
    ]);

    expect([quads[0], quads[1], quads[3]]).toEqual(
      [laid.pens[0], laid.pens[1], 20].map(Math.fround),
    );
    expect([quads[4], quads[5], quads[7]]).toEqual(
      [laid.pens[2], laid.pens[3], 20].map(Math.fround),
    );
  });

  it('starts every label white at the origin, and moves one with patch', () => {
    const { recording, root, labels } = setup();
    const rows = () => floats(recording.bytes(root.unwrap(labels.labels)));

    expect(rows().slice(0, 8)).toEqual([0, 0, 0, 0, 1, 1, 1, 1]);

    labels.labels.patch({ 1: { position: d.vec3f(1, 2, 3) } });
    expect(rows().slice(8, 16)).toEqual([1, 2, 3, 0, 1, 1, 1, 1]);
    expect(recording.log.writes.at(-1)!.size).toBeLessThanOrEqual(16);
  });

  it('places with the mode it is given, reading the camera and viewport only when it needs them', async () => {
    const { recording, root, view, sans, into } = setup();
    const options = { capacity: 1, length: 1 } as const;

    await warmed(into, text.create(root, sans, { ...options, place: text.screen }));
    expect(recording.log.shaders.at(-1)).not.toMatch(/viewProjection/);

    await warmed(into, text.create(root, sans, { camera: view, ...options, place: text.pinned }));
    expect(recording.log.shaders.at(-1)).toMatch(/viewProjection/);

    const raised = text.create(root, sans, {
      camera: view,
      ...options,
      place: tgpu
        .fn((position: d.v3f, offset: d.v2f) => {
          'use gpu';
          return text.facing(position + d.vec3f(0, lift.$, 0), offset);
        })
        .pipe((cfg) => cfg.with(lift, 2)),
    });
    await warmed(into, raised);
    expect(recording.log.shaders.at(-1)).toMatch(/\+ vec3f\(0, 2/);

    const swell = root.createReadonly(d.arrayOf(d.f32, 1));
    const swelled = text.create(root, sans, {
      camera: view,
      ...options,
      place: (position, offset, glyph) => {
        'use gpu';
        return text.pinned(position, offset * swell.$[glyph.label]);
      },
    });
    await warmed(into, swelled);
    expect(recording.log.shaders.at(-1)).toMatch(
      /Placed\(label, k, \(\*quad\)\.pen, \(\*quad\)\.em\)/,
    );
  });

  it('tells shade which glyph of the label a pixel is of, and where in the label it is', async () => {
    const { recording, root, view, sans, into } = setup();
    const shown = root.createUniform(d.f32);

    await warmed(
      into,
      text.create(root, sans, {
        camera: view,
        capacity: 1,
        length: 4,
        shade: (fragment) => {
          'use gpu';
          const alpha = std.saturate(shown.$ - d.f32(fragment.glyph)) * fragment.local.x;
          return d.vec4f(fragment.color.xyz * alpha, alpha);
        },
      }),
    );
    expect(recording.log.shaders.at(-1)).toMatch(/@location\(\d\) @interpolate\(flat\) glyph: u32/);
    expect(recording.log.shaders.at(-1)).toMatch(/local: vec2f/);
  });

  it('draws into the translucency targets in place of blending, as lines do', () => {
    const { root, view, sans } = setup();
    const { drawing } = text.create(root, sans, {
      camera: view,
      capacity: 1,
      length: 1,
      blend: 'translucent',
    });

    expect(drawing.translucent).toBe(true);
    expect(drawing.blend).toBeUndefined();
  });

  it('blends over by default or adds, under its label or `text`', async () => {
    const { root, size, view, sans } = setup();
    const created = vi.spyOn(root.device, 'createRenderPipelineAsync');
    const options = { capacity: 1, length: 1 } as const;
    const glow = text.create(root, sans, { camera: view, ...options, blend: 'add', label: 'glow' });

    expect(text.create(root, sans, { ...options, camera: view }).drawing.label).toBe('text');
    expect(glow.drawing.label).toBe('glow');
    const screen = targets.create(root, size, { color: { format: 'rgba8unorm' } });
    await warmed(passes.create(root, screen, { color: passes.color }), glow);
    const [descriptor] = created.mock.calls.map(([each]) => each);
    expect([...descriptor!.fragment!.targets][0]!.blend!.color).toEqual({
      srcFactor: 'one',
      dstFactor: 'one',
      operation: 'add',
    });
  });

  it('shades with a function of the fragment, such as an outline from the fourth channel', async () => {
    const { recording, root, view, sans, into } = setup();

    await warmed(
      into,
      text.create(root, sans, {
        camera: view,
        capacity: 1,
        length: 1,
        shade: outlined,
      }),
    );
    expect(recording.log.shaders.at(-1)).toMatch(/fn outlined\(/);
  });

  it('tests the depth of its passes without writing it, and blends over', async () => {
    const { root, size, view, sans } = setup();
    const created = vi.spyOn(root.device, 'createRenderPipelineAsync');
    const msaa = targets.create(root, size, {
      color: { format: 'rgba8unorm', samples: 4 },
      depth: { format: 'depth24plus', samples: 4 },
    });
    const behind = text.create(root, sans, {
      camera: view,
      capacity: 1,
      length: 1,
    });
    await warmed(passes.create(root, msaa, { color: passes.color, depth: passes.depth }), behind);

    const [descriptor] = created.mock.calls.map(([each]) => each);
    expect(descriptor!.depthStencil).toEqual({
      format: 'depth24plus',
      depthWriteEnabled: false,
      depthCompare: 'less',
    });
    expect(descriptor!.multisample).toMatchObject({ count: 4 });
    expect([...descriptor!.fragment!.targets][0]!.blend!.color).toEqual({
      srcFactor: 'one',
      dstFactor: 'one-minus-src-alpha',
      operation: 'add',
    });
  });

  it('refuses a label it does not have, and sizes below 1', () => {
    const { root, view, sans, labels } = setup();

    expect(() => labels.write(3, 'A', { em: 1 })).toThrow(
      new RangeError('The text set has no label 3; there are 3.'),
    );
    expect(() =>
      text.create(root, sans, {
        camera: view,
        capacity: 1,
        length: 0,
      }),
    ).toThrow(
      new RangeError('text.create requires `length` to be an integer of at least 1. Received 0.'),
    );
  });
});
