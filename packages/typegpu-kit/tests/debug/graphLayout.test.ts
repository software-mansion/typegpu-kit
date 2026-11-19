import { describe, expect, it } from 'vitest';
import {
  createGraphLayout,
  palette,
  premultiplied,
  rectFloats,
} from '../../src/debug/graphLayout.ts';
import type { Bar } from '../../src/debug/recorder.ts';

const sum = (parts: Record<string, number>) => Object.values(parts).reduce((a, b) => a + b, 0);

/** A column averaging `frames` frames that each spent these milliseconds on the GPU and CPU. */
function bar(time: number, gpu: Record<string, number>, cpu: Record<string, number>): Bar {
  const frames = 2;
  const segments = (parts: Record<string, number>) =>
    Object.entries(parts).map(([name, ms]) => ({ name, ms: ms * frames, calls: frames }));
  return {
    time,
    frames: { gpu: frames, cpu: frames },
    segments: { gpu: segments(gpu), cpu: segments(cpu) },
    peak: { gpu: sum(gpu), cpu: sum(cpu) },
  };
}

function rectsOf(rects: Float32Array, count: number) {
  return Array.from({ length: count }, (_, index) => {
    const [x, y, w, h, r, g, b, a, top, bottom] = rects.subarray(
      index * rectFloats,
      (index + 1) * rectFloats,
    );
    return { x, y, w, h, color: [r, g, b, a], top, bottom };
  });
}

describe('graph layout', () => {
  it('fits as many columns as the web graph in each section', () => {
    const layout = createGraphLayout(64);
    expect(layout.columns(360)).toBe(27);
    expect(layout.columns(1000)).toBe(48);
  });

  it('stacks each column’s segments in their colours, with the newest on the right', () => {
    const layout = createGraphLayout(4096);
    const bars = [
      bar(0, { shade: 2 }, { frame: 1 }),
      bar(100, { shade: 4, blur: 2 }, { frame: 3 }),
    ];
    const { rects, count, labels } = layout.layout(bars, 360, 150, 1);
    const drawn = rectsOf(rects, count);

    const shade = premultiplied(layout.colorOf('shade'));
    const blur = premultiplied(layout.colorOf('blur'));
    expect(layout.colorOf('shade')).toBe(palette[0]);
    const columns = drawn.filter(({ w }) => w === 4);
    const newest = columns.filter(({ x }) => x === 10 + 137 - 5);
    expect(newest.length).toBeGreaterThan(0);
    const shadeRect = newest.find(({ color }) =>
      color.every((v, i) => Math.abs(v - shade[i]) < 1e-6),
    );
    const blurRect = newest.find(({ color }) =>
      color.every((v, i) => Math.abs(v - blur[i]) < 1e-6),
    );
    expect(shadeRect && blurRect && blurRect.y + blurRect.h).toBeCloseTo(shadeRect!.y, 5);

    expect(labels.map(({ text }) => text)).toEqual(
      expect.arrayContaining(['gpu 4.00ms at vsync', 'cpu 2.00ms', 'shade', 'blur', 'frame']),
    );
  });

  it('notes a device without timestamp queries in the GPU section', () => {
    const layout = createGraphLayout(256);
    const { labels } = layout.layout([], 360, 150, 1, {
      note: 'The device does not support timestamp-query.',
    });
    const note = labels.find(({ text }) => text.startsWith('The device'));
    expect(note && note.y).toBeLessThan(75);
  });

  it('links each legend row to its segment and lists a hovered column', () => {
    const layout = createGraphLayout(4096);
    const bars = [bar(0, { shade: 2 }, { frame: 1 }), bar(100, { blur: 4 }, { frame: 3 })];
    const live = layout.layout(bars, 360, 150, 1);
    const links = rectsOf(live.rects, live.count).filter(({ top, bottom }) => top || bottom);
    // Two GPU rows and one CPU row, each from the graph's edge to its swatch
    expect(links.map(({ x, w }) => [x, w])).toEqual([
      [148, 33],
      [148, 33],
      [148, 33],
    ]);

    // The older column, under the pointer in the GPU section, dims the newest and fills the legends
    const older = layout.columnAt(10 + 137 - 7, 40);
    expect(older).toBe(1);
    expect(layout.columnAt(5, 40)).toBeUndefined();
    const { rects, count, labels } = layout.layout(bars, 360, 150, 1, { hovered: older });
    const texts = labels.map(({ text }) => text);
    expect(texts).toEqual(expect.arrayContaining(['gpu 2.00ms at vsync', 'shade', 'cpu 1.00ms']));
    expect(texts).not.toContain('blur');
    const blur = premultiplied(layout.colorOf('blur'));
    const newest = rectsOf(rects, count).find(
      ({ x, w, color }) =>
        x === 10 + 137 - 5 && w === 4 && Math.abs(color[0] - blur[0] * 0.55) < 1e-6,
    );
    expect(newest).toBeDefined();
  });

  it('folds the smallest segments into one legend row and stops at capacity', () => {
    const many = Object.fromEntries(Array.from({ length: 12 }, (_, i) => [`pass ${i}`, i + 1]));
    const layout = createGraphLayout(30);
    const { count, labels } = layout.layout([bar(0, many, { frame: 1 })], 360, 150, 1);

    expect(count).toBe(30);
    expect(labels.some(({ text }) => /^\d+ more$/.test(text))).toBe(true);
  });

  it('premultiplies hex and rgba colours', () => {
    expect(premultiplied('#ff0000')).toEqual([1, 0, 0, 1]);
    expect(premultiplied('rgba(255, 255, 255, 0.5)')).toEqual([0.5, 0.5, 0.5, 0.5]);
  });
});
