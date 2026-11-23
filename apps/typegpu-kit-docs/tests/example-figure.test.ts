import { frames, type Frame } from '@typegpu/kit';
import { offscreenSurface, recordingRoot } from '@typegpu/kit/testing';
import { describe, expect, it, vi } from 'vitest';
import { examples } from '../src/examples/examples.ts';
import type { Settings } from 'typegpu-kit-examples/settings.ts';
import {
  build,
  controlsOf,
  optionsOf,
  start,
  writerFor,
  type Embed,
} from '../src/components/viz/example.ts';
import { connect } from '../src/examples/host.ts';
import * as loupe from '../src/components/viz/loupe.ts';
import { driven, fromDisk, recorded } from 'typegpu-kit-examples/testing';

/** Settings as an example's scene gives them, with a record of what it wrote. */
function settingsWith() {
  const written: (string | undefined)[] = [];
  const values: { speed: number; view: string; taa: boolean; hour: number; size: number } = {
    speed: 0.5,
    view: 'picture',
    taa: true,
    hour: 12,
    size: 512,
  };
  const settings: Settings<typeof values> = {
    values,
    knobs: {
      speed: { min: 0, max: 1, step: 0.01 },
      view: { options: ['picture', 'motion'] },
      taa: { label: 'TAA', disabled: 'no device' },
      hour: { min: 0, max: 24, step: 1, live: true },
      size: { options: [256, 512] },
    },
    write: (key) => void written.push(key),
    actions: { reseed: () => {} },
    readouts: { frames: () => 3 },
  };
  return { settings, written };
}

describe('the controls', () => {
  it('shows a number on a slider, a name or listed number as a choice, a flag as a toggle, an action as a button and a readout', () => {
    const { settings } = settingsWith();
    const controls = controlsOf('test', settings, [
      'view',
      'reseed',
      'frames',
      'taa',
      'speed',
      'hour',
      'size',
    ]);

    expect(controls).toEqual([
      { kind: 'choice', name: 'view', label: 'view', live: false, options: ['picture', 'motion'] },
      { kind: 'action', name: 'reseed', label: 'reseed' },
      { kind: 'readout', name: 'frames', label: 'frames' },
      { kind: 'toggle', name: 'taa', label: 'TAA', live: false, disabled: 'no device' },
      { kind: 'slider', name: 'speed', label: 'speed', live: false, min: 0, max: 1, step: 0.01 },
      { kind: 'slider', name: 'hour', label: 'hour', live: true, min: 0, max: 24, step: 1 },
      { kind: 'choice', name: 'size', label: 'size', live: false, options: [256, 512] },
    ]);
    expect(controlsOf('test', settings, [])).toEqual([]);
  });

  it('names what the settings have when asked for what they lack', () => {
    const { settings } = settingsWith();
    expect(() => controlsOf('test', settings, ['tea'])).toThrow(
      'test has no setting, action or readout tea; it has speed, view, taa, hour, size, reseed, frames.',
    );
    expect(() => controlsOf('test', undefined, ['speed'])).toThrow(
      'test has no setting, action or readout speed; it has none.',
    );
  });
});

it.each(['touch', 'pen'])(
  'leaves %s strokes to scrolling until tapped, and disarms off screen',
  (pointerType) => {
    const captured = new Set<number>();
    const attributes = new Set<string>();
    const canvas = Object.assign(new EventTarget(), {
      style: { touchAction: '' },
      clientWidth: 64,
      clientHeight: 36,
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 64, height: 36 }),
      setPointerCapture: vi.fn((pointer: number) => void captured.add(pointer)),
      hasPointerCapture: (pointer: number) => captured.has(pointer),
      toggleAttribute: (name: string, on: boolean) =>
        on ? attributes.add(name) : attributes.delete(name),
      hasAttribute: (name: string) => attributes.has(name),
      getAttribute: () => null,
      setAttribute: (name: string) => void attributes.add(name),
    }) as unknown as HTMLCanvasElement;
    const observers: ((entries: { isIntersecting: boolean }[]) => void)[] = [];
    const disconnect = vi.fn();
    vi.stubGlobal(
      'IntersectionObserver',
      class {
        constructor(change: (entries: { isIntersecting: boolean }[]) => void) {
          observers.push(change);
        }
        observe() {}
        disconnect = disconnect;
      },
    );
    const drag = vi.fn();
    const listening = connect(
      canvas,
      { pointing: { move() {}, drag }, frame() {}, destroy() {} },
      { page: true },
    );
    const point = (type: string, options: Partial<PointerEvent> = {}) =>
      canvas.dispatchEvent(
        Object.assign(new Event(type), {
          pointerId: 1,
          pointerType,
          isPrimary: true,
          button: 0,
          clientX: 32,
          clientY: 18,
          ...options,
        }),
      );
    try {
      point('pointerdown');
      point('pointermove', { clientY: 60 });
      point('pointercancel');
      expect(drag).not.toHaveBeenCalled();
      expect(canvas.hasAttribute('data-armed')).toBe(false);

      point('pointerdown');
      point('pointerup');
      expect(canvas.hasAttribute('data-armed')).toBe(true);
      expect(canvas.style.touchAction).toBe('none');
      expect(drag).not.toHaveBeenCalled();
      point('pointerdown');
      point('pointermove', { clientX: 40 });
      point('pointermove', { pointerId: 2, isPrimary: false });
      expect(drag.mock.calls.map((call) => call[2])).toEqual([true, false]);

      expect(observers).toHaveLength(1);
      observers[0]([{ isIntersecting: false }, { isIntersecting: true }]);
      expect(canvas.hasAttribute('data-armed')).toBe(true);
      observers[0]([{ isIntersecting: false }]);
      expect(canvas.hasAttribute('data-armed')).toBe(false);
      expect(canvas.style.touchAction).not.toBe('none');
      point('pointerdown');
      expect(drag).toHaveBeenCalledTimes(2);
    } finally {
      listening[Symbol.dispose]();
      expect(disconnect).toHaveBeenCalledOnce();
      vi.unstubAllGlobals();
    }
  },
);

/** Resolves once pending promises have settled. */
const settled = () => new Promise((resolve) => setTimeout(resolve));

describe('a control that changes a setting', () => {
  it('writes the latest value once the write before it ends, and stops at a scene let go', async () => {
    const { settings } = settingsWith();
    const sent: number[] = [];
    /** Ends the running write, failing it with `error` if given. */
    let finish: ((error?: Error) => void) | undefined;
    settings.write = () => {
      sent.push(settings.values.speed);
      return new Promise<void>((resolve, reject) => {
        finish = (error) => (error ? reject(error) : resolve());
      });
    };
    let held: Settings | undefined = settings as unknown as Settings;
    const changed = vi.fn();
    const onError = vi.fn();
    const write = writerFor(() => held, changed, onError);

    write('speed', 0.1);
    write('speed', 0.2);
    write('speed', 0.3);
    expect(sent).toEqual([0.1]);
    finish?.();
    await settled();
    expect(sent).toEqual([0.1, 0.3]);
    finish?.();
    await settled();
    expect(changed).toHaveBeenCalledOnce();

    // Let go while its write runs, the scene gets no other, and its failure goes unsaid
    write('speed', 0.4);
    write('speed', 0.5);
    held = undefined;
    finish?.(new Error('The scene was destroyed.'));
    await settled();
    expect(sent).toEqual([0.1, 0.3, 0.4]);
    expect(onError).not.toHaveBeenCalled();
  });
});

describe('the values a figure starts from', () => {
  it('sets and writes each', async () => {
    const { settings, written } = settingsWith();
    await start('test', settings, { view: 'motion', speed: 1 });

    expect(settings.values).toMatchObject({ view: 'motion', speed: 1 });
    expect(written).toEqual(['view', 'speed']);
  });

  it("leaves a disabled knob's value as it is", async () => {
    const { settings, written } = settingsWith();
    await start('test', settings, { taa: false });

    expect(settings.values.taa).toBe(true);
    expect(written).toEqual([]);
  });

  it('throws on a name it lacks, another type, a number out of range and a name not offered', async () => {
    const { settings, written } = settingsWith();

    await expect(start('test', settings, { sped: 1 })).rejects.toThrow('test has no setting sped');
    await expect(start('test', settings, { speed: 'fast' })).rejects.toThrow(
      `test's speed is a number, not "fast".`,
    );
    await expect(start('test', settings, { speed: 2 })).rejects.toThrow(
      `test's speed goes from 0 to 1, not 2.`,
    );
    await expect(start('test', settings, { view: 'depth' })).rejects.toThrow(
      `test's view is one of picture, motion, not depth.`,
    );
    await expect(start('test', settings, { size: 300 })).rejects.toThrow(
      `test's size is one of 256, 512, not 300.`,
    );
    expect(written).toEqual([]);
  });
});

it("runs an example with its card's options, the figure's over them", () => {
  expect(optionsOf({ snow: 8000, motes: 6 }, { snow: 200 })).toEqual({ snow: 200, motes: 6 });
  expect(optionsOf(undefined, { snow: 200 })).toEqual({ snow: 200 });
  expect(optionsOf({ snow: 8000 }, undefined)).toEqual({ snow: 8000 });
});

describe('the loupe', () => {
  it('keeps its square inside the canvas', () => {
    expect(loupe.cornerOf(40, 64, 16)).toBe(32);
    expect(loupe.cornerOf(2, 64, 16)).toBe(0);
    expect(loupe.cornerOf(63, 64, 16)).toBe(48);
  });

  it('copies the square and draws it in each frame, creates nothing in a steady one and frees what it made', () => {
    const { root, log } = recordingRoot();
    const source = offscreenSurface(root, { width: 64, height: 40, format: 'bgra8unorm' });
    const target = offscreenSurface(root, { width: 32, height: 32, format: 'bgra8unorm' });
    const before = log.alive();

    const lens = loupe.create(root, source.format, 16);
    const loop = frames.create(root, (frame) => lens.draw(frame, source.texture, target, 60, 2));
    loop.run(0);
    const steady = log.counts();

    expect(recorded(log, () => loop.run(1 / 60))).toEqual([
      'copy-texture',
      'begin-render',
      'draw',
      'end-render',
      'submit',
    ]);
    expect(log.counts()).toEqual(steady);

    // Smaller than the square, a source draws nothing.
    const small = offscreenSurface(root, { width: 8, height: 8, format: 'bgra8unorm' });
    const smallLoop = frames.create(root, (frame) => lens.draw(frame, small.texture, target, 4, 4));
    expect(recorded(log, () => smallLoop.run(0))).toEqual(['submit']);
    small.destroy();

    loop.destroy();
    lens.destroy();
    expect(log.alive()).toEqual(before);
  });
});

/** Builds `embed` as an ExampleFigure does, on a recording root, and runs it until it is ready. */
async function embedded(embed: Embed, texels?: number) {
  const recording = recordingRoot({ features: ['indirect-first-instance'] });
  const { root } = recording;
  const surface = offscreenSurface(root, { width: 64, height: 40, format: 'bgra8unorm' });
  const target = offscreenSurface(root, { width: 32, height: 32, format: 'bgra8unorm' });

  const built = await build(embed, root, surface, fromDisk);
  const { scene } = built;
  const lens = texels === undefined ? undefined : loupe.create(root, surface.format, texels);
  const loop = frames.create(root, (frame: Frame) => {
    scene.frame(frame);
    lens?.draw(frame, surface.texture, target, 32, 20);
  });

  const { run, warmUp } = driven(loop, scene.ready);
  await warmUp();

  const destroy = () => {
    loop.destroy();
    lens?.destroy();
    scene.destroy?.();
    target.destroy();
    surface.destroy();
  };
  return { ...recording, built, run, destroy };
}

describe('an example in a figure', () => {
  it('starts the lattice from its values, with its controls and a loupe', async () => {
    const embed: Embed = {
      key: 'lattice',
      options: {},
      controls: ['taa', 'speed', 'view'],
      values: { view: 'motion', taa: false },
    };
    const { built, log, run, destroy } = await embedded(embed, 24);

    expect(built.controls.map((control) => control.kind)).toEqual(['toggle', 'slider', 'choice']);
    expect(built.scene.settings?.values).toMatchObject({ view: 'motion', taa: false });

    await run(4);
    const counts = log.counts();
    const events = await recorded(log, () => run(2));
    expect(log.counts()).toEqual(counts);
    expect(events.filter((event) => event === 'copy-texture')).toHaveLength(2);
    expect(events.filter((event) => event === 'submit')).toHaveLength(2);

    destroy();
  });

  it('fails on a control the example lacks', async () => {
    const { root } = recordingRoot();
    const surface = offscreenSurface(root, { width: 64, height: 40, format: 'bgra8unorm' });
    const embed: Embed = { key: 'flow-field', options: {}, controls: ['tea'], values: {} };

    await expect(build(embed, root, surface, fromDisk)).rejects.toThrow(
      'flow-field has no setting, action or readout tea; it has speed, scale, pointSize, paused, reseed, settings writes.',
    );
  });
});

/** The value of each attribute of each `<ExampleFigure>` in `source`, a page's MDX. */
function figuresIn(source: string): Record<string, unknown>[] {
  const figures: Record<string, unknown>[] = [];
  const tag = '<ExampleFigure';

  for (let at = source.indexOf(tag); at !== -1; at = source.indexOf(tag, at + tag.length)) {
    const attributes: Record<string, unknown> = {};
    let i = at + tag.length;

    const skip = () => {
      while (/\s/.test(source[i] ?? '')) i++;
    };

    for (skip(); source[i] !== '>' && !source.startsWith('/>', i); skip()) {
      const name = /^[A-Za-z]+/.exec(source.slice(i))?.[0];
      if (!name) throw new Error(`Can't read the ExampleFigure at ${source.slice(at, i + 20)}`);
      i += name.length;
      if (source[i] !== '=') {
        attributes[name] = true;
        continue;
      }

      i += 1;
      if (source[i] === '"') {
        const end = source.indexOf('"', i + 1);
        attributes[name] = source.slice(i + 1, end);
        i = end + 1;
        continue;
      }

      // An expression in braces, which may hold strings with braces of their own.
      const from = i;
      let depth = 0;
      let quote: string | undefined;
      do {
        const char = source[i];
        if (quote) {
          if (char === '\\') i++;
          else if (char === quote) quote = undefined;
        } else if (`'"\``.includes(char)) quote = char;
        else if (char === '{') depth++;
        else if (char === '}') depth--;
        i++;
      } while (depth > 0);
      attributes[name] = new Function(`return (${source.slice(from + 1, i - 1)});`)();
    }

    figures.push(attributes);
  }
  return figures;
}

const pages = import.meta.glob<string>('../src/content/docs/**/*.mdx', {
  eager: true,
  query: '?raw',
  import: 'default',
});

const figures = Object.entries(pages).flatMap(([path, source]) =>
  figuresIn(source).map((attributes, i): [string, Record<string, unknown>] => [
    `${path.replace('../src/content/docs/', '')} #${i + 1}`,
    attributes,
  ]),
);

it('reads the attributes of a figure in MDX', () => {
  const source = `<ExampleFigure example="benthos" controls={['a', "b}"]} options={{ snow: 9 }} loupe />`;
  expect(figuresIn(source)).toEqual([
    { example: 'benthos', controls: ['a', 'b}'], options: { snow: 9 }, loupe: true },
  ]);
});

describe.runIf(figures.length > 0).each(figures)('the figure in %s', (_name, attributes) => {
  it('runs its example with its controls, values and loupe, and frees it', async () => {
    const key = attributes.example as string;
    const example = examples.find((each) => each.key === key);
    expect(example, `an example ${key}`).toBeDefined();

    const texels = attributes.loupe ? loupe.TEXELS : undefined;
    const embed: Embed = {
      key,
      options: optionsOf(example?.card, attributes.options as Record<string, unknown>),
      controls: (attributes.controls as string[] | undefined) ?? [],
      values: (attributes.values as Embed['values'] | undefined) ?? {},
    };

    const { run, destroy } = await embedded(embed, texels);
    await run(3);
    destroy();
  });
});
