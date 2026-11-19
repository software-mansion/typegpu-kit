// Share one GPU root across figures on the page
import { finish, targets, type Frame, type Surface } from '@typegpu/kit';
import { stage, type Staged } from '@typegpu/kit/hosting';
import * as web from '@typegpu/kit/web';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import type { Loading } from 'typegpu-kit-examples/contract';
import { loading } from '../../examples/files.ts';
import { COLORS, colorsOf, type Colors } from '../site/dither.ts';
import { animation, note, reducedMotion, type Animation, type LiveOptions } from './figure.ts';
import * as loupe from './loupe.ts';
import { orbitInPage } from './orbit.ts';
import { printColor } from './print.ts';

type PageGpu = { readonly root: TgpuRoot } | { readonly root?: undefined; readonly why: string };

let shared: Promise<PageGpu> | undefined;
const losing = new Set<() => void>();

/**
 * Calls `listener` whenever the page's device is lost, as iOS loses it for a page in the
 * background; the next `pageGpu` starts another.
 */
export function onLost(listener: () => void) {
  losing.add(listener);
}

/** Creates the shared root on first use, requesting optional `indirect-first-instance`. Returns the reason if initialization fails. */
export function pageGpu(): Promise<PageGpu> {
  shared ??= navigator.gpu
    ? tgpu.init({ device: { optionalFeatures: ['indirect-first-instance'] } }).then(
        (root) => {
          void root.device.lost.then((info) => {
            if (info.reason === 'destroyed') return;
            console.warn('The GPU device was lost:', info.message);
            shared = undefined;
            for (const listener of losing) listener();
          });
          return { root };
        },
        (error: unknown) => {
          console.error('WebGPU did not start:', error);
          return { why: 'This figure draws with WebGPU, which did not start in this browser.' };
        },
      )
    : Promise.resolve({ why: 'This figure draws with WebGPU, which this browser does not have.' });
  return shared;
}

/** The page's root, or undefined without WebGPU. */
export const pageRoot = () => pageGpu().then((gpu) => gpu.root);

/** Settles once `root`'s device is lost, even before this is called; never if the page destroys it. */
export const lostOf = (root: TgpuRoot): Promise<void> =>
  root.device.lost.then((info) =>
    info.reason === 'destroyed' ? new Promise<void>(() => {}) : undefined,
  );

/** Prints display colours in CSS-pixel cells on the site's ground. */
export function createPrint(root: TgpuRoot, target: Surface, cell = 1, colors: Colors = COLORS) {
  const screen = targets.create(root, target, {
    picture: { format: 'rgba8unorm', sampled: true },
  });
  const palette = root.createUniform(d.struct({ ground: d.vec3f }), {
    ground: d.vec3f(...colors.grey[0]),
  });
  const size = root.createUniform(d.vec2f);
  const written = [0, 0];
  const grade = (_color: d.v3f, uv: d.v2f) => {
    'use gpu';
    const at = std.floor((uv * size.$) / cell);
    const color = finish.picture(((at + 0.5) * cell) / size.$);
    return printColor(color, d.vec2u(at), palette.$.ground);
  };
  const final = finish.create(root, {
    format: target.format,
    source: () => screen.textures.picture,
    tone: null,
    encode: false,
    alpha: 'premultiplied',
    grade,
    settings: { vignette: 0, dither: 0 },
  });
  const destroy = () => {
    screen.destroy();
    palette.buffer.destroy();
    size.buffer.destroy();
  };
  const surface: Surface = {
    format: 'rgba8unorm',
    get width() {
      return target.width;
    },
    get height() {
      return target.height;
    },
    get pixelRatio() {
      return target.pixelRatio;
    },
    view() {
      screen.fit();
      const width = target.width / target.pixelRatio;
      const height = target.height / target.pixelRatio;
      if (written[0] !== width || written[1] !== height) {
        written[0] = width;
        written[1] = height;
        size.write(d.vec2f(width, height));
      }
      return screen.views.picture;
    },
    destroy,
    [Symbol.dispose]: destroy,
  };
  return {
    surface,
    draw(frame: Frame) {
      const pass = frame.encoder.beginRenderPass({
        colorAttachments: { view: target.view(frame) },
      });
      final.draw(pass);
      pass.end();
    },
    destroy,
  };
}

/** A figure's scene module: `createScene`, and `load` when the scene reads files. */
export interface SceneModule<Options, S extends web.Scene> {
  readonly createScene: web.CreateScene<Options, S>;
  /** Reads the scene's files into its options, only where the figure can draw. */
  readonly load?: (files: Loading) => Promise<Options>;
}

/** One scene a figure holds, from when it builds until it lets it go. */
export interface Built<S extends web.Scene> {
  readonly scene: S;
  readonly running: Animation;
}

/**
 * Runs `createScene` with the shared root, the figure's canvas and the optional loupe and print,
 * reading the module's files first when it has `load`. Returns undefined without WebGPU, saying
 * why, or once the scene has failed. `report` takes the first error from a frame, readiness or the
 * device. Destroying `running` destroys the scene before its surface.
 */
export async function build<Options, S extends web.Scene>(
  figure: HTMLElement,
  canvas: HTMLCanvasElement,
  module: SceneModule<Options, S>,
  options: Options,
  hosting: LiveOptions<S>,
  report: (error: unknown) => void,
): Promise<Built<S> | undefined> {
  const gpu = await pageGpu();
  if (!gpu.root) {
    canvas.hidden = true;
    note(figure, gpu.why);
    return undefined;
  }

  let given = options;
  if (module.load) {
    if (hosting.loading) note(figure, hosting.loading);
    given = await module.load(loading());
    if (hosting.loading) figure.querySelector(':scope > .viz-note')?.remove();
  }

  const { root } = gpu;
  const { device } = root;
  const magnified = figure.querySelector<HTMLElement>('[data-loupe]');
  const surface = web.surface(root, canvas, {
    alphaMode: 'premultiplied',
    ...hosting.surface,
    ...(magnified ? { usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC } : {}),
  });

  let time = 0;
  let failed = false;
  let handle: Animation | undefined;
  let lens: ReturnType<typeof loupe.attach> | undefined;

  // `stage` reports the first error once, from a frame, the device or `staged.fail`
  const fails = (error: unknown) => {
    failed = true;
    handle?.destroy();
    report(error);
  };

  const { onFrame } = hosting;
  const print = hosting.grade
    ? createPrint(root, surface, hosting.grade, colorsOf(figure))
    : undefined;
  let staged: Staged<S, web.Surface>;
  try {
    staged = await stage(
      root,
      surface,
      (gpuRoot, target, settings) =>
        module.createScene(gpuRoot, print?.surface ?? target, settings),
      given,
      {
        schedule: false,
        watch: device,
        onError: fails,
        onFrame: (held, frame) => {
          print?.draw(frame);
          lens?.draw(frame);
          onFrame?.(frame, held);
        },
      },
    );
  } catch (error) {
    print?.destroy();
    surface.destroy();
    canvas.hidden = true;
    throw error;
  }

  const { scene, loop, fail: stop } = staged;
  canvas.hidden = false;
  lens = magnified
    ? loupe.attach(magnified, canvas, root, surface, () => handle?.redraw())
    : undefined;
  const resized = new ResizeObserver(() => handle?.redraw());
  // A figure that isn't playing draws only when asked, with no time to ease over, so its view
  // goes straight to where it was turned
  const { view } = scene as web.Scene & { view?: { goal: web.OrbitGoal; jump(): void } };
  const orbit =
    hosting.orbit && view
      ? orbitInPage(view, canvas, () => {
          if (!handle?.playing) view.jump();
          handle?.redraw();
        })
      : undefined;

  const running = animation(figure, {
    step: hosting.animate ? (dt) => (time += dt) : undefined,
    playing: hosting.playing ?? (hosting.animate ? !reducedMotion() : false),
    play: hosting.play,

    // An uncaptured error doesn't say whose it was; a scope around each frame does
    draw() {
      device.pushErrorScope('validation');
      try {
        loop.run(time);
      } finally {
        void device.popErrorScope().then(
          (error) => error && stop(new Error(error.message)),
          () => {},
        );
      }
    },

    destroy() {
      resized.disconnect();
      orbit?.destroy();
      staged.destroy();
      print?.destroy();
      lens?.destroy();
      surface.destroy();
    },

    onError: stop,
  });
  handle = running;
  if (failed) {
    running.destroy();
    return undefined;
  }
  resized.observe(canvas);
  return { scene, running };
}
