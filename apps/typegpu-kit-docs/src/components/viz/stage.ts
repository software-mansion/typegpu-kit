// The lit stage most 3D figures share: an orbit camera, HDR targets, passes and a finish
import { srgbToLinear } from '@typegpu/color';
import {
  bloom,
  cameras,
  finish,
  passes,
  targets,
  type Frame,
  type Injectable,
  type Surface,
} from '@typegpu/kit';
import { d, type TgpuRoot } from 'typegpu';
import { quat } from 'wgpu-matrix';

/** The figures' sky: a cool top over a warm ground. */
export const SKY = { skyTop: [0.32, 0.38, 0.5], skyBottom: [0.1, 0.09, 0.08] } as const;

/** A colour picked in sRGB, as the shaders take it. */
export const linear = (r: number, g: number, b: number) => srgbToLinear(d.vec3f(r, g, b));

/** A rotation of `angle` radians around `axis`, as instances take it. */
export const turn = (axis: readonly [number, number, number], angle: number) => {
  const [x, y, z, w] = quat.fromAxisAngle(axis, angle);
  return [x!, y!, z!, w!] as const;
};

/**
 * The world point under `u`, `v`, from 0 to 1 across `surface` with y down, for a flat camera
 * `height` tall centred on the origin.
 */
export const flatPoint = (surface: Surface, height: number, u: number, v: number) => {
  const aspect = surface.width / Math.max(surface.height, 1);
  return [(u - 0.5) * height * aspect, (0.5 - v) * height] as const;
};

const HDR = 'rgba16float';

interface LitStageOptions {
  /** The orbit camera's options, sized by the surface. */
  readonly camera: Omit<cameras.OrbitOptions, 'size'>;
  /** Samples a pixel, resolved into the HDR picture. Default 1. */
  readonly samples?: number;
  /** Default opaque black. */
  readonly clear?: readonly [number, number, number, number];
  /** Adds a bloom of the HDR picture to the finish, which `present` records. Default false. */
  readonly glow?: boolean;
  /** The finish's options beside its format, source and glow. Default the finish's defaults. */
  readonly finish?: Omit<finish.Options, 'format' | 'source' | 'glow'>;
}

/** Creates an orbit camera, HDR colour and depth targets and a finish onto `surface`. */
export function litStage(root: TgpuRoot, surface: Surface, options: LitStageOptions) {
  const { samples = 1, clear = [0, 0, 0, 1] } = options;
  const view = cameras.orbit(root, { size: surface, ...options.camera });
  const writes = { color: passes.color, depth: passes.depth } as const;

  const resolved =
    samples > 1
      ? targets.create(root, surface, {
          color: { format: HDR, samples },
          depth: { format: 'depth24plus', samples },
          hdr: { format: HDR, sampled: true },
        })
      : undefined;
  const single = resolved
    ? undefined
    : targets.create(root, surface, {
        color: { format: HDR, sampled: true },
        depth: { format: 'depth24plus', transient: true },
      });
  const screen = resolved ?? single!;
  const hdr = () => (resolved ? resolved.textures.hdr : single!.textures.color);
  const glow = options.glow ? bloom.create(root, { source: hdr }) : undefined;
  const final = finish.create(root, {
    format: surface.format,
    source: hdr,
    ...(glow ? { glow } : {}),
    ...options.finish,
  });

  return {
    view,
    screen,
    final,
    glow,
    /** The HDR picture the passes draw, which the finish reads. */
    hdr,

    /** Passes through the stage's camera, lit by `light`. A new lighting needs new passes. */
    lit(light: Injectable) {
      const described = resolved
        ? passes.create(root, resolved, writes, { resolve: { color: 'hdr' }, clear })
        : passes.create(root, single!, writes, { clear });
      return described.pipe(view.inject()).beneath(light.inject());
    },

    /** Records the glow, if any, and draws the finish onto the surface. */
    present(frame: Frame) {
      glow?.record(frame);
      const pass = frame.encoder.beginRenderPass({
        colorAttachments: { view: surface.view(frame) },
      });
      final.draw(pass);
      pass.end();
    },

    destroy() {
      screen.destroy();
      glow?.destroy();
    },
  };
}
