import {
  ao,
  bloom,
  finish,
  lighting,
  passes,
  pipelines,
  shading,
  taa,
  targets,
  tone,
  type cameras,
  type Frame,
  type instances,
  type Surface,
} from '@typegpu/kit';
import { d, std, type TgpuRoot } from 'typegpu';

/** The stages of the look. Each turns off without rebuilding anything. */
export interface Stages {
  shadows: boolean;
  occlusion: boolean;
  taa: boolean;
  bloom: boolean;
}

export interface LookOptions {
  /** How far the sun's shadow reaches, a little past the farthest prop the view shows. */
  readonly far: number;
  /** The size of the gaps the occlusion darkens, in world units. */
  readonly crease: number;
  readonly light: Partial<d.InferInput<typeof lighting.Settings>>;
  /** The stages that run, read every frame. Call `apply` after changing `taa` or `bloom`. */
  readonly stages: Stages;
}

/** How strongly the glow adds to the picture. */
const GLOW = 0.2;

/** Sky light, dimmed where nearby surfaces hide the sky. */
const occludedSky = (material: shading.Material, point: shading.Point) => {
  'use gpu';
  return shading.hemisphere(material, point) * ao.at(point.position);
};

/** Cools the shadows and warms the lights after the tone curve. */
const dusk = (color: d.v3f, _uv: d.v2f) => {
  'use gpu';
  const light = std.saturate(tone.luminance(color) * 2);
  return color * std.mix(d.vec3f(0.84, 0.93, 1.14), d.vec3f(1.02, 1, 0.96), light);
};

/** The lighting and every pass from the prepass to the canvas, seen through `view`. */
export function createLook(
  root: TgpuRoot,
  surface: Surface,
  view: cameras.OrbitCamera,
  options: LookOptions,
) {
  const { stages } = options;
  // Half-resolution depth of the solid props, drawn first for the occlusion to read
  const early = targets.create(root, surface, {
    depth: { format: 'depth32float', scale: 0.5, sampled: true },
  });
  const prepass = passes.create(root, early, { depth: passes.depth }).pipe(view.inject());
  const occlusion = ao.create(root, view, {
    depth: () => early.textures.depth,
    temporal: true, // turns its directions each frame for TAA to average
    settings: { radius: options.crease, power: 3 },
    enabled: () => stages.occlusion,
  });
  const light = lighting.create(root, {
    shading: { ambient: occludedSky },
    shadow: {
      eye: view,
      cascades: 2,
      far: options.far,
      filter: 9,
      enabled: () => stages.shadows,
    },
  });
  light.settings.patch(options.light);

  const screen = targets.create(root, surface, {
    hdr: { format: 'rgba16float', sampled: true },
    depth: { format: 'depth24plus', sampled: true },
  });
  const aa = taa.create(root, {
    source: () => screen.textures.hdr,
    depth: () => screen.textures.depth,
    camera: view,
  });
  const scene = passes
    .create(
      root,
      screen,
      { hdr: passes.color, depth: passes.depth },
      { jitter: aa.jitter, shared: { ao: occlusion.part } },
    )
    .pipe(view.inject())
    .beneath(light.inject());
  const glow = bloom.create(root, { source: aa });
  const final = finish.create(root, { format: surface.format, source: aa, glow, grade: dusk });

  const shapes: passes.Drawn[] = [];
  const casters: passes.Drawn[] = [];
  const drawn: passes.Drawn[] = [];

  return {
    light,

    /** Draws `renderer` in the prepass, the sun's shadow and the scene. */
    add(renderer: instances.Renderer) {
      shapes.push(prepass.add(renderer));
      casters.push(light.shadow.add(renderer));
      drawn.push(scene.add(renderer));
    },

    /** Applies `stages` to TAA and the final pass, which keep their own state. */
    apply() {
      aa.enable(stages.taa);
      final.settings.patch({ glowIntensity: stages.bloom ? GLOW : 0 });
    },

    initAsync: () => pipelines.compile([prepass, occlusion, light, scene, aa, glow, final]),

    record(frame: Frame) {
      early.fit();
      screen.fit();
      light.shadow.record(frame, casters); // a depth pass per cascade
      if (stages.occlusion) prepass.record(frame, shapes);
      occlusion.record(frame); // reads the prepass
      scene.record(frame, drawn); // reads the shadow and the occlusion
      aa.record(frame);
      if (stages.bloom) glow.record(frame); // reads TAA's picture
      final.record(frame, surface.view(frame));
    },

    destroy() {
      scene.destroy();
      prepass.destroy();
      light.destroy();
      glow.destroy();
      aa.destroy();
      occlusion.destroy();
      screen.destroy();
      early.destroy();
    },
  };
}
