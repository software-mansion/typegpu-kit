import {
  cameras,
  Camera,
  passes,
  targets,
  type CameraBase,
  type Frame,
  type Lighting,
  type SizeSource,
} from '@typegpu/kit';
import type { Inject } from '@typegpu/kit/inject';
import { meshes } from '@typegpu/kit/geometry';
import { perlin2d } from '@typegpu/noise';
import { d, std, tgpu, type TgpuRoot, type TgpuUniform } from 'typegpu';
import { pictureGroups, sample } from './picture.ts';
import { HORIZON } from './sky.ts';

/** True for reflection pipelines. Excludes geometry below the water. */
export const mirrored = tgpu.slot(false);

/** Reflection resolution as a fraction of screen width and height. */
const MIRROR_SCALE = 0.5;
/** The square of water, 6.4 km across. */
const plane = meshes.unindexed(meshes.plane({ width: 6400, depth: 6400 }));
const DEEP = d.vec3f(0.0015, 0.0025, 0.004);
/** Water reflectance at normal incidence. */
const REFLECTANCE = 0.02;
const RIPPLE_METRES = [9, 2.6] as const;
const RIPPLE_HEIGHT = [0.5, 0.22] as const;
const RIPPLE_DRIFT_METRES_PER_SECOND = [d.vec2f(0.6, 0.25), d.vec2f(-0.35, 0.5)] as const;
const RIPPLE_STEP_METRES = 0.08;
/** Ripple displacement in screen widths at a distance of 10 m. */
const RIPPLE_SHIFT = 0.18;
const RIPPLE_SHIFT_METRES = 10;
/** Distance beyond which subpixel ripples become a still reflection. */
const CALM_METRES = 900;
const HAZE_METRES = 2600;

/** Reflects `view` across water at y = 0 and flips the image horizontally. */
const reflectionOf = (view: d.Infer<typeof Camera>) => {
  'use gpu';
  // Negating the y column reflects the world. Negating the x row flips the image and preserves
  // triangle winding for back-face culling
  const flip = d.vec4f(-1, 1, 1, 1);
  const m = view.viewProjection;
  return Camera({
    viewProjection: d.mat4x4f(
      m.columns[0] * flip,
      m.columns[1] * flip * -1,
      m.columns[2] * flip,
      m.columns[3] * flip,
    ),
    position: view.position * d.vec3f(1, -1, 1),
    right: view.right * d.vec3f(-1, 1, -1),
    up: view.up * d.vec3f(1, -1, 1),
  });
};

/** Ripple height in metres from two drifting noise layers. */
const ripplesAt = (place: d.v2f, time: number) => {
  'use gpu';
  let height = d.f32(0);
  for (const i of tgpu.unroll([0, 1])) {
    const drift = RIPPLE_DRIFT_METRES_PER_SECOND[i] * time;
    height += perlin2d.sample((place + drift) / RIPPLE_METRES[i]) * RIPPLE_HEIGHT[i];
  }
  return height;
};

const Corner = d.struct({ position: d.vec4f, world: d.vec3f, clip: d.vec4f });
const Seen = d.struct({ color: d.vec4f });

interface LakeOptions {
  /** The scene's camera, whose reflection the mirror draws. */
  readonly view: CameraBase;
  /** The scene's lighting, which lights what the mirror draws. */
  readonly light: Lighting;
  readonly size: SizeSource;
  readonly time: TgpuUniform<d.F32>;
  /** Cached Perlin gradients for ripples. */
  readonly noise: { inject(): Inject };
}

/**
 * Creates reflection passes for above-water geometry through a reflected scene camera. The
 * water drawing samples the result with ripple distortion.
 */
export function createLake(root: TgpuRoot, { view, light, size, time, noise }: LakeOptions) {
  const mirror = targets.create(root, size, {
    hdr: { format: 'rgba16float', sampled: true, scale: MIRROR_SCALE },
    depth: { format: 'depth24plus', scale: MIRROR_SCALE },
  });
  const reflectedView = () => {
    'use gpu';
    return reflectionOf(view.uniform.$);
  };
  const reflected: Inject = (cfg) => cfg.with(cameras.current, reflectedView).with(mirrored, true);
  const reflection = passes
    .create(root, mirror, { hdr: passes.color, depth: passes.depth }, { label: 'mirror' })
    .pipe(reflected)
    .beneath(light.inject());

  const groupOf = pictureGroups(root);

  const water: passes.Drawing<typeof Corner, d.Infer<typeof Seen>> = {
    label: 'water',
    Vertex: Corner,
    at: (vertexIndex: number, _instanceIndex: number) => {
      'use gpu';
      const world = plane.vertexAt(vertexIndex).position;
      const clip = cameras.toClip(world);
      return Corner({ position: clip, world, clip });
    },
    pixelOf: (corner) => {
      'use gpu';
      const toEye = cameras.current.$.position - corner.world;
      const distance = std.length(toEye);
      const toward = toEye / distance;
      const calm = std.saturate(1 - distance / CALM_METRES);

      const place = corner.world.xz;
      const here = ripplesAt(place, time.$);
      const dx = ripplesAt(place + d.vec2f(RIPPLE_STEP_METRES, 0), time.$) - here;
      const dz = ripplesAt(place + d.vec2f(0, RIPPLE_STEP_METRES), time.$) - here;
      const normal = std.normalize(d.vec3f(-dx * calm, RIPPLE_STEP_METRES, -dz * calm));

      const screen = (corner.clip.xy / corner.clip.w) * d.vec2f(-0.5, -0.5) + 0.5;
      const shift =
        (normal.xz * (RIPPLE_SHIFT * RIPPLE_SHIFT_METRES)) / std.max(distance, RIPPLE_SHIFT_METRES);
      const above = sample(screen + shift).xyz;

      const cosine = std.saturate(std.dot(normal, toward));
      const fresnel = REFLECTANCE + (1 - REFLECTANCE) * std.pow(1 - cosine, 5);
      const haze = std.saturate(distance / HAZE_METRES);
      const color = std.mix(DEEP + above * fresnel, HORIZON, haze * haze);
      return Seen({ color: d.vec4f(color, 1) });
    },
    color: passes.shaded,
    inject: () => noise.inject(),
    draw(pass) {
      pass.setBindGroup(groupOf(mirror.textures.hdr));
      pass.draw(plane.vertexCount);
    },
  };

  return {
    /** Add above-water geometry to these reflection passes. */
    reflection,
    water,
    /** Draws the reflection before the scene samples it. */
    record(frame: Frame, drawn: readonly passes.Drawn[]) {
      mirror.fit();
      reflection.record(frame, drawn);
    },
    initAsync: () => reflection.initAsync(),
    destroy() {
      mirror.destroy();
    },
  };
}
