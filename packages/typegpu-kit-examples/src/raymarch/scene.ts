import {
  cameras,
  finish,
  fxaa,
  pipelines,
  targets,
  tone,
  type Frame,
  type Surface,
} from '@typegpu/kit';
import { oklabToLinearRgb } from '@typegpu/color';
import { opSmoothUnion, sdPlane, sdRoundedBox3d, sdSphere } from '@typegpu/sdf';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import type { Settings } from '../settings.ts';
import { createRaymarch, UNSHADED } from './raymarch.ts';

const HDR = 'rgba16float';
/** Display-space target for FXAA after the final pass. */
const LDR = 'rgba8unorm';
const TAU = Math.PI * 2;

/** Cube centre height, half-size and edge radius in world units. */
const CUBE = { height: 0.7, half: 0.42, rounding: 0.08 };
const BALLS = 5;
/** World units from a ball's centre to its surface. */
const BALL = 0.3;
/** The balls' orbit: its centre, and radius, radial swing and vertical bob in world units. */
const ORBIT = { middle: d.vec3f(0, 0.75, 0), out: 0.85, swing: 0.3, bob: 0.4 };
/**
 * Ball `k` circles at `first + k × faster` radians per unit turned, and swings out and bobs `swing`
 * and `bob` times as fast as it circles.
 */
const PACE = { first: 0.7, faster: 0.13, swing: 0.9, bob: 1.6 };
/** Bounding radius around the orbit's centre, including the cube. */
const REACH = Math.hypot(ORBIT.out + ORBIT.swing, ORBIT.bob) + BALL;
/** Floor checker frequency per world unit and two grey levels. */
const FLOOR = { squares: 2, dark: 0.16, light: 0.22 };
/** Oklab lightness, chroma and hue rotation in radians per world unit along x, y and z. */
const BLOB = { lightness: 0.66, chroma: 0.16, hues: d.vec3f(1.4, 1.8, 0.9) };

/** Balls melting into a cube, a distance field marched once a pixel and smoothed with FXAA. */
export async function createScene(root: TgpuRoot, surface: Surface) {
  const screen = targets.create(root, surface, {
    hdr: { format: HDR, sampled: true },
    ldr: { format: LDR, sampled: true },
  });
  const view = cameras.orbit(root, {
    size: surface,
    target: [0, CUBE.height, 0],
    distance: 5,
    pitch: 0.35,
    pitchRange: [0.05, 1.4],
    distanceRange: [2, 12],
  });

  const values = {
    /** Smooth-union blend distance. */
    blend: 0.35,
    /** Turns a second of the balls' orbits. */
    speed: 0.6,
    /** Smooths the stair-steps along edges. */
    fxaa: true,
  };
  const blend = root.createUniform(d.f32, values.blend);
  /** Ball centres, placed once a frame rather than at every ray step. */
  const balls = root.createUniform(d.arrayOf(d.vec4f, BALLS));
  const centres = new Float32Array(BALLS * 4);
  let turned = 0;

  /**
   * Spaces balls evenly round the orbit's centre, with increasing orbital speed, radial swing and
   * vertical motion.
   */
  const place = () => {
    for (let k = 0; k < BALLS; k++) {
      const angle = turned * (PACE.first + k * PACE.faster) + (k * TAU) / BALLS;
      const out = ORBIT.out + Math.sin(angle * PACE.swing + k) * ORBIT.swing;
      centres[k * 4] = ORBIT.middle.x + Math.cos(angle) * out;
      centres[k * 4 + 1] = ORBIT.middle.y + Math.sin(angle * PACE.bob) * ORBIT.bob;
      centres[k * 4 + 2] = ORBIT.middle.z + Math.sin(angle) * out;
    }
    balls.write(centres);
  };
  place();

  const blobs = (point: d.v3f) => {
    'use gpu';
    // Each smooth union expands the bound by at most a quarter of the blend. Beyond the shadow
    // range, use distance to the bounding sphere instead of evaluating every blob
    const outside = std.length(point - ORBIT.middle) - REACH - blend.$ * (BALLS / 4);
    if (outside > UNSHADED) return outside;

    let shape = sdRoundedBox3d(
      point - d.vec3f(0, CUBE.height, 0),
      d.vec3f(CUBE.half),
      CUBE.rounding,
    );
    for (const k of tgpu.unroll(std.range(BALLS))) {
      shape = opSmoothUnion(shape, sdSphere(point - balls.$[k].xyz, BALL), blend.$);
    }
    return shape;
  };

  const field = (point: d.v3f) => {
    'use gpu';
    return std.min(blobs(point), sdPlane(point, d.vec3f(0, 1, 0), 0));
  };

  const paint = (point: d.v3f) => {
    'use gpu';
    if (point.y < blobs(point)) {
      const cell = std.floor(point.xz * FLOOR.squares);
      const checker = std.fract((cell.x + cell.y) * 0.5) * 2;
      return d.vec3f(std.mix(FLOOR.dark, FLOOR.light, checker));
    }
    const hue = std.dot(point, BLOB.hues);
    const chroma = d.vec2f(std.cos(hue), std.sin(hue)) * BLOB.chroma;
    return oklabToLinearRgb(d.vec3f(BLOB.lightness, chroma));
  };

  const march = createRaymarch(root, { camera: view, format: HDR, field, paint });
  const look = { source: () => screen.textures.hdr, tone: tone.aces, settings: { vignette: 0.3 } };
  const final = finish.create(root, { ...look, format: LDR });
  const plain = finish.create(root, { ...look, format: surface.format });
  const smooth = fxaa.create(root, { format: surface.format, source: () => screen.textures.ldr });
  await pipelines.compile([march, final, plain, smooth]);

  const settings: Settings<typeof values> = {
    values,
    knobs: {
      blend: { min: 0.01, max: 1, step: 0.01 },
      speed: { min: 0, max: 2, step: 0.01 },
      fxaa: { label: 'FXAA' },
    },

    write() {
      blend.write(values.blend);
    },
  };

  return {
    view,
    settings,

    frame(frame: Frame) {
      view.update(frame.dt);
      screen.fit();
      if (values.speed !== 0) {
        turned += frame.dt * values.speed;
        place();
      }

      const scene = frame.encoder.beginRenderPass({ colorAttachments: { view: screen.views.hdr } });
      march.draw(scene);
      scene.end();

      if (values.fxaa) {
        final.record(frame.encoder, screen.views.ldr);
        smooth.record(frame.encoder, surface.view(frame));
      } else {
        plain.record(frame.encoder, surface.view(frame));
      }
    },

    destroy() {
      screen.destroy();
    },
  };
}
