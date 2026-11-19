import {
  bloom,
  cameras,
  clock,
  environment,
  exposure,
  finish,
  pipelines,
  targets,
  tone,
  type Frame,
  type Surface,
} from '@typegpu/kit';
import { d, type TgpuRoot } from 'typegpu';
import { mat4 } from 'wgpu-matrix';
import type { Settings } from '../settings.ts';
import { createAtmosphere } from './atmosphere.ts';
import { createClouds } from './clouds.ts';
import { createFoam } from './foam.ts';
import { foamingAt, spectrumAt } from './spectrum.ts';
import { createSurface } from './surface.ts';
import { createWaves } from './waves.ts';

const HDR = 'rgba16float';
const DEPTH = 'depth32float';
const UP = [0, 1, 0] as const;
/** The waves travel towards +z, at the camera, out of the low sun at −z. */
const WIND_DIRECTION = Math.PI / 2;
/** The sun sits 15° right of the view's centre, so its glitter runs across the sea. */
const SUN_AZIMUTH = 0.26;
/**
 * The eye stands `height` metres above the still sea, or `clear` above the highest crest beneath
 * it. Over the sea's loop that crest stays below 0.78 of the significant wave height at any wind,
 * and is `crest` of it at the default. Angles are in radians.
 */
const EYE = {
  height: 4,
  clear: 1.5,
  crest: 0.7312573,
  pitch: 0.028,
  pitchRange: [-0.35, 0.6] as const,
  across: (55 * Math.PI) / 180,
  tallest: 1.1,
  turning: 14,
  near: 0.5,
  far: 20_000,
};
const ENVIRONMENT_SIZE = 128;
const ENVIRONMENT_TEXEL = Math.PI / (2 * ENVIRONMENT_SIZE);
/** The deck's drift in metres per second, and the seconds between the sky's rebakes. */
const DRIFT = { speed: 14, sky: 2 };
const EXPOSURE = { key: 0.15, lowest: 1, highest: 3000 };
/** The key darkens to `share` as the sun sets through `fade` degrees, so dusk stays dark. */
const DUSK = { share: 0.25, fade: [-4, 10] as const };
/** The sea opens `START` seconds into its loop, after `HISTORY.steps` steps of foam history. */
const START = 195.4;
const HISTORY = { steps: 24, step: 0.125 };
/** Seconds a frame may advance the sea, so a stalled tab resumes calmly. */
const LONGEST_STEP = 0.1;

/**
 * A near gale at dusk: FFT waves with foam where the crests compress, and a low sun under a cloud
 * deck lighting the backs of the raised waves and laying a glint path to the eye.
 */
export async function createScene(root: TgpuRoot, surface: Surface) {
  const values = { wind: 20, sun: 0.5, clouds: 1 };
  const screen = targets.create(root, surface, {
    hdr: { format: HDR, sampled: true },
    depth: { format: DEPTH },
  });
  const view = cameras.custom(root);
  // Hosts turn the goal as an orbit's; the eye stays put and looks the other way, at the target
  const goal = { yaw: 0, pitch: EYE.pitch, distance: 1 };
  const look = { ...goal };

  const waves = createWaves(root);
  const foam = createFoam(root, { direction: WIND_DIRECTION });
  const atmosphere = createAtmosphere(root);
  const clouds = createClouds(root, atmosphere, view, surface);
  // Reflections and ambient light see the sky and deck from the still sea, metres below the eye
  const light = environment.create(root, { size: ENVIRONMENT_SIZE });
  const skySource = light.add((direction: d.v3f) => {
    'use gpu';
    const deck = clouds.march(d.vec3f(), direction, ENVIRONMENT_TEXEL);
    return atmosphere.skyRadiance(direction) * deck.w + deck.xyz;
  });
  const sea = createSurface(root, {
    view,
    waves,
    atmosphere,
    clouds,
    foam,
    light,
    format: HDR,
    depth: DEPTH,
  });

  const glow = bloom.create(root, {
    source: () => screen.textures.hdr,
    settings: { threshold: 3, knee: 0.6 },
    levels: 3,
  });
  const metering = exposure.create(root, { source: () => screen.textures.hdr, settings: EXPOSURE });
  const final = finish.create(root, {
    format: surface.format,
    source: () => screen.textures.hdr,
    glow,
    exposure: metering,
    tone: tone.agx,
    settings: { exposure: 1, vignette: 0.3, glowIntensity: 0.08 },
  });
  await pipelines.compile([waves, foam, atmosphere, clouds, light, sea, glow, metering, final]);

  let rebuild = true;
  let rebake = true;
  let skyBaked = -Infinity;
  const eye = new Float32Array(3);
  const ahead = new Float32Array(3);
  const viewMatrix = mat4.create();
  const projection = mat4.create();

  const write = (key?: keyof typeof values) => {
    if (key === undefined || key === 'wind') {
      waves.foaming.write(foamingAt(values.wind));
      const spectrum = spectrumAt(values.wind, WIND_DIRECTION);
      waves.spectrum.write(spectrum);
      eye[1] = Math.max(EYE.height, EYE.crest * spectrum.waveHeight + EYE.clear);
      // Changes between frames coalesce: the next frame rebuilds the sea once
      rebuild = true;
    }
    if (key === undefined || key === 'sun') {
      const elevation = (values.sun * Math.PI) / 180;
      const sun = d.vec3f(
        Math.cos(elevation) * Math.sin(SUN_AZIMUTH),
        Math.sin(elevation),
        -Math.cos(elevation) * Math.cos(SUN_AZIMUTH),
      );
      atmosphere.place(sun, EYE.height);
      const [from, to] = DUSK.fade;
      const day = Math.min(Math.max((values.sun - from) / (to - from), 0), 1);
      metering.settings.patch({
        key: EXPOSURE.key * (DUSK.share + (1 - DUSK.share) * day * day),
      });
      rebake = true;
    }
    if (key === undefined || key === 'clouds') {
      clouds.cover(values.clouds);
      rebake = true;
    }
  };
  write();
  sea.place(eye[0], eye[2]);
  const settings: Settings<typeof values> = {
    values,
    knobs: {
      wind: { label: 'wind (m/s)', min: 4, max: 24, step: 0.5 },
      sun: { label: 'sun height (°)', min: -4, max: 60, step: 0.5 },
      clouds: { label: 'cloud cover', min: 0, max: 1, step: 0.05 },
    },
    write,
  };

  // The clouds' first bake also fills their noise volume, longer than a frame, so it runs at load
  const warm = root['~unstable'].createCommandEncoder();
  const warming = warm.beginComputePass();
  clouds.bake(warming);
  foam.bake(warming);
  waves.rebuild(warming);
  warming.end();
  warm.submit();
  for (let step = 0; step < HISTORY.steps; step++) {
    waves.step.patch({ time: START - (HISTORY.steps - step) * HISTORY.step, dt: HISTORY.step });
    const encoder = root['~unstable'].createCommandEncoder();
    const history = encoder.beginComputePass();
    waves.record(history);
    history.end();
    encoder.submit();
  }
  rebuild = false;

  const aim = () => {
    goal.pitch = Math.min(Math.max(goal.pitch, EYE.pitchRange[0]), EYE.pitchRange[1]);
    const level = Math.cos(look.pitch);
    ahead[0] = -level * Math.sin(look.yaw);
    ahead[1] = -Math.sin(look.pitch);
    ahead[2] = -level * Math.cos(look.yaw);
    mat4.lookAt(eye, [eye[0] + ahead[0], eye[1] + ahead[1], eye[2] + ahead[2]], UP, viewMatrix);

    const aspect = surface.width / surface.height;
    const tall = Math.min(2 * Math.atan(Math.tan(EYE.across / 2) / aspect), EYE.tallest);
    mat4.perspective(tall, aspect, EYE.near, EYE.far, projection);
    view.write(viewMatrix, projection);
  };

  return {
    view: {
      goal,
      /** Turns the eye to its goal at once. */
      jump() {
        Object.assign(look, goal);
      },
    },
    settings,
    frame(frame: Frame) {
      const turning = clock.ease(frame.dt, EYE.turning);
      look.yaw += (goal.yaw - look.yaw) * turning;
      look.pitch += (goal.pitch - look.pitch) * turning;
      aim();
      screen.fit();
      const dt = Math.min(frame.dt, LONGEST_STEP);
      waves.step.patch({ time: START + frame.time, dt });
      const drifted = frame.time * DRIFT.speed;
      clouds.move([Math.cos(WIND_DIRECTION) * drifted, Math.sin(WIND_DIRECTION) * drifted]);
      const bakeSky = rebake || frame.time - skyBaked >= DRIFT.sky;

      const compute = frame.encoder.beginComputePass();
      if (rebake) {
        atmosphere.bake(compute);
        clouds.bake(compute);
      }
      clouds.record(compute);
      if (rebuild) waves.rebuild(compute);
      waves.record(compute);
      compute.end();
      rebuild = false;
      if (bakeSky) {
        light.bake(frame, skySource);
        skyBaked = frame.time;
      }
      rebake = false;

      const pass = frame.encoder.beginRenderPass({
        colorAttachments: [{ view: screen.views.hdr, loadOp: 'clear', storeOp: 'store' }],
        depthStencilAttachment: {
          view: screen.views.depth,
          depthClearValue: 1,
          depthLoadOp: 'clear',
          depthStoreOp: 'discard',
        },
      });
      sea.draw(pass);
      pass.end();

      glow.record(frame);
      metering.record(frame);
      final.record(frame, surface.view(frame));
    },

    destroy() {
      for (const made of [screen, waves, foam, atmosphere, clouds, light, sea, glow, metering])
        made.destroy();
    },
  };
}
