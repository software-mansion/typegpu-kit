import {
  bloom,
  cameras,
  clock,
  finish,
  passes,
  pipelines,
  targets,
  video,
  type Frame,
  type Surface,
} from '@typegpu/kit';
import type { TgpuRoot } from 'typegpu';
import type { Settings } from '../settings.ts';
import type { Camera } from '../ui.ts';
import { createEmbers, SHEDDING } from './embers.ts';
import { createMirror } from './mirror.ts';

/** The feed's long and short sides in texels. The feed is also the world the embers rise through. */
const LONG = 640;
const SHORT = 360;
const STEP_SECONDS = 1 / 120;
/** Smoke repaints at a camera's pace, so it sheds as a camera would. */
const SMOKE_HZ = 30;
/** Seconds without a camera frame before the smoke returns. */
const CAMERA_GONE_SECONDS = 1;

const LOOK = { exposure: 1.1, glowIntensity: 0.3, vignette: 0.45 };
const BLOOM = { threshold: 0.7 };
const SAMPLED = { format: 'rgba16float', sampled: true } as const;

/** Your camera as a dark mirror whose moving edges shed embers. */
export async function createScene(root: TgpuRoot, surface: Surface) {
  const values = {
    /** Change in brightness that counts as motion. Lower sheds from gentler movement. */
    sensitivity: SHEDDING.sensitivity,
    /** Chance a moving edge sheds an ember each camera frame. */
    rate: SHEDDING.rate,
    /** Mixes ember colour from fire to the colour of what shed it. */
    tint: SHEDDING.tint,
    /** Brightness of the reflection behind the embers. */
    reflection: 0.12,
    paused: false,
  };
  // A portrait screen gets a portrait feed, so a phone camera isn't cropped to a landscape band
  const portrait = surface.height > surface.width;
  const size = portrait ? ([SHORT, LONG] as const) : ([LONG, SHORT] as const);
  const [feedWidth, feedHeight] = size;
  const feed = video.create(root, { size });
  // The embers compare frames in the small feed; the reflection samples the camera in place, where
  // the feed's crop lies
  const live = video.live(root, { crop: { size } });
  const sim = clock.fixedStep({ step: STEP_SECONDS });
  const smoke = clock.cadence({ hz: SMOKE_HZ });

  // Shows the middle of the feed, as large as covers the screen
  const view = cameras.flat(root, {
    size: surface,
    fit: { width: feedWidth, height: feedHeight, mode: 'cover' },
    center: [feedWidth / 2, feedHeight / 2],
  });
  const fire = createEmbers(root, { camera: view, step: STEP_SECONDS, feed, size });
  const { embers } = fire;
  const mirror = createMirror(root, { feed, live, view, format: SAMPLED.format });
  mirror.framing.patch({ brightness: values.reflection });

  const screen = targets.create(root, surface, { hdr: SAMPLED });
  const scene = passes
    .create(root, screen, { hdr: passes.color }, { shared: { camera: live.part } })
    .pipe(view.inject());
  const sparks = scene.add(embers.drawing);
  const hdr = () => screen.textures.hdr;
  const glow = bloom.create(root, { source: hdr, settings: BLOOM });
  const final = finish.create(root, { format: surface.format, source: hdr, glow, settings: LOOK });

  let lensed = false;

  let seen = feed.frames;
  let sinceCamera = Infinity;
  let pictures = 0;
  let shed = 0;

  const settings: Settings<typeof values> = {
    values,
    knobs: {
      sensitivity: { min: 0.02, max: 0.3, step: 0.01 },
      rate: { label: 'shedding', min: 0, max: 1, step: 0.05 },
      tint: { label: 'camera colour', min: 0, max: 1, step: 0.05 },
      reflection: { min: 0, max: 0.5, step: 0.01 },
      paused: {},
    },
    write() {
      const { sensitivity, rate, tint, reflection, paused } = values;
      sim.paused = paused;
      mirror.framing.patch({ brightness: reflection });
      fire.settings.patch({ sensitivity, rate, tint });
    },
    readouts: {
      /** Counts camera frames the feed has written. */
      'camera frames': () => feed.frames,
      'pictures shed from': () => shed,
    },
  };

  const camera: Camera = {
    attach(source) {
      feed.attach(source);
      live.attach(source);
    },
    get frames() {
      return feed.frames;
    },
  };

  await pipelines.compile([feed, fire, mirror, scene, glow, final]);

  return {
    settings,
    ui: { camera },

    frame(frame: Frame) {
      view.update(frame.dt);
      screen.fit();

      feed.update(frame);
      live.update(frame);
      if (feed.frames !== seen) {
        // The camera came back, so the smoke's last picture must not count as its first's past
        if (sinceCamera >= CAMERA_GONE_SECONDS) fire.reset();
        seen = feed.frames;
        sinceCamera = 0;
        pictures += 1;
      } else {
        sinceCamera += frame.dt;
        if (sinceCamera >= CAMERA_GONE_SECONDS && smoke.due(frame.dt)) {
          mirror.paint(frame.encoder, frame.time);
          pictures += 1;
        }
      }

      // While a camera streams, the reflection samples it at full resolution
      const streaming = sinceCamera < CAMERA_GONE_SECONDS && feed.frames > 0;
      if (streaming !== lensed) {
        lensed = streaming;
        mirror.framing.patch({ live: streaming ? 1 : 0 });
      }

      const steps = sim.update(frame.dt);
      const compute = frame.encoder.beginComputePass();
      if (pictures !== shed && !values.paused) {
        fire.shed(compute, sim.time);
        shed = pictures;
      }
      for (let i = 0; i < steps; i++) embers.step(compute);
      embers.spawn(compute);
      embers.compact(compute, sim.ahead);
      compute.end();

      const pass = scene.begin(frame);
      mirror.draw(pass);
      sparks.draw(pass);
      pass.end();

      glow.record(frame);
      final.record(frame, surface.view(frame));
    },

    destroy() {
      for (const made of [feed, live, fire, screen, glow]) made.destroy();
    },
  };
}
