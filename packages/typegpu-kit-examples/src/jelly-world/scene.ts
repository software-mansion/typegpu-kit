import {
  cameras,
  clock,
  pipelines,
  profiler,
  dynamicResolution,
  targets,
  type Frame,
  type Surface,
  font,
  finish,
  taa,
  type Baked,
} from '@typegpu/kit';
import { d, std, type TgpuRoot } from 'typegpu';
import type { Pointing } from '../contract.ts';
import type { Settings } from '../settings.ts';
import { createSlider, RANGE, STEP } from './slider.ts';
import { createJelly, Look, Moment, MOTION } from './jelly.ts';
import { createLabel } from './label.ts';
import { createLand } from './land.ts';
import { createWorld } from './world.ts';
import { createCritters } from './critters.ts';
import { createCamera } from './camera.ts';
import type { V3 } from './tray.ts';

export interface SceneOptions {
  /** The font of the value under the jelly, as `font.create` takes it. */
  readonly font: Baked;
}

/** Jelly colours in display values. */
const FLAVOURS = {
  orange: [1, 0.45, 0.075],
  lime: [0.55, 0.9, 0.12],
  berry: [0.9, 0.16, 0.42],
  blue: [0.15, 0.55, 1],
} satisfies Record<string, V3>;
type Flavour = keyof typeof FLAVOURS;
/** How fast the jelly's colour eases to a new flavour, as a rate per second. */
const FLAVOUR_EASE = 8;

/** World x of the jelly world's front before and after the lever reveals it, past the view. */
const FRONT = [-2.7, 2.7] as const;
/** How fast the front sweeps to where it is going, as a rate per second. */
const SWEEP = 1.4;
/** Where the lever sends the jelly's end, as progress: from the upper half, then from the lower. */
const SENDS = [0.05, 1] as const;

/** The light's direction at either end of the `light` setting. */
const LIGHT = [d.vec3f(0.19, -0.24, 0.75), d.vec3f(-0.5, -0.14, -0.8)] as const;

/**
 * The share of the display's frame interval the frame's GPU time keeps under while `auto` sets
 * the detail, and the detail's range.
 */
const DETAIL = { share: 0.75, range: [0.3, 1] as const };

/** How near a ray must pass the jelly's spine for a stroke to drag the jelly. */
const REACH = 0.12;
/** Pointer rays nearer level than this find no point on the planes below. */
const LEVEL = { grazing: 1e-4 };
/** Height over the tray that critters look at under the pointer. */
const WATCH_HEIGHT = 0.08;
/** Height of the plane the pointer drags the jelly's end along. */
const DRAG_PLANE = 0.05;

/** How fast the jelly's end eases to where it is going, as a rate per second. */
const FOLLOW = 5;
/** The idle sway of the jelly's end: its rate in radians a second, and its ends as progress. */
const SWAY = { speed: 0.6, low: 0.2, high: 0.95 };

/** Where the jelly's end sits in world x at `progress` from 0 to 1. */
const endAt = (progress: number) => RANGE[0] + (RANGE[1] - RANGE[0]) * progress;
/** Where the idle sway puts the jelly's end, `swing` from 0 to 1 between its ends. */
const swayed = (swing: number) => endAt(SWAY.low + (SWAY.high - SWAY.low) * swing);

/** A jelly slider marched per pixel over a tray, whose lever grows a jelly world of critters. */
export async function createScene(root: TgpuRoot, surface: Surface, options: SceneOptions) {
  const gpu = profiler.create(root);
  const marchTime = gpu.timer('march');
  const outputTime = gpu.timer('output');
  const resolution = dynamicResolution.create(gpu.frame, {
    budget: (interval) => DETAIL.share * interval,
    range: DETAIL.range,
  });

  const values = {
    /** Where the jelly's end sits from 0 to 1. Setting it stops the idle sway. */
    progress: 1,
    flavour: 'orange' as Flavour,
    /** Swings the light from behind the jelly to in front of it. */
    light: 0,
    /** Blurs the floor seen through the jelly. */
    frosted: false,
    /** Fraction of each side of the screen marched, where `auto` starts. */
    detail: 1,
    /**
     * Sets `detail` to keep the frame's GPU time under three quarters of the display's frame
     * interval. Needs `timestamp-query`.
     */
    auto: gpu.available,
  };

  const camera = createCamera(root, surface);
  const view = camera.view;
  const screen = targets.create(
    root,
    surface,
    {
      color: { format: 'rgba16float', sampled: true },
      motion: { format: MOTION, sampled: true },
      depth: { format: 'depth24plus', sampled: true },
    },
    { resolution: () => resolution.scale },
  );
  const region = screen.regions.color;
  const aa = taa.create(root, {
    source: () => screen.textures.color,
    motion: () => screen.textures.motion,
    depth: () => screen.textures.depth,
    region,
    camera: view,
  });
  const final = finish.create(root, {
    format: surface.format,
    source: aa,
    tone: (color) => {
      'use gpu';
      return std.tanh(color * 1.3);
    },
    encode: false,
    settings: { vignette: 0 },
  });

  const sim = clock.fixedStep({ step: STEP });
  /** Where the jelly's end is going and where it is, starting in the middle of the sway. */
  let goal = swayed(0.5);
  let at = goal;
  const slider = createSlider(root, { start: at });
  const sans = font.create(root, options.font);
  const label = createLabel(root, sans);
  /** The jelly's colour, easing to the flavour's. */
  const colour = d.vec4f(...FLAVOURS[values.flavour], 1);
  const lookValues = () => ({
    light: std.normalize(std.mix(LIGHT[0], LIGHT[1], values.light)),
    blur: values.frosted ? 1 : 0,
    jelly: colour,
  });
  const look = root.createUniform(Look, lookValues());
  const world = createWorld(root);
  const now = root.createUniform(Moment);
  const critters = createCritters(root, world);
  const land = createLand(root, { camera: view, world, critters, now, look });
  const jelly = createJelly(root, {
    camera: view,
    look,
    field: slider.field,
    end: slider.end,
    moved: slider.moved,
    label: label.view,
    offset: aa.offset,
    now,
    scenery: land.scenery,
  });
  await pipelines.compile([aa, final, slider, label, land, jelly]);

  const moment = { front: 0, time: 0, seed: d.vec2f() };
  const ray: { origin: V3; direction: V3 } = { origin: [0, 0, 0], direction: [0, 0, 1] };
  const grip: V3 = [0, 0, 0];
  const eye: V3 = [0, 0, 0];
  const point: V3 = [0, 0, 0];
  /** Whether the jelly's end has been set by hand, which stops the idle sway. */
  let touched = false;
  let time = 0;
  /** Whether the spine's field is still to be baked for the first frame. */
  let unbaked = true;
  /** Whether the floor's baked shadows match the light. */
  let lit = false;
  let revealed = false;
  /** Where the jelly world's front is in world x, sweeping towards `FRONT`'s end. */
  let front: number = FRONT[0];
  /** Whether the stroke started away from the jelly, the lever and the critters, which it ignores. */
  let idle = false;

  const settings: Settings<typeof values> = {
    values,
    knobs: {
      progress: { min: 0, max: 1, step: 0.01 },
      flavour: { options: Object.keys(FLAVOURS) as Flavour[] },
      light: { min: 0, max: 1, step: 0.01 },
      frosted: {},
      detail: { min: DETAIL.range[0], max: DETAIL.range[1], step: 0.05, live: true },
      auto: { live: true, ...(gpu.available ? {} : { disabled: 'needs timestamp-query' }) },
    },
    write(key) {
      // Moving the detail by hand turns `auto` off; either restarts its timing from the detail
      if (key === 'detail') values.auto = false;
      if (key === undefined || key === 'detail' || key === 'auto') {
        resolution.scale = values.detail;
      }
      look.patch(lookValues());
      if (key === undefined || key === 'light') lit = false;
      if (key === 'progress') {
        touched = true;
        goal = endAt(values.progress);
      }
    },
  };

  const pointing: Pointing = {
    move(x, y) {
      view.rayAt(x, y, ray);
      if (cameras.hitHeight(ray, WATCH_HEIGHT, point, LEVEL) !== undefined) critters.watch(point);
    },
    leave() {
      critters.watch(undefined);
    },
    drag(x, y, start) {
      if (start) idle = false;
      if (idle) return;

      view.rayAt(x, y, ray);
      const { origin, direction } = ray;
      if (start && world.grab(origin, direction)) return;
      if (world.holding) {
        world.pull(origin, direction);
        return;
      }
      if (start && critters.pick(origin, direction)) return;
      if (critters.holding) {
        critters.hold(origin, direction);
        return;
      }
      if (start && !slider.passes(origin, direction, REACH)) {
        idle = true;
        return;
      }

      if (cameras.hitHeight(ray, DRAG_PLANE, grip, LEVEL) === undefined) return;
      touched = true;
      goal = grip[0];
    },
    /** Lets go of the lever or a critter the pointer carries. */
    end() {
      world.release();
      critters.drop();
    },
  };

  /** Sets the detail from GPU time while `auto`. */
  const pace = (frame: Frame) => {
    if (!values.auto) return;
    resolution.update(frame);
    values.detail = resolution.scale;
  };

  /** Eases the jelly's colour towards its flavour's, writing it while it moves. */
  const easeFlavour = (dt: number) => {
    const flavour = FLAVOURS[values.flavour];
    const blend = clock.ease(dt, FLAVOUR_EASE);
    let easing = false;
    for (let i = 0; i < 3; i++) {
      const left = flavour[i] - colour[i];
      colour[i] += left * blend;
      easing ||= Math.abs(left) > 1e-4;
    }
    if (easing) look.patch({ jelly: colour });
  };

  /**
   * Sweeps the front and swings the lever. Its first pull reveals the jelly world, and later pulls
   * send the jelly's end, now at `progress`, to the other side of the tray.
   */
  const reveal = (dt: number, progress: number) => {
    front += (FRONT[revealed ? 1 : 0] - front) * clock.ease(dt, SWEEP);
    if (!world.update(dt, front)) return;
    if (revealed) {
      touched = true;
      goal = endAt(progress > 0.5 ? SENDS[0] : SENDS[1]);
    }
    revealed = true;
    critters.hop();
  };

  return {
    view,
    settings,
    pointing,

    frame(frame: Frame) {
      const dt = Math.min(frame.dt, 0.1);
      time += dt;
      view.update(dt);
      pace(frame);
      if (screen.fit()) aa.reset();

      if (!touched) goal = swayed(0.5 + 0.5 * Math.sin(time * SWAY.speed));
      at += (goal - at) * clock.ease(dt, FOLLOW);
      slider.drag(at);
      easeFlavour(dt);

      let stepped = unbaked;
      unbaked = false;
      for (let steps = sim.update(dt); steps > 0; steps--) {
        slider.step();
        critters.feel(slider.spine);
        stepped = true;
      }

      const progress = std.saturate((slider.target - RANGE[0]) / (RANGE[1] - RANGE[0]));
      reveal(dt, progress);
      camera.follow(dt, revealed);

      moment.front = front;
      moment.time = time;
      moment.seed.x = Math.random() * 2 - 1;
      moment.seed.y = Math.random() * 2 - 1;
      now.write(moment);

      label.show(frame, Math.round(progress * 100));
      cameras.eyeOf(view.pose, eye);
      critters.update(dt, time, front, slider.spine, eye);
      critters.weigh(slider.loads, slider.spine);

      slider.track();
      const shading = !lit || world.moving;
      lit = true;
      const compute = frame.encoder.beginComputePass();
      if (stepped) slider.bake(compute);
      if (shading) land.bakeShadows(compute);
      land.cull(compute, region.width, region.height);
      compute.end();

      const march = frame.encoder.beginRenderPass({
        colorAttachments: [{ view: screen.views.color }, { view: screen.views.motion }],
        depthStencilAttachment: {
          view: screen.views.depth,
          depthClearValue: 1,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
        timestampWrites: marchTime.writes(),
      });
      march.setViewport(0, 0, region.width, region.height, 0, 1);
      jelly.draw(march);
      march.end();
      aa.record(frame);

      final.record(frame, surface.view(frame), { timestampWrites: outputTime.writes() });
      gpu.capture(frame);
    },

    destroy() {
      for (const made of [gpu, screen, aa, slider, sans, label, world, critters, land, jelly]) {
        made.destroy();
      }
    },
  };
}
