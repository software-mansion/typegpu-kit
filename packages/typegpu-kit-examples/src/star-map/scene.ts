import {
  cameras,
  passes,
  picker,
  pipelines,
  targets,
  bloom,
  font,
  finish,
  tone,
  type Baked,
  type Frame,
  type Surface,
} from '@typegpu/kit';
import { d, std, type TgpuRoot } from 'typegpu';
import type { Pointing } from '../contract.ts';
import type { Settings } from '../settings.ts';
import { createLabel } from './label.ts';
import { createStars, galaxy, Star, type StarData } from './stars.ts';

const HDR = 'rgba16float';
/** The dark between the stars, linear. */
const SPACE = [0.004, 0.005, 0.012, 1];
/** Radians a second the view turns round the galaxy on its own. */
const DRIFT = 0.03;
/** Brightness multipliers for hovered and selected stars. */
const HOVERED = 1;
const CHOSEN = 1.5;

export interface SceneOptions {
  /** The label's font. */
  readonly font: Baked;
  /** Default 20 000. */
  readonly count?: number;
  /** Chooses the star nearest this point from the start. Default none. */
  readonly choose?: readonly [x: number, y: number, z: number];
}

/** Star selected by a press, with its index. */
export interface Choice {
  readonly index: number;
  readonly star: StarData;
}

/** Index of the star nearest `point`. */
function nearestTo(data: readonly StarData[], [x, y, z]: readonly [number, number, number]) {
  let best = 0;
  let least = Infinity;
  for (const [index, { position }] of data.entries()) {
    const apart = Math.hypot(position[0] - x, position[1] - y, position[2] - z);
    if (apart < least) [best, least] = [index, apart];
  }
  return best;
}

/**
 * Views stars with an orbit camera and picks the hovered star on the GPU. A press chooses it, and
 * a label pinned beside it names it and its distance from the core.
 */
export async function createScene(root: TgpuRoot, surface: Surface, options: SceneOptions) {
  const { count = 20_000 } = options;
  const screen = targets.create(root, surface, { hdr: { format: HDR, sampled: true } });
  const view = cameras.orbit(root, {
    size: surface,
    target: [0, 0, 0],
    distance: 11,
    pitch: 0.7,
    distanceRange: [2, 30],
    damping: 6,
  });

  const data = galaxy(count);
  const stars = root.createReadonly(d.arrayOf(Star, count), data);
  const chosen = root.createUniform(d.u32, picker.none);
  let pointer: [x: number, y: number] | undefined;

  const values = {
    /** Picker margin beyond a star edge, in points. */
    reach: picker.defaults.reach,
    /** Chooses the nearest candidate by screen distance or eye distance. */
    nearest: 'pointer' as 'pointer' | 'eye',
  };

  const byEye = root.createUniform(d.u32, values.nearest === 'eye' ? 1 : 0);
  const picks = picker.create(root, view, {
    capacity: count,
    size: surface,
    position: (i) => {
      'use gpu';
      return d.vec3f(stars.$[i].position);
    },
    radius: (i) => {
      'use gpu';
      return stars.$[i].radius;
    },
    rank: (candidate) => {
      'use gpu';
      return std.select(picker.byPointer(candidate), picker.byEye(candidate), byEye.$ === 1);
    },
    hz: 20,
    settings: { reach: values.reach },
  });

  // Highlight the hovered star in the picker frame and keep the selected star lit
  const field = createStars(root, view, {
    stars,
    format: HDR,
    glow: (i: number) => {
      'use gpu';
      const hovered = std.select(d.f32(0), d.f32(HOVERED), i === picks.found());
      return hovered + std.select(d.f32(0), d.f32(CHOSEN), i === chosen.$);
    },
  });

  const glow = bloom.create(root, {
    source: () => screen.textures.hdr,
    settings: { threshold: 0.9, spread: 1.2 },
  });
  const sans = font.create(root, options.font);
  const label = createLabel(root, sans, view);
  const overlay = passes.create(root, surface);
  const callout = overlay.add(label.shapes);
  const labelled = overlay.add(label.text);
  const final = finish.create(root, {
    format: surface.format,
    source: () => screen.textures.hdr,
    tone: tone.aces,
    settings: { glowIntensity: 0.35, vignette: 0.45 },
    glow,
  });
  await pipelines.compile([picks, field, glow, overlay, final]);

  let drawnFrame = -1;
  let time = 0;

  const settings: Settings<typeof values> = {
    values,
    knobs: {
      reach: { label: 'pick margin (points)', min: 0, max: 64, step: 1 },
      nearest: { label: 'pick priority', options: ['pointer', 'eye'] },
    },

    write() {
      picks.settings.patch({ reach: values.reach });
      byEye.write(values.nearest === 'eye' ? 1 : 0);
    },

    readouts: {
      /** The last pick read back: `none yet`, `null` for no star, or a star index. */
      result: () => {
        const { latest, latestFrame } = picks.result;
        if (latestFrame < 0) return 'none yet';
        return latest === null ? 'null' : `#${latest}`;
      },
      /** Frame index when the readback result was picked. */
      'picked in frame': () => {
        const { latestFrame } = picks.result;
        return latestFrame < 0 ? 'pending' : latestFrame;
      },
      'drawn in frame': () => drawnFrame,
    },
  };

  let choice: Choice | null = null;
  const choose = (index: number | null) => {
    chosen.write(index ?? picker.none);
    choice = index === null ? null : { index, star: data[index] };
    label.show(choice && { index: choice.index, position: choice.star.position }, time);
  };
  if (options.choose) choose(nearestTo(data, options.choose));

  let presses = 0;
  let open = true;
  let pressed: [x: number, y: number] | undefined;
  const pointing = {
    move(x: number, y: number) {
      pointer = [x, y];
    },
    leave() {
      pointer = undefined;
      picks.clear();
    },
    /** Selects the star at the pressed point, or clears selection, once its pick comes back. */
    press(x: number, y: number) {
      const press = ++presses;
      pressed = [x, y];
      picks.result.next().then(
        (found) => {
          if (!open || press !== presses) return;
          pressed = undefined;
          choose(found);
        },
        () => {},
      );
    },
    get reach() {
      return values.reach;
    },
  } satisfies Pointing;

  return {
    view,
    settings,
    pointing,

    /** Selected star, or null. */
    get choice() {
      return choice;
    },

    frame(frame: Frame) {
      drawnFrame = frame.index;
      time = frame.time;
      label.tick(time);
      view.goal.yaw += frame.dt * DRIFT;
      view.update(frame.dt);
      screen.fit();

      const compute = frame.encoder.beginComputePass();
      const at = pressed ?? pointer;
      if (at) picks.pick(compute, at[0], at[1]);
      compute.end();
      picks.result.capture(frame);

      const scene = frame.encoder.beginRenderPass({
        colorAttachments: { view: screen.views.hdr, clearValue: SPACE },
      });
      field.draw(scene);
      scene.end();

      glow.record(frame);

      // The label draws over the finished frame, so neither bloom nor tone mapping touches it
      const output = overlay.begin(frame);
      final.draw(output);
      callout.draw(output);
      labelled.draw(output);
      output.end();
    },

    destroy() {
      open = false;
      for (const made of [screen, stars.buffer, picks, glow, sans, label]) made.destroy();
    },
  };
}
