import { cameras, clock, passes, type Frame, type Surface, trails } from '@typegpu/kit';
import { caps, joins } from '@typegpu/kit/geometry';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { flatPoint } from './stage.ts';

/** World units from the bottom of the view to the top. */
const HEIGHT = 10;

/** Seconds per fixed step, which pushes once in that mode. */
export const STEP = 1 / 60;

/** The screens a push per frame is shown on, in frames a second. */
export const SCREENS = [30, 60, 120, 240] as const;

export const JOINS = { round: joins.round, miter: joins.miter } as const;
export const CAPS = { round: caps.round, arrow: caps.arrow, butt: caps.butt } as const;

export type Join = keyof typeof JOINS;
export type Cap = keyof typeof CAPS;
export type Blend = 'add' | 'over';
/** When the kernel pushes: once a fixed step, or once a frame of the screen. */
export type Push = 'step' | 'frame';

export interface Look {
  readonly length: number;
  readonly join: Join;
  readonly cap: Cap;
  readonly blend: Blend;
}

export const START = {
  length: 48,
  width: 0.3,
  join: 'round',
  cap: 'round',
  blend: 'add',
  push: 'step',
  screen: 120,
} as const satisfies Look & { width: number; push: Push; screen: number };

/** Points the pointer's thread keeps at most: the longest `length` the figure offers. */
export const LONGEST = 128;

/** World units the pointer moves before the thread takes another point. */
export const GAP = 0.3;

/** The thread is trail 0; the orbits are the trails after it. */
const THREAD = 0;

const Orbit = d.struct({
  /** World units from the centre, and radians a second around it. */
  radius: d.f32,
  speed: d.f32,
  /** A smaller circle's radius, which it rounds `turns` times a turn. */
  swirl: d.f32,
  turns: d.f32,
});

const ORBITS = [
  { radius: 3.4, speed: 1.1, swirl: 0, turns: 0 },
  { radius: 2.4, speed: -1.4, swirl: 0.6, turns: 5 },
  { radius: 3.9, speed: 0.7, swirl: 0.9, turns: -3 },
  { radius: 1.9, speed: 1.9, swirl: 0.35, turns: 4 },
];
const ORBITERS = ORBITS.length;

/** Each trail's colour and opacity: the thread first, then the orbits'. */
const HUES = [
  d.vec4f(0.2, 0.85, 0.92, 1),
  d.vec4f(0.5, 0.66, 0.98, 1),
  d.vec4f(0.97, 0.71, 0.31, 1),
  d.vec4f(0.42, 0.82, 0.65, 1),
  d.vec4f(0.8, 0.65, 0.95, 1),
];

/** A five-pointed star drawn as one closed line, its first point repeated at the end. */
function star(radius: number) {
  const points: number[] = [];
  for (let i = 0; i <= 5; i++) {
    const angle = Math.PI / 2 + (i * 4 * Math.PI) / 5;
    points.push(Math.cos(angle) * radius, Math.sin(angle) * radius, 0);
  }
  return points;
}

/**
 * Trails for the trails page: orbits a kernel pushes, once a fixed step or once a frame of a
 * screen it stands in for, and a thread written from the CPU.
 */
export function createScene(root: TgpuRoot, surface: Surface) {
  const view = cameras.flat(root, { size: surface, height: HEIGHT });
  const canvas = passes.create(root, surface).pipe(view.inject());

  const orbits = root.createReadonly(d.arrayOf(Orbit, ORBITERS), ORBITS);
  const angles = root.createMutable(
    d.arrayOf(d.f32, ORBITERS),
    ORBITS.map((_, i) => i * 1.7),
  );
  const hues = root.createReadonly(d.arrayOf(d.vec4f, ORBITERS + 1), HUES);
  /** Seconds each push moves the orbits on. */
  const tick = root.createUniform(d.f32, STEP);

  const glow = (point: trails.Sample, across: number) => {
    'use gpu';
    const hue = hues.$[point.trail];
    const alpha = hue.w * (1 - across * across) * (1 - point.along);
    return d.vec4f(hue.xyz * alpha, alpha);
  };

  let width: number = START.width;
  let push: Push = START.push;
  let screen = clock.fixedStep({ step: 1 / START.screen, maxSteps: 8 });
  const steps = clock.fixedStep({ step: STEP, maxSteps: 8 });

  const thread = new Float32Array(LONGEST * 3);
  thread.set(star(1.3));
  let held = 6;
  const line = () => thread.subarray(0, held * 3);

  const make = (look: Look) => {
    const set = trails.create(root, {
      capacity: ORBITERS + 1,
      length: look.length,
      blend: look.blend,
      join: JOINS[look.join],
      startCap: CAPS[look.cap],
      endCap: look.cap === 'arrow' ? caps.butt : CAPS[look.cap],
      shade: glow,
      settings: { width },
    });

    const orbit = tgpu.computeFn({
      workgroupSize: [8],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      if (gid.x >= ORBITERS) return;

      const path = orbits.$[gid.x];
      const angle =
        std.fract((angles.$[gid.x] + path.speed * tick.$) / (2 * Math.PI)) * 2 * Math.PI;
      angles.$[gid.x] = angle;

      const around = d.vec2f(std.cos(angle), std.sin(angle)) * path.radius;
      const swirl = d.vec2f(std.cos(angle * path.turns), std.sin(angle * path.turns)) * path.swirl;
      set.push(gid.x + 1, d.vec3f(around + swirl, 0));
    });

    const moving = root.pipe(set.inject()).createComputePipeline({ compute: orbit });
    set.write(THREAD, line());
    return { set, moving, drawn: canvas.add(set.drawing) };
  };

  let shown = make(START);

  /** Adds a point to the thread once it is `GAP` from the last. */
  const extend = (x: number, y: number) => {
    const last = (held - 1) * 3;
    if (Math.hypot(x - thread[last], y - thread[last + 1]) < GAP) return;

    if (held === LONGEST) {
      thread.copyWithin(0, 3);
      held--;
    }
    thread.set([x, y, 0], held * 3);
    held++;
    shown.set.write(THREAD, line());
  };

  /** Where the stroke went down, until it starts the thread again a `GAP` away. */
  let pressed: readonly [number, number] | undefined;

  return {
    /** A stroke writes the thread, starting it again at its first move, so a scroll leaves it. */
    pointing: {
      drag(u: number, v: number, start: boolean) {
        const [x, y] = flatPoint(surface, HEIGHT, u, v);
        if (start) {
          pressed = [x, y];
          return;
        }
        if (pressed) {
          if (Math.hypot(x - pressed[0], y - pressed[1]) < GAP) return;
          thread.set([...pressed, 0]);
          held = 1;
          pressed = undefined;
        }
        extend(x, y);
      },
    },

    /** Makes the set again: a new length, join, cap or blend means new pipelines. */
    remake(look: Look) {
      shown.drawn.destroy();
      shown.set.destroy();
      shown = make(look);
    },

    /** World units across a trail's newest point. */
    widen(across: number) {
      width = across;
      shown.set.settings.patch({ width });
    },

    /** Pushes once a fixed step, or once a frame of a screen of `hz` frames a second. */
    pushEvery(mode: Push, hz: number) {
      push = mode;
      screen = clock.fixedStep({ step: 1 / hz, maxSteps: 8 });
      tick.write(mode === 'step' ? STEP : 1 / hz);
    },

    /** Seconds of motion a full trail spans. */
    get seconds() {
      return shown.set.length * (push === 'step' ? STEP : screen.step);
    },

    frame(frame: Frame) {
      view.update(frame.dt);

      const pushes = (push === 'step' ? steps : screen).update(frame.dt);
      if (pushes > 0) {
        const compute = frame.encoder.beginComputePass();
        compute.setPipeline(shown.moving);
        for (let i = 0; i < pushes; i++) compute.dispatchWorkgroups(1);
        compute.end();
      }

      canvas.record(frame, shown.drawn);
    },

    destroy() {
      shown.drawn.destroy();
      shown.set.destroy();
      orbits.buffer.destroy();
      angles.buffer.destroy();
      hues.buffer.destroy();
      tick.buffer.destroy();
      canvas.destroy();
    },
  };
}

export type TrailsFigure = ReturnType<typeof createScene>;
