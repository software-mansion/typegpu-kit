import { cameras, lines, passes, type Frame, type Surface } from '@typegpu/kit';
import { caps, joins } from '@typegpu/kit/geometry';
import { d, type TgpuRoot } from 'typegpu';

export const JOINS = { round: joins.round, miter: joins.miter, bevel: lines.joins.bevel } as const;
export const CAPS = {
  round: caps.round,
  square: caps.square,
  butt: caps.butt,
  arrow: caps.arrow,
} as const;

export type Join = keyof typeof JOINS;
export type Cap = keyof typeof CAPS;
export type Units = 'world' | 'points';

export interface Look {
  readonly join: Join;
  readonly cap: Cap;
  readonly units: Units;
}

export const START = {
  join: 'round',
  cap: 'round',
  units: 'world',
  width: 0.2,
  dashed: false,
} as const satisfies Look & { width: number; dashed: boolean };

/** World units from the eye to the middle row, which the view turns about. */
const DISTANCE = 9;
const FOV = 0.8;
/** Dash and gap along each line, in world units. */
const DASH = 0.36;
const GAP = 0.2;

/** One zigzag with sharp and shallow corners, x and y, which each row places at its depth. */
const ZIGZAG = [
  -5.4, -0.5, -3.7, 0.6, -2.6, -0.6, -0.5, 0.4, 0.8, -0.4, 2.1, 0.65, 4, 0.1, 5.4, 0.7,
];
const POINTS = ZIGZAG.length / 2;

/** Each row's height and depth, the nearest first, and its colour. */
const ROWS = [
  { y: -1.25, z: 3, color: [0.2, 0.85, 0.92, 1] },
  { y: -0.2, z: 0, color: [0.97, 0.71, 0.31, 1] },
  { y: 0.95, z: -3, color: [0.42, 0.82, 0.65, 1] },
  { y: 2.15, z: -6, color: [0.8, 0.65, 0.95, 1] },
] as const;

const Shape = d.struct({ dashed: d.u32 });

/**
 * Rows of the same zigzag at four depths for the lines page, drawn in one set whose join, end caps
 * and width units the page chooses. In points, every row keeps the width the middle row has in
 * world units.
 */
export function createScene(root: TgpuRoot, surface: Surface) {
  const view = cameras.orbit(root, {
    size: surface,
    target: [0, 0, 0],
    distance: DISTANCE,
    pitch: 0.18,
    yaw: 0,
    fov: FOV,
  });
  const canvas = passes.create(root, surface).pipe(view.inject());
  const shape = root.createUniform(Shape, { dashed: START.dashed ? 1 : 0 });

  const ink = (fragment: lines.Fragment) => {
    'use gpu';
    let alpha = fragment.coverage * (1 - 0.3 * fragment.across * fragment.across);
    if (shape.$.dashed === 1) alpha *= lines.dash(fragment, DASH, GAP);
    return d.vec4f(fragment.color.xyz * alpha, alpha);
  };

  let width: number = START.width;
  let look: Look = START;
  let time = 0;
  let measured = [0, 0];
  const row = new Float32Array(POINTS * 3);

  /** Points a world unit spans at the middle row's distance. */
  const pointsPerUnit = () => {
    const ratio = surface.pixelRatio ?? 1;
    return surface.height / ratio / (2 * DISTANCE * Math.tan(FOV / 2));
  };

  const write = (set: lines.Lines) => {
    const across = look.units === 'points' ? width * pointsPerUnit() : width;
    for (const [k, { y, z, color }] of ROWS.entries()) {
      for (let i = 0; i < POINTS; i++) row.set([ZIGZAG[i * 2], ZIGZAG[i * 2 + 1] + y, z], i * 3);
      set.write(k, row, { width: across, color });
    }
    measured = [surface.width, surface.height];
  };

  const make = () => {
    const set = lines.create(root, {
      capacity: ROWS.length,
      length: POINTS,
      units: look.units,
      join: JOINS[look.join],
      startCap: look.cap === 'arrow' ? caps.butt : CAPS[look.cap],
      endCap: CAPS[look.cap],
      shade: ink,
      label: 'zigzags',
    });
    write(set);
    return { set, drawn: canvas.add(set.drawing) };
  };

  let shown = make();

  return {
    /** Makes the set again: a new join, cap or units means new pipelines. */
    remake(next: Look) {
      look = next;
      shown.drawn.destroy();
      shown.set.destroy();
      shown = make();
    },

    /** World units across the lines, or across the middle row in points. */
    widen(across: number) {
      width = across;
      write(shown.set);
    },

    dash(dashed: boolean) {
      shape.patch({ dashed: dashed ? 1 : 0 });
    },

    frame(frame: Frame) {
      time += frame.dt;
      view.goal.yaw = 0.55 * Math.sin(time * 0.35);
      view.update(frame.dt);
      if (
        look.units === 'points' &&
        (measured[0] !== surface.width || measured[1] !== surface.height)
      ) {
        write(shown.set);
      }
      canvas.record(frame, shown.drawn);
    },

    destroy() {
      shown.drawn.destroy();
      shown.set.destroy();
      shape.buffer.destroy();
      canvas.destroy();
    },
  };
}

export type LinesFigure = ReturnType<typeof createScene>;
