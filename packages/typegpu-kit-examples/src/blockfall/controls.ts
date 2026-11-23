import type { Tap } from '../ui.ts';
import type { createBlockfall } from './game.ts';

const QUARTER = Math.PI / 2;

type Direction = readonly [x: number, z: number];
type Blockfall = ReturnType<typeof createBlockfall>;

/** Rotation axis and turn for tipping a piece along a grid direction. */
const axisAlong = ([x, z]: Direction): [axis: number, turn: number] => (x !== 0 ? [0, x] : [2, z]);

/**
 * The game's buttons, as `ui.ts` declares them. Moves follow the camera's goal, so they turn with a
 * camera turn at once.
 */
export function createControls(game: Blockfall, view: { readonly goal: { yaw: number } }) {
  const along = (act: (away: Direction, right: Direction) => void): Tap => ({
    press() {
      const quarter = Math.round(view.goal.yaw / QUARTER);
      const sin = Math.round(Math.sin(quarter * QUARTER));
      const cos = Math.round(Math.cos(quarter * QUARTER));
      act([-sin, -cos], [cos, -sin]);
    },
  });
  const turn = (by: number): Tap => ({ press: () => (view.goal.yaw += by * QUARTER) });

  return {
    away: along((away) => game.move(...away)),
    near: along(([x, z]) => game.move(-x, -z)),
    left: along((_, [x, z]) => game.move(-x, -z)),
    right: along((_, right) => game.move(...right)),
    spin: { press: () => game.rotate(1, 1) },
    roll: along((_, right) => game.rotate(...axisAlong(right))),
    tip: along((away) => game.rotate(...axisAlong(away))),
    drop: { press: () => game.drop() },
    turnLeft: turn(-1),
    turnRight: turn(1),
  };
}
