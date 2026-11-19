import { cameras } from '@typegpu/kit';
import * as web from '@typegpu/kit/web';
import { onKeys } from './figure.ts';

/** Pixels of drag an arrow key turns an orbit view by. */
export const TURN = 16;

/**
 * Connects `web.orbitControls` and calls `moved` after camera changes. Vertical swipes and ordinary
 * wheel input scroll the page, and arrow keys turn the focused canvas as a drag does.
 */
export function orbitInPage(
  view: { readonly goal: web.OrbitGoal },
  canvas: HTMLCanvasElement,
  moved: () => void,
): web.OrbitControls {
  const keys = new AbortController();
  const turn = (dx: number, dy: number) => {
    cameras.turn(view.goal, dx * TURN, dy * TURN);
    moved();
  };
  onKeys(canvas, { arrow: turn }, 'arrow keys turn it', keys.signal);
  const controls = web.orbitControls(view, canvas, { page: true, onChange: moved });
  const destroy = () => {
    keys.abort();
    controls.destroy();
  };
  return { destroy, [Symbol.dispose]: destroy };
}
