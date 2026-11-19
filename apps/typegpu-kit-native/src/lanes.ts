// The touch routers of the screens on show, by lane. touch.ts imports this file relatively from its
// worklets, so the UI runtime loads it again and keeps the routers there between touch events.

import type { TouchRouter } from './touches.ts';
import type { Target } from './touch.ts';

export interface Lane {
  readonly router: TouchRouter;
  /** Where the router's actions go, replaced on every event as the screen re-renders. */
  target: Target;
}

export const lanes = new Map<number, Lane>();
