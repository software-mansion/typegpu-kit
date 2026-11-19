// What the example screen shows over the canvas: the example's UI elements and where they sit

import type { Ui, UiElement } from 'typegpu-kit-examples/ui.ts';

type ElementOf<K extends UiElement['kind']> = Extract<UiElement, { readonly kind: K }>;

/** The elements of kind `kind` in `ui`, as name and element pairs in declaration order. */
export function elementsOf<K extends UiElement['kind']>(
  ui: Ui | undefined,
  kind: K,
): [name: string, element: ElementOf<K>][] {
  const found: [string, ElementOf<K>][] = [];
  for (const [name, element] of Object.entries(ui ?? {})) {
    if (element.kind === kind) found.push([name, element as ElementOf<K>]);
  }
  return found;
}

/** Points from the safe area's top to what sits under the top bar. */
export const UNDER_BAR = 52;

/** Points from the safe area's edges to the corner texts and pads. */
export const EDGE = 16;
