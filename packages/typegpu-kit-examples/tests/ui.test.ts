import { describe, expect, expectTypeOf, it } from 'vitest';
import type { ExampleScene, Picture } from '../src/contract.ts';
import {
  buttons,
  image,
  text,
  type AnyButtonsKind,
  type Bound,
  type Held,
  type Tap,
  type Ui,
} from '../src/ui.ts';

const ui = {
  score: text({ corner: 'top-right' }),
  controls: buttons(
    { left: { label: '◀', keys: ['ArrowLeft'], held: true }, drop: { label: 'drop' } },
    { left: [['left']], right: [['drop']] },
  ),
  picture: image('choose image'),
};

describe('UI declarations', () => {
  it('builds plain data', () => {
    expect(structuredClone(ui)).toEqual(ui);
    expect(ui.controls.pads.right).toEqual([['drop']]);
  });

  it('is a UI, and hosts take its buttons whatever they are named', () => {
    expectTypeOf(ui).toExtend<Ui>();
    expectTypeOf(ui.controls).toExtend<AnyButtonsKind>();
  });

  it('types the handlers a scene binds', () => {
    expectTypeOf<Bound<typeof ui>['score']>().toEqualTypeOf<() => string>();
    expectTypeOf<Bound<typeof ui>['controls']['left']>().toEqualTypeOf<Held>();
    expectTypeOf<Bound<typeof ui>['controls']['drop']>().toEqualTypeOf<Tap>();
    expectTypeOf<Bound<typeof ui>['picture']>().toEqualTypeOf<(picture: Picture) => void>();
  });

  it('rejects a pad naming a missing button and a held button without release', () => {
    // @ts-expect-error `jump` is not a button
    buttons({ drop: { label: 'drop' } }, { left: [['jump']] });

    const bound: Bound<typeof ui> = {
      score: () => '0',
      // @ts-expect-error a held button needs `release`
      controls: { left: { press() {} }, drop: { press() {} } },
      picture: () => {},
    };
    // Any binding is an example scene's `ui`
    const scene: Pick<ExampleScene, 'ui'> = { ui: bound };
    expect(scene.ui).toBe(bound);
  });
});
