// @ts-nocheck: the fixture's packages are no dependency, so nothing resolves them.
import * as fixture from '@kit-fixtures/fixture';
import type * as shade from '@kit-fixtures/fixture/shade';

export const make = (root: unknown) => fixture.spinner.create(root, { label: 'one' });
export const later = () => import('@kit-fixtures/fixture/tool');
export type Shade = typeof shade;
