// @ts-nocheck: the fixture's packages are no dependency, so nothing resolves them.
// Its exports carry @uses, so an example counts only the modules of the names it imports.
import { spinner } from '@kit-fixtures/fixture';
import * as vendored from '@kit-fixtures/fixture/vendored';

/** @uses fixture/shapes */
export const load = () => vendored.shapes.cube(1);

/** @uses fixture/spinner */
export const spin = (root: unknown) => spinner.create(root, { label: 'shared' });

/** @uses fixture/ghost */
export const haunt = () => undefined;

export const plain = 1;
