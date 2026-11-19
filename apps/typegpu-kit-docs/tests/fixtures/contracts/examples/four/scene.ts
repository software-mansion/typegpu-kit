// @ts-nocheck: the fixture's packages are no dependency, so nothing resolves them.
const { spinner } = await import('@kit-fixtures/fixture');

export const make = (root: unknown) => spinner.create(root, { label: 'four' });
