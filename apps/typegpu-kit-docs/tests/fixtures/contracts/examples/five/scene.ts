// @ts-nocheck: the fixture's packages are no dependency, so nothing resolves them.
// Uses the kit only through files beside the examples, as examples share a loader.
import { haunt, load, plain } from '../shared.ts';
import type { Later } from '../typed.ts';
import { waited } from '../waits.ts';

export const make = () => [load(), haunt(), plain, waited];
export type Made = Later;
