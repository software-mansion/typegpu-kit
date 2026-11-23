import { readFileSync } from 'node:fs';
import { font, type Baked } from '../../src/index.ts';

/** The committed Inter bake, as an app loads it: the JSON and the atlas's bytes. */
export const inter: Baked = font.read(
  readFileSync(new URL('../../fonts/inter.json', import.meta.url), 'utf8'),
  readFileSync(new URL('../../fonts/inter.rgba', import.meta.url)),
);
