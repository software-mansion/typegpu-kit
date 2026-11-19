// Writes src/lines.json: each example's non-blank lines of TypeScript, which set its rarity. `pnpm
// fix` runs it, and the pull's tests fail while it is stale.

import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const examples = fileURLToPath(new URL('../../typegpu-kit-examples/src/', import.meta.url));
const out = fileURLToPath(new URL('../src/lines.json', import.meta.url));

/** Non-blank lines of TypeScript in the example `key`'s directory and below. */
const linesOf = (key: string) =>
  readdirSync(join(examples, key), { recursive: true, encoding: 'utf8' })
    .filter((file) => file.endsWith('.ts'))
    .reduce(
      (sum, file) =>
        sum +
        readFileSync(join(examples, key, file), 'utf8')
          .split('\n')
          .filter((line) => line.trim()).length,
      0,
    );

/** Every example's lines by key, sorted by key. */
export const countLines = () =>
  Object.fromEntries(
    readdirSync(examples, { withFileTypes: true })
      .filter((entry) => entry.isDirectory() && existsSync(join(examples, entry.name, 'meta.json')))
      .map((entry) => entry.name)
      .toSorted()
      .map((key) => [key, linesOf(key)]),
  );

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  writeFileSync(out, `${JSON.stringify(countLines(), null, 2)}\n`);
}
