import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { expect, it } from 'vitest';

const read = (path: string) => readFileSync(join(import.meta.dirname, '..', path), 'utf8');
/** A file of the scene the initializer writes for `name`. */
const written = (name: string, file: string) =>
  read(`../../packages/typegpu-create-kit/scenes/${name}/${file}`);
const page = read('src/content/docs/starter-scenes.mdx');

it('shows each scene the initializer writes, whole', () => {
  const blocks = [...page.matchAll(/```ts twoslash title="src\/scene\.ts"\n([\s\S]*?\n)```/g)];
  expect(blocks.map(([, body]) => body)).toEqual(
    ['shader', 'lit', 'finished', 'simulation'].map((name) => written(name, 'scene.ts')),
  );
});

it('shows the entry file of the scenes that orbit', () => {
  const [, entry] =
    /```ts twoslash title="src\/index\.ts"\n[\s\S]*?\/\/ ---cut---\n([\s\S]*?\n)```/.exec(page)!;
  for (const name of ['lit', 'finished', 'simulation'])
    expect(written(name, 'index.ts')).toBe(entry);
});
