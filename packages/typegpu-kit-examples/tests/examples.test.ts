import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { categories, exampleOf, examples } from '../src/index.ts';

/** Where the models and textures an example ships come from, each under CC0. */
const SOURCES = [
  'Blender demo files',
  'KayKit',
  'Kenney',
  'Poly Haven',
  'Quaternius',
  'ambientCG',
  'for the kit',
];

/**
 * The marks of a hash written by hand, where `@typegpu/noise` belongs: sine-fract's 43758, PCG's
 * multipliers, Wang's 0x27d4eb2d and the golden ratio's 0x9e3779b9.
 */
const HASH = /43758|747796405|2891336453|0x27d4eb2d|0x9e3779b9|fract\((?:std\.)?sin\(/i;

const at = (path: string) => new URL(path, import.meta.url);

/** Every file in an example's directory and below, as absolute paths. */
const filesOf = (key: string) =>
  readdirSync(at(`../src/${key}/`), { recursive: true, withFileTypes: true })
    .filter((entry) => entry.isFile())
    .map((entry) => join(entry.parentPath, entry.name));

const directories = (url: URL) =>
  readdirSync(url, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name);

/** Every example directory on disk, `<name>`, as the module contracts find them too. */
const found = directories(at('../src/'));

const text = (value: unknown) => typeof value === 'string' && value.length > 0;
const record = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);
const optional = (check: (value: unknown) => boolean) => (value: unknown) =>
  value === undefined || check(value);

/** Each `meta.json` field and what its value must be. */
const META: Readonly<Record<string, (value: unknown) => boolean>> = {
  title: text,
  category: text,
  rarity: text,
  description: text,
  line: text,
  order: Number.isInteger,
  card: record,
  features: (value) => Array.isArray(value) && value.every(text),
  hint: text,
  poster: (value) =>
    record(value) &&
    Object.keys(value).every((field) => field === 'seconds' || field === 'quality') &&
    optional((seconds) => typeof seconds === 'number' && seconds >= 0)(value.seconds) &&
    optional(
      (quality) => Number.isInteger(quality) && Number(quality) >= 0 && Number(quality) <= 100,
    )(value.quality),
};

const REQUIRED = ['title', 'category', 'rarity', 'description', 'line', 'order'];

describe('the examples', () => {
  it('are every directory with a meta.json, in the gallery once each', () => {
    expect(examples.map((example) => example.key).toSorted()).toEqual(found.toSorted());

    const orders = examples.map((example) => example.order);
    expect(new Set(orders).size, 'two examples share an order').toBe(orders.length);

    expect(new Set(examples.map((example) => example.category))).toEqual(
      new Set(categories.map((category) => category.name)),
    );
  });

  it('keep to the Meta fields in meta.json, which JSON imports leave unchecked', () => {
    const wrong = found.flatMap((key) => {
      const meta = JSON.parse(readFileSync(at(`../src/${key}/meta.json`), 'utf8')) as object;
      return [
        ...REQUIRED.filter((field) => !(field in meta)).map((field) => `${key} lacks ${field}`),
        ...Object.entries(meta)
          .filter(([field, value]) => !META[field]?.(value))
          .map(([field]) => `${key} has a bad or unknown ${field}`),
      ];
    });
    expect(wrong).toEqual([]);
  });

  it.each(found)('%s has a scene, a poster and a test, and no page code', async (key) => {
    for (const file of ['meta.json', 'scene.ts', 'poster.webp']) {
      expect(existsSync(at(`../src/${key}/${file}`)), `${key}/${file}`).toBe(true);
    }
    for (const file of ['index.ts', 'index.html']) {
      expect(existsSync(at(`../src/${key}/${file}`)), `${key}/${file}`).toBe(false);
    }
    expect(existsSync(at(`./${key}.test.ts`)), `tests/${key}.test.ts`).toBe(true);

    const example = exampleOf(key);
    expect((await example.scene()).createScene).toBeTypeOf('function');
    expect(!!example.assets, 'listed with its assets.ts').toBe(
      existsSync(at(`../src/${key}/assets.ts`)),
    );
  });

  it.each(found)('%s names the CC0 source of the files it ships', (key) => {
    const shipped = filesOf(key).filter(
      (path) => !/\.(?:ts|json|html)$/.test(path) && !path.endsWith('/poster.webp'),
    );
    if (shipped.length === 0) return;

    const { description } = exampleOf(key);
    expect(description, `${key}/meta.json`).toContain('CC0');
    const named = SOURCES.filter((source) => description.includes(source));
    expect(named, `${key}/meta.json names none of ${SOURCES.join(', ')}`).not.toEqual([]);
  });

  it.each(found)('%s takes its randomness from @typegpu/noise, with no hash of its own', (key) => {
    for (const path of filesOf(key).filter((each) => each.endsWith('.ts'))) {
      expect(readFileSync(path, 'utf8'), path).not.toMatch(HASH);
    }
  });
});
