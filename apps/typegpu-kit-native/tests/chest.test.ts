/// <reference types="node" />
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { font, type FontData } from '@typegpu/kit';
import { expect, it } from 'vitest';
import { captionOf, type Label } from '../src/chest/caption.ts';
import { entries } from '../src/chest/cards.ts';

const baked = (at: string) =>
  font.read(
    JSON.parse(readFileSync(path.resolve(import.meta.dirname, `${at}.json`), 'utf8')),
    readFileSync(path.resolve(import.meta.dirname, `${at}.rgba`)),
  ).data;

const lacking = (data: FontData, labels: readonly Label[]) =>
  labels.flatMap(({ text }) => [...text]).filter((character) => !(character in data.glyphs));

it("bakes every character the chest's cards print; its fonts README bakes more", () => {
  const display = baked('../../../packages/typegpu-kit/fonts/inter');
  const mono = baked('../assets/fonts/mono');
  const [displayMetrics, monoMetrics] = [font.metrics(display), font.metrics(mono)];

  const missing = entries.flatMap((entry) => {
    const caption = captionOf(entry, 340, 476, 1 / 3, displayMetrics, monoMetrics);
    return [...lacking(display, caption.display), ...lacking(mono, caption.mono)];
  });
  expect([...new Set(missing)].join('')).toBe('');
});
