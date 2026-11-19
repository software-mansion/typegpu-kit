import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { expect, it } from 'vitest';
import { setupPage } from '../src/contracts/agents.ts';
import { repoRoot } from '../src/contracts/extract.ts';

it('names every patch the kit ships, and no other', () => {
  const shipped = readdirSync(join(repoRoot, 'packages/typegpu-kit/patches'));
  const page = readFileSync(setupPage, 'utf8');
  const from = page.indexOf('{/* BEGIN:setup');
  const to = page.indexOf('{/* END:setup */}');
  const setup = page.slice(from, to);
  const named = new Set([...setup.matchAll(/[\w.@-]+\.patch\b/g)].map(([file]) => file));

  expect(from).toBeGreaterThanOrEqual(0);
  expect(to).toBeGreaterThan(from);
  expect(shipped.length).toBeGreaterThan(0);
  expect([...named].toSorted()).toEqual(shipped.toSorted());
});
