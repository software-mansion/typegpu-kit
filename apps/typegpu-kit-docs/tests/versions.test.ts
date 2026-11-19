import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { skillFile } from '../src/contracts/agents.ts';
import { repoRoot } from '../src/contracts/extract.ts';

/** The workspace's default catalog: what the repo builds and tests with. */
function catalog() {
  const workspace = readFileSync(join(repoRoot, 'pnpm-workspace.yaml'), 'utf8');
  const block = /^catalog:\n((?: {2}.*\n)+)/m.exec(workspace)![1];
  return new Map(
    [...block.matchAll(/^ {2}'?([^':]+)'?: (.+)$/gm)].map(([, name, version]) => [name!, version!]),
  );
}

const pages = join(repoRoot, 'apps/typegpu-kit-docs/src/content/docs');
const files = [
  skillFile,
  ...readdirSync(pages, { recursive: true, encoding: 'utf8' })
    .filter((file) => /\.mdx?$/.test(file))
    .map((file) => join(pages, file)),
];

describe('install lines', () => {
  it('name the versions the repo tests with', () => {
    const versions = catalog();
    const named = files.flatMap((file) =>
      [
        ...readFileSync(file, 'utf8').matchAll(
          /(@typegpu\/[\w-]+|typegpu)@(\d+\.\d+\.\d+(?:-[\w.]+)?)/g,
        ),
      ].map(([, name, version]) => ({ file, name: name!, version: version! })),
    );

    expect(named.length).toBeGreaterThan(0);
    for (const { file, name, version } of named) {
      expect(`${name}@${version} in ${file}`).toBe(`${name}@${versions.get(name)} in ${file}`);
    }
  });
});
