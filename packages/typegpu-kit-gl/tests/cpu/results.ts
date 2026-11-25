import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

export function recordResults(section: string, value: unknown): void {
  const file = new URL('./results.json', import.meta.url);
  const previous = existsSync(file)
    ? (JSON.parse(readFileSync(file, 'utf8')) as Record<string, unknown>)
    : {};
  previous[section] = value;
  writeFileSync(file, `${JSON.stringify(previous, null, 2)}\n`);
  execFileSync(fileURLToPath(new URL('../../../../node_modules/.bin/oxfmt', import.meta.url)), [
    fileURLToPath(file),
  ]);
}

export function recordExampleResults(section: string, value: { example: string }[]): void {
  const file = new URL('./results.json', import.meta.url);
  const previous = existsSync(file)
    ? (JSON.parse(readFileSync(file, 'utf8')) as Record<string, unknown>)
    : {};
  const old = (previous[section] ?? []) as { example: string }[];
  recordResults(section, [
    ...old.filter((entry) => !value.some((current) => current.example === entry.example)),
    ...value,
  ]);
}
