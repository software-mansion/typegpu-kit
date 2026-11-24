import { existsSync, readFileSync, writeFileSync } from 'node:fs';

export function recordResults(section: string, value: unknown): void {
  const file = new URL('./results.json', import.meta.url);
  const previous = existsSync(file)
    ? (JSON.parse(readFileSync(file, 'utf8')) as Record<string, unknown>)
    : {};
  previous[section] = value;
  writeFileSync(file, `${JSON.stringify(previous, null, 2)}\n`);
}
