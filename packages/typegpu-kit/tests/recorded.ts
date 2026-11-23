import type { RecordingLog } from '@typegpu/kit/testing';

/**
 * Returns the events `log` records while `body` runs, such as one frame's passes and submit, and
 * leaves the log whole. Resolves after an async `body`.
 */
export function recorded(log: RecordingLog, body: () => Promise<unknown>): Promise<string[]>;
export function recorded(log: RecordingLog, body: () => unknown): string[];
export function recorded(log: RecordingLog, body: () => unknown) {
  const from = log.events.length;
  const result = body();
  const since = () => log.events.slice(from);
  return result instanceof Promise ? result.then(since) : since();
}
