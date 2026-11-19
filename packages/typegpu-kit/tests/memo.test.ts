import { expect, it, vi } from 'vitest';
import { memo } from '../src/memo.ts';

it('compares every argument', () => {
  const make = vi.fn((scene: object, bloom: object) => [scene, bloom]);
  const groups = memo(make);
  const scene = {};

  groups(scene, {});
  groups(scene, {});

  expect(make).toHaveBeenCalledTimes(2);
});

it('remembers an undefined argument and a call without arguments', () => {
  const split = vi.fn((writes?: object) => ({ writes }));
  const once = vi.fn(() => ({}));
  const splitter = memo(split);
  const only = memo(once);

  splitter(undefined);
  splitter(undefined);
  only();
  only();

  expect(split).toHaveBeenCalledTimes(1);
  expect(once).toHaveBeenCalledTimes(1);
});

it('keeps the last result when the function throws', () => {
  let fail = false;
  const make = memo((key: object) => {
    if (fail) throw new Error('lost device');
    return { key };
  });
  const key = {};
  const kept = make(key);

  fail = true;
  expect(() => make({})).toThrow('lost device');
  expect(make(key)).toBe(kept);
});

it('keeps two results, so arguments that alternate run it twice', () => {
  const make = vi.fn((texture: object) => ({ texture }));
  const group = memo(make);
  const even = {};
  const odd = {};

  const a = group(even);
  const b = group(odd);
  for (let i = 0; i < 4; i++) {
    expect(group(even)).toBe(a);
    expect(group(odd)).toBe(b);
  }
  expect(make).toHaveBeenCalledTimes(2);

  group({});
  expect(group(odd)).toBe(b);
  expect(group(even)).not.toBe(a);
  expect(make).toHaveBeenCalledTimes(4);
});
