import { expect, it } from 'vitest';
import { patches } from '../../src/debug/patch.ts';

it('keeps a newer patch when an older one is removed, and leaves no own property behind', () => {
  const target = Object.create({ name: () => 'original' }) as { name(): string };
  const older = patches();
  const newer = patches();
  older.patch(target, 'name', (_self, args, next) => `older ${next(...args)}`);
  newer.patch(target, 'name', (_self, args, next) => `newer ${next(...args)}`);

  older.restore();
  expect(target.name()).toBe('newer original');

  newer.restore();
  expect(target.name()).toBe('original');
  expect(Object.hasOwn(target, 'name')).toBe(false);
});
