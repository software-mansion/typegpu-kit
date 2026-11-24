import { expect, it } from 'vitest';
import { completeCall, projectRecordBody, usedRecordFields } from '../../src/cpu/record-fields.ts';

it('distinguishes complete calls and record escapes', () => {
  expect(completeCall('make(a,call(b),"(")')).toEqual({
    name: 'make',
    arguments: ['a', 'call(b)', '"("'],
  });
  expect(completeCall('make(a)+other(b)')).toBeUndefined();
  expect(usedRecordFields('return r.position[0]+r.member;', 'r')).toEqual(['member', 'position']);
  expect(usedRecordFields('return f(r);', 'r')).toBeUndefined();
});

it('does not project dependent or escaping records', () => {
  for (const body of [
    '{let result = cpuType();\n(result.a) = (result.b);\nreturn result;}',
    '{let result = cpuType();\nescape(result);\nreturn result;}',
    '{let result = cpuType();\nresult[index] = 3;\nreturn result;}',
  ])
    expect(
      projectRecordBody(
        body,
        ['a'],
        true,
        () => '0',
        () => undefined,
      ),
    ).toBeUndefined();
});

it('loads a selected field before later source mutations', () => {
  const body =
    '{\nlet result = cpuType();\n(result.a) = (source[0]);\n(result.b) = (source[1]);\nreturn result;\n}';
  const projected = projectRecordBody(
    body,
    ['a'],
    true,
    () => '0',
    () => undefined,
  )!;
  expect(projected).not.toContain('source[1]');
  const source = [3, 4];
  const value = new Function('source', `return (function()${projected})();`)(source);
  source[0] = 10;
  expect(value).toBe(3);
});
