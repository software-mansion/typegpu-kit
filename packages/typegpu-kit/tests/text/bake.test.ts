import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, symlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { ASCII, bake, format } from '../../src/text/bake.ts';

const fonts = (name: string) => new URL(`../../fonts/${name}`, import.meta.url);

describe('bake', () => {
  it('makes the committed Inter byte for byte', async () => {
    const { data, atlas } = await bake({
      input: fileURLToPath(fonts('InterDisplay-SemiBold.ttf')),
      characters: '…',
    });

    expect(format(data)).toBe(readFileSync(fonts('inter.json'), 'utf8'));
    expect(Buffer.compare(atlas, readFileSync(fonts('inter.rgba')))).toBe(0);

    expect(new Set(Object.keys(data.glyphs))).toEqual(new Set([...ASCII, '…']));
    expect(data.glyphs[' ']).toEqual({ advance: expect.any(Number) });
    expect(data.kerning.AV).toBeLessThan(0);
    expect(data.capHeight).toBeCloseTo(1490 / 2048, 6);
  });

  it('runs as the command a package manager links', () => {
    const directory = mkdtempSync(path.join(tmpdir(), 'typegpu-kit-bin-'));
    try {
      const linked = path.join(directory, 'typegpu-kit-bake-font');
      symlinkSync(fileURLToPath(new URL('../../src/text/bake.ts', import.meta.url)), linked);
      expect(execFileSync(process.execPath, [linked, '--help'], { encoding: 'utf8' })).toMatch(
        /^Usage: typegpu-kit-bake-font/,
      );
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it('bakes at an em size and range of its own', async () => {
    const { data } = await bake({
      input: fileURLToPath(fonts('InterDisplay-SemiBold.ttf')),
      emSize: 32,
      pixelRange: 4,
    });

    expect(data.source).toMatchObject({ emSize: 32, pixelRange: 4 });
    expect(data.atlas.distanceRange).toBe(4);
    expect(data.glyphs.A?.advance).toBeGreaterThan(0);
    expect(data.kerning.AV).toBeLessThan(0);
  });
});
