import { decodeHalf, encodeHalf } from './textures.ts';
import { vector } from './values.ts';

export function pack(values: number[], bits: number, kind: string): number {
  let result = 0;
  const mask = 2 ** bits - 1;
  for (let i = 0; i < 32 / bits; i++) {
    let value = values[i]!;
    if (kind === 'float') value = encodeHalf(value);
    else if (kind === 'unorm') value = Math.floor(Math.min(Math.max(value, 0), 1) * mask + 0.5);
    else if (kind === 'snorm')
      value = Math.floor(Math.min(Math.max(value, -1), 1) * (2 ** (bits - 1) - 1) + 0.5);
    result |= (value & mask) << (i * bits);
  }
  return result >>> 0;
}

export function unpack(value: number, bits: number, kind: string): number[] {
  const mask = 2 ** bits - 1;
  return vector(
    Array.from({ length: 32 / bits }, (_, i) => {
      const word = (value >>> (i * bits)) & mask;
      if (kind === 'float') return decodeHalf(word);
      if (kind === 'unorm') return Math.fround(word / mask);
      if (kind === 'snorm')
        return Math.fround(
          Math.max(-1, ((word << (32 - bits)) >> (32 - bits)) / (2 ** (bits - 1) - 1)),
        );
      return kind === 'i' ? (word << (32 - bits)) >> (32 - bits) : word;
    }),
  );
}
