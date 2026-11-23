import { d, std } from 'typegpu';

// Unsigned comparisons of these reordered bits preserve float ordering, so atomic minimum and
// maximum work on floats

export const ordered = (value: number) => {
  'use gpu';
  const bits = std.bitcast(d.f32, d.u32)(value);
  return std.select(bits ^ 0x80000000, bits ^ 0xffffffff, bits >= 0x80000000);
};

export const unordered = (bits: number) => {
  'use gpu';
  return std.bitcast(
    d.u32,
    d.f32,
  )(std.select(bits ^ 0xffffffff, bits ^ 0x80000000, bits >= 0x80000000));
};

export const orderedOnCpu = (() => {
  const float = new Float32Array(1);
  const bits = new Uint32Array(float.buffer);

  return (value: number) => {
    float[0] = value;
    const x = bits[0];
    return (x >= 0x80000000 ? ~x : x ^ 0x80000000) >>> 0;
  };
})();
