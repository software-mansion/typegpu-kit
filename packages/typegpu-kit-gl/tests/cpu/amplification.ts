import type { compareCaptures, ExampleDifference } from './example-comparison.ts';

interface Mechanism {
  example: string;
  resource: RegExp;
  mechanism: string;
  evidence: string;
}

const mechanisms: Mechanism[] = [
  {
    example: 'censer',
    resource: /^texture:smoke advected:host#0$/,
    mechanism:
      'normalized-coordinate rounding crosses hardware filter weight thresholds; position contraction feeds force/carry cancellation',
    evidence:
      'CPU_ADVECTION_TRACE replay: all 85 residual texels traced, 595 samples fit 8-bit interpolation plus half-result rounding within one half ulp; substituting sampled values and world position puts every final store within one half ulp. Worst texel (93,26,49) crosses fraction 0.998046875 with a one-ulp coordinate change',
  },
  {
    example: 'survey-model',
    resource: /^buffer:contours_points:host#0$/,
    mechanism: 'height rounding crosses the final contour-bisection threshold',
    evidence:
      'contour-thresholds.test.gpu.ts: both differing points change the tenth bisection decision. Identical-point height errors are at most 5.22e-7; row 1094 straddles -0.2 and row 3384 straddles 0.28',
  },
  {
    example: 'stampede',
    resource: /^buffer:(?:positionData|attributeData):typegpu-kit\/src\/render\/meshPool.ts#0$/,
    mechanism:
      'polar-coordinate builtin accuracy and finite-difference height cancellation affect packed normals',
    evidence:
      'terrain-thresholds.test.gpu.ts reproduces the first differing position. On identical positions all sampled heights differ by at most 7.17e-7 relative; subtracting adjacent heights and packing normalized derivatives amplifies these differences',
  },
  {
    example: 'fireworks',
    resource: /^buffer:(?:positionData|attributeData):typegpu-kit\/src\/render\/meshPool.ts#1$/,
    mechanism:
      'atan2 has unbounded accuracy on zero axes; polar-coordinate and finite-difference calculations amplify bounded differences elsewhere',
    evidence:
      'thresholds and terrain-thresholds probes: CPU atan2(+0,+0)=0 and atan2(+0,-0)=pi, while Dawn returns NaN; land heights are 8.156269 or 5.843728 versus 7. Signed-zero axis samples also reverse two derivative taps. Non-axis same-input height samples stay within the field policy',
  },
  {
    example: 'jelly-world',
    resource: /^texture:texture:host#0$/,
    mechanism:
      'cubic distance-root cancellation and a nearest-segment tie amplify permitted float differences',
    evidence:
      'curve-thresholds.test.gpu.ts: texel 23060 has sqrt(h)-q equal to zero on the CPU and -1.86e-7 on Dawn before the cube root; texel 19052 selects different adjacent segments whose distances differ by one f32 ulp',
  },
  {
    example: 'spider',
    resource: /^buffer:(?:grid:host#0|positionData:typegpu-kit\/src\/render\/meshPool.ts#0)$/,
    mechanism:
      'coordinate division and contraction differences are scaled through noise octaves and cancellation',
    evidence:
      'noise-stages.test.gpu.ts reproduces all four selected differing heights. Identical coordinates have exact fractional parts, gradient differences at most 5.96e-8 and final noise differences at most 2.38e-7; the octave inputs differ before noise evaluation',
  },
  {
    example: 'sph-sculpt',
    resource: /^buffer:heights:host#0$/,
    mechanism:
      'rim length rounding is amplified by a narrow smoothstep and terrain-height cancellation',
    evidence:
      'noise-stages.test.gpu.ts: row 3960 accumulated height agrees at 3.1578631401, rim differs by two f32 ulps (0.9804323316 / 0.9804324508), and the final mix differs by 9.06e-6; identical-input noise stays within 2.38e-7',
  },
  {
    example: 'jellies',
    resource: /^buffer:(?:pointBuffer|lastBuffer):typegpu-kit-examples\/src\/jellies\/swarm.ts#0$/,
    mechanism:
      'repeated Verlet integration and strand constraints propagate builtin and contraction differences during prewarm',
    evidence:
      'replay:jellies:trailing:15 passes the ordinary field policy from identical captured inputs; whole-scene differences occur after repeated prewarm steps, not in that isolated dispatch',
  },
  {
    example: 'ocean',
    resource: /^buffer:omegas:host#0$/,
    mechanism: 'one-ulp dispersion error crosses an integer frequency-quantization threshold',
    evidence:
      'thresholds.test.gpu.ts: input length is exactly equal; omega differs by one f32 ulp, omega/omega0 is 88 versus 87.99999237, so floor chooses adjacent frequencies',
  },
  {
    example: 'kinetic',
    resource: /^(?:texture:<unnamed>:host#[01]|buffer:shBuffer:host#0)$/,
    mechanism:
      'atmosphere radius-minus-Earth cancellation followed by half stores, mipmaps and harmonic projection',
    evidence:
      'thresholds.test.gpu.ts atmosphericHeights: affected cube texels have at most one f32 ulp radius difference; subtracting 6360 amplifies it to 0.00048828125 km (up to 8.3% of height). Replay of procedural daylight has no texture sampling and differs in only 73 base-level channels per output',
  },
  {
    example: 'flow-field',
    resource: /^buffer:particles:/,
    mechanism:
      'permitted division and noise rounding changes a float hash seed and the respawn branch',
    evidence:
      'thresholds.test.gpu.ts: identical initial particles; motion within accuracy; identical updated seeds produce identical randf; 7287 shifted seeds and 61 changed respawn decisions among 65536 rows',
  },
];

/** Only source-traced mechanisms are classified; a transcendental call alone is not evidence. */
export function classifyAmplification(example: string, result: ReturnType<typeof compareCaptures>) {
  const amplified: (ExampleDifference & { mechanism: string; evidence: string })[] = [];
  const differences = result.differences.filter((difference) => {
    if (difference.order.length) return true;
    const known = mechanisms.find(
      (item) => item.example === example && item.resource.test(difference.key),
    );
    if (!known) return true;
    amplified.push({ ...difference, mechanism: known.mechanism, evidence: known.evidence });
    return false;
  });
  return {
    ...result,
    differences,
    amplified,
    summary: {
      ...result.summary,
      differsReview: differences.filter((item) => !item.order.length).length,
      amplifiedNumerics: amplified.length,
    },
  };
}
