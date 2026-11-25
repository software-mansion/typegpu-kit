import type { compareCaptures, ExampleDifference } from './example-comparison.ts';

interface Mechanism {
  example: string;
  resource: RegExp;
  mechanism: string;
  evidence: string;
}

const mechanisms: Mechanism[] = [
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
