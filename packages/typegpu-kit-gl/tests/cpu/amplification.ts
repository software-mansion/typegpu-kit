import type { compareCaptures, ExampleDifference } from './example-comparison.ts';

interface Mechanism {
  example: string;
  resource: RegExp;
  mechanism: string;
  evidence: string;
}

const mechanisms: Mechanism[] = [
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
