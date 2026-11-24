import { defineConfig } from 'oxlint';

/** Calls the examples never make, each with the rule it breaks. */
const NEVER = [
  ...['add', 'sub', 'mul', 'div'].map((property) => ({
    object: 'std',
    property,
    message: 'Write vector arithmetic with operators.',
  })),
  ...['createGuardedComputePipeline', 'dispatchThreads'].map((property) => ({
    property,
    message:
      'Write a tgpu.computeFn with workgroupSize and a bounds check, and dispatchWorkgroups.',
  })),
  {
    property: 'writePartial',
    message: 'TypeGPU deprecates writePartial: use patch.',
  },
  {
    property: 'toSorted',
    message: 'React Native’s Hermes lacks it: copy the array, then sort the copy.',
  },
  {
    object: 'Promise',
    property: 'withResolvers',
    message: 'React Native’s Hermes lacks it: resolve from the executor.',
  },
];

export default defineConfig({
  categories: {
    correctness: 'error',
    suspicious: 'error',
  },
  rules: {
    'no-duplicate-imports': 'error',
  },
  ignorePatterns: [
    '**/dist/**',
    '**/.astro/**',
    '.claude/**',
    'apps/typegpu-kit-native/vendor/**',
    'packages/typegpu-kit/src/geometry/**',
    'packages/typegpu-kit/src/sort/**',
    'packages/typegpu-gl/src/**',
  ],
  overrides: [
    {
      // The examples are the kit's checks, read line by line, and run on React Native too, as
      // do the pull and the app that hosts them there.
      files: [
        'packages/typegpu-kit-examples/src/**',
        'packages/typegpu-kit-pull/src/**',
        'apps/typegpu-kit-docs/src/examples/**',
        'apps/typegpu-kit-native/**',
      ],
      rules: {
        'no-console': 'error',
        'no-warning-comments': 'error',
        'typescript/no-explicit-any': 'error',
        'typescript/ban-ts-comment': 'error',
        'no-restricted-properties': ['error', ...NEVER],
      },
    },
    {
      // The kit runs on React Native too, and follows the rules it gives the examples.
      files: ['packages/typegpu-kit/src/**'],
      rules: {
        'no-restricted-properties': ['error', ...NEVER],
      },
    },
  ],
});
