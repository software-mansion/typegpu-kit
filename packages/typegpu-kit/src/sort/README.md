# Radix sort (vendored)

Copied from TypeGPU's `packages/typegpu-algorithm/src` at `717142f23`. Edit only for compiler or platform compatibility, or for a performance change recorded below. `sorter.ts` publishes the radix sort; change the kit's API there, not here.

## Local differences

- `scan/prefixScan.ts` copies and reverses the levels for Hermes.
- `runPass.ts` sets each pipeline on a TypeGPU pass with `setPipeline` in place of `pipeline.with(pass)`, which made an object per dispatch.
