# CPU compute verification

The normal suites compare language features, textures, Kit data structures and rendering modules against Dawn. The expression sweep checks 13 416 components, with exact comparisons where WGSL requires them and accuracy bounds for division and transcendental expressions. The seeded skeletons comparison checks poses and joints after 100 frames. The noise tests compare float seeds bit for bit and the Perlin gradient cache within four ulp.

## Whole scenes and controls

`CPU_SWEEP=1` enables the 33-example compilation sweep. Add `CPU_EXECUTE=1` to dispatch their kernels and `CPU_INTERACT=1` to drive pointer moves, presses, drags, buttons, image uploads, setting extremes and options, actions and camera goals. Each action gets a 60 Hz frame and the end gets six more frames for readbacks. `undispatched` lists each compiled entry that received no nonzero dispatch and the source-specific reason. Camera attachment needs a real external texture; the synthetic image is a 2×2 RGBA pattern. Files use the examples' Node loader, whose decoded images are white pixels.

`CPU_EXAMPLE_COMPARE=1` enables `examples.test.gpu.ts`. It runs two frames from the same seed, with a plain Dawn root and a branded CPU fallback root. Render outputs and resources downstream of them are excluded. Atomic allocation order is reported separately and checked by the module invariant tests. Ocean's `lattice` is an excluded backend-specific FFT intermediate; its final maps remain compared.

Integer fields stay exact. Float fields use the larger of four ulp, 1e-5 of their vector's component magnitude and 1e-6 absolute. Half stores allow one half ulp. Packed normalized fields allow one unit per channel. Octahedral normals are compared after seam canonicalization or decoding because opposite edge signs can represent the same normal. These policies live only in tests.

`amplification.ts` lists only mechanisms traced at their source. They appear in the report as `amplified`, not `passesWithinPolicy`; unexplained differences remain failures. A shader merely calling a transcendental is not enough to classify its output.

## Identical-input replay

`CPU_REPLAY=1 CPU_EXAMPLES=<example> CPU_REPLAY_PIPELINE=<regex> CPU_REPLAY_RESOURCE=<regex>` enables `replay.test.gpu.ts`. It snapshots bindings before a CPU dispatch and runs that module and the corresponding WGSL pipeline on identical bytes. `CPU_REPLAY_SKIP` selects a later matching dispatch. Both roots use the fallback branch for this test; the whole-scene comparison uses different backend branches deliberately.

`CPU_REPLAY_BATCH=1` probes spider terrain, survey contours, stampede land, jellies trails, sph-sculpt heights and censer advection. These probes retain unresolved failures. `CPU_THRESHOLDS=1` enables diagnostics for ocean quantization, fireworks' undefined origin angle, flow-field's hash input and atmospheric cancellation.

`CPU_CAPTURE=1 CPU_EXAMPLES=jelly-world CPU_REPLAY_PIPELINE=fill CPU_CAPTURE_DIRECTORY=<directory>` with the unit `replay.test.ts` writes a numeric input capture. Point `CPU_CURVE_INPUT` at that capture's JSON to enable `curve-thresholds.test.gpu.ts`; it skips when the path is absent. The capture is not a repository fixture or required for the normal tests.

## Measured numeric mechanisms

- Flow-field: initial particles agree exactly. Permitted division and noise differences shift 7 287 updated float seeds among 65 536 particles; 61 respawn decisions change. Identical updated seeds produce identical random values. The position differences are hash amplification, not an incorrect integer RNG.
- Kinetic: replaying its procedural daylight bake from identical inputs leaves 73 differing base-level channels per output. It samples no textures. At affected face-0 texels, atmospheric radii differ by at most one f32 ulp; subtracting Earth's radius of 6 360 km amplifies that to 0.00048828125 km, up to 8.3% of the small height. Half stores, mipmaps and projection propagate the difference.
- Every finite half-float bit pattern round-trips, and every finite midpoint rounds to even for both signs. The encoder does not imitate a particular GPU's truncation or filter precision.

## Performance and runtime contract

The accepted paired 31-sample measurements before f32 semantics and after optimization were:

| CPU kernels | Before f32 | Optimized f32 |
| ----------- | ---------: | ------------: |
| forest      |   9.667 ms |      8.758 ms |
| fishtank    |  28.465 ms |     26.622 ms |
| skeletons   |  11.866 ms |     11.443 ms |
| stampede    |  51.131 ms |     56.652 ms |

Retaining compiled kernel closures reduced falling-sand's first-50-dispatch median from 5.147 to 3.130 ms, steady dispatches from 4.527 to 2.983 ms for 16 128 invocations, and its 450-step CPU prewarm from 3.333 to 2.459 s. Fixed dispatch overhead was 1.80 microseconds. The strict-f32 particle microbenchmark measured 9.30 million invocations/s; it overstates the penalty on complete kernels.

These are CPU-only timings, including kernels that the real fallback can run as fragment passes. The closure is retained, but bindings and uniform-derived caches are reset for every dispatch and released afterward. Uniform writes, copies, changed binding ranges and other kernels therefore cannot reuse stale cached values. There is no device versioning contract and no change to `src/cpu/types.ts` for this optimization.
