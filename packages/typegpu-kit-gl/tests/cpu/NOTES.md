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

## Hardening findings

The interaction sweep found one runtime error: conditional expressions could emit an unresolved `select` identifier because the base generator bypassed `emitCall`. Commit `3b70f435` routes those expressions through the CPU emitter; the calls fixture covers scalar and vector ternaries, and the expression suite remains green. Interaction controls now also drag while a held UI button remains pressed, reaching sph-sculpt's build-stroke height-map kernel.

The following differences have source-level evidence and are reported as numerical amplification, separately from resources that meet the ordinary field policy:

- Survey-model: both traced contour points choose different sides on the tenth bisection. Identical-point height errors are at most 5.22e-7; values straddle the levels -0.2 and 0.28. Earlier versions of this diagnostic incorrectly used floating division for the ring number; the corrected probe uses `std.intdiv` and passed Dawn locally.
- Spider: octave inputs already differ from coordinate division/contraction. Re-evaluating identical inputs gives exact fractional coordinates, gradient differences at most 5.96e-8, quintic differences at most 4.18e-7 and noise differences at most 2.39e-7. Scaling coordinates through four octaves and cancelling their contributions reproduces the differing heights.
- Sph-sculpt heights: at cell 3960 the accumulated noise height agrees at 3.15786314. The rim differs by two f32 ulps, and the narrow smoothstep followed by a mix with -4 amplifies that to about 9e-6. This classifies the height buffer only, not the environment or material volumes listed below.
- Jelly-world: cubic-distance arithmetic cancels `sqrt(h)-q` to zero on the CPU and a tiny negative value on Dawn before taking a cube root. The resulting root moves. Another texel chooses between adjacent curve segments whose distances differ by one f32 ulp. The fixture records the coefficients, root inputs and selected segment, including finite-difference taps.
- Stampede: the polar-coordinate probe reproduces the differing mesh position. At identical positions the sampled heights differ by at most 7.17e-7 relative. Subtracting nearby heights for normals and quantizing those normals amplifies the errors. These are distinct from culling or state-machine errors.
- Fireworks: the CPU returns zero for `atan2(+0,+0)` and pi for `atan2(+0,-0)`; Dawn returns NaN at the origin. This gives terrain heights 8.156269 or 5.843728 versus 7. Signed-zero axis inputs also reverse two finite-difference taps. Non-axis identical-coordinate heights differ by at most 1.525e-6 relative. WGSL gives no accuracy bound for `atan2` when x is zero, or y is zero/subnormal; this is broader than just the origin. The diagnostic records these inputs separately and asserts the bound only within the specified accuracy range.
- Ocean frequencies: equal input lengths produce dispersion frequencies one f32 ulp apart. Dividing by the base frequency gives 88 versus 87.99999237, so `floor` chooses adjacent bins. This classifies `omegas` only; it does not excuse every downstream texture automatically.
- Jellies: replaying `trailing` dispatch 15 from identical inputs meets the ordinary field policy. The whole-scene point and previous-point buffers differ after repeated prewarm integration and strand constraints. This is evidence for accumulation in that path, not an exhaustive comparison of every prewarm dispatch.

The numerical accuracy references are [WGSL concrete expression accuracy](https://www.w3.org/TR/WGSL/#floating-point-accuracy) and [atan2](https://www.w3.org/TR/WGSL/#atan2-builtin). Rounding and classification rules live in the tests; the production runtime has no comparison tolerance or device-specific precision mode.

### Half stores and filtering

The orchestrator's 4096-value half-store ramp found 6069 differing channels out of 12288: every difference was exactly one half ulp, with the CPU's magnitude greater. The CPU rounds to nearest even; this Dawn/Metal device truncates texture stores. Both choices are allowed. Every finite half value and midpoint is checked independently in the unit suite.

The two-texel filtering probe found 3327 differing components. All 3327 match a model with 8-bit interpolation coordinates and a half-rounded sample result; 2307 already match just the coordinate model. All are bounded by half a weight step times the tap range plus one half ulp. This is measured filter precision, not a reason to make the CPU imitate Metal.

Censer's identical-input advection replay differs in 8868 components beyond the one-store allowance. A test-only 8-bit coordinate model reduces that to 232; also rounding sample results to half reduces it to 110. Those 110 components, with a maximum distance of 62 half ulps, remain unexplained. The models are injected into a separate compiled test module with restored input bytes; production texture sampling is unchanged. Kinetic's remaining 73 base-level components per output are explained by the separately traced atmospheric cancellation; that kernel samples no textures.

### Unresolved comparisons

These are still review failures in the opt-in whole-example comparison, not accepted tolerances:

- Censer: the 110 components left by the filter model need a finer trace of advection arithmetic or hardware interpolation. Whole-scene differences also propagate into smoke velocity/curls and the pressure hierarchy.
- Ocean: sky/cloud, environment, foam and wave-map outputs beyond the classified frequency buffer have not all been traced. Half-store chains and frequency quantization are known sources, but their presence alone does not prove every reported output difference.
- Sph-sculpt: the two environment textures, harmonic buffer, and `origin`/`volume` material textures remain unclassified. The height buffer's rim mechanism is established separately.

The detailed resource keys, differing-word counts and first values remain in `results.json`. Render-dependent resources, backend-specific scratch storage and resources with atomic allocation order are listed separately; their exclusion is not a claim of equality.
