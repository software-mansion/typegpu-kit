# CPU compute verification

The normal suites compare language features, textures, Kit data structures and rendering modules against Dawn. The expression sweep checks 13 416 components, with exact comparisons where WGSL requires them and accuracy bounds for division and transcendental expressions. The seeded skeletons comparison checks poses and joints after 100 frames. The noise tests compare float seeds bit for bit and the Perlin gradient cache within four ulp.

## Numerical follow-up result

The fresh targeted Dawn run of censer, sph-sculpt and ocean passes all three comparisons with zero review failures. No production CPU arithmetic or sampler change was needed. Two replay-harness defects were corrected: matching ping-pong resources without their read/write roles, and reusing explicit layouts with unused bindings absent from a snapshot. Both have a normal Dawn regression.

| Scene      | Within ordinary field policy | Named amplification entries | Review failures |
| ---------- | ---------------------------: | --------------------------: | --------------: |
| censer     |                           10 |                          13 |               0 |
| sph-sculpt |                           14 |                           6 |               0 |
| ocean      |                            5 |                          18 |               0 |

These are resource comparison counts, not bit-identical scenes. Allocation-order and render-dependent exclusions remain listed separately. Detailed traces cover the selected worst or residual cells; full-resource replays and controlled runs have their own assertions. Other examples retain their earlier captures; this follow-up reran these three, not all 33.

Validation at this checkpoint: 67 CPU unit tests and 28 normal Dawn comparisons pass; the targeted writer, filtering, geometry, history and whole-scene probes pass; repository type and style gates pass. The sampler models and source-accuracy intervals exist only in tests.

## Whole scenes and controls

`CPU_SWEEP=1` enables the 33-example compilation sweep. Add `CPU_EXECUTE=1` to dispatch their kernels and `CPU_INTERACT=1` to drive pointer moves, presses, drags, buttons, image uploads, setting extremes and options, actions and camera goals. Each action gets a 60 Hz frame and the end gets six more frames for readbacks. `undispatched` lists each compiled entry that received no nonzero dispatch and the source-specific reason. Camera attachment needs a real external texture; the synthetic image is a 2×2 RGBA pattern. Files use the examples' Node loader, whose decoded images are white pixels.

`CPU_EXAMPLE_COMPARE=1` enables `examples.test.gpu.ts`. It runs two frames from the same seed, with a plain Dawn root and a branded CPU fallback root. Render outputs and resources downstream of them are excluded. Atomic allocation order is reported separately and checked by the module invariant tests. Ocean's `lattice` is an excluded backend-specific FFT intermediate; its final maps remain compared. Its foam history is transposed into matching cascade/x/y order before comparison, with that decoding recorded separately.

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

`compile` accepts the self-describing compute string and ignores unknown directive comments, including GPU-form metadata. `dispatch` takes workgroup counts; the device reads the three u32 words for indirect dispatch before calling it. Buffer offsets and sizes are bytes. Texture mip levels are relative to the bound view, tightly packed with rows top to bottom and slices/layers in order; half textures contain raw `Uint16Array` bits. Samplers use `GPUSamplerDescriptor`. Binding read/write flags conservatively reflect declared access: mutable buffers are both read and written, even if a particular kernel only stores. The device must synchronize those resources accordingly. The original `types.ts` contract is unchanged.

## Initial hardening checkpoint

The initial 33-scene run plus the fishtank readiness rerun has 22 scenes whose compared resources meet the field policy, eight with only source-traced amplification, and three with unresolved differences (ocean, censer, sph-sculpt). All 26 normal Dawn tests pass. Fishtank initially exceeded a 100-tick shader-compilation wait; the harness now waits up to 30 seconds without advancing simulation time, and its rerun passes. The normal unit suite passes 65 tests with four opt-in tests skipped; the all-example compilation sweep passes.

The eight amplified scenes are jelly-world, fireworks, kinetic, stampede, jellies, spider, survey-model and flow-field. Some of the other 22 scenes have no compute-written resources left after render-dependency exclusions; their status is not a render comparison.

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

Censer's identical-input advection replay differs in 8868 components beyond the one-store allowance. A test-only 8-bit coordinate model reduces that to 232; also rounding sample results to half reduces it to 110. The follow-up advection trace below localizes those 110 components, whose maximum distance was 62 half ulps. The models are injected into a separate compiled test module with restored input bytes; production texture sampling is unchanged. Kinetic's remaining 73 base-level components per output are explained by the separately traced atmospheric cancellation; that kernel samples no textures.

### Censer advection residual, traced

`CPU_ADVECTION_TRACE=1 CPU_REPLAY_BATCH=1 CPU_EXAMPLES=censer` with `replay.test.gpu.ts` traces every texel left by the test-only filter model (85 texels, 110 components). `results.json` retains the ten worst full traces and a summary for all 85. Tags 0–15 name dt, at, back, restored, corrected velocity, held matter, curl, curl gradient, normalized gradient, confinement, world position, lift, force sum, dissipation result, carry, and final mix. Tags 100–111 hold normalized coordinates, texel coordinates, fractions, eight clamped taps with raw half bits, and sampled result; 112 is the pre-store value and 113 holds dissipation, sponge and carry-mix factors.

The largest residual is at (93,26,49): normalized y is 0.2365896850824356 on the CPU and 0.2365896999835968 on Dawn (one f32 ulp). The texel-space fractions are 0.9980449676513672 and 0.998046875, straddling the 8-bit filter rounding midpoint. All eight raw taps agree. The predicted x velocity is 6.37769699e-6 under the CPU-coordinate hardware model and 2.68220901e-6 on Dawn, producing the 62-half-ulp difference after dissipation. The samples are interior, not a clamp, out-of-volume or depth-normalization defect.

All 595 samples in the 85 traces agree with trilinear interpolation of Dawn's recorded taps using 8-bit weights and half-rounded results to within one half ulp. Replaying Dawn's sampled values through the CPU module leaves only (75,33,40) two half ulps apart: world position differs by one f32 ulp from contraction; `force`/`carry` amplify it and the final mix cancels to about -9.7e-6. Replaying that world-position input as well puts every affected component within one half-store ulp. These substitutions are diagnostic only; neither the production sampler nor the arithmetic changes. The opt-in test asserts these source bounds and does not claim bit-exact whole-scene equality.

`CPU_CENSER_KERNELS=1 CPU_CENSER_TRACE=1` with `censer-kernels.test.gpu.ts` checks downstream writers after the second advection starts, avoiding the initially empty smoke field. It keeps distinct pipeline labels even when guarded entry functions share the generated name `item`. There are 27 writer/shape cases: 25 meet the ordinary policy from identical inputs. Matter transport's ten worst texels have no misses against the measured sample envelope; substituting Dawn samples puts every final store within one half ulp. Curl, averaging, divergence, every pressure level and projection all pass independently. The fallback matter branch also averages already half-stored values in a separate pass; the WebGPU branch averages pre-store workgroup values, so that branch adds a quantization point before pressure processing.

### Sph-sculpt material and environment traces

`CPU_MATERIAL_TRACE=1` with replay entries `originating` and `generating` (skip 1 for the latter) traces all 192 origin and 269 mip-volume residual texels. Every pre-round search coordinate differs by one f32 ulp and selects a different integer center. For origin (10,13,37), the CPU's 20.500001907348633 rounds to 21 while Dawn's 20.5 rounds to 20: shifting the finite search window loses the nearest candidate. Substituting only Dawn's center makes all 461 encoded output bytes exact. `results.json` retains every summary and the ten worst full traces.

`CPU_SKY_TRACE=1` on the base environment bake traces the ten worst texels. Atmospheric radii agree within three f32 ulps; subtracting the Earth radius 6360 amplifies this. The near-horizon ray/ground intersection also subtracts large nearly equal terms. Substituting Dawn ray lengths, radii and heights makes the final half encodings exact for all ten. Optional `CPU_SKY_DISCRIMINANT=1` materializes multiplication terms and thereby changes Dawn's contraction: it is a separate diagnostic, not an unchanged-kernel comparison. Replays of `halve`, `projectRow` and `projectRows` pass the ordinary field policy from identical bytes.

`CPU_PREFILTER_TRACE=1` traces the ten worst reflection texels and their 640 cube samples. Substituting Dawn's samples makes every final half encoding exact. The largest sample discrepancy occurs at low mip levels, where a bilinear footprint crosses both cube-face axes. The ambiguous corner tap differs: for a one-texel cube with incident faces 3, 12 and 48, CPU filtering gives 18.75, Dawn 16.5, and three-face averaging would give 21. The [Vulkan cube-map edge handling rules](https://docs.vulkan.org/spec/latest/chapters/textures.html#textures-cubemapedge) permit other corner methods, constrained to preserve equal incident values; WebGPU's sampler description refers to Vulkan filtering. The dedicated `cube-corners` probe records all eight corners, cube arrays and sizes 1, 2 and 4, and the unit test checks the equal-incident invariant. An averaging experiment was disproved and removed; production sampling is unchanged.

A test-only corner policy selecting the X incident face at an ambiguous corner, combined with 8-bit interpolation, predicts the large sampled differences. All 640 Dawn samples fit tap-derived intervals allowing one 8-bit step in spatial/LOD weights and adjacent-half rounding at each filter stage. A fixed nearest-weight model leaves 11 samples 3–4 half ulps apart; flooring LOD reduces that to three samples at three half ulps, so the tests assert the bounded interval rather than claim exact knowledge of Metal's internal sequence. These are measured sampling differences from identical coordinates and texture bytes, not an unexplained arithmetic error or an added production tolerance.

### Follow-up scope

The initial whole-example run left these review failures. The investigations below supersede that initial status without changing production tolerances:

- Censer: advection and matter transport are source-traced; the follow-up replays all 27 distinct nonzero-step writer/shape cases. The other 25 pass independently, including curl, averaging, divergence, all pressure levels and projection. Their whole-scene differences inherit the bounded transport inputs and half stores.
- Ocean: the follow-up below isolates sky/cloud, environment, foam and wave-map writers separately; equalizing frequencies alone does not clear them.
- Sph-sculpt: the five environment/material resources are source-traced in the probes below. Allocation-order and render-dependent outputs remain separately excluded.

The detailed resource keys, differing-word counts and first values remain in `results.json`. Render-dependent resources, backend-specific scratch storage and resources with atomic allocation order are listed separately; their exclusion is not a claim of equality.

## Ocean follow-up: separate inputs, representation and sampling

Equalizing the 16 differing `omegas` words before the wave evolution dispatch does **not** clear the other 17 whole-scene resource entries. `ocean-inputs.test.gpu.ts` records the baseline and equal-frequency captures independently. Cloud, atmosphere, foam-tile and environment kernels do not depend on that frequency buffer.

`ocean-waves.test.gpu.ts` starts both branches with equal amplitudes and frequencies. The plain WebGPU branch stores the private foam history at `(cascade*N+x)*N+y`; the fallback branch stores it at `(cascade*N+y)*N+x`. Transposing the fallback buffer clears its differences under the ordinary field policy. This is a representation difference between Kit implementations, not a CPU arithmetic defect. Every one of the 35 selected identical-input dispatches passes: one evolution, three row stages, three column stages, one packing stage and 27 mip fills.

Injecting the final Dawn FFT lattice before the last fallback packing dispatch puts every base displacement and derivative component within one half-store ulp. Three foam channels remain two to four half ulps apart: permitted multiply-add contraction changes the Jacobian by one f32 ulp before subtracting the whitecap threshold. The recorded contraction candidates bracket all three Dawn halves within the one-store allowance. Injecting identical base half bytes makes all wave mip levels bit-exact. Without that control, mip sums of signed values cancel and amplify adjacent base-half differences.

The whole-scene foam history additionally runs 24 prewarm steps. Decoding the transpose leaves 826 differing words; equal spectra and frequencies reduce this to 114, on both the plain and fallback-shaped Dawn paths. `CPU_OCEAN_HISTORY=1 CPU_OCEAN_HISTORY_TRACE=1` records ten of these residual cells through all 26 steps, including the two frame updates. Every CPU update reconstructs bit-exactly from the recorded FFT signals and prior foam. Inverting the active Dawn foam-growth expression gives an effective Jacobian within 0.489 f32 ulp of the separately rounded arithmetic on the same recorded inputs. This is an inferred intermediate, not a claim to know Metal's instruction sequence. A one-ulp Jacobian interval plus the `exp` accuracy bound contains every recorded Dawn update. Incoming deformation signals differ by at most 9.69e-8 absolute (2.40e-6 relative to their vector magnitude) after the FFT. The whitecap subtraction multiplies these small differences by `67.5 * 0.125 = 8.4375` before carrying history forward. [WGSL permits reassociation and fusion](https://www.w3.org/TR/WGSL/#reassociation-and-fusion); no comparison tolerance was added to the runtime.

The independent foam tile is another store chain. Its base and each of ten mip dispatches pass from identical inputs. After injecting Dawn's base bytes, a test-only truncating-half-store model makes all eleven levels bit-exact (`CPU_OCEAN_FOAM=1`). The production CPU still rounds half stores to nearest even.

### Clouds and atmosphere

The additional rgba8unorm filtering ramp in `filter-precision.test.gpu.ts` has 4100 components. Every Dawn component exactly matches 8-bit interpolation coordinates followed by a conversion step of `1/(255*16)`. The probe records the model; the production sampler keeps its full-precision interpolation. This conversion quantization matters when cloud density subtracts a threshold from several sampled noise octaves.

`ocean-kernels.test.gpu.ts` captures the independent writer dispatches and traces the ten worst texels for density, lighting, each cloud-view target, and the environment base. It records each sample's coordinates, input identity, output and the filtering envelope. Replaying Dawn's samples puts these selected final outputs within one half-store ulp; the standalone sunlight buffer becomes bit-exact. These checks isolate arithmetic after sampling rather than granting a scene-wide tolerance.

Atmospheric transmittance has radii at most three f32 ulps apart before subtracting Earth radius 6360. Sky-view radii differ by at most one ulp, and near-zero sphere intersections change 26 selected ground-shadow decisions. For example, the origin near `(1.0997744, 6360, 1.1484978)` yields intersection 0 on the CPU and -8.239858e-7 on Dawn. The subtraction cancels terms around 56; the largest recorded result gap is 0.831 ulp of that dot-product term. The sign test changes whether the shader samples sunlight. Injecting the recorded intersection/radius values and sampled values puts all selected stores within one half ulp. `ocean-atmosphere.test.gpu.ts` asserts the coordinate-radius bound, sample envelope and controlled-store bound separately.

### Environment cube filtering

The ten worst base environment texels use 506 sky/cloud samples. Every sample lies within the measured spatial filtering and half-result bounds; substituting Dawn samples puts the final stores within one half ulp. Every identical-input `halve` and harmonic `projectRows` replay passes the ordinary field policy.

The four reflection levels retain 40 worst texels and 2062 executed samples. Mip precision was the missing term in the earlier model: the independent cube-LOD ramp matches `floor(f16(requestedLOD)*64)/64` in all 3075 components. Every ocean reflection sample fits an envelope allowing incident-face cube corners, one 8-bit spatial weight step, the measured 6-bit mip weights and adjacent-half results. Injecting Dawn samples by Hammersley iteration makes all 40 final half vectors bit-exact. A fixed single-corner/quantization model is not asserted exact for arbitrary texture data; the tap-derived envelope is the check.

[Vulkan mip level selection](https://docs.vulkan.org/spec/latest/chapters/textures.html#textures-level-selection) explicitly quantizes the interpolation fraction to implementation mip precision. Cube-corner behavior likewise permits differing incident-face policies. Neither model is installed in the production sampler.

These controls close the named ocean mechanisms, with limits on what was measured: the detailed intermediate traces cover the selected worst texels, while whole-resource stage replays, equal-input controls, and whole-scene comparisons retain their own counts. They do not establish bit-identical whole-scene simulation. The raw reports remain available rather than being overwritten by a generic pass flag.

### Reproducing the ocean controls

Run the relevant file with `npx vitest run --project 'typegpu-kit-gl (gpu)' tests/cpu/<file>` and these opt-in flags:

- `ocean-inputs.test.gpu.ts`: `CPU_OCEAN_INPUTS=1` compares whole captures with and without equalized frequencies.
- `ocean-waves.test.gpu.ts`: `CPU_OCEAN_WAVES=1 CPU_OCEAN_STAGES=1 CPU_OCEAN_STORE_CONTROLS=1` checks all 35 stage replays and the final-lattice/base-half controls.
- `ocean-foam.test.gpu.ts`: `CPU_OCEAN_FOAM=1` checks the tile's eleven half-store levels.
- `ocean-kernels.test.gpu.ts`: `CPU_OCEAN_KERNELS=1 CPU_OCEAN_TRACE_KEYS='^compute:|^bakeLighting:' CPU_OCEAN_GEOMETRY=1 CPU_OCEAN_PREFILTER=1` captures the independent writers and checks their worst-texel traces.
- `ocean-atmosphere.test.gpu.ts`: `CPU_OCEAN_ATMOSPHERE=1 CPU_ATMOSPHERE_ENTRY='bakeTransmittance|bakeSkyView|bakeLighting' CPU_ATMOSPHERE_GEOMETRY=1 CPU_SAMPLE_TRACE=1` runs the smaller atmosphere-only control.
- `cube-lod.test.gpu.ts` and `filter-precision.test.gpu.ts`: `CPU_THRESHOLDS=1` measures this Dawn device's mip and spatial filter precision. These are device diagnostics, not claims that every GPU has the same quantization.

### Replay harness correction

The old replay matcher could select the opposite direction of a ping-pong dispatch because both directions mention the same resources. Matching now includes each resource's read/write role, and binding reconstruction prefers the matching group and binding. `replay-matching.test.gpu.ts` is a normal Dawn regression with two opposed texture dispatches and an unused explicit-layout binding. Replays compile the captured WGSL with an inferred layout because snapshots contain statically used bindings, whereas the original explicit layout can contain additional unused entries. Invalid earlier cloud replay results are superseded by the role-matched runs.

## Execution coverage at the hardening checkpoint

All 33 examples passed compilation and the controls sweep. The table counts shader modules created; identical generated code may occur in more than one module, and dispatch tracking is by code. The remaining entries are listed explicitly rather than counted as executed. Pointer controls, all settings, images, UI buttons and held-button strokes were exercised. Camera attachment still needs a real external texture.

The costs below are two-frame CPU-only means from the interaction runs, at the driver's reduced scene capacities and 64×36 surface. They are not benchmark medians or full-stack frame times; concurrent work ran on the same machine, and fragment-eligible kernels also run on the CPU fake device. The paired 31-sample medians above remain the performance comparison. Per-pipeline invocation counts and times are in `results.json`.

| Example      | Compiled modules | Undispatched entries               | Calls/frame | Invocations/frame | CPU ms/frame |
| ------------ | ---------------: | ---------------------------------- | ----------: | ----------------: | -----------: |
| censer       |               17 | none                               |         112 |          41387008 |     48147.56 |
| falling-sand |               65 | `spawnEach`, `advanceKernel`       |         150 |           7840708 |      8303.32 |
| ocean        |               30 | `compute` ×2                       |          51 |          81805441 |      3155.19 |
| classroom    |                7 | none                               |           7 |           1010625 |      2496.97 |
| sph-sculpt   |               38 | `compute` ×2                       |          49 |           1664966 |       280.68 |
| stampede     |               36 | none                               |         118 |            790194 |       243.37 |
| jelly-world  |                4 | none                               |           2 |             32832 |       236.34 |
| jump-flood   |                4 | none                               |          12 |           2883648 |       149.70 |
| fishtank     |               30 | `spawnEach`                        |        97.5 |            311589 |       102.03 |
| slime-mould  |                5 | none                               |         7.5 |            145600 |       101.23 |
| benthos      |                6 | none                               |         4.5 |             84128 |        68.90 |
| flow-field   |                2 | none                               |         2.5 |            163840 |        52.55 |
| jellies      |               10 | none                               |         9.5 |              2160 |        13.66 |
| ember-mirror |                6 | `spawnEach`, `advanceKernel`       |           7 |             96384 |        13.22 |
| fireworks    |               31 | `spawnEach` ×4, `advanceKernel` ×4 |        30.5 |           88897.5 |         8.10 |
| night-market |               15 | none                               |          15 |             27522 |         5.21 |
| dominoes     |               13 | `spawnEach`, `advanceKernel`       |          17 |             27712 |         4.07 |
| skeletons    |               34 | none                               |        32.5 |              6821 |         3.65 |
| kinetic      |               11 | `compute`                          |           2 |              4097 |         3.33 |
| lattice      |                5 | none                               |         4.5 |              5440 |         1.96 |
| quiver-trees |               46 | none                               |          10 |                10 |         0.51 |
| platformer   |               19 | `spawnEach`                        |          27 |              5701 |         0.32 |
| forest       |                6 | none                               |           2 |                 2 |         0.30 |
| blockfall    |               13 | `spawnEach`, `advanceKernel`       |           9 |              1440 |         0.20 |
| solar-system |                5 | none                               |         3.5 |               224 |         0.09 |
| afterimage   |                0 | none                               |           0 |                 0 |         0.00 |
| cloister     |                0 | none                               |           0 |                 0 |         0.00 |
| hex-isle     |                1 | `move_1`                           |           0 |                 0 |         0.00 |
| spider       |                8 | none                               |           0 |                 0 |         0.00 |
| robot-arm    |               15 | none                               |           0 |                 0 |         0.00 |
| star-map     |                1 | none                               |           0 |                 0 |         0.00 |
| survey-model |                6 | none                               |           0 |                 0 |         0.00 |
| raymarch     |                0 | none                               |           0 |                 0 |         0.00 |

Undispatched particle `spawnEach`/`advanceKernel` entries belong to pools whose examples use their emitter or device-side spawning instead of queued host bursts; the module tests exercise those facilities. Hex-isle supplies a custom depth-of-field radius and never enables its optional `move_1` autofocus. The remaining environment `compute` entries are precompiled panorama/cube source variants for which the scene supplies no corresponding source. The sweep reports those reasons and module indexes.

## Generated kernels versus their JavaScript source

`CPU_SOURCE_BENCHMARK=1 npx vitest run --project 'typegpu-kit-gl (unit)' tests/cpu/source-performance.test.ts` runs the three shared bodies in `source-performance.ts`. The test launches one native Node process with `unplugin-typegpu/node`; every measured path runs in that process. Native imports matter: Vitest's module runner severely distorted the scalar TypeGPU baseline in an initial discarded measurement. There is no GPU, compilation is outside timing, and generated timings call the same `CpuModule.dispatch` used by the device, including binding/reset and invocation traversal.

Each body reads immutable, exactly representable inputs for 16,384 rows and writes separate outputs. Generated code reads packed buffers through typed-array views. Direct source calls use TypeGPU's own vector/matrix classes in JS arrays and put returned values in a JS output array; they do not incur buffer serialization. `tgpu.fn` calls use that same body with TypeGPU's normal argument/return copies and casts. `tgpu['~unstable'].simulate` runs the buffer-reading entry in a manual invocation loop within one simulation context per measured batch, including its initial buffer materialization. This version's simulation API models one thread; it has no dispatch simulator. The explicit `std.add`/`std.mul` particle body performs the same thirty vector operations as the existing particle benchmark while remaining directly callable JavaScript.

Apple M5 Pro, Node v26.13.1, 2026-11-25: eight warm-up batches per path, then fifteen samples per path, each over 20 ms. Sample order rotates between paths to distribute shared-machine and thermal variation. Throughputs below are **million invocations/second**, from the final interleaved run; all timings are recorded under `sourcePerformance` in `results.json`.

| Kernel                                            | Generated f32 | Direct source | `tgpu.fn` source | `simulate` | Generated / direct | Generated without intermediate rounding |
| ------------------------------------------------- | ------------: | ------------: | ---------------: | ---------: | -----------------: | --------------------------------------: |
| Particle step, 30 vector operations               |         10.05 |        0.0318 |           0.0298 |     0.0272 |               316× |                                   12.95 |
| Integer hash, eight multiply/add/xor/shift rounds |         62.71 |         31.15 |            10.27 |      1.891 |              2.01× |                                   62.09 |
| Matrix × vector, two normalizations, sin/cos, mix |         8.216 |        0.1031 |           0.0728 |     0.0562 |              79.7× |                                   12.46 |

The particle advantage comes from scalarizing its hot vector loop: TypeGPU's JS `std` path repeatedly classifies operands, constructs intermediate argument arrays and allocates vector objects. The integer kernel has none of that vector allocation, so its smaller gain comes from emitting wrapping integer operations directly instead of repeated schema calls/checks and JS floating-point multiplication. The transform kernel removes generic matrix/vector dispatch and most intermediate vector objects; both versions still pay for transcendental functions. These are comparisons against TypeGPU's current JS implementations, **not** against hand-optimized typed-array JavaScript and not a general JavaScript compiler speedup claim.

The rounding diagnostic changes only a test copy of the generated string, removing `Math.fround` calls. F32 literals and typed-array stores remain rounded; it is not a production option. Intermediate rounding adds 28.8% execution time for this particle fixture and 51.7% for this transform fixture. The integer fixture has zero rounding calls, so its 1% difference is timing variation. These small kernels do not replace the earlier whole-frame f32 cost measurements. There are 130 sampled rows per fixture checked before and after timing across all paths: integer results are exact; maximum float discrepancies are 9.54e-7 for particles and 1.19e-7 for transforms.

The compiler is useful beyond the fallback as a basis for GPU-free kernel tests, worker/server simulations and debugging supported kernels against GPU output. A standalone CPU root would still need packaging around the current generator/runtime and device adapter. It does not replace GPU parallel throughput or optimized scalar JS, provide robust out-of-bounds shader semantics, or run under a policy that forbids `new Function`/`unsafe-eval`.

## CPU semantics audit, 2026-11-25

The six regressions have independent fixes on `kit-gl/cpu-semantics`: float-to-integer saturation (scalar and vector, preserving integer reinterpretation); single evaluation of compound-store destinations, including tracked buffer writes; signed integer builtin results before comparisons or float conversion; scalar coordinates and dimensions for 1D textures; scalar zero for a missing depth mip; and lexical lifetime for local aliases. Each failing regression was run before its fix. `scopes.test.gpu.ts` also compares the sibling-arm, loop-body, nested-shadowing, switch-case and vector/scalar name-reuse fixture against Dawn. Before the scope fix, the copied-record else paths returned NaN: emitted code treated the copied object as the sibling arm's buffer address.

Local address/layout metadata, scalarized vector components and owned-value metadata now live in `#addresses`, `#vectors` and `#ownedLocals` scoped snapshots. References are lowered before their declaring scope ends. The other maps hold global function/resource/schema keys, globally unique generated-expression identifiers or proven uniform expressions, rather than reusable lexical identifiers. No barrier algorithm or general composite-copy rewrite was attempted in this audit; their existing comparisons pass, but those paths have not been exhaustively proven.

`integer-builtins.test.ts` confirms that `firstLeadingBit`, `extractBits` and `insertBits` fail during generation with `unsupported builtin <name>`. This audit does not add those builtins.

The conversion performance check uses the shared particle body in `source-performance.ts` and an additional `d.u32(coordinate * 0.125)` grid-index body. Both generators run against the same current runtime and independent identical buffers. The baseline is `49d175b2`; the updated generator is `dd8facbf`. All old/new outputs agree exactly on these inputs. Native Node v26.13.1 on Apple M5 Pro, eight warm-up batches followed by fifteen rotating, interleaved samples per path, each over 20 ms; raw samples are under `conversionPerformancePaired` in `results.json`.

| Kernel                         | Before, M invocations/s | After, M invocations/s | Execution-time change |
| ------------------------------ | ----------------------: | ---------------------: | --------------------: |
| Particle, 30 vector operations |                  10.791 |                 10.701 |                +0.85% |
| Float to unsigned grid index   |                  62.966 |                 58.516 |                +7.60% |

The conversion-specific difference is about 1.2 ns per invocation. The particle fixture is a control without floating-to-integer casts. An earlier sequential A/B run was discarded because its conversion timing depended strongly on run order. These figures use the interleaved old/new comparison, not that run, and do not imply performance relative to the earlier source-versus-generated table measured under different machine load.

To reproduce, temporarily copy `git show 49d175b2:packages/typegpu-kit-gl/src/cpu/generator.ts` next to the current generator, import its class after installing `unplugin-typegpu/node`, and invoke `runSourcePerformance({ previousGenerator: BaselineCpuGenerator, kernels: ['particles', 'floatIndices'], section: 'conversionPerformancePaired' })` in native Node. Remove that temporary copy afterward; neither production source nor generated source needs to be edited for the comparison. The normal `CPU_SOURCE_BENCHMARK=1` test retains its original three-kernel default.

Verification after all six fixes: repository types and style pass; 1,959 unit tests and 199 WebGL tests pass; the enabled CPU/Dawn suite passes 29 tests. The compile sweep accepts all 491 compute modules from all 33 examples.
