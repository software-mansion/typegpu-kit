# What the WebGL fallback cannot do

`@typegpu/kit-gl` runs a TypeGPU Kit program on WebGL 2, which has no compute stage, no storage buffers and fewer texture kinds than WebGPU. This file lists what differs from WebGPU. A feature that is missing throws an error that names it; nothing here fails silently unless the entry says so.

`ARCHITECTURE.md` explains the mechanisms these limits come from.

## Compute

A compute pipeline runs one of three ways. The generator picks per pipeline, and the device falls back to the CPU per dispatch when a GPU form's conditions do not hold.

**As fragment passes on the GPU**, when every write is provably to the invocation's own cell: its own array element or texel, a linear index with a constant width, an index plus an offset every invocation shares, a constant stride, a column of a 3d or array texture, a binding only one chosen invocation writes, or atomics on its own cell.

**As blocks on the GPU**, when each invocation owns a constant block of up to 8 cells of a grid, as a 2 × 2 Margolus step or a pair of cells does: it writes them through a helper at `row * width + col` of a vector parameter, which returns early unless a function of the parameter proves `0 <= col < width`, and each call names its cell as a step per axis times the invocation plus a constant offset and a phase every invocation shares, the offsets of an axis within one step. A read of a cell of another invocation's block returns the contents from before the dispatch, the same rule as for single cells.

**As points on the GPU**, when a kernel writes cells it computes: each write writes a whole element or texel with `=` or `textureStore`, in the entry or in a function it calls, but never inside a runtime loop or in a function called inside one (`tgpu.unroll` loops are fine), and the kernel never reads what it writes this way. Where one resource's writes need more than 255 bytes an invocation, cell and value, the stride WebGL reads attributes at, the kernel runs on the CPU.

**As sums on the GPU**, for a buffer the kernel only adds to with `atomicAdd` or `atomicSub` whose results it drops, as a counter's `add` does: the kernel reads and writes that buffer no other way, in any function it calls, except with `atomicLoad` where no addition to the buffer can come before it on any path, and makes no addition inside a runtime loop, its own or in a function it calls. Such a load reads the buffer as it was before the dispatch, as if every invocation's loads ran before any invocation's additions, which WebGPU allows; a load that may follow an addition, even through a helper or an `if` arm that falls through, runs the kernel on the CPU. Additions are counted along the path that makes the most. An invocation's additions take 12 bytes each of the 255 a record holds, but those to a buffer of a fixed size share one per word, so a kernel that adds more than 21 times to a runtime-sized array runs on the CPU. Additions to the same cell sum exactly, wrapping at 2^32 as atomics do, however many a dispatch makes: a dispatch whose invocations times additions could pass 1 118 481 is drawn in batches of that many, each added to the last, at the cost of one more combining draw a batch. The device needs `EXT_color_buffer_float` and `EXT_float_blend` for it, and runs such kernels on the CPU without them.

**Takes of a counter**, as a particle system's `emit` makes, also run on the GPU: `atomicAdd` of 1 to one whole `u32` counter whose result the kernel uses, as long as the binding is exactly one `atomic<u32>`, the kernel never loads it, and no value it takes, nor a variable stored from one in any way, decides a branch or a loop, is returned, or is passed to a function of the kernel, which must not write its arguments either. Invocations take values in the order of their index, and an invocation's own takes in the order it makes them: one of the orders WebGPU's atomics may give, so which ring slot an emitter gets within a dispatch can differ from a WebGPU run, where it differs from run to run too. A particle system with `born`, which receives the slot, keeps its emitters on the CPU. A dispatch of more than 65 536 invocations, or with more than 15 takes of one counter an invocation, runs on the CPU.

All of these run at GPU speed, and differ from WebGPU in these ways:

- A read of a cell another invocation writes in the same dispatch returns the value from before the dispatch. WebGPU leaves such a read a data race, so a kernel that is correct there does not depend on it.
- Where two invocations write one cell, either value may win, as in WebGPU.
- A point outside its target, or outside a buffer's bound range, is dropped. WebGPU may drop such a write or move it inside the binding.
- A fragment pass runs the whole kernel once per written resource, and once more for each texel a buffer element spans beyond the first; it writes four layers of a 3d or array target a draw. A pass of points runs it once per invocation, captured with transform feedback, then draws the captured writes once per write site, and for a buffer once per word of the element and per channel of a data texel unless the element fits a texel. A kernel that writes many resources or large structs costs a multiple of its WebGPU cost.
- A linear index with a constant width needs a dispatch no wider than that width, or an early return past it at the top of the kernel, such as `if (gid.x >= width) return;` with a constant `width`. A wider dispatch runs on the CPU.

**On the CPU**, as generated JavaScript, for everything else: shared atomics, `workgroupBarrier` and workgroup variables, writes to computed cells inside loops, and writes to computed cells the kernel also reads. A kernel that uses the value an atomic returns stays here, except for takes of a counter, below. A small GPU form whose outputs the CPU reads back may also move here, when its inputs are on the CPU and the move saves waiting for the GPU.

- Workgroups run in order, with x varying fastest. Within a workgroup, invocations run in order until a barrier; all lanes reach it before any continues. Atomics follow that sequential order.
- A simple kernel costs about 0.04 microseconds per invocation, one with barriers or atomics 0.2 to 2. Tens of thousands of rows a frame are fine. A texture-sized kernel that stays on the CPU is not.
- A CPU kernel that reads a texture or buffer a draw or a GPU-run kernel wrote waits for the GPU and reads it back, every dispatch. A kernel that meters or sums a render target pays this each frame. The wait includes every other page's GPU work in the same browser, so two such pages drawing at once slow each other far more than they share the GPU: `falling-sand` and `stampede`, at 120 and 107 frames a second alone in Chromium here, ran at 28 and 42 side by side.
- Buffer accesses must be in bounds: the CPU path does not implement WebGPU's robust out-of-bounds access rules. Passing a pointer to a runtime-sized array is unsupported.
- Kernels are compiled with `new Function`, so a page whose content security policy forbids `unsafe-eval` cannot run them.

CPU `f32` literals, conversions and arithmetic round to 32 bits at each operation, including vector and matrix components. Integers preserve 32-bit wraparound, and half-float stores round to nearest, ties to even. A GPU may instead truncate a store to a 16-bit float texture, so a field fed back into itself can drift by a half-float ulp per step relative to that GPU's WebGPU result. GPU reassociation, multiply-add contraction and builtin accuracy can also differ, and simulations can amplify those differences. Filtered samples differ too, because a GPU weights its taps and mip levels with a few bits of fraction where a CPU kernel uses the exact one; use `textureLoad` where a simulation needs exact texel values. For repeatable random streams, seed from integer ids and explicit seeds rather than computed positions. Avoid subtracting nearly equal large floats, and handle zero axes explicitly when using `atan2`: its accuracy there is unbounded, and the CPU's `atan2(0, 0)` returns zero while WebGPU need not return the same result.

`backends.of(root).compute` is `'cpu'` on the fallback. A module reads it to pick kernels the fallback runs on the GPU or that suit a CPU, as the Kit's culling, `exposure`, `environment`, `grid`, `liveLists`, `particles` and `poses` do, and an example's `meta.json` may give `fallback` scene options that size it. On the fallback, particles from one burst fill the ring in burst order, where WebGPU's atomics fill it in any order.

## What still needs a branch

Most examples run on the fallback as they are. Four choose other kernels with `backends.of(root)`, because the kernels they run on WebGPU would stay on the CPU here and be too slow there. Each waits on a form the fallback does not have:

- `ocean`, `censer` and `sph-sculpt` share workgroup memory behind barriers: the FFT and its mip reductions, the smoke's coarse average, the water's tally and occupancy. Such a kernel can be split at its barriers into passes, with the workgroup's memory spilled to a scratch texture between them, but for these three that costs more passes than their branches: `ocean`'s FFT would take 26 passes, ten of them scatters of 1.2M points, against its branch's 11, and its mips 45 against 27. Their branches are the faster design and stay. In the Kit, `poses` is the kernel the split would speed up, since it now recomputes each joint's ancestors in every thread that needs them.
- `jump-flood` places seeds with `atomicMin` used as a minimum and links regions with `atomicOr` into rows of bits. No form covers either.

In the Kit, the culling, `exposure`, `environment`, `grid`, `liveLists`, `particles` and `poses` branch for the same reasons: workgroup memory, counters per cell whose results are used, and a ring slot taken inside a loop or passed on to a callback.

## Shaders

- A function written as a WGSL string cannot be translated. `common.fullScreenTriangle` and TypeGPU's own mipmap and resampling shaders are recognised and replaced; any other throws.
- No `textureGather` or `textureGatherCompare`, no `textureNumSamples`, no multisampled or cube-array texture bindings.
- No `sample_index`, `primitive_index`, `clip_distances` or sample-mask input. A `sample_mask` output discards the fragment when bit 0 is clear, which is exact only without multisampling.
- No `extractBits` or `insertBits` in generated GLSL (CPU kernels support both), no `f16`, no `bool` inside a buffer, and no array of arrays other than a buffer's runtime-sized array of them.
- A render stage cannot write a buffer or a storage texture, and cannot use atomics other than `atomicLoad`.
- A uniform buffer over 16 KiB is read through a texture instead of a uniform block. Reading a whole runtime-sized array as one value throws.
- A stage may use 16 texture units, and a program 32. Each storage buffer, each large uniform buffer and each texture and sampler pair takes one. A program that needs more fails when it links, with a message that counts them.

## Device

- No timestamp or occlusion queries, and no immediate data.
- No cube-array textures. A multisampled texture can be rendered and resolved, not sampled.
- A 6-layer texture is stored as a cube or as an array, whichever its first view asks for. A view of the other kind, or of part of a texture's layers, samples a copy the device refreshes after each change.
- Two views of one texture with different mip ranges cannot be bound in the same draw.
- Without `OES_draw_buffers_indexed`, a pipeline whose colour targets blend or mask differently draws once for each group of targets that agree. Each later draw writes neither depth nor stencil, and one that wrote depth keeps only the fragments that won it, so a later group that blends sees one fragment a pixel where WebGPU blends every fragment that passed. A pipeline that also writes stencil is an error that names the extension.
- Depth bias has no clamp.
- A device created on a canvas presents to that canvas without a copy and to no other. A device created without one presents to any number, each through a bitmap transfer.
- `firstInstance` and `baseVertex` are emulated with a uniform and offset attribute pointers.
- WebGL 2 has no indirect draws, so `drawIndirect` and `drawIndexedIndirect` read their arguments on the CPU, and a draw record a GPU-run kernel wrote waits for the GPU each frame. A kernel that writes draw records is cheapest on the CPU here. So the Kit's culling and live lists test each row on the GPU, where the rows are, and list the rows that passed on the CPU, which then holds the counts. The CPU still waits once a frame for the GPU to hand over those marks, and that wait is as long as the frame's drawing: a scene that culls or lists every frame runs its CPU and GPU work one after the other. Drawing a known upper bound of instances and letting the vertex stage collapse those past a count it reads from the buffer would keep the records on the GPU and remove the wait; it is not implemented.

## Extensions and limits

The device asks for the extensions below and keeps to the limits the context reports, down to WebGL 2's minimums: 2048 for textures, 256 for 3d sizes and layers, 4 draw buffers and samples (with 4, a pass of sums draws its records twice, to the same bound), 16 texture units a stage and 32 a program, 24 uniform bindings of 16 KiB each, 64 interleaved transform feedback components. `tests/device/minimum.ts` emulates such a context, and `scripts/examples.ts --minimum --hide all` runs the examples on it. It reports the minimums to the device, but the driver behind it keeps its own limits, so a program over a limit the device leaves to the driver still links there. At the minimums with every extension present, every example runs, with 2048 texels for textures as with 4096.

- `EXT_color_buffer_half_float` or `EXT_color_buffer_float`: rendering to 16-bit float targets, which every Kit pass with an HDR target does. Without both, creating such a texture for rendering is an error that names them, and almost every example stops there.
- `EXT_color_buffer_float`: rendering to 32-bit float and `rg11b10ufloat` targets, GPU forms that write float textures and blending or additive sums into them. Without it, those kernels run on the CPU, as the device's `fallbacks` say, and the render targets are errors that name it. Reading float, depth and signed-normalised textures back needs no extension.
- `EXT_float_blend`: blending into 32-bit float targets, and the additive GPU forms. Without it, a pipeline that blends into such a target is an error that names it, and additive kernels run on the CPU.
- `OES_texture_float_linear`: filtering 32-bit float textures. Without it, the device lacks the `float32-filterable` feature, as a WebGPU adapter without it does, and a filtering sampler that reads such a texture is an error that names it; the draw then samples with nearest filtering.
- `OES_draw_buffers_indexed`: different blend states or write masks per colour target in one draw. Without it, the draw repeats once per group of targets that agree, at the cost of running the fragment stage again, as the Device section says.
- `EXT_clip_control`: depth from 0 to 1 natively. Without it, the vertex stage maps depth into GL's range, at a small cost in depth precision.
- `KHR_parallel_shader_compile`: linking without blocking. Without it, programs link when first used and block then.
- `EXT_texture_filter_anisotropic`: anisotropic filtering, ignored without it. `EXT_disjoint_timer_query_webgl2`, `WEBGL_multi_draw` and `WEBGL_debug_renderer_info` change nothing the Kit needs.
- A scatter whose writes need more transform feedback components than the context records, or whose capture fails to link, runs on the CPU, with the reason in `fallbacks`. GPU forms draw into at most 4 buffers at once, WebGL 2's minimum. Texture units are counted against the program's 32; a stage that needs more than its 16 fails when its program links, with the driver's message.

## Browsers

- Checked in Chromium with WebGPU removed from the page and in Playwright's WebKit, which has no WebGPU. WebKit draws at the display's 60 Hz.
- Also checked in Chromium on two other GL backends, with `scripts/examples.ts --flag=--use-angle=gl` or `--flag=--use-angle=swiftshader`. On ANGLE over Apple's OpenGL 4.1 driver, which lacks `EXT_clip_control` and `KHR_parallel_shader_compile`, every example runs at the usual frame rates. On SwiftShader, the software Vulkan Chromium uses where it blocks the GPU, every example draws at 0.1 to 19 frames a second, `falling-sand` takes two minutes to get ready, and `ocean` crashes Chromium's GPU process during its first frame, at any canvas size, which the device reports as a lost device. No flush or synchronous call after each GL call avoids it; `tests/device/NOTES.md` lists what was ruled out.
- Not checked in Safari itself: `safaridriver` needs Safari's "Allow remote automation" setting, and without it a session times out connecting to Safari.
- Not checked in Firefox, whose WebGL on macOS does not go through the shader translator Chromium and WebKit share, so it is the likeliest browser to reject a shader the others accept. On macOS 27, Playwright's Firefox build 1532 never finishes starting, headless or headed, under Playwright or launched alone, with Firefox's sandboxes off, without its GPU process and without content processes: after `RenderCompositorSWGL failed mapping default framebuffer` its main thread spins in the event loop. Build 1555 exits at once with `Could not find profile folder`. A release Firefox was not available to run.
