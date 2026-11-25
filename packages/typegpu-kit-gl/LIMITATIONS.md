# What the WebGL fallback cannot do

`@typegpu/kit-gl` runs a TypeGPU Kit program on WebGL 2, which has no compute stage, no storage buffers and fewer texture kinds than WebGPU. This file lists what differs from WebGPU. A feature that is missing throws an error that names it; nothing here fails silently unless the entry says so.

`ARCHITECTURE.md` explains the mechanisms these limits come from.

## Compute

A compute pipeline runs one of three ways. The generator picks per pipeline, and the device falls back to the CPU per dispatch when a GPU form's conditions do not hold.

**As fragment passes on the GPU**, when every write is provably to the invocation's own cell: its own array element or texel, a linear index with a constant width, an index plus an offset every invocation shares, a constant stride, a column of a 3d or array texture, a binding only one chosen invocation writes, or atomics on its own cell.

**As points on the GPU**, when a kernel writes cells it computes: each write writes a whole element or texel with `=` or `textureStore`, in the entry or in a function it calls, but never inside a runtime loop or in a function called inside one (`tgpu.unroll` loops are fine), and the kernel never reads what it writes this way.

**As sums on the GPU**, for a buffer the kernel only adds to with `atomicAdd` or `atomicSub` whose results it drops, as a counter's `add` does: the kernel reads and writes that buffer no other way, and makes no addition inside a runtime loop, its own or in a function it calls. Additions to the same cell sum exactly, wrapping at 2^32 as atomics do, up to 65 536 records a cell in a dispatch: a dispatch whose invocations times additions could pass that runs on the CPU. The device needs `EXT_color_buffer_float` and `EXT_float_blend` for it, and runs such kernels on the CPU without them.

**Takes of a counter**, as a particle system's `emit` makes, also run on the GPU: `atomicAdd` of 1 to one whole `u32` counter whose result the kernel uses, as long as no value it takes, nor a variable made from one, decides a branch or a loop, is returned, or is passed to a function of the kernel. Invocations take values in the order of their index, and an invocation's own takes in the order it makes them: one of the orders WebGPU's atomics may give, so which ring slot an emitter gets within a dispatch can differ from a WebGPU run, where it differs from run to run too. A particle system with `born`, which receives the slot, keeps its emitters on the CPU. A dispatch of more than 65 536 invocations, or with more than 15 takes of one counter an invocation, runs on the CPU.

All of these run at GPU speed, and differ from WebGPU in these ways:

- A read of a cell another invocation writes in the same dispatch returns the value from before the dispatch. WebGPU leaves such a read a data race, so a kernel that is correct there does not depend on it.
- Where two invocations write one cell, either value may win, as in WebGPU.
- A point outside its target, or outside a buffer's bound range, is dropped. WebGPU may drop such a write or move it inside the binding.
- A fragment pass runs the whole kernel once per written resource, and once more for each texel a buffer element spans beyond the first; it writes four layers of a 3d or array target a draw. A pass of points runs it once per invocation, captured with transform feedback, then draws the captured writes once per write site, and for a buffer once per word of the element and per channel of a data texel unless the element fits a texel. A kernel that writes many resources or large structs costs a multiple of its WebGPU cost.
- A linear index with a constant width needs a dispatch no wider than that width, or an early return past it at the top of the kernel, such as `if (gid.x >= width) return;` with a constant `width`. A wider dispatch runs on the CPU.

**On the CPU**, as generated JavaScript, for everything else: shared atomics, `workgroupBarrier` and workgroup variables, writes to computed cells inside loops, and writes to computed cells the kernel also reads. A kernel that uses the value an atomic returns stays here, except for takes of a counter, below. A small GPU form whose outputs the CPU reads back may also move here, when its inputs are on the CPU and the move saves waiting for the GPU.

- Workgroups run in order, with x varying fastest. Within a workgroup, invocations run in order until a barrier; all lanes reach it before any continues. Atomics follow that sequential order.
- A simple kernel costs about 0.04 microseconds per invocation, one with barriers or atomics 0.2 to 2. Tens of thousands of rows a frame are fine. A texture-sized kernel that stays on the CPU is not.
- A CPU kernel that reads a texture or buffer a draw or a GPU-run kernel wrote waits for the GPU and reads it back, every dispatch. A kernel that meters or sums a render target pays this each frame.
- Buffer accesses must be in bounds: the CPU path does not implement WebGPU's robust out-of-bounds access rules. Passing a pointer to a runtime-sized array is unsupported.
- Kernels are compiled with `new Function`, so a page whose content security policy forbids `unsafe-eval` cannot run them.

CPU `f32` literals, conversions and arithmetic round to 32 bits at each operation, including vector and matrix components. Integers preserve 32-bit wraparound, and half-float stores round to nearest, ties to even. A GPU may instead truncate a store to a 16-bit float texture, so a field fed back into itself can drift by a half-float ulp per step relative to that GPU's WebGPU result. GPU multiply-add contraction and builtin accuracy can also differ, and simulations can amplify those differences. For repeatable random streams, seed from integer ids and explicit seeds rather than computed positions. Avoid subtracting nearly equal large floats, and guard builtin inputs outside their specified domain: the CPU's `atan2(0, 0)` returns zero, while WebGPU need not return the same result.

`backends.of(root).compute` is `'cpu'` on the fallback. A module reads it to pick kernels the fallback runs on the GPU or that suit a CPU, as the Kit's culling, `exposure`, `environment`, `grid`, `particles` and `poses` do, and an example's `meta.json` may give `fallback` scene options that size it. On the fallback, particles from one burst fill the ring in burst order, where WebGPU's atomics fill it in any order.

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
- Different blend states per colour target need `OES_draw_buffers_indexed`.
- Depth bias has no clamp.
- Reading back a depth or signed-normalised texture needs `EXT_color_buffer_float`.
- A device created on a canvas presents to that canvas without a copy and to no other. A device created without one presents to any number, each through a bitmap transfer.
- `firstInstance` and `baseVertex` are emulated with a uniform and offset attribute pointers.
- WebGL 2 has no indirect draws, so `drawIndirect` and `drawIndexedIndirect` read their arguments on the CPU, and a draw record a GPU-run kernel wrote waits for the GPU each frame. A kernel that writes draw records is cheapest on the CPU here, so the Kit's culling walks its rows there. Drawing a known upper bound of instances and letting the vertex stage collapse those past a count it reads from the buffer would keep the records on the GPU; it is not implemented.

## Browsers

- Checked in Chromium with WebGPU removed from the page and in WebKit, which has no WebGPU. WebKit draws at the display's 60 Hz.
- Not checked in Firefox: the test browser did not start under automation here.
