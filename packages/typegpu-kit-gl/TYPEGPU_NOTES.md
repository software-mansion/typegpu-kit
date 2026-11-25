# Notes for TypeGPU

What building the WebGL fallback showed could be better in TypeGPU itself. `PATCHES.md` lists the changes this repository already applies, each with its diff. This file lists what is not a ready patch: limits that needed a workaround here, with the reason, a small reproduction and a way to fix each upstream. An entry that a patch already covers says so and stays short.

Each entry names where the workaround lives, so it can be deleted once TypeGPU changes.

## Generators

### 1. A root's generator class must claim to generate WGSL

`tgpu.init` and `tgpu.initFromDevice` type `unstable_shaderGeneratorClass` as `WgslGeneratorClass`, a class whose instances have `languageKey: 'wgsl'`. A root over another device accepts another language, which is the point of passing a generator.

```ts
class Glsl extends WgslGenerator {
  override languageKey = 'glsl';
}
// Type 'typeof Glsl' is not assignable to type 'ShaderGeneratorClass<ShaderGenerator & { languageKey: "wgsl" }>'
tgpu.initFromDevice({ device, unstable_shaderGeneratorClass: Glsl });
```

**Upstream.** Type the option as `ShaderGeneratorClass`, or make the root generic over the language its device accepts. In `packages/typegpu/src/core/root/init.ts`.

**Workaround.** `KitGenerator` in `src/generator.ts` declares `languageKey = 'wgsl' as const`.

### 2. A generator is not told what it is generating

TypeGPU builds one generator per pipeline with `new root.shaderGeneratorClass()`: no arguments, no pipeline kind, no stage. A generator that emits GLSL for render pipelines and something else for compute pipelines has to find out the kind by itself.

- The first call it receives is not always `functionDefinition` for the entry: `declareGlobalVar` or `emitTypeAnnotation` can come first. `KitGenerator` reads `ctx.topFunctionScope?.functionType`. This is exposed by the internal `ResolutionCtx` type, but couples backend selection to the resolver's current stack.
- A compute entry written as a WGSL template never calls `functionDefinition` at all (see entry 6).

```ts
class Either implements ShaderGenerator {
  initGenerator(ctx: ResolutionCtx) {
    // Render or compute? Nothing here says.
  }
}
```

**Upstream.** Pass what is being resolved to the generator, for example `initGenerator(ctx, { kind: 'render' | 'compute', item })`, or let a root take a generator per pipeline kind. In `core/pipeline/renderPipeline.ts` and `computePipeline.ts`, where `resolve(this, { shaderGenerator: new root.shaderGeneratorClass() })` is called.

**Workaround.** `KitGenerator.#use` in `src/generator.ts`.

### 3. A render pipeline is one module, so one generator run must emit both stages

`RenderPipelineCore.resolveAndCreateShaderModule` resolves once and gives the same `GPUShaderModule` to `vertex` and `fragment`. GLSL needs a source per stage, and a vertex source that merely contains an unused function calling `dFdx` or `discard` does not compile. Upstream's `GlslGenerator` takes the stage as a constructor argument and needs two resolutions (`dualGlOptions()`), which a root cannot do.

```ts
const pipeline = root.createRenderPipeline({ vertex, fragment, targets: { format } });
root.unwrap(pipeline); // one createShaderModule call, one string, used for both stages
```

**Upstream.** Either let a generator ask for one module per stage, or have `GlslGenerator` emit both stages from one resolution with the chunks marked, as `KitGlslGenerator` does. The second needs no core change.

**Workaround.** `src/glsl/generator.ts` tags each declaration as shared, vertex or fragment, and `parse` in `src/glsl/renderModule.ts` keeps per stage what its `main` reaches.

### 4. The only channel from generator to device is the shader text

A device that is not WebGPU needs more than code: which GLSL sampler pairs which texture and sampler, which uniform reads which binding, the workgroup size, which bindings a kernel writes. TypeGPU passes `code` to `device.createShaderModule` and nothing else, so the fallback encodes all of it as `//#` comment lines and parses them back. It also relies on TypeGPU replacing bind group placeholders in the whole string, comments included.

```ts
// All a device receives for a pipeline's shaders:
device.createShaderModule({ code }); // so `code` carries `//#buffer 0 1 …` lines
```

**Upstream.** Let a generator return data beside the code, and pass it on, for example as an extra field of the `GPUShaderModuleDescriptor` that a custom device can read. In `resolutionCtx.ts` `resolve` and the two pipeline cores.

**Workaround.** The directive formats in `src/glsl/renderModule.ts`, `src/glsl/computeModule.ts` and `src/cpu/index.ts`.

### 5. A generator cannot resolve the same pipeline a second way

The fallback wants two forms of a compute pipeline: JavaScript for the CPU, and GLSL fragment passes where the kernel allows. TypeGPU resolves once. The generator captures the pipeline by wrapping `ctx.resolve` on the context instance, then calls `tgpu.resolveWithContext([item], { unstable_shaderGenerator })` for the second form.

```ts
initGenerator(ctx: ResolutionCtx) {
  const resolve = ctx.resolve.bind(ctx);
  let first = true;
  (ctx as { resolve: unknown }).resolve = (item: unknown, schema?: unknown) => {
    if (first) (first = false), (this.item = item); // the pipeline core
    return resolve(item as never, schema as never);
  };
}
```

The second resolution gives the same `@group` and `@binding` numbers only because both walk the functions in the same order: the catch-all group numbers fixed resources in order of first use. The device cross-checks the two binding tables and drops the second form on a mismatch.

**Upstream.** Give the generator the item (entry 2), and make catch-all binding numbers independent of visiting order, or expose them so a second resolution can reuse them.

**Workaround.** `KitGenerator` in `src/generator.ts`; the cross-check in `src/device/shader.ts`.

### 6. Code written as WGSL text is invisible to other generators

A function implemented with a WGSL template goes into the output as written. Three places in TypeGPU itself do this:

- `createGuardedComputePipeline`'s entry. Patched: `PATCHES.md`, core patch 6. The source comment says why it is WGSL: TypeGPU does not run `unplugin-typegpu` on its own package. Running the plugin on TypeGPU's source at build time, or attaching the metadata by hand as the patch does, removes the whole class.
- `common.fullScreenTriangle`. Not patched: `src/glsl/generator.ts` recognises it and substitutes GLSL.
- The mipmap and resampling shaders in `core/texture/textureUtils.ts`, which `texture.generateMipmaps()` and a resizing `texture.write()` compile through `device.createShaderModule` directly, bypassing the root's generator. Not patched: the device recognises the fullscreen vertex shader and the sampling and gathering fragment shaders by fragments of their WGSL text and substitutes GLSL programs of its own, so any edit to those strings upstream silently breaks mipmaps on the fallback.

```ts
const root = tgpu.initFromDevice({ device, unstable_shaderGeneratorClass: Glsl });
root.createRenderPipeline({ vertex: common.fullScreenTriangle, fragment, targets });
// The module the device receives still contains `@vertex fn fullScreenTriangle(...)`
```

**Upstream.** Write the three as `'use gpu'` functions. For the mipmap shaders, build them with the root's generator like any other pipeline.

**Workaround.** As listed per item; the shader substitution is `builtin` in `src/device/shader.ts`.

### 7. `WgslGenerator` has few seams for another target

Retargeting by subclassing works well where a hook exists (`emitCall`, `emitBinaryOp`, `typeInstantiation`, `numericLiteral`, `_emitVarDecl`, `functionDefinition`, `declareGlobalVar`). Where none exists a subclass overrides `_expression` or `_statement` and inspects AST nodes itself. Cases the fallback hit: unary operators, member and index access on buffer memory, assignment to a place that is not a variable, post-increment, `return` in an entry function, and builtins that were emitted as text rather than through `emitCall` (patched: `PATCHES.md`, core patches 3 to 5).

```ts
class Target extends WgslGenerator {
  // No hook for `-x`, `a[i].b = v` or `i++`: override `_expression` and `_statement` and match
  // tinyest nodes by hand
}
```

**Upstream.** Split `_expressionImpl` and `_statement` into one overridable method per node kind, and route every operator and access through a typed hook the way binary operators are.

**Workaround.** The overrides in `src/cpu/generator.ts` and `packages/typegpu-gl/src/glslGenerator.ts`.

### 8. Shader-function calls bypass `emitCall`

User functions use `stitch` to assemble `name(args)` directly. Overriding `emitCall` therefore changes builtin calls that use that hook, but does not change calls to `tgpu.fn` or shellless shader functions. A backend that passes vector components as separate numeric parameters must change both the definition and every call.

```ts
const twice = tgpu.fn([d.vec2f], d.vec2f)((v) => v * 2);
const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
  'use gpu';
  output.$ = twice(d.vec2f(1, 2));
});
// A backend defining twice(x, y) still receives a generated twice(vec2f(1, 2)) call.
```

**Upstream.** Route the `codegenImpl` in `core/function/tgpuFn.ts` and shellless calls through `ctx.gen.emitCall`, after argument conversion and function resolution. Keep the converted argument snippets typed until that hook.

**Workaround.** `CpuGenerator.call` resolves the call result and `callArguments` in `src/cpu/references.ts` splits its text before re-emitting it. Remove this parsing once all function calls use the hook.

### 9. Atomic references arrive as pointer-shaped text

The generic call hook has a typed snippet, but its lvalue has already become WGSL pointer syntax. A backend using indexed typed arrays needs the target array and index separately to implement an atomic operation without reading or writing adjacent fields.

```ts
const row = rows.$[id.x];
std.atomicAdd(row.count, 1);
// The backend must recover the address from an expression such as (&row.count).
```

**Upstream.** Expose a typed lvalue representation and an atomic-operation hook in `tgsl/shaderGenerator.ts`. Keep member/index accesses structured through the pointer conversion in `tgsl/wgslGenerator.ts` and `data/ref.ts`.

**Workaround.** `addressParts` and `lowerReferences` in `src/cpu/references.ts`, plus atomic handling in `CpuGenerator.call`, parse and lower pointer text.

### 10. Barrier suspension and its call graph are not generator metadata

A CPU backend must suspend every lane at a barrier, including barriers reached through helpers. There is no function property or barrier hook reporting that requirement, so the fallback recognises emitted barrier calls and propagates generator-function status through helper calls. A loose substring check even misclassified a helper named `yields` as a suspension.

```ts
const sync = () => {
  'use gpu';
  std.workgroupBarrier();
};
const kernel = tgpu.computeFn({ workgroupSize: [64] })(() => {
  'use gpu';
  sync();
});
// The entry must suspend although its body contains no direct barrier call.
```

**Upstream.** Add typed barrier emission and function effects, including reachable barriers, to `tgsl/shaderGenerator.ts`; preserve calls as structured nodes until function emission in `tgsl/wgslGenerator.ts`.

**Workaround.** `CpuGenerator.functionDefinition` lowers barrier text to `yield`, then propagates `yield*`. `src/cpu/phase-helpers.ts` parses and inlines straight-line scalar barrier helpers; `src/cpu/phases.ts` splits unconditional entry barriers into plain phase functions. It turns entry returns into completion flags, which `src/cpu/index.ts` checks across lanes before the next phase. Other control-flow barriers retain generator functions. Structured function effects and barrier nodes would remove these text passes.

### 11. Local names are reusable across sibling blocks, without scope hooks

Reusing names is valid WGSL, but a backend that caches scalar components or buffer offsets by emitted name needs to retire that cache on block exit. The generator exposes no enter/leave-block hooks with stable variable identities. Reusing a cached temporary can leak an address from a previous sibling block and produce `ReferenceError: cpuAddress_4 is not defined` in generated JavaScript.

```ts
if (first) {
  const item = rows.$[i];
  consume(item.position);
}
if (second) {
  const item = rows.$[j];
  consume(item.position);
}
// Both blocks may receive the same local identifier; they are different bindings.
```

**Upstream.** Expose block-scope lifecycle hooks or stable variable identities in `tgsl/shaderGenerator.ts` and `resolutionCtx.ts`. Backend metadata should be keyed by binding identity, not printed text.

**Workaround.** Compiler-owned address/vector/argument temporaries in `src/cpu/generator.ts` use `ctx.makeUniqueIdentifier(..., 'global')`, with maps saved and restored per function. User-local names are never substituted throughout the whole function body.

The GLSL compute generator hits the same limit with facts rather than caches: "this local holds `gid.x`" keyed by the emitted name would carry into a sibling block's `const i = gid.x + 1` and prove a write own that is not. `KitGlslComputeGenerator._emitVarDecl` in `src/glsl/compute.ts` forgets a name whenever any declaration reuses it. `ctx.getById` cannot help: it takes the JavaScript identifier, not the emitted one.

### 12. A function is generated once per argument type, never per argument value

The shellless repository keys variants by argument types, and a `tgpu.fn` with a shell is generated once for all callers. GLSL needs a body per call site in two cases: a texture or sampler argument, because GLSL samples through one combined uniform, so `blurred(a, s, uv)` and `blurred(b, s, uv)` need two bodies; and an argument that is the invocation's own index, because a helper writing `buf[i]` runs as a fragment pass only for calls that pass the own index. Tagging the argument's schema does not make a variant: textures compare equal by sample type, samplers entirely.

```ts
const blurred = (map: d.texture2d<d.F32>, s: d.sampler, uv: d.v2f) => {
  'use gpu';
  return std.textureSample(map, s, uv);
};
blurred(layout.$.a, layout.$.s, uv);
blurred(layout.$.b, layout.$.s, uv); // reuses the body generated for `a`
```

**Upstream.** A generator hook that adds a key per argument to the variant lookup, such as `variantKey(arg: Snippet): string | undefined`, used by `ShelllessRepository.get` in `tgsl/shellless.ts` and by shelled function resolution in `core/function/fnCore.ts`.

**Workaround.** For shellless functions, `KitGlslGenerator._callShellless` in `src/glsl/generator.ts` swaps `ctx.shelllessRepo.cache` while the lookup runs. For shelled functions, such as a guarded pipeline's callback, there is no seam: `KitGlslComputeGenerator` in `src/glsl/compute.ts` generates the function for its first call's own arguments and rejects the kernel when a later call passes others to a function that depends on them.

### 13. Buffer access becomes text before a generator sees what it reads

`layout.$.items[i].position.x` resolves the binding to its identifier and `accessProp`/`accessIndex` stitch the access together. A backend that reads buffers another way (the fallback reads words from a data texture, and must not load a whole element to read one field) gets no structured place. Overriding `_expression` catches access in the AST, but access through a value proxy at the JavaScript level is stitched at once. A generator cannot hold its own object as a snippet's value either: an object without `$comptimeValueOf` counts as known at compile time, so `place < 1` throws "requires numeric operands", and the symbol is not exported.

```ts
const x = layout.$.items[i].position.x; // reaches the generator as the text `items[i].position.x`
```

**Upstream.** Represent buffer, member and index access as a place the generator turns into code (`emitLoad(place)`), and export `$comptimeValueOf` or `runtimeOnly` from `typegpu/~internal`. In `tgsl/accessProp.ts`, `tgsl/accessIndex.ts` and `types.ts`.

**Workaround.** `KitGlslGenerator._expression` in `src/glsl/generator.ts` narrows places itself and keeps each place's load expression as the snippet's value, with a map from that text back to the place.

### 14. A ternary is written as WGSL's `select`

`_expressionImpl` emits a JavaScript ternary as `select(f, t, c)` through `stitch`, past `emitCall`. GLSL has no `select`.

```ts
const y = x < 0.5 ? 1 : 2; // select(2, 1, (x < 0.5)) in every target language
```

**Upstream.** Emit the conditional through `emitCall('select', …)` or an `emitConditional` hook, in `tgsl/wgslGenerator.ts`.

**Workaround.** `SELECT_OVERLOADS` in `packages/typegpu-gl/src/glslGenerator.ts`, declared when a ternary's output starts with `select(`. CPU `CpuGenerator._expression` also detects this resolved conditional and routes its arguments through `emitCall('select', ...)`, after the base generator has validated the branches. Without that step, a rare runtime branch in falling-sand reaches an undefined JavaScript `select`.

### 15. `functionDefinition` must call `determineReturnType`, which nothing says

A generator that builds an entry function itself and never calls `options.determineReturnType()` fails with `Failed to determine return type` from `resolveFunction`, far from the cause.

```ts
override functionDefinition(options: FunctionDefinitionOptions) {
  return myOwnEntry(options); // throws later unless it calls options.determineReturnType()
}
```

**Upstream.** Document it on `FunctionDefinitionOptions`, or call it in `resolveFunction` when the generator did not. In `tgsl/shaderGenerator.ts` and `resolutionCtx.ts`.

**Workaround.** `KitGlslComputeGenerator.computeEntry` in `src/glsl/compute.ts` calls it.

### 16. Generators reach into `ResolutionCtx` members its type does not promise

The GLSL generators use `ctx.topFunctionScope` (which function is being generated), `ctx.shelllessRepo.cache` (entry 12), `ctx.setBlockExternals` to bind an already generated snippet to a temporary name, because `defineVariable` rejects snippets of unknown type such as a namespace or a comptime object, and replacements of `ctx.resolve` (entry 5) and `ctx.addDeclaration` (to translate `common.fullScreenTriangle`, entry 6) on the context instance. Each can break in a patch release.

```ts
const scope = (ctx as unknown as { topFunctionScope: unknown }).topFunctionScope;
```

**Upstream.** Add what a generator needs to the public `ResolutionCtx` type, and let `_expression` continue from a snippet instead of an AST node, such as `_access(target: Snippet, node)`. In `types.ts` and `tgsl/wgslGenerator.ts`.

**Workaround.** `src/glsl/generator.ts`, `src/glsl/compute.ts` and `src/generator.ts`.

### 17. A second resolution of a pipeline numbers its bindings its own way

The catch-all group's bindings are numbered in the order a resolution reaches its resources (`allocateFixedEntry`), and a layout without an explicit index takes the next free group in the order it is reached. A generator that resolves the same pipeline again in another order, as the GPU form of a compute pipeline does while it analyses what the kernel writes, gets other numbers than the resolution the pipeline is created from, and nothing says so: the form bound the wrong textures in two examples until a check compared it with the WGSL.

```ts
const a = root.createReadonly(d.arrayOf(d.f32, 4));
const b = root.createMutable(d.arrayOf(d.f32, 4));
// WGSL: `a` at @binding(0), `b` at @binding(1); a generator that reaches `b` first numbers them the other way
```

**Upstream.** Let a nested resolution inherit its parent's binding numbers, for example by taking the parent's `fixedBindings` and placeholders as an option of `resolveWithContext`, or give each fixed binding a stable index when it is created. In `resolutionCtx.ts`.

**Workaround.** `KitGlslComputeGenerator.#numbered` in `src/glsl/compute.ts` maps each binding to the outer resolution's catch-all index or layout placeholder, through `fixedBindings` and `bindGroupLayoutsToPlaceholderMap`, and writes placeholders the outer resolution replaces.

### 18. A vertex output position cannot be typed `@invariant`

`d.invariant(d.builtin.position)` is a valid vertex output, and TypeGPU emits `@invariant` for it, but the types of `tgpu.vertexFn`'s `out` accept only `d.builtin.position` with its one attribute.

```ts
tgpu.vertexFn({ out: { position: d.invariant(d.builtin.position) } }); // type error
```

**Upstream.** Accept `Decorated<Vec4f, [Builtin<'position'>, Invariant]>` in `AnyVertexOutputBuiltin`. In `core/function/tgpuVertexFn.ts` and `data/wgslTypes.ts`.

**Workaround.** The Kit's `passes.ts` builds its vertex shell untyped; tests cast through `unknown`.

### 19. Swizzle mutation validation depends on the emitted value's class

`validateSnippetMutation` is private and recognises `instanceof Swizzle`. A swizzle is a fresh value when read, but assignment validates its base's mutability. Replacing it with a target-language string loses that distinction: keeping its runtime origin rejects valid writes; changing its origin to the base's makes valid reads look like reference assignments.

```ts
let value = d.vec4f(1, 2, 3, 4);
value.yx = d.vec2f(value.xy); // Valid, but a string-valued swizzle is rejected as a non-reference.
value = value.yxwz; // Valid, but giving the swizzle a local-variable origin rejects this copy.
```

Constant vector indexing takes this same path: `state.$[0]` becomes a `Swizzle` in `tgsl/accessIndex.ts` and resolves to `state.x`. A subclass that only retargets member-access AST nodes misses indexed reads and writes. On an array-valued CPU backend this produced undefined reads in `randf`'s private state and an all-zero random stream.

**Upstream.** Keep lvalue provenance separate from emitted text, and expose swizzle emission with the original base and components. Mutation validation in `tgsl/wgslGenerator.ts` should consume that provenance without requiring a concrete `tgsl/swizzle.ts` instance.

**Workaround.** `CpuGenerator.#lowerSwizzle` clones the original Swizzle instance and uses `makeResolvable` to change only its emitted code. This retains readonly and nested-swizzle validation while lowering components to JavaScript indices.

### 20. Whether a local is reassigned is decided after its declaration is generated

A `let` whose reassignment may sit in a pruned branch reaches `_emitVarDecl` with a placeholder keyword, `#VAR_n#`, which the function scope replaces once the body is done; a JavaScript `const` that holds a struct copy gets one too. A generator that wants to know whether a variable keeps its first value, such as the GPU compute form deciding whether `view.cursor` still holds a uniform's value at a condition, cannot ask at that point.

```ts
const view = View(layout.$.view); // `#VAR_0# view_1 = …`: let or var is decided later
if (id.x === d.u32(view.cursor.x)) pick.$ = …;
```

**Upstream.** Pass the generator the variable's final mutability, or a hook called when placeholders are resolved, with each variable and whether it was reassigned. In `tgsl/wgslGenerator.ts` and the function scope in `resolutionCtx.ts`.

**Workaround.** `KitGlslComputeGenerator` in `src/glsl/compute.ts` records every assignment's root name per function and rejects the kernel at the function's end when a choice of invocation read a variable that was reassigned.

### 21. A generator cannot ask what a function reads

An accessor filled with a function, such as `tgpu.accessor(Pass, () => Pass(layout.$.pass))`, reaches a kernel as a call to a generated function. A generator that needs to know whether a value is the same for every invocation, as the GPU compute form does for an offset like `cells[own + (1 - current.$.source) * n]`, sees only the call. TypeGPU records which bindings a resolution reaches (`bindingUseProbe`) but not per function, nor whether a function reads private variables or the invocation's builtins.

```ts
const current = tgpu.accessor(Pass, () => {
  'use gpu';
  return Pass(layout.$.pass); // reads a uniform only, but the caller sees `current()`
});
```

**Upstream.** Record, per generated function, the bindings, private variables and builtins it reads, transitively, and expose it to generators beside the function's snippet. In `core/function/fnCore.ts` and `resolutionCtx.ts`.

**Workaround.** `KitGlslComputeGenerator` in `src/glsl/compute.ts` marks a function as shared when it has no parameters and its generated body names no private variable, written resource, invocation id or function that is not shared itself.

### 22. Local liveness and callee effects are not exposed together

Function arguments expose `used`, but local declarations have no equivalent hook and function definitions carry no reusable purity summary. When a specialised callback ignores an argument, a backend can omit the call's argument but still computes the local that supplied it. In stampede's culling this left an unused default bounding sphere, including its transforms, in every row's visibility check.

```ts
const fallback = sphereOf(row);
return customBounds(row, fallback); // This fill never reads its second argument.
// A backend can emit customBounds(row, undefined) but still emits sphereOf(row).
return placementOf(row).position; // The helper still computes rotation and scale.
```

**Upstream.** Expose local use counts or stable binding identities, field use paths on record arguments and returns, and a function effect summary covering writes and barriers, through `tgsl/shaderGenerator.ts` and `resolutionCtx.ts`. Then a backend can remove unused pure initializers and record fields before emitting text.

**Workaround.** `CpuGenerator` tracks function effects; `src/cpu/dead-locals.ts` conservatively removes unused generated declarations whose calls are known pure. `src/cpu/record-fields.ts` projects pure helpers onto the record fields their callers use, preserving eager snapshots and argument evaluation. Unknown calls, mutations, escaping records and field dependencies keep the original calculation.

## Roots and devices

### 23. `root.configureContext` assumes a WebGPU canvas context

It calls `canvas.getContext('webgpu')` and `navigator.gpu.getPreferredCanvasFormat()`. A root over a device that presents another way, such as the fallback's, cannot be configured through it, and `navigator.gpu` does not exist where the fallback is needed.

```ts
const root = tgpu.initFromDevice({ device: webglDevice });
root.configureContext({ canvas }); // Unable to initialize 'webgpu' context on the provided canvas.
```

**Upstream.** Let the device supply the context, for example an optional `configureCanvas` member TypeGPU prefers when present.

**Workaround.** `backends.of(root).configure` in `packages/typegpu-kit/src/backends.ts`, which `web.surface` calls.

### 24. TypeGPU reads WebGPU's constant namespaces from the global scope

`GPUBufferUsage`, `GPUTextureUsage`, `GPUShaderStage`, `GPUMapMode` and `GPUColorWrite` are read as globals when a buffer, texture or layout is created. A browser without WebGPU does not define them, so creating a uniform throws `ReferenceError: GPUBufferUsage is not defined` before any device is involved.

```ts
tgpu.initFromDevice({ device: webglDevice }).createUniform(d.f32); // ReferenceError without WebGPU
```

**Upstream.** Keep the constants in TypeGPU. The specification fixes their values.

**Workaround.** `defineConstants` in `src/globals.ts` defines the missing ones.

### 25. Running a kernel on the CPU needs a second code generator

TypeGPU can call a `'use gpu'` function in JavaScript, but not a whole kernel: `tgpu.simulate` runs one invocation, the texture functions throw `MissingCpuImplError`, atomics have no CPU implementation, the barriers do nothing, and values are generic vector objects. So the fallback generates its own JavaScript from the typed AST. [TypeGPU #3220](https://github.com/software-mansion/TypeGPU/pull/3220) adds slots and accessors to CPU execution; the rest is open.

```ts
const kernel = tgpu.computeFn({ workgroupSize: [64] })(() => {
  'use gpu';
  std.workgroupBarrier(); // nothing on the CPU runs this kernel, its barriers or its atomics
});
```

**Upstream.** A supported CPU executor for compute pipelines, taking bound resources and a dispatch size, would let `tgpu.simulate`, tests and fallbacks share one implementation.

**Workaround.** `src/cpu`.

### 26. A texture's cube or array viewing is unknown when it is created

WebGPU lets a 6-layer texture be viewed as `cube` or `2d-array`; WebGL 2 must pick `TEXTURE_CUBE_MAP` or `TEXTURE_2D_ARRAY` when it allocates storage. WebGPU's compatibility mode solves the same problem with `textureBindingViewDimension` on `GPUTextureDescriptor`, but TypeGPU never passes it: `INTERNAL_createTexture` materialises the texture from `props` alone, and views are created later, when a bind group first unwraps them. A texture written before its first cube view is created ends up stored as an array, and every cube view of it then samples a copy the device refreshes whenever the texture changes.

```ts
const sky = root
  .createTexture({ size: [256, 256, 6], format: 'rgba16float' })
  .$usage('sampled', 'render');
sky.write(faces); // allocates storage now, before any view exists
const cube = sky.createView(d.textureCube(d.f32)); // on WebGL: a copy, re-blitted after each write
```

**Upstream.** Accept `textureBindingViewDimension` in texture props, or infer it from the first view schema created before materialisation, and pass it to `device.createTexture`. In `core/texture/texture.ts`.

**Workaround.** `sampled` and `derivedCopy` in `src/device/texture.ts`: storage is chosen from the views created before allocation, reallocated if the first sampled view disagrees and nothing was written yet, and otherwise copied per view.

### 27. TypeGPU's internal blits allocate per call and submit their own encoder

`blit` in `core/texture/textureUtils.ts` creates a bind group on every call, and `generateTextureMipmaps` and `writeImages` each create a command encoder and submit it at once. A module that regenerates mipmaps every frame, such as one mipmapping a rendered target, allocates bind groups each frame and submits outside the frame's encoder, so its work cannot be ordered with the frame's passes.

```ts
target.generateMipmaps(); // its own encoder and submit, a new bind group per level and layer
```

**Upstream.** Cache the bind group per source view, and give `generateMipmaps` an optional encoder to record into. In `core/texture/textureUtils.ts` and `core/texture/texture.ts`.

**Workaround.** None in the fallback; the device runs each submit as it comes.

### 28. A copy of a local array element is still a reference when assigned back

Inside a `'use gpu'` function, `d.vec2u(group[1])` makes a copy, and binding it to a `const` should leave a value. Assigning that `const` to another element of the same local array is rejected as a reference assignment, so a compare-and-swap over a local array needs a second copy constructor at each store. A bitonic step does exactly this: it reads two elements, compares them and writes them back swapped.

```ts
const Pairs = d.arrayOf(d.vec2u, 8);
const swap = tgpu.fn(
  [],
  d.u32,
)(() => {
  'use gpu';
  const group = Pairs();
  const b = d.vec2u(group[1]);
  group[0] = b; // "references cannot be assigned. Try 'group[0] = vec2u(b)'"
  return group[0].x;
});
```

**Upstream.** Track the result of a schema call as a value, not a reference, through a `const` binding, so assigning it copies without another constructor. In the code generator's assignment check.

**Workaround.** Wrap each stored value in its constructor again, `group[0] = d.vec2u(b)`, as the grid's bitonic `merge` kernel does.

### 29. A shift of module constants is not folded where a schema needs a count

`d.arrayOf(T, n)` inside a `'use gpu'` function needs `n` known when the shader resolves. `2 ** N` with a module constant `N` folds and resolves, but `1 << N` does not: it is resolved as a runtime expression and the schema call fails. A module that sizes a local array by a bit count has to write the power or hoist the schema.

```ts
const N = 3;
const f = tgpu.fn(
  [],
  d.u32,
)(() => {
  'use gpu';
  const group = d.arrayOf(d.vec2u, 1 << N)(); // "Called comptime function with runtime-known values: '1 << N'"
  return group[0].x;
});
```

**Upstream.** Fold bitwise and shift operators on known numbers as the other arithmetic operators are folded. In the code generator's binary-expression handling.

**Workaround.** Hoist the schema to module scope, `const Gathered = d.arrayOf(d.vec2u, 2 ** STEPS)`, and call `Gathered()` in the body, as `grid.ts` does.

### 30. WGSL language features come from the global GPU, not the root's device

`root.enabledWgslLanguageFeatures` returns `navigator.gpu.wgslLanguageFeatures`, even for a root made with `initFromDevice` over another backend. A CPU or WebGL root can therefore advertise immediates merely because the machine also has a WebGPU adapter, despite its own `maxImmediateSize` being zero.

```ts
const root = tgpu.initFromDevice({ device: fallbackDevice });
root.device.limits.maxImmediateSize; // 0
root.enabledWgslLanguageFeatures.has('immediate_address_space'); // true on some WebGPU hosts
// An example selects immediateVar and fails to compile on the fallback.
```

**Upstream.** Accept supported language features in `initFromDevice`, or derive them from the selected generator/device instead of the global adapter, in `packages/typegpu/src/core/root/init.ts`.

**Workaround.** `src/webgl.ts` and the CPU test root in `tests/cpu/device.ts` supply an empty `enabledWgslLanguageFeatures` set. Device limits alone cannot affect the current getter.

## Source transforms

### 31. Multiple declarations in one shader statement fail before resolution

`tinyest-for-wgsl` rejects a JavaScript declaration containing more than one declarator. The generator never sees it, although its equivalent separate statements work. Copying a numerical reference implementation with `const a = ..., b = ...` into a shader therefore fails during the build.

```ts
const kernel = tgpu.fn(
  [],
  d.f32,
)(() => {
  'use gpu';
  const a = d.f32(1),
    b = d.f32(2);
  return a + b;
});
// Error: Currently only one declaration in a statement is supported.
```

**Upstream.** Lower the declarators to separate tinyest declaration statements in source order, preserving initializer dependencies, in `tinyest-for-wgsl`'s `transpilers.ts` variable-declaration handler.

**Workaround.** Split declarations in shader bodies, as in `tests/cpu/curve-thresholds.test.gpu.ts`. This is needed before either the WGSL or CPU generator can run.

## Value types

### 32. Matrix column references are writable in WGSL but readonly in TypeScript

The matrix `columns` tuple has a readonly index signature. Replacing a column of a local matrix therefore fails TypeScript checking even though TypeGPU resolves the assignment and Dawn executes it correctly. Component assignments through the same view are accepted by the types.

```ts
const kernel = tgpu.fn(
  [],
  d.mat2x2f,
)(() => {
  'use gpu';
  let matrix = d.mat2x2f();
  matrix.columns[0] = d.vec2f(7, 8);
  return d.mat2x2f(matrix);
});
// TS2542: Index signature in type 'readonly [v2f, v2f]' only permits reading.
```

**Upstream.** Expose a writable column view, with matching CPU assignment semantics, in `packages/typegpu/src/data/wgslTypes.ts` and `data/matrix.ts`, or provide a typed column setter that lowers to a WGSL column assignment.

**Workaround.** The matrix-column regression in `tests/cpu/indexed.ts` casts the columns to a mutable tuple. The CPU generator already writes the underlying column components.

## Name resolution

### 33. Captured names can resolve to inherited Object prototype methods

Function argument lookup indexes `layer.argAccess[id]` without checking whether the property belongs to that argument table. A captured function named `valueOf` instead finds `Object.prototype.valueOf` and invokes it with no receiver, failing before either shader generator can resolve the captured function.

```ts
const valueOf = () => {
  'use gpu';
  return d.u32(23);
};
const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
  'use gpu';
  output.$ = valueOf();
});
// Cannot convert undefined or null to object, in ItemStateStackImpl.getSnippetById.
```

**Upstream.** Use a null-prototype argument table or an own-property check in `packages/typegpu/src/resolutionCtx.ts`, in `ItemStateStackImpl.getSnippetById`.

**Workaround.** The effectful RHS helper in `tests/cpu/indexed.ts` is named `nextValue` instead of `valueOf`.
