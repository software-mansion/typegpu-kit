# TypeGPU Kit

Rendering, post-processing and simulation modules on TypeGPU for the web and React Native, recorded into a frame you own.

## Layout

- `packages/typegpu-kit` contains `@typegpu/kit`. Core modules live in `src/*.ts`; `post/`, `particles/`, `render/`, `text/` and `gltf/` contain the other areas. `/testing` exports test helpers, `/debug` a GPU debug panel, and `typegpu-kit-bake-font` bakes the MSDF fonts in `fonts/`.
- `packages/typegpu-kit/src/geometry` contains TypeGPU's geometry, public at `@typegpu/kit/geometry`, and `src/sort` its radix sort, which only `sorter` imports. Edit them only as their READMEs say.
- `packages/typegpu-create-kit` contains `@typegpu/create-kit`, run by `npm create @typegpu/kit`, which writes `template/` with one of `scenes/`, the Starter scenes page's scenes, at the kit's version.
- `packages/typegpu-kit-pull` shares the box and card animation between the docs and native app.
- `packages/typegpu-kit-examples` contains the examples, private and shared by the docs and the native app in `apps/typegpu-kit-native`. Each lives in `src/<name>/`, its test in `tests/`.
- `apps/typegpu-kit-docs` contains the Starlight docs, with module headers in `src/contracts` and the example pages' web host in `src/examples`.

Post-processing, particles, rendering, text, glTF and geometry import only public core files, the subpath entries and files re-exported flat such as `inject.ts`, by relative path. Core imports none of them, and they reach the radix sort through `sorter`. `layering.test.ts` enforces this.

## Examples

Each example directory contains `meta.json`, modules, `scene.ts`, and its posters `poster.webp` and `poster-small.webp`. Add `assets.ts` when the scene needs files, and `ui.ts` when it shows texts, game buttons or an image picker. List the example in the package's `src/index.ts`, and give its `meta.json` a `category` from the `categories` exported there, `games` for a game, and a card `rarity` from `RARITIES` in `packages/typegpu-kit-pull/src/rarity.ts`. Its directory name is its key in figures, tests and URLs: `/examples/lattice/`. After changing how an example looks, render posters on dawn.node with `pnpm --filter typegpu-kit-examples posters`, or some with `POSTERS=forest,blockfall`.

Export `createScene(root, surface, options?)` from `scene.ts`, returning the `ExampleScene` that the package's `src/contract.ts` describes. Draw the whole frame into `surface.view(frame)`. Keep the animation loop and DOM outside the scene. In `destroy()`, free targets, large buffers, models, the modules the scene made and anything it registered with something that outlives it. Leave small uniforms, samplers, bind groups and descriptions to the garbage collector. An unreferenced resource is not a leak.

One host runs every example page, gallery card and figure, so examples carry no page code. Test each example with `describeExample` from `tests/utils/examples.ts`, which checks its steady frames and settings on `recordingRoot()`, and pin its passes and pipeline count.

An example that ships models or textures names their CC0 source in its description. Lint rejects `console`, TODO comments and `any` in examples.

## Commands

```sh
pnpm dev          # serve docs and examples for a WebGPU browser
pnpm dev:host     # serve them over LAN HTTPS, self-signed, for phones
pnpm test         # types, lint, format, unit and GPU tests, docs build with link checks
pnpm test:unit    # headless Vitest tests on recordingRoot
pnpm test:gpu     # *.test.gpu.ts on dawn.node, including every example's WGSL
pnpm build        # build docs and examples
pnpm preview      # serve the last build
pnpm fix          # fix lint and formatting
```

A push to `release` deploys the docs to https://docs.swmansion.com/typegpu-kit/ through `.github/workflows/docs.yml`. Pages link from the root, such as `/kit/frames/`, and components through `components/site/urls.ts`; the build prefixes the base, which `DOCS_BASE=/typegpu-kit/ pnpm build` reproduces locally.

The docs build keeps every address in `apps/typegpu-kit-docs/src/build/addresses.json`, the pages and module cards the site has published. A page that moves names where it is in `src/build/moved.ts`, which leaves a redirect and a Markdown pointer at its old address, and a card that moves leaves its anchor. Local builds record new pages and cards automatically; CI leaves the list unchanged. The build rejects a missing address; accept its removal with `ADDRESSES=record pnpm build`. The build also fails on a page without one description.

The kit ships the React Native patches in `packages/typegpu-kit/patches/`. "Setting up a project" on the getting-started page, between `BEGIN:setup` and `END:setup`, names exactly those files; the gate checks that they match.

Run `pnpm --filter typegpu-kit-docs agents` to regenerate the untracked `packages/typegpu-kit/docs/`. Write the kit's agent guide, the skill `packages/typegpu-kit/skills/typegpu-kit/SKILL.md`, by hand, in two parts:

- Its "Find it before you build it" index grows with the kit. Each line gives one need, the modules for it and a file to start from. Give a new module or capability a line, and split a line that passes 256 bytes.
- Everything outside the index is rules, 8 KB together. Add one only for a mistake an agent makes after reading the module's page. At the limit, move a rule to its page; never shorten other lines to fit.

The gate checks both sizes and rejects a module the skill never names, a `namespace.member` the kit no longer exports, and an example link whose example stops using the module its sentence names. The kit's build opens each module's `.d.ts` with its guide and examples.

## Module rules

Follow these rules when you add GPU functionality or change a public API. [How the kit works](apps/typegpu-kit-docs/src/content/docs/concepts/how-the-kit-works.mdx) teaches the same seven.

1. The program owns the frame: the device, loop, encoder, passes and submit. Modules hold no frame state, encoder or pass, and never submit. A recording method takes the smallest piece of the frame it needs: the pass it adds to, TypeGPU's or raw as `AnyComputePass` or `AnyRenderPass`, through `adopt`; `Frame | TgpuCommandEncoder`, through `encoderOf`, when it records passes or copies of its own; the `Frame` when it works after the submit; numbers for maths on the CPU. It reads `frame.encoder` on each call.
2. A module is a factory that borrows the root, takes options and returns its TypeGPU resources, `'use gpu'` functions, recording methods and, if it draws the scene, a drawing. It destroys only what it allocated.
3. Costs are stated and checked. Factories allocate, `initAsync` compiles, frames record. A steady frame creates no GPU objects except those a header's `@allocates` names. A module's header lists what it owns, binds and records, and the gate checks it on `recordingRoot()`.
4. Modules connect through TypeGPU's functions, slots, accessors and `pipe`, and return TypeGPU types. An instance's members carry their fill: make them in `create` over that instance's resources, so one shader reads two instances and a pipeline that reads them needs no fill. A placeholder is filled by the pipeline: keep a definition's members and hooks that pipelines fill differently, such as cameras and shading slots, at module scope, filled by the instance's `inject()` or with `cfg.with`, each with an `unfilled` default naming what fills it. Put a resource that is replaced, such as a resizing map, in a `bindings.layout` part.
5. Identity is JS identity. A definition's name, as in `table.define('Enemy', …)`, only labels types and WGSL. Module scope holds schemas, hooks and pure functions, never resources, registries or string keys. The kit has no build transforms of its own. Only the tools that watch a program, `@typegpu/kit/debug` and `@typegpu/kit/testing/dawn`, and the worker runtime's timer workaround may patch globals, one at a time, undone when disposed.
6. Many of a thing are rows in a buffer with stable ids, not objects on the CPU.
7. A module that draws returns a `Drawing` or a `Compound`, and descriptions compile the pipelines. A module never creates a render pipeline for a target it was not handed.

### How we build

- Each module is one readable source file, built and tested for a use real programs have, even before an example calls it.
- Check on `recordingRoot()` what a module allocates, uploads and records, and that a steady frame creates no GPU objects. Pin exact commands only where their order or count is the contract.
- Compile a module's WGSL on dawn.node with `gpuRoot()` from `@typegpu/kit/testing/dawn`, and assert a result where correctness depends on execution.

### What a release keeps

"Versions" on the getting-started page states it for users.

- The contract changes only by addition. `packages/typegpu-kit/tests/contract.test.ts` prints its declarations and fails when they differ from `api/contract.api.md`. Update the report only for an addition.
- A module keeps its import path, namespace, factory names and recording methods through a minor release.
- Tag a module `@unstable` in its header while its names or shape may still change.

### API shape

- Export each module as a namespace, such as `export * as frames from './frames.ts'`, from the package entry and its own subpath; platform modules and tools only from subpaths. Export its options, types and `'use gpu'` functions beside the factory, and key types flat too. Name exports with identifiers: React Native's Babel preset cannot compile string names in `export * as`.
- Factories allocate and are the only members that take a root, first: `create(root, …)`, or a name for the variant one makes, such as `cameras.orbit(root, …)`. `define(…)` holds code, the hooks. Every other member is a value or a `'use gpu'` function.
- A factory takes at most three inputs by position after the root, then one options object of values. Only a group that switches a feature on, such as a lighting's `shadow`, or a pair that is one value, such as a pass's `bias`, nests, one level deep. Inputs that read better by name, such as a capacity or a callback, are required fields.
- An invalid option never changes the type of what a factory returns. A factory that infers its options literal types the parameter with `Checked`, so the mistake is one error at the literal.
- A module the factory's own code reads comes by position, as a camera does, or as an option field, never in a list; a callback carries its own fills. A drawing reads what its pass description supplies, such as the camera and the lighting given to `beneath`, from the description's `inject()`, and takes an option field only to override it.

### Names

- Name a namespace for what `create` gives you, in a graphics programmer's word: `bloom`, `picker` rather than `picking`. Use the plural for many rows or kinds, such as `particles` and `cameras`, and the singular for one facility, such as `grid`. Name a library of `'use gpu'` functions for its field, such as `shading`, and a platform module for its platform or role, such as `web`.
- Use a word WebGPU or TypeGPU defines, such as pass, pipeline, buffer or binding, only for exactly that thing. Abbreviate only as graphics programmers do, as in `taa` and `gltf`.
- PascalCase values are TypeGPU schemas and table definitions, named for their record, such as `PointLight`, and React components. Fixed specs are capitals, such as `HDR` and `WORKGROUP`; settings `defaults` and descriptions such as `passes.color` stay lowercase values. Handle types, such as `Lighting`, are types only and name what the factory gives you. The kit has no classes.
- A name means one thing across the kit, never a schema in one module and a handle type in another. Two values never differ only in case, so a `Camera` accessor is `cameras.current`, not `camera`, but a type may share its maker's word: `bloom` gives a `Bloom`.
- Name an instance for what it holds, such as `const glow = bloom.create(…)`, and import namespaces under their own names.
- A verb means one thing too:

  | Verb                          | Does                                                                                     |
  | ----------------------------- | ---------------------------------------------------------------------------------------- |
  | `create`, or a variant's name | allocates, from a root                                                                   |
  | `define`                      | returns a definition with its hooks, without a root                                      |
  | `add`                         | puts an item into a collection and returns its handle                                    |
  | `clear`                       | empties a collection: at once on the CPU, or recorded given a frame or encoder           |
  | `write`, `patch`              | set values from the CPU and never record: at once, or at the next `flush`                |
  | `set<Field>`                  | changes one setting that has follow-on work, such as a refit or resize                   |
  | `flush`                       | uploads the writes waiting since the last one                                            |
  | `step`                        | runs one fixed step, in a compute pass you opened when it is on the GPU                  |
  | `draw`                        | draws into a render pass you opened                                                      |
  | `record`                      | records passes or copies of its own                                                      |
  | `capture`                     | copies to the CPU after the submit                                                       |
  | `bake`                        | precomputes a texture from a source, as the baker, environment light and font atlases do |
  | `update`                      | advances the module once a frame, from `dt` or the frame                                 |
  | `inject`                      | returns the fills a pipeline needs to read the module                                    |
  | `initAsync`                   | compiles what the module built                                                           |
  | `destroy`                     | frees what the module allocated                                                          |

## Writing GPU code

Follow the kit's agent guide, `packages/typegpu-kit/skills/typegpu-kit/SKILL.md`, through its GPU code and platform notes, and read the `typegpu` skill, `node_modules/typegpu/skills/typegpu/SKILL.md`, before writing TypeGPU code. In addition, in the kit's modules:

- Export, inject and fill as the module rules and API shape above say, with members over an instance's own resources, an `unfilled` default on every placeholder other modules read and swappable resources in `bindings.layout` parts.
- Tag a module's header with `@binds` lines, one per group of accessors a shader reads, giving its storage and uniform buffers per stage. modules.md's table and the binding budget come from them.
- Fill an accessor of a buffer that both kernels and vertex stages read with `byStage(buffer)`, read-only in the vertex stage.

## Style

- Keep comments for public API docs, invariants the types cannot express, platform constraints and suppression reasons.
- Read `apps/typegpu-kit-docs/WRITING.md` before you write a page or TSDoc, or add a module.
- Write each commit message as one line of at most 72 characters: `type(scope): title` or `type: title`, with no body or trailers. Types are `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `build`, `ci`, `chore`, `revert` and `merge`. Scopes name a kit module, an example or an area such as `docs`, `native`, `web` or `examples`, in kebab-case. Titles start with a lowercase imperative verb, keep proper names' spelling and end without a period. Use `build(deps)` for dependency upgrades and `merge: integrate <topic>` for merges.
- Title each pull request as a commit message. Keep its description to a few short lines: what changes and the docs pages that show it, without headings, checklists or test reports.
- Omit AI attribution from commits and pull requests.
