# Writing the TypeGPU Kit docs

Readers want to build a game or demo. Module pages assume TypeGPU basics. They explain what a module does, when to use it and its constraints. Tutorials assume no graphics or TypeGPU knowledge.

## Voice

- Write in second person and present tense. Use short, plain sentences. Explain the problem before the API, then show how the module solves it.
- State concrete facts. Keep a measurement only when it directly supports a rule the reader follows. Omit measurement tables and device or test stories.
- Use no dashes as asides, curly quotes, puffery, closing summaries or exclamation marks. Use colons only before lists, code blocks or examples.
- State platform requirements and constraints. Omit project history, design rationale and coverage disclaimers.
- Say what a release keeps in one place, [Versions](src/content/docs/getting-started.mdx) on the getting-started page. A module that may change in any way carries `@unstable` in its header, which its card shows. Add no other stability words to prose.
- Identifiers in backticks. Write thousands in prose with a space: 4 096. Link another page the first time it comes up.
- A word the kit uses in a sense of its own gets an entry in `src/components/site/terms.ts`: the word, one sentence a TypeGPU beginner can read and the section that teaches it. The build marks its first use on each page with that sentence, and [How to read the docs](src/content/docs/how-to-read.mdx) lists every word. Don't link a term on first use only to define it.
- Say a thing once, and give each rule one home. Don't restate the code, the module header or another page.
- Write prose that stands without its figures. Agents read each page as Markdown that keeps only the captions.

## Tutorials

A tutorial, `src/content/docs/tutorials/<name>.mdx`, builds one example from an empty canvas, in `<Steps>`. Its reader may never have written a shader or used TypeGPU.

Use figures where they explain more than prose, and keep code blocks short.

- Each step adds one idea. It says what the reader is about to make and why it works, shows the code that idea adds, and ends by describing the result on screen.
- Show every line the reader writes, exactly once, in blocks of 10 to 40 visible lines of added or changed code. Keep the rest of the file hidden in the block so twoslash checks it. Use `// ---cut---`, `// ---cut-start---` and `// ---cut-end---`, or `// ---cut-after---`. The finished files are one link away, in the example.
- Teach the kit, never the docs host. The reader's code imports nothing from the docs app.
- Give invisible mechanisms a figure, such as thread workgroups, full-screen triangles, noise or fixed steps. Reuse a kit page's figure when it explains the idea. Otherwise, add one under `src/components/viz/`.
- Show each visual change with an `<ExampleFigure>` of the finished example. Use its `options` and `values` to reproduce the step, such as disabling glow. If no setting reproduces it, make a figure from the example's modules.
- A GPU term gets one plain sentence the first time it comes up (buffer, kernel, thread, workgroup, draw, vertex and fragment, uniform, encoder, pass, submit), and a link to the page that covers it.
- End with a tree of the files the reader wrote and what to try next. Leave out the repository's plumbing: `meta.json`, the poster, the example's test, its registration and the example page's host.

## Page shape (module pages)

A module page lives at `src/content/docs/kit/<slug>.mdx`, where `<slug>` is the kebab-case name of its lead module, the first its frontmatter lists: `kit/mesh-pool` for `meshPool`. List the page's natural subject first. `kit/` holds nothing else. Pages on how the kit works go in `concepts/`, and recipes that combine modules with each other or with other libraries for one job go in `recipes/`.

1. Frontmatter: `title`, a noun phrase; `description`, one sentence on what you get, in words a TypeGPU beginner knows; `modules`, the modules the page covers.
2. One paragraph on the problem it solves, in words a TypeGPU beginner knows.
3. The figure, if any. Put the figure or the usage block before the first section.
4. The piece in use: one `ts twoslash` block, typechecked, 10 to 30 visible lines, setup above `// ---cut---`.
5. Order sections by use, with setup, per-frame work, edge cases and teardown. Put option and member facts in TSDoc for the generated card. Explain an option or member on the page only where its behaviour needs context, without duplicating its table.
6. A list holds at most five items and a bullet at most 40 words. Past that, name each rule in two or three bold words and wrap the list in `<Rules>` from `src/components/site/Rules.astro`, which sets the names in a column, or make it a table or a section.
7. Use `## Limits` only for constraints a reader must design around. Omit coverage notes, dated findings and project history. Delete the section if nothing remains.

## The module header

A page with `modules: [kit/frames]` gets a card generated by `src/contracts/` from package source. The card lists signatures, grouped exports, resource facts, the options that factories and the methods of their results take, and the members of what factories return. Exports group into GPU functions, accessors, outputs, schemas, constants, helpers and types. TSDoc tags provide facts the types cannot express. Put tags in the module's `@module` block or on a factory. The gate rejects tags elsewhere and checks them against the code:

- `@module` marks the block; its first sentence is the module's summary, or else its factory's
- `@kind <platform|tool|module|cpu|shader>` where the code can't tell, `@platform <web|native>` for one platform only, `@flat` for a subpath imported by name
- `@injects <hook|definition|member> [note]`, `@binds <n> storage[, <n> uniform]: <what>`
- `@requires <feature> <required|optional|option> [note]`, `@submits <setup|always> [note]`
- `@allocates <when>[: what]`, `@owns <items>[: note]`, `@records <note>`
- On a factory, use `@param <name|options.field> <words>` for positional inputs other than the root and options. Use `@returns <words>` to describe the result and require cleanup if it owns GPU resources. On any export or member of a returned type, `@hidden` omits it from the card.

Options tables come from types named `Options` or `…Options` declared in the module's files. A parameter of a factory or of a method of its result, or a field of another options type, must use that type. A member of a returned type shows the first sentence of its TSDoc. Every required field needs a TSDoc description. Every optional field needs a closing `Default <value>.` sentence.

`src/contracts/overrides.ts` provides vendored module facts. Its `flat` gathers the names a vendored subpath exports flat into modules. Correct a wrong card in source, tags or `src/contracts/`.

A module's strip shows its import, its main call, the facts that differ from a plain module and a link to its card. A single-module page shows the strip under the title. A page with several modules places each strip under its section heading using `<Module id="kit/bloom" />` from `src/components/site/Module.astro`, and the lead module's at the top of the body. The first listed module is the lead. Its index row opens the page, while other rows open their strips. The build requires each listed module exactly once.

Every page ends with its cards under a generated `Reference` heading, in the order `modules` lists them. In a card, a name in a call opens its definition: the result its members, `options` its fields, a module its page. Never write a `## Reference` section by hand.

[How to read the docs](src/content/docs/how-to-read.mdx) is the key to the strip, the card and their marks. A new fact or mark gets its sentence in `meanings` in `src/contracts/facets.ts`, which the card's tooltips and that page share.

## Components and figures

- Starlight asides (at most two or three a page), `<Steps>`, `<Tabs syncKey="platform">`, `<FileTree>`; `<LiveGrid keys={[…]} />` for live example cards; `<ModuleMap />` for the map.
- Put figures in `src/components/viz/Name.astro` with `<figure class="viz">`, plain controls and a caption that opens with what to try, after `<b class="viz-try">Try</b>`. Give a figure a `<p class="viz-title">` line when the picture alone doesn't say what it shows. Use `Slider.astro`, `Choice.astro` or buttons for controls, and a `<select>` only for options too many or too long for one row. Initialise with `each` from `./figure.ts` so failures appear in the figure.
- A figure that draws SVG builds it with `./svg.ts`: `fitWidth` redraws it when its width changes, `onStroke` drags its handles, `pointIn` finds a pointer in its units, and a drawing that takes arrow keys through `onKeys` carries `class="viz-keys"` for its focus ring. Clamp and ease with `clamp` and `smooth` from `../home/geometry.ts`, and write numbers with `rounded` and counts with `plural` from `./figure.ts`. It keeps its kit code in `Name.program.ts`, whose `createProgram()` runs with no arguments, on `recordingRoot()` when it makes GPU resources. `tests/figures.test.ts` runs every program. A figure states only what the kit's source makes true, so the program calls the kit's own functions. Where the mechanism runs only in a shader, the program mirrors its maths, a comment names the kit function it mirrors, and a test checks the mirror on a case with a known answer. Two figures that share a drawing keep it in one module beside them, as `timeline.ts` and `frame-strip.ts` do.
- A figure that shows pixels draws with WebGPU from `Name.scene.ts`, which exports `createScene(root, surface, options?)` as an example does, through `live` from `./figure.ts`; a lit 3D scene starts from `litStage` in `./stage.ts`. Pass `live` the module as `() => import('./Name.scene.ts')` and import nothing else from it in the figure's script, so the page fetches the scene only as the figure nears the screen. A scene that needs files also exports `load(loading)`, as an example's `assets.ts` does, which `live` calls only once the figure has WebGPU. `tests/figures.test.ts` and `figures.test.gpu.ts` run every scene on `recordingRoot()` and on dawn.node, with what its `load` reads. Never shade pixels on the CPU.
- `live` lets a scene go once its figure stays far from the screen, and builds it again as the figure comes back or after the device is lost. Wire controls once, before `live`, and change the scene through `shown.update`; give each scene what the reader set in the `ready` option. Pass `orbit: true` to turn the scene's `view` by dragging.
- `<ExampleFigure example="lattice" controls={['taa', 'view']} loupe caption="…" />` runs an example scene on the page. `controls` selects settings, actions and readouts, with readouts updated each frame. It adds orbit dragging, `pointing` input with a `reach` ring, strokes when pointing supports `drag`, and a link to the full example. `options` overrides card options; `values` sets initial settings; `aspect` sets the canvas ratio. `tests/example-figure.test.ts` builds every page figure and rejects control names the scene lacks.
- `Loupe.astro`, right after the canvas in a `.viz-screen`, magnifies a square of a live figure's canvas pixel for pixel, under the pointer or at `at`, to show aliasing, TAA or FXAA; `live` runs it.
- An animated figure steps on the page's one animation frame through `animation` from `./figure.ts`: it stops off screen and starts paused under `prefers-reduced-motion`. Everything stays legible at 360 px. The site is dark only.
- Styles use the tokens in `src/styles/tokens.css` and `viz.css`, never raw colours or sizes. Facet colours (`--f-*`) mean a facet, never decoration. What a call does with a name shows in its underline, by style and by a colour no facet has (`--r-*`): dotted grey borrows, plain yellow takes, double blue returns. A module it needs takes that facet's green.

## Look

- Use the home page's near-black ground, grey text and 1 px rules. Pages and figures have no filled panels, gradients, glows or shadows; the site's sidebar glow, header rail and footer horizon stay. Corners are at most 2 px.
- The accent's blue marks what is live, pointed at, selected or new in the step, with its ice at the brightest; text in the accent reads through `--link`. A figure's categories take the colours `viz.css` maps at its top, one meaning each, such as reads sky and writes amber. Red marks errors, and on a card or a strip an exception to a rule every other module keeps: it submits, it allocates after setup, or its API may still change.
- Headings use Aeonik 500 with −0.02em spacing. Labels use JetBrains Mono at 11 to 12 px, with a muted note. Join a label to its subject with a hairline and a small dot; the accent when active.
- Print area fills through the site's 8×8 Bayer dither or `--screen-door` mask, in cells 1 to 4 px across. Numbers use 1 px boxed badges. Show textures and buffers as labelled plates where it helps.
- Code has no fill and a 2 px left rule. Use the home page's draggable number for a value in code: dotted accent underline, horizontal drag and arrow keys.
- Hover never reflows. Transitions take 0.12 to 0.3 s. Nothing flickers or oscillates; reduced motion holds still. Keep every figure legible at 360 px, including its fallback without WebGPU.

## Process

- Every page shares the site components, route data, CSS and Astro and Expressive Code configuration. Build addresses through `src/components/site/urls.ts`. Add a page that isn't a module page to its group in `guides` in `nav.ts`. Module pages get rows from `modules` frontmatter. `areas` in `nav.ts` places every module in one labelled section of an area, also a module without a page; the build fails on a module no section places. Name areas and sections for what a program uses their modules for, never after a package. Only sidebar groups fold. Set `wide: true` for a full-width page, as the map does. A guide that reads badly without another page names it in `readFirst` frontmatter, with its slug as `page` and what the reader takes from it as `for`. Each section in `areas` names the guide that teaches the idea behind its modules as `concept`, which their pages link beside their label. A page ends with the pages beside it in its own group, so the order of a group in `nav.ts` is its reading order.
- `src/build/` checks the built site: `links.ts` fails the build on a broken link or anchor and on a page nothing links to, `parts.ts` on a part of Starlight's markup the site restyles that went missing, and `tables.ts` labels table cells for phones. After a Starlight upgrade, look at asides, tabs, steps, cards, file trees, badges and search results by eye.
- Keep dated findings and measurements out of the docs.
- Check with `pnpm build` (twoslash typechecks every block) and `pnpm test`. After changing `ec.config.mjs`, delete `node_modules/.astro/data-store.json` first.
- Look at what you changed at 1440 and 390 px wide on `pnpm preview`.
