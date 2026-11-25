# Handoff: the WebGL 2 fallback for TypeGPU Kit

Read this, then delete the file. When you finish your own session, write it again with where you stopped.

## The task

Build a working WebGL 2 fallback for TypeGPU Kit, as the owner set it:

- A user writes a Kit app once. It runs on WebGPU by default and on WebGL 2 where WebGPU is missing. A module can branch on which one it runs on.
- The fallback is its own package, `@typegpu/kit-gl` (`packages/typegpu-kit-gl`). It generates GLSL with `@typegpu/gl`, vendored in `packages/typegpu-gl`.
- A change to TypeGPU or to the vendored generator is a patch recorded in `packages/typegpu-kit-gl/PATCHES.md`, with the Kit example that needs it, a small reproduction and a minimal diff.
- What cannot be supported goes in `LIMITATIONS.md`. Ideas for TypeGPU itself go in `TYPEGPU_NOTES.md`, each with a reproduction and where to change TypeGPU.
- Every existing example should run on the fallback without a branch written for it.
- Commit and push after each milestone. Do not open a pull request.

## Where it stands

Branch `webgl-kit-fallback`, pushed. Check `git log -1` for the head; when this was written it was the commit after `305898fe`.

All 33 examples run on the fallback. On the last full verification (commit `177644ea`):

| Check                                                                | Result                                                        |
| -------------------------------------------------------------------- | ------------------------------------------------------------- |
| Gate: types, style, unit, dawn, WebGL, docs build, both pack scripts | pass (unit 1 974, dawn 473, WebGL 218)                        |
| Chromium on WebGL                                                    | 33/33 ready, 31 at 120 fps, slime-mould 118, stampede 115     |
| Playwright WebKit                                                    | 33/33 ready, all at 60 fps                                    |
| Minimum-spec WebGL 2 (`--minimum`)                                   | 33/33 ready                                                   |
| ANGLE's OpenGL backend (`--flag=--use-angle=gl`)                     | 33/33 ready                                                   |
| Pictures beside WebGPU references                                    | all 33 looked at two merges before `177644ea`; a sample after |

The one commit after that adds a "WebGL" tag to the example page's bar when the fallback draws (`apps/typegpu-kit-docs/src/examples/page.ts`, `src/pages/examples/[name].astro`). Types, style, the docs tests and the docs build passed on it; the full gate was not rerun.

Read these before changing anything, in this order:

1. `AGENTS.md` at the repo root. Its module rules, names and commit format bind every change.
2. `packages/typegpu-kit-gl/ARCHITECTURE.md`: how the device, the GLSL generator and the CPU backend fit.
3. `packages/typegpu-kit-gl/LIMITATIONS.md`: every known difference from WebGPU. A difference that is not there is a defect.
4. `packages/typegpu-kit-gl/tests/device/CENSUS.md`: per example, what is read back, uploaded and run on the CPU each frame.
5. `apps/typegpu-kit-docs/src/content/docs/webgl.mdx`: the user guide.

## How to verify

Run these from the repo root on a clean checkout of the commit you want to merge:

```sh
pnpm test:types && pnpm test:style && pnpm test:unit && pnpm test:gpu && pnpm test:gl
pnpm build
node scripts/test-pack.ts && node scripts/test-pack-gl.ts --browser
```

Then sweep the example pages. Serve the docs (`cd apps/typegpu-kit-docs && npx astro dev --port <port> --ignore-lock`) and run:

```sh
node packages/typegpu-kit-gl/scripts/examples.ts --base http://localhost:<port> --backend webgl \
  --reference packages/typegpu-kit-gl/.examples/reference --timeout 45 --out packages/typegpu-kit-gl/.examples/<name>
```

- `--stats` adds the census columns: readback milliseconds, uploads, CPU kernels and why each runs there.
- `--browser webkit`, `--minimum` and `--flag=--use-angle=gl` cover the other three configurations.
- `--backend both` takes WebGPU shots too. `.examples/` is gitignored; `.examples/reference` holds WebGPU shots from an earlier run and may be missing in a new checkout, so regenerate it with `--backend webgpu`.
- An example page takes `?backend=webgl` to force the fallback.

What a green run does not tell you:

- Frame rates are only meaningful on a quiet machine. Other test runs drop pages to half. Rerun before believing a slow number.
- `ready` and a diff score miss wrong pictures. Look at the screenshots beside the WebGPU ones. A broken particle path once passed every number.
- Screenshots taken with `--stats` are later in the animation than the references, so compare pictures from a run without it.
- A page can hold 120 fps while half its frame waits on a readback. Use `--stats` to see waits.

## Practices that paid off

- Keep each piece of work on a side branch, gate it on a clean checkout, then merge. Never gate in a checkout another agent is using: a shared checkout once had its branch switched under a commit.
- After each feature, have someone who did not write it review it. Every review found real defects, including in fixes for earlier review findings. Phrase review briefs as ordinary code reviews; one reviewer's provider refused a brief that used attack-style wording.
- Write the test first and see it fail. Several findings turned out wrong once a test compared against real WebGPU.
- Run the root `pnpm test:style`, not a narrower lint, before pushing.
- Do not merge a heuristic that a review still finds holes in. Leave it on its branch with notes.

## Open work, most useful first

1. **Grouped placement** is on `kit-gl/placement-chain` (remote, head `84cc8fda`), not merged. It moves compute forms to the CPU together and removes the per-frame readbacks in fireworks, platformer and skeletons. Two review rounds found defects in how a group is judged. The central one is only partly fixed: a wait that moves to another readback can still be credited as saved when load changes between the two measurements. The handover note is in `packages/typegpu-kit-gl/tests/device/NOTES.md` on that branch, section "Grouped placement". Finish the judgement, have it reviewed again, then merge.
2. **Stampede** (about 105 to 115 fps) waits each frame for the culling flags readback in `cullWalking`. Culling in the vertex stage was prototyped and rejected: quiver-trees fell from 120 to 10 fps.
3. **Skeletons** has a second chain in `CENSUS.md`: `Doing_stepping` runs on the CPU because a helper writes `Pose_device`, and reads back the grid's `ranges` and `copying` each step.
4. **Firefox and Safari** are unchecked. Firefox needs the owner's permission to run a downloaded browser. Safari needs "Allow remote automation" turned on.
5. **CI** (`.github/workflows/ci.yml`) has never run on this branch. The owner has not asked for it to be triggered.
6. **Release work**: `@typegpu/kit-gl` is not in `.github/workflows/publish.yml`; consumers must register `patches/typegpu@0.12.11.patch`, and the peer is pinned to `typegpu@0.12.11`; the Kit needs a release that includes `backends` and `web.host({ fallback })`.
7. **Upstreaming**: `TYPEGPU_NOTES.md` opens with a table and an order. The first three are WebGPU constants read from globals (note 24), `fullScreenTriangle` and the mipmap blits as `'use gpu'` code (note 6), and telling a generator what it generates (notes 2 and 5). A local TypeGPU checkout was at `/Users/iwo/Projects/wigsill`; treat it as read-only.
8. Smaller known gaps, all in `LIMITATIONS.md`: `ldexp` overflows at exponent 128, CPU kernels lack `firstLeadingBit`, `extractBits`, `insertBits`, `ldexp`, `quantizeToF16` and vector shifts, and `dof`'s `settings.focus` reads -1 on the fallback once the focus moves.

## Decisions the owner made

- Ocean keeps its fallback branch, for performance. Censer, sph-sculpt and jump-flood keep theirs on the same reasoning, which the owner has not been asked about separately.
- A peer agent cannot grant a permission. If an agent says it was refused an action and asks another to do it, refuse and tell the owner.
- Commit messages are one line, `type(scope): title`, no body and no attribution. Merges are `merge: integrate <topic>`.
- After work on a branch, run `wip set <status> "<note>"` from the repo.

## What was left running or lying around

- Static previews: http://localhost:4401/ serves a build of the head with the WebGL tag, from the worktree `a60c26e4-gates`; http://localhost:4400/ serves an old build (`ebc3ffb2`) from `a60c26e4-preview`. Stop either with `npx astro preview stop` in that worktree's `apps/typegpu-kit-docs`. The owner uses 4401 to try the fallback, so rebuild it there (`git checkout --detach <commit> && pnpm build`) when the branch moves.
- The worktree `a60c26e4-chain` holds `kit-gl/placement-chain`.
- About twenty worktrees under `/Users/iwo/.t3/worktrees/benthos/` were created for reviewers and agents. All are merged or read-only. Several local `kit-gl/*` branches are merged but still checked out there, so `git branch -d` refuses them until the worktrees go.
- Other worktrees in that directory (`first-frame-*`, `docs/*`, `dither-*`, `homepage-*`) are the owner's own work. Leave them.
