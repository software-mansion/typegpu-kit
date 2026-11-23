# Contributing to TypeGPU Kit

## Reporting issues and suggesting features

Report bugs in [Issues](https://github.com/software-mansion/typegpu-kit/issues/new).
For a feature idea, join an [existing discussion](https://github.com/software-mansion/typegpu-kit/discussions)
or [start a discussion](https://github.com/software-mansion/typegpu-kit/discussions/new/choose).

## Development

1. Fork the repository and create a branch from `main`.
2. Install Node.js 24.2.0 or later and pnpm 11.18.0, then run `pnpm install`.
3. Run `pnpm dev` and open the docs and examples in a WebGPU browser.
4. Read [AGENTS.md](AGENTS.md) for the full contributor rules. Write each commit message as
   one line of at most 72 characters in the form `type(scope): title` or `type: title`.
   Start the title with a lowercase verb in the imperative, keep proper names' spelling,
   and omit the final period, body and trailers.
5. Run `pnpm test` and make sure all checks pass.
6. Open a [pull request](https://github.com/software-mansion/typegpu-kit/pulls) for review.
