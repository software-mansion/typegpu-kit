# typegpu-kit-pull

The animation that opens a box or a foil pack and reveals an example's card. The docs site's overlay and album in `apps/typegpu-kit-docs/src/components/` and the native app's chest in `apps/typegpu-kit-native/src/chest` both use it. The package is private and has no page or screen of its own:

- `timeline.ts` times a pull as pure functions of milliseconds, and moves a vessel, the box or the docs' foil pack, from its resting spot into the overlay.
- `box.ts` poses the box at rest and through a pull, and lays out its stage. `box.scene.ts` draws it.
- `card.ts` gives the card's finish by rarity, the seal on its back, its size and its pose through a flight.
- `foil.scene.ts` prints a card face's poster in cells under its holo foil.
- `vessel.ts` and `vessel.scene.ts` hold the frames and camera that the box and the docs' foil pack share.
- `dither.ts` holds the Bayer cells and greys that every canvas prints through.
- `rarity.ts` holds the rarity tiers each example's `meta.json` names and each example's count of lines of code from `lines.json`, and draws a random card.

`pnpm fix` writes `lines.json` with `scripts/lines.ts`, and the tests fail while it is stale.
