# typegpu-kit-pull

The pull from the docs site's box, shared by the docs site's overlay and album in `apps/typegpu-kit-docs/src/components/` and the native app's chest in `apps/typegpu-kit-native/src/chest`. It is private, and holds what both draw and time alike, with no page or screen of its own:

- `timeline.ts` gives a pull as pure functions of milliseconds, and a vessel's way from its resting spot into the overlay.
- `box.ts` poses the box at rest and through a pull, and lays out its stage. `box.scene.ts` draws it.
- `card.ts` gives the card's finish by rarity, the seal on its back, its size and its pose through a flight.
- `foil.scene.ts` prints a card face's poster in cells under its holo foil.
- `vessel.ts` and `vessel.scene.ts` hold the frames and camera that the box and the docs' foil pack share.
- `dither.ts` holds the Bayer cells and greys that every canvas prints through.
- `rarity.ts` sets the examples' rarities from their lines of code in `lines.json`, and draws a card.

`pnpm fix` writes `lines.json` with `scripts/lines.ts`, and the tests fail while it is stale.
