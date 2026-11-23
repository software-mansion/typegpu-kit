# TypeGPU Kit native

An Expo app that runs the docs site's examples on an iPhone, straight from the `typegpu-kit-examples` package in `packages/typegpu-kit-examples`. Every example opens on the same screen. It hosts the scene with `useScene` on a worker runtime all examples share, through one worklet that can load any example on the package's list. The screen wires up the gestures, pads and corner texts the example declares, and shows its settings as native controls in a sheet. Expo Go can't load native WebGPU, so the app runs as a development build.

## Running it on an iPhone

Connect the iPhone over USB, with Developer Mode on. Build with Xcode 27.1 or later, whose iOS 27 SDK has the API names react-native-screens uses. Then, from the repository root:

```sh
pnpm install
cd apps/typegpu-kit-native
npx expo prebuild --platform ios --clean
pnpm ios
```

`pnpm ios` builds a Debug app, installs it on the phone you pick, and starts Metro, which serves the JavaScript. Keep the phone on the same network as the Mac. Set the `APPLE_TEAM_ID` environment variable to your Apple Developer team ID for signing. To build from Xcode instead, open `ios/TypeGPUKit.xcworkspace`, select the phone, and run; start Metro with `pnpm start`. `metro.config.js` serves whole bundles when React Native asks for lazy ones, so a worker runtime finds the scene it opens in the bundle instead of fetching it.

A Release build embeds the bundle and needs no Metro:

```sh
npx expo run:ios --device --configuration Release
```

Write `Release` with a capital R. Run `prebuild` again after changing `app.json`, `package.json` or the patches; `ios/` is generated and not committed.

## What you should see

- **Examples**, a gallery of posters under a banner of dithered smoke that lights up under your finger. A row of highlights comes first, then sections such as **Worlds** and **Games**. Tap the box on the banner to draw a random example's card, and tap the card to open it.
- Opening one shows its poster in dithered cells with the loading status, until a band of accent cells sweeps across and uncovers the first frame.
- Orbiting scenes turn with one finger and zoom with a pinch. Scenes that take strokes, such as falling sand, paint with one finger, and those that also orbit, such as SPH sculpt, turn and zoom with two; a tap presses where it lands.
- Examples with buttons show them as pads at the bottom, such as Blockfall's moves, rotations and drop, the platformer's run and jump, and the fish tank's feed. Games show their score in a corner.
- The bar chart button shows the GPU debug panel over the scene. Its Log and Trace buttons open the share sheet.
- **⋯** opens the settings sheet. It has sliders, switches and pickers for the scene's settings, buttons for its actions, photos and camera, and its readouts. Domino run lays out a photo from your library.
- **‹** returns to the gallery. An error replaces the scene with its message and a **Restart** button.

## Checks

```sh
pnpm test:types              # type-check the app
npx expo export --platform ios   # bundle with Metro, Babel and Bundle Mode
```
