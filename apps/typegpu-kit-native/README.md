# TypeGPU Kit native

An Expo app that runs the docs site's examples on an iPhone, straight from the `typegpu-kit-examples` package in `packages/typegpu-kit-examples`. A gallery opens any example on one screen, which hosts its scene on a shared worker runtime through `useScene` and one worklet that loads any example from the package's list. The screen connects the gestures, pads and corner texts the example declares, and its settings as native controls in a sheet. Expo Go can't load native WebGPU, so the app runs as a development build.

## Running it on an iPhone

Connect the iPhone over USB, with Developer Mode on. Build with Xcode 27.1 or later: react-native-screens uses an iOS 27 API that the 27.0 beta's SDK names differently. Then, from the repository root:

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

- **Examples**, a gallery of posters in sections such as **Worlds** and **Games**, below a banner of dithered smoke that lights under a finger.
- Opening one shows its poster printed in dithered cells, with loading text, until a front of accent cells lifts it off the first frame.
- Orbiting scenes turn with one finger and zoom with a pinch. Scenes that take strokes, such as falling sand, paint with one finger, and those that also orbit, such as SPH sculpt, turn and zoom with two; a tap presses where it lands.
- Games show their score in a corner and pads at the bottom: Blockfall moves and rotates pieces, the platformer runs and jumps.
- **≋** shows the GPU debug panel over the scene; its Log and Trace open the share sheet.
- **⋯** opens the settings sheet: sliders, switches and pickers that write the scene's settings, its actions, readouts and image pickers. Domino run lays out a photo from the library.
- **‹** returns to the gallery. An error replaces the scene with its message and a **Restart** button.

## Checks

```sh
pnpm test:types              # type-check the app
npx expo export --platform ios   # bundle with Metro, Babel and Bundle Mode
```
