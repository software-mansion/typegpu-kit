/** @type {import('react-native-worklets/plugin').PluginOptions} */
const workletsPluginOptions = {
  bundleMode: true,
  strictGlobal: true,
  importForwarding: {
    // Packages that worklets import from, loaded again on the runtime that runs them: the kit, the
    // examples package whose list loads the opened scene there, and the pull's box and card.
    moduleNames: ['typegpu', '@typegpu/kit', 'typegpu-kit-examples', 'typegpu-kit-pull'],
    // Files whose worklets import relatively, matched by path segments: the kit's and the
    // examples' workspace sources, which no module name matches, this app's worklets, its chrome's
    // worklets, the chest's scenes and overlay, and its touch lane, whose routers live in a module
    // the UI runtime loads again.
    relativePaths: [
      'packages/typegpu-kit/src',
      'packages/typegpu-kit-examples/src',
      'apps/typegpu-kit-native/src/worklets.ts',
      'apps/typegpu-kit-native/src/chrome/worklets.ts',
      'apps/typegpu-kit-native/src/chest',
      'apps/typegpu-kit-native/src/touch.ts',
    ],
  },
};

module.exports = function (api) {
  api.cache(true);
  return {
    presets: [['babel-preset-expo', { worklets: false, reanimated: false }]],
    plugins: ['unplugin-typegpu/babel', ['react-native-worklets/plugin', workletsPluginOptions]],
  };
};
