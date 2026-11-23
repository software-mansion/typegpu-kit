const { getDefaultConfig } = require('expo/metro-config');
const { getBundleModeMetroConfig } = require('react-native-worklets/bundleMode');

const config = getBundleModeMetroConfig(getDefaultConfig(__dirname));

// The examples' models and HDR images and the kit's font atlas load as assets
config.resolver.assetExts.push('glb', 'hdr', 'rgba');

// Google's JavaScript Draco decoder requires `node:fs` only when it runs in Node
const resolve = config.resolver.resolveRequest;
config.resolver.resolveRequest = (context, name, platform) =>
  name === 'node:fs' && context.originModulePath.includes('/vendor/draco/')
    ? { type: 'empty' }
    : (resolve ?? context.resolveRequest)(context, name, platform);

// React Native asks for lazy bundles in development, which leave out modules reached only through
// `import()`. Worker runtimes load the same bundle and can't fetch the rest, so serve it whole.
const rewrite = config.server.rewriteRequestUrl ?? ((url) => url);
config.server.rewriteRequestUrl = (url) =>
  rewrite(url).replace(/([?&])lazy=true(&|$)/, '$1lazy=false$2');

module.exports = config;
