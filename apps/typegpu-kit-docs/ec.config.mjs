import ecTwoSlash from 'expressive-code-twoslash';
import ts from 'typescript';
import { instrument } from './src/styles/instrument.mjs';

/**
 * Exclude hover popups from search so excerpts show only the written code.
 * @type {import('@astrojs/starlight/expressive-code').ExpressiveCodePlugin}
 */
const searchSkipsPopups = {
  name: 'search-skips-popups',
  hooks: {
    postprocessRenderedBlock({ renderData }) {
      const elements = [renderData.blockAst];
      for (let element = elements.pop(); element; element = elements.pop()) {
        const classes = element.properties.className;
        if (Array.isArray(classes) && classes.includes('twoslash-popup-container')) {
          element.properties.dataPagefindIgnore = '';
          continue;
        }
        for (const child of element.children) if (child.type === 'element') elements.push(child);
      }
    },
  },
};

/** @type {import('@astrojs/starlight/expressive-code').StarlightExpressiveCodeOptions} */
export default {
  themes: [instrument],
  // Shell blocks too: a terminal frame keeps its title bar's height with nothing in it
  defaultProps: { frame: 'code' },
  styleOverrides: {
    borderRadius: '0px',
    borderColor: 'var(--rule)',
    codeFontFamily: 'var(--font-mono)',
    codeFontSize: '14px',
    codeLineHeight: '1.65',
    codePaddingBlock: '14px',
    codePaddingInline: '18px',
    uiFontFamily: 'var(--font-mono)',
    uiFontSize: '12px',
    focusBorder: 'var(--accent)',
    scrollbarThumbColor: 'hsl(214 16% 30% / 0.6)',
    scrollbarThumbHoverColor: 'hsl(214 16% 40% / 0.8)',
    frames: {
      frameBoxShadowCssValue: 'none',
      editorBackground: 'var(--ground)',
      editorTabBarBackground: 'var(--ground)',
      editorTabBarBorderColor: 'var(--rule)',
      editorTabBarBorderBottomColor: 'var(--rule)',
      editorActiveTabBackground: 'var(--ground)',
      editorActiveTabForeground: 'var(--text-2)',
      editorActiveTabBorderColor: 'transparent',
      editorActiveTabIndicatorTopColor: 'transparent',
      editorActiveTabIndicatorBottomColor: 'transparent',
      editorActiveTabIndicatorHeight: '1px',
      inlineButtonForeground: 'var(--text-2)',
      inlineButtonBorder: 'var(--rule-strong)',
      inlineButtonBackground: 'var(--ground)',
      tooltipSuccessBackground: 'var(--accent)',
      tooltipSuccessForeground: 'var(--ground)',
    },
    // LCH hues for the accent, the needs green and the exception coral; Expressive Code derives
    // the backgrounds and keeps the code legible on them
    textMarkers: {
      markHue: '285',
      insHue: '155',
      delHue: '30',
      defaultChroma: '20',
      defaultLuminance: '24%',
      backgroundOpacity: '38%',
    },
    twoSlash: {
      borderColor: 'var(--rule-strong)',
      background: 'var(--ground-3)',
      hoverUnderlineColor: 'var(--muted)',
      textColor: 'var(--text-2)',
      tagColor: 'var(--muted)',
      linkColor: 'var(--link)',
      linkColorHover: 'var(--accent-high)',
      linkColorVisited: 'var(--link)',
      linkColorActive: 'var(--accent-high)',
      errorColor: 'var(--f-exception)',
      warnColor: 'var(--f-requires)',
      suggestionColor: 'var(--f-needs)',
      messageColor: 'var(--f-records)',
      cursorColor: 'var(--text)',
      completionBoxBackground: 'var(--ground-3)',
      completionBoxBorder: 'var(--rule-strong)',
      completionBoxColor: 'var(--text-2)',
      completionBoxMatchedColor: 'var(--link)',
      completionBoxHoverBackground: 'var(--ground-4)',
    },
  },
  plugins: [
    ecTwoSlash({
      twoslashOptions: {
        compilerOptions: {
          strict: true,
          target: ts.ScriptTarget.ES2024,
          module: ts.ModuleKind.ESNext,
          moduleResolution: ts.ModuleResolutionKind.Bundler,
          allowImportingTsExtensions: true,
          noEmit: true,
          lib: [
            'lib.es2024.d.ts',
            'lib.dom.d.ts',
            'lib.dom.iterable.d.ts',
            'lib.esnext.disposable.d.ts',
          ],
          types: ['@webgpu/types'],
        },
      },
    }),
    searchSkipsPopups,
  ],
};
