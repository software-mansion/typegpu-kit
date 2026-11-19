// @ts-check
import { satteri } from '@astrojs/markdown-satteri';
import starlight from '@astrojs/starlight';
import basicSsl from '@vitejs/plugin-basic-ssl';
import { defineConfig } from 'astro/config';
import typegpu from 'unplugin-typegpu/vite';
import { baseLinks, linksIntegration } from './src/build/links.ts';
import { markdownIntegration } from './src/build/markdown.ts';
import { moved } from './src/build/moved.ts';
import { partsIntegration } from './src/build/parts.ts';
import { tableLabels } from './src/build/tables.ts';
import { termMarks } from './src/build/terms.ts';

const site = (/** @type {string} */ name) => `./src/components/site/${name}.astro`;

// The release workflow builds for GitHub Pages: DOCS_SITE=https://docs.swmansion.com, DOCS_BASE=/typegpu-kit/
const base = (process.env.DOCS_BASE ?? '/').replace(/\/?$/, '/');

export default defineConfig({
  ...(process.env.DOCS_SITE && { site: process.env.DOCS_SITE }),
  base,
  scopedStyleStrategy: 'where',
  // Astro sends a redirect to its destination as written, without the base
  redirects: Object.fromEntries(
    Object.entries(moved).map(([from, to]) => [`/${from}`, `${base}${to}`]),
  ),
  // Examples use top-level await
  // WebGPU requires HTTPS off localhost; dev:host uses a self-signed certificate
  vite: {
    plugins: [typegpu(), process.env.DEV_HOST ? basicSsl() : null],
    build: { target: 'esnext' },
  },
  markdown: { processor: satteri({ hastPlugins: [tableLabels, termMarks(), baseLinks(base)] }) },
  integrations: [
    starlight({
      title: 'TypeGPU Kit',
      description: 'GPU modules built on TypeGPU that record into your command encoder',
      favicon: '/favicon.svg',
      customCss: ['./src/styles/theme.css', './src/styles/viz.css'],
      routeMiddleware: './src/routeData.ts',
      components: {
        Head: site('Head'),
        Header: site('Header'),
        Sidebar: site('Sidebar'),
        MobileMenuToggle: site('MobileMenuToggle'),
        PageTitle: site('PageTitle'),
        MarkdownContent: site('MarkdownContent'),
        PageSidebar: site('PageSidebar'),
        Footer: site('Footer'),
        Hero: site('Hero'),
        ThemeProvider: site('Nothing'),
      },
    }),
    markdownIntegration(),
    linksIntegration(),
    partsIntegration(),
  ],
});
