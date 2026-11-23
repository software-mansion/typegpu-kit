import { defineRouteMiddleware } from '@astrojs/starlight/route-data';
import { REFERENCE_ID } from './components/site/modules.ts';

export const onRequest = defineRouteMiddleware((context) => {
  const route = context.locals.starlightRoute;

  // MarkdownContent.astro ends a module page with its cards under this heading
  if (route.toc && (route.entry.data.modules ?? []).length > 0) {
    route.toc.items.push({ depth: 2, slug: REFERENCE_ID, text: 'Reference', children: [] });
  }

  // Hide a table of contents that links only to the page title
  if (route.toc && route.toc.items.length <= 1) route.toc = undefined;
});
