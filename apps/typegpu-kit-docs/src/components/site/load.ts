import { getCollection } from 'astro:content';
import { moduleById, withPages, type Module } from './modules.ts';

/** Public modules in contract order, with the page that lists each module. */
export async function loadModules(): Promise<Module[]> {
  const [modules, docs] = await Promise.all([getCollection('modules'), getCollection('docs')]);
  return withPages(
    modules.map((entry) => entry.data),
    docs.map((entry) => ({ id: entry.id, modules: entry.data.modules })),
  );
}

/** The modules a page's frontmatter lists, in order, and every module, which their links need. */
export async function loadListed(listing: { modules?: readonly { id: string }[] | undefined }) {
  const ids = (listing.modules ?? []).map((each) => each.id);
  const modules = ids.length > 0 ? await loadModules() : [];
  return { modules, listed: ids.map((id) => moduleById(modules, id)) };
}
