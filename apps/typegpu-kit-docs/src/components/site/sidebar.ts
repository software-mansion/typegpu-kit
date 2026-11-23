import { getCollection } from 'astro:content';
import { buildIndex, rowsOf, type IndexGroup, type Page } from './nav.ts';

/** The index's groups, and every docs page as the index knows it. */
export async function loadIndex() {
  const [modules, docs] = await Promise.all([getCollection('modules'), getCollection('docs')]);
  const pages: Page[] = docs.map((entry) => ({
    id: entry.id === 'index' ? '' : entry.id,
    title: entry.data.title,
    description: entry.data.description,
    modules: entry.data.modules,
  }));

  return {
    groups: buildIndex(
      modules.map((entry) => entry.data),
      pages,
    ),
    pages,
  };
}

/** Whether `group` has a row for the page `current`. */
export const holds = (group: IndexGroup, current: string) =>
  rowsOf(group).some((link) => link.page === current);
