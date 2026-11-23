// A phone stacks a table's rows (theme.css), so each body cell of a Markdown table carries its
// column's heading as `data-label`, and a table whose first column names a group only on the
// group's first row is marked `data-grouped`.

import type { SatteriProcessorOptions } from '@astrojs/markdown-satteri';

// Sätteri's hast types, by way of the processor's options: `satteri` itself isn't a dependency.
type HastPlugin = Exclude<
  NonNullable<SatteriProcessorOptions['hastPlugins']>[number],
  ((...args: never[]) => unknown) | readonly unknown[] | false | null | undefined
>;
type ElementVisitor = Extract<NonNullable<HastPlugin['element']>, { filter: string[] }>;
type Element = Parameters<ElementVisitor['visit']>[0];

const elements = (node: Element | undefined, tagName: string) =>
  (node?.children ?? []).filter(
    (child): child is Element => child.type === 'element' && child.tagName === tagName,
  );

/** The Sätteri plugin, for `markdown.processor`. */
export const tableLabels: HastPlugin = {
  name: 'typegpu-kit-table-labels',
  element: {
    filter: ['table'],
    visit(table, ctx) {
      const [headRow] = elements(elements(table, 'thead')[0], 'tr');
      if (!headRow) return;

      const labels = elements(headRow, 'th').map((cell) => ctx.textContent(cell).trim());
      const rows = elements(table, 'tbody').flatMap((body) => elements(body, 'tr'));
      let grouped = false;

      for (const row of rows) {
        elements(row, 'td').forEach((cell, column) => {
          const label = labels[column];
          if (label) ctx.setProperty(cell, 'data-label', label);
          if (column === 0 && cell.children.length === 0) grouped = true;
        });
      }

      if (grouped) ctx.setProperty(table, 'data-grouped', '');
    },
  },
};
