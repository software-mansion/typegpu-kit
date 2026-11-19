const posters = import.meta.glob<string>(
  '../../../../../packages/typegpu-kit-examples/src/*/poster.webp',
  {
    // As files, so the small ones aren't inlined into every page that names a poster
    query: '?url&no-inline',
    import: 'default',
    eager: true,
  },
);

/** The address of the still frame of the example `key` (`blockfall`), which its veil prints. */
export const posterOf = (key: string | undefined) =>
  key ? posters[`../../../../../packages/typegpu-kit-examples/src/${key}/poster.webp`] : undefined;
