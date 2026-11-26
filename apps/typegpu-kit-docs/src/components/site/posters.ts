const posters = import.meta.glob<string>(
  '../../../../../packages/typegpu-kit-examples/src/*/poster{,-small}.webp',
  {
    // As files, so the small ones aren't inlined into every page that names a poster
    query: '?url&no-inline',
    import: 'default',
    eager: true,
  },
);

const at = (key: string | undefined, name: string) =>
  key ? posters[`../../../../../packages/typegpu-kit-examples/src/${key}/${name}`] : undefined;

/** The address of the sharp still frame of the example `key` (`blockfall`) that its veil shows. */
export const posterOf = (key: string | undefined) => at(key, 'poster.webp');

/** The address of the small still frame of the example `key`, for cells and thumbnails. */
export const smallPosterOf = (key: string | undefined) => at(key, 'poster-small.webp');
