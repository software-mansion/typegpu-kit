import { plain } from './dither.ts';

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

/**
 * The address of the still frame of the example `key` (`blockfall`) that its veil shows: sharp when
 * the reader turned the dither off, and otherwise small, for the veil to print in cells.
 */
export const posterOf = (key: string | undefined) =>
  at(key, plain ? 'poster.webp' : 'poster-small.webp');

/** The address of the small still frame of the example `key`, for cells and thumbnails. */
export const smallPosterOf = (key: string | undefined) => at(key, 'poster-small.webp');
