import { z } from 'astro/zod';

export const shellFields = {
  /** Widens the page beside the index to the width of a page without it, for a map or a grid. */
  wide: z.boolean().optional(),
  /** The page this one builds on, for a page that reads badly without it, and what it gives. */
  readFirst: z
    .object({
      /** Its slug: `concepts/how-the-kit-works`. */
      page: z.string(),
      /** What the reader takes from it, after a comma: `for the frame and the seven rules`. */
      for: z.string(),
    })
    .optional(),
};
