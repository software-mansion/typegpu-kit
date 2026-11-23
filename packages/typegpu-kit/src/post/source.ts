import type { SizeSource } from '../targets.ts';
import type { Rendered, Source } from './post.ts';

/**
 * Reads an effect's `source` and `region` options as one rendered picture, the `region` option
 * before a rendered source's own. Call it once at creation, then take `output` and `region` once at
 * every recording, in that order.
 */
export function sourceOf(options: { source: Source; region?: SizeSource | undefined }): Rendered {
  const { source, region } = options;
  if (typeof source === 'function') {
    return {
      get output() {
        return source();
      },
      region,
    };
  }
  if (region === undefined) return source;
  return {
    get output() {
      return source.output;
    },
    region,
  };
}
