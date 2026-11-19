import { glob } from 'astro/loaders';
import { docsSchema } from '@astrojs/starlight/schema';
import { defineCollection, reference } from 'astro:content';
import { z } from 'astro/zod';
import { shellFields } from './components/site/frontmatter.ts';
import { contractsLoader, moduleIds } from './contracts/loader.ts';
import { moduleContract } from './contracts/contract.ts';

const docs = glob({
  base: './src/content/docs',
  pattern: '**/[^_]*.{md,mdx}',
});

export const collections = {
  docs: defineCollection({
    loader: docs,
    schema: docsSchema({
      // The modules a page covers, as ids of the `modules` collection
      extend: z
        .object({
          modules: z
            .array(reference('modules'))
            .superRefine((listed, context) => {
              const known = moduleIds();
              listed.forEach(({ id }, i) => {
                if (!known.has(id))
                  context.addIssue({ code: 'custom', path: [i], message: `No module ${id}.` });
              });
            })
            .optional(),
        })
        .extend(shellFields),
    }),
  }),
  modules: defineCollection({ loader: contractsLoader(), schema: moduleContract }),
};
