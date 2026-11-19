import { categories as sections, examples, type Example } from 'typegpu-kit-examples';

export interface Category {
  readonly name: string;
  readonly title: string;
  readonly examples: readonly Example[];
}

export const categories: readonly Category[] = sections
  .map(({ name, title }) => ({
    name,
    title,
    examples: examples.filter((example) => example.category === name),
  }))
  .filter((category) => category.examples.length > 0);
