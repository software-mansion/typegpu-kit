// Keep these collection helpers independent of Astro so tests can import them

import { moduleById, type ModuleContract } from '../../contracts/contract.ts';
import type { Card } from '../../contracts/facets.ts';
import type { Line } from '../../contracts/signature.ts';
import { pageHref } from './urls.ts';

export { moduleById, type ModuleContract };

/** The page a module leads, by the kebab-case form of its id: `kit/mesh-pool`. */
export const pageLedBy = (module: ModuleContract) =>
  module.id.replace(/[A-Z]/g, (upper) => `-${upper.toLowerCase()}`);

/** A module as the site shows it: its contract, and the docs page that covers it. */
export type Module = ModuleContract & { readonly page?: string | undefined };

/** A docs page by its collection id (`kit/frames`), and the modules its frontmatter lists. */
export interface PageModules {
  readonly id: string;
  readonly modules?: readonly { readonly id: string }[] | undefined;
}

/** Modules that no page covers on purpose. Every other module needs a page that lists it. */
export const pageless: readonly string[] = ['kit/inject', 'kit/owner'];

/**
 * The contracts with the page that lists each. Throws on a page that lists an unknown module, a
 * module on two pages, and a module on none that `pageless` doesn't name. A page that lists modules
 * lives at its lead's `pageLedBy`, and `kit/` holds only such pages.
 */
export function withPages(
  contracts: readonly ModuleContract[],
  pages: readonly PageModules[],
): Module[] {
  const ids = new Set(contracts.map((contract) => contract.id));
  const pageOf = new Map<string, string>();

  for (const page of pages) {
    const lead = page.modules?.[0]?.id;
    if (lead === undefined && page.id.startsWith('kit/')) {
      throw new Error(`The page ${page.id} lists no modules, but kit/ holds only module pages.`);
    }
    const home = lead && ids.has(lead) ? pageLedBy(moduleById(contracts, lead)) : undefined;
    if (home !== undefined && page.id !== home) {
      throw new Error(
        `The page ${page.id} leads with ${lead}, so it belongs at ${home}.mdx; move it, or list its natural subject first.`,
      );
    }

    for (const { id } of page.modules ?? []) {
      const other = pageOf.get(id);
      if (!ids.has(id)) throw new Error(`The page ${page.id} lists ${id}, which is no module.`);
      if (other !== undefined) throw new Error(`${id} is on two pages, ${other} and ${page.id}.`);
      pageOf.set(id, page.id);
    }
  }

  for (const id of pageless) {
    if (!ids.has(id)) throw new Error(`pageless names ${id}, which is no module.`);
    if (pageOf.has(id)) throw new Error(`${id} is on ${pageOf.get(id)}; take it off pageless.`);
  }

  const missing = [...ids].filter((id) => !pageOf.has(id) && !pageless.includes(id));
  if (missing.length > 0) {
    throw new Error(
      `No page lists ${missing.join(', ')}: add each to a page's modules, or to pageless in modules.ts.`,
    );
  }

  return contracts.map((contract) => {
    const page = pageOf.get(contract.id);
    return page === undefined ? contract : { ...contract, page };
  });
}

/** The id of a module's strip on its page: where its row in the index and other pages' links land. */
export const anchorOf = (module: ModuleContract) => `module-${module.namespace}`;

/** The id of a module's card, under its page's Reference. */
export const referenceOf = (module: ModuleContract) => `reference-${module.namespace}`;

/** The id of a call on a module's card. */
export const callId = (module: ModuleContract, name: string) =>
  `${referenceOf(module)}-call-${name}`;

/** The id of an options type or a result on a module's card, or of one of its fields or members. */
export const typeId = (module: ModuleContract, type: string, field?: string) =>
  `${referenceOf(module)}-${type}${field === undefined ? '' : `-${field}`}`;

/**
 * Where a type name a card prints leads: its own options, results and schemas to their tables, and
 * another module's schema to its table there, as `namespace.Name` or, when no other module's
 * schema shares it, by its name alone.
 */
export function typeHrefs(module: ModuleContract, modules: readonly ModuleContract[], card: Card) {
  const hrefs = new Map<string, string>();
  const shared = new Map<string, number>();
  for (const other of modules) {
    for (const { name } of other.structs) shared.set(name, (shared.get(name) ?? 0) + 1);
  }
  for (const other of modules) {
    const page = other === module ? '' : hrefOf(modules, other)?.replace(/#.*$/, '');
    if (page === undefined) continue;
    for (const { name } of other.structs) {
      const href = `${page}#${typeId(other, name)}`;
      hrefs.set(`${other.namespace}.${name}`, href);
      if (shared.get(name) === 1) hrefs.set(name, href);
    }
  }
  for (const { name } of [...card.options, ...card.handles, ...module.structs]) {
    hrefs.set(name, `#${typeId(module, name)}`);
  }
  return hrefs;
}

/** The first word of a method or a member as a card writes it: `shadow` of `shadow.record`. */
const firstName = (text: string) => /^[\w$]+/.exec(text)?.[0];

/** Where a member of what a module's calls return sits on its card, by a method that names it. */
export function memberHref(module: ModuleContract, card: Card, method: string) {
  // `shadow.record` has a row of its own; a method without one leads to the member that holds it
  for (const name of [method.replace(/\(.*$/, ''), firstName(method)]) {
    const handle = card.handles.find((each) => each.members.some((member) => member.name === name));
    if (handle && name) return `#${typeId(module, handle.name, name)}`;
  }
  return undefined;
}

/**
 * Where each token of a call leads on its module's card: its result to that type's members, an
 * options object to its table, and an option to its row, or to its own table when it has one.
 */
export function tokenHrefs(module: ModuleContract, card: Card, line: Line) {
  const tables = card.options.filter((table) => table.takenBy.includes(line.name));
  const top = tables.filter((table) => !table.parameter.includes('.'));
  let braced = false;

  return line.tokens.map((token, at): string | undefined => {
    if (token.text.includes('{')) braced = true;
    if (token.text.includes('}')) braced = false;

    if (at > 0 && line.tokens[at - 1].text === ' → ') {
      const handle = card.handles.find(
        (each) =>
          each.from.includes(line.name) && new RegExp(`\\b${each.name}\\b`).test(token.text),
      );
      return handle && `#${typeId(module, handle.name)}`;
    }

    const name = token.text.replace(/\?$/, '');
    if (!/^[\w$]+$/.test(name)) return undefined;
    if (!braced) {
      const own = top.find((table) => table.parameter === name);
      return own && `#${typeId(module, own.name)}`;
    }
    for (const table of top) {
      if (!table.fields.some((field) => field.name === name)) continue;
      const nested = tables.find((each) => each.parameter === `${table.parameter}.${name}`);
      return `#${nested ? typeId(module, nested.name) : typeId(module, table.name, name)}`;
    }
    return undefined;
  });
}

/** The id of the section that ends a module page with its cards. */
export const REFERENCE_ID = 'reference';

/** The module whose row in the index is its page's own: the first its frontmatter lists. */
export const leadOf = (page: PageModules | undefined) => page?.modules?.[0]?.id;

/**
 * The ids a page's MDX places with `<Module id="…" />`, in order, outside code. Any other spelling
 * of the component throws, so a card is never dropped unseen.
 */
export function placedIn(body: string) {
  const prose = body.replace(/^(`{3,}|~{3,})[\s\S]*?^\1/gm, '').replace(/`[^`\n]*`/g, '');
  const tags = prose.match(/<Module\b[^>]*>/g) ?? [];

  return tags.map((tag) => {
    const id = /^<Module\s+id="([^"]+)"(?:\s+was="[^"]+")?\s*\/>$/.exec(tag)?.[1];
    if (!id) throw new Error(`Write a module's card as <Module id="kit/name" />, not ${tag}.`);
    return id;
  });
}

/**
 * Throws unless a page shows each module it lists exactly once: one module's strip under the title,
 * and several each where the page places it. Every card follows under the page's Reference.
 */
export function checkPlacement(page: string, listed: readonly string[], placed: readonly string[]) {
  const where = `The page ${page}`;

  if (listed.length < 2) {
    if (placed.length > 0) {
      throw new Error(
        `${where} places ${placed.join(', ')} with <Module>, but a page with one module shows its strip under the title and places none.`,
      );
    }
    return;
  }

  for (const id of placed) {
    if (!listed.includes(id))
      throw new Error(`${where} places ${id}, which its modules don't list.`);
  }
  for (const id of listed) {
    const count = placed.filter((each) => each === id).length;
    if (count !== 1) {
      throw new Error(
        `${where} places ${id} ${count === 0 ? 'nowhere' : `${count} times`}; give its section one <Module id="${id}" />.`,
      );
    }
  }
}

/** Where a module's page shows it: the page, and its own card when the page covers several. */
export function hrefOf(modules: readonly Module[], module: ModuleContract) {
  const page = modules.find((each) => each.id === module.id)?.page;
  if (page === undefined) return undefined;
  const shared = modules.filter((each) => each.page === page).length > 1;
  return `${pageHref(page)}${shared ? `#${anchorOf(module)}` : ''}`;
}

const byName = (a: ModuleContract, b: ModuleContract) =>
  a.namespace.localeCompare(b.namespace, 'en', { sensitivity: 'base' });

/**
 * What sets an example apart: the modules it uses that not every example does, the rarest first,
 * then by name. `examples` are the keys of every example.
 */
export function distinctOf(
  modules: readonly ModuleContract[],
  key: string,
  examples: readonly string[],
  count = 4,
) {
  const users = (module: ModuleContract) =>
    module.usedBy.filter((each) => examples.includes(each)).length;

  return modules
    .filter((module) => module.usedBy.includes(key) && users(module) < examples.length)
    .toSorted((a, b) => users(a) - users(b) || byName(a, b))
    .slice(0, count);
}

/** One relation for the map: `from` takes `to`; `fills` when `to` fills accessors in `from`'s pipelines. */
export interface Relation {
  readonly from: string;
  readonly to: string;
  readonly fills: boolean;
}

export const relationsOf = (modules: readonly ModuleContract[]): Relation[] =>
  modules.flatMap((module) =>
    module.needs.map((to) => ({
      from: module.id,
      to,
      fills: moduleById(modules, to).injects.length > 0,
    })),
  );
