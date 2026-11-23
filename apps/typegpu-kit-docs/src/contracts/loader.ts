import type { Loader } from 'astro/loaders';
import { extractContracts, repo, type Extraction } from './extract.ts';

/** How long dev waits after the last change before it reads the contracts again. */
const SETTLE_MS = 150;

let latest: Extraction | undefined;

/** The contracts as last read, read now if they haven't been. */
const current = () => (latest ??= extractContracts());

/**
 * The ids of every public module, for the docs schema: Astro only logs a reference to an entry that
 * doesn't exist, and a page that lists an unknown module should fail the sync.
 */
export const moduleIds = () => new Set(current().contracts.map((contract) => contract.id));

/** Load public module contracts keyed by id. Build and sync reject mismatches with the code. Development warns and reloads after package or example changes. */
export function contractsLoader(): Loader {
  return {
    name: 'typegpu-kit-contracts',

    async load({ store, parseData, generateDigest, logger, watcher }) {
      const sync = async (extraction: Extraction) => {
        latest = extraction;
        const { contracts, issues } = extraction;

        if (issues.length > 0 && !watcher) {
          const list = issues.map((issue) => `- ${issue}`).join('\n');
          throw new Error(`The module contracts disagree with the code:\n${list}`);
        }
        for (const issue of issues) logger.warn(issue);

        const ids = new Set(contracts.map((contract) => contract.id));
        for (const id of store.keys()) if (!ids.has(id)) store.delete(id);

        for (const contract of contracts) {
          const data = await parseData({ id: contract.id, data: { ...contract } });
          store.set({ id: contract.id, data, digest: generateDigest(data) });
        }
      };

      await sync(current());
      if (!watcher) return;

      let pending: ReturnType<typeof setTimeout> | undefined;
      const changed = (path: string) => {
        const relevant =
          (path.startsWith(`${repo.package}/`) && /\/(src\/.*\.ts|package\.json)$/.test(path)) ||
          ([repo.examples, repo.pages].some((dir) => dir && path.startsWith(dir)) &&
            /\.(ts|json)$/.test(path));
        if (!relevant) return;

        clearTimeout(pending);
        pending = setTimeout(() => {
          Promise.resolve()
            .then(() => sync(extractContracts(repo, latest)))
            .catch((error: unknown) => logger.error(String(error)));
        }, SETTLE_MS);
      };

      watcher.add([repo.package, repo.examples, ...(repo.pages ? [repo.pages] : [])]);
      for (const event of ['change', 'add', 'unlink'] as const) watcher.on(event, changed);
    },
  };
}
