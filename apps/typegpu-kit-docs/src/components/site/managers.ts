/** The package managers install blocks offer, in tab order; `name` is the tab's label. */
export const MANAGERS = [
  { name: 'npm', add: 'npm i', dev: 'npm i -D', exec: 'npx', dlx: 'npx' },
  { name: 'pnpm', add: 'pnpm add', dev: 'pnpm add -D', exec: 'pnpm', dlx: 'pnpm dlx' },
  { name: 'yarn', add: 'yarn add', dev: 'yarn add -D', exec: 'yarn', dlx: 'yarn dlx' },
  { name: 'bun', add: 'bun add', dev: 'bun add -d', exec: 'bunx', dlx: 'bunx' },
] as const;

/** Where Starlight keeps the tab chosen for the sync key `pkg`. */
export const MANAGER_KEY = 'starlight-synced-tabs__pkg';

/**
 * What an install block runs: packages to add, dev packages, a binary of the project, a one-off,
 * an initializer with its arguments.
 */
export interface Commands {
  readonly add?: string | undefined;
  readonly dev?: string | undefined;
  readonly exec?: string | undefined;
  readonly dlx?: string | undefined;
  readonly create?: string | undefined;
}

/** The lines `manager` runs for `commands`. */
export function linesOf(manager: (typeof MANAGERS)[number], commands: Commands) {
  const { add, dev, exec, dlx, create } = commands;
  return [
    add && `${manager.add} ${add}`,
    dev && `${manager.dev} ${dev}`,
    exec && `${manager.exec} ${exec}`,
    dlx && `${manager.dlx} ${dlx}`,
    create && `${manager.name} create ${create}`,
  ].filter((line): line is string => Boolean(line));
}
