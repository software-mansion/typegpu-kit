// React's hooks for one component, for tests that call a hook without a renderer. A render runs
// the component, then the effects whose dependencies changed: every cleanup first, then every
// effect, in order. A state update renders again in a microtask; unmounting runs every cleanup.

type Deps = readonly unknown[] | undefined;

interface Effect {
  deps: Deps | null;
  cleanup: (() => void) | undefined;
}

interface Due {
  readonly effect: Effect;
  readonly deps: Deps;
  readonly run: () => void | (() => void);
}

interface Instance {
  readonly hooks: unknown[];
  readonly effects: Effect[];
  readonly due: Due[];
  index: number;
  update(): void;
}

let rendering: Instance | undefined;

function hook<H>(make: (instance: Instance) => H): H {
  const instance = rendering;
  if (!instance) throw new Error('A hook was called outside a render.');

  const index = instance.index++;
  if (index === instance.hooks.length) instance.hooks.push(make(instance));
  return instance.hooks[index] as H;
}

// `null` stands for an effect or memo that never ran.
const changed = (previous: Deps | null, next: Deps) =>
  previous === null ||
  previous === undefined ||
  next === undefined ||
  previous.length !== next.length ||
  previous.some((value, i) => !Object.is(value, next[i]));

export function useState<T>(initial?: T | (() => T)) {
  const cell = hook((instance) => {
    const state = {
      value: typeof initial === 'function' ? (initial as () => T)() : initial,
      set: (next: T | ((previous: T) => T)) => {
        const value =
          typeof next === 'function' ? (next as (previous: T) => T)(state.value as T) : next;
        if (Object.is(value, state.value)) return;
        state.value = value;
        instance.update();
      },
    };
    return state;
  });

  return [cell.value, cell.set] as const;
}

export const useRef = <T>(initial: T) => hook(() => ({ current: initial }));

export function useMemo<T>(make: () => T, deps: Deps): T {
  const memo = hook(() => ({ deps: null as Deps | null, value: undefined as T | undefined }));

  if (changed(memo.deps, deps)) {
    memo.deps = deps;
    memo.value = make();
  }

  return memo.value as T;
}

export const useCallback = <F>(fn: F, deps: Deps): F => useMemo(() => fn, deps);

export function useEffect(run: () => void | (() => void), deps?: Deps) {
  const effect = hook((instance) => {
    const created: Effect = { deps: null, cleanup: undefined };
    instance.effects.push(created);
    return created;
  });

  if (changed(effect.deps, deps)) rendering!.due.push({ effect, deps, run });
}

export interface Mounted<Props, Result> {
  /** What the last render returned. */
  readonly current: Result;
  /** Renders again with new props, at once. */
  render(props: Props): void;
  unmount(): void;
}

/** Renders `component` once and keeps it mounted until `unmount`. */
export function mount<Props, Result>(
  component: (props: Props) => Result,
  props: Props,
): Mounted<Props, Result> {
  let current: Result;
  let mounted = true;
  let scheduled = false;

  const instance: Instance = {
    hooks: [],
    effects: [],
    due: [],
    index: 0,

    update() {
      if (scheduled || !mounted) return;
      scheduled = true;
      queueMicrotask(() => {
        scheduled = false;
        if (mounted) render(props);
      });
    },
  };

  const render = (next: Props) => {
    props = next;
    instance.index = 0;
    rendering = instance;
    try {
      current = component(props);
    } finally {
      rendering = undefined;
    }

    const due = instance.due.splice(0);
    for (const { effect } of due) {
      effect.cleanup?.();
      effect.cleanup = undefined;
    }
    for (const { effect, deps, run } of due) {
      effect.deps = deps;
      const cleanup = run();
      effect.cleanup = typeof cleanup === 'function' ? cleanup : undefined;
    }
  };

  render(props);

  return {
    get current() {
      return current;
    },

    render,

    unmount() {
      mounted = false;
      for (const effect of instance.effects) effect.cleanup?.();
    },
  };
}

/** An element as a plain record, for tests that read what a component returned. */
export interface Element {
  readonly type: unknown;
  readonly props: Record<string, unknown> & { readonly children: unknown[] };
}

export function createElement(
  type: unknown,
  props: Record<string, unknown> | null,
  ...children: unknown[]
): Element {
  return { type, props: { ...props, children } };
}
