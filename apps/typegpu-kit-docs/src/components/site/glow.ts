const putOut = () => delete document.documentElement.dataset.glow;

export interface LightOptions {
  /** The entry the light sets out from. */
  before?: HTMLElement | null | undefined;
  /** The elements whose left borders the light draws as rules and runs along (theme.css hides them). */
  rules?: (() => Iterable<Element>) | undefined;
}

/** Resolves once the page's own scripts have run and it idles, so the light starts after them. */
const settled = () =>
  new Promise((resume) => {
    if ('requestIdleCallback' in window) requestIdleCallback(resume, { timeout: 1000 });
    else setTimeout(resume, 300);
  });

/** Lights `current()` inside `list` once the page idles. */
export function light(
  list: HTMLElement,
  current: () => HTMLElement | null,
  options: LightOptions = {},
) {
  if (!navigator.gpu) return;

  void settled()
    .then(() => import('./glow.follow.ts'))
    .then(({ follow }) => follow(list, current, options))
    .then(
      (lit) => {
        if (!lit) putOut();
      },
      (error: unknown) => {
        console.warn('The current entry keeps its plain marker:', error);
        putOut();
      },
    );
}
