// A marked term (build/terms.ts) opens its popover on a click, a tap, Enter or Space, and closes it
// on Escape or a click elsewhere, all through the Popover API. This opens it under a mouse that
// rests on the term too, and places it where CSS can't anchor it.

/** How long a mouse rests on a term before its popover opens, in ms. */
const REST = 150;
/** How long a mouse has to cross from a term to its popover before the popover closes, in ms. */
const CROSS = 120;

const termOf = (popover: Element) =>
  document.querySelector<HTMLElement>(`.term[popovertarget="${popover.id}"]`);
const popoverOf = (term: Element) =>
  document.getElementById(term.getAttribute('popovertarget') ?? '');
const isOpen = (popover: Element) => popover.matches(':popover-open');

/** The popover a resting mouse opened, which closes once the mouse leaves it and its term. */
let pointed: HTMLElement | null = null;
let timer = 0;

const after = (ms: number, run: () => void) => {
  clearTimeout(timer);
  timer = window.setTimeout(run, ms);
};

const leave = () => {
  if (!pointed) return clearTimeout(timer);
  after(CROSS, () => {
    if (pointed && isOpen(pointed)) pointed.hidePopover();
    pointed = null;
  });
};

function point(event: PointerEvent) {
  if (event.pointerType !== 'mouse') return;
  const at =
    event.target instanceof Element ? event.target.closest<HTMLElement>('.term, .term-pop') : null;
  const popover = at?.matches('.term') ? popoverOf(at) : at;

  if (popover && popover === pointed) return clearTimeout(timer);
  if (!popover || !at?.matches('.term') || isOpen(popover)) return leave();

  after(REST, () => {
    if (isOpen(popover)) return;
    popover.showPopover();
    pointed = popover;
  });
}

if ('showPopover' in HTMLElement.prototype) {
  document.addEventListener('pointerover', point);
  document.documentElement.addEventListener('pointerleave', leave);

  // A click on a term whose popover the mouse opened keeps it open, as a click on a closed one does
  document.addEventListener('click', (event) => {
    const term = event.target instanceof Element ? event.target.closest('.term') : null;
    if (!term) return;
    clearTimeout(timer);
    if (pointed && popoverOf(term) === pointed) {
      event.preventDefault();
      pointed = null;
    }
  });

  // `toggle` doesn't bubble
  document.addEventListener(
    'toggle',
    (event) => {
      if (event.target === pointed && (event as ToggleEvent).newState === 'closed') pointed = null;
    },
    true,
  );
}

// Without CSS anchor positioning a popover opens in the middle of the window, so it is placed here:
// under its term, or above it where there is no room below, and inside the window's edges
if (!CSS.supports('top: anchor(bottom)')) {
  const GAP = 4;
  const EDGE = 16;

  const place = (popover: HTMLElement) => {
    const term = termOf(popover);
    if (!term) return;
    const box = term.getBoundingClientRect();
    // Measured before it opens, so it never shows in the middle first
    popover.style.display = 'block';
    const { offsetWidth: width, offsetHeight: height } = popover;
    popover.style.display = '';

    const view = document.documentElement;
    const below = box.bottom + GAP;
    const fits = below + height <= view.clientHeight - EDGE || box.top - GAP - height < 0;
    Object.assign(popover.style, {
      inset: 'auto',
      margin: '0',
      top: `${fits ? below : box.top - GAP - height}px`,
      left: `${Math.max(EDGE, Math.min(box.left, view.clientWidth - width - EDGE))}px`,
    });
  };
  const placeOpen = () => {
    for (const popover of document.querySelectorAll<HTMLElement>('.term-pop:popover-open')) {
      place(popover);
    }
  };

  document.addEventListener(
    'beforetoggle',
    (event) => {
      const popover = event.target;
      if (!(popover instanceof HTMLElement) || !popover.matches('.term-pop')) return;
      if ((event as ToggleEvent).newState === 'open') place(popover);
    },
    true,
  );
  addEventListener('scroll', placeOpen, { passive: true, capture: true });
  addEventListener('resize', placeOpen);
}
