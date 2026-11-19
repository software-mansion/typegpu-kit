// The reader's album: the examples they have found as cards, kept in this browser. Imports
// nothing, so every page can read it.

const KEY = 'typegpu-kit:album';
const CHANGED = 'album:changed';

/** One card in the album: when it was found, and the page it was found on. */
interface Found {
  readonly at: number;
  readonly source: string;
}

interface AlbumState {
  readonly version: 1;
  readonly found: Readonly<Record<string, Found>>;
}

const EMPTY: AlbumState = { version: 1, found: {} };

/** The album as stored, or empty when there is none or storage is closed. */
function readAlbum(): AlbumState {
  try {
    const stored = JSON.parse(localStorage.getItem(KEY) ?? 'null') as Partial<AlbumState> | null;
    if (stored?.version !== 1 || typeof stored.found !== 'object' || !stored.found) return EMPTY;
    return { version: 1, found: stored.found };
  } catch {
    return EMPTY;
  }
}

/** What this page last wrote when storage refused it, so the page stays consistent. */
let memory: AlbumState | undefined;
const current = () => memory ?? readAlbum();

const write = (next: AlbumState) => {
  try {
    localStorage.setItem(KEY, JSON.stringify(next));
    memory = undefined;
  } catch {
    // The page keeps the find; the next page will not
    memory = next;
  }
  dispatchEvent(new CustomEvent(CHANGED));
};

/** The cards in the album, by example key. */
export const foundCards = () => current().found;

export const has = (key: string) => key in current().found;

export const count = () => Object.keys(current().found).length;

/** Adds `key`, found on `source`; a card already there stays as it was. Returns whether it is new. */
export function addCard(key: string, source: string) {
  const album = current();
  if (key in album.found) return false;
  write({ ...album, found: { ...album.found, [key]: { at: Date.now(), source } } });
  return true;
}

/** Calls `listener` whenever the album changes here, in another tab, or on a return to the page. */
export function onAlbum(listener: () => void) {
  addEventListener(CHANGED, listener);
  addEventListener('storage', (event) => {
    if (event.key === KEY) listener();
  });
  addEventListener('pageshow', (event) => {
    if (event.persisted) listener();
  });
}
