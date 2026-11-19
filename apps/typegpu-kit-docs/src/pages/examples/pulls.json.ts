// The cards a pull draws from, fetched when a page first opens a box or a pack.
import { loadEntries } from '../../components/pull/entries.ts';

export async function GET() {
  return Response.json(await loadEntries());
}
