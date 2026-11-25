/**
 * The order of a compute kernel's atomic loads and additions, read from the first resolution's
 * code. TypeGPU generates a helper in the middle of the statement that first calls it, so the
 * generator cannot follow program order as it goes; this reads each finished function instead.
 *
 * The first resolution writes an addition to a summed buffer as `kit_ad__<id>()` and an atomic
 * load as `kit_ld__<id>(<place>)`. A function's code is parsed into statements, `if` arms, loops and
 * returns; a call to another function of the kernel applies that function's summary where its
 * arguments end. Whatever this cannot order, it orders the cautious way: a ternary's or a
 * `switch`'s arms count as if each ran one after another, and a loop's body as if it ran twice.
 * Inside a `switch` or a loop, where `break` and `continue` can leave a path before a `return`
 * ends it, a `return` ends nothing: every way out of them carries no more than the whole body.
 */

/** Marks an addition to a summed buffer in the first resolution: `kit_ad__<id>()`. */
export const ADD_MARK = 'kit_ad__';
/** Marks an atomic load in the first resolution: `kit_ld__<id>(<place>)`. */
export const LOAD_MARK = 'kit_ld__';

export interface Flow {
  /** Buffers some load of which may follow an addition to them in the same invocation. */
  readonly late: ReadonlySet<string>;
  /** The most additions to each buffer one path through the kernel makes. */
  readonly most: ReadonlyMap<string, number>;
}

interface Summary {
  /** Buffers the function may load, anywhere. */
  readonly loads: ReadonlySet<string>;
  /** Buffers the function may add to before it returns. */
  readonly adds: ReadonlySet<string>;
  /** The most additions to each buffer one path through the function makes. */
  readonly most: ReadonlyMap<string, number>;
  readonly late: ReadonlySet<string>;
}

type Event =
  | { readonly kind: 'load'; readonly id: string }
  | { readonly kind: 'add'; readonly id: string }
  | { readonly kind: 'call'; readonly name: string };

type Statement =
  | { readonly kind: 'events'; readonly events: readonly Event[] }
  | { readonly kind: 'return'; readonly events: readonly Event[] }
  | { readonly kind: 'block'; readonly body: readonly Statement[] }
  | {
      readonly kind: 'if';
      readonly events: readonly Event[];
      readonly yes: Statement;
      readonly otherwise: Statement | undefined;
    }
  | { readonly kind: 'loop'; readonly events: readonly Event[]; readonly body: Statement }
  | { readonly kind: 'switch'; readonly events: readonly Event[]; readonly body: Statement };

interface State {
  readonly added: Set<string>;
  readonly counts: Map<string, number>;
}

const TOKENS = /\/\/[^\n]*|[A-Za-z_]\w*|\d[\w.]*|\S/g;

function tokensOf(code: string): string[] {
  return (code.match(TOKENS) ?? []).filter((token) => !token.startsWith('//'));
}

/** The index of the bracket that closes the one at `from`. */
function closing(tokens: readonly string[], from: number): number {
  const open = tokens[from] as string;
  const close = open === '(' ? ')' : open === '{' ? '}' : ']';
  let depth = 0;
  for (let i = from; i < tokens.length; i++) {
    if (tokens[i] === open) depth++;
    if (tokens[i] === close && --depth === 0) return i;
  }
  throw new Error('@typegpu/kit-gl: unbalanced code in a compute form');
}

/** Loads, additions and calls in `tokens` in the order they run: a call once its arguments ran. */
function eventsOf(tokens: readonly string[], functions: ReadonlySet<string>): Event[] {
  const events: Event[] = [];
  const pending: (Event | undefined)[] = [];
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i] as string;
    if (token === '(') {
      const name = tokens[i - 1] ?? '';
      pending.push(
        name.startsWith(ADD_MARK)
          ? { kind: 'add', id: name.slice(ADD_MARK.length) }
          : name.startsWith(LOAD_MARK)
            ? { kind: 'load', id: name.slice(LOAD_MARK.length) }
            : functions.has(name)
              ? { kind: 'call', name }
              : undefined,
      );
    } else if (token === ')') {
      const event = pending.pop();
      if (event) events.push(event);
    }
  }
  return events;
}

function parseStatement(
  tokens: readonly string[],
  at: number,
  functions: ReadonlySet<string>,
): [Statement, number] {
  const token = tokens[at];
  if (token === '{') {
    const end = closing(tokens, at);
    const body: Statement[] = [];
    for (let i = at + 1; i < end;) {
      const [statement, next] = parseStatement(tokens, i, functions);
      body.push(statement);
      i = next;
    }
    return [{ kind: 'block', body }, end + 1];
  }
  if (token === 'if' || token === 'for' || token === 'while' || token === 'switch') {
    const end = closing(tokens, at + 1);
    const events = eventsOf(tokens.slice(at + 1, end + 1), functions);
    const [body, next] = parseStatement(tokens, end + 1, functions);
    if (token === 'if') {
      if (tokens[next] === 'else') {
        const [otherwise, after] = parseStatement(tokens, next + 1, functions);
        return [{ kind: 'if', events, yes: body, otherwise }, after];
      }
      return [{ kind: 'if', events, yes: body, otherwise: undefined }, next];
    }
    if (token === 'switch') return [{ kind: 'switch', events, body }, next];
    return [{ kind: 'loop', events, body }, next];
  }
  if (token === 'do') {
    const [body, next] = parseStatement(tokens, at + 1, functions);
    const end = closing(tokens, next + 1);
    const events = eventsOf(tokens.slice(next + 1, end + 1), functions);
    return [{ kind: 'loop', events, body }, end + 2];
  }
  if (token === 'else') throw new Error('@typegpu/kit-gl: an else without an if in a compute form');
  if (token === 'case' || token === 'default') {
    let i = at;
    while (i < tokens.length && tokens[i] !== ':') i++;
    return [{ kind: 'events', events: [] }, i + 1];
  }
  // An expression or declaration, up to its `;`
  let i = at;
  while (i < tokens.length && tokens[i] !== ';') {
    i =
      tokens[i] === '(' || tokens[i] === '[' || tokens[i] === '{' ? closing(tokens, i) + 1 : i + 1;
  }
  const events = eventsOf(tokens.slice(at, i), functions);
  return [{ kind: token === 'return' ? 'return' : 'events', events }, i + 1];
}

/** Runs a statement from `state`: the state where it falls through, or undefined if it cannot. */
function run(
  statement: Statement,
  state: State,
  summaries: ReadonlyMap<string, Summary>,
  out: { late: Set<string>; most: Map<string, number>; loads: Set<string> },
  lenient = false,
): State | undefined {
  const apply = (events: readonly Event[], into: State) => {
    for (const event of events) {
      if (event.kind === 'load') {
        out.loads.add(event.id);
        if (into.added.has(event.id)) out.late.add(event.id);
      } else if (event.kind === 'add') {
        into.added.add(event.id);
        into.counts.set(event.id, (into.counts.get(event.id) ?? 0) + 1);
      } else {
        const summary = summaries.get(event.name);
        if (!summary) continue;
        for (const id of summary.late) out.late.add(id);
        for (const id of summary.loads) {
          out.loads.add(id);
          if (into.added.has(id)) out.late.add(id);
        }
        for (const id of summary.adds) into.added.add(id);
        for (const [id, count] of summary.most) {
          into.counts.set(id, (into.counts.get(id) ?? 0) + count);
        }
      }
    }
  };
  const ends = (at: State) => {
    for (const [id, count] of at.counts) out.most.set(id, Math.max(out.most.get(id) ?? 0, count));
  };
  switch (statement.kind) {
    case 'events':
      apply(statement.events, state);
      return state;
    case 'return':
      apply(statement.events, state);
      ends(state);
      return lenient ? state : undefined;
    case 'block': {
      let at: State | undefined = state;
      for (const each of statement.body) {
        if (!at) break;
        at = run(each, at, summaries, out, lenient);
      }
      return at;
    }
    case 'if': {
      apply(statement.events, state);
      const arms = [statement.yes, statement.otherwise].map((arm) =>
        arm ? run(arm, copy(state), summaries, out, lenient) : copy(state),
      );
      const through = arms.filter((arm): arm is State => arm !== undefined);
      if (through.length === 0) return undefined;
      return {
        added: new Set(through.flatMap((arm) => [...arm.added])),
        counts: new Map(
          [...new Set(through.flatMap((arm) => [...arm.counts.keys()]))].map((id) => [
            id,
            Math.max(...through.map((arm) => arm.counts.get(id) ?? 0)),
          ]),
        ),
      };
    }
    case 'loop': {
      // Twice through the condition and the body: a second turn sees what the first added
      let at: State = state;
      for (let turn = 0; turn < 2; turn++) {
        apply(statement.events, at);
        const after = run(statement.body, copy(at), summaries, out, true);
        if (after) at = merge(at, after);
      }
      return at;
    }
    case 'switch': {
      // Its cases one after another, none ending the path: a superset of any one case's events
      apply(statement.events, state);
      return run(statement.body, state, summaries, out, true);
    }
  }
}

function copy(state: State): State {
  return { added: new Set(state.added), counts: new Map(state.counts) };
}

function merge(a: State, b: State): State {
  const counts = new Map(a.counts);
  for (const [id, count] of b.counts) counts.set(id, Math.max(counts.get(id) ?? 0, count));
  return { added: new Set([...a.added, ...b.added]), counts };
}

/**
 * The flow of a kernel whose functions' code `codes` holds by name, `entry` among them. Functions
 * are summarised before their callers, in the order TypeGPU finished them.
 */
export function flowOf(codes: ReadonlyMap<string, string>, entry: string): Flow {
  const functions = new Set(codes.keys());
  const summaries = new Map<string, Summary>();
  const summarise = (name: string): Summary => {
    const known = summaries.get(name);
    if (known) return known;
    const code = codes.get(name) as string;
    const tokens = tokensOf(code.slice(code.indexOf('{')));
    const [body] = parseStatement(tokens, 0, functions);
    const calls = eventsOf(tokens, functions).flatMap((event) =>
      event.kind === 'call' && event.name !== name ? [event.name] : [],
    );
    for (const callee of calls) summarise(callee);
    const out = {
      late: new Set<string>(),
      most: new Map<string, number>(),
      loads: new Set<string>(),
    };
    const end = run(body, { added: new Set(), counts: new Map() }, summaries, out);
    if (end)
      for (const [id, count] of end.counts)
        out.most.set(id, Math.max(out.most.get(id) ?? 0, count));
    const summary: Summary = {
      loads: out.loads,
      adds: new Set(out.most.keys()),
      most: out.most,
      late: out.late,
    };
    summaries.set(name, summary);
    return summary;
  };
  const { late, most } = summarise(entry);
  return { late, most };
}
