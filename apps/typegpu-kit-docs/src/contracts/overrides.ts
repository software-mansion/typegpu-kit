import type { ModuleContract } from './contract.ts';

/** What a vendored module's tags would say, if TypeGPU Kit edited its source. */
export type Override = Partial<
  Pick<
    ModuleContract,
    | 'summary'
    | 'kind'
    | 'platform'
    | 'requires'
    | 'injects'
    | 'binds'
    | 'submits'
    | 'allocates'
    | 'owns'
    | 'order'
  >
> & {
  /** Exported functions the card lists as helpers, not factories: what the factories are made with. */
  readonly helpers?: readonly string[];
};

export interface Overrides {
  /** Facts for vendored modules, by id. A module of TypeGPU Kit's own carries its facts as tags. */
  readonly modules: Readonly<Record<string, Override>>;
  /**
   * Modules a vendored subpath's root exports flat, by its specifier and then the module's:
   * the files their names come from, relative to the root's directory.
   */
  readonly flat?: Readonly<Record<string, Readonly<Record<string, readonly string[]>>>>;
}

/** Vendored module facts must match source feature checks, submissions and exports. */
export const overrides: Overrides = {
  modules: {
    'kit/polylines': {
      summary:
        "`'use gpu'` functions that place the vertices of lines, joins, caps and circles on the screen.",
      kind: 'shader',
      helpers: [
        'circleVertexCount',
        'lineSegmentIndices',
        'lineSegmentWireframeIndices',
        'lineSegmentLeftIndices',
      ],
    },
    'kit/meshes': {
      summary:
        'Shapes with a vertex count and `vertexAt(i)`, computed in a vertex shader or baked into buffers.',
      injects: [
        {
          kind: 'member',
          note: "a baked shape's vertices and indices, for the pipeline that draws it",
        },
      ],
      binds: [
        {
          storage: 1,
          uniform: 0,
          note: "a baked shape's `vertexAt` or `indexAt`; drawing through `inject()` uses vertex and index buffers without storage bindings",
        },
      ],
      submits: {
        when: 'setup',
        note: 'a bake without an encoder or a pass submits a compute pass of its own',
      },
      owns: {
        items: ['buffers', 'a pipeline per baked shape'],
        note: "a baked shape's vertices and indices, and the pass that fills them; a computed shape owns nothing",
      },
      helpers: [
        'attach',
        'concat',
        'map',
        'transform',
        'isIndexed',
        'parametric',
        'triangleVertexCount',
        'triangleIndexCount',
        'triangleIndices',
        'tessellate',
      ],
    },
  },
  flat: {
    '@typegpu/kit/geometry': { polylines: ['circle.ts', 'lines/index.ts'] },
  },
};
