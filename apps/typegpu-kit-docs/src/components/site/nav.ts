import type { ModuleContract } from '../../contracts/contract.ts';
import { hrefOf, leadOf, withPages, type Module, type PageModules } from './modules.ts';
import { pageHref } from './urls.ts';

export interface IndexLink {
  readonly label: string;
  readonly href: string;
  /** The page it opens, as a slug: `kit/frames`, or `''` for the overview. */
  readonly page: string;
  /** Set for a module's row, which shows its namespace as code. */
  readonly module?: string;
  /**
   * Set for a module other than the lead of a page that covers several. Its row opens its card on
   * the page, and is neither marked as the page nor counted in the page order.
   */
  readonly block?: boolean;
  /** For such a row, the name of the page it sits in: `finish` for `bloom`. */
  readonly within?: string;
}

/** Rows under a label inside a group, which doesn't fold. */
export interface IndexSection {
  readonly label: string;
  readonly links: readonly IndexLink[];
}

export interface IndexGroup {
  readonly label: string;
  /** Set for a group of guides: what each of its pages is, as the label above its title says. */
  readonly kind?: string;
  readonly links: readonly IndexLink[];
  /** Labelled rows after `links`. */
  readonly sections?: readonly IndexSection[];
}

/** A docs page: its slug, its title, its description and the modules its frontmatter lists. */
export interface Page extends PageModules {
  readonly title: string;
  readonly description?: string | undefined;
}

/** The groups before the module index, in reading order. Map and Examples sit in the header. */
const guides = [
  {
    label: 'Start',
    kind: 'Start',
    links: [
      { label: 'Overview', page: '' },
      { label: 'Getting started', page: 'getting-started' },
      { label: 'How to read the docs', page: 'how-to-read' },
      { label: 'Starter scenes', page: 'starter-scenes' },
      { label: 'Platforms', page: 'platforms' },
    ],
  },
  {
    // Each builds one example from an empty canvas
    label: 'Tutorials',
    kind: 'Tutorial',
    links: [
      { label: 'Slime mould', page: 'tutorials/slime-mould' },
      { label: 'Domino run', page: 'tutorials/domino-run' },
      { label: 'Write a module', page: 'tutorials/write-a-module' },
    ],
  },
  {
    // How the kit works
    label: 'Concepts',
    kind: 'Concept',
    links: [
      { label: 'How the kit works', page: 'concepts/how-the-kit-works' },
      { label: 'Modules', page: 'concepts/modules' },
      { label: 'Connecting modules', page: 'concepts/connecting-modules' },
      { label: 'Scenes and passes', page: 'concepts/scenes-and-passes' },
      { label: 'Data on the GPU', page: 'concepts/data-on-the-gpu' },
    ],
  },
  {
    // Modules combined with each other or with other libraries for one job
    label: 'Recipes',
    kind: 'Recipe',
    links: [
      { label: 'Light from textures', page: 'recipes/light-from-textures' },
      { label: 'Finished model scenes', page: 'recipes/lit-models' },
      { label: 'Dark scenes', page: 'recipes/dark-scenes' },
      { label: 'Effects you can scrub', page: 'recipes/scrubbed-effects' },
      { label: 'Beside other renderers', page: 'recipes/other-renderers' },
    ],
  },
];

/** Rows under a label in an area: the modules a program uses for one thing, in reading order. */
export interface Section {
  readonly label: string;
  /** Where the idea behind these modules is taught: a guide's slug, with a section's anchor. */
  readonly concept: string;
  readonly modules: readonly string[];
}

/** A group of the index's modules, in labelled sections. */
export interface Area {
  readonly label: string;
  readonly sections: readonly Section[];
}

/**
 * The kit's modules by area, and each area's by what a program uses them for, in reading order.
 * Every module sits in exactly one section, also a module without a page; the index shows the ones
 * that have one.
 */
export const areas: readonly Area[] = [
  {
    label: 'Core',
    sections: [
      {
        label: 'Frame',
        concept: 'concepts/how-the-kit-works#you-own-the-frame',
        modules: [
          'kit/frames',
          'kit/clock',
          'kit/coroutines',
          'kit/pipelines',
          'kit/inject',
          'kit/owner',
        ],
      },
      {
        label: 'Passes and targets',
        concept: 'concepts/scenes-and-passes',
        modules: [
          'kit/passes',
          'kit/history',
          'kit/targets',
          'kit/baker',
          'kit/mipmaps',
          'kit/memo',
          'kit/cameras',
          'kit/bindings',
        ],
      },
      {
        label: 'Tables and lists',
        concept: 'concepts/data-on-the-gpu#rows',
        modules: ['kit/table', 'kit/liveLists', 'kit/machines', 'kit/grid'],
      },
      {
        label: 'Kernels',
        concept: 'concepts/data-on-the-gpu#a-fixed-step',
        modules: ['kit/workgroup', 'kit/sorter'],
      },
      {
        label: 'Reading back',
        concept: 'concepts/data-on-the-gpu#results-back',
        modules: ['kit/readback', 'kit/counters', 'kit/picker', 'kit/dynamicResolution'],
      },
      {
        label: 'Hosts',
        concept: 'platforms',
        modules: ['kit/web', 'kit/native', 'kit/hosting', 'kit/host', 'kit/react', 'kit/runtimes'],
      },
      {
        label: 'Testing and debugging',
        concept: 'concepts/how-the-kit-works#nothing-is-hidden',
        modules: [
          'kit/testing',
          'kit/recording',
          'kit/gpuDebug',
          'kit/gpuDebugPanel',
          'kit/profiler',
          'kit/census',
        ],
      },
    ],
  },
  {
    label: 'Rendering',
    sections: [
      {
        label: 'Shapes',
        concept: 'concepts/scenes-and-passes#every-renderer-is-a-drawing',
        modules: ['kit/meshes', 'kit/meshPool'],
      },
      {
        label: 'Renderers',
        concept: 'concepts/scenes-and-passes#every-renderer-is-a-drawing',
        modules: ['kit/instances', 'kit/impostors', 'kit/poses', 'kit/lines', 'kit/polylines'],
      },
      {
        label: 'Light',
        concept: 'concepts/connecting-modules#a-module-others-read',
        modules: ['kit/lighting', 'kit/environment', 'kit/materials', 'kit/shading', 'kit/ao'],
      },
    ],
  },
  {
    label: 'Post-processing',
    sections: [
      {
        label: 'Final pass',
        concept: 'concepts/scenes-and-passes#after-the-passes',
        modules: ['kit/finish', 'kit/tone', 'kit/bloom', 'kit/exposure', 'kit/post'],
      },
      {
        label: 'Antialiasing',
        concept: 'concepts/scenes-and-passes#after-the-passes',
        modules: ['kit/fxaa', 'kit/taa'],
      },
      {
        label: 'Lens',
        concept: 'concepts/scenes-and-passes#after-the-passes',
        modules: ['kit/dof'],
      },
    ],
  },
  {
    label: 'Particles',
    sections: [
      {
        label: 'Points',
        concept: 'concepts/data-on-the-gpu#only-what-is-alive',
        modules: ['kit/particles'],
      },
      {
        label: 'Ribbons',
        concept: 'concepts/data-on-the-gpu#rows',
        modules: ['kit/trails'],
      },
    ],
  },
  {
    label: 'Text',
    sections: [
      {
        label: 'Placing text',
        concept: 'concepts/scenes-and-passes#every-renderer-is-a-drawing',
        modules: ['kit/text', 'kit/typeset'],
      },
      {
        label: 'Fonts',
        concept: 'concepts/modules#lifecycle',
        modules: ['kit/font', 'kit/fontBaker'],
      },
    ],
  },
  {
    label: 'Assets',
    sections: [
      {
        label: 'Models',
        concept: 'concepts/modules#lifecycle',
        modules: ['kit/gltf'],
      },
      {
        label: 'Images',
        concept: 'concepts/modules#lifecycle',
        modules: ['kit/images'],
      },
      {
        label: 'Camera and video',
        concept: 'concepts/modules#lifecycle',
        modules: ['kit/video'],
      },
    ],
  },
];

/**
 * What kind of page this is, as the label above its title says it: a guide by its group and its
 * place in it, and a module page by the area and section of `lead`, the first module it lists,
 * with the guide that teaches the idea behind it.
 */
export function kindOf(page: string, lead?: string) {
  if (lead !== undefined) {
    const area = areas.find((each) => places(each, lead));
    const section = area?.sections.find((each) => each.modules.includes(lead));
    if (!area || !section) return { kind: 'Module', place: undefined, concept: undefined };

    const [slug, anchor] = section.concept.split('#');
    return {
      kind: 'Module',
      place: `${area.label} / ${section.label}`,
      concept: {
        label: guideLabel(slug)!,
        href: pageHref(slug) + (anchor ? `#${anchor}` : ''),
      },
    };
  }
  const guide = guides.find((each) => each.links.some((link) => link.page === page));
  if (!guide) return undefined;
  const at = guide.links.findIndex((link) => link.page === page);
  return { kind: guide.kind, place: `${at + 1} of ${guide.links.length}`, concept: undefined };
}

/** A guide's name in the index, by its slug. */
export const guideLabel = (page: string) =>
  guides.flatMap((each) => each.links).find((each) => each.page === page)?.label;

/** A label that names a package, such as `@typegpu/kit/geometry`, rather than its modules' use. */
const packageLike = (label: string) => label.includes('@');

/**
 * Throws unless `layout` places each of `modules` in exactly one section, names no other module,
 * gives every area a section and every section a module, and labels none with a package's name.
 */
export function checkAreas(
  modules: readonly ModuleContract[],
  layout: readonly Area[] = areas,
): void {
  const problems: string[] = [];
  for (const area of layout) {
    if (packageLike(area.label)) {
      problems.push(`the area ${area.label} is named like a package; name it for its modules' use`);
    }
    if (area.sections.length === 0) problems.push(`the area ${area.label} has no sections`);
    for (const section of area.sections) {
      if (packageLike(section.label)) {
        problems.push(`the section ${section.label} is named like a package`);
      }
      if (section.modules.length === 0) problems.push(`the section ${section.label} is empty`);
      if (guideLabel(section.concept.split('#')[0]) === undefined) {
        problems.push(`the section ${section.label} names ${section.concept}, which is no guide`);
      }
    }
  }

  const ids = new Set(modules.map((module) => module.id));
  const placed = layout.flatMap((area) => area.sections.flatMap((section) => section.modules));
  const unknown = placed.filter((id) => !ids.has(id));
  const missing = [...ids].filter((id) => !placed.includes(id));
  const twice = placed.filter((id, at) => placed.indexOf(id) !== at);
  if (unknown.length > 0) problems.push(`unknown ${unknown.join(', ')}`);
  if (missing.length > 0) problems.push(`missing ${missing.join(', ')}`);
  if (twice.length > 0) problems.push(`twice ${twice.join(', ')}`);

  if (problems.length > 0) {
    throw new Error(
      `The areas in nav.ts must place each module in one section of an area: ${problems.join('; ')}.`,
    );
  }
}

/** Whether one of an area's sections places the module with `id`. */
export const places = (area: Area, id: string) =>
  area.sections.some((section) => section.modules.includes(id));

/** The areas that hold some of `modules`, in reading order, after checking where they sit. */
export function areasOf(modules: readonly ModuleContract[]) {
  checkAreas(modules);
  return areas.filter((area) => modules.some((module) => places(area, module.id)));
}

const pageLink = ({ label, page }: { label: string; page: string }): IndexLink => ({
  label,
  page,
  href: pageHref(page),
});

export function buildIndex(
  contracts: readonly ModuleContract[],
  pages: readonly Page[],
): IndexGroup[] {
  const modules = withPages(contracts, pages);

  const leadOn = (page: string) => leadOf(pages.find((each) => each.id === page));
  const byId = new Map(modules.map((module) => [module.id, module]));

  const row = (module: Module): IndexLink => {
    const lead = leadOn(module.page!) === module.id;
    return {
      label: module.namespace,
      href: lead ? pageHref(module.page!) : hrefOf(modules, module)!,
      page: module.page!,
      module: module.namespace,
      ...(lead ? {} : { block: true, within: byId.get(leadOn(module.page!)!)!.namespace }),
    };
  };

  const groups = areasOf(modules).map((area): IndexGroup => {
    const sections = area.sections
      .map((section) => ({
        label: section.label,
        links: section.modules.flatMap((id) => {
          const module = byId.get(id);
          return module?.page === undefined ? [] : [row(module)];
        }),
      }))
      .filter((section) => section.links.length > 0);

    return { label: area.label, links: [], sections };
  });

  return [
    ...guides.map((guide) => ({
      label: guide.label,
      kind: guide.kind,
      links: guide.links.map(pageLink),
    })),
    ...groups,
  ];
}

/** A group's rows, in reading order. */
export const rowsOf = (group: IndexGroup): IndexLink[] => [
  ...group.links,
  ...(group.sections ?? []).flatMap((section) => section.links),
];

/** Every link in the index, in reading order. */
export const linksOf = (groups: readonly IndexGroup[]): IndexLink[] => groups.flatMap(rowsOf);

/** Whether a row is the one for `current`; a page's other modules' rows open their blocks. */
export const isCurrent = (link: IndexLink, current: string) =>
  !link.block &&
  (link.page === current || (link.page === 'examples' && current.startsWith('examples/')));

/** A page beside the current one: its slug, its title, and what it is about. */
export interface Neighbour {
  readonly page: string;
  readonly label: string;
  readonly description?: string | undefined;
  /** What it is to the current page: `Next concept`, `Next in Core`, `Tutorials`. */
  readonly kicker: string;
}

/**
 * The pages before and after `current` in its own group, titled as the pages are. The last page of
 * a group of guides leads on to the first of the next one, and the last module of an area nowhere.
 */
export function neighbours(
  groups: readonly IndexGroup[],
  pages: readonly Page[],
  current: string,
): { prev?: Neighbour | undefined; next?: Neighbour | undefined } {
  const pagesOf = (group: IndexGroup) => [
    ...new Set(
      rowsOf(group)
        .filter((each) => !each.block)
        .map((each) => each.page),
    ),
  ];
  const group = groups.find((each) => pagesOf(each).includes(current));
  if (!group) return {};

  const titled = (page: string | undefined, kicker: string): Neighbour | undefined => {
    if (page === undefined) return undefined;
    const found = page === '' ? undefined : pages.find((each) => each.id === page);
    return {
      page,
      label: found?.title ?? guideLabel(page) ?? page,
      description: found?.description,
      kicker,
    };
  };

  const order = pagesOf(group);
  const at = order.indexOf(current);
  const of =
    group.kind === 'Start'
      ? ''
      : group.kind
        ? ` ${group.kind.toLowerCase()}`
        : ` in ${group.label}`;
  const onward = group.kind ? groups[groups.indexOf(group) + 1] : undefined;

  return {
    prev: titled(order[at - 1], `Previous${of}`),
    next:
      titled(order[at + 1], `Next${of}`) ??
      (onward?.kind ? titled(pagesOf(onward)[0], onward.label) : undefined),
  };
}
