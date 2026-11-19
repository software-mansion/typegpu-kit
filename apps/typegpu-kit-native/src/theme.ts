// The docs site's tokens, from `apps/typegpu-kit-docs/src/styles/tokens.css`

export const colors = {
  ground: '#0a0d12',
  ground2: '#0e1118',
  ground3: '#14181f',
  ground4: '#1b1f28',
  rule: '#1f242d',
  ruleStrong: '#313844',
  text: '#e9ecf2',
  text2: '#c1c7d2',
  muted: '#929baa',
  /** The mark's blue, 5:1 on the ground: text that must read takes `link` or `accentHigh`. */
  accent: '#4d79ff',
  accentHigh: '#dbf0ff',
  accentLow: 'rgba(77, 121, 255, 0.16)',
  link: '#94b0ff',
  exception: '#f37b72',
  /** The pulled card's ground, its back's hatching, and the dark that shades it as it turns. */
  cardGround: '#0d1016',
  cardHatch: '#181d25',
  cardShade: '#030407',
} as const;

/** The rarities' colours, by the site token each is: `f-kind` for common. */
export const tokens = {
  'f-kind': '#c5cbd3',
  'f-needs': '#6cd0a5',
  'f-records': '#7ec7e7',
  'f-injects': '#c7a6f2',
  'f-requires': '#f8b44f',
} as const;

/** Inter Display SemiBold, which the kit ships for its baked font. Loaded in the root layout. */
export const DISPLAY = 'InterDisplay';

export const fonts = {
  display: { fontFamily: DISPLAY },
  mono: { fontFamily: 'Menlo' },
} as const;
