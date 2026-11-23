import { random, type RandomGenerator } from 'math/random';
import { Pattern, type Launch } from './fireworks.ts';

type Vec3 = [number, number, number];

/** Linear colours the stars burn in. */
const PALETTE: Vec3[] = [
  [1, 0.55, 0.16],
  [1, 0.1, 0.06],
  [0.16, 1, 0.3],
  [0.18, 0.36, 1],
  [0.72, 0.24, 1],
  [1, 0.92, 0.82],
  [0.2, 0.9, 1],
];
const GOLD = PALETTE[0];
const SILVER = PALETTE[5];
const PATTERNS = [
  Pattern.peony,
  Pattern.chrysanthemum,
  Pattern.willow,
  Pattern.palm,
  Pattern.ring,
  Pattern.crackle,
];
/** Shell launch sites on barges behind the castle in the initial view. */
const SITES: Vec3[] = [
  [-170, 0.5, -70],
  [-95, 0.5, -120],
  [-20, 0.5, -150],
  [55, 0.5, -130],
  [130, 0.5, -95],
  [-60, 0.5, -60],
  [80, 0.5, -55],
];
const GRAVITY = 9.81;
/** Launch-speed multiplier for clicked shells relative to shells bursting at their apex. */
const CLICKED_BOOST = 1.6;
const HEIGHT_METRES = [72, 125] as const;
/** Metres a second a burst throws its stars, which sets how wide it opens. */
const BURST_SPEED = [38, 62] as const;
/** Metres a burst may land from above its site, along x and z. */
const LEAN_METRES = 14;
/** Seconds between cues, divided by the rate. */
const GAP_SECONDS = [0.7, 2.2] as const;
/** A finale `every` seconds bursts `shells` shells over `seconds`. */
const FINALE = { every: 70, shells: 26, seconds: 6 };
/** Shots that can wait to launch at once. */
const SHOTS = 64;
/** Burst-speed multipliers that open palms wider and willows narrower than other patterns. */
const OPENING: Partial<Record<Pattern, number>> = { [Pattern.palm]: 1.3, [Pattern.willow]: 0.8 };

/** A shell waiting to launch `at` a time on the show's clock, or free at `Infinity`. */
interface Shot {
  at: number;
  readonly shell: { -readonly [K in keyof Launch]: Launch[K] };
}

/** The barge nearest to the point under `burst`. */
function siteUnder(burst: Vec3) {
  let nearest = SITES[0];
  let closest = Infinity;
  for (const site of SITES) {
    const distance = Math.hypot(site[0] - burst[0], site[2] - burst[2]);
    if (distance < closest) {
      nearest = site;
      closest = distance;
    }
  }
  return nearest;
}

/**
 * Schedules salvos, pairs, fans, chains and periodic finales on the CPU. The GPU simulates
 * shells after launch.
 */
export function createShow(rng: RandomGenerator = Math.random) {
  const shots: Shot[] = Array.from({ length: SHOTS }, () => ({
    at: Infinity,
    shell: {
      position: [0, 0, 0],
      velocity: [0, 0, 0],
      fuse: 0,
      color: [0, 0, 0],
      pattern: Pattern.peony,
      burstSpeed: 0,
    },
  }));
  let clock = 0;
  let nextCue = 0;
  let nextFinale = FINALE.every;

  const pick = <T>(items: T[]) => random.choice(rng, items);
  const free = () => shots.find((shot) => shot.at === Infinity);

  const colorOf = (pattern: Pattern) => {
    if (pattern === Pattern.willow || pattern === Pattern.palm) return GOLD;
    if (pattern === Pattern.crackle) return SILVER;
    return pick(PALETTE);
  };

  /**
   * Aims from `site` toward `burst`. Bursts at the apex, or during ascent when `boost`
   * increases launch speed.
   */
  const aim = (
    shot: Shot,
    site: Vec3,
    burst: Vec3,
    pattern: Pattern,
    color: Vec3,
    burstSpeed: number,
    boost = 1,
  ) => {
    const height = Math.max(burst[1] - site[1], 1);
    const rise = boost * Math.sqrt(2 * GRAVITY * height);
    const climb = (rise - Math.sqrt(Math.max(rise * rise - 2 * GRAVITY * height, 0))) / GRAVITY;
    const { shell } = shot;
    shell.position = [...site];
    shell.velocity = [(burst[0] - site[0]) / climb, rise, (burst[2] - site[2]) / climb];
    shell.fuse = climb;
    shell.color = [...color];
    shell.pattern = pattern;
    shell.burstSpeed = burstSpeed;
  };

  /** Launches a shell from `site` after `delay` seconds, bursting near `height` above it. */
  const queue = (
    delay: number,
    site: Vec3,
    pattern: Pattern,
    color: Vec3,
    height = random.float(rng, ...HEIGHT_METRES),
  ) => {
    const shot = free();
    if (!shot) return;
    const burst: Vec3 = [
      site[0] + (rng() - 0.5) * LEAN_METRES * 2,
      height,
      site[2] + (rng() - 0.5) * LEAN_METRES * 2,
    ];
    aim(
      shot,
      site,
      burst,
      pattern,
      color,
      random.float(rng, ...BURST_SPEED) * (OPENING[pattern] ?? 1),
    );
    shot.at = clock + delay;
  };

  /** Queues the next cue's shells and returns the seconds until the cue after it. */
  const cue = () => {
    if (clock >= nextFinale) {
      for (let i = 0; i < FINALE.shells; i++) {
        const pattern = pick(PATTERNS);
        queue(rng() * FINALE.seconds, pick(SITES), pattern, colorOf(pattern));
      }
      nextFinale = clock + FINALE.every;
      return FINALE.seconds + GAP_SECONDS[1] * 2;
    }

    const pattern = pick(PATTERNS);
    const color = colorOf(pattern);
    const roll = rng();
    if (roll < 0.3) {
      // One shell, as high as they go
      queue(0, pick(SITES), pattern, color, HEIGHT_METRES[1]);
    } else if (roll < 0.5) {
      // A pair from barges either side of the middle one
      const left = random.int(rng, 0, 1);
      queue(0, SITES[left], pattern, color);
      queue(0, SITES[4 - left], pattern, color);
    } else if (roll < 0.7) {
      // A fan across the first five barges, each a little later and higher
      for (let i = 0; i < 5; i++)
        queue(i * 0.22, SITES[i], pattern, color, HEIGHT_METRES[0] + i * 6);
    } else if (roll < 0.85) {
      // A chain from one barge, each shell a new colour
      const site = pick(SITES);
      for (let i = 0; i < 4; i++) {
        queue(i * 0.45, site, pattern, pick(PALETTE), HEIGHT_METRES[0] + i * 10);
      }
    } else {
      queue(0, pick(SITES), Pattern.palm, GOLD);
      queue(0.6, pick(SITES), Pattern.crackle, SILVER);
      queue(1.1, pick(SITES), Pattern.willow, GOLD);
    }
    return random.float(rng, ...GAP_SECONDS);
  };

  return {
    /**
     * Advances by `dt` seconds and launches due shells. `rate` scales time between cues. With
     * `auto` off, launches only queued shells.
     */
    advance(dt: number, rate: number, auto: boolean, launch: (shell: Launch) => void) {
      clock += dt;
      if (!auto) nextCue = Math.max(nextCue, clock);
      else if (rate > 0) {
        while (clock >= nextCue) nextCue += cue() / rate;
      }
      for (const shot of shots) {
        if (shot.at > clock) continue;
        launch(shot.shell);
        shot.at = Infinity;
      }
    },

    /** Launches from the nearest barge to burst at `burst`. */
    fireAt(burst: Vec3) {
      const site = siteUnder(burst);
      const shot = free();
      if (!shot) return;
      const pattern = pick(PATTERNS);
      aim(
        shot,
        site,
        burst,
        pattern,
        colorOf(pattern),
        random.float(rng, ...BURST_SPEED),
        CLICKED_BOOST,
      );
      shot.at = clock;
    },
  };
}
