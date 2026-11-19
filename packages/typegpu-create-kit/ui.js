// The initializer's look: the logo dithered into characters over a dithered rule, and prompts in the
// kit's blue and ice.

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { styleText } from 'node:util';
import { isCancel, SelectPrompt, TextPrompt } from '@clack/core';

/** @typedef {readonly [number, number, number]} Rgb */

const out = process.stdout;
/** Bits of colour the terminal shows: 24 and 8 get the kit's colours, 4 plain styles, 1 none. */
const depth = out.isTTY ? out.getColorDepth() : 1;
/** The dither's characters, and a prompt's marks: heavy where it is live, faint for its rail. */
const [HEAVY, LIGHT, FAINT] = ['#', ':', '.'];

/** @type {Rgb} */
const BLUE = [76, 119, 255];
/** @type {Rgb} */
const ICE = [217, 239, 255];
/** @type {Rgb} */
const DEEP = [41, 64, 178];
/** Lifts the logo's tones, so its lightest colours print as solid characters. */
const GAIN = 1.25;

const luminance = (/** @type {Rgb} */ [r, g, b]) => (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
const mean = (/** @type {Rgb} */ a, /** @type {Rgb} */ b) =>
  /** @type {Rgb} */ ([(a[0] + b[0]) >> 1, (a[1] + b[1]) >> 1, (a[2] + b[2]) >> 1]);
/** A channel in the 256-colour cube, whose levels are 0, 95, 135, 175, 215 and 255. */
const level = (/** @type {number} */ v) => (v < 48 ? 0 : Math.max(1, Math.round((v - 55) / 40)));

/**
 * Text in `rgb` where the terminal has 24-bit colour, in the nearest of 256 colours where it has
 * those, and in `plain` where it has 16.
 * @param {Rgb} rgb
 * @param {Parameters<typeof styleText>[0]} plain
 * @returns {(text: string) => string}
 */
const paint = (rgb, plain) => (text) => {
  if (depth >= 24) return `\x1b[38;2;${rgb.join(';')}m${text}\x1b[39m`;
  if (depth >= 8) {
    const [r, g, b] = rgb.map(level);
    return `\x1b[38;5;${16 + 36 * r + 6 * g + b}m${text}\x1b[39m`;
  }
  return depth > 1 ? styleText(plain, text) : text;
};
const blue = paint(BLUE, 'blue');
const ice = paint(ICE, 'cyan');
const deep = paint(DEEP, 'blue');
/** @param {Parameters<typeof styleText>[0]} style @returns {(text: string) => string} */
const styled = (style) => (text) => (depth > 1 ? styleText(style, text) : text);
const dim = styled('dim');
const bold = styled('bold');
const warn = styled('yellow');
const inverse = styled('inverse');

/** The logo's pixels, read from the site's `logo.svg`, whose paths are one-pixel-high runs. */
function logo() {
  const svg = readFileSync(path.join(import.meta.dirname, 'logo.svg'), 'utf8');
  const [, width, height] = /viewBox="0 0 (\d+) (\d+)"/.exec(svg) ?? [];
  /** @type {(Rgb | undefined)[][]} */
  const pixels = Array.from({ length: Number(height) }, () => Array(Number(width)).fill(undefined));
  for (const [, fill, d] of svg.matchAll(/<path fill="#(\w+)" d="([^"]+)"/g)) {
    const hex = fill.length === 3 ? fill.replace(/./g, '$&$&') : fill;
    const channel = (/** @type {number} */ at) => parseInt(hex.slice(at, at + 2), 16);
    /** @type {Rgb} */
    const rgb = [channel(0), channel(2), channel(4)];
    for (const [, x, y, run] of d.matchAll(/M(\d+) (\d+)h(\d+)/g))
      for (let at = Number(x); at < Number(x) + Number(run); at++) pixels[Number(y)][at] = rgb;
  }
  return pixels;
}

const BAYER = [
  [0, 32, 8, 40, 2, 34, 10, 42],
  [48, 16, 56, 24, 50, 18, 58, 26],
  [12, 44, 4, 36, 14, 46, 6, 38],
  [60, 28, 52, 20, 62, 30, 54, 22],
  [3, 35, 11, 43, 1, 33, 9, 41],
  [51, 19, 59, 27, 49, 17, 57, 25],
  [15, 47, 7, 39, 13, 45, 5, 37],
  [63, 31, 55, 23, 61, 29, 53, 21],
];

/**
 * A tone from 0 to 1 as a character in `rgb`, dithered: the 8×8 Bayer matrix picks between a
 * heavy, a light and a faint one, so a tone between two characters comes out as a pattern of both.
 * @param {number} tone
 * @param {Rgb} rgb
 * @param {number} x
 * @param {number} y
 */
function dither(tone, rgb, x, y) {
  const threshold = (BAYER[y % 8][x % 8] + 0.5) / 64;
  if (tone * 6 <= threshold) return ' ';
  const character = tone > threshold ? HEAVY : tone * 2.2 > threshold ? LIGHT : FAINT;
  return paint(rgb, luminance(rgb) > 0.6 ? 'cyan' : 'blue')(character);
}

/**
 * The logo in characters, one for a pixel and the pixel under it, in their mean colour.
 * @param {(Rgb | undefined)[][]} pixels
 */
function art(pixels) {
  const lines = [];
  for (let y = 0; y < pixels.length; y += 2) {
    const cells = pixels[y].map((top, x) => {
      const bottom = pixels[y + 1]?.[x];
      const ink = top && bottom ? mean(top, bottom) : (top ?? bottom);
      if (!ink) return ' ';
      const tone = Math.min(1, (0.2 + 0.8 * luminance(ink)) * GAIN);
      return dither(top && bottom ? tone : tone / 2, ink, x, y / 2);
    });
    lines.push(cells.join(''));
  }
  return lines;
}

/** A rule that thins out to the right, ice at its front. */
const rule = (/** @type {number} */ width) =>
  Array.from({ length: width }, (_, x) => {
    const cover = 1 - x / (width - 1);
    return dither(cover * cover, cover < 0.42 ? ICE : cover < 0.7 ? BLUE : DEEP, x, 0);
  }).join('');

/** Prints the logo and the kit's version, or the version alone where the logo would not fit. */
export function banner(/** @type {string} */ version) {
  const title = `${bold(ice('TypeGPU Kit'))} ${dim(`${version} · new project`)}`;
  const pixels = logo();
  const width = pixels[0].length;
  const lines = out.columns >= width + 4 ? ['', ...art(pixels), '', rule(width), ''] : [''];
  out.write(`${[...lines, title, '', ''].map((line) => `  ${line}`.trimEnd()).join('\n')}`);
}

/**
 * One prompt: its mark and message, then `body` beside the rail.
 * @param {string} state
 * @param {string} message
 * @param {string[]} body
 */
const frame = (state, message, body) =>
  [
    state === 'submit' || state === 'cancel'
      ? `${blue(LIGHT)}  ${message}`
      : `${ice(HEAVY)}  ${bold(message)}`,
    ...body.map((line) => `${deep(FAINT)}  ${line}`),
    deep(FAINT),
  ]
    .map((line) => `  ${line}`)
    .join('\n');

/**
 * Ends the run after a cancelled prompt.
 * @returns {never}
 */
function stop() {
  out.write(`  ${blue(HEAVY)}  ${dim('Stopped. Nothing was written.')}\n\n`);
  process.exit(1);
}

/**
 * Asks for a line of text. An empty answer is `fallback`.
 * @param {string} message
 * @param {string} fallback
 * @param {(value: string) => string | undefined} refuse the reason a value will not do
 */
export async function ask(message, fallback, refuse) {
  const prompt = new TextPrompt({
    defaultValue: fallback,
    validate: (value) => refuse(value || fallback),
    render() {
      if (this.state === 'submit') return frame(this.state, message, [dim(this.value ?? '')]);
      if (this.state === 'cancel') return frame(this.state, message, [dim(this.userInput)]);
      const typed = this.userInput
        ? this.userInputWithCursor
        : `${inverse(fallback[0])}${dim(fallback.slice(1))}`;
      return frame(
        this.state,
        message,
        this.state === 'error' ? [typed, warn(this.error)] : [typed],
      );
    },
  });
  const answer = await prompt.prompt();
  if (answer === undefined || isCancel(answer)) stop();
  return answer;
}

/**
 * Asks for one of `options`, each a name with a hint beside it.
 * @template {string} T
 * @param {string} message
 * @param {readonly { name: T; title: string; hint: string }[]} options
 */
export async function choose(message, options) {
  const wide = Math.max(...options.map(({ title }) => title.length));
  const prompt = new SelectPrompt({
    options: options.map(({ name }) => ({ value: name })),
    render() {
      const picked = options[this.cursor];
      if (this.state === 'submit' || this.state === 'cancel')
        return frame(this.state, message, [dim(picked.title)]);
      return frame(
        this.state,
        message,
        options.map(({ title, hint }, at) => {
          const name = title.padEnd(wide);
          return `${at === this.cursor ? ice(`${HEAVY} ${name}`) : `${dim(FAINT)} ${name}`}  ${dim(hint)}`;
        }),
      );
    },
  });
  const answer = await prompt.prompt();
  if (answer === undefined || isCancel(answer)) stop();
  return answer;
}

/**
 * Prints the closing line and the commands to run next.
 * @param {string} line
 * @param {string[]} steps
 */
export function finish(line, steps) {
  const lead = out.isTTY ? `  ${blue(HEAVY)}  ` : '';
  const indent = out.isTTY ? '     ' : '  ';
  out.write(`${lead}${line}\n\n${steps.map((step) => `${indent}${ice(step)}`).join('\n')}\n\n`);
}
