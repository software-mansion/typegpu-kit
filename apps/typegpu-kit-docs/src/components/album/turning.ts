// How the sheets move, without a page or a GPU: each eases over to its side of the spread, its free
// edge trailing the turn. Times are in seconds.

import { restLeft, restRight } from './book.ts';

/** Seconds a sheet takes to turn. */
const TURN = 0.6;

const PI = Math.PI;
const inOut = (k: number) => (k < 0.5 ? 4 * k * k * k : 1 - (-2 * k + 2) ** 3 / 2);

interface Sheet {
  /** Its turn from the right, at 0, to the left, at π, the bend that trails it, and its height. */
  angle: number;
  bend: number;
  height: number;
  /** The turn it eases from and to, and how far along it is. */
  from: number;
  to: number;
  k: number;
}

/** A sheet lying on the right. */
const lying = (): Sheet => ({ angle: 0, bend: 0, height: 0, from: 0, to: 0, k: 1 });

/** `count` sheets lying open at the first spread; `still` drops the motion. */
export function createTurning(count: number, still: boolean) {
  const sheets = Array.from({ length: count }, lying);
  let spread = 0;

  /** Sends each sheet to its side of the spread, turning over or there at once. */
  const settle = (now: boolean) => {
    sheets.forEach((sheet, i) => {
      const to = i < spread ? PI : 0;
      if (to === sheet.to) return;
      sheet.from = now ? to : sheet.angle;
      sheet.to = to;
      sheet.k = now || still ? 1 : 0;
    });
    step(0);
  };

  /** Moves the sheets on by `dt`; whether any still moves. */
  const step = (dt: number) => {
    sheets.forEach((sheet, i) => {
      sheet.k = Math.min(1, sheet.k + dt / TURN);
      const k = inOut(sheet.k);
      sheet.angle = sheet.from + (sheet.to - sheet.from) * k;
      sheet.bend = 0.2 * Math.sign(sheet.from - sheet.to) * Math.sin(PI * k);
      const lifted = Math.sin(sheet.angle);
      sheet.height =
        restRight(i, count) +
        (restLeft(i) - restRight(i, count)) * (sheet.angle / PI) +
        0.05 * lifted;
    });
    return !landed();
  };

  const landed = () => sheets.every(({ k }) => k >= 1);
  step(0);

  return {
    sheets: sheets as readonly Readonly<Sheet>[],
    get spread() {
      return spread;
    },
    /** Whether every sheet lies down. */
    get landed() {
      return landed();
    },
    step,
    /** Lies open at `target` at once. */
    turnTo(target: number) {
      spread = Math.min(Math.max(target, 0), count);
      settle(true);
    },
    /** Turns the sheet at the spread's edge over; whether there was one. */
    turn(forward: boolean) {
      if (forward ? spread >= count : spread <= 0) return false;
      spread += forward ? 1 : -1;
      settle(false);
      return true;
    },
  };
}
