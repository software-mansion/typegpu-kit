/**
 * Turns a number on the GPU, a module for the contracts' tests.
 *
 * @module
 * @owns a buffer: the speed
 * @records `spin` in a compute pass.
 */

import { d, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';
import { helper } from './stray.ts';

export interface Options {
  /** What the spinner is called. */
  label: string;
  /** Turns a second, e.g. one. Default `1`. */
  speed?: number;
  /** Turns it backwards. */
  backwards?: boolean;
  /** Eases each turn. @gpu */
  ease?: (t: number) => number;
  /** Runs after each turn. */
  turned?: () => void;
}

export interface SpinOptions {
  /** Turns this many times. Default `1`. */
  turns?: number;
}

export interface Spinner {
  /** Counts the turns so far. It starts at 0. */
  readonly turns: number;
  /** Spins it in a compute pass. */
  spin(pass: TgpuComputePass, options?: SpinOptions): void;
  /** @hidden */
  reset(): void;
  destroy(): void;
}

export const Settings = d.struct({ speed: d.f32 });

/** Makes a spinner. */
export function create(root: TgpuRoot, options: Options): Spinner;
export function create(root: TgpuRoot, options: Options, label: string): Spinner;
export function create(root: TgpuRoot, options: Options, _label?: string): Spinner {
  const buffer = root.createBuffer(Settings, { speed: options.speed ?? helper() });
  return {
    turns: 0,
    spin: (pass) => void pass,
    reset: () => {},
    destroy: () => buffer.destroy(),
  };
}

export const turn = (x: number) => {
  'use gpu';
  return x * 2;
};

export const speed = tgpu.accessor(d.f32);

export const defaults = { speed: 1, axis: d.vec3f(0, 1, 0) };

export const presets = { fast: () => 2 };

/** @hidden */
export const internal = 3;

export const registry = new Map<string, number>();
