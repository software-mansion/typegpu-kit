// What an example shows over its canvas besides settings: plain data each host renders the same
// way, which the scene binds handlers to in its `ui` field

import type { Source } from '@typegpu/kit/video';
import type { Picture } from './contract.ts';

export type Corner = 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';

/** Text over the canvas at a corner, such as a game's score. Line breaks are kept. */
export interface TextKind {
  readonly kind: 'text';
  readonly corner: Corner;
}

export interface Button {
  /** The pad's text: a word, or a symbol such as `▲`. */
  readonly label: string;
  /** What a screen reader says for a symbol label. Default `label`. */
  readonly name?: string;
  /** `KeyboardEvent.code` values that press it on the web, such as `ArrowUp`. Default none. */
  readonly keys?: readonly string[];
  /** Pressed once per hold, with `release` as the key comes up or the finger lifts. Default false. */
  readonly held?: boolean;
  /**
   * On a touch pad, a tap holds a `held` button down and the next tap releases it, for a mode
   * kept while the other fingers work. Its keys still hold it only while down. Default false.
   */
  readonly latch?: boolean;
}

/** Rows of button names on one side; `null` leaves a gap one button wide. */
export type PadRows<N extends string> = readonly (readonly (N | null)[])[];

export interface Pads<N extends string> {
  readonly left?: PadRows<N>;
  readonly right?: PadRows<N>;
}

export type Buttons = Readonly<Record<string, Button>>;

/**
 * Any `buttons` element, whatever its buttons are named, as hosts take it. `ButtonsKind<B>` reads
 * `B` both ways through `keyof B`, so a declared element is no `ButtonsKind`.
 */
export interface AnyButtonsKind {
  readonly kind: 'buttons';
  readonly buttons: Buttons;
  readonly pads: Pads<string>;
}

/** Buttons pressed by keys on the web and by pads on a touch screen. */
export interface ButtonsKind<B extends Buttons = Buttons> extends AnyButtonsKind {
  readonly buttons: B;
  readonly pads: Pads<keyof B & string>;
}

/** An image the reader picks: a file picker or a drop on the web, the photo library on native. */
export interface ImageKind {
  readonly kind: 'image';
  /** The button's text, such as `choose image`. */
  readonly label: string;
}

/**
 * A camera the host asks for when the example opens: `getUserMedia` on the web, VisionCamera on
 * native. The host attaches its source to the `Camera` the scene binds, and a button turns it off
 * and on. A reader who declines keeps whatever the scene shows without a camera.
 */
export interface CameraKind {
  readonly kind: 'camera';
  /** The button's text, such as `use camera`. */
  readonly label: string;
  /** Prefers the camera facing the reader, shown mirrored. Default true. */
  readonly selfie?: boolean;
}

/**
 * Short sounds the scene plays, such as a jump's, by name. The web host decodes each file with Web
 * Audio and plays it when the scene asks, once the reader has pressed a key or touched the page;
 * the native host stays silent.
 */
export interface SoundsKind<N extends string = string> {
  readonly kind: 'sounds';
  /** Each sound's file, by path from the example's directory, such as `sounds/jump.wav`. */
  readonly files: Readonly<Record<N, string>>;
}

export type UiElement = TextKind | AnyButtonsKind | ImageKind | CameraKind | SoundsKind;
export type Ui = Readonly<Record<string, UiElement>>;

export function text(options: { readonly corner: Corner }): TextKind {
  return { kind: 'text', corner: options.corner };
}

export function buttons<const B extends Buttons>(
  named: B,
  pads: Pads<NoInfer<keyof B & string>>,
): ButtonsKind<B> {
  return { kind: 'buttons', buttons: named, pads };
}

export function image(label: string): ImageKind {
  return { kind: 'image', label };
}

export function camera(label: string, options: { readonly selfie?: boolean } = {}): CameraKind {
  return { kind: 'camera', label, ...options };
}

export function sounds<const N extends string>(files: Readonly<Record<N, string>>): SoundsKind<N> {
  return { kind: 'sounds', files };
}

/** Where a sound sits in the mix. */
export interface Placement {
  /** From -1, all left, to 1, all right. Default 0, centred. */
  readonly pan?: number;
  /** Its loudness, from 0, silent, to 1, as recorded. Default 1. */
  readonly gain?: number;
}

/** A sound playing, which can be moved in the mix and cut short. */
export interface Voice {
  /** Whether the sound still plays. */
  readonly playing: boolean;
  /** Moves the sound in the mix, as a sound that moves or is heard from somewhere else would. */
  place(placement: Placement): void;
  /** Fades the sound out over `fade` seconds, then stops it. Does nothing once it has ended. */
  stop(fade: number): void;
}

/**
 * Plays sound `name` once, now, placed in the mix as `placement` says. Returns the playing sound,
 * or nothing when it isn't heard.
 */
export type Play<N extends string = string> = (
  name: N,
  placement?: Placement,
) => Voice | undefined | void;

/** What a scene binds to a sounds element: where the host attaches its player. */
export interface Speaker<N extends string = string> {
  /** Takes the host's player, or `undefined` to fall silent, as the host leaves. */
  attach(play: Play<N> | undefined): void;
}

/** What a scene binds to a camera element: where the host attaches the camera's source. */
export interface Camera {
  attach(source: Source | undefined): void;
  /** Counts camera frames the scene has received. */
  readonly frames: number;
}

export interface Tap {
  press(): void;
}

export interface Held {
  press(): void;
  release(): void;
}

/** The handlers a scene binds to a declaration. */
export type Bound<U extends Ui> = { readonly [K in keyof U]: Handler<U[K]> };

/** What a scene binds to one element. Distributes, so `Bound<Ui>` accepts any binding. */
export type Handler<E extends UiElement> = E extends TextKind
  ? () => string
  : E extends ButtonsKind<infer B>
    ? { readonly [N in keyof B]: B[N] extends { readonly held: true } ? Held : Tap }
    : E extends ImageKind
      ? (picture: Picture) => void
      : E extends CameraKind
        ? Camera
        : E extends SoundsKind<infer N>
          ? Speaker<N>
          : never;
