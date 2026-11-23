/** Height of the tray's top, where the jelly world stands. */
export const GROUND = 0.03;
/** Half the length and width of the slot the jelly lies in, and the height of its floor. */
export const SLOT = { x: 1.02, z: 0.22, floor: -0.06 };
/** The middle of the tray, which the view turns round. */
export const MIDDLE = [0, 0.05, 0] as const;

/** A point or direction in world units, which the CPU's vector math writes in place. */
export type V3 = [number, number, number];
