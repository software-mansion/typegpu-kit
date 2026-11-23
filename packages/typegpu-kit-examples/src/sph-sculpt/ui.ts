import { buttons } from '../ui.ts';

export const ui = {
  controls: buttons(
    {
      build: { label: 'build', keys: ['ShiftLeft', 'ShiftRight'], held: true, latch: true },
    },
    { right: [['build']] },
  ),
};
