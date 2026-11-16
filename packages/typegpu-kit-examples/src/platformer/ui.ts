import { buttons, text } from '../ui.ts';

export const ui = {
  score: text({ corner: 'top-right' }),
  controls: buttons(
    {
      left: { label: '◀', name: 'left', keys: ['ArrowLeft', 'KeyA'], held: true },
      right: { label: '▶', name: 'right', keys: ['ArrowRight', 'KeyD'], held: true },
      down: { label: '▼', name: 'duck', keys: ['ShiftLeft', 'ShiftRight'], held: true },
      jump: { label: 'jump', keys: ['ArrowUp', 'KeyW', 'Space'], held: true },
    },
    { left: [['left', 'down', 'right']], right: [['jump']] },
  ),
};
