import { buttons, text } from '../ui.ts';

export const ui = {
  score: text({ corner: 'top-right' }),
  controls: buttons(
    {
      left: { label: '◀', name: 'left', keys: ['ArrowLeft', 'KeyA'], held: true },
      right: { label: '▶', name: 'right', keys: ['ArrowRight', 'KeyD'], held: true },
      jump: { label: 'jump', keys: ['ArrowUp', 'KeyW', 'Space'], held: true },
    },
    { left: [['left', 'right']], right: [['jump']] },
  ),
};
