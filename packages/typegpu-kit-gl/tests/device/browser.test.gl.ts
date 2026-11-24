import { expect, test } from 'vitest';

test('the test browser has WebGL 2', () => {
  const gl = new OffscreenCanvas(4, 4).getContext('webgl2');
  expect(gl).not.toBeNull();
  if (!gl) return;
  expect(gl.getParameter(gl.MAX_COLOR_ATTACHMENTS)).toBeGreaterThanOrEqual(4);
});

test('the test browser has WebGPU to compare against', async () => {
  const adapter = await navigator.gpu?.requestAdapter();
  expect(adapter).toBeTruthy();
});
