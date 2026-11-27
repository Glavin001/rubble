// A page's World and its WebGPU device and canvas: the World (engine-world-wasm) opens the
// device, the page renders on it into `canvas` (sized to its CSS box, device pixel ratio up
// to 2, on every resize), a lost device or an uncaptured GPU error stops the page with the
// browser's reason. (softbody.html, frozen, keeps its own inline copy.)

import { createScene } from './rubble-world.mjs';
import { showFatal } from './load-progress.mjs';

/** The World for scene `def` (sceneFromQuery's), its device and `canvas`'s context. */
export async function openWorld(def, { canvas, fail }) {
  const world = await createScene(def);
  const device = world.device();
  // A lost device (the GPU process restarted, a GPU reset or watchdog, memory): say so with
  // the browser's reason, and offer the reload that makes a new one.
  device.lost.then((i) => {
    showFatal({
      title: 'The GPU device was lost',
      detail: `${i.reason}: ${i.message || '(no message)'}`,
      hint: 'The browser took the WebGPU device away (its GPU process restarted, the GPU reset, or it ran out of memory). Reloading opens a new device. If it happens again, chrome://gpu has the browser\'s log.',
    });
    fail('device lost: ' + i.message);
  });
  device.addEventListener('uncapturederror', (e) => fail('gpu: ' + ((e.error && e.error.message) || e.error)));
  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  const configure = () => {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    canvas.height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    context.configure({ device, format, alphaMode: 'opaque' });
  };
  configure();
  addEventListener('resize', configure);
  return { world, device, context, format };
}
