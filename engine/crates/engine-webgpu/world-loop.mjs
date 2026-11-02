// The pages' World frame bookkeeping (rubble-world.mjs re-exports it): polling, and
// stepping as fast as readbacks allow. No wasm here, so unit tests can drive it with a mock
// world.

// One macrotask: lets the browser run GPU map callbacks.
const channel = new MessageChannel();
let waiting = [];
// Node (unit tests): the port keeps the process alive only while a wait is pending.
channel.port1.onmessage = () => { const w = waiting; waiting = []; channel.port1.unref?.(); for (const r of w) r(); };
channel.port1.unref?.();
export const macrotask = () => new Promise((r) => { waiting.push(r); if (waiting.length === 1) { channel.port1.ref?.(); channel.port2.postMessage(0); } });

/** Polls the world, hands the frame to `onFrame` and frees it; tracks the last step seen. */
export function pollFrame(world, onFrame) {
  const f = world.poll();
  if (!f.empty) world.lastPolledStep = f.lastStep;
  onFrame(f);
  f.free();
}

/**
 * `n` steps as fast as readbacks allow (automation, checks), handing each poll's
 * frame to `onFrame`; resolves once every step's results have arrived.
 * `beforeStep()` runs right before each step is taken (per-step inputs).
 */
export async function stepFrames(world, n, onFrame, beforeStep = null) {
  for (let i = 0; i < n;) {
    if (world.canStep()) {
      beforeStep?.();
      world.step();
      i++;
    } else await macrotask();
    pollFrame(world, onFrame);
  }
  await drain(world, onFrame);
}

/** Resolves once the frames of every step taken so far have been polled. */
export async function drain(world, onFrame) {
  const last = world.stepCount() - 1;
  while ((world.lastPolledStep ?? -1) < last) {
    await macrotask();
    pollFrame(world, onFrame);
  }
}
