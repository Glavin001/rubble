(() => {
  // Drive the page's frame loop from a 120 Hz timer instead of the display.
  const HZ = 120, T = 1000 / HZ;
  let next = performance.now() + T;
  window.requestAnimationFrame = (cb) => {
    const now = performance.now();
    if (next < now) next = now;
    const at = next; next += T;
    return setTimeout(() => cb(performance.now()), Math.max(0, at - now));
  };
  window.__abLog = ['rAF replaced by a 120 Hz timer'];
  return 'ok';
})()
