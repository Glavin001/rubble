// Drawing against physics on the vehicle lab, live: an expression for the recorder's
// --eval (it runs in the page once it is ready).
//
//   LIMIT=20 bash engine/tools/vehicle-lab/run.sh --probe --secs 1 --eval engine/tools/vehicle-lab/physics-first.js \
//     --out engine/out/physics-first "derby4:mode=derby&car=gen&surface=0&derbyCars=4"
//
// window.__pf (set before with --query or in the URL is not possible: an option of the run
// is read from the page's own `pf*` globals, defaulting to): WALL seconds of the run, the
// page's real-time loop (drawing as the page draws it); then, with the World still
// stepping, DRAWOFF seconds with nothing drawn (window.__drawOff: the World alone); then
// the page's per-second rows (perfLog: World steps a second, frames drawn, frame interval
// p50 / p95 / worst, steps refused, GPU ms a World step, GPU ms a drawn frame by part)
// with the derby's state (the most damage of any car: contact) a second.
// Returns the rows as text, one a second, and a summary of windows by simulated time.
(async () => {
  const WALL = window.__pfWall ?? 30, DRAWOFF = window.__pfDrawOff ?? 8;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const derby = [];
  const tick = setInterval(() => {
    const g = window.__derby?.();
    if (g) derby.push({ t: g.t, dmg: Math.max(0, ...g.cars.map((c) => c.damage || 0)) });
  }, 250);
  const t0 = performance.now();
  await sleep(WALL * 1000);
  const offAt = performance.now() / 1000;
  window.__drawOff = true;
  await sleep(DRAWOFF * 1000);
  window.__drawOff = false;
  await sleep(1500);
  clearInterval(tick);
  const rows = window.__perfLog();
  const dmgAt = (t) => { let d = 0; for (const r of derby) if (r.t <= t) d = r.dmg; return d; };
  const f = (x, d = 1) => (x == null ? '-' : Number(x).toFixed(d));
  const out = ['page s | sim s | off | rAF/s | steps/s | refused | drawn/s | frame p50/p95/max ms | GPU ms/step | GPU ms/drawn frame skin/shadow/picture | scale | CPU ms step/poll/draw (mean, max) | max damage'];
  for (const r of rows) {
    if (r.page * 1000 < t0 - 500) continue;
    const off = r.page >= offAt - 0.01 && r.page < offAt + DRAWOFF - 0.5;
    out.push([f(r.page), f(r.sim, 2), off ? 'OFF' : '', r.fps, r.steps, r.refused, r.drawn, r.dt.join('/'), f(r.gpu, 2), r.gpuDraw ? r.gpuDraw.join('/') : '-', r.scale, `${r.cpuStep.join(',')}/${r.cpuPoll.join(',')}/${r.cpuDraw.join(',')}`, f(100 * dmgAt(r.sim), 0) + '%'].join(' | '));
  }
  return out.join('\n');
})()
