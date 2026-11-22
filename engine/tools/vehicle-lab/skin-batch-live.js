// Before and after the skin batch in one run of the page's own real-time loop (one GPU
// lock hold a scene): ROUNDS rounds of WINDOW seconds "before" (one by one with the
// reference kernel at window.__skinRefUrl: skin-batch-live.sh puts the base commit's
// there) then WINDOW seconds "after" (the page's default). An expression for the
// recorder's --eval. Returns the page's per-second rows (perfLog) tagged with the
// setting, and the derby's most damage of any car (contact) a second; summarised by
// skin-batch-live.py.
(async () => {
  const ROUNDS = window.__sblRounds ?? 3, WINDOW = window.__sblWindow ?? 10;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const ref = (await import(window.__skinRefUrl)).SHOWCASE_SKIN_WGSL.skin;
  const set = (mode) => { const c = document.getElementById('skinBatch'); c.value = mode; c.dispatchEvent(new Event('change', { bubbles: true })); };
  const derby = [];
  const tick = setInterval(() => {
    const g = window.__derby?.();
    if (g) derby.push({ t: g.t, dmg: Math.max(0, ...g.cars.map((c) => c.damage || 0)) });
  }, 250);
  const windows = [];
  for (let r = 1; r <= ROUNDS; r++) {
    for (const side of ['before', 'after']) {
      if (side === 'before') { set('off'); window.__skinCode(ref); } else { window.__skinCode(null); set('wide'); }
      const at = performance.now() / 1000;
      await sleep(WINDOW * 1000);
      // (The first second of a window is the switch's: left out.)
      windows.push({ side, r, from: at + 1, to: performance.now() / 1000 });
    }
  }
  await sleep(1500);
  clearInterval(tick);
  const rows = window.__perfLog();
  const dmgAt = (t) => { let d = 0; for (const x of derby) if (x.t <= t) d = x.dmg; return d; };
  const f = (x, d = 1) => (x == null ? '-' : Number(x).toFixed(d));
  const out = ['side | round | page s | sim s | steps/s | refused | drawn/s | frame p50/p95/max ms | GPU ms/step | GPU ms/drawn frame skin/shadow/picture | max damage'];
  for (const r of rows) {
    const w = windows.find((x) => r.page >= x.from && r.page < x.to);
    if (!w) continue;
    out.push([w.side, w.r, f(r.page), f(r.sim, 2), r.steps, r.refused, r.drawn, r.dt.join('/'), f(r.gpu, 2), r.gpuDraw ? r.gpuDraw.join('/') : '-', f(100 * dmgAt(r.sim), 0) + '%'].join(' | '));
  }
  return out.join('\n');
})()
