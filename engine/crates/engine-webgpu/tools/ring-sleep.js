// The 110k ring with sleep on, in the page's own loop: samples demo.html's status and
// window.__RATES__ every 2 s until every body sleeps for 10 s (or 5 min). Run with
//   CHECK=1 READY='!!window.__RATES__' EXPR="$(cat tools/ring-sleep.js)" \
//   node tools/page-bench.mjs 'demo.html?scene=ring110k&sleep=on'
(async () => {
  const t0 = performance.now(), out = [];
  const read = () => {
    const s = document.getElementById('status').textContent;
    const step = +(s.match(/step (\d+)/)?.[1] ?? -1), awake = +(s.match(/awake (\d+)/)?.[1] ?? -1);
    return { t: +((performance.now() - t0) / 1000).toFixed(1), step, awake, rates: window.__RATES__ ? { ...window.__RATES__ } : null };
  };
  let asleepSince = null;
  while (performance.now() - t0 < 300000) {
    await new Promise((r) => setTimeout(r, 2000));
    const r = read();
    out.push(r);
    if (r.awake === 0 && r.step > 480) { asleepSince ??= r.t; if (r.t - asleepSince >= 10) break; } else asleepSince = null;
  }
  return { samples: out.filter((_, i) => i % 3 === 0 || i >= out.length - 6) };
})()
