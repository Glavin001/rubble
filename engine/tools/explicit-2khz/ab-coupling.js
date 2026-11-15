// The coupling's paths in the running page, interleaved (vehicle-lab-record.mjs --probe
// --eval tools/explicit-2khz/ab-coupling.js): from t 20 s, 1 s samples of the GPU ms per
// World step with the default (candidate list and folded step), the list off (header bit
// 128: every node against every shape, four phases) and the fold off (bit 1024), rounds to
// t 41 s; then once the coupling off (lossy, measurement: the vehicle no longer feels the
// bodies). The list and the fold are lossless: the trajectory is the same either way.
(() => {
  window.__abLog = [];
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const mean = (a) => a.reduce((x, y) => x + y, 0) / Math.max(a.length, 1);
  const sample = async (label) => {
    const t0 = window.__state().t;
    while (window.__state().t < t0 + 1.0) await sleep(50);
    const s = window.__state(), st = window.__stats().car;
    window.__abLog.push(`${label}: ${mean(window.__gpu().total.slice(-30)).toFixed(2)} ms (t ${s.t.toFixed(1)}, shapes ${st.coupling?.shapes})`);
  };
  (async () => {
    while (!(window.__state?.().t > 20)) await sleep(200);
    const w = window.__world;
    while (window.__state().t < 41) {
      await sample('default');
      w.debugExplicitCouplingBits(128);
      await sample('list off');
      w.debugExplicitCouplingBits(0);
      await sample('default');
      w.debugExplicitCouplingBits(1024);
      await sample('fold off');
      w.debugExplicitCouplingBits(0);
    }
    // Last (it changes the run from here): the coupling off, then on again.
    await sample('default');
    w.setExplicitKnob('coupling', false);
    await sample('coupling off');
    w.setExplicitKnob('coupling', true);
    await sample('default');
    window.__abLog.push('done');
  })().catch((e) => window.__abLog.push('failed: ' + e.message));
  return 'started';
})()
