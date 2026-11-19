// Where a browser frame's milliseconds go: the pages' profiler panel and the headless
// harness (tools/webgpu-profile/run.sh) read the same numbers from here.
//
//   import { attachProfiler } from './profiler.mjs';
//   const prof = attachProfiler(() => W?.world, { parent: controlsElement });
//   prof.span('page.draw', () => draw(...));   // page work worth naming
//
// Per display frame it keeps: the rAF callbacks' JS time, the World's host spans (Rust,
// engine_webgpu::host_profile: world.step and its parts, sim.encode_dispatches, world.poll,
// ...), the page's named spans, and the GPU times of the steps whose results arrived
// (World gpuTiming 'step' or 'phases'). A span that waited on the GPU through JSPI (it
// suspended to the event loop) is counted apart, never inside the frame's JS time.
// Kernel replay (WgpuSim::kernel_replay) runs from here with the page's frame loop held.
// No physics here: the numbers come from Rust; this file collects and shows them.
import { setHostProfile, hostSpans, wasmExports } from './rubble-world.mjs';

const median = (a) => { if (!a.length) return 0; const s = [...a].sort((x, y) => x - y); return s[(s.length - 1) >> 1]; };
const quant = (a, q) => { if (!a.length) return 0; const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.round((s.length - 1) * q))]; };
const stat = (a) => ({ median: median(a), p90: quant(a, 0.9), max: a.length ? Math.max(...a) : 0, n: a.length });

let installed = null;

/**
 * Attach the profiler to a page whose World is `getWorld()` (a getter: pages that restart
 * build a new World). `opts.parent`: where the panel's toggle goes (a UI control, so the
 * profiler never needs a URL flag); `opts.keep`: frames kept (default 600).
 */
export function attachProfiler(getWorld, opts = {}) {
  if (installed) return installed;
  const keep = opts.keep ?? 600;
  const frames = [];
  let on = false, measures = false, held = null, cur = null, wrapped = new WeakSet();
  const rafNative = window.requestAnimationFrame.bind(window);
  const pageSpans = new Map();
  let gpuPending = [];

  // Every rAF callback timed; callbacks of one display frame share its timestamp.
  window.requestAnimationFrame = (cb) => rafNative((now) => {
    if (held) { held.push(cb); return; }
    if (!on) { cb(now); return; }
    if (!cur || cur.now !== now) { finishFrame(); cur = { now, js: 0, callbacks: 0 }; }
    const t0 = performance.now();
    try { cb(now); } finally { cur.js += performance.now() - t0; cur.callbacks++; }
  });

  // The World's frames carry their steps' GPU times: read them as the page polls.
  function watchWorld() {
    const w = getWorld?.();
    if (!w || wrapped.has(w)) return;
    wrapped.add(w);
    const poll = w.poll.bind(w);
    w.poll = () => {
      const f = poll();
      if (on) { try { for (const t of f.gpuTimes() || []) gpuPending.push(t); } catch {} }
      return f;
    };
  }

  function finishFrame() {
    if (!cur) return;
    const spans = JSON.parse(hostSpans());
    const host = {}; let suspendedMs = 0, suspended = 0;
    for (const s of spans) {
      if (s.y > 0) { suspended++; suspendedMs += s.d; host[`${s.n} (suspended)`] = (host[`${s.n} (suspended)`] || 0) + s.d; continue; }
      host[s.n] = (host[s.n] || 0) + s.d;
    }
    const page = Object.fromEntries(pageSpans); pageSpans.clear();
    const gpu = gpuPending; gpuPending = [];
    const prev = frames[frames.length - 1];
    frames.push({ now: cur.now, dt: prev ? cur.now - prev.now : 0, js: cur.js, callbacks: cur.callbacks, host, page, suspended, suspendedMs,
      gpu: gpu.map((t) => ({ step: t.step, totalMs: t.totalMs, segments: t.segments })) });
    if (frames.length > keep) frames.shift();
    cur = null;
  }

  const api = {
    get on() { return on; },
    get frames() { return frames; },
    /** Start or stop recording; `withMeasures` also emits performance.measure per host span (traces). */
    start(withMeasures = false) { watchWorld(); measures = withMeasures; setHostProfile(true, measures); on = true; frames.length = 0; hostSpans(); },
    stop() { finishFrame(); on = false; setHostProfile(false, false); },
    reset() { frames.length = 0; hostSpans(); gpuPending = []; },
    /** Time page work `fn` as span `name` (also a performance.measure while recording). */
    span(name, fn) {
      if (!on) return fn();
      const t0 = performance.now();
      try { return fn(); } finally {
        const t1 = performance.now();
        pageSpans.set(name, (pageSpans.get(name) || 0) + (t1 - t0));
        if (measures) try { performance.measure(name, { start: t0, end: t1 }); } catch {}
      }
    },
    /** Aggregate of the kept frames: medians, p90s and max per key, plus rates. */
    summary() {
      finishFrame();
      const fr = frames.slice(1);
      const keys = (field) => [...new Set(fr.flatMap((f) => Object.keys(f[field])))];
      const col = (field, k) => fr.map((f) => f[field][k] || 0);
      const host = Object.fromEntries(keys('host').map((k) => [k, stat(col('host', k))]));
      const page = Object.fromEntries(keys('page').map((k) => [k, stat(col('page', k))]));
      const gpuTotal = fr.flatMap((f) => f.gpu.map((g) => g.totalMs));
      const segs = new Map();
      for (const f of fr) for (const g of f.gpu) {
        const per = new Map();
        for (const [n, ms] of g.segments || []) per.set(n, (per.get(n) || 0) + ms);
        for (const [n, ms] of per) { if (!segs.has(n)) segs.set(n, []); segs.get(n).push(ms); }
      }
      const steps = fr.reduce((a, f) => a + f.gpu.length, 0);
      const span = fr.length ? (fr[fr.length - 1].now - fr[0].now) / 1000 : 0;
      const hostTop = (f) => (f.host['world.step'] || 0) + (f.host['world.poll'] || 0);
      return {
        frames: fr.length, seconds: span, rafHz: span ? fr.length / span : 0, stepsTimed: steps,
        visibility: document.visibilityState,
        frameDt: stat(fr.map((f) => f.dt)),
        js: stat(fr.map((f) => f.js)),
        jsOther: stat(fr.map((f) => Math.max(0, f.js - hostTop(f) - Object.values(f.page).reduce((a, b) => a + b, 0)))),
        suspendedMs: stat(fr.map((f) => f.suspendedMs)),
        host, page,
        gpuStep: stat(gpuTotal),
        gpuSegments: Object.fromEntries([...segs].map(([n, a]) => [n, stat(a)])),
      };
    },
    /**
     * Kernel replay of the next step (WgpuSim::kernel_replay): holds the page's frame
     * loop, runs the replay through WebAssembly.promising (it waits on the GPU), releases
     * the loop. Returns the report (JSON object).
     */
    async replay(reps = 15) {
      const w = getWorld?.();
      if (!w) throw new Error('no world');
      if (!('Suspending' in WebAssembly)) throw new Error('kernel replay needs JSPI (Chrome 137+)');
      const wasm = await wasmExports();
      const raw = wasm.webworld_kernelReplayRun;
      if (!raw) throw new Error('this wasm build has no kernelReplayRun');
      held = [];
      // Let the frame in flight finish its callbacks before borrowing the world.
      await new Promise((r) => setTimeout(r, 50));
      try {
        await WebAssembly.promising(raw)(w.__wbg_ptr, reps);
        return JSON.parse(w.kernelReplayResult());
      } finally {
        const q = held; held = null;
        for (const cb of q) window.requestAnimationFrame(cb);
      }
    },
    /** Name the step's GPU work after the plan (debug groups `phase/kernel`, for GPU traces). */
    setDebugLabels(v) { getWorld?.()?.setDebugLabels(v); },
    setGpuTiming(mode) { getWorld?.()?.setGpuTiming(mode); },
  };
  installed = api;
  window.__rubbleProfiler = api;
  if (opts.parent) panel(api, opts.parent);
  return api;
}

const f2 = (x) => (x ?? 0).toFixed(2);

/** The in-page panel: a toggle in `parent`, a live table of the breakdown. */
function panel(prof, parent) {
  const btn = document.createElement('button');
  btn.textContent = 'Profiler';
  btn.title = 'Where each frame’s milliseconds go: JS, the World’s host work (Rust), the GPU per step and phase. Kernel replay times every GPU dispatch of one step alone.';
  const box = document.createElement('div');
  box.id = 'rubble-profiler';
  box.hidden = true;
  box.style.cssText = 'position:fixed;right:8px;bottom:8px;z-index:50;max-height:70vh;overflow:auto;font:11px/1.35 ui-monospace,monospace;background:rgba(10,12,16,.88);color:#dfe6ee;padding:8px 10px;border-radius:6px;min-width:300px;max-width:min(560px,calc(100vw - 32px))';
  box.innerHTML = `<div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-bottom:6px">
    <b style="margin-right:auto">Profiler</b>
    <label title="GPU timestamps per step: one span, or a pass per phase (the split itself adds a little GPU time)">GPU <select data-k="timing"><option value="off">off</option><option value="step" selected>step</option><option value="phases">phases</option></select></label>
    <label title="Name every dispatch phase/kernel in GPU captures (no change to the work)"><input type="checkbox" data-k="labels"> labels</label>
    <button data-k="replay" title="Replay the next step one dispatch at a time (15 reps each) and time every kernel alone. Holds the page for a moment; the simulation is left exactly as it was.">Kernel replay</button>
    <button data-k="copy" title="Copy the breakdown as JSON">Copy</button></div><pre data-k="out" style="margin:0;white-space:pre-wrap"></pre><pre data-k="rep" style="margin:6px 0 0;white-space:pre"></pre>`;
  document.body.append(box);
  parent.append(btn);
  const q = (k) => box.querySelector(`[data-k="${k}"]`);
  let timer = null;
  btn.onclick = () => {
    box.hidden = !box.hidden;
    if (!box.hidden) { prof.start(false); prof.setGpuTiming(q('timing').value); timer = setInterval(render, 500); }
    else { clearInterval(timer); prof.stop(); }
  };
  q('timing').onchange = (e) => prof.setGpuTiming(e.target.value);
  q('labels').onchange = (e) => prof.setDebugLabels(e.target.checked);
  q('copy').onclick = () => navigator.clipboard?.writeText(JSON.stringify(prof.summary(), null, 1));
  q('replay').onclick = async () => {
    q('rep').textContent = 'replaying…';
    try { const r = await prof.replay(15); q('rep').textContent = r.error ? `replay: ${r.error}` : r.table; }
    catch (e) { q('rep').textContent = `replay failed: ${e.message ?? e}`; }
  };
  function render() {
    const s = prof.summary();
    const rows = [
      `frames ${s.frames} · rAF ${s.rafHz.toFixed(0)} Hz · ${s.visibility} · dt ${f2(s.frameDt.median)} ms`,
      `JS per frame ${f2(s.js.median)} ms (p90 ${f2(s.js.p90)}) · other JS ${f2(s.jsOther.median)}`,
      s.suspendedMs.median > 0 ? `suspended on GPU waits ${f2(s.suspendedMs.median)} ms` : '',
      'host (Rust)        median    p90',
      ...Object.entries(s.host).sort((a, b) => b[1].median - a[1].median).slice(0, 14).map(([k, v]) => `  ${k.padEnd(30)} ${f2(v.median).padStart(6)} ${f2(v.p90).padStart(6)}`),
      ...Object.entries(s.page).map(([k, v]) => `  ${k.padEnd(30)} ${f2(v.median).padStart(6)} ${f2(v.p90).padStart(6)}`),
      `GPU per step ${f2(s.gpuStep.median)} ms (p90 ${f2(s.gpuStep.p90)}, ${s.stepsTimed} steps)`,
      ...Object.entries(s.gpuSegments).sort((a, b) => b[1].median - a[1].median).slice(0, 12).map(([k, v]) => `  ${k.padEnd(30)} ${f2(v.median).padStart(6)} ${f2(v.p90).padStart(6)}`),
    ];
    q('out').textContent = rows.filter(Boolean).join('\n');
    prof.reset();
  }
}
