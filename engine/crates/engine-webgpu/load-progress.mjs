// A page's loading card (the current stage, its elapsed time, the stages done and
// each GPU kernel's compile state) and its fatal-error card (a lost WebGPU device
// or a failed load, with a retry button). UI only: the stages are named by the code
// paths that run them, and the kernel list comes from Rust (engine-world-wasm
// `Hear`: the lines `loading.progress` takes).
//
//   const loading = createLoading({ cancel });   // shows at once
//   loading.stage('Building the car');    // ends the previous stage
//   world.explicitSimAsync(spec, loading.progress);   // Rust's load events
//   await loading.paint();                // let the card draw before blocking work
//   loading.done();                       // first frame on screen: hide
//   loading.fail({ title, detail, hint });   // the load failed: say why, offer a retry
//   showFatal({ title, detail, hint });   // after loading: the device is gone
//
// While kernels compile (one at a time, as Rust starts them) the card lists every
// kernel: queued, compiling (its own running clock, and how long it took on this
// browser last time), or done (its time). The times are kept in localStorage per kernel and kernel version (the
// `key` Rust sends), so the estimate is what this machine measured, never a guess.

const STYLE = `
.rb-card { position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); z-index: 20;
  width: max-content; min-width: min(560px, calc(100vw - 32px)); max-width: min(720px, calc(100vw - 32px)); max-height: calc(100vh - 32px); overflow: auto; box-sizing: border-box;
  background: rgba(12, 16, 24, 0.94); border: 1px solid rgba(255,255,255,0.12); border-radius: 10px;
  padding: 16px 18px; color: #e6e9ef; font: 13px/1.45 system-ui, sans-serif; box-shadow: 0 8px 30px rgba(0,0,0,0.5); }
.rb-card h2 { margin: 0 0 8px; font-size: 15px; font-weight: 600; }
.rb-card .rb-now { display: flex; justify-content: space-between; gap: 12px; font-weight: 600; }
.rb-card .rb-time { font: 12px ui-monospace, monospace; color: #9aa4b2; font-variant-numeric: tabular-nums; white-space: nowrap; }
.rb-card .rb-bar { height: 3px; margin: 8px 0 10px; background: #1d2533; border-radius: 2px; overflow: hidden; }
/* A transform animation runs on the compositor: it keeps moving while Rust blocks the main thread. */
.rb-card .rb-bar i { display: block; height: 100%; width: 30%; background: #5b9cff; border-radius: 2px; will-change: transform; animation: rb-slide 1.2s ease-in-out infinite; }
.rb-card.rb-failed .rb-bar i { animation: none; width: 100%; background: #c0392b; }
@keyframes rb-slide { 0% { transform: translateX(-100%); } 100% { transform: translateX(340%); } }
@keyframes rb-spin { to { transform: rotate(360deg); } }
.rb-card ol { margin: 0; padding: 0; list-style: none; font: 12px ui-monospace, monospace; color: #9aa4b2; }
.rb-card ol li { display: flex; justify-content: space-between; gap: 12px; }
.rb-card ol li span:last-child { white-space: nowrap; font-variant-numeric: tabular-nums; }
.rb-card ol.rb-kernels { margin: 0 0 8px; padding: 6px 8px; background: rgba(255,255,255,0.04); border-radius: 6px; }
.rb-card ol.rb-kernels:empty { display: none; }
.rb-card .rb-k span:first-child::before { content: '·'; display: inline-block; width: 12px; height: 12px; margin-right: 6px; text-align: center; line-height: 12px; }
.rb-card .rb-k.rb-run { color: #e6e9ef; }
.rb-card .rb-k.rb-run span:first-child::before { content: ''; box-sizing: border-box; border: 2px solid #33415a; border-top-color: #5b9cff; border-radius: 50%; vertical-align: -2px; will-change: transform; animation: rb-spin 0.8s linear infinite; }
.rb-card .rb-k.rb-done span:first-child::before { content: '✓'; color: #5fd08a; }
.rb-card .rb-k.rb-fail { color: #ff8a80; }
.rb-card .rb-k.rb-fail span:first-child::before { content: '✕'; }
.rb-card .rb-note { margin-top: 8px; color: #9aa4b2; font-size: 12px; }
.rb-card .rb-note:empty { display: none; }
.rb-card pre { white-space: pre-wrap; word-break: break-word; margin: 8px 0; font: 12px ui-monospace, monospace; color: #ff8a80; max-height: 40vh; overflow: auto; }
.rb-card button { margin-top: 8px; margin-right: 8px; background: #2b5fb8; color: #fff; border: 0; border-radius: 6px; padding: 6px 14px; font: 600 13px system-ui, sans-serif; cursor: pointer; }
.rb-card button:hover { background: #3a70cc; }
.rb-card button.rb-quiet { background: #2a3242; }
.rb-card button.rb-quiet:hover { background: #374157; }
`;
let styled = false;
const style = () => {
  if (styled) return;
  styled = true;
  const s = document.createElement('style');
  s.textContent = STYLE;
  document.head.appendChild(s);
};
const secs = (ms) => `${(ms / 1000).toFixed(1)} s`;
// A duration as a person says it: "0.4 s", "12 s", "1 min 35 s".
const about = (ms) => ms < 10000 ? secs(ms) : ms < 60000 ? `${Math.round(ms / 1000)} s` : `${Math.floor(ms / 60000)} min ${Math.round((ms % 60000) / 1000)} s`;

// This browser's last compile time per kernel: { [name]: { key, ms, cold } }.
// `key` is the kernel version the time belongs to, `ms` the last load's time, and
// `cold` the longest seen for that version (its uncached compile). Storage can be
// absent or throw (private windows): the card then simply has no estimates.
const STORE = 'rubble.kernelCompile.v1';
const readTimes = () => { try { return JSON.parse(localStorage.getItem(STORE)) || {}; } catch { return {}; } };
const writeTimes = (t) => { try { localStorage.setItem(STORE, JSON.stringify(t)); } catch { /* no estimates next time */ } };

/** Whether an error's text says the browser lost its WebGPU device. */
export const isDeviceLost = (text) => /lost the GPU device|device lost|device is lost|external Instance reference no longer exists/i.test(String(text));

/**
 * The loading card. `note` (optional) is shown under the stages. `cancel`
 * (optional) `{ label, hint, run }`: a button shown while kernels compile, which
 * stops waiting by running `run` (a page's lighter variant of the scene, such as
 * the same scene without the option whose kernels are slow).
 */
export function createLoading({ title = 'Loading', note = '', cancel = null } = {}) {
  style();
  const card = document.createElement('div');
  card.className = 'rb-card';
  card.id = 'loading';
  card.innerHTML = `<h2></h2><div class="rb-now"><span class="rb-stage"></span><span class="rb-time"></span></div><div class="rb-bar"><i></i></div><ol class="rb-kernels"></ol><div class="rb-note rb-knote"></div><ol class="rb-stages"></ol><div class="rb-note rb-pnote"></div><div class="rb-actions"></div>`;
  card.querySelector('h2').textContent = title;
  card.querySelector('.rb-pnote').textContent = note;
  document.body.appendChild(card);
  const t0 = performance.now();
  let current = null; // { name, at }
  let failed = false;
  const log = []; // [name, ms]
  const stageEl = card.querySelector('.rb-stage'), timeEl = card.querySelector('.rb-time'), listEl = card.querySelector('.rb-stages');
  const kernelsEl = card.querySelector('.rb-kernels'), knoteEl = card.querySelector('.rb-knote'), actionsEl = card.querySelector('.rb-actions');
  // Kernels: name -> { key, deferred, state: 'queued' | 'run' | 'later' | 'done' |
  // 'fail' | 'stopped', at (its compile's start), ms, last, li }.
  const kernels = new Map();
  const times = readTimes();
  const waited = () => [...kernels.values()].filter((k) => !k.deferred);
  const kernelLine = () => {
    if (!kernels.size || failed || !current) return;
    const all = waited(), left = all.filter((k) => k.state === 'run' || k.state === 'queued');
    if (!left.length) { stageEl.textContent = `GPU kernels compiled (${all.length}); building the scene on the GPU`; knoteEl.textContent = ''; actionsEl.textContent = ''; return; }
    const n = all.length - left.length;
    const now = all.find((k) => k.state === 'run');
    current.name = `Compiling ${all.length} GPU kernels`;
    stageEl.textContent = `Compiling GPU kernel ${Math.min(n + 1, all.length)} of ${all.length}${now ? `: ${now.name}` : ''}`;
    // What is left, from this browser's own record of the same kernel versions: the
    // one compiling (its last time less its clock) and the ones queued, one after another.
    const known = left.every((k) => k.last?.same);
    const eta = left.reduce((sum, k) => sum + (k.last?.same ? Math.max(0, k.last.ms - (k.state === 'run' ? performance.now() - k.at : 0)) : 0), 0);
    const over = now?.last?.same && performance.now() - now.at > now.last.ms + 500;
    knoteEl.textContent = (known
      ? (over ? `${now.name} is taking longer than on this browser's last load.` : `About ${about(eta)} left, going by this browser's last load.`)
      : 'No estimate: this browser has not compiled this version of these kernels before. The clocks show that it is working.')
      + (known ? '' : ' The browser caches each compiled kernel, so the next load of this version skips the wait.');
  };
  const kernelRow = (k) => {
    const [a, b] = k.li.children;
    k.li.className = `rb-k rb-${k.state === 'later' ? 'queued' : k.state}`;
    a.textContent = k.name;
    const last = k.last ? (k.last.same ? `last time ${about(k.last.ms)}` : `changed; the previous version took ${about(k.last.cold)}`) : 'first time here';
    b.textContent = k.state === 'run' ? `compiling ${secs(performance.now() - k.at)} · ${last}`
      : k.state === 'queued' ? `queued · ${last}`
        : k.state === 'later' ? 'queued: compiles after the first frame'
          : k.state === 'done' ? `done ${secs(k.ms)}`
            : k.state === 'stopped' ? `interrupted at ${secs(k.ms)}`
              : k.state === 'never' ? 'not started'
                : `failed at ${secs(k.ms)}`;
  };
  const tick = () => {
    const now = performance.now();
    timeEl.textContent = current ? `${secs(now - current.at)} · total ${secs(now - t0)}` : secs(now - t0);
    for (const k of kernels.values()) if (k.state === 'run') kernelRow(k);
    kernelLine();
  };
  const timer = setInterval(tick, 100);
  const end = () => {
    if (!current) return;
    const ms = performance.now() - current.at;
    log.push([current.name, ms]);
    const li = document.createElement('li');
    li.innerHTML = '<span></span><span></span>';
    li.firstChild.textContent = `✓ ${current.name}`;
    li.lastChild.textContent = secs(ms);
    listEl.appendChild(li);
    current = null;
    actionsEl.textContent = '';
    knoteEl.textContent = '';
  };
  const api = {
    /** Start stage `name` (ending the one before). */
    stage(name) {
      if (failed) return;
      end();
      current = { name, at: performance.now() };
      stageEl.textContent = name;
      tick();
      console.info(`[loading] ${name} (at ${secs(current.at - t0)})`);
    },
    /** Within the current stage: part `name` finished `ms` after the stage began,
     * listed under it; `label` (optional) renames the stage. */
    item(name, ms, label) {
      const li = document.createElement('li');
      li.innerHTML = '<span></span><span></span>';
      li.firstChild.textContent = `  · ${name}`;
      li.lastChild.textContent = secs(ms);
      listEl.appendChild(li);
      if (label && current) { current.name = label; stageEl.textContent = label; }
      console.info(`[loading]   ${name} ${secs(ms)}`);
    },
    /** The kernels about to compile (`[{ name, key, deferred }]`, in any order): starts
     * the compile stage and lists them queued, in the order they compile (the
     * smallest first, `bytes`). */
    kernels(list) {
      if (failed) return;
      api.stage(`Compiling ${list.filter((k) => !k.deferred).length} GPU kernels`);
      kernels.clear();
      kernelsEl.textContent = '';
      const order = list.slice().sort((a, b) => (!!a.deferred - !!b.deferred) || ((a.bytes ?? 0) - (b.bytes ?? 0)));
      for (const { name, key, deferred } of order) {
        const rec = times[name];
        const li = document.createElement('li');
        li.innerHTML = '<span></span><span></span>';
        const k = { name, key, deferred: !!deferred, state: deferred ? 'later' : 'queued', at: 0, ms: 0, li,
          last: rec ? { same: rec.key === key, ms: rec.ms, cold: rec.cold ?? rec.ms } : null };
        kernels.set(name, k);
        kernelsEl.appendChild(li);
        kernelRow(k);
      }
      if (cancel) {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'rb-quiet';
        b.textContent = cancel.label;
        if (cancel.hint) b.title = cancel.hint;
        b.onclick = () => { b.disabled = true; cancel.run(); };
        actionsEl.appendChild(b);
      }
      tick();
    },
    /** Kernel `name`'s compile starts. */
    compiling(name) {
      const k = kernels.get(name);
      console.info(`[loading]   ${name} compiling`);
      if (!k) return;
      k.state = 'run';
      k.at = performance.now();
      if (!card.isConnected || failed) return;
      kernelRow(k);
      kernelLine();
    },
    /** Kernel `name` finished (`ok`) or was rejected, `ms` after it started. */
    kernel(name, ms, ok = true) {
      const k = kernels.get(name);
      console.info(`[loading]   ${name} ${ok ? 'compiled' : 'failed'} ${secs(ms)}`);
      if (!k) return;
      // (A deferred kernel reports after the page is running: recorded, not shown.)
      k.state = ok ? 'done' : 'fail';
      k.ms = ms;
      if (ok) {
        const rec = times[name];
        // Re-read before writing: another tab may have recorded other kernels.
        const all = readTimes();
        all[name] = { key: k.key, ms, cold: rec && rec.key === k.key ? Math.max(rec.cold ?? rec.ms, ms) : ms };
        writeTimes(all);
      }
      if (!card.isConnected || failed) return;
      kernelRow(k);
      kernelLine();
    },
    /** One line from Rust's load (engine-world-wasm `Hear`): pass this function as
     * the `progress` argument of `explicitSimAsync`, `createShowcase` and the like. */
    progress(line) {
      const i = line.indexOf(' ');
      const [what, rest] = i < 0 ? [line, ''] : [line.slice(0, i), line.slice(i + 1)];
      if (what === 'spawn') api.stage('Building the vehicles (Rust)');
      else if (what === 'kernels') api.kernels(JSON.parse(rest));
      else if (what === 'compiling') api.compiling(rest);
      else if (what === 'compiled' || what === 'failed') {
        const [name, ms] = rest.split(' ');
        api.kernel(name, Number(ms), what === 'compiled');
        if (what === 'failed') console.warn(`[loading] ${rest}`);
      }
    },
    /** Resolves once the card has been drawn (before blocking work), or after
     * 100 ms in a hidden or occluded tab, where no frame comes. */
    paint: () => new Promise((r) => {
      let done = false;
      const go = () => { if (!done) { done = true; setTimeout(r, 0); } };
      requestAnimationFrame(go);
      setTimeout(go, 100);
    }),
    /** The stages so far: [[name, ms], ...], each kernel's [name, ms, state], and the total (ms). */
    times: () => ({ stages: log.slice(), kernels: [...kernels.values()].map((k) => [k.name, k.ms, k.state]), total: performance.now() - t0 }),
    /** Whether the card is still up (loading, not failed). */
    active: () => card.isConnected && !failed,
    /** Loaded: hide the card. */
    done() {
      end();
      clearInterval(timer);
      const total = performance.now() - t0;
      console.info(`[loading] done in ${secs(total)}: ${log.map(([n, ms]) => `${n} ${secs(ms)}`).join(' · ')}`);
      card.remove();
      return { stages: log.slice(), kernels: [...kernels.values()].map((k) => [k.name, k.ms, k.state]), total };
    },
    /** The load failed: the card stays, with what was done, the kernels' states as
     * they stood, the error, and a retry button (`retry`: its action; default a
     * reload). A lost device is named as such. */
    fail({ title, detail = '', hint = '', retry = null, retryLabel = 'Retry' } = {}) {
      if (failed) return;
      failed = true;
      clearInterval(timer);
      const lost = isDeviceLost(detail);
      const now = performance.now();
      for (const k of kernels.values()) {
        if (k.state === 'run') { k.state = 'stopped'; k.ms = now - k.at; kernelRow(k); }
        else if (k.state === 'queued') { k.state = 'never'; kernelRow(k); }
      }
      card.id = 'fatal';
      card.classList.add('rb-failed');
      card.setAttribute('role', 'alert');
      card.querySelector('h2').textContent = title ?? (lost ? 'The browser lost the GPU device' : 'Loading failed');
      stageEl.textContent = current ? `Stopped while: ${current.name}` : 'Stopped';
      timeEl.textContent = secs(now - t0);
      knoteEl.textContent = '';
      actionsEl.textContent = '';
      const pre = document.createElement('pre');
      pre.textContent = detail;
      if (detail) actionsEl.before(pre);
      card.querySelector('.rb-pnote').textContent = hint || (lost
        ? 'The browser\'s GPU process stopped while the kernels compiled; nothing on this page is running. Retry reloads the page: kernels that had finished are in the browser\'s shader cache and do not compile again. (macOS may go on compiling the interrupted kernel in the background for some minutes, which slows the retry.)'
        : 'The full error is in the console.');
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = retryLabel;
      b.onclick = retry ?? (() => location.reload());
      actionsEl.appendChild(b);
      if (cancel && lost) {
        const c = document.createElement('button');
        c.type = 'button';
        c.className = 'rb-quiet';
        c.textContent = cancel.label;
        c.onclick = () => cancel.run();
        actionsEl.appendChild(c);
      }
    },
    /** Failed: remove the card (a caller that shows its own error). */
    stop() { clearInterval(timer); card.remove(); },
  };
  return api;
}

/** The fatal-error card: `title`, the browser's `detail`, a `hint`, and a reload button. */
export function showFatal({ title, detail = '', hint = '' }) {
  style();
  if (document.getElementById('fatal')) return;
  document.getElementById('loading')?.remove();
  const card = document.createElement('div');
  card.className = 'rb-card';
  card.id = 'fatal';
  card.setAttribute('role', 'alert');
  card.innerHTML = `<h2></h2><pre></pre><div class="rb-note"></div><button type="button">Reload the page</button>`;
  card.querySelector('h2').textContent = title;
  card.querySelector('pre').textContent = detail;
  if (!detail) card.querySelector('pre').remove();
  card.querySelector('.rb-note').textContent = hint;
  card.querySelector('button').onclick = () => location.reload();
  document.body.appendChild(card);
}
