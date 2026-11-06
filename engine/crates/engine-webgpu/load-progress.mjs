// A page's loading card (the current stage, its elapsed time and the stages done)
// and its fatal-error card (a lost WebGPU device or a failed load, with a reload
// button). UI only: the stages are named by the code paths that run them.
//
//   const loading = createLoading();      // shows at once
//   loading.stage('Building the car');    // ends the previous stage
//   await loading.paint();                // let the card draw before blocking work
//   loading.done();                       // first frame on screen: hide
//   showFatal({ title, detail, hint });   // the device is gone: say why, offer a reload

const STYLE = `
.rb-card { position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); z-index: 20;
  min-width: 320px; max-width: min(560px, calc(100vw - 32px)); box-sizing: border-box;
  background: rgba(12, 16, 24, 0.94); border: 1px solid rgba(255,255,255,0.12); border-radius: 10px;
  padding: 16px 18px; color: #e6e9ef; font: 13px/1.45 system-ui, sans-serif; box-shadow: 0 8px 30px rgba(0,0,0,0.5); }
.rb-card h2 { margin: 0 0 8px; font-size: 15px; font-weight: 600; }
.rb-card .rb-now { display: flex; justify-content: space-between; gap: 12px; font-weight: 600; }
.rb-card .rb-time { font: 12px ui-monospace, monospace; color: #9aa4b2; font-variant-numeric: tabular-nums; }
.rb-card .rb-bar { height: 3px; margin: 8px 0 10px; background: #1d2533; border-radius: 2px; overflow: hidden; }
/* A transform animation runs on the compositor: it keeps moving while Rust blocks the main thread. */
.rb-card .rb-bar i { display: block; height: 100%; width: 30%; background: #5b9cff; border-radius: 2px; will-change: transform; animation: rb-slide 1.2s ease-in-out infinite; }
@keyframes rb-slide { 0% { transform: translateX(-100%); } 100% { transform: translateX(340%); } }
.rb-card ol { margin: 0; padding: 0; list-style: none; font: 12px ui-monospace, monospace; color: #9aa4b2; }
.rb-card ol li { display: flex; justify-content: space-between; gap: 12px; }
.rb-card .rb-note { margin-top: 8px; color: #9aa4b2; font-size: 12px; }
.rb-card pre { white-space: pre-wrap; word-break: break-word; margin: 8px 0; font: 12px ui-monospace, monospace; color: #ff8a80; max-height: 40vh; overflow: auto; }
.rb-card button { margin-top: 6px; background: #2b5fb8; color: #fff; border: 0; border-radius: 6px; padding: 6px 14px; font: 600 13px system-ui, sans-serif; cursor: pointer; }
.rb-card button:hover { background: #3a70cc; }
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

/** The loading card. `note` (optional) is shown under the stages. */
export function createLoading({ title = 'Loading', note = '' } = {}) {
  style();
  const card = document.createElement('div');
  card.className = 'rb-card';
  card.id = 'loading';
  card.innerHTML = `<h2></h2><div class="rb-now"><span class="rb-stage"></span><span class="rb-time"></span></div><div class="rb-bar"><i></i></div><ol></ol><div class="rb-note"></div>`;
  card.querySelector('h2').textContent = title;
  card.querySelector('.rb-note').textContent = note;
  document.body.appendChild(card);
  const t0 = performance.now();
  let current = null; // { name, at }
  const log = []; // [name, ms]
  const stageEl = card.querySelector('.rb-stage'), timeEl = card.querySelector('.rb-time'), listEl = card.querySelector('ol');
  const tick = () => {
    const now = performance.now();
    timeEl.textContent = current ? `${secs(now - current.at)} · total ${secs(now - t0)}` : secs(now - t0);
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
  };
  const api = {
    /** Start stage `name` (ending the one before). */
    stage(name) {
      end();
      current = { name, at: performance.now() };
      stageEl.textContent = name;
      tick();
      console.info(`[loading] ${name} (at ${secs(current.at - t0)})`);
    },
    /** Resolves once the card has been drawn (before blocking work), or after
     * 100 ms in a hidden or occluded tab, where no frame comes. */
    paint: () => new Promise((r) => {
      let done = false;
      const go = () => { if (!done) { done = true; setTimeout(r, 0); } };
      requestAnimationFrame(go);
      setTimeout(go, 100);
    }),
    /** The stages so far: [[name, ms], ...] and the total (ms). */
    times: () => ({ stages: log.slice(), total: performance.now() - t0 }),
    /** Loaded: hide the card. */
    done() {
      end();
      clearInterval(timer);
      const total = performance.now() - t0;
      console.info(`[loading] done in ${secs(total)}: ${log.map(([n, ms]) => `${n} ${secs(ms)}`).join(' · ')}`);
      card.remove();
      return { stages: log.slice(), total };
    },
    /** Failed: stop the clock (the fatal card takes over). */
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
