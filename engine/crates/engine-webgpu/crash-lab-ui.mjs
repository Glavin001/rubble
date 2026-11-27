// The crash lab's layout (softbody.html ?solver=explicit): its panels, where they sit,
// how they collapse and the focus mode. Layout only: no control here does anything by
// itself; crash-lab.mjs, crash-ref.mjs and softbody.html put their controls in the panels.
//
// One responsibility per panel, and each control in exactly one panel:
//   Scene    (left, Shift+S)  what is simulated: the scene picker (reference crashes,
//            crash-test presets, scenes); a reference crash's contact setting and run; for
//            the free scenes the vehicle (car picker, converted BeamNG vehicles, skin
//            binding, Load .truck, Load JBeam folder), the scene's knobs, the solver's
//            options and the solver switch. What does not apply to a reference crash is
//            hidden there, not disabled.
//   View     (left, Shift+V)  how and from where it is drawn: the one camera group (the
//            scenario's own cameras, side, top, ¾, front, follow, auto-orbit and the
//            car-relative views), skeleton, nodes, stubs, colour, skin, studio, grab and its
//            strength.
//   Playback (bottom, Shift+P) time: restart, play/pause, frame, step, slow motion, the
//            recorded run's scrubber and live, and the focus toggle.
//   Results  (right, Shift+R) what was measured: a reference crash's criteria (collapsed:
//            one summary line) and the live readouts (speed, damage, solver and GPU stats).
// The studio HUD is not a panel: the pills (top left), the clock (top right), the node
// and beam counts and the colour legend (above the playback bar). Focus (H, Esc leaves;
// the recorders' ?clean=1) hides every panel and leaves the 3D view and the HUD exactly
// where the clean recordings have always had them.
//
// Every panel collapses to a labelled tab and remembers its state (localStorage). Below
// 1000 px wide only one of Scene, View and Results is open at a time (a switchable
// drawer: the vehicle stays in view). A panel taller than its room scrolls inside itself.

const STORE = 'rubble-crash-lab-ui';
const DEFAULT = { scene: true, view: true, playback: true, results: false };
const KEYS = { scene: 'S', view: 'V', results: 'R', playback: 'P' };
const TITLES = { scene: 'Scene', view: 'View', results: 'Results', playback: 'Playback' };

const CSS = `
.lab-col { position: absolute; display: flex; flex-direction: column; gap: 8px; pointer-events: none; min-height: 0; }
#labLeft { top: 18px; left: 20px; bottom: var(--lab-reserve, 120px); align-items: flex-start; }
#labRight { top: 14px; right: 22px; bottom: var(--lab-reserve, 120px); align-items: flex-end; }
#labLeft > #labPills, #labRight > #labTime { position: static; flex: none; }
.lab-dock { pointer-events: auto; display: flex; flex-direction: column; min-height: 0; flex: 0 1 auto; box-sizing: border-box; background: rgba(10, 16, 30, 0.84); border: 1px solid rgba(160, 190, 255, 0.16); border-radius: 10px; font: 12px/1.4 system-ui, sans-serif; color: #dfe6f3; letter-spacing: normal; backdrop-filter: blur(8px); }
#labLeft .lab-dock { width: clamp(260px, 24vw, 340px); }
#labRight .lab-dock { width: clamp(280px, 27vw, 420px); }
.lab-dock.closed { width: auto !important; flex: none; }
.lab-dock > .lab-head { flex: none; display: flex; align-items: center; gap: 8px; margin: 0; padding: 0; font: inherit; }
.lab-toggle { flex: 1; display: flex; align-items: center; gap: 8px; min-height: 30px; min-width: 0; padding: 4px 10px; background: none; border: 0; border-radius: 10px; color: #e9eef7; font: 600 12px/1.3 system-ui, sans-serif; letter-spacing: 0.06em; text-align: left; cursor: pointer; }
.lab-dock .lab-toggle:hover { background: rgba(43, 79, 138, 0.35); }
.lab-toggle .t { text-transform: uppercase; }
.lab-toggle .s { flex: 1; min-width: 0; color: #aab8d4; font-weight: 500; letter-spacing: 0.02em; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lab-toggle .s:empty { display: none; }
.lab-toggle kbd, .lab-dock kbd { background: #222b3a; color: #c4cfe4; border-radius: 3px; padding: 0 4px; font: 11px ui-monospace, monospace; letter-spacing: 0; }
.lab-toggle .chev { color: #c4cfe4; font-size: 12px; }
#solverBtn { white-space: normal; text-align: left; }
.lab-dock.closed .lab-body { display: none; }
.lab-body { min-height: 0; overflow: auto; overscroll-behavior: contain; padding: 2px 10px 8px; border-top: 1px solid rgba(160, 190, 255, 0.10); }
.lab-body .r { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 5px 0; }
.lab-body .lbl { color: #9fb0cf; min-width: 52px; }
.lab-body .sub { margin: 8px 0 2px; color: #9fb0cf; font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; }
.lab-dock button, .lab-dock select, .lab-dock label.file { box-sizing: border-box; min-height: 26px; background: #18233a; color: #e6ecf7; border: 1px solid #2c3b5c; border-radius: 6px; padding: 3px 8px; font: 12px/1.3 system-ui, sans-serif; cursor: pointer; }
.lab-dock select { max-width: 100%; }
.lab-dock button:hover, .lab-dock label.file:hover { background: #22324f; }
.lab-dock button.on, .lab-dock select.on { background: #2b4f8a; border-color: #4a78c4; }
.lab-dock .lab-toggle { background: none; border: 0; min-height: 30px; }
.lab-dock label { color: #9fb0cf; }
.lab-dock label.file { color: #e6ecf7; display: inline-flex; align-items: center; }
.lab-dock label.file:focus-within { outline: 2px solid #8fc1ff; outline-offset: 1px; }
.lab-dock .muted { color: #9aa6be; }
.lab-dock a { color: #8fc1ff; }
.lab-dock :focus-visible, #labUnfocus:focus-visible { outline: 2px solid #8fc1ff; outline-offset: 1px; }
.lab-dock input[type=range] { accent-color: #5b9cff; }
.lab-body .row { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px; align-items: center; }
.lab-body .row label.p { color: #9fb0cf; }
.lab-body .row label.p:has(input[type=range]) { display: flex; align-items: center; gap: 6px; width: 100%; }
.lab-body .row input[type=range] { flex: 1; min-width: 100px; }
.lab-body .row .pv { min-width: 36px; text-align: right; color: #e6e9ef; font-variant-numeric: tabular-nums; }
.lab-body .row label.p input[type=checkbox] { width: 15px; height: 15px; margin: 0 2px 0 0; vertical-align: -3px; accent-color: #5b9cff; }
/* Playback: one compact bar at the bottom, its toggle the first item. */
#labPanel.lab-dock { position: absolute; left: 50%; bottom: 14px; transform: translateX(-50%); width: min(980px, calc(100vw - 32px)); flex-direction: row; align-items: center; }
#labPanel.lab-dock.closed { width: auto; }
#labPanel > .lab-head .lab-toggle { flex: none; padding: 4px 8px; }
#labPanel.lab-dock:not(.closed) > .lab-head .t, #labPanel.lab-dock:not(.closed) > .lab-head kbd { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); }
#labPanel > .lab-body { flex: 1; min-width: 0; border-top: 0; border-left: 1px solid rgba(160, 190, 255, 0.10); padding: 4px 8px; overflow: visible; }
#labPanel .r { margin: 0; }
/* The HUD above the playback bar while the panels show (focus keeps the clean frames' places). */
body.lab-on:not(.lab-clean) #labCounts { left: 22px; right: auto; bottom: var(--lab-hud, 70px); text-align: left; font-size: 16px; }
body.lab-on:not(.lab-clean) #labCounts small { font-size: 11px; margin-top: 2px; }
body.lab-on:not(.lab-clean) #labLegend { bottom: var(--lab-hud, 70px); max-width: calc(100vw - 380px); flex-wrap: wrap; justify-content: flex-end; row-gap: 4px; }
body.lab-on:not(.lab-clean) #labPills { max-width: calc(100vw - 400px); }
body.lab-on:not(.lab-clean) #labPills .lab-pill { white-space: normal; }
body.lab-on #info, body.lab-on #hud, body.lab-on #status { display: none !important; }
.lab-clean .lab-dock { display: none !important; }
/* Narrow windows, focus included (the recorders' 1280 x 720 frames are not narrow): the
   pills wrap before the clock and the counts give the legend room. */
@media (max-width: 999px) {
  #labPills { max-width: calc(100vw - 400px); }
  #labPills .lab-pill { white-space: normal; }
  body.lab-clean #labCounts { font-size: 22px; }
}
/* Focus: a way back for the pointer, invisible until hovered or reached by Tab. */
#labUnfocus { display: none; position: absolute; left: 50%; bottom: 14px; transform: translateX(-50%); padding: 6px 14px; min-height: 30px; border-radius: 999px; background: rgba(10, 16, 30, 0.84); color: #e6ecf7; border: 1px solid #2c3b5c; font: 12px system-ui, sans-serif; cursor: pointer; opacity: 0; }
.lab-clean #labUnfocus { display: block; }
#labUnfocus:hover, #labUnfocus:focus-visible { opacity: 1; }
`;

const load = () => {
  try { const s = JSON.parse(localStorage.getItem(STORE) || 'null'); if (s && typeof s === 'object') return { ...DEFAULT, ...s }; } catch { /* storage off */ }
  return { ...DEFAULT };
};
const save = (s) => { try { localStorage.setItem(STORE, JSON.stringify(s)); } catch { /* storage off */ } };

/**
 * The lab's panels. `opts.pills`, `opts.time`: the HUD's pill and clock elements (placed at
 * the top of the left and right columns); `opts.counts`, `opts.legend`: the HUD lines
 * kept above the playback bar. `opts.focus()`/`opts.setFocus(on)`: the lab's clean mode.
 * Returns each panel's body element (`bodies.scene` …), `summary(name, html)` (the text a
 * collapsed panel shows beside its name) and `open`/`toggle`/`state`/`restore`.
 * Its key handler is registered before the lab's (capture): Shift+S/V/R/P never reach the
 * lab's own S, V, R, P keys.
 */
export function createLabLayout(opts) {
  const style = document.createElement('style');
  style.textContent = CSS;
  document.head.appendChild(style);
  const el = (html, parent = document.body) => { const t = document.createElement('template'); t.innerHTML = html.trim(); return parent.appendChild(t.content.firstChild); };
  const left = el('<div class="lab-col" id="labLeft"></div>');
  const right = el('<div class="lab-col" id="labRight"></div>');
  if (opts.pills) left.appendChild(opts.pills);
  if (opts.time) right.appendChild(opts.time);
  const section = (name, parent, id) => {
    const s = el(`<section class="lab-dock" data-dock="${name}" id="${id}" aria-label="${TITLES[name]}">
      <h2 class="lab-head"><button type="button" class="lab-toggle" data-dock-toggle="${name}" aria-controls="${id}Body" aria-keyshortcuts="Shift+${KEYS[name]}"
        title="${TITLES[name]}: show or hide (Shift+${KEYS[name]})"><span class="t">${TITLES[name]}</span><span class="s"></span><kbd aria-hidden="true">⇧${KEYS[name]}</kbd><span class="chev" aria-hidden="true"></span></button></h2>
      <div class="lab-body" id="${id}Body"></div></section>`, parent);
    return s;
  };
  const secs = {
    scene: section('scene', left, 'labScene'),
    view: section('view', left, 'labView'),
    results: section('results', right, 'labResults'),
    playback: section('playback', document.body, 'labPanel'),
  };
  const bodies = Object.fromEntries(Object.entries(secs).map(([k, s]) => [k, s.querySelector('.lab-body')]));
  const unfocus = el('<button type="button" id="labUnfocus" title="Show the panels (H or Esc)">show panels · H</button>');
  unfocus.onclick = () => opts.setFocus(false);

  let st = load();
  const narrow = () => innerWidth < 1000;
  /** The drawer rule: which open panels close when `name` opens (or on a resize). */
  function exclusive(name) {
    const group = narrow() ? ['scene', 'view', 'results'] : [];
    if (!group.includes(name)) return;
    for (const k of group) if (k !== name) st[k] = false;
  }
  function apply() {
    for (const [k, s] of Object.entries(secs)) {
      const on = !!st[k];
      s.classList.toggle('closed', !on);
      const b = s.querySelector('.lab-toggle');
      b.setAttribute('aria-expanded', String(on));
      b.querySelector('.chev').textContent = on ? '▾' : '▸';
    }
    relayout();
  }
  function open(name, on) {
    if (!(name in secs)) return;
    st[name] = !!on;
    if (on) exclusive(name);
    save(st); apply();
  }
  const toggle = (name) => open(name, !st[name]);
  for (const b of document.querySelectorAll('[data-dock-toggle]')) b.onclick = () => toggle(b.dataset.dockToggle);
  // At most one open in a drawer group (the first open, in panel order) after a resize.
  function fit() {
    const first = ['scene', 'view', 'results'].find((k) => st[k]);
    if (first) exclusive(first);
    apply();
  }

  // The columns end above the playback bar and the HUD lines over it.
  const root = document.documentElement.style;
  function relayout() {
    const bar = secs.playback.getBoundingClientRect();
    const barTop = Math.max(0, innerHeight - bar.top);
    root.setProperty('--lab-hud', `${Math.round(barTop + 8)}px`);
    const hud = Math.max(opts.counts?.offsetHeight ?? 0, opts.legend?.offsetHeight ?? 0);
    root.setProperty('--lab-reserve', `${Math.round(barTop + 8 + (hud ? hud + 8 : 0))}px`);
  }
  addEventListener('resize', fit);
  if (typeof ResizeObserver === 'function') {
    const ro = new ResizeObserver(() => relayout());
    for (const e of [secs.playback, opts.counts, opts.legend]) if (e) ro.observe(e);
  }

  // Shift+S/V/R/P (capture, ahead of the lab's and the page's keys).
  addEventListener('keydown', (e) => {
    const t = e.target;
    if (!e.shiftKey || e.ctrlKey || e.metaKey || e.altKey || e.repeat || t?.isContentEditable || ['INPUT', 'SELECT', 'TEXTAREA'].includes(t?.tagName)) return;
    const name = Object.keys(KEYS).find((k) => e.code === `Key${KEYS[k]}`);
    if (!name) return;
    e.preventDefault(); e.stopImmediatePropagation();
    if (opts.focus()) opts.setFocus(false);
    toggle(name);
  }, { capture: true });

  fit();
  const api = {
    bodies,
    sections: secs,
    /** The line a panel shows beside its name (collapsed or not). */
    summary(name, html) { const s = secs[name]?.querySelector('.lab-toggle .s'); if (s && s.innerHTML !== html) s.innerHTML = html; },
    open, toggle, relayout,
    state: () => ({ ...st }),
    restore(s) { st = { ...DEFAULT, ...s }; save(st); apply(); },
  };
  window.__labUi = api;
  return api;
}
