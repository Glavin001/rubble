// Crash-test lab UI for softbody.html ?solver=explicit: time control (slow motion,
// pause, frame and physics-step stepping, restart), impact detection and the time
// readout, a recorded run to scrub back through, the studio HUD (scene pills, node
// and beam counts, strain legend), view toggles, camera presets, a clean recording
// mode and BeamNG-style node grabbing.
//
// Physics stays exact: slow motion only takes fewer 0.5 ms steps per displayed
// frame; impact detection and the grab are the solver's (ExplicitSim
// readFrameState / setGrab). This module wires UI to them and to the renderer
// (crash-lab-render.mjs).

import { COLOR_MODES, SKIN_MODES, PART_COLORS, partCss, strainGradientCss } from './crash-lab-render.mjs';
import { createLabLayout } from './crash-lab-ui.mjs';
import { pickerOptions } from './crash-lab-scene.mjs';

// The one-click presets (wall56, pole29, ...) and the lab's scenes are Rust's catalogue
// (engine-world-wasm lab_scenes.rs, `crashLabCatalog`); crash-lab-scene.mjs `labTables`
// gives the page its tables from it.

/** Slow-motion factors (displayed time per simulated time). */
export const SLOW = [1, 2, 4, 10, 20, 50];

/** Camera presets: orbit yaw (0 = in front, +Z), pitch, distance as a multiple of the scene's size. */
export const CAMS = {
  side: { yaw: Math.PI / 2, pitch: 0.14, dist: 0.95 },
  top: { yaw: Math.PI / 2, pitch: 1.5, dist: 1.4 },
  three: { yaw: 2.25, pitch: 0.3, dist: 1.05 },
  // The other three-quarter (from ahead and to the right of +Z travel).
  threeFront: { yaw: 0.6, pitch: 0.3, dist: 1.05 },
  front: { yaw: 0.0, pitch: 0.12, dist: 1.3 },
};

const STEP_RATE = 2000;
const RECORD_FRAMES = 1200;
const RECORD_BYTES = 384 << 20;

const fmtInt = (n) => n.toLocaleString('en-US');

const CSS = `
.lab-ui { position: absolute; pointer-events: none; font-family: "Inter", system-ui, sans-serif; color: #e9eef7; letter-spacing: 0.04em; }
#labPills { top: 18px; left: 20px; display: flex; flex-direction: column; gap: 8px; align-items: flex-start; }
.lab-pill { background: rgba(14, 22, 40, 0.72); border: 1px solid rgba(160, 190, 255, 0.22); border-radius: 999px; padding: 6px 14px; font-weight: 650; font-size: 14px; letter-spacing: 0.12em; backdrop-filter: blur(6px); }
.lab-pill.sub { font-weight: 500; font-size: 12px; color: #aab8d4; }
#labTime { top: 14px; right: 22px; text-align: right; }
#labTime .big { font-size: 64px; font-weight: 750; line-height: 1; font-variant-numeric: tabular-nums; letter-spacing: 0.01em; text-shadow: 0 2px 18px rgba(0,0,0,0.5); }
#labTime .big.pre { color: #8fa0bf; }
#labTime .sub { margin-top: 6px; font-size: 12px; font-weight: 600; letter-spacing: 0.18em; color: #9fb0cf; }
#labCounts { left: 22px; bottom: 18px; font-weight: 750; font-size: 30px; line-height: 1.1; letter-spacing: 0.02em; text-shadow: 0 2px 14px rgba(0,0,0,0.5); }
#labCounts small { display: block; font-size: 12px; font-weight: 600; letter-spacing: 0.18em; color: #9fb0cf; margin-top: 4px; }
#labLegend { right: 22px; bottom: 20px; display: flex; align-items: center; gap: 10px; font-size: 12px; font-weight: 650; letter-spacing: 0.16em; color: #c4cfe4; }
#labLegend .bar { width: 170px; height: 10px; border-radius: 5px; }
#labLegend .chip { display: inline-flex; align-items: center; gap: 5px; margin-left: 8px; }
#labLegend .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
#labScrub { flex: 1; min-width: 120px; }
#labPanel .sep { width: 1px; height: 18px; background: #2c3b5c; margin: 0 2px; }
#labPanel .tag { color: #ffcf5a; font-weight: 700; letter-spacing: 0.1em; }
#labPanel .tag:empty { display: none; }
.lab-clean #info, .lab-clean #hud, .lab-clean #status, .lab-clean #labPanel { display: none !important; }
/* While the engine loads: the panels' final places, their controls to come; the progress
   card centred over the 3D area right of the left column (never over a panel). */
.lab-shell-wait { color: #9aa6be; padding: 4px 0; }
body.lab-on .rb-card { left: calc(50% + 150px); min-width: min(560px, calc(100vw - 340px)); max-width: min(720px, calc(100vw - 340px)); }
`;

/**
 * The lab's frame before the engine loads (softbody.html calls it at once in lab mode): the
 * HUD's places, the panels (crash-lab-ui.mjs) with a line saying what they will hold, the
 * raw page panels hidden. `createCrashLab({ shell })` fills it. `title`: the scene's name
 * when known.
 */
export function createLabShell({ title = '' } = {}) {
  const style = document.createElement('style');
  style.textContent = CSS;
  document.head.appendChild(style);
  document.body.classList.add('lab-on');
  const el = (html) => { const t = document.createElement('template'); t.innerHTML = html.trim(); return document.body.appendChild(t.content.firstChild); };
  const pillsEl = el('<div class="lab-ui" id="labPills"></div>');
  const timeEl = el('<div class="lab-ui" id="labTime"><div class="big pre">0.000 s</div><div class="sub">LOADING</div></div>');
  const countsEl = el('<div class="lab-ui" id="labCounts"></div>');
  const legendEl = el('<div class="lab-ui" id="labLegend"></div>');
  let clean = false;
  const shell = { pillsEl, timeEl, countsEl, legendEl, onFocus: null };
  shell.ui = createLabLayout({ pills: pillsEl, time: timeEl, counts: countsEl, legend: legendEl, focus: () => shell.focus?.() ?? clean, setFocus: (on) => (shell.onFocus ? shell.onFocus(on) : (clean = on)) });
  const wait = (t) => `<div class="lab-shell-wait">${t}</div>`;
  shell.ui.bodies.scene.innerHTML = wait(title ? `${title.replace(/[&<>]/g, '')}: loading…` : 'Loading the scene…');
  shell.ui.bodies.view.innerHTML = wait('Cameras and drawing: once the engine is up.');
  shell.ui.bodies.playback.innerHTML = `<div class="r">${['restart', 'pause', 'frame ▸', 'step ▸'].map((t) => `<button disabled>${t}</button>`).join('')}<span class="lab-shell-wait">loading the engine…</span></div>`;
  shell.ui.bodies.results.innerHTML = wait('Measured once the run starts.');
  // Not a lab page after all (the scene turned out to be a plain view): the page as it was.
  shell.destroy = () => { document.body.classList.remove('lab-on'); for (const id of ['labLeft', 'labRight', 'labPanel', 'labUnfocus', 'labCounts', 'labLegend']) document.getElementById(id)?.remove(); style.remove(); };
  shell.setTitle = (t) => { const w = shell.ui.bodies.scene.querySelector('.lab-shell-wait'); if (w) w.textContent = `${t}: loading…`; };
  shell.ui.relayout();
  return shell;
}

/**
 * @param {object} ctx
 * @param {() => any} ctx.sim the current ExplicitSim
 * @param {() => any} ctx.layout its layout()
 * @param {any} ctx.renderer createCrashLabRenderer(...)
 * @param {HTMLCanvasElement} ctx.canvas
 * @param {URLSearchParams} ctx.qs
 * @param {string | null} ctx.preset LAB_PRESETS key
 * @param {string} ctx.xscene
 * @param {(o: object) => void} ctx.setOrbit orbit camera ({ yaw, pitch, dist, target }), or { chase: true }
 * @param {() => {vp: Float32Array, eye: number[]}} ctx.view the last frame's view-projection and eye
 * @param {(m: Float32Array) => Float32Array} ctx.inverse 4×4 inverse
 * @param {() => void} ctx.restart rebuild the scene
 * @param {(e: any) => void} ctx.fail
 * @param {boolean} [ctx.drive] the scene is driven (WASD, Space handbrake, chase camera)
 * @param {any} ctx.pacer the page's FramePacer (engine-world-wasm): steps per displayed frame
 */
export function createCrashLab(ctx) {
  const { qs, canvas, renderer } = ctx;
  // The frame built before the engine loaded (createLabShell), or one now.
  const shell = ctx.shell ?? createLabShell();
  const { pillsEl, timeEl, countsEl, legendEl, ui } = shell;
  shell.focus = () => st.clean;
  shell.onFocus = (on) => setClean(on);
  for (const b of Object.values(ui.bodies)) b.replaceChildren();
  const preset = ctx.preset ? ctx.presets?.[ctx.preset] : null;
  // A crash reference run's first contact (a step): its scenario's, which a change of its
  // values moves.
  const impactAt = () => (typeof ctx.impactAt === 'function' ? ctx.impactAt() : ctx.impactAt);
  // 'body' only when the page draws an authored body (ctx.body).
  const skinModes = ctx.body ? SKIN_MODES : SKIN_MODES.filter((m) => m !== 'body');

  // ---------------------------------------------------------------- state
  const P = (k, d) => (qs.has(k) && qs.get(k) !== '' ? qs.get(k) : d);
  const slowParam = P('slow', preset || ctx.crash ? 'auto' : '1');
  const st = {
    auto: slowParam === 'auto',
    slow: slowParam === 'auto' ? Number(P('autoSlow', 20)) : Math.max(1, Number(slowParam) || 1),
    paused: qs.get('paused') === '1',
    pending: 0, // steps queued by the step buttons
    impactStep: null,
    lastState: null,
    reads: 0,
    ring: [],
    ringBytes: 0,
    replay: -1, // index into ring, -1 = live
    grabTool: false,
    grab: null,
    grabK: Number(P('grabK', 5e4)),
    spin: qs.get('spin') === '1',
    // Driving (ctx.drive): the chase camera.
    cam: P('cam3', preset?.cam ?? (ctx.drive ? 'follow' : 'three')),
    clean: qs.get('clean') === '1',
    focus: null,
    size: 5,
  };
  renderer.set({
    colorMode: COLOR_MODES.includes(qs.get('color')) ? qs.get('color') : 'strain',
    skin: skinModes.includes(qs.get('skin3')) ? qs.get('skin3') : ctx.body ? 'body' : 'ghost',
    // With an authored body the skeleton and nodes start hidden (K, N show them).
    skeleton: qs.has('skeleton') ? qs.get('skeleton') !== '0' : !ctx.body,
    nodes: qs.has('nodes') ? qs.get('nodes') !== '0' : !ctx.body,
    stubs: qs.get('stubs') !== '0',
    studio: qs.get('studio') !== '0',
    tubeR: Number(P('tube', 0.009)),
    nodeR: Number(P('node', 0.017)),
  });
  if (st.clean) document.body.classList.add('lab-clean');

  // ---------------------------------------------------------------- panels
  // The layout (crash-lab-ui.mjs: each panel's one responsibility): the scene picker in
  // Scene, time in Playback, how it is drawn and from where in View. The page and
  // crash-ref.mjs add theirs (lab.ui.bodies).
  // Grouped by scenario family (crash-lab-scene.mjs): each family's reference crashes, the
  // lab's presets and scenes that stage it, and the family itself to configure.
  const current = ctx.familyId ? `f:${ctx.familyId}` : ctx.refId ? `r:${ctx.refId}` : ctx.preset ? `p:${ctx.preset}` : `s:${ctx.xscene}`;
  const sceneOpts = pickerOptions({ registry: ctx.registry ?? { scenarios: ctx.refs ?? [], families: [] }, catalog: ctx.catalog ?? { scenes: [], presets: [] }, current });
  const into = (parent, html) => { const t = document.createElement('template'); t.innerHTML = html.trim(); parent.append(...t.content.childNodes); };
  into(ui.bodies.scene, `<div class="r"><select id="labPreset" aria-label="Scene" title="What is simulated, by kind of crash: its reference crashes, the lab's presets and scenes, or the family itself to configure (its parameters below)">${sceneOpts}</select></div>`);
  into(ui.bodies.playback, `<div class="r">
      <button id="labRestart" title="Restart the run (R)" aria-keyshortcuts="R">restart</button>
      <button id="labPlay" title="Play or pause (Space)" aria-keyshortcuts="Space">pause</button>
      <button id="labFrame" title="One displayed frame (.)" aria-keyshortcuts=".">frame ▸</button>
      <button id="labStep" title="One 0.5 ms physics step (,)" aria-keyshortcuts=",">step ▸</button>
      <span class="sep"></span>
      <label for="labSlow">slow</label><select id="labSlow">${['auto', ...SLOW].map((s) => `<option value="${s}">${s === 'auto' ? 'auto (1x → 20x at impact)' : s === 1 ? '1x (real time)' : `${s}x slower`}</option>`).join('')}</select>
      <span class="sep"></span>
      <input id="labScrub" type="range" min="0" max="0" value="0" step="1" aria-label="Recorded run" title="Recorded run: drag to scrub back" />
      <button id="labLive" title="Back to the live simulation">live</button>
      <span id="labReplayTag" class="tag"></span>
      <span class="sep"></span>
      <button id="labClean" title="Focus: hide every panel, keep the view and the clock (H; Esc leaves)" aria-keyshortcuts="H">focus</button>
    </div>`);
  // The one camera group: the scene's own cameras (a reference crash's), the presets, the
  // chase, auto-orbit; the page adds its car-relative views (#labCamMore).
  const refCams = Object.keys(ctx.cams ?? {});
  into(ui.bodies.view, `<div role="group" aria-label="Camera" id="labCams">
    ${refCams.length ? `<div class="r"><span class="lbl" title="The reference crash's own cameras">scenario</span>${refCams.map((c) => `<button data-cam="${c}" title="The scenario's own camera (fixed)">${c.replace(/^ref:/, '')}</button>`).join('')}</div>` : ''}
    <div class="r"><span class="lbl" title="Cameras that follow the vehicles">camera</span>
      ${['side', 'top', 'three', 'front'].map((c, i) => `<button data-cam="${c}" title="${c === 'three' ? 'Three-quarter' : c[0].toUpperCase() + c.slice(1)} view, following the vehicles (${i + 1})" aria-keyshortcuts="${i + 1}">${c === 'three' ? '¾' : c}</button>`).join('')}
      <button id="labFollow" title="Chase the vehicle (5)" aria-keyshortcuts="5">follow</button>
      <button id="labSpin" title="Orbit slowly around the subject (O)" aria-keyshortcuts="O">auto-orbit</button>
    </div>
    <div class="r" id="labCamMore"><span class="lbl" title="Cameras on the vehicle">on car</span></div>
    </div>
    <div class="r"><span class="lbl">draw</span>
      <button id="labSkel" title="The beams (K or V)" aria-keyshortcuts="K">skeleton</button>
      <button id="labNodes" title="The nodes (N)" aria-keyshortcuts="N">nodes</button>
      <button id="labStubs" title="Broken-beam stubs">stubs</button>
      <button id="labStudio" title="The studio floor and backdrop">studio</button>
    </div>
    <div class="r"><label class="lbl" for="labColor">colour</label><select id="labColor" title="Beam colour (B cycles)">${COLOR_MODES.map((m) => `<option>${m}</option>`).join('')}</select>
      <label for="labSkin">skin</label><select id="labSkin" title="The body drawn (M cycles)">${skinModes.map((m) => `<option>${m}</option>`).join('')}</select></div>
    <div class="r"><span class="lbl">grab</span>
      <button id="labGrab" title="Drag a node on a spring (G, or Ctrl+drag a node)" aria-keyshortcuts="G">grab</button>
      <label for="labGrabK">strength</label><input id="labGrabK" type="range" min="3" max="6.5" step="0.05" style="width:90px" />
      <span id="labGrabKv"></span>
    </div>`);
  const view = ui.bodies.view;
  const $ = (id) => document.getElementById(id);
  const toggle = (b, on) => { b.classList.toggle('on', !!on); if (b.tagName === 'BUTTON') b.setAttribute('aria-pressed', String(!!on)); };
  const syncPanel = () => {
    $('labPlay').textContent = st.paused || st.replay >= 0 ? 'play ▶' : 'pause ❚❚';
    $('labSlow').value = st.auto ? 'auto' : String(st.slow);
    const o = renderer.opts;
    toggle($('labSkel'), o.skeleton); toggle($('labNodes'), o.nodes); toggle($('labStubs'), o.stubs); toggle($('labStudio'), o.studio);
    $('labColor').value = o.colorMode; $('labSkin').value = o.skin;
    toggle($('labGrab'), st.grabTool); toggle($('labSpin'), st.spin);
    // The active camera: a preset or a scenario camera while the page orbits, follow while
    // it chases (the page's car-relative views mark their own select).
    const mode = ctx.camMode?.() ?? (st.cam === 'follow' ? 'chase' : 'orbit');
    for (const b of view.querySelectorAll('[data-cam]')) toggle(b, mode === 'orbit' && b.dataset.cam === st.cam);
    toggle($('labFollow'), mode === 'chase');
    $('labGrabK').value = Math.log10(st.grabK);
    $('labGrabKv').textContent = `${st.grabK >= 1e6 ? (st.grabK / 1e6).toFixed(1) + ' MN/m' : (st.grabK / 1e3).toFixed(0) + ' kN/m'}`;
    $('labReplayTag').textContent = st.replay >= 0 ? 'REPLAY' : '';
    pills();
    legend();
  };
  // After a click the shortcuts go back to the page (a focused control keeps its own
  // keys); a control reached by the keyboard keeps the focus.
  let pointerAt = -1e9;
  addEventListener('pointerdown', () => { pointerAt = performance.now(); }, { capture: true });
  const byPointer = () => performance.now() - pointerAt < 1500;
  const blur = (e) => { if (byPointer()) e?.target?.blur?.(); };
  $('labPreset').onchange = (e) => {
    const [kind, k] = e.target.value.split(':');
    if (kind === 's') { location.href = ctx.sceneUrl(k); return; }
    if (kind === 'r') { location.href = ctx.refUrl(k); return; }
    if (kind === 'f') { location.href = ctx.familyUrl(k); return; }
    // The page's vehicle and options (softbody.html keepVehicle) carry across.
    const u = new URL(location.href); const s = new URLSearchParams(); window.__keepVehicle?.(s); s.set('solver', 'explicit'); s.set('preset', k); u.search = s.toString(); location.href = u.href;
  };
  $('labPlay').onclick = (e) => { playPause(); blur(e); };
  $('labFrame').onclick = (e) => { frameStep(); blur(e); };
  $('labStep').onclick = (e) => { physicsStep(); blur(e); };
  $('labRestart').onclick = (e) => { restart(); blur(e); };
  $('labSlow').onchange = (e) => { const v = e.target.value; st.auto = v === 'auto'; if (!st.auto) st.slow = Number(v); else st.slow = Number(P('autoSlow', 20)); syncPanel(); blur(e); };
  $('labScrub').oninput = (e) => { enterReplay(Number(e.target.value)); };
  $('labLive').onclick = (e) => { st.replay = -1; syncPanel(); blur(e); };
  $('labSkel').onclick = (e) => { renderer.set({ skeleton: !renderer.opts.skeleton }); syncPanel(); blur(e); };
  $('labNodes').onclick = (e) => { renderer.set({ nodes: !renderer.opts.nodes }); syncPanel(); blur(e); };
  $('labStubs').onclick = (e) => { renderer.set({ stubs: !renderer.opts.stubs }); syncPanel(); blur(e); };
  $('labStudio').onclick = (e) => { renderer.set({ studio: !renderer.opts.studio }); syncPanel(); blur(e); };
  $('labColor').onchange = (e) => { renderer.set({ colorMode: e.target.value }); syncPanel(); blur(e); };
  $('labSkin').onchange = (e) => { renderer.set({ skin: e.target.value }); syncPanel(); blur(e); };
  for (const b of view.querySelectorAll('[data-cam]')) b.onclick = (e) => { camera(b.dataset.cam); blur(e); };
  $('labFollow').onclick = (e) => { camera('follow'); blur(e); };
  $('labSpin').onclick = (e) => { st.spin = !st.spin; syncPanel(); blur(e); };
  $('labGrab').onclick = (e) => { st.grabTool = !st.grabTool; syncPanel(); blur(e); };
  $('labGrabK').oninput = (e) => { st.grabK = 10 ** Number(e.target.value); syncPanel(); };
  // From the keyboard the focus moves to the (otherwise invisible) way back.
  $('labClean').onclick = (e) => { const kb = !byPointer(); setClean(true); blur(e); if (kb) $('labUnfocus')?.focus(); };

  // ---------------------------------------------------------------- HUD
  function pills() {
    const custom = qs.get('pills');
    // A preset's own pills while its values are the preset's; else the scene's, from its values
    // now (ctx.pill may be a function of them).
    const own = typeof ctx.pill === 'function' ? ctx.pill() : ctx.pill;
    const list = custom ? custom.split(',').map((s) => s.trim()).filter(Boolean)
      : [...((!ctx.modified?.() && preset?.pills) || [own ?? ctx.xscene.toUpperCase()])];
    const o = renderer.opts;
    const sub = custom ? [] : [o.skin === 'off' ? (o.nodes ? 'NODE + BEAM' : 'THE SKELETON') : o.skin === 'ghost' ? 'THE SKELETON · GHOST BODY' : o.skin === 'body' ? 'BODY SKINNED TO THE NODES' : 'COLLISION SKIN'];
    pillsEl.innerHTML = list.map((t) => `<div class="lab-pill">${t}</div>`).join('') + sub.map((t) => `<div class="lab-pill sub">${t}</div>`).join('');
  }
  function legend() {
    const o = renderer.opts;
    if (!o.skeleton) { legendEl.innerHTML = ''; return; }
    if (o.colorMode === 'strain') legendEl.innerHTML = `<span>BEAM STRAIN</span><span class="bar" style="background:${strainGradientCss()}"></span><span>BREAKS</span>`;
    else if (o.colorMode === 'damage') legendEl.innerHTML = `<span>BEAMS</span>${[['INTACT', '#7587a1'], ['BENT', 'linear-gradient(90deg,#ffd91a,#ff1a0d)'], ['BROKEN', '#f21ae6']].map(([t, c]) => `<span class="chip"><span class="dot" style="background:${c}"></span>${t}</span>`).join('')}`;
    else {
      const names = ctx.sim()?.labParts() ?? [];
      legendEl.innerHTML = `<span>PARTS</span>${names.map((n, i) => `<span class="chip"><span class="dot" style="background:${partCss(i)}"></span>${n.toUpperCase()}</span>`).join('')}`;
    }
  }
  function counts() {
    const l = ctx.layout();
    const v = l.vehicles;
    // One vehicle's counts (the reference shots); a pair shows "2 x".
    const same = v.every((x) => x.nodes === v[0].nodes && x.beams === v[0].beams);
    const pre = v.length > 1 && same ? `${v.length} x ` : '';
    const n = same ? v[0].nodes : l.nodes, b = same ? v[0].beams : l.beams;
    countsEl.innerHTML = `${pre}${fmtInt(n)} NODES / ${fmtInt(b)} BEAMS<small>${v[0].name.split(' (')[0].toUpperCase()}</small>`;
  }
  const subtitle = () => {
    const tail = 'FROM THE SIMULATION';
    if (st.replay >= 0) return `REPLAY · ${tail}`;
    if (st.paused) return `PAUSED · ${tail}`;
    const slow = slowNow();
    // Below real pace (a GPU that cannot keep up: the steps capped): the share shown.
    const sp = ctx.pacer.speed();
    const pace = sp < 0.97 ? ` · ${Math.round(sp * 100)} % PACE (GPU)` : '';
    return `${slow === 1 ? 'REAL TIME' : `${slow}x SLOWER`}${pace} · ${tail}`;
  };
  /** The time readout: ms from the first impact (− before it), else time since the start. */
  function timeText(step) {
    if (st.impactStep != null) {
      const ms = (step - st.impactStep) * 1000 / STEP_RATE;
      const r = Math.round(ms);
      return { text: `${r > 0 ? '+' : r < 0 ? '−' : ''}${Math.abs(r)} ms`, pre: false, ms };
    }
    return { text: `${(step / STEP_RATE).toFixed(3)} s`, pre: true, ms: null };
  }
  function updateHud() {
    const step = st.replay >= 0 ? st.ring[st.replay].step : ctx.sim().steps();
    const t = timeText(step);
    const big = timeEl.firstChild;
    big.textContent = t.text;
    big.classList.toggle('pre', t.pre);
    timeEl.lastChild.textContent = t.pre && st.replay < 0 ? `${subtitle()} · AWAITING IMPACT` : subtitle();
    const scrub = $('labScrub');
    scrub.max = String(Math.max(0, st.ring.length - 1));
    if (st.replay < 0) scrub.value = scrub.max;
  }

  // ---------------------------------------------------------------- time control
  function playPause() {
    if (st.replay >= 0) { st.replay = -1; st.paused = false; } else st.paused = !st.paused;
    syncPanel();
  }
  const frameSteps = () => Math.max(1, Math.round(STEP_RATE / 60 / (st.auto && st.impactStep == null ? 1 : st.slow)));
  function frameStep() { st.replay = -1; st.paused = true; st.pending += frameSteps(); syncPanel(); }
  function physicsStep() { st.replay = -1; st.paused = true; st.pending += 1; syncPanel(); }
  function clearRun() {
    releaseGrab();
    st.impactStep = null; st.ring = []; st.ringBytes = 0; st.replay = -1; st.pending = 0; st.lastState = null; st.focus = null; st.focusAt = null;
  }
  function restart() {
    clearRun();
    ctx.restart();
    ctx.pacer.reset(slowNow());
    syncPanel();
  }
  /** The slow-motion factor asked for now (auto: real time until the impact). */
  const slowNow = () => (st.auto && st.impactStep == null ? 1 : st.slow);
  /** Steps to take this displayed frame (`dt` wall seconds), at most `max`: the
   * page's FramePacer (Rust: accumulator, caps, eased slow motion). */
  function stepsFor(dt, max) {
    if (st.replay >= 0) return 0;
    if (st.paused) { const k = Math.min(st.pending, max); st.pending -= k; return k; }
    return ctx.pacer.advance(dt, slowNow(), max);
  }
  /** Chunks of a frame's steps: before the impact in auto mode, small chunks so the
   * impact read lands within a few milliseconds of it. */
  function chunks(steps) {
    // A crash reference run (ctx.sampleSteps): every chunk ends on a sample's step (the
    // page reads each sample's state after its chunk).
    if (ctx.sampleSteps) {
      const out = [];
      for (let at = ctx.sim().steps(), left = steps; left > 0;) {
        const k = Math.min(left, ctx.sampleSteps - (at % ctx.sampleSteps));
        out.push(k); at += k; left -= k;
      }
      return out;
    }
    if (!(st.auto && st.impactStep == null) || steps <= 8) return [steps];
    const l = ctx.layout();
    const size = l.nodes * 48 + l.beams * 32 > 200000 ? 17 : 8;
    const out = [];
    for (let s = steps; s > 0; s -= size) out.push(Math.min(size, s));
    return out;
  }
  // Frame-state reads: at most 6 in flight; each resolves with the state after the
  // frames submitted when it was issued.
  let inFlight = 0;
  function read() {
    const sim = ctx.sim();
    if (inFlight >= 6) return null;
    inFlight++;
    let p;
    try { p = sim.readFrameState(); } catch (e) { inFlight--; ctx.fail(e); return null; }
    return p.then((s) => {
      inFlight--;
      if (sim !== ctx.sim()) return;
      st.reads++;
      onState(s);
    }, (e) => { inFlight--; ctx.fail(e); });
  }
  function onState(s) {
    // A crash reference run's first contact is its scenario's (ctx.impactAt, a step).
    const at = impactAt();
    if (st.impactStep == null && (at != null ? s.step >= at : s.impact)) { st.impactStep = at ?? s.step; syncPanel(); }
    if (!st.lastState || s.step >= st.lastState.step) st.lastState = s;
    // The scene's focus: the box centre of every node (a flung piece ignored by
    // using the 2nd–98th percentile would cost a sort; the box is fine for presets).
    const p = s.pos;
    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < p.length; i += 3) for (let a = 0; a < 3; a++) { const x = p[i + a]; if (x < lo[a]) lo[a] = x; if (x > hi[a]) hi[a] = x; }
    const c = lo.map((x, a) => (x + hi[a]) / 2);
    if (c.every(Number.isFinite) && (!st.focusAt || s.step >= st.focusAt.step)) {
      // The focus's velocity from the previous read (the camera carries the focus
      // forward to the step on display: reads arrive a few frames late).
      // Over at least 24 steps (reads come every 8 or 17 steps before the impact:
      // the box centre of a deforming body is noisy over fewer).
      const f = st.focusAt, ref = f?.ref && s.step - f.ref.step <= 400 ? f.ref : f;
      const dstep = ref ? s.step - ref.step : 0;
      const vel = dstep > 0 ? c.map((x, a) => (x - ref.p[a]) * STEP_RATE / dstep) : (f?.v ?? [0, 0, 0]);
      const keep = ref && s.step - ref.step < 24 ? ref : { p: c, step: s.step };
      st.focusAt = { p: c, v: dstep >= 24 || !f?.v ? vel : f.v, step: s.step, ref: keep };
      st.focus = c;
      st.size = Math.max(4, Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]));
    }
    // Record (one entry per step reached).
    const last = st.ring[st.ring.length - 1];
    if (!last || s.step > last.step) {
      const bytes = s.pos.byteLength + s.rest.byteLength;
      st.ring.push({ step: s.step, pos: s.pos, rest: s.rest, bent: s.bent, broken: s.broken });
      st.ringBytes += bytes;
      while (st.ring.length > RECORD_FRAMES || st.ringBytes > RECORD_BYTES) {
        const x = st.ring.shift();
        st.ringBytes -= x.pos.byteLength + x.rest.byteLength;
        if (st.replay > 0) st.replay--;
      }
    }
  }
  function enterReplay(i) {
    if (!st.ring.length) return;
    releaseGrab();
    st.replay = Math.max(0, Math.min(st.ring.length - 1, i));
    syncPanel();
  }

  // ---------------------------------------------------------------- camera
  // `zoom` scales the preset's distance, `yaw` overrides its yaw (the
  // recorder's close-ups).
  function camera(name, zoom = 1, yaw = null) {
    st.cam = name;
    st.camZoom = zoom;
    st.camYaw = yaw;
    if (name === 'follow') { ctx.setOrbit({ chase: true }); syncPanel(); return; }
    // A fixed camera of the scene's own (ctx.cams: a crash reference scenario's eye and
    // target, world coordinates): it does not follow the focus.
    const fixed = ctx.cams?.[name];
    if (fixed) { ctx.setOrbit({ ...fixed, dist: fixed.dist * zoom }); syncPanel(); return; }
    const c = CAMS[name];
    ctx.setOrbit({ yaw: yaw ?? c.yaw, pitch: c.pitch, dist: c.dist * st.size * zoom, target: null });
    syncPanel();
  }

  // ---------------------------------------------------------------- grab
  function releaseGrab() {
    if (!st.grab) return;
    st.grab = null;
    try { ctx.sim().clearGrab(); } catch (e) { ctx.fail(e); }
    renderer.set({ grab: null });
  }
  const ndcOf = (e) => {
    const r = canvas.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * 2 - 1, 1 - ((e.clientY - r.top) / r.height) * 2];
  };
  const mul4 = (m, v) => [0, 1, 2, 3].map((r) => m[r] * v[0] + m[4 + r] * v[1] + m[8 + r] * v[2] + m[12 + r] * v[3]);
  function ray(ndc) {
    const inv = ctx.inverse(ctx.view().vp);
    const a = mul4(inv, [ndc[0], ndc[1], 0, 1]), b = mul4(inv, [ndc[0], ndc[1], 1, 1]);
    const p0 = a.slice(0, 3).map((x) => x / a[3]), p1 = b.slice(0, 3).map((x) => x / b[3]);
    const d = p1.map((x, i) => x - p0[i]); const l = Math.hypot(...d);
    return { o: p0, d: d.map((x) => x / l) };
  }
  /** The node nearest the screen point (within `px` pixels), from the last frame state. */
  function pick(ndc, px = 24) {
    const s = st.lastState;
    if (!s) return null;
    const vp = ctx.view().vp, r = canvas.getBoundingClientRect();
    let best = null;
    for (let i = 0, n = s.pos.length / 3; i < n; i++) {
      const c = mul4(vp, [s.pos[3 * i], s.pos[3 * i + 1], s.pos[3 * i + 2], 1]);
      if (c[3] <= 0) continue;
      const dx = (c[0] / c[3] - ndc[0]) * r.width / 2, dy = (c[1] / c[3] - ndc[1]) * r.height / 2;
      const d2 = dx * dx + dy * dy, z = c[2] / c[3];
      if (d2 > px * px) continue;
      // Nearest on screen, the closer to the camera among near-ties.
      const score = Math.sqrt(d2) + z * 4;
      if (!best || score < best.score) best = { i, score };
    }
    return best?.i ?? null;
  }
  function grabNode(global) {
    const l = ctx.layout();
    let v = 0;
    for (let k = 0; k < l.vehicles.length; k++) if (global >= l.vehicles[k].nodeOffset) v = k;
    const p = [0, 1, 2].map((a) => st.lastState.pos[3 * global + a]);
    const eye = ctx.view().eye;
    const n = p.map((x, i) => x - eye[i]); const nl = Math.hypot(...n);
    st.grab = { global, vehicle: v, node: global - l.vehicles[v].nodeOffset, plane: { p, n: n.map((x) => x / nl) }, target: p };
    applyGrab();
  }
  function applyGrab() {
    const g = st.grab;
    if (!g) return;
    ctx.sim().setGrab(g.vehicle, g.node, g.target[0], g.target[1], g.target[2], st.grabK);
    renderer.set({ grab: { node: g.global, target: g.target } });
  }
  function dragTo(ndc) {
    const g = st.grab;
    if (!g) return;
    const { o, d } = ray(ndc);
    const den = d[0] * g.plane.n[0] + d[1] * g.plane.n[1] + d[2] * g.plane.n[2];
    if (Math.abs(den) < 1e-6) return;
    const t = ((g.plane.p[0] - o[0]) * g.plane.n[0] + (g.plane.p[1] - o[1]) * g.plane.n[1] + (g.plane.p[2] - o[2]) * g.plane.n[2]) / den;
    if (t <= 0) return;
    g.target = o.map((x, i) => x + d[i] * t);
    applyGrab();
  }
  // Capture phase: a grab press is taken before the page's orbit drag sees it.
  canvas.addEventListener('pointerdown', (e) => {
    if (!(e.ctrlKey || e.metaKey || st.grabTool) || e.button !== 0) return;
    if (st.replay >= 0) { st.replay = -1; syncPanel(); }
    const i = pick(ndcOf(e));
    if (i == null) return;
    e.stopImmediatePropagation();
    e.preventDefault();
    canvas.setPointerCapture?.(e.pointerId);
    grabNode(i);
  }, { capture: true });
  addEventListener('pointermove', (e) => { if (st.grab) dragTo(ndcOf(e)); });
  addEventListener('pointerup', () => releaseGrab());

  // ---------------------------------------------------------------- keys
  function setClean(on) {
    const back = !on && document.activeElement?.id === 'labUnfocus';
    st.clean = on; document.body.classList.toggle('lab-clean', on);
    if (back) $('labClean').focus();
    ui.relayout();
  }
  addEventListener('keydown', (e) => {
    if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT' || e.repeat) return;
    const k = e.code;
    // Driving: Space is the handbrake (the page's), not pause.
    if (k === 'Space' && !ctx.drive) { playPause(); e.preventDefault(); e.stopImmediatePropagation(); }
    else if (k === 'Period') frameStep();
    else if (k === 'Comma') physicsStep();
    else if (k === 'KeyH' || (k === 'Escape' && st.clean)) setClean(!st.clean);
    else if (k === 'KeyK' || k === 'KeyV') { renderer.set({ skeleton: !renderer.opts.skeleton }); syncPanel(); e.stopImmediatePropagation(); }
    else if (k === 'KeyN') { renderer.set({ nodes: !renderer.opts.nodes }); syncPanel(); }
    else if (k === 'KeyB') { const i = COLOR_MODES.indexOf(renderer.opts.colorMode); renderer.set({ colorMode: COLOR_MODES[(i + 1) % COLOR_MODES.length] }); syncPanel(); }
    else if (k === 'KeyM') { const i = skinModes.indexOf(renderer.opts.skin); renderer.set({ skin: skinModes[(i + 1) % skinModes.length] }); syncPanel(); e.stopImmediatePropagation(); }
    else if (k === 'KeyG') { st.grabTool = !st.grabTool; syncPanel(); }
    else if (k === 'KeyO') { st.spin = !st.spin; syncPanel(); }
    else if (k === 'KeyR') { restart(); e.stopImmediatePropagation(); }
    else if (k >= 'Digit1' && k <= 'Digit5') camera(['side', 'top', 'three', 'front', 'follow'][Number(k.slice(5)) - 1]);
  }, { capture: true });

  counts();
  syncPanel();
  let camSet = false;

  /** The orbit cameras follow the focus. A live frame (`ahead`: its lead, the
   * simulated seconds it shows past the last step) follows it where it is on
   * display: the last read carried forward by its velocity (at most 0.1 s).
   * Otherwise (`ahead` null: automation, frame-stepped; a replay) as read.
   * `spin` radians of yaw. */
  function followFocus(spin = 0, ahead = null) {
    if (!st.focus || st.cam === 'follow' || ctx.cams?.[st.cam]) return;
    const f = st.focusAt;
    const lead = f && ahead != null && st.replay < 0 ? Math.min(0.1, Math.max(0, ctx.sim().steps() - f.step) / STEP_RATE + ahead) : 0;
    const p = lead > 0 ? f.p.map((x, a) => x + f.v[a] * lead) : st.focus;
    ctx.setOrbit({ follow: p, spin });
  }

  return {
    st,
    stepsFor,
    followFocus,
    chunks,
    read,
    /** Per displayed frame (before drawing): replay upload or live gather (`lead`
     * simulated seconds past the last step: FramePacer.lead), HUD, camera. */
    beforeDraw(enc, dt, lead = null) {
      if (st.replay >= 0 && st.ring[st.replay]) renderer.upload(st.ring[st.replay]);
      else renderer.gather(enc, st.paused || lead == null ? 0 : lead);
      if (st.focus) {
        renderer.set({ focus: [st.focus[0], st.focus[2]] });
        if (!camSet) { camSet = true; camera(st.cam, st.camZoom ?? 1, st.camYaw ?? null); }
        followFocus(st.spin ? dt * 0.12 : 0, st.paused && lead != null ? 0 : lead);
      }
      if (st.grab) applyGrab();
      updateHud();
    },
    /** The scene was rebuilt (restart): rebind and recount. */
    rebuilt() { st.focusAt = null; renderer.setSim(ctx.sim(), ctx.layout()); counts(); legend(); },
    /** The scene was placed again with other values (the page rebuilt the sim): a new run
     * from its start, the camera kept. */
    replaced() { clearRun(); ctx.pacer.reset(slowNow()); st.focusAt = null; renderer.setSim(ctx.sim(), ctx.layout()); counts(); syncPanel(); },
    timeText,
    /** The solver step on display (the replayed frame's, else the live one). */
    shownStep: () => (st.replay >= 0 && st.ring[st.replay] ? st.ring[st.replay].step : ctx.sim().steps()),
    impactStep: () => st.impactStep,
    grabAt(global, target) { if (!st.lastState) return false; grabNode(global); st.grab.target = target; applyGrab(); return true; },
    release: releaseGrab,
    pick,
    playPause, frameStep, physicsStep, restart, enterReplay, camera, setClean,
    /** The layout (crash-lab-ui.mjs): its panels' bodies, summaries, open/close. */
    ui,
    setSlow(v) { st.auto = v === 'auto'; st.slow = st.auto ? 20 : v; syncPanel(); },
    setPaused(on) { st.paused = on; syncPanel(); },
    sync: syncPanel,
  };
}
