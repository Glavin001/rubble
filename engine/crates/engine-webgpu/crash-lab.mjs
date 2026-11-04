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

/** One-click presets: the explicit scene, its fields (XPARAMS keys), pills, camera. */
export const LAB_PRESETS = {
  wall56: { label: 'Wall · 56 km/h', xscene: 'wall', params: { speed: 56 }, pills: ['WALL · 56 KM/H'], cam: 'three' },
  // An oblique side pole: sliding at 75° to the car's axis into a pole at the door,
  // 0.8 m away (a car sliding on its tyres would stop short of the usual run-up).
  pole29: { label: 'Pole · 29 km/h', xscene: 'sidepole', params: { speed: 29, angle: 75, at: 0.3, runUp: 0.8 }, pills: ['POLE · 29 KM/H'], cam: 'threeFront' },
  headon50: { label: 'Head-on · 2 x 50 km/h', xscene: 'headon', params: { speed: 50, speed2: 50 }, pills: ['HEAD-ON · 2 x 50 KM/H'], cam: 'side' },
  drop15: { label: 'Dropped from 15 m', xscene: 'drop', params: { height: 15 }, pills: ['DROPPED FROM 15 M'], cam: 'side' },
};

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
#labPanel { position: absolute; left: 50%; bottom: 14px; transform: translateX(-50%); width: min(980px, calc(100vw - 32px)); background: rgba(10, 16, 30, 0.82); border: 1px solid rgba(160, 190, 255, 0.16); border-radius: 10px; padding: 8px 10px; font: 12px/1.4 system-ui, sans-serif; color: #dfe6f3; backdrop-filter: blur(8px); }
#labPanel .r { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 3px 0; }
#labPanel button, #labPanel select { background: #18233a; color: #e6ecf7; border: 1px solid #2c3b5c; border-radius: 6px; padding: 3px 8px; font: inherit; cursor: pointer; }
#labPanel button:hover { background: #22324f; }
#labPanel button.on { background: #2b4f8a; border-color: #4a78c4; }
#labPanel label { color: #9fb0cf; }
#labPanel input[type=range] { vertical-align: middle; }
#labPanel .sep { width: 1px; height: 18px; background: #2c3b5c; margin: 0 4px; }
#labScrub { flex: 1; min-width: 200px; }
#labPanel .tag { color: #ffcf5a; font-weight: 700; letter-spacing: 0.1em; }
.lab-clean #info, .lab-clean #hud, .lab-clean #status, .lab-clean #labPanel { display: none !important; }
body.lab-on #hud { top: 118px; }
body.lab-on #info { top: auto; bottom: 118px; max-width: 360px; font-size: 12px; opacity: 0.92; }
body.lab-on #infoText, body.lab-on #xkeys { display: none !important; }
body.lab-on #status { display: none; }
body.lab-on:not(.lab-clean) #labCounts, body.lab-on:not(.lab-clean) #labLegend { bottom: 118px; }
body.lab-on:not(.lab-clean) #labCounts { left: auto; right: 22px; bottom: 146px; text-align: right; font-size: 20px; }
`;

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
 */
export function createCrashLab(ctx) {
  const { qs, canvas, renderer } = ctx;
  const style = document.createElement('style');
  style.textContent = CSS;
  document.head.appendChild(style);
  document.body.classList.add('lab-on');
  const el = (html) => { const t = document.createElement('template'); t.innerHTML = html.trim(); return document.body.appendChild(t.content.firstChild); };
  const pillsEl = el('<div class="lab-ui" id="labPills"></div>');
  const timeEl = el('<div class="lab-ui" id="labTime"><div class="big">0 ms</div><div class="sub"></div></div>');
  const countsEl = el('<div class="lab-ui" id="labCounts"></div>');
  const legendEl = el('<div class="lab-ui" id="labLegend"></div>');
  const preset = ctx.preset ? LAB_PRESETS[ctx.preset] : null;

  // ---------------------------------------------------------------- state
  const P = (k, d) => (qs.has(k) && qs.get(k) !== '' ? qs.get(k) : d);
  const slowParam = P('slow', preset ? 'auto' : '1');
  const st = {
    auto: slowParam === 'auto',
    slow: slowParam === 'auto' ? Number(P('autoSlow', 20)) : Math.max(1, Number(slowParam) || 1),
    paused: qs.get('paused') === '1',
    pending: 0, // steps queued by the step buttons
    acc: 0,
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
    cam: P('cam3', preset?.cam ?? 'three'),
    clean: qs.get('clean') === '1',
    focus: null,
    size: 5,
  };
  renderer.set({
    colorMode: COLOR_MODES.includes(qs.get('color')) ? qs.get('color') : 'strain',
    skin: SKIN_MODES.includes(qs.get('skin3')) ? qs.get('skin3') : 'ghost',
    skeleton: qs.get('skeleton') !== '0',
    nodes: qs.get('nodes') !== '0',
    stubs: qs.get('stubs') !== '0',
    studio: qs.get('studio') !== '0',
    tubeR: Number(P('tube', 0.009)),
    nodeR: Number(P('node', 0.017)),
  });
  if (st.clean) document.body.classList.add('lab-clean');

  // ---------------------------------------------------------------- panel
  const panel = el(`<div id="labPanel">
    <div class="r">
      <select id="labPreset" title="Crash-test presets"><option value="">preset…</option>${Object.entries(LAB_PRESETS).map(([k, p]) => `<option value="${k}"${k === ctx.preset ? ' selected' : ''}>${p.label}</option>`).join('')}</select>
      <span class="sep"></span>
      <button id="labPlay" title="Space">pause</button>
      <button id="labFrame" title="One displayed frame (.)">frame ▸</button>
      <button id="labStep" title="One 0.5 ms physics step (,)">step ▸</button>
      <button id="labRestart" title="R">restart</button>
      <label>slow</label><select id="labSlow">${['auto', ...SLOW].map((s) => `<option value="${s}">${s === 'auto' ? 'auto (1x → 20x at impact)' : s === 1 ? '1x (real time)' : `${s}x slower`}</option>`).join('')}</select>
      <span class="sep"></span>
      <input id="labScrub" type="range" min="0" max="0" value="0" step="1" title="Recorded run: drag to scrub back" />
      <button id="labLive" title="Back to the live simulation">live</button>
      <span id="labReplayTag" class="tag"></span>
    </div>
    <div class="r">
      <button id="labSkel" title="K">skeleton</button>
      <button id="labNodes" title="N">nodes</button>
      <button id="labStubs" title="broken-beam stubs">stubs</button>
      <label>colour</label><select id="labColor">${COLOR_MODES.map((m) => `<option>${m}</option>`).join('')}</select>
      <label>skin</label><select id="labSkin">${SKIN_MODES.map((m) => `<option>${m}</option>`).join('')}</select>
      <button id="labStudio">studio</button>
      <span class="sep"></span>
      <label>camera</label>${['side', 'top', 'three', 'front'].map((c) => `<button data-cam="${c}">${c === 'three' ? '¾' : c}</button>`).join('')}<button id="labFollow">follow</button><button id="labSpin" title="O">orbit</button>
      <span class="sep"></span>
      <button id="labGrab" title="G, or Ctrl+drag a node">grab</button>
      <label>strength</label><input id="labGrabK" type="range" min="3" max="6.5" step="0.05" />
      <span id="labGrabKv"></span>
      <span class="sep"></span>
      <button id="labClean" title="H: hide the panels (recording)">clean</button>
    </div>
  </div>`);
  const $ = (id) => document.getElementById(id);
  const toggle = (b, on) => b.classList.toggle('on', !!on);
  const syncPanel = () => {
    $('labPlay').textContent = st.paused || st.replay >= 0 ? 'play ▶' : 'pause ❚❚';
    $('labSlow').value = st.auto ? 'auto' : String(st.slow);
    const o = renderer.opts;
    toggle($('labSkel'), o.skeleton); toggle($('labNodes'), o.nodes); toggle($('labStubs'), o.stubs); toggle($('labStudio'), o.studio);
    $('labColor').value = o.colorMode; $('labSkin').value = o.skin;
    toggle($('labGrab'), st.grabTool); toggle($('labSpin'), st.spin);
    for (const b of panel.querySelectorAll('[data-cam]')) toggle(b, b.dataset.cam === st.cam);
    toggle($('labFollow'), st.cam === 'follow');
    $('labGrabK').value = Math.log10(st.grabK);
    $('labGrabKv').textContent = `${st.grabK >= 1e6 ? (st.grabK / 1e6).toFixed(1) + ' MN/m' : (st.grabK / 1e3).toFixed(0) + ' kN/m'}`;
    $('labReplayTag').textContent = st.replay >= 0 ? 'REPLAY' : '';
    pills();
    legend();
  };
  const blur = (e) => e?.target?.blur?.();
  $('labPreset').onchange = (e) => { const k = e.target.value; if (!k) return; const u = new URL(location.href); const keep = ['car', 'truck', 'tyres', 'drag']; const s = new URLSearchParams(); s.set('solver', 'explicit'); s.set('preset', k); for (const x of keep) if (qs.has(x)) s.set(x, qs.get(x)); u.search = s.toString(); location.href = u.href; };
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
  for (const b of panel.querySelectorAll('[data-cam]')) b.onclick = (e) => { camera(b.dataset.cam); blur(e); };
  $('labFollow').onclick = (e) => { camera('follow'); blur(e); };
  $('labSpin').onclick = (e) => { st.spin = !st.spin; syncPanel(); blur(e); };
  $('labGrab').onclick = (e) => { st.grabTool = !st.grabTool; syncPanel(); blur(e); };
  $('labGrabK').oninput = (e) => { st.grabK = 10 ** Number(e.target.value); syncPanel(); };
  $('labClean').onclick = (e) => { setClean(true); blur(e); };

  // ---------------------------------------------------------------- HUD
  function pills() {
    const custom = qs.get('pills');
    const list = custom ? custom.split(',').map((s) => s.trim()).filter(Boolean)
      : [...(preset?.pills ?? [ctx.xscene.toUpperCase()])];
    const o = renderer.opts;
    const sub = custom ? [] : [o.skin === 'off' ? (o.nodes ? 'NODE + BEAM' : 'THE SKELETON') : o.skin === 'ghost' ? 'THE SKELETON · GHOST BODY' : 'COLLISION SKIN'];
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
    const slow = st.auto && st.impactStep == null ? 1 : st.slow;
    return `${slow === 1 ? 'REAL TIME' : `${slow}x SLOWER`} · ${tail}`;
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
  function restart() {
    releaseGrab();
    st.impactStep = null; st.ring = []; st.ringBytes = 0; st.replay = -1; st.acc = 0; st.pending = 0; st.lastState = null; st.focus = null;
    ctx.restart();
    syncPanel();
  }
  /** Steps to take this displayed frame (`dt` wall seconds), at most `max`. */
  function stepsFor(dt, max) {
    if (st.replay >= 0) return 0;
    if (st.paused) { const k = Math.min(st.pending, max); st.pending -= k; return k; }
    const slow = st.auto && st.impactStep == null ? 1 : st.slow;
    st.acc += Math.max(0, dt) * STEP_RATE / slow;
    const per = STEP_RATE / 60 / slow;
    const k = Math.max(0, Math.min(Math.floor(st.acc), Math.ceil(2 * per), max));
    st.acc = Math.min(st.acc - k, Math.max(1, per));
    return k;
  }
  /** Chunks of a frame's steps: before the impact in auto mode, small chunks so the
   * impact read lands within a few milliseconds of it. */
  function chunks(steps) {
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
    if (st.impactStep == null && s.impact) { st.impactStep = s.step; syncPanel(); }
    if (!st.lastState || s.step >= st.lastState.step) st.lastState = s;
    // The scene's focus: the box centre of every node (a flung piece ignored by
    // using the 2nd–98th percentile would cost a sort; the box is fine for presets).
    const p = s.pos;
    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < p.length; i += 3) for (let a = 0; a < 3; a++) { const x = p[i + a]; if (x < lo[a]) lo[a] = x; if (x > hi[a]) hi[a] = x; }
    const c = lo.map((x, a) => (x + hi[a]) / 2);
    if (c.every(Number.isFinite)) {
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
  function camera(name) {
    st.cam = name;
    if (name === 'follow') { ctx.setOrbit({ chase: true }); syncPanel(); return; }
    const c = CAMS[name];
    ctx.setOrbit({ yaw: c.yaw, pitch: c.pitch, dist: c.dist * st.size, target: null });
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
  function setClean(on) { st.clean = on; document.body.classList.toggle('lab-clean', on); }
  addEventListener('keydown', (e) => {
    if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT' || e.repeat) return;
    const k = e.code;
    if (k === 'Space') { playPause(); e.preventDefault(); e.stopImmediatePropagation(); }
    else if (k === 'Period') frameStep();
    else if (k === 'Comma') physicsStep();
    else if (k === 'KeyH' || (k === 'Escape' && st.clean)) setClean(!st.clean);
    else if (k === 'KeyK' || k === 'KeyV') { renderer.set({ skeleton: !renderer.opts.skeleton }); syncPanel(); e.stopImmediatePropagation(); }
    else if (k === 'KeyN') { renderer.set({ nodes: !renderer.opts.nodes }); syncPanel(); }
    else if (k === 'KeyB') { const i = COLOR_MODES.indexOf(renderer.opts.colorMode); renderer.set({ colorMode: COLOR_MODES[(i + 1) % COLOR_MODES.length] }); syncPanel(); }
    else if (k === 'KeyM') { const i = SKIN_MODES.indexOf(renderer.opts.skin); renderer.set({ skin: SKIN_MODES[(i + 1) % SKIN_MODES.length] }); syncPanel(); e.stopImmediatePropagation(); }
    else if (k === 'KeyG') { st.grabTool = !st.grabTool; syncPanel(); }
    else if (k === 'KeyO') { st.spin = !st.spin; syncPanel(); }
    else if (k === 'KeyR') { restart(); e.stopImmediatePropagation(); }
    else if (k >= 'Digit1' && k <= 'Digit5') camera(['side', 'top', 'three', 'front', 'follow'][Number(k.slice(5)) - 1]);
  }, { capture: true });

  counts();
  syncPanel();
  let camSet = false;

  return {
    st,
    stepsFor,
    chunks,
    read,
    /** Per displayed frame (before drawing): replay upload or live gather, HUD, camera. */
    beforeDraw(enc, dt) {
      if (st.replay >= 0 && st.ring[st.replay]) renderer.upload(st.ring[st.replay]);
      else renderer.gather(enc);
      if (st.focus) {
        renderer.set({ focus: [st.focus[0], st.focus[2]] });
        if (!camSet) { camSet = true; camera(st.cam); }
        if (st.cam !== 'follow') ctx.setOrbit({ follow: st.focus, spin: st.spin ? dt * 0.12 : 0 });
      }
      if (st.grab) applyGrab();
      updateHud();
    },
    /** The scene was rebuilt (restart): rebind and recount. */
    rebuilt() { renderer.setSim(ctx.sim(), ctx.layout()); counts(); legend(); },
    timeText,
    impactStep: () => st.impactStep,
    grabAt(global, target) { if (!st.lastState) return false; grabNode(global); st.grab.target = target; applyGrab(); return true; },
    release: releaseGrab,
    pick,
    playPause, frameStep, physicsStep, restart, enterReplay, camera, setClean,
    setSlow(v) { st.auto = v === 'auto'; st.slow = st.auto ? 20 : v; syncPanel(); },
    setPaused(on) { st.paused = on; syncPanel(); },
    sync: syncPanel,
  };
}
