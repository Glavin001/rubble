// The playground shell (play.html): every preset of the rubble::play registry on one page,
// drawn by three.js. Rust owns the presets, the scenes, their scripts and actions
// (Director) and what to draw (the render manifest); three.js draws on the World's own
// device (rubble-three.mjs); this file only wires the page: the menu, the camera, the
// HUD, the performance readouts, input and the frame loop. A family module
// (play-<family>.mjs) adds its controls, HUD fields, camera and checks.
//
// Switching presets and changing a scene's options rebuild the session in place (a new
// World and renderer; the page, its menu and the URL stay): no reload.
//
// A family module exports `setup(ctx)` (and optionally `settings(sceneDef)`: WorldDef
// fields over the scene's own; `create(sceneDef, settings)`: a world it builds itself).
// `ctx` (see `startSession`): world, placed, sceneDef, preset, THREE, view, camera,
// controls, rubble, renderer, ui, held, aim, lookFrom, setShadowRadius, counters,
// stepOne, drain, on, restart. setup returns hooks: onFrame, beforeStep, frame, keydown,
// hud, userCar, stepsPerFrame, dispose, hooks (window functions for checks).
import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { positionWorldDirection, mix, color, smoothstep } from 'three/tsl';
import { createScene, sceneFromQuery, presetList, pollFrame, stepFrames, drain, macrotask, errorText } from './rubble-world.mjs';
import { RubbleThree } from './rubble-three.mjs';
import * as PERF from './perf-meter.mjs';

const $ = (id) => document.getElementById(id);
const statusEl = $('status');
// The first failure is the cause (later GPU errors are usually "invalid due to a previous
// error"): it stays on screen; every one goes to the console.
let failures = 0;
const fail = (e) => {
  console.error('playground:', e);
  if (failures++ === 0) {
    $('err').textContent = errorText(e);
    window.__failed = String(e?.message ?? e);
  } else $('err').dataset.more = ` (+${failures - 1} more in the console)`;
  statusEl.textContent = `stopped${$('err').dataset.more ?? ''}`;
};
const clearFailure = () => { failures = 0; $('err').textContent = ''; delete $('err').dataset.more; delete window.__failed; };
addEventListener('unhandledrejection', (e) => { fail(e.reason); e.preventDefault(); });
addEventListener('error', (e) => fail(e.error || e.message));

/** Families the shell runs (their presets open here; the rest open their own page). */
const FAMILIES = new Set(['rigid', 'motion', 'terrain', 'vehicle']);
const HUBS = ['Scale & motion', 'Destruction', 'Vehicles', 'Terrain', 'Studio'];

if (!navigator.gpu) throw new Error('WebGPU is not available in this browser');
const adapterInfo = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
const wantTimestamps = !!adapterInfo?.features.has('timestamp-query');

// ---------------------------------------------------------------- the menu (the Rust registry)
const presets = await presetList();
const here = (p) => p.placed && FAMILIES.has(p.family);
const hrefOf = (p) => (here(p) ? `play.html?preset=${p.name}` : p.query ? `${p.page}?${p.query}` : p.page);
function buildMenu(current) {
  const menu = $('menu');
  menu.replaceChildren();
  const filter = $('search').value.trim().toLowerCase();
  const match = (p) => !filter || `${p.title} ${p.name} ${p.hub}`.toLowerCase().includes(filter);
  for (const hub of HUBS) {
    const list = presets.filter((p) => p.hub === hub && p.menu !== 'alias' && match(p));
    if (!list.length) continue;
    const h = document.createElement('div');
    h.className = 'hub';
    h.textContent = hub;
    menu.append(h);
    const entry = (p) => {
      const a = document.createElement('a');
      a.textContent = p.title;
      a.href = hrefOf(p);
      if (!here(p)) { a.className = 'away'; a.title = `opens ${p.page} (its own page)`; }
      else a.onclick = (e) => { if (e.metaKey || e.ctrlKey || e.shiftKey) return; e.preventDefault(); go(p.name); };
      if (p === current) a.className = 'here';
      return a;
    };
    for (const p of list.filter((p) => p.menu === 'main')) menu.append(entry(p));
    const more = list.filter((p) => p.menu === 'more');
    if (more.length) {
      // Collapsed unless searching (or the current preset is in it).
      const d = document.createElement('details');
      d.open = !!filter || more.includes(current);
      const s = document.createElement('summary');
      s.textContent = `More (${more.length})`;
      d.append(s, ...more.map(entry));
      menu.append(d);
    }
  }
  if (!menu.children.length) menu.textContent = 'No demo matches.';
}
$('search').addEventListener('input', () => buildMenu(session?.preset));

// ---------------------------------------------------------------- sessions
let session = null;
let switching = Promise.resolve();
/** Open preset `name` with the page's other switches `extra` (no reload). */
function go(name, extra = new URLSearchParams()) {
  const u = new URLSearchParams(extra);
  u.set('preset', name);
  history.pushState(null, '', `play.html?${u}`);
  return open();
}
addEventListener('popstate', () => open());
/** (Re)build the session from the URL. */
function open() {
  switching = switching.then(async () => {
    try {
      await session?.dispose();
      session = null;
      clearFailure();
      session = await startSession(new URLSearchParams(location.search));
      window.__ready = true;
    } catch (e) { fail(e); }
  });
  return switching;
}

async function startSession(qs) {
  window.__ready = false;
  const preset = presets.find((p) => p.name === qs.get('preset') && here(p)) ?? presets.find(here);
  qs.delete('preset');
  // The page's own switches first (a scene reads a switch's first value), then the preset's.
  const query = [qs.toString(), preset.query].filter(Boolean).join('&');
  document.title = `Rubble · ${preset.title}`;
  $('title').textContent = preset.title;
  buildMenu(preset);
  for (const id of ['controls', 'engine', 'help', 'famHud', 'famHudLine', 'perfExtra', 'perfButtons', 'perf']) $(id).replaceChildren();
  statusEl.textContent = 'building the world…';
  const cleanups = [];
  const on = (target, type, fn, opts) => { target.addEventListener(type, fn, opts); cleanups.push(() => target.removeEventListener(type, fn, opts)); };

  const family = await import(`./play-${preset.family}.mjs`);
  const sceneDef = await sceneFromQuery(preset.family, query);
  const liveTiming = wantTimestamps ? (qs.get('timing') ?? 'step') : 'off';
  const settings = { gpuTiming: liveTiming, ...(family.settings?.(sceneDef, new URLSearchParams(query)) ?? {}) };
  // A family may build its own world (the rigid family's snapshot scenes); otherwise the
  // scene's world, placed in Rust.
  const made = await family.create?.(sceneDef, settings);
  const world = made?.world ?? await createScene(sceneDef, settings);
  const device = world.device();
  let disposed = false;
  device.lost.then((i) => { if (!disposed) fail('device lost: ' + i.message); });
  on(device, 'uncapturederror', (e) => fail('gpu: ' + ((e.error && e.error.message) || e.error)));
  const placed = made ? made.placed : world.placeScene(sceneDef);

  // ---- three.js on the World's device
  const renderer = new THREE.WebGPURenderer({ device, antialias: true });
  await renderer.init();
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  $('stage').replaceChildren(renderer.domElement);
  const view = new THREE.Scene();
  // A sky gradient from the horizon (the fog's colour) up to the zenith.
  view.backgroundNode = mix(color(0xb7c6d6), color(0x5d86b8), smoothstep(0.0, 0.6, positionWorldDirection.y));
  view.fog = new THREE.Fog(new THREE.Color(0xb7c6d6), 250, 1100);
  view.add(new THREE.HemisphereLight(0xe8f0ff, 0x4a4032, 1.25));
  const sun = new THREE.DirectionalLight(0xfff1dc, 2.8);
  sun.castShadow = $('shadows').checked;
  sun.shadow.mapSize.set(4096, 4096);
  sun.shadow.bias = -0.0004;
  view.add(sun, sun.target);
  const rubble = new RubbleThree(renderer, world);
  view.add(rubble.object);
  rubble.colorMode.value = Number($('color').value);
  const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 4000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  const lookFrom = (eye, target) => { camera.position.set(...eye); controls.target.set(...target); controls.update(); };
  // The sun's shadow box follows the camera's target: `radius` m around it.
  let shadowRadius = 40;
  function placeSun() {
    const t = controls.target;
    sun.position.set(t.x + shadowRadius * 0.8, t.y + shadowRadius * 1.6, t.z + shadowRadius * 0.5);
    sun.target.position.copy(t);
    const s = sun.shadow.camera;
    s.left = s.bottom = -shadowRadius; s.right = s.top = shadowRadius; s.near = 0.1; s.far = shadowRadius * 6;
    s.updateProjectionMatrix();
  }
  // The drawing buffer follows the canvas's laid-out size each frame; a canvas with no
  // size (hidden, not laid out yet) is not drawn: a zero-sized swapchain texture fails the
  // frame's command buffer.
  let sized = [0, 0];
  function fitCanvas() {
    const c = renderer.domElement, w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return false;
    if (w !== sized[0] || h !== sized[1]) {
      sized = [w, h];
      renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    return true;
  }

  // ---- controls and input
  const keys = {};
  /** Rebuild this preset in place with switches changed (`{ name: value | null }`). */
  const restart = (changes) => {
    const u = new URLSearchParams(location.search);
    for (const [k, v] of Object.entries(changes)) { if (v === null || v === undefined || v === '') u.delete(k); else u.set(k, v); }
    history.replaceState(null, '', `play.html?${u}`);
    return open();
  };
  const mirror = (k, v) => { const u = new URLSearchParams(location.search); if (v === null) u.delete(k); else u.set(k, v); history.replaceState(null, '', `play.html?${u}`); };
  const labelled = (label, el, title) => { const l = document.createElement('label'); if (title) l.title = title; l.append(`${label} `, el); return l; };
  const ui = {
    /** A button with an optional key (`KeyF`). */
    button(label, key, fn, { into = 'controls' } = {}) {
      const b = document.createElement('button');
      b.innerHTML = key ? `${label} <kbd>${key.replace(/^Key/, '')}</kbd>` : label;
      b.onclick = () => { fn(); b.blur(); };
      $(into).append(b);
      if (key) keys[key] = fn;
      return b;
    },
    /** A select over switch `param`. `apply`: 'restart' rebuilds the scene in place (the
     * default), or a function applied live (the URL mirrors either way). */
    select(label, param, options, current, { apply = 'restart', into = 'controls', title } = {}) {
      const s = document.createElement('select');
      for (const [v, t] of options) s.add(new Option(t, v, false, String(v) === String(current)));
      s.onchange = () => { if (apply === 'restart') restart({ [param]: s.value }); else { mirror(param, s.value); apply(s.value); } s.blur(); };
      $(into).append(labelled(label, s, title));
      return s;
    },
    /** A checkbox over switch `param` (`on`/`off`), applied as `select`'s. */
    checkbox(label, param, checked, { apply = 'restart', into = 'controls', title, values = ['on', 'off'] } = {}) {
      const c = document.createElement('input');
      c.type = 'checkbox';
      c.checked = checked;
      c.onchange = () => { const v = c.checked ? values[0] : values[1]; if (apply === 'restart') restart({ [param]: v }); else { mirror(param, v); apply(c.checked); } };
      const l = document.createElement('label');
      if (title) l.title = title;
      l.append(c, ` ${label}`);
      $(into).append(l);
      return c;
    },
    /** A slider over switch `param` (applied as `select`'s, on release for restarts). */
    range(label, param, { min, max, step = 1, value, apply = 'restart', into = 'controls', title, format = (v) => v } = {}) {
      const r = document.createElement('input');
      Object.assign(r, { type: 'range', min, max, step, value });
      const out = document.createElement('span');
      out.className = 'muted';
      out.textContent = format(value);
      r.oninput = () => { out.textContent = format(Number(r.value)); if (apply !== 'restart') { mirror(param, r.value); apply(Number(r.value)); } };
      r.onchange = () => { if (apply === 'restart') restart({ [param]: r.value }); };
      const l = labelled(label, r, title);
      l.append(' ', out);
      $(into).append(l);
      return r;
    },
    /** Raw markup into a panel ('controls', 'engine', 'help', 'famHud'). */
    html(html, into = 'help') { const d = document.createElement('div'); d.innerHTML = html; $(into).append(d); return d; },
    help(html) { $('help').innerHTML = html; },
    /** A HUD block a family writes to. */
    block() { const d = document.createElement('div'); $('famHud').append(d); return d; },
  };
  const held = new Set();
  let scene = null;
  on(window, 'keydown', (e) => {
    if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
    held.add(e.code);
    scene?.keydown?.(e);
    if (!e.repeat && keys[e.code]) { e.preventDefault(); keys[e.code](); }
  });
  on(window, 'keyup', (e) => held.delete(e.code));
  on(window, 'blur', () => held.clear());
  /** The camera's ray (eye and unit direction), for aimed actions. */
  const aim = () => { const d = new THREE.Vector3(); camera.getWorldDirection(d); return { eye: camera.position.toArray(), dir: d.toArray() }; };

  // ---- frames, rates and measurements (perf-meter.mjs, as demo.html's)
  let counters = null, gpuMs = 0, frameTimes = [], hostMs = 0, drawMs = 0;
  const submitted = new Map();
  let lastCompletion = 0;
  const rates = PERF.createRateMeter((r) => {
    if (r.measuring) { $('perf').textContent = 'measuring…'; return; }
    window.__RATES__ = { ...r, labels: undefined };
    $('perf').innerHTML = `<b>TPS</b> ${r.tps.toFixed(0)} · <b title="1000 ÷ physics step cost (${r.labels.maxTpsSub || '—'})">Max TPS</b> ${r.labels.maxTps} `
      + `<span class="muted">${r.labels.step}</span> · <b>FPS</b> ${r.fps.toFixed(0)} · <b title="1000 ÷ CPU time per frame">Max FPS</b> ${r.maxFps > 0 ? r.maxFps.toFixed(0) : '—'} `
      + `<span class="muted">${r.frameMs.toFixed(1)} ms/frame · host ${hostMs.toFixed(1)} ms · draw ${drawMs.toFixed(1)} ms</span>`;
  });
  const onFrame = (f) => {
    scene?.onFrame?.(f);
    const completedAt = performance.now();
    const times = f.gpuTimes();
    for (const t of times) {
      gpuMs = t.totalMs;
      frameTimes.push(t);
      rates.addStepCost(PERF.liveStepSample({ gpuMs: t.totalMs, submittedAt: submitted.get(t.step), completedAt, previousCompletedAt: lastCompletion }));
    }
    if (!f.empty) {
      if (!times.length && submitted.has(f.lastStep))
        rates.addStepCost(PERF.liveStepSample({ gpuMs: 0, submittedAt: submitted.get(f.lastStep), completedAt, previousCompletedAt: lastCompletion }));
      for (const s of submitted.keys()) if (s <= f.lastStep) submitted.delete(s);
      lastCompletion = completedAt;
    }
    if (frameTimes.length > 64) frameTimes.splice(0, frameTimes.length - 64);
    const c = f.counters();
    if (c) counters = { contact: c.contacts, candidate: c.candidates, awake: c.awake, overflow: c.overflow, colors: c.colors };
  };
  const step = () => { submitted.set(world.stepCount(), performance.now()); world.step(); };
  // ?paused=1: the loop draws but takes no steps until __run() (checks start from the placed state).
  let manual = qs.get('paused') === '1';
  // ?stopAt=N: playback freezes at step N (still drawing), for comparable stills.
  const stopAt = Number(qs.get('stopAt') ?? 0);
  const stepOne = async () => {
    manual = true;
    while (!world.canStep()) { await macrotask(); pollFrame(world, onFrame); }
    scene?.beforeStep?.();
    step();
    pollFrame(world, onFrame);
  };
  const drainAll = () => drain(world, onFrame);

  scene = await family.setup({
    world, placed, sceneDef, preset, query, qs, THREE, view, camera, controls, rubble, renderer, ui, held, aim, lookFrom,
    setShadowRadius: (r) => { shadowRadius = r; },
    counters: () => counters, stepOne, drain: drainAll, on, restart, rates,
    measuring: (v) => { rates.measuring = v; if (v) rates.showMeasuring(); else rates.reset(); manual = v; },
    liveTiming, wantTimestamps, frameTimes: () => frameTimes, clearFrameTimes: () => { frameTimes = []; },
  });
  placeSun();

  // The GPU phase profile (any scene): World times each phase of its plan for 10 steps.
  if (wantTimestamps) {
    ui.button('Profile GPU phases', null, async () => {
      rates.measuring = true; rates.showMeasuring(); manual = true;
      try {
        const { summary, report } = await PERF.profilePhases({
          world, stepOne, drain: drainAll, liveTiming, wantTimestamps,
          frameTimes: () => frameTimes, clearFrameTimes: () => { frameTimes = []; },
        });
        window.__PHASES__ = report;
        $('perfExtra').textContent = summary || 'no phases (nothing awake)';
      } finally { rates.measuring = false; rates.reset(); manual = false; }
    }, { into: 'perfButtons' });
  }

  function draw() {
    rubble.update();
    controls.update();
    placeSun();
    if (!fitCanvas()) return;
    const t = performance.now();
    renderer.render(view, camera);
    drawMs = performance.now() - t;
  }
  let lastT = performance.now();
  function frame() {
    const now = performance.now();
    const dt = Math.min(0.05, (now - lastT) / 1000);
    lastT = now;
    if (manual) return;
    const frozen = stopAt > 0 && world.simSteps() >= stopAt;
    world.beforeFrame(dt, scene.userCar?.() ?? -1);
    // Steps only while a readback slot is free (never blocks), each after the family's
    // per-step input.
    let stepped = 0;
    for (let k = 0; !frozen && k < (scene.stepsPerFrame ?? 1) && world.canStep(); k++) {
      scene.beforeStep?.();
      step();
      stepped++;
    }
    world.afterFrame();
    pollFrame(world, onFrame);
    scene.frame?.(dt);
    hostMs = performance.now() - now;
    draw();
    rates.frame(stepped, performance.now() - now);
    const c = counters;
    $('stats').innerHTML = `${(world.bodyCount() - 1).toLocaleString()} bodies · step ${world.simSteps()}${frozen ? ' (stopped: stopAt)' : ''} · GPU step ${gpuMs.toFixed(2)} ms`
      + (c ? ` · contacts ${c.contact.toLocaleString()} · awake ${c.awake.toLocaleString()}${c.overflow ? ` · <span class="bad">overflow 0x${c.overflow.toString(16)}</span>` : ''}` : '');
    $('famHudLine').innerHTML = scene.hud?.() ?? '';
  }
  renderer.setAnimationLoop(frame);

  // ---- automation
  // __step(n): n physics steps, frame-locked, then a draw; the family's hooks besides.
  window.__step = async (n = 1) => {
    manual = true;
    await stepFrames(world, n, onFrame, scene.beforeStep ?? null);
    scene.frame?.(0);
    draw();
    return { steps: world.stepCount(), counters };
  };
  // __renderBench(n): the draw alone, `n` times, each waited for on the GPU (ms per draw).
  window.__renderBench = async (n = 60) => {
    manual = true;
    const ts = [];
    for (let i = 0; i < n; i++) {
      const t = performance.now();
      draw();
      await device.queue.onSubmittedWorkDone();
      ts.push(performance.now() - t);
    }
    ts.sort((a, b) => a - b);
    return { p50: ts[n >> 1], p90: ts[Math.floor(n * 0.9)], min: ts[0], ...rubble.stats() };
  };
  window.__run = () => { manual = false; };
  const hookNames = Object.keys(scene.hooks ?? {});
  Object.assign(window, scene.hooks ?? {});
  window.__world = world;
  window.__preset = preset;
  statusEl.textContent = '';

  return {
    preset, world, rubble,
    setColorMode: (m) => { rubble.colorMode.value = m; },
    setShadows: (v) => { sun.castShadow = v; },
    async dispose() {
      renderer.setAnimationLoop(null);
      await scene?.dispose?.();
      for (const c of cleanups) c();
      for (const k of hookNames) delete window[k];
      disposed = true;
      rubble.dispose();
      controls.dispose();
      renderer.dispose();
      world.free();
    },
  };
}

// Page-wide view options (kept across sessions).
$('color').onchange = () => session?.setColorMode(Number($('color').value));
$('shadows').onchange = () => session?.setShadows($('shadows').checked);
$('color').value = new URLSearchParams(location.search).get('color') ?? '0';
await open();
