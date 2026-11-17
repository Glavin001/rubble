// The playground shell (play.html): every preset of the rubble::play registry on one page,
// drawn by three.js. Rust owns the presets, the scenes, their scripts and actions
// (Director) and what to draw (the render manifest); three.js draws on the World's own
// device (rubble-three.mjs); this file only wires the page: the menu, the camera, the
// HUD, input and the frame loop. A family module (play-<family>.mjs) adds its controls,
// HUD fields, camera and checks.
import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createScene, sceneFromQuery, presetList, pollFrame, stepFrames, errorText } from './rubble-world.mjs';
import { RubbleThree } from './rubble-three.mjs';

const $ = (id) => document.getElementById(id);
const statusEl = $('status');
const fail = (e) => {
  $('err').textContent = errorText(e);
  statusEl.textContent = 'stopped';
  window.__failed = String(e?.message ?? e);
};
addEventListener('unhandledrejection', (e) => { fail(e.reason); e.preventDefault(); });
addEventListener('error', (e) => fail(e.error || e.message));

/** Families the shell runs (their presets open here; the rest still open their page). */
const FAMILIES = new Set(['rigid', 'motion', 'terrain', 'vehicle']);
const HUBS = ['Scale & motion', 'Terrain', 'Vehicles', 'Destruction'];

// ---------------------------------------------------------------- the preset and menu
const presets = await presetList();
const here = (p) => p.placed && FAMILIES.has(p.family);
const qs = new URLSearchParams(location.search);
const preset = presets.find((p) => p.name === qs.get('preset') && here(p)) ?? presets.find(here);
qs.delete('preset');
// The page's own switches first (a scene reads a switch's first value), then the preset's.
const query = [qs.toString(), preset.query].filter(Boolean).join('&');
document.title = `Rubble · ${preset.title}`;
$('title').textContent = preset.title;
const menu = $('menu');
for (const hub of HUBS) {
  const list = presets.filter((p) => p.hub === hub);
  if (!list.length) continue;
  const h = document.createElement('div');
  h.className = 'hub';
  h.textContent = hub;
  menu.append(h);
  for (const p of list) {
    const a = document.createElement('a');
    a.textContent = p.title;
    if (here(p)) a.href = `play.html?preset=${p.name}`;
    else {
      // Not on the shell yet: its own page.
      a.href = p.query ? `${p.page}?${p.query}` : p.page;
      a.className = 'away';
      a.title = `opens ${p.page}`;
    }
    if (p === preset) a.className = 'here';
    menu.append(a);
  }
}

if (!navigator.gpu) throw new Error('WebGPU is not available in this browser');
const family = await import(`./play-${preset.family}.mjs`);
const sceneDef = await sceneFromQuery(preset.family, query);
statusEl.textContent = 'building the world…';
const world = await createScene(sceneDef, { gpuTiming: 'step', ...(family.settings?.(sceneDef) ?? {}) });
const device = world.device();
device.lost.then((i) => fail('device lost: ' + i.message));
device.addEventListener('uncapturederror', (e) => fail('gpu: ' + ((e.error && e.error.message) || e.error)));
const placed = world.placeScene(sceneDef);

// ---------------------------------------------------------------- three.js on the World's device
const renderer = new THREE.WebGPURenderer({ device, antialias: true });
await renderer.init();
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
document.body.prepend(renderer.domElement);

const view = new THREE.Scene();
const sky = new THREE.Color(0x9fb3c8);
view.background = sky;
view.fog = new THREE.Fog(sky, 200, 900);
view.add(new THREE.HemisphereLight(0xe8f0ff, 0x4a4032, 1.25));
const sun = new THREE.DirectionalLight(0xfff1dc, 2.8);
sun.castShadow = true;
sun.shadow.mapSize.set(4096, 4096);
sun.shadow.bias = -0.0004;
view.add(sun, sun.target);

const rubble = new RubbleThree(renderer, world);
view.add(rubble.object);

const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.05, 4000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
/** Point the camera from `eye` at `target` (family camera hints). */
const lookFrom = (eye, target) => {
  camera.position.set(...eye);
  controls.target.set(...target);
  controls.update();
};
// The sun's shadow box follows the camera's target: `radius` m around it.
let shadowRadius = 40;
function placeSun() {
  const t = controls.target;
  sun.position.set(t.x + shadowRadius * 0.8, t.y + shadowRadius * 1.6, t.z + shadowRadius * 0.5);
  sun.target.position.copy(t);
  const s = sun.shadow.camera;
  s.left = s.bottom = -shadowRadius;
  s.right = s.top = shadowRadius;
  s.near = 0.1;
  s.far = shadowRadius * 6;
  s.updateProjectionMatrix();
}
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// ---------------------------------------------------------------- controls and input
const keys = {};
const ui = {
  /** A button with an optional key (`KeyF`). */
  button(label, key, fn) {
    const b = document.createElement('button');
    b.innerHTML = key ? `${label} <kbd>${key.replace(/^Key/, '')}</kbd>` : label;
    b.onclick = () => { fn(); b.blur(); };
    $('controls').append(b);
    if (key) keys[key] = fn;
    return b;
  },
  /** A select whose change reloads with switch `param` (scene-building switches). */
  select(label, param, options, current) {
    const l = document.createElement('label');
    l.textContent = `${label} `;
    const s = document.createElement('select');
    for (const [v, t] of options) s.add(new Option(t, v, false, String(v) === String(current)));
    s.onchange = () => { const u = new URL(location.href); u.searchParams.set(param, s.value); location.href = u.href; };
    l.append(s);
    $('controls').append(l);
    return s;
  },
  help(html) { $('help').innerHTML = html; },
  /** A HUD block a family writes to. */
  block() { const d = document.createElement('div'); $('famHud').append(d); return d; },
};
const colorSel = ui.select('Colour', 'color', [[0, 'material'], [1, 'awake / asleep'], [2, 'speed']], qs.get('color') ?? 0);
colorSel.onchange = () => { rubble.colorMode.value = Number(colorSel.value); };
rubble.colorMode.value = Number(colorSel.value);
const shadowBox = document.createElement('label');
shadowBox.innerHTML = '<input type="checkbox" checked> shadows';
shadowBox.firstChild.onchange = (e) => { sun.castShadow = e.target.checked; };
$('controls').append(shadowBox);

let scene = null;
const held = new Set();
addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  held.add(e.code);
  scene?.keydown?.(e);
  if (!e.repeat) keys[e.code]?.();
});
addEventListener('keyup', (e) => held.delete(e.code));
addEventListener('blur', () => held.clear());

/** The camera's ray (eye and unit direction), for aimed actions. */
const aim = () => {
  const d = new THREE.Vector3();
  camera.getWorldDirection(d);
  return { eye: camera.position.toArray(), dir: d.toArray() };
};

scene = await family.setup({
  world, placed, sceneDef, preset, THREE, view, camera, controls, rubble, renderer, ui, held, aim, lookFrom,
  setShadowRadius: (r) => { shadowRadius = r; },
});
placeSun();

// ---------------------------------------------------------------- the frame loop
let counters = null, gpuMs = 0;
const onFrame = (f) => {
  scene.onFrame?.(f);
  for (const t of f.gpuTimes()) gpuMs = t.totalMs;
  const c = f.counters();
  if (c) counters = { contact: c.contacts, awake: c.awake, overflow: c.overflow };
};
function draw() {
  rubble.update();
  controls.update();
  placeSun();
  renderer.render(view, camera);
}
let manual = false;
let lastT = performance.now(), fpsAcc = 0, fpsN = 0, fps = 0;
function frame() {
  const now = performance.now();
  const dt = Math.min(0.05, (now - lastT) / 1000);
  lastT = now;
  fpsAcc += dt; fpsN++;
  if (fpsAcc > 0.5) { fps = fpsN / fpsAcc; fpsAcc = 0; fpsN = 0; }
  if (manual) return;
  world.beforeFrame(dt, scene.userCar?.() ?? -1);
  // Steps only while a readback slot is free (never blocks), each after the family's
  // per-step input.
  for (let k = 0; k < (scene.stepsPerFrame ?? 1) && world.canStep(); k++) {
    scene.beforeStep?.();
    world.step();
  }
  world.afterFrame();
  pollFrame(world, onFrame);
  scene.frame?.(dt);
  draw();
  const c = counters;
  $('stats').innerHTML = `${(world.bodyCount() - 1).toLocaleString()} bodies · step ${world.stepCount()} · ${fps.toFixed(0)} fps · GPU step ${gpuMs.toFixed(2)} ms`
    + (c ? ` · contacts ${c.contact.toLocaleString()} · awake ${c.awake.toLocaleString()}${c.overflow ? ` · <span class="bad">overflow 0x${c.overflow.toString(16)}</span>` : ''}` : '');
  $('famHudLine').innerHTML = scene.hud?.() ?? '';
}
renderer.setAnimationLoop(frame);

// ---------------------------------------------------------------- automation
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
Object.assign(window, scene.hooks ?? {});
window.__world = world;
window.__preset = preset;
window.__ready = true;
statusEl.textContent = '';
