// The vehicle family in the playground shell (play.html): raycast-wheel vehicles
// (rubble::play::vehicles; their scripts and actions are rubble::play::director and
// rubble::play::vehicle_director: the gauntlet's shelling, the meteor range's volleys and
// scoring, the meteor run's director and judge, directed debris, the cannon). This module
// adds the player's input (keys, gamepad), the chase / orbit camera, the scene's keys and
// options, the dashboard, the parts & wheels panel and the scenes' HUDs; vehicles, parts
// and wheels are drawn from the World by rubble-three.mjs. Everything vehicle.html offers
// is here (so vehicle.html can open play.html?preset=<scene>), except its WGSL renderer
// and the hybrid (soft-frame) buggies, which rubble-three does not draw.
import { createScene, saveState, macrotask } from './rubble-world.mjs';

const SCENES = [
  ['playground', 'Playground'], ['destruction', 'Destruction derby'], ['traffic', 'Traffic'], ['slab', 'Slab'],
  ['seesaw', 'Seesaw'], ['carry', 'Carry'], ['sidewall', 'Side wall'], ['bridge', 'Bridge'], ['tanklab', 'Tank lab'],
  ['buggies', 'Buggies'], ['vibeland', 'Vibe-land garage'], ['gauntlet', 'Gauntlet'], ['meteors', 'Meteor range'], ['run', 'Meteor run'],
];

const qrot = (q, v) => {
  const [x, y, z, w] = q;
  const ux = y * v[2] - z * v[1], uy = z * v[0] - x * v[2], uz = x * v[1] - y * v[0];
  const tx = 2 * ux, ty = 2 * uy, tz = 2 * uz;
  return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
};
const norm3 = (a) => { const l = Math.hypot(...a) || 1; return a.map((x) => x / l); };
const presetOptions = (presets) => presets.filter((p) => !p.name.endsWith('_trailer')).map((p) => [p.name, p.name.replaceAll('_', ' ')]);

// The family's own panels (dashboard, parts, the run's game HUD): fixed overlays, removed
// with the session.
const CSS = `
  .pv-panel { position: fixed; background: rgba(13, 17, 23, 0.84); border: 1px solid #30363d; border-radius: 8px; padding: 10px 12px; }
  #pv-dash { right: 12px; bottom: 12px; width: 15rem; font-variant-numeric: tabular-nums; }
  #pv-speed { font-size: 34px; font-weight: 700; line-height: 1; }
  #pv-speed small { font-size: 13px; font-weight: 500; color: #8b949e; }
  #pv-gear { font-size: 26px; font-weight: 700; float: right; }
  .pv-bar { height: 6px; background: #252c3d; border-radius: 3px; overflow: hidden; margin: 6px 0; }
  .pv-bar > div { height: 100%; width: 0; background: linear-gradient(90deg, #4fb0ff, #ffb04f 75%, #ff5050); }
  #pv-wheels { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; margin-top: 6px; }
  .pv-wheel { border: 1px solid #333c55; border-radius: 4px; padding: 2px 6px; font-size: 11px; }
  .pv-wheel.contact { border-color: #4fb06a; } .pv-wheel.slide { border-color: #ffb04f; }
  .pv-wheel.gone { opacity: 0.35; text-decoration: line-through; }
  #pv-parts { right: 12px; top: 12px; width: 15rem; max-height: 50vh; overflow: auto; }
  .pv-part { display: flex; justify-content: space-between; align-items: center; margin: 3px 0; gap: 6px; }
  .pv-part.lost > span:first-child { opacity: 0.4; text-decoration: line-through; }
  @media (max-width: 700px) { #pv-parts { display: none; } #pv-dash { width: 11rem; } }
  #pv-game { position: fixed; right: 12px; top: 12px; width: min(20rem, calc(100vw - 24px)); text-align: right; pointer-events: none; font-variant-numeric: tabular-nums; text-shadow: 0 1px 3px #000; }
  @media (max-width: 700px) { #pv-game { top: auto; bottom: 150px; } }
  #pv-game .score { font-size: 32px; font-weight: 800; line-height: 1.1; letter-spacing: 0.02em; }
  #pv-game .meter { height: 9px; background: rgba(40, 26, 26, 0.85); border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 5px; overflow: hidden; margin: 5px auto 0; }
  #pv-hp { height: 100%; width: 100%; background: linear-gradient(90deg, #ff4d3d, #ffb04f 45%, #5fd36a); transition: width 0.25s; }
  #pv-prog { height: 100%; width: 0; background: linear-gradient(90deg, #6fa8ff, #c7dcff); }
  #pv-ginfo { margin-top: 4px; font-size: 12px; color: #d6dbe6; }
  #pv-toasts { position: fixed; left: 50%; top: 30%; transform: translateX(-50%); pointer-events: none; text-align: center; }
  .pv-toast { font-weight: 850; font-size: 22px; letter-spacing: 0.04em; text-shadow: 0 2px 8px #000; animation: pv-toast 1.2s ease-out forwards; white-space: nowrap; }
  .pv-toast.good { color: #ffd84f; } .pv-toast.big { color: #ff9a3c; font-size: 30px; } .pv-toast.bad { color: #ff5a4a; }
  @keyframes pv-toast { from { opacity: 1; transform: translateY(0) scale(1.2); } to { opacity: 0; transform: translateY(-50px) scale(1); } }
  #pv-card { position: fixed; inset: 0; display: flex; align-items: center; justify-content: center; pointer-events: none; }
  #pv-card[hidden], #pv-game[hidden], #pv-parts[hidden], #pv-dash[hidden] { display: none; }
  #pv-card > div { background: rgba(16, 10, 9, 0.84); border: 1px solid #7a412c; border-radius: 14px; padding: 20px 28px; text-align: center; max-width: min(30rem, calc(100vw - 40px)); line-height: 1.5; }
  #pv-card h2 { margin: 0 0 6px; font-size: 30px; color: #ff9a5a; letter-spacing: 0.06em; }
  #pv-card table { margin: 8px auto 0; border-collapse: collapse; font-variant-numeric: tabular-nums; }
  #pv-card td { padding: 1px 10px; text-align: right; } #pv-card td:first-child { text-align: left; color: #b9a49a; }
`;

/**
 * The scene with the car the menu picked: `?car=` (the shell's `?preset=` names the
 * playground preset), Rust's `VehicleSceneDef.preset`.
 */
export async function create(sceneDef, settings) {
  const car = new URLSearchParams(location.search).get('car');
  if (car) sceneDef.preset = car;
  const world = await createScene(sceneDef, settings);
  return { world, placed: world.placeScene(sceneDef) };
}

export async function setup(ctx) {
  const { world, placed, sceneDef, qs, ui, held, aim, camera, controls, setShadowRadius, THREE, view, on, stepOne, drain, restart } = ctx;
  const scene = sceneDef.scene ?? 'playground';
  const street = scene === 'gauntlet', range = scene === 'meteors', run = scene === 'run';
  const vehicles = placed.vehicles;
  const disposers = [];
  const entries = [];
  const entry = (i) => (entries[i] ??= world.vehicleEntry(vehicles[i].h));
  // The player drives the first vehicle no script drives (Tab cycles through all).
  let player = Math.max(0, vehicles.findIndex((v) => !v.ai));
  const handle = () => vehicles[player].h;

  // ---- the family's panels
  const style = document.createElement('style');
  style.textContent = CSS;
  document.head.append(style);
  disposers.push(() => style.remove());
  const overlay = (html, attrs = {}) => {
    const d = document.createElement('div');
    Object.assign(d, attrs);
    d.innerHTML = html;
    document.body.append(d);
    disposers.push(() => d.remove());
    return d;
  };
  overlay('<span id="pv-gear">N</span><div id="pv-speed">0 <small>km/h</small></div><div class="pv-bar"><div id="pv-rpm"></div></div><div class="muted" id="pv-rpmText">— rpm</div><div id="pv-wheels"></div>',
    { id: 'pv-dash', className: 'pv-panel', hidden: !vehicles.length });
  overlay('<div class="row" style="justify-content: space-between"><strong>Parts &amp; wheels</strong><span class="muted">sever / disable</span></div><div id="pv-partList"></div>',
    { id: 'pv-parts', className: 'pv-panel', hidden: run || !vehicles.length });
  const $ = (id) => document.getElementById(id);
  const speedEl = $('pv-speed'), gearEl = $('pv-gear'), rpmEl = $('pv-rpm'), rpmText = $('pv-rpmText'), wheelsEl = $('pv-wheels');
  const partList = $('pv-partList');

  // ---- options: every one a control; scene switches rebuild in place, the rest apply live
  const presets = await (await fetch('./generated/vehicle_presets.json', { cache: 'no-store' })).json();
  const playerName = (vehicles[player]?.name ?? '').replace(/_shells$/, '');
  ui.select('Scene', 'scene', SCENES, scene, { apply: (v) => restart({ scene: v, car: null }) });
  ui.select('Car', 'car', presetOptions(presets), qs.get('car') ?? playerName, { title: 'The player\'s vehicle preset (the scene\'s own by default)' });
  const desc0 = vehicles.length ? entry(player).desc : null;
  // ?strength=k scales every mount's break limits (VehicleSceneDef.strength).
  if (scene === 'buggies' || scene === 'vibeland' || (desc0?.parts ?? []).some((p, i) => i > 0 && p.mount?.break_force > 0)) {
    ui.select('Mounts', 'strength', [[1, 'rated'], [0.5, '½ strength'], [0.25, '¼ strength'], [0.1, 'fragile']], Number(qs.get('strength') || 1),
      { title: 'Scales every mount\'s break limits' });
  }
  if (desc0?.tracks || qs.get('tracks') === 'shells') {
    ui.checkbox('Wheel shells (no track pods)', 'tracks', qs.get('tracks') === 'shells', { values: ['shells', ''], title: 'The tank as it was before track pods, for comparisons' });
  }
  if (scene === 'traffic') ui.range('AI vehicles', 'vehicles', { min: 1, max: 60, value: Number(qs.get('vehicles') || 48) });
  if (scene === 'bridge') ui.select('Heavy vehicle', 'heavy', [['none', 'none (the car alone)'], ...presetOptions(presets)], qs.get('heavy') || 'monster_truck');
  // Engine switches (VehicleSceneDef).
  ui.checkbox('Sleep', 'sleep', qs.get('sleep') !== 'off', { into: 'engine', values: ['', 'off'], title: 'Bodies at rest go to sleep' });
  if (placed.cannon?.length) ui.checkbox('Cannonball CCD', 'ccd', qs.get('ccd') !== '0', { into: 'engine', values: ['1', '0'], title: 'Continuous collision for the cannonballs' });
  if (scene === 'carry') ui.checkbox('Resting-pair carry', 'rest', qs.get('rest') === 'carry', { into: 'engine', values: ['carry', ''] });

  // Directed debris and the scripts' own firing (gauntlet, meteor range, meteor run):
  // live, through the director's actions.
  let guideBox = null, guideSel = null, shellBox = null, routesBox = null;
  const guideOn = () => guideBox?.checked ?? false;
  const applyGuidance = () => world.act({ action: 'guide', on: guideBox.checked, strength: Number(guideSel.value) });
  if (street || range || run) {
    guideBox = ui.checkbox('Directed debris', 'guide', sceneDef.guide, { values: ['1', '0'], apply: () => { applyGuidance(); runHud(); } });
    // Its own switch, not ?strength= (the mounts'): the director takes it live.
    guideSel = ui.select('Strength', 'guideStrength', [[0, 'subtle'], [1, 'cinematic'], [2, 'exaggerated']], qs.get('guideStrength') ?? String(sceneDef.guideStrength),
      { apply: () => applyGuidance() });
    if (qs.has('guideStrength')) applyGuidance();
    if (range) {
      const moving = ui.checkbox('Moving targets', 'targets', sceneDef.targetsMoving, {
        values: ['moving', 'fixed'],
        apply: (v) => { world.act({ action: 'targets', moving: v }); routesBox.parentElement.hidden = !v; resetScene(); },
      });
      routesBox = ui.checkbox('Known routes', 'routes', sceneDef.routes, {
        values: ['1', '0'], title: 'Steering knows each cart\'s route, turns included',
        apply: (v) => { world.act({ action: 'targets', routes: v }); resetScene(); },
      });
      routesBox.parentElement.hidden = !moving.checked;
    }
    // The director's autofire (the gauntlet's shelling, the range's volleys, the run's meteors).
    shellBox = ui.checkbox(run ? 'Meteors' : range ? 'Volleys' : 'Shelling', 'autofire', qs.get('autofire') !== '0', { values: ['1', '0'], apply: (v) => world.setAutofire(v) });
    world.setAutofire(shellBox.checked);
  }
  const autofire = () => world.setAutofire(shellBox?.checked ?? true);

  // ---- the player's vehicle: chase distance from its size (a motorcycle vs a tank)
  let chaseDist = 8, maxRpm = 0, playerDesc = null;
  let state = null, chassis = null, loads = null, partsKey = '';
  const severed = new Set();
  function selectPlayer(i) {
    player = i;
    playerDesc = entry(player).desc;
    maxRpm = playerDesc.drive.type === 'drivetrain' ? playerDesc.drive.engine.max_rpm : 0;
    const extent = Math.max(
      ...playerDesc.wheels.map((w) => Math.hypot(w.position[0], w.position[2]) + w.radius),
      ...playerDesc.parts.flatMap((p) => p.colliders.map((k) => Math.hypot(k.position[0], k.position[2])
        + (k.shape.type === 'box' ? Math.hypot(k.shape.half[0], k.shape.half[2]) : k.shape.radius))),
    );
    chaseDist = Math.max(4.5, Math.min(16, 2.5 + 2.6 * extent));
    state = null; chassis = null; loads = null; partsKey = '';
    severed.clear();
    renderParts();
  }
  setShadowRadius(range ? 70 : 45);
  const initial = await saveState(world);

  // ---- input
  const key = (...c) => c.some((k) => held.has(k));
  // Scripted input (the hooks) holds until a key is pressed.
  let scripted = false;
  function readInput() {
    let forward = (key('KeyW', 'ArrowUp') ? 1 : 0) - (key('KeyS', 'ArrowDown') ? 1 : 0);
    let steer = (key('KeyD', 'ArrowRight') ? 1 : 0) - (key('KeyA', 'ArrowLeft') ? 1 : 0);
    let handbrake = key('Space') ? 1 : 0, brake = 0;
    for (const pad of navigator.getGamepads ? navigator.getGamepads() : []) {
      if (!pad) continue;
      if (Math.abs(pad.axes[0] || 0) > 0.12) steer = pad.axes[0];
      const rt = pad.buttons[7]?.value || 0, lt = pad.buttons[6]?.value || 0;
      if (rt > 0.05 || lt > 0.05) forward = rt - lt;
      if (pad.buttons[0]?.pressed) handbrake = 1;
      if (pad.buttons[1]?.pressed) brake = 1;
    }
    return { forward, steer, brake, handbrake };
  }

  // ---- meteor range: touchdown markers coloured on target / off it (visual only: the
  // director places the marker bodies, the colour is the page's).
  const markerMeshes = [];
  const markerIndex = new Map((placed.markers ?? []).map((m, i) => [m.collider, i]));
  let markerHit = null, markerMiss = null;
  if (range && placed.markers?.length) {
    const geo = new THREE.BoxGeometry(0.54, 0.14, 0.54);
    markerHit = new THREE.MeshStandardMaterial({ color: 0x33e64d, roughness: 0.6 });
    markerMiss = new THREE.MeshStandardMaterial({ color: 0xf22626, roughness: 0.6 });
    for (let i = 0; i < placed.markers.length; i++) {
      const m = new THREE.Mesh(geo, markerMiss);
      m.visible = false;
      view.add(m);
      markerMeshes.push(m);
    }
    disposers.push(() => { for (const m of markerMeshes) view.remove(m); geo.dispose(); markerHit.dispose(); markerMiss.dispose(); });
  }
  function markTouchdown(d) {
    const m = markerMeshes[markerIndex.get(d.marker) ?? -1];
    if (!m) return;
    m.material = d.hit ? markerHit : markerMiss;
    m.position.set(d.at[0], 0.07, d.at[2]);
    m.visible = true;
  }

  // ---- the scripts' events: shells, volleys, touchdowns (marked and scored), the run's
  // judged near misses and hits, its status.
  const recent = [];
  let lastShell = 0, runStatus = null, runLog = null, rangeLog = null;
  let rangeScore = [[0, 0], [0, 0]];
  const blocksLaunched = new Set();
  const guideStatsEl = (street || range) ? ui.block() : null;
  if (guideStatsEl) guideStatsEl.className = 'muted';
  function onEvents() {
    for (const e of world.takeEvents()) {
      const d = e.data;
      if (e.what !== 'touchdown' && e.what !== 'judge' && e.what !== 'status') { recent.unshift(e.what); recent.length = Math.min(recent.length, 6); }
      if (e.what === 'shell' || e.what === 'volley') lastShell = e.step;
      if ((e.what === 'shell' || e.what === 'blast') && d) for (const i of d.blocks) blocksLaunched.add(i);
      else if (e.what === 'touchdown') {
        rangeScore = d.score;
        rangeLog?.push({ lane: d.lane, pad: d.pad, at: d.at, hit: d.hit, turned: d.turned, body: d.body });
        markTouchdown(d);
        const [a, b] = rangeScore;
        if (guideStatsEl) guideStatsEl.innerHTML = `<b>On target</b> · steered lane (left, x &gt; 0): <b>${a[0]}/${a[1]}</b> · unsteered lane (right): <b>${b[0]}/${b[1]}</b>`
          + (guideOn() ? '' : ' · steering off');
      } else if (e.what === 'judge') {
        callOut(d.event);
        runLog?.push({ t: Math.round(d.t * 100) / 100, event: d.event, meteor: d.meteor, index: d.index, at: d.at.map((x) => Math.round(x * 10) / 10), car: d.car.map((x) => Math.round(x * 10) / 10) });
      } else if (e.what === 'status') { runStatus = d.status; runHud(); }
    }
  }

  // ---- meteor run: score, health and progress, the call-outs and the cards
  if (run) {
    overlay('<div class="score" id="pv-score">0</div><div class="meter" title="Health"><div id="pv-hp"></div></div><div class="meter" style="height: 5px" title="Distance to the finish"><div id="pv-prog"></div></div><div id="pv-ginfo"></div>', { id: 'pv-game' });
    overlay('', { id: 'pv-toasts' });
    overlay('<div id="pv-cardBody"></div>', { id: 'pv-card', hidden: true });
  }
  const toastsEl = $('pv-toasts');
  // Call-outs: one per kind, counting up a burst for up to 0.9 s, then a new one.
  const CALLOUTS = { 1: ['NEAR MISS', 25, 'good'], 2: ['HIT', 0, 'bad'], 3: ['CLOSE CALL', 150, 'big'], 4: ['METEOR HIT', -35, 'bad'] };
  const callOuts = new Map(); // code -> {el, n, born}
  function callOut(code) {
    if (!toastsEl) return;
    const [text, points, cls] = CALLOUTS[code] || ['?', 0, 'good'];
    const now = performance.now();
    let c = callOuts.get(code);
    if (!c || now - c.born > 900 || !c.el.isConnected) {
      const el = document.createElement('div');
      el.className = `pv-toast ${cls}`;
      el.addEventListener('animationend', () => el.remove());
      toastsEl.append(el);
      c = { el, n: 0, born: now };
      callOuts.set(code, c);
    }
    c.n += 1;
    const sum = points * c.n;
    c.el.textContent = `${text}${c.n > 1 ? ` ×${c.n}` : ''}${sum > 0 ? ` +${sum}` : sum < 0 ? ` −${-sum}` : ''}`;
    // Restart its rise.
    c.el.style.animation = 'none';
    void c.el.offsetWidth;
    c.el.style.animation = '';
    while (toastsEl.childElementCount > 4) toastsEl.firstChild.remove();
  }
  let runCardKey = '';
  function runHud() {
    const st = runStatus;
    if (!run || !st) return;
    $('pv-score').textContent = (st.state === 'finished' ? st.total : st.score).toLocaleString();
    $('pv-hp').style.width = `${st.health}%`;
    $('pv-prog').style.width = `${Math.min(100, (100 * st.distance) / st.finish)}%`;
    $('pv-ginfo').textContent = `${st.near} near misses · ${st.close} close calls · ${st.hits + st.meteorHits} hits · ${st.elapsed.toFixed(1)} s`
      + ` · directed debris ${guideOn() ? 'ON' : 'OFF'} (G)`;
    const k = `${st.state}:${guideOn()}`;
    if (k === runCardKey) return;
    runCardKey = k;
    const steer = guideOn()
      ? 'Directed debris is <b>on</b>: the blasts\' masonry and the falling meteors are nudged to land around you, not on you.'
      : 'Directed debris is <b>off</b>: everything flies where the blasts throw it.';
    const rows = (r) => `<table>${r.map(([a, b]) => `<tr><td>${a}</td><td>${b}</td></tr>`).join('')}</table>`;
    const body = $('pv-cardBody');
    $('pv-card').hidden = st.state === 'running';
    if (st.state === 'ready') {
      body.innerHTML = `<h2>METEOR GAUNTLET</h2>Drive to the end of the street through a meteor shower.<br>Near misses score; hits cost health.<br><br>`
        + `<kbd>W</kbd> to go · <kbd>A</kbd><kbd>D</kbd> steer · <kbd>S</kbd> brake · <kbd>G</kbd> directed debris · <kbd>R</kbd> restart<br><br><span class="muted">${steer}</span>`;
    } else if (st.state === 'wrecked') {
      body.innerHTML = `<h2>WRECKED</h2>${rows([['Distance', `${st.distance.toFixed(0)} / ${st.finish.toFixed(0)} m`], ['Near misses', st.near], ['Close calls', st.close], ['Score', st.score]])}<br>Press <kbd>R</kbd> to drive again`;
    } else if (st.state === 'finished') {
      body.innerHTML = `<h2>MADE IT</h2>${rows([['Near misses', `${st.near} × 25`], ['Close calls', `${st.close} × 150`], ['Health left', `+${st.healthBonus}`], [`Time ${st.elapsed.toFixed(1)} s`, `+${st.timeBonus}`], ['<b>Total</b>', `<b>${st.total.toLocaleString()}</b>`]])}<br>Press <kbd>R</kbd> to drive again · <kbd>G</kbd> to try it with directed debris ${guideOn() ? 'off' : 'on'}`;
    }
  }

  // ---- the gauntlet's directed-debris stats (a non-blocking read every 30 steps)
  let guideBusy = false, guideStatsStep = -1;
  function pollGuide() {
    if (!street || guideBusy || world.simSteps() - guideStatsStep < 30) return;
    guideStatsStep = world.simSteps();
    guideBusy = true;
    world.readGuidance().then((s) => {
      guideBusy = false;
      const claims = s.claims;
      const cars = claims.slice(2).reduce((a, b) => a + b, 0);
      guideStatsEl.textContent = guideOn()
        ? `steering ${s.steering} · steered so far ${s.done} (arrived ${s.ended.arrived}, knocked off course ${s.ended.pushed}) · aimed: around car ${claims[0]}, behind ${claims[1]}, parked cars ${cars}`
        : 'directed debris off: blocks fly as thrown';
    });
  }

  // ---- breaks: the bridge's bolts (joints) and mounts torn off under load
  const bridgeBolts = placed.bridgeBolts ?? 0;
  let bolts = { broken: 0, lastForce: 0 };
  const breakable = vehicles.some((_, i) => entry(i).desc.parts.some((p, k) => k > 0 && (p.mount?.break_force > 0 || p.mount?.break_torque > 0)));
  // Counted from the mount events (vehicle.html counted its buggy meshes' latched masks).
  let mounts = { parts: 0, wheels: 0, last: null };
  function onBreak(e) {
    if (e.type === 'joint') { bolts.broken++; bolts.lastForce = e.force; }
    else if (e.type === 'mount') {
      const vi = vehicles.findIndex((v) => v.h === e.vehicle);
      const desc = vi >= 0 ? entry(vi).desc : null;
      if (e.wheel) mounts.wheels++; else mounts.parts++;
      mounts.last = `${e.wheel ? `wheel ${e.index}` : desc?.parts[e.index]?.name ?? `part ${e.index}`} at ${(e.force / 1000).toFixed(0)} kN`;
    }
  }
  // Each mount's load against its limit, a few times a second (non-blocking read).
  let loadStep = -1, loadBusy = false;
  function pollLoads() {
    if (!breakable || !vehicles.length || loadBusy || world.simSteps() - loadStep < 15) return;
    loadStep = world.simSteps();
    loadBusy = true;
    const who = handle();
    world.readMountLoads(who).then((w) => { loadBusy = false; if (who === handle()) loads = world.vehicleMountLoads(who, w); });
  }
  const mountText = () => (breakable ? ` · broken off: ${mounts.parts} parts, ${mounts.wheels} wheels${mounts.last ? ` (last ${mounts.last})` : ''}` : '');
  const boltText = () => (bridgeBolts ? ` · bolts broken ${bolts.broken}/${bridgeBolts}${bolts.broken ? ` (last at ${(bolts.lastForce / 1000).toFixed(0)} kN)` : ''}` : '');

  // ---- parts & wheels: sever / disable each, the engine override
  let engineOn = true;
  function renderParts() {
    if (!vehicles.length) return;
    const desc = entry(player).desc, h = handle();
    partList.textContent = '';
    const lost = state ? state.partsLost : 0;
    const row = (html, gone, buttons) => {
      const r = document.createElement('div');
      r.className = 'pv-part' + (gone ? ' lost' : '');
      r.innerHTML = html;
      const btns = document.createElement('span');
      for (const [label, fn, disabled] of buttons) {
        const b = document.createElement('button');
        b.textContent = label;
        b.disabled = !!disabled;
        b.onclick = () => { fn(); b.blur(); };
        btns.append(b, ' ');
      }
      r.append(btns);
      partList.append(r);
    };
    desc.parts.forEach((p, i) => {
      const gone = (lost >> i) & 1;
      row(`<span>${p.name} <span class="muted">(${p.role.type.replace('_', ' ')})</span> <span class="muted load" data-part="${i}"></span></span>`, gone,
        [['sever', () => world.vehicleDamage(h, 'detachPart', i), gone], ['disable', () => world.vehicleDamage(h, 'disablePart', i), gone]]);
    });
    const names = desc.wheels.length === 4 ? ['FL', 'FR', 'RL', 'RR'] : desc.wheels.map((_, i) => 'W' + i);
    names.forEach((n, i) => {
      const gone = state && (state.wheels[i]?.detached || state.wheels[i]?.disabled);
      row(`<span>wheel ${n} <span class="muted load" data-wheel="${i}"></span></span>`, gone,
        [['sever', () => world.vehicleDamage(h, 'detachWheel', i), gone], ['remove', () => world.vehicleDamage(h, 'disableWheel', i), gone]]);
    });
    row('<span>engine override</span>', false, [[engineOn ? 'cut' : 'restore', () => {
      engineOn = !engineOn;
      world.vehicleSetFunctionalState(h, { engineConnected: engineOn });
      renderParts();
    }]]);
  }
  // TODO(rust): which wheel X / Z picks (a random attached one) and which part P severs
  // next are the page's choices, as vehicle.html made them.
  function severWheel(spawn) {
    const st = state;
    if (!st) return;
    const alive = st.wheels.map((w, i) => (!w.detached && !w.disabled ? i : -1)).filter((i) => i >= 0);
    if (!alive.length) return;
    const w = alive[Math.floor(Math.random() * alive.length)];
    world.vehicleDamage(handle(), spawn ? 'detachWheel' : 'disableWheel', w);
  }
  function severPart() {
    const desc = entry(player).desc;
    for (let i = 0; i < desc.parts.length; i++) {
      if (desc.parts.length > 1 && i === 0) continue; // keep the main body
      if (severed.has(i)) continue;
      severed.add(i);
      world.vehicleDamage(handle(), 'detachPart', i);
      renderParts();
      return;
    }
  }

  // ---- actions (the scene's director)
  const fireCannon = (a = aim()) => { if (placed.cannon?.length) { world.act({ action: 'cannon', aim: a }); onEvents(); } };
  // The scene's fire key: the gauntlet's shell, the range's volley.
  const fireShell = () => { if (street || range) { world.act({ action: 'fire' }); onEvents(); } };
  function resetScene() {
    world.restoreState(initial);
    if (run) { toastsEl.replaceChildren(); runCardKey = ''; }
    rangeScore = [[0, 0], [0, 0]];
    blocksLaunched.clear();
    for (const m of markerMeshes) m.visible = false;
    // The scripts start over (the street, the run, the range; directed debris re-armed).
    world.act({ action: 'reset' });
    onEvents();
    lastShell = 0;
    bolts = { broken: 0, lastForce: 0 };
    mounts = { parts: 0, wheels: 0, last: null };
    severed.clear();
    renderParts();
  }
  if (street || range) ui.button(range ? 'Volley' : 'Shell', 'KeyF', fireShell);
  if (placed.cannon?.length) ui.button('Cannon', 'KeyB', () => fireCannon());
  ui.button('Reset', 'KeyR', resetScene);
  // The meteor range opens on an orbit over both lanes; the rest chase the car.
  let chase = !range, pinnedOrbit = null;
  ui.button('Camera', 'KeyC', () => { chase = !chase; pinnedOrbit = null; });
  if (vehicles.length > 1) ui.button('Next car', 'Tab', () => {
    world.setVehicleInput(handle(), { forward: 0, steer: 0, brake: 1 });
    selectPlayer((player + 1) % vehicles.length);
  });
  if (vehicles.length) {
    ui.button('Sever wheel', 'KeyX', () => severWheel(true));
    ui.button('Remove wheel', 'KeyZ', () => severWheel(false));
    ui.button('Sever part', 'KeyP', severPart);
  }
  if (run) ui.button('Directed debris', 'KeyG', () => guideBox.click());
  // Dragging the view leaves the chase camera for the orbit (as vehicle.html's drag).
  on(controls, 'start', () => { chase = false; pinnedOrbit = null; });
  ui.help('<kbd>W</kbd><kbd>S</kbd> drive/brake · <kbd>A</kbd><kbd>D</kbd> steer · <kbd>Space</kbd> handbrake · drag to orbit (<kbd>C</kbd> chase / orbit)');

  // ---- the chassis pose, read a frame late (never waits)
  let reading = null;
  function readChassis() {
    if (reading || !vehicles.length) return reading;
    const who = player;
    reading = world.readBodies([entry(who).chassis]).then((f) => {
      reading = null;
      if (who !== player) return; // read before a player switch
      chassis = { p: [f[0], f[1], f[2]], q: [f[3], f[4], f[5], f[6]], v: [f[7], f[8], f[9]] };
    });
    return reading;
  }

  // ---- the dashboard: speed, gear, rpm, each wheel's load and slip
  function updateHud() {
    const st = state;
    if (!st) return;
    speedEl.innerHTML = `${Math.abs(st.forwardSpeed * 3.6).toFixed(0)} <small>km/h</small>`;
    gearEl.textContent = maxRpm ? (st.gear < 0 ? 'R' : st.gear === 0 ? 'N' : String(st.gear)) : 'D';
    rpmEl.style.width = maxRpm ? `${Math.min(100, (100 * st.engineRpm) / maxRpm)}%` : '0';
    rpmText.textContent = maxRpm ? `${st.engineRpm.toFixed(0)} rpm · clutch ${(st.clutch * 100).toFixed(0)}%${st.engineConnected ? '' : ' · ENGINE LOST'}` : `direct drive${st.engineConnected ? '' : ' · ENGINE LOST'}`;
    wheelsEl.innerHTML = st.wheels.map((w, i) => {
      const cls = w.detached || w.disabled ? 'gone' : w.sliding ? 'slide' : w.inContact ? 'contact' : '';
      return `<div class="pv-wheel ${cls}">#${i} ${w.detached ? 'severed' : w.disabled ? 'removed' : `${(w.suspensionForce / 1000).toFixed(1)}kN · κ${w.slipRatio.toFixed(2)}`}</div>`;
    }).join('');
    const k = `${st.partsLost}/${st.wheels.map((w) => +w.detached + 2 * +w.disabled).join('')}`;
    if (k !== partsKey) { partsKey = k; renderParts(); }
    if (loads) {
      // Each mount's heaviest load so far, as a share of its limit.
      for (const el of partList.querySelectorAll('.load')) {
        const l = el.dataset.part !== undefined ? loads.find((x) => !x.wheel && x.index === +el.dataset.part) : loads.find((x) => x.wheel && x.index === +el.dataset.wheel);
        el.textContent = l && l.breakForce < 1e29 ? `peak ${(100 * l.peakRatio).toFixed(0)}%` : '';
      }
    }
  }

  // ---- camera: chase behind the car, or orbit around it (drag), following it
  let snapped = false, recordDt = 0;
  function frame(dt) {
    if (!dt && recordDt) dt = recordDt;
    readChassis();
    onEvents();
    pollGuide();
    pollLoads();
    updateHud();
    if (!chassis) return;
    const f = qrot(chassis.q, [0, 0, 1]);
    const h = Math.hypot(f[0], f[2]) || 1;
    const flat = [f[0] / h, 0, f[2] / h];
    const p = chassis.p;
    // The meteor run looks further up the street, into the sky they come from.
    const ahead = run && chase ? 8 : 0, lift = run && chase ? 1.8 : 1.0;
    const wantT = [p[0] + flat[0] * ahead, p[1] + lift, p[2] + flat[2] * ahead];
    const orbitEye = (at, { yaw, pitch, dist }) => {
      const cp = Math.cos(pitch);
      return [at[0] + dist * cp * Math.sin(yaw), at[1] + dist * Math.sin(pitch), at[2] + dist * cp * Math.cos(yaw)];
    };
    const back = chaseDist + ahead;
    const chaseOff = [-flat[0] * back, chaseDist * (run ? 0.48 : 0.38) - lift, -flat[2] * back];
    const t = controls.target, e = camera.position;
    if (!snapped) {
      snapped = true;
      t.set(...wantT);
      e.set(...(chase ? wantT.map((x, i) => x + chaseOff[i])
        : orbitEye(wantT, pinnedOrbit ?? (range ? { yaw: Math.PI, pitch: 0.75, dist: 52 } : { yaw: 0.6, pitch: 0.35, dist: 16 }))));
      return;
    }
    // Smooth the eye's offset from the target (not its world position), so a fast car
    // stays the same size on screen.
    const k = 1 - Math.pow(0.0015, dt);
    const off = [e.x - t.x, e.y - t.y, e.z - t.z];
    const nt = [t.x, t.y, t.z].map((x, i) => x + (wantT[i] - x) * Math.min(1, k * 4));
    t.set(...nt);
    if (chase) e.set(...nt.map((x, i) => x + off[i] + (chaseOff[i] - off[i]) * k));
    else if (pinnedOrbit) e.set(...orbitEye(nt, pinnedOrbit));
    // Orbit: the view the user dragged to, carried along with the car.
    else e.set(...nt.map((x, i) => x + off[i]));
  }

  if (vehicles.length) selectPlayer(player);

  // ---- automation: vehicle.html's hooks (__step is the shell's)
  const resume = () => { scripted = false; autofire(); window.__run?.(); };
  // n steps frame-locked through the shell (its frame handler sees every poll), each
  // after `each()`; then the chassis read.
  const advance = async (n, each = null) => {
    world.setAutofire(false);
    for (let i = 0; i < n; i++) { each?.(); await stepOne(); }
    await drain();
    await readChassis();
  };
  // Frame-locked runs: the scripts' reads taken (the chassis, touchdowns, the run's judge).
  const settleScripts = async () => {
    world.prepareFrame();
    while (world.scriptPending()) await macrotask();
    world.beforeFrame(0, -1);
    onEvents();
  };
  const setGuide = (guided, strength) => { guideBox.checked = guided; guideSel.value = String(strength); applyGuidance(); };
  // TODO(rust): the takes' drivers (cruise control, lane keeping) and measurements below
  // are vehicle.html's JS, ported as they were.
  const hooks = {
    __vehicleState: () => state,
    __setVehicleInput: (h, input) => { if (h === handle()) scripted = true; world.setVehicleInput(h, input); },
    // A ball fired along `v` from 3 m short of `p` (the cannon fires from 3 m ahead of the
    // eye at 45 m/s).
    __cannon: (p, v) => { const d = norm3(v); fireCannon({ eye: p.map((x, k) => x - d[k] * 3), dir: d }); },
    // Chassis poses of every vehicle: [{name, p, up, speed}] (one GPU read).
    __chassisPoses: async () => {
      const f = await world.readBodies(vehicles.map((_, i) => entry(i).chassis));
      return vehicles.map((v, i) => {
        const o = i * 14;
        const q = [f[o + 3], f[o + 4], f[o + 5], f[o + 6]];
        return { name: v.name, p: [f[o], f[o + 1], f[o + 2]], up: qrot(q, [0, 1, 0]), speed: Math.hypot(f[o + 7], f[o + 8], f[o + 9]) };
      });
    },
    // Parts / wheels broken off under load, and the latest break (no hybrid frames here:
    // `loose` and `groups` stay empty).
    __mounts: () => ({ parts: mounts.parts, wheels: mounts.wheels, loose: [], groups: [], last: mounts.last }),
    // __record(script, {name, every, width, upload}) plays a scripted drive frame-locked
    // (one physics step per video frame at 60 Hz, through the shell's __step) and PUTs
    // each composited frame (canvas + caption) as JPEG to `${upload}/${name}/f00000.jpg`.
    // Script items: {n, input?, cruise?, act?: 'wheel:<i>'|'remove:<i>'|'part'|'cannon'|
    // 'reset'|'player:<i>', cam?: 'chase' | {orbit: [yaw, pitch, dist], spin?}, caption?}.
    __record: async (script, opts = {}) => {
      scripted = true;
      world.setAutofire(false);
      const canvas = ctx.renderer.domElement;
      const name = opts.name || 'take', every = opts.every || 1, upload = opts.upload || '/rec';
      const W = Math.min(opts.width || 1280, canvas.width);
      const H = Math.round((W * canvas.height) / canvas.width / 2) * 2;
      const out = new OffscreenCanvas(W, H);
      const g = out.getContext('2d');
      let frameNo = 0, caption = '', spin = 0;
      const pending = new Set();
      const t0 = performance.now();
      recordDt = 1 / 60;
      try {
        for (const seg of script) {
          if (seg.input) world.setVehicleInput(handle(), seg.input);
          if (seg.caption !== undefined) caption = seg.caption;
          if (seg.cam === 'chase') { chase = true; pinnedOrbit = null; spin = 0; }
          else if (seg.cam?.orbit) { chase = false; const [yaw, pitch, dist] = seg.cam.orbit; pinnedOrbit = { yaw, pitch, dist }; spin = seg.cam.spin || 0; }
          if (seg.act) {
            const [a, arg] = seg.act.split(':');
            if (a === 'wheel') world.vehicleDamage(handle(), 'detachWheel', Number(arg));
            else if (a === 'remove') world.vehicleDamage(handle(), 'disableWheel', Number(arg));
            else if (a === 'part') severPart();
            else if (a === 'cannon') fireCannon();
            else if (a === 'reset') resetScene();
            else if (a === 'player') { world.setVehicleInput(handle(), { forward: 0, steer: 0, brake: 1 }); selectPlayer(Number(arg)); }
          }
          for (let i = 0; i < (seg.n || 0); i++) {
            if (pinnedOrbit) pinnedOrbit.yaw += spin;
            if (seg.cruise !== undefined && state) {
              // Hold a target forward speed (m/s) with throttle / brake.
              const e = seg.cruise - state.forwardSpeed;
              world.setVehicleInput(handle(), { ...(seg.input || {}), forward: Math.max(0, Math.min(1, e * 0.4)), brake: e < -1 ? Math.min(1, -e * 0.2) : 0 });
            }
            if (frameNo === 0) snapped = false;
            // The shell's __step: the step, this module's frame (camera, HUD), a draw.
            await window.__step(1);
            await readChassis();
            if (world.simSteps() % every !== 0) continue;
            g.drawImage(canvas, 0, 0, W, H);
            const st = state;
            g.fillStyle = 'rgba(10,14,22,0.62)';
            g.fillRect(12, H - 74, Math.min(W - 24, 720), 62);
            g.fillStyle = '#fff';
            g.font = '600 22px system-ui, sans-serif';
            g.fillText(caption || playerDesc.name.replaceAll('_', ' '), 24, H - 46);
            g.font = '16px ui-monospace, monospace';
            if (st) {
              const gear = maxRpm ? (st.gear < 0 ? 'R' : st.gear === 0 ? 'N' : String(st.gear)) : 'D';
              g.fillText(bridgeBolts
                ? `${playerDesc.name.replaceAll('_', ' ')}  ${Math.abs(st.forwardSpeed * 3.6).toFixed(0).padStart(3)} km/h  bolts broken ${bolts.broken}/${bridgeBolts}  step ${world.simSteps()}`
                : `${Math.abs(st.forwardSpeed * 3.6).toFixed(0).padStart(3)} km/h  gear ${gear}  ${maxRpm ? st.engineRpm.toFixed(0) + ' rpm' : ''}  yaw ${st.yawRate.toFixed(2)}  contacts ${st.contacts}  step ${world.simSteps()}`, 24, H - 22);
            }
            const blob = await out.convertToBlob({ type: 'image/jpeg', quality: 0.88 });
            const p = fetch(`${upload}/${name}/f${String(frameNo++).padStart(5, '0')}.jpg`, { method: 'PUT', body: blob }).then(() => pending.delete(p));
            pending.add(p);
            if (pending.size > 16) await Promise.race(pending);
          }
        }
        await Promise.all(pending);
      } finally { recordDt = 0; autofire(); }
      return { name, frames: frameNo, width: W, height: H, seconds: (performance.now() - t0) / 1000, state, chassis };
    },
  };
  // Gauntlet A/B take: from a reset, drive down the street at `cruise` m/s holding its
  // lane, shelling every `every` steps, with directed debris on or off. Each launched
  // block's first touchdown (sampled every 6 steps) is measured from the car at that
  // moment: `landings` histograms the plan-view distances (m), `near` counts those within
  // `near` m and `close` those within 2 m.
  if (street) hooks.__gauntletTake = async ({ steps: total = 720, guided = true, strength = 1, cruise = 11, every = 70, near = 6 } = {}) => {
    scripted = true;
    world.setAutofire(false);
    setGuide(guided, strength);
    resetScene();
    const slots = [entry(0).chassis, ...placed.blocks.map((b) => b.slot)];
    const landed = new Map(); // block -> distance from the car at touchdown
    let shells = 0;
    const trace = [];
    try {
      for (let s = 0; s < total; s += 6) {
        const f = await world.readBodies(slots);
        chassis = { p: [f[0], f[1], f[2]], q: [f[3], f[4], f[5], f[6]], v: [f[7], f[8], f[9]] };
        for (let i = 0; i < placed.blocks.length; i++) {
          const o = (i + 1) * 14;
          if (landed.has(i) || !blocksLaunched.has(i) || f[o + 1] > 0.6) continue;
          landed.set(i, Math.hypot(f[o] - f[0], f[o + 2] - f[2]));
        }
        const fw = qrot(chassis.q, [0, 0, 1]);
        const speed = fw[0] * chassis.v[0] + fw[2] * chassis.v[2];
        if (s % 60 === 0) trace.push([Math.round(chassis.p[0] * 10) / 10, Math.round(chassis.p[2]), Math.round(speed * 10) / 10]);
        const steerIn = Math.max(-1, Math.min(1, 0.25 * (chassis.p[0] + 2.5) + 2.5 * fw[0]));
        const e = cruise - speed;
        world.setVehicleInput(handle(), { forward: Math.max(0, Math.min(1, e * 0.4)), steer: steerIn, brake: e < -1 ? Math.min(1, -e * 0.2) : 0, handbrake: 0 });
        if (chassis.p[2] > -12 && chassis.p[2] < 150 && world.simSteps() - lastShell >= every) { await settleScripts(); fireShell(); shells++; }
        await advance(6);
      }
    } finally { resume(); }
    const { claims, ...states } = await world.readGuidance();
    const d = [...landed.values()];
    const landings = [2, 4, 6, 8, 12, 20, Infinity].map((r, k, a) => d.filter((x) => x < r && x >= (a[k - 1] ?? 0)).length);
    return { guided, strength, shells, landed: d.length, near: d.filter((x) => x < near).length, close: d.filter((x) => x < 2).length, landings, z: chassis.p[2], trace, states, claims: claims.slice(0, world.scriptState().zones) };
  };
  // Meteor range take: from a reset, `volleys` volleys two seconds apart with steering on
  // or off; returns each lane's [on target, landed].
  if (range) hooks.__meteorTake = async ({ volleys = 10, guided = true, strength = 1, moving = false, routes = true } = {}) => {
    scripted = true;
    world.setAutofire(false);
    guideBox.checked = guided; guideSel.value = String(strength);
    world.act({ action: 'targets', moving, routes });
    applyGuidance();
    resetScene();
    rangeLog = [];
    let s = 0;
    const each = () => { if (s % 120 === 0 && s < volleys * 120) fireShell(); s++; };
    try {
      while (s < volleys * 120 + 240) {
        await advance(6, each);
        await settleScripts();
      }
    } finally { resume(); }
    const log = rangeLog;
    rangeLog = null;
    return { guided, strength, moving, routes, steered: rangeScore[0], unsteered: rangeScore[1], log };
  };
  // Meteor run take: from a reset, drive the street at `cruise` m/s holding the lane at
  // x = `lane`, with directed debris on or off, until the run ends or `steps` steps;
  // returns the run's status.
  if (run) hooks.__runTake = async ({ steps: total = 2400, guided = true, strength = 1, cruise = 11, lane = -2.5 } = {}) => {
    scripted = true;
    setGuide(guided, strength);
    resetScene();
    // The run's director meteors on its own; the take steers and reads its judge.
    world.setAutofire(true);
    runLog = [];
    let st = null;
    try {
      for (let s = 0; s < total; s += 6) {
        await readChassis();
        const fw = qrot(chassis.q, [0, 0, 1]);
        const speed = fw[0] * chassis.v[0] + fw[2] * chassis.v[2];
        const e = cruise - speed;
        world.setVehicleInput(handle(), { forward: Math.max(0, Math.min(1, e * 0.4)), steer: Math.max(-1, Math.min(1, 0.25 * (chassis.p[0] - lane) + 2.5 * fw[0])), brake: e < -1 ? Math.min(1, -e * 0.2) : 0, handbrake: 0 });
        await settleScripts();
        st = runStatus;
        if (st && (st.state === 'wrecked' || st.state === 'finished')) break;
        for (let i = 0; i < Math.min(6, total - s); i++) await stepOne();
        await drain();
      }
    } finally { resume(); }
    const log = runLog;
    runLog = null;
    const { claims, ...states } = await world.readGuidance();
    return { guided, strength, ...st, z: chassis.p[2], states, hits: log.filter((e) => e.event === 2 || e.event === 4) };
  };

  return {
    beforeStep() { if (vehicles.length && !scripted) world.setVehicleInput(handle(), readInput()); },
    onFrame(f) {
      if (f.empty) return;
      if (vehicles.length) state = f.vehicleState(handle()) ?? state;
      for (const e of f.destructionEvents()) onBreak(e);
    },
    frame,
    keydown(e) {
      if (['Tab', 'Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code)) e.preventDefault();
      scripted = false; // a key takes the car back from scripted input
    },
    hud() {
      const st = state;
      const speed = st ? Math.abs(st.forwardSpeed * 3.6) : chassis ? Math.hypot(...chassis.v) * 3.6 : 0;
      return `${vehicles.length} vehicles · ${vehicles[player]?.name?.replaceAll('_', ' ') ?? ''} · ${speed.toFixed(0)} km/h · ${chase ? 'chase' : 'orbit'} cam`
        + boltText() + mountText() + (recent.length ? ` · <span class="muted">${recent.join(', ')}</span>` : '');
    },
    hooks,
    dispose() { for (const d of disposers) d(); },
  };
}
