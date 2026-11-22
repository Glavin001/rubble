// The motion family in the playground shell (play.html): the CCD gallery, the character
// playground and bullets through walls (rubble::play::motion; their script and actions are
// rubble::play::director). Its menus, keys, HUD (the lanes' verdicts, the player, the log),
// camera and checks (__check) — motion.html's, moved. Bodies, characters and projectile
// tracers are drawn from the World by rubble-three.mjs.

import { Poses } from './rubble-world.mjs';

const SCENES = { ccd: 'Tunnelling gallery: CCD off vs on', character: 'Character playground', bullets: 'Bullets through walls' };
const SPEEDS = [60, 140, 250];
const DT = 1 / 60; // the scene's step, in f64 for the page's clocks
const HELP = {
  ccd: '<kbd>F</kbd> fire a volley · <kbd>R</kbd> reset · drag to orbit · scroll to zoom',
  character: '<kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> walk · <kbd>Space</kbd> jump · <kbd>F</kbd> bullet · <kbd>B</kbd> CCD rocket · <kbd>R</kbd> reset · drag to look',
  bullets: '<kbd>F</kbd> fire · <kbd>R</kbd> reset · drag to orbit',
};

export async function setup({ world, placed, sceneDef, ui, aim: shellAim, lookFrom, setShadowRadius, camera, controls: orbit }) {
  const scene = sceneDef.scene;
  const shotSpeed = sceneDef.speed, lookAhead = sceneDef.lookAhead;
  // Menus: scene, shot speed, look-ahead (each reloads with its switch).
  const go = (k, v) => { const u = new URL(location.href); u.searchParams.set(k, v); location.href = u.href; };
  const controls = document.getElementById('controls');
  const sceneSel = document.createElement('select');
  for (const [id, t] of Object.entries(SCENES)) sceneSel.add(new Option(t, id, false, id === scene));
  sceneSel.onchange = () => go('scene', sceneSel.value);
  const speedSel = document.createElement('select');
  for (const v of SPEEDS) speedSel.add(new Option(`${v} m/s`, v, false, v === shotSpeed));
  speedSel.onchange = () => go('speed', speedSel.value);
  const la = document.createElement('label');
  la.innerHTML = '<input type="checkbox"> look-ahead';
  la.firstChild.checked = lookAhead;
  la.firstChild.onchange = (e) => go('lookahead', e.target.checked ? '1' : '0');
  controls.append(sceneSel, speedSel, la);
  ui.help(HELP[scene]);
  const logEl = document.createElement('pre');
  logEl.className = 'muted';
  logEl.style.cssText = 'max-height: 140px; overflow: auto; margin: 4px 0 0; font-size: 11px;';
  document.getElementById('hud').append(logEl);

  // Characters: handle -> { start, jumps sent }.
  const characters = new Map(placed.characters.map((c) => [c.h, { start: c.start, jumps: 0 }]));
  // Input with a jump counter (a new value jumps once), as the HUD keys count jumps.
  const setCharacterInput = (h, v, jumpSeq) => {
    const c = characters.get(h);
    world.characterSetInput(h, v, jumpSeq !== c.jumps);
    c.jumps = jumpSeq;
  };
  // Scene state the HUD and the checks read.
  const S = {
    lanes: placed.lanes.map((L) => ({ ...L, result: { control: '—', ccd: '—' }, tally: { control: 0, ccd: 0, shots: 0 } })),
    shower: placed.shower, player: placed.player ?? -1, npcs: placed.npcs, platform: placed.platform, lift: placed.lift,
    rockets: placed.rockets, walls: placed.walls, targets: placed.targets, log: [],
  };
  const say = (m) => { S.log.unshift(`${world.stepCount()}: ${m}`); S.log.length = Math.min(S.log.length, 40); };

  // The script fires volleys and bursts on its cadence and the player's actions; the HUD
  // follows what it reports.
  function onEvent(e) {
    if (e.what === 'volley') {
      for (const L of S.lanes) {
        L.fired = e.step;
        L.result = { control: 'flying', ccd: 'flying' };
        L.tally.shots++;
      }
      if (S.shower) S.shower.fired = e.step;
      say(`volley at ${shotSpeed} m/s`);
    } else if (e.what === 'burst') say('3 bullets at 900 m/s (5,000 J/m to cross a wall)');
    else if (e.what === 'bullet') say('bullet at 600 m/s');
    else if (e.what === 'rocket') say('CCD rocket at 150 m/s');
  }
  const takeEvents = () => { for (const e of world.takeEvents()) onEvent(e); };

  // The camera: each scene's view (motion.html's), orbiting its target.
  const CAM = {
    ccd: { yaw: 2.75, pitch: 0.42, dist: 30, target: [0, 1, 24] },
    bullets: { yaw: 2.3, pitch: 0.4, dist: 18, target: [0, 1, 16] },
    character: { yaw: 0.6, pitch: 0.4, dist: 9, target: [0, 1, 0] },
  }[scene];
  const cp = Math.cos(CAM.pitch);
  lookFrom([CAM.target[0] + CAM.dist * cp * Math.sin(CAM.yaw), CAM.target[1] + CAM.dist * Math.sin(CAM.pitch), CAM.target[2] + CAM.dist * cp * Math.cos(CAM.yaw)], CAM.target);
  setShadowRadius(scene === 'ccd' ? 50 : 30);

  // Keys: held ones for walking; F, B, R, Space.
  const keys = new Set();
  let jumpSeq = 0;
  addEventListener('keyup', (e) => keys.delete(e.code));
  addEventListener('blur', () => keys.clear());
  const key = (...c) => c.some((k) => keys.has(k));
  const aim = shellAim;
  function fire() {
    world.act({ action: 'fire', aim: scene === 'character' ? aim() : undefined });
    takeEvents();
  }
  function rocket() {
    if (scene !== 'character') return;
    world.act({ action: 'cannon', aim: aim() });
    takeEvents();
  }
  function resetScene() {
    // Every body and character back to its start, at rest (parked bodies asleep again).
    world.act({ action: 'reset' });
    takeEvents();
    for (const L of S.lanes) { L.result = { control: '—', ccd: '—' }; L.tally = { control: 0, ccd: 0, shots: 0 }; }
    S.log.length = 0;
    jumpSeq = 0;
  }
  function keydown(e) {
    keys.add(e.code);
    if (['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code)) e.preventDefault();
    if (e.repeat) return;
    if (e.code === 'KeyR') resetScene();
    else if (e.code === 'KeyF') fire();
    else if (e.code === 'KeyB') rocket();
    else if (e.code === 'Space') jumpSeq++;
  }
  // The player's input before each step; the NPCs, platform and lift are the script's.
  function beforeStep() {
    if (scene !== 'character') return;
    const fwd = (key('KeyW', 'ArrowUp') ? 1 : 0) - (key('KeyS', 'ArrowDown') ? 1 : 0);
    const side = (key('KeyD', 'ArrowRight') ? 1 : 0) - (key('KeyA', 'ArrowLeft') ? 1 : 0);
    // Forward: the camera's heading on the ground.
    const d = aim().dir;
    const h = Math.hypot(d[0], d[2]) || 1;
    const fx = d[0] / h, fz = d[2] / h;
    const speed = key('ShiftLeft') ? 6 : 3.5;
    let vx = (fx * fwd - fz * side), vz = (fz * fwd + fx * side);
    const l = Math.hypot(vx, vz) || 1;
    if (fwd || side) { vx = (vx / l) * speed; vz = (vz / l) * speed; } else { vx = 0; vz = 0; }
    if (!S.scripted) setCharacterInput(S.player, [vx, 0, vz], jumpSeq);
  }

  // Frames: poses, characters and projectiles, their events, the lanes' verdicts.
  const poses = new Poses(4096);
  let lastCounters = null;
  const bodyState = { get: (id) => ({ p: poses.position(world.bodySlot(id)) }) };
  function onFrame(f) {
    takeEvents();
    if (f.empty) return;
    poses.apply(f);
    const c = f.counters();
    if (c) {
      lastCounters = { contact: c.contacts, candidate: c.candidates, awake: c.awake, overflow: c.overflow };
      if (c.overflow) say(`overflow flags 0x${c.overflow.toString(16)}`);
    }
    const motion = { characters: f.characters(), projectiles: f.projectiles() };
    if (motion.characters.length || motion.projectiles.length || !S.motion) {
      motion.characters = Object.fromEntries(motion.characters.map((c) => [c.id, c]));
      S.motion = { ...S.motion, ...motion };
    }
    for (const e of f.projectileEvents()) {
      let collider = '?';
      try { collider = world.shapeCollider(e.shape); } catch {}
      say(`projectile ${e.id % 2 ** 32} hit collider ${collider}${e.type === 'stopped' ? ' and stopped' : ` (${e.thickness.toFixed(2)} m through)`}`);
    }
    for (const e of f.characterEvents()) say(`character ${e.id % 2 ** 32} ${e.type}`);
    judge();
  }
  // Lanes: a ball past the wall's far face tunnelled; one stopped short of it held.
  function judge() {
    for (const L of S.lanes) {
      if (L.fired === undefined) continue;
      for (const side of ['control', 'ccd']) {
        const st = bodyState.get(L[side]);
        if (!st || L.result[side] !== 'flying') continue;
        const far = L.wallZ + L.depth;
        const r = 0.15;
        if (st.p[2] - r > far + 0.05) {
          const moved = (L[side + 'Bricks'] || []).some((b) => { const q = bodyState.get(b); return q && q.p[2] > L.wallZ + 0.15; });
          L.result[side] = L.dynamic && moved ? 'smashed through' : 'TUNNELLED';
          if (side === 'ccd' && L.result[side] === 'TUNNELLED') say(`${L.name}: CCD ball tunnelled`);
        } else if (world.stepCount() - L.fired > 30 && st.p[2] < far) { L.result[side] = st.p[2] + r <= L.wallZ + 0.03 ? 'stopped at wall' : 'embedded'; }
      }
    }
    if (S.shower && S.shower.fired !== undefined) {
      for (const side of ['control', 'ccd']) S.shower[side + 'Through'] = S.shower[side].filter((c) => (bodyState.get(c.body)?.p[1] ?? 99) < 5.9 && (bodyState.get(c.body)?.p[1] ?? 99) > -100).length;
    }
  }
  const cls = (r) => r === 'TUNNELLED' ? 'bad' : r === 'flying' || r === '—' ? 'muted' : r === 'embedded' ? 'warn' : 'ok';
  function hud() {
    let h = '';
    if (scene === 'ccd') {
      h += '<table><tr><th>wall</th><th>CCD off</th><th>CCD on</th></tr>';
      for (const L of S.lanes) h += `<tr><td>${L.name}</td><td class="${cls(L.result.control)}">${L.result.control}</td><td class="${cls(L.result.ccd)}">${L.result.ccd}</td></tr>`;
      h += '</table>';
      if (S.shower?.fired !== undefined) h += `<div class="row">40 m/s debris through the slab: <span class="bad">off ${S.shower.controlThrough ?? 0}/49</span> · <span class="ok">on ${S.shower.ccdThrough ?? 0}/49</span></div>`;
      h += `<div class="muted">0.1 m balls at ${shotSpeed} m/s (${(shotSpeed * DT).toFixed(2)} m per step) · look-ahead ${lookAhead ? 'on' : 'off'}</div>`;
    } else if (S.motion) {
      const c = S.motion.characters[S.player];
      if (c) h += `<div>player: <b class="${c.ground === 'ground' ? 'ok' : c.ground === 'steep' ? 'warn' : ''}">${c.ground}</b> · (${c.position.map((x) => x.toFixed(2)).join(', ')}) · ${Math.hypot(c.velocity[0], c.velocity[2]).toFixed(2)} m/s${c.push[1] < 0 ? ` · ${(-c.push[1]).toFixed(0)} N on body ${c.groundBody}` : ''}</div>`;
      const alive = S.motion.projectiles.filter((p) => p.alive).length;
      h += `<div class="muted">${characters.size} characters · ${alive}/${S.motion.projectiles.length} projectiles in flight</div>`;
    }
    logEl.textContent = S.log.join('\n');
    return h + `<span class="muted">bodies ${world.bodyCount()}</span>`;
  }
  // The camera follows the player in the character playground.
  function frame(dt) {
    if (scene === 'character' && S.motion?.characters[S.player]) {
      const p = S.motion.characters[S.player].position;
      const k = Math.min(1, dt * 8);
      const t = orbit.target;
      const dx = (p[0] - t.x) * k, dy = (p[1] + 0.8 - t.y) * k, dz = (p[2] - t.z) * k;
      t.set(t.x + dx, t.y + dy, t.z + dz);
      camera.position.set(camera.position.x + dx, camera.position.y + dy, camera.position.z + dz);
    }
  }

  // __check(): the scene's end-to-end script, run frame-locked; returns metrics.
  async function __check() {
    world.setAutofire(false);
    if (scene === 'ccd') {
      await window.__step(5);
      fire();
      await window.__step(90);
      const lanes = S.lanes.map((L) => {
        const st = (side) => bodyState.get(L[side]);
        return { name: L.name, control: L.result.control, ccd: L.result.ccd, controlZ: st('control').p[2], ccdZ: st('ccd').p[2] };
      });
      return { scene, speed: shotSpeed, lookAhead, lanes, showerThrough: { control: S.shower.controlThrough, ccd: S.shower.ccdThrough }, counters: lastCounters };
    }
    if (scene === 'bullets') {
      await window.__step(5);
      fire();
      await window.__step(10);
      const hits = S.log.filter((l) => l.includes('hit collider'));
      const st = await world.readBodies(S.targets.map((t) => world.bodySlot(t)));
      const targets = S.targets.map((t, k) => ({ p: Array.from(st.subarray(k * 14, k * 14 + 3)), v: Array.from(st.subarray(k * 14 + 7, k * 14 + 10)) }));
      return { scene, projectiles: S.motion.projectiles, hitLines: hits, targets };
    }
    // Character: walk to the stairs and up them, jump on top, ride the platform.
    S.scripted = true;
    const ground = [];
    const walk = async (v, n) => { setCharacterInput(S.player, v, jumpSeq); for (let i = 0; i < n; i += 4) { await window.__step(4); ground.push(S.motion.characters[S.player].ground); } };
    await walk([0, 0, 0], 30);
    await walk([0, 0, 3], 120);
    const atLane = S.motion.characters[S.player].position.slice();
    await walk([3, 0, 0], 200);
    const top = S.motion.characters[S.player];
    jumpSeq++;
    await walk([0, 0, 0], 60);
    return { scene, atLane, top: { position: top.position, ground: top.ground }, grounds: ground, events: S.log.slice(0, 12), counters: lastCounters };
  }

  return {
    onFrame, beforeStep, frame, keydown, hud,
    hooks: { __check, __scene: S },
  };
}
