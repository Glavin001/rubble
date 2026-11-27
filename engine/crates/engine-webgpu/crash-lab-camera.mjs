// The crash lab page's camera and input (crash-lab.html): the orbit camera (drag, wheel) and
// the views on the vehicle (chase, front, rear, sides, top, driver's seat, mounted: #camSel,
// ?cam=, C cycles), each frame's eye and target eased toward the subject's pose carried to
// the step on display; the keys (R restart, L launch, C camera, V beams) and the driver's
// input (WASD / arrows, Space handbrake, a gamepad). State: page.cam, page.keys, page.labView.

import { qrot, norm3, viewProj, quatFromBasis } from './view-math.mjs';

const XRATE = 2000;
const CAM_MODES = [['orbit', 'orbit (free)'], ['chase', 'chase'], ['front', 'front'], ['rear', 'rear'], ['left', 'left side (port)'], ['right', 'right side (starboard)'], ['top', 'top down'], ['driver', "driver's seat"], ['mount', 'mounted on… looking at target']];

/** page.cam, the camera selects, the pointer and keys, page.updateCamera and page.readInput. */
export function createCamera(page) {
  const { cfg, qs, canvas } = page;
  const boxes = page.layout.boxes;
  // Driving: the chase camera behind the player; otherwise the orbit (the arena from high up).
  const cam = page.cam = {
    mode: cfg.drive ? 'chase' : 'orbit',
    orbit: cfg.arena ? { yaw: 0.7, pitch: 0.55, dist: 650, target: [0, 1, boxes.length ? boxes.reduce((a, b) => a + b[2], 0) / boxes.length : 0] } : { yaw: 2.2, pitch: 0.3, dist: 11, target: null },
    eye: [0, 6, -14], target: [0, 1, 0],
    // Camera subjects (?camTarget=, ?camFrom= by name); `mount`: where a camera rides on it.
    subjects: [{ name: 'car 0 (player, explicit)', key: 'car0', body: -1, mount: [0.36, 0.95, 0.05] }],
    targetIdx: 0, fromIdx: 0, targetPose: null, fromPose: null,
  };
  const findSubject = (k, d = 0) => { const i = cam.subjects.findIndex((x) => x.key === k || x.name === k); return i < 0 ? d : i; };
  page.findSubject = findSubject;
  cam.targetIdx = findSubject(qs.get('camTarget'), 0);
  cam.fromIdx = findSubject(qs.get('camFrom'), Math.max(0, cam.subjects.length - 1));
  if (qs.get('cam') && CAM_MODES.some(([m]) => m === qs.get('cam'))) cam.mode = qs.get('cam');
  selects(page);
  // Orbit: drag turns it (and leaves the chase), the wheel zooms.
  let drag = null;
  canvas.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY }; });
  addEventListener('pointerup', () => { drag = null; });
  addEventListener('pointermove', (e) => {
    if (!drag) return;
    cam.orbit.yaw -= (e.clientX - drag.x) * 0.005;
    cam.orbit.pitch = Math.max(-0.2, Math.min(1.4, cam.orbit.pitch + (e.clientY - drag.y) * 0.005));
    drag = { x: e.clientX, y: e.clientY };
    if (cam.mode === 'chase') { cam.mode = 'orbit'; page.sync(); }
  });
  canvas.addEventListener('wheel', (e) => { e.preventDefault(); cam.orbit.dist = Math.max(3, Math.min(cfg.arena ? 3000 : 120, cam.orbit.dist * Math.exp(e.deltaY * 0.001))); }, { passive: false });
  keyboard(page);
  page.updateCamera = (dt, snap) => { updateCamera(page, dt, snap); page.draw.writeFrame(); };
  page.camModes = CAM_MODES;
}

// #camSel (the car-relative views; in the lab the orbit cameras and the chase are the lab's
// own group), #camTargetSel and #camFromSel. page.sync redraws them (the lab adds its own).
function selects(page) {
  const { cam, el, cfg } = page;
  const { camSel, camTargetSel: tSel, camFromSel: fSel } = el;
  const LAB = cfg.lab;
  if (LAB) { camSel.add(new Option('car views…', '')); camSel.title = 'Cameras on the vehicle (C cycles every camera)'; }
  for (const [m, label] of CAM_MODES) if (!LAB || (m !== 'orbit' && m !== 'chase')) camSel.add(new Option(label, m));
  cam.subjects.forEach((x, i) => { tSel.add(new Option(x.name, i)); fSel.add(new Option(x.name, i)); });
  const sync = () => {
    camSel.value = LAB && (cam.mode === 'orbit' || cam.mode === 'chase') ? '' : cam.mode; tSel.value = cam.targetIdx; fSel.value = cam.fromIdx;
    camSel.classList.toggle('on', camSel.value !== '');
    const mount = cam.mode === 'mount';
    fSel.style.display = mount ? '' : 'none'; el.camFromLbl.style.display = mount ? '' : 'none';
    tSel.style.display = cam.mode === 'orbit' || (LAB && camSel.value === '') ? 'none' : '';
  };
  camSel.onchange = (e) => { if (!camSel.value) { sync(); return; } cam.mode = camSel.value; if (cam.mode === 'orbit') cam.orbit.target = null; sync(); e.target.blur(); };
  tSel.onchange = (e) => { cam.targetIdx = Number(tSel.value); e.target.blur(); };
  fSel.onchange = (e) => { cam.fromIdx = Number(fSel.value); e.target.blur(); };
  page.sync = sync;
  window.__camSync = () => page.sync();
  sync();
}

function keyboard(page) {
  const { cam } = page;
  const keys = page.keys = new Set();
  addEventListener('keydown', (e) => {
    if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
    keys.add(e.code);
    if (['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code)) e.preventDefault();
    if (e.repeat) return;
    if (e.code === 'KeyR') page.resetScene();
    else if (e.code === 'KeyL') page.launch();
    else if (e.code === 'KeyC') { const i = CAM_MODES.findIndex(([m]) => m === cam.mode); cam.mode = CAM_MODES[(i + 1) % CAM_MODES.length][0]; if (cam.mode === 'orbit') cam.orbit.target = null; page.sync(); }
    else if (e.code === 'KeyV') page.beamView = !page.beamView;
  });
  addEventListener('keyup', (e) => keys.delete(e.code));
  addEventListener('blur', () => keys.clear());
  const key = (...c) => c.some((k) => keys.has(k));
  page.readInput = () => {
    let fwd = (key('KeyW', 'ArrowUp') ? 1 : 0) - (key('KeyS', 'ArrowDown') ? 1 : 0);
    let steer = (key('KeyD', 'ArrowRight') ? 1 : 0) - (key('KeyA', 'ArrowLeft') ? 1 : 0);
    let hand = key('Space') ? 1 : 0;
    for (const pad of navigator.getGamepads ? navigator.getGamepads() : []) {
      if (!pad) continue;
      if (Math.abs(pad.axes[0] || 0) > 0.12) steer = pad.axes[0];
      const rt = pad.buttons[7]?.value || 0, lt = pad.buttons[6]?.value || 0;
      if (rt > 0.05 || lt > 0.05) fwd = rt - lt;
      if (pad.buttons[0]?.pressed) hand = 1;
    }
    return { forward: fwd, steer, brake: 0, handbrake: hand };
  };
}

// The camera for this frame: views on the subject (its pose read back each frame) level (yaw
// only) except the driver's seat and the mount, which ride its full orientation. Poses are a
// few steps old by the time they arrive: carried forward to the step on display.
function updateCamera(page, dt, snap) {
  const { cam, cfg } = page;
  const sim = page.sim, manual = page.manual;
  const age = (P) => P.xstep != null && sim ? Math.max(0, sim.steps() - P.xstep) / XRATE + (manual ? 0 : page.lead) : Math.max(0, page.simSteps() - (P.step ?? page.simSteps())) / cfg.hz;
  const lead = (P) => {
    if (!P) return P;
    const a = age(P), v = P.vel ?? P.v;
    if (!(a > 0)) return P;
    // A drive pose turns too: its axes carried by their rates.
    const q = P.dfw ? quatFromBasis(P.fw.map((x, i) => x + P.dfw[i] * a), P.up.map((x, i) => x + P.dup[i] * a)) : P.q;
    return { ...P, p: P.p.map((x, i) => x + v[i] * a), q };
  };
  const T = lead(cam.targetPose), F = lead(cam.fromPose);
  const lp = page.lastPlayer;
  let at = cam.mode === 'orbit' && cam.orbit.target ? cam.orbit.target : lp ? [lp.p[0], lp.p[1] + 0.6, lp.p[2]] : cam.target;
  let desiredE;
  let rigid = false; // a mounted view follows without lag
  if (cam.mode !== 'orbit' && T) {
    const fq = qrot(T.q, [0, 0, 1]);
    const f = norm3([fq[0], 0, fq[2]]), l = [f[2], 0, -f[0]];
    const c = [T.p[0], T.p[1] + 0.6, T.p[2]];
    const off = (fw, side, up) => [c[0] + f[0] * fw + l[0] * side, c[1] + up, c[2] + f[2] * fw + l[2] * side];
    at = c;
    const m = cam.mode;
    if (m === 'chase') desiredE = off(-9, 0, 2.6);
    else if (m === 'front') desiredE = off(8, 0, 0.9);
    else if (m === 'rear') desiredE = off(-8, 0, 0.9);
    else if (m === 'left') desiredE = off(0.3, 7, 0.8);
    else if (m === 'right') desiredE = off(0.3, -7, 0.8);
    else if (m === 'top') desiredE = off(-1.5, 0, 13);
    else if (m === 'driver') {
      const e = qrot(T.q, [0.36, 0.95, 0.05]);
      desiredE = [T.p[0] + e[0], T.p[1] + e[1], T.p[2] + e[2]];
      const d = qrot(T.q, [0, -0.05, 1]);
      at = [desiredE[0] + d[0] * 10, desiredE[1] + d[1] * 10, desiredE[2] + d[2] * 10];
      rigid = true;
    } else if (m === 'mount' && F) {
      const mo = qrot(F.q, cam.subjects[cam.fromIdx]?.mount ?? [0, 1.6, 2.6]);
      desiredE = [F.p[0] + mo[0], F.p[1] + mo[1], F.p[2] + mo[2]];
      rigid = true;
    }
  }
  if (!desiredE) {
    const o = cam.orbit, cp = Math.cos(o.pitch);
    desiredE = [at[0] + o.dist * cp * Math.sin(o.yaw), at[1] + o.dist * Math.sin(o.pitch), at[2] + o.dist * cp * Math.cos(o.yaw)];
  }
  // Exponential follow, the same per second at any frame rate: the eye keeps 0.2 % of its
  // offset after a second, the target four times as fast. Automation (frame-stepped
  // recordings) keeps its own easing, unchanged.
  const k = snap || rigid ? 1 : 1 - Math.pow(0.002, dt);
  const kt = snap || rigid ? 1 : manual ? Math.min(1, k * 4) : 1 - Math.pow(0.002, 4 * dt);
  cam.target = cam.target.map((x, i) => x + (at[i] - x) * kt);
  cam.eye = cam.eye.map((x, i) => x + (desiredE[i] - x) * k);
  const canvas = page.canvas;
  const vp = viewProj(cam.eye, cam.target, canvas.width / canvas.height, cfg.arena ? 0.5 : 0.05, cfg.arena ? 4000 : 900);
  page.labView = { vp, eye: cam.eye.slice() };
}
