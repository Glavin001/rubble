// The vehicle family in the playground shell (play.html): raycast-wheel vehicles
// (rubble::play::vehicles; their scripts and actions are rubble::play::director). This
// module adds the player's input (keys, gamepad), the chase camera, the scene's keys and
// the HUD; vehicles, parts and wheels are drawn from the World by rubble-three.mjs.
import { saveState } from './rubble-world.mjs';

const SCENES = [
  ['playground', 'Playground'], ['destruction', 'Destruction derby'], ['traffic', 'Traffic'], ['slab', 'Slab'],
  ['seesaw', 'Seesaw'], ['carry', 'Carry'], ['sidewall', 'Side wall'], ['bridge', 'Bridge'], ['tanklab', 'Tank lab'],
  ['buggies', 'Buggies'], ['gauntlet', 'Gauntlet'], ['meteors', 'Meteor range'], ['run', 'Meteor run'],
];

const qrot = (q, v) => {
  const [x, y, z, w] = q;
  const ux = y * v[2] - z * v[1], uy = z * v[0] - x * v[2], uz = x * v[1] - y * v[0];
  const tx = 2 * ux, ty = 2 * uy, tz = 2 * uz;
  return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
};

export async function setup({ world, placed, sceneDef, ui, held, aim, camera, controls, setShadowRadius }) {
  const scene = new URLSearchParams(location.search).get('scene') ?? sceneDef.scene ?? 'playground';
  ui.select('Scene', 'scene', SCENES, scene);
  const vehicles = placed.vehicles;
  // The player drives the first vehicle no script drives.
  let player = Math.max(0, vehicles.findIndex((v) => !v.ai));
  const entry = (i) => world.vehicleEntry(vehicles[i].h);
  // Chase distance from the vehicle's size (a motorcycle vs a tank).
  let chaseDist = 8;
  function selectPlayer(i) {
    player = i;
    const desc = entry(player).desc;
    const extent = Math.max(
      ...desc.wheels.map((w) => Math.hypot(w.position[0], w.position[2]) + w.radius),
      ...desc.parts.flatMap((p) => p.colliders.map((k) => Math.hypot(k.position[0], k.position[2])
        + (k.shape.type === 'box' ? Math.hypot(k.shape.half[0], k.shape.half[2]) : k.shape.radius))),
    );
    chaseDist = Math.max(4.5, Math.min(16, 2.5 + 2.6 * extent));
  }
  if (vehicles.length) selectPlayer(player);
  setShadowRadius(45);
  const initial = await saveState(world);

  const key = (...c) => c.some((k) => held.has(k));
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

  // Actions (the scene's director): cannon along the camera ray, the scene's fire key,
  // reset to the placed state with the scripts started over.
  const events = [];
  const take = () => { for (const e of world.takeEvents()) { events.unshift(e.what); events.length = Math.min(events.length, 6); } };
  ui.button('Cannon', 'KeyB', () => { if (placed.cannon?.length) { world.act({ action: 'cannon', aim: aim() }); take(); } });
  ui.button('Fire', 'KeyF', () => { world.act({ action: 'fire' }); take(); });
  ui.button('Reset', 'KeyR', () => { world.restoreState(initial); world.act({ action: 'reset' }); take(); });
  let chase = true;
  ui.button('Camera', 'KeyC', () => { chase = !chase; });
  if (vehicles.length > 1) ui.button('Next car', 'Tab', () => {
    world.setVehicleInput(vehicles[player].h, { forward: 0, steer: 0, brake: 1 });
    selectPlayer((player + 1) % vehicles.length);
  });
  ui.help('<kbd>W</kbd><kbd>S</kbd> drive · <kbd>A</kbd><kbd>D</kbd> steer · <kbd>Space</kbd> handbrake · drag to orbit (C: chase / free)');

  // The chassis pose, read a frame late (never waits).
  let chassis = null, reading = null, state = null;
  function readChassis() {
    if (reading || !vehicles.length) return;
    reading = world.readBodies([entry(player).chassis]).then((f) => {
      reading = null;
      chassis = { p: [f[0], f[1], f[2]], q: [f[3], f[4], f[5], f[6]], v: [f[7], f[8], f[9]] };
    });
  }
  let snapped = false;
  function frame(dt) {
    readChassis();
    if (!chase || !chassis) return;
    const f = qrot(chassis.q, [0, 0, 1]);
    const h = Math.hypot(f[0], f[2]) || 1;
    const flat = [f[0] / h, 0, f[2] / h];
    const p = chassis.p;
    const wantT = [p[0], p[1] + 1, p[2]];
    const wantE = [p[0] - flat[0] * chaseDist, p[1] + chaseDist * 0.38, p[2] - flat[2] * chaseDist];
    // Smooth the eye's offset from the target, so a fast car stays the same size.
    const k = snapped ? 1 - Math.pow(0.0015, dt) : 1;
    snapped = true;
    const t = controls.target, e = camera.position;
    const off = [e.x - t.x, e.y - t.y, e.z - t.z], wantOff = wantE.map((x, i) => x - wantT[i]);
    t.set(...[t.x, t.y, t.z].map((x, i) => x + (wantT[i] - x) * Math.min(1, k * 4)));
    e.set(...[t.x, t.y, t.z].map((x, i) => x + off[i] + (wantOff[i] - off[i]) * k));
  }

  return {
    beforeStep() { if (vehicles.length) world.setVehicleInput(vehicles[player].h, readInput()); },
    onFrame(f) { if (!f.empty && vehicles.length) state = f.vehicleState(vehicles[player].h) ?? state; },
    frame,
    keydown(e) { if (e.code === 'Tab' || e.code === 'Space') e.preventDefault(); },
    hud() {
      const s = state;
      const speed = chassis ? Math.hypot(...chassis.v) * 3.6 : 0;
      return `${vehicles.length} vehicles · ${vehicles[player]?.name ?? ''} · ${speed.toFixed(0)} km/h${s?.gear !== undefined ? ` · gear ${s.gear}` : ''}${events.length ? ` · <span class="muted">${events.join(', ')}</span>` : ''}`;
    },
    hooks: { __vehicleState: () => state },
  };
}
