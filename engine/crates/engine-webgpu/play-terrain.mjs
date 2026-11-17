// The terrain family in the playground shell (play.html): bodies on the heightfield
// collider (rubble::play::hills). Its controls (rain, crater, reset: the scene's actions),
// its HUD, its camera and its check (__check) — terrain.html's, moved. The heightfield and
// the bodies are drawn from the World's render manifest (rubble-three.mjs).

import { Poses } from './rubble-world.mjs';

const COUNTS = [500, 2000, 5000, 10000];

export async function setup({ world, placed, sceneDef, ui, lookFrom, setShadowRadius }) {
  const terrain = placed.terrain;
  const NX = placed.nx, NZ = placed.nz, DX = placed.dx, DZ = placed.dz;
  const X0 = -0.5 * (NX - 1) * DX, Z0 = -0.5 * (NZ - 1) * DZ;
  const bodies = placed.bodies;
  // Every body's pose as the last polled frame left it.
  const poses = new Poses(sceneDef.n + 8);
  for (const b of bodies) poses.pose.set([...b.p, ...b.q], b.slot * 7);
  const surfaceAt = (x, z) => { const h = world.heightFieldSurface(terrain, x, z); return Number.isNaN(h) ? null : h; };

  lookFrom([25, 24, 32], [0, 0, 0]);
  setShadowRadius(60);

  // Craters, rain and reset are the scene's actions (rubble::play::director).
  let craters = 0;
  const act = (a) => { world.act(a); for (const e of world.takeEvents()) { if (e.what === 'crater') craters++; else if (e.what === 'reset') craters = 0; } };
  const crater = (at) => act({ action: 'crater', at });
  ui.select('Bodies', 'n', COUNTS.map((n) => [n, n.toLocaleString()]), sceneDef.n);
  ui.button('Rain', 'KeyB', () => act({ action: 'rain' }));
  ui.button('Crater', 'KeyC', () => crater());
  ui.button('Reset', 'KeyR', () => act({ action: 'reset' }));
  ui.help('Drag to orbit · scroll to zoom. The crater is an authored height edit plus a blast impulse (T1); computed cratering comes with the granular solver (T2).');

  // __check(): settle, dig a crater under the pile, settle again; every body must stay on
  // or above the terrain (centre above the surface under it).
  const __check = async () => {
    const margin = 1.0;
    const below = () => {
      let n = 0, off = 0, deepest = 0;
      const cases = [];
      for (const b of bodies) {
        const [x, y, z] = poses.position(b.slot);
        const h = Math.abs(x) < -X0 - margin && Math.abs(z) < -Z0 - margin ? surfaceAt(x, z) : null;
        if (h === null) { off++; continue; }
        const d = h - y;
        if (d > 0) {
          n++; deepest = Math.max(deepest, d);
          if (cases.length < 6) cases.push({ slot: b.slot, p: [x, y, z].map((v) => +v.toFixed(2)), surface: +h.toFixed(2) });
        }
      }
      return { below: n, off, deepest, cases };
    };
    const step = async (n) => (await window.__step(n)).counters;
    let counters = await step(300);
    const settled = { ...below(), counters };
    crater([0, 0]);
    counters = await step(240);
    const afterCrater = { ...below(), counters };
    return { bodies: bodies.length, settled, afterCrater };
  };

  return {
    onFrame: (f) => poses.apply(f),
    hud: () => `${bodies.length} bodies · ${craters} craters`,
    hooks: { __check },
  };
}
