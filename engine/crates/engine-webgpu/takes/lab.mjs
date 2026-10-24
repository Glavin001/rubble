// Fracture lab takes (record-takes.mjs), run through fracture/director.mjs `__record`.
// Segment fields: n (video frames at 60 fps), slow (1/slow speed, lossless), caption,
// cam ({orbit: [yaw, pitch, dist], to?, spin?, target?}), act ('fire' | {shoot: [u, v]} |
// {ray: [origin, direction]}), view ({bonds, color}), mass / speed (the shot sliders,
// powers of two).

// The lab's page state for a take: its Scene menu key and the menus' values, then a load.
const setup = ({ scene, bonds = 'none', hz = 0, radius = 0, strength = 0, level }) => `
  const $ = (id) => document.getElementById(id);
  $('bond-mode').value = ${JSON.stringify(bonds)};
  $('physics-hz').value = ${JSON.stringify(String(hz))};
  $('ball-radius').value = ${JSON.stringify(String(radius))};
  $('bond-strength').value = ${JSON.stringify(String(strength))};
  await window.__FRACTURE_DEMO__.load(${JSON.stringify(scene)});
  ${level === undefined ? '' : `await window.__FRACTURE_DEMO__.selectLevel(${level});`}
  const err = document.getElementById('status');
  if (err.classList.contains('error')) throw new Error(err.textContent);`;

const take = (state, script) => ({ page: 'fracture/index.html', setup: setup(state), script });

export const LAB_TAKES = {
  // Threshold fracture of the baked veneer house: one heavy ball, the impact again at
  // quarter speed, then the settle.
  lab_house_threshold: take({ scene: 'house', bonds: 'threshold', level: 1 }, [
    { n: 50, caption: 'Veneer house, Shardwright bake: bonds break when a contact force beats their strength', cam: { orbit: [0.25, 0.22, 16], to: [0.4, 0.2, 15] } },
    { n: 1, act: { shoot: [0.5, 0.6] }, mass: 5, speed: 2, caption: 'A 32x ball at 4x speed' },
    { n: 14, caption: 'Incoming' },
    { n: 160, slow: 4, caption: 'Quarter speed: the wall lets go in large pieces first, then they break again', cam: { orbit: [0.4, 0.2, 15], to: [0.95, 0.26, 14] } },
    { n: 180, caption: 'Real time: the roof comes down on the debris', cam: { orbit: [0.95, 0.26, 14], to: [1.6, 0.35, 18] } },
  ]),
  // The quasi-static stress solver (blast_parity) with the bond graph drawn: the load
  // path before and after the shot.
  lab_house_stress_xray: take({ scene: 'house', bonds: 'stress' }, [
    { n: 60, view: { bonds: true, color: 2 }, caption: 'Stress solver (blast_parity): every bond, and one colour per rigid body', cam: { orbit: [0.4, 0.3, 24], to: [0.6, 0.3, 22] } },
    { n: 1, act: { shoot: [0.5, 0.6] }, mass: 5, caption: 'A 32x ball into the front wall' },
    { n: 160, slow: 2, caption: 'Half speed: bonds that fail drop out of the graph; each new body gets its own colour', cam: { orbit: [0.6, 0.3, 22], to: [1.1, 0.35, 22] } },
    { n: 120, view: { bonds: false, color: 0 }, caption: 'Materials view', cam: { orbit: [1.1, 0.35, 22], to: [1.5, 0.3, 26] } },
  ]),
  // The lab's wrecking ball (200 t at 40 m/s) through the 7 x 7 grid of cuboid houses.
  lab_cuboid_grid_ball: take({ scene: 'cuboidGrid' }, [
    { n: 60, caption: '49 cuboid houses, every box a rigid body', cam: { orbit: [0.35, 0.35, 140], to: [0.5, 0.3, 120] } },
    { n: 1, act: 'fire', caption: 'The wrecking ball: 200 t at 40 m/s' },
    { n: 300, caption: 'Through the grid', cam: { orbit: [0.5, 0.3, 120], to: [0.9, 0.25, 110] } },
    { n: 180, slow: 2, caption: 'Half speed', cam: { orbit: [0.9, 0.25, 110], to: [1.3, 0.4, 130] } },
  ]),
};
