// vehicle.html takes (record-takes.mjs). Each take: the page's query string and a
// `__record` script. Segment fields (vehicle.html __record): n (steps), input, cruise
// (hold m/s), act ('wheel:<i>' | 'remove:<i>' | 'part' | 'cannon' | 'reset' | 'player:<i>'),
// cam ('chase' | {orbit: [yaw, pitch, dist], spin}), caption.
const loop = (k, f) => Array.from({ length: k }, (_, i) => f(i));
const TAKES = {
  sedan_ramps: {
    query: 'preset=sedan&spawn=0,0,0',
    script: [
      { n: 40, input: { forward: 0, steer: 0 }, caption: 'Sedan: raycast wheels solved as rows inside AVBD', cam: 'chase' },
      { n: 120, input: { forward: 1, steer: 0 }, caption: 'Full throttle: auto gearbox, open diff' },
      { n: 150, cruise: 16, input: { steer: 0 }, caption: 'Through the bollards and over the ramps' },
      { n: 90, cruise: 16, input: { steer: 0 }, caption: 'Down the far ramp', cam: { orbit: [-1.2, 0.25, 11], spin: 0.01 } },
      { n: 40, cruise: 16, input: { steer: 0 }, caption: 'Handbrake turn', cam: 'chase' },
      { n: 70, input: { forward: 0, steer: -1, handbrake: 1 }, caption: 'Handbrake turn: rear wheels lock and slide' },
      { n: 60, input: { forward: 1, steer: -0.3 }, caption: 'Power out' },
      { n: 120, cruise: 12, input: { steer: 0.35 }, caption: 'Steady right turn: anti-roll bars, Ackermann steering' },
      { n: 100, input: { forward: 0, steer: 0, brake: 1 }, caption: 'Brake to a stop' },
    ],
  },
  motorcycle_slalom: {
    query: 'preset=motorcycle&spawn=0,-8,3.14159',
    script: [
      { n: 60, input: { forward: 0, steer: 0 }, caption: 'Motorcycle: the lean controller holds it upright at rest', cam: 'chase' },
      { n: 150, cruise: 12, input: { steer: 0 }, caption: 'Accelerate to 43 km/h (Jolt MotorcycleTest drivetrain)' },
      ...loop(6, (k) => ({ n: 70, cruise: 12, input: { steer: k % 2 ? 1 : -1 }, caption: 'Slalom: leans into each turn, steering limited by max lean' })),
      { n: 240, cruise: 10, input: { steer: -1 }, caption: 'Sweeping left turn', cam: { orbit: [0, 0.18, 5.5], spin: 0.012 } },
      { n: 90, input: { forward: 0.8, steer: 0 }, caption: 'Straighten up and accelerate', cam: 'chase' },
      { n: 150, input: { forward: 0, steer: 0, brake: 1 }, caption: 'Full brake to a stop' },
    ],
  },
  tank_wall: {
    query: 'preset=tank&spawn=-15,42,0',
    script: [
      { n: 60, input: { forward: 0, steer: 0 }, caption: 'Tank: two 7-wheel tracks, each one shared spin', cam: { orbit: [2.4, 0.35, 16], spin: 0.004 } },
      { n: 130, input: { forward: 0, steer: -1 }, caption: 'Pivot turn: tracks counter-rotate (no throttle, below pivot speed)' },
      { n: 130, input: { forward: 0, steer: 1 }, caption: 'Pivot back the other way' },
      { n: 40, input: { forward: 0, steer: 0, brake: 1 }, caption: 'Settle' },
      { n: 200, cruise: 6, input: { steer: 0 }, caption: 'Drive into the brick wall (box friction tracks, Jolt drivetrain)', cam: 'chase' },
      { n: 150, cruise: 6, input: { steer: 0 }, caption: 'Breaching the wall', cam: { orbit: [1.9, 0.3, 18], spin: 0.006 } },
      { n: 180, cruise: 8, input: { steer: 0.7 }, caption: 'Steer by track ratio: steering brake on the inner track, torque to the outer', cam: 'chase' },
      { n: 90, input: { forward: 0, steer: 0, brake: 1 }, caption: 'Brake' },
    ],
  },
  monster_stack: {
    query: 'preset=monster_truck&spawn=-10,22,0',
    script: [
      { n: 40, input: { forward: 0, steer: 0 }, caption: 'Monster truck: sphere-cast wheels, 4WD with limited-slip diffs', cam: 'chase' },
      { n: 150, cruise: 11, input: { steer: 0 }, caption: 'Smash through the box stack' },
      { n: 120, cruise: 11, input: { steer: 0 }, caption: 'Boxes are ordinary AVBD bodies; the truck is one chassis body', cam: { orbit: [2.6, 0.3, 16], spin: 0.008 } },
      // Full lock at 7 m/s needs ~0.75 g. The truck's static stability factor is
      // ~0.92 g (half-track 1.45 m / CG 1.58 m), so at 10 m/s it slides at the limit
      // and debris under a tyre trips it over, which is physical but not this take.
      { n: 150, cruise: 7, input: { steer: 1 }, caption: 'Hard right turn', cam: 'chase' },
      { n: 210, cruise: 12, input: { steer: 0.15 }, caption: 'Back toward the ramps' },
      { n: 100, input: { forward: 0, steer: 0, brake: 1 }, caption: 'Brake' },
    ],
  },
  kart_ice: {
    query: 'preset=kart&spawn=-30,-6,3.14159',
    script: [
      { n: 40, input: { forward: 0, steer: 0 }, caption: 'Kart: direct drive, arcade assists', cam: 'chase' },
      { n: 100, cruise: 12, input: { steer: 0 }, caption: 'Onto the ice patch (μ 0.08)' },
      { n: 120, cruise: 12, input: { steer: -1 }, caption: 'Steer on ice: the tyres saturate and slide', cam: { orbit: [0.6, 0.45, 12], spin: 0.006 } },
      { n: 60, input: { forward: 1, steer: 1, handbrake: 1 }, caption: 'Handbrake spin' },
      { n: 120, input: { forward: 1, steer: 0.4 }, caption: 'Back onto asphalt: grip returns', cam: 'chase' },
      { n: 80, input: { forward: 0, steer: 0, brake: 1 }, caption: 'Brake' },
    ],
  },
  destruction: {
    query: 'scene=destruction&spawn=0,-15,3.14159',
    script: [
      { n: 40, input: { forward: 0, steer: 0 }, caption: 'Destructible sedan: 12 parts on one chassis body', cam: 'chase' },
      { n: 90, cruise: 9, input: { steer: 0 }, caption: 'Drive' },
      { n: 60, act: 'part', cruise: 9, input: { steer: 0 }, caption: 'Sever a part: it becomes a free rigid body with the chassis velocity' },
      { n: 60, act: 'part', cruise: 9, input: { steer: 0 }, caption: 'Sever another part' },
      { n: 90, act: 'wheel:0', cruise: 9, input: { steer: 0 }, caption: 'Sever the front-left wheel: it keeps its spin and rolls away', cam: { orbit: [2.2, 0.3, 12], spin: 0.01 } },
      { n: 120, cruise: 7, input: { steer: 0 }, caption: 'Three wheels left: suspension load redistributes' },
      { n: 90, act: 'wheel:3', cruise: 7, input: { steer: 0 }, caption: 'Sever a rear wheel (drive lost on that side)', cam: 'chase' },
      { n: 120, input: { forward: 0.8, steer: 0.2 }, caption: 'Limping on two wheels' },
      ...loop(4, () => ({ n: 20, act: 'part', input: { forward: 0.5, steer: 0 }, caption: 'Strip the body panels' })),
      { n: 120, input: { forward: 0, steer: 0, brake: 1 }, caption: 'Wreck', cam: { orbit: [0.5, 0.35, 12], spin: 0.008 } },
    ],
  },
  buggy_jump: {
    query: 'scene=buggies',
    script: [
      { n: 60, input: { forward: 0, steer: 0 }, caption: 'Dune buggy (vibe-land): 17 parts on breakable mounts, a rigged suspension', cam: { orbit: [0.8, 0.22, 7], spin: 0.006 } },
      { n: 60, input: { forward: 0, steer: 0 }, caption: 'Coilovers, wishbones and steering posed from the raycast wheels', cam: 'chase' },
      { n: 170, cruise: 15, input: { steer: 0 }, caption: 'Over the kicker and the tabletop: nothing comes off on a landing' },
      { n: 120, cruise: 10, input: { steer: 0.5 }, caption: 'Turn back', cam: { orbit: [-1.0, 0.25, 9], spin: 0.006 } },
      { n: 100, input: { forward: 0, steer: 0, brake: 1 }, caption: 'Brake', cam: 'chase' },
    ],
  },
  buggy_cannon: {
    query: 'scene=buggies',
    script: [
      { n: 60, input: { forward: 0, steer: 0, brake: 1 }, caption: 'Parked: each mount rated from the measured joints of the recipe', cam: { orbit: [1.57, 0.12, 9] } },
      { n: 100, act: 'cannon', caption: '400 kg at 45 m/s into the open side: the struck seat tears off its rails' },
      { n: 60, caption: 'The loose part is a body of its own; the mesh follows it', cam: { orbit: [1.2, 0.3, 9], spin: 0.01 } },
      { n: 30, cam: { orbit: [0.0, 0.1, 10] } },
      { n: 120, act: 'cannon', caption: 'Head on: bumper, nose and lamps carry the blow' },
      { n: 120, caption: 'Broken mounts are severed before the next step: forward-only, like joints', cam: { orbit: [0.6, 0.35, 11], spin: 0.008 } },
    ],
  },
  vibeland: {
    query: 'scene=vibeland&spawn=0,-4,3.1416',
    script: [
      { n: 90, input: { forward: 0, steer: 0, brake: 1 }, caption: "vibe-land's garage: seven vehicles from one parametric recipe", cam: { orbit: [3.14, 0.18, 26], spin: 0.004 } },
      ...loop(7, (k) => ({ n: 75, act: `player:${k + 1}`, input: { forward: 0, steer: 0, brake: 1 }, caption: ['Sand rail', 'Baja trophy truck', 'Rally hatchback', 'Monster truck', 'Derby sedan', 'Sprint car', 'Semi and its hitched flatbed'][k], cam: { orbit: [2.6, 0.22, 9], spin: 0.008 } })),
      { n: 180, input: { forward: 0.7, steer: 0, brake: 0 }, caption: 'The semi pulls its trailer', cam: 'chase' },
    ],
  },
  buggy_hybrid: {
    query: 'scene=buggies&build=hybrid',
    script: [
      { n: 60, input: { forward: 0, steer: 0, brake: 1 }, caption: 'Hybrid buggy: node-beam tube frame on the raycast core, bolted parts on joints', cam: { orbit: [0.9, 0.2, 8], spin: 0.006 } },
      { n: 30, cam: { orbit: [0.0, 0.1, 10] } },
      { n: 120, act: 'cannon', caption: 'Head on: the bumper bends and tears off, the nose dents' },
      { n: 30, cam: { orbit: [1.57, 0.12, 9] } },
      { n: 120, act: 'cannon', caption: 'Into the side: the cage dents, a seat tears off its rails' },
      { n: 150, caption: 'The safety cell keeps its shape', cam: { orbit: [0.7, 0.35, 9], spin: 0.008 } },
    ],
  },
  traffic: {
    query: 'scene=traffic&spawn=0,0,0',
    script: [
      { n: 480, input: { forward: 0.4, steer: 0.2 }, caption: 'Traffic: 49 vehicles + 400 boxes, all on the GPU', cam: { orbit: [0.3, 0.55, 70], spin: 0.002 } },
      { n: 240, input: { forward: 0.6, steer: -0.3 }, caption: 'Chase cam in the crowd', cam: 'chase' },
    ],
  },
  // Two-way ground coupling labs (each mirrors a test in tests/vehicle.rs).
  slab_ice: {
    query: 'scene=slab',
    script: [
      { n: 60, input: { forward: 0, steer: 0 }, caption: 'Two-way coupling: sedan on a 2.5 t slab resting on frictionless skids', cam: { orbit: [1.57, 0.35, 22] } },
      { n: 200, input: { forward: 0.5, steer: 0 }, caption: 'Throttle: the tyres push the slab backward (momentum is conserved)' },
      { n: 120, input: { forward: 0, steer: 0, brake: 1 }, caption: 'Brake: the tyre force reverses and both come to rest together' },
      { n: 60, input: { forward: 0, steer: 0, brake: 1 }, caption: 'Slab displaced; the static cones show how far', cam: { orbit: [1.2, 0.5, 30] } },
    ],
  },
  seesaw: {
    query: 'scene=seesaw',
    script: [
      { n: 40, input: { forward: 0, steer: 0 }, caption: 'Two-way coupling: a 250 kg seesaw plank on a pivot', cam: { orbit: [1.57, 0.25, 20] } },
      { n: 330, cruise: 3.5, input: { steer: 0 }, caption: 'The wheel loads act on the plank: it tips once the car passes the pivot' },
      { n: 120, cruise: 3.5, input: { steer: 0 }, caption: 'Down the other side', cam: { orbit: [1.9, 0.3, 18] } },
      { n: 80, input: { forward: 0, steer: 0, brake: 1 }, caption: 'Brake' },
    ],
  },
  carry: {
    query: 'scene=carry',
    script: [
      { n: 90, input: { forward: 0, steer: 0 }, caption: 'Two-way coupling: a parked kart riding on a monster truck', cam: { orbit: [2.4, 0.35, 14] } },
      { n: 180, input: { forward: 0.15, steer: 0 }, caption: 'The kart’s tyres load the truck’s roof; friction carries it along', cam: 'chase' },
      { n: 240, input: { forward: 0.15, steer: 0.35 }, caption: 'Turning: the kart’s lateral tyre forces hold it on', cam: { orbit: [2.4, 0.35, 16], spin: 0.006 } },
      { n: 120, input: { forward: 0, steer: 0, brake: 0.4 }, caption: 'Gentle brake' },
    ],
  },
  // Breakable joints (tests/vehicle.rs bridge tests): the sedan crosses, the truck does not.
  bridge: {
    query: 'scene=bridge',
    script: [
      { n: 45, input: { forward: 0, steer: 0 }, caption: 'Plank bridge: five 2.6 m planks, 16 ball-joint bolts rated 255 kN', cam: { orbit: [1.25, 0.3, 24] } },
      { n: 330, cruise: 4, input: { steer: 0 }, caption: 'A 1.3 t sedan crosses: its tyre rows load the deck, the bolts hold', cam: { orbit: [1.5, 0.22, 20] } },
      { n: 45, input: { forward: 0, steer: 0, brake: 1 }, caption: 'Parked on the far side' },
      { n: 20, act: 'player:1', input: { forward: 0, steer: 0 }, caption: 'Now a 3 t monster truck', cam: { orbit: [1.5, 0.22, 22] } },
      { n: 420, cruise: 3, input: { steer: 0 }, caption: 'Over 255 kN at midspan: the bolts break (EVENT_BREAK) and the deck folds' },
      { n: 150, input: { forward: 0, steer: 0, brake: 1 }, caption: 'The truck falls through with the planks', cam: { orbit: [1.9, 0.35, 18], spin: 0.004 } },
    ],
  },
  // Same bridge, truck at 43 km/h: it is past midspan when the bolts let go.
  bridge_fast: {
    query: 'scene=bridge&preset=monster_truck&heavy=none&spawn=0,-27,0',
    script: [
      { n: 30, input: { forward: 0, steer: 0 }, caption: 'Same bridge, monster truck with a run-up', cam: { orbit: [1.3, 0.25, 20] } },
      { n: 150, cruise: 12, input: { steer: 0 }, caption: 'Up to 43 km/h on the approach' },
      { n: 210, cruise: 12, input: { steer: 0 }, caption: 'The bolts still break, but behind the truck: momentum carries it over', cam: { orbit: [1.6, 0.2, 18] } },
      { n: 150, input: { forward: 0, steer: 0, brake: 1 }, caption: 'Made it. The deck is gone', cam: { orbit: [2.2, 0.35, 22], spin: 0.004 } },
    ],
  },
  // Track pods (tests/vehicle.rs tank tests): the same drive with and without them.
  ...Object.fromEntries([['tank_course', ''], ['tank_course_shells', '&tracks=shells']].map(([name, q]) => [name, {
    query: `scene=tanklab${q}`,
    script: [
      { n: 45, input: { forward: 0, steer: 0 }, caption: q ? 'Before: rays + wheel-shell spheres, 0.3 m tracks, belly 1 m up' : 'After: solid 0.6 m track pods with a belt, belly 0.5 m up', cam: { orbit: [2.3, 0.3, 15] } },
      { n: 300, cruise: 4, input: { steer: 0 }, caption: 'Rubble in the track paths at 4 m/s', cam: { orbit: [1.35, 0.22, 13] } },
      { n: 240, cruise: 4, input: { steer: 0 }, caption: 'Two 250 kg logs across the lane', cam: { orbit: [1.45, 0.2, 13] } },
      { n: 420, cruise: 4, input: { steer: 0 }, caption: 'A 0.6 m ledge: twice the road wheels\u2019 reach', cam: { orbit: [1.5, 0.15, 14] } },
      { n: 120, input: { forward: 0, steer: 0, brake: 1 }, caption: 'Brake', cam: { orbit: [2.2, 0.3, 16] } },
    ],
  }])),
  sidewall: {
    query: 'scene=sidewall',
    script: [
      { n: 40, input: { forward: 0, steer: 0 }, caption: 'Wheel shells: crates beside the tyres, off the suspension ray lines', cam: { orbit: [3.5, 0.45, 12] } },
      { n: 300, cruise: 6, input: { steer: 0 }, caption: 'The tyre outlines shove the crates aside (wheels are physical, not just rays)' },
      { n: 90, input: { forward: 0, steer: 0, brake: 1 }, caption: 'Brake', cam: { orbit: [3.0, 0.5, 18] } },
    ],
  },
};

export const VEHICLE_TAKES = Object.fromEntries(Object.entries(TAKES).map(([name, t]) => [name, { page: 'vehicle.html', ...t }]));
