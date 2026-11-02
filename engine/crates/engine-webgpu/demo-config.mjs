// demo.html's scenes and URL switches, as the World it runs on takes them
// (engine-world-wasm `WorldDef`: `contacts`, `features`, `solver`, `capacity`).
// Pure: tests/persistence_schedule.test.mjs checks the defaults and the errors.
//
// The step itself is Rust's (`World::step`, engine-exec `rigid_plan`). The URL's
// kernel switches map onto the engine's own `ContactStorage` and
// `EngineFeatures` (the native `ENGINE_*` benchmark variables); the JS host's
// experiments with no Rust path are refused with the reason
// (`DROPPED_SWITCHES`).

// Allocate enough contacts for the benchmark; the 2M default bounds memory for
// smaller scenes.
const DEFAULT_CONTACT_CAP = 1 << 21;
const nextPow2 = (n) => { let p = 1; while (p < n) p <<= 1; return p; };
export function sceneCap(bodies, extras = {}) {
  const b = nextPow2(Math.max(bodies, 4096));
  return {
    maxBodies: b,
    maxColliders: nextPow2(Math.max(bodies * 2, 8192)),
    maxContacts: Math.min(nextPow2(Math.max(bodies * 16, 32768)), DEFAULT_CONTACT_CAP),
    maxJoints: extras.maxJoints ?? 64,
    maxCandidates: Math.min(nextPow2(Math.max(bodies * 32, 65536)), DEFAULT_CONTACT_CAP),
    maxColors: 32,
    // Teleports, velocity sets and drives are World commands (the drag joint,
    // the projectile); capacity only.
    maxCommands: 2048,
  };
}

// Capacities mirror three-avbd bench-cases tiers (small / paper / large / steady).
export function demoScenes(qs = new URLSearchParams()) {
  return {
    wall: {
      title: 'wall smash 2k, 10 iterations',
      lead: 30, iterations: 10, settled: false, sleepRounds: 8, friction: 0.5,
      eye: [0, 16, -42], look: [0, 8, 0], far: 400, ground: 48,
      cap: sceneCap(4096),
    },
    ring: {
      title: 'brick ring 9k, 4 iterations',
      lead: 60, iterations: 4, settled: false, sleepRounds: 8, friction: 0.5,
      eye: [32, 22, 14], look: [0, 5, 0], far: 400, ground: 22, flags: 1, sleepTime: 1e6,
      cap: sceneCap(16384), // contacts 262144 — was 131072 (overflow mid-smash)
    },
    gables: {
      title: 'brick gables 6.7k, 3 iterations',
      lead: 60, iterations: 3, settled: false, sleepRounds: 8, friction: 0.5,
      eye: [0, 28, -40], look: [0, 6, 0], far: 400, ground: 44, flags: 1,
      cap: sceneCap(16384),
    },
    pyramid: {
      // levels via ?levels=N (default 30). Match three-avbd: reuseContacts + full 4-pt faces.
      title: 'settled pyramid',
      lead: 180, iterations: 10, settled: true, sleepRounds: 8, friction: 0.5,
      eye: [28, 22, 28], look: [0, 6, 0], far: 400, ground: 64,
      levels: Math.max(2, Math.min(50, Number(qs.get('levels') || 30) || 30)),
      flags: 1, // FLAG_REUSE_CONTACTS — three-avbd default
      cap: sceneCap(16384),
    },
    pile: {
      title: 'settled pile 32k, 4 iterations',
      lead: 180, iterations: 4, settled: true, sleepRounds: 1, friction: 0.7,
      eye: [48, 36, 48], look: [0, 8, 0], far: 800, ground: 96,
      // Box Pile scene defaults (DECISIONS 8af–8av)
      flags: 1, alpha: 0.95, betaLin: 2000, sleepLin: 0.5, sleepAng: 0.5, sleepTime: 0.25,
      cap: sceneCap(65536),
    },
    columns: {
      title: 'box columns 100k, 4 iterations',
      lead: 180, iterations: 4, settled: true, sleepRounds: 8, friction: 0.5,
      eye: [90, 40, 90], look: [0, 6, 0], far: 1200, ground: 200,
      flags: 1, // three-avbd reuseContacts default
      cap: sceneCap(131072),
    },
    ring110k: {
      title: 'brick ring 110k, 4 iterations',
      lead: 60, iterations: 4, settled: false, sleepRounds: 8, friction: 0.5,
      eye: [140, 60, 40], look: [0, 12, 0], far: 2000, ground: 90, flags: 1, sleepTime: 1e6,
      cap: sceneCap(131072),
    },
    gables506k: {
      title: 'brick gables 506k, 3 iterations',
      lead: 60, iterations: 3, settled: false, sleepRounds: 8, friction: 0.5,
      eye: [0, 80, -120], look: [0, 10, 0], far: 4000, ground: 256, flags: 1, sleepTime: 1e6,
      cap: { ...sceneCap(524288), maxContacts: 6 * 1048576, maxCandidates: 4 * 1048576 },
    },
    // Joint scenes: built by engine-scenes in Rust, exported as RUBSNAP1 records
    // (cargo run -p engine-scenes --release --bin scene_export), imported by World.
    joints: {
      title: 'joints showcase', snapshot: 'joints',
      lead: 60, settled: false, sleepRounds: 8, friction: 0.5,
      eye: [0, 12, -22], look: [0, 2.5, 1], far: 400, ground: 60,
    },
    ragdolls: {
      title: 'ragdolls 40', snapshot: 'ragdolls',
      lead: 60, settled: false, sleepRounds: 8, friction: 0.6,
      eye: [6, 6, -11], look: [0, 1.5, 0], far: 400, ground: 40,
    },
    ragdolls10k: {
      title: 'ragdoll pile 11k bodies, 10k joints', snapshot: 'ragdolls10k',
      lead: 60, settled: false, sleepRounds: 8, friction: 0.6,
      eye: [16, 14, -22], look: [0, 1, 0], far: 600, ground: 60,
    },
    hingechains10k: {
      title: 'hinge chains 10k bodies, 10k joints', snapshot: 'hingechains10k',
      lead: 60, settled: false, sleepRounds: 8, friction: 0.5,
      eye: [40, 30, -90], look: [0, 8, 0], far: 800, ground: 120,
    },
    motorfield10k: {
      title: 'motor field 10k bodies, 2.5k motors', snapshot: 'motorfield10k',
      lead: 60, settled: false, sleepRounds: 8, friction: 0.5,
      eye: [30, 30, -70], look: [0, 2, 0], far: 800, ground: 80,
    },
    // Legacy rubble web demo scenes (engine-scenes `legacy`): the same bodies, masses,
    // velocities and per-body friction as crates/rubble-scenes, for side-by-side
    // comparison. Each brings its own ground; gravity comes from the snapshot (−9.81).
    // Static friction draws no floor: legacy placed its boxes on the part of the ramp
    // below the floor slab (y ≈ −5), and the port keeps that geometry.
    legacyEmpty: {
      title: 'legacy · empty', snapshot: 'legacy-empty',
      lead: 60, settled: false, sleepRounds: 8, friction: 0.5, frictionLabel: 'per body (legacy)', groundY: 0.5,
      eye: [0, 6, -20], look: [0, 0, 0], far: 600, ground: 50,
    },
    legacyGround: {
      title: 'legacy · ground', snapshot: 'legacy-ground',
      lead: 60, settled: false, sleepRounds: 8, friction: 0.5, frictionLabel: 'per body (legacy)', groundY: 0.5,
      eye: [0, 6, -16], look: [0, 1, 0], far: 600, ground: 50,
    },
    legacyDynamicFriction: {
      title: 'legacy · dynamic friction (μ 5 → 0)', snapshot: 'legacy-dynamic-friction',
      lead: 60, settled: false, sleepRounds: 8, friction: 0.5, frictionLabel: 'per body (legacy)', groundY: 0.5,
      eye: [10, 18, -58], look: [10, 0, -20], far: 800, ground: 120,
    },
    legacyStaticFriction: {
      title: 'legacy · static friction ramp (μ 0.25 → 0.5)', snapshot: 'legacy-static-friction',
      lead: 60, settled: false, sleepRounds: 8, friction: 0.5, frictionLabel: 'per body (legacy)', groundY: 0.5,
      eye: [-34, 8, -34], look: [-8, -3, 0], far: 600, ground: 0,
    },
    legacyPyramid: {
      title: 'legacy · pyramid 136', snapshot: 'legacy-pyramid',
      lead: 60, settled: false, sleepRounds: 8, friction: 0.5, frictionLabel: 'per body (legacy)', groundY: 0,
      eye: [0, 8, -26], look: [0, 6, 0], far: 600, ground: 50,
    },
    legacyStack: {
      title: 'legacy · stack 10', snapshot: 'legacy-stack',
      lead: 60, settled: false, sleepRounds: 8, friction: 0.5, frictionLabel: 'per body (legacy)', groundY: 0.5,
      eye: [0, 8, -22], look: [0, 7, 0], far: 600, ground: 50,
    },
    legacyStackRatio: {
      title: 'legacy · stack ratio 1:8', snapshot: 'legacy-stack-ratio',
      lead: 60, settled: false, sleepRounds: 8, friction: 0.5, frictionLabel: 'per body (legacy)', groundY: 0.5,
      eye: [0, 12, -36], look: [0, 8, 0], far: 600, ground: 50,
    },
    legacyScatter: {
      title: 'legacy · scatter 3k (spheres, boxes, capsules)', snapshot: 'legacy-scatter',
      lead: 60, settled: false, sleepRounds: 8, friction: 0.5, frictionLabel: 'per body (legacy)', groundY: 0,
      eye: [24, 18, 32], look: [0, 3, 0], far: 1200, ground: 300,
    },
    legacyScatterBoxes: {
      title: 'legacy · scatter boxes 4k', snapshot: 'legacy-scatter-boxes',
      lead: 60, settled: false, sleepRounds: 8, friction: 0.5, frictionLabel: 'per body (legacy)', groundY: 0,
      eye: [26, 20, 34], look: [0, 3, 0], far: 1200, ground: 300,
    },
    legacyGridBoxes: {
      title: 'legacy · grid boxes 1.5k', snapshot: 'legacy-grid-boxes',
      lead: 60, settled: false, sleepRounds: 8, friction: 0.5, frictionLabel: 'per body (legacy)', groundY: 0,
      eye: [10, 10, 14], look: [0, 5, 0], far: 1200, ground: 300,
    },
    legacySlantedGrid: {
      title: 'legacy · slanted grid 3.2k', snapshot: 'legacy-slanted-grid',
      lead: 60, settled: false, sleepRounds: 8, friction: 0.5, frictionLabel: 'per body (legacy)', groundY: 0,
      eye: [14, 10, 18], look: [0, 5, 0], far: 1200, ground: 300,
    },
    legacyGrid10k: {
      title: 'legacy · grid 10k', snapshot: 'legacy-grid10k',
      lead: 60, settled: false, sleepRounds: 8, friction: 0.5, frictionLabel: 'per body (legacy)', groundY: 0,
      eye: [22, 16, 28], look: [0, 5, 0], far: 1200, ground: 300,
    },
    // Not in the legacy demo: its 10k grid scaled to 100k (50 × 40 × 50).
    legacyGrid100k: {
      title: 'grid 100k (legacy grid ×10)', snapshot: 'legacy-grid100k',
      lead: 60, settled: false, sleepRounds: 8, friction: 0.5, frictionLabel: 'per body (legacy)', groundY: 0,
      eye: [46, 30, 58], look: [0, 9, 0], far: 2000, ground: 300,
    },
  };
}

// The JS host's kernel experiments with no path in Rust's plan: their values
// other than Rust's own are refused (an old URL fails with the reason instead
// of silently running something else).
export const DROPPED_SWITCHES = {
  coloring: { keep: 'reuse', why: 'coloring=rebuild (re-colouring from scratch each step); the plan always repairs the previous colouring' },
  prefix: { keep: 'parallel', why: 'prefix=serial (one-workgroup colour-tile prefix); the plan runs the parallel prefix' },
  packing: { keep: 'stable', why: 'packing=atomic (pack_color_lists); the plan packs colour tiles in stable order' },
  adjacency: { keep: 'physical', why: 'adjacency=append (csr_sort_append); the plan sorts adjacency physically (csr_sort)' },
  profile: { keep: null, why: 'profile=detail|graph|colors (the JS host split its own passes); World times its plan phases (gpuTiming: phases), which bench=1 reports' },
};

export const SLEEP_OFF_TIME = 1e6;

// URL → { scene, cfg, worldDef pieces, page switches }. Throws on any invalid
// or dropped value, before the world is made.
export function parseDemoQuery(search) {
  const qs = new URLSearchParams(search);
  const SCENES = demoScenes(qs);
  const scene = qs.get('scene') || 'wall';
  const cfg = SCENES[scene];
  if (!cfg) throw new Error(`unknown scene ${scene}; try ${Object.keys(SCENES).join(', ')}`);
  for (const [key, d] of Object.entries(DROPPED_SWITCHES)) {
    if (qs.has(key) && qs.get(key) !== d.keep)
      throw new Error(`${key}=${qs.get(key)} was a JS-host experiment with no path in World's step: ${d.why}`);
  }
  const benchOnly = qs.get('bench') === 'only';
  const bench = qs.get('bench') === '1' || benchOnly;
  const check = qs.get('check');
  // Contact storage (ENGINE_CONTACTS): persistent on every scene by default.
  const contacts = qs.get('contacts') || 'persistent';
  if (!['persistent', 'compact', 'canonical'].includes(contacts)) throw new Error('contacts must be persistent, compact or canonical');
  const persistent = contacts === 'persistent';
  // EngineFeatures: only the fields given differ from the shipped defaults.
  const features = {};
  // Ring-only measured default; sparse=32 retains the separate-colour control.
  const sparse = qs.has('sparse') ? Number(qs.get('sparse')) : scene === 'ring110k' ? 10 : 32;
  if (!Number.isInteger(sparse) || sparse < 0 || sparse > 32)
    throw new Error('sparse must be an integer color index from 0 through 32');
  features.sparseColors = sparse;
  // colors=full / rounds=full: the fixed-palette controls (ENGINE_COLOR_HEADROOM=full).
  if (qs.get('colors') === 'full') features.colorHeadroom = null;
  if (qs.get('rounds') === 'full') features.colorRoundHeadroom = null;
  const persistence = qs.get('persistence') || 'contact';
  if (!['contact', 'manifold', 'fused'].includes(persistence)) throw new Error('persistence must be contact, manifold or fused');
  if (persistence === 'fused' && !(contacts !== 'canonical')) throw new Error('persistence=fused requires compact or persistent contacts');
  features.manifoldPersistence = { contact: 'contacts', manifold: 'manifolds', fused: 'fused' }[persistence];
  const hashInsert = qs.get('hashInsert') || 'record';
  if (!['contact', 'manifold', 'record'].includes(hashInsert)) throw new Error('hashInsert must be contact, manifold or record');
  features.warmHash = hashInsert;
  const reuse = qs.get('reuse') || (persistent && persistence === 'contact' ? 'copy' : 'pass');
  if (!['pass', 'copy'].includes(reuse)) throw new Error('reuse must be pass or copy');
  if (reuse === 'copy' && (!persistent || persistence !== 'contact'))
    throw new Error('reuse=copy requires persistent contacts with per-contact persistence');
  features.reuseCopy = reuse === 'copy';
  const np = qs.get('np') || (reuse === 'copy' ? 'split' : 'fused');
  if (!['fused', 'split'].includes(np)) throw new Error('np must be fused or split');
  if (np === 'split' && reuse !== 'copy') throw new Error('np=split requires reuse=copy');
  features.npSplit = np === 'split';
  const persist = qs.get('persist') || (reuse === 'copy' ? 'list' : 'scan');
  if (!['scan', 'list'].includes(persist)) throw new Error('persist must be scan or list');
  if (persist === 'list' && reuse !== 'copy') throw new Error('persist=list requires reuse=copy');
  features.persistList = persist === 'list';
  // Resting contacts in a sleeping world: the sleep archive (default), the resting-pair
  // carry (opt-out) or neither (measurement).
  const rest = qs.get('rest') || 'archive';
  if (!['archive', 'carry', 'drop'].includes(rest)) throw new Error('rest must be archive, carry or drop');
  features.restingContacts = rest;
  features.carryRest = rest !== 'drop';
  const repair = qs.get('repair') || 'local';
  if (!['rounds', 'local'].includes(repair)) throw new Error('repair must be rounds or local');
  features.localRepair = repair === 'local';
  const grid = qs.get('grid') || 'split';
  if (!['fixed', 'scaled', 'split'].includes(grid)) throw new Error('grid must be fixed, scaled or split');
  features.grid = grid;
  // ?gridlevels=on|off: the hierarchical grid (and with it the awake-indexed broadphase)
  // or the uniform one; unset: hierarchical for convex scenes only.
  const gridLevels = qs.get('gridlevels');
  if (gridLevels !== null && !['on', 'off'].includes(gridLevels)) throw new Error('gridlevels must be on or off');
  if (gridLevels !== null) features.gridLevels = gridLevels === 'on';
  // ?broadphase=full: no awake-indexed broadphase; =hier: on hierarchical grids only.
  const broadphase = qs.get('broadphase');
  if (broadphase !== null && !['full', 'hier', 'awake'].includes(broadphase)) throw new Error('broadphase must be full, hier or awake');
  if (broadphase === 'full') features.awakeBroadphase = false;
  if (broadphase === 'hier') features.awakeUniform = false;
  // ?costfloor=off: the per-step work over every collider and body slot as before the
  // cost-floor passes (each gives the same results; for A/B timing). A comma list turns
  // off only those parts: uniform, incremental, ccd, body, tail.
  const costFloor = qs.get('costfloor');
  const floorParts = { uniform: 'awakeUniform', incremental: 'restIncremental', ccd: 'ccdByCollider', body: 'awakeBodyPasses', tail: 'primalTail' };
  if (costFloor !== null && costFloor !== 'on') {
    const off = costFloor === 'off' ? Object.keys(floorParts) : costFloor.split(',');
    for (const part of off) {
      if (!(part in floorParts)) throw new Error(`costfloor must be on, off or a list of ${Object.keys(floorParts).join(', ')}`);
      features[floorParts[part]] = false;
    }
  }
  // CSR endpoints follow the contact storage in Rust's plan (pairs when persistent).
  const endpoints = qs.get('endpoints') || (persistent ? 'pairs' : 'manifold');
  if (!['manifold', 'pairs'].includes(endpoints)) throw new Error('endpoints must be manifold or pairs');
  if (endpoints !== (persistent ? 'pairs' : 'manifold'))
    throw new Error(`endpoints=${endpoints} with ${contacts} contacts was a JS-host experiment; World's plan reads ${persistent ? 'solver pairs' : 'manifolds'} for these contacts`);
  // Continuous collision for the smash balls and the F projectile: on by default;
  // the ring/gables checks compare against native runs without CCD (?ccd=1 turns it on).
  const ccd = qs.has('ccd') ? qs.get('ccd') !== '0' : !check;
  const sleepQuery = qs.get('sleep');
  if (sleepQuery !== null && !['on', 'off'].includes(sleepQuery)) throw new Error('sleep must be on or off');
  const sceneSleepTime = cfg.sleepTime ?? 0.5;
  const sleepEnabledTime = sceneSleepTime < SLEEP_OFF_TIME ? sceneSleepTime : 0.5;
  const sleepOn = sleepQuery !== null ? sleepQuery === 'on' : sceneSleepTime < SLEEP_OFF_TIME;
  if (check && !['ring', 'gables'].includes(check)) throw new Error('check must be ring or gables');
  const spatialOrder = qs.get('order') === 'morton' && !scene.match(/^(joints|ragdolls|hingechains|motorfield|legacy)/);
  return {
    qs, scene, cfg, scenes: SCENES, contacts, features, bench, benchOnly, check, ccd,
    sleepOn, sleepEnabledTime, spatialOrder,
    stopAt: Math.max(0, Number(qs.get('stopAt')) || 0),
    batchSize: qs.get('batch') === '1' ? 1 : 10,
    // What a benchmark report labels its configuration with.
    modes: { contactLayout: contacts, sparseStart: sparse, persistence, hashInsert, reuse, np, persist, repair, rest, endpoints, grid,
      colorDispatch: qs.get('colors') === 'full' ? 'full' : 'adaptive', roundDispatch: qs.get('rounds') === 'full' ? 'full' : 'adaptive' },
  };
}

// The solver constants a scene runs with (`AvbdParams`, camelCase).
export function demoSolver(cfg, sleepTime, snapshotMeta = null) {
  return {
    dt: 1 / 60, iterations: cfg.iterations ?? snapshotMeta?.iterations, flags: cfg.flags || 0,
    alpha: cfg.alpha ?? 0.99, betaLin: cfg.betaLin ?? 10000,
    sleepLin: cfg.sleepLin ?? 0.15, sleepAng: cfg.sleepAng ?? 0.15, sleepTime,
  };
}
