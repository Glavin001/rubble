// demo.html's URL switches, as the World it runs on takes them (engine-world-wasm
// `WorldDef`: `contacts`, `features`, `solver`), over the scene table Rust writes.
// Pure: tests/persistence_schedule.test.mjs checks the defaults and the errors.
//
// The step itself is Rust's (`World::step`, engine-exec `rigid_plan`). The URL's
// kernel switches map onto the engine's own `ContactStorage` and
// `EngineFeatures` (the native `ENGINE_*` benchmark variables); the JS host's
// experiments with no Rust path are refused with the reason
// (`DROPPED_SWITCHES`).

// The scene table (engine-scenes `web::scenes`, written by scene_export): titles,
// cameras, solver constants and capacities, in menu order. The built scenes are made
// in Rust when placed (`rubble::play::SceneDef`, `WebWorld.placeScene`); the rest are snapshots.
import table from './generated/scenes/demo.json' with { type: 'json' };

export function demoScenes(qs = new URLSearchParams()) {
  const scenes = Object.fromEntries(structuredClone(table));
  // ?levels=N (2..50) for the pyramid.
  scenes.pyramid.levels = Math.max(2, Math.min(50, Number(qs.get('levels') || scenes.pyramid.levels) || scenes.pyramid.levels));
  return scenes;
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
  // carry (opt-out), neither (measurement), or the touch query with cold wakes in
  // canonical (threshold) worlds too (opt-in; the archive keeps main's warm wakes there).
  const rest = qs.get('rest') || 'archive';
  if (!['archive', 'carry', 'drop', 'touch'].includes(rest)) throw new Error('rest must be archive, carry, drop or touch');
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
  // off only those parts: uniform, incremental, ccd, body, tail, loop.
  const costFloor = qs.get('costfloor');
  const floorParts = { uniform: 'awakeUniform', incremental: 'restIncremental', ccd: 'ccdByCollider', body: 'awakeBodyPasses', tail: 'primalTail', loop: 'solveLoop' };
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
