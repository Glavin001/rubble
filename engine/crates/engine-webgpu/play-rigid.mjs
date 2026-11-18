// The rigid family in the playground shell (play.html): stacks, rings, piles and joint
// rigs (rubble::play::rigid, as demo.html places them, and the table's RUBSNAP1 snapshot
// scenes). Physics, scene and solver constants are Rust's (the scene's own WorldDef);
// demo.html's switches (contact storage, engine features, sleep, CCD, order, stopAt, kill)
// are read by demo-config.mjs, the one parser both pages use. This module adds the camera
// the scene table names, the scene menu, the engine panel, the joint rigs' controls, the
// benchmark protocol and the regression checks (demo.html's ?bench= and ?check=).
import { demoScenes, demoSolver, parseDemoQuery, SLEEP_OFF_TIME, DEMO_SWITCHES } from './demo-config.mjs';
import { createWorld, saveState } from './rubble-world.mjs';
import { installJointControls } from './joint-interaction.mjs';
import { runRingRegression } from './ring-regression.mjs';
import { runGablesRegression } from './gables-regression.mjs';
import { restoreSceneOrder } from './spatial-order.mjs';
import { measurePhysicsBatches, assertBenchmarkWorkload } from './physics-benchmark.mjs?v=3';
import * as PERF from './perf-meter.mjs';

// Every scene of the table (built ones placed in Rust, the rest RUBSNAP1 snapshots).
const SCENES = Object.keys(demoScenes());
const sceneId = (sceneDef, q) => q?.get('scene') ?? sceneDef.scene;
const queryOf = (q) => `?${q.toString()}`;

/** WorldDef fields over the scene's own, as demo.html builds its world: the URL's contact
 * storage, engine features and sleep timer (demo-config.mjs), and where the world ends
 * (`kill`: bodies thrown off the ground below y = −50 are removed, so a settled scene can
 * stop stepping; `kill=off`, `kill=<y>`). */
export function settings(sceneDef, q) {
  const Q = parseDemoQuery(queryOf(q));
  const kill = q.get('kill');
  const killBelow = kill === 'off' ? null : Number(kill ?? -50);
  if (killBelow !== null && !Number.isFinite(killBelow)) throw new Error(`kill=${kill}: a height or off`);
  return {
    contacts: Q.contacts, features: Q.features,
    solver: { sleepTime: Q.sleepOn ? Q.sleepEnabledTime : SLEEP_OFF_TIME },
    ...(killBelow === null ? {} : { killBelow }),
  };
}

// RUBSNAP1 scenes (engine-scenes scene_export): the World imports their records.
async function loadSnapshot(id) {
  const base = `./generated/scenes/${id}`;
  const meta = await (await fetch(`${base}.json`)).json();
  const bytes = new Uint8Array(await (await fetch(`${base}.bin`)).arrayBuffer());
  const [nb, nc, nm, nj] = new Uint32Array(bytes.buffer, 8, 4);
  const view = (off, n) => bytes.subarray(off, off + n);
  const bodiesAt = 24, collidersAt = bodiesAt + nb * 112, jointsAt = collidersAt + nc * 64 + nm * 16;
  return { meta, bytes, bodies: view(bodiesAt, nb * 112), colliders: view(collidersAt, nc * 64), joints: view(jointsAt, nj * meta.jointLayout.stride) };
}

/** A snapshot scene's world (its records imported), or null for a built scene. */
export async function create(sceneDef, settings) {
  const cfg = demoScenes()[sceneDef.scene];
  if (!cfg?.snapshot) return null;
  const snap = await loadSnapshot(cfg.snapshot);
  const solver = { ...demoSolver({ ...cfg, iterations: cfg.iterations ?? snap.meta.iterations }, settings.solver.sleepTime, snap.meta) };
  const world = await createWorld({ gravity: [0, snap.meta.params?.gravityY ?? -10, 0], capacity: snap.meta.cap, ...settings, solver });
  const handles = world.importSnapshot(snap.bytes);
  return { world, placed: { snapshot: snap, handles } };
}

export async function setup(ctx) {
  const { world, placed, sceneDef, lookFrom, setShadowRadius, ui, counters, camera, controls, renderer, stepOne, drain, on, restart, query } = ctx;
  const q = new URLSearchParams(query);
  const Q = parseDemoQuery(queryOf(q));
  const id = Q.scene;
  const cfg = Q.cfg;
  if (cfg?.eye && cfg?.look) {
    lookFrom(cfg.eye, cfg.look);
    setShadowRadius(Math.max(20, Math.hypot(...cfg.eye.map((v, i) => v - cfg.look[i])) * 0.8));
  }
  ui.select('Scene', 'scene', SCENES.map((s) => [s, demoScenes()[s]?.title ?? s]), id);
  if (id === 'pyramid') ui.range('Levels', 'levels', { min: 2, max: 50, value: cfg.levels });
  ui.help('Drag to orbit · scroll to zoom · right-drag to pan.');

  // ---- the engine panel: sleep and resting pairs live; the rest rebuild in place.
  let sleepTime = Q.sleepOn ? Q.sleepEnabledTime : SLEEP_OFF_TIME;
  ui.checkbox('Sleep', 'sleep', Q.sleepOn, {
    into: 'engine', title: 'Bodies at rest fall asleep (scenes matching three-avbd default to never)',
    apply: (onNow) => { sleepTime = onNow ? Q.sleepEnabledTime : SLEEP_OFF_TIME; world.setSolver({ sleepTime }); },
  });
  for (const s of DEMO_SWITCHES) {
    const current = q.get(s.key) ?? '';
    const options = s.values.map((v) => [v, v === '' ? 'default' : v]);
    if (s.key === 'rest') {
      // Archive and carry switch live (World setCarryRest); drop and touch rebuild.
      const live = new Set(['archive', 'carry']);
      let was = current || 'archive';
      const sel = ui.select(s.label, s.key, options.filter(([v]) => v !== ''), was, {
        into: 'engine', title: s.title,
        apply: (v) => {
          if (live.has(v) && live.has(was)) { world.setCarryRest(v === 'carry'); was = v; } else restart({ rest: v });
        },
      });
      sel.value = was;
      continue;
    }
    ui.select(s.label, s.key, options, current, { into: 'engine', title: s.title });
  }
  ui.range('Stop at step', 'stopAt', { min: 0, max: 2000, step: 10, value: Q.stopAt, into: 'engine', title: 'Freeze playback at this step, still drawing (0: never)', format: (v) => (v ? v : 'never') });
  ui.select('Kill below', 'kill', [['', 'y = −50'], ['-10', 'y = −10'], ['-200', 'y = −200'], ['off', 'never']], q.get('kill') ?? '', { into: 'engine', title: 'Bodies below this height are removed (a settled scene can stop stepping)' });

  // ---- snapshot rigs (joints, ragdolls, chains, motors): demo.html's grab, fire, motors.
  let joints = null;
  if (placed.snapshot) {
    const canvas = renderer.domElement;
    joints = installJointControls({
      world, snap: placed.snapshot, handles: placed.handles, canvas,
      camera: { eye: () => camera.position.toArray(), target: () => controls.target.toArray(), fovY: (camera.fov * Math.PI) / 180 },
    });
    on(canvas, 'pointerdown', (e) => {
      if ((e.ctrlKey || e.altKey || e.metaKey) && e.button === 0) { controls.enabled = false; joints.startGrab(e).catch(() => {}); }
    });
    on(canvas, 'pointermove', (e) => joints.moveGrab(e));
    on(canvas, 'pointerup', (e) => { if (joints.endGrab(e)) controls.enabled = true; });
    ui.help(`Drag to orbit · ${joints.legend}`);
  }

  // The placed state (before any step): reset and the checks start from it.
  const initialState = await saveState(world);
  const reset = () => { world.restoreState(initialState); world.setSolver({ sleepTime }); joints?.reset(); };
  ui.button('Reset', 'KeyR', reset);

  // ---- the benchmark (demo.html's fair three-avbd protocol, perf-meter.mjs): blind
  // lead-in, median of 10-step batches, workload checks outside the timed window, then the
  // GPU phase profile; the scene restarts afterwards.
  const bodyCount = world.bodyCount();
  const solver = world.solver();
  async function __bench() {
    ctx.measuring(true);
    try {
      reset();
      const buckets = () => PERF.readColorBuckets({ world, maxColors: cfg.cap?.maxColors ?? 64, awake: counters()?.awake ?? 0 });
      const samples = await PERF.benchWall({
        lead: cfg.lead ?? 0, stepOne, drain, counters: () => ({ ...counters() }), buckets,
        validate: (stage) => assertBenchmarkWorkload({ scene: id, totalBodies: bodyCount, iterations: solver.iterations, dt: solver.dt, counters: counters(), stage, allowSleep: sleepTime < SLEEP_OFF_TIME }),
        measure: measurePhysicsBatches,
        progress: (i, n) => { document.getElementById('perfExtra').textContent = `benchmark lead-in ${i}/${n}…`; },
        meta: { scene: id, totalBodies: bodyCount, dynamicBodies: bodyCount - 1, iterations: solver.iterations, sleep: sleepTime < SLEEP_OFF_TIME, ...Q.modes, dt: solver.dt, userAgent: navigator.userAgent },
      });
      const { summary, report } = await PERF.profilePhases({
        world, stepOne, drain, liveTiming: ctx.liveTiming, wantTimestamps: ctx.wantTimestamps, frameTimes: ctx.frameTimes, clearFrameTimes: ctx.clearFrameTimes,
      });
      const out = { scene: id, bodies: bodyCount - 1, iterations: solver.iterations, wallMs: samples.wallMs, phases: summary, timingSamples: samples, phaseSamples: report };
      window.__BENCH__ = out;
      document.getElementById('perfExtra').textContent = `benchmark: ${samples.wallMs.toFixed(2)} ms/step wall (${bodyCount - 1} bodies)${summary ? ` · ${summary}` : ''}`;
      return out;
    } finally { reset(); ctx.measuring(false); }
  }
  if (!placed.snapshot) ui.button('Benchmark', null, () => __bench(), { into: 'perfButtons' });

  if (placed.snapshot) return { beforeStep: () => joints.beforeStep(world.simSteps() / 60), hooks: { __bench } };

  // ---- __regression(): demo.html's ?check= oracles (ring110k: ring_hold; gables506k: the
  // pinned three-avbd checkpoints) through this page's World, sleep off, from the placed
  // state, steps submitted as demo.html submits them.
  const sceneBodies = Array.from({ length: placed.initial.length / 3 }, (_, i) => ({ p: Array.from(placed.initial.subarray(3 * i, 3 * i + 3)) }));
  async function __regression() {
    const kind = id === 'gables506k' ? 'gables' : id === 'ring110k' ? 'ring' : null;
    if (!kind) throw new Error(`no regression oracle for scene ${id} (ring110k, gables506k)`);
    // The oracles are native runs without CCD: the scene must be placed as ?check= places it.
    if (sceneDef.ccd) throw new Error(`the ${kind} oracle runs without CCD: load with &check=${kind} (or &ccd=0)`);
    // Restored, unless no step was taken yet (?paused=1).
    if (world.stepCount() > 0) { await drain(); world.restoreState(initialState); }
    world.setSolver({ sleepTime: SLEEP_OFF_TIME });
    const read = async () => {
      await drain();
      const words = await world.readBodyRecords(0, bodyCount);
      const bytes = words.buffer.slice(words.byteOffset, words.byteOffset + words.byteLength);
      return { bytes: restoreSceneOrder(bytes, placed.gpuToScene), counters: { ...counters() } };
    };
    const run = kind === 'gables' ? runGablesRegression : runRingRegression;
    return run({ initial: sceneBodies, submit: stepOne, read, batchSize: Q.batchSize, progress: async () => {} });
  }
  // demo.html's ?check=ring|gables and ?bench=1 run on load (their checks read __CHECK__ / __BENCH__).
  if (Q.check) queueMicrotask(async () => {
    ctx.measuring(true);
    try { window.__CHECK__ = await __regression(); document.getElementById('perfExtra').textContent = `PASS · ${window.__CHECK__.bodies} bodies · ${window.__CHECK__.steps} steps`; }
    catch (e) { window.__CHECK__ = { passed: false, error: String(e) }; throw e; }
    finally { reset(); ctx.measuring(false); }
  });
  else if (Q.bench) queueMicrotask(() => __bench());
  return { hooks: { __regression, __bench } };
}
