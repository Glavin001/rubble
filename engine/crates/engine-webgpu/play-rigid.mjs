// The rigid family in the playground shell (play.html): stacks, rings, piles and joint
// rigs (rubble::play::rigid, as demo.html places them, and the table's RUBSNAP1 snapshot
// scenes). Physics, scene and solver constants are Rust's (the scene's own WorldDef);
// this module adds the camera the scene table names, the scene menu, the joint rigs'
// controls and the regression checks (demo.html's ?check=).
import { demoScenes, demoSolver, SLEEP_OFF_TIME } from './demo-config.mjs';
import { createWorld, saveState } from './rubble-world.mjs';
import { installJointControls } from './joint-interaction.mjs';
import { runRingRegression } from './ring-regression.mjs';
import { runGablesRegression } from './gables-regression.mjs';
import { restoreSceneOrder } from './spatial-order.mjs';

// Every scene of the table (built ones placed in Rust, the rest RUBSNAP1 snapshots).
const SCENES = Object.keys(demoScenes());
const sceneId = (sceneDef) => new URLSearchParams(location.search).get('scene') ?? sceneDef.scene;

/** WorldDef fields over the scene's own (as demo.html runs it): bodies thrown off the
 * ground below y = −50 are removed, so a settled scene can stop stepping. */
export const settings = () => ({ killBelow: -50 });

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
  const cfg = demoScenes()[sceneId(sceneDef)];
  if (!cfg?.snapshot) return null;
  const snap = await loadSnapshot(cfg.snapshot);
  const solver = demoSolver({ ...cfg, iterations: cfg.iterations ?? snap.meta.iterations }, SLEEP_OFF_TIME, snap.meta);
  const world = await createWorld({ gravity: [0, snap.meta.params?.gravityY ?? -10, 0], capacity: snap.meta.cap, solver, ...settings });
  const handles = world.importSnapshot(snap.bytes);
  return { world, placed: { snapshot: snap, handles } };
}

export async function setup({ world, placed, sceneDef, lookFrom, setShadowRadius, ui, counters, camera, controls, renderer, stepOne, drain }) {
  const qs = new URLSearchParams(location.search);
  const id = sceneId(sceneDef);
  const cfg = demoScenes(qs)[id];
  if (cfg?.eye && cfg?.look) {
    lookFrom(cfg.eye, cfg.look);
    setShadowRadius(Math.max(20, Math.hypot(...cfg.eye.map((v, i) => v - cfg.look[i])) * 0.8));
  }
  ui.select('Scene', 'scene', SCENES.map((s) => [s, demoScenes()[s]?.title ?? s]), id);
  ui.help('Drag to orbit · scroll to zoom · right-drag to pan.');

  // Snapshot rigs (joints, ragdolls, chains, motors): demo.html's grab, fire and motors.
  let joints = null;
  if (placed.snapshot) {
    const canvas = renderer.domElement;
    joints = installJointControls({
      world, snap: placed.snapshot, handles: placed.handles, canvas,
      camera: { eye: () => camera.position.toArray(), target: () => controls.target.toArray(), fovY: (camera.fov * Math.PI) / 180 },
    });
    canvas.addEventListener('pointerdown', (e) => {
      if ((e.ctrlKey || e.altKey || e.metaKey) && e.button === 0) { controls.enabled = false; joints.startGrab(e).catch(() => {}); }
    });
    canvas.addEventListener('pointermove', (e) => joints.moveGrab(e));
    canvas.addEventListener('pointerup', (e) => { if (joints.endGrab(e)) controls.enabled = true; });
    ui.help(`Drag to orbit · ${joints.legend}`);
  }

  // The placed state (before any step): reset and the checks start from it.
  const initialState = await saveState(world);
  ui.button('Reset', 'KeyR', () => { world.restoreState(initialState); joints?.reset(); });
  if (placed.snapshot) return { beforeStep: () => joints.beforeStep(world.simSteps() / 60) };

  // __regression(): demo.html's ?check= oracles (ring110k: ring_hold; gables506k: the
  // pinned three-avbd checkpoints) through this page's World, sleep off, from the placed
  // state, steps submitted as demo.html submits them.
  const sceneBodies = Array.from({ length: placed.initial.length / 3 }, (_, i) => ({ p: Array.from(placed.initial.subarray(3 * i, 3 * i + 3)) }));
  async function __regression() {
    const kind = id === 'gables506k' ? 'gables' : id === 'ring110k' ? 'ring' : null;
    if (!kind) throw new Error(`no regression oracle for scene ${id} (ring110k, gables506k)`);
    // The oracles are native runs without CCD: the scene must be placed as ?check= places it.
    if (sceneDef.ccd) throw new Error(`the ${kind} oracle runs without CCD: load with &check=${kind} (or &ccd=0)`);
    // Restored, unless no step was taken yet (?paused=1).
    if (world.stepCount() > 0) { await window.__step(0); world.restoreState(initialState); }
    world.setSolver({ sleepTime: SLEEP_OFF_TIME });
    const bodyCount = world.bodyCount();
    const read = async () => {
      await drain();
      const words = await world.readBodyRecords(0, bodyCount);
      const bytes = words.buffer.slice(words.byteOffset, words.byteOffset + words.byteLength);
      return { bytes: restoreSceneOrder(bytes, placed.gpuToScene), counters: { ...counters() } };
    };
    const run = kind === 'gables' ? runGablesRegression : runRingRegression;
    return run({ initial: sceneBodies, submit: stepOne, read, progress: async () => {} });
  }
  return { hooks: { __regression } };
}
