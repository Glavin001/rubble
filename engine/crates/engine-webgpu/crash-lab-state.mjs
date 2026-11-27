// The crash lab page's state (crash-lab.html): the one object every page module reads and
// writes. Each field is set by the module named beside it; the methods at the bottom are
// installed by those modules, so a module calls another through the page and none of them
// keeps page state of its own. The window.__* automation hooks (crash-lab-hooks.mjs) read
// this object; nothing else is global.

/** A fresh page state (crash-lab.html makes one). */
export function createPageState({ loading, loadCancel, fail }) {
  return {
    // ---------------------------------------------------------------- bootstrap (crash-lab.html)
    loading, // load-progress.mjs card
    fail, // (error) => void: the page stops with this error
    loadCancel, // the card's Cancel { label, hint, run } (crash-lab-sim.mjs sets it)

    // ---------------------------------------------------------------- crash-lab-config.mjs
    qs: null, // URLSearchParams: the URL's, with a preset's or reference crash's values merged in
    cfg: null, // what the URL asks for: scenes, preset, options, the explicit solver's modes (see readConfig)
    crash: null, // a reference crash or family: { reg, scenario, folders, sel, run, contact } (null for a free scene)
    shell: null, // the lab's frame built before the engine (crash-lab.mjs createLabShell), or null

    // ---------------------------------------------------------------- crash-lab-vehicle.mjs
    vehicle: null, // { importSrc, carKind, cars, source(), showcase(), skinBind }

    // ---------------------------------------------------------------- crash-lab-controls.mjs
    el: null, // the page's own controls and readouts by id (carSel, optRow, speed, xstats, ...)

    // ---------------------------------------------------------------- world-device.mjs
    world: null, device: null, canvas: null, context: null, format: null,
    sceneDef: null, // the World's scene (sceneFromQuery): ground and the drawn static boxes

    // ---------------------------------------------------------------- crash-lab-sim.mjs
    sim: null, // the ExplicitSim (engine-world-wasm), replaced on a restart or respawn
    layout: null, // sim.layout()
    stop: null, // stability-stop.mjs watcher of the current sim
    pacer: null, // FramePacer (2 kHz)
    lead: 0, // the live frame's lead: simulated seconds past its last step (0 in automation)
    player: 0, // the vehicle the camera, HUD and keys follow
    lastPlayer: null, // its last read pose { p, q, v, step, xstep, ... }
    damage: null, breaks: 0, // the last damage read (per vehicle) and broken beams
    frameStats: null, // { n, gpuMs, gpuApprox, lastSteps, ... } (crash-lab-sim.mjs)
    drawnBoxes: null, // the World's bodies drawing the sim's static shapes { key, bodies, n }
    driveOverride: null, // automation's held drive input, else the keys
    wheelsDrawn: 0, // raycast wheels drawn at the last read

    // ---------------------------------------------------------------- crash-lab-camera.mjs
    cam: null, // { mode, orbit, eye, target, subjects, targetIdx, fromIdx, targetPose, fromPose }
    keys: null, // Set of held key codes
    labView: { vp: new Float32Array(16), eye: [0, 0, 0] }, // the last frame's view (the lab's picking)
    beamView: false, // the plain view's beam drawing (V)

    // ---------------------------------------------------------------- crash-lab-draw.mjs, crash-lab-body.mjs
    draw: null, // pipelines, bind groups, the lab renderer (labR) and the body renderer (bodyR)

    // ---------------------------------------------------------------- crash-lab-mount.mjs
    lab: null, // createCrashLab(...) (null in the plain views: arena, traffic, ?lab=0)
    crashRef: null, // createCrashRef(...) in a reference crash
    sceneParams: null, // mountSceneParams(...)

    // ---------------------------------------------------------------- crash-lab-loop.mjs
    manual: false, // automation drives the steps (no rAF stepping)
    labDt: 0, // the last frame's wall time (the lab's clock)
    frameTrace: null, // window.__frameTrace's log
    realTime: 1, fps: 0,

    // ---------------------------------------------------------------- methods (installed by)
    labelProfile: null, // crash-lab-controls.mjs
    // crash-lab-sim.mjs: simSteps (the sim's steps in the ?hz= rate's units), spec, watch,
    // anyDrivable, noWheelsNotice, resetScene, respawn, launch, frameSteps, readPoses,
    // readDamage, damageTick, poseRead, damageRead (the reads in flight)
    simSteps: null, spec: null, watch: null, anyDrivable: null, noWheelsNotice: null, resetScene: null, respawn: null, launch: null,
    frameSteps: null, readPoses: null, readDamage: null, damageTick: null, poseRead: null, damageRead: null,
    // crash-lab-camera.mjs (sync: redraw the camera selects, and the lab's group)
    updateCamera: null, readInput: null, sync: null, findSubject: null, camModes: null,
    drawScene: null, // crash-lab-draw.mjs
    updateHud: null, updateStatus: null, startLoop: null, // crash-lab-loop.mjs
  };
}
