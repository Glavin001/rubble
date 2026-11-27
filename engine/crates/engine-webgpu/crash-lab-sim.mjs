// The crash lab page's simulation driver (crash-lab.html): the World that holds the ground
// and draws the static shapes, the explicit 2 kHz solver (engine-world-wasm ExplicitSim on
// the World's device) spawned from the scene's spec (a reference crash's, a family's, or a
// lab scene's with its knobs and the solver's options), its restart (R, the options),
// respawn (the Scene panel's values: no reload), the steps of each frame (FramePacer, or the
// lab's clock in chunks), the launch key, the pose / drive / damage reads, the raycast
// wheels' stand-in boxes and the stability stop. Physics is Rust's; this module only calls it.

import { sceneFromQuery, FramePacer } from './rubble-world.mjs';
import { crashRefFolders } from './crash-ref.mjs';
import { createStabilityStop } from './stability-stop.mjs';
import { openWorld } from './world-device.mjs';
import { cross3, quatFromBasis } from './view-math.mjs';

const XRATE = 2000;
const CAR_FRONT = 2.2; // the sedan's bumper face ahead of its spawn centre

/** The World and the first sim (page.world, page.device, page.sim, page.layout, ...). */
export async function startSim(page) {
  const { qs, cfg, loading, loadCancel } = page;
  const { importSrc, carKind } = page.vehicle;
  // The World's scene (rubble::play::softbody on the explicit solver): the ground, and the
  // sim's static shapes drawn (placeScene below). (qs: a preset has set its scene and knobs.)
  const sq = new URLSearchParams(qs); sq.set('solver', 'explicit');
  page.sceneDef = await sceneFromQuery('soft', sq.toString());
  if (importSrc) Object.assign(page.sceneDef, { car: 'import', import: importSrc.kind === 'jbeam' ? { jbeam: importSrc.files, pc: importSrc.pc ?? null } : { truck: importSrc.text } });
  loading.stage('Loading the physics engine (wasm) and opening the GPU');
  await loading.paint();
  page.canvas = document.getElementById('c');
  Object.assign(page, await openWorld(page.sceneDef, { canvas: page.canvas, fail: page.fail }));
  page.pacer = new FramePacer(XRATE, Number(qs.get('minFps')) || 0);
  page.frameStats = { n: 0, gpuMs: null, gpuApprox: false, timesRead: null, lastSteps: 0, stepsSince: 0, callsSince: 0 };
  installSim(page);
  // First build: the vehicle spawned in Rust, then its GPU kernels compiled without blocking
  // (explicitSimAsync, so Chrome's GPU watchdog never fires); a restart's build is a cache hit.
  if (carKind === 'showcase') { loading.stage('Building the car (Rust)'); await loading.paint(); page.vehicle.showcase(); }
  // Cancel: with surface contact on, reload the same scene without it (its kernels are the
  // slow ones); otherwise there is no lighter variant to offer.
  if (cfg.opt.surfaceContact && !page.crash) {
    loadCancel.label = 'Cancel: load without surface contact';
    loadCancel.hint = 'Stops waiting and reloads this scene with surface contact off (the Surface contact option turns it back on).';
    loadCancel.run = () => { const s = new URLSearchParams(location.search); s.set('surfaceContact', '0'); location.search = s.toString(); };
  } else {
    loadCancel.label = 'Cancel';
    loadCancel.hint = 'Stops waiting. The kernels keep compiling in the browser until the page is closed or reloaded.';
    loadCancel.run = () => page.fail(new Error('Loading cancelled.'));
  }
  page.sim = await page.world.explicitSimAsync(page.spec(), loading.progress);
  page.watch();
  page.layout = page.sim.layout();
  page.labelProfile();
  page.noWheelsNotice();
  // The sim collides with its own copies of its static shapes; the World holds them only to
  // draw (not obstacles: the arena has hundreds), slots 1..n after the ground, until the
  // scene is placed again with other values (placeBoxes).
  page.world.placeScene({ ...page.sceneDef, explicitBoxes: page.layout.boxes });
  page.drawnBoxes = { key: JSON.stringify(page.layout.boxes), bodies: null, n: page.layout.boxes.length };
}

// The sim's methods on the page.
function installSim(page) {
  const { qs, cfg } = page;
  const P = cfg.P;
  // The damage reads' last step and wall time (damageTick).
  let damageStep = -1, damageWall = -1e9;
  // The sim's spec: a crash reference run's id, run, contact and designs' files (Rust builds
  // the rest: crash_ref.rs), or the lab scene's vehicle, knobs and options.
  page.spec = () => {
    if (page.crash) return { crashRef: { ...page.crash.sel, run: page.crash.run, contact: page.crash.contact, designs: page.crash.folders.designs }, timing: true };
    const [x, z] = cfg.spawn;
    const xscene = cfg.xscene, opt = cfg.opt;
    return {
      vehicles: [{ ...page.vehicle.source(), at: [x, z + CAR_FRONT], front: true, yaw: 0 }], scene: xscene, speed: cfg.xspeed,
      // The scene's fields (XPARAMS), km/h to m/s.
      ...Object.fromEntries((cfg.tables.XPARAMS[xscene] || []).filter(([k]) => k !== 'n' && k !== 'xspeed').map(([k, label, d]) => [k, P(k, d) / (label.includes('km/h') ? 3.6 : 1)])),
      // arena: ?n= copies of the vehicle at 90–150 km/h into walls, poles and (?headOn= share of
      // lanes) each other, or (?layout=ring) all at the centre from concentric rings
      // (?ringRadius=); ?cols=, ?hitMax= s (the solver's defaults otherwise).
      ...(xscene === 'arena' ? { count: P('n', 256), headOn: P('headOn', 0), ...(qs.has('cols') ? { cols: P('cols', 0) } : {}), ...(qs.has('hitMax') ? { hitMax: P('hitMax', 1.2) } : {}), layout: qs.get('layout') === 'ring' ? 'ring' : 'rows', ringRadius: P('ringRadius', 30) } : {}),
      // ?runUp= m: the crash scenes' gap to what they hit (default 1 + v/2 s).
      ...(qs.has('runUp') ? { runUp: P('runUp', 0) } : {}),
      ...(xscene === 'traffic' && qs.has('ai') ? { ai: qs.get('ai') } : {}),
      // ?drag=0: no per-node air drag or turbulence (as the acceptance runs).
      drag: opt.drag,
      // Surface contact: collision triangles meet the boxes and shapes by their surface too
      // (jbeam-semantics-spec.md §38).
      surfaceContact: opt.surfaceContact,
      // Centred damping (§39) off or on; unset, the profile's default.
      ...(opt.centredDamping === null ? {} : { centredDamping: opt.centredDamping }),
    };
  };
  // The simulation's clock in the ?hz= rate's units (the soft scenes' cadence of reads).
  page.simSteps = () => Math.floor((page.sim?.steps() ?? 0) * cfg.hz / XRATE);
  page.anyDrivable = () => !!page.sim && !!page.layout && page.layout.vehicles.some((_, i) => page.sim.drivable(i));
  // Raycast wheels asked for but the vehicle has none (a boat, a bare body): it spawned
  // without them; say so (a notice, not an error).
  page.noWheelsNotice = () => {
    if (!cfg.raycast || page.anyDrivable()) return;
    const text = 'This vehicle has no wheels: it is not drivable here (try the 15 m drop).';
    const d = document.createElement('div');
    d.textContent = text;
    d.style.cssText = 'position:fixed;top:12px;left:50%;transform:translateX(-50%);z-index:50;padding:8px 14px;border-radius:8px;background:rgba(20,28,48,.92);color:#fff;font:13px system-ui;';
    document.body.appendChild(d);
    console.warn(text);
  };
  // The stop on an unstable simulation or a lost device (stability-stop.mjs; the judgement
  // is Rust's, read-only): once stopped, no more steps and no more frames.
  page.watch = () => { page.stop = createStabilityStop(page.sim.health(), { params: { scene: cfg.scene, query: location.search } }); };
  // After a new sim: the frame's counters and reads start over.
  const fresh = () => {
    page.pacer.reset(1);
    Object.assign(page.frameStats, { n: 0, gpuMs: null });
    page.damage = null; damageStep = -1; page.lastPlayer = null;
  };
  page.resetScene = () => {
    if (page.stop?.stopped) return;
    page.sim?.free();
    page.sim = page.world.explicitSim(page.spec());
    page.layout = page.sim.layout();
    page.labelProfile();
    page.noWheelsNotice();
    page.watch();
    fresh();
    if (page.lab) { page.lab.rebuilt(); page.lab.read(); }
    page.crashRef?.reset();
  };
  // The World's drawn copies of the sim's static shapes, placed again when they changed.
  const placeBoxes = (rows) => {
    const key = JSON.stringify(rows), d = page.drawnBoxes;
    if (key === d.key) return;
    const old = d.bodies ?? Array.from({ length: d.n }, (_, i) => page.world.bodyAt(1 + i)).filter((b) => b !== undefined);
    for (const b of old) page.world.destroyBody(b);
    page.drawnBoxes = { key, bodies: page.world.placeExplicitBoxes(rows), n: rows.length };
    page.draw.colDirty = true;
  };
  // The lab's scene placed again with other values (crash-lab-scene.mjs; `crash`: a reference
  // crash's or family's new entry, its designs' folders loaded first): a restart whose
  // kernels compile without blocking (a new design may need its own), then the drawn boxes.
  page.respawn = async (crash) => {
    if (page.stop?.stopped) return;
    if (crash) {
      const folders = await crashRefFolders(crash.info);
      page.crash.scenario = crash.info; page.crash.folders = folders;
      page.crashRef?.setScenario(crash.info);
    }
    const next = await page.world.explicitSimAsync(page.spec(), null);
    page.sim?.free(); page.sim = next; page.layout = page.sim.layout(); page.labelProfile(); page.watch();
    fresh();
    placeBoxes(page.layout.boxes);
    if (page.lab) { page.lab.replaced(); page.lab.read(); }
    page.crashRef?.reset();
  };
  page.launch = () => { page.stop?.launched(cfg.xspeed); return page.sim.launch(cfg.xspeed).catch(page.fail); };
  installFrame(page);
  installReads(page);
  // Damage a few times a second of wall time (a slow frame takes many steps).
  page.damageTick = () => {
    if (page.simSteps() - damageStep >= 30 && performance.now() - damageWall >= 250) { damageStep = page.simSteps(); damageWall = performance.now(); page.readDamage().catch(page.fail); }
  };
}

// The steps of one displayed frame: 2,000 a second of wall time, paced by Rust's FramePacer
// (frame_pacer.rs: whole steps with the remainder carried, real time down to 20 fps, slow
// motion eased), or the lab's clock (slow motion, pause, stepping) in chunks, each followed by
// a frame-state read (impact, recording, picking). Every chunk is one submit; its GPU time from
// timestamps a few times a second. ?minFps=N: a frame-rate floor instead.
function installFrame(page) {
  const fs = page.frameStats;
  let userUntil = 0;
  page.frameSteps = (dt) => {
    const sim = page.sim;
    let steps;
    const t0 = performance.now();
    // Driving: the keys and gamepad go to the player's raycast wheels (Rust does the rest).
    if (page.cfg.drive && sim.drivable(page.player)) {
      const i = page.driveOverride ?? page.readInput();
      // Self-driving scenes: the keys take over the player's car for 3 s after a touch.
      if (i.forward || i.steer || i.handbrake) userUntil = performance.now() + 3000;
      const user = !sim.selfDriving() || performance.now() < userUntil || page.driveOverride;
      if (user) sim.setDriveInput(page.player, i.forward ?? 0, i.steer ?? 0, i.brake ?? 0, i.handbrake ?? 0);
      if (sim.selfDriving()) sim.autopilot(Math.max(0, dt), user ? page.player : -1);
    }
    if (page.lab) {
      steps = page.lab.stepsFor(dt, sim.maxFrameSteps());
      for (const k of page.lab.chunks(steps)) if (k) { sim.frame(k); fs.callsSince++; page.lab.read(); page.crashRef?.sampled(); }
    } else {
      steps = page.pacer.advance(dt, 1, sim.maxFrameSteps());
      sim.frame(steps);
      if (steps) fs.callsSince++;
    }
    fs.lastSteps = steps;
    fs.stepsSince += steps;
    if (++fs.n % 15 === 0 && !fs.timesRead && steps) {
      // The budget's GPU cost: the timed solver frames' mean (a lab frame submits its steps
      // in chunks) for their mean steps.
      const calls = fs.callsSince, taken = fs.stepsSince;
      fs.stepsSince = 0; fs.callsSince = 0;
      // Without timestamp queries: submit-to-done time of this frame (an upper bound).
      fs.timesRead = sim.timed()
        ? sim.readFrameTimes().then((t) => { if (t.length && sim === page.sim) { fs.gpuMs = t.reduce((a, b) => a + b, 0) / t.length; page.pacer.reportGpu(fs.gpuMs, taken / calls); } })
        : page.device.queue.onSubmittedWorkDone().then(() => { fs.gpuMs = performance.now() - t0; fs.gpuApprox = true; page.pacer.reportGpu(fs.gpuMs, steps); });
      fs.timesRead.finally(() => { fs.timesRead = null; }).catch(page.fail);
    }
    return steps;
  };
}

// Non-blocking reads (one of each in flight): the player's pose (or the drive state, which
// also poses the wheels' stand-in boxes) and every vehicle's beam damage.
function installReads(page) {
  let poseRead = null, damageRead = null;
  // Raycast wheels are drawn as static boxes of the World (render only), posed from readDrive.
  const wheelBodies = [];
  const drawWheels = (d) => {
    const { world } = page;
    page.wheelsDrawn = d.vehicles.reduce((n, v) => n + (v?.wheels.length ?? 0), 0);
    // A body that draws its own wheels (the showcase body, or an import's wheel pieces) needs no boxes.
    const own = page.draw.ownWheels();
    const opt = page.cfg.opt;
    if (opt.wheelBoxes === null) page.el.optRow.querySelector('#optWheelBoxes').checked = !own;
    if (!(opt.wheelBoxes ?? !own)) {
      // Off: boxes made earlier go out of sight (far below the ground).
      for (const list of wheelBodies) for (const b of list ?? []) world.bodySetType(b, 'static', [0, -1000, 0], [0, 0, 0, 1], [0, 0, 0], [0, 0, 0]);
      return;
    }
    d.vehicles.forEach((v, i) => {
      if (!v) return;
      wheelBodies[i] ??= [];
      v.wheels.forEach((w, k) => {
        // A box R × R × width, its local +X along the axle, turned about it by the wheel's spin.
        const fw = cross3(w.axle, w.up), c = Math.cos(w.spin), s = Math.sin(w.spin);
        const z = [0, 1, 2].map((a) => fw[a] * c - w.up[a] * s), y = [0, 1, 2].map((a) => w.up[a] * c + fw[a] * s);
        const q = quatFromBasis(z, y);
        if (!wheelBodies[i][k]) {
          const body = world.createBody({ position: w.centre, rotation: q });
          // Render only (the prop material's friction, as the scene's static props).
          world.createShape(body, { material: { friction: 0.6 } }, { box: { half: [w.width / 2, w.radius, w.radius] } });
          wheelBodies[i][k] = body;
        }
        world.bodySetType(wheelBodies[i][k], 'static', w.centre, q, [0, 0, 0], [0, 0, 0]);
      });
    });
  };
  page.readPoses = () => {
    if (poseRead) return poseRead;
    const sim = page.sim, step = page.simSteps(), xstep = sim.steps();
    const cam = page.cam;
    if (page.cfg.raycast && page.anyDrivable()) {
      poseRead = sim.readDrive().then((d) => {
        poseRead = null;
        if (sim !== page.sim) return;
        drawWheels(d);
        const me = d.vehicles[page.player];
        if (!me) return;
        page.frameTrace?.poses.push({ step: xstep, p: me.centre.slice(), speed: me.speed, at: performance.now() });
        // The rates the camera carries the pose forward with: from the previous read, the
        // body's own velocity (sliding too) and how its axes turn.
        const last = page.lastPlayer;
        const prev = last?.xstep < xstep && xstep - last.xstep <= 400 && last.fw ? last : null;
        const per = prev ? XRATE / (xstep - prev.xstep) : 0;
        const rate = (a, b) => a.map((x, i) => (x - b[i]) * per);
        page.lastPlayer = { p: me.centre, q: quatFromBasis(me.forward, me.up), v: me.forward.map((x) => x * me.speed), step, xstep, speed: me.speed,
          fw: me.forward, up: me.up, vel: prev ? rate(me.centre, prev.p) : null, dfw: prev ? rate(me.forward, prev.fw) : null, dup: prev ? rate(me.up, prev.up) : null };
        cam.targetPose = cam.fromPose = page.lastPlayer;
      }, (e) => { poseRead = null; throw e; });
      return poseRead;
    }
    poseRead = sim.readPoses().then((f) => {
      poseRead = null;
      if (sim !== page.sim) return;
      page.lastPlayer = { p: [f[0], f[1], f[2]], q: [0, 0, 0, 1], v: [f[3], f[4], f[5]], step, xstep };
      cam.targetPose = cam.fromPose = page.lastPlayer;
    }, (e) => { poseRead = null; throw e; });
    return poseRead;
  };
  page.readDamage = () => {
    if (damageRead) return damageRead;
    const sim = page.sim, layout = page.layout;
    // The display lists a few vehicles: a big fleet reads only the first 8.
    damageRead = sim.readDamage(layout.vehicles.length > 16 ? 8 : undefined).then((d) => {
      damageRead = null;
      if (sim !== page.sim) return page.damage;
      page.damage = d.map((x, i) => ({ name: layout.vehicles[i].name, beams: x.beams, yielded: x.bent, broken: x.broken }));
      d.forEach((x, i) => page.stop?.damage(i, x.broken, x.bent, x.beams));
      page.breaks = d.reduce((n, x) => n + x.broken, 0);
      return page.damage;
    }, (e) => { damageRead = null; throw e; });
    return damageRead;
  };
  // The reads in flight, if any (automation waits for them before its own).
  page.poseRead = () => poseRead;
  page.damageRead = () => damageRead;
}
