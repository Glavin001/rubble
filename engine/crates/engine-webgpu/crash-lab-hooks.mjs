// The crash lab page's automation hooks (crash-lab.html): the window.__* functions the
// recorders and checks call (crash-lab-record.mjs, crash-lab-check.mjs, crash-ref-record.mjs,
// drive-check.mjs, explicit-check.mjs, ...). Each reads or drives page state
// (crash-lab-state.mjs); none holds any of its own. The lab's own (__lab, __labUi) are
// crash-lab-mount.mjs's and crash-lab-ui.mjs's.
//   __step(n)        n solver steps (2 kHz), each frame of up to 33 waited on; then a drawn
//                    frame; returns { steps, player, damage, breaks, counters, solver }
//   __live(ms)       the page's own loop for ms of wall time: { fps, realTime }
//   __setView(o)     { cam: { mode, target, from }, beams, orbit: { yaw, pitch, dist, target }, chase }
//   __setCaption(t)  a caption over the frame (recordings), the panels hidden
//   __renderBench(n) ms per drawn frame (no physics)
//   __frameTrace(on) a per-frame pacing log (returned when turned off)
//   __explicit       { sim(), layout(), launch(), gpuMs(), drive(input), readDrive(), wheelBodies() }
//   __crashRef       a reference run's { scenario, registry, info(), progress(), report() }
//   __world, __device, __options, __keepVehicle, __solver ('explicit'), __ready, __failed

import { drain } from './rubble-world.mjs';

const XSTEPS = Math.round(2000 / 60);

/** The hooks, once the page is up (window.__ready last). */
export function installHooks(page) {
  const { device, world } = page;
  const drawNow = async () => {
    const enc = device.createCommandEncoder();
    page.drawScene(enc);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  };
  window.__step = async (n = 1) => {
    page.manual = true;
    const sim = page.sim;
    for (let i = 0; i < n;) {
      const k = Math.min(XSTEPS, n - i);
      sim.frame(k); page.frameStats.lastSteps = k; i += k;
      const r = page.lab?.read();
      await device.queue.onSubmittedWorkDone();
      if (r) await r;
    }
    const times = await sim.readFrameTimes();
    if (times.length) page.frameStats.gpuMs = times.reduce((a, b) => a + b, 0) / times.length;
    if (page.poseRead()) await page.poseRead().catch(() => {});
    await page.readPoses();
    if (page.damageRead()) await page.damageRead().catch(() => {});
    await page.readDamage();
    page.updateHud();
    page.updateStatus();
    page.updateCamera(0.1, false);
    await drawNow();
    return { steps: page.sim.steps(), player: page.lastPlayer, damage: page.damage, breaks: page.breaks, counters: null, solver: 'explicit' };
  };
  window.__setView = (o) => {
    const cam = page.cam;
    if (o.cam) {
      cam.mode = o.cam.mode ?? cam.mode;
      if (o.cam.target !== undefined) cam.targetIdx = page.findSubject(o.cam.target, cam.targetIdx);
      if (o.cam.from !== undefined) cam.fromIdx = page.findSubject(o.cam.from, cam.fromIdx);
      page.sync();
    }
    if (o.beams !== undefined) page.beamView = o.beams;
    if (o.orbit) { cam.mode = 'orbit'; cam.orbit = { ...cam.orbit, ...o.orbit }; }
    if (o.chase) cam.mode = 'chase';
  };
  window.__setCaption = (text, clean = true) => {
    const c = document.getElementById('caption');
    c.textContent = text || ''; c.style.display = text ? 'block' : 'none';
    for (const id of ['info', 'status']) { const e = document.getElementById(id); if (e) e.style.display = clean ? 'none' : ''; }
  };
  window.__world = world;
  window.__device = device;
  window.__live = async (ms = 3000) => {
    page.manual = false;
    await new Promise((r) => setTimeout(r, ms));
    page.manual = true;
    await drain(world, () => {});
    return { fps: page.fps, realTime: page.realTime };
  };
  window.__renderBench = async (n = 60) => {
    page.manual = true;
    const t = performance.now();
    for (let i = 0; i < n; i++) { page.updateCamera(1 / 60, false); await drawNow(); }
    return (performance.now() - t) / n;
  };
  window.__frameTrace = (on = true) => { const t = page.frameTrace; page.frameTrace = on ? { frames: [], poses: [] } : null; return t; };
  window.__options = page.cfg.opt;
  window.__keepVehicle = page.cfg.keepVehicle;
  window.__crashRef = page.crashRef ? { scenario: page.crash.scenario, registry: page.crash.reg, info: () => page.crashRef.info(), progress: () => page.crashRef.progress(), report: () => page.crashRef.report() } : null;
  window.__solver = 'explicit';
  window.__explicit = {
    sim: () => page.sim, layout: () => page.layout, launch: () => page.sim.launch(page.cfg.xspeed), gpuMs: () => page.frameStats.gpuMs,
    // Driving automation: hold an input ({ forward, steer, brake, handbrake }; null: the keys
    // again) and read the vehicles' drive state.
    drive: (i) => { page.driveOverride = i; if (i && page.sim.drivable(page.player)) page.sim.setDriveInput(page.player, i.forward ?? 0, i.steer ?? 0, i.brake ?? 0, i.handbrake ?? 0); },
    readDrive: () => (page.anyDrivable() ? page.sim.readDrive() : Promise.resolve({ vehicles: [] })), wheelBodies: () => page.wheelsDrawn,
  };
  window.__ready = true;
}
