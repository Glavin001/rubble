// The crash lab mounted on the page (crash-lab.html): the lab (crash-lab.mjs: time control,
// impact, the recorded run, the studio HUD, grabbing) on the page's sim, renderer and camera;
// a reference crash's criteria and report (crash-ref.mjs); the scene's parameters
// (crash-lab-scene.mjs); the page's own controls (crash-lab-controls.mjs) placed in the lab's
// panels (crash-lab-ui.mjs: one place each); and the lab's automation hook window.__lab.

import { crashReferenceRegistry, crashReferenceScenario } from './rubble-world.mjs';
import { crashRefCams, createCrashRef } from './crash-ref.mjs';
import { mountSceneParams } from './crash-lab-scene.mjs';
import { createCrashLab } from './crash-lab.mjs';
import { mat4Inv } from './view-math.mjs';

/** page.lab, page.crashRef, page.sceneParams and window.__lab. */
export async function mountLab(page) {
  const { cfg, qs, cam } = page;
  const { XSCENES, XCRASH, XPILL, LAB_PRESETS } = cfg.tables;
  // The scene picker's reference crashes (the registry, Rust's) and a run's own cameras,
  // sample steps and first contact.
  const reg = page.crash?.reg ?? await crashReferenceRegistry();
  const scenario = () => page.crash?.scenario ?? null;
  const lab = page.lab = createCrashLab({
    sim: () => page.sim, layout: () => page.layout, renderer: page.draw.labR, canvas: page.canvas, qs, preset: page.crash ? null : cfg.preset, presets: LAB_PRESETS, xscene: cfg.xscene, body: !!page.draw.bodyR, shell: page.shell,
    scenes: XSCENES, sceneUrl: cfg.urls.xscene, presetUrl: cfg.urls.preset,
    pill: () => (scenario() ? `${scenario().id} · ${scenario().title}`.toUpperCase() : XPILL[cfg.xscene]?.() ?? cfg.xscene.toUpperCase()), modified: () => !!page.sceneParams?.modified(), crash: !!page.crash || XCRASH.has(cfg.xscene), drive: cfg.drive,
    refs: reg.scenarios, refId: cfg.refId ? page.crash.scenario.id : null, familyId: cfg.family, registry: reg, catalog: cfg.catalog, refUrl: cfg.urls.ref, familyUrl: cfg.urls.family,
    ...(page.crash ? { cams: crashRefCams(page.crash.scenario), sampleSteps: reg.sampleSteps, impactAt: () => page.sim.crashInfo()?.impactStep } : {}),
    setOrbit: (o) => {
      if (o.chase) { cam.mode = 'chase'; page.sync(); return; }
      if (o.follow) { if (cam.mode === 'orbit') { cam.orbit.target = o.follow; cam.orbit.yaw += o.spin; } return; }
      // (A fixed camera, a reference crash's own, brings its target.)
      cam.mode = 'orbit'; cam.orbit = { ...cam.orbit, yaw: o.yaw, pitch: o.pitch, dist: o.dist, target: o.target ?? cam.orbit.target }; page.sync();
    },
    view: () => page.labView, inverse: mat4Inv, restart: () => page.resetScene(), fail: page.fail, pacer: page.pacer, camMode: () => cam.mode,
  });
  lab.read();
  const ui = lab.ui;
  if (page.crash) {
    const c = page.crash;
    page.crashRef = createCrashRef({ registry: reg, scenario: c.scenario, sim: () => page.sim, lab, contact: c.contact, run: c.run, fail: page.fail,
      url: (change) => cfg.urls.ref(c.scenario.id, change),
      mount: { results: ui.bodies.results, scene: ui.bodies.scene, summary: (html) => ui.summary('results', html) } });
  }
  placeControls(page, ui, reg);
  installLabHook(page);
}

// The page's own controls into the lab's panels. Scene: the scene's parameters (a reference
// crash's or family's, or the lab scene's; a change places the scene again: respawn), and for
// the free scenes the vehicle and the solver's options and link (a reference crash's are its
// scenario's: hidden there, not disabled). View: the car-relative cameras join the lab's
// camera group. Results: the live readouts under a reference crash's criteria.
function placeControls(page, ui, reg) {
  const { el, cfg, qs, lab } = page;
  const div = (cls, ...kids) => { const d = document.createElement('div'); d.className = cls; d.append(...kids); return d; };
  const sub = (t) => { const d = div('sub'); d.textContent = t; return d; };
  const scn = ui.bodies.scene;
  page.sceneParams = mountSceneParams({ mount: scn, qs, registry: reg, catalog: cfg.catalog, crash: page.crash?.sel ?? null, info: () => page.crash?.scenario ?? null, resolve: crashReferenceScenario, xscene: cfg.xscene, preset: cfg.preset,
    respawn: (crash) => page.respawn(crash).then(() => lab.sync()).catch(page.fail) });
  if (!page.crash) {
    scn.append(sub('vehicle'), div('r', el.carSel, el.modSel, el.bindSel, el.truckLabel, el.jbeamLabel), el.truckInfo);
    scn.append(sub('solver'), el.optRow, div('r', el.solverBtn));
  }
  const title = el.infoTitle;
  title.classList.add('muted');
  title.style.marginTop = '6px';
  scn.append(title);
  document.getElementById('labCamMore').append(el.camSel, el.camTargetSel, el.camFromLbl, el.camFromSel);
  const camSync = page.sync;
  page.sync = () => { camSync(); lab.sync(); };
  page.sync();
  const live = document.createElement('details');
  live.className = 'lab-live';
  live.open = !page.crash;
  live.innerHTML = '<summary class="sub">live readouts</summary>';
  live.append(el.speed, el.xstats, el.damage, el.events);
  ui.bodies.results.append(live);
  if (!page.crash) {
    // The collapsed Results line of a free scene: speed and the player's damage.
    const line = () => {
      const d = (el.damage.innerHTML.split('<br>')[0] ?? '').replace(/<[^>]*>/g, '').replace(/^.*?: /, '');
      ui.summary('results', `${el.speed.textContent.trim()}${d && d !== '—' ? ` · ${d}` : ''}`);
    };
    line();
    setInterval(line, 300);
  }
  ui.relayout();
}

// window.__lab: n steps in frames of `chunk` (each read: impact to that chunk), then a drawn
// frame; the lab's state.
function installLabHook(page) {
  const { lab, device } = page;
  const $ = (id) => document.getElementById(id);
  const labState = () => {
    const s = lab.st, step = s.replay >= 0 ? s.ring[s.replay].step : page.sim.steps();
    return { step, impactStep: s.impactStep, time: lab.timeText(step).text, reads: s.reads, recorded: s.ring.length, replay: s.replay,
      bent: s.lastState?.bent ?? null, broken: s.lastState?.broken ?? null, grab: s.grab && { node: s.grab.global, target: s.grab.target },
      opts: { ...page.draw.labR.opts, grab: undefined }, slow: s.auto ? 'auto' : s.slow, paused: s.paused, counts: $('labCounts').textContent,
      readout: $('labTime').innerText.replace(/\n/g, ' | '), pills: $('labPills').innerText.replace(/\n/g, ' | ') };
  };
  const drawNow = async () => {
    // Driving: the wheels' poses and the chase camera's subject first.
    if (page.cfg.raycast) await page.readPoses();
    page.updateCamera(0.1, false);
    const enc = device.createCommandEncoder();
    page.drawScene(enc);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  };
  window.__lab = {
    lab, state: labState, draw: drawNow, view: () => ({ vp: Array.from(page.labView.vp), eye: page.labView.eye.slice() }),
    async advance(n, chunk = 17) {
      page.manual = true;
      const reads = [];
      for (let i = 0; i < n;) {
        const k = Math.min(chunk, n - i);
        // A crash reference run: chunks that end on its sample steps, each sample read.
        for (const c of page.crashRef ? lab.chunks(k) : [k]) { page.sim.frame(c); const r = lab.read(); if (r) reads.push(r); const q = page.crashRef?.sampled(); if (q) reads.push(q); }
        i += k; await device.queue.onSubmittedWorkDone();
      }
      await Promise.all(reads);
      await drawNow();
      return labState();
    },
    async frames(n) { page.manual = false; for (let i = 0; i < n; i++) await new Promise((r) => requestAnimationFrame(r)); page.manual = true; await device.queue.onSubmittedWorkDone(); return labState(); },
  };
}
