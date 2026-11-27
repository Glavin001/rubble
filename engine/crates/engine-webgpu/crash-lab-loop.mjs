// The crash lab page's frame loop (crash-lab.html): each animation frame the sim's steps
// (crash-lab-sim.mjs frameSteps), the pose read, the camera, one submit of the drawing, the
// HUD (#xstats, #speed, #damage, #events) and the status line. At most two frames in flight
// on the GPU (as a game loop keeps it, so fps and real time count finished work). A stopped
// page (an unstable state, a lost device, a failure) submits nothing more; automation
// (page.manual: crash-lab-hooks.mjs) steps it itself and the loop only draws.

const XRATE = 2000;

/** page.updateHud, page.updateStatus; startLoop(page) runs the first frame. */
export function createLoop(page) {
  const { el, qs } = page;
  const fs = page.frameStats;
  // ?car=buggy: the buggy has rigid parts, the sedan runs here.
  const xnote = page.vehicle.carKind === 'buggy' ? '<br>(the buggy has rigid parts: the sedan runs here)' : '';
  page.updateHud = () => {
    const d = page.damage?.[0], pacer = page.pacer;
    const pace = !page.manual && pacer.speed() < 0.97 ? ` · <b>${Math.round(pacer.speed() * 100)} % pace</b> (GPU behind)` : '';
    el.xstats.innerHTML = `explicit 2 kHz · ${fs.lastSteps} steps/frame${page.layout.wide ? ' · wide' : ''}${pace}<br>GPU ${fs.gpuMs == null ? '…' : (fs.gpuApprox ? '≈' : '') + fs.gpuMs.toFixed(2)} ms/frame`
      + (d ? `<br>${d.broken} broken · ${d.yielded} bent / ${d.beams} beams` : '') + xnote;
    if (page.lastPlayer) el.speed.innerHTML = `${(Math.hypot(...page.lastPlayer.v) * 3.6).toFixed(0)} <small>km/h</small>`;
    page.damageTick();
    const damage = page.damage;
    if (damage) {
      el.damage.innerHTML = damage.map((x, i) => ({ ...x, i })).filter((x) => x.i === page.player || (x.beams > 200 && (damage.length < 6 || x.yielded > 0))).slice(0, 8)
        .map((x) => `${x.i === page.player ? '▶ ' : ''}${x.name.split(' (')[0].slice(0, 22)}: ${x.yielded} bent · ${x.broken} broken / ${x.beams}`).join('<br>');
      el.events.innerHTML = `${page.breaks} beam-break events`;
    }
  };
  page.updateStatus = () => {
    const n = page.sim.steps(), lay = page.layout;
    el.status.textContent = `explicit beams · ${lay.vehicles[0].name.split(' (')[0]}: ${lay.nodes} nodes, ${lay.beams} beams, ${lay.triangles} triangles · 2000 Hz · step ${n} (${(n / XRATE).toFixed(1)} s)`
      + (page.manual ? '' : ` · ${page.fps.toFixed(0)} fps · ${(page.realTime * 100).toFixed(0)}% real time`);
  };
  // ?stopAt=N: the page's own loop takes no steps past step N (in ?hz= units; automation from
  // a known state: ?stopAt=0 leaves every step to __step).
  const stopAt = qs.has('stopAt') ? Number(qs.get('stopAt')) : Infinity;
  let lastT = performance.now(), fpsAcc = 0, fpsN = 0, simAcc = 0, wallAcc = 0, inFlight = 0;
  const submit = () => {
    const enc = page.device.createCommandEncoder();
    page.drawScene(enc);
    page.device.queue.submit([enc.finish()]);
  };
  const frame = (now) => {
    const { sim, lab } = page;
    if (page.stop?.stopped || window.__failed) return;
    if (!page.manual) { if (lab?.st.grab) page.stop?.disturbed(); page.stop?.check(sim.steps() / XRATE); }
    if (!page.manual && inFlight >= 2) { page.frameTrace?.frames.push({ now, skipped: 1 }); requestAnimationFrame(frame); return; }
    const dt = Math.min(0.05, (now - lastT) / 1000);
    lastT = now;
    fpsAcc += dt; fpsN++;
    if (fpsAcc > 0.5) { page.fps = fpsN / fpsAcc; fpsAcc = 0; fpsN = 0; }
    if (page.manual || page.simSteps() >= stopAt) {
      page.updateCamera(dt, false);
      submit();
      requestAnimationFrame(frame);
      return;
    }
    page.labDt = dt;
    const steps = page.frameSteps(dt);
    simAcc += steps / XRATE; wallAcc += dt;
    if (wallAcc > 0.5) { page.realTime = simAcc / wallAcc; simAcc = 0; wallAcc = 0; }
    page.readPoses().catch(page.fail);
    // Paced (not paused, stepped or replayed): the frame shows the state its lead past the
    // last step, and the camera eases by the wall time that motion stands for; otherwise as
    // stepped, by the frame's wall time. (The lab's gather carries the nodes forward, the
    // plain views' skin and beams do in the shader.)
    const paced = !lab || (!lab.st.paused && lab.st.replay < 0);
    page.lead = paced ? page.pacer.lead() : 0;
    // The orbit cameras' subject at the step on display, before the camera moves.
    lab?.followFocus(0, page.lead);
    page.updateCamera(paced ? page.pacer.shownDt() : dt, false);
    submit();
    inFlight++;
    page.device.queue.onSubmittedWorkDone().then(() => { inFlight--; });
    const pacer = page.pacer, c = page.cam;
    page.frameTrace?.frames.push({ now, dt, steps, step: page.sim.steps(), slow: lab ? (lab.st.auto && lab.st.impactStep == null ? 1 : lab.st.slow) : 1,
      eye: c.eye.slice(), at: c.target.slice(), focus: lab?.st.focus?.slice() ?? null, focusStep: lab?.st.lastState?.step ?? null, gpu: fs.gpuMs, scale: pacer.scale(), speed: pacer.speed(), msPerStep: pacer.msPerStep(), share: pacer.share(), lead: page.lead });
    page.updateHud();
    page.updateStatus();
    requestAnimationFrame(frame);
  };
  page.startLoop = () => {
    // The first frame on screen ends the loading card (its stages: window.__loadTimes).
    page.loading.stage('First frame');
    requestAnimationFrame((t) => {
      frame(t);
      page.device.queue.onSubmittedWorkDone().then(() => { if (!window.__failed) window.__loadTimes = page.loading.done(); });
    });
  };
}
