// The lab's director: frame-locked takes (window.__record, record-takes.mjs) and instant
// replay, in every bond mode. It only sequences what the modes already do: steps,
// shots along rays, the orbit camera and the view. The physics is the mode's own.
//
// Frame-locked: each video frame advances a fixed amount of simulated time (1/60 s,
// divided by a segment's `slow`), whatever the wall clock does, so a take is the same
// physics as live play at real time; slow motion steps fewer physics ticks per video
// frame at the same dt (lossless).
//
// Replay re-runs the scene from its load: the lab's modes repeat a run exactly (same
// build, same GPU), so the same shots at the same ticks give the same result. It
// catches up without drawing, plays the last moments slowly from another angle, then
// carries on live from where the run was. The replay never feeds anything back into a
// run: it is a fresh one. The status reports whether it reached the same state.
//
// lab: {stressOn, stress, sim, renderer, scene, canvas, stepHz, showBonds, pause,
//   resume, reload, setStatus, stats, shotSettings, applyShotSettings, shoot}

const VIDEO_HZ = 60;

export function createDirector(lab) {
  // Shots since the scene loaded: {tick, origin, direction, settings}, {tick, fire: true,
  // settings} for the stress modes' ball along the structure's line of attack, or {tick,
  // detonate: name} for a pack's demolition charges ('' every charge, each at its delay), or
  // {tick, launch: name} for a pack's launched bonded bodies ('' every one, each at its delay).
  let shots = [];
  let busy = false;
  // Fractional physics ticks owed (ticks per video frame need not be whole).
  let credit = 0;

  const tickHz = () => lab.stressOn ? lab.stress.stats.frameHz : lab.stepHz();
  const tick = () => lab.stressOn ? (lab.stress.stats.frames ?? 0) : lab.sim.steps;

  // Advance `ticks` physics ticks and draw (stress: one call steps and draws).
  async function advance(ticks, draw = true) {
    if (lab.stressOn) { await lab.stress.frameSteps(ticks); return; }
    for (let done = 0; done < ticks; done += 4) await lab.sim.step(Math.min(4, ticks - done));
    if (draw) lab.renderer.draw(lab.showBonds());
  }
  // One video frame of 1/(60 slow) s simulated.
  async function videoFrame(slow = 1) {
    credit += tickHz() / (VIDEO_HZ * slow);
    const ticks = Math.floor(credit + 1e-9);
    credit -= ticks;
    await advance(ticks);
  }
  // The pending shot (a ray in the lab's frame) fired now, as live play does.
  function fire(shot) {
    if (shot.detonate !== undefined) return lab.stress.detonate(shot.detonate);
    if (shot.launch !== undefined) return lab.stress.launch(shot.launch);
    lab.applyShotSettings(shot.settings);
    if (shot.fire) return lab.stress.fire();
    if (lab.stressOn) return lab.stress.shootRay(shot.origin, shot.direction);
    lab.sim.launch(shot.origin, shot.direction);
    return true;
  }
  // The scene's own orbit when the take starts: `rel` / `relTo` are offsets from it
  // ([yaw + a, pitch + b, distance × c]), so a take frames any structure.
  let base = null;
  const relative = (r) => r && [base[0] + r[0], base[1] + r[1], base[2] * r[2]];
  // `at` / `atTo` (stress modes): the orbit's target as a point of the scene as it was built
  // (the stress frame, Z-up: a Rust layout's own coordinates), eased from `at` to `atTo`
  // across the segment: a camera that follows a launched vehicle in.
  const built = (p) => {
    const s = lab.stress.stats.placement || [0, 0, 0];
    return [p[0] + s[0], p[2] + s[1], -p[1] + s[2]];
  };
  function camera(cam, t) {
    if (!cam) return;
    const from = cam.orbit ?? relative(cam.rel), to = cam.to ?? relative(cam.relTo);
    const [yaw, pitch, dist] = to ? from.map((v, k) => v + (to[k] - v) * t) : from;
    const set = lab.stressOn ? lab.stress : lab.renderer;
    let target = cam.target || [];
    if (cam.at && lab.stressOn) {
      const a = built(cam.at), b = cam.atTo ? built(cam.atTo) : a;
      target = a.map((v, k) => v + (b[k] - v) * t);
    }
    set.setOrbit(yaw + (cam.spin || 0), pitch, dist, target);
  }

  // A frame-composited JPEG (canvas + caption + stats line), PUT to the recorder.
  function compositor(opts) {
    const canvas = lab.canvas();
    const W = Math.min(opts.width || 1280, canvas.width), H = Math.round((W * canvas.height) / canvas.width / 2) * 2;
    const out = new OffscreenCanvas(W, H), g = out.getContext('2d');
    const pending = new Set();
    let n = 0;
    return {
      async frame(caption) {
        g.drawImage(lab.canvas(), 0, 0, W, H);
        if (caption !== '') {
          g.fillStyle = 'rgba(10,14,22,0.62)';
          g.fillRect(12, H - 74, Math.min(W - 24, 760), 62);
          g.fillStyle = '#fff';
          g.font = '600 22px system-ui, sans-serif';
          g.fillText(caption, 24, H - 46);
          g.font = '16px ui-monospace, monospace';
          g.fillText(lab.stats(), 24, H - 22);
        }
        const blob = await out.convertToBlob({ type: 'image/jpeg', quality: 0.88 });
        const p = fetch(`${opts.upload || '/rec'}/${opts.name || 'take'}/f${String(n++).padStart(5, '0')}.jpg`, { method: 'PUT', body: blob }).then(() => pending.delete(p));
        pending.add(p);
        if (pending.size > 16) await Promise.race(pending);
      },
      async done() { await Promise.all(pending); return n; },
    };
  }

  // Script segments: {n (video frames at 60 fps), slow?, caption?, cam?: {orbit: [yaw,
  // pitch, dist], to?: [yaw, pitch, dist] (eased across the segment), or rel / relTo
  // (offsets from the scene's orbit: [yaw + a, pitch + b, distance × c]), spin? (rad per
  // frame), target?: [x, y, z]}, act?: 'fire' (the row shot; in the stress modes the ball
  // along the structure's line of attack) | {shoot: [u, v]}
  // (through the viewport) | {ray: [origin, direction]} | {detonate: name} (a pack's
  // charges; '' sets off every one at its delay) | {launch: name} (a pack's launched bonded
  // bodies: a car, a bus, an aircraft; '' every one at its delay), view?: {bonds, color}, mass?,
  // speed? (shot scales, powers of two as the sliders)}.
  async function record(script, opts = {}) {
    await lab.pause();
    busy = true;
    const t0 = performance.now();
    const out = compositor(opts);
    base = (lab.stressOn ? lab.stress : lab.renderer).orbit;
    let caption = '', spin = 0;
    try {
      for (const seg of script) {
        if (seg.caption !== undefined) caption = seg.caption;
        if (seg.view) lab.setView(seg.view);
        if (seg.mass !== undefined || seg.speed !== undefined) lab.applyShotSettings({ ...lab.shotSettings(), ...(seg.mass !== undefined && { mass: seg.mass }), ...(seg.speed !== undefined && { speed: seg.speed }) });
        if (seg.cam) spin = 0;
        if (seg.act?.detonate !== undefined) {
          // A Topple pack's charges, cut between frames (stress modes).
          const shot = { tick: tick(), detonate: seg.act.detonate };
          if (!lab.stressOn || !fire(shot)) throw new Error(`take: no charge ${JSON.stringify(seg.act.detonate)} to set off`);
          shots.push(shot);
        } else if (seg.act?.launch !== undefined) {
          // A pack's launched bonded bodies, set moving between frames (stress modes).
          const shot = { tick: tick(), launch: seg.act.launch };
          if (!lab.stressOn || !fire(shot)) throw new Error(`take: no launch ${JSON.stringify(seg.act.launch)} to release`);
          shots.push(shot);
        } else if (seg.act === 'fire' && lab.stressOn) {
          // The stress modes' ball along the structure's line of attack (a Wrecking Yard pack's).
          const shot = { tick: tick(), fire: true, settings: lab.shotSettings() };
          if (!fire(shot)) throw new Error('take: this scene has no line of attack to fire along');
          shots.push(shot);
        } else if (seg.act) {
          const ray = seg.act === 'fire' ? lab.scene?.rowShot
            : seg.act.shoot ? lab.aim(...seg.act.shoot)
            : seg.act.ray ? { origin: seg.act.ray[0], direction: seg.act.ray[1] } : null;
          if (!ray) throw new Error(`take: no shot for ${JSON.stringify(seg.act)}`);
          const shot = { tick: tick(), origin: [...ray.origin], direction: [...ray.direction], settings: lab.shotSettings() };
          fire(shot);
          shots.push(shot);
        }
        const n = seg.n || 0;
        for (let i = 0; i < n; i++) {
          if (seg.cam) { camera({ ...seg.cam, spin: spin }, n > 1 ? i / (n - 1) : 1); spin += seg.cam.spin || 0; }
          await videoFrame(seg.slow || 1);
          await out.frame(caption);
        }
      }
      const frames = await out.done();
      return { frames, seconds: (performance.now() - t0) / 1000, summary: await lab.summary() };
    } finally {
      busy = false;
      lab.resume();
    }
  }

  // Instant replay: re-run from the load, catch up to `lead` s before the last shot, play
  // `window` s from there at 1/`slow` speed while the camera swings `turn` rad round to
  // the live view, catch up to where the run was, and resume live.
  async function replay({ lead = 0.5, window: span = 2.5, slow = 4, turn = 0.9, onProgress = () => {} } = {}) {
    if (busy || !shots.length) return null;
    busy = true;
    try {
      await lab.pause();
      const log = shots, end = tick(), before = await lab.summary();
      const view = () => lab.stressOn ? lab.stress : lab.renderer;
      const [yaw, pitch, dist, ...target] = view().orbit;
      const hz = tickHz();
      const from = Math.max(0, log.at(-1).tick - Math.round(lead * hz));
      const until = Math.min(end, from + Math.round(span * hz));
      await lab.reload();
      shots = [];
      let next = 0;
      const fireDue = () => { while (next < log.length && log[next].tick <= tick()) { fire(log[next]); shots.push(log[next]); next++; } };
      // Fast, drawing now and then: up to `to`, firing each shot at its tick.
      const catchUp = async (to, p0, p1) => {
        for (let t = tick(); t < to; t = tick()) {
          fireDue();
          const stop = next < log.length ? Math.min(to, log[next].tick) : to;
          await advance(Math.max(1, Math.min(8, stop - t)), false);
          onProgress(p0 + (p1 - p0) * (tick() / Math.max(1, to)));
        }
      };
      await catchUp(from, 0, 0.3);
      // The window, slowed and paced at 60 video frames per wall second.
      credit = 0;
      const frames = Math.max(1, Math.ceil(((until - tick()) / hz) * VIDEO_HZ * slow));
      const t0 = performance.now();
      for (let i = 0; tick() < until; i++) {
        fireDue();
        view().setOrbit(yaw + turn * (1 - Math.min(1, i / frames)), pitch, dist, target);
        credit += hz / (VIDEO_HZ * slow);
        const ticks = Math.min(until - tick(), Math.floor(credit + 1e-9));
        credit -= ticks;
        if (ticks) await advance(ticks); else if (lab.stressOn) await lab.stress.frameSteps(0); else lab.renderer.draw(lab.showBonds());
        onProgress(0.3 + 0.6 * Math.min(1, i / frames));
        const wait = t0 + ((i + 1) * 1000) / VIDEO_HZ - performance.now();
        if (wait > 0) await new Promise((r) => setTimeout(r, wait));
      }
      view().setOrbit(yaw, pitch, dist, target);
      await catchUp(end, 0.9, 1);
      fireDue();
      if (!lab.stressOn) lab.renderer.draw(lab.showBonds());
      const after = await lab.summary();
      return { same: JSON.stringify(after) === JSON.stringify(before), before, after };
    } finally {
      busy = false;
      lab.resume();
    }
  }

  return {
    record, replay,
    get busy() { return busy; },
    get shots() { return shots.length; },
    // A pack's charges set off ('' every one), for replays.
    logDetonate(name) { shots.push({ tick: tick(), detonate: name }); },
    // A pack's launched bonded bodies released ('' every one), for replays.
    logLaunch(name) { shots.push({ tick: tick(), launch: name }); },
    // The stress modes' ball fired along the line of attack, for replays.
    logFire() { shots.push({ tick: tick(), fire: true, settings: lab.shotSettings() }); },
    // A live shot (click or the row's ball), for replays.
    logShot(origin, direction) { shots.push({ tick: tick(), origin: [...origin], direction: [...direction], settings: lab.shotSettings() }); },
    // A new load: the log starts over.
    reset() { shots = []; credit = 0; },
  };
}
