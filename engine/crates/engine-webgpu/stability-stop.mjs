// A page's stop on an unstable simulation or a lost device. All judgement is in
// Rust (engine-world-wasm stability.rs: `ExplicitSim.health()`,
// `WebWorld.explicitHealth()`); this file only asks it once per displayed frame,
// remembers the answer and shows the panel.
//
// Detection and stop only: nothing here (or in the Rust guard) changes the
// simulation. It reads the vehicles' state back; when the state is impossible for
// correct physics (a non-finite value, kinetic energy far past what the scene
// supplied, a body exploding with rising energy) the page stops submitting steps,
// keeps its last frame on screen, and reports an ENGINE ERROR with the numbers and
// a copyable report. It never resumes by itself: the button reloads the page.
//
//   const stop = createStabilityStop(xsim.health(), { params: spec });
//   // each displayed frame, before stepping:
//   if (stop.stopped) return;            // no more steps, no more frames
//   stop.check(simulatedSeconds);
//   // optional (what the check cannot see for itself): stop.damage(i, broken, bent, beams);
//   // stop.launched(speed); stop.supplied(joules); stop.disturbed();

const STYLE = `
.rb-stop { position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); z-index: 30;
  width: min(760px, calc(100vw - 32px)); max-height: calc(100vh - 32px); overflow: auto; box-sizing: border-box;
  background: rgba(24, 10, 10, 0.96); border: 1px solid #c0392b; border-radius: 10px; padding: 16px 18px;
  color: #f2e9e9; font: 13px/1.45 system-ui, sans-serif; box-shadow: 0 8px 30px rgba(0,0,0,0.6); }
.rb-stop h2 { margin: 0 0 8px; font-size: 16px; font-weight: 700; color: #ff8a80; }
.rb-stop p { margin: 0 0 8px; }
.rb-stop pre { white-space: pre-wrap; word-break: break-word; margin: 8px 0; padding: 8px; background: rgba(0,0,0,0.35); border-radius: 6px;
  font: 12px/1.4 ui-monospace, monospace; color: #ffd0cc; max-height: 40vh; overflow: auto; user-select: text; }
.rb-stop button { margin: 4px 8px 0 0; background: #2b5fb8; color: #fff; border: 0; border-radius: 6px; padding: 6px 14px; font: 600 13px system-ui, sans-serif; cursor: pointer; }
.rb-stop button.rb-quiet { background: #3a2a2a; }
`;

const lostText = (text) => /lost the GPU device|device lost/i.test(text);

/**
 * The stop for one simulation. `health`: its `VehicleHealth` (null: only a lost
 * device stops it, through `lost`). `params`: the scene's parameters, for the
 * report. `onStop(message)`: the page's own bookkeeping (pause flags, HUD).
 */
export function createStabilityStop(health, { params = {}, onStop = null } = {}) {
  const api = {
    /** The stop's message once stopped (final), else null. */
    stopped: null,
    /** Ask Rust about the state after the steps submitted so far (`t` simulated
     * seconds). Never waits; a stop shows up in `stopped` a frame or two later. */
    check(t) {
      if (api.stopped || !health) return;
      const now = health.stopped();
      if (now) { stop(now); return; }
      health.check(t).then((m) => { if (m) stop(m); }, (e) => { console.warn('[rubble] stability check failed', e); });
    },
    /** A vehicle's damage as the page read it (for the report). */
    damage(i, broken, bent, beams) { if (!api.stopped) health?.damage(i, broken, bent, beams); },
    /** The page launched the vehicles at `speed` m/s (energy the scene supplied). */
    launched(speed) { if (!api.stopped) health?.launched(speed); },
    /** Something outside the vehicles put up to `joules` into them (a shot's energy). */
    supplied(joules) { if (!api.stopped) health?.supplied(joules); },
    /** The scene has inputs the check cannot bound (rigid bodies shot or pressed at
     * the vehicles): only non-finite values stop it while `on`. */
    unbounded(on) { if (!api.stopped) health?.setUnbounded(on); },
    /** An input the check cannot bound acts this frame (a hand's grab): call it
     * every frame while it does. */
    disturbed() { if (!api.stopped) health?.disturbed(); },
    /** The device was lost (the page's `device.lost`): stop with that message. */
    lost(message) { stop(`The browser lost the GPU device (${message}). Nothing on this page is running; the simulation was not restarted.`); },
  };
  function stop(message) {
    if (api.stopped) return;
    api.stopped = message;
    window.__stopped = message;
    const lost = lostText(message) && !message.startsWith('ENGINE ERROR');
    const report = `${message}\nscene: ${location.href}\nparameters: ${JSON.stringify(params)}\nbrowser: ${navigator.userAgent}`;
    console.error(`[rubble] simulation stopped\n${report}`);
    try { onStop?.(message); } catch (e) { console.warn(e); }
    if (!document.getElementById('rb-stop-style')) {
      const s = document.createElement('style');
      s.id = 'rb-stop-style';
      s.textContent = STYLE;
      document.head.appendChild(s);
    }
    document.getElementById('fatal')?.remove();
    const card = document.createElement('div');
    card.className = 'rb-stop';
    card.id = 'stopped';
    card.setAttribute('role', 'alert');
    card.innerHTML = '<h2></h2><p></p><pre></pre><button type="button" class="rb-restart"></button><button type="button" class="rb-quiet rb-copy">Copy the report</button>';
    card.querySelector('h2').textContent = lost ? 'The browser lost the GPU device' : 'Engine error: the simulation became unstable and was stopped';
    card.querySelector('p').textContent = lost
      ? 'The page stopped. It does not rebuild by itself: reload when you are ready.'
      : 'This is a bug in the simulation, not a limit on what you may do: the state below cannot come from correct physics. Stepping was stopped (nothing was changed or damped) so the broken state does not overload the GPU; the picture is the last frame. Please report it: the report has the scene link and the numbers.';
    card.querySelector('pre').textContent = report;
    const restart = card.querySelector('.rb-restart');
    restart.textContent = lost ? 'Reload the page' : 'Restart (reload the scene)';
    restart.onclick = () => location.reload();
    const copy = card.querySelector('.rb-copy');
    copy.onclick = () => navigator.clipboard?.writeText(report).then(() => { copy.textContent = 'Copied'; }, () => { copy.textContent = 'Select the text to copy'; });
    document.body.appendChild(card);
  }
  return api;
}
