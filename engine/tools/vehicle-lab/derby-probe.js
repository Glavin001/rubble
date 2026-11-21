// The derby's scripted check (vehicle-lab-record.mjs --probe --eval this file):
// a scripted player (the page's own key handlers: W held, A and D in turn every second)
// and, every second, the round as the lab (Rust) reports it into window.__abLog, which
// the probe prints at the end: each car's plan or why it is out, km/h, damage %, target,
// position and how many of its wheels touch something (0 while it should move: beached).
//
//   engine/tools/vehicle-lab/run.sh --probe --secs 60 --eval engine/tools/vehicle-lab/derby-probe.js \
//     'derby4:mode=derby&derbyCars=4'
(() => {
  window.__abLog = [];
  const key = (code, down) => dispatchEvent(new KeyboardEvent(down ? 'keydown' : 'keyup', { code }));
  const player = new URLSearchParams(location.search).get('derbyPlayer') !== '0';
  if (player) key('KeyW', true);
  let k = 0;
  setInterval(() => {
    k++;
    if (player) {
      key('KeyA', false); key('KeyD', false);
      if (k % 3 === 1) key('KeyA', true); else if (k % 3 === 2) key('KeyD', true);
    }
    const g = window.__derby?.();
    const s = window.__state?.();
    if (g) window.__abLog.push(JSON.stringify({
      t: +g.t.toFixed(1), followed: Array.from(window.__lab.followPose()).slice(0, 3).map((x) => +x.toFixed(1)), sim: s && +s.simRate.toFixed(0), fps: s && +s.fps.toFixed(0), left: g.left, over: g.over, winner: g.winner, bodies: s?.bodies,
      cars: g.cars.map((c) => [c.out ? `OUT:${c.out}` : c.plan, +c.kmh.toFixed(0), +(100 * c.damage).toFixed(0), c.target, c.pos && c.pos.map((x) => +x.toFixed(1)), c.grounded]),
      events: g.events.slice(0, 4).map((e) => `${e.t.toFixed(1)} ${e.text}`),
    }));
  }, 1000);
  return `derby probe: ${player ? 'scripted player on' : 'spectator'}`;
})()
