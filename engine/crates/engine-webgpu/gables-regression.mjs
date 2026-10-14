// Full-size counterpart to tests/gables_hold.rs. Reference capture and bounds:
// docs/avbd-regression-notes.md; tests/reference-gables.html runs the pinned GPU solver.
const checkpoints = [1, 10, 20, 60, 120, 240];
const referenceForward = [-183.28572, -178.78572, -173.78595, -154.0713, -126.1219, -74.0568];

export async function runGablesRegression({ initial, submit, read, progress, batchSize = 10 }) {
  if (initial.length !== 505923) throw new Error(`gables fixture: expected 505923 total bodies, got ${initial.length}`);
  let steps = 0;
  const samples = [];
  const firstBall = initial.length - 2;
  for (const [checkpoint, target] of checkpoints.entries()) {
    while (steps < target) {
      const end = Math.min(target, steps + batchSize);
      while (steps < end) { submit(); steps++; }
      await progress(steps);
    }
    const { bytes, counters } = await read();
    const view = new DataView(bytes);
    const require = (ok, message) => { if (!ok) throw new Error(`gables regression at step ${steps}: ${message}`); };
    require(counters.overflow === 0, `capacity/coloring overflow 0x${counters.overflow.toString(16)}`);
    require(counters.awake === initial.length - 1, `active workload slept: awake=${counters.awake}`);
    if (steps === 1) require(counters.contact === 3916802,
      `lost initial supports: contacts=${counters.contact}, expected 3916802; check exact touch, clipping and tiled dispatch`);
    let total = 0, count = 0, maxDrift = 0, worst = 0;
    for (let id = 1; id < initial.length; id++) {
      const o = id * 112;
      for (const offset of [0, 4, 8, 16, 20, 24, 28, 48, 52, 56, 64, 68, 72])
        require(Number.isFinite(view.getFloat32(o + offset, true)), `body ${id}, field ${offset} is nonfinite`);
      if (id >= firstBall) continue;
      const drift = Math.abs(view.getFloat32(o + 4, true) - initial[id].p[1]);
      if (steps <= 20) require(drift < 0.3, `brick ${id} drifted ${drift} m before impact`);
      const column = Math.floor((id - 1) / (68 * 465));
      if (column === 6 || column === 9) continue;
      total += drift; count++;
      if (drift > maxDrift) { maxDrift = drift; worst = id; }
    }
    const meanDrift = total / count;
    require(meanDrift < 0.2 && maxDrift < 0.6,
      `unhit columns collapsed: mean drift=${meanDrift} m, max=${maxDrift} m at brick ${worst}; reference at 240 is 0.118/0.328 m`);
    const balls = [firstBall, firstBall + 1].map(id => [0, 4, 8].map(offset => view.getFloat32(id * 112 + offset, true)));
    for (const [side, pos] of balls.entries())
      require(Math.abs(pos[0] - initial[firstBall + side].p[0]) < 1 && pos[1] > 4.6 && pos[1] < 4.95
        && Math.abs(pos[2] - referenceForward[checkpoint]) < 1.5,
      `impactor ${side} diverged: actual=${pos}, reference forward=${referenceForward[checkpoint]} ±1.5 m; check rolling/friction/warm starts`);
    const sample = { steps, meanDrift, maxDrift, balls, ...counters };
    samples.push(sample);
    console.log('GABLES_CHECK', JSON.stringify(sample));
  }
  return { passed: true, bodies: initial.length - 1, steps, batchSize, samples };
}
