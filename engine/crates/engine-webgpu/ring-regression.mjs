// Browser counterpart to tests/ring_hold.rs. Run through the actual demo encoder,
// including ten-step batches without CPU counter readbacks and scene reset.
const checkpoints = [1, 30, 60, 120, 240, 480];

export async function runRingRegression({ initial, submit, read, progress, batchSize = 10 }) {
  const samples = [];
  let steps = 0;
  for (const target of checkpoints) {
    while (steps < target) {
      const end = Math.min(target, steps + batchSize);
      while (steps < end) { submit(); steps++; }
      // Yield for progress without introducing a counter readback. Warm-start
      // metadata must stay correct when the CPU's cached contact count is stale.
      await progress(steps);
    }
    const { bytes, counters } = await read();
    const view = new DataView(bytes);
    const require = (ok, message) => { if (!ok) throw new Error(`ring regression at step ${steps}: ${message}`); };
    require(counters.overflow === 0, `capacity overflow 0x${counters.overflow.toString(16)}`);
    require(counters.awake === initial.length - 1, `active workload slept: awake=${counters.awake}`);
    let maxDrift = 0, total = 0, flanks = 0, worst = 0;
    for (let id = 1; id < initial.length; id++) {
      const o = id * 112;
      for (const offset of [0, 4, 8, 16, 20, 24, 28, 48, 52, 56, 64, 68, 72])
        require(Number.isFinite(view.getFloat32(o + offset, true)), `body ${id}, field ${offset} is nonfinite`);
      if (Math.abs(initial[id].p[0]) > 40) {
        const drift = Math.abs(view.getFloat32(o + 4, true) - initial[id].p[1]);
        total += drift; flanks++;
        if (drift > maxDrift) { maxDrift = drift; worst = id; }
      }
    }
    const meanDrift = total / flanks;
    const ballOffset = (initial.length - 1) * 112;
    const ball = [0, 4, 8].map(offset => view.getFloat32(ballOffset + offset, true));
    const meanLimit = steps <= 120 ? 0.2 : steps <= 240 ? 0.5 : 1.6;
    require(meanDrift < meanLimit, `widespread collapse: mean flank drift ${meanDrift} m exceeds ${meanLimit} m`);
    require(steps > 120 || maxDrift < 0.5, `flank body ${worst} drifted ${maxDrift} m; check support contacts and dispatch coverage`);
    require(ball[1] > 7.5 && ball[1] < 9.5, `sphere left the floor: ${ball}`);
    if (steps === 1) require(counters.contact > 1_250_000 && counters.contact < 1_320_000,
      `first-frame contacts=${counters.contact}; reference is ~1.285M (exact-touch/polygon coverage)`);
    if (steps === 480) require(ball[2] > 90, `sphere failed to traverse both walls: ${ball}`);
    const sample = { steps, maxDrift, meanDrift, ball, ...counters };
    samples.push(sample);
    console.log('RING_CHECK', JSON.stringify(sample));
  }
  return { passed: true, bodies: initial.length - 1, steps, batchSize, samples };
}
