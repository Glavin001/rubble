// Match pinned three-avbd compare(... rounds:2, batches:5, perBatch:10,
// warmup:0). The caller builds the scene and completes its 60-step lead-in.
// Keep every sample: the best round alone hides throttling and competing work.
// Validate outside timed batches: a fast result with missing work is not a result.
// allowSleep: the run measures the production configuration with sleeping on, so
// fewer awake bodies is expected; the result is labelled, not compared with
// three-avbd (whose benchmark keeps every body awake).
export function assertBenchmarkWorkload({ scene, totalBodies, iterations, dt, counters, stage, allowSleep = false }) {
  const reject = reason => { throw new Error(`Physics benchmark ${scene} ${stage}: ${reason}; timing comparison rejected.`); };
  if (!Number.isInteger(counters.overflow) || counters.overflow !== 0) {
    reject(`overflow=${counters.overflow}; constraints may have been discarded`);
  }
  const expected = { ring110k: [110442, 4], gables506k: [505923, 3] }[scene];
  if (!expected) return; // Settled scenes intentionally allow sleeping bodies.
  if (totalBodies !== expected[0] || iterations !== expected[1] || Math.abs(dt - 1 / 60) > 1e-8 || !Number.isFinite(dt)) {
    reject(`expected ${expected[0]} total bodies, ${expected[1]} iterations and dt=1/60; got ${totalBodies}, ${iterations}, ${dt}`);
  }
  if (!allowSleep && counters.awake !== totalBodies - 1) {
    reject(`expected all ${totalBodies - 1} dynamic bodies awake, got ${counters.awake}; sleeping or omitted bodies reduce measured work`);
  }
  for (const key of ['contact', 'manifold', 'color']) {
    if (!Number.isInteger(counters[key]) || counters[key] <= 0) reject(`invalid ${key} count ${counters[key]}; full active collision/solve work is required`);
  }
}

export async function measurePhysicsBatches({ step, drain, now = () => performance.now() }) {
  await drain();
  const rounds = [];
  for (let round = 0; round < 2; round++) {
    for (let i = 0; i < 3; i++) step();
    await drain();
    const samplesMs = [];
    for (let batch = 0; batch < 5; batch++) {
      const start = now();
      for (let i = 0; i < 10; i++) step();
      await drain();
      const ms = (now() - start) / 10;
      if (!Number.isFinite(ms) || ms <= 0) {
        throw new Error(`Physics timing round ${round + 1}, batch ${batch + 1}: invalid ${ms} ms/step; GPU completion and a monotonic clock are required.`);
      }
      samplesMs.push(ms);
    }
    rounds.push({ samplesMs, medianMs: [...samplesMs].sort((a, b) => a - b)[2] });
  }
  return {
    protocol: 'three-avbd-2x5x10',
    stepsPerBatch: 10,
    untimedStepsPerRound: 3,
    measuredSteps: 100,
    rounds,
    wallMs: Math.min(...rounds.map(r => r.medianMs)),
    // This is a timing result, never a declaration of scene correctness/parity.
    budgetMs: 1000 / 120,
    allBatchesWithin120Hz: rounds.every(r => r.samplesMs.every(ms => ms <= 1000 / 120)),
  };
}
