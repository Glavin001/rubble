// The pages' performance measurements, one implementation (demo.html and the playground
// shell use it; tests/demo_profiling.test.mjs tests it): the live rates (TPS, Max TPS from
// the step's own cost, FPS, Max FPS from CPU time per frame), the GPU phase profile and
// the fair benchmark protocol. GPU timestamps come from World (gpuTiming 'step' /
// 'phases'; decoded and tested in Rust, engine-world events.rs).

// Live HUD step cost. Playback steps once per rAF, so completion-to-completion intervals
// are pinned to the display refresh, not the engine. With 'timestamp-query' World times
// each live step's pass (GPU time, render excluded). Otherwise (Safari/iOS) busy time
// stands in: frame arrival minus the later of submit and the previous arrival. Arrivals
// are seen at polls (once per frame), so the estimate reads high and is labelled as one.
export function liveStepSample({ gpuMs, submittedAt, completedAt, previousCompletedAt }) {
  if (gpuMs > 0) return { ms: gpuMs, source: 'gpu' };
  return { ms: completedAt - Math.max(submittedAt ?? completedAt, previousCompletedAt ?? 0), source: 'estimate' };
}
// Median over a rate window; GPU samples win, so an occasional untimed step cannot mix
// estimate latency into a GPU-timed readout.
export function summarizeStepCost(samples) {
  const gpu = samples.filter((sample) => sample.source === 'gpu');
  const used = (gpu.length ? gpu : samples).map((sample) => sample.ms).filter((ms) => ms > 0).sort((a, b) => a - b);
  if (!used.length) return { stepMs: 0, maxTps: 0, source: 'none' };
  const stepMs = used[used.length >> 1];
  return { stepMs, maxTps: 1000 / stepMs, source: gpu.length ? 'gpu' : 'estimate' };
}
export function stepCostLabels({ stepMs, maxTps, source }) {
  if (source === 'gpu') return { step: `${stepMs.toFixed(1)} ms/step GPU`, maxTps: maxTps.toFixed(0), maxTpsSub: 'GPU-timed' };
  if (source === 'estimate') return { step: `≈${stepMs.toFixed(1)} ms/step est.`, maxTps: `≈${maxTps.toFixed(0)}`, maxTpsSub: 'estimate' };
  return { step: '', maxTps: '—', maxTpsSub: '' };
}

/**
 * Rolling 1 s windows: TPS = physics steps/s, FPS = presents/s (observed, so
 * display-bound). Max TPS = 1000 / measured physics step cost (`liveStepSample`); Max
 * FPS = 1000 / CPU time per frame callback. `onWindow(readout)` gets each closed window:
 * `{ tps, fps, maxTps, maxFps, stepMs, stepSource, frameMs, labels }`, or
 * `{ measuring: true }` while a benchmark owns the queue.
 */
export function createRateMeter(onWindow, now = () => performance.now()) {
  let start = now(), steps = 0, frames = 0, frameMsSum = 0, costs = [], measuring = false;
  const reset = () => { start = now(); steps = 0; frames = 0; frameMsSum = 0; costs = []; };
  return {
    get measuring() { return measuring; },
    set measuring(v) { measuring = !!v; },
    reset,
    /** A step's cost (`liveStepSample`): ignored while measuring. */
    addStepCost(sample) { if (!measuring) costs.push(sample); },
    /** A benchmark pauses live steps and saturates the queue: no live rate means anything. */
    showMeasuring() { onWindow({ measuring: true }); },
    /** One frame callback: whether it stepped, its CPU time (ms). Closes 1 s windows. */
    frame(steppedCount, frameMs) {
      frames++;
      frameMsSum += frameMs;
      steps += steppedCount;
      const elapsed = now() - start;
      if (elapsed < 1000) return;
      const sec = elapsed / 1000;
      const frameMsAvg = frames > 0 ? frameMsSum / frames : 0;
      const cost = summarizeStepCost(costs);
      onWindow({
        tps: steps / sec, fps: frames / sec, maxTps: cost.maxTps, maxFps: frameMsAvg > 1e-6 ? 1000 / frameMsAvg : 0,
        stepMs: cost.stepMs, stepSource: cost.source, frameMs: frameMsAvg, labels: stepCostLabels(cost),
      });
      reset();
    },
  };
}

// GPU split of the step: World times each phase of its plan in its own pass (gpuTiming:
// phases) for 10 steps, each waited for. Per step, a phase that repeats (each iteration's
// primal and dual) is summed.
export function summarizePhases(samples) {
  const median = (values) => values.slice().sort((a, b) => a - b)[values.length >> 1];
  const names = [...new Set(samples.flatMap((s) => Object.keys(s.phasesMs)))];
  const phasesMs = Object.fromEntries(names.map((name) => [name, median(samples.map((s) => s.phasesMs[name] ?? 0))]));
  const totals = Object.fromEntries(['gpuPhasesMs', 'gpuSpanMs', 'completionMs'].map((k) => [k, median(samples.map((s) => s[k]))]));
  return { phasesMs, ...totals };
}
export function phaseSample(t, completionMs) {
  const phasesMs = {};
  for (const [name, ms] of t.segments) phasesMs[name] = (phasesMs[name] ?? 0) + ms;
  return { step: t.step, phasesMs, gpuPhasesMs: t.totalMs, gpuSpanMs: t.spanMs, completionMs };
}
/**
 * The phase profile: `{ summary (text), report }`, or `{ summary: '', report: undefined }`
 * without timestamp queries. `frameTimes()` returns (and `clearFrameTimes()` empties) the
 * GPU times the page's frames carried since the last clear.
 */
export async function profilePhases({ world, stepOne, drain, liveTiming, wantTimestamps, frameTimes, clearFrameTimes, now = () => performance.now() }) {
  if (!wantTimestamps) return { summary: '', report: undefined };
  world.setGpuTiming('phases');
  const samples = [];
  try {
    for (let i = 0; i < 10; i++) {
      await drain();
      clearFrameTimes();
      const t0 = now();
      await stepOne();
      await drain();
      const t = frameTimes().find((x) => x.segments.length > 1);
      // A step that dispatched nothing (World skips the GPU step once nothing is awake or
      // written) has no phases: every phase reads zero.
      if (!t) samples.push({ step: null, phasesMs: {}, gpuPhasesMs: 0, gpuSpanMs: 0, completionMs: now() - t0, skipped: true });
      else samples.push(phaseSample(t, now() - t0));
    }
  } finally {
    world.setGpuTiming(liveTiming);
  }
  const summary = summarizePhases(samples);
  const report = {
    protocol: 'world-phase-timestamps-v1',
    detail: 'plan phases',
    caveat: 'One pass per plan phase and a wait per step perturb execution. These are not the batched throughput window. Phase medians need not sum to the median total.',
    measuredAt: new Date().toISOString(), ...summary, samples,
  };
  const text = Object.entries(summary.phasesMs).map(([name, value]) => `${name} ${value.toFixed(2)}`).join(' · ')
    + ` · profiled GPU span ${summary.gpuSpanMs.toFixed(2)} / completion ${summary.completionMs.toFixed(2)} ms`;
  return { summary: text, report };
}

/** The colour buckets the awake bodies fall in (benchmark coverage diagnostics). */
export async function readColorBuckets({ world, maxColors, awake }) {
  const counts = Array.from(await world.readColorCounts()).slice(0, maxColors);
  const total = counts.reduce((sum, count) => sum + count, 0);
  if (total !== awake) throw new Error(`Color buckets cover ${total} bodies, but awake count is ${awake}; refusing invalid benchmark coverage.`);
  return counts;
}

/**
 * The fair three-avbd protocol (blind lead-in + median of 10-step batches): every step
 * through World, its readback included. Colour buckets are read outside the timed window.
 * Returns the timing samples (`wallMs` and the workload around it).
 */
export async function benchWall({ lead, stepOne, drain, counters, buckets, validate, measure, meta = {}, progress = () => {} }) {
  for (let i = 0; i < lead; i++) {
    await stepOne();
    if (i % 10 === 9) { await drain(); progress(i + 1, lead); }
  }
  await drain();
  const workloadStart = { ...counters(), colorBuckets: await buckets() };
  validate('after lead-in');
  const samples = await measure({ step: stepOne, drain });
  Object.assign(samples, { measuredAt: new Date().toISOString(), leadSteps: lead, host: 'World (engine-world-wasm)', ...meta });
  await drain();
  validate('after timed window');
  samples.workload = { start: workloadStart, end: { ...counters(), colorBuckets: await buckets() } };
  return samples;
}
