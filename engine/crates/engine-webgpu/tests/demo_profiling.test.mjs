import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  liveStepSample, summarizeStepCost, stepCostLabels, createRateMeter, profilePhases, readColorBuckets, benchWall,
} from '../perf-meter.mjs';

// The pages' benchmark and HUD measurements (perf-meter.mjs, used by demo.html and the
// playground shell). GPU timestamps come from World (gpuTiming 'step' / 'phases'; their
// decoding is tested in Rust, engine-world events::tests::gpu_time_sums_segments_and_spans_gaps);
// the pages keep the protocol around them: coverage checks outside the timed window, the
// per-phase summary and the live HUD.
const plain = (value) => JSON.parse(JSON.stringify(value));

test('demo.html and the shell measure through perf-meter.mjs', () => {
  for (const page of ['../demo.html', '../play.mjs']) {
    const src = readFileSync(new URL(page, import.meta.url), 'utf8');
    assert.match(src, /perf-meter\.mjs/, `${page} imports the shared measurements`);
  }
});

for (const scenario of ['valid', 'coverage mismatch', 'device loss']) {
  test(`color-bucket diagnostics validate GPU coverage: ${scenario}`, async () => {
    const world = { async readColorCounts() {
      if (scenario === 'device loss') throw new Error('simulated bucket readback device loss');
      return new Uint32Array([19, 0, 7, 5, 99, 99]); // past the palette: max degrees
    } };
    const result = readColorBuckets({ world, maxColors: 4, awake: scenario === 'coverage mismatch' ? 32 : 31 });
    if (scenario === 'valid') assert.deepEqual(plain(await result), [19, 0, 7, 5]);
    else await assert.rejects(result, scenario === 'device loss'
      ? /simulated bucket readback device loss/
      : /Color buckets cover 31 bodies, but awake count is 32/);
  });
}

test('benchmark captures color buckets before and after timing without reads inside the measured interval', async () => {
  const events = [];
  let measuring = false;
  const report = await benchWall({
    lead: 0, stepOne: async () => {}, drain: async () => { events.push('drain'); },
    counters: () => ({ awake: 110441 }),
    buckets: async () => { assert.equal(measuring, false); events.push('buckets'); return [60000, 50441]; },
    validate: () => {},
    measure: async ({ step, drain }) => {
      events.push('timing start'); measuring = true; await step(); await drain(); measuring = false;
      events.push('timing end'); return { wallMs: 12 };
    },
  });
  assert.deepEqual(events, ['drain', 'buckets', 'timing start', 'drain', 'timing end', 'drain', 'buckets'],
    'bucket diagnostics must not contaminate throughput batches or sample a different physics step');
  assert.deepEqual(plain(report.workload), {
    start: { awake: 110441, colorBuckets: [60000, 50441] }, end: { awake: 110441, colorBuckets: [60000, 50441] },
  });
  assert.equal(report.host, 'World (engine-world-wasm)');
});

// World times each phase of its plan in its own pass. The page sums a phase that repeats
// (each iteration's primal and dual), takes medians over 10 waited steps and puts the live
// timing back afterwards, even when a sample is invalid.
function profileFixture({ invalid = false } = {}) {
  const modes = [];
  let clock = 0, step = 0, times = [];
  const args = {
    wantTimestamps: true, liveTiming: 'step', now: () => clock,
    world: { setGpuTiming(m) { modes.push(m); } },
    drain: async () => {},
    frameTimes: () => times, clearFrameTimes: () => { times = []; },
    stepOne: async () => {
      clock += 20;
      step++;
      if (invalid) return;
      times.push({
        step, totalMs: 9, spanMs: 10,
        segments: [['broadphase', 1], ['narrowphase', 2], ['primal', 2], ['dual', 0.5], ['primal', 2], ['dual', 0.5], ['sleep', 1]],
      });
    },
  };
  return { args, modes };
}
test('phase profile sums repeated phases, keeps every sample and restores live timing', async () => {
  const { args, modes } = profileFixture();
  const { summary, report } = await profilePhases(args);
  assert.deepEqual(modes, ['phases', 'step']);
  assert.equal(report.samples.length, 10, 'retain individual samples so phase/CPU discrepancies cannot be hidden by medians');
  assert.deepEqual(plain(report.phasesMs), { broadphase: 1, narrowphase: 2, primal: 4, dual: 1, sleep: 1 });
  assert.equal(report.gpuPhasesMs, 9); assert.equal(report.gpuSpanMs, 10); assert.equal(report.completionMs, 20);
  assert.match(report.caveat, /not the batched throughput window/);
  assert.match(summary, /primal 4\.00 · dual 1\.00/);
  assert.match(summary, /profiled GPU span 10\.00 \/ completion 20\.00 ms/);
});
test('a profiled step that dispatched nothing reads zero for every phase, and live timing comes back', async () => {
  const { args, modes } = profileFixture({ invalid: true });
  const { summary, report } = await profilePhases(args);
  assert.deepEqual(modes, ['phases', 'step']);
  assert.equal(report.samples.length, 10);
  assert.ok(report.samples.every((s) => s.skipped && s.gpuSpanMs === 0));
  assert.match(summary, /profiled GPU span 0\.00 \/ completion 20\.00 ms/);
});

// Live HUD step cost. Playback steps once per rAF, so completion intervals are
// display-bound; Max TPS must come from the step's own cost. The numbers mirror ring110k
// on an M3 Max: ~11 ms GPU per step behind a 60 Hz display.
test('live step cost uses the step\'s GPU time, not the display-bound completion interval', () => {
  assert.deepEqual(plain(liveStepSample({ gpuMs: 11.1, submittedAt: 1000, completedAt: 1016.7, previousCompletedAt: 1000 })), { ms: 11.1, source: 'gpu' });
});

test('without GPU time the live estimate is busy time and is labelled', () => {
  assert.deepEqual(plain(liveStepSample({ gpuMs: 0, submittedAt: 1000, completedAt: 1012.5, previousCompletedAt: 990 })), { ms: 12.5, source: 'estimate' });
  assert.deepEqual(plain(liveStepSample({ gpuMs: 0, submittedAt: 1000, completedAt: 1030, previousCompletedAt: 1018 })), { ms: 12, source: 'estimate' });
});

test('step cost window reports a median, prefers GPU samples and derives Max TPS from cost', () => {
  const gpu = [11.0, 11.2, 10.9, 25.0, 11.1].map((ms) => ({ ms, source: 'gpu' }));
  const mixed = [...gpu, { ms: 16.7, source: 'estimate' }, { ms: 16.7, source: 'estimate' }];
  const result = plain(summarizeStepCost(mixed));
  assert.equal(result.source, 'gpu');
  assert.equal(result.stepMs, 11.1, 'a hitch must not move the readout, and estimates must not mix into GPU time');
  assert.ok(Math.abs(result.maxTps - 1000 / 11.1) < 1e-9);
  assert.deepEqual(plain(stepCostLabels(result)), { step: '11.1 ms/step GPU', maxTps: '90', maxTpsSub: 'GPU-timed' });
  const estimate = plain(summarizeStepCost([{ ms: 14, source: 'estimate' }, { ms: 13, source: 'estimate' }, { ms: 15, source: 'estimate' }]));
  assert.deepEqual(estimate, { stepMs: 14, maxTps: 1000 / 14, source: 'estimate' });
  assert.deepEqual(plain(stepCostLabels(estimate)), { step: '≈14.0 ms/step est.', maxTps: '≈71', maxTpsSub: 'estimate' });
  assert.deepEqual(plain(stepCostLabels(summarizeStepCost([]))), { step: '', maxTps: '—', maxTpsSub: '' });
});

test('live rates keep TPS observed, Max TPS from step cost, and report measuring while a benchmark runs', () => {
  let clock = 0, last = null;
  const meter = createRateMeter((r) => { last = r; }, () => clock);
  const playSecond = () => {
    for (let frame = 0; frame < 60; frame++) {
      clock += 16.7;
      meter.addStepCost({ ms: 11.1, source: 'gpu' });
      meter.frame(1, 2);
    }
  };
  playSecond();
  assert.equal(last.tps.toFixed(0), '60', 'TPS is the observed, display-bound step rate');
  assert.equal(last.labels.maxTps, '90', 'Max TPS is 1000 / GPU step cost, not 1000 / 16.7 ms');
  assert.equal(last.labels.maxTpsSub, 'GPU-timed');
  assert.equal(last.labels.step, '11.1 ms/step GPU');
  assert.equal(last.stepSource, 'gpu');
  assert.equal(last.maxFps.toFixed(0), '500', 'Max FPS is 1000 / CPU time per frame (2 ms)');

  meter.measuring = true;
  meter.showMeasuring();
  assert.deepEqual(last, { measuring: true });
  // Late live frames and the benchmark's saturated queue must not reach the HUD.
  meter.addStepCost({ ms: 45, source: 'estimate' });
  meter.measuring = false;
  meter.reset();
  playSecond();
  assert.equal(last.labels.step, '11.1 ms/step GPU');
  assert.equal(last.labels.maxTps, '90');
});
