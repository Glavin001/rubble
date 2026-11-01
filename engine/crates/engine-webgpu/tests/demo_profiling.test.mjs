import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';

// Execute the shipped benchmark and HUD code of demo.html. GPU timestamps come
// from World (gpuTiming 'step' / 'phases'; their decoding is tested in Rust,
// engine-world events::tests::gpu_time_sums_segments_and_spans_gaps); the page
// keeps the protocol around them: coverage checks outside the timed window, the
// per-phase summary and the live HUD.
const html = readFileSync(new URL('../demo.html', import.meta.url), 'utf8');
const slice = (from, to) => {
  const a = html.indexOf(from), b = html.indexOf(to, a);
  assert.ok(a >= 0 && b > a, `demo.html keeps ${from} … ${to}`);
  return html.slice(a, b);
};
const plain = value => JSON.parse(JSON.stringify(value));

const bucketSource = slice('async function readColorBuckets()', '// Fair three-avbd protocol');
for (const scenario of ['valid', 'coverage mismatch', 'device loss']) {
  test(`color-bucket diagnostics validate GPU coverage: ${scenario}`, async () => {
    const context = {
      cfg: { cap: { maxColors: 4 } }, last: { awake: scenario === 'coverage mismatch' ? 32 : 31 }, Array,
      world: { async readColorCounts() {
        if (scenario === 'device loss') throw new Error('simulated bucket readback device loss');
        return new Uint32Array([19, 0, 7, 5, 99, 99]); // past the palette: max degrees
      } },
    };
    const result = runInNewContext(`${bucketSource}\nreadColorBuckets()`, context);
    if (scenario === 'valid') assert.deepEqual(plain(await result), [19, 0, 7, 5]);
    else await assert.rejects(result, scenario === 'device loss'
      ? /simulated bucket readback device loss/
      : /Color buckets cover 31 bodies, but awake count is 32/);
  });
}

test('benchmark captures color buckets before and after timing without reads inside the measured interval', async () => {
  const wallSource = slice('let timingSamples;', 'let phaseSamples;');
  const events = [];
  let measuring = false;
  const context = {
    lead: 0, scene: 'ring110k', bodyCount: 110442, solver: { iterations: 4 }, dt: 1 / 60, sleepTime: 1e6, SLEEP_OFF_TIME: 1e6,
    Q: { modes: { contactLayout: 'persistent' } }, info: {}, navigator: { userAgent: 'test' }, statusEl: {},
    last: { awake: 110441 }, async stepOne() {}, async drainAll() { events.push('drain'); },
    async readColorBuckets() { assert.equal(measuring, false); events.push('buckets'); return [60000, 50441]; },
    assertBenchmarkWorkload() {},
    async measurePhysicsBatches({ step, drain }) {
      events.push('timing start'); measuring = true; await step(); await drain(); measuring = false;
      events.push('timing end'); return { wallMs: 12 };
    },
  };
  const report = await runInNewContext(`${wallSource}\n(async () => { await benchWall(); return timingSamples; })()`, context);
  assert.deepEqual(events, ['drain', 'buckets', 'timing start', 'drain', 'timing end', 'drain', 'buckets'],
    'bucket diagnostics must not contaminate throughput batches or sample a different physics step');
  assert.deepEqual(plain(report.workload), {
    start: { awake: 110441, colorBuckets: [60000, 50441] }, end: { awake: 110441, colorBuckets: [60000, 50441] },
  });
  assert.equal(report.host, 'World (engine-world-wasm)');
});

// World times each phase of its plan in its own pass. The page sums a phase that
// repeats (each iteration's primal and dual), takes medians over 10 waited steps
// and puts the live timing back afterwards, even when a sample is invalid.
const profileSource = slice('let phaseSamples;', '// Live HUD step cost.');
function profileFixture({ invalid = false } = {}) {
  const modes = [];
  let clock = 0, step = 0;
  const context = {
    wantTimestamps: true, liveTiming: 'step', performance: { now: () => clock }, Error, Object, Set,
    frameTimes: [],
    world: { setGpuTiming(m) { modes.push(m); } },
    async drainAll() {},
    async stepOne() {
      clock += 20;
      step++;
      if (invalid) return;
      context.frameTimes.push({
        step, totalMs: 9, spanMs: 10,
        segments: [['broadphase', 1], ['narrowphase', 2], ['primal', 2], ['dual', 0.5], ['primal', 2], ['dual', 0.5], ['sleep', 1]],
      });
    },
  };
  return { context, modes };
}
test('phase profile sums repeated phases, keeps every sample and restores live timing', async () => {
  const { context, modes } = profileFixture();
  const { summary, report } = await runInNewContext(`${profileSource}\n(async () => ({ summary: await profilePhases(), report: phaseSamples }))()`, context);
  assert.deepEqual(modes, ['phases', 'step']);
  assert.equal(report.samples.length, 10, 'retain individual samples so phase/CPU discrepancies cannot be hidden by medians');
  assert.deepEqual(plain(report.phasesMs), { broadphase: 1, narrowphase: 2, primal: 4, dual: 1, sleep: 1 });
  assert.equal(report.gpuPhasesMs, 9); assert.equal(report.gpuSpanMs, 10); assert.equal(report.completionMs, 20);
  assert.match(report.caveat, /not the batched throughput window/);
  assert.match(summary, /primal 4\.00 · dual 1\.00/);
  assert.match(summary, /profiled GPU span 10\.00 \/ completion 20\.00 ms/);
});
test('a profiled step without phase timestamps is rejected, and live timing comes back', async () => {
  const { context, modes } = profileFixture({ invalid: true });
  await assert.rejects(runInNewContext(`${profileSource}\nprofilePhases()`, context), /no phase timestamps/);
  assert.deepEqual(modes, ['phases', 'step']);
});

// Live HUD step cost. Playback steps once per rAF, so completion intervals are
// display-bound; Max TPS must come from the step's own cost. The numbers mirror
// ring110k on an M3 Max: ~11 ms GPU per step behind a 60 Hz display.
const costSource = slice('// Live HUD step cost.', '// End of the step-cost helpers');
const liveCost = expr => runInNewContext(`${costSource}\n${expr}`, { Error });

test('live step cost uses the step\'s GPU time, not the display-bound completion interval', () => {
  const sample = liveCost(`liveStepSample({gpuMs: 11.1, submittedAt: 1000, completedAt: 1016.7, previousCompletedAt: 1000})`);
  assert.deepEqual(plain(sample), { ms: 11.1, source: 'gpu' });
});

test('without GPU time the live estimate is busy time and is labelled', () => {
  const idle = liveCost(`liveStepSample({gpuMs: 0, submittedAt: 1000, completedAt: 1012.5, previousCompletedAt: 990})`);
  assert.deepEqual(plain(idle), { ms: 12.5, source: 'estimate' });
  const busy = liveCost(`liveStepSample({gpuMs: 0, submittedAt: 1000, completedAt: 1030, previousCompletedAt: 1018})`);
  assert.deepEqual(plain(busy), { ms: 12, source: 'estimate' });
});

test('step cost window reports a median, prefers GPU samples and derives Max TPS from cost', () => {
  const gpu = [11.0, 11.2, 10.9, 25.0, 11.1].map(ms => ({ ms, source: 'gpu' }));
  const mixed = [...gpu, { ms: 16.7, source: 'estimate' }, { ms: 16.7, source: 'estimate' }];
  const result = plain(liveCost(`summarizeStepCost(${JSON.stringify(mixed)})`));
  assert.equal(result.source, 'gpu');
  assert.equal(result.stepMs, 11.1, 'a hitch must not move the readout, and estimates must not mix into GPU time');
  assert.ok(Math.abs(result.maxTps - 1000 / 11.1) < 1e-9);
  assert.deepEqual(plain(liveCost(`stepCostLabels(${JSON.stringify(result)})`)),
    { step: '11.1 ms/step GPU', maxTps: '90', maxTpsSub: 'GPU-timed' });
  const estimate = plain(liveCost(`summarizeStepCost([{ms: 14, source: 'estimate'}, {ms: 13, source: 'estimate'}, {ms: 15, source: 'estimate'}])`));
  assert.deepEqual(estimate, { stepMs: 14, maxTps: 1000 / 14, source: 'estimate' });
  assert.deepEqual(plain(liveCost(`stepCostLabels(${JSON.stringify(estimate)})`)),
    { step: '≈14.0 ms/step est.', maxTps: '≈71', maxTpsSub: 'estimate' });
  assert.deepEqual(plain(liveCost(`stepCostLabels(summarizeStepCost([]))`)), { step: '', maxTps: '—', maxTpsSub: '' });
});

test('live HUD keeps TPS observed, Max TPS from step cost, and shows measuring while benchWall runs', () => {
  const rates = slice('    // Rolling 1 s windows', '    // End of the live rate window');
  const el = () => ({ textContent: '' });
  const dom = { tpsVal: el(), fpsVal: el(), maxTpsVal: el(), maxTpsSub: el(), maxFpsVal: el(), stepMsVal: el(), frameMsVal: el() };
  let clock = 0;
  const context = { ...dom, performance: { now: () => clock }, window: {}, Error, measuring: false };
  const hud = runInNewContext(`${costSource}\n${rates}\n({bumpRates, addStepCost, showMeasuring, resetRates, set measuring(v) { measuring = v; }})`, context);
  const playSecond = () => {
    for (let frame = 0; frame < 60; frame++) {
      clock += 16.7;
      hud.addStepCost({ ms: 11.1, source: 'gpu' });
      hud.bumpRates(true, 2);
    }
  };
  playSecond();
  assert.equal(dom.tpsVal.textContent, '60', 'TPS is the observed, display-bound step rate');
  assert.equal(dom.maxTpsVal.textContent, '90', 'Max TPS is 1000 / GPU step cost, not 1000 / 16.7 ms');
  assert.equal(dom.maxTpsSub.textContent, 'GPU-timed');
  assert.equal(dom.stepMsVal.textContent, '11.1 ms/step GPU');
  assert.equal(context.window.__RATES__.stepSource, 'gpu');

  hud.measuring = true;
  hud.showMeasuring();
  for (const id of ['tpsVal', 'maxTpsVal', 'fpsVal', 'maxFpsVal']) assert.equal(dom[id].textContent, '—');
  assert.equal(dom.maxTpsSub.textContent, 'measuring…');
  // Late live frames and benchWall's saturated queue must not reach the HUD.
  hud.addStepCost({ ms: 45, source: 'estimate' });
  hud.measuring = false;
  hud.resetRates();
  playSecond();
  assert.equal(dom.stepMsVal.textContent, '11.1 ms/step GPU');
  assert.equal(dom.maxTpsVal.textContent, '90');
});
