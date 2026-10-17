import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';

// Execute the shipped profiler. Delayed fake queue/readback completion and
// timestamps with inter-pass gaps distinguish phase work, GPU span and CPU time.
const html = readFileSync(new URL('../demo.html', import.meta.url), 'utf8');
const start = html.indexOf('let phaseSamples;');
const end = html.indexOf('async function sampleSettle()', start);
assert.ok(start >= 0 && end > start, 'demo profiler entry points must remain testable');
const source = html.slice(start, end);
// The encoder's dispatch-count constants (8ev), read from the shipped page, with the
// fixed-palette controls on (colors=full, rounds=full): these tests check the full
// schedule; adaptive counts are exercised by the browser regressions.
const dispatchSource = html.slice(html.indexOf('const COLOR_ROUND_LIMIT'), html.indexOf('const colorsFull'));
const DISPATCH = {
  ...runInNewContext(`${dispatchSource}({COLOR_ROUND_LIMIT, COLOR_ROUND_HEADROOM, COLOR_HEADROOM, SLEEP_NEVER_S})`),
  colorsFull: true, roundsFull: true,
};
const plain = value => JSON.parse(JSON.stringify(value));

const bucketSource = html.slice(html.indexOf('async function readColorBuckets()'), html.indexOf('function restoreScene()'));
for (const scenario of ['valid', 'coverage mismatch', 'device loss']) {
  test(`color-bucket diagnostics validate GPU coverage and release readback: ${scenario}`, async () => {
    const counts = new Uint32Array([19, 0, 7, 5]);
    const state = { mapped: false, destroyed: 0, copied: false, submitted: false };
    const buffers = Array.from({length: 18}, () => ({}));
    const readback = {
      async mapAsync(mode) {
        assert.equal(mode, 1); assert.ok(state.submitted);
        if (scenario === 'device loss') throw new Error('simulated bucket readback device loss');
        state.mapped = true;
      },
      getMappedRange() { assert.ok(state.mapped); return counts.buffer; },
      unmap() { state.mapped = false; },
      destroy() { state.destroyed++; },
    };
    const encoder = {
      copyBufferToBuffer(src, srcOffset, dst, dstOffset, bytes) {
        assert.equal(src, buffers[17], 'read the solver color counts, not stale counters or history');
        assert.equal(dst, readback); assert.equal(srcOffset, 0); assert.equal(dstOffset, 0);
        assert.equal(bytes, 16, 'copy the entire configured palette, including empty buckets');
        state.copied = true;
      },
      finish() { assert.ok(state.copied); return this; },
    };
    const context = {...DISPATCH,
      cap: {maxColors: 4}, last: {awake: scenario === 'coverage mismatch' ? 32 : 31}, bufs: buffers,
      GPUBufferUsage: {COPY_DST: 2, MAP_READ: 4}, GPUMapMode: {READ: 1}, Uint32Array,
      device: {
        createBuffer(options) { assert.deepEqual(plain(options), {size: 16, usage: 6}); return readback; },
        createCommandEncoder() { return encoder; },
        queue: {submit(commands) { assert.equal(commands[0], encoder); state.submitted = true; }},
      },
    };
    const result = runInNewContext(`${bucketSource}\nreadColorBuckets()`, context);
    if (scenario === 'valid') assert.deepEqual(plain(await result), [19, 0, 7, 5]);
    else await assert.rejects(result, scenario === 'device loss'
      ? /simulated bucket readback device loss/
      : /Color buckets cover 31 bodies, but awake count is 32/);
    assert.equal(state.destroyed, 1, 'success or failure must release diagnostic GPU allocation');
    assert.equal(state.mapped, false);
  });
}

test('benchmark captures color buckets before and after timing without readbacks inside the measured interval', async () => {
  const wallSource = html.slice(html.indexOf('let timingSamples;'), html.indexOf('let phaseSamples;'));
  const events = [];
  let measuring = false;
  const context = {...DISPATCH,
    lead: 0, scene: 'ring110k', bodyCount: 110442,
    params: {iterations: 4, dt: 1/60}, SLEEP_OFF_TIME: 1e6, stablePacking: true, persistentContacts: true,
    compactContacts: true, sparseStart: 32, parallelColorPrefix:false, physicalAdjacency: true, spatialOrder: false,
    fusedPersistence: false, manifoldPersistence: false, hashInsertMode: 'contact', reuseMode: 'pass', npMode: 'fused', persistMode: 'scan', repairMode: 'rounds', restMode: 'drop', endpointsMode: 'manifold', gridMode: 'fixed', info: {}, navigator: {userAgent: 'test'},
    last: {awake: 110441}, submitPhysics() {}, device: {queue: {async onSubmittedWorkDone() {}}},
    async readCounters() { assert.equal(measuring, false); events.push('counters'); },
    async readColorBuckets() { assert.equal(measuring, false); events.push('buckets'); return [60000,50441]; },
    assertBenchmarkWorkload() {},
    async measurePhysicsBatches({step, drain}) {
      events.push('timing start'); measuring = true; step(); await drain(); measuring = false;
      events.push('timing end'); return {wallMs: 12};
    },
  };
  const report = await runInNewContext(`${wallSource}\n(async () => { await benchWall(); return timingSamples; })()`, context);
  assert.deepEqual(events, ['counters','buckets','timing start','timing end','counters','buckets'],
    'bucket diagnostics must not contaminate throughput batches or sample a different physics step');
  assert.deepEqual(plain(report.workload), {
    start: {awake:110441, colorBuckets:[60000,50441]}, end: {awake:110441, colorBuckets:[60000,50441]},
  });
});

function fixture({ graphProfile = false, detailedProfile = false, reuse = true, compactContacts = false, passDelta = 0, corruptTimestamp = false, rejectMap = false } = {}) {
  const iterations = 4, phases = 6 + iterations * 2 + (graphProfile ? 8 : 0) + (detailedProfile ? (reuse ? 3 : 2) : 0);
  const state = {encodes: 0, submissions: 0, passes: 0, destroyed: 0, mapped: false, drained: false};
  let clock = 0;
  const stamps = new BigUint64Array(phases * 2);
  for (let i = 0; i < phases; i++) {
    stamps[i * 2] = BigInt(i * 2) * 1_000_000n;
    stamps[i * 2 + 1] = stamps[i * 2] + 1_000_000n;
  }
  if (corruptTimestamp) stamps[3] = 0n;
  const encoder = {
    beginComputePass({timestampWrites}) {
      const index = state.passes % (phases + passDelta);
      assert.equal(timestampWrites.beginningOfPassWriteIndex, index * 2);
      assert.equal(timestampWrites.endOfPassWriteIndex, index * 2 + 1);
      state.passes++;
      return {end() {}};
    },
    resolveQuerySet(_q, first, count) { assert.equal(first, 0); assert.equal(count, stamps.length); },
    copyBufferToBuffer() {}, finish() { return this; },
  };
  const context = {...DISPATCH,
    params: {iterations, flags: reuse ? 1 : 0}, detailedProfile, graphProfile, colorProfile: false, compactContacts, parity: 0, steps: 166, wantTimestamps: true,
    performance: {now() { return clock; }},
    GPUBufferUsage: {QUERY_RESOLVE: 1, COPY_SRC: 2, MAP_READ: 4, COPY_DST: 8},
    GPUMapMode: {READ: 1}, BigUint64Array,
    device: {
      createQuerySet({count}) { assert.equal(count, stamps.length); return {destroy() { state.destroyed++; }}; },
      createBuffer() { return {
        async mapAsync() {
          await Promise.resolve();
          clock += 40; // Includes queue execution + readback delivery, not just phases.
          if (rejectMap) throw new Error('simulated device loss during profiler readback');
          state.mapped = true;
        },
        getMappedRange() { assert.ok(state.mapped); return stamps.buffer; },
        unmap() { state.mapped = false; }, destroy() { state.destroyed++; },
      }; },
      createCommandEncoder() {
        assert.ok(state.drained, 'profiling must drain earlier GPU work before starting its CPU interval');
        state.drained = false;
        return encoder;
      },
      queue: {
        async onSubmittedWorkDone() { await Promise.resolve(); clock += 100; state.drained = true; },
        submit(commands) { assert.equal(commands.length, 1); assert.equal(commands[0], encoder); state.submissions++; },
      },
    },
    encodeStep(enc, render, begin) {
      assert.equal(enc, encoder); assert.equal(render, false); state.encodes++; clock += 3;
      for (let i = 0; i < phases + passDelta; i++) begin().end();
    },
    dispWork() { throw new Error('Timestamp reduction dispatched physics after submission; compact preparation belongs inside encodeStep.'); },
  };
  return {context, state};
}
for (const compactContacts of [false, true]) {
  test(`profiler pairs GPU phases with the same submission and excludes prior queued work (compact=${compactContacts})`, async () => {
    const {context, state} = fixture({compactContacts});
    const result = await runInNewContext(`${source}\nprofileOneStep()`, context);
    assert.deepEqual(plain(result), {
      step:167, passCount:14,
      phasesMs:{aabbs:1,grid:1,narrowphase:1,warm:1,graph:1,primal:4,dual:4,sleep:1},
      gpuPhasesMs:14, gpuSpanMs:27, encodeMs:3, submitToReadbackMs:40, completionMs:43,
    }, 'phase sum must exclude gaps; GPU span must include them; completion must include encoding/readback but exclude the preceding 100 ms drain');
    assert.equal(state.encodes,1); assert.equal(state.submissions,1); assert.equal(state.passes,14);
    assert.equal(context.steps,167); assert.equal(context.parity,1);
    assert.equal(state.destroyed,3,'query, resolve and readback resources must be released');
  });
}
test('profile report retains all paired raw samples and explicit perturbation caveat', async () => {
  const {context, state} = fixture();
  const {summary, report} = await runInNewContext(`${source}\n(async () => ({summary: await profilePhases(), report: phaseSamples}))()`, context);
  assert.equal(report.samples.length,10,'retain individual samples so phase/CPU discrepancies cannot be hidden by independently aggregated medians');
  assert.deepEqual(Array.from(report.samples, s => s.step), [167,168,169,170,171,172,173,174,175,176]);
  assert.equal(report.gpuPhasesMs,14); assert.equal(report.gpuSpanMs,27); assert.equal(report.completionMs,43);
  assert.match(report.caveat,/not the batched throughput window/);
  assert.match(summary,/profiled GPU span 27.00 \/ completion 43.00 ms/);
  assert.equal(state.submissions,10); assert.equal(state.destroyed,30);
});
for (const failure of [
  {options:{passDelta:-1}, message:/expected 14 passes, encoded 13/, submissions:0},
  {options:{corruptTimestamp:true}, message:/timestamp 3 precedes 2/, submissions:1},
  {options:{rejectMap:true}, message:/simulated device loss/, submissions:1},
]) {
  test(`profiler rejects invalid evidence and releases resources: ${failure.message}`, async () => {
    const {context, state} = fixture(failure.options);
    await assert.rejects(runInNewContext(`${source}\nprofileOneStep()`, context),failure.message);
    assert.equal(state.submissions,failure.submissions);
    assert.equal(state.destroyed,3,'a failed profile must not leak GPU resources into later benchmarks');
  });
}

for (const reuse of [false,true]) {
  test(`detailed collision phases retain correct primal/dual offsets (reuse=${reuse})`,async()=>{
    const {context,state}=fixture({detailedProfile:true,reuse});
    const r=await runInNewContext(`${source}\nprofileOneStep()`,context);
    assert.deepEqual(plain(r.phasesMs),{aabbs:1,grid:1,...(reuse?{refs:1}:{}),hashClear:1,hashInsert:1,narrowphase:1,warm:1,graph:1,primal:4,dual:4,sleep:1});
    assert.equal(r.passCount,reuse?17:16);
    assert.equal(r.gpuPhasesMs,reuse?17:16);
    assert.equal(r.gpuSpanMs,reuse?33:31);
    assert.equal(state.submissions,1); assert.equal(state.destroyed,3);
  });
}

for (const reuse of [false, true]) {
  test(`graph profiling preserves solver offsets and accounts for each construction stage (reuse=${reuse})`, async () => {
    const {context, state} = fixture({graphProfile:true, reuse});
    const {summary, report} = await runInNewContext(`${source}\n(async () => ({summary:await profilePhases(), report:phaseSamples}))()`, context);
    assert.equal(report.detail, 'graph');
    assert.deepEqual(plain(report.phasesMs), {
      aabbs:1, grid:1, narrowphase:1, warm:1,
      awake:1, csrDegree:1, csrPrefix:1, csrScatter:1, csrSort:1,
      colorInit:1, colorRepair:1, colorPack:1, predict:1,
      primal:4, dual:4, sleep:1,
    }, 'graph detail must not charge graph passes to primal or lose any construction work');
    assert.ok(report.samples.every(s => s.passCount === 22));
    assert.equal(report.gpuPhasesMs, 22); assert.equal(report.gpuSpanMs, 43);
    assert.match(summary, /csrSort 1.00/); assert.match(report.caveat, /not the batched throughput window/);
    assert.equal(state.submissions, 10); assert.equal(state.destroyed, 30);
  });
}

// Live HUD step cost. Playback steps once per rAF, so completion intervals are
// display-bound; Max TPS must come from the step's own cost. The numbers mirror
// ring110k on an M3 Max: ~11 ms GPU per step behind a 60 Hz display.
const liveCost = expr => runInNewContext(`${source}\n${expr}`, {BigUint64Array, Error});
const ns = ms => BigInt(Math.round(ms * 1e6));

test('live step cost uses the step pass timestamps, not the display-bound completion interval', () => {
  // Completions 16.7 ms apart, as at 60 Hz: the old HUD reported exactly that.
  const sample = liveCost(`liveStepSample({stamps: new BigUint64Array([${ns(500)}n, ${ns(511.1)}n]), submittedAt: 1000, completedAt: 1016.7, previousCompletedAt: 1000})`);
  assert.equal(sample.source, 'gpu');
  assert.ok(Math.abs(sample.ms - 11.1) < 1e-9);
});

test('without valid timestamps the live estimate is submission busy time and is labelled', () => {
  // Idle queue: the GPU could not start before submit, so busy = completion - submit.
  const idle = liveCost(`liveStepSample({stamps: null, submittedAt: 1000, completedAt: 1012.5, previousCompletedAt: 990})`);
  assert.deepEqual(plain(idle), {ms: 12.5, source: 'estimate'});
  // Saturated queue: this step waited for the previous one, so the interval counts.
  const busy = liveCost(`liveStepSample({stamps: null, submittedAt: 1000, completedAt: 1030, previousCompletedAt: 1018})`);
  assert.deepEqual(plain(busy), {ms: 12, source: 'estimate'});
  // Zeroed or reversed timestamp pairs are not GPU evidence.
  for (const pair of ['0n, 0n', `${ns(12)}n, ${ns(11)}n`]) {
    const bad = liveCost(`liveStepSample({stamps: new BigUint64Array([${pair}]), submittedAt: 1000, completedAt: 1013, previousCompletedAt: 0})`);
    assert.equal(bad.source, 'estimate');
  }
});

test('step cost window reports a median, prefers GPU samples and derives Max TPS from cost', () => {
  const gpu = [11.0, 11.2, 10.9, 25.0, 11.1].map(ms => ({ms, source: 'gpu'}));
  const mixed = [...gpu, {ms: 16.7, source: 'estimate'}, {ms: 16.7, source: 'estimate'}];
  const result = plain(liveCost(`summarizeStepCost(${JSON.stringify(mixed)})`));
  assert.equal(result.source, 'gpu');
  assert.equal(result.stepMs, 11.1, 'a hitch must not move the readout, and estimates must not mix into GPU time');
  assert.ok(Math.abs(result.maxTps - 1000 / 11.1) < 1e-9);
  assert.deepEqual(plain(liveCost(`stepCostLabels(${JSON.stringify(result)})`)),
    {step: '11.1 ms/step GPU', maxTps: '90', maxTpsSub: 'GPU-timed'});
  const estimate = plain(liveCost(`summarizeStepCost([{ms: 14, source: 'estimate'}, {ms: 13, source: 'estimate'}, {ms: 15, source: 'estimate'}])`));
  assert.deepEqual(estimate, {stepMs: 14, maxTps: 1000 / 14, source: 'estimate'});
  assert.deepEqual(plain(liveCost(`stepCostLabels(${JSON.stringify(estimate)})`)),
    {step: '≈14.0 ms/step est.', maxTps: '≈71', maxTpsSub: 'estimate'});
  assert.deepEqual(plain(liveCost(`stepCostLabels(summarizeStepCost([]))`)), {step: '', maxTps: '—', maxTpsSub: ''});
});

test('live HUD keeps TPS observed, Max TPS from step cost, and shows measuring while benchWall runs', () => {
  const rates = html.slice(html.indexOf('    // Rolling 1 s windows'), html.indexOf('    // Live playback pipelines counter readback'));
  const el = () => ({textContent: ''});
  const dom = {tpsVal: el(), fpsVal: el(), maxTpsVal: el(), maxTpsSub: el(), maxFpsVal: el(), stepMsVal: el(), frameMsVal: el()};
  let clock = 0;
  const context = {...dom, performance: {now: () => clock}, window: {}, BigUint64Array, Error, measuring: false};
  const hud = runInNewContext(`${source}\n${rates}\n({bumpRates, addStepCost, showMeasuring, resetRates, set measuring(v) { measuring = v; }})`, context);
  const playSecond = () => {
    for (let frame = 0; frame < 60; frame++) {
      clock += 16.7;
      hud.addStepCost({ms: 11.1, source: 'gpu'});
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
  // Late live completions and benchWall's saturated queue must not reach the HUD.
  hud.addStepCost({ms: 45, source: 'estimate'});
  hud.measuring = false;
  hud.resetRates();
  playSecond();
  assert.equal(dom.stepMsVal.textContent, '11.1 ms/step GPU');
  assert.equal(dom.maxTpsVal.textContent, '90');
});

test('live readback decodes the step pass timestamps after the counters', () => {
  const block = html.slice(html.indexOf('const LIVE_IN_FLIGHT'), html.indexOf('async function frame('));
  const bytes = new ArrayBuffer(96);
  new Uint32Array(bytes, 0, 20)[5] = 110441;
  new BigUint64Array(bytes, 80, 2).set([ns(500), ns(511.1)]);
  const sizes = [];
  const device = {
    createBuffer({size}) { sizes.push(size); return {getMappedRange: () => bytes.slice(0), unmap() {}}; },
    createQuerySet({type, count}) { assert.equal(type, 'timestamp'); assert.equal(count, 2); return {}; },
  };
  const ctx = runInNewContext(`${block}\n({liveFree, readLiveCounters, get last() { return last; }})`, {
    device, wantTimestamps: true, last: null, Promise, Set, Array, Uint32Array, BigUint64Array, Error,
    GPUBufferUsage: {MAP_READ: 1, COPY_DST: 8, QUERY_RESOLVE: 16, COPY_SRC: 4},
  });
  assert.deepEqual(sizes, [16, 96, 96], 'one shared resolve buffer; each readback holds counters + two timestamps');
  const stamps = ctx.readLiveCounters(ctx.liveFree[0]);
  assert.equal(ctx.last.awake, 110441);
  assert.deepEqual(Array.from(stamps), [ns(500), ns(511.1)]);
  assert.ok(Math.abs(liveCost(`liveStepSample({stamps: new BigUint64Array([${stamps[0]}n, ${stamps[1]}n])}).ms`) - 11.1) < 1e-9);
});
