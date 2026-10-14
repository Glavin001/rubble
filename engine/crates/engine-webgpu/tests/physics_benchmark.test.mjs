import { test } from 'node:test';
import assert from 'node:assert/strict';
import { measurePhysicsBatches, assertBenchmarkWorkload } from '../physics-benchmark.mjs';

test('full-scale impact timings reject missing work with a diagnostic instead of reporting a false speedup', () => {
  for (const [scene, totalBodies, iterations] of [['ring110k', 110442, 4], ['gables506k', 505923, 3]]) {
    const valid = { scene, totalBodies, iterations, dt: 1 / 60, stage: 'after timed window',
      counters: { awake: totalBodies - 1, contact: 1_170_000, manifold: 215_000, color: 11, overflow: 0 } };
    assert.doesNotThrow(() => assertBenchmarkWorkload(valid));
    for (const [patch, diagnostic] of [
      [{totalBodies: totalBodies - 1}, /total bodies/],
      [{iterations: iterations - 1}, /iterations/],
      [{dt: 1 / 120}, /dt=1\/60/],
      [{dt: NaN}, /dt=1\/60/],
      [{counters: {...valid.counters, awake: totalBodies - 2}}, /sleeping or omitted bodies/],
      [{counters: {...valid.counters, overflow: 1}}, /constraints may have been discarded/],
      [{counters: {...valid.counters, contact: 0}}, /invalid contact count/],
      [{counters: {...valid.counters, manifold: undefined}}, /invalid manifold count/],
      [{counters: {...valid.counters, color: NaN}}, /invalid color count/],
    ]) {
      assert.throws(() => assertBenchmarkWorkload({...valid, ...patch}), error => {
        assert.match(error.message, new RegExp(`${scene} after timed window`));
        assert.match(error.message, diagnostic);
        return true;
      });
    }
  }
  assert.doesNotThrow(() => assertBenchmarkWorkload({ scene:'columns', stage:'settled', counters:{awake:0,contact:0,overflow:0} }),
    'The full-active rule must not reject intentionally settled/sleeping scenes.');
});

test('physics timing drains actual GPU work and matches the pinned 2 × 5 × 10 protocol', async () => {
  let time = 0, pending = 7, steps = 0;
  const batches = [], clockReads = [];
  const result = await measurePhysicsBatches({
    step() { pending++; steps++; time += 1; }, // CPU encode cost is included
    async drain() {
      const work = pending;
      await Promise.resolve(); // GPU completion must be awaited before reading time
      batches.push(work);
      time += work * 2;
      pending = 0;
    },
    now() {
      assert.equal(pending, 0, 'Timing read before GPU completion would measure submission speed rather than physics.');
      clockReads.push(time);
      return time;
    },
  });
  assert.deepEqual(batches, [7, 3, 10, 10, 10, 10, 10, 3, 10, 10, 10, 10, 10],
    'Exclude existing work and the three warm-up steps; never include rendering or readbacks inside timed batches.');
  assert.equal(steps, 106, 'Do not change the moving ring trajectory/window relative to pinned three-avbd.');
  assert.equal(clockReads.length, 20);
  assert.equal(result.wallMs, 3, 'Wall time must include CPU encoding plus completed GPU work, divided by ten steps.');
  assert.equal(result.allBatchesWithin120Hz, true);
});

test('raw samples reveal slow batches even when the best median meets 120 Hz', async () => {
  const samples = [6, 8, 7, 9, 10, 4, 5, 6, 7, 30];
  let calls = 0, time = 0;
  const result = await measurePhysicsBatches({
    step() {}, async drain() {},
    now() { if (calls++ % 2) time += samples[(calls >> 1) - 1] * 10; return time; },
  });
  assert.deepEqual(result.rounds.map(r => r.samplesMs), [samples.slice(0, 5), samples.slice(5)]);
  assert.deepEqual(result.rounds.map(r => r.medianMs), [8, 6]);
  assert.equal(result.wallMs, 6, 'Retain pinned best-of-two median semantics for fair comparisons.');
  assert.equal(result.allBatchesWithin120Hz, false, 'A good median must not hide stalls or imply sustained 120 Hz.');
});

test('broken clocks and GPU failures cannot produce successful benchmark reports', async () => {
  for (const value of [0, NaN, Infinity, -1]) {
    let calls = 0;
    await assert.rejects(measurePhysicsBatches({ step() {}, async drain() {}, now: () => calls++ ? value : 0 }),
      /round 1, batch 1: invalid/, `Invalid clock ${value} needs an actionable diagnostic.`);
  }
  await assert.rejects(measurePhysicsBatches({ step() {}, async drain() { throw new Error('device lost'); } }), /device lost/);
});
