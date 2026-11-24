#!/usr/bin/env node
// Before/after of two profiles, for proving an optimisation:
//
//   node tools/webgpu-profile/diff.mjs out/webgpu-profile/A/summary.json out/webgpu-profile/B/summary.json
//   node tools/webgpu-profile/diff.mjs A.json B.json --json        # machine-readable
//
// Takes the browser harness's summary.json (profile-webgpu.mjs) or a native report
// (examples/webgpu_profile.rs --json). Every metric is a median; a row is a change only
// past the noise threshold the scenario matrix uses (5 % and 0.05 ms, both), otherwise
// "same". The verdict: "faster" if something improved and nothing got slower, "slower"
// if anything got slower, else "same". Exit 0 always (the verdict is in the output).
import { readFileSync } from 'node:fs';

const [a, b] = process.argv.slice(2).filter((x) => !x.startsWith('--'));
if (!a || !b) { console.error('usage: diff.mjs BEFORE.json AFTER.json [--json] [--rel 0.05] [--abs 0.05]'); process.exit(2); }
const opt = (k, d) => { const i = process.argv.indexOf(k); return i >= 0 ? Number(process.argv[i + 1]) : d; };
const REL = opt('--rel', 0.05), ABS = opt('--abs', 0.05);

/** Flatten a profile into `metric -> ms` (medians). */
function metrics(p) {
  const m = new Map();
  const s = p.summary;
  if (s) {
    m.set('frame.dt', s.frameDt?.median);
    m.set('frame.js', s.js?.median);
    m.set('frame.js_other', s.jsOther?.median);
    for (const [k, v] of Object.entries(s.host || {})) m.set(`host.${k}`, v.median);
    for (const [k, v] of Object.entries(s.page || {})) m.set(`page.${k}`, v.median);
    m.set('gpu.step', s.gpuStep?.median);
    for (const [k, v] of Object.entries(s.gpuSegments || {})) m.set(`gpu.segment.${k}`, v.median);
  }
  for (const [k, v] of Object.entries(p.native || {})) if (typeof v === 'number') m.set(`native.${k}`, v);
  const r = p.replay;
  if (r && !r.error) {
    m.set('replay.whole_step', r.whole_step_ms?.median);
    m.set('replay.phases_sum', r.phases_sum_ms?.median);
    m.set('replay.isolated_sum', r.isolated_sum_ms);
    m.set('replay.empty_pass', r.empty_pass_ms?.median);
    // Dispatches keyed by phase/kernel and their order within it (plans may differ in length).
    const seen = new Map();
    for (const d of r.dispatches || []) {
      const base = `${d.phase}/${d.kernel}`; const n = (seen.get(base) || 0) + 1; seen.set(base, n);
      m.set(`replay.kernel.${base}#${n}`, d.isolated_ms?.median);
    }
  }
  for (const [k, v] of m) if (v == null || Number.isNaN(v)) m.delete(k);
  return m;
}

const A = metrics(JSON.parse(readFileSync(a, 'utf8')));
const B = metrics(JSON.parse(readFileSync(b, 'utf8')));
const rows = [];
for (const k of new Set([...A.keys(), ...B.keys()])) {
  const x = A.get(k), y = B.get(k);
  if (x == null || y == null) { rows.push({ metric: k, before: x ?? null, after: y ?? null, delta: null, pct: null, verdict: x == null ? 'new' : 'gone' }); continue; }
  const d = y - x, pct = x > 0 ? d / x : 0;
  const changed = Math.abs(d) > ABS && Math.abs(pct) > REL;
  rows.push({ metric: k, before: x, after: y, delta: d, pct, verdict: changed ? (d < 0 ? 'faster' : 'slower') : 'same' });
}
// Frame interval is the display's: a shorter one is not a speed-up of our work, but a longer
// one is a regression (frames missed). Timing keys only count where we own the work.
const owned = rows.filter((r) => r.metric !== 'frame.dt' || r.verdict === 'slower');
const slower = owned.filter((r) => r.verdict === 'slower');
const faster = owned.filter((r) => r.verdict === 'faster');
const verdict = slower.length ? 'slower' : faster.length ? 'faster' : 'same';
if (process.argv.includes('--json')) {
  console.log(JSON.stringify({ before: a, after: b, threshold: { rel: REL, abs: ABS }, verdict, rows }, null, 1));
} else {
  const f = (x) => (x == null ? '—' : x.toFixed(3));
  console.log(`before ${a}\nafter  ${b}\nthreshold: ${(REL * 100).toFixed(0)} % and ${ABS} ms\n`);
  console.log(`${'metric'.padEnd(56)} ${'before'.padStart(9)} ${'after'.padStart(9)} ${'delta'.padStart(9)} ${'%'.padStart(7)}  verdict`);
  const order = { slower: 0, faster: 1, new: 2, gone: 3, same: 4 };
  for (const r of rows.sort((p, q) => order[p.verdict] - order[q.verdict] || Math.abs(q.delta ?? 0) - Math.abs(p.delta ?? 0))) {
    console.log(`${r.metric.slice(0, 56).padEnd(56)} ${f(r.before).padStart(9)} ${f(r.after).padStart(9)} ${f(r.delta).padStart(9)} ${r.pct == null ? '      —' : `${(100 * r.pct).toFixed(1).padStart(6)}%`}  ${r.verdict}`);
  }
  console.log(`\nverdict: ${verdict} (${faster.length} faster, ${slower.length} slower, threshold ${(REL * 100).toFixed(0)} % and ${ABS} ms)`);
}
