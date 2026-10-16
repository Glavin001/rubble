// Collider-pair cost benchmark (engine/docs/collider-cost-model.md).
// Runs headless.mjs scene=pairs for each shape combination at two counts and
// reports the marginal GPU cost of one live pair (slope, ns per tick), by stage.
//   node shape-cost.mjs . out.json [reuse=0|1]   (COMBOS=box/box,hull8/hull8 to pick)
import {execFileSync} from 'node:child_process';
import {writeFileSync} from 'node:fs';
const dir = process.argv[2];
const out = process.argv[3];
const reuse = process.argv[4] ?? '0';
const combos = process.env.COMBOS ? process.env.COMBOS.split(',').map(c => c.split('/')) : [
  ['box', 'box'], ['box', 'sphere'], ['box', 'capsule'], ['hull4', 'sphere'], ['hull16', 'sphere'], ['hull16', 'capsule'],
  ['box', 'hull4'], ['hull4', 'hull4'], ['box', 'hull8'], ['hull8', 'hull8'], ['box', 'hull16'], ['hull16', 'hull16'],
  ['box', 'hull32'], ['hull32', 'hull32'], ['box', 'box4'], ['hull4', 'box4'],
];
const counts = [8192, 32768];
const stage = (k) =>
  /^(build_aabbs|broadphase|write_moved|rest_|write_finer)/.test(k) ? 'broadphase' :
  /^(narrowphase|convex_|contact_hash|persist_contacts|update_refs|reuse_all)/.test(k) ? 'narrowphase' :
  /^(solve_|pack_solver|predict_bodies|extract_velocity)/.test(k) ? 'solver' :
  /^(csr_|color_|init_colors|count_color|prefix_color|scatter_color|record_color|write_color|build_awake)/.test(k) ? 'graph' :
  /^(sleep_|write_sleep)/.test(k) ? 'sleep' : 'other';
const rows = [];
for (const [a, b] of combos) {
  const runs = [];
  for (const count of counts) {
    const txt = execFileSync('node', ['headless.mjs', 'pairs', `a=${a}`, `b=${b}`, `count=${count}`, 'warm=240', 'measure=120', `reuse=${reuse}`], {cwd: dir, encoding: 'utf8', maxBuffer: 1 << 26});
    const r = JSON.parse(txt);
    if (r.error || r.gpuError) throw new Error(`${a}/${b}: ${r.error ?? r.gpuError}`);
    const st = {};
    for (const [k, v] of Object.entries(r.kernels ?? {})) st[stage(k)] = (st[stage(k)] ?? 0) + v;
    runs.push({count, ms: r.msPerTick, st, c: r.counters});
  }
  const [lo, hi] = runs, dn = hi.count - lo.count;
  const per = (x, y) => +(((y - x) / dn) * 1e6).toFixed(1); // ns per pair
  const row = {pair: `${a}/${b}`, contactsPerPair: +(hi.c.contact / hi.count).toFixed(2), manifoldsPerPair: +(hi.c.manifold / hi.count).toFixed(2),
    awake: hi.c.awake === hi.count, totalNs: per(lo.ms, hi.ms)};
  for (const s of ['broadphase', 'narrowphase', 'graph', 'solver', 'sleep', 'other']) row[s + 'Ns'] = per(lo.st[s] ?? 0, hi.st[s] ?? 0);
  rows.push(row);
  console.log(JSON.stringify(row));
  writeFileSync(out, JSON.stringify(rows, null, 2));
}
