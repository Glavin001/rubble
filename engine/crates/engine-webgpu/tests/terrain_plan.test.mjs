// terrain-host.mjs executes engine_terrain::plan through the wasm TerrainStep: the
// words it decodes must be the plan's blocks, and the region's lifecycle (open,
// coupled frames, the counters' check, settling, bake) is Rust's. No GPU here.
// Run `cargo run -p engine-webgpu --bin wgsl_export` first (writes generated/).
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { wasm } from '../vehicle.mjs';

const { TerrainStep } = wasm;
const KERNELS = JSON.parse(TerrainStep.kernels());
const T = JSON.parse(TerrainStep.constants());

// The decoder terrain-host.mjs uses (it imports world-host.mjs, which needs a
// browser; keep the two in step).
function decodeBlocks(words, at = 0) {
  const blocks = [];
  const n = words[at++];
  for (let b = 0; b < n; b++) {
    const nw = words[at++], nc = words[at++];
    const writes = [], cmds = [];
    for (let i = 0; i < nw; i++) {
      const len = words[at + 2];
      writes.push({ binding: words[at], offset: words[at + 1], words: words.slice(at + 3, at + 3 + len) });
      at += 3 + len;
    }
    for (let i = 0; i < nc; i++, at += 3) cmds.push({ op: words[at], kernel: KERNELS[words[at + 1]], value: words[at + 2] });
    blocks.push({ writes, cmds });
  }
  return { blocks, end: at };
}

const SETTINGS = { config: { max_particles: 1 << 20, max_blocks: 8192, max_parcels: 1 << 16 }, region_margin: 0.6, settle_speed: 0.05, settle_frames: 2 };
const FIELD = Float32Array.of(193, 193, 1.5, 1.5, 0, 0, 0);
const counters = (o = {}) => {
  const c = new Uint32Array(T.COUNTER_WORDS);
  const f = new Float32Array(c.buffer);
  c[3] = o.particles ?? 1000;
  f[2] = o.speed ?? 0;
  if (o.flags) c[T.C_FLAGS] = o.flags;
  return c;
};

test('every kernel the plan names has a WGSL module', () => {
  for (const k of KERNELS) assert.ok(existsSync(new URL(`../generated/wgsl/${k}.wgsl`, import.meta.url)), k);
});

test('an impact opens a region (activation, then source) and runs coupled frames until it settles', () => {
  const p = new TerrainStep(JSON.stringify(SETTINGS));
  const pred = JSON.parse(p.predict_impact(2, 3000, 20000, 9.81));
  assert.ok(pred.crater.radius > 30 && pred.crater.radius < 70, `R = ${pred.crater.radius}`);
  const { blocks, end } = decodeBlocks(p.impact(0, FIELD, Float32Array.of(0, 0, 0), 2, 3000, 20000, 9.81));
  assert.equal(blocks.length, 2);
  // The default ground is absorbing: activation also lays out its boundary nodes.
  assert.deepEqual(blocks[0].cmds.map((c) => c.kernel), ['mpm_activate_count', 'mpm_activate_scan', 'mpm_activate_emit', 'mpm_activate_lower', 'mpm_absorb_init']);
  assert.equal(blocks[0].writes[0].binding, T.PARAMS);
  assert.deepEqual(blocks[1].cmds.map((c) => c.kernel), ['mpm_write_args', 'mpm_source']);
  assert.ok(end > 0);
  const region = JSON.parse(p.region());
  assert.ok(Math.abs(region.dx - pred.dx) < 1e-6);

  // Coupled frames: [kind 1, step, blocks…] with the exchange inside each substep.
  const f0 = p.frame(1 / 60, -9.81, 1);
  assert.equal(f0[0], 1);
  const frame = decodeBlocks(f0, 2).blocks[0];
  const names = frame.cmds.map((c) => c.kernel);
  const subs = names.filter((k) => k === 'mpm_p2g').length;
  assert.ok(subs >= 1);
  assert.equal(names.filter((k) => k === 'mpm_couple_apply').length, subs);
  const g = names.indexOf('mpm_grid');
  assert.deepEqual(names.slice(g, g + 4), ['mpm_grid', 'mpm_couple_gather', 'mpm_couple_apply', 'mpm_g2p']);
  assert.deepEqual(names.slice(-2), ['mpm_couple_emit', 'mpm_couple_finish']);
  // The source frame needs more substeps than a calm one.
  JSON.parse(p.report(f0[1], counters({ speed: 0.01 })));
  const calm = decodeBlocks(p.frame(1 / 60, -9.81, 1), 2).blocks[0].cmds.filter((c) => c.kernel === 'mpm_p2g').length;
  assert.ok(calm < subs, `${calm} substeps calm, ${subs} at the source`);
  JSON.parse(p.report(f0[1] + 1, counters({ speed: 0.01 })));
  // Two quiet frames: the next frame bakes (kind 2) and the region closes.
  const bake = p.frame(1 / 60, -9.81, 1);
  assert.equal(bake[0], 2);
  assert.deepEqual(decodeBlocks(bake, 2).blocks[0].cmds.map((c) => c.kernel),
    ['mpm_write_args', 'mpm_bake_bin', 'mpm_bake_raise', 'mpm_bake_header', 'mpm_deposit_raise', 'mpm_deposit_header']);
  assert.equal(p.region(), 'null');
  assert.equal(p.frame(1 / 60, -9.81, 1)[0], 0);
  p.free();
});

test('a GPU flag is the frame\'s error, by name', () => {
  const p = new TerrainStep(JSON.stringify(SETTINGS));
  p.impact(0, FIELD, Float32Array.of(0, 0, 0), 2, 3000, 20000, 9.81);
  const f = p.frame(1 / 60, -9.81, 1);
  assert.throws(() => p.report(f[1], counters({ flags: 4 })), /left the active region/);
  assert.throws(() => p.report(f[1], counters({ flags: 32 })), /max_particles/);
  // One region at a time.
  assert.throws(() => p.impact(0, FIELD, Float32Array.of(0, 0, 0), 2, 3000, 20000, 9.81), /open terrain regions/);
  p.free();
});

test('settings are checked: unknown fields and bad values fail', () => {
  assert.throws(() => new TerrainStep(JSON.stringify({ config: { max_particle: 5 } })), /unknown field/);
  assert.throws(() => new TerrainStep(JSON.stringify({ settle_speed: 0 })), /settle_speed/);
  assert.throws(() => new TerrainStep(JSON.stringify({ config: { max_blocks: 70000 } })), /dispatch row/);
});

test('the absorbing ground sets up its node records at activation and the geostatic start in the first frame', () => {
  const p = new TerrainStep(JSON.stringify({ ...SETTINGS, config: { ...SETTINGS.config, boundary: 'absorbing', initial_stress: 'geostatic' } }));
  const s = JSON.parse(p.settings());
  assert.equal(s.config.boundary, 'absorbing');
  assert.equal(s.config.initial_stress, 'geostatic');
  const { blocks } = decodeBlocks(p.impact(0, FIELD, Float32Array.of(0, 0, 0), 2, 3000, 20000, 9.81));
  assert.deepEqual(blocks[0].cmds.map((c) => c.kernel),
    ['mpm_activate_count', 'mpm_activate_scan', 'mpm_activate_emit', 'mpm_activate_lower', 'mpm_absorb_init']);
  const f0 = p.frame(1 / 60, -9.81, 1);
  const first = decodeBlocks(f0, 2).blocks[0].cmds.map((c) => c.kernel);
  assert.equal(first.filter((k) => k === 'mpm_geostatic').length, 1);
  assert.ok(first.indexOf('mpm_geostatic') < first.indexOf('mpm_p2g'));
  JSON.parse(p.report(f0[1], counters({ speed: 1 })));
  const next = decodeBlocks(p.frame(1 / 60, -9.81, 1), 2).blocks[0].cmds.map((c) => c.kernel);
  assert.ok(!next.includes('mpm_geostatic'));
  p.free();
  // The absorbing ground carries the geostatic traction: it needs that start.
  assert.throws(() => new TerrainStep(JSON.stringify({ config: { boundary: 'absorbing', initial_stress: 'unstressed' } })), /geostatic/i);
  assert.throws(() => new TerrainStep(JSON.stringify({ config: { boundary: 'soft' } })), /boundary/);
});
