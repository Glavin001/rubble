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
function decodeBlocks(words, at = 0, kernels = KERNELS) {
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
    for (let i = 0; i < nc; i++, at += 3) cmds.push({ op: words[at], kernel: kernels[words[at + 1]], value: words[at + 2] });
    blocks.push({ writes, cmds });
  }
  return { blocks, end: at };
}

const SETTINGS = { config: { max_particles: 1 << 20, max_blocks: 8192, max_parcels: 1 << 16 }, region_margin: 0.6, settle_speed: 0.05, settle_frames: 2 };
const FIELD = Float32Array.of(193, 193, 1.5, 1.5, 0, 0, 0);
// An area's bake on the absorbing ground: the edge columns' give (a counting
// sort of their particles, then the shift), the splat and the deposits.
const BAKE = ['mpm_write_args', 'mpm_bake_edge_clear', 'mpm_bake_edge_count', 'mpm_bake_edge_scan',
  'mpm_bake_edge_scatter', 'mpm_bake_edge_sort', 'mpm_bake_shift', 'mpm_write_args', 'mpm_bake_bin',
  'mpm_bake_raise', 'mpm_bake_header', 'mpm_deposit_raise', 'mpm_deposit_header'];
// The last area's bake also lands any parcels still in flight before the pool
// empties (a forced close; settling waits for them).
const LAST_BAKE = [...BAKE.slice(0, -2), 'mpm_parcels_land', ...BAKE.slice(-2)];
const counters = (o = {}) => {
  const c = new Uint32Array(T.COUNTER_WORDS);
  const f = new Float32Array(c.buffer);
  c[3] = o.particles ?? 1000;
  f[2] = o.speed ?? 0;
  // Each area's own top speed (slot k at C_AREA_SPEED + k): `areas` by slot,
  // else every area at `speed`.
  for (let k = 0; k < T.MAX_AREAS; k++) f[T.C_AREA_SPEED + k] = o.areas ? (o.areas[k] ?? 0) : (o.speed ?? 0);
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
  assert.deepEqual(blocks[0].cmds.map((c) => c.kernel), ['mpm_activate_count', 'mpm_activate_scan', 'mpm_write_args', 'mpm_activate_emit', 'mpm_activate_lower', 'mpm_absorb_init']);
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
  assert.deepEqual(names.slice(g, g + 5), ['mpm_grid', 'mpm_couple_gather', 'mpm_couple_apply', 'mpm_g2p', 'mpm_absorb_faces']);
  assert.deepEqual(names.slice(-2), ['mpm_couple_emit', 'mpm_couple_finish']);
  // The source frame needs more substeps than a calm one.
  JSON.parse(p.report(f0[1], counters({ speed: 0.01 })));
  const calm = decodeBlocks(p.frame(1 / 60, -9.81, 1), 2).blocks[0].cmds.filter((c) => c.kernel === 'mpm_p2g').length;
  assert.ok(calm < subs, `${calm} substeps calm, ${subs} at the source`);
  JSON.parse(p.report(f0[1] + 1, counters({ speed: 0.01 })));
  // Two quiet frames: the next frame bakes (kind 2) and the region closes.
  const bake = p.frame(1 / 60, -9.81, 1);
  assert.equal(bake[0], 2);
  assert.deepEqual(decodeBlocks(bake, 2).blocks[0].cmds.map((c) => c.kernel), LAST_BAKE);
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
  p.free();
});

test('several impacts: far apart they open areas that bake on their own; near one they grow it', () => {
  const p = new TerrainStep(JSON.stringify(SETTINGS));
  decodeBlocks(p.impact(0, FIELD, Float32Array.of(-125, 0, 0), 2, 3000, 20000, 9.81));
  // A second meteor far away: a second area, activated and with its source.
  const second = decodeBlocks(p.impact(0, FIELD, Float32Array.of(125, 0, 0), 2, 3000, 20000, 9.81)).blocks;
  assert.equal(second.length, 2);
  assert.deepEqual(second[1].cmds.map((c) => c.kernel), ['mpm_write_args', 'mpm_source']);
  let areas = JSON.parse(p.areas());
  assert.equal(areas.length, 2);
  assert.notEqual(areas[0].slot, areas[1].slot);
  assert.equal(areas[0].dx, areas[1].dx, 'one grid');
  // One frame steps both (one coupled block).
  const f0 = p.frame(1 / 60, -9.81, 1);
  assert.equal(f0[0], 1);
  assert.equal(decodeBlocks(f0, 2).blocks.length, 1);
  // A third inside the first one's forming crater grows it: its source finds the surface of
  // the material that has moved.
  const r = areas[0].prediction.radius;
  const rim = decodeBlocks(p.impact(0, FIELD, Float32Array.of(-125 + 0.5 * r, 0, 0), 2, 3000, 20000, 9.81)).blocks;
  assert.deepEqual(rim.at(-1).cmds.map((c) => c.kernel), ['mpm_write_args', 'mpm_source_surface', 'mpm_source']);
  areas = JSON.parse(p.areas());
  assert.equal(areas.length, 2, 'the second hit grew the first area');
  const grown = areas.find((a) => a.origin[0] < 0);
  assert.ok(grown.end[0] > -125 + 0.5 * r, 'the grown box covers the new hit');
  // The far area settles while the grown one still moves: kind 3, its bake
  // then the others' coupled substeps.
  const speeds = (far, near) => { const a = []; a[areas.find((x) => x.origin[0] > 0).slot] = far; a[grown.slot] = near; return a; };
  let f = p.frame(1 / 60, -9.81, 1);
  for (let i = 0; i < 3; i++) {
    JSON.parse(p.report(f[1], counters({ speed: 50, areas: speeds(0.01, 50) })));
    f = p.frame(1 / 60, -9.81, 1);
    if (f[0] === 3) break;
  }
  assert.equal(f[0], 3);
  const blocks = decodeBlocks(f, 2).blocks;
  assert.equal(blocks.length, 2);
  assert.deepEqual(blocks[0].cmds.map((c) => c.kernel), BAKE);
  assert.ok(blocks[1].cmds.some((c) => c.kernel === 'mpm_p2g'));
  assert.equal(JSON.parse(p.areas()).length, 1);
  // Bake the rest now.
  assert.equal(decodeBlocks(p.close()).blocks.length, 1);
  assert.equal(p.region(), 'null');
  p.free();
});

test('a later event too coarse for the open grid fails by name', () => {
  const p = new TerrainStep(JSON.stringify(SETTINGS));
  p.impact(0, FIELD, Float32Array.of(-125, 0, 0), 2, 3000, 20000, 9.81);
  // A much smaller impactor asks for far finer cells than the open grid's.
  assert.throws(() => p.impact(0, FIELD, Float32Array.of(125, 0, 0), 0.5, 3000, 20000, 9.81), /too coarse/);
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
    ['mpm_activate_count', 'mpm_activate_scan', 'mpm_write_args', 'mpm_activate_emit', 'mpm_activate_lower', 'mpm_absorb_init']);
  const f0 = p.frame(1 / 60, -9.81, 1);
  // The first frame's payload sets the geostatic F (counter word 11).
  const geostatic = (words) => decodeBlocks(words, 2).blocks[0].writes.some((w) => w.binding === T.COUNTERS && w.offset === 44 && w.words[0] === 1);
  assert.ok(geostatic(f0));
  JSON.parse(p.report(f0[1], counters({ speed: 1 })));
  assert.ok(!geostatic(p.frame(1 / 60, -9.81, 1)));
  p.free();
  // The absorbing ground carries the geostatic traction: it needs that start.
  assert.throws(() => new TerrainStep(JSON.stringify({ config: { boundary: 'absorbing', initial_stress: 'unstressed' } })), /geostatic/i);
  assert.throws(() => new TerrainStep(JSON.stringify({ config: { boundary: 'soft' } })), /boundary/);
});

test('the render-only surface pass rebuilds the open region\'s ground and fails past its limit by name', () => {
  const SURFACE_KERNELS = JSON.parse(TerrainStep.surface_kernels());
  for (const k of SURFACE_KERNELS) assert.ok(existsSync(new URL(`../generated/wgsl/${k}.wgsl`, import.meta.url)), k);
  const names = (words) => {
    assert.equal(words[0], 1);
    const nw = words[1], nc = words[2];
    let at = 3;
    const writes = [];
    for (let i = 0; i < nw; i++) { writes.push({ binding: words[at], words: words.slice(at + 3, at + 3 + words[at + 2]) }); at += 3 + words[at + 2]; }
    const cmds = [];
    for (let i = 0; i < nc; i++, at += 3) cmds.push(SURFACE_KERNELS[words[at + 1]]);
    return { writes, cmds };
  };
  const p = new TerrainStep(JSON.stringify(SETTINGS));
  assert.deepEqual(Array.from(p.surface(9.81)), [0], 'no region, no pass');
  p.impact(0, FIELD, Float32Array.of(0, 0, 0), 2, 3000, 20000, 9.81);
  const { writes, cmds } = names(p.surface(9.81));
  assert.deepEqual(cmds, ['terrain_surface_clear', 'mpm_write_args', 'terrain_surface_shift', 'terrain_surface_shift_mean', 'terrain_surface_bin', 'terrain_surface_resolve', 'terrain_surface_debris']);
  assert.equal(writes[0].binding, T.SURFACE);
  assert.ok(p.surface_bytes() >= 4 * writes[0].words[1]);
  p.free();
  const small = new TerrainStep(JSON.stringify({ ...SETTINGS, surface_cells: 1000 }));
  small.impact(0, FIELD, Float32Array.of(0, 0, 0), 2, 3000, 20000, 9.81);
  assert.throws(() => small.surface(9.81), /surface max_cells/);
  small.free();
});

test('the surface pass shows every open area, each in its own view and patch', () => {
  const p = new TerrainStep(JSON.stringify(SETTINGS));
  p.set_surface_patches(Uint32Array.of(5, 128, 128, 6, 128, 128));
  p.impact(0, FIELD, Float32Array.of(-125, 0, 0), 2, 3000, 20000, 9.81);
  p.impact(0, FIELD, Float32Array.of(125, 0, 0), 2, 3000, 20000, 9.81);
  const areas = JSON.parse(p.areas());
  assert.equal(areas.length, 2);
  const [block] = decodeBlocks(p.surface(9.81), 0, JSON.parse(TerrainStep.surface_kernels())).blocks;
  const table = block.writes.find((w) => w.offset === 4 * T.SURFACE_VIEW_WORD).words;
  const view = (s) => table.slice(s * T.SURFACE_VIEW_WORDS, (s + 1) * T.SURFACE_VIEW_WORDS);
  for (const a of areas) {
    const v = view(a.slot);
    assert.equal(v[0] & T.SURFACE_VIEW_OPEN, T.SURFACE_VIEW_OPEN);
    assert.deepEqual(Array.from(v.slice(T.SURFACE_VIEW_WINDOW, T.SURFACE_VIEW_WINDOW + 4)), a.window);
    assert.equal(v[5], a.slot === 0 ? 5 : 6, 'its slot\'s patch');
  }
  assert.equal(view(2)[0], 0);
  // Every patch is published (a slot not open sinks its own).
  const publish = block.cmds.at(-1);
  assert.equal(publish.kernel, "terrain_surface_publish");
  assert.equal(publish.value, 2 * Math.ceil((128 * 128) / 4));
  p.free();
  // An area past the last patch fails by name.
  const one = new TerrainStep(JSON.stringify(SETTINGS));
  one.set_surface_patches(Uint32Array.of(5, 128, 128));
  one.impact(0, FIELD, Float32Array.of(-125, 0, 0), 2, 3000, 20000, 9.81);
  one.impact(0, FIELD, Float32Array.of(125, 0, 0), 2, 3000, 20000, 9.81);
  assert.throws(() => one.surface(9.81), /surface patches/);
  one.free();
});

test('the game profile sets its settings, and the page may override them', () => {
  const settings = (config) => JSON.parse(new TerrainStep(JSON.stringify({ config })).settings()).config;
  const game = settings({ profile: 'game' });
  assert.equal(game.cells_per_radius, 8);
  assert.ok(Math.abs(game.courant - 0.45) < 1e-6);
  assert.equal(game.bed_depth, 1.25);
  assert.equal(game.speed_growth, 1);
  const physical = settings({});
  assert.equal(physical.cells_per_radius, 16);
  assert.equal(physical.bed_depth, 1.5);
  assert.equal(physical.speed_growth, 1.5);
  assert.equal(settings({ profile: 'game', cells_per_radius: 12 }).cells_per_radius, 12);
  assert.throws(() => new TerrainStep(JSON.stringify({ config: { profile: 'fast' } })), /profile/);
});
