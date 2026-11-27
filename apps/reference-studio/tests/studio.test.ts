// Unit tests (node --test): workspace paths, image headers, the data check, and the data
// tools on an in-memory workspace. The engine (Rust in wasm) is a stub here: what it computes
// is tested in Rust (engine-softbody, engine-softbody-lab tests) and through the page in the
// smoke test.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { test } from 'node:test';
import { checkData } from '../shared/check-data.ts';
import type { ScenarioDesc } from '../shared/model.ts';
import { Ops, type EngineApi } from '../shared/ops.ts';
import { Store, StudioError, checkId, imageSize } from '../shared/store.ts';
import { MemoryWorkspace, checkPath } from '../shared/workspace.ts';
import { DiskWorkspace, WORKSPACE } from '../scripts/node-workspace.ts';

const stub: EngineApi = {
  registry: (data) => ({ scenarios: data.map((d) => ({ id: d.id, title: d.title, reference: { video: '', atS: [], links: [], realTest: '' }, designs: [], folderDesigns: [], secsAfterImpact: d.secs_after_impact, cameras: d.cameras, moments: d.moments, criteria: [], recreated: true, contact: d.contact })), families: [], contacts: ['discrete', 'edge', 'continuous'], runs: 8, sampleSteps: 8, dt: 0.0005 }),
  pick: (m) => ({ moment: m[0]?.name ?? null, camera: null }),
  validId: (id) => {
    if (/^S\d\d$/i.test(id)) throw new Error(`scenario id ${id}: a built-in scenario has it`);
  },
  checkDesc: (d) => {
    if (!d.bodies.length) throw new Error('a scenario needs a body');
  },
  exportBuiltin: async () => {
    throw new Error('no engine in node');
  },
  recorded: async () => {
    throw new Error('no engine in node');
  },
  outline: () => ({}),
  compare: () => ({}),
  rankDesigns: async () => [],
};

async function scratch() {
  const ws = new MemoryWorkspace();
  for (const s of ['vehicle-side', 'truss-side']) await ws.write(`structures/${s}.json`, readFileSync(join(WORKSPACE, 'structures', `${s}.json`)));
  const store = await Store.open(ws);
  return { ws, store, ops: new Ops(store, stub) };
}

// A PNG header (imageSize reads only IHDR).
function pngHeader(w: number, h: number): Uint8Array {
  const b = new Uint8Array(33);
  b.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13, 0x49, 0x48, 0x44, 0x52]);
  new DataView(b.buffer).setUint32(16, w);
  new DataView(b.buffer).setUint32(20, h);
  return b;
}

test('paths stay inside the workspace; ids are file names', () => {
  for (const bad of ['../x.png', 'a/../../x.png', '/etc/passwd', '', 'a\0b.png', '..', 'a/..', 'a\\b', 'a//b']) assert.throws(() => checkPath(bad), Error, bad);
  assert.deepEqual(checkPath('library/S01/a.webp'), ['library', 'S01', 'a.webp']);
  for (const bad of ['../x', 'a/b', '.hidden', 'x'.repeat(200), 'a..b']) assert.throws(() => checkId(bad), StudioError, bad);
  assert.equal(checkId('S01_1m52s_side-02.v2'), 'S01_1m52s_side-02.v2');
});

test('image sizes from headers', () => {
  assert.deepEqual(imageSize(pngHeader(1280, 720)), { width: 1280, height: 720 });
  const webp = new Uint8Array(40);
  webp.set(new TextEncoder().encode('RIFF'), 0);
  webp.set(new TextEncoder().encode('WEBPVP8X'), 8);
  webp.set([1999 & 255, 1999 >> 8, 0], 24);
  webp.set([1630 & 255, 1630 >> 8, 0], 27);
  assert.deepEqual(imageSize(webp), { width: 2000, height: 1631 });
  assert.equal(imageSize(new TextEncoder().encode('not an image at all, just text')), null);
});

test('the committed workspace holds points and text only', async () => {
  assert.deepEqual(await checkData(new DiskWorkspace(WORKSPACE)), []);
});

test('the data check finds embedded images, and skips library/ and recordings/', async () => {
  const { ws } = await scratch();
  await ws.write('frames/bad.json', JSON.stringify({ id: 'bad', regions: [], note: 'data:image/png;base64,iVBORw0KGgo' }));
  await ws.write('frames/pic.png', pngHeader(2, 2));
  await ws.write('frames/blob.json', JSON.stringify({ x: 'A'.repeat(400) }));
  await ws.write('library/S01/ok.png', pngHeader(2, 2));
  await ws.write('recordings/abc.rrec', new Uint8Array([1, 2, 3]));
  const p = (await checkData(ws)).join('\n');
  assert.match(p, /bad\.json: a data: URI/);
  assert.match(p, /pic\.png: an image or video file/);
  assert.match(p, /blob\.json: a long base64 run/);
  assert.doesNotMatch(p, /^(library|recordings)\//m);
});

test('editing tools: regions, keypoints, polylines, visibility, undo by put_frame; writes reach the workspace', async () => {
  const { ops, store, ws } = await scratch();
  const shot = ops.t_create_shot({ title: 'Truss under load' });
  assert.equal(shot.id, 'Truss-under-load');
  const f = await ops.addImage('../../evil name.png', pngHeader(1000, 500), { shot: shot.id });
  assert.equal(f.image!.file, 'Truss-under-load/evil_name.png', 'the file name is cleaned and kept inside the library');
  assert.equal((await ops.addImage('again.png', pngHeader(1000, 500), {})).id, f.id, 'the same bytes are the same frame');
  assert.throws(() => ops.t_set_keypoint({ frame: f.id, region: 'real', element: 'x', x: 0, y: 0 }), /no region/);
  const r = ops.t_mark_region({ frame: f.id, source: 'other', x0: 0.5, y0: 0, x1: 0, y1: 1 });
  assert.deepEqual(r.crop, [0, 0, 0.5, 1]);
  ops.t_set_region_schema({ frame: f.id, region: r.id, schema: 'truss-side' });
  ops.t_set_keypoint({ frame: f.id, region: r.id, element: 'left_support', x: 0.1, y: 0.9 });
  assert.throws(() => ops.t_set_keypoint({ frame: f.id, region: r.id, element: 'top_chord', x: 0.1, y: 0.9 }), /polyline/);
  for (const [x, y] of [[0.1, 0.5], [0.9, 0.5]]) ops.t_add_polyline_point({ frame: f.id, region: r.id, element: 'top_chord', x, y });
  ops.t_add_polyline_point({ frame: f.id, region: r.id, element: 'top_chord', x: 0.5, y: 0.45, index: 1 });
  ops.t_move_polyline_point({ frame: f.id, region: r.id, element: 'top_chord', index: 1, x: 0.5, y: 0.4 });
  const before = store.frame(f.id);
  ops.t_set_visibility({ frame: f.id, region: r.id, element: 'top_chord', vis: 'blurred' });
  ops.t_remove_point({ frame: f.id, region: r.id, element: 'top_chord', index: 0 });
  let t = ops.t_get_tracing({ frame: f.id, region: r.id })[0];
  assert.deepEqual(t.instances[0].elements.top_chord, { pts: [[0.5, 0.4], [0.9, 0.5]], vis: 'blurred' });
  assert.throws(() => ops.t_set_keypoint({ frame: f.id, region: r.id, element: 'left_support', x: 7, y: 0 }), /normalised/);
  ops.t_put_frame({ frame: before });
  t = ops.t_get_tracing({ frame: f.id, region: r.id })[0];
  assert.equal(t.instances[0].elements.top_chord.pts!.length, 3, 'put_frame restores the earlier document');
  assert.deepEqual(ops.summaries()[0].status, ['traced']);
  assert.equal(ops.t_list_untraced().length, 0);
  await store.flush();
  const saved = JSON.parse(new TextDecoder().decode((await ws.read(`frames/${f.id}.json`))!));
  assert.equal(saved.regions[0].tracing.instances[0].elements.top_chord.pts.length, 3, 'the workspace has the document');
  const again = await Store.open(ws);
  assert.equal(again.frame(f.id).regions.length, 1, 'a reopened workspace reads it back');
  assert.ok(again.imageExists(f.image!.file));
});

test('copy the previous frame of a shot', async () => {
  const { ops } = await scratch();
  const s = ops.t_create_shot({ title: 'two frames' });
  const a = await ops.addImage('a.png', pngHeader(10, 10), { shot: s.id, moment: 'before' });
  const b = await ops.addImage('b.png', pngHeader(10, 11), { shot: s.id, moment: 'peak' });
  for (const f of [a, b]) ops.t_mark_region({ frame: f.id, source: 'real' });
  ops.t_set_region_schema({ frame: a.id, region: 'real', schema: 'vehicle-side' });
  ops.t_set_keypoint({ frame: a.id, region: 'real', element: 'door_sill_corner', x: 0.3, y: 0.8 });
  const r = ops.t_copy_tracing_from_previous({ frame: b.id, region: 'real' }) as { copied_from: string };
  assert.equal(r.copied_from, a.id);
  assert.equal(ops.beforeOf(ops.store.frame(b.id))?.id, a.id, 'the before frame: a "before" label first');
});

test('scenario tools: create, boxes, bodies, info; pins and cameras on frames', async () => {
  const { ops, store, ws } = await scratch();
  const d = (await ops.t_create_scenario({ title: 'Car into wall' })) as ScenarioDesc;
  assert.equal(d.id, 'car-into-wall');
  await assert.rejects(ops.t_create_scenario({ id: 'S02' }), /built-in/);
  ops.t_add_box({ scenario: d.id, name: 'pole', centre: [1, 1, 9], half: [0.15, 1, 0.15], yaw_deg: 30 });
  ops.t_set_box({ scenario: d.id, box: 'wall', centre: [0, 1, 7.0004] });
  ops.t_set_body({ scenario: d.id, body: 'car', speed_kmh: 56.3, yaw_deg: 2, design: 'suv_4x4' });
  ops.t_add_body({ scenario: d.id, name: 'other', design: 'sedan', at: [3, 0, 0] });
  assert.throws(() => ops.t_set_box({ scenario: d.id, box: 'nope' }), /no box/);
  ops.t_remove_body({ scenario: d.id, body: 'other' });
  assert.throws(() => ops.t_remove_body({ scenario: d.id, body: 'car' }), /at least one/);
  ops.t_set_scenario_info({ scenario: d.id, secs_after_impact: 0.3, contact: 'continuous' });
  const now = store.scenario(d.id);
  assert.deepEqual(now.boxes.map((b) => b.name), ['wall', 'pole']);
  assert.equal(now.boxes[0].centre[2], 7, 'stored to the millimetre');
  assert.equal(now.bodies[0].design, 'suv_4x4');
  assert.equal(now.contact, 'continuous');
  await store.flush();
  assert.ok(await ws.read(`scenarios/${d.id}.json`), 'saved in the workspace');
  const listed = ops.t_list_scenarios() as { id: string; recreated: boolean }[];
  assert.ok(listed.some((x) => x.id === d.id && x.recreated));
  // A frame pinned at a time with its camera; another image copies the camera.
  const s = ops.t_create_shot({ title: 'pins' });
  const a = await ops.addImage('a.png', pngHeader(1280, 720), { shot: s.id });
  const b = await ops.addImage('b.png', pngHeader(1280, 721), { shot: s.id });
  ops.t_set_frame_camera({ frame: a.id, eye: [4, 1, 0], target: [0, 0.8, 0], fov_y_deg: 50 });
  ops.t_set_frame_pin({ frame: a.id, at_ms: 70.123, recording: 'abc', label: 'peak' });
  ops.t_copy_camera({ from: a.id, to: b.id });
  const pins = ops.t_list_pins({ shot: s.id });
  assert.deepEqual(pins.map((p) => [p.frame, p.pinned, p.at_ms]), [[a.id, true, 70.12], [b.id, false, null]]);
  assert.equal(store.frame(b.id).camera!.fov_y_deg, 50);
  assert.ok(ops.summaries().find((x) => x.id === a.id)!.status.includes('pinned'));
});
