// Unit tests of the server side (node --test): path safety, image headers, the data check,
// and the editing tools on a scratch data folder. No Rust, no browser.

import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { checkData } from '../server/check-data.ts';
import { Ops } from '../server/ops.ts';
import { Rust } from '../server/rust.ts';
import { ENGINE, Store, StudioError, checkId, imageSize, safeJoin } from '../server/store.ts';

const REAL = join(ENGINE, 'docs', 'crash-reference', 'studio');

function scratch() {
  const root = mkdtempSync(join(tmpdir(), 'studio-test-'));
  const data = join(root, 'data');
  mkdirSync(join(data, 'structures'), { recursive: true });
  for (const s of ['vehicle-side', 'truss-side']) writeFileSync(join(data, 'structures', `${s}.json`), readFileSync(join(REAL, 'structures', `${s}.json`)));
  const store = new Store({ data, library: join(root, 'library'), cache: join(root, 'cache') });
  return { root, data, store, ops: new Ops(store, new Rust(join(root, 'cache'), () => {})) };
}

// A 3 x 2 PNG (header only matters to imageSize; the bytes after IHDR are not read).
function pngHeader(w: number, h: number): Buffer {
  const b = Buffer.alloc(33);
  b.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13, 0x49, 0x48, 0x44, 0x52]);
  b.writeUInt32BE(w, 16);
  b.writeUInt32BE(h, 20);
  return b;
}

test('paths cannot leave their folder', () => {
  const base = '/tmp/studio-base';
  for (const bad of ['../x.png', 'a/../../x.png', '/etc/passwd', '', 'a\0b.png', '..', 'a/..']) {
    assert.throws(() => safeJoin(base, bad), StudioError, bad);
  }
  assert.equal(safeJoin(base, 'S01/a.webp'), '/tmp/studio-base/S01/a.webp');
  for (const bad of ['../x', 'a/b', '.hidden', 'x'.repeat(200), 'a..b']) assert.throws(() => checkId(bad), StudioError, bad);
  assert.equal(checkId('S01_1m52s_side-02.v2'), 'S01_1m52s_side-02.v2');
});

test('image sizes from headers', () => {
  assert.deepEqual(imageSize(pngHeader(1280, 720)), { width: 1280, height: 720 });
  const webp = Buffer.alloc(40);
  webp.write('RIFF', 0);
  webp.write('WEBPVP8X', 8);
  webp.writeUIntLE(1999, 24, 3);
  webp.writeUIntLE(1630, 27, 3);
  assert.deepEqual(imageSize(webp), { width: 2000, height: 1631 });
  assert.equal(imageSize(Buffer.from('not an image at all, just text')), null);
});

test('the committed data holds points and text only', () => {
  assert.deepEqual(checkData(REAL), []);
});

test('the data check finds embedded images', () => {
  const { data } = scratch();
  mkdirSync(join(data, 'frames'), { recursive: true });
  writeFileSync(join(data, 'frames', 'bad.json'), JSON.stringify({ id: 'bad', regions: [], note: 'data:image/png;base64,iVBORw0KGgo' }));
  writeFileSync(join(data, 'frames', 'pic.png'), pngHeader(2, 2));
  writeFileSync(join(data, 'frames', 'blob.json'), JSON.stringify({ x: 'A'.repeat(400) }));
  const p = checkData(data).join('\n');
  assert.match(p, /bad\.json: a data: URI/);
  assert.match(p, /pic\.png: an image or video file/);
  assert.match(p, /blob\.json: a long base64 run/);
});

test('editing tools: regions, keypoints, polylines, visibility, undo by put_frame', async () => {
  const { ops, store } = scratch();
  const shot = ops.t_create_shot({ title: 'Truss under load' });
  assert.equal(shot.id, 'Truss-under-load');
  const f = ops.addImage('../../evil name.png', pngHeader(1000, 500), { shot: shot.id });
  assert.equal(f.image!.file, 'Truss-under-load/evil_name.png', 'the file name is cleaned and kept inside the library');
  assert.equal(ops.addImage('again.png', pngHeader(1000, 500), {}).id, f.id, 'the same bytes are the same frame');
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
  const sums = ops.summaries();
  assert.deepEqual(sums[0].status, ['traced']);
  assert.equal(ops.t_list_untraced().length, 0);
});

test('copy the previous frame of a shot', () => {
  const { ops } = scratch();
  const s = ops.t_create_shot({ title: 'two frames' });
  const a = ops.addImage('a.png', pngHeader(10, 10), { shot: s.id, moment: 'before' });
  const b = ops.addImage('b.png', pngHeader(10, 11), { shot: s.id, moment: 'peak' });
  for (const f of [a, b]) ops.t_mark_region({ frame: f.id, source: 'real' });
  ops.t_set_region_schema({ frame: a.id, region: 'real', schema: 'vehicle-side' });
  ops.t_set_keypoint({ frame: a.id, region: 'real', element: 'door_sill_corner', x: 0.3, y: 0.8 });
  const r = ops.t_copy_tracing_from_previous({ frame: b.id, region: 'real' }) as { copied_from: string };
  assert.equal(r.copied_from, a.id);
  assert.equal(ops.beforeOf(ops.store.frame(b.id))?.id, a.id, 'the before frame: a "before" label first');
});
