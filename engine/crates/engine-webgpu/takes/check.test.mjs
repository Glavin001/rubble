// The take checks (takes/check.mjs) on made-up recordings: each failure they exist for is
// reported, and a clean take passes.
import test from 'node:test';
import assert from 'node:assert/strict';
import { checkTake, BLACK_MEAN, FLICKER, UNIFORM_STD } from './check.mjs';

const take = { script: [{ n: 2, cam: { rel: [0, 0, 1] } }, { n: 3, cam: { rel: [0, 0, 1], relTo: [0.5, 0, 1] } }] };
const image = (mean = 120, std = 40) => ({ mean, std, px: new Uint8Array(4) });
// Five frames of a clean recording: the physics and the camera move, the view changes.
const clean = () => ({
  frames: 5,
  overflow: [],
  log: [0, 1, 2, 3, 4].map((i) => ({ frames: i + 1, simSeconds: (i + 1) / 60, brokenBonds: i, chunks: 10, d1: i ? 1 : null, d2: i > 1 ? 1.5 : null, std: 30, cam: [i < 2 ? 0 : 0.1 * i, 0, 10] })),
});
const extra = (rec) => ({ uploaded: rec.frames, errors: [], images: Array.from({ length: rec.frames }, () => image()) });

test('a clean take passes', () => {
  const rec = clean();
  assert.deepEqual(checkTake(take, rec, extra(rec)), []);
});

test('frame counts that disagree fail', () => {
  const rec = clean();
  const fails = checkTake(take, { ...rec, frames: 4, log: rec.log.slice(0, 4) }, { ...extra(rec), uploaded: 3 });
  assert.ok(fails.some((f) => f.includes('the script asks for 5')));
  assert.ok(fails.some((f) => f.includes('3 uploaded')));
  assert.ok(fails.some((f) => f.includes('the MP4 holds 5')));
});

test('black and uniform frames, page errors and overflowing captions fail', () => {
  const rec = { ...clean(), overflow: ['a very long caption'] };
  const x = extra(rec);
  x.images[1] = image(BLACK_MEAN - 1, 0);
  x.images[3] = image(200, 1);
  x.errors = ['GPUValidationError: bad'];
  const fails = checkTake(take, rec, x);
  assert.ok(fails.includes('black frames: 1'));
  assert.ok(fails.includes('near-uniform frames: 3'));
  assert.ok(fails.includes('page error: GPUValidationError: bad'));
  assert.ok(fails.some((f) => f.startsWith('overlay wider than the frame')));
});

test('bonds broken going back fails', () => {
  const rec = clean();
  rec.log[3].brokenBonds = 0;
  assert.ok(checkTake(take, rec, extra(rec)).some((f) => f.startsWith('brokenBonds went back at frame 3')));
});

test('a frozen frame and a flickering one fail', () => {
  const rec = clean();
  rec.log[3].d1 = 0;
  assert.ok(checkTake(take, rec, extra(rec)).some((f) => f.startsWith('frozen frames') && f.endsWith(': 3')));
  const flick = clean();
  flick.log[2].d1 = FLICKER + 5;
  flick.log[3].d1 = FLICKER + 5;
  flick.log[3].d2 = 0.5;
  assert.ok(checkTake(take, flick, extra(flick)).some((f) => f.startsWith('flicker') && f.endsWith(': 2')));
});

test('an empty view and a camera that is not a number fail', () => {
  // The caption keeps the MP4 frame from looking uniform; the page's probe of the view does not.
  const rec = clean();
  rec.log[2].std = UNIFORM_STD - 1;
  rec.log[3].std = UNIFORM_STD - 1;
  assert.ok(checkTake(take, rec, extra(rec)).includes('near-uniform views (nothing in shot): 2-3'));
  // NaN is logged as null once the log is JSON.
  const lost = clean();
  lost.log[4].cam = [null, null, null];
  assert.ok(checkTake(take, lost, extra(lost)).includes('cameras that are not numbers: 4'));
});
