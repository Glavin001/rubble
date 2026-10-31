// Checks on a recorded take (record-takes.mjs --check): what a viewer would call broken
// in the video, found without watching it.
//   - frames: the take's segments ask for sum(n) video frames; the page must report that
//     many, upload that many and the MP4 must hold that many;
//   - black or near-uniform frames (a frame drawn before the scene, a lost device, a
//     camera inside a wall), from the MP4 decoded small and grey, and frames whose view
//     alone (the page's probe, without the caption that keeps a frame from looking
//     uniform) is near uniform: nothing in shot;
//   - a camera that is not a number (`cam` logged with NaN or null: a script's relative
//     camera taken from a scene orbit the view did not report);
//   - page errors: exceptions, console errors, WebGPU validation messages;
//   - counters going backwards (per video frame, `lab.counters()`): bonds broken, the
//     physics frame and simulated time never decrease;
//   - frozen frames: two video frames with the same view while simulated time advanced
//     and the camera moved (a draw that did not happen), and flicker: one frame unlike
//     both its neighbours, which match (from the page's exact view differences);
//   - a caption or stats line wider than the frame.
// Pure functions of what the recorder collected, plus ffmpeg for the frames.
import { spawnSync } from 'node:child_process';
import { FFMPEG } from './harness.mjs';

// The decoded frames' size (grey, one byte a pixel).
const W = 64, H = 36;
// A frame darker than this mean (0..255) is black; one whose standard deviation is under
// this is near uniform.
export const BLACK_MEAN = 10;
export const UNIFORM_STD = 4;

// Per frame of `mp4`: {mean, std} of the grey image, and its bytes (for comparisons).
export function frameStats(mp4) {
  const r = spawnSync(FFMPEG, ['-loglevel', 'error', '-i', mp4, '-vf', `scale=${W}:${H}:flags=area,format=gray`, '-f', 'rawvideo', '-'], { maxBuffer: 1 << 30 });
  if (r.status !== 0) throw new Error(`ffmpeg could not decode ${mp4}: ${r.stderr}`);
  const n = W * H, out = [];
  for (let o = 0; o + n <= r.stdout.length; o += n) {
    const px = r.stdout.subarray(o, o + n);
    let sum = 0, sq = 0;
    for (const v of px) { sum += v; sq += v * v; }
    const mean = sum / n;
    out.push({ mean, std: Math.sqrt(Math.max(0, sq / n - mean * mean)), px });
  }
  return out;
}

// Each decoded frame's MD5 (ffmpeg framemd5): two recordings of a take must match.
export function frameHashes(mp4) {
  const r = spawnSync(FFMPEG, ['-loglevel', 'error', '-i', mp4, '-f', 'framemd5', '-'], { maxBuffer: 1 << 28, encoding: 'utf8' });
  if (r.status !== 0) throw new Error(`ffmpeg could not hash ${mp4}: ${r.stderr}`);
  return r.stdout.split('\n').filter((l) => l && !l.startsWith('#')).map((l) => l.split(',').at(-1).trim());
}

// A frame this different (mean absolute, 0..255, from `lab`'s view probe) from both
// neighbours while they match each other flickers.
export const FLICKER = 3;

// The checks over one take. `take`: its definition (script); `rec`: what `__record`
// returned ({frames, log, overflow}); `uploaded`: frames the server received; `errors`:
// page errors; `images`: frameStats of its MP4. Returns a list of failure strings.
export function checkTake(take, rec, { uploaded, errors, images }) {
  const fail = [];
  const asked = take.script.reduce((n, s) => n + (s.n || 0), 0);
  if (rec.frames !== asked) fail.push(`frames: the script asks for ${asked}, the page recorded ${rec.frames}`);
  if (uploaded !== rec.frames) fail.push(`frames: ${rec.frames} recorded, ${uploaded} uploaded`);
  if (images.length !== rec.frames) fail.push(`frames: ${rec.frames} recorded, the MP4 holds ${images.length}`);
  if (rec.log && rec.log.length !== rec.frames) fail.push(`frames: ${rec.frames} recorded, ${rec.log.length} logged`);
  for (const e of errors) fail.push(`page error: ${e}`);
  for (const o of rec.overflow || []) fail.push(`overlay wider than the frame: ${JSON.stringify(o)}`);
  const runs = (pred) => {
    // Consecutive frame ranges where pred(i) holds, as "a-b".
    const out = [];
    for (let i = 0; i < images.length; i++) {
      if (!pred(i)) continue;
      let j = i;
      while (j + 1 < images.length && pred(j + 1)) j++;
      out.push(i === j ? `${i}` : `${i}-${j}`);
      i = j;
    }
    return out;
  };
  const black = runs((i) => images[i].mean < BLACK_MEAN);
  if (black.length) fail.push(`black frames: ${black.join(', ')}`);
  const flat = runs((i) => images[i].mean >= BLACK_MEAN && images[i].std < UNIFORM_STD);
  if (flat.length) fail.push(`near-uniform frames: ${flat.join(', ')}`);
  const log = rec.log || [];
  const empty = runs((i) => typeof log[i]?.std === 'number' && log[i].std < UNIFORM_STD);
  if (empty.length) fail.push(`near-uniform views (nothing in shot): ${empty.join(', ')}`);
  const lost = runs((i) => Array.isArray(log[i]?.cam) && log[i].cam.some((v) => !Number.isFinite(v)));
  if (lost.length) fail.push(`cameras that are not numbers: ${lost.join(', ')}`);
  for (const key of ['brokenBonds', 'frames', 'simSeconds']) {
    for (let i = 1; i < log.length; i++) {
      if (log[i][key] === undefined || log[i - 1][key] === undefined) continue;
      if (log[i][key] < log[i - 1][key]) fail.push(`${key} went back at frame ${i}: ${log[i - 1][key]} → ${log[i][key]}`);
    }
  }
  if (log.length && log[0].chunks === 0) fail.push('the first frame was drawn before the scene loaded (0 chunks)');
  // Frozen: the view (exact pixels, before encoding: `d1`) did not change although the
  // physics advanced and the camera moved, so the frame was not drawn.
  const camMoved = (i) => {
    const a = log[i]?.cam, b = log[i - 1]?.cam;
    return !!(a && b && a.some((v, k) => Math.abs(v - b[k]) > 1e-9));
  };
  const frozen = runs((i) => i > 0 && log[i]?.d1 === 0 && camMoved(i) && log[i].frames > log[i - 1].frames);
  if (frozen.length) fail.push(`frozen frames (the physics and the camera moved, the view did not): ${frozen.join(', ')}`);
  // Flicker: one frame unlike both its neighbours, which are alike (a frame drawn from
  // the wrong state, or a missing piece for one frame).
  const flicker = runs((i) => i > 0 && i + 1 < log.length && log[i].d1 > FLICKER && log[i + 1].d1 > FLICKER && log[i + 1].d2 !== null && log[i + 1].d2 < 0.25 * Math.min(log[i].d1, log[i + 1].d1));
  if (flicker.length) fail.push(`flicker (a frame unlike both neighbours): ${flicker.join(', ')}`);
  return fail;
}
