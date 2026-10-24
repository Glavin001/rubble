#!/usr/bin/env node
// Records demo takes to MP4 (showcase videos and play-testing), from any demo page that
// implements the take contract:
//   window.__record(script, {name, upload, width}) plays `script` frame-locked (physics
//   advances a fixed amount per video frame, independent of wall-clock speed) and PUTs
//   each composited frame as JPEG to `${upload}/${name}/f00000.jpg`; it resolves to
//   {frames, ...}. The page's segment fields are its own (see takes/*.mjs).
//
//   node --experimental-websocket record-takes.mjs [take ...] [--page vehicle|lab|showcase] [--out DIR] [--reel] [--list] [--check]
//
// --check runs takes/check.mjs on each take (frame counts, black or uniform frames, page
// and WebGPU errors, counters going backwards, frozen frames, overlay overflow), writes
// <take>_log.json (the per-frame counters) and exits 1 when any take fails. --repeat (with
// --check) records each take a second time and fails it unless the two decode to the same
// frames (a take must repeat exactly, dust and all).
//
// Serves this directory, launches headless Chrome (WebGPU on), runs each take and encodes
// recordings/<take>.mp4 plus <take>_sheet.jpg (one frame per second) for quick review.
// --reel concatenates the takes into recordings/takes_reel.mp4. Needs Chrome and ffmpeg
// (CHROME / FFMPEG env), `cargo run -p engine-webgpu --bin wgsl_export` and, for lab
// takes, the fracture-lab and stress-web wasm builds (fracture/README.md).
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { openHarness } from './takes/harness.mjs';
import { VEHICLE_TAKES } from './takes/vehicle.mjs';
import { LAB_TAKES } from './takes/lab.mjs';
import { SHOWCASE_TAKES } from './takes/showcase.mjs';
import { checkTake, frameStats, frameHashes } from './takes/check.mjs';
import { rmSync, writeFileSync } from 'node:fs';

const ROOT = dirname(fileURLToPath(import.meta.url));
// Each page: how to tell it is ready to record and where it shows a startup error.
const PAGES = {
  'vehicle.html': { ready: 'typeof window.__record === "function" && !!window.__vehicleState()', error: 'document.getElementById("err")?.textContent || ""' },
  'fracture/index.html': { ready: 'window.__FRACTURE_DEMO__?.ready && typeof window.__record === "function"', error: 'document.querySelector(".error")?.textContent || ""' },
  'showcase/index.html': { ready: 'window.__SHOWCASE__?.ready && typeof window.__record === "function"', error: 'document.querySelector(".error")?.textContent || ""' },
};
const GROUPS = { vehicle: VEHICLE_TAKES, lab: LAB_TAKES, showcase: SHOWCASE_TAKES };
const ALL = { ...VEHICLE_TAKES, ...LAB_TAKES, ...SHOWCASE_TAKES };

export async function recordTakes(names, { out = join(ROOT, 'recordings'), reel = false, check = false, repeat = false } = {}) {
  for (const t of names) if (!ALL[t]) throw new Error(`unknown take ${t}; have ${Object.keys(ALL).join(', ')}`);
  const h = await openHarness(ROOT);
  const failed = {};
  try {
    for (const name of names) {
      const take = ALL[name];
      const page = PAGES[take.page];
      h.takeErrors();
      await h.open(`${take.page}?${take.query || ''}`, take.ready || page.ready, { errorExpr: page.error });
      if (take.setup) await h.evaluate(`(async () => { ${take.setup} })()`, true);
      const record = async (as) => {
        if (as !== name) {
          await h.open(`${take.page}?${take.query || ''}`, take.ready || page.ready, { errorExpr: page.error });
          if (take.setup) await h.evaluate(`(async () => { ${take.setup} })()`, true);
        }
        return h.evaluate(`window.__record(${JSON.stringify(take.script)}, { name: ${JSON.stringify(as)} })
          .then((r) => ({ frames: r.frames, seconds: r.seconds, summary: r.summary ?? null, log: r.log ?? null, overflow: r.overflow ?? [] }))`, true);
      };
      const r = await record(name);
      if (!r?.frames) throw new Error(`${name}: no frames recorded`);
      const file = h.encode(name, out);
      console.log(`${name}: ${r.frames} frames (${(r.frames / 60).toFixed(1)} s video) in ${r.seconds.toFixed(1)} s → ${file}${r.summary ? `\n  ${JSON.stringify(r.summary)}` : ''}`);
      if (check) {
        if (r.log) writeFileSync(join(out, `${name}_log.json`), JSON.stringify(r.log));
        const fails = checkTake(take, r, { uploaded: h.uploaded(name), errors: h.takeErrors(), images: frameStats(file) });
        if (repeat) {
          const again = `${name}__repeat`;
          await record(again);
          const file2 = h.encode(again, out);
          const [a, b] = [frameHashes(file), frameHashes(file2)];
          const first = a.findIndex((x, i) => x !== b[i]);
          if (a.length !== b.length || first >= 0) fails.push(`does not repeat: a second recording differs from frame ${first >= 0 ? first : Math.min(a.length, b.length)}`);
          for (const f of [file2, join(out, `${again}_sheet.jpg`)]) rmSync(f, { force: true });
          for (const e of h.takeErrors()) fails.push(`page error (second recording): ${e}`);
        }
        if (fails.length) failed[name] = fails;
        console.log(fails.length ? `  check FAILED:\n${fails.map((f) => `    - ${f}`).join('\n')}` : '  check passed');
      }
    }
    if (reel) console.log(`reel → ${h.reel(names, out, typeof reel === 'string' ? reel : 'takes_reel.mp4')}`);
  } finally {
    h.close();
  }
  return failed;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  const value = (flag) => { const i = args.indexOf(flag); return i >= 0 ? args[i + 1] : undefined; };
  const flagged = new Set(['--out', '--page'].flatMap((f) => { const i = args.indexOf(f); return i >= 0 ? [i, i + 1] : []; }));
  const names = args.filter((a, i) => !a.startsWith('--') && !flagged.has(i));
  const group = value('--page');
  if (group && !GROUPS[group]) throw new Error(`unknown page group ${group}; have ${Object.keys(GROUPS).join(', ')}`);
  const pool = group ? GROUPS[group] : ALL;
  if (args.includes('--list')) {
    for (const [name, t] of Object.entries(pool)) console.log(`${name.padEnd(22)} ${t.page}?${t.query || ''}`);
    process.exit(0);
  }
  const check = args.includes('--check');
  const failed = await recordTakes(names.length ? names : Object.keys(pool), { out: value('--out'), reel: args.includes('--reel'), check, repeat: args.includes('--repeat') });
  if (check) console.log(Object.keys(failed).length ? `checks failed: ${Object.keys(failed).join(', ')}` : 'every take passed its checks');
  process.exit(check && Object.keys(failed).length ? 1 : 0);
}
