#!/usr/bin/env node
// Records demo takes to MP4 (showcase videos and play-testing), from any demo page that
// implements the take contract:
//   window.__record(script, {name, upload, width}) plays `script` frame-locked (physics
//   advances a fixed amount per video frame, independent of wall-clock speed) and PUTs
//   each composited frame as JPEG to `${upload}/${name}/f00000.jpg`; it resolves to
//   {frames, ...}. The page's segment fields are its own (see takes/*.mjs).
//
//   node --experimental-websocket record-takes.mjs [take ...] [--page vehicle|lab] [--out DIR] [--reel] [--list]
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

const ROOT = dirname(fileURLToPath(import.meta.url));
// Each page: how to tell it is ready to record and where it shows a startup error.
const PAGES = {
  'vehicle.html': { ready: 'typeof window.__record === "function" && !!window.__vehicleState()', error: 'document.getElementById("err")?.textContent || ""' },
  'fracture/index.html': { ready: 'window.__FRACTURE_DEMO__?.ready && typeof window.__record === "function"', error: 'document.querySelector(".error")?.textContent || ""' },
};
const GROUPS = { vehicle: VEHICLE_TAKES, lab: LAB_TAKES };
const ALL = { ...VEHICLE_TAKES, ...LAB_TAKES };

export async function recordTakes(names, { out = join(ROOT, 'recordings'), reel = false } = {}) {
  for (const t of names) if (!ALL[t]) throw new Error(`unknown take ${t}; have ${Object.keys(ALL).join(', ')}`);
  const h = await openHarness(ROOT);
  try {
    for (const name of names) {
      const take = ALL[name];
      const page = PAGES[take.page];
      await h.open(`${take.page}?${take.query || ''}`, take.ready || page.ready, { errorExpr: page.error });
      if (take.setup) await h.evaluate(`(async () => { ${take.setup} })()`, true);
      const r = await h.evaluate(`window.__record(${JSON.stringify(take.script)}, { name: ${JSON.stringify(name)} })
        .then((r) => ({ frames: r.frames, seconds: r.seconds, summary: r.summary ?? null }))`, true);
      if (!r?.frames) throw new Error(`${name}: no frames recorded`);
      const file = h.encode(name, out);
      console.log(`${name}: ${r.frames} frames (${(r.frames / 60).toFixed(1)} s video) in ${r.seconds.toFixed(1)} s → ${file}${r.summary ? `\n  ${JSON.stringify(r.summary)}` : ''}`);
    }
    if (reel) console.log(`reel → ${h.reel(names, out, typeof reel === 'string' ? reel : 'takes_reel.mp4')}`);
  } finally {
    h.close();
  }
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
  await recordTakes(names.length ? names : Object.keys(pool), { out: value('--out'), reel: args.includes('--reel') });
  process.exit(0);
}
