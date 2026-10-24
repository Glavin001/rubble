#!/usr/bin/env node
// Records the vehicle demo takes to MP4 (play-testing and showcase videos).
//
//   node --experimental-websocket record-vehicle-demos.mjs [take ...] [--out DIR] [--reel]
//
// The vehicle.html takes of record-takes.mjs (takes/vehicle.mjs), run through
// vehicle.html's frame-locked `__record` hook: one physics step per 60 fps frame,
// independent of wall-clock speed. Writes recordings/<take>.mp4 plus a <take>_sheet.jpg
// contact sheet; --reel concatenates the takes into recordings/vehicles_reel.mp4. Needs
// Chrome and ffmpeg on PATH (or CHROME / FFMPEG env), and
// `cargo run -p engine-webgpu --bin wgsl_export` first.
import { recordTakes } from './record-takes.mjs';
import { VEHICLE_TAKES } from './takes/vehicle.mjs';

const args = process.argv.slice(2);
const outIdx = args.indexOf('--out');
const names = args.filter((a, i) => !a.startsWith('--') && args[i - 1] !== '--out');
await recordTakes(names.length ? names : Object.keys(VEHICLE_TAKES), {
  out: outIdx >= 0 ? args[outIdx + 1] : undefined,
  reel: args.includes('--reel') && 'vehicles_reel.mp4',
});
process.exit(0);
