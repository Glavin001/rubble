// Side-by-side crash video of the comparison manifest's cars
// (vehicle-compare-cars.json, the CPU harness `vehicle_compare`'s list): each
// car recorded by crash-lab-record.mjs in the same preset, from the same camera
// and the same timing (real time to 30 ms before its impact, 20x slow motion
// through it, real time after), labelled, then stacked side by side (two cars
// in a row, more in a grid) by ffmpeg. Cars are recorded one after another
// (one GPU job at a time), each under a kill timer.
//
//   node --experimental-websocket crash-lab-compare.mjs [--cars sedan1,etkk] [--preset wall56] \
//       [--size 960x540] [--out recordings/compare/wall56.mp4] [--timeout 600]
//
// Needs the page built (wgsl_export, wasm-pack; see crash-lab-record.mjs), Chrome
// and ffmpeg. A manifest car whose files are absent is skipped.
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const manifest = JSON.parse(readFileSync(join(ROOT, 'vehicle-compare-cars.json'), 'utf8')).cars;
const names = opt('--cars', manifest.filter((c) => !c.optional).map((c) => c.name).join(',')).split(',');
const preset = opt('--preset', 'wall56');
const size = opt('--size', '960x540');
const out = opt('--out', join(ROOT, 'recordings', 'compare', `${preset}.mp4`));
const timeout = Number(opt('--timeout', '600')) * 1000;
mkdirSync(dirname(out), { recursive: true });

const clips = [];
// The first car's impact time; the others are recorded on it (the same scene
// places every car's front at the same distance), so the clips line up even
// where a car's impact detection fires early (a lid springing open at spawn).
let impactMs = opt('--impact', null);
for (const name of names) {
  const car = manifest.find((c) => c.name === name);
  if (!car) throw new Error(`--cars: ${name} is not in vehicle-compare-cars.json`);
  if (car.mod && !existsSync(join(ROOT, '__mods', car.mod))) { console.log(`- ${name}: __mods/${car.mod} absent, skipped`); continue; }
  const clip = join(dirname(out), `${preset}-${name}.mp4`);
  console.log(`${name}: recording ${preset}…`);
  const child = spawn(process.execPath, ['--experimental-websocket', join(ROOT, 'crash-lab-record.mjs'), '--car', name, '--size', size,
    '--label', car.label || name, '--out', clip, ...(impactMs != null ? ['--impact', String(impactMs)] : []), preset], { stdio: ['ignore', 'pipe', 'inherit'] });
  child.stdout.on('data', (d) => {
    process.stdout.write(d);
    const m = /impact at (\d+) ms/.exec(d.toString());
    if (m && impactMs == null) impactMs = Number(m[1]);
  });
  // The kill timer: a hung page or GPU never outlives `timeout`.
  const timer = setTimeout(() => { console.log(`${name}: timed out, killed`); child.kill('SIGKILL'); }, timeout);
  const code = await new Promise((r) => child.on('exit', r));
  clearTimeout(timer);
  if (code !== 0 || !existsSync(clip)) throw new Error(`${name}: the recorder failed (${code})`);
  clips.push(clip);
}
if (!clips.length) throw new Error('no car recorded');
// Pad every clip to the longest (holding its last frame), then stack.
const dur = (f) => Number(spawnSync('ffprobe', ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', f]).stdout.toString().trim());
const longest = Math.max(...clips.map(dur));
const pads = clips.map((f, i) => `[${i}:v]tpad=stop_mode=clone:stop_duration=${(longest - dur(f) + 0.05).toFixed(2)}[v${i}]`);
const n = clips.length;
const stack = n === 1 ? '[v0]copy[o]'
  : n === 2 ? '[v0][v1]hstack=inputs=2[o]'
  : `${clips.map((_, i) => `[v${i}]`).join('')}xstack=inputs=${n}:layout=${clips.map((_, i) => `${i % 2 === 0 ? '0' : 'w0'}_${Math.floor(i / 2) === 0 ? '0' : 'h0'}`).join('|')}:fill=black[o]`;
const enc = spawnSync('ffmpeg', ['-loglevel', 'error', '-y', ...clips.flatMap((f) => ['-i', f]), '-filter_complex', [...pads, stack].join(';'),
  '-map', '[o]', '-c:v', 'libx264', '-preset', 'medium', '-crf', '21', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out], { stdio: 'inherit' });
if (enc.status !== 0) throw new Error('ffmpeg failed');
console.log(`→ ${out} (${longest.toFixed(1)} s, ${clips.length} cars)`);
