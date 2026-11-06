#!/usr/bin/env node
// Every converted community vehicle (community_cache's __mods/index.json) in the
// browser crash lab (softbody.html ?solver=explicit, JBeam mode, raycast wheels):
// loads, rests 2 s (its broken and bent beams), then the 56 km/h wall (wall56) for
// 1 s. Prints a table (vehicle, configuration, MB, nodes, beams, rest, wall) and
// writes a still of each wall run. Local content only: the folder holds no mods
// in git; skips when there is no index.
//
//   node --experimental-websocket community-check.mjs [--all] [--out DIR] [folder ...]
//
// --all: every configuration (default: each vehicle's representative one). Needs Chrome
// (CHROME to override).
import { chromeArgs } from './check-chrome.mjs';
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const ALL = args.includes('--all');
const OUT = opt('--out', join(ROOT, 'recordings', 'community'));
const only = args.filter((a, i) => !a.startsWith('--') && !args[i - 1]?.startsWith('--out'));
const INDEX = join(ROOT, '__mods', 'index.json');
if (!existsSync(INDEX)) { console.log('- no __mods/index.json (community_cache): skipped'); process.exit(0); }
mkdirSync(OUT, { recursive: true });
const cars = JSON.parse(readFileSync(INDEX, 'utf8')).filter((c) => !only.length || only.includes(c.folder));
const CHROME = process.env.CHROME || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');

const TYPES = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wgsl': 'text/plain', '.wasm': 'application/wasm' };
const server = createServer((req, res) => {
  const p = normalize(join(ROOT, decodeURIComponent(new URL(req.url, 'http://x').pathname)));
  if (!p.startsWith(ROOT) || !existsSync(p) || statSync(p).isDirectory()) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': TYPES[extname(p)] || 'application/octet-stream', 'cache-control': 'no-store' });
  res.end(readFileSync(p));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

const profile = mkdtempSync(join(tmpdir(), 'community-chrome-'));
const port = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, chromeArgs({ stock: false, port, profile, width: 1280, height: 720 }), { stdio: 'ignore' });
process.on('exit', () => { try { chrome.kill('SIGKILL'); } catch {} try { rmSync(profile, { recursive: true, force: true }); } catch {} });
let targets = [];
for (let i = 0; i < 100 && !targets.length; i++) {
  try { targets = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).filter((t) => t.type === 'page'); } catch {}
  if (!targets.length) await new Promise((r) => setTimeout(r, 100));
}
const ws = new WebSocket(targets[0].webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener('open', r));
let nextId = 0;
const waiting = new Map();
let errors = [];
ws.addEventListener('message', (e) => {
  const m = JSON.parse(e.data);
  if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); }
  if (m.method === 'Runtime.exceptionThrown') errors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') errors.push(m.params.args.map((a) => a.value ?? a.description).join(' '));
});
const send = (method, params = {}) => new Promise((r) => { const id = ++nextId; waiting.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true, timeout: 600_000 });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text);
  return r.result?.result?.value;
};
await send('Runtime.enable');
await send('Page.enable');
const open = async (q) => {
  errors = [];
  await send('Page.navigate', { url: `${base}/softbody.html?${q}` });
  for (let i = 0; i < 1800; i++) {
    if (await evaluate('window.__ready === true').catch(() => false)) return;
    if (errors.length) throw new Error(errors.join('; '));
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error('page not ready in 180 s');
};
const step = (n) => evaluate(`window.__step(${n}).then((r) => r.damage?.[0] ?? null)`);

const rows = [];
for (const c of cars) {
  // Each vehicle's representative configuration (else its first).
  const PREFER = ['etkk/kr6.pc', 'ccf/f4_sport_M.pc', 'gavril_vertex/gavril_vertex.pc', 'fullsuv/mid-sport.pc', 'boathauler/gooseneck.pc', 'bh_yacht48/yacht48.pc'];
  const first = c.configs.find((k) => PREFER.includes(k.pc)) ?? c.configs[0];
  for (const k of ALL ? c.configs : [first]) {
    const q = `solver=explicit&mod=/__mods/${c.folder}&pc=${encodeURIComponent(k.pc)}${(k.wheels ?? 1) > 0 ? '&wheels=raycast' : ''}`;
    const row = { car: c.label, pc: k.pc, mb: c.megabytes, nodes: k.nodes, beams: k.beams, rest: '', wall: '', url: `/softbody.html?${q}&preset=${(k.wheels ?? 1) > 0 ? 'wall56' : 'drop15'}` };
    try {
      await open(`${q}&xscene=idle&cam=left`);
      const d = await step(4000);
      row.rest = `${d.broken} broken, ${d.yielded} bent`;
      const still = await send('Page.captureScreenshot', { format: 'jpeg', quality: 80 });
      writeFileSync(join(OUT, `${c.folder}-${k.pc.split('/').pop().replace(/\.pc$/, '')}-rest.jpg`), Buffer.from(still.result.data, 'base64'));
      // A vehicle without wheels (the yacht) takes the 15 m drop.
      const crash = (k.wheels ?? 1) > 0 ? 'wall56' : 'drop15';
      await open(`${q}&preset=${crash}`);
      // (The drop: 15 m takes 1.75 s to fall.)
      const w = await step(crash === 'drop15' ? 6000 : 3000);
      row.wall = `${w.broken} broken, ${w.yielded} bent`;
      const shot = await send('Page.captureScreenshot', { format: 'jpeg', quality: 80 });
      writeFileSync(join(OUT, `${c.folder}-${k.pc.split('/').pop().replace(/\.pc$/, '')}-${crash}.jpg`), Buffer.from(shot.result.data, 'base64'));
      row.wall = `${crash}: ${row.wall}`;
      // The vehicle stays across scene and preset changes (the lab's menu).
      for (const v of (k.wheels ?? 1) > 0 ? ['s:sidepole', 's:yard', 'p:pole29'] : ['s:sidepole', 'p:pole29']) {
        await evaluate(`(() => { const s = document.getElementById('labPreset'); s.value = '${v}'; s.dispatchEvent(new Event('change')); })()`);
        await new Promise((r) => setTimeout(r, 1500));
        for (let i = 0; i < 1800 && !(await evaluate('window.__ready === true').catch(() => false)); i++) await new Promise((r) => setTimeout(r, 100));
        const href = await evaluate('location.search');
        if (!href.includes(encodeURIComponent(`/__mods/${c.folder}`)) && !href.includes(`/__mods/${c.folder}`)) throw new Error(`${v} lost the vehicle: ${href}`);
      }
      row.wall += (k.wheels ?? 1) > 0 ? '; kept across sidepole, yard, pole29' : '; kept across sidepole, pole29';
    } catch (e) {
      row.rest ||= `error: ${String(e.message ?? e).slice(0, 160)}`;
      row.wall ||= 'not run';
    }
    console.log(`| ${row.car} | ${row.pc} | ${row.mb} | ${row.nodes} | ${row.beams} | ${row.rest} | ${row.wall} |`);
    rows.push(row);
  }
}
writeFileSync(join(OUT, 'community-check.json'), JSON.stringify(rows, null, 2));
process.exit(0);
