// Shared harness for recording demo takes (record-takes.mjs): serves engine-webgpu,
// launches headless Chrome with WebGPU (throwaway profile), talks CDP over a WebSocket,
// collects the frames a page PUTs to /rec/<take>/<frame>.jpg, and encodes them with
// ffmpeg. Needs Chrome and ffmpeg on PATH (or CHROME / FFMPEG env) and Node with
// --experimental-websocket.
import { createServer } from 'node:http';
import { spawn, spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { extname, join, normalize } from 'node:path';

const TYPES = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wgsl': 'text/plain', '.wasm': 'application/wasm', '.bin': 'application/octet-stream', '.glb': 'model/gltf-binary' };

export const FFMPEG = process.env.FFMPEG || 'ffmpeg';
export const CHROME = process.env.CHROME || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');

export function ffmpeg(args) {
  const r = spawnSync(FFMPEG, ['-loglevel', 'error', '-y', ...args], { stdio: 'inherit' });
  if (r.status !== 0) throw new Error(`ffmpeg failed: ${args.join(' ')}`);
}

// Serves `root` (static files, no caching) and accepts frame uploads; opens one Chrome page.
// Returns {base, frames, send, evaluate, open, encode, reel, close}.
export async function openHarness(root, { width = 1280, height = 720 } = {}) {
  const frames = mkdtempSync(join(tmpdir(), 'take-frames-'));
  const server = createServer((req, res) => {
    const url = new URL(req.url, 'http://x');
    const m = req.method === 'PUT' && url.pathname.match(/^\/rec\/([\w-]+)\/([\w.-]+)$/);
    if (m) {
      const chunks = [];
      req.on('data', (c) => chunks.push(c));
      req.on('end', () => {
        mkdirSync(join(frames, m[1]), { recursive: true });
        writeFileSync(join(frames, m[1], m[2]), Buffer.concat(chunks));
        res.writeHead(204).end();
      });
      return;
    }
    const path = normalize(join(root, decodeURIComponent(url.pathname)));
    if (!path.startsWith(root) || !existsSync(path) || statSync(path).isDirectory()) { res.writeHead(404).end(); return; }
    res.writeHead(200, { 'content-type': TYPES[extname(path)] || 'application/octet-stream', 'cache-control': 'no-store' });
    res.end(readFileSync(path));
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const base = `http://127.0.0.1:${server.address().port}`;

  const profile = mkdtempSync(join(tmpdir(), 'take-chrome-'));
  const debugPort = 9300 + Math.floor(Math.random() * 500);
  const chrome = spawn(CHROME, [
    '--headless=new', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`,
    '--enable-unsafe-webgpu', `--window-size=${width},${height}`, '--no-first-run', '--no-default-browser-check',
    '--disable-background-timer-throttling', '--disable-renderer-backgrounding', 'about:blank',
  ], { stdio: 'ignore' });
  const cleanup = () => { try { chrome.kill('SIGKILL'); } catch {} rmSync(profile, { recursive: true, force: true }); rmSync(frames, { recursive: true, force: true }); };
  process.on('exit', cleanup);

  let targets = [];
  for (let i = 0; i < 100 && !targets.length; i++) {
    try { targets = (await (await fetch(`http://127.0.0.1:${debugPort}/json`)).json()).filter((t) => t.type === 'page'); } catch {}
    if (!targets.length) await new Promise((r) => setTimeout(r, 100));
  }
  if (!targets.length) throw new Error(`Chrome did not start (${CHROME})`);
  const ws = new WebSocket(targets[0].webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener('open', r));
  let nextId = 0;
  const waiting = new Map();
  ws.addEventListener('message', (e) => {
    const m = JSON.parse(e.data);
    if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); }
    if (m.method === 'Runtime.exceptionThrown') console.error('[page]', m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
    if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') console.error('[page]', m.params.args.map((a) => a.value ?? a.description).join(' '));
  });
  const send = (method, params = {}) => new Promise((r) => { const id = ++nextId; waiting.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
  const evaluate = async (expression, awaitPromise = false) => {
    const r = await send('Runtime.evaluate', { expression, awaitPromise, returnByValue: true, timeout: 3_600_000 });
    if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text);
    return r.result?.result?.value;
  };
  await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: false });

  return {
    base, frames, send, evaluate,
    // Navigate to `path` (relative to root, with its query) and wait until `ready` (a JS
    // expression) is truthy; throws with the page's error text (`errorExpr`) otherwise.
    async open(path, ready, { timeoutMs = 120_000, errorExpr = '""' } = {}) {
      await send('Page.navigate', { url: `${base}/${path}` });
      const until = Date.now() + timeoutMs;
      let ok = false;
      while (!ok && Date.now() < until) {
        await new Promise((r) => setTimeout(r, 200));
        ok = await evaluate(`(() => { try { return !!(${ready}); } catch { return false; } })()`).catch(() => false);
      }
      const err = await evaluate(errorExpr).catch(() => '');
      if (!ok || err) throw new Error(`${path}: page did not become ready${err ? `: ${err}` : ''}`);
    },
    // Encode the frames uploaded under `name` to <out>/<name>.mp4 and a contact sheet
    // (one frame per second, 4x4) <out>/<name>_sheet.jpg.
    encode(name, out, fps = 60) {
      mkdirSync(out, { recursive: true });
      const input = join(frames, name, 'f%05d.jpg');
      ffmpeg(['-framerate', String(fps), '-i', input, '-c:v', 'libx264', '-preset', 'slow', '-crf', '21', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', join(out, `${name}.mp4`)]);
      ffmpeg(['-i', input, '-vf', `select='not(mod(n\\,${fps}))',scale=480:-1,tile=4x4`, '-frames:v', '1', join(out, `${name}_sheet.jpg`)]);
      return join(out, `${name}.mp4`);
    },
    // Concatenate already-encoded takes into one reel.
    reel(names, out, file) {
      const list = join(frames, `${file}.txt`);
      writeFileSync(list, names.map((t) => `file '${join(out, `${t}.mp4`)}'`).join('\n'));
      ffmpeg(['-f', 'concat', '-safe', '0', '-i', list, '-c', 'copy', '-movflags', '+faststart', join(out, file)]);
      return join(out, file);
    },
    close() { ws.close(); server.close(); cleanup(); },
  };
}
