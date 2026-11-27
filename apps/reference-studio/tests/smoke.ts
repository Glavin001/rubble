// Smoke test: drive the studio headlessly through window.studio and WebMCP, as an agent would.
//
//   npm run smoke                 (from apps/reference-studio; needs Chrome and the Rust tools)
//   npm run smoke -- --real       screenshots of the real library instead (third-party footage: do not share)
//
// Starts its own studio server on 127.0.0.1:8771 with a scratch data folder and library
// (engine/out/reference-studio/smoke/), draws two test images itself (a car silhouette
// before and after a frontal crash: no third-party footage), creates a shot, adds both
// frames, marks regions, traces them, links the shot to Rubble's S02 with a hand-set camera,
// reads the comparison table, and checks the same tools through document.modelContext and
// the HTTP API. Screenshots of every view at 1280x720 and 800 px wide, light and dark, go to
// engine/out/reference-studio/screenshots/<generated|real>/. Chrome runs headless without
// the GPU. Exits 1 on the first failed check.

import { spawn, type ChildProcess } from 'node:child_process';
import { copyFileSync, mkdirSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { deflateSync } from 'node:zlib';
import { ENGINE, PACKAGE } from '../server/store.ts';

const REAL = process.argv.includes('--real');
const PORT = 8771;
const BASE = `http://127.0.0.1:${PORT}`;
const OUT = join(ENGINE, 'out', 'reference-studio');
const SCRATCH = join(OUT, 'smoke');
const SHOTS = join(OUT, 'screenshots', REAL ? 'real' : 'generated');
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const procs: ChildProcess[] = [];
let failures = 0;

function check(ok: unknown, what: string): void {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${what}`);
  if (!ok) failures++;
}

// ---------------------------------------------------------------- a PNG drawn here

const CRC = new Int32Array(256).map((_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c;
});
function crc32(b: Buffer): number {
  let c = -1;
  for (const x of b) c = CRC[(c ^ x) & 0xff] ^ (c >>> 8);
  return (c ^ -1) >>> 0;
}
function png(w: number, h: number, px: (x: number, y: number) => [number, number, number]): Buffer {
  const raw = Buffer.alloc((w * 3 + 1) * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const [r, g, b] = px(x, y);
      raw.set([r, g, b], y * (w * 3 + 1) + 1 + x * 3);
    }
  const chunk = (type: string, data: Buffer) => {
    const t = Buffer.concat([Buffer.from(type), data]);
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length);
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(crc32(t));
    return Buffer.concat([len, t, crc]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr.set([8, 2, 0, 0, 0], 8);
  return Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}
type P = [number, number];
function inside(poly: P[], x: number, y: number): boolean {
  let c = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c;
  }
  return c;
}
/** A side-view car silhouette, front at the left; `crush` 0..1 folds the hood and pushes the nose and front wheel back. */
function car(crush: number) {
  const k = crush;
  const body: P[] = [[180 + 70 * k, 520], [180 + 70 * k, 430], [260 + 60 * k, 400 - 40 * k], [360 + 10 * k, 385 - 50 * k], [470, 380], [600, 262], [880, 262], [1010, 380], [1100, 400], [1100, 520]];
  const window: P[] = [[610, 285], [730, 285], [730, 372], [500, 378]];
  const wheels: [number, number, number][] = [[330 + 45 * k, 520, 70], [950, 520, 70]];
  const image = png(1280, 720, (x, y) => {
    for (const [cx, cy, r] of wheels) {
      const d = Math.hypot(x - cx, y - cy);
      if (d < r * 0.45) return [170, 175, 185];
      if (d < r) return [28, 30, 34];
    }
    if (inside(window, x, y)) return [120, 160, 200];
    if (inside(body, x, y)) return k > 0 && x < 470 ? [205, 120, 60] : [200, 70, 60];
    if (y > 590) return [120, 122, 128];
    return [232, 236, 242];
  });
  const n = (p: P): P => [+(p[0] / 1280).toFixed(4), +(p[1] / 720).toFixed(4)];
  return {
    image,
    hood: [body[2], body[3], body[4]].map(n),
    bumper: [body[1], body[0]].map(n),
    windscreen: [body[4], body[5]].map(n),
    roof: [body[5], body[6]].map(n),
    sill: [[300 + 45 * k + 90, 520], [880, 520]].map((p) => n(p as P)),
    door: [[500, 300], [470, 520]].map((p) => n(p as P)),
    corner: n([470, 520]),
    wheels: wheels.map(([cx, cy, r]) => ({ c: n([cx, cy]), rim: n([cx + r, cy]) })),
  };
}

// ---------------------------------------------------------------- Chrome over CDP

async function chrome() {
  const port = 9400 + Math.floor(Math.random() * 400);
  const profile = join(SCRATCH, 'chrome-profile');
  mkdirSync(profile, { recursive: true });
  procs.push(spawn(CHROME, ['--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, '--window-size=1280,720', 'about:blank'], { stdio: 'ignore' }));
  let target: { webSocketDebuggerUrl: string } | undefined;
  for (let i = 0; i < 80 && !target; i++) {
    await sleep(250);
    try {
      target = ((await (await fetch(`http://127.0.0.1:${port}/json`)).json()) as { type: string; webSocketDebuggerUrl: string }[]).find((t) => t.type === 'page');
    } catch {
      /* not up yet */
    }
  }
  if (!target) throw new Error('Chrome did not start');
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r, j) => ((ws.onopen = r), (ws.onerror = j)));
  let id = 0;
  const pending = new Map<number, (d: Record<string, unknown>) => void>();
  const errors: string[] = [];
  ws.onmessage = (m) => {
    const d = JSON.parse(String(m.data));
    if (d.id && pending.has(d.id)) {
      pending.get(d.id)!(d);
      pending.delete(d.id);
    }
    if (d.method === 'Runtime.exceptionThrown') errors.push(d.params.exceptionDetails.exception?.description ?? d.params.exceptionDetails.text);
  };
  const send = (method: string, params: Record<string, unknown> = {}) =>
    new Promise<Record<string, any>>((r) => {
      pending.set(++id, r);
      ws.send(JSON.stringify({ id, method, params }));
    });
  const evaluate = async <T = any>(expression: string): Promise<T> => {
    const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text);
    return r.result?.result?.value;
  };
  await send('Runtime.enable');
  await send('Page.enable');
  return { send, evaluate, errors, close: () => ws.close() };
}

// ---------------------------------------------------------------- main

async function main() {
  rmSync(join(SCRATCH, 'data'), { recursive: true, force: true });
  rmSync(join(SCRATCH, 'library'), { recursive: true, force: true });
  mkdirSync(join(SCRATCH, 'data', 'structures'), { recursive: true });
  mkdirSync(SHOTS, { recursive: true });
  const real = join(ENGINE, 'docs', 'crash-reference', 'studio');
  for (const f of readdirSync(join(real, 'structures'))) copyFileSync(join(real, 'structures', f), join(SCRATCH, 'data', 'structures', f));
  const env = { ...process.env, STUDIO_PORT: String(PORT), ...(REAL ? {} : { STUDIO_DATA: join(SCRATCH, 'data'), STUDIO_LIBRARY: join(SCRATCH, 'library') }) };
  const vite = spawn(process.execPath, [join(PACKAGE, 'node_modules', 'vite', 'bin', 'vite.js')], { cwd: PACKAGE, env, stdio: ['ignore', 'pipe', 'pipe'] });
  procs.push(vite);
  let up = false;
  for (let i = 0; i < 60 && !up; i++) {
    await sleep(250);
    up = await fetch(`${BASE}/api/tools`).then((r) => r.ok, () => false);
  }
  if (!up) throw new Error('the studio server did not start');
  const c = await chrome();
  const go = async (hash: string, ms = 1200) => {
    await c.send('Page.navigate', { url: `${BASE}/${hash}` });
    await sleep(ms);
  };
  const shoot = async (name: string, wait = 600) => {
    for (const [w, h, tag] of [[1280, 720, '1280'], [800, 900, '800']] as const)
      for (const scheme of ['light', 'dark']) {
        await c.send('Emulation.setDeviceMetricsOverride', { width: w, height: h, deviceScaleFactor: 1, mobile: false });
        await c.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: scheme }] });
        await sleep(wait);
        const s = await c.send('Page.captureScreenshot', { format: 'png' });
        writeFileSync(join(SHOTS, `${name}-${tag}-${scheme}.png`), Buffer.from(s.result.data, 'base64'));
      }
    await c.send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 720, deviceScaleFactor: 1, mobile: false });
  };
  await go('#/library');
  await c.evaluate('new Promise((r) => { const t = () => (window.studio ? r(true) : setTimeout(t, 100)); t(); })');
  check(await c.evaluate('window.studio.tools.length') >= 30, 'window.studio lists the tools');

  if (REAL) {
    const f = await c.evaluate<string>("window.studio.list_frames({status: 'paired'}).then((l) => l.find((x) => x.id.startsWith('S02') && x.moment === 'peak crush')?.id ?? l[0].id)");
    await shoot('library');
    await go(`#/annotate/${f}`, 2000);
    await shoot('annotate');
    await go(`#/link/${f}`, 12000);
    await shoot('link', 1500);
    await go(`#/compare/${f}`, 6000);
    await shoot('compare', 1200);
    return finish(c);
  }

  // Empty state.
  await shoot('empty');
  const S = (expr: string) => c.evaluate(`(async () => { const s = window.studio; ${expr} })()`);
  const before = car(0);
  const after = car(1);
  const shot = await S("return s.create_shot({ title: 'Generated test car into a wall' })");
  check(shot.id === 'Generated-test-car-into-a-wall', `create_shot made ${shot.id}`);
  const add = (name: string, img: Buffer, moment: string) =>
    S(`return s.add_image({ name: ${JSON.stringify(name)}, data_base64: ${JSON.stringify(img.toString('base64'))}, shot: ${JSON.stringify(shot.id)}, view: 'side', moment: ${JSON.stringify(moment)} })`);
  const f0 = await add('test-car-before.png', before.image, 'before');
  const f1 = await add('test-car-after.png', after.image, 'peak crush');
  check(f0.image?.width === 1280 && f1.shot === shot.id, `add_image saved ${f0.image?.file} and ${f1.image?.file}`);
  const untraced = await S('return s.list_untraced()');
  check(untraced.length === 2, 'list_untraced lists both new frames');

  // Through WebMCP (document.modelContext): the polyfill's getTools/executeTool, as an agent's browser would.
  const mc = await c.evaluate<{ n: number; has: boolean; shots: number }>(`(async () => {
    const tools = await document.modelContext.getTools();
    const t = tools.find((x) => x.name === 'list_shots');
    const r = JSON.parse(await document.modelContext.executeTool(t, {}));
    return { n: tools.length, has: !!t, shots: r.length };
  })()`).catch((e) => ({ n: 0, has: false, shots: -1, e: String(e) }));
  check(mc.n === (await c.evaluate('window.studio.tools.length')) && mc.has && mc.shots === 1, `WebMCP: ${mc.n} tools registered (${await c.evaluate('window.studio.webmcp')}), list_shots through executeTool gives ${mc.shots} shot`);

  // Trace both frames through the same tools.
  for (const [f, shape] of [[f0, before], [f1, after]] as const) {
    await S(`await s.mark_region({ frame: '${f.id}', source: 'real', label: 'test image' }); return s.set_region_schema({ frame: '${f.id}', region: 'real', schema: 'vehicle-side', front: 'left' })`);
    const ops: [string, Record<string, unknown>][] = [];
    const base = { frame: f.id, region: 'real' };
    shape.wheels.forEach((w, k) => {
      const element = k ? 'rear_wheel' : 'front_wheel';
      ops.push(['set_keypoint', { ...base, element, x: w.c[0], y: w.c[1], part: 'c' }], ['set_keypoint', { ...base, element, x: w.rim[0], y: w.rim[1], part: 'rim' }]);
    });
    for (const [element, pts] of [['hood', shape.hood], ['bumper_front', shape.bumper], ['windscreen', shape.windscreen], ['roofline', shape.roof], ['sill', shape.sill], ['door_front_edge', shape.door]] as const)
      for (const [x, y] of pts) ops.push(['add_polyline_point', { ...base, element, x, y }]);
    ops.push(['set_keypoint', { ...base, element: 'door_sill_corner', x: shape.corner[0], y: shape.corner[1] }]);
    for (const [name, input] of ops) await S(`return s.${name}(${JSON.stringify(input)})`);
  }
  await S(`return s.set_visibility({ frame: '${f1.id}', region: 'real', element: 'hood', vis: 'blurred' })`);
  const tr = await S(`return s.get_tracing({ frame: '${f1.id}', region: 'real' })`);
  check(tr[0].instances[0].elements.hood.pts.length === 3 && tr[0].instances[0].elements.hood.vis === 'blurred', 'get_tracing returns the traced hood (3 points, blurred)');
  // Move and remove a point, then put it back.
  await S(`await s.add_polyline_point({ frame: '${f1.id}', region: 'real', element: 'roofline', x: 0.6, y: 0.37, index: 1 }); return s.remove_point({ frame: '${f1.id}', region: 'real', element: 'roofline', index: 1 })`);

  // Page tools.
  await S(`return s.open_frame({ frame: '${f1.id}' })`);
  await sleep(1200);
  await S("return s.select_element({ element: 'rear_face' })");
  const view = await S('return s.describe_view()');
  check(view.view === 'annotate' && view.frame?.id === f1.id && view.element === 'rear_face', `describe_view: ${view.view} ${view.frame?.id} element ${view.element}`);
  await shoot('annotate');

  // Link to Rubble's S02 with this frame's own camera (the scenario's side camera, set by hand).
  await S(`return s.link_scenario({ shot: '${shot.id}', scenario: 'S02' })`);
  await S(`return s.set_frame_camera({ frame: '${f1.id}', eye: [4.6, 1.1, -0.3], target: [0, 0.8, -0.3], fov_y_deg: 60 })`);
  const outline = await S(`return s.get_rubble_outline({ frame: '${f1.id}' })`);
  check(outline.own_camera && outline.camera === `frame_${f1.id}` && outline.instance.elements.front_wheel?.c, `get_rubble_outline: ${outline.scenario} ${outline.moment} through ${outline.camera}`);
  const table = await S(`return s.get_comparison({ frame: '${f1.id}' })`);
  const row = (id: string) => table.rows.find((r: { id: string }) => r.id === id)?.values;
  const fold = row('hood_fold')?.real;
  const setback = row('wheel_setback')?.real;
  check(table.frame?.id === 'wheelbase' && table.before_frame === f0.id, `get_comparison: frame ${table.frame?.id}, before ${table.before_frame}`);
  // The drawn crush (worked by hand): after, the hood (320,360) (370,335) (470,380) has its middle point 31.4 px above its chord;
  // over the 210.95 px chord before that is 0.1488. The front wheel moves back 45 px of a 620 px wheelbase.
  check(fold && Math.abs(fold.v - 0.1488) < 0.001 && fold.vis === 'blurred', `hood fold height ${fold?.v?.toFixed(4)} (by hand: 0.1488), flagged ${fold?.vis}`);
  check(setback && Math.abs(setback.v - 45 / 620) < 1e-3, `front wheel setback ${setback?.v?.toFixed(4)} wheelbases (drawn: ${(45 / 620).toFixed(4)})`);
  check(row('hood_fold')?.rubble != null, `Rubble's hood fold ${row('hood_fold')?.rubble?.v?.toFixed(3)} in the same table`);

  // The HTTP route, as a script without the page would use it.
  const http = await fetch(`${BASE}/api/tool/list_frames`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ shot: shot.id }) }).then((r) => r.json());
  check(http.ok && http.result.length === 2 && http.result.every((f: { status: string[] }) => f.status.includes('linked')), 'HTTP: POST /api/tool/list_frames: both frames traced and linked');
  const bad = await fetch(`${BASE}/library/..%2F..%2Fpackage.json`).then((r) => r.status);
  check(bad === 403, `path traversal on /library refused (${bad})`);
  const simple = await fetch(`${BASE}/api/tool/list_shots`, { method: 'POST', headers: { 'content-type': 'text/plain' }, body: '{}' }).then((r) => r.status);
  check(simple === 415, `a non-JSON (cross-site form style) post refused (${simple})`);

  await go('#/library');
  await shoot('library');
  await go(`#/link/${f1.id}`, 2500);
  await shoot('link', 1200);
  await go(`#/compare/${f1.id}`, 2500);
  await shoot('compare', 1000);
  await go(`#/annotate/${f1.id}`, 1500);
  await c.evaluate("document.querySelector('.topbar .btn').click()");
  await shoot('help', 500);
  check(c.errors.length === 0, `no page exceptions${c.errors.length ? `: ${c.errors.join(' | ')}` : ''}`);
  return finish(c);
}

function finish(c: { close: () => void }) {
  c.close();
  console.log(`screenshots: ${SHOTS}`);
  return failures;
}

main()
  .then((n) => {
    console.log(n ? `${n} check(s) failed` : 'smoke test passed');
    process.exitCode = n ? 1 : 0;
  })
  .catch((e) => {
    console.error(e);
    process.exitCode = 1;
  })
  .finally(() => {
    for (const p of procs) p.kill();
    setTimeout(() => process.exit(), 300);
  });
