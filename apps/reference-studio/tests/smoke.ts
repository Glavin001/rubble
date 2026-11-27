// Smoke test: the built static site (dist/) in headless Chrome, driven through window.studio
// and WebMCP as an agent would, through the whole recreate flow.
//
//   npm run build && npm run smoke     (from apps/reference-studio; needs Chrome)
//   npm run smoke -- --webgpu          the same flow on the page's WebGPU (Chrome with its GPU): run it under the
//                                      GPU lock, python3 engine/scripts/with-gpu-lock.py --limit 300 -- npm run smoke -- --webgpu
//   npm run smoke -- --real [--webgpu] S02 recreated as data with the repo workspace's S02 frames (third-party
//                                      footage: screenshots go to .../step2/real*, not for sharing)
//
// No server but a static file server for dist/ on 127.0.0.1:8771. Storage is the origin
// private file system (?workspace=opfs) in a fresh Chrome profile. Headless Chrome here has
// no GPU, so the run uses the wasm CPU reference (?cpu-fallback: the studio labels it
// "CPU fallback (tests)"); the WebGPU path is the same Rust behind the same tools and is
// checked by hand in a browser with a GPU (README "Checks").
//
// It draws two test images (a car silhouette before and after a frontal crash: no
// third-party footage), creates a shot, traces both, creates a data scenario (one generated
// sedan, one wall), simulates it, pins both images at times of the recording with a camera,
// links the shot, reads Rubble's outline and the comparison table (checked against values
// worked by hand), ranks the designs against the tracing, moves the wall live, and saves
// screenshots of every view at 1280x720 and 800 px wide, light and dark, to
// engine/out/reference-studio/screenshots/step2/generated/. Exits 1 on the first failed check.

import { spawn, type ChildProcess } from 'node:child_process';
import { createReadStream, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize } from 'node:path';
import { deflateSync } from 'node:zlib';
import { ENGINE, PACKAGE } from '../scripts/node-workspace.ts';

const REAL = process.argv.includes('--real');
const WEBGPU = process.argv.includes('--webgpu');
const PORT = 8771;
const BASE = `http://127.0.0.1:${PORT}`;
const OUT = join(ENGINE, 'out', 'reference-studio');
const SCRATCH = join(OUT, 'smoke');
const SHOTS = join(OUT, 'screenshots', 'step2', `${REAL ? 'real' : 'generated'}${WEBGPU ? '-webgpu' : ''}`);
const DIST = join(PACKAGE, 'dist');
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const procs: ChildProcess[] = [];
let failures = 0;

function check(ok: unknown, what: string): void {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${what}`);
  if (!ok) failures++;
}

// ---------------------------------------------------------------- a static file server

const TYPES: Record<string, string> = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm', '.map': 'application/json', '.wgsl': 'text/plain', '.txt': 'text/plain', '.png': 'image/png' };
function serve(): Promise<() => void> {
  const server = createServer((req, res) => {
    const path = normalize(decodeURIComponent(new URL(req.url ?? '/', BASE).pathname)).replace(/^(\.\.[/\\])+/, '');
    let p = join(DIST, path);
    if (!p.startsWith(DIST)) return res.writeHead(403).end();
    if (existsSync(p) && statSync(p).isDirectory()) p = join(p, 'index.html');
    if (!existsSync(p)) return res.writeHead(404).end();
    res.writeHead(200, { 'content-type': TYPES[extname(p)] ?? 'application/octet-stream' });
    createReadStream(p).pipe(res);
  });
  return new Promise((r) => server.listen(PORT, '127.0.0.1', () => r(() => server.close())));
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
    rear: [[1100, 400], [1100, 520]].map((p) => n(p as P)),
    wheels: wheels.map(([cx, cy, r]) => ({ c: n([cx, cy]), rim: n([cx + r, cy]) })),
  };
}

// ---------------------------------------------------------------- Chrome over CDP

async function chrome() {
  const port = 9400 + Math.floor(Math.random() * 400);
  const profile = join(SCRATCH, 'chrome-profile');
  mkdirSync(profile, { recursive: true });
  procs.push(spawn(CHROME, ['--headless=new', ...(WEBGPU ? ['--enable-unsafe-webgpu', '--disable-background-timer-throttling', '--disable-renderer-backgrounding'] : ['--disable-gpu', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']), '--no-first-run', '--no-default-browser-check', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, '--window-size=1280,720', 'about:blank'], { stdio: 'ignore' }));
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
  if (!existsSync(join(DIST, 'index.html')) || !existsSync(join(DIST, 'rubble', 'world-wasm'))) throw new Error('no dist/ with the engine: npm run engine && npm run designs && npm run build');
  mkdirSync(SHOTS, { recursive: true });
  rmSync(join(SCRATCH, 'chrome-profile'), { recursive: true, force: true });
  const stop = await serve();
  procs.push({ kill: stop } as unknown as ChildProcess);
  const c = await chrome();
  const q = WEBGPU ? '?workspace=opfs' : '?workspace=opfs&cpu-fallback';
  const executor = WEBGPU ? 'webgpu' : 'cpu-fallback';
  const go = async (hash: string, ms = 1200) => {
    await c.send('Page.navigate', { url: `${BASE}/${q}${hash}` });
    await sleep(ms);
    await c.evaluate('new Promise((r) => { const t = () => (window.studio && document.querySelector(".topbar .badge") ? r(true) : setTimeout(t, 100)); t(); })');
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
  await go('#/library', 2500);
  const ws = await c.evaluate<string>("document.querySelector('[data-testid=workspace]')?.textContent ?? ''");
  check(/this browser only/.test(ws), `the workspace is this browser's private storage (${ws})`);
  check((await c.evaluate<number>('window.studio.tools.length')) >= 60, `window.studio lists the tools (${await c.evaluate('window.studio.tools.length')})`);
  const S = <T = any>(expr: string) => c.evaluate<T>(`(async () => { const s = window.studio; ${expr} })()`);
  if (REAL) {
    await seed(c, (rel) => rel.startsWith('structures/') || rel === 'scenarios/s02-data.json' || rel === 'shots/S02.json' || rel.startsWith('frames/S02') || rel.startsWith('library/S02/'));
    await go('#/library', 2500);
    await S("return s.link_scenario({ shot: 'S02', scenario: 's02-data' })");
    const o = await S("return s.open_scene({ scenario: 's02-data', shot: 'S02' })");
    check(o.bodies[0]?.design === 'suv_4x4', `S02 as data opened: ${o.bodies[0]?.nodes} nodes, first contact ${o.first_contact_ms?.toFixed(1)} ms`);
    const r = await S('return s.simulate({})');
    check(r.fresh && r.executor === executor, `simulated on ${r.executor}: ${r.samples} samples in ${r.wall_s?.toFixed(1)} s`);
    // The real side frame at peak crush, through the scenario's side camera (a first guess: its crop is the real half).
    const f = 'S02_2m12s_side_01';
    await S(`await s.set_frame_camera({ frame: '${f}', eye: [4.6, 1.1, -0.3], target: [0, 0.8, -0.3], fov_y_deg: 60, region: 'real' }); await s.jump_to_pin({ frame: '${f}' }); await s.seek({ ms: 70 }); return s.pin_image({ frame: '${f}' })`);
    await sleep(1200);
    await shoot('s02-real-image', 1000);
    const t = await S(`return s.get_comparison({ frame: '${f}' })`);
    check(t.rubble && !t.rubble.error, `get_comparison of ${f} with Rubble at ${t.rubble?.at_ms} ms (${t.rubble?.recording?.executor})`);
    console.log(JSON.stringify(t.rows?.map((x: { id: string; values: Record<string, { v: number } | null> }) => [x.id, Object.fromEntries(Object.entries(x.values).map(([k, v]) => [k, v ? +v.v.toFixed(3) : null]))])));
    return finish(c);
  }
  await shoot('empty');

  // A shot from a drawn silhouette pair, traced.
  const before = car(0);
  const after = car(1);
  const shot = await S("return s.create_shot({ title: 'Generated test car into a wall' })");
  check(shot.id === 'Generated-test-car-into-a-wall', `create_shot made ${shot.id}`);
  const add = (name: string, img: Buffer, moment: string) =>
    S(`return s.add_image({ name: ${JSON.stringify(name)}, data_base64: ${JSON.stringify(img.toString('base64'))}, shot: ${JSON.stringify(shot.id)}, view: 'side', moment: ${JSON.stringify(moment)} })`);
  const f0 = await add('test-car-before.png', before.image, 'before');
  const f1 = await add('test-car-after.png', after.image, 'peak crush');
  check(f0.image?.width === 1280 && f1.shot === shot.id, `add_image saved ${f0.image?.file} and ${f1.image?.file} in the workspace`);
  const mc = await c.evaluate<{ n: number; has: boolean; shots: number }>(`(async () => {
    const tools = await document.modelContext.getTools();
    const t = tools.find((x) => x.name === 'list_shots');
    const r = JSON.parse(await document.modelContext.executeTool(t, {}));
    return { n: tools.length, has: !!t, shots: r.length };
  })()`).catch((e) => ({ n: 0, has: false, shots: -1, e: String(e) }));
  check(mc.n === (await c.evaluate('window.studio.tools.length')) && mc.has && mc.shots === 1, `WebMCP: ${mc.n} tools registered (${await c.evaluate('window.studio.webmcp')}), list_shots through executeTool gives ${mc.shots} shot`);
  for (const [f, shape] of [[f0, before], [f1, after]] as const) {
    await S(`await s.mark_region({ frame: '${f.id}', source: 'real', label: 'test image' }); return s.set_region_schema({ frame: '${f.id}', region: 'real', schema: 'vehicle-side', front: 'left' })`);
    const ops: [string, Record<string, unknown>][] = [];
    const base = { frame: f.id, region: 'real' };
    shape.wheels.forEach((w, k) => {
      const element = k ? 'rear_wheel' : 'front_wheel';
      ops.push(['set_keypoint', { ...base, element, x: w.c[0], y: w.c[1], part: 'c' }], ['set_keypoint', { ...base, element, x: w.rim[0], y: w.rim[1], part: 'rim' }]);
    });
    for (const [element, pts] of [['hood', shape.hood], ['bumper_front', shape.bumper], ['windscreen', shape.windscreen], ['roofline', shape.roof], ['sill', shape.sill], ['door_front_edge', shape.door], ['rear_face', shape.rear]] as const)
      for (const [x, y] of pts) ops.push(['add_polyline_point', { ...base, element, x, y }]);
    ops.push(['set_keypoint', { ...base, element: 'door_sill_corner', x: shape.corner[0], y: shape.corner[1] }]);
    for (const [name, input] of ops) await S(`return s.${name}(${JSON.stringify(input)})`);
  }
  await S(`return s.set_visibility({ frame: '${f1.id}', region: 'real', element: 'hood', vis: 'blurred' })`);
  await S(`return s.open_frame({ frame: '${f1.id}' })`);
  await sleep(1200);
  await shoot('annotate');

  // A data scenario (one car, one wall), opened in the Scene view and simulated.
  const sc = await S("return s.create_scenario({ title: 'Test car into a wall' })");
  check(sc.id === 'test-car-into-a-wall' && sc.bodies.length === 1 && sc.boxes.length === 1, `create_scenario: ${sc.id}, ${sc.bodies[0].design} at ${sc.bodies[0].speed_kmh} km/h, a wall`);
  await S(`return s.link_scenario({ shot: '${shot.id}', scenario: '${sc.id}' })`);
  const opened = await S(`return s.open_scene({ scenario: '${sc.id}', shot: '${shot.id}' })`);
  check(opened.bodies[0]?.surface_triangles > 100 && opened.first_contact_ms > 0, `open_scene: ${opened.bodies[0]?.nodes} nodes, ${opened.bodies[0]?.surface_triangles} surface triangles, first contact ${opened.first_contact_ms?.toFixed(1)} ms (Rust), placed in ${opened.place_ms?.toFixed(1)} ms`);
  await sleep(800);
  await S("return s.select_object({ kind: 'box', name: 'wall' })");
  await shoot('scene-setup');
  // Live edit: the wall 1 m further moves the first contact by 1 m at 50 km/h (72 ms).
  const t0 = Date.now();
  await S(`return s.set_box({ scenario: '${sc.id}', box: 'wall', centre: [0, 1, 4.6] })`);
  let moved: any = null;
  for (let i = 0; i < 50; i++) {
    moved = await S('return s.describe_scene()');
    if (Math.abs(moved.first_contact_ms - opened.first_contact_ms) > 1) break;
    await sleep(50);
  }
  check(Math.abs(moved.first_contact_ms - opened.first_contact_ms - 72) < 1, `set_box re-placed the scene live: first contact ${opened.first_contact_ms.toFixed(1)} -> ${moved.first_contact_ms.toFixed(1)} ms (Rust ${moved.place_ms?.toFixed(1)} ms, ${Date.now() - t0} ms round trip, no respawn)`);
  await S('return s.undo_scene()');
  await sleep(300);
  const undone = await S('return s.describe_scene()');
  check(Math.abs(undone.first_contact_ms - opened.first_contact_ms) < 0.01, 'undo_scene puts the wall back');
  const sim = await S('return s.simulate({})');
  check(sim.present && sim.fresh && sim.executor === executor && sim.samples > 100, `simulate (${executor}${WEBGPU ? '' : ', labelled'}): ${sim.samples} samples, ${sim.duration_s?.toFixed(2)} s simulated in ${sim.wall_s?.toFixed(1)} s`);
  const rep = await S('return s.get_report()');
  check(rep.report?.results?.every((r: { pass: boolean; asserted: boolean }) => r.pass || !r.asserted), `get_report: ${rep.report?.results?.map((r: { name: string; pass: boolean }) => `${r.name} ${r.pass ? 'pass' : 'FAIL'}`).join(', ')}`);
  const cam = { eye: [6.0, 1.0, 1.0], target: [0, 0.6, 1.0], fov_y_deg: 45 };
  await S(`await s.seek({ ms: -10 }); await s.set_scene_camera(${JSON.stringify(cam)}); return s.pin_image({ frame: '${f0.id}' })`);
  await S(`await s.seek({ ms: 72.5 }); return s.pin_image({ frame: '${f1.id}' })`);
  const pins = await S(`return s.list_pins({ shot: '${shot.id}' })`);
  check(pins.length === 2 && pins.every((p: { pinned: boolean }) => p.pinned) && Math.abs(pins.find((p: { frame: string }) => p.frame === f1.id).at_ms - 72.5) < 0.01, `pin_image stored both times and cameras: ${pins.map((p: { frame: string; at_ms: number }) => `${p.frame} ${p.at_ms} ms`).join(', ')}`);
  await S(`await s.seek({ ms: 0 }); return s.jump_to_pin({ frame: '${f1.id}' })`);
  const jumped = await S('return s.describe_scene()');
  check(Math.abs(jumped.time_ms - 72.5) < 0.01 && jumped.camera.fov_y_deg === 45 && jumped.image === f1.id, `jump_to_pin: ${jumped.time_ms} ms, fov ${jumped.camera.fov_y_deg}, image ${jumped.image}`);
  await sleep(1000);
  await shoot('scene-review-image', 800);
  const st = await S('return s.get_recording_status()');
  check(st.fresh && st.start_ms < -50 && st.end_ms > 400, `get_recording_status: fresh, ${st.start_ms.toFixed(0)} to ${st.end_ms.toFixed(0)} ms`);

  // Rubble's outline at the pin through the pinned camera, and the comparison table.
  const outline = await S(`return s.get_rubble_outline({ frame: '${f1.id}' })`);
  check(outline.pinned && outline.own_camera && Math.abs(outline.at_ms - 72.5) < 0.01 && outline.instance.elements.front_wheel?.c && outline.recording.executor === executor, `get_rubble_outline: ${outline.scenario} at ${outline.at_ms} ms (pinned) through the frame's camera, from the ${outline.recording.executor} recording`);
  const table = await S(`return s.get_comparison({ frame: '${f1.id}' })`);
  const row = (id: string) => table.rows?.find((r: { id: string }) => r.id === id)?.values;
  const fold = row('hood_fold')?.real;
  const setback = row('wheel_setback')?.real;
  check(table.frame?.id === 'wheelbase' && table.before_frame === f0.id, `get_comparison: frame ${table.frame?.id}, before ${table.before_frame}`);
  // The drawn crush (worked by hand): after, the hood (320,360) (370,335) (470,380) has its middle point 31.4 px above its chord;
  // over the 210.95 px chord before that is 0.1488. The front wheel moves back 45 px of a 620 px wheelbase.
  check(fold && Math.abs(fold.v - 0.1488) < 0.001 && fold.vis === 'blurred', `hood fold height ${fold?.v?.toFixed(4)} (by hand: 0.1488), flagged ${fold?.vis}`);
  check(setback && Math.abs(setback.v - 45 / 620) < 1e-3, `front wheel setback ${setback?.v?.toFixed(4)} wheelbases (drawn: ${(45 / 620).toFixed(4)})`);
  check(row('hood_fold')?.rubble != null && table.rubble?.pinned, `Rubble's hood fold ${row('hood_fold')?.rubble?.v?.toFixed(3)} in the same table, at its pin (${table.rubble?.at_ms} ms)`);

  // Matching a design from the "before" tracing (Rust ranks every generated design).
  const ranked = await S(`return s.list_design_matches({ frame: '${f0.id}' })`);
  check(Array.isArray(ranked) && ranked.length >= 6 && ranked[0].distance <= ranked[1].distance && ranked[0].rows.length === 4, `list_design_matches: ${ranked.slice(0, 3).map((m: { design: string; distance: number }) => `${m.design} ${m.distance.toFixed(3)}`).join(', ')}`);

  // Re-simulating after a setup change keeps the pins, flagged.
  await S(`return s.set_body({ scenario: '${sc.id}', body: 'car', speed_kmh: 45 })`);
  await sleep(400);
  const stale = await S('return s.get_recording_status()');
  check(stale.present && !stale.fresh, 'a setup edit marks the recording stale');
  await S('return s.simulate({})');
  const flagged = await S(`return s.get_rubble_outline({ frame: '${f1.id}' })`);
  check(flagged.check === true && flagged.pinned, 'after re-simulating, the pinned image keeps its time and camera, flagged "check after re-simulation"');
  await S(`await s.seek({ ms: 72.5 }); await s.select_object({ kind: 'body', name: 'car' }); return s.jump_to_pin({ frame: '${f1.id}' })`);
  await sleep(800);
  await shoot('scene-review-flagged', 800);
  await S(`await s.select_image({ frame: '${f0.id}' }); await s.set_mode({ mode: 'setup' }); return true`);
  await sleep(600);
  await S("return s.select_object({ kind: 'body', name: 'car' })");
  await shoot('scene-setup-body', 800);

  // S02 exported as data, in the Scene view: editing its wall, and mid-crash.
  const s02 = await S("return s.create_scenario({ id: 's02-data', from: 'S02', title: 'S02 as data: SUV into a full-width rigid wall, 50 km/h' })");
  check(s02.bodies[0].design === 'suv_4x4' && s02.bodies[0].speed_kmh === 50, `create_scenario from S02: ${s02.bodies[0].design} at ${s02.bodies[0].speed_kmh} km/h, wall at z ${s02.boxes[0].centre[2].toFixed(3)}`);
  await S("await s.open_scene({ scenario: 's02-data' }); return s.set_scene_camera({ eye: [4.6, 1.1, -0.3], target: [0, 0.8, -0.3], fov_y_deg: 60 })");
  const s02sim = await S('return s.simulate({})');
  check(s02sim.fresh && s02sim.executor === executor, `S02 as data simulated (${s02sim.executor}): ${s02sim.samples} samples in ${s02sim.wall_s?.toFixed(1)} s`);
  const s02rep = await S('return s.get_report()');
  check(s02rep.report?.results?.every((r: { pass: boolean }) => r.pass), `S02 as data, law criteria: ${s02rep.report?.results?.map((r: { name: string; value: number; pass: boolean }) => `${r.name} ${r.value?.toPrecision(3)} ${r.pass ? 'pass' : 'FAIL'}`).join(', ')}; crush peak ${s02rep.report?.metrics?.['car.crush_peak']?.toFixed(3)} m`);
  await S("await s.seek({ ms: 70 }); return s.set_mode({ mode: 'review' })");
  await sleep(800);
  await shoot('s02-midcrash', 800);
  await S("await s.set_mode({ mode: 'setup' }); await s.select_object({ kind: 'box', name: 'box0' }); return s.set_scene_camera({ eye: [5.5, 3.2, -3.5], target: [0, 0.8, 0.8], fov_y_deg: 55 })");
  await sleep(800);
  await shoot('s02-setup-box', 800);

  await go('#/library');
  await shoot('library');
  await go(`#/link/${f1.id}`, 2500);
  await shoot('link', 1500);
  await go(`#/compare/${f1.id}`, 2500);
  await shoot('compare', 1500);
  check(c.errors.length === 0, `no page exceptions${c.errors.length ? `: ${c.errors.join(' | ')}` : ''}`);
  return finish(c);
}

/** Copy repo workspace files (as `keep` chooses) into the page's OPFS, as a user copying a folder would. */
async function seed(c: { evaluate: <T = any>(e: string) => Promise<T> }, keep: (rel: string) => boolean) {
  const root = join(PACKAGE, 'workspace');
  const walk = (dir: string): string[] => readdirSync(join(root, dir)).flatMap((f) => (statSync(join(root, dir, f)).isDirectory() ? walk(`${dir}${f}/`) : [`${dir}${f}`]));
  for (const rel of walk('').filter(keep)) {
    const b64 = readFileSync(join(root, rel)).toString('base64');
    await c.evaluate(`(async () => {
      let d = await navigator.storage.getDirectory();
      const parts = ${JSON.stringify(rel)}.split('/');
      for (const p of parts.slice(0, -1)) d = await d.getDirectoryHandle(p, { create: true });
      const w = await (await d.getFileHandle(parts[parts.length - 1], { create: true })).createWritable();
      await w.write(Uint8Array.from(atob(${JSON.stringify(b64)}), (x) => x.charCodeAt(0)));
      await w.close();
    })()`);
  }
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
