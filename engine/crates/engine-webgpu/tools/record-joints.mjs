// Record the Rust-exported joint scenes running in Chrome WebGPU (Slang→WGSL).
//
//   cargo run -p engine-scenes --release --bin scene_export
//   cargo run -p engine-webgpu --release --bin wgsl_export
//   node engine/crates/engine-webgpu/tools/record-joints.mjs [scene ...]
//
// Serves this crate on a free port, drives each scene (orbit, grab, fire,
// motor toggle) and writes test-results/joints/<scene>.mp4 (webm if ffmpeg
// is missing) plus the page's live TPS / step-time readout per scene.

import { createRequire } from 'node:module';
import { createServer } from 'node:http';
import { readFile, mkdir, rename, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const crate = path.resolve(here, '..');
const repo = path.resolve(crate, '../../..');
const require = createRequire(path.join(repo, 'web/package.json'));
const { chromium } = require('playwright');

const MIME = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.wgsl': 'text/plain', '.json': 'application/json', '.bin': 'application/octet-stream' };
const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://x');
    const file = path.join(crate, decodeURIComponent(url.pathname));
    if (!file.startsWith(crate)) throw new Error('outside crate');
    const body = await readFile(file);
    res.writeHead(200, { 'content-type': MIME[path.extname(file)] || 'application/octet-stream' });
    res.end(body);
  } catch {
    res.writeHead(404);
    res.end();
  }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

const W = 1280, H = 720;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function orbit(page, dx, dy, steps = 30) {
  await page.mouse.move(W / 2, H / 2);
  await page.mouse.down();
  for (let i = 1; i <= steps; i++) {
    await page.mouse.move(W / 2 + (dx * i) / steps, H / 2 + (dy * i) / steps);
    await sleep(16);
  }
  await page.mouse.up();
}

async function grabDrag(page, from, to, ms = 1500) {
  await page.keyboard.down('Alt');
  await page.mouse.move(from[0], from[1]);
  await page.mouse.down();
  await sleep(300); // async pick
  const steps = Math.max(10, ms / 30);
  for (let i = 1; i <= steps; i++) {
    await page.mouse.move(from[0] + ((to[0] - from[0]) * i) / steps, from[1] + ((to[1] - from[1]) * i) / steps);
    await sleep(30);
  }
  await sleep(400);
  await page.mouse.up();
  await page.keyboard.up('Alt');
}

const SCRIPTS = {
  async joints(page) {
    await sleep(5000);
    await grabDrag(page, [W * 0.5, H * 0.55], [W * 0.45, H * 0.3]);
    await sleep(1500);
    await page.keyboard.press('m');
    await sleep(2000);
    await page.keyboard.press('m');
    await orbit(page, -260, 40);
    await page.keyboard.press('f');
    await sleep(1200);
    await page.keyboard.press('f');
    await sleep(3000);
    await orbit(page, 420, 0, 40);
    await sleep(4000);
  },
  async ragdolls(page) {
    await sleep(4000);
    await grabDrag(page, [W * 0.5, H * 0.6], [W * 0.5, H * 0.2], 2000);
    await sleep(1000);
    await page.keyboard.press('f');
    await sleep(4000);
  },
  async ragdolls10k(page) {
    await sleep(9000);
    await page.keyboard.press('f');
    await sleep(4000);
  },
  async hingechains10k(page) {
    await sleep(9000);
    await orbit(page, 200, 30);
    await sleep(3000);
  },
  async motorfield10k(page) {
    await sleep(5000);
    await page.keyboard.press('m');
    await sleep(2500);
    await page.keyboard.press('m');
    await sleep(4000);
  },
};

const out = path.join(repo, 'test-results/joints');
await mkdir(out, { recursive: true });
const scenes = process.argv.slice(2).length ? process.argv.slice(2) : Object.keys(SCRIPTS);
const browser = await chromium.launch({
  headless: true,
  channel: existsSync('/Applications/Google Chrome.app') ? 'chrome' : undefined,
  args: ['--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=metal'],
});
const results = {};
try {
  for (const scene of scenes) {
    const ctx = await browser.newContext({ viewport: { width: W, height: H }, recordVideo: { dir: out, size: { width: W, height: H } } });
    const page = await ctx.newPage();
    const errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    await page.goto(`${base}/demo.html?scene=${scene}&sleep=on`);
    await page.waitForFunction(() => /step \d+/.test(document.getElementById('status').textContent), null, { timeout: 120000 });
    const err = await page.$eval('#err', (e) => e.textContent);
    if (err) throw new Error(`${scene}: ${err}`);
    await SCRIPTS[scene](page);
    const status = await page.$eval('#status', (e) => e.textContent);
    const rates = await page.evaluate(() => document.body.innerText.split('\n').filter((l) => /TPS|FPS/.test(l)).join(' | '));
    results[scene] = { status, rates, errors };
    console.log(`${scene}: ${rates}\n  ${status}${errors.length ? `\n  errors: ${errors.join('; ')}` : ''}`);
    const video = page.video();
    await ctx.close();
    const webm = path.join(out, `${scene}.webm`);
    await rename(await video.path(), webm);
    try {
      execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-i', webm, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', path.join(out, `${scene}.mp4`)]);
      await rm(webm);
      console.log(`  -> ${path.join(out, `${scene}.mp4`)}`);
    } catch {
      console.log(`  -> ${webm} (ffmpeg unavailable)`);
    }
  }
} finally {
  await browser.close();
  server.close();
}
