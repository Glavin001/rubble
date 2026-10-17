// Headless Chrome benchmark runner for when the in-app browser pane is hidden.
// Uses system Chrome (Playwright channel 'chrome', WebGPU on the real adapter) and one
// fresh page per run, so runs stay serial and each starts cold.
//
//   node tools/headless-bench.mjs demo "scene=ring110k&bench=only" "scene=ring110k&bench=only&np=split"
//   node tools/headless-bench.mjs ref paper large      # pinned three-avbd bench3d tiers
//
// Serve demo.html on 127.0.0.1:8765 and the reference Vite server on 127.0.0.1:5317 first.
// Prints one JSON line per run. Discard runs while another GPU client is busy.
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(path.resolve(here, '../../../../web/package.json'));
const { chromium } = require('playwright');

const [mode, ...items] = process.argv.slice(2);
const browser = await chromium.launch({
  channel: 'chrome', headless: true,
  args: ['--enable-unsafe-webgpu', '--disable-background-timer-throttling', '--disable-renderer-backgrounding'],
});

async function demo(q) {
  const page = await browser.newPage();
  const result = new Promise(resolve => page.on('console', m => {
    const t = m.text();
    if (t.startsWith('BENCH ') || t.startsWith('REGRESSION ')) resolve(JSON.parse(t.slice(t.indexOf(' ') + 1)));
  }));
  await page.goto(`http://127.0.0.1:8765/demo.html?${q}`);
  const r = await Promise.race([result, new Promise(resolve => setTimeout(() => resolve(null), 900000))]);
  const status = await page.evaluate(() => document.querySelector('#status')?.textContent?.slice(0, 400));
  await page.close();
  if (!r) return { q, timeout: true, status };
  const s = r.timingSamples;
  return {
    q, wallMs: r.wallMs, phases: r.phases, contacts: r.contacts ?? r.counters?.contact, awake: r.awake, colors: r.colors,
    overflow: r.overflow, rounds: s?.rounds?.map(x => x.medianMs), workloadEnd: s?.workload?.end,
    ...(q.includes('check=') ? { status } : {}),
  };
}

async function ref(tier) {
  const page = await browser.newPage();
  await page.goto('http://127.0.0.1:5317/bench3d.html');
  await page.waitForSelector('#run');
  await page.evaluate(t => { for (const e of document.querySelectorAll('input[name=tier]')) e.checked = e.value === t; }, tier);
  await page.click('#run');
  await page.waitForFunction(() => /^(Done|Failed)/.test(document.querySelector('#status').textContent), null, { timeout: 900000, polling: 1000 });
  const rows = await page.evaluate(() => [...document.querySelectorAll('#results tr')].map(r => [...r.children].map(c => c.textContent)));
  const status = await page.evaluate(() => document.querySelector('#status').textContent);
  await page.close();
  return { tier, status, rows };
}

for (const item of items) console.log(JSON.stringify(mode === 'ref' ? await ref(item) : await demo(item)));
await browser.close();
// Playwright can keep node alive for minutes after close; exit explicitly.
process.exit(0);
