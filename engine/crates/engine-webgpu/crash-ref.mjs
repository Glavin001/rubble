// The crash reference suite in the crash lab (softbody.html?solver=explicit&crashRef=S02
// &run=0&contact=edge): the scenario's mod folders, its controls (contact setting, run,
// its own cameras), the reference footage's links and the criteria panel.
//
// Nothing about a scenario is here: the registry, the fleet, the sampling and the report
// are Rust (engine-world-wasm crash_ref.rs over engine_softbody::crash_reference). This
// module fetches the files Rust asks for, sends the id, calls `crashSample` at the
// sample steps Rust names and draws the report Rust returns.

import { loadMod } from './beamng-mod.mjs';

/** The registry (Rust's `crashReferenceRegistry`, parsed) and the scenario `id` in it. */
export function crashRefScenario(registry, id) {
  const s = registry.scenarios.find((x) => x.id.toLowerCase() === String(id).toLowerCase());
  if (!s) throw new Error(`no crash reference scenario ${id}: ${registry.scenarios.map((x) => x.id).join(', ')}`);
  return s;
}

/**
 * The mod folders of the scenario's `folderDesigns` (the site build's `__mods/<design>_new`,
 * named by `__mods/designs.json` when it is there): each loaded with `loadMod` (the first
 * is the body the page draws), and the files Rust builds each template from
 * (`[[path, text]]`: its `.jbeam` parts, its `.pc` and `crash.json`).
 */
export async function crashRefFolders(scenario) {
  let listed = [];
  try { const r = await fetch('/__mods/designs.json'); if (r.ok) listed = await r.json(); } catch { /* not listed: the default folder */ }
  const mods = [], designs = {};
  for (const d of scenario.folderDesigns) {
    const row = listed.find((x) => x.name === d);
    const folder = row?.folder ? `/__mods/${row.folder}` : `/__mods/${d}_new`;
    const mod = await loadMod(folder, { pc: row?.pc ?? null });
    const crash = mod.list.find((f) => f.endsWith('crash.json'));
    const files = [...mod.files];
    if (mod.pcName && mod.pc != null) files.push([mod.pcName, mod.pc]);
    if (crash) { const r = await fetch(`${folder}/${crash}`); if (!r.ok) throw new Error(`${folder}/${crash}: ${r.status}`); files.push([crash, await r.text()]); }
    designs[d] = files;
    mods.push({ design: d, folder, pc: mod.pcName, mod });
  }
  return { mods, designs };
}

/** The orbit camera (crash-lab.mjs `cams`: yaw, pitch, dist about a fixed target) that puts
 * the eye at a scenario camera's `eye`, looking at its `target` (world coordinates). */
export function orbitOf(cam) {
  const d = cam.eye.map((x, i) => x - cam.target[i]);
  const dist = Math.hypot(...d);
  return { yaw: Math.atan2(d[0], d[2]), pitch: Math.asin(d[1] / dist), dist, target: cam.target.slice() };
}

/** The lab's extra cameras (crash-lab.mjs `ctx.cams`): each of the scenario's cameras as a
 * fixed orbit, named `ref:<name>`. */
export function crashRefCams(scenario) {
  return Object.fromEntries(scenario.cameras.map((c) => [`ref:${c.name}`, orbitOf(c)]));
}

const CSS = `
#crashRef { position: absolute; top: 118px; right: 16px; width: min(440px, calc(100vw - 32px)); max-height: calc(100vh - 260px); overflow: auto; background: rgba(10, 16, 30, 0.86); border: 1px solid rgba(160, 190, 255, 0.16); border-radius: 10px; padding: 10px 12px; font: 12px/1.4 system-ui, sans-serif; color: #dfe6f3; backdrop-filter: blur(8px); z-index: 5; }
#crashRef h3 { margin: 0 0 4px; font-size: 13px; letter-spacing: 0.02em; }
#crashRef .r { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 5px 0; }
#crashRef button, #crashRef select { background: #18233a; color: #e6ecf7; border: 1px solid #2c3b5c; border-radius: 6px; padding: 2px 7px; font: inherit; cursor: pointer; }
#crashRef button.on { background: #2b4f8a; border-color: #4a78c4; }
#crashRef label { color: #9fb0cf; }
#crashRef a { color: #8fc1ff; }
#crashRef table { width: 100%; border-collapse: collapse; margin-top: 6px; }
#crashRef td { padding: 3px 4px; vertical-align: top; border-top: 1px solid #1f2b45; }
#crashRef td.v { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
#crashRef .m { padding: 1px 6px; border-radius: 4px; font-weight: 700; font-size: 11px; white-space: nowrap; }
#crashRef .pass { background: #1d5a37; } #crashRef .fail { background: #8a1f2b; }
#crashRef .in { background: #26405f; } #crashRef .out { background: #5a4a1d; } #crashRef .wait { background: #2a3344; color: #9fb0cf; }
#crashRef details { color: #aab8d4; } #crashRef summary { cursor: pointer; }
#crashRef .muted { color: #8d9bb8; }
.lab-clean #crashRef { display: none !important; }
`;

const fmt = (v) => (v == null || !Number.isFinite(v) ? '—' : Math.abs(v) >= 1000 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(2) : v.toFixed(3));
const band = (c) => (c.lo != null && c.hi != null ? `[${c.lo}, ${c.hi}]` : c.lo != null ? `≥ ${c.lo}` : c.hi != null ? `≤ ${c.hi}` : 'any');
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const basisText = (b) => (b?.kind === 'law' ? `law: ${b.source}` : b?.kind === 'cited' ? `cited: ${b.source}` : 'estimate');

/**
 * The run's panel and sampling. `ctx`: { registry, scenario, sim() (the ExplicitSim), lab
 * (createCrashLab's), url(params) (this page with crashRef params changed), contact, run,
 * fail }.
 */
export function createCrashRef(ctx) {
  const { registry, scenario: s } = ctx;
  const style = document.createElement('style');
  style.textContent = CSS;
  document.head.appendChild(style);
  const el = document.createElement('div');
  el.id = 'crashRef';
  document.body.appendChild(el);
  const links = s.reference.links.map((u, i) => {
    const t = s.reference.atS[i];
    return `<a href="${esc(u)}" target="_blank" rel="noopener">${Math.floor(t / 60)}:${String(t % 60).padStart(2, '0')}</a>`;
  }).join(' · ');
  el.innerHTML = `<h3>${esc(s.id)} · ${esc(s.title)}</h3>
    <div class="muted">${esc(s.reference.realTest)} · reference video ${links}</div>
    <div class="r"><label>contact</label><select id="crContact">${registry.contacts.map((c) => `<option${c === ctx.contact ? ' selected' : ''}>${c}</option>`).join('')}</select>
      <label>run</label><select id="crRun" title="Run 0 is nominal; runs 1–${registry.runs - 1} jitter the speed, the path and the heading slightly (Jitter::run)">${Array.from({ length: registry.runs }, (_, k) => `<option value="${k}"${k === ctx.run ? ' selected' : ''}>${k}${k === 0 ? ' (nominal)' : ''}</option>`).join('')}</select></div>
    <div class="r"><label>camera</label>${s.cameras.map((c) => `<button data-crcam="${esc(c.name)}">${esc(c.name)}</button>`).join('')}</div>
    <div class="r"><label>moments</label>${s.moments.map((m) => `<span class="muted">${esc(m.name)} ${m.after_impact_ms} ms</span>`).join(' · ')}</div>
    <div class="r" id="crStatus"></div>
    <table id="crTable"></table>`;
  const $ = (id) => el.querySelector('#' + id);
  $('crContact').onchange = (e) => { location.href = ctx.url({ contact: e.target.value }); };
  $('crRun').onchange = (e) => { location.href = ctx.url({ run: e.target.value }); };
  const camName = (c) => `ref:${c}`;
  for (const b of el.querySelectorAll('[data-crcam]')) b.onclick = (e) => { ctx.lab.camera(camName(b.dataset.crcam)); sync(); e.target.blur(); };
  const sync = () => { for (const b of el.querySelectorAll('[data-crcam]')) b.classList.toggle('on', ctx.lab.st.cam === camName(b.dataset.crcam)); };

  // ------------------------------------------------------------ sampling and the report
  let info = null, pushed = 0, report = null, sim = null, error = null;
  // A new run (the first, a restart): its sample 0, the state at spawn, before any frame.
  const reset = () => { sim = ctx.sim(); info = sim.crashInfo(); pushed = 0; report = null; error = null; draw(); sampled(); };
  /** After every frame the page submits: the sample of these steps, when they are one. */
  function sampled() {
    const x = ctx.sim();
    if (x !== sim) { reset(); return null; }
    if (report || error) return null;
    let p;
    try { p = x.crashSample(); } catch (e) { error = String(e?.message ?? e); draw(); ctx.fail(e); return null; }
    return p.then((n) => {
      if (x !== sim || n == null) return;
      pushed = Math.max(pushed, n);
      if (pushed >= info.samples && !report) { report = x.crashReport(); window.__crashRefReport = report; draw(); }
      else if (pushed % 25 === 0) draw();
    }, (e) => { error = String(e?.message ?? e); draw(); ctx.fail(e); });
  }
  function draw() {
    const status = error ? `<span class="m fail">error</span> ${esc(error)}`
      : report ? `<span class="m pass">done</span> the report of ${info.samples} samples (every ${info.sampleSteps} steps, ${(info.totalSteps / 2000).toFixed(2)} s), judged by Rust`
      : `<span class="m wait">running</span> sample ${pushed} of ${info?.samples ?? '…'} · impact at ${info ? (info.impactS * 1000).toFixed(0) : '…'} ms`;
    $('crStatus').innerHTML = status;
    const results = report?.results ?? null;
    const rows = s.criteria.map((c, i) => {
      const r = results?.[i];
      const mark = !r ? ['wait', '…'] : r.asserted ? (r.pass ? ['pass', 'pass'] : ['fail', 'FAIL']) : (r.pass ? ['in', 'in band'] : ['out', 'out of band']);
      const tier = c.asserted ? c.tier : `${c.tier}, reported`;
      return `<tr><td><span class="m ${mark[0]}">${mark[1]}</span></td><td><div>${esc(c.name)}</div>
        <details><summary class="muted">${esc(c.metric)} · ${tier}</summary><div>${esc(c.why)}</div><div><i>${esc(basisText(c.basis))}</i></div></details></td>
        <td class="v">${r ? fmt(r.value) : '—'}<div class="muted">${band(c)}</div></td></tr>`;
    });
    $('crTable').innerHTML = rows.join('');
  }
  reset();
  sync();
  return {
    sampled,
    reset,
    sync,
    info: () => info,
    report: () => report,
    progress: () => ({ pushed, samples: info?.samples ?? 0, error }),
  };
}
