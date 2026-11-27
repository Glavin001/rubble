// The crash reference suite in the crash lab (crash-lab.html?crashRef=S02
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
    // Each design's folder is fetched once a page (a changed design parameter loads only
    // the new one).
    if (!folderCache.has(d)) folderCache.set(d, loadFolder(d, listed));
    const { folder, mod, files } = await folderCache.get(d);
    designs[d] = files;
    mods.push({ design: d, folder, pc: mod.pcName, mod });
  }
  return { mods, designs };
}
const folderCache = new Map();
async function loadFolder(d, listed) {
  // A local imported vehicle (Rust's `family::LOCAL`): `local:<folder>/<path of its .pc>`.
  const local = d.startsWith('local:') ? d.slice(6) : null;
  const row = listed.find((x) => x.name === d);
  const folder = local ? `/__mods/${local.split('/')[0]}` : row?.folder ? `/__mods/${row.folder}` : `/__mods/${d}_new`;
  const mod = await loadMod(folder, { pc: local ? local.slice(local.indexOf('/') + 1) : row?.pc ?? null });
  const crash = mod.list.find((f) => f.endsWith('crash.json'));
  const files = [...mod.files];
  if (mod.pcName && mod.pc != null) files.push([mod.pcName, mod.pc]);
  if (crash) { const r = await fetch(`${folder}/${crash}`); if (!r.ok) throw new Error(`${folder}/${crash}: ${r.status}`); files.push([crash, await r.text()]); }
  return { folder, mod, files };
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

// The results' styles. The lab's layout (crash-lab-ui.mjs) gives the panel its frame;
// without one (ctx.mount absent) the panel stands alone at the right, as it used to.
const CSS = `
#crashRef.alone { position: absolute; top: 118px; right: 16px; width: min(440px, calc(100vw - 32px)); max-height: calc(100vh - 260px); overflow: auto; background: rgba(10, 16, 30, 0.86); border: 1px solid rgba(160, 190, 255, 0.16); border-radius: 10px; padding: 10px 12px; font: 12px/1.4 system-ui, sans-serif; color: #dfe6f3; backdrop-filter: blur(8px); z-index: 5; }
#crashRef .cr-head { margin: 4px 0 6px; }
#crashRef .cr-head b { display: block; font-size: 12.5px; color: #e9eef7; }
#crashRef .muted { color: #9aa6be; }
#crashRef a { color: #8fc1ff; }
#crashRef .r { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 5px 0; }
#crashRef .note { margin: 6px 0; padding: 5px 8px; border-radius: 6px; background: rgba(42, 51, 68, 0.7); color: #c4cfe4; }
#crashRef .grp { margin: 8px 0 2px; color: #9fb0cf; font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; }
#crashRef ul { list-style: none; margin: 0; padding: 0; }
#crashRef li { border-top: 1px solid #1f2b45; }
#crashRef li > details > summary { display: flex; align-items: baseline; gap: 6px; padding: 4px 2px; cursor: pointer; list-style: none; }
#crashRef li > details > summary::-webkit-details-marker { display: none; }
#crashRef li > details > summary:hover { background: rgba(43, 79, 138, 0.22); }
#crashRef .n { flex: 1; min-width: 0; color: #dfe6f3; }
#crashRef .v { font-variant-numeric: tabular-nums; white-space: nowrap; text-align: right; }
#crashRef .b { color: #9aa6be; font-variant-numeric: tabular-nums; white-space: nowrap; text-align: right; }
#crashRef .vb { display: flex; flex-direction: column; align-items: flex-end; flex: none; max-width: 45%; }
#crashRef .vb .b { font-size: 11px; }
#crashRef .st { margin: 4px 0; }
#crashRef .d { padding: 0 4px 6px 4px; color: #aab8d4; }
#crashRef .d code { color: #c4cfe4; font: 11px ui-monospace, monospace; }
#crashRef .m { padding: 1px 6px; border-radius: 4px; font-weight: 700; font-size: 11px; white-space: nowrap; }
#crashRef .pass, .lab-toggle .m.pass { background: #1d5a37; } #crashRef .fail, .lab-toggle .m.fail { background: #8a1f2b; } #crashRef .known, .lab-toggle .m.known { background: #7a5a12; }
#crashRef .in { background: #26405f; } #crashRef .out { background: #5a4a1d; } #crashRef .wait, .lab-toggle .m.wait { background: #2a3344; color: #c4cfe4; }
.lab-toggle .m { padding: 0 5px; border-radius: 4px; font-weight: 700; font-size: 11px; color: #e9eef7; }
#crashRef .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #3a4660; flex: none; align-self: center; }
#crashRef details.rep > summary { cursor: pointer; margin: 8px 0 2px; color: #9fb0cf; font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; }
.lab-clean #crashRef { display: none !important; }
`;

const fmt = (v) => (v == null || !Number.isFinite(v) ? '—' : Math.abs(v) >= 1000 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(2) : v.toFixed(3));
const band = (c) => (c.lo != null && c.hi != null ? `[${c.lo}, ${c.hi}]` : c.lo != null ? `≥ ${c.lo}` : c.hi != null ? `≤ ${c.hi}` : 'any');
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const basisText = (b) => (b?.kind === 'law' ? `law: ${b.source}` : b?.kind === 'cited' ? `cited: ${b.source}` : 'estimate');

/**
 * The run's panel and sampling. `ctx`: { registry, scenario, sim() (the ExplicitSim), lab
 * (createCrashLab's), url(params) (this page with crashRef params changed), contact, run,
 * fail, mount? }. `mount` (the lab's layout): `{ results, scene, summary(html) }`: the
 * criteria go in `results`, the contact and run pickers in `scene`, and the one-line state
 * to `summary` (what the collapsed Results panel shows). The scenario's cameras are the
 * lab's (its camera group, ctx.cams).
 */
export function createCrashRef(ctx) {
  const { registry } = ctx;
  let s = ctx.scenario;
  const style = document.createElement('style');
  style.textContent = CSS;
  document.head.appendChild(style);
  const el = document.createElement('div');
  el.id = 'crashRef';
  if (!ctx.mount) el.className = 'alone';
  (ctx.mount?.results ?? document.body).appendChild(el);
  // The heading: what the run is and where its criteria come from. A reference crash with
  // its values changed, or a family opened directly, carries the law-based criteria only.
  const head = () => {
    const links = s.reference.links.map((u, i) => {
      const t = s.reference.atS[i];
      return `<a href="${esc(u)}" target="_blank" rel="noopener">${Math.floor(t / 60)}:${String(t % 60).padStart(2, '0')}</a>`;
    }).join(' · ');
    const source = s.modifiedFrom ? `<div class="note">Modified from ${esc(s.modifiedFrom)}: its reference criteria (bands derived for its own speed, masses and geometry) are not applied; the law-based criteria below are, from this run's own numbers.</div>`
      : !s.reference.realTest ? `<div class="note">A scenario family opened directly: the law-based criteria, from this run's own numbers.</div>`
      : `<div class="muted">${esc(s.reference.realTest)}${links ? ` · reference video ${links}` : ''}</div>`;
    return `<b>${esc(s.id)} · ${esc(s.title)}</b>${source}<div class="muted">moments: ${s.moments.map((m) => `${esc(m.name)} ${m.after_impact_ms} ms`).join(' · ')}</div>`;
  };
  // The run's own settings: what is simulated (the Scene panel).
  const pick = document.createElement('div');
  pick.className = 'r';
  pick.innerHTML = `<label class="lbl" for="crContact">contact</label><select id="crContact" title="The contact setting this run uses (registry contacts)">${registry.contacts.map((c) => `<option${c === ctx.contact ? ' selected' : ''}>${c}</option>`).join('')}</select>
      <label for="crRun">run</label><select id="crRun" title="Run 0 is nominal; runs 1–${registry.runs - 1} jitter the speed, the path and the heading slightly (Jitter::run)">${Array.from({ length: registry.runs }, (_, k) => `<option value="${k}"${k === ctx.run ? ' selected' : ''}>${k}${k === 0 ? ' (nominal)' : ''}</option>`).join('')}</select>`;
  (ctx.mount?.scene ?? el).appendChild(pick);
  el.insertAdjacentHTML('beforeend', `<div class="cr-head" id="crHead">${head()}</div>
    <div class="st" id="crStatus"></div>
    <div id="crTable"></div>`);
  const $ = (id) => document.getElementById(id);
  $('crContact').onchange = (e) => { location.href = ctx.url({ contact: e.target.value }); };
  $('crRun').onchange = (e) => { location.href = ctx.url({ run: e.target.value }); };
  // (The cameras are the lab's: its camera group marks the active one.)
  const sync = () => ctx.lab.sync?.();

  // ------------------------------------------------------------ sampling and the report
  let info = null, pushed = 0, report = null, sim = null, error = null;
  // A new run (the first, a restart): its sample 0, the state at spawn, before any frame.
  const reset = () => { sim = ctx.sim(); info = sim.crashInfo(); pushed = 0; report = null; error = null; tableKey = null; draw(); sampled(); };
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
  let tableKey = null;
  function draw() {
    const results = report?.results ?? null;
    const status = error ? `<span class="m fail">error</span> ${esc(error)}`
      : report ? `<span class="m pass">done</span> the report of ${info.samples} samples (every ${info.sampleSteps} steps, ${(info.totalSteps / 2000).toFixed(2)} s), judged by Rust`
      : `<span class="m wait">running</span> sample ${pushed} of ${info?.samples ?? '…'} · impact at ${info ? (info.impactS * 1000).toFixed(0) : '…'} ms`;
    $('crStatus').innerHTML = status;
    // The collapsed panel's line: the run's progress, then its verdict.
    const asserted = s.criteria.map((c, i) => [c, results?.[i]]).filter(([c]) => c.asserted);
    const reported = s.criteria.map((c, i) => [c, results?.[i]]).filter(([c]) => !c.asserted);
    if (ctx.mount?.summary) {
      if (error) ctx.mount.summary('<span class="m fail">error</span>');
      else if (!results) ctx.mount.summary(`running · sample ${pushed} of ${info?.samples ?? '…'}`);
      else {
        const pass = asserted.filter(([, r]) => r?.pass).length;
        const known = asserted.filter(([, r]) => r && !r.pass && r.known).length, fail = asserted.length - pass - known;
        const inBand = reported.filter(([, r]) => r?.pass).length;
        ctx.mount.summary(`<span class="m ${fail ? 'fail' : known ? 'known' : 'pass'}">${pass} pass${known ? ` · ${known} known failure${known > 1 ? 's' : ''}` : ''} · ${fail} fail</span> ${reported.length} reported (${inBand} in band)`);
      }
    }
    // One row per criterion: its mark, name, value and band on a line; the metric, tier,
    // basis and why behind it. Before the report no value is known (Rust judges the run
    // at its end): one note says so, the rows show their bands.
    const row = ([c, r]) => {
      const mark = !r ? '<span class="dot" aria-hidden="true"></span>' : `<span class="m ${r.asserted ? (r.pass ? 'pass' : r.known ? 'known' : 'fail') : r.pass ? 'in' : 'out'}">${r.asserted ? (r.pass ? 'pass' : r.known ? 'known failure' : 'FAIL') : r.pass ? 'in band' : 'out of band'}</span>`;
      return `<li><details><summary>${mark}<span class="n">${esc(c.name)}</span><span class="vb">${r ? `<span class="v">${fmt(r.value)}</span>` : ''}<span class="b">${band(c)}</span></span></summary>
        <div class="d"><code>${esc(c.metric)}</code> · ${c.asserted ? c.tier : `${c.tier}, reported`} · <i>${esc(basisText(c.basis))}</i><div>${esc(c.why)}</div>${r?.known_note ? `<div>known failure (${esc(r.known_occurs ?? '')}): ${esc(r.known_note)}</div>` : ''}</div></details></li>`;
    };
    const key = `${results ? 'r' : error ? 'e' : 'w'}${info?.samples}`;
    if (key === tableKey) return;
    tableKey = key;
    const open = $('crTable').querySelector('details.rep')?.open ?? false;
    $('crTable').innerHTML = (results || error ? '' : `<div class="note">Values are measured when the run ends (sample ${info?.samples ?? '…'}); the bands are below.</div>`)
      + `<div class="grp">Asserted · ${asserted.length} · failing one fails the run</div><ul>${asserted.map(row).join('')}</ul>`
      + (reported.length ? `<details class="rep"${open ? ' open' : ''}><summary>Reported · ${reported.length} · never a failure</summary><ul>${reported.map(row).join('')}</ul></details>` : '');
  }
  reset();
  sync();
  return {
    sampled,
    reset,
    sync,
    /** The scenario placed again with other values (Rust's entry for it: its criteria,
     * `modifiedFrom`): the heading and the table follow, then a new run starts (`reset`). */
    setScenario(info) { s = info; $('crHead').innerHTML = head(); tableKey = null; },
    scenario: () => s,
    info: () => info,
    report: () => report,
    progress: () => ({ pushed, samples: info?.samples ?? 0, error }),
  };
}
