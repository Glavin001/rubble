// The crash lab's scenes as configurations (softbody.html ?solver=explicit): the scene picker
// grouped by scenario family, and a scene's parameters as controls in the Scene panel.
//
// Nothing about a scene is defined here. The reference crashes and the scenario families
// (their parameters, defaults, ranges, units, presets) are Rust's registry
// (`crashReferenceRegistry`: engine_softbody::crash_reference::family, families); the lab's
// own scenes and one-click presets are Rust's catalogue (`crashLabCatalog`: engine-world-wasm
// lab_scenes.rs). This module draws controls for the parameters it is given and hands the
// values back; the page places the scene again from them (no reload).

// (No import of the engine: crash-lab.mjs, which builds the lab's frame before the engine
// loads, imports this module.)

/** A crash reference scenario's family values from the page's URL (`f.<key>=<value>`: the
 * values a changed control wrote). */
export function crashRefParams(qs) {
  return Object.fromEntries([...qs].filter(([k]) => k.startsWith('f.')).map(([k, v]) => [k.slice(2), v]));
}

/** `qs` with the family values `values` that differ from `base` written as `f.<key>` (the
 * others dropped). */
export function crashRefParamsUrl(qs, values, base) {
  for (const k of [...qs.keys()]) if (k.startsWith('f.')) qs.delete(k);
  for (const [k, v] of Object.entries(values)) if (String(v) !== String(base[k])) qs.set(`f.${k}`, String(v));
  return qs;
}

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

/** A parameter's default (`ParamKind`). */
export const paramDefault = (p) => p.kind.default;

/**
 * The page's tables from the lab's catalogue, in the shapes softbody.html reads them:
 * `XSCENES` (id → title), `XPARAMS` (id → [[key, "label unit", default, [min, max, step]]]:
 * km/h ones go to the solver in m/s), `XCRASH`, `XDRIVE` (sets of ids), `XPILL` (id → () =>
 * the pill from the values now, `P(key, default)`) and `LAB_PRESETS` (id → { label, xscene,
 * params, pills, cam }).
 */
export function labTables(catalog, P) {
  const XSCENES = {}, XPARAMS = {}, XPILL = {};
  for (const s of catalog.scenes) {
    XSCENES[s.id] = s.title;
    if (s.params.length) XPARAMS[s.id] = s.params.map((p) => [p.key, `${p.label} ${p.unit}`.trim(), paramDefault(p), [p.kind.min, p.kind.max, p.kind.step]]);
    XPILL[s.id] = () => s.pill.replace(/\{(\w+)\}/g, (_, k) => String(P(k, paramDefault(s.params.find((p) => p.key === k)))));
  }
  const LAB_PRESETS = Object.fromEntries(catalog.presets.map((p) => [p.id, { label: p.label, xscene: p.scene, params: p.values, pills: p.pills, cam: p.cam }]));
  return {
    XSCENES, XPARAMS, XPILL, LAB_PRESETS,
    XCRASH: new Set(catalog.scenes.filter((s) => s.crash).map((s) => s.id)),
    XDRIVE: new Set(catalog.scenes.filter((s) => s.drive).map((s) => s.id)),
  };
}

/**
 * The scene picker's options, grouped by scenario family: under each family its reference
 * crashes (`r:<id>`), the lab's presets (`p:<id>`) and scenes (`s:<id>`) that stage it, and
 * the family itself with the generated designs (`f:<id>`); then the lab's scenes no family
 * stages. Every entry keeps its value and name, except a scene that is a pure duplicate of a
 * preset (`sameAs`: the same fleet word for word, Rust's catalogue says which): the preset
 * stands for both (`?xscene=` still opens the scene, listed while it is open). `current`:
 * the selected value.
 */
export function pickerOptions({ registry, catalog, current }) {
  const opt = (value, text, title = '') => `<option value="${esc(value)}"${value === current ? ' selected' : ''}${title ? ` title="${esc(title)}"` : ''}>${esc(text)}</option>`;
  const sceneOf = Object.fromEntries(catalog.scenes.map((s) => [s.id, s]));
  const listed = (s) => !s.sameAs || current === `s:${s.id}`;
  const groups = [];
  for (const f of registry.families ?? []) {
    const refs = registry.scenarios.filter((s) => s.family === f.id && !s.recreated);
    const scenes = catalog.scenes.filter((s) => s.family === f.id && listed(s));
    const presets = catalog.presets.filter((p) => sceneOf[p.scene]?.family === f.id);
    const rows = [
      ...refs.map((s) => opt(`r:${s.id}`, `${s.id} · ${s.title}`, 'A reference crash: its criteria, cameras and footage')),
      ...presets.map((p) => opt(`p:${p.id}`, `preset · ${p.label}`, 'A crash-test preset: the scene below with these values, on the car loaded')),
      ...scenes.map((s) => opt(`s:${s.id}`, `scene · ${s.title}`, 'A lab scene: any car loaded, its knobs in this panel')),
      opt(`f:${f.id}`, `configure · ${f.title.toLowerCase()}`, `${f.about} Generated designs; every parameter a control; law-based criteria.`),
    ];
    groups.push(`<optgroup label="${esc(f.title)}">${rows.join('')}</optgroup>`);
  }
  const rest = catalog.scenes.filter((s) => !s.family && listed(s));
  const restPresets = catalog.presets.filter((p) => !sceneOf[p.scene]?.family);
  if (rest.length || restPresets.length) groups.push(`<optgroup label="Rigs, fleets and driving">${[...restPresets.map((p) => opt(`p:${p.id}`, `preset · ${p.label}`)), ...rest.map((s) => opt(`s:${s.id}`, `scene · ${s.title}`))].join('')}</optgroup>`);
  const data = registry.scenarios.filter((s) => s.recreated);
  if (data.length) groups.push(`<optgroup label="Recreated in the Reference Studio">${data.map((s) => opt(`r:${s.id}`, `${s.id} · ${s.title}`)).join('')}</optgroup>`);
  return groups.join('');
}

const CSS = `
.lab-params .ph { display: flex; align-items: center; gap: 6px; margin: 8px 0 2px; }
.lab-params .ph .sub { margin: 0; flex: 1; }
.lab-params .mod { padding: 0 6px; border-radius: 4px; font-size: 11px; font-weight: 700; background: #7a5a12; color: #fff3d6; }
.lab-params .mod[hidden] { display: none; }
.lab-params .fam { color: #9aa6be; font-size: 11px; margin: 0 0 4px; }
.lab-params .pr { display: grid; grid-template-columns: minmax(60px, 30%) minmax(40px, 1fr) auto; gap: 6px; align-items: center; margin: 3px 0; }
.lab-params .pr label { color: #9fb0cf; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lab-params .pr.changed label { color: #ffd27a; }
.lab-params .pr > span { white-space: nowrap; }
.lab-params .pr input[type=range] { width: 100%; min-width: 40px; accent-color: #5b9cff; }
.lab-params .pr input[type=number] { width: 54px; box-sizing: border-box; background: #18233a; color: #e6ecf7; border: 1px solid #2c3b5c; border-radius: 6px; padding: 2px 4px; font: 12px system-ui, sans-serif; }
.lab-params .pr select { grid-column: 2 / 4; }
.lab-params .pr .u { color: #9aa6be; font-size: 11px; min-width: 24px; }
.lab-params .note { margin: 6px 0; padding: 5px 8px; border-radius: 6px; background: rgba(42, 51, 68, 0.7); color: #c4cfe4; }
.lab-params .note:empty { display: none; }
.lab-params .busy { color: #9fb0cf; }
`;
let styled = false;

/**
 * A scene's parameters as controls in `mount` (the Scene panel). `params`: Rust's
 * declarations (`Param[]`: key, label, unit, help, kind); `values`: the values now; `base`:
 * the preset's (or the defaults), which "reset" restores and against which a change is
 * marked; `title`: the heading; `apply(values, key)`: place the scene again (may return a
 * promise: the controls wait for it). `note(modified)`: a line under the controls. Returns
 * `{ values(), set(values), el }`.
 */
export function createParamControls({ mount, params, values, base, title, about = '', apply, note = () => '' }) {
  if (!styled) { const st = document.createElement('style'); st.textContent = CSS; document.head.appendChild(st); styled = true; }
  const el = document.createElement('div');
  el.className = 'lab-params';
  mount.appendChild(el);
  let cur = { ...values };
  let busy = false;
  const same = (a, b) => (typeof a === 'number' || typeof b === 'number' ? Math.abs(Number(a) - Number(b)) < 1e-9 : String(a) === String(b));
  const changed = (k) => !same(cur[k], base[k] ?? paramDefault(params.find((p) => p.key === k)));
  const modified = () => params.some((p) => changed(p.key));
  const fmt = (p, v) => (p.kind.kind === 'number' ? String(Math.round(Number(v) * 1e4) / 1e4) : String(v));
  const row = (p) => {
    const id = `lp_${p.key}`;
    const help = `${p.label}${p.unit ? ` (${p.unit})` : ''}${p.help ? `: ${p.help}` : ''}`;
    if (p.kind.kind === 'number') {
      const { min, max, step } = p.kind;
      return `<div class="pr" data-k="${esc(p.key)}" title="${esc(help)}"><label for="${id}">${esc(p.label)}</label><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" aria-label="${esc(p.label)}"><span><input type="number" min="${min}" max="${max}" step="any" aria-label="${esc(p.label)} value"> <span class="u">${esc(p.unit)}</span></span></div>`;
    }
    if (p.kind.kind === 'flag') return `<div class="pr" data-k="${esc(p.key)}" title="${esc(help)}"><label for="${id}">${esc(p.label)}</label><input type="checkbox" id="${id}"></div>`;
    return `<div class="pr" data-k="${esc(p.key)}" title="${esc(help)}"><label for="${id}">${esc(p.label)}</label><select id="${id}">${p.kind.choices.map((c) => `<option>${esc(c)}</option>`).join('')}</select></div>`;
  };
  el.innerHTML = `<div class="ph"><div class="sub">parameters</div><span class="mod" hidden title="These values differ from the preset's">modified</span><button type="button" class="reset" title="Back to the preset's values">reset</button></div>
    <div class="fam" title="${esc(about)}">${esc(title)}</div>${params.map(row).join('')}<div class="note"></div>`;
  const sync = () => {
    for (const p of params) {
      const r = el.querySelector(`.pr[data-k="${p.key}"]`);
      if (!r) continue;
      const v = cur[p.key] ?? paramDefault(p);
      if (p.kind.kind === 'number') { r.querySelector('input[type=range]').value = String(v); const n = r.querySelector('input[type=number]'); if (document.activeElement !== n) n.value = fmt(p, v); }
      else if (p.kind.kind === 'flag') r.querySelector('input').checked = !!v;
      else r.querySelector('select').value = String(v);
      r.classList.toggle('changed', changed(p.key));
      for (const c of r.querySelectorAll('input, select')) c.disabled = busy;
    }
    el.querySelector('.mod').hidden = !modified();
    el.querySelector('.reset').disabled = busy || !modified();
    el.querySelector('.note').textContent = busy ? 'placing the scene…' : note(modified());
  };
  const commit = async (k, v) => {
    const p = params.find((x) => x.key === k);
    if (p.kind.kind === 'number') { v = Number(v); if (!Number.isFinite(v)) { sync(); return; } v = Math.min(p.kind.max, Math.max(p.kind.min, v)); }
    if (same(cur[k], v)) { sync(); return; }
    cur = { ...cur, [k]: v };
    await run(k);
  };
  const run = async (k) => {
    busy = true; sync();
    try { await apply({ ...cur }, k); } finally { busy = false; sync(); }
  };
  for (const p of params) {
    const r = el.querySelector(`.pr[data-k="${p.key}"]`);
    const stop = (e) => e.stopPropagation();
    for (const c of r.querySelectorAll('input, select')) c.addEventListener('keydown', stop);
    if (p.kind.kind === 'number') {
      const range = r.querySelector('input[type=range]'), num = r.querySelector('input[type=number]');
      range.oninput = () => { num.value = fmt(p, range.value); };
      range.onchange = () => commit(p.key, range.value);
      num.onchange = () => commit(p.key, num.value);
    } else if (p.kind.kind === 'flag') r.querySelector('input').onchange = (e) => commit(p.key, e.target.checked);
    else r.querySelector('select').onchange = (e) => commit(p.key, e.target.value);
  }
  el.querySelector('.reset').onclick = () => { cur = { ...Object.fromEntries(params.map((p) => [p.key, paramDefault(p)])), ...base }; run(null); };
  sync();
  return { el, values: () => ({ ...cur }), modified, set(v) { cur = { ...v }; sync(); }, sync };
}

/**
 * The Scene panel's parameters for what is open (`mount`: the panel's body, `at`: the
 * element they go before, or the end):
 * - a reference crash or a family opened directly (`crash`: `{ id, family, params }`, the
 *   page's crashRef selection, `info()`: Rust's entry for the scenario now): the family's
 *   parameters (`registry.families`), starting from the scenario's values; a change asks Rust
 *   for the scenario with those values (`resolve`: rubble-world.mjs `crashReferenceScenario`,
 *   its designs and criteria,
 *   `modifiedFrom` when they are not the preset's) and `respawn({ info })` places it; the
 *   values go to the URL as `f.<key>`.
 * - a lab scene or preset (`xscene`, `preset`): the scene's parameters (`catalog.scenes`),
 *   starting from the URL's values (a preset's merged in), against the preset's; a change
 *   writes the URL (and `qs`, which the page's spawn reads) and `respawn(null)` places it.
 * No reload either way. Returns the controls (`modified()`: the values are not the preset's).
 */
export function mountSceneParams({ mount, at = null, qs, registry, catalog, crash, info, resolve, xscene, preset, respawn }) {
  const box = document.createElement('div');
  mount.insertBefore(box, at);
  const url = (edit) => { const u = new URL(location.href); edit(u.searchParams); history.replaceState(null, '', u.href); };
  if (crash) {
    const s0 = info();
    const fam = registry.families.find((f) => f.id === s0.family);
    if (!fam) return null;
    // The preset's values (a reference crash's as the registry lists it), or the defaults.
    const base = registry.scenarios.find((x) => x.id === crash.id)?.values ?? fam.defaults;
    return createParamControls({
      mount: box, params: fam.params, values: s0.values, base, title: `family: ${fam.title}`, about: fam.about,
      note: (m) => (crash.id ? (m ? `Modified from ${crash.id}: its reference criteria are not applied (law-based criteria only).` : `${crash.id}'s values: its reference criteria apply.`) : 'A family opened directly: law-based criteria.'),
      apply: async (values) => {
        const next = await resolve({ id: crash.id, family: crash.family, params: values });
        crash.params = values;
        url((q) => crashRefParamsUrl(q, values, base));
        await respawn({ info: next });
      },
    });
  }
  const scene = catalog.scenes.find((s) => s.id === xscene);
  if (!scene?.params.length) return null;
  const pv = preset ? catalog.presets.find((p) => p.id === preset)?.values ?? {} : {};
  const base = Object.fromEntries(scene.params.map((p) => [p.key, pv[p.key] ?? paramDefault(p)]));
  const num = (k, d) => (qs.has(k) && qs.get(k) !== '' && Number.isFinite(Number(qs.get(k))) ? Number(qs.get(k)) : d);
  const values = Object.fromEntries(scene.params.map((p) => [p.key, num(p.key, base[p.key])]));
  return createParamControls({
    mount: box, params: scene.params, values, base, title: `lab scene: ${scene.title} · on the car loaded`,
    about: 'The lab\'s own scene: any vehicle the page loads, placed by its collision skin, its run-up 1 m plus half a second of travel unless set.',
    note: (m) => (preset && m ? 'Modified from the preset (its pills now show the values).' : ''),
    apply: async (v) => {
      for (const [k, x] of Object.entries(v)) qs.set(k, String(x));
      url((q) => { for (const [k, x] of Object.entries(v)) q.set(k, String(x)); });
      await respawn(null);
    },
  });
}
