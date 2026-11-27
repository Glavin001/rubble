// The crash lab's vehicle (crash-lab.html): where it comes from and the controls that pick it.
// A reference crash's generated designs (their mod folders), ?truck=<url> (a Rigs of Rods
// .truck), ?mod=<folder>&pc= (a BeamNG folder: beamng-mod.mjs), a file loaded on this tab
// (?car=import, kept in sessionStorage), Rubble's authored showcase sedan (?car=showcase,
// ?style=, ?res=), the hi-res JBeam test car (?car=hires) or the procedural sedan. Owns the
// car picker (#carSel), the converted BeamNG vehicles (#modSel), the skin binding select
// (#bindSel; crash-lab-body.mjs wires it), Load .truck (#truckFile), Load JBeam folder
// (#jbeamDir) and #truckInfo. Result: page.vehicle (crash-lab-state.mjs).

import { hiresCarJbeam, explicitSedanTruck, showcaseCar } from './rubble-world.mjs';
import { loadMod, LOCAL_MODS } from './beamng-mod.mjs';

const IMPORT_KEY = 'rubble-softbody-import';
// Rubble's generated sedan (the vehicle lab's default car, at __mods/sedan_new: build-web.sh
// makes it by running the generator): listed when it is there.
const GEN = { mod: '/__mods/sedan_new', pc: 'sedan/sedan.pc' };

const make = (html) => { const t = document.createElement('template'); t.innerHTML = html.trim(); return t.content.firstChild; };

/** page.vehicle: the vehicle's source, and its controls (page.el). */
export async function loadVehicle(page) {
  const { qs, crash } = page;
  let importSrc = null; // { kind: 'truck', text, label } | { kind: 'jbeam', files, pc, label, ... }
  if (crash?.folders.mods.length) {
    // The scenario's first generated design (vehicle 0): the body drawn on its vehicles.
    importSrc = crash.folders.mods[0].mod;
  } else if (qs.get('truck')) {
    const url = qs.get('truck');
    const r = await fetch(url);
    if (!r.ok) throw new Error(`?truck=${url}: ${r.status} ${r.statusText}`);
    importSrc = { kind: 'truck', text: await r.text(), label: url.split('/').pop() };
  } else if (qs.get('mod')) {
    // ?mod=<folder url>: a BeamNG vehicle folder served with a files.json (its files' paths):
    // its .jbeam parts, a .pc (?pc= its name, else the first) and its .dae meshes, or a glTF
    // render model with weighted skins (beamng-mod.mjs).
    try { importSrc = await loadMod(qs.get('mod'), { pc: qs.get('pc') }); } catch (e) {
      console.warn(`${e.message} The sedan instead.`);
      const note = make('<div class="panel" style="bottom: 80px; left: 10px; max-width: 60ch; font: 12px system-ui, sans-serif"></div>');
      note.textContent = `?mod=${qs.get('mod')} not found: the sedan instead. ${LOCAL_MODS}`;
      document.getElementById('err').before(note);
    }
  } else if (qs.get('car') === 'import') {
    try { importSrc = JSON.parse(sessionStorage.getItem(IMPORT_KEY) || 'null'); } catch { importSrc = null; }
  }
  // (The dune buggy has rigid parts: ?car=buggy runs the sedan here.)
  const cars = { sedan: 'Car: procedural sedan (soft body)', hires: 'Car: hi-res JBeam test car (1,110 nodes)', showcase: 'Car: Rubble showcase sedan (authored)' };
  if (importSrc) cars.import = `Car: ${importSrc.label} (imported)`;
  const carKind = importSrc ? 'import' : cars[qs.get('car')] || qs.get('car') === 'buggy' ? qs.get('car') : 'sedan';
  // The authored showcase sedan (engine_softbody::authoring): its textured body skinned to
  // the nodes; ?bind=, ?alt=0 are measurement switches (a visual A/B on the same physics).
  let showcaseCache = null;
  const v = page.vehicle = {
    importSrc, carKind, cars,
    skinBind: qs.get('bind') ?? '', // the skin binding (#bindSel; ?bind= at load)
    showcaseOf: (bind) => showcaseCar({ style: qs.get('style') ?? 'sedan', resolution: qs.get('res') ?? 'medium',
      ...(page.cfg.raycast ? { wheels: 'raycast' } : {}),
      ...(bind ? { bind } : {}), ...(qs.get('alt') === '0' ? { alt: false } : {}) }),
    showcase: () => showcaseCache ??= v.showcaseOf(v.skinBind),
    showcaseBuilt: () => showcaseCache,
    source: null,
  };
  const hiresJbeam = carKind === 'hires' ? [['hires.jbeam', hiresCarJbeam()]] : null;
  const base = () => carKind === 'hires' ? { jbeam: hiresJbeam }
    // JBeam content runs in JBeam mode (BeamNG's rules: desc order, pressure wheels or
    // raycast wheels); ?jbeamMode=truck: through the .truck export (RoR's rules).
    : carKind === 'import' ? (importSrc.kind === 'jbeam' ? { jbeam: importSrc.files, pc: importSrc.pc, native: page.cfg.jbeamNative } : { truck: importSrc.text })
    : carKind === 'showcase' ? { truck: v.showcase().truck }
    // The sedan as the acceptance runs build it (?tyres=0: its bare body).
    : { truck: explicitSedanTruck(qs.get('tyres') !== '0') };
  // The vehicle's source for the explicit solver (crash-lab-sim.mjs spawns it).
  v.source = () => (page.cfg.raycast ? { ...base(), wheels: 'raycast' } : base());
}

/** The vehicle's controls (#carSel, #modSel, #bindSel, #truckFile, #jbeamDir, #truckInfo) into page.el. */
export function vehicleControls(page) {
  const { qs } = page;
  const { cars, carKind } = page.vehicle;
  const el = page.el;
  el.carSel = make('<select id="carSel" title="The vehicle every scene crashes"></select>');
  el.modSel = make('<select id="modSel" title="Converted BeamNG vehicles (community_cache: engine-webgpu/__mods)" hidden></select>');
  el.bindSel = make(`<select id="bindSel" title="The showcase car's skin binding (a visual A/B on the same physics; ?bind= sets it at load)" hidden>
    <option value="">skin: node frames, cut at breaks</option><option value="triangles">skin: triangle binding (before)</option><option value="legacy">skin: legacy binding</option></select>`);
  el.truckLabel = make('<label class="file">Load .truck <input id="truckFile" type="file" accept=".truck,.load,.car,.trailer,.boat,.airplane,.fixed,.train" hidden /></label>');
  el.jbeamLabel = make('<label class="file" title="A BeamNG vehicle folder (.jbeam parts, optional .pc config)">Load JBeam folder <input id="jbeamDir" type="file" webkitdirectory multiple hidden /></label>');
  el.truckInfo = make('<div class="muted" id="truckInfo"></div>');
  const { carSel, modSel, truckInfo } = el;
  for (const [id, t] of Object.entries(cars)) carSel.add(new Option(t, id, false, id === carKind));
  fetch(`${GEN.mod}/files.json`, { method: 'HEAD' }).then((r) => {
    if (!r.ok) return;
    carSel.add(new Option('Car: Rubble generated sedan', 'gen', false, qs.get('mod') === GEN.mod));
    if (qs.get('mod') === GEN.mod) for (const o of [...carSel.options]) if (o.value === 'import') o.remove();
  }).catch(() => {});
  carSel.onchange = () => {
    const u = new URL(location.href);
    if (carSel.value === 'gen') {
      for (const k of ['car', 'truck', 'jbeamMode']) u.searchParams.delete(k);
      u.searchParams.set('mod', GEN.mod); u.searchParams.set('pc', GEN.pc); u.searchParams.set('wheels', 'raycast');
      location.href = u.href; return;
    }
    u.searchParams.set('car', carSel.value); if (carSel.value !== 'import') for (const k of ['truck', 'mod', 'pc', 'wheels']) u.searchParams.delete(k); location.href = u.href;
  };
  // Converted community vehicles (community_cache writes __mods/index.json): every car and
  // configuration; picking one loads it in JBeam mode on raycast wheels in this scene.
  fetch('/__mods/index.json').then((r) => (r.ok ? r.json() : [])).then((list) => {
    if (!list.length) return;
    modSel.hidden = false;
    modSel.add(new Option('Converted BeamNG vehicles…', ''));
    for (const c of list) {
      const g = document.createElement('optgroup');
      g.label = `${c.label} (${c.megabytes} MB)${c.note ? ` · ${c.note}` : ''}`;
      for (const k of c.configs) {
        const mod = `/__mods/${c.folder}`;
        g.appendChild(new Option(`${c.label} · ${k.pc.split('/').pop().replace(/\.pc$/, '')} (${k.nodes} nodes)${c.note || (k.rest && (k.rest.broken > 0 || k.rest.fastest > 1 || !k.rest.finite)) ? ' · marginal at rest' : ''}`, `${mod}|${k.pc}|${k.wheels ?? 1}`, false, qs.get('mod') === mod && qs.get('pc') === k.pc));
      }
      modSel.appendChild(g);
    }
  }).catch(() => {});
  modSel.onchange = () => {
    if (!modSel.value) return;
    const [mod, pc, wheels] = modSel.value.split('|');
    const u = new URL(location.href);
    for (const k of ['car', 'truck', 'jbeamMode']) u.searchParams.delete(k);
    u.searchParams.set('mod', mod);
    u.searchParams.set('pc', pc);
    // Raycast wheels when it has wheels; none (the yacht): its scene is the 15 m drop.
    if (Number(wheels) > 0) u.searchParams.set('wheels', 'raycast');
    else { u.searchParams.delete('wheels'); u.searchParams.delete('xscene'); u.searchParams.set('preset', 'drop15'); }
    location.href = u.href;
  };
  // A loaded file is kept for this tab (?car=import) and the scene rebuilt with it.
  const importUrl = () => { const u = new URL(location.href); u.searchParams.set('car', 'import'); u.searchParams.delete('truck'); return u; };
  const remember = (src) => {
    try { sessionStorage.setItem(IMPORT_KEY, JSON.stringify(src)); history.replaceState(null, '', importUrl().href); return true; } catch {
      truckInfo.textContent += ' · too large to keep for the solver toggle';
      return false;
    }
  };
  el.truckLabel.querySelector('input').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try { if (remember({ kind: 'truck', text: await file.text(), label: file.name })) { location.href = importUrl().href; return; } } catch (err) { truckInfo.textContent = `${file.name}: ${err.message || err}`; }
    e.target.value = '';
  });
  el.jbeamLabel.querySelector('input').addEventListener('change', async (e) => {
    const all = [...e.target.files];
    try {
      const files = await Promise.all(all.filter((f) => f.name.endsWith('.jbeam')).map(async (f) => [f.webkitRelativePath || f.name, await f.text()]));
      const pcs = all.filter((f) => f.name.endsWith('.pc'));
      const pcFile = pcs.find((f) => f.name === 'start.pc') || pcs[0];
      const pc = pcFile ? await pcFile.text() : undefined;
      if (remember({ kind: 'jbeam', files, pc, label: `${files.length} jbeam files${pcFile ? ' + ' + pcFile.name : ''}` })) { location.href = importUrl().href; return; }
    } catch (err) { truckInfo.textContent = `JBeam: ${err.message || err}`; }
    e.target.value = '';
  });
  // A reference crash's vehicles are its scenario's (Rust): the pickers do not apply.
  if (page.crash) for (const c of [carSel, modSel]) { c.disabled = true; c.title = `The vehicles of ${page.crash.scenario.id} are its scenario's (the crash reference panel has its controls)`; }
}
