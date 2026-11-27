// What the crash lab page's URL asks for (crash-lab.html): a reference crash or scenario
// family (?crashRef=, ?crashFamily=, their ?f.<key>= values, ?run=, ?contact=), a one-click
// preset (?preset=), an explicit scene (?xscene=, or a soft scene's name in ?scene=) and its
// knobs, the explicit solver's modes (driving, raycast wheels, the plain views) and options
// (surface contact, drag, centred damping, wheel boxes), and the page URLs the pickers go to.
// Reads only; the scenes, presets and families are Rust's (crashLabCatalog, the crash
// reference registry). Results go to page.qs, page.cfg and page.crash (crash-lab-state.mjs).

import { crashReferenceRegistry, crashReferenceScenario, crashLabCatalog } from './rubble-world.mjs';
import { crashRefFolders } from './crash-ref.mjs';
import { labTables, crashRefParams } from './crash-lab-scene.mjs';

// What a scene or preset change keeps: the vehicle (built-in car, .truck, imported mod and
// configuration, wheels, JBeam mode) and the view and contact options.
const KEEP_KEYS = ['car', 'style', 'res', 'truck', 'mod', 'pc', 'wheels', 'jbeamMode', 'tyres', 'drag', 'lab', 'drive', 'tex', 'skin', 'surfaceContact', 'fullContact', 'centredDamping', 'wheelBoxes'];
// The soft solver's scene names (?scene=, as softbody.html's links write them), each mapped
// to the nearest explicit scene.
const XMAP = { yard: 'yard', crates: 'idle', traffic: 'traffic', drop: 'drop', crusher: 'idle', ramp: 'coast', rollover: 'coast',
  wall: 'wall', pole: 'pole', sidepole: 'sidepole', tbone: 'tbone', headon: 'headon', limbo: 'limbo', sandwich: 'wall', import: 'wall' };
// Keys only the explicit page understands: the soft page's link drops them.
const LAB_ONLY = ['solver', 'preset', 'crashRef', 'crashFamily'];

/**
 * The URL's scene: page.qs, page.crash, page.shell and page.cfg (all but the options, which
 * need the vehicle: readOptions).
 */
export async function readConfig(page) {
  const qs = page.qs = new URLSearchParams(location.search);
  const keepVehicle = (s) => { for (const k of KEEP_KEYS) if (qs.has(k)) s.set(k, qs.get(k)); return s; };
  // Crash parameters (?speed= km/h, ?overlap=, ...): a finite number, else the default.
  const P = (k, d) => (qs.has(k) && qs.get(k) !== '' && Number.isFinite(Number(qs.get(k))) ? Number(qs.get(k)) : d);
  // A crash reference scenario (?crashRef=S02&run=0&contact=edge&cam=side; crash-ref.mjs,
  // engine-world-wasm crash_ref.rs): the suite's own fleet, built in Rust from the scenario's
  // id, with raycast wheels; the generated designs' mod folders (the body drawn, the files
  // Rust builds their templates from). A family opened directly (?crashFamily=block): a
  // crashRef run of the family's layout; either one's values are ?f.<key>= (the Scene panel).
  const refId = qs.get('crashRef');
  const family = refId ? null : qs.get('crashFamily');
  // The lab's panels from the first paint (built by crash-lab.html's first module).
  page.shell = window.__labShell ?? null;
  // The lab's own scenes and presets: Rust's catalogue (engine-world-wasm lab_scenes.rs).
  const catalog = await crashLabCatalog();
  if (refId || family) {
    qs.set('wheels', 'raycast'); qs.set('xscene', 'idle');
    for (const k of ['preset', 'truck', 'mod', 'pc', 'car']) qs.delete(k);
    const reg = await crashReferenceRegistry();
    const sel = { id: refId, family, params: crashRefParams(qs) };
    const scenario = await crashReferenceScenario(sel);
    page.shell?.setTitle(`${scenario.id} · ${scenario.title}`);
    const folders = await crashRefFolders(scenario);
    // The scenario's own camera (?cam=<its name>, else its first; a family has none).
    if (!qs.has('cam3') && scenario.cameras.length) qs.set('cam3', `ref:${scenario.cameras.some((c) => c.name === qs.get('cam')) ? qs.get('cam') : scenario.cameras[0].name}`);
    page.crash = { reg, scenario, folders, sel,
      run: Math.max(0, Math.min(reg.runs - 1, Math.floor(Number(qs.get('run')) || 0))),
      contact: reg.contacts.includes(qs.get('contact')) ? qs.get('contact') : 'edge' };
  }
  // Crash-lab presets (?preset=): the explicit scene and its fields (the query's own values
  // win), labels and camera. A ?xscene= other than the preset's own wins: the preset is
  // dropped (it would otherwise force its scene back). The tables are the catalogue's.
  const tables = labTables(catalog, (k, d) => P(k, d));
  const { XSCENES, XDRIVE, LAB_PRESETS } = tables;
  const preset = LAB_PRESETS[qs.get('preset')] && (!qs.has('xscene') || qs.get('xscene') === LAB_PRESETS[qs.get('preset')].xscene)
    ? qs.get('preset') : null;
  if (preset) {
    const p = LAB_PRESETS[preset];
    qs.set('xscene', p.xscene);
    for (const [k, v] of Object.entries(p.params)) if (!qs.has(k)) qs.set(k, String(v));
  }
  // The soft scene's name (?scene=: its spawn offsets and the World's scene), and the
  // explicit scene: ?xscene=, else the soft scene's nearest.
  const scene = qs.get('scene') || 'yard';
  const xscene = XSCENES[qs.get('xscene')] ? qs.get('xscene') : XMAP[scene];
  if (!xscene) throw new Error(`unknown scene ${scene}`);
  const arena = xscene === 'arena';
  // Driving: raycast wheels (engine_softbody::explicit_drive) in place of the beam tyres,
  // WASD / gamepad, chase camera. The drive scenes use them; ?drive=1 turns them on in any
  // scene (beam tyres stay the crash scenes' default).
  const drive = qs.has('drive') ? qs.get('drive') === '1' : XDRIVE.has(xscene);
  // Raycast wheels: the drive scenes', or any scene with ?wheels=raycast (a crash on the
  // solver's own wheels, the body's rims riding them).
  const raycast = drive || qs.get('wheels') === 'raycast';
  // The lab (crash-lab.mjs, crash-lab-render.mjs): tube beams, node spheres, a ghost body,
  // time control, impact time and node grabbing. The plain views: ?lab=0, the arena, and the
  // traffic derby (more cars than the lab reads per frame).
  const lab = !arena && xscene !== 'traffic' && qs.get('lab') !== '0';
  const runup = Number(qs.get('runup')) || 0;
  page.cfg = {
    refId, family, catalog, tables, preset, scene, xscene, arena, drive, raycast, lab, P, keepVehicle,
    // ?hz=: the rate the page's step counts are in (the soft scenes' 240 Hz: ?stopAt= and
    // the damage reads' cadence).
    hz: Number(qs.get('hz')) || 240,
    runup,
    // The launch key's speed (?xspeed= km/h) and the soft scenes' spawn offsets.
    xspeed: (Number(qs.get('xspeed')) || 100) / 3.6,
    spawn: { wall: [0, -2 - runup], pole: [0, -2 - runup], limbo: [0, -2 - runup], headon: [0, -3 - runup / 2] }[scene] ?? [0, 0],
    // JBeam imports run in JBeam mode (BeamNG's rules) unless ?jbeamMode=truck.
    jbeamNative: qs.get('jbeamMode') !== 'truck',
    opt: null, // readOptions
    urls: urlBuilders(page, keepVehicle),
  };
  // Not a lab after all (a plain view): the frame built before the engine goes.
  if (page.shell && !lab) { page.shell.destroy(); page.shell = null; document.documentElement.classList.remove('lab-page'); }
  if (lab && !page.crash) page.shell?.setTitle(preset ? LAB_PRESETS[preset].label : XSCENES[xscene]);
}

/**
 * The solver's options, each a control in the Scene panel and a URL key so a link keeps it
 * (page.cfg.opt; window.__options). Surface contact is on by default for a generated car (a
 * mod with weighted skins: its collision skin is closed) and off for the others.
 */
export function readOptions(page) {
  const { qs } = page;
  page.cfg.opt = {
    surfaceContact: qs.has('surfaceContact') ? qs.get('surfaceContact') === '1' : !!page.vehicle.importSrc?.skin,
    drag: qs.get('drag') !== '0',
    centredDamping: qs.has('centredDamping') ? qs.get('centredDamping') !== '0' : null,
    // null: boxes unless the car draws its own wheels.
    wheelBoxes: qs.has('wheelBoxes') ? qs.get('wheelBoxes') === '1' : null,
  };
}

// The pages the pickers and links go to.
function urlBuilders(page, keepVehicle) {
  const { qs } = page;
  const here = (s) => { const u = new URL(location.href); u.search = s.toString(); return u.href; };
  // A scene with its default knobs: keeps the vehicle (and the lab's view options), drops
  // any preset and the old scene's knobs.
  const xscene = (id) => { const s = keepVehicle(new URLSearchParams()); s.set('xscene', id); return here(s); };
  // A reference crash: its run (the same scenario's), contact, the view options, and the
  // same scenario's changed family values (?f.<key>=).
  const ref = (id, change = {}) => {
    const s = new URLSearchParams();
    s.set('crashRef', id);
    const keep = { run: page.crash?.scenario.id === id ? String(page.crash.run) : null, contact: page.crash?.contact ?? 'edge', ...change };
    for (const [k, v] of Object.entries(keep)) if (v != null && !(k === 'run' && v === '0')) s.set(k, v);
    for (const k of ['skin3', 'color', 'skeleton', 'nodes']) if (qs.has(k)) s.set(k, qs.get(k));
    if (page.crash?.sel.id === id) for (const [k, v] of qs) if (k.startsWith('f.')) s.set(k, v);
    return here(s);
  };
  const family = (id) => { const u = new URL(ref('')); u.searchParams.delete('crashRef'); u.searchParams.set('crashFamily', id); return u.href; };
  // A preset, keeping the vehicle.
  const preset = (id) => { const s = keepVehicle(new URLSearchParams()); s.set('preset', id); return here(s); };
  // The soft (AVBD) solver's page with this URL's scene and vehicle (softbody.html).
  const soft = () => { const s = new URLSearchParams(location.search); for (const k of LAB_ONLY) s.delete(k); const q = s.toString(); return `softbody.html${q ? `?${q}` : ''}${location.hash}`; };
  return { xscene, ref, family, preset, soft };
}
