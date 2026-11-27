// The crash lab page's own controls and readouts (crash-lab.html), made here with the ids the
// scripts use: the solver's options (#optRow: surface contact, raycast wheels, drag, wheel
// boxes, centred damping), the link to the soft solver's page (#solverBtn), the title and
// description (#infoTitle, #infoText), the key help (#xkeys), the camera selects (#camSel,
// #camTargetSel, #camFromLbl, #camFromSel: crash-lab-camera.mjs fills them) and the live
// readouts (#speed, #xstats, #damage, #events). In the lab, crash-lab-mount.mjs puts them in
// the lab's panels; the plain views (arena, traffic, ?lab=0) get the classic #info and #hud
// panels (plainPanels), with the scene select, the scene's knobs and beams / reset buttons.

import { vehicleControls } from './crash-lab-vehicle.mjs';

const make = (html) => { const t = document.createElement('template'); t.innerHTML = html.trim(); return t.content.firstChild; };

/** page.el: every control, made (not yet placed: plainPanels or crash-lab-mount.mjs). */
export function makeControls(page) {
  const { cfg } = page;
  const el = page.el = {};
  vehicleControls(page);
  el.optRow = make(`<div class="row" id="optRow">
      <label class="p" title="Collision triangles and their edges meet walls, poles and shapes by their surface (not only by their nodes). On by default for a generated car (a mod with weighted skins)."><input type="checkbox" id="optSurface" /> surface contact</label>
      <label class="p" title="The solver's raycast wheels and suspension on the car's hub nodes (reloads the page: the scene is rebuilt on other wheels)."><input type="checkbox" id="optRaycast" /> raycast wheels</label>
      <label class="p" title="Air drag and turbulence on the nodes"><input type="checkbox" id="optDrag" /> drag</label>
      <label class="p" title="Stand-in boxes at the raycast wheels (a car that draws its own wheels needs none)"><input type="checkbox" id="optWheelBoxes" /> wheel boxes</label>
      <label class="p" title="Beam damping about the pair's centre of mass (the BeamNG profile's)">damping <select id="optCentred"><option value="">profile default</option><option value="1">centred</option><option value="0">not centred</option></select></label>
    </div>`);
  // The soft (AVBD) solver lives on its own page: a link there with this scene and vehicle.
  el.solverBtn = make('<a id="solverBtn" class="btn" title="The soft (AVBD) solver\'s page with this scene and vehicle (softbody.html)">solver: explicit 2 kHz ⇄ soft</a>');
  el.solverBtn.href = cfg.urls.soft();
  el.infoTitle = make('<div id="infoTitle"><b>Soft-body vehicles</b> <span class="muted">— explicit node–beam solver</span></div>');
  el.infoText = make('<div class="muted" id="infoText"></div>');
  el.infoText.textContent = cfg.drive
    ? 'Fixed 0.5 ms steps (2 kHz) of springs, plastic and breakable beams and self-collision, on raycast wheels (no tyre nodes: the soft solver\'s raycast vehicle as forces on the hub nodes, so the body bends and stays bent; the wheels are drawn from its state), every step of a frame in one GPU submit.'
    : 'Fixed 0.5 ms steps (2 kHz) of springs, plastic and breakable beams, soft tyres and self-collision, every step of a frame in one GPU submit; rendered from its buffers. Crash scenes with knobs (speeds, angles, overlap), static boxes, vehicle–vehicle contact and no drivetrain: driving, moving rigid bodies and the dune buggy are on the soft solver (link above).';
  el.xkeys = make('<div class="muted" style="margin-top:6px" id="xkeys"></div>');
  el.xkeys.innerHTML = cfg.drive
    ? '<kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> drive · <kbd>Space</kbd> handbrake · <kbd>R</kbd> restart · <kbd>V</kbd> beams · <kbd>C</kbd> camera · drag to orbit, wheel to zoom'
    : 'Explicit solver: no drivetrain. <kbd>L</kbd> launch at 100 km/h · <kbd>R</kbd> restart · <kbd>V</kbd> beams (green intact, yellow→red bent, magenta broken) · <kbd>C</kbd> camera · drag to orbit, wheel to zoom';
  el.camSel = make('<select id="camSel"></select>');
  el.camTargetSel = make('<select id="camTargetSel" title="what it looks at"></select>');
  el.camFromLbl = make('<label class="p" id="camFromLbl">from</label>');
  el.camFromSel = make('<select id="camFromSel" title="what it rides on"></select>');
  el.speed = make('<div id="speed">0 <small>km/h</small></div>');
  el.xstats = make('<div class="damage" id="xstats"></div>');
  el.damage = make('<div class="damage" id="damage">—</div>');
  el.events = make('<div class="damage muted" id="events"></div>');
  el.status = document.getElementById('status');
  el.err = document.getElementById('err');
  wireOptions(page);
  // The profile each vehicle actually runs (layout.vehicles[].profile), on the solver link
  // and the title: after the first build and every restart (crash-lab-sim.mjs).
  page.labelProfile = () => {
    const prof = [...new Set(page.layout.vehicles.map((v) => ({ beamng: 'BeamNG profile', ror: 'RoR parity' })[v.profile] ?? v.profile))].join(' + ');
    el.solverBtn.textContent = `solver: explicit 2 kHz (${prof}) ⇄ soft`;
    el.infoTitle.innerHTML = `<b>Soft-body vehicles</b> <span class="muted">— explicit node–beam solver, ${prof}</span>`;
  };
}

// The options: each change goes to the URL (a link keeps it) and restarts the scene with it
// (no reload), except the wheels (the scene's build: a reload).
function wireOptions(page) {
  const { qs, cfg } = page;
  const opt = cfg.opt;
  const $ = (id) => page.el.optRow.querySelector(`#${id}`);
  const setUrl = (k, v) => { const u = new URL(location.href); if (v === null) u.searchParams.delete(k); else u.searchParams.set(k, v); history.replaceState(null, '', u.href); qs.delete(k); if (v !== null) qs.set(k, v); };
  const box = (id, key, get, set) => { const c = $(id); c.checked = get(); c.onchange = () => { set(c.checked); setUrl(key, c.checked ? '1' : '0'); page.resetScene(); }; };
  box('optSurface', 'surfaceContact', () => opt.surfaceContact, (v) => { opt.surfaceContact = v; });
  box('optDrag', 'drag', () => opt.drag, (v) => { opt.drag = v; });
  const wb = $('optWheelBoxes');
  wb.checked = opt.wheelBoxes ?? false;
  wb.onchange = () => { opt.wheelBoxes = wb.checked; setUrl('wheelBoxes', wb.checked ? '1' : '0'); };
  const cd = $('optCentred');
  cd.value = opt.centredDamping === null ? '' : opt.centredDamping ? '1' : '0';
  cd.onchange = () => { opt.centredDamping = cd.value === '' ? null : cd.value === '1'; setUrl('centredDamping', cd.value === '' ? null : cd.value); page.resetScene(); };
  const rc = $('optRaycast');
  rc.checked = cfg.raycast;
  rc.disabled = cfg.drive;
  rc.onchange = () => { const u = new URL(location.href); if (rc.checked) u.searchParams.set('wheels', 'raycast'); else u.searchParams.delete('wheels'); location.href = u.href; };
  // A reference crash's options are its scenario's.
  if (page.crash) page.el.optRow.style.display = 'none';
}

/**
 * The plain views' panels (the arena, the traffic derby, ?lab=0): #info (title, vehicle,
 * scene, the scene's knobs, options, camera, keys) and #hud (the solver link and readouts).
 */
export function plainPanels(page) {
  const { el, cfg, qs } = page;
  const { XSCENES, XPARAMS, LAB_PRESETS } = cfg.tables;
  const row = (...kids) => { const d = make('<div class="row"></div>'); d.append(...kids); return d; };
  el.xsceneSel = make('<select id="xsceneSel" title="The explicit solver\'s scene"></select>');
  for (const [id, t] of Object.entries(XSCENES)) el.xsceneSel.add(new Option(t, id, false, id === cfg.xscene));
  el.xsceneSel.onchange = () => { location.href = cfg.urls.xscene(el.xsceneSel.value); };
  el.beamsBtn = make('<button id="beamsBtn">beams (V)</button>');
  el.beamsBtn.onclick = (e) => { page.beamView = !page.beamView; e.target.blur(); };
  el.resetBtn = make('<button id="resetBtn">reset (R)</button>');
  el.resetBtn.onclick = (e) => { if (page.lab) page.lab.restart(); else page.resetScene(); e.target.blur(); };
  // The scene's knobs (XPARAMS: the catalogue's); a change rebuilds the page with them.
  const params = make('<div class="row" id="paramRow"></div>');
  const here = new URLSearchParams(location.search);
  const fields = XPARAMS[cfg.xscene] || [];
  for (const [k, unit, d, range] of fields) {
    const l = make('<label class="p"></label>'); l.textContent = unit + ' ';
    const inp = document.createElement('input'); inp.id = 'p_' + k;
    const v = here.has(k) ? here.get(k) : qs.has(k) ? qs.get(k) : d;
    if (range) {
      // A slider with its value beside it; the scene rebuilds on release.
      const [min, max, step] = range;
      Object.assign(inp, { type: 'range', min, max, step, value: v });
      const out = make('<span class="pv"></span>'); out.textContent = v;
      inp.oninput = () => { out.textContent = inp.value; };
      inp.onchange = () => run();
      inp.onkeydown = (e) => e.stopPropagation();
      l.append(inp, out);
    } else {
      Object.assign(inp, { type: 'number', step: 'any', value: v });
      inp.onkeydown = (e) => { e.stopPropagation(); if (e.key === 'Enter') run(); };
      l.appendChild(inp);
    }
    params.appendChild(l);
  }
  const run = () => {
    const u = new URLSearchParams(location.search); u.set('xscene', cfg.xscene);
    for (const [k] of fields) u.set(k, document.getElementById('p_' + k).value);
    if (cfg.preset) { const p = LAB_PRESETS[cfg.preset]; u.delete('preset'); for (const [k, v] of Object.entries(p.params)) if (!u.has(k)) u.set(k, String(v)); if (p.cam && !u.has('cam3')) u.set('cam3', p.cam); }
    location.search = u.toString();
  };
  if (fields.length) { const b = make('<button>restart</button>'); b.onclick = run; params.appendChild(b); }
  const info = make('<div class="panel" id="info"></div>');
  info.append(el.infoTitle, el.infoText,
    row(el.carSel, el.modSel, el.bindSel, el.xsceneSel, el.truckLabel, el.jbeamLabel, el.beamsBtn, el.resetBtn),
    params, el.optRow, row(make('<label class="p">camera</label>'), el.camSel, el.camTargetSel, el.camFromLbl, el.camFromSel), el.xkeys, el.truckInfo);
  const hud = make('<div class="panel" id="hud"></div>');
  const sw = document.createElement('div'); sw.append(el.solverBtn);
  hud.append(sw, el.xstats, el.speed, el.damage, el.events);
  document.getElementById('c').after(info, hud);
}
