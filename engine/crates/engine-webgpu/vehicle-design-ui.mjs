// The vehicle lab's Design panel: pick a registered design, edit its parameters (controls
// generated from the design's own schema: engine-vehiclegen's `param_schema_json`, whose
// ranges are the ones its `validate()` enforces; no design or parameter is named here),
// watch a live preview, press Apply to rig it into a soft-body car and drive it.
//
// The Rust runs in a worker (vehicle-design-worker.mjs): the preview is built on every
// edit (about 0.4 s), the rig on Apply only (4 to 5 s for the six designs, a 70-90 MB mod).
//
//   const panel = createDesignPanel(root, { onApply({ design, title, params, files, ms }) {} });
//   panel.ready  // resolves when the worker has listed the designs
import { createDesignPreview } from './design-preview.mjs';

const label = (k) => k.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase());
const srgb = (l) => Math.round(255 * (l <= 0.0031308 ? 12.92 * l : 1.055 * l ** (1 / 2.4) - 0.055)).toString(16).padStart(2, '0');
const lin = (h) => { const c = parseInt(h, 16) / 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
const toHex = (rgb) => `#${rgb.map((l) => srgb(Math.min(1, Math.max(0, l)))).join('')}`;
const fromHex = (h) => [lin(h.slice(1, 3)), lin(h.slice(3, 5)), lin(h.slice(5, 7))];
const decimals = (step) => Math.max(0, Math.min(4, Math.ceil(-Math.log10(step) - 1e-9)));

export function createDesignPanel(root, { onApply }) {
  const el = (tag, props = {}, ...kids) => { const e = Object.assign(document.createElement(tag), props); e.append(...kids); return e; };
  const worker = new Worker(new URL('./vehicle-design-worker.mjs', import.meta.url), { type: 'module' });
  let seq = 0;
  const waiting = new Map();
  worker.onmessage = (e) => {
    const m = e.data, w = waiting.get(m.id);
    if (m.id === -1) { for (const [, x] of waiting) x.reject(new Error(m.error)); waiting.clear(); return; }
    if (!w) return;
    if (m.progress) { w.progress?.(m.progress); return; }
    waiting.delete(m.id);
    if (m.error) w.reject(new Error(m.error)); else w.resolve(m);
  };
  worker.onerror = (e) => { for (const [, x] of waiting) x.reject(new Error(`the design worker failed: ${e.message || 'no message'}`)); waiting.clear(); };
  const call = (msg, progress) => new Promise((resolve, reject) => { const id = ++seq; waiting.set(id, { resolve, reject, progress }); worker.postMessage({ id, ...msg }); });

  // ---- the panel's DOM
  const select = el('select', { id: 'designPick', title: 'The generated design to edit: every design the build registers (engine-vehiclegen), each with the parameters it has.' });
  const controlsBox = el('div', { id: 'designControls' });
  const canvas = el('canvas', { id: 'designPreview', style: 'width:100%;height:190px;border-radius:6px;background:#10141c;touch-action:none;cursor:grab', title: 'Drag to orbit, wheel to zoom' });
  const stats = el('div', { className: 'note', textContent: 'starting the Rust modules…' });
  const msg = el('div', { className: 'note', style: 'color:#ffb4a0;white-space:pre-wrap' });
  const apply = el('button', { id: 'designApply', textContent: 'Apply and drive', disabled: true, title: 'Rigs this design with the soft-body physics (4 to 5 s) and makes it a car of the lab: pick it under Car ("Your design"), or in Mixed it is one more type. The lab rebuilds with it.' });
  const reset = el('button', { id: 'designReset', textContent: 'Reset', title: 'The design\'s default parameters' });
  const status = el('span', { className: 'note', id: 'designStatus' });
  root.append(el('div', { className: 'row' }, el('label', { className: 'muted', htmlFor: 'designPick', textContent: 'Design' }), select), canvas, stats, controlsBox, msg,
    el('div', { className: 'row' }, apply, reset, status));

  let preview = null;
  try { preview = createDesignPreview(canvas); } catch (e) { stats.textContent = `no preview: ${e.message}`; }
  new ResizeObserver(() => preview?.resize()).observe(canvas);

  let designs = [], design = null, defaults = null, params = null, valid = false, rigging = false, previewGen = 0, timer = 0;
  const title = () => designs.find((d) => d.name === design)?.title ?? design;

  async function refresh() {
    clearTimeout(timer);
    const gen = ++previewGen;
    const v = await call({ op: 'validate', design, params });
    if (gen !== previewGen) return;
    valid = !v.invalid;
    msg.textContent = v.invalid ? `${v.invalid}\nThis combination is not built: change a parameter.` : '';
    apply.disabled = !valid || rigging;
    if (!valid) return;
    try {
      const b = await call({ op: 'preview', design, params });
      if (gen !== previewGen || b.superseded) return;
      preview?.show(b);
      stats.textContent = `${title()}: ${b.triangles.toLocaleString()} triangles, built in ${b.ms.toFixed(0)} ms (Rust, WebAssembly)`;
    } catch (e) { msg.textContent = e.message; }
  }
  const later = () => { clearTimeout(timer); timer = setTimeout(() => refresh().catch((e) => { msg.textContent = e.message; }), 60); };

  function buildControls(schema) {
    controlsBox.textContent = '';
    for (const c of schema) {
      const id = `dp_${c.key}`;
      const set = (v) => { params[c.key] = v; later(); };
      let row;
      if (c.kind === 'number') {
        const dec = decimals(c.step), out = el('b', { textContent: (+params[c.key]).toFixed(dec) });
        const input = el('input', { type: 'range', id, min: c.min, max: c.max, step: c.step, value: params[c.key] });
        input.oninput = () => { out.textContent = (+input.value).toFixed(dec); set(+input.value); };
        row = el('div', { className: 'row' }, el('label', { className: 'muted', htmlFor: id, textContent: label(c.key), style: 'min-width:8.5em' }), input, out);
      } else if (c.kind === 'choice') {
        const input = el('select', { id }, ...c.options.map((o) => new Option(o, o)));
        input.value = params[c.key];
        input.onchange = () => set(input.value);
        row = el('div', { className: 'row' }, el('label', { className: 'muted', htmlFor: id, textContent: label(c.key), style: 'min-width:8.5em' }), input);
      } else if (c.kind === 'bool') {
        const input = el('input', { type: 'checkbox', id, checked: !!params[c.key] });
        input.onchange = () => set(input.checked);
        row = el('div', { className: 'row' }, el('label', {}, input, ` ${label(c.key)}`));
      } else if (c.kind === 'color') {
        const input = el('input', { type: 'color', id, value: toHex(params[c.key]) });
        input.oninput = () => set(fromHex(input.value));
        row = el('div', { className: 'row' }, el('label', { className: 'muted', htmlFor: id, textContent: label(c.key), style: 'min-width:8.5em' }), input);
      }
      if (row) controlsBox.append(row);
    }
  }
  let schema = null;
  async function choose(name) {
    design = name;
    select.value = name;
    const r = await call({ op: 'schema', design: name });
    schema = r.schema; defaults = r.defaults; params = structuredClone(defaults);
    buildControls(schema);
    await refresh();
  }
  select.onchange = () => choose(select.value).catch((e) => { msg.textContent = e.message; });
  reset.onclick = () => { params = structuredClone(defaults); buildControls(schema); refresh(); };
  apply.onclick = async () => {
    if (!valid || rigging) return;
    rigging = true; apply.disabled = true; select.disabled = true;
    const t0 = performance.now();
    const tick = setInterval(() => { status.textContent = `rigging… ${((performance.now() - t0) / 1000).toFixed(0)} s`; }, 250);
    status.textContent = 'rigging…';
    try {
      const r = await call({ op: 'rig', design, params: structuredClone(params) }, (t) => { status.dataset.step = t; });
      status.textContent = `rigged in ${(r.ms / 1000).toFixed(1)} s (${(r.files.reduce((a, f) => a + f[1].length, 0) / 1e6).toFixed(0)} MB): building the lab…`;
      await onApply({ design, title: title(), params: structuredClone(params), files: r.files, ms: r.ms, setStatus: (t) => { status.textContent = t; } });
    } catch (e) {
      msg.textContent = e.message; status.textContent = '';
    } finally {
      clearInterval(tick); rigging = false; apply.disabled = !valid; select.disabled = false;
    }
  };

  const ready = call({ op: 'hello' }).then(async (h) => {
    designs = h.designs;
    for (const d of designs) select.add(new Option(d.title, d.name));
    await choose(designs[0].name);
  }).catch((e) => { stats.textContent = ''; msg.textContent = `The design tools did not load: ${e.message}. Build them with engine/scripts/build-web.sh.`; });
  return {
    ready,
    /** Switch to a registered design by name (no-op for a name that is not one). */
    setDesign: async (name) => { await ready; if (designs.some((d) => d.name === name) && name !== design) await choose(name); },
    get design() { return design; },
    get params() { return params; },
    destroy() { worker.terminate(); preview?.destroy(); },
  };
}
