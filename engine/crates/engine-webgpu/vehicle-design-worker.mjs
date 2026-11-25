// Worker for the vehicle lab's Design panel: the Rust generator and rigger (compiled to
// wasm: engine-vehiclegen-wasm builds the preview's render buffers, engine-vehiclerig-wasm
// lists the registered designs, their controls, validates parameters and rigs a design)
// off the main thread. Messages in: `{ id, op, ... }`; messages out: `{ id, ... }` or
// `{ id, error }`, and `{ id, progress }` while a rig runs.
//   hello                    -> { designs: [{ name, title }], materials }
//   schema { design }        -> { schema: [control], defaults }  (controls: vehicle-design-ui.mjs)
//   validate { design, params } -> {} or { invalid: the validator's message }
//   preview { design, params }  -> { parts, ms } (only the newest preview request is built)
//   rig { design, params }      -> { files: [[path, Uint8Array]], ms }: the soft-body mod's
//                                  files (the lab loads them with modFromFiles)

import initGen, { build_vehicle, material_names } from './generated/vehiclegen-wasm/engine_vehiclegen_wasm.js';
import initRig, { design_list, default_params, param_schema, validate_params, rig_vehicle } from './generated/vehiclerig-wasm/engine_vehiclerig_wasm.js';

const ready = Promise.all([initGen(), initRig()]);
let pendingPreview = null;
let busy = false;
const queue = [];

const msg = (e) => (e && e.message ? e.message : String(e));

self.onmessage = (e) => {
  const req = e.data;
  if (req.op === 'preview') {
    // Newest wins: a slider dragged while a build runs replaces the queued request.
    if (pendingPreview) self.postMessage({ id: pendingPreview.id, superseded: true });
    pendingPreview = req;
  } else {
    queue.push(req);
  }
  if (!busy) run();
};

async function run() {
  busy = true;
  try {
    await ready;
    while (queue.length || pendingPreview) {
      const req = queue.length ? queue.shift() : pendingPreview;
      if (req === pendingPreview) pendingPreview = null;
      try {
        self.postMessage({ id: req.id, ...handle(req) }, transfers);
      } catch (err) {
        self.postMessage({ id: req.id, error: msg(err) });
      }
      transfers = [];
      // (let queued messages in between builds)
      await new Promise((r) => setTimeout(r));
    }
  } catch (err) {
    self.postMessage({ id: -1, error: `the Rust modules did not load: ${msg(err)}` });
  }
  busy = false;
}

let transfers = [];

function handle(req) {
  switch (req.op) {
    case 'hello':
      return { designs: JSON.parse(design_list()), materials: material_names() };
    case 'schema':
      return { schema: JSON.parse(param_schema(req.design)), defaults: JSON.parse(default_params(req.design)) };
    case 'validate':
      try { validate_params(req.design, JSON.stringify(req.params)); return {}; } catch (e) { return { invalid: msg(e) }; }
    case 'preview': {
      const t0 = performance.now();
      const b = build_vehicle(req.design, JSON.stringify(req.params), false);
      const parts = [];
      for (let i = 0; i < b.part_count(); i++) {
        const p = { name: b.part_name(i), positions: b.positions(i), indices: b.indices(i), groups: b.groups(i), normals: b.normals(i), colors: b.colors(i) };
        parts.push(p);
        transfers.push(p.positions.buffer, p.indices.buffer, p.groups.buffer, p.normals.buffer, p.colors.buffer);
      }
      const out = { parts, paint: b.paint(), triangles: b.triangle_count(), ms: performance.now() - t0 };
      b.free();
      return out;
    }
    case 'rig': {
      const t0 = performance.now();
      self.postMessage({ id: req.id, progress: 'rigging the lattice and binding the body (Rust, WebAssembly)…' });
      const rf = rig_vehicle(req.design, JSON.stringify(req.params));
      const tRig = performance.now() - t0;
      const files = [];
      for (let i = 0; i < rf.count(); i++) {
        const bytes = rf.take(i);
        files.push([rf.path(i), bytes]);
        transfers.push(bytes.buffer);
      }
      rf.free();
      return { files, ms: performance.now() - t0, rigMs: tRig };
    }
    default:
      throw new Error(`unknown request '${req.op}'`);
  }
}
