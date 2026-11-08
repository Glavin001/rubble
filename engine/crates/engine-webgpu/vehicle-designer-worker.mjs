// Worker for vehicle-designer.html: builds vehicles with the Rust generator
// (engine-vehiclegen, compiled to wasm) off the main thread. Only the newest
// request is built: while a build runs, newer slider values replace any
// queued one, so dragging a slider never piles up work.

import init, { build_vehicle, default_params, material_names } from './generated/vehiclegen-wasm/engine_vehiclegen_wasm.js';

const ready = init();
let pending = null;
let busy = false;

self.onmessage = (e) => {
  const req = e.data;
  if (req.type === 'hello') {
    ready.then(() => self.postMessage({ type: 'hello', materials: material_names(), defaults: JSON.parse(default_params(req.design)) }))
      .catch((err) => self.postMessage({ type: 'error', error: String(err) }));
    return;
  }
  pending = req;
  if (!busy) run();
};

async function run() {
  busy = true;
  await ready;
  while (pending) {
    const req = pending;
    pending = null;
    try {
      const t0 = performance.now();
      const b = build_vehicle(req.design, JSON.stringify(req.params), !!req.uv);
      const ms = performance.now() - t0;
      if (req.type === 'glb') {
        const bytes = b.to_glb();
        b.free();
        self.postMessage({ type: 'glb', id: req.id, ms, glb: bytes }, [bytes.buffer]);
        continue;
      }
      const parts = [];
      const transfer = [];
      for (let i = 0; i < b.part_count(); i++) {
        const p = {
          name: b.part_name(i),
          positions: b.positions(i),
          normals: b.normals(i),
          colors: b.colors(i),
          indices: b.indices(i),
          groups: b.groups(i),
        };
        parts.push(p);
        transfer.push(p.positions.buffer, p.normals.buffer, p.colors.buffer, p.indices.buffer, p.groups.buffer);
      }
      const msg = { type: 'build', id: req.id, ms, parts, triangles: b.triangle_count(), markers: JSON.parse(b.markers_json()) };
      b.free();
      self.postMessage(msg, transfer);
    } catch (err) {
      self.postMessage({ type: 'error', id: req.id, error: err && err.message ? err.message : String(err) });
    }
  }
  busy = false;
}
