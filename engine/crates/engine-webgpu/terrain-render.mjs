// WGSL for drawing deformable terrain as one ground: the heightfield's own
// records (engine_schema::heightfield) outside the open areas' windows, and
// inside each the live surface the engine rebuilds every frame from that
// area's particles (engine_terrain::surface, binding T.SURFACE; see
// terrain-host.mjs { surface: true }). Each open area is a view of the
// surface buffer (its window, its heights). Material above that surface by
// more than a layer has left the ground: a page draws it, and the parcels, as
// debris. No terrain logic here, only reads of what the GPU wrote.
//
//   const wgsl = GROUND_WGSL({ hfBase, surfaceBinding: 10 });
//   // needs `hulls: array<vec4u>` (binding 18's records) declared before it.
//   groundHeight(i, j, live)   sample (i, j): live surface or heightfield
//   groundAt(x, z, live)       bilinear height at local (x, z)
//   groundNormal(i, j, live)
//   surfViews(), surfOpen(v), surfViewWindow(v)   the open areas' windows
//   debrisParticle(k)          the k-th particle the engine listed as debris
//                              (more than a layer above the surface); draw them
//                              with drawIndirect(surfaceBuffer, 4 * T.SURFACE_DEBRIS_ARGS_WORD)
//                              after writing the vertex count to that word
import { T } from './terrain-host.mjs';

/** Words of the surface buffer (engine_terrain::surface). Header: [0] views
 * to scan, [4] the debris list's first word, [14] frames, [15] valid,
 * [16..20) the debris draw's arguments; then a view record per area slot. */
export const SURFACE_WORDS = {
  views: 0, debrisBase: 4, frames: 14, valid: T.SURFACE_VALID_WORD, debrisArgs: T.SURFACE_DEBRIS_ARGS_WORD, debrisCount: T.SURFACE_DEBRIS_ARGS_WORD + 1,
  view: T.SURFACE_VIEW_WORD, viewWords: T.SURFACE_VIEW_WORDS, maxViews: T.MAX_SURFACE_VIEWS,
  // In a view record: flags (bit T.SURFACE_VIEW_OPEN), layers, heights' first word, …, window, bed, layer height.
  vFlags: 0, vHeights: 2, vWindow: T.SURFACE_VIEW_WINDOW, vBed: T.SURFACE_VIEW_WINDOW + 4, vLayer: T.SURFACE_VIEW_WINDOW + 5,
};
const W = SURFACE_WORDS;

/**
 * How far the last live surface is from the heightfield now (call right after
 * a bake: the bake does not touch the surface buffer, so this is the jump a
 * viewer would see at the switch). Test tooling: reads both, compares.
 * `slots`: the area slots that baked (default: every view the last surface
 * pass showed). Returns { rms, mean, worst, samples } in metres over their
 * windows.
 */
export async function surfaceSwitchError(device, surfaceBuffer, hullBuffer, hfBase, nx, slots = null) {
  const read = async (buf, offset, words) => {
    const rb = device.createBuffer({ size: words * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, offset, rb, 0, words * 4);
    device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = rb.getMappedRange().slice(0);
    rb.unmap(); rb.destroy();
    return out;
  };
  const table = new Uint32Array(await read(surfaceBuffer, 4 * W.view, W.viewWords * W.maxViews));
  let sum = 0, sq = 0, worst = 0, n = 0;
  for (let s = 0; s < W.maxViews; s++) {
    const v = table.subarray(s * W.viewWords, (s + 1) * W.viewWords);
    if (!(v[W.vFlags] & T.SURFACE_VIEW_OPEN) || (slots && !slots.includes(s))) continue;
    const [i0, j0, ni, nj] = v.subarray(W.vWindow, W.vWindow + 4);
    const live = new Float32Array(await read(surfaceBuffer, 4 * v[W.vHeights], ni * nj));
    const rows = new Float32Array(await read(hullBuffer, (hfBase + 2) * 16 + j0 * nx * 4, Math.ceil((nj * nx) / 4) * 4));
    for (let j = 0; j < nj; j++) for (let i = 0; i < ni; i++) {
      const d = live[j * ni + i] - rows[j * nx + i0 + i];
      sum += d; sq += d * d; n++;
      if (Math.abs(d) > Math.abs(worst)) worst = d;
    }
  }
  return { rms: n ? Math.sqrt(sq / n) : 0, mean: n ? sum / n : 0, worst, samples: n };
}

export const GROUND_WGSL = ({ hfBase, surfaceBinding }) => /* wgsl */`
@group(0) @binding(${surfaceBinding}) var<storage, read> surf: array<u32>;
const HF_BASE = ${hfBase}u;
struct HfDims { nx: u32, nz: u32, dx: f32, dz: f32, x0: f32, z0: f32 };
fn hfDims() -> HfDims {
  let head = hulls[HF_BASE];
  let d = bitcast<vec4f>(hulls[HF_BASE + 1u]);
  return HfDims(head.y, head.z, d.x, d.y, -0.5 * f32(head.y - 1u) * d.x, -0.5 * f32(head.z - 1u) * d.y);
}
fn hfH(i: u32, j: u32, nx: u32) -> f32 { let k = j * nx + i; let r = hulls[HF_BASE + 2u + (k >> 2u)]; return bitcast<f32>(r[k & 3u]); }
// The live surface: valid while an area is open (the host clears it at the last bake).
fn surfValid() -> bool { return surf[${W.valid}u] == 1u; }
// Views to scan (area slots up to the highest open one).
fn surfViews() -> u32 { return min(surf[${W.views}u], ${W.maxViews}u); }
fn surfView(v: u32, f: u32) -> u32 { return surf[${W.view}u + v * ${W.viewWords}u + f]; }
fn surfOpen(v: u32) -> bool { return surfValid() && (surfView(v, ${W.vFlags}u) & ${T.SURFACE_VIEW_OPEN}u) != 0u; }
fn surfViewWindow(v: u32) -> vec4u { return vec4u(surfView(v, ${W.vWindow}u), surfView(v, ${W.vWindow + 1}u), surfView(v, ${W.vWindow + 2}u), surfView(v, ${W.vWindow + 3}u)); }
// The open view whose window holds sample (i, j), or 0xFFFFFFFF.
fn surfViewAt(i: u32, j: u32) -> u32 {
  for (var v = 0u; v < surfViews(); v++) {
    if (!surfOpen(v)) { continue; }
    let w = surfViewWindow(v);
    if (i >= w.x && j >= w.y && i < w.x + w.z && j < w.y + w.w) { return v; }
  }
  return 0xFFFFFFFFu;
}
// The first open view (0xFFFFFFFF: none), for single-area tools.
fn surfFirst() -> u32 {
  for (var v = 0u; v < surfViews(); v++) { if (surfOpen(v)) { return v; } }
  return 0xFFFFFFFFu;
}
// The first open view's window (zero without one).
fn surfWindow() -> vec4u { let v = surfFirst(); if (v == 0xFFFFFFFFu) { return vec4u(0u); } return surfViewWindow(v); }
// The layer height (one cell size over the domain: any open view's).
fn surfLayer() -> f32 { let v = surfFirst(); if (v == 0xFFFFFFFFu) { return 0.0; } return bitcast<f32>(surfView(v, ${W.vLayer}u)); }
fn inSurf(i: u32, j: u32, live: bool) -> bool { return live && surfViewAt(i, j) != 0xFFFFFFFFu; }
fn groundHeight(i: u32, j: u32, live: bool) -> f32 {
  let d = hfDims();
  if (live) {
    let v = surfViewAt(i, j);
    if (v != 0xFFFFFFFFu) {
      let w = surfViewWindow(v);
      return bitcast<f32>(surf[surfView(v, ${W.vHeights}u) + (j - w.y) * w.z + (i - w.x)]);
    }
  }
  return hfH(i, j, d.nx);
}
fn groundNormal(i: u32, j: u32, live: bool) -> vec3f {
  let d = hfDims();
  let hx0 = groundHeight(max(i, 1u) - 1u, j, live); let hx1 = groundHeight(min(i + 1u, d.nx - 1u), j, live);
  let hz0 = groundHeight(i, max(j, 1u) - 1u, live); let hz1 = groundHeight(i, min(j + 1u, d.nz - 1u), live);
  return normalize(vec3f(-(hx1 - hx0) / (2.0 * d.dx), 1.0, -(hz1 - hz0) / (2.0 * d.dz)));
}
fn groundAt(x: f32, z: f32, live: bool) -> f32 {
  let d = hfDims();
  let fx = clamp((x - d.x0) / d.dx, 0.0, f32(d.nx - 1u) - 0.001);
  let fz = clamp((z - d.z0) / d.dz, 0.0, f32(d.nz - 1u) - 0.001);
  let i = u32(fx); let j = u32(fz);
  let u = fx - f32(i); let v = fz - f32(j);
  return mix(mix(groundHeight(i, j, live), groundHeight(i + 1u, j, live), u),
             mix(groundHeight(i, j + 1u, live), groundHeight(i + 1u, j + 1u, live), u), v);
}
// The k-th particle terrain_surface_debris listed (k < the debris count).
fn debrisParticle(k: u32) -> u32 { return surf[surf[${W.debrisBase}u] + k]; }
`;
