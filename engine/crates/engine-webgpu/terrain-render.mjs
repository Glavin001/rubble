// WGSL for drawing deformable terrain as one ground: the heightfield's own
// records (engine_schema::heightfield) outside an open region's window, and
// inside it the live surface the engine rebuilds every frame from the
// region's particles (engine_terrain::surface, binding T.SURFACE; see
// terrain-host.mjs { surface: true }). Material above that surface by more
// than a layer has left the ground: a page draws it, and the parcels, as
// debris. No terrain logic here, only reads of what the GPU wrote.
//
//   const wgsl = GROUND_WGSL({ hfBase, surfaceBinding: 10 });
//   // needs `hulls: array<vec4u>` (binding 18's records) declared before it.
//   groundHeight(i, j, live)   sample (i, j): live surface or heightfield
//   groundAt(x, z, live)       bilinear height at local (x, z)
//   groundNormal(i, j, live)
//   debrisParticle(k)          the k-th particle the engine listed as debris
//                              (more than a layer above the surface); draw them
//                              with drawIndirect(surfaceBuffer, 4 * T.SURFACE_DEBRIS_ARGS_WORD)
//                              after writing the vertex count to that word

/** Header words of the surface buffer (engine_terrain::surface). */
export const SURFACE_WORDS = { debrisBase: 4, window: 8, bed: 12, layer: 13, frames: 14, valid: 15, debrisArgs: 16, debrisCount: 17, heights: 24 };

/**
 * How far the last live surface is from the heightfield now (call right after
 * a bake: the bake does not touch the surface buffer, so this is the jump a
 * viewer would see at the switch). Test tooling: reads both, compares.
 * Returns { rms, mean, worst, samples } in metres over the window.
 */
export async function surfaceSwitchError(device, surfaceBuffer, hullBuffer, hfBase, nx) {
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
  const head = new Uint32Array(await read(surfaceBuffer, 0, SURFACE_WORDS.heights));
  const [i0, j0, ni, nj] = head.slice(SURFACE_WORDS.window, SURFACE_WORDS.window + 4);
  const live = new Float32Array(await read(surfaceBuffer, 4 * SURFACE_WORDS.heights, ni * nj));
  const first = j0 * nx;
  const rows = new Float32Array(await read(hullBuffer, (hfBase + 2) * 16 + first * 4, Math.ceil((nj * nx) / 4) * 4));
  let sum = 0, sq = 0, worst = 0;
  for (let j = 0; j < nj; j++) for (let i = 0; i < ni; i++) {
    const d = live[j * ni + i] - rows[j * nx + i0 + i];
    sum += d; sq += d * d;
    if (Math.abs(d) > Math.abs(worst)) worst = d;
  }
  const n = ni * nj;
  return { rms: Math.sqrt(sq / n), mean: sum / n, worst, samples: n };
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
// The live surface: valid while a region is open (the host clears it at the bake).
fn surfValid() -> bool { return surf[${SURFACE_WORDS.valid}u] == 1u; }
fn surfWindow() -> vec4u { return vec4u(surf[${SURFACE_WORDS.window}u], surf[${SURFACE_WORDS.window + 1}u], surf[${SURFACE_WORDS.window + 2}u], surf[${SURFACE_WORDS.window + 3}u]); }
fn surfLayer() -> f32 { return bitcast<f32>(surf[${SURFACE_WORDS.layer}u]); }
fn inSurf(i: u32, j: u32, live: bool) -> bool {
  if (!live || !surfValid()) { return false; }
  let w = surfWindow();
  return i >= w.x && j >= w.y && i < w.x + w.z && j < w.y + w.w;
}
fn groundHeight(i: u32, j: u32, live: bool) -> f32 {
  let d = hfDims();
  if (inSurf(i, j, live)) {
    let w = surfWindow();
    return bitcast<f32>(surf[${SURFACE_WORDS.heights}u + (j - w.y) * w.z + (i - w.x)]);
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
fn debrisParticle(k: u32) -> u32 { return surf[surf[${SURFACE_WORDS.debrisBase}u] + k]; }
`;
