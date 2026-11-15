// Renderer of the authored showcase vehicles (engine_softbody::authoring): a dense
// visual mesh skinned on the GPU from the explicit solver's node buffer, flexbody
// style. All geometry, materials and bindings come from the wasm
// (`showcaseCar(params).visual`); this file only uploads them, deforms them in a
// compute pass and shades them.
//
// Interface
// ---------
//   import { createShowcaseCarRenderer } from './showcase-car-render.mjs';
//   const car = showcaseCar({ style: 'sedan', resolution: 'medium' });   // wasm export
//   const sim = world.explicitSim({ vehicles: [{ truck: car.truck, at: [0, 0] }], scene: 'idle' });
//   const r = createShowcaseCarRenderer(device, { format, depthFormat: 'depth24plus', sampleCount: 1 });
//   r.setSolver(sim.nodesBuffer(), sim.miscBuffer(), sim.beamsBuffer());  // again whenever the sim is rebuilt
//   const v = layout.vehicles[i];
//   const id = r.addCar(car.visual, { nodeOffset: v.nodeOffset, beamOffset: v.beamOffset, vehicle: i });
//   // each frame, after sim.frame(steps):
//   r.update(encoder, { dt, floorY, boxes });                // compute: deform every car, debris
//   r.drawReflection(pass, { viewProj, eye, light });        // optional: after a floor at y 0
//   r.draw(pass, { viewProj, eye, light });                  // in a render pass with depth
//   r.removeCar(id); r.destroy();
//
// `viewProj` is a column-major Float32Array(16) (WebGPU clip space, z in 0..1), `eye`
// the camera position [x, y, z], `light` the direction toward the sun.
//
// Node buffer (ExplicitSim.nodesBuffer): 3 vec4<f32> per node, every vehicle's nodes
// in order: (position relative to the vehicle's origin, mass), velocity, force.
// Misc buffer (ExplicitSim.miscBuffer): vec4<u32> per vehicle, its origin's f32 bits.
//
// Binding (per vertex, from the wasm): the solver nodes n0 n1 n2 (local indices; the
// vehicle's tyre nodes included), (α, β, h), the rest normal in the triangle's frame
// (t, b, n) and the triangle's rest edge lengths r1, r2:
//   e1 = P1 − P0, e2 = P2 − P0, N = normalize(e1 × e2)
//   position = P0 + α e1 + β e2 + h N
//   T = normalize(e1), B = N × T, normal = t T + b B + n N
// A vertex whose triangle has stretched past 2.5 × a rest edge is torn: its triangles
// are discarded (a tear, not a stretched sheet). Body vertices (`visual.smooth`) shade
// smoothly: their rest normal moved by the change of their nodes' skin normals,
// weighted (1 − α − β, α, β); node normals are computed each frame from the skin
// triangles (`normalTris`, per node `nodeTriStart`/`nodeTriList`, `nodeRestNormal`).
//
// Damage follows the physics (engine/docs/beamng-realism-plan.md P5): each
// damage unit (a glass pane, lamp or mirror: `visual.unit*`) is a set of the
// solver's beams; a pass reads their broken flags and rest lengths from the
// beam buffer every update, and a unit is broken when one of its beams is, or
// when its live beams' plastic strain passes its trigger (tempered glass's
// break strain, a lens's). Then its glass, lenses and mirror glass are
// dropped, its lamps go dark and its shards are thrown (skin.rs
// unit_states() is the same on the CPU). The laminated windscreen cracks
// where its beams yielded past annealed glass's strain. Per node the pass
// also keeps its plastic strain (the paint loses its gloss) and the metres it
// slid in contact with the floor or a static box (scratches to primer and
// steel; metal throws sparks).
//
// Debris (fx-particles.mjs, shared with car-render.mjs): sparks from metal
// nodes sliding on a surface faster than 2 m/s, glass and lens shards from a
// breaking unit (area × skin.rs SHARD_DENSITY), paint and plastic chips where
// the skin tears or a panel yields suddenly, dust on hard landings. Only
// with `update`'s dt > 0 (simulated seconds since the last update).
//
// Materials (ids, `visual.materials`): paint (metallic flake base under a
// clearcoat), glass, chrome, trim (gloss black plastic), headlight (projector,
// emissive), taillight, tyre, rim, gap, underbody, plate, grille, seat, dash,
// doorcard, headliner, carpet, lens (clear lamp lens), drl (emissive white),
// reflector, tail_lens (red), tail_led (emissive red), brake, caliper,
// tail_back. Glass and both lenses are drawn blended in a second pass. Base
// colours in `visual.colors` (linear RGB; the paint from the wasm's `paint`
// param), baked ambient occlusion in `visual.ao`. Reflections come from a
// procedural studio (cyclorama, overhead softbox, side strips, a key box).

import { FX_STRUCT_WGSL, FX_COMPUTE_WGSL, FX_DRAW_WGSL, FX_BLEND, fxBufferSize } from './fx-particles.mjs';

const WG = 64;

// Debris particles (fx-particles.mjs) in the ring buffer shared by every car.
const FX_N = 8192;
// Static surfaces the sparks and scrapes know about (layout.boxes rows).
const FX_BOXES = 16;
// Annealed glass's strain at which the windscreen cracks (skin.rs
// LAMINATED_CRACK_STRAIN: 45 MPa over 70 GPa).
const LAMINATED_CRACK_STRAIN = 45e6 / 70e9;
// Crease sharpness (skin.rs CREASE_SHARP): the envelope's temperature over
// the binding triangle's size. A fold whose planes differ by θ switches
// planes over about CREASE_SHARP · L / sin θ.
const CREASE_SHARP = 0.08;
// The most the envelope moves a vertex off its triangle beyond its rest
// offset (skin.rs ENVELOPE_MAX): half the 2 cm contact band.
const ENVELOPE_MAX = 0.01;
// Level of detail: the finest level whose median edge spans this many pixels
// (3: at 1080p the 10-car row at 16 m and 30 m looks the same on the second
// level as on the full mesh; finer triangles only add quad overshading).
const LOD_PX = 3.0;
// A node's skin splits into two planes (a fold through it) where its most
// deviant triangle is more than 20° from its mean normal (skin.rs SPLIT_COS).
const SPLIT_COS = Math.cos(20 * Math.PI / 180);

const SKIN_WGSL = /* wgsl */ `
struct Bind { n0: u32, n1: u32, n2: u32, vehicle: u32, a: f32, b: f32, h: f32, r1: f32, nt: f32, nb: f32, nn: f32, r2: f32, rx: f32, ry: f32, rz: f32, flags: u32 };
// topo (u32 words) holds the car's tables at these offsets: per node its
// triangle range (nodeCount + 1 starts), the triangle list (listAt), the
// triangles' nodes and rest perimeter (4 words each, trisAt), rest normals and
// positions (8 words a node, restNAt), per node its beams (nodeCount + 1 starts then the
// list, nodeBeamAt), every beam's spawn rest length (rest0At), the damage
// units (4 words each: beam list start and end in topo, kind, strain trigger;
// then the lists, unitAt) and per node its surface material (nodeMatAt).
// Shut-line fallbacks: a Bind's word 3 (\`vehicle\`, unused here) is 0 or
// the index in binds of its own-part binding (after the count vertices),
// whose word 3 is its frame's death record (4 beam ids at deathAt).
struct Params { count: u32, nodeOffset: u32, nodeCount: u32, vehicle: u32, listAt: u32, trisAt: u32, beamOffset: u32, level: u32,
  restNAt: u32, nodeBeamAt: u32, rest0At: u32, unitAt: u32, unitCount: u32, nodeMatAt: u32, paint: u32, stateAt: u32, deathAt: u32,
  wheelAt: u32 };
// The frame's effects: simulated seconds since the last update (0: nothing
// moves, spawns or accumulates), a seed, the floor and the static boxes
// (centre, kind; half extents) the sparks and scrapes test.
// lead.x: seconds the shown state is carried past the solver's (node velocity ×
// lead; FramePacer.lead: smooth motion between fixed steps in slow motion).
struct Fx { dt: f32, seed: u32, floorY: f32, boxes: u32, lead: vec4f, box: array<vec4f, ${2 * FX_BOXES}> };
@group(0) @binding(0) var<storage, read> nodes: array<vec4f>;
@group(0) @binding(1) var<storage, read> misc: array<vec4u>;
@group(0) @binding(2) var<storage, read> binds: array<Bind>;
// Per vertex 2 vec4: (position, tear and damage as two f16), (normal,
// plastic strain and scrape as two f16): OUT_LAYOUT.
@group(0) @binding(3) var<storage, read_write> outv: array<vec4f>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read> topo: array<u32>;
@group(0) @binding(6) var<storage, read> beams: array<vec4f>;
// Per node NS (6) vec4: (plane a, tear), (last world position, valid), (plastic
// strain, broken beams, paint worn (µm), 0); then per damage unit
// (broken, strain, shards due, broken as of the last moving update) at
// stateAt; then the level of detail state (picked, picked last update) at
// stateAt + unitCount.
@group(0) @binding(7) var<storage, read_write> nodeN: array<vec4f>;
@group(0) @binding(8) var<uniform> fxu: Fx;
${FX_STRUCT_WGSL(FX_N)}
@group(0) @binding(9) var<storage, read_write> fx: FxParticles;
${FX_COMPUTE_WGSL(FX_N, 'fxu.seed')}
// Level of detail: per car its levels' sizes and switch distance (lod), the
// eye (view, from the last draw), and the indirect args the pick writes: per
// level a dispatch (3 words) and four indexed draws (opaque, glass, the
// opaque triangles near the floor for the contact shadows, and those under
// 0.95 m for the floor reflection, which has faded to 3 % there; 5 words each).
// Per level (up to 3): vertices, opaque and glass index counts, the floor
// passes' counts, the median edge (m); then the level count.
struct Lod { n: vec4u, opq: vec4u, gl: vec4u, low: vec4u, mid: vec4u, edge: vec4f, levels: vec4u };
@group(1) @binding(0) var<uniform> lod: Lod;
@group(1) @binding(1) var<storage, read_write> args: array<u32>;
@group(1) @binding(2) var<uniform> view: vec4f;

const LAMINATED = 1u;
const CREASE_SHARP = ${CREASE_SHARP};
const ENVELOPE_MAX = ${ENVELOPE_MAX};
const SPLIT_COS = ${SPLIT_COS};
// Per node: (plane a now, tear), (last world position, valid), (plastic
// strain, broken beams, scrape, 0), (plane b now, two planes), plane a at
// rest (and two), plane b at rest.
const NS = 6u;
const ANNEALED_CRACK = ${LAMINATED_CRACK_STRAIN};
// A node within this of a surface (its collision range) is in contact.
const CONTACT = 0.03;
// Paint wear (Archard: worn depth = K · p · s / H, s the distance slid in
// contact). K = 0.05: two-body abrasive wear against a rough counterface
// (concrete, a road) is K ≈ 10⁻² … 10⁻¹ (Rabinowicz, Friction and Wear of
// Materials, 2nd ed., 1995); the middle of that range is a Choice. H = 0.2
// GPa: a crosslinked automotive clearcoat's indentation hardness, a Choice
// (published nanoindentation studies measure these coats; we found no value
// to cite). p = 1 MPa: a Choice, because the solver does not expose its
// contact forces (a 10 kN contact on 1 cm² of real contact). Depth per metre
// slid: K p / H = 250 µm. Layers (typical OEM build-up, Streitberger &
// Dössel, Automotive Paints and Coatings, 2nd ed., 2008): clearcoat 45 µm,
// basecoat 20 µm, primer 25 µm, e-coat 20 µm: primer shows past 65 µm worn,
// bare steel past 110 µm.
const WEAR_K = 0.05;
const WEAR_P = 1.0e6;
const WEAR_H = 0.2e9;
const WEAR_UM_PER_M = ${1e6} * WEAR_K * WEAR_P / WEAR_H;

fn origin() -> vec3f {
  let o = misc[params.vehicle];
  return vec3f(bitcast<f32>(o.x), bitcast<f32>(o.y), bitcast<f32>(o.z));
}
fn nodePos(i: u32) -> vec3f { let k = (params.nodeOffset + i) * 3u; return nodes[k].xyz + nodes[k + 1u].xyz * fxu.lead.x; }
fn beamBroken(j: u32) -> bool { return beams[(params.beamOffset + j) * 2u + 1u].z != 0.0; }
fn beamRest(j: u32) -> f32 { return beams[(params.beamOffset + j) * 2u].x; }
fn restN(i: u32) -> vec3f {
  let k = params.restNAt + 8u * i;
  return vec3f(bitcast<f32>(topo[k]), bitcast<f32>(topo[k + 1u]), bitcast<f32>(topo[k + 2u]));
}
fn restP(i: u32) -> vec3f {
  let k = params.restNAt + 8u * i + 4u;
  return vec3f(bitcast<f32>(topo[k]), bitcast<f32>(topo[k + 1u]), bitcast<f32>(topo[k + 2u]));
}

// Creases (skin.rs crease_envelope; the CPU mirror). Each binding node has
// a tangent plane through it (its skin normal), or two where a fold runs
// through the node (its skin triangles fall in two groups more than
// SPLIT_COS apart: one plane per side, so a node on the fold line carries
// both panels' planes, not a bevel between them). The vertex's point on the
// binding triangle (barycentric w) sits at height d over each plane, and the
// surface follows the plane nearest it: at a fold the two sides' planes
// cross along a straight line, so the fold is straight and sharp across
// while smooth along it, whatever the binding triangles' grid; where the
// planes barely differ (a mild bend) the weights are the barycentric ones
// (Phong-like, smooth). Weights, per node i and plane c:
//   s = w_i exp(−(|t| − min|t|) / (CREASE_SHARP · L)), t the distance to the plane along u
// The point moves along u (the planes' barycentric mean normal) to the
// plane it follows (e). Returns the weighted normal n, u and e.
struct Env { n: vec3f, u: vec3f, e: f32 };
fn envelope(p: array<vec3f, 3>, na: array<vec3f, 3>, nb: array<vec4f, 3>, w: vec3f, l: f32) -> Env {
  let pl = w.x * p[0] + w.y * p[1] + w.z * p[2];
  let k = 1.0 / (CREASE_SHARP * max(l, 1e-4));
  // The direction the point moves in: the planes' barycentric mean normal
  // (continuous across the binding triangles).
  let ww = max(w, vec3f(0.0));
  let u = normalize(ww.x * (na[0] + nb[0].xyz) + ww.y * (na[1] + nb[1].xyz) + ww.z * (na[2] + nb[2].xyz) + vec3f(1e-9));
  var score: array<f32, 6>;
  var t: array<f32, 6>;
  var best = 1e30;
  for (var i = 0u; i < 3u; i++) {
    let lw = log(max(w[i], 1e-6));
    // The distance along u from the point to each plane.
    t[2u * i] = -dot(pl - p[i], na[i]) / max(dot(na[i], u), 0.2);
    t[2u * i + 1u] = -dot(pl - p[i], nb[i].xyz) / max(dot(nb[i].xyz, u), 0.2);
    score[2u * i] = abs(t[2u * i]) * k - lw;
    score[2u * i + 1u] = select(1e30, abs(t[2u * i + 1u]) * k - lw, nb[i].w > 0.5);
    best = min(best, min(score[2u * i], score[2u * i + 1u]));
  }
  var acc = vec3f(0.0);
  var off = 0.0;
  var sum = 0.0;
  for (var i = 0u; i < 3u; i++) {
    let sa = exp(best - score[2u * i]);
    let sb = exp(best - score[2u * i + 1u]);
    acc += sa * na[i] + sb * nb[i].xyz;
    off += sa * t[2u * i] + sb * t[2u * i + 1u];
    sum += sa + sb;
  }
  return Env(normalize(acc + vec3f(1e-9)), u, off / sum);
}
// A node's planes from its skin triangles (rest: their rest positions):
// (normal, 1 when there are two) and the second's normal. Two when its most
// deviant triangle is more than SPLIT_COS from the mean: then the triangles
// nearer that one than the one most opposite it form the second.
fn nodePlanes(i: u32, rest: bool) -> array<vec4f, 2> {
  var acc = vec3f(0.0);
  for (var k = topo[i]; k < topo[i + 1u]; k++) {
    let t = params.trisAt + 4u * topo[params.listAt + k];
    acc += triCross(t, rest);
  }
  let m = normalize(acc + vec3f(1e-12));
  if (dot(acc, acc) < 1e-24) { return array<vec4f, 2>(vec4f(0.0), vec4f(0.0)); }
  var lo = 2.0;
  var sb = m;
  for (var k = topo[i]; k < topo[i + 1u]; k++) {
    let c = triCross(params.trisAt + 4u * topo[params.listAt + k], rest);
    let cl = length(c);
    if (cl < 1e-12) { continue; }
    let q = dot(c / cl, m);
    if (q < lo) { lo = q; sb = c / cl; }
  }
  if (lo > SPLIT_COS) { return array<vec4f, 2>(vec4f(m, 0.0), vec4f(m, 0.0)); }
  var hi = 2.0;
  var sa = m;
  for (var k = topo[i]; k < topo[i + 1u]; k++) {
    let c = triCross(params.trisAt + 4u * topo[params.listAt + k], rest);
    let cl = length(c);
    if (cl < 1e-12) { continue; }
    let q = dot(c / cl, sb);
    if (q < hi) { hi = q; sa = c / cl; }
  }
  var aa = vec3f(0.0);
  var ab = vec3f(0.0);
  for (var k = topo[i]; k < topo[i + 1u]; k++) {
    let c = triCross(params.trisAt + 4u * topo[params.listAt + k], rest);
    if (dot(c, sa) >= dot(c, sb)) { aa += c; } else { ab += c; }
  }
  return array<vec4f, 2>(vec4f(normalize(aa + vec3f(1e-12)), 1.0), vec4f(normalize(ab + vec3f(1e-12)), 1.0));
}
fn triCross(t: u32, rest: bool) -> vec3f {
  if (rest) {
    let a = restP(topo[t]);
    return cross(restP(topo[t + 1u]) - a, restP(topo[t + 2u]) - a);
  }
  let a = nodePos(topo[t]);
  return cross(nodePos(topo[t + 1u]) - a, nodePos(topo[t + 2u]) - a);
}
// v rotated by the shortest rotation taking unit a to unit b.
fn rotTo(a: vec3f, b: vec3f, v: vec3f) -> vec3f {
  let k = cross(a, b);
  let c = dot(a, b);
  if (c < -0.999) { return reflect(v, normalize(a + b + vec3f(1e-6, 2e-6, 3e-6))); }
  return v * c + cross(k, v) + k * (dot(k, v) / (1.0 + c));
}

// Distance from p to the static surfaces (the floor, boxes, cylinders) and
// that surface's normal there.
fn surface(p: vec3f) -> vec4f {
  var best = vec4f(0.0, 1.0, 0.0, p.y - fxu.floorY);
  for (var k = 0u; k < min(fxu.boxes, ${FX_BOXES}u); k++) {
    let c = fxu.box[2u * k];
    let h = fxu.box[2u * k + 1u].xyz;
    let d = p - c.xyz;
    if (c.w > 1.5 && c.w < 2.5) {
      // A vertical cylinder (radius h.x, half height h.y).
      let r = length(d.xz);
      let sd = max(r - h.x, abs(d.y) - h.y);
      if (sd < best.w) { best = vec4f(normalize(vec3f(d.x, 0.0, d.z) + vec3f(1e-6)), sd); }
    } else {
      let q = abs(d) - h;
      let sd = length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
      if (sd < best.w) {
        var nrm = vec3f(sign(d.x), 0.0, 0.0);
        if (q.y > q.x && q.y > q.z) { nrm = vec3f(0.0, sign(d.y), 0.0); }
        if (q.z > q.x && q.z > q.y) { nrm = vec3f(0.0, 0.0, sign(d.z)); }
        best = vec4f(nrm, sd);
      }
    }
  }
  return best;
}

// Per node (normals, tears, plastic strain, contact: sparks, dust, chips,
// scrapes), then per damage unit (its state from its beams).
@compute @workgroup_size(${WG})
fn nodeNormals(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.nodeCount) {
    let u = i - params.nodeCount;
    if (u >= params.unitCount) { return; }
    let rec = params.unitAt + 4u * u;
    var broken = false;
    var strain = 0.0;
    for (var k = topo[rec]; k < topo[rec + 1u]; k++) {
      let j = topo[k];
      if (beamBroken(j)) { broken = true; continue; }
      let r0 = bitcast<f32>(topo[params.rest0At + j]);
      strain = max(strain, abs(beamRest(j) - r0) / max(abs(r0), 1e-6));
    }
    broken = broken || strain > bitcast<f32>(topo[rec + 3u]);
    let old = nodeN[params.stateAt + u];
    var st = vec4f(select(0.0, 1.0, broken), strain, 0.0, old.w);
    // Shards are thrown once, on the first moving update after it broke.
    if (fxu.dt > 0.0) { st.z = select(0.0, 1.0, broken && old.w < 0.5); st.w = st.x; }
    nodeN[params.stateAt + u] = st;
    return;
  }
  var stretch = 0.0;
  for (var k = topo[i]; k < topo[i + 1u]; k++) {
    let t = params.trisAt + 4u * topo[params.listAt + k];
    let a = nodePos(topo[t]);
    let b = nodePos(topo[t + 1u]);
    let c = nodePos(topo[t + 2u]);
    // Tear: the triangle's perimeter against its rest perimeter (word 3).
    let rest = bitcast<f32>(topo[t + 3u]);
    if (rest > 0.0) { stretch = max(stretch, (length(b - a) + length(c - b) + length(a - c)) / rest); }
  }
  // w: how torn the node's skin is (0 whole, 1 torn), smooth across nodes so a
  // tear cuts the panel along a smooth line rather than the mesh's grid.
  // The node's planes now and at rest (nodePlanes: one, or two either side
  // of a fold through it).
  let pn = nodePlanes(i, false);
  let pr = nodePlanes(i, true);
  nodeN[NS * i] = vec4f(pn[0].xyz, smoothstep(1.35, 1.8, stretch));
  nodeN[NS * i + 3u] = pn[1];
  nodeN[NS * i + 4u] = pr[0];
  nodeN[NS * i + 5u] = pr[1];
  // Plastic strain: the node's beams' largest permanent length change, and
  // its broken beams.
  var plastic = 0.0;
  var nb = 0.0;
  let bl = params.nodeBeamAt + params.nodeCount + 1u;
  for (var k = topo[params.nodeBeamAt + i]; k < topo[params.nodeBeamAt + i + 1u]; k++) {
    let j = topo[bl + k];
    if (beamBroken(j)) { nb += 1.0; continue; }
    let r0 = bitcast<f32>(topo[params.rest0At + j]);
    plastic = max(plastic, abs(beamRest(j) - r0) / max(abs(r0), 1e-6));
  }
  let was = nodeN[NS * i + 2u];
  var scrape = was.z;
  let p = nodePos(i) + origin();
  let prev = nodeN[NS * i + 1u];
  if (fxu.dt > 0.0) {
    let mat = topo[params.nodeMatAt + i];
    if (prev.w > 0.5 && distance(p, prev.xyz) < 1.0) {
      let v = (p - prev.xyz) / fxu.dt;
      let sf = surface(p);
      let vn = dot(v, sf.xyz);
      let vt = length(v - sf.xyz * vn);
      let seed = i * 7919u + params.vehicle * 104729u;
      if (sf.w < CONTACT && mat != 3u) {
        // Sliding contact: metal throws sparks (about 4 a second per node
        // and m/s), a hard landing dust. The paint wears (µm, WEAR_*) only
        // where the solver's own contact rule holds: the node at or inside
        // the surface (it pushes a node out of the ground and boxes only
        // there), not merely near it.
        if (sf.w <= 0.0) { scrape += vt * fxu.dt * WEAR_UM_PER_M; }
        if (mat == 0u && vt > 2.0 && fxRnd(seed, 1u) < min(vt * 4.0 * fxu.dt, 0.5)) {
          let dir = normalize(v - sf.xyz * vn + vec3f(1e-5));
          fxSpawn(p + sf.xyz * 0.01, v * 0.7 + dir * 2.0 + fxRndDir(seed, 2u) * 1.8 + sf.xyz * 1.5, 0.2 + 0.4 * fxRnd(seed, 3u), 0u);
        }
        if (vn < -3.0 && fxRnd(seed, 5u) < 0.25) {
          fxSpawn(p, v * 0.2 + fxRndDir(seed, 6u) * 0.5 + vec3f(0.0, 0.4, 0.0), 1.2, 3u);
        }
      }
      // Hard impacts: where the panel yields suddenly, paint and plastic
      // flake off (chips in the panel's colour; black from plastic).
      let jump = plastic - was.x;
      if (jump > 0.02 && mat < 2u && fxRnd(seed, 8u) < min(jump * 2.0, 0.15)) {
        let col = select(params.paint, 0x0b0b0cu, mat == 1u && fxRnd(seed, 9u) < 0.5);
        fxSpawn(p, v * 0.7 + fxRndDir(seed, 10u) * 1.6 + vec3f(0.0, 0.8, 0.0), 4.0, 5u | (col << 8u));
      }
    }
    nodeN[NS * i + 1u] = vec4f(p, 1.0);
  }
  nodeN[NS * i + 2u] = vec4f(plastic, nb, scrape, 0.0);
}


// Pick the car's level: the finest whose median edge is at least LOD_PX
// pixels at the car's distance (its first node) from the last frame's eye
// (view.w: pixels per unit of tan, i.e. viewport height / 2 / tan(fov / 2)).
// Finer triangles than a pixel change nothing on screen but cost overshading.
const LOD_PX = ${LOD_PX};
@compute @workgroup_size(1)
fn pick() {
  let nl = lod.levels.x;
  var sel = 0u;
  if (nl > 1u && view.w > 0.0) {
    let d = max(distance(nodePos(0u) + origin(), view.xyz), 1e-3);
    sel = nl - 1u;
    for (var k = 0u; k < nl; k++) {
      if (lod.edge[k] * view.w / d >= LOD_PX) { sel = k; break; }
    }
  }
  let ls = params.stateAt + params.unitCount;
  // (picked, picked last update): a level just switched to has stale
  // vertices, so it spawns nothing this update.
  nodeN[ls] = vec4f(f32(sel), nodeN[ls].x, 0.0, 0.0);
  for (var k = 0u; k < nl; k++) {
    let on = select(0u, 1u, k == sel);
    let b = k * 23u;
    args[b] = on * ((lod.n[k] + ${WG - 1}u) / ${WG}u); args[b + 1u] = 1u; args[b + 2u] = 1u;
    args[b + 3u] = lod.opq[k]; args[b + 4u] = on; args[b + 5u] = 0u; args[b + 6u] = 0u; args[b + 7u] = 0u;
    args[b + 8u] = lod.gl[k]; args[b + 9u] = on; args[b + 10u] = lod.opq[k]; args[b + 11u] = 0u; args[b + 12u] = 0u;
    args[b + 13u] = lod.low[k]; args[b + 14u] = on; args[b + 15u] = 0u; args[b + 16u] = 0u; args[b + 17u] = 0u;
    args[b + 18u] = lod.mid[k]; args[b + 19u] = on; args[b + 20u] = 0u; args[b + 21u] = 0u; args[b + 22u] = 0u;
  }
}

// Raycast wheels (skin.rs wheel_vertex, the CPU mirror): a wheel piece's
// vertex (flags bit 23; bit 24 it spins) rides drive wheel n0, whose misc
// records (state, centre, axle, up: ExplicitSim.wheelRecords) are 4 words
// at wheelAt + 4·n0; (a, b, h) and (nt, nb, nn) its rest offset and normal
// in the wheel's frame (axle A, up U, forward F = A × U), turned about the
// axle by the spin angle.
fn wheelVertex(i: u32, bd: Bind) {
  let r = params.wheelAt + 4u * bd.n0;
  let st = bitcast<vec4f>(misc[topo[r]]);
  let centre = bitcast<vec4f>(misc[topo[r + 1u]]).xyz;
  let a = normalize(bitcast<vec4f>(misc[topo[r + 2u]]).xyz + vec3f(1e-9));
  let up = bitcast<vec4f>(misc[topo[r + 3u]]).xyz;
  let u = normalize(up - a * dot(a, up) + vec3f(1e-9));
  let f = cross(a, u);
  let spin = select(0.0, st.y, (bd.flags & (1u << 24u)) != 0u);
  let c = cos(spin);
  let s = sin(spin);
  let ur = u * c + f * s;
  let fr = f * c - u * s;
  let pos = centre + a * bd.a + ur * bd.b + fr * bd.h;
  let nrm = normalize(a * bd.nt + ur * bd.nb + fr * bd.nn + vec3f(1e-9));
  let k = i * 2u;
  outv[k] = vec4f(pos, bitcast<f32>(pack2x16float(vec2f(0.0, 0.0))));
  outv[k + 1u] = vec4f(nrm, bitcast<f32>(pack2x16float(vec2f(0.0, 0.0))));
}

// Weighted skins (engine_softbody::flexskin, a Rubble extension; the CPU twin
// is flexskin::eval). A weighted vertex's Bind (flags bit 25): n0 its entry
// count, vehicle its first entry slot in binds (4 entries a 64-byte slot,
// each (node, A, dA/dT, dA/dB)), (a, b, h) its detail offset, (nt, nb, nn) its
// rest normal's coefficients in the frame (U, V, W):
// P = sum A q, U = sum a q, V = sum b q, W = (U x V)/|U x V|;
// x = P + off.x U/|U| + off.y (W x U/|U|) + off.z W;
// n = normalize(nt (V x W) + nb (W x U) + nn (U x V)).
struct Weighted { pos: vec3f, nrm: vec3f, ids: vec3u };
fn wEntry(slot: u32, e: u32) -> vec4u {
  let b = binds[slot + e / 4u];
  let k = e % 4u;
  if (k == 0u) { return vec4u(b.n0, b.n1, b.n2, b.vehicle); }
  if (k == 1u) { return vec4u(bitcast<u32>(b.a), bitcast<u32>(b.b), bitcast<u32>(b.h), bitcast<u32>(b.r1)); }
  if (k == 2u) { return vec4u(bitcast<u32>(b.nt), bitcast<u32>(b.nb), bitcast<u32>(b.nn), bitcast<u32>(b.r2)); }
  return vec4u(bitcast<u32>(b.rx), bitcast<u32>(b.ry), bitcast<u32>(b.rz), b.flags);
}
fn weightedVertex(bd: Bind) -> Weighted {
  var p = vec3f(0.0); var u = vec3f(0.0); var v = vec3f(0.0);
  var ids = vec3u(0u);
  for (var e = 0u; e < bd.n0; e++) {
    let w = wEntry(bd.vehicle, e);
    let q = nodePos(w.x);
    p += q * bitcast<f32>(w.y);
    u += q * bitcast<f32>(w.z);
    v += q * bitcast<f32>(w.w);
    if (e < 3u) { ids[e] = w.x; }
  }
  if (bd.n0 < 2u) { ids.y = ids.x; }
  if (bd.n0 < 3u) { ids.z = ids.y; }
  let c = cross(u, v);
  let wn = c * (1.0 / max(length(c), 1e-20));
  let un = u * (1.0 / max(length(u), 1e-20));
  let vn = cross(wn, un);
  let pos = p + un * bd.a + vn * bd.b + wn * bd.h;
  let n = cross(v, wn) * bd.nt + cross(wn, u) * bd.nb + c * bd.nn;
  return Weighted(pos, n * (1.0 / max(length(n), 1e-20)), ids);
}

@compute @workgroup_size(${WG})
fn skin(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.count) { return; }
  var bd = binds[i];
  if ((bd.flags & (1u << 23u)) != 0u) { wheelVertex(i, bd); return; }
  let weighted = (bd.flags & (1u << 25u)) != 0u;
  var pos: vec3f;
  var n: vec3f;
  var w: vec3f;
  var tear: f32;
  var nrm: vec3f;
  if (weighted) {
    let r = weightedVertex(bd);
    pos = r.pos + origin();
    nrm = r.nrm;
    n = r.nrm;
    w = vec3f(1.0 / 3.0);
    tear = 0.0;
    bd.n0 = r.ids.x; bd.n1 = r.ids.y; bd.n2 = r.ids.z;
  } else {
  if (bd.vehicle != 0u) {
    // On a shut line's frame: its own part's frame for good once the frame
    // has died (a beam of its death record broke) or torn (skin.rs on_alt).
    let al = binds[bd.vehicle];
    let q0 = nodePos(bd.n0);
    var dead = length(nodePos(bd.n1) - q0) > 2.5 * bd.r1 || length(nodePos(bd.n2) - q0) > 2.5 * bd.r2;
    let d = params.deathAt + 4u * al.vehicle;
    for (var k = 0u; k < 4u; k++) {
      let j = topo[d + k];
      dead = dead || (j != 0xffffffffu && beamBroken(j));
    }
    if (dead) { bd = al; }
  }
  let org = origin();
  let p0 = nodePos(bd.n0) + org;
  let e1 = nodePos(bd.n1) - nodePos(bd.n0);
  let e2 = nodePos(bd.n2) - nodePos(bd.n0);
  let c = cross(e1, e2);
  let cl = length(c);
  n = select(vec3f(0.0, 1.0, 0.0), c / cl, cl > 1e-12);
  let l1 = length(e1);
  let l2 = length(e2);
  let t = e1 / max(l1, 1e-9);
  let b = cross(n, t);
  let s1 = l1 / max(bd.r1, 1e-6);
  let s2 = l2 / max(bd.r2, 1e-6);
  let torn = s1 > 2.5 || s2 > 2.5 || cl <= 1e-12;
  let rigid = (bd.flags & 2u) != 0u;
  let t0 = nodeN[NS * bd.n0].w; let t1 = nodeN[NS * bd.n1].w; let t2 = nodeN[NS * bd.n2].w;
  w = vec3f(1.0 - bd.a - bd.b, bd.a, bd.b);
  if (rigid) {
    // Rigid parts (mirrors, handles, lamps, tempered panes): the rest shape
    // in the triangle's orthonormal frame, never stretched; they go whole.
    pos = p0 + t * bd.a + b * bd.b + n * bd.h;
    tear = max(t0, max(t1, t2));
    w = vec3f(1.0 / 3.0);
  } else {
    pos = p0 + bd.a * e1 + bd.b * e2 + bd.h * n;
    // The tear field interpolated over the binding nodes: the fragment stage
    // cuts where it crosses 0.5, a smooth line through the panel.
    tear = w.x * t0 + w.y * t1 + w.z * t2;
  }
  tear = select(clamp(tear, 0.0, 1.0), 1.0, torn);
  // The rest normal carried by the triangle's frame (faceted: keeps kinks).
  let nf = normalize(bd.nt * t + bd.nb * b + bd.nn * n + vec3f(1e-9));
  nrm = nf;
  if ((bd.flags & 1u) != 0u && !rigid) {
    // The body's skin: the crease envelope (envelope()) now and at rest. The
    // vertex moves by the change of its offset (the rest pose and any rigid
    // motion are exact), and its rest normal turns with its planes' normals,
    // weighted as they are: smooth over a mild bend, switching sharply along
    // a fold's straight line. (No faceted fallback: the binding triangle's
    // frame normal drew its edges as a sawtooth along crushed lips.)
    let ids = array<u32, 3>(bd.n0, bd.n1, bd.n2);
    var pn: array<vec3f, 3>; var an: array<vec3f, 3>; var bn: array<vec4f, 3>;
    var pr: array<vec3f, 3>; var ar: array<vec3f, 3>; var br: array<vec4f, 3>;
    var ok = !torn;
    for (var c = 0u; c < 3u; c++) {
      let q = NS * ids[c];
      pn[c] = nodePos(ids[c]) + org; an[c] = nodeN[q].xyz; bn[c] = nodeN[q + 3u];
      pr[c] = restP(ids[c]); let r = nodeN[q + 4u]; ar[c] = r.xyz; br[c] = vec4f(nodeN[q + 5u].xyz, r.w);
      ok = ok && dot(an[c], an[c]) > 0.5 && dot(ar[c], ar[c]) > 0.5;
    }
    if (ok) {
      let l = 0.5 * (bd.r1 + bd.r2);
      let now = envelope(pn, an, bn, w, l);
      let rest = envelope(pr, ar, br, w, l);
      pos += clamp(now.e - rest.e, -ENVELOPE_MAX, ENVELOPE_MAX) * now.u;
      nrm = normalize(rotTo(rest.n, now.n, vec3f(bd.rx, bd.ry, bd.rz)));
    }
  }
  }
  // The binding nodes' plastic strain, broken beams and scrape.
  let q0 = nodeN[NS * bd.n0 + 2u]; let q1 = nodeN[NS * bd.n1 + 2u]; let q2 = nodeN[NS * bd.n2 + 2u];
  let plastic = w.x * q0.x + w.y * q1.x + w.z * q2.x;
  let scrape = w.x * q0.z + w.y * q1.z + w.z * q2.z;
  // Damage: −1 on a broken unit (its glass, lenses and mirror glass are
  // dropped, its lamps dark); on the laminated windscreen its cracks (0..1)
  // where its beams yielded past annealed glass's strain or broke.
  let unit = (bd.flags >> 2u) & 0x3fffu;
  let mat = (bd.flags >> 17u) & 63u;
  var dmg = 0.0;
  var due = false;
  if (unit != 0u) {
    let st = nodeN[params.stateAt + unit - 1u];
    if (topo[params.unitAt + 4u * (unit - 1u) + 2u] == LAMINATED) {
      let broke = select(0.0, 1.0, q0.y + q1.y + q2.y > 0.0);
      dmg = max(smoothstep(ANNEALED_CRACK, 20.0 * ANNEALED_CRACK, plastic), broke);
    } else {
      dmg = select(0.0, -1.0, st.x > 0.5);
      due = st.z > 0.5 && ((bd.flags >> 16u) & 1u) != 0u;
    }
  }
  let k = i * 2u;
  let was0 = outv[k];
  let was = vec4f(was0.xyz, unpack2x16float(bitcast<u32>(was0.w)).x);
  // Debris from this vertex: a shard as its pane breaks, a chip of paint
  // where the skin tears (not on a level just switched to: stale vertices).
  let ls = nodeN[params.stateAt + params.unitCount];
  if (fxu.dt > 0.0 && u32(ls.y) == params.level && u32(ls.x) == params.level) {
    let v = select(vec3f(0.0), (pos - was.xyz) / fxu.dt, distance(pos, was.xyz) < 1.0);
    let seed = i * 2654435761u + params.vehicle * 40503u;
    if (due) {
      var tint = 0u;
      if (mat == 27u) { tint = 1u; }
      if (mat == 20u) { tint = 2u; }
      if (mat == 17u || mat == 2u) { tint = 3u; }
      fxSpawn(pos, v + fxRndDir(seed, 1u) * (0.6 + 1.6 * fxRnd(seed, 2u)) + n * 0.8, 5.0 + 3.0 * fxRnd(seed, 3u), 4u | (tint << 8u));
    }
    if (was.w < 0.5 && tear >= 0.5 && (mat == 0u || mat == 3u) && fxRnd(seed, 5u) < 0.012) {
      let col = select(params.paint, 0x0b0b0cu, mat == 3u);
      fxSpawn(pos, v * 0.8 + fxRndDir(seed, 6u) * 1.2 + vec3f(0.0, 0.8, 0.0), 4.0, 5u | (col << 8u));
    }
  }
  outv[k] = vec4f(pos, bitcast<f32>(pack2x16float(vec2f(tear, dmg))));
  outv[k + 1u] = vec4f(nrm, bitcast<f32>(pack2x16float(vec2f(plastic, min(scrape, 60000.0)))));
}

// The particles' step (once per update, after every car's spawns).
@compute @workgroup_size(64)
fn fxStepAll(@builtin(global_invocation_id) gid: vec3u) { fxStep(gid.x, fxu.dt, fxu.floorY); }
`;

// The same skinning from world-space node positions (xyz f32 per node, every
// vehicle's nodes in order): the crash lab's gathered buffer, live or replayed
// (crash-lab-render.mjs positionsBuffer), and the beams' state from its
// compact rest lengths (negative when broken: restBuffer). Only the node and
// beam fetches differ; binding 1 (misc) is bound but unused (positions are
// already in world space).
const SKIN_POS_WGSL = (() => {
  let s = SKIN_WGSL;
  const swap = (a, b) => { if (!s.includes(a)) throw new Error('showcase skinPos: missing ' + a); s = s.replace(a, b); };
  swap('@group(0) @binding(0) var<storage, read> nodes: array<vec4f>;', '@group(0) @binding(0) var<storage, read> pos: array<f32>;');
  swap('fn nodePos(i: u32) -> vec3f { let k = (params.nodeOffset + i) * 3u; return nodes[k].xyz + nodes[k + 1u].xyz * fxu.lead.x; }',
    'fn nodePos(i: u32) -> vec3f { let k = (params.nodeOffset + i) * 3u; return vec3f(pos[k], pos[k + 1u], pos[k + 2u]); }');
  swap('@group(0) @binding(6) var<storage, read> beams: array<vec4f>;', '@group(0) @binding(6) var<storage, read> beams: array<f32>;');
  swap('fn beamBroken(j: u32) -> bool { return beams[(params.beamOffset + j) * 2u + 1u].z != 0.0; }', 'fn beamBroken(j: u32) -> bool { return beams[params.beamOffset + j] < 0.0; }');
  swap('fn beamRest(j: u32) -> f32 { return beams[(params.beamOffset + j) * 2u].x; }', 'fn beamRest(j: u32) -> f32 { return abs(beams[params.beamOffset + j]); }');
  s = s.replace(/fn origin\(\) -> vec3f \{[\s\S]*?\n\}\n/, 'fn origin() -> vec3f { return vec3f(0.0); }\n');
  return s;
})();

// The skinned vertex buffer: per vertex (position, tear | damage as two
// f16), (normal, plastic strain | scrape as two f16).
const OUT_STRIDE = 32;
const OUT_LAYOUT = { arrayStride: OUT_STRIDE, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 8, offset: 12, format: 'uint32' }, { shaderLocation: 1, offset: 16, format: 'float32x3' }, { shaderLocation: 7, offset: 28, format: 'uint32' }] };

// Debris particles: drawn blended after the glass.
const FX_DRAW_MODULE = /* wgsl */ `
struct Frame { viewProj: mat4x4f, eye: vec4f, light: vec4f };
@group(0) @binding(0) var<uniform> frame: Frame;
${FX_STRUCT_WGSL(FX_N)}
@group(0) @binding(1) var<storage, read> fx: FxParticlesRead;
fn tonemap(c: vec3f) -> vec3f {
  let x = c * 0.8;
  let m = clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
  return pow(m, vec3f(1.0 / 2.2));
}
${FX_DRAW_WGSL}
// A chip's colour: its panel's linear RGB (8 bits each) in kind's upper bits,
// lit as a flake turning in the studio.
fn fxChip(kind: u32, h: f32) -> vec3f {
  let c = vec3f(f32((kind >> 8u) & 255u), f32((kind >> 16u) & 255u), f32(kind >> 24u)) / 255.0;
  return tonemap(c * (0.25 + 0.6 * h));
}
@vertex fn vsFx(@builtin(vertex_index) vi: u32, @builtin(instance_index) inst: u32) -> FxOut { return fxVertex(vi, inst, frame.eye.xyz, frame.viewProj); }
@fragment fn fsFx(i: FxOut) -> @location(0) vec4f { return fxFragment(i); }
`;

const DRAW_WGSL = /* wgsl */ `
struct Frame { viewProj: mat4x4f, eye: vec4f, light: vec4f };
@group(0) @binding(0) var<uniform> frame: Frame;
// Imported vehicles' colour maps (setTextures): one array, a layer each.
@group(0) @binding(1) var texArr: texture_2d_array<f32>;
@group(0) @binding(2) var texSamp: sampler;
// Their data maps (setDataTextures, linear: normal, roughness, AO), and the
// texture sets (skin.rs tex_sets): per set its colour, normal, roughness
// and AO layers + 1 (0 none); a vertex's tex is its set + 1.
@group(0) @binding(3) var texData: texture_2d_array<f32>;
@group(0) @binding(4) var<storage, read> texSets: array<vec4u>;

struct VOut {
  @builtin(position) @invariant clip: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) color: vec3f,
  @location(3) @interpolate(flat) mat: u32,
  @location(4) torn: f32,
  @location(5) ao: f32,
  @location(6) strain: f32,
  @location(7) rest: vec3f,
  @location(8) uv: vec2f,
  // (plastic strain, paint worn through: µm).
  @location(9) wear: vec2f,
  // Texture set + 1 (0 none): an imported vehicle's maps.
  @location(10) @interpolate(flat) tex: u32,
};

@vertex
fn vs(@location(0) p3: vec3f, @location(8) tw: u32, @location(1) n3: vec3f, @location(7) ww: u32, @location(2) color: vec3f, @location(3) mat: u32, @location(4) ao: f32, @location(5) rest: vec3f, @location(6) uv: vec2f) -> VOut {
  let td = unpack2x16float(tw);
  let pt = vec4f(p3, td.x);
  let nrm = vec4f(n3, td.y);
  let wear = unpack2x16float(ww);
  var o: VOut;
  o.clip = frame.viewProj * vec4f(pt.xyz, 1.0);
  o.world = pt.xyz;
  o.normal = nrm.xyz;
  o.color = color;
  // The attribute's low byte is the material, its high half the texture.
  o.mat = mat & 0xffu;
  o.tex = mat >> 16u;
  o.torn = pt.w;
  o.ao = ao;
  o.strain = nrm.w;
  o.rest = rest;
  o.uv = uv;
  o.wear = wear;
  return o;
}

// A procedural showroom: a cyclorama with a bright horizon band fading to a
// dark ceiling, three long softbox strips overhead (crisp lines along the
// body), two low side strips (the flanks' horizon line), a key box in front
// and the sun. rough widens every edge (a cheap prefilter), so sharp
// clearcoat reflections show crisp strips and rough metal soft blobs.
fn softbox(d: vec3f, c: vec3f, r: vec3f, u: vec3f, ha: f32, hb: f32, w: f32) -> f32 {
  let z = dot(d, c);
  if (z <= 0.05) { return 0.0; }
  let x = dot(d, r) / z;
  let y = dot(d, u) / z;
  let mx = 1.0 - smoothstep(ha - w, ha + w, abs(x));
  let my = 1.0 - smoothstep(hb - w, hb + w, abs(y));
  return mx * my * (ha * hb) / ((ha + w) * (hb + w));
}
// The studio's palette (shared with the page's backdrop).
const HORIZON = vec3f(0.5, 0.52, 0.56);
const CEILING = vec3f(0.045, 0.05, 0.06);
const FLOOR_NEAR = vec3f(0.13, 0.135, 0.145);
// The studio's light frames (centre, right, up).
const SC0 = vec3f(-0.4819187, 0.8762159, 0.0000000);
const SR0 = vec3f(0.8762159, 0.4819187, 0.0000000);
const SC1 = vec3f(0.0000000, 1.0000000, 0.0000000);
const SR1 = vec3f(1.0000000, 0.0000000, 0.0000000);
const SC2 = vec3f(0.4819187, 0.8762159, 0.0000000);
const SR2 = vec3f(0.8762159, -0.4819187, 0.0000000);
const SA = vec3f(0.9428509, 0.3017123, 0.1414276);
const SA_R = vec3f(0.1483405, 0.0000000, -0.9889364);
const SA_U = vec3f(-0.2983743, 0.9533990, -0.0447561);
const SB = vec3f(-0.9356015, 0.2993925, -0.1871203);
const SB_R = vec3f(-0.1961161, 0.0000000, 0.9805807);
const SB_U = vec3f(0.2935785, 0.9541300, 0.0587157);
const KB = vec3f(0.2931977, 0.4607393, 0.8377078);
const KB_R = vec3f(0.9438584, 0.0000000, -0.3303504);
const KB_U = vec3f(-0.1522054, 0.8875355, -0.4348726);
// The shaded point (world): env() reflects the ground plane (y = 0) from it.
var<private> gP: vec3f;
fn sky(d: vec3f, rough: f32) -> vec3f {
  let w = 0.012 + rough * 1.1;
  let up = max(d.y, 0.0);
  var c = mix(HORIZON, CEILING, smoothstep(0.0, 0.75, up));
  c = c + vec3f(0.4) * exp(-up * (22.0 / (1.0 + rough * 6.0)));   // horizon glow
  // Three long strips overhead, along the car (frames are module constants:
  // the same lights, no per-pixel normalising).
  let strip = vec3f(3.6, 3.55, 3.4);
  c += strip * softbox(d, SC0, SR0, vec3f(0.0, 0.0, 1.0), 0.07, 1.4, w);
  c += strip * softbox(d, SC1, SR1, vec3f(0.0, 0.0, 1.0), 0.07, 1.4, w);
  c += strip * softbox(d, SC2, SR2, vec3f(0.0, 0.0, 1.0), 0.07, 1.4, w);
  // Side strips (long, horizontal, a little above the horizon).
  c += vec3f(2.8) * softbox(d, SA, SA_R, SA_U, 1.7, 0.05, w);
  c += vec3f(2.4) * softbox(d, SB, SB_R, SB_U, 1.7, 0.05, w);
  // Key box in front, high.
  c += vec3f(2.6, 2.5, 2.35) * softbox(d, KB, KB_R, KB_U, 0.30, 0.20, w);
  // Sun (the scene's light) as a small hot disc.
  let s = max(dot(d, normalize(frame.light.xyz)), 0.0);
  c += vec3f(1.0, 0.95, 0.85) * pow(s, 400.0 / (1.0 + rough * 300.0)) * (6.0 / (1.0 + rough * 30.0));
  return c;
}
fn env(d: vec3f, rough: f32) -> vec3f {
  let up = sky(d, rough);
  // Below the horizon: the floor, reflected from the shaded point. Short rays
  // land in the car's own shadow (under the sills, the bumpers, the arches);
  // far rays fade into the cyclorama's horizon.
  let dy = min(d.y, -1e-3);
  let t = max(gP.y, 0.0) / -dy;
  let occ = 1.0 - 0.9 * exp(-t * 1.6);
  let g = gP.xz + d.xz * t;
  let grid = abs(fract(g) - 0.5);
  let line = 1.0 - 0.08 * smoothstep(0.47, 0.5, max(grid.x, grid.y)) * (1.0 - rough);
  let far = smoothstep(4.0, 30.0, t);
  let floor = mix(FLOOR_NEAR * line * occ, HORIZON * 0.8, far);
  // A crisp horizon on a glossy panel, softer as it roughens.
  let hw = 0.006 + rough * 0.35;
  return mix(floor, up, smoothstep(-hw, hw * 0.5, d.y));
}

// A 5 × 7 pixel font for the printed parts (tyre sidewalls, plates): glyph
// rows packed 5 bits each (rows 0-3 in x, 4-6 in y; bit 4 the leftmost pixel).
const FONT = array<vec2u, 21>(vec2u(1001022u, 18004u), vec2u(575025u, 14897u), vec2u(1001022u, 31281u), vec2u(541200u, 32272u), vec2u(999967u, 32272u), vec2u(475663u, 30753u), vec2u(1001022u, 16912u), vec2u(575022u, 14897u), vec2u(135327u, 4228u), vec2u(708142u, 14905u), vec2u(135556u, 14468u), vec2u(67118u, 32004u), vec2u(459838u, 30753u), vec2u(600258u, 2143u), vec2u(64031u, 14881u), vec2u(999686u, 14897u), vec2u(133183u, 8456u), vec2u(476718u, 14897u), vec2u(509486u, 12353u), vec2u(133185u, 16648u), vec2u(0u, 0u));
const TYRE_TEXT = array<u32, 31>(0u, 1u, 2u, 2u, 3u, 4u, 20u, 20u, 5u, 6u, 7u, 0u, 8u, 20u, 20u, 20u, 11u, 13u, 14u, 19u, 12u, 14u, 20u, 0u, 11u, 9u, 20u, 20u, 20u, 20u, 20u);
const PLATE_TEXT = array<u32, 7>(0u, 2u, 3u, 20u, 13u, 16u, 9u);
// Coverage of glyph g at p (pixels: x 0..5 right, y 0..7 down), soft edged.
fn glyph(g: u32, p: vec2f) -> f32 {
  if (p.x < 0.0 || p.x >= 5.0 || p.y < 0.0 || p.y >= 7.0) { return 0.0; }
  let c = vec2u(floor(p));
  let w = FONT[g];
  let word = select(w.y, w.x, c.y < 4u);
  let row = select(c.y - 4u, c.y, c.y < 4u);
  let bit = (word >> (row * 5u + (4u - c.x))) & 1u;
  let f = fract(p) - 0.5;
  let edge = 1.0 - smoothstep(0.42, 0.56, max(abs(f.x), abs(f.y)));
  return f32(bit) * mix(1.0, edge, 0.6);
}
// Diffuse irradiance (a few env taps round the normal).
// Analytic (the studio blurred to a cosine lobe, fitted to env(n, 1)): the
// bright cyclorama, the sun, and the floor in the car's shadow below.
fn irradiance(n: vec3f) -> vec3f {
  let up = n.y;
  let skyc = mix(vec3f(1.2, 1.22, 1.26), vec3f(1.0, 1.0, 1.02), smoothstep(0.0, 0.9, up))
    + vec3f(0.19, 0.18, 0.16) * max(dot(n, normalize(frame.light.xyz)), 0.0);
  let t = max(gP.y, 0.0) / max(-up, 0.05);
  let floorc = FLOOR_NEAR * (1.0 - 0.9 * exp(-t * 1.6)) * 1.3;
  return mix(floorc, skyc, smoothstep(-0.6, 0.25, up)) * 0.55;
}

fn ggx(ndh: f32, a: f32) -> f32 {
  let a2 = a * a;
  let d = ndh * ndh * (a2 - 1.0) + 1.0;
  return a2 / (3.14159 * d * d);
}
fn hash3(p: vec3f) -> f32 {
  return fract(sin(dot(p, vec3f(127.1, 311.7, 74.7))) * 43758.5453);
}
fn noise3(p: vec3f) -> f32 {
  let i = floor(p); let f = fract(p); let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hash3(i), hash3(i + vec3f(1, 0, 0)), u.x), mix(hash3(i + vec3f(0, 1, 0)), hash3(i + vec3f(1, 1, 0)), u.x), u.y),
             mix(mix(hash3(i + vec3f(0, 0, 1)), hash3(i + vec3f(1, 0, 1)), u.x), mix(hash3(i + vec3f(0, 1, 1)), hash3(i + vec3f(1, 1, 1)), u.x), u.y), u.z);
}
// Whether material m on a broken (non-laminated) unit is gone: glass, lenses,
// a mirror's glass (skin.rs shatters()). i.strain < 0 marks a broken unit.
fn shattered(m: u32, d: f32) -> bool { return d < 0.0 && (m == 1u || m == 27u || m == 17u || m == 20u || m == 2u); }

// A normal map's tangent-space normal t (DirectX: green along +v of the
// image, rows top first) on n, in the cotangent frame of the surface's and
// the map's screen-space derivatives (no tangents stored; Schüler 2013's
// construction), its texels as stored (0..1).
fn normalMapped(n: vec3f, pdx: vec3f, pdy: vec3f, tdx: vec2f, tdy: vec2f, t: vec3f) -> vec3f {
  let q = t * 2.0 - 1.0;
  let dp2perp = cross(pdy, n);
  let dp1perp = cross(n, pdx);
  let tt = dp2perp * tdx.x + dp1perp * tdy.x;
  let bb = dp2perp * tdx.y + dp1perp * tdy.y;
  let m = max(dot(tt, tt), dot(bb, bb));
  if (m < 1e-20) { return n; }
  let s = inverseSqrt(m);
  return normalize(tt * s * q.x + bb * s * q.y + n * max(q.z, 1e-3));
}

// Per material: rough, metal (F0 scale), clearcoat.
fn props(m: u32) -> vec3f {
  switch m {
    case 0u: { return vec3f(0.38, 0.0, 1.0); }   // paint (base under clearcoat)
    case 2u: { return vec3f(0.04, 1.0, 0.0); }   // chrome
    case 3u: { return vec3f(0.16, 0.0, 0.0); }   // trim: gloss black plastic
    case 29u: { return vec3f(0.42, 0.0, 0.0); }  // pillar applique: satin black
    case 6u: { return vec3f(0.62, 0.0, 0.0); }   // tyre (satin sidewall)
    case 7u: { return vec3f(0.32, 1.0, 0.0); }   // rim (gunmetal)
    case 28u: { return vec3f(0.26, 1.0, 0.0); }  // machined spoke faces (satin bright)
    case 8u: { return vec3f(0.5, 0.0, 0.0); }    // a shut line's wall
    case 9u: { return vec3f(0.95, 0.0, 0.0); }   // underbody
    case 10u: { return vec3f(0.45, 0.0, 0.0); }  // plate
    case 11u: { return vec3f(0.55, 0.0, 0.0); }  // grille back
    case 19u: { return vec3f(0.12, 1.0, 0.0); }  // reflector
    case 22u: { return vec3f(0.45, 1.0, 0.0); }  // brake disc / barrel
    case 23u: { return vec3f(0.3, 0.0, 0.6); }   // caliper
    case 24u: { return vec3f(0.2, 0.0, 0.5); }   // tail lamp back
    case 30u: { return vec3f(0.55, 0.0, 0.25); } // inner stamping: e-coat, a little sheen
    case 31u: { return vec3f(0.5, 1.0, 0.0); }   // engine: cast alloy
    case 32u: { return vec3f(0.62, 0.0, 0.0); }  // grained black plastic (satin)
    default: { return vec3f(0.8, 0.0, 0.0); }
  }
}

fn tonemap(c: vec3f) -> vec3f {
  let x = c * 0.8;
  let m = clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
  return pow(m, vec3f(1.0 / 2.2));
}

// Glass cracks: a Voronoi web in rest space (cells ~6 cm) plus rays from the
// nearest impact-like centre; 1 on a crack line, 0 off it.
fn hash2(p: vec2f) -> vec2f {
  return fract(sin(vec2f(dot(p, vec2f(127.1, 311.7)), dot(p, vec2f(269.5, 183.3)))) * 43758.5453);
}
fn cracks(q: vec2f) -> f32 {
  let g = floor(q);
  let f = fract(q);
  var d1 = 8.0; var d2 = 8.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let o = vec2f(f32(x), f32(y));
      let r = o + hash2(g + o) - f;
      let d = dot(r, r);
      if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) { d2 = d; }
    }
  }
  let edge = sqrt(d2) - sqrt(d1);
  return 1.0 - smoothstep(0.015, 0.05, edge);
}

@fragment
fn fs(i: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  return shade(i, front);
}

// The car mirrored in the studio floor (drawn with viewProj · diag(1, −1, 1)),
// blended over the floor, fading with height above it (a polished floor's
// reflection, not a mirror's). frame.light.w is the floor's reflectance at the
// contact.
@fragment
fn fsDepth() -> @location(0) vec4f { return vec4f(0.0); }

// Contact shadows: whatever is within 7 cm of the floor (the tyres' contact
// patches, a scraping sill) flattened onto it and darkened by its height, so
// the car sits on the floor wherever it is and however it has turned.
// frame.eye.w: the floor's height; frame.light.w: the opacity per layer.
struct COut { @builtin(position) clip: vec4f, @location(0) a: f32 };
@vertex
fn vsContact(@location(0) p3: vec3f, @location(8) tw: u32, @location(1) n3: vec3f, @location(2) color: vec3f, @location(3) mat: u32) -> COut {
  let pt = vec4f(p3, unpack2x16float(tw).x);
  let nrm = vec4f(n3, 0.0);
  var o: COut;
  let hgt = pt.y - frame.eye.w;
  // Spread along the surface's horizontal normal (an occluder darkens the
  // floor it faces), wider the higher it is: the shadow shows past the
  // tyre's own silhouette, a tight core under a soft skirt.
  let nh = vec2f(nrm.x, nrm.z);
  let sp = nh / max(length(nh), 1e-3) * min(length(nh), 1.0) * (0.02 + 0.2 * max(hgt, 0.0));
  o.clip = frame.viewProj * vec4f(pt.x + sp.x, frame.eye.w + 0.002, pt.z + sp.y, 1.0);
  // Glass, lenses and torn skin cast none.
  let m = mat & 0xffu;
  let skip = pt.w > 0.5 || m == 1u || m == 17u || m == 20u || m == 25u || m == 27u;
  let fall = 1.0 - smoothstep(0.0, 0.15, hgt);
  o.a = select(frame.light.w * fall, 0.0, skip || hgt < -0.05);
  // Far above the floor: collapsed to one point outside the view, so the
  // flattened car costs no fragments (only what is near the floor draws).
  if (hgt > 0.25) { o.clip = vec4f(2.0, 2.0, 2.0, 1.0); }
  return o;
}
@fragment
fn fsContact(i: COut) -> @location(0) vec4f {
  if (i.a < 0.01) { discard; }
  return vec4f(0.0, 0.0, 0.0, i.a);
}

@fragment
fn fsMirror(i: VOut) -> @location(0) vec4f {
  // Cheap: where the reflection has faded, nothing; elsewhere the base colour
  // under the studio's diffuse light, the paint with a hint of its sheen (a
  // blurred floor reflection shows no more), tears and shattered glass cut.
  // frame.eye.w: the floor's height (the mirror plane).
  let a = frame.light.w * exp(-max(i.world.y - frame.eye.w, 0.0) * 2.2);
  if (a < 0.015 || i.torn > 0.5) { discard; }
  if (shattered(i.mat, i.strain)) { discard; }
  gP = i.world - vec3f(0.0, frame.eye.w, 0.0);
  let n = normalize(i.normal);
  var c = i.color * irradiance(n) * (0.35 + 0.65 * clamp(i.ao, 0.0, 1.0));
  if (i.mat == 18u || i.mat == 21u) { c = i.color * 3.0; }
  if (i.mat == 0u) { c += vec3f(0.06) * smoothstep(-0.2, 0.6, n.y); }
  return vec4f(tonemap(c), a);
}

// Zebra (highlight-quality check, \`?zebra=1|2|3\`): paint shows the reflection
// of evenly spaced light stripes (1 parallel to the car's length, 2 across
// it, 3 level bands), so a smooth surface shows smooth, continuous lines and
// a wobble or kink in the surface shows as one in the lines.
const ZEBRA: f32 = 0.0;
fn zebraCoord(n: vec3f, w: vec3f) -> f32 {
  let r = reflect(normalize(w - frame.eye.xyz), n);
  if (ZEBRA < 1.5) { return atan2(r.y, r.x) * 9.0 / 3.14159265; }
  if (ZEBRA < 2.5) { return atan2(r.y, r.z) * 9.0 / 3.14159265; }
  return asin(clamp(r.y, -1.0, 1.0)) * 18.0 / 3.14159265;
}
fn shade(i: VOut, front: bool) -> vec4f {
  // Texture-coordinate footprint (taken in uniform control flow): printed
  // text fades to its average ink when too small to read.
  let uvw = fwidth(i.uv);
  // The colour map's sampling footprint (v flipped: images are stored top
  // row first, Collada's t points up).
  let tdx = dpdx(i.uv) * vec2f(1.0, -1.0);
  let tdy = dpdy(i.uv) * vec2f(1.0, -1.0);
  // The surface's footprint (a normal map's tangent frame).
  let pdx = dpdx(i.world);
  let pdy = dpdy(i.world);
  var zc = 0.0;
  var zw = 1.0;
  if (ZEBRA > 0.0) {
    zc = zebraCoord(normalize(i.normal), i.world);
    zw = fwidth(zc);
  }
  // Tears: the far half of a torn triangle is cut away (through the triangle,
  // not along the grid); a band at the cut shows the bare metal edge.
  if (i.torn > 0.5) { discard; }
  gP = i.world;
  let glassy = i.mat == 1u || i.mat == 25u || i.mat == 27u;
  // Tempered glass, lenses and mirror glass are gone once their physics unit
  // has broken (skin.rs damage units).
  if (shattered(i.mat, i.strain)) { discard; }
  // A thin lip (the field's last 6 %: a few millimetres), not a dark band.
  if (i.torn > 0.47 && !glassy && i.mat != 6u && i.wear.x > 0.02) {
    // Torn sheet steel: a bright, rough-polished lip (paint and e-coat
    // ripped off) catching the studio from every side.
    var nn = normalize(i.normal);
    let vv = normalize(frame.eye.xyz - i.world);
    if (dot(nn, vv) < 0.0) { nn = -nn; }
    let lit = 0.45 + 0.55 * max(dot(nn, normalize(frame.light.xyz)), 0.0);
    let steel = vec3f(0.62, 0.62, 0.64);
    // (The analytic diffuse lobe, not env(): a second inlined environment
    // lookup costs the whole shader occupancy.)
    let rr = reflect(-vv, nn);
    let sky = mix(0.3, 1.3, smoothstep(-0.3, 0.6, rr.y));
    return vec4f(tonemap(steel * (sky + 0.35 * lit)), 1.0);
  }
  var n = normalize(i.normal);
  let v = normalize(frame.eye.xyz - i.world);
  // The inside of a body panel (seen through a window or a tear): bare, dark.
  // A back face whose skinned normal still faces the eye is not an inside but
  // a sliver folded over in a crush (a rolled edge, a crease): paint, so a
  // buckled seam reads as a sharp fold rather than a dark sawtooth.
  if (!front && i.mat == 0u && dot(n, v) < 0.0) {
    let d = vec3f(0.055, 0.056, 0.06) * (0.4 + 0.6 * clamp(i.ao, 0.0, 1.0));
    return vec4f(tonemap(d * (irradiance(-n) + vec3f(0.2))), 1.0);
  }
  // Two-sided: light the side facing the camera.
  if (dot(n, v) < 0.0) { n = -n; }
  let l = normalize(frame.light.xyz);
  var ao = clamp(i.ao, 0.0, 1.0);
  // An imported material's data maps: its normal map tilts n, its roughness
  // map replaces the material's, its AO map darkens.
  var roughMap = -1.0;
  if (i.tex > 0u) {
    let ts = texSets[i.tex - 1u];
    let st = vec2f(i.uv.x, 1.0 - i.uv.y);
    if (ts.y > 0u) { n = normalMapped(n, pdx, pdy, tdx, tdy, textureSampleGrad(texData, texSamp, st, ts.y - 1u, tdx, tdy).xyz); }
    if (ts.z > 0u) { roughMap = textureSampleGrad(texData, texSamp, st, ts.z - 1u, tdx, tdy).r; }
    if (ts.w > 0u) { ao *= textureSampleGrad(texData, texSamp, st, ts.w - 1u, tdx, tdy).r; }
  }
  // Wheels (they turn, so nothing is baked): the arch shades their tops and
  // the car's body their inner faces.
  if (i.mat == 6u || i.mat == 7u || i.mat == 22u || i.mat == 23u || i.mat == 28u) {
    ao *= mix(1.0, 0.42, smoothstep(0.30, 0.66, i.world.y));
  }
  // Grained plastic: a fine stipple (about 1 mm, in rest space) tilting the
  // normal, faded to the satin average where a grain is under a pixel.
  if (i.mat == 32u) {
    let q = i.rest * 1100.0;
    let gr = vec3f(noise3(q), noise3(q + vec3f(17.3)), noise3(q + vec3f(41.7))) - vec3f(0.5);
    let px = distance(frame.eye.xyz, i.world) * 1.2;
    n = normalize(n + gr * 0.5 * (1.0 - smoothstep(0.4, 1.6, px)));
  }
  let ndl = max(dot(n, l), 0.0);
  let ndv = max(dot(n, v), 1e-3);
  let h = normalize(v + l);
  let ndh = max(dot(n, h), 0.0);
  let r = reflect(-v, n);
  let sun = vec3f(1.0, 0.96, 0.9) * 1.6;
  let fres = pow(1.0 - ndv, 5.0);
  // Wear (P11): where the panel yielded the clearcoat crazes and dulls in
  // proportion to its plastic strain; where it slid along the ground or a
  // wall it is scratched through to primer, then bare steel, in streaks
  // along the surface, more with every metre slid. Worn spots shade through
  // the generic paths (bare steel as metal, primer as matt), adding no
  // shading code of their own (the paint path is occupancy bound).
  let plastic = i.wear.x;
  let scr = i.wear.y;
  var m = i.mat;
  var albedo = i.color;
  if (i.tex > 0u) {
    let tc = texSets[i.tex - 1u].x;
    if (tc > 0u) { albedo *= textureSampleGrad(texArr, texSamp, vec2f(i.uv.x, 1.0 - i.uv.y), tc - 1u, tdx, tdy).rgb; }
  }
  // Worn depth (µm, WEAR_*): through clear and base to primer past 65 µm,
  // through primer and e-coat to bare steel past 110 µm; in streaks (the
  // contact is a few asperities wide), wider as it deepens.
  if ((m == 0u || m == 3u || m == 9u) && scr > 65.0) {
    let q = i.rest * vec3f(60.0, 60.0, 4.0);
    let streak = noise3(q) * 0.65 + noise3(q * 3.1) * 0.35;
    let amount = smoothstep(65.0, 400.0, scr);
    if (streak > 0.815 - 0.3 * amount && scr > 110.0) { m = 7u; albedo = vec3f(0.56, 0.56, 0.58); }
    else if (m == 0u && streak > 0.74 - 0.35 * amount) { m = 254u; albedo = vec3f(0.30, 0.30, 0.29); }
  }
  // Round a tear, the paint has cracked off the stretched steel (grey
  // primer) inside the bare lip: the tear reads as pressed steel, not paper.
  if (m == 0u && i.torn > 0.44 && plastic > 0.04) { m = 254u; albedo = vec3f(0.16, 0.16, 0.155); }

  // Emissive lamps (dark once their lamp has broken: its lens gone, the
  // reflector and light guides behind it unlit).
  let dead = i.strain < 0.0;
  if (m == 18u && !dead) { return vec4f(tonemap(albedo * 6.0 + env(r, 0.05) * 0.04), 1.0); }
  if (m == 21u && !dead) { return vec4f(tonemap(albedo * vec3f(2.6, 0.5, 0.5)), 1.0); }
  if ((m == 18u || m == 21u) && dead) {
    let f = 0.04 + 0.96 * fres;
    let c0 = select(vec3f(0.05), vec3f(0.08, 0.01, 0.01), m == 21u);
    return vec4f(tonemap(c0 * irradiance(n) * ao + irradiance(r) * f * 0.5), 1.0);
  }
  if (m == 4u) {
    // Projector lens: dark polished glass with a cool inner glint.
    let f = 0.04 + 0.96 * fres;
    let glint = pow(max(dot(n, v), 0.0), 24.0) * vec3f(0.25, 0.3, 0.4);
    return vec4f(tonemap(vec3f(0.01, 0.012, 0.016) + env(r, 0.02) * (0.08 + f) + glint), 1.0);
  }
  // Transparent: glass, clear lens, red lens (blended pass).
  if (m == 1u || m == 17u || m == 20u || m == 25u || m == 27u) {
    let f = 0.04 + 0.96 * fres;
    if (m == 25u && i.strain > 0.02) {
      // Laminated windscreen: spider cracks where its beams yielded past
      // annealed glass's break strain (or broke): i.strain the crack amount.
      let amt = i.strain;
      let q = i.rest.xy * 18.0 + i.rest.z * 7.0;
      let c = max(cracks(q), cracks(q * 2.7) * step(0.5, amt)) * amt;
      if (c > 0.02) {
        return vec4f(tonemap(vec3f(0.55, 0.57, 0.6) * (0.4 + env(r, 0.3) * 0.25)), clamp(0.5 + c * 0.5, 0.0, 1.0));
      }
    }
    let refl = env(r, 0.0) * (0.05 + 0.95 * fres) * select(1.0, 0.16, m == 20u);
    var tint = vec3f(0.008, 0.01, 0.012);
    var a = 0.6;
    if (m == 17u) { tint = vec3f(0.0); a = 0.08; }
    // Privacy glass (rear doors, quarter lights, backlight): dark tint.
    if (m == 27u) { tint = vec3f(0.002, 0.0025, 0.003); a = 0.88; }
    // Ruby lens: deep red, thin enough for the LEDs behind it to glow through.
    if (m == 20u) { tint = vec3f(0.11, 0.0, 0.001); a = 0.6; }
    let alpha = clamp(a + f * (1.0 - a), 0.0, 1.0);
    return vec4f(tonemap((tint * a + refl) / alpha), alpha);
  }
  if (m == 26u) {
    // A lit display: the instrument cluster (u 0..1: two dials and a centre
    // readout) or the centre screen (u 1..2: a map with a route); under glass.
    let f = 0.04 + 0.96 * fres;
    var col = vec3f(0.004, 0.006, 0.012);
    if (i.uv.x < 1.0) {
      let q = vec2f(i.uv.x * 3.1, i.uv.y);
      for (var k = 0; k < 2; k++) {
        let c0 = vec2f(select(0.62, 2.48, k == 1), 0.5);
        let d = q - c0;
        let r = length(d);
        let ang = atan2(d.y, d.x);
        let ring = (1.0 - smoothstep(0.012, 0.03, abs(r - 0.4))) * select(1.0, 0.0, abs(ang + 1.5708) < 0.8);
        let ticks = (1.0 - smoothstep(0.25, 0.45, abs(fract(ang * 4.0 / 3.1416 + 0.5) - 0.5) * 6.0)) * step(0.33, r) * step(r, 0.39) * select(1.0, 0.0, abs(ang + 1.5708) < 0.8);
        let needle_a = select(0.6, 1.4, k == 1);
        let nd = vec2f(cos(needle_a), sin(needle_a));
        let along = dot(d, nd);
        let needle = (1.0 - smoothstep(0.006, 0.016, abs(d.x * nd.y - d.y * nd.x))) * step(0.0, along) * step(along, 0.36);
        col += vec3f(0.35, 0.6, 1.0) * ring * 0.9 + vec3f(0.8, 0.85, 0.9) * ticks * 0.7 + vec3f(1.0, 0.35, 0.08) * needle * 1.6;
      }
      // Centre readout: a few bars.
      let cq = q - vec2f(1.55, 0.5);
      let bars = step(abs(cq.x), 0.35) * (1.0 - smoothstep(0.015, 0.03, abs(fract(cq.y * 5.0) - 0.5) * 0.2)) * step(abs(cq.y), 0.3);
      col += vec3f(0.5, 0.75, 1.0) * bars * 0.25;
    } else {
      let q = vec2f((i.uv.x - 1.0) * 1.75, i.uv.y);
      let grid = abs(fract(q * vec2f(9.0, 5.0)) - 0.5);
      col = vec3f(0.035, 0.04, 0.05) + vec3f(0.05) * (1.0 - smoothstep(0.44, 0.49, max(grid.x, grid.y)));
      let route = 1.0 - smoothstep(0.012, 0.025, abs(q.y - 0.45 - 0.18 * sin(q.x * 3.5)));
      col = mix(col, vec3f(0.2, 0.55, 1.0), route);
      let bar = step(0.86, q.y);
      col = mix(col, vec3f(0.08, 0.09, 0.11), bar);
    }
    return vec4f(tonemap(col * 2.2 + env(r, 0.02) * f * 0.5), 1.0);
  }
  if (m >= 12u && m <= 16u) {
    // Cabin trim (seats, dash, door cards, headliner, carpet): under the roof,
    // lit by what comes through the glass; a soft sheen, no studio reflection.
    let sheen = pow(1.0 - ndv, 3.0) * 0.08 + sun.x * ggx(ndh, select(0.25, roughMap, roughMap >= 0.0)) * ndl * 0.02;
    let cab = albedo * (irradiance(n) * 0.55 * ao + sun * ndl / 3.14159 * 0.35 * ao) + vec3f(sheen) * ao;
    return vec4f(tonemap(cab), 1.0);
  }
  if (ZEBRA > 0.0 && m == 0u) {
    let q = abs(fract(zc) - 0.5) * 2.0;
    let e = clamp(zw, 1e-4, 0.5);
    let band = smoothstep(0.5 - e, 0.5 + e, q);
    return vec4f(vec3f(mix(0.04, 0.92, band) * mix(0.6, 1.0, ao)), 1.0);
  }
  var pr = props(m);
  if (roughMap >= 0.0) { pr.x = roughMap; }
  var albedo2 = albedo;
  var ao2 = ao;
  if (m == 10u) {
    // Plate: white with a dark border and its registration.
    let uv = i.uv;
    let border = step(0.035, uv.x) * step(uv.x, 0.965) * step(0.07, uv.y) * step(uv.y, 0.93);
    let inner = step(0.05, uv.x) * step(uv.x, 0.95) * step(0.1, uv.y) * step(uv.y, 0.9);
    var ink = border * (1.0 - inner);
    let nch = f32(7);
    let tw = 0.84;
    let tx = (uv.x - (1.0 - tw) * 0.5) / tw * nch;
    if (tx >= 0.0 && tx < nch) {
      let ci = u32(floor(tx));
      let px = vec2f(fract(tx) * 6.0 - 0.5, (0.86 - uv.y) / 0.72 * 7.0);
      ink = max(ink, glyph(PLATE_TEXT[ci], px));
    }
    // Too small to read: its average ink instead of aliased pixels.
    ink = mix(ink, 0.3, smoothstep(0.08, 0.3, length(uvw) * 7.0));
    albedo2 = mix(albedo, vec3f(0.015, 0.02, 0.04), ink);
  }
  if (m == 6u) {
    // Tyre: raised lettering on the outer sidewall, tread blocks on the
    // shoulders (sipes between them).
    let uv = i.uv;
    if (uv.y > 0.48 && uv.y < 0.84) {
      let nch = f32(62);
      let tx = fract(uv.x) * nch;
      let ci = u32(floor(tx)) % 31u;
      let px = vec2f(fract(tx) * 6.0 - 0.5, (0.81 - uv.y) / 0.30 * 7.0);
      let ink = mix(glyph(TYRE_TEXT[ci], px), 0.25, smoothstep(0.3, 1.0, uvw.x * nch));
      albedo2 = mix(albedo, vec3f(0.07), ink);
      pr = vec3f(mix(0.72, 0.45, ink), 0.0, 0.0);
    }
    if (uv.y >= 2.0) {
      let w = uv.y - 2.0;
      let shoulder = step(w, 0.38) + step(1.62, w);
      let blocks = abs(fract(uv.x * 72.0 + select(0.0, 0.5, w > 1.0)) - 0.5);
      let groove = (1.0 - smoothstep(0.06, 0.1, blocks)) * shoulder;
      let sipe = (1.0 - smoothstep(0.02, 0.05, abs(fract(uv.x * 144.0) - 0.5))) * (1.0 - shoulder) * 0.6;
      ao2 = ao * (1.0 - 0.75 * max(groove, sipe));
    }
  }
  if (m == 11u) {
    // Grille mesh: a honeycomb in rest space (2 cm cells): gloss-black rims
    // round dark holes.
    let q = vec2f(i.rest.x, i.rest.y) * 50.0;
    let hx = vec2f(1.0, 1.7320508);
    let a1 = q - hx * floor(q / hx + 0.5);
    let b1 = q - hx * floor((q - hx * 0.5) / hx + 0.5) - hx * 0.5;
    let d = min(dot(a1, a1), dot(b1, b1));
    let hole = 1.0 - smoothstep(0.12, 0.2, d);
    ao2 = ao * mix(1.0, 0.15, hole);
    pr = vec3f(mix(0.18, 0.9, hole), 0.0, 0.0);
    albedo2 = albedo * mix(1.0, 0.3, hole);
  }
  let rough = pr.x; let metal = pr.y; let coat = pr.z;
  let a = max(rough * rough, 0.0015);
  let k = a * 0.5;
  let g = ndl / (ndl * (1.0 - k) + k) * ndv / (ndv * (1.0 - k) + k);
  var c: vec3f;
  if (m == 0u) {
    // Car paint: a tinted metallic-flake base under a clearcoat.
    let gloss = 1.0 - smoothstep(0.004, 0.06, plastic);
    let cf = (0.04 + 0.96 * fres) * mix(0.35, 1.0, gloss);
    // Flop: the colour deepens toward grazing (the flakes face away).
    let flop = mix(1.0, 0.45, pow(1.0 - ndv, 1.5));
    let diffuse = albedo * flop * (irradiance(n) * ao + sun * ndl / 3.14159);
    let flakeF0 = albedo * 0.55 + vec3f(0.03);
    // The flakes' rough reflection: the blurred studio (the analytic lobe, not
    // a second environment lookup).
    let flake = flakeF0 * (irradiance(r) * 1.7 * ao * 0.9 + sun * ggx(ndh, 0.17) * g / max(4.0 * ndv, 1e-3));
    // Sparkle: flakes fixed in the paint (rest space, so they never swim as
    // the body moves or deforms), glinting near the sun's highlight; faded
    // to nothing where a flake is under a pixel (no shimmer at a distance).
    let fpx = distance(frame.eye.xyz, i.world) * 6.3e-1;
    let hx = hash3(floor(i.rest * 900.0));
    let sp = step(0.988, hx) * pow(ndh, 60.0) * ndl * 1.6 * (1.0 - smoothstep(0.6, 2.0, fpx));
    let base = diffuse * 0.75 + flake + (albedo * 0.6 + vec3f(0.4)) * sp;
    // Clearcoat: a mirror-sharp sun glint over a broader sheen, and the
    // studio reflected crisply (horizon, softboxes, the floor).
    let coatSpec = sun * (ggx(ndh, 0.0016) * 0.22 * gloss + ggx(ndh, 0.02) * 0.06) * cf / max(4.0 * ndv, 1e-3) * ndl;
    // The coat's sky term eased at grazing (cf rises to 1 there; a real
    // coat with a rough base reads milky long before that).
    let cfe = cf * mix(1.0, 0.8, smoothstep(0.5, 0.95, 1.0 - ndv));
    // A crazed coat reflects the studio blurred (the analytic lobe: a
    // roughness-dependent env() would not fold to the sharp coat's constants).
    var refl = env(r, 0.003);
    if (gloss < 0.999) { refl = mix(irradiance(r) * 1.6, refl, gloss); }
    c = base * (1.0 - cfe) + refl * cfe * mix(0.45, 1.0, ao) + coatSpec;
  } else if (metal > 0.5) {
    let f0 = albedo;
    let f = f0 + (vec3f(1.0) - f0) * fres;
    c = f * (env(r, rough) * mix(0.35, 1.0, ao) + sun * ggx(ndh, a) * g / max(4.0 * ndv, 1e-3) * ndl);
  } else {
    let f = 0.04 + 0.96 * fres;
    let spec = (env(r, rough) * f * mix(0.3, 1.0, ao2) + sun * ggx(ndh, a) * g * f / max(4.0 * ndv, 1e-3) * ndl) * min(1.0, ao2 * 3.0) * select(1.0, ao2 * 0.5, m == 9u);
    c = albedo2 * (irradiance(n) * ao2 + sun * ndl / 3.14159 * mix(0.5, 1.0, ao2)) * (1.0 - f) + spec;
    if (coat > 0.0) {
      let cf = 0.04 + 0.96 * fres;
      c = c * (1.0 - cf * coat) + env(r, 0.03) * cf * coat * ao2;
    }
  }
  return vec4f(tonemap(c), 1.0);
}
`;

/**
 * A renderer for showcase cars on `device`. See the file header for the interface.
 * `opts`: `{ format, depthFormat = 'depth24plus', sampleCount = 1, cullMode = 'none', zebra = 0 }`
 * (`zebra` 1–3 draws the paint as reflected stripes: the highlight check).
 */
export function createShowcaseCarRenderer(device, opts = {}) {
  const { format, depthFormat = 'depth24plus', sampleCount = 1, cullMode = 'none', zebra = 0 } = opts;
  // ?crease= / ?splitDeg= (showcase-car.html): other crease constants, for comparison.
  const crease = (code) => code.replace(`const LOD_PX = ${LOD_PX};`, `const LOD_PX = ${opts.lodPx || LOD_PX};`).replace(`const CREASE_SHARP = ${CREASE_SHARP};`, `const CREASE_SHARP = ${opts.creaseSharp || CREASE_SHARP};`)
    .replace(`const SPLIT_COS = ${SPLIT_COS};`, `const SPLIT_COS = ${opts.splitDeg ? Math.cos(opts.splitDeg * Math.PI / 180) : SPLIT_COS};`);
  const skinModule = device.createShaderModule({ label: 'showcase-skin', code: crease(SKIN_WGSL) });
  let drawCode = zebra ? DRAW_WGSL.replace('const ZEBRA: f32 = 0.0;', `const ZEBRA: f32 = ${Number(zebra).toFixed(1)};`) : DRAW_WGSL;
  // Profiling (showcase-car.html ?perfFlat=1, ?perfSkip=opaque,glass,reflect,contact):
  // a flat fragment shader (vertex and raster cost alone), or parts left out.
  if (opts.perfFlat) drawCode = drawCode.replace('  return shade(i, front);\n}', '  return vec4f(i.color, 1.0);\n}');
  // ?debugWear=1: red the tear field, green plastic strain (×10), blue metres slid (×10).
  if (opts.debugWear) drawCode = drawCode.replace('  return shade(i, front);\n}', '  return vec4f(i.torn, min(i.wear.x * 10.0, 1.0), min(i.wear.y * 10.0, 1.0), 1.0);\n}');
  // ?perfNoDiscard=1: the fragment discards (tears, shattered glass) left out (measures HSR).
  if (opts.perfNoDiscard) drawCode = drawCode.replace('if (i.torn > 0.5) { discard; }', '').replace('if (shattered(i.mat, i.strain)) { discard; }', '');
  const skip = new Set(opts.perfSkip ?? []);
  const drawModule = device.createShaderModule({ label: 'showcase-draw', code: drawCode });
  // 0 nodes, 1 misc, 2 binds, 3 out (rw), 4 params, 5 topo, 6 beams, 7 node
  // state (rw), 8 effects, 9 particles (rw): 8 storage buffers, the default limit.
  const skinLayout = device.createBindGroupLayout({
    entries: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((binding) => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: binding === 4 || binding === 8 ? 'uniform' : binding === 3 || binding === 7 || binding === 9 ? 'storage' : 'read-only-storage' },
    })),
  });
  const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: [skinLayout] });
  const skinPipe = device.createComputePipeline({ label: 'showcase-skin', layout: pipeLayout, compute: { module: skinModule, entryPoint: 'skin' } });
  const normalPipe = device.createComputePipeline({ label: 'showcase-node-normals', layout: pipeLayout, compute: { module: skinModule, entryPoint: 'nodeNormals' } });
  const fxStepPipe = device.createComputePipeline({ label: 'showcase-fx-step', layout: pipeLayout, compute: { module: skinModule, entryPoint: 'fxStepAll' } });
  // Debris: one ring buffer for every car, and the effects uniform (dt,
  // seed, floor, static boxes).
  const fxBuf = device.createBuffer({ label: 'showcase-fx', size: fxBufferSize(FX_N), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const fxuBuf = device.createBuffer({ label: 'showcase-fx-frame', size: 32 + 32 * FX_BOXES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const fxu = new ArrayBuffer(32 + 32 * FX_BOXES);
  let fxSeed = 0, fxLive = false;
  // drawDepth's pipeline (made on first use, per depth format).
  let depthOnly = null;
  const fxModule = device.createShaderModule({ label: 'showcase-fx-draw', code: FX_DRAW_MODULE });
  const fxPipe = device.createRenderPipeline({
    label: 'showcase-fx',
    layout: 'auto',
    vertex: { module: fxModule, entryPoint: 'vsFx' },
    fragment: { module: fxModule, entryPoint: 'fsFx', targets: [{ format, blend: FX_BLEND }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: depthFormat, depthWriteEnabled: false, depthCompare: 'less' },
    multisample: { count: sampleCount },
  });
  // The pick also writes the indirect args (group 1), so the skin dispatches,
  // which read them as indirect args, must not bind them.
  const pickLayout = device.createBindGroupLayout({
    entries: [0, 1, 2].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: binding === 1 ? 'storage' : 'uniform' } })),
  });
  const pickPipeLayout = device.createPipelineLayout({ bindGroupLayouts: [skinLayout, pickLayout] });
  const pickPipe = device.createComputePipeline({ label: 'showcase-lod-pick', layout: pickPipeLayout, compute: { module: skinModule, entryPoint: 'pick' } });
  // The eye of the last draw and its pixel scale (w, 0 before the first draw): picks levels of detail.
  const viewBuf = device.createBuffer({ label: 'showcase-view', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const drawPipe = device.createRenderPipeline({
    label: 'showcase-draw',
    layout: 'auto',
    vertex: {
      module: drawModule,
      entryPoint: 'vs',
      buffers: [
        OUT_LAYOUT,
        { arrayStride: 40, attributes: [{ shaderLocation: 2, offset: 0, format: 'float32x3' }, { shaderLocation: 4, offset: 12, format: 'float32' }, { shaderLocation: 3, offset: 16, format: 'uint32' }, { shaderLocation: 5, offset: 20, format: 'float32x3' }, { shaderLocation: 6, offset: 32, format: 'float32x2' }] },
      ],
    },
    fragment: { module: drawModule, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'triangle-list', cullMode },
    depthStencil: { format: depthFormat, depthWriteEnabled: true, depthCompare: 'less' },
    multisample: { count: sampleCount },
  });
  // Glass: blended over what is behind it (the interior), after the opaque draw.
  const glassPipe = device.createRenderPipeline({
    label: 'showcase-glass',
    layout: 'auto',
    vertex: {
      module: drawModule,
      entryPoint: 'vs',
      buffers: [
        OUT_LAYOUT,
        { arrayStride: 40, attributes: [{ shaderLocation: 2, offset: 0, format: 'float32x3' }, { shaderLocation: 4, offset: 12, format: 'float32' }, { shaderLocation: 3, offset: 16, format: 'uint32' }, { shaderLocation: 5, offset: 20, format: 'float32x3' }, { shaderLocation: 6, offset: 32, format: 'float32x2' }] },
      ],
    },
    fragment: { module: drawModule, entryPoint: 'fs', targets: [{ format, blend: { color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } } }] },
    primitive: { topology: 'triangle-list', cullMode },
    depthStencil: { format: depthFormat, depthWriteEnabled: false, depthCompare: 'less' },
    multisample: { count: sampleCount },
  });
  // The floor reflection: the opaque level again, mirrored, blended by height.
  const mirrorPipe = device.createRenderPipeline({
    label: 'showcase-mirror',
    layout: 'auto',
    vertex: {
      module: drawModule,
      entryPoint: 'vs',
      buffers: [
        OUT_LAYOUT,
        { arrayStride: 40, attributes: [{ shaderLocation: 2, offset: 0, format: 'float32x3' }, { shaderLocation: 4, offset: 12, format: 'float32' }, { shaderLocation: 3, offset: 16, format: 'uint32' }, { shaderLocation: 5, offset: 20, format: 'float32x3' }, { shaderLocation: 6, offset: 32, format: 'float32x2' }] },
      ],
    },
    fragment: { module: drawModule, entryPoint: 'fsMirror', targets: [{ format, blend: { color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'zero', dstFactor: 'one' } } }] },
    primitive: { topology: 'triangle-list', cullMode },
    // Only the nearest surface (laid down by the depth pass): blended layers
    // drawn in any order would show the cabin through the body.
    depthStencil: { format: depthFormat, depthWriteEnabled: false, depthCompare: 'equal' },
    multisample: { count: sampleCount },
  });
  const mirrorDepthPipe = device.createRenderPipeline({
    label: 'showcase-mirror-depth',
    layout: 'auto',
    vertex: {
      module: drawModule,
      entryPoint: 'vs',
      buffers: [
        OUT_LAYOUT,
        { arrayStride: 40, attributes: [{ shaderLocation: 2, offset: 0, format: 'float32x3' }, { shaderLocation: 4, offset: 12, format: 'float32' }, { shaderLocation: 3, offset: 16, format: 'uint32' }, { shaderLocation: 5, offset: 20, format: 'float32x3' }, { shaderLocation: 6, offset: 32, format: 'float32x2' }] },
      ],
    },
    fragment: { module: drawModule, entryPoint: 'fsDepth', targets: [{ format, writeMask: 0 }] },
    primitive: { topology: 'triangle-list', cullMode },
    depthStencil: { format: depthFormat, depthWriteEnabled: true, depthCompare: 'less' },
    multisample: { count: sampleCount },
  });
  const contactPipe = device.createRenderPipeline({
    label: 'showcase-contact',
    layout: 'auto',
    vertex: {
      module: drawModule,
      entryPoint: 'vsContact',
      buffers: [
        { arrayStride: OUT_STRIDE, attributes: OUT_LAYOUT.attributes.slice(0, 3) },
        { arrayStride: 40, attributes: [{ shaderLocation: 2, offset: 0, format: 'float32x3' }, { shaderLocation: 3, offset: 16, format: 'uint32' }] },
      ],
    },
    // Multiplies what is under it (the floor and its reflection) by 1 − a.
    fragment: { module: drawModule, entryPoint: 'fsContact', targets: [{ format, blend: { color: { srcFactor: 'zero', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'zero', dstFactor: 'one' } } }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: depthFormat, depthWriteEnabled: false, depthCompare: 'always' },
    multisample: { count: sampleCount },
  });
  const contactBuf = device.createBuffer({ label: 'showcase-contact-frame', size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const contactBind = device.createBindGroup({ layout: contactPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: contactBuf } }] });
  const mirrorBuf = device.createBuffer({ label: 'showcase-mirror-frame', size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const mirrorBind = device.createBindGroup({ layout: mirrorPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: mirrorBuf } }] });
  const mirrorDepthBind = device.createBindGroup({ layout: mirrorDepthPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: mirrorBuf } }] });
  const frameBuf = device.createBuffer({ label: 'showcase-frame', size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  // Colour maps: one texture array shared by every car (setTextures), a
  // white layer until then.
  const texSamp = device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat', maxAnisotropy: 8 });
  let texArr = device.createTexture({ size: [1, 1, 1], format: 'rgba8unorm-srgb', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture: texArr }, new Uint8Array([255, 255, 255, 255]), { bytesPerRow: 4 }, [1, 1, 1]);
  let texLayers = 0;
  // Data maps (normal, roughness, AO; linear): a flat layer until setDataTextures.
  let texData = device.createTexture({ size: [1, 1, 1], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture: texData }, new Uint8Array([128, 128, 255, 255]), { bytesPerRow: 4 }, [1, 1, 1]);
  // Texture sets of every car's levels (DRAW_WGSL texSets): 4 words each, set
  // k at 4·k; rebuilt as cars are added.
  let setWords = [0, 0, 0, 0];
  let setsBuf = null;
  const syncSets = () => {
    setsBuf?.destroy();
    setsBuf = upload('showcase-tex-sets', new Uint32Array(setWords));
  };
  const texEntries = () => [{ binding: 1, resource: texArr.createView({ dimension: '2d-array' }) }, { binding: 2, resource: texSamp },
    { binding: 3, resource: texData.createView({ dimension: '2d-array' }) }, { binding: 4, resource: { buffer: setsBuf } }];
  const makeDrawBinds = () => [drawPipe, glassPipe].map((p) => device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: frameBuf } }, ...texEntries()] }));
  let drawBind = null, glassBind = null;
  const fxBind = device.createBindGroup({ layout: fxPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: frameBuf } }, { binding: 1, resource: { buffer: fxBuf } }] });
  const cars = new Map();
  let nextId = 1;
  let nodesBuf = null;
  let miscBuf = null;
  let beamsBuf = null;
  let posBuf = null, restBuf = null, posMisc = null, posPipes = null;
  const bindFor = (car, lv, src0, src1, beams) => device.createBindGroup({
    layout: skinLayout,
    entries: [src0, src1, lv.binds, lv.out, lv.params, car.topo, beams, car.nodeN, fxuBuf, fxBuf]
      .map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const makePosBind = (car) => posBuf && car.levels.map((lv) => bindFor(car, lv, posBuf, posMisc ?? posBuf, restBuf));
  const makeSkinBind = (car) => car.levels.map((lv) => bindFor(car, lv, nodesBuf, miscBuf, beamsBuf));
  const upload = (label, data, usage = GPUBufferUsage.STORAGE) => {
    const b = device.createBuffer({ label, size: Math.max(16, Math.ceil(data.byteLength / 4) * 4), usage: usage | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, data);
    return b;
  };
  syncSets();
  [drawBind, glassBind] = makeDrawBinds();
  // Upload `layers` (decodeTexture's mipped RGBA8, null: `fill`) as a new
  // `format` array of `size` × `size`.
  const layerArray = (layers, size, format, fill) => {
    const mips = Math.log2(size) + 1;
    if (!Number.isInteger(mips)) throw new Error('showcase textures: size must be a power of two');
    const t = device.createTexture({ size: [size, size, Math.max(1, layers.length)], mipLevelCount: mips, format, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    layers.forEach((data, z) => {
      let off = 0;
      for (let l = 0, sz = size; l < mips; l++, sz = Math.max(1, sz >> 1)) {
        const n = sz * sz * 4;
        let px = data ? data.subarray(off, off + n) : null;
        if (!px) { px = new Uint8Array(n); for (let k = 0; k < n; k += 4) px.set(fill, k); }
        device.queue.writeTexture({ texture: t, mipLevel: l, origin: [0, 0, z] }, px, { bytesPerRow: sz * 4, rowsPerImage: sz }, [sz, sz, 1]);
        off += n;
      }
    });
    return t;
  };

  return {
    /**
     * The colour maps (wasm `decodeTexture(bytes, size)` per file of a
     * visual's `textures`, null for a missing one: white), `size` × `size`
     * with mips. Returns the first layer's index: pass it as addCar's
     * `texBase`. Replaces any earlier set (cars added before keep theirs only
     * if this set starts with the same layers).
     */
    setTextures(layers, size) {
      texArr.destroy();
      texArr = layerArray(layers, size, 'rgba8unorm-srgb', [255, 255, 255, 255]);
      texLayers = layers.length;
      [drawBind, glassBind] = makeDrawBinds();
      return 0;
    },
    /**
     * The data maps (normal, roughness, AO: linear; `decodeTexture` per file of a
     * visual's `dataTextures`, null for a missing one: flat), as setTextures; pass
     * the first layer's index as addCar's `dataBase`.
     */
    setDataTextures(layers, size) {
      texData.destroy();
      texData = layerArray(layers, size, 'rgba8unorm', [128, 128, 255, 255]);
      [drawBind, glassBind] = makeDrawBinds();
      return 0;
    },
    /**
     * Skin from world-space node positions instead (xyz f32 per node, all vehicles in
     * order; e.g. the crash lab's positionsBuffer(), so replays deform the body too),
     * with the beams' state from `rest` (f32 per beam, all vehicles in order, negative
     * when broken: the lab's restBuffer()). Call again when either buffer is replaced;
     * null returns to the solver buffers. `misc`: the solver's miscBuffer(), for raycast
     * wheels (their pieces are placed from its drive records).
     */
    setPositions(buf, rest, misc = null) {
      if (buf && !rest) throw new Error('showcase setPositions: the beams\' rest buffer is needed (crash-lab restBuffer())');
      posBuf = buf;
      restBuf = rest ?? null;
      posMisc = misc;
      if (buf && !posPipes) {
        const m = device.createShaderModule({ label: 'showcase-skin-pos', code: crease(SKIN_POS_WGSL) });
        posPipes = {
          pick: device.createComputePipeline({ label: 'showcase-lod-pick-pos', layout: pickPipeLayout, compute: { module: m, entryPoint: 'pick' } }),
          skin: device.createComputePipeline({ label: 'showcase-skin-pos', layout: pipeLayout, compute: { module: m, entryPoint: 'skin' } }),
          normal: device.createComputePipeline({ label: 'showcase-node-normals-pos', layout: pipeLayout, compute: { module: m, entryPoint: 'nodeNormals' } }),
        };
      }
      for (const car of cars.values()) car.posBind = makePosBind(car);
    },
    /** The explicit solver's node, misc and beam buffers (call again after a rebuild). */
    setSolver(nodes, misc, beams) {
      if (!beams) throw new Error('showcase setSolver: the solver\'s beamsBuffer() is needed (damage units)');
      nodesBuf = nodes;
      miscBuf = misc;
      beamsBuf = beams;
      for (const car of cars.values()) car.skinBind = makeSkinBind(car);
    },
    /**
     * Add a car: `visual` from the wasm (`showcaseCar(...).visual`), `nodeOffset` its
     * vehicle's first node in the solver (`layout.vehicles[i].nodeOffset`), `vehicle`
     * its index (for its origin). Returns an id.
     */
    addCar(visual, { nodeOffset = 0, vehicle = 0, beamOffset = 0, lod = null, texBase = 0, dataBase = 0, wheelRecords = null } = {}) {
      const nodeCount = visual.nodeCount;
      const nt = visual.normalTris, start = visual.nodeTriStart, list = visual.nodeTriList;
      const units = visual.unitKind?.length ?? 0;
      const nbs = visual.nodeBeamStart, nbl = visual.nodeBeamList, r0 = visual.beamRest0;
      if (!nbs || !r0) throw new Error('showcase addCar: the visual has no beam tables (rebuild the wasm)');
      // topo's tables (SKIN_WGSL's Params documents the layout).
      const listAt = start.length, trisAt = listAt + list.length;
      const restNAt = trisAt + (nt.length / 3) * 4;
      const nodeBeamAt = restNAt + nodeCount * 8;
      const rest0At = nodeBeamAt + nbs.length + nbl.length;
      const unitAt = rest0At + r0.length;
      const unitListAt = unitAt + units * 4;
      const nodeMatAt = unitListAt + (units ? visual.unitBeams.length : 0);
      const deathAt = nodeMatAt + nodeCount;
      const death = visual.skinDeath ?? new Uint32Array(0);
      // Raycast wheels' misc records (ExplicitSim.wheelRecords), 4 per wheel.
      const wheelAt = deathAt + death.length;
      const wheelRecs = wheelRecords ?? new Uint32Array(0);
      const topo = new Uint32Array(wheelAt + Math.max(wheelRecs.length, 1));
      topo.set(death, deathAt);
      topo.set(wheelRecs, wheelAt);
      const tf = new Float32Array(topo.buffer);
      topo.set(start, 0);
      topo.set(list, listAt);
      const nr = visual.nodeRest;
      for (let t = 0; t < nt.length / 3; t++) {
        topo.set(nt.subarray(3 * t, 3 * t + 3), trisAt + 4 * t);
        if (nr) {
          // Word 3: the triangle's rest perimeter (tears).
          const q = (k) => [nr[3 * nt[3 * t + k]], nr[3 * nt[3 * t + k] + 1], nr[3 * nt[3 * t + k] + 2]];
          const d = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
          const [a, b, c] = [q(0), q(1), q(2)];
          tf[trisAt + 4 * t + 3] = d(a, b) + d(b, c) + d(c, a);
        }
      }
      for (let i = 0; i < nodeCount; i++) {
        tf.set(visual.nodeRestNormal.subarray(3 * i, 3 * i + 3), restNAt + 8 * i);
        if (nr) tf.set(nr.subarray(3 * i, 3 * i + 3), restNAt + 8 * i + 4);
      }
      topo.set(nbs, nodeBeamAt);
      topo.set(nbl, nodeBeamAt + nbs.length);
      tf.set(r0, rest0At);
      for (let u = 0; u < units; u++) {
        topo[unitAt + 4 * u] = unitListAt + visual.unitStart[u];
        topo[unitAt + 4 * u + 1] = unitListAt + visual.unitStart[u + 1];
        topo[unitAt + 4 * u + 2] = visual.unitKind[u];
        tf[unitAt + 4 * u + 3] = visual.unitStrain[u];
      }
      if (units) topo.set(visual.unitBeams, unitListAt);
      if (visual.nodeMaterial) topo.set(visual.nodeMaterial, nodeMatAt);
      // The paint's colour (the first painted vertex's, linear 8-bit RGB) for chips.
      let paint = 0x808080;
      for (let i = 0; i < visual.vertexCount; i++) {
        if (visual.material[i] === 0) {
          const c = [0, 1, 2].map((k) => Math.max(0, Math.min(255, Math.round(visual.colors[3 * i + k] * 255))));
          paint = c[0] | (c[1] << 8) | (c[2] << 16);
          break;
        }
      }
      const stateAt = nodeCount * 6;
      const level = (v, lvIndex) => {
        const n = v.vertexCount;
        // Shut-line fallbacks after the n bindings (word 3 links them).
        const na = v.alt && v.alt.length === n ? v.altTri.length : 0;
        // Weighted skins (flexskin): 4 entries a 64-byte slot after the
        // fallbacks; per vertex its entries' first slot.
        const ws = v.wtStart && v.wtStart.length === n + 1 ? v.wtStart : null;
        const wslot = ws ? new Uint32Array(n) : null;
        let nslots = 0;
        if (ws) for (let i = 0; i < n; i++) { wslot[i] = n + na + nslots; nslots += Math.ceil((ws[i + 1] - ws[i]) / 4); }
        const bindData = new ArrayBuffer((n + na + nslots) * 64);
        const u = new Uint32Array(bindData);
        const f = new Float32Array(bindData);
        for (let i = 0; i < n; i++) {
          const o = i * 16;
          u[o] = v.bindNodes[3 * i];
          u[o + 1] = v.bindNodes[3 * i + 1];
          u[o + 2] = v.bindNodes[3 * i + 2];
          u[o + 3] = vehicle;
          f[o + 4] = v.bindCoords[3 * i];
          f[o + 5] = v.bindCoords[3 * i + 1];
          f[o + 6] = v.bindCoords[3 * i + 2];
          f[o + 7] = v.bindRest[2 * i];
          f[o + 8] = v.bindNormal[3 * i];
          f[o + 9] = v.bindNormal[3 * i + 1];
          f[o + 10] = v.bindNormal[3 * i + 2];
          f[o + 11] = v.bindRest[2 * i + 1];
          f[o + 12] = v.normals[3 * i];
          f[o + 13] = v.normals[3 * i + 1];
          f[o + 14] = v.normals[3 * i + 2];
          // Flags: smooth, rigid, damage unit + 1 (bits 2-15), shard seed
          // (16), material (17-22).
          const un = v.unit ? v.unit[i] : 0xffffffff;
          // Rigid kinds (skin.rs rigid_kind): 1 rigid, 2 a raycast wheel's
          // (bit 23, spinning: 24), 3 its caliper (bit 23).
          const rk = v.rigid ? v.rigid[i] : 0;
          if (rk >= 2 && !wheelRecords) throw new Error('showcase addCar: raycast wheel pieces need wheelRecords (ExplicitSim.wheelRecords)');
          u[o + 15] = (v.smooth ? v.smooth[i] : 0) | (rk === 1 ? 2 : 0) | (rk >= 2 ? 1 << 23 : 0) | (rk === 2 ? 1 << 24 : 0)
            | (un === 0xffffffff ? 0 : ((un + 1) & 0x3fff) << 2) | (v.shard && v.shard[i] ? 1 << 16 : 0) | (v.material[i] << 17);
          const a = na ? v.alt[i] : 0;
          u[o + 3] = a ? n + a - 1 : 0;
          if (ws && ws[i + 1] > ws[i]) {
            // Entries sorted by |A| (the first three are the vertex's plastic,
            // scrape and damage nodes), packed (node, A, dA/dT, dA/dB).
            const k0 = ws[i], k1 = ws[i + 1];
            const ord = [];
            for (let k = k0; k < k1; k++) ord.push(k);
            ord.sort((x, y) => Math.abs(v.wtW[3 * y]) - Math.abs(v.wtW[3 * x]));
            u[o] = k1 - k0;
            u[o + 3] = wslot[i];
            f[o + 4] = v.wtOff[3 * i]; f[o + 5] = v.wtOff[3 * i + 1]; f[o + 6] = v.wtOff[3 * i + 2];
            f[o + 8] = v.wtNrm[3 * i]; f[o + 9] = v.wtNrm[3 * i + 1]; f[o + 10] = v.wtNrm[3 * i + 2];
            u[o + 15] = (u[o + 15] & ~3) | (1 << 25);
            ord.forEach((k, e) => {
              const q = wslot[i] * 16 + 4 * e;
              u[q] = v.wtNode[k];
              f[q + 1] = v.wtW[3 * k]; f[q + 2] = v.wtW[3 * k + 1]; f[q + 3] = v.wtW[3 * k + 2];
            });
          }
          if (a) {
            // The fallback: its own-part frame, the same vertex otherwise.
            const q = (n + a - 1) * 16, k = a - 1;
            u.set(u.subarray(o, o + 16), q);
            u[q] = v.altNodes[3 * k]; u[q + 1] = v.altNodes[3 * k + 1]; u[q + 2] = v.altNodes[3 * k + 2];
            u[q + 3] = v.altTri[k];
            f[q + 4] = v.altCoords[3 * k]; f[q + 5] = v.altCoords[3 * k + 1]; f[q + 6] = v.altCoords[3 * k + 2];
            f[q + 7] = v.altRest[2 * k];
            f[q + 8] = v.altNormal[3 * k]; f[q + 9] = v.altNormal[3 * k + 1]; f[q + 10] = v.altNormal[3 * k + 2];
            f[q + 11] = v.altRest[2 * k + 1];
          }
        }
        const binds = upload('showcase-binds', bindData);
        // Deformed (position | torn, normal | damage, plastic | scrape) per
        // vertex: the compute pass writes it, the draw reads it as a vertex
        // buffer. Starts at the rest pose.
        const out = device.createBuffer({ label: 'showcase-out', size: n * OUT_STRIDE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        const rest = new Float32Array(n * 8);
        for (let i = 0; i < n; i++) {
          rest.set(v.positions.subarray(3 * i, 3 * i + 3), 8 * i);
          rest.set(v.normals.subarray(3 * i, 3 * i + 3), 8 * i + 4);
        }
        device.queue.writeBuffer(out, 0, rest);
        // Per vertex: colour, AO, material, rest position (the glass's crack
        // pattern is drawn in rest space so it rides the glass).
        // and texture coordinates (printed parts: tyres, plates, displays).
        // The level's texture sets into the shared table, its files offset
        // by the car's layer bases (a visual without sets: one per colour map).
        const vs = v.texSets ?? Uint16Array.from((v.textures ?? []).flatMap((_, k) => [k + 1, 0, 0, 0]));
        const setBase = setWords.length / 4;
        for (let k = 0; k < vs.length; k += 4) {
          setWords.push(vs[k] ? vs[k] + texBase : 0, vs[k + 1] ? vs[k + 1] + dataBase : 0, vs[k + 2] ? vs[k + 2] + dataBase : 0, vs[k + 3] ? vs[k + 3] + dataBase : 0);
        }
        const attr = new ArrayBuffer(n * 40);
        const af = new Float32Array(attr);
        const au = new Uint32Array(attr);
        for (let i = 0; i < n; i++) {
          af.set(v.colors.subarray(3 * i, 3 * i + 3), 10 * i);
          af[10 * i + 3] = v.ao ? v.ao[i] : 1;
          // Material (low byte), texture set + 1 (high half; 0 none).
          const t = v.tex ? v.tex[i] : 0;
          au[10 * i + 4] = v.material[i] | (t ? (t + setBase) << 16 : 0);
          af.set(v.positions.subarray(3 * i, 3 * i + 3), 10 * i + 5);
          if (v.uvs) af.set(v.uvs.subarray(2 * i, 2 * i + 2), 10 * i + 8);
        }
        // Opaque triangles first, the blended ones (glass, lenses) after, in
        // the mesh's order (the exterior before the cabin: early depth
        // rejection). A second index list for the floor passes: the opaque
        // triangles within 25 cm of the ground at rest (tyres, sill and bumper
        // edges: what can touch the floor; not the underbody plate or the
        // cabin) for the contact shadows, then the rest under 0.95 m for the
        // floor reflection (faded to under 3 % above that).
        const tri = v.indices;
        const TRANSPARENT = new Set([1, 17, 20, 25, 27]);
        const LOW = 0.25, MID = 0.95;
        const NO_CONTACT = new Set([9, 12, 13, 14, 15, 16, 26]);
        let yMin = Infinity;
        for (let i = 0; i < n; i++) if (v.positions[3 * i + 1] < yMin) yMin = v.positions[3 * i + 1];
        const minY = (t) => Math.min(v.positions[3 * tri[t] + 1], v.positions[3 * tri[t + 1] + 1], v.positions[3 * tri[t + 2] + 1]) - yMin;
        // Class: 0 low, 1 mid, 2 high (opaque), 3 blended.
        const cls = (t) => TRANSPARENT.has(v.material[tri[t]]) ? 3 : minY(t) < LOW && !NO_CONTACT.has(v.material[tri[t]]) ? 0 : minY(t) < MID ? 1 : 2;
        const count = [0, 0, 0, 0];
        const cl = new Uint8Array(tri.length / 3);
        for (let t = 0; t < tri.length; t += 3) { cl[t / 3] = cls(t); count[cl[t / 3]] += 3; }
        const nl = count[0], nm = count[0] + count[1], no = nm + count[2];
        const ordered = new Uint32Array(tri.length);
        const floorList = new Uint32Array(Math.max(nm, 3));
        let io = 0, ig = no;
        const fat = [0, nl];
        // The hidden layers last among the opaque (the cabin, inner stampings
        // and the bay, behind intact panels and glass): the depth test rejects
        // them before shading.
        const HIDDEN = new Set([12, 13, 14, 15, 16, 26, 30, 31]);
        for (const late of [false, true]) {
          for (let t = 0; t < tri.length; t += 3) {
            const k = cl[t / 3];
            if (k === 3 || HIDDEN.has(v.material[tri[t]]) !== late) continue;
            ordered.set(tri.subarray(t, t + 3), io); io += 3;
          }
        }
        for (let t = 0; t < tri.length; t += 3) {
          const k = cl[t / 3];
          if (k === 3) { ordered.set(tri.subarray(t, t + 3), ig); ig += 3; }
          if (k < 2) { floorList.set(tri.subarray(t, t + 3), fat[k]); fat[k] += 3; }
        }
        const attrs = device.createBuffer({ label: 'showcase-attrs', size: attr.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(attrs, 0, attr);
        const index = device.createBuffer({ label: 'showcase-index', size: ordered.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(index, 0, ordered);
        const floorIndex = device.createBuffer({ label: 'showcase-floor-index', size: floorList.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(floorIndex, 0, floorList);
        const params = device.createBuffer({ label: 'showcase-params', size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const pw = () => new Uint32Array([n, car.nodeOffset, nodeCount, vehicle, listAt, trisAt, car.beamOffset, lvIndex, restNAt, nodeBeamAt, rest0At, unitAt, units, nodeMatAt, paint, stateAt, deathAt, wheelAt, 0, 0]);
        return { n, binds, out, attrs, index, floorIndex, params, pw, opaqueCount: no, glassCount: tri.length - no, lowCount: nl, midCount: nm };
      };
      const car = { nodeOffset, beamOffset };
      const extra = (Array.isArray(lod) ? lod : [lod]).filter(Boolean).slice(0, 2);
      const levels = [visual, ...extra].map((v, k) => level(v, k));
      syncSets();
      [drawBind, glassBind] = makeDrawBinds();
      for (const lv of levels) device.queue.writeBuffer(lv.params, 0, lv.pw());
      // SKIN_WGSL's Lod: per level n, opaque, glass, low, mid counts, edge.
      const lodData = new ArrayBuffer(112);
      const lu = new Uint32Array(lodData), lf = new Float32Array(lodData);
      levels.forEach((lv, k) => {
        lu[k] = lv.n; lu[4 + k] = lv.opaqueCount; lu[8 + k] = lv.glassCount; lu[12 + k] = lv.lowCount; lu[16 + k] = lv.midCount;
        lf[20 + k] = [visual, ...extra][k].edgeLength ?? 0;
      });
      lu[24] = levels.length;
      const argWords = new Uint32Array(levels.length * 23);
      levels.forEach((lv, k) => {
        const on = k === 0 ? 1 : 0;
        argWords.set([on * Math.ceil(lv.n / WG), 1, 1, lv.opaqueCount, on, 0, 0, 0, lv.glassCount, on, lv.opaqueCount, 0, 0, lv.lowCount, on, 0, 0, 0, lv.midCount, on, 0, 0, 0], k * 23);
      });
      Object.assign(car, {
        nodeCount, vehicle, levels, units, stateAt, unitNames: visual.unitNames ?? [], skinBind: null, posBind: null,
        topo: upload('showcase-topology', topo),
        // Per node 6 vec4, per unit 1, the level state 1 (zeros: nothing yet).
        nodeN: device.createBuffer({ label: 'showcase-node-state', size: (stateAt + units + 1) * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }),
        lod: upload('showcase-lod', lodData, GPUBufferUsage.UNIFORM),
        // Indirect args (23 words a level); level 0 drawn until the first pick.
        args: upload('showcase-lod-args', argWords, GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT),
      });
      car.pickBind = device.createBindGroup({
        layout: pickLayout,
        entries: [car.lod, car.args, viewBuf].map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      if (nodesBuf) car.skinBind = makeSkinBind(car);
      car.posBind = makePosBind(car);
      const id = nextId++;
      cars.set(id, car);
      return id;
    },
    /** Move a car to another vehicle slot (after a rebuild with a new layout). */
    /** Point a car at another first node (same vehicle index; re-add to change it). */
    setNodeOffset(id, nodeOffset, beamOffset) {
      const car = cars.get(id);
      if (!car) return;
      car.nodeOffset = nodeOffset;
      if (beamOffset !== undefined) car.beamOffset = beamOffset;
      for (const lv of car.levels) device.queue.writeBuffer(lv.params, 0, lv.pw());
    },
    removeCar(id) {
      const car = cars.get(id);
      if (!car) return;
      for (const lv of car.levels) for (const b of [lv.binds, lv.out, lv.attrs, lv.index, lv.floorIndex, lv.params]) b.destroy();
      for (const b of [car.topo, car.nodeN, car.lod, car.args]) b.destroy();
      cars.delete(id);
      // The last car gone: the texture sets' table starts over.
      if (!cars.size) setWords = [0, 0, 0, 0];
    },
    /**
     * A car's whole per-node and per-unit render state (normals, plastic strain, paint
     * worn by sliding, broken units: the history `update` keeps), copied in `encoder`
     * and resolved once mapped to a Float32Array, for a snapshot of the car as drawn
     * (crash-orbit.html). `writeNodeState(id, data)` puts one back.
     */
    readNodeState(encoder, id) {
      const car = cars.get(id);
      if (!car) return async () => null;
      const size = car.nodeN.size;
      const rb = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      encoder.copyBufferToBuffer(car.nodeN, 0, rb, 0, size);
      return async () => {
        await rb.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(rb.getMappedRange().slice(0));
        rb.destroy();
        return out;
      };
    },
    writeNodeState(id, data) {
      const car = cars.get(id);
      if (!car) throw new Error(`showcase writeNodeState: no car ${id}`);
      if (data.byteLength !== car.nodeN.size) throw new Error(`showcase writeNodeState: ${data.byteLength} bytes for a ${car.nodeN.size}-byte state`);
      device.queue.writeBuffer(car.nodeN, 0, data);
    },
    /**
     * The damage units' state as the renderer holds it (for checks): per car id
     * `{ names, state: Float32Array(4 per unit: broken, strain, shards due,
     * broken as of the last moving update) }`, copied in `encoder` (after
     * update) and resolved once mapped.
     */
    readUnits(encoder) {
      const reads = [];
      for (const [id, car] of cars) {
        if (!car.units) continue;
        const size = car.units * 16;
        const rb = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        encoder.copyBufferToBuffer(car.nodeN, car.stateAt * 16, rb, 0, size);
        reads.push([id, car, rb]);
      }
      return async () => {
        const out = new Map();
        for (const [id, car, rb] of reads) {
          await rb.mapAsync(GPUMapMode.READ);
          out.set(id, { names: car.unitNames, state: new Float32Array(rb.getMappedRange().slice(0)) });
          rb.destroy();
        }
        return out;
      };
    },
    /**
     * Deform every car from the solver's current node positions (or the positions
     * buffer given to setPositions): node normals, the level-of-detail pick (by
     * the last draw's eye), then the picked level's skin.
     */
    update(encoder, { dt = 0, floorY = 0, boxes = null, lead = 0 } = {}) {
      const usePos = !!posBuf;
      if (!usePos && !nodesBuf) return;
      // Effects: dt the simulated seconds since the last update (0: nothing
      // spawns, moves or wears), the floor, the static boxes (layout.boxes rows).
      const uw = new Uint32Array(fxu), uf = new Float32Array(fxu);
      const live = dt > 0 && dt < 0.25;
      uf[0] = live ? dt : 0;
      uw[1] = ++fxSeed;
      uf[2] = floorY;
      uf[4] = lead;
      const nb = boxes ? Math.min(boxes.length, FX_BOXES) : 0;
      uw[3] = nb;
      for (let k = 0; k < nb; k++) {
        const [cx, cy, cz, hx, hy, hz, , , , , kind] = boxes[k];
        uf.set([cx, cy, cz, kind, hx, hy, hz, 0], 8 + 8 * k);
      }
      device.queue.writeBuffer(fxuBuf, 0, fxu);
      if (live) fxLive = true;
      const pipes = usePos ? posPipes : { normal: normalPipe, skin: skinPipe, pick: pickPipe };
      const pass = encoder.beginComputePass({ label: usePos ? 'showcase-skin-pos' : 'showcase-skin' });
      let anyGroup = null;
      for (const car of cars.values()) {
        const groups = usePos ? car.posBind : car.skinBind;
        anyGroup = groups[0];
        pass.setBindGroup(0, groups[0]);
        pass.setPipeline(pipes.normal);
        pass.dispatchWorkgroups(Math.ceil((car.nodeCount + car.units) / WG));
        pass.setPipeline(pipes.pick);
        pass.setBindGroup(1, car.pickBind);
        pass.dispatchWorkgroups(1);
        pass.setPipeline(pipes.skin);
        car.levels.forEach((lv, k) => {
          pass.setBindGroup(0, groups[k]);
          pass.dispatchWorkgroupsIndirect(car.args, k * 92);
        });
      }
      if (live && anyGroup) {
        pass.setPipeline(fxStepPipe);
        pass.setBindGroup(0, anyGroup);
        pass.dispatchWorkgroups(Math.ceil(FX_N / 64));
      }
      pass.end();
    },
    /**
     * The cars' reflection in a floor at y = `floorY` (reflectance `strength`
     * at the contact, fading with height), then the contact shadows (each
     * layer within 7 cm of the floor darkens it by up to `contact`): draw it
     * after the floor (which must not write depth) and before the cars. Uses
     * its own uniforms, so it can share a command encoder with `draw`.
     */
    drawReflection(pass, { viewProj, eye, light = [0.4, 0.8, 0.35], strength = 0.22, floorY = 0, contact = 0.3 }) {
      // viewProj · M, M the mirror in y = floorY: y → 2·floorY − y.
      const vp = Float32Array.from(viewProj);
      for (let k = 0; k < 4; k++) { vp[12 + k] += 2 * floorY * vp[4 + k]; vp[4 + k] = -vp[4 + k]; }
      const fr = new Float32Array(24);
      fr.set(vp, 0);
      fr.set([eye[0], 2 * floorY - eye[1], eye[2], floorY], 16);
      fr.set([light[0], light[1], light[2], strength], 20);
      device.queue.writeBuffer(mirrorBuf, 0, fr);
      for (const [pipe, bind] of skip.has('reflect') ? [] : [[mirrorDepthPipe, mirrorDepthBind], [mirrorPipe, mirrorBind]]) {
        pass.setPipeline(pipe);
        pass.setBindGroup(0, bind);
        for (const car of cars.values()) {
          car.levels.forEach((lv, k) => {
            pass.setVertexBuffer(0, lv.out);
            pass.setVertexBuffer(1, lv.attrs);
            pass.setIndexBuffer(lv.floorIndex, 'uint32');
            pass.drawIndexedIndirect(car.args, k * 92 + 72);
          });
        }
      }
      if (contact > 0 && !skip.has('contact')) {
        const cf = new Float32Array(24);
        cf.set(viewProj, 0);
        cf.set([eye[0], eye[1], eye[2], floorY], 16);
        cf.set([light[0], light[1], light[2], contact], 20);
        device.queue.writeBuffer(contactBuf, 0, cf);
        pass.setPipeline(contactPipe);
        pass.setBindGroup(0, contactBind);
        for (const car of cars.values()) {
          car.levels.forEach((lv, k) => {
            pass.setVertexBuffer(0, lv.out);
            pass.setVertexBuffer(1, lv.attrs);
            pass.setIndexBuffer(lv.floorIndex, 'uint32');
            pass.drawIndexedIndirect(car.args, k * 92 + 52);
          });
        }
      }
    },
    /** Draw every car into `pass` (a render pass with a colour target of `format` and depth). */
    draw(pass, { viewProj, eye, light = [0.4, 0.8, 0.35], height = 1080 }) {
      const fr = new Float32Array(24);
      fr.set(viewProj, 0);
      fr.set([eye[0], eye[1], eye[2], 1], 16);
      fr.set([light[0], light[1], light[2], 0], 20);
      device.queue.writeBuffer(frameBuf, 0, fr);
      // The pick's pixel scale: viewport height / 2 × the projection's y scale
      // (the length of viewProj's second row; a rotation keeps it).
      const py = Math.hypot(viewProj[1], viewProj[5], viewProj[9]);
      device.queue.writeBuffer(viewBuf, 0, new Float32Array([eye[0], eye[1], eye[2], 0.5 * height * py]));
      for (const [pipe, bind, at] of [[drawPipe, drawBind, 12, 'opaque'], [glassPipe, glassBind, 32, 'glass']].filter((d) => !skip.has(d[3]))) {
        pass.setPipeline(pipe);
        pass.setBindGroup(0, bind);
        for (const car of cars.values()) {
          car.levels.forEach((lv, k) => {
            pass.setVertexBuffer(0, lv.out);
            pass.setVertexBuffer(1, lv.attrs);
            pass.setIndexBuffer(lv.index, 'uint32');
            pass.drawIndexedIndirect(car.args, k * 92 + at);
          });
        }
      }
      // Debris (6 vertices a particle; dead ones collapse outside the view).
      if (fxLive && !skip.has('fx')) {
        pass.setPipeline(fxPipe);
        pass.setBindGroup(0, fxBind);
        pass.draw(6, FX_N);
      }
    },
    /**
     * Every car's opaque level into a depth-only pass (a shadow map): `viewProj`
     * the light's, `format` the pass's depth format. Uses its own uniform.
     */
    drawDepth(pass, { viewProj, format: depthFmt = 'depth32float' }) {
      if (!depthOnly || depthOnly.format !== depthFmt) {
        const module = device.createShaderModule({
          label: 'showcase-depth',
          code: '@group(0) @binding(0) var<uniform> vp: mat4x4f;\n@vertex fn vs(@location(0) p: vec3f, @location(8) tw: u32) -> @builtin(position) vec4f {\n  // A torn vertex (its triangles discarded in the colour pass) collapses outside.\n  if (unpack2x16float(tw).x > 0.5) { return vec4f(2.0, 2.0, 2.0, 1.0); }\n  return vp * vec4f(p, 1.0);\n}',
        });
        const pipe = device.createRenderPipeline({
          label: 'showcase-depth', layout: 'auto',
          vertex: { module, entryPoint: 'vs', buffers: [{ arrayStride: OUT_STRIDE, attributes: OUT_LAYOUT.attributes.slice(0, 2) }] },
          primitive: { topology: 'triangle-list', cullMode: 'none' },
          depthStencil: { format: depthFmt, depthWriteEnabled: true, depthCompare: 'less', depthBias: 2, depthBiasSlopeScale: 2.5 },
        });
        const buf = device.createBuffer({ label: 'showcase-depth-vp', size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        depthOnly = { format: depthFmt, pipe, buf, bind: device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] }) };
      }
      device.queue.writeBuffer(depthOnly.buf, 0, Float32Array.from(viewProj));
      pass.setPipeline(depthOnly.pipe);
      pass.setBindGroup(0, depthOnly.bind);
      for (const car of cars.values()) {
        car.levels.forEach((lv, k) => {
          pass.setVertexBuffer(0, lv.out);
          pass.setIndexBuffer(lv.index, 'uint32');
          pass.drawIndexedIndirect(car.args, k * 92 + 12);
        });
      }
    },
    destroy() {
      for (const id of [...cars.keys()]) this.removeCar(id);
      frameBuf.destroy();
      mirrorBuf.destroy();
      viewBuf.destroy();
      fxBuf.destroy();
      fxuBuf.destroy();
    },
  };
}
