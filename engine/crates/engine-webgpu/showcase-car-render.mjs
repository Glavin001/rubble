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
//   r.setSolver(sim.nodesBuffer(), sim.miscBuffer());        // again whenever the sim is rebuilt
//   const id = r.addCar(car.visual, { nodeOffset: layout.vehicles[i].nodeOffset, vehicle: i });
//   // each frame, after sim.frame(steps):
//   r.update(encoder);                                       // compute: deform every car
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
// Materials (ids, `visual.materials`): paint (metallic flake base under a
// clearcoat), glass, chrome, trim (gloss black plastic), headlight (projector,
// emissive), taillight, tyre, rim, gap, underbody, plate, grille, seat, dash,
// doorcard, headliner, carpet, lens (clear lamp lens), drl (emissive white),
// reflector, tail_lens (red), tail_led (emissive red), brake, caliper,
// tail_back. Glass and both lenses are drawn blended in a second pass. Base
// colours in `visual.colors` (linear RGB; the paint from the wasm's `paint`
// param), baked ambient occlusion in `visual.ao`. Reflections come from a
// procedural studio (cyclorama, overhead softbox, side strips, a key box).

const WG = 64;

const SKIN_WGSL = /* wgsl */ `
struct Bind { n0: u32, n1: u32, n2: u32, vehicle: u32, a: f32, b: f32, h: f32, r1: f32, nt: f32, nb: f32, nn: f32, r2: f32, rx: f32, ry: f32, rz: f32, flags: u32 };
struct Params { count: u32, nodeOffset: u32, nodeCount: u32, vehicle: u32, listAt: u32, trisAt: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> nodes: array<vec4f>;
@group(0) @binding(1) var<storage, read> misc: array<vec4u>;
@group(0) @binding(2) var<storage, read> binds: array<Bind>;
@group(0) @binding(3) var<storage, read_write> outv: array<vec4f>;
@group(0) @binding(4) var<uniform> params: Params;
// Node normals: per node (start, end) into the triangle list, the skin triangles,
// rest normals, and the normals now (written by nodeNormals).
// topo: per node its triangle range (nodeCount + 1 starts), the triangle list (at
// listAt), the triangles' nodes (4 words each, at trisAt).
@group(0) @binding(5) var<storage, read> topo: array<u32>;
@group(0) @binding(6) var<storage, read> restN: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> nodeN: array<vec4f>;
// Level of detail: per car its levels' sizes and switch distance (lod), the
// eye (view, from the last draw), and the indirect args the pick writes: per
// level a dispatch (3 words) and two indexed draws (opaque, glass; 5 words).
struct Lod { n0: u32, n1: u32, opq0: u32, gl0: u32, opq1: u32, gl1: u32, levels: u32, dist: f32 };
@group(1) @binding(0) var<uniform> lod: Lod;
@group(1) @binding(1) var<storage, read_write> args: array<u32>;
@group(1) @binding(2) var<uniform> view: vec4f;

fn origin() -> vec3f {
  let o = misc[params.vehicle];
  return vec3f(bitcast<f32>(o.x), bitcast<f32>(o.y), bitcast<f32>(o.z));
}
fn nodePos(i: u32) -> vec3f { return nodes[(params.nodeOffset + i) * 3u].xyz; }

@compute @workgroup_size(${WG})
fn nodeNormals(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.nodeCount) { return; }
  var acc = vec3f(0.0);
  var stretch = 0.0;
  for (var k = topo[i]; k < topo[i + 1u]; k++) {
    let t = params.trisAt + 4u * topo[params.listAt + k];
    let a = nodePos(topo[t]);
    let b = nodePos(topo[t + 1u]);
    let c = nodePos(topo[t + 2u]);
    acc += cross(b - a, c - a);
    // Tear: the triangle's perimeter against its rest perimeter (word 3).
    let rest = bitcast<f32>(topo[t + 3u]);
    if (rest > 0.0) { stretch = max(stretch, (length(b - a) + length(c - b) + length(a - c)) / rest); }
  }
  let l = length(acc);
  // w: how torn the node's skin is (0 whole, 1 torn), smooth across nodes so a
  // tear cuts the panel along a smooth line rather than the mesh's grid.
  nodeN[i] = vec4f(select(vec3f(0.0), acc / l, l > 1e-12), smoothstep(1.35, 1.8, stretch));
}


// Pick the car's level from its distance (its first node) to the eye.
@compute @workgroup_size(1)
fn pick() {
  var sel = 0u;
  if (lod.levels > 1u && view.w > 0.5 && distance(nodePos(0u) + origin(), view.xyz) > lod.dist) { sel = 1u; }
  let ns = array<u32, 2>(lod.n0, lod.n1);
  let opq = array<u32, 2>(lod.opq0, lod.opq1);
  let gl = array<u32, 2>(lod.gl0, lod.gl1);
  for (var k = 0u; k < 2u; k++) {
    let on = select(0u, 1u, k == sel);
    let b = k * 13u;
    args[b] = on * ((ns[k] + ${WG - 1}u) / ${WG}u); args[b + 1u] = 1u; args[b + 2u] = 1u;
    args[b + 3u] = opq[k]; args[b + 4u] = on; args[b + 5u] = 0u; args[b + 6u] = 0u; args[b + 7u] = 0u;
    args[b + 8u] = gl[k]; args[b + 9u] = on; args[b + 10u] = opq[k]; args[b + 11u] = 0u; args[b + 12u] = 0u;
  }
}

@compute @workgroup_size(${WG})
fn skin(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.count) { return; }
  let bd = binds[i];
  let org = origin();
  let p0 = nodePos(bd.n0) + org;
  let e1 = nodePos(bd.n1) - nodePos(bd.n0);
  let e2 = nodePos(bd.n2) - nodePos(bd.n0);
  let c = cross(e1, e2);
  let cl = length(c);
  let n = select(vec3f(0.0, 1.0, 0.0), c / cl, cl > 1e-12);
  let l1 = length(e1);
  let l2 = length(e2);
  let t = e1 / max(l1, 1e-9);
  let b = cross(n, t);
  // Strain of the binding triangle (stretch or squash of either edge).
  let s1 = l1 / max(bd.r1, 1e-6);
  let s2 = l2 / max(bd.r2, 1e-6);
  let strain = max(max(s1, 1.0 / max(s1, 1e-6)), max(s2, 1.0 / max(s2, 1e-6))) - 1.0;
  let torn = s1 > 2.5 || s2 > 2.5 || cl <= 1e-12;
  let rigid = (bd.flags & 2u) != 0u;
  var pos: vec3f;
  let t0 = nodeN[bd.n0].w; let t1 = nodeN[bd.n1].w; let t2 = nodeN[bd.n2].w;
  var tear: f32;
  if (rigid) {
    // Rigid parts (mirrors, handles, lamps): the rest shape in the triangle's
    // orthonormal frame, never stretched; they go whole.
    pos = p0 + t * bd.a + b * bd.b + n * bd.h;
    tear = max(t0, max(t1, t2));
    // Mounted: where its mount (a skin triangle) has moved more than 4 cm
    // from where the part holds this vertex, the part breaks here.
    let mount = bd.flags >> 2u;
    if (mount != 0u) {
      let mt = params.trisAt + 4u * (mount - 1u);
      let q0 = nodePos(topo[mt]);
      let q1 = nodePos(topo[mt + 1u]);
      let q2 = nodePos(topo[mt + 2u]);
      let qc = cross(q1 - q0, q2 - q0);
      let qn = qc / max(length(qc), 1e-12);
      let w = q0 + org + bd.rx * (q1 - q0) + bd.ry * (q2 - q0) + bd.rz * qn;
      if (distance(w, pos) > 0.04) { tear = 1.0; }
    }
  } else {
    pos = p0 + bd.a * e1 + bd.b * e2 + bd.h * n;
    // The tear field interpolated over the binding nodes: the fragment stage
    // cuts where it crosses 0.5, a smooth line through the panel.
    tear = (1.0 - bd.a - bd.b) * t0 + bd.a * t1 + bd.b * t2;
  }
  tear = select(clamp(tear, 0.0, 1.0), 1.0, torn);
  // The rest normal carried by the triangle's frame (faceted: keeps kinks).
  let nf = normalize(bd.nt * t + bd.nb * b + bd.nn * n + vec3f(1e-9));
  var nrm = nf;
  if ((bd.flags & 1u) != 0u && !rigid) {
    // Smooth: the rest normal moved by the binding nodes' normal changes. Where
    // it disagrees with the frame's by more than ~15°, the panel has kinked
    // here: keep the faceted normal so the crease reads sharp.
    let w0 = 1.0 - bd.a - bd.b;
    let ns = normalize(vec3f(bd.rx, bd.ry, bd.rz)
      + w0 * (nodeN[bd.n0].xyz - restN[bd.n0].xyz)
      + bd.a * (nodeN[bd.n1].xyz - restN[bd.n1].xyz)
      + bd.b * (nodeN[bd.n2].xyz - restN[bd.n2].xyz) + vec3f(1e-9));
    nrm = normalize(mix(ns, nf, 1.0 - smoothstep(0.90, 0.97, dot(ns, nf))));
  }
  let k = i * 2u;
  outv[k] = vec4f(pos, tear);
  outv[k + 1u] = vec4f(nrm, strain);
}
`;

// The same skinning from world-space node positions (xyz f32 per node, every
// vehicle's nodes in order): the crash lab's gathered buffer, live or replayed
// (crash-lab-render.mjs positionsBuffer). Only the node fetch differs; binding 1
// (misc) is bound but unused (positions are already in world space).
const SKIN_POS_WGSL = (() => {
  let s = SKIN_WGSL;
  const swap = (a, b) => { if (!s.includes(a)) throw new Error('showcase skinPos: missing ' + a); s = s.replace(a, b); };
  swap('@group(0) @binding(0) var<storage, read> nodes: array<vec4f>;', '@group(0) @binding(0) var<storage, read> pos: array<f32>;');
  swap('fn nodePos(i: u32) -> vec3f { return nodes[(params.nodeOffset + i) * 3u].xyz; }',
    'fn nodePos(i: u32) -> vec3f { let k = (params.nodeOffset + i) * 3u; return vec3f(pos[k], pos[k + 1u], pos[k + 2u]); }');
  s = s.replace(/fn origin\(\) -> vec3f \{[\s\S]*?\n\}\n/, 'fn origin() -> vec3f { return vec3f(0.0); }\n');
  return s;
})();

const DRAW_WGSL = /* wgsl */ `
struct Frame { viewProj: mat4x4f, eye: vec4f, light: vec4f };
@group(0) @binding(0) var<uniform> frame: Frame;

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) color: vec3f,
  @location(3) @interpolate(flat) mat: u32,
  @location(4) torn: f32,
  @location(5) ao: f32,
  @location(6) strain: f32,
  @location(7) rest: vec3f,
  @location(8) uv: vec2f,
};

@vertex
fn vs(@location(0) pt: vec4f, @location(1) nrm: vec4f, @location(2) color: vec3f, @location(3) mat: u32, @location(4) ao: f32, @location(5) rest: vec3f, @location(6) uv: vec2f) -> VOut {
  var o: VOut;
  o.clip = frame.viewProj * vec4f(pt.xyz, 1.0);
  o.world = pt.xyz;
  o.normal = nrm.xyz;
  o.color = color;
  o.mat = mat;
  o.torn = pt.w;
  o.ao = ao;
  o.strain = nrm.w;
  o.rest = rest;
  o.uv = uv;
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
// The shaded point (world): env() reflects the ground plane (y = 0) from it.
var<private> gP: vec3f;
fn sky(d: vec3f, rough: f32) -> vec3f {
  let w = 0.012 + rough * 1.1;
  let up = max(d.y, 0.0);
  var c = mix(HORIZON, CEILING, smoothstep(0.0, 0.75, up));
  c = c + vec3f(0.4) * exp(-up * (22.0 / (1.0 + rough * 6.0)));   // horizon glow
  // Three long strips overhead, along the car.
  for (var k = -1; k <= 1; k++) {
    let xo = f32(k) * 0.55;
    let cc = normalize(vec3f(xo, 1.0, 0.0));
    let rr = normalize(vec3f(1.0, -xo, 0.0));
    c += vec3f(3.6, 3.55, 3.4) * softbox(d, cc, rr, vec3f(0.0, 0.0, 1.0), 0.07, 1.4, w);
  }
  // Side strips (long, horizontal, a little above the horizon).
  let sA = normalize(vec3f(1.0, 0.32, 0.15));
  let sB = normalize(vec3f(-1.0, 0.32, -0.2));
  c += vec3f(2.8) * softbox(d, sA, normalize(cross(vec3f(0.0, 1.0, 0.0), sA)), normalize(cross(sA, cross(vec3f(0.0, 1.0, 0.0), sA))), 1.7, 0.05, w);
  c += vec3f(2.4) * softbox(d, sB, normalize(cross(vec3f(0.0, 1.0, 0.0), sB)), normalize(cross(sB, cross(vec3f(0.0, 1.0, 0.0), sB))), 1.7, 0.05, w);
  // Key box in front, high.
  let k = normalize(vec3f(0.35, 0.55, 1.0));
  c += vec3f(2.6, 2.5, 2.35) * softbox(d, k, normalize(cross(vec3f(0.0, 1.0, 0.0), k)), normalize(cross(k, cross(vec3f(0.0, 1.0, 0.0), k))), 0.30, 0.20, w);
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
const TYRE_TEXT = array<u32, 31>(0u, 1u, 2u, 2u, 3u, 4u, 20u, 20u, 5u, 6u, 7u, 0u, 8u, 20u, 20u, 20u, 11u, 13u, 14u, 19u, 13u, 9u, 20u, 0u, 10u, 18u, 20u, 20u, 20u, 20u, 20u);
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

// Per material: rough, metal (F0 scale), clearcoat.
fn props(m: u32) -> vec3f {
  switch m {
    case 0u: { return vec3f(0.38, 0.0, 1.0); }   // paint (base under clearcoat)
    case 2u: { return vec3f(0.04, 1.0, 0.0); }   // chrome
    case 3u: { return vec3f(0.16, 0.0, 0.0); }   // trim: gloss black plastic
    case 6u: { return vec3f(0.62, 0.0, 0.0); }   // tyre (satin sidewall)
    case 7u: { return vec3f(0.32, 1.0, 0.0); }   // rim (gunmetal)
    case 28u: { return vec3f(0.1, 1.0, 0.0); }   // machined spoke faces
    case 8u: { return vec3f(0.5, 0.0, 0.0); }    // a shut line's wall
    case 9u: { return vec3f(0.95, 0.0, 0.0); }   // underbody
    case 10u: { return vec3f(0.45, 0.0, 0.0); }  // plate
    case 11u: { return vec3f(0.55, 0.0, 0.0); }  // grille back
    case 19u: { return vec3f(0.12, 1.0, 0.0); }  // reflector
    case 22u: { return vec3f(0.45, 1.0, 0.0); }  // brake disc / barrel
    case 23u: { return vec3f(0.3, 0.0, 0.6); }   // caliper
    case 24u: { return vec3f(0.2, 0.0, 0.5); }   // tail lamp back
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
  // Texture-coordinate footprint (taken in uniform control flow): printed
  // text fades to its average ink when too small to read.
  let uvw = fwidth(i.uv);
  // Tears: the far half of a torn triangle is cut away (through the triangle,
  // not along the grid); a band at the cut shows the bare metal edge.
  if (i.torn > 0.5) { discard; }
  gP = i.world;
  let glassy = i.mat == 1u || i.mat == 25u || i.mat == 27u;
  // Side and rear glass (tempered) shatters and drops once its frame strains.
  if ((i.mat == 1u || i.mat == 27u) && i.strain > 0.12) { discard; }
  if (i.torn > 0.3 && !glassy) {
    let nn = normalize(i.normal);
    let lit = 0.35 + 0.65 * max(dot(nn, normalize(frame.light.xyz)), 0.0);
    return vec4f(tonemap(vec3f(0.42, 0.42, 0.44) * lit), 1.0);
  }
  var n = normalize(i.normal);
  let v = normalize(frame.eye.xyz - i.world);
  // The inside of a body panel (seen through a window or a tear): bare, dark.
  if (!front && i.mat == 0u) {
    let d = vec3f(0.035, 0.035, 0.038) * (0.4 + 0.6 * clamp(i.ao, 0.0, 1.0));
    return vec4f(tonemap(d * (irradiance(-n) + vec3f(0.2))), 1.0);
  }
  // Two-sided: light the side facing the camera.
  if (dot(n, v) < 0.0) { n = -n; }
  let l = normalize(frame.light.xyz);
  var ao = clamp(i.ao, 0.0, 1.0);
  // Wheels (they turn, so nothing is baked): the arch shades their tops and
  // the car's body their inner faces.
  if (i.mat == 6u || i.mat == 7u || i.mat == 22u || i.mat == 23u || i.mat == 28u) {
    ao *= mix(1.0, 0.42, smoothstep(0.30, 0.66, i.world.y));
  }
  let ndl = max(dot(n, l), 0.0);
  let ndv = max(dot(n, v), 1e-3);
  let h = normalize(v + l);
  let ndh = max(dot(n, h), 0.0);
  let r = reflect(-v, n);
  let sun = vec3f(1.0, 0.96, 0.9) * 1.6;
  let fres = pow(1.0 - ndv, 5.0);
  let albedo = i.color;
  let m = i.mat;
  // Emissive lamps.
  if (m == 18u) { return vec4f(tonemap(albedo * 6.0 + env(r, 0.05) * 0.04), 1.0); }
  if (m == 21u) { return vec4f(tonemap(albedo * vec3f(2.6, 0.5, 0.5)), 1.0); }
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
      // Laminated windscreen: spider cracks grow with the frame's strain.
      let amt = smoothstep(0.02, 0.15, i.strain);
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
    let sheen = pow(1.0 - ndv, 3.0) * 0.08 + sun.x * ggx(ndh, 0.25) * ndl * 0.02;
    let cab = albedo * (irradiance(n) * 0.55 * ao + sun * ndl / 3.14159 * 0.35 * ao) + vec3f(sheen) * ao;
    return vec4f(tonemap(cab), 1.0);
  }
  var pr = props(m);
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
    let cf = 0.04 + 0.96 * fres;
    // Flop: the colour deepens toward grazing (the flakes face away).
    let flop = mix(1.0, 0.45, pow(1.0 - ndv, 1.5));
    let diffuse = albedo * flop * (irradiance(n) * ao + sun * ndl / 3.14159);
    let flakeF0 = albedo * 0.55 + vec3f(0.03);
    // The flakes' rough reflection: the blurred studio (the analytic lobe, not
    // a second environment lookup).
    let flake = flakeF0 * (irradiance(r) * 1.7 * ao * 0.9 + sun * ggx(ndh, 0.17) * g / max(4.0 * ndv, 1e-3));
    // Sparkle: tiny flakes catching the sun.
    let cell = floor(i.world * 900.0);
    let sp = step(0.992, hash3(cell)) * pow(ndh, 40.0) * ndl * 2.5;
    let base = diffuse * 0.75 + flake + albedo * sp;
    // Clearcoat: a mirror-sharp sun glint over a broader sheen, and the
    // studio reflected crisply (horizon, softboxes, the floor).
    let coatSpec = sun * (ggx(ndh, 0.0016) * 0.22 + ggx(ndh, 0.02) * 0.06) * cf / max(4.0 * ndv, 1e-3) * ndl;
    c = base * (1.0 - cf) + env(r, 0.005) * cf * mix(0.45, 1.0, ao) + coatSpec;
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
 * `opts`: `{ format, depthFormat = 'depth24plus', sampleCount = 1, cullMode = 'none' }`.
 */
export function createShowcaseCarRenderer(device, opts = {}) {
  const { format, depthFormat = 'depth24plus', sampleCount = 1, cullMode = 'none' } = opts;
  const skinModule = device.createShaderModule({ label: 'showcase-skin', code: SKIN_WGSL });
  const drawModule = device.createShaderModule({ label: 'showcase-draw', code: DRAW_WGSL });
  const skinLayout = device.createBindGroupLayout({
    entries: [0, 1, 2, 3, 4, 5, 6, 7].map((binding) => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: binding === 4 ? 'uniform' : binding === 3 || binding === 7 ? 'storage' : 'read-only-storage' },
    })),
  });
  const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: [skinLayout] });
  const skinPipe = device.createComputePipeline({ label: 'showcase-skin', layout: pipeLayout, compute: { module: skinModule, entryPoint: 'skin' } });
  const normalPipe = device.createComputePipeline({ label: 'showcase-node-normals', layout: pipeLayout, compute: { module: skinModule, entryPoint: 'nodeNormals' } });
  // The pick also writes the indirect args (group 1), so the skin dispatches,
  // which read them as indirect args, must not bind them.
  const pickLayout = device.createBindGroupLayout({
    entries: [0, 1, 2].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: binding === 1 ? 'storage' : 'uniform' } })),
  });
  const pickPipeLayout = device.createPipelineLayout({ bindGroupLayouts: [skinLayout, pickLayout] });
  const pickPipe = device.createComputePipeline({ label: 'showcase-lod-pick', layout: pickPipeLayout, compute: { module: skinModule, entryPoint: 'pick' } });
  // The eye of the last draw (w = 1 once set): picks levels of detail.
  const viewBuf = device.createBuffer({ label: 'showcase-view', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const drawPipe = device.createRenderPipeline({
    label: 'showcase-draw',
    layout: 'auto',
    vertex: {
      module: drawModule,
      entryPoint: 'vs',
      buffers: [
        { arrayStride: 32, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }, { shaderLocation: 1, offset: 16, format: 'float32x4' }] },
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
        { arrayStride: 32, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }, { shaderLocation: 1, offset: 16, format: 'float32x4' }] },
        { arrayStride: 40, attributes: [{ shaderLocation: 2, offset: 0, format: 'float32x3' }, { shaderLocation: 4, offset: 12, format: 'float32' }, { shaderLocation: 3, offset: 16, format: 'uint32' }, { shaderLocation: 5, offset: 20, format: 'float32x3' }, { shaderLocation: 6, offset: 32, format: 'float32x2' }] },
      ],
    },
    fragment: { module: drawModule, entryPoint: 'fs', targets: [{ format, blend: { color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } } }] },
    primitive: { topology: 'triangle-list', cullMode },
    depthStencil: { format: depthFormat, depthWriteEnabled: false, depthCompare: 'less' },
    multisample: { count: sampleCount },
  });
  const frameBuf = device.createBuffer({ label: 'showcase-frame', size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const drawBind = device.createBindGroup({ layout: drawPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: frameBuf } }] });
  const glassBind = device.createBindGroup({ layout: glassPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: frameBuf } }] });
  const cars = new Map();
  let nextId = 1;
  let nodesBuf = null;
  let miscBuf = null;
  let posBuf = null, posPipes = null;
  const bindFor = (car, lv, src0, src1) => device.createBindGroup({
    layout: skinLayout,
    entries: [src0, src1, lv.binds, lv.out, lv.params, car.topo, car.restN, car.nodeN]
      .map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const makePosBind = (car) => posBuf && car.levels.map((lv) => bindFor(car, lv, posBuf, posBuf));
  const makeSkinBind = (car) => car.levels.map((lv) => bindFor(car, lv, nodesBuf, miscBuf));
  const upload = (label, data, usage = GPUBufferUsage.STORAGE) => {
    const b = device.createBuffer({ label, size: Math.max(16, Math.ceil(data.byteLength / 4) * 4), usage: usage | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, data);
    return b;
  };

  return {
    /**
     * Skin from world-space node positions instead (xyz f32 per node, all vehicles in
     * order; e.g. the crash lab's positionsBuffer(), so replays deform the body too).
     * Call again when that buffer is replaced; null returns to the solver buffers.
     */
    setPositions(buf) {
      posBuf = buf;
      if (buf && !posPipes) {
        const m = device.createShaderModule({ label: 'showcase-skin-pos', code: SKIN_POS_WGSL });
        posPipes = {
          pick: device.createComputePipeline({ label: 'showcase-lod-pick-pos', layout: pickPipeLayout, compute: { module: m, entryPoint: 'pick' } }),
          skin: device.createComputePipeline({ label: 'showcase-skin-pos', layout: pipeLayout, compute: { module: m, entryPoint: 'skin' } }),
          normal: device.createComputePipeline({ label: 'showcase-node-normals-pos', layout: pipeLayout, compute: { module: m, entryPoint: 'nodeNormals' } }),
        };
      }
      for (const car of cars.values()) car.posBind = makePosBind(car);
    },
    /** The explicit solver's node and misc buffers (call again after a rebuild). */
    setSolver(nodes, misc) {
      nodesBuf = nodes;
      miscBuf = misc;
      for (const car of cars.values()) car.skinBind = makeSkinBind(car);
    },
    /**
     * Add a car: `visual` from the wasm (`showcaseCar(...).visual`), `nodeOffset` its
     * vehicle's first node in the solver (`layout.vehicles[i].nodeOffset`), `vehicle`
     * its index (for its origin). Returns an id.
     */
    addCar(visual, { nodeOffset = 0, vehicle = 0, lod = null, lodDistance = 18 } = {}) {
      const nodeCount = visual.nodeCount;
      const nt = visual.normalTris, start = visual.nodeTriStart, list = visual.nodeTriList;
      const listAt = start.length, trisAt = listAt + list.length;
      const topo = new Uint32Array(trisAt + (nt.length / 3) * 4);
      topo.set(start, 0);
      topo.set(list, listAt);
      const nr = visual.nodeRest;
      const tf = new Float32Array(topo.buffer);
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
      const rn = new Float32Array(nodeCount * 4);
      for (let i = 0; i < nodeCount; i++) rn.set(visual.nodeRestNormal.subarray(3 * i, 3 * i + 3), 4 * i);
      const level = (v) => {
        const n = v.vertexCount;
        const bindData = new ArrayBuffer(n * 64);
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
          u[o + 15] = (v.smooth ? v.smooth[i] : 0) | (v.rigid && v.rigid[i] ? 2 : 0);
          // A mounted rigid part's mount: (skin triangle + 1) in the flags'
          // upper bits, its (α, β, h) in the (unused) rest-normal words.
          if (v.rigid && v.rigid[i] && v.anchor && v.anchor[i] !== 0xffffffff) {
            u[o + 15] |= (v.anchor[i] + 1) << 2;
            f[o + 12] = v.anchorCoords[3 * i];
            f[o + 13] = v.anchorCoords[3 * i + 1];
            f[o + 14] = v.anchorCoords[3 * i + 2];
          }
        }
        const binds = upload('showcase-binds', bindData);
        // Deformed (position | torn, normal | strain) per vertex: the compute pass
        // writes it, the draw reads it as a vertex buffer. Starts at the rest pose.
        const out = device.createBuffer({ label: 'showcase-out', size: n * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
        const rest = new Float32Array(n * 8);
        for (let i = 0; i < n; i++) {
          rest.set(v.positions.subarray(3 * i, 3 * i + 3), 8 * i);
          rest.set(v.normals.subarray(3 * i, 3 * i + 3), 8 * i + 4);
        }
        device.queue.writeBuffer(out, 0, rest);
        // Per vertex: colour, AO, material, rest position (the glass's crack
        // pattern is drawn in rest space so it rides the glass).
        // and texture coordinates (printed parts: tyres, plates, displays).
        const attr = new ArrayBuffer(n * 40);
        const af = new Float32Array(attr);
        const au = new Uint32Array(attr);
        for (let i = 0; i < n; i++) {
          af.set(v.colors.subarray(3 * i, 3 * i + 3), 10 * i);
          af[10 * i + 3] = v.ao ? v.ao[i] : 1;
          au[10 * i + 4] = v.material[i];
          af.set(v.positions.subarray(3 * i, 3 * i + 3), 10 * i + 5);
          if (v.uvs) af.set(v.uvs.subarray(2 * i, 2 * i + 2), 10 * i + 8);
        }
        // Opaque triangles first, the blended ones (glass, lenses) after.
        const tri = v.indices;
        const TRANSPARENT = new Set([1, 17, 20, 25, 27]);
        let no = 0;
        for (let t = 0; t < tri.length; t += 3) if (!TRANSPARENT.has(v.material[tri[t]])) no += 3;
        const ordered = new Uint32Array(tri.length);
        let io = 0, ig = no;
        for (let t = 0; t < tri.length; t += 3) {
          if (TRANSPARENT.has(v.material[tri[t]])) { ordered[ig++] = tri[t]; ordered[ig++] = tri[t + 1]; ordered[ig++] = tri[t + 2]; }
          else { ordered[io++] = tri[t]; ordered[io++] = tri[t + 1]; ordered[io++] = tri[t + 2]; }
        }
        const attrs = device.createBuffer({ label: 'showcase-attrs', size: attr.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(attrs, 0, attr);
        const index = device.createBuffer({ label: 'showcase-index', size: ordered.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(index, 0, ordered);
        const params = device.createBuffer({ label: 'showcase-params', size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(params, 0, new Uint32Array([n, nodeOffset, nodeCount, vehicle, listAt, trisAt, 0, 0]));
        return { n, binds, out, attrs, index, params, opaqueCount: no, glassCount: tri.length - no };
      };
      const levels = [level(visual)];
      if (lod) levels.push(level(lod));
      const lodData = new ArrayBuffer(32);
      const lu = new Uint32Array(lodData);
      const l1 = levels[1] ?? levels[0];
      lu.set([levels[0].n, l1.n, levels[0].opaqueCount, levels[0].glassCount, l1.opaqueCount, l1.glassCount, levels.length]);
      new Float32Array(lodData)[7] = lodDistance;
      const car = {
        nodeCount, vehicle, nodeOffset, levels, listAt, trisAt, skinBind: null, posBind: null,
        topo: upload('showcase-topology', topo),
        restN: upload('showcase-rest-normals', rn),
        nodeN: device.createBuffer({ label: 'showcase-node-normals', size: nodeCount * 16, usage: GPUBufferUsage.STORAGE }),
        lod: upload('showcase-lod', lodData, GPUBufferUsage.UNIFORM),
        // Indirect args (2 levels × 13 words); level 0 drawn until the first pick.
        args: upload('showcase-lod-args', new Uint32Array([
          Math.ceil(levels[0].n / WG), 1, 1, levels[0].opaqueCount, 1, 0, 0, 0, levels[0].glassCount, 1, levels[0].opaqueCount, 0, 0,
          0, 1, 1, l1.opaqueCount, 0, 0, 0, 0, l1.glassCount, 0, l1.opaqueCount, 0, 0,
        ]), GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT),
      };
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
    setNodeOffset(id, nodeOffset) {
      const car = cars.get(id);
      if (!car) return;
      car.nodeOffset = nodeOffset;
      for (const lv of car.levels) device.queue.writeBuffer(lv.params, 0, new Uint32Array([lv.n, nodeOffset, car.nodeCount, car.vehicle, car.listAt, car.trisAt, 0, 0]));
    },
    removeCar(id) {
      const car = cars.get(id);
      if (!car) return;
      for (const lv of car.levels) for (const b of [lv.binds, lv.out, lv.attrs, lv.index, lv.params]) b.destroy();
      for (const b of [car.topo, car.restN, car.nodeN, car.lod, car.args]) b.destroy();
      cars.delete(id);
    },
    /**
     * Deform every car from the solver's current node positions (or the positions
     * buffer given to setPositions): node normals, the level-of-detail pick (by
     * the last draw's eye), then the picked level's skin.
     */
    update(encoder) {
      const usePos = !!posBuf;
      if (!usePos && !nodesBuf) return;
      const pipes = usePos ? posPipes : { normal: normalPipe, skin: skinPipe, pick: pickPipe };
      const pass = encoder.beginComputePass({ label: usePos ? 'showcase-skin-pos' : 'showcase-skin' });
      for (const car of cars.values()) {
        const groups = usePos ? car.posBind : car.skinBind;
        pass.setBindGroup(0, groups[0]);
        pass.setPipeline(pipes.normal);
        pass.dispatchWorkgroups(Math.ceil(car.nodeCount / WG));
        pass.setPipeline(pipes.pick);
        pass.setBindGroup(1, car.pickBind);
        pass.dispatchWorkgroups(1);
        pass.setPipeline(pipes.skin);
        car.levels.forEach((lv, k) => {
          pass.setBindGroup(0, groups[k]);
          pass.dispatchWorkgroupsIndirect(car.args, k * 52);
        });
      }
      pass.end();
    },
    /** Draw every car into `pass` (a render pass with a colour target of `format` and depth). */
    draw(pass, { viewProj, eye, light = [0.4, 0.8, 0.35] }) {
      const fr = new Float32Array(24);
      fr.set(viewProj, 0);
      fr.set([eye[0], eye[1], eye[2], 1], 16);
      fr.set([light[0], light[1], light[2], 0], 20);
      device.queue.writeBuffer(frameBuf, 0, fr);
      device.queue.writeBuffer(viewBuf, 0, new Float32Array([eye[0], eye[1], eye[2], 1]));
      for (const [pipe, bind, at] of [[drawPipe, drawBind, 12], [glassPipe, glassBind, 32]]) {
        pass.setPipeline(pipe);
        pass.setBindGroup(0, bind);
        for (const car of cars.values()) {
          car.levels.forEach((lv, k) => {
            pass.setVertexBuffer(0, lv.out);
            pass.setVertexBuffer(1, lv.attrs);
            pass.setIndexBuffer(lv.index, 'uint32');
            pass.drawIndexedIndirect(car.args, k * 52 + at);
          });
        }
      }
    },
    destroy() {
      for (const id of [...cars.keys()]) this.removeCar(id);
      frameBuf.destroy();
      viewBuf.destroy();
    },
  };
}
