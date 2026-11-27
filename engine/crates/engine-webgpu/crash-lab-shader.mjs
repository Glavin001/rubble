// The crash lab page's scene shader (crash-lab.html, crash-lab-draw.mjs): the sky, the asphalt
// ground, the World's colliders (the sim's static shapes drawn, the raycast wheels' stand-in
// boxes) and, in the plain views (arena, traffic, ?lab=0), the explicit solver's collision
// skin and beams read straight from its buffers. The lab draws its vehicles itself
// (crash-lab-render.mjs). One WGSL module; the shadow pass uses it with the light's matrix.
// (softbody.html, frozen, keeps its own copy with the soft solver's passes.)

import { LIGHTING_WGSL } from './car-render.mjs';
import { WHEEL_RIM_MATERIAL } from './vehicle.mjs';

export const SCENE_WGSL = /* wgsl */`
struct Body { position: vec3f, invMass: f32, orientation: vec4f, invInertia: vec3f, flags: u32,
  linearVel: vec3f, sleepTimer: f32, angularVel: vec3f, generation: u32, initialPos: vec3f, initialPad: f32, initialRot: vec4f };
struct Collider { body: u32, shape: u32, material: u32, generation: u32, localPos: vec3f, density: f32, localRot: vec4f, params: vec4f };
${LIGHTING_WGSL}
@group(0) @binding(0) var<storage, read> bodies: array<Body>;
@group(0) @binding(2) var<storage, read> cols: array<Collider>;
fn qrot(q: vec4f, v: vec3f) -> vec3f {
  let u = q.xyz;
  return u * (2.0 * dot(u, v)) + v * (q.w * q.w - dot(u, u)) + cross(u, v) * (2.0 * q.w);
}
fn hash(n: u32) -> f32 { var x = n * 747796405u + 2891336453u; x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u; return f32((x >> 22u) ^ x) / 4294967295.0; }
fn shadeR(c: vec3f, n: vec3f, w: vec3f, rough: f32, metal: f32) -> vec3f { return tonemap(surface(toLinear(c), n, w, rough, metal, 0.0)); }
fn shade(c: vec3f, n: vec3f, w: vec3f) -> vec3f { return shadeR(c, n, w, 0.6, 0.0); }
// Sky: a full-screen triangle at the far plane.
struct SkyOut { @builtin(position) clip: vec4f, @location(0) ndc: vec2f };
@vertex fn vsSky(@builtin(vertex_index) vi: u32) -> SkyOut {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u)) * 2.0 - 1.0;
  var o: SkyOut; o.clip = vec4f(p, 0.9999, 1.0); o.ndc = p; return o;
}
@fragment fn fsSky(i: SkyOut) -> @location(0) vec4f {
  let far = frame.invViewProj * vec4f(i.ndc, 1.0, 1.0);
  let d = normalize(far.xyz / far.w - frame.eye.xyz);
  return vec4f(tonemap(skyColor(d)), 1.0);
}

struct Out { @builtin(position) clip: vec4f, @location(0) n: vec3f, @location(1) c: vec3f, @location(2) w: vec3f, @location(3) center: vec3f, @location(4) r: f32 };
const HIDE = vec4f(0.0, 0.0, 2.0, 1.0);

@vertex fn vsCol(@location(0) corner: vec3f, @location(1) normal: vec3f, @builtin(instance_index) inst: u32) -> Out {
  let ci = colIdx[inst];
  let col = cols[ci];
  let b = bodies[col.body];
  var o: Out;
  o.r = 0.0;
  // Hidden: wheel rims, parked bodies, soft-body node spheres and rigid cores (negative group).
  let group = i32(col.material) >> 16;
  let hidden = col.material == ${WHEEL_RIM_MATERIAL}u || b.position.y < -1000.0 || group < 0;
  var half = col.params.xyz;
  if (col.shape == 0u) { half = vec3f(col.params.x); o.r = col.params.x; }
  if (col.shape == 1u) { half = vec3f(col.params.x, col.params.y + col.params.x, col.params.x); }
  let q = vec4f(b.orientation);
  let lq = col.localRot;
  let local = col.localPos + qrot(lq, corner * half * 2.0);
  let w = b.position + qrot(q, local);
  o.w = w;
  o.center = b.position + qrot(q, col.localPos);
  o.n = qrot(q, qrot(lq, normal));
  o.clip = select(frame.viewProj * vec4f(w, 1.0), HIDE, hidden);
  let dyn = b.invMass > 0.0 && (b.flags & 1u) != 0u;
  if (!dyn) {
    o.c = select(vec3f(0.55, 0.56, 0.6), vec3f(0.85, 0.75, 0.2), col.shape == 1u);
    // The crash lab's studio (opts.y): dark slate obstacles.
    if (frame.opts.y > 0.5) { o.c = vec3f(0.2, 0.24, 0.32); }
  } else if (col.shape == 0u) {
    o.c = vec3f(0.2, 0.22, 0.25);
  } else {
    o.c = mix(vec3f(0.85, 0.55, 0.25), vec3f(0.9, 0.78, 0.4), hash(col.body));
  }
  return o;
}

@fragment fn fsCol(i: Out) -> @location(0) vec4f {
  var n = normalize(i.n);
  if (i.r > 0.0) {
    let rd = normalize(i.w - frame.eye.xyz);
    let oc = frame.eye.xyz - i.center;
    let b = dot(oc, rd);
    let h = b * b - (dot(oc, oc) - i.r * i.r);
    if (h < 0.0) { discard; }
    n = normalize(frame.eye.xyz + rd * (-b - sqrt(h)) - i.center);
  }
  return vec4f(shade(i.c, n, i.w), 1.0);
}
@group(0) @binding(11) var<storage, read> colIdx: array<u32>; // drawable colliders (vsCol instances)
@vertex fn vsGround(@location(0) p: vec3f) -> Out {
  var o: Out;
  o.w = p; o.n = vec3f(0.0, 1.0, 0.0); o.c = vec3f(0.3); o.center = p; o.r = 0.0;
  o.clip = frame.viewProj * vec4f(p, 1.0) + vec4f(0.0) * bodies[0].position.x;
  return o;
}
@fragment fn fsGround(i: Out) -> @location(0) vec4f {
  // Asphalt: grain at two scales, faint 8 m slab lines.
  let p = vec3f(i.w.x, 0.0, i.w.z);
  let grain = noise3(p * 9.0) * 0.5 + noise3(p * 1.3) * 0.35 + noise3(p * 0.15) * 0.15;
  let g = abs(fract(i.w.xz / 8.0) - 0.5);
  let line = smoothstep(0.485, 0.5, max(g.x, g.y));
  let base = mix(vec3f(0.2, 0.2, 0.21), vec3f(0.31, 0.31, 0.32), grain) * (1.0 - 0.25 * line);
  return vec4f(shadeR(base, vec3f(0.0, 1.0, 0.0), i.w, 0.85, 0.0), 1.0);
}
// The skin and beam colours (the plain views' collision skin and beams).
@fragment fn fsSkin(i: Out, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  let n = select(-normalize(i.n), normalize(i.n), front);
  return vec4f(shade(i.c, n, i.w), 1.0);
}
@fragment fn fsBeam(i: Out) -> @location(0) vec4f { return vec4f(i.c, 1.0); }

// Explicit solver (?solver=explicit): read straight from its buffers. Nodes are 3
// vec4 each (position relative to the vehicle's origin first); the origin is the
// vehicle's misc record. Triangles: (a, b, c, vehicle) then their spawn edges.
// Beams: records (a, b, vehicle | tyre << 16, spawn rest) and state ((rest, ..),
// (strength, force, broken, 0)).
@group(0) @binding(13) var<storage, read> xnodes: array<vec4f>;
@group(0) @binding(14) var<storage, read> xmisc: array<vec4u>;
@group(0) @binding(15) var<storage, read> xtris: array<vec4u>;
@group(0) @binding(16) var<storage, read> xrecs: array<vec4u>;
@group(0) @binding(17) var<storage, read> xbeams: array<vec4f>;
// frame.opts.z: the frame's lead (FramePacer.lead, simulated seconds past the last
// step): positions carried forward by their velocity (visual only).
fn xpos(i: u32, v: u32) -> vec3f {
  let p = xnodes[i * 3u].xyz + bitcast<vec3f>(xmisc[v].xyz);
  return select(p, p + xnodes[i * 3u + 1u].xyz * frame.opts.z, frame.opts.z > 0.0);
}
// The collision skin: panels go dark with strain, a triangle past 1.8× its spawn
// edges reads as torn and is dropped.
@vertex fn vsXSkin(@builtin(vertex_index) vi: u32) -> Out {
  let t = xtris[(vi / 3u) * 2u];
  let r = bitcast<vec4f>(xtris[(vi / 3u) * 2u + 1u]);
  let pa = xpos(t.x, t.w);
  let pb = xpos(t.y, t.w);
  let pc = xpos(t.z, t.w);
  let k = vi % 3u;
  let p = select(select(pc, pb, k == 1u), pa, k == 0u);
  let la = length(pa - pb) / max(r.x, 1e-4);
  let lb = length(pb - pc) / max(r.y, 1e-4);
  let lc = length(pc - pa) / max(r.z, 1e-4);
  let strain = max(max(abs(la - 1.0), abs(lb - 1.0)), abs(lc - 1.0));
  let torn = max(max(la, lb), lc) > 1.8;
  var o: Out;
  o.w = p; o.center = p; o.r = 0.0;
  o.n = normalize(cross(pb - pa, pc - pa));
  let base = mix(vec3f(0.95, 0.55, 0.12), vec3f(0.1, 0.62, 0.6), step(0.5, hash(t.w * 7u + 3u)));
  o.c = mix(base, vec3f(0.18, 0.16, 0.15), clamp(strain * 6.0, 0.0, 0.85));
  o.clip = select(frame.viewProj * vec4f(p, 1.0), HIDE, torn);
  return o;
}
// Beams: tyres always (dark); with the beam view every beam, green intact, yellow
// to red bent (rest length off its spawn length by 0.1 % to 8 %), magenta broken.
// Four vertices per beam: 0–1 the beam (2–3 hidden); a broken beam instead draws a
// short stub at each of its nodes (0–1 at one end, 2–3 at the other), since its
// two nodes may since have flown metres apart.
// One draw of 4 vertices per beam (not an instance per beam: millions of 4-vertex
// instances cost far more than the vertices themselves).
@vertex fn vsXBeam(@builtin(vertex_index) vall: u32) -> Out {
  let inst = vall / 4u;
  let vi = vall % 4u;
  let rec = xrecs[inst];
  let v = rec.z & 0xffffu;
  let tyre = (rec.z >> 16u) != 0u;
  let pa = xpos(rec.x, v);
  let pb = xpos(rec.y, v);
  let s0 = xbeams[inst * 2u];
  let s1 = xbeams[inst * 2u + 1u];
  let broken = s1.z != 0.0;
  let rest0 = bitcast<f32>(rec.w);
  let bend = abs(s0.x - rest0) / max(rest0, 1e-6);
  let q0 = select(pa, pb, vi >= 2u);
  let q1 = select(pb, pa, vi >= 2u);
  let stub = min(0.25 * rest0, 0.06) / max(length(q1 - q0), 1e-6);
  let far = select(q1, mix(q0, q1, min(stub, 1.0)), broken);
  let p = select(far, q0, (vi & 1u) == 0u);
  var o: Out;
  o.w = p; o.center = p; o.r = 0.0; o.n = vec3f(0.0, 1.0, 0.0);
  o.c = select(vec3f(0.15, 0.85, 0.35), vec3f(0.12, 0.12, 0.13), tyre && frame.opts.x < 0.5);
  if (bend > 1e-3) { o.c = mix(vec3f(1.0, 0.85, 0.1), vec3f(1.0, 0.1, 0.05), clamp(bend / 0.08, 0.0, 1.0)); }
  if (broken) { o.c = vec3f(0.95, 0.1, 0.9); }
  let shown = select(tyre && !broken, true, frame.opts.x > 0.5) && (broken || vi < 2u);
  o.clip = select(HIDE, frame.viewProj * vec4f(p, 1.0), shown);
  return o;
}
`;
