// The one-World showcase's scene renderer (one-world.html): sky, ground and road, the
// World's rigid colliders (house chunks, crates, the slab, the wrecking ball and its
// rod) straight from the World's GPU buffers, lit by a sun with a shadow map that the
// skinned cars (showcase-car-render.mjs `drawDepth`) cast into too. Rendering only:
// every pose comes from the World's body buffer as the step left it, carried forward
// by `lead` seconds along each body's velocity (smooth slow motion between steps).
//
//   const env = createSceneRenderer(device, { format, depthFormat: 'depth32float' });
//   env.frame({ viewProj, invViewProj, eye, focus, ground, lead, time, road, rods })
//   env.shadowPass(encoder, world, (pass, lightVP) => cars.drawDepth(pass, { viewProj: lightVP }))
//   env.draw(pass, world)              // sky, ground, colliders, rods

const SHADOW = 2048;
// Triangles drawn per collider (a box 12; a hull its faces' fans).
const HULL_TRIS = 60;
// The shadow map's half width (m) about the focus.
const SHADOW_HALF = 26;

const COMMON = /* wgsl */ `
struct Body { position: vec3f, invMass: f32, orientation: vec4f, invInertia: vec3f, flags: u32,
  linearVel: vec3f, sleepTimer: f32, angularVel: vec3f, generation: u32, initialPos: vec3f, initialPad: f32, initialRot: vec4f };
struct Collider { body: u32, shape: u32, material: u32, generation: u32, localPos: vec3f, density: f32, localRot: vec4f, params: vec4f };
struct Frame {
  viewProj: mat4x4f, invViewProj: mat4x4f, lightVP: mat4x4f,
  eye: vec4f,       // xyz, time
  sun: vec4f,       // direction toward the sun, lead (s)
  ground: vec4f,    // y, road half width, road centre x, road length
  rod: vec4f,       // anchor xyz, ball body slot (bits; 0xffffffff none)
  focus: vec4f,     // xyz, fog distance
  cut: vec4f,       // the ghost cut's centre xyz, its radius about it (m; 0: no cut)
  cut2: vec4f,      // the cut's radius along the line to the eye (m), ghost opacity
};
@group(0) @binding(0) var<uniform> frame: Frame;
// The ghost cut (rendering only): a body whose centre is within cut.w of the cut's
// vertical axis (any height: what lies on or over the car), or within cut2.x of the
// line from the cut's centre to the eye (what stands between the camera and the car).
fn ghosted(p: vec3f) -> bool {
  if (frame.cut.w <= 0.0) { return false; }
  if (length(p.xz - frame.cut.xz) < frame.cut.w) { return true; }
  let a = frame.cut.xyz;
  let d = frame.eye.xyz - a;
  let t = clamp(dot(p - a, d) / max(dot(d, d), 1e-6), 0.0, 1.0);
  return length(p - (a + d * t)) < frame.cut2.x;
}

fn qrot(q: vec4f, v: vec3f) -> vec3f {
  let u = q.xyz;
  return u * (2.0 * dot(u, v)) + v * (q.w * q.w - dot(u, u)) + cross(u, v) * (2.0 * q.w);
}
fn qmul(a: vec4f, b: vec4f) -> vec4f {
  return vec4f(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}
fn hash(n: u32) -> f32 { var x = n * 747796405u + 2891336453u; x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u; return f32((x >> 22u) ^ x) / 4294967295.0; }
fn hash2(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453); }
fn noise(p: vec2f) -> f32 {
  let i = floor(p); let f = fract(p); let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash2(i), hash2(i + vec2f(1, 0)), u.x), mix(hash2(i + vec2f(0, 1)), hash2(i + vec2f(1, 1)), u.x), u.y);
}
fn fbm(p: vec2f) -> f32 { var s = 0.0; var a = 0.5; var q = p; for (var i = 0; i < 4; i++) { s += a * noise(q); q = q * 2.03 + 17.0; a *= 0.5; } return s; }
// A body's pose carried forward by the frame's lead along its velocity.
fn bodyPos(b: Body) -> vec3f { return b.position + b.linearVel * frame.sun.w; }
fn bodyRot(b: Body) -> vec4f {
  let w = b.angularVel * frame.sun.w;
  return normalize(b.orientation + 0.5 * qmul(vec4f(w, 0.0), b.orientation));
}
fn skyColor(d: vec3f) -> vec3f {
  let h = clamp(d.y, -1.0, 1.0);
  var c = mix(vec3f(0.78, 0.84, 0.9), vec3f(0.33, 0.52, 0.8), pow(max(h, 0.0), 0.55));
  c = mix(c, vec3f(0.55, 0.57, 0.58), smoothstep(0.0, -0.2, h));
  let s = max(dot(d, frame.sun.xyz), 0.0);
  c += vec3f(1.0, 0.85, 0.6) * (pow(s, 600.0) * 6.0 + pow(s, 12.0) * 0.18);
  return c;
}
fn fog(c: vec3f, p: vec3f) -> vec3f {
  let d = distance(p, frame.eye.xyz);
  let f = 1.0 - exp(-max(d - 30.0, 0.0) / frame.focus.w);
  return mix(c, skyColor(normalize(p - frame.eye.xyz)) * 0.95, clamp(f, 0.0, 0.85));
}
`;

const SHADE = /* wgsl */ `
@group(0) @binding(4) var shadowMap: texture_depth_2d;
@group(0) @binding(5) var shadowSamp: sampler_comparison;
fn shadow(p: vec3f, n: vec3f) -> f32 {
  let q = frame.lightVP * vec4f(p + n * 0.03, 1.0);
  let uv = q.xy * vec2f(0.5, -0.5) + 0.5;
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) || q.z > 1.0) { return 1.0; }
  let t = 1.0 / ${SHADOW}.0;
  var s = 0.0;
  for (var y = -1; y <= 1; y++) { for (var x = -1; x <= 1; x++) {
    s += textureSampleCompareLevel(shadowMap, shadowSamp, uv + vec2f(f32(x), f32(y)) * t * 1.2, q.z - 0.0015);
  } }
  return s / 9.0;
}
fn lit(albedo: vec3f, n: vec3f, p: vec3f, ao: f32, spec: f32) -> vec3f {
  let l = frame.sun.xyz;
  let sh = shadow(p, n);
  let diff = max(dot(n, l), 0.0) * sh;
  let sky = 0.5 + 0.5 * n.y;
  let v = normalize(frame.eye.xyz - p);
  let hv = normalize(l + v);
  let sp = pow(max(dot(n, hv), 0.0), 48.0) * spec * sh;
  let bounce = 0.25 + 0.2 * max(-n.y, 0.0);
  let c = albedo * (vec3f(1.0, 0.95, 0.86) * 1.9 * diff + vec3f(0.45, 0.52, 0.64) * 1.25 * sky * ao + vec3f(0.4, 0.36, 0.3) * bounce) + vec3f(1.0, 0.95, 0.85) * sp;
  return c;
}
`;

const COLLIDERS = /* wgsl */ `
${COMMON}
@group(0) @binding(1) var<storage, read> bodies: array<Body>;
@group(0) @binding(2) var<storage, read> cols: array<Collider>;
@group(0) @binding(3) var<storage, read> hulls: array<vec4u>;
${SHADE}
struct Out { @builtin(position) clip: vec4f, @location(0) w: vec3f, @location(1) local: vec3f, @location(2) @interpolate(flat) half: vec3f,
  @location(3) @interpolate(flat) kind: vec2u, @location(4) n: vec3f, @location(5) bary: vec3f, @location(6) @interpolate(flat) edges: vec3f };

// Triangles a collider instance may draw: a box takes 12, a hull its faces' fans
// (at most 2V − 4 for V vertices; the fracture's chunks have far fewer than 48).
const TRIS: u32 = ${HULL_TRIS}u;
fn hullWord(base: u32, w: u32) -> u32 { let r = hulls[base + w / 4u]; return r[w % 4u]; }
// Collider ci's triangle corner vi (world), its face normal, its corner of the
// triangle (barycentric) and which of the triangle's edges are the face's own.
fn solidWorld(ci: u32, vi: u32, o: ptr<function, Out>) -> bool {
  let col = cols[ci];
  if (col.shape != 2u && col.shape != 3u) { return false; }
  let tri = vi / 3u;
  let corner = vi % 3u;
  var lp = vec3f(0.0);
  var nl = vec3f(0.0);
  var half = col.params.xyz;
  var edges = vec3f(1.0);
  if (col.shape == 2u) {
    if (tri >= 12u) { return false; }
    // 6 faces × 2 triangles, each face's own normal.
    let face = tri / 2u;
    let k = array(0u, 1u, 2u, 0u, 2u, 3u)[(tri % 2u) * 3u + corner];
    let axis = face / 2u;
    let sgn = select(-1.0, 1.0, face % 2u == 1u);
    var uv = array(vec2f(-1, -1), vec2f(1, -1), vec2f(1, 1), vec2f(-1, 1))[k];
    if (sgn < 0.0) { uv = vec2f(uv.y, uv.x); }
    var c = vec3f(0.0);
    if (axis == 0u) { c = vec3f(sgn, uv.x, uv.y); nl = vec3f(sgn, 0, 0); }
    else if (axis == 1u) { c = vec3f(uv.y, sgn, uv.x); nl = vec3f(0, sgn, 0); }
    else { c = vec3f(uv.x, uv.y, sgn); nl = vec3f(0, 0, sgn); }
    lp = c * half;
    edges = select(vec3f(1.0, 1.0, 0.0), vec3f(1.0, 0.0, 1.0), tri % 2u == 0u);
  } else {
    // The hull's faces as fans (engine_schema::convex's v2 records).
    let base = bitcast<u32>(col.params.x) & 0x1FFFFFFFu;
    let scale = col.params.y;
    let hdr = hulls[base];
    let nv = hdr.x & 0xffu;
    let nf = (hdr.x >> 8u) & 0xffu;
    var t = tri;
    var f = 0u;
    var count = 0u;
    var start = 0u;
    loop {
      if (f >= nf) { return false; }
      let fw = hullWord(base, (4u + nv + nf) * 4u + f);
      count = fw >> 16u;
      start = fw & 0xffffu;
      if (count >= 3u && t < count - 2u) { break; }
      t -= max(count, 2u) - 2u;
      f++;
    }
    let idx = select(select(t + 2u, t + 1u, corner == 1u), 0u, corner == 0u);
    let b = start + idx;
    let lw = hullWord(base, hdr.y + b / 4u);
    let v = (lw >> (8u * (b % 4u))) & 0xffu;
    lp = bitcast<vec3f>(hulls[base + 4u + v].xyz) * scale;
    nl = bitcast<vec4f>(hulls[base + 4u + nv + f]).xyz;
    half = bitcast<vec3f>(hulls[base + 2u].xyz) * scale;
    edges = vec3f(1.0, select(0.0, 1.0, t + 3u == count), select(0.0, 1.0, t == 0u));
  }
  let b = bodies[col.body];
  let q = bodyRot(b);
  (*o).w = bodyPos(b) + qrot(q, col.localPos + qrot(col.localRot, lp));
  (*o).n = qrot(qmul(q, col.localRot), nl);
  (*o).local = lp;
  (*o).half = half;
  (*o).kind = vec2u(bitcast<u32>(col.density), col.body);
  (*o).bary = array(vec3f(1, 0, 0), vec3f(0, 1, 0), vec3f(0, 0, 1))[corner];
  (*o).edges = edges;
  return true;
}
fn colGhosted(ci: u32) -> bool { return ghosted(bodyPos(bodies[cols[ci].body])); }
@vertex fn vsBox(@builtin(vertex_index) vi: u32, @builtin(instance_index) ci: u32) -> Out {
  var o: Out;
  if (colGhosted(ci) || !solidWorld(ci, vi, &o)) { o.clip = vec4f(2.0, 2.0, 2.0, 1.0); return o; }
  o.clip = frame.viewProj * vec4f(o.w, 1.0);
  return o;
}
// The ghosted colliders only (drawn last, blended, no depth write).
@vertex fn vsBoxGhost(@builtin(vertex_index) vi: u32, @builtin(instance_index) ci: u32) -> Out {
  var o: Out;
  if (!colGhosted(ci) || !solidWorld(ci, vi, &o)) { o.clip = vec4f(2.0, 2.0, 2.0, 1.0); return o; }
  o.clip = frame.viewProj * vec4f(o.w, 1.0);
  return o;
}
@vertex fn vsBoxShadow(@builtin(vertex_index) vi: u32, @builtin(instance_index) ci: u32) -> @builtin(position) vec4f {
  var o: Out;
  if (colGhosted(ci) || !solidWorld(ci, vi, &o)) { return vec4f(2.0, 2.0, 2.0, 1.0); }
  return frame.lightVP * vec4f(o.w, 1.0);
}
// Spheres (the wrecking ball): a 24 × 16 latitude–longitude mesh, among the first colliders.
fn sphereWorld(ci: u32, vi: u32, o: ptr<function, Out>) -> bool {
  let col = cols[ci];
  if (col.shape != 0u) { return false; }
  let b = bodies[col.body];
  let quad = vi / 6u;
  let k = array(vec2u(0, 0), vec2u(1, 0), vec2u(1, 1), vec2u(0, 0), vec2u(1, 1), vec2u(0, 1))[vi % 6u];
  let lon = f32(quad % 24u + k.x) / 24.0 * 6.2831853;
  let lat = (f32(quad / 24u + k.y) / 16.0 - 0.5) * 3.14159265;
  let d = vec3f(cos(lat) * cos(lon), sin(lat), cos(lat) * sin(lon));
  let q = bodyRot(b);
  let r = col.params.x;
  (*o).w = bodyPos(b) + qrot(q, col.localPos + d * r);
  (*o).n = qrot(q, d);
  (*o).local = d * r;
  (*o).half = vec3f(r);
  (*o).kind = vec2u(bitcast<u32>(col.density), col.body);
  (*o).edges = vec3f(0.0);
  return true;
}
@vertex fn vsSphere(@builtin(vertex_index) vi: u32, @builtin(instance_index) ci: u32) -> Out {
  var o: Out;
  if (!sphereWorld(ci, vi, &o)) { o.clip = vec4f(2.0, 2.0, 2.0, 1.0); return o; }
  o.clip = frame.viewProj * vec4f(o.w, 1.0);
  return o;
}
@vertex fn vsSphereShadow(@builtin(vertex_index) vi: u32, @builtin(instance_index) ci: u32) -> @builtin(position) vec4f {
  var o: Out;
  if (!sphereWorld(ci, vi, &o)) { return vec4f(2.0, 2.0, 2.0, 1.0); }
  return frame.lightVP * vec4f(o.w, 1.0);
}
// The wrecking ball's rod: a 6 cm bar from the anchor to the ball's centre.
fn rodWorld(vi: u32, o: ptr<function, Out>) -> bool {
  let slot = bitcast<u32>(frame.rod.w);
  if (slot == 0xffffffffu) { return false; }
  let a = frame.rod.xyz;
  let b = bodyPos(bodies[slot]);
  let z = normalize(b - a);
  let x = normalize(cross(z, select(vec3f(0, 1, 0), vec3f(1, 0, 0), abs(z.y) > 0.9)));
  let y = cross(z, x);
  let face = vi / 6u;
  let k = array(0u, 1u, 2u, 0u, 2u, 3u)[vi % 6u];
  let ang = (f32(face) + select(0.0, 1.0, k == 1u || k == 2u)) * 1.5707963;
  let along = select(0.0, 1.0, k >= 2u);
  let rd = cos(ang) * x + sin(ang) * y;
  (*o).w = mix(a, b, along) + rd * 0.05;
  (*o).n = normalize(cos(ang + 0.785) * x + sin(ang + 0.785) * y);
  (*o).local = vec3f(0.0);
  (*o).half = vec3f(1.0);
  (*o).kind = vec2u(bitcast<u32>(7800.0), 0u);
  (*o).edges = vec3f(0.0);
  return face < 4u;
}
@vertex fn vsRod(@builtin(vertex_index) vi: u32) -> Out {
  var o: Out;
  if (!rodWorld(vi, &o)) { o.clip = vec4f(2.0, 2.0, 2.0, 1.0); return o; }
  o.clip = frame.viewProj * vec4f(o.w, 1.0);
  return o;
}
@vertex fn vsRodShadow(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var o: Out;
  if (!rodWorld(vi, &o)) { return vec4f(2.0, 2.0, 2.0, 1.0); }
  return frame.lightVP * vec4f(o.w, 1.0);
}

// Materials from the collider's density (the cuboid house's: brick 1900, concrete
// 2300, timber 550, glass 2500; the showcase's crates 260, slab 2400, ball 5000).
@fragment fn fsSolid(i: Out, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  let density = bitcast<f32>(i.kind.x);
  let h = hash(i.kind.y * 7u + 3u);
  let h2 = hash(i.kind.y * 13u + 5u);
  let n = normalize(i.n);
  // The face's own edges darkened (about a pixel and a half): chunk outlines.
  let px = i.bary / max(fwidth(i.bary), vec3f(1e-6));
  let dEdge = min(min(select(1e9, px.x, i.edges.x > 0.5), select(1e9, px.y, i.edges.y > 0.5)), select(1e9, px.z, i.edges.z > 0.5));
  let edge = 1.0 - 0.38 * (1.0 - smoothstep(0.6, 1.8, dEdge));
  var albedo = vec3f(0.5);
  var spec = 0.08;
  var ao = 1.0;
  if (density > 4000.0) {
    // Steel: the ball and its rod.
    albedo = vec3f(0.16, 0.17, 0.19); spec = 0.9;
  } else if (density > 2450.0) {
    // Glass.
    albedo = vec3f(0.35, 0.5, 0.58); spec = 1.5;
  } else if (density > 2350.0) {
    // The precast slab: rough concrete with stains.
    let t = fbm(i.local.xz * 3.0 + i.local.y * 2.0);
    albedo = vec3f(0.56, 0.55, 0.52) * (0.8 + 0.35 * t); spec = 0.05;
  } else if (density > 2000.0) {
    albedo = vec3f(0.62, 0.6, 0.56) * (0.9 + 0.15 * h);
  } else if (density > 700.0) {
    // Brick: per brick red-brown, a little soot and grain.
    let base = mix(vec3f(0.55, 0.22, 0.14), vec3f(0.68, 0.33, 0.2), h);
    let grain = fbm(vec2f(dot(i.local, vec3f(7.0, 11.0, 5.0)), i.local.y * 9.0));
    albedo = mix(base, vec3f(0.42, 0.3, 0.25), 0.25 * h2) * (0.85 + 0.3 * grain);
  } else if (density > 280.0) {
    // Timber: long thin pieces white trim, broad thin boards dark shingle, the rest stained wood.
    let lmax = max(i.half.x, max(i.half.y, i.half.z));
    let lmin = min(i.half.x, min(i.half.y, i.half.z));
    let grain = noise(vec2f(dot(i.local, vec3f(1.0, 0.0, 1.0)) * 1.5, dot(i.local, vec3f(0.0, 40.0, 0.0))));
    if (lmin < 0.05 && lmax > 0.6) { albedo = vec3f(0.3, 0.27, 0.26) * (0.8 + 0.3 * h); }
    else if (lmin < 0.06) { albedo = vec3f(0.86, 0.85, 0.8); }
    else { albedo = vec3f(0.5, 0.34, 0.2) * (0.75 + 0.35 * grain); }
  } else {
    // A crate: slats along its faces, a darker frame, a stencil-like band.
    let a = abs(i.local) / i.half;
    let fx = a.x >= max(a.y, a.z);
    let fy = !fx && a.y >= a.z;
    let t = select(select(i.local.x, i.local.z, fx), i.local.x, fy);
    let slat = 0.5 + 0.5 * sin(t * 26.0);
    let frameW = select(select(max(a.x, a.y), max(a.x, a.z), fy), max(a.y, a.z), fx);
    albedo = mix(vec3f(0.72, 0.55, 0.32), vec3f(0.62, 0.45, 0.25), slat) * (0.85 + 0.25 * h);
    if (frameW > 0.84) { albedo = vec3f(0.45, 0.3, 0.16); }
    ao = 0.9;
  }
  var c = lit(albedo * edge, n, i.w, ao, spec);
  return vec4f(fog(c, i.w), 1.0);
}
// A ghosted collider: a faint pale fill and its face outlines (cut2.y the fill's
// opacity), so the bricks over and around the car read as a glass model of the pile.
@fragment fn fsGhost(i: Out) -> @location(0) vec4f {
  let px = i.bary / max(fwidth(i.bary), vec3f(1e-6));
  let dEdge = min(min(select(1e9, px.x, i.edges.x > 0.5), select(1e9, px.y, i.edges.y > 0.5)), select(1e9, px.z, i.edges.z > 0.5));
  let line = 1.0 - smoothstep(0.5, 1.6, dEdge);
  let n = normalize(i.n);
  let shade = 0.75 + 0.25 * max(dot(n, frame.sun.xyz), 0.0);
  let c = mix(vec3f(0.86, 0.6, 0.48), vec3f(1.0, 0.93, 0.85), line) * shade;
  // Faded out well above the car (the wall over it) and right at the eye (bricks the
  // camera passes through), so the outlines near the car carry the picture.
  let keep = (1.0 - smoothstep(frame.cut.y + 1.5, frame.cut.y + 4.5, i.w.y)) * smoothstep(1.2, 3.5, distance(i.w, frame.eye.xyz));
  return vec4f(c, (frame.cut2.y + (1.0 - frame.cut2.y) * 0.4 * line) * keep);
}
`;

const ENV = /* wgsl */ `
${COMMON}
${SHADE}
struct SkyOut { @builtin(position) clip: vec4f, @location(0) ndc: vec2f };
@vertex fn vsSky(@builtin(vertex_index) vi: u32) -> SkyOut {
  let p = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3))[vi];
  var o: SkyOut; o.clip = vec4f(p, 1.0, 1.0); o.ndc = p; return o;
}
@fragment fn fsSky(i: SkyOut) -> @location(0) vec4f {
  let a = frame.invViewProj * vec4f(i.ndc, 0.0, 1.0);
  let b = frame.invViewProj * vec4f(i.ndc, 1.0, 1.0);
  let d = normalize(b.xyz / b.w - a.xyz / a.w);
  var c = skyColor(d);
  // Thin clouds.
  if (d.y > 0.0) {
    let uv = d.xz / (d.y + 0.08) * 2.0 + vec2f(frame.eye.w * 0.01, 0.0);
    let cl = smoothstep(0.5, 0.85, fbm(uv));
    c = mix(c, vec3f(0.97, 0.97, 0.98), cl * 0.6 * smoothstep(0.0, 0.15, d.y));
  }
  return vec4f(c, 1.0);
}
struct GOut { @builtin(position) clip: vec4f, @location(0) w: vec3f };
@vertex fn vsGround(@builtin(vertex_index) vi: u32) -> GOut {
  let xz = array(vec2f(-1, -1), vec2f(1, -1), vec2f(1, 1), vec2f(-1, -1), vec2f(1, 1), vec2f(-1, 1))[vi] * 600.0 + frame.focus.xz;
  var o: GOut;
  o.w = vec3f(xz.x, frame.ground.x, xz.y);
  o.clip = frame.viewProj * vec4f(o.w, 1.0);
  return o;
}
@fragment fn fsGround(i: GOut) -> @location(0) vec4f {
  let p = i.w.xz;
  let roadHalf = frame.ground.y;
  let rx = abs(p.x - frame.ground.z);
  // Grass: two scales of noise, patches of dry earth.
  let g = fbm(p * 0.35);
  let g2 = noise(p * 4.0);
  var albedo = mix(vec3f(0.2, 0.3, 0.11), vec3f(0.33, 0.38, 0.16), g) * (0.85 + 0.25 * g2);
  albedo = mix(albedo, vec3f(0.38, 0.32, 0.22), smoothstep(0.62, 0.75, fbm(p * 0.08 + 3.0)) * 0.7);
  var spec = 0.02;
  // The road (along z): asphalt, a dashed centre line, edge lines, a gravel verge.
  if (roadHalf > 0.0) {
    let asphalt = vec3f(0.16, 0.16, 0.17) * (0.85 + 0.3 * noise(p * 9.0)) * (0.9 + 0.2 * fbm(p * 0.5));
    let verge = vec3f(0.42, 0.39, 0.34) * (0.8 + 0.4 * noise(p * 6.0));
    let inRoad = 1.0 - smoothstep(roadHalf - 0.02, roadHalf + 0.02, rx);
    let inVerge = 1.0 - smoothstep(roadHalf + 0.6, roadHalf + 1.1 + 0.4 * noise(p * 1.3), rx);
    albedo = mix(albedo, verge, inVerge);
    albedo = mix(albedo, asphalt, inRoad);
    let fw = fwidth(p.x) + 0.002;
    let edge = smoothstep(roadHalf - 0.35 - fw, roadHalf - 0.33, rx) * (1.0 - smoothstep(roadHalf - 0.2, roadHalf - 0.2 + fw, rx));
    let dash = step(0.5, fract(p.y / 6.0)) * (1.0 - smoothstep(0.08, 0.08 + fw, rx));
    albedo = mix(albedo, vec3f(0.85, 0.84, 0.78), max(edge, dash) * inRoad * 0.9);
    spec = mix(spec, 0.15, inRoad);
  }
  var c = lit(albedo, vec3f(0, 1, 0), i.w, 1.0, spec);
  return vec4f(fog(c, i.w), 1.0);
}
`;

function mat4mul(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
    o[c * 4 + r] = s;
  }
  return o;
}

/** A light view-projection: orthographic along `sun` (toward the sun), `half` m about `focus`. */
export function lightViewProj(sun, focus, half) {
  const z = sun; // the light looks along −sun
  const up = Math.abs(z[1]) > 0.95 ? [1, 0, 0] : [0, 1, 0];
  const nx = [up[1] * z[2] - up[2] * z[1], up[2] * z[0] - up[0] * z[2], up[0] * z[1] - up[1] * z[0]];
  const ln = Math.hypot(...nx); const x = nx.map((v) => v / ln);
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  const eye = focus.map((f, i) => f + sun[i] * 80);
  const view = new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
    -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]), -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]), -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]), 1]);
  const near = 1, far = 200;
  const proj = new Float32Array([1 / half, 0, 0, 0, 0, 1 / half, 0, 0, 0, 0, 1 / (near - far), 0, 0, 0, near / (near - far), 1]);
  return mat4mul(proj, view);
}

// `hullTris`: triangles drawn per collider instance (a scene of boxes only: 12);
// `firstCollider`: colliders before it are not drawn (a ground slab the sky's ground
// stands for); `brick`: light dynamic colliders (below 700 kg/m³) are shaded as brick.
export function createSceneRenderer(device, { format, depthFormat = 'depth32float', sampleCount = 1, hullTris = HULL_TRIS, firstCollider = 0, brick = false }) {
  const mod = (label, code) => device.createShaderModule({ label, code });
  const colMod = mod('one-world colliders', brick
    ? COLLIDERS.replace('let density = bitcast<f32>(i.kind.x);', 'let raw = bitcast<f32>(i.kind.x);\n  let density = select(raw, 1900.0, raw > 0.0 && raw < 700.0);')
    : COLLIDERS);
  const envMod = mod('one-world env', ENV);
  const frameBuf = device.createBuffer({ size: 64 * 3 + 16 * 7, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const frameData = new Float32Array((64 * 3 + 16 * 7) / 4);
  const shadowTex = device.createTexture({ size: [SHADOW, SHADOW], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  const shadowView = shadowTex.createView();
  const shadowSamp = device.createSampler({ compare: 'less', magFilter: 'linear', minFilter: 'linear' });
  const depthStencil = { format: depthFormat, depthWriteEnabled: true, depthCompare: 'less' };
  const shadowDepth = { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBias: 2, depthBiasSlopeScale: 2.5 };
  // One layout for every pipeline: the frame, bodies, colliders, hulls, the shadow map.
  const V = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
  const bgl = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: V, buffer: { type: 'uniform' } },
    ...[1, 2, 3].map((binding) => ({ binding, visibility: V, buffer: { type: 'read-only-storage' } })),
    { binding: 4, visibility: V, texture: { sampleType: 'depth' } },
    { binding: 5, visibility: V, sampler: { type: 'comparison' } },
  ] });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
  const pipe = (module, vs, fs, depth) => device.createRenderPipeline({
    layout, vertex: { module, entryPoint: vs },
    fragment: fs ? { module, entryPoint: fs, targets: [{ format }] } : undefined,
    primitive: { cullMode: 'none' }, depthStencil: depth,
    multisample: { count: fs ? sampleCount : 1 },
  });
  const P = {
    sky: pipe(envMod, 'vsSky', 'fsSky', { format: depthFormat, depthWriteEnabled: false, depthCompare: 'always' }),
    ground: pipe(envMod, 'vsGround', 'fsGround', depthStencil),
    box: pipe(colMod, 'vsBox', 'fsSolid', depthStencil),
    sphere: pipe(colMod, 'vsSphere', 'fsSolid', depthStencil),
    rod: pipe(colMod, 'vsRod', 'fsSolid', depthStencil),
    boxS: pipe(colMod, 'vsBoxShadow', null, shadowDepth),
    sphereS: pipe(colMod, 'vsSphereShadow', null, shadowDepth),
    rodS: pipe(colMod, 'vsRodShadow', null, shadowDepth),
    // The ghost cut's colliders: blended over what is drawn, no depth write.
    ghost: device.createRenderPipeline({
      layout, vertex: { module: colMod, entryPoint: 'vsBoxGhost' },
      fragment: { module: colMod, entryPoint: 'fsGhost', targets: [{ format, blend: {
        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' } } }] },
      primitive: { cullMode: 'none' }, depthStencil: { format: depthFormat, depthWriteEnabled: false, depthCompare: 'less' },
      multisample: { count: sampleCount },
    }),
  };
  // The ghost cut (env.cut): off until set; rendering only.
  let cut = null;
  const res = (b) => ({ buffer: b });
  let bound = [], group = null;
  // The shadow pass cannot sample the map it writes: its group binds a dummy.
  const dummy = device.createTexture({ size: [1, 1], format: 'depth32float', usage: GPUTextureUsage.TEXTURE_BINDING }).createView();
  const make = (bufs, view) => device.createBindGroup({ layout: bgl, entries: [{ binding: 0, resource: res(frameBuf) }, { binding: 1, resource: res(bufs[0]) }, { binding: 2, resource: res(bufs[1]) }, { binding: 3, resource: res(bufs[2]) },
    { binding: 4, resource: view }, { binding: 5, resource: shadowSamp }] });
  const bind = (world) => {
    const bufs = [world.bodyBuffer(), world.colliderBuffer(), world.hullArenaBuffer()];
    if (group && bufs.every((b, i) => b === bound[i])) return group;
    bound = bufs;
    group = { box: make(bufs, shadowView), boxS: make(bufs, dummy) };
    group.sphere = group.rod = group.box;
    group.sphereS = group.rodS = group.boxS;
    return group;
  };
  let lightVP = null;
  let sphereCols = 64;
  const api = {
    sun: (() => { const s = [0.55, 0.72, 0.42]; const l = Math.hypot(...s); return s.map((x) => x / l); })(),
    lightVP: () => lightVP,
    /** The frame's uniforms. */
    frame({ viewProj, invViewProj, eye, focus, ground, lead = 0, time = 0, road = [3.6, 0, 400], rod = null, fogDist = 140 }) {
      lightVP = lightViewProj(api.sun, focus, SHADOW_HALF);
      frameData.set(viewProj, 0);
      frameData.set(invViewProj, 16);
      frameData.set(lightVP, 32);
      frameData.set([eye[0], eye[1], eye[2], time], 48);
      frameData.set([api.sun[0], api.sun[1], api.sun[2], lead], 52);
      frameData.set([ground, road[0], road[1], road[2]], 56);
      frameData.set([rod ? rod.anchor[0] : 0, rod ? rod.anchor[1] : 0, rod ? rod.anchor[2] : 0, 0], 60);
      new Uint32Array(frameData.buffer, 63 * 4, 1)[0] = rod ? rod.body : 0xffffffff;
      frameData.set([focus[0], focus[1], focus[2], fogDist], 64);
      frameData.set(cut ? [cut.centre[0], cut.centre[1], cut.centre[2], cut.radius, cut.towardEye ?? 0, cut.opacity ?? 0.06, 0, 0] : [0, 0, 0, 0, 0, 0, 0, 0], 68);
      device.queue.writeBuffer(frameBuf, 0, frameData);
    },
    /**
     * The ghost cut, rendering only: colliders whose body centre is within `radius` m of
     * `centre`'s vertical axis, or within `towardEye` m of the line from `centre` to the
     * eye, are left out of the solid and shadow passes and drawn by `drawGhosts` as
     * outlines over a faint fill (`opacity`). `null` turns it off. Read at `frame`.
     */
    setCut(c) { cut = c && c.radius > 0 ? c : null; },
    /** The ghost cut's colliders, blended: after everything solid (the cars too). */
    drawGhosts(pass, world) {
      if (!cut) return;
      const g = bind(world);
      const n = world.colliderCount();
      pass.setPipeline(P.ghost); pass.setBindGroup(0, g.box); pass.draw(3 * hullTris, Math.max(0, n - firstCollider), 0, firstCollider);
    },
    /** The shadow map: colliders, then `extra(pass, lightVP)` (the cars). */
    shadowPass(encoder, world, extra) {
      const g = bind(world);
      const pass = encoder.beginRenderPass({ colorAttachments: [], depthStencilAttachment: { view: shadowView, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' } });
      const n = world.colliderCount();
      pass.setPipeline(P.boxS); pass.setBindGroup(0, g.boxS); pass.draw(3 * hullTris, Math.max(0, n - firstCollider), 0, firstCollider);
      pass.setPipeline(P.sphereS); pass.setBindGroup(0, g.sphereS); pass.draw(24 * 16 * 6, Math.min(n, sphereCols));
      pass.setPipeline(P.rodS); pass.setBindGroup(0, g.rodS); pass.draw(36);
      extra?.(pass, lightVP);
      pass.end();
    },
    /** Sky, ground, colliders and the rod into `pass`. */
    draw(pass, world) {
      const g = bind(world);
      const n = world.colliderCount();
      pass.setPipeline(P.sky); pass.setBindGroup(0, g.box); pass.draw(3);
      pass.setPipeline(P.ground); pass.setBindGroup(0, g.box); pass.draw(6);
      pass.setPipeline(P.box); pass.setBindGroup(0, g.box); pass.draw(3 * hullTris, Math.max(0, n - firstCollider), 0, firstCollider);
      pass.setPipeline(P.sphere); pass.setBindGroup(0, g.sphere); pass.draw(24 * 16 * 6, Math.min(n, sphereCols));
      pass.setPipeline(P.rod); pass.setBindGroup(0, g.rod); pass.draw(36);
    },
  };
  return api;
}
