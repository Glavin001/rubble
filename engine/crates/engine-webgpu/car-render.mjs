// Demo renderer of the procedural soft-body car (engine_softbody::visual): a
// detailed, multi-part mesh bound to the car's parts through the engine API
// (SoftHost.bindMesh) and skinned by the engine kernel (soft-skin.mjs), shaded
// here as car paint, glass, chrome, trim, lamps, interior and engine. An
// integration replaces this file with its own renderer; the binding, skinning,
// strain and tear data come from the engine.
//
// Optional particle effects (sparks, glass, chips, dust) spawn from per-triangle
// strain changes; off unless the caller passes a dt.

import { createSoftSkinner } from './soft-skin.mjs';

const FX_N = 16384;

export const LIGHTING_WGSL = /* wgsl */`
struct Frame { viewProj: mat4x4f, eye: vec4f, light: vec4f, opts: vec4f, lightVP: mat4x4f, invViewProj: mat4x4f };
@group(0) @binding(1) var<uniform> frame: Frame;
@group(0) @binding(9) var shadowTex: texture_depth_2d;
@group(0) @binding(10) var shadowSamp: sampler_comparison;

fn toLinear(c: vec3f) -> vec3f { return pow(max(c, vec3f(0.0)), vec3f(2.2)); }
fn tonemap(c: vec3f) -> vec3f {
  let x = c * 0.85;
  let m = clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
  return pow(m, vec3f(1.0 / 2.2));
}
fn sunDir() -> vec3f { return normalize(frame.light.xyz); }
fn skyColor(d: vec3f) -> vec3f {
  let up = clamp(d.y, -1.0, 1.0);
  let zenith = vec3f(0.18, 0.34, 0.66);
  let horizon = vec3f(0.62, 0.72, 0.84);
  let groundC = vec3f(0.16, 0.16, 0.15);
  var c = mix(horizon, zenith, pow(clamp(up, 0.0, 1.0), 0.6));
  c = select(c, mix(horizon * 0.8, groundC, clamp(-up * 4.0, 0.0, 1.0)), up < 0.0);
  let s = max(dot(d, sunDir()), 0.0);
  return c + vec3f(1.0, 0.92, 0.8) * (pow(s, 900.0) * 30.0 + pow(s, 12.0) * 0.25);
}
// Reflection of the environment, blurred toward the hemisphere average by roughness.
fn envLight(r: vec3f, rough: f32) -> vec3f {
  let sharp = skyColor(r);
  let blur = mix(vec3f(0.2, 0.2, 0.19), vec3f(0.45, 0.55, 0.7), clamp(r.y * 0.5 + 0.5, 0.0, 1.0));
  return mix(sharp, blur, clamp(rough * 1.4, 0.0, 1.0));
}
fn shadowAt(w: vec3f, n: vec3f) -> f32 {
  let p = frame.lightVP * vec4f(w + n * 0.03, 1.0);
  let uv = p.xy * vec2f(0.5, -0.5) + 0.5;
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || p.z > 1.0) { return 1.0; }
  let texel = 1.0 / 2048.0;
  var s = 0.0;
  for (var x = -1; x <= 1; x++) {
    for (var y = -1; y <= 1; y++) {
      s += textureSampleCompareLevel(shadowTex, shadowSamp, uv + vec2f(f32(x), f32(y)) * texel * 1.5, p.z - 0.0015);
    }
  }
  return s / 9.0;
}
// albedo linear; rough 0..1; metal 0..1; coat = clearcoat strength.
fn surface(albedo: vec3f, n: vec3f, w: vec3f, rough: f32, metal: f32, coat: f32) -> vec3f {
  let v = normalize(frame.eye.xyz - w);
  let l = sunDir();
  let h = normalize(v + l);
  let ndl = max(dot(n, l), 0.0);
  let ndv = max(dot(n, v), 1e-3);
  let ndh = max(dot(n, h), 0.0);
  let a = max(rough * rough, 0.002);
  let d = a * a / (3.14159 * pow(ndh * ndh * (a * a - 1.0) + 1.0, 2.0));
  let f0 = mix(vec3f(0.04), albedo, metal);
  let fres = f0 + (1.0 - f0) * pow(1.0 - ndv, 5.0);
  let k = a * 0.5;
  let g = ndl / (ndl * (1.0 - k) + k) * ndv / (ndv * (1.0 - k) + k);
  let spec = fres * d * g / max(4.0 * ndl * ndv, 1e-3);
  let sh = shadowAt(w, n);
  let sun = vec3f(1.0, 0.95, 0.86) * 3.2;
  let diffuse = albedo * (1.0 - metal);
  let amb = mix(vec3f(0.14, 0.14, 0.13), vec3f(0.36, 0.44, 0.58), n.y * 0.5 + 0.5) * (0.6 + 0.4 * sh);
  let r = reflect(-v, n);
  var c = diffuse * (sun * ndl * sh / 3.14159 + amb) + spec * sun * ndl * sh + envLight(r, rough) * fres * (0.35 + 0.65 * metal);
  // Clearcoat: a sharp, weak reflection layer on top (car paint).
  let cf = 0.04 + 0.96 * pow(1.0 - ndv, 5.0);
  c = c * (1.0 - coat * cf) + coat * cf * envLight(r, 0.03) * (0.5 + 0.5 * sh);
  let hc = pow(ndh, 600.0) * ndl * sh * coat;
  c += vec3f(hc * 6.0);
  let fog = clamp(length(frame.eye.xyz - w) / 320.0, 0.0, 1.0);
  return mix(c, skyColor(normalize(w - frame.eye.xyz)) * 0.9, fog * fog);
}
fn hash3(p: vec3f) -> f32 { return fract(sin(dot(p, vec3f(127.1, 311.7, 74.7))) * 43758.5453); }
fn noise3(p: vec3f) -> f32 {
  let i = floor(p); let f = fract(p); let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hash3(i), hash3(i + vec3f(1, 0, 0)), u.x), mix(hash3(i + vec3f(0, 1, 0)), hash3(i + vec3f(1, 1, 0)), u.x), u.y),
             mix(mix(hash3(i + vec3f(0, 0, 1)), hash3(i + vec3f(1, 0, 1)), u.x), mix(hash3(i + vec3f(0, 1, 1)), hash3(i + vec3f(1, 1, 1)), u.x), u.y), u.z);
}
`;

const COMPUTE_WGSL = /* wgsl */`
struct Body { position: vec3f, invMass: f32, orientation: vec4f, invInertia: vec3f, flags: u32,
  linearVel: vec3f, sleepTimer: f32, angularVel: vec3f, generation: u32, initialPos: vec3f, initialPad: f32, initialRot: vec4f };
struct Counts { verts: u32, tris: u32, seed: u32, dt: f32 };
struct Particle { pos: vec3f, life: f32, vel: vec3f, kind: u32 };
struct Particles { head: atomic<u32>, pad0: u32, pad1: u32, pad2: u32, p: array<Particle, ${FX_N}> };
@group(0) @binding(0) var<storage, read> bodies: array<Body>;
@group(0) @binding(2) var<storage, read> tris: array<vec4u>;
@group(0) @binding(3) var<storage, read> faces: array<vec4f>;
@group(0) @binding(4) var<storage, read> vpos: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> prevStrain: array<f32>;
@group(0) @binding(6) var<storage, read> vmeta: array<vec4u>;
@group(0) @binding(7) var<uniform> counts: Counts;
@group(0) @binding(8) var<storage, read_write> fx: Particles;

fn rnd(a: u32, b: u32) -> f32 {
  var x = a * 747796405u + b * 2891336453u + counts.seed * 1664525u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return f32((x >> 22u) ^ x) / 4294967295.0;
}
fn rndDir(a: u32, b: u32) -> vec3f {
  let z = rnd(a, b) * 2.0 - 1.0;
  let ph = rnd(a, b + 7u) * 6.28318;
  let r = sqrt(max(1.0 - z * z, 0.0));
  return vec3f(r * cos(ph), z, r * sin(ph));
}
// kind: 0 spark, 1 glass, 2 paint chip, 3 dust; car index << 8.
fn spawn(pos: vec3f, vel: vec3f, life: f32, kind: u32) {
  let i = atomicAdd(&fx.head, 1u) % ${FX_N}u;
  fx.p[i] = Particle(pos, life, vel, kind);
}

// Effects from what changed since the last frame (engine faces: w = strain, −1
// torn): a tear sheds a chip, a pane that lets go sheds a shard, a sudden
// crumple throws a spark and a little dust.
@compute @workgroup_size(64) fn spawnFx(@builtin(global_invocation_id) id: vec3u) {
  let t = id.x;
  if (t >= counts.tris) { return; }
  let cur = faces[t].w;
  let prev = prevStrain[t];
  prevStrain[t] = cur;
  if (counts.dt <= 0.0 || prev < 0.0) { return; }
  let f = tris[t];
  let cen = (vpos[f.x].xyz + vpos[f.y].xyz + vpos[f.z].xyz) / 3.0;
  let vel = bodies[vmeta[f.x * 2u].y].linearVel;
  let car = vmeta[f.x * 2u].x >> 16u;
  if (cur < 0.0) {
    if (f.w == 0u && rnd(t, 1u) < 0.025) { spawn(cen, vel * 0.85 + rndDir(t, 2u) * 1.2 + vec3f(0.0, 0.8, 0.0), 5.0, 2u | (car << 8u)); }
    return;
  }
  if (f.w == 1u && cur > 0.06 && prev <= 0.06) {
    if (rnd(t, 10u) < 0.5) { spawn(cen, vel * 0.8 + rndDir(t, 11u) * 1.5 + vec3f(0.0, 0.6, 0.0), 4.0, 1u); }
    return;
  }
  let jump = cur - prev;
  if (f.w == 0u && jump > 0.02 && rnd(t, 3u) < min(jump * 1.2, 0.08)) {
    spawn(cen, vel * 0.6 + rndDir(t, 4u) * (2.0 + 4.0 * rnd(t, 5u)) + vec3f(0.0, 1.5, 0.0), 0.25 + 0.35 * rnd(t, 6u), 0u);
    if (rnd(t, 8u) < 0.04) { spawn(cen, vel * 0.3 + rndDir(t, 9u) * 0.4, 1.0, 3u); }
  }
}

// Particles: gravity, drag, a bouncing ground at y = 0.
@compute @workgroup_size(64) fn particles(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= ${FX_N}u) { return; }
  var q = fx.p[i];
  if (q.life <= 0.0 || counts.dt <= 0.0) { return; }
  let dt = counts.dt;
  let kind = q.kind & 255u;
  let drag = select(select(0.4, 3.0, kind == 3u), 1.2, kind == 0u);
  q.vel = q.vel * exp(-drag * dt) + vec3f(0.0, select(-9.81, 0.4, kind == 3u), 0.0) * dt;
  q.pos += q.vel * dt;
  if (q.pos.y < 0.01) {
    q.pos.y = 0.01;
    q.vel = vec3f(q.vel.x * 0.55, abs(q.vel.y) * select(0.25, 0.4, kind == 1u), q.vel.z * 0.55);
    if (kind == 0u) { q.life -= 0.1; }
  }
  q.life -= dt;
  fx.p[i] = q;
}
`;

const RENDER_WGSL = LIGHTING_WGSL + /* wgsl */`
@group(0) @binding(2) var<storage, read> tris: array<vec4u>;
@group(0) @binding(3) var<storage, read> tinfo: array<vec4f>;
@group(0) @binding(4) var<storage, read> vpos: array<vec4f>;
@group(0) @binding(5) var<storage, read> vnrm: array<vec4f>;
@group(0) @binding(6) var<storage, read> vmeta: array<vec4u>; // 2 per vertex: flags, centre body | rest xyz bits

struct Out { @builtin(position) clip: vec4f, @location(0) n: vec3f, @location(1) w: vec3f, @location(2) rest: vec3f,
  @location(3) @interpolate(flat) mat: u32, @location(4) dmg: f32, @location(5) seam: f32, @location(6) @interpolate(flat) car: u32,
  @location(7) tstrain: f32 };
const HIDE = vec4f(0.0, 0.0, 2.0, 1.0);

fn carVertex(vi: u32, vp: mat4x4f) -> Out {
  let t = vi / 3u;
  let f = tris[t];
  let k = vi % 3u;
  let v = select(select(f.z, f.y, k == 1u), f.x, k == 0u);
  let ti = tinfo[t];
  let vflags = vmeta[v * 2u].x;
  var o: Out;
  o.w = vpos[v].xyz;
  o.n = vnrm[v].xyz;
  o.dmg = vnrm[v].w;
  o.tstrain = max(ti.w, 0.0);
  o.rest = bitcast<vec4f>(vmeta[v * 2u + 1u]).xyz;
  o.mat = f.w;
  o.seam = select(0.0, 1.0, ((vflags >> 8u) & 1u) != 0u);
  o.car = vflags >> 16u;
  // Glass shatters once its panel has crumpled a few percent.
  let shattered = f.w == 1u && ti.w > 0.06;
  o.clip = select(vp * vec4f(o.w, 1.0), HIDE, ti.w < 0.0 || shattered || frame.opts.x > 0.5);
  return o;
}
@vertex fn vsCar(@builtin(vertex_index) vi: u32) -> Out { return carVertex(vi, frame.viewProj); }

struct Particle { pos: vec3f, life: f32, vel: vec3f, kind: u32 };
struct Particles { head: u32, pad0: u32, pad1: u32, pad2: u32, p: array<Particle, ${FX_N}> };
@group(0) @binding(7) var<storage, read> fx: Particles;
struct FxOut { @builtin(position) clip: vec4f, @location(0) c: vec4f, @location(1) uv: vec2f };
@vertex fn vsFx(@builtin(vertex_index) vi: u32, @builtin(instance_index) inst: u32) -> FxOut {
  let q = fx.p[inst];
  let kind = q.kind & 255u;
  var o: FxOut;
  let corner = array<vec2f, 6>(vec2f(-1, -1), vec2f(1, -1), vec2f(1, 1), vec2f(-1, -1), vec2f(1, 1), vec2f(-1, 1))[vi];
  o.uv = corner;
  let toEye = normalize(frame.eye.xyz - q.pos);
  var right = normalize(cross(toEye, vec3f(0.0, 1.0, 0.0)));
  var up = cross(right, toEye);
  var p = q.pos;
  let h = f32(inst * 2654435761u % 1000u) / 1000.0;
  if (kind == 0u) {
    // Spark: a streak along its velocity.
    let v = q.vel;
    let len = clamp(length(v) * 0.02, 0.02, 0.25);
    let d = normalize(v + vec3f(1e-4));
    let side = normalize(cross(d, toEye));
    p = q.pos - d * len * (corner.y * 0.5 + 0.5) + side * corner.x * 0.006;
    let fade = clamp(q.life * 3.0, 0.0, 1.0);
    o.c = vec4f(vec3f(1.0, 0.7, 0.3) * 1.8 * fade, 0.0);
  } else if (kind == 1u) {
    let s = 0.012 + 0.015 * h;
    p = q.pos + (right * corner.x + up * corner.y) * s;
    o.c = vec4f(vec3f(0.75, 0.85, 0.95) * 0.8, 0.85);
  } else if (kind == 2u) {
    let s = 0.025 + 0.04 * h;
    let ang = h * 6.28 + q.life * 4.0;
    let r2 = right * cos(ang) + up * sin(ang);
    let u2 = cross(r2, toEye);
    p = q.pos + (r2 * corner.x + u2 * corner.y * 0.6) * s;
    let car = q.kind >> 8u;
    o.c = vec4f(tonemap(toLinear(paintColor(car)) * (0.4 + 0.6 * h)), 1.0);
  } else {
    let s = 0.1 + (1.0 - q.life) * 0.3;
    p = q.pos + (right * corner.x + up * corner.y) * s;
    let a = 0.1 * clamp(q.life, 0.0, 1.0);
    o.c = vec4f(vec3f(0.45, 0.43, 0.4) * a, a);
  }
  o.clip = select(frame.viewProj * vec4f(p, 1.0), HIDE, q.life <= 0.0);
  return o;
}
@fragment fn fsFx(i: FxOut) -> @location(0) vec4f {
  let r = length(i.uv);
  if (i.c.a > 0.0 && i.c.a < 0.5 && r > 1.0) { discard; } // round dust
  return i.c;
}
@vertex fn vsCarShadow(@builtin(vertex_index) vi: u32) -> Out { return carVertex(vi, frame.lightVP); }

fn paintColor(car: u32) -> vec3f {
  let k = car % 5u;
  if (k == 0u) { return vec3f(0.62, 0.04, 0.03); }
  if (k == 1u) { return vec3f(0.85, 0.86, 0.88); }
  if (k == 2u) { return vec3f(0.04, 0.16, 0.48); }
  if (k == 3u) { return vec3f(0.08, 0.08, 0.09); }
  return vec3f(0.9, 0.55, 0.05);
}

@fragment fn fsCar(i: Out, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  var n = normalize(i.n);
  if (!front) { n = -n; }
  let p = i.rest;
  var albedo = vec3f(0.5);
  var rough = 0.5; var metal = 0.0; var coat = 0.0; var emit = vec3f(0.0);
  switch i.mat {
    case 0u: { // paint
      albedo = toLinear(paintColor(i.car));
      rough = 0.32; coat = 1.0;
      // Crumpled metal: the clearcoat dulls, creases darken, and thin scrapes
      // (long along the car) cut through to bare steel where it crumpled most.
      let d = smoothstep(0.015, 0.25, max(i.dmg, i.tstrain));
      let streak = noise3(p * vec3f(90.0, 90.0, 6.0)) * 0.7 + noise3(p * vec3f(30.0, 30.0, 3.0)) * 0.3;
      let bare = smoothstep(0.86, 0.9, streak) * smoothstep(0.25, 0.8, d) + smoothstep(0.55, 1.0, d) * step(0.55, noise3(p * 18.0));
      let b = clamp(bare, 0.0, 1.0);
      albedo = mix(albedo * (1.0 - 0.45 * d), vec3f(0.5, 0.51, 0.53), b);
      metal = b; rough = mix(mix(rough, 0.55, d), 0.38, b); coat = coat * (1.0 - d) * (1.0 - b);
    }
    case 1u: { albedo = vec3f(0.01, 0.012, 0.016); rough = 0.04; coat = 0.0; } // glass (blended pass)
    case 2u: { albedo = vec3f(0.92, 0.92, 0.95); rough = 0.12; metal = 1.0; } // chrome
    case 3u: { albedo = vec3f(0.025, 0.025, 0.028); rough = 0.55; } // trim, bumpers
    case 4u: { albedo = vec3f(0.9); rough = 0.08; metal = 0.6; emit = vec3f(1.6, 1.5, 1.2); } // headlight
    case 5u: { albedo = vec3f(0.3, 0.01, 0.01); rough = 0.1; emit = vec3f(1.1, 0.04, 0.03); } // taillight
    case 6u: { albedo = vec3f(0.035); rough = 0.85; } // underbody
    case 7u: { albedo = vec3f(0.045, 0.045, 0.05); rough = 0.9; } // interior
    case 8u: { albedo = vec3f(0.22, 0.22, 0.23); rough = 0.4; metal = 0.7; } // engine
    case 9u: { // grille: dark honeycomb
      let g = abs(fract(p.xy * vec2f(28.0, 28.0)) - 0.5);
      albedo = select(vec3f(0.01), vec3f(0.08), max(g.x, g.y) > 0.4); rough = 0.5;
    }
    default: {}
  }
  if (!front && i.mat == 0u) {
    // Inside of a panel: grey primer.
    albedo = vec3f(0.18, 0.18, 0.17); metal = 0.0; coat = 0.0; rough = 0.8;
  }
  var c = surface(albedo, n, i.w, rough, metal, coat) + emit;
  // Panel gaps: a thin dark line along seams between parts.
  c *= 1.0 - 0.85 * smoothstep(0.82, 0.97, i.seam);
  return vec4f(tonemap(c), 1.0);
}

@fragment fn fsGlass(i: Out, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  var n = normalize(i.n);
  if (!front) { n = -n; }
  let v = normalize(frame.eye.xyz - i.w);
  let ndv = max(dot(n, v), 0.0);
  let fres = 0.04 + 0.96 * pow(1.0 - ndv, 5.0);
  let refl = envLight(reflect(-v, n), 0.02) * (0.4 + 0.6 * shadowAt(i.w, n));
  // Stress cracks before the pane lets go.
  // Thin lines along the noise's mid-level set: a web of cracks.
  let nz = noise3(i.rest * 22.0) * 0.7 + noise3(i.rest * 61.0) * 0.3;
  let crack = smoothstep(0.012, 0.05, i.tstrain) * (1.0 - smoothstep(0.0, 0.018, abs(nz - 0.5)));
  let base = vec3f(0.015, 0.02, 0.025);
  let c = mix(base, refl, clamp(fres + 0.08, 0.0, 1.0)) + vec3f(0.7) * crack;
  let alpha = clamp(0.5 + fres * 0.5 + crack, 0.0, 1.0);
  return vec4f(tonemap(c), alpha);
}
`;

// Visual part ids (engine_softbody::visual::part) → the procedural car's part names.
const PART_NAMES = ['body', 'bonnet', 'boot', 'door_fl', 'door_rl', 'door_fr', 'door_rr', 'bumper_front', 'bumper_rear', 'body', 'engine', 'roof'];

/**
 * Car renderer. `visual` = proceduralCarVisual() words; `soft` = the world's
 * SoftHost (bindMesh); `bodies` = the body buffer (binding 0 of the step).
 */
export function createCarRenderer(device, { format, sampleCount, bodies, frameBuf, shadowView, shadowSampler, visual, soft }) {
  const nv0 = visual[0], nt0 = visual[1];
  const bindW = visual.subarray(2, 2 + nv0 * 8);
  const restW = visual.subarray(2 + nv0 * 8, 2 + nv0 * 11);
  const triW = visual.subarray(2 + nv0 * 11, 2 + nv0 * 11 + nt0 * 4);
  const rest0 = new Float32Array(restW.buffer, restW.byteOffset, restW.length);
  // Vertices per part (the part id is the low byte of the visual's flags).
  const byPart = new Map();
  for (let v = 0; v < nv0; v++) {
    const pt = bindW[v * 8 + 3] & 255;
    if (!byPart.has(pt)) byPart.set(pt, []);
    byPart.get(pt).push(v);
  }
  // Triangles: opaque first, then glass (drawn blended after everything else).
  const order = [];
  for (const glassPass of [false, true]) for (let t = 0; t < nt0; t++) if ((triW[t * 4 + 3] === 1) === glassPass) order.push(t);
  const opaquePerCar = order.findIndex((t) => triW[t * 4 + 3] === 1);
  const compMod = device.createShaderModule({ label: 'car-fx', code: COMPUTE_WGSL });
  const rendMod = device.createShaderModule({ label: 'car-render', code: RENDER_WGSL });
  const cp = (entry) => device.createComputePipeline({ layout: 'auto', compute: { module: compMod, entryPoint: entry } });
  const pSpawn = cp('spawnFx'), pFx = cp('particles');
  const depthStencil = { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' };
  const opaque = device.createRenderPipeline({
    layout: 'auto', vertex: { module: rendMod, entryPoint: 'vsCar' },
    fragment: { module: rendMod, entryPoint: 'fsCar', targets: [{ format }] },
    primitive: { cullMode: 'none' }, depthStencil, multisample: { count: sampleCount },
  });
  const glass = device.createRenderPipeline({
    layout: 'auto', vertex: { module: rendMod, entryPoint: 'vsCar' },
    fragment: { module: rendMod, entryPoint: 'fsGlass', targets: [{ format, blend: {
      color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' } } }] },
    primitive: { cullMode: 'none' }, depthStencil: { ...depthStencil, depthWriteEnabled: false }, multisample: { count: sampleCount },
  });
  const fxPipe = device.createRenderPipeline({
    layout: 'auto', vertex: { module: rendMod, entryPoint: 'vsFx' },
    fragment: { module: rendMod, entryPoint: 'fsFx', targets: [{ format, blend: {
      color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' } } }] },
    primitive: { cullMode: 'none' }, depthStencil: { ...depthStencil, depthWriteEnabled: false }, multisample: { count: sampleCount },
  });
  // Particles persist across rebuilds (sparks keep flying when a car spawns).
  const fxBuf = device.createBuffer({ size: 16 + FX_N * 32, usage: GPUBufferUsage.STORAGE });
  const shadow = device.createRenderPipeline({
    layout: 'auto', vertex: { module: rendMod, entryPoint: 'vsCarShadow' },
    primitive: { cullMode: 'none' }, depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBias: 2, depthBiasSlopeScale: 2.5 },
  });
  const r = { verts: 0, tris: 0, opaqueTris: 0, groups: null, skinner: null };
  let bufs = [];
  let generation = 0;
  const buf = (data, usage) => {
    const b = device.createBuffer({ size: Math.max(16, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST });
    if (data.byteLength) device.queue.writeBuffer(b, 0, data);
    bufs.push(b);
    return b;
  };
  const bg = (p, list) => device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: list.map(([binding, resource]) => ({ binding, resource: resource instanceof GPUBuffer ? { buffer: resource } : resource })) });
  /** cars: [{ h, index }]: one mesh per procedural car, bound with soft.bindMesh. */
  r.rebuild = async (cars) => {
    const gen = ++generation;
    const nv = nv0 * cars.length, nt = nt0 * cars.length;
    if (!nt) { r.groups = null; return; }
    // Engine bindings, part by part (a door's mesh follows only the door's nodes).
    const bindings = new Uint32Array(nv * 8);
    const vmeta = new Uint32Array(nv * 8);
    const rest = new Float32Array(nv * 3);
    cars.forEach((car, ci) => {
      for (const [pt, verts] of byPart) {
        const xyz = new Float32Array(verts.length * 3);
        verts.forEach((v, k) => xyz.set(rest0.subarray(v * 3, v * 3 + 3), k * 3));
        let w;
        try { w = soft.bindMesh(car.h, xyz, PART_NAMES[pt] ?? 'body'); } catch { w = soft.bindMesh(car.h, xyz); } // a desc without parts
        verts.forEach((v, k) => {
          const o = ci * nv0 + v;
          bindings.set(w.subarray(k * 8, k * 8 + 8), o * 8);
          vmeta[o * 8] = (bindW[v * 8 + 3] & 0xffff) | (car.index << 16);
          vmeta[o * 8 + 1] = w[k * 8];
          vmeta.set(new Uint32Array(rest0.buffer, rest0.byteOffset + v * 12, 3), o * 8 + 4);
          rest.set(rest0.subarray(v * 3, v * 3 + 3), o * 3);
        });
      }
    });
    const tris = new Uint32Array(nt * 4);
    let k = 0;
    for (const glassPass of [false, true]) {
      cars.forEach((car, ci) => {
        for (const t of order) {
          if ((triW[t * 4 + 3] === 1) !== glassPass) continue;
          tris.set([ci * nv0 + triW[t * 4], ci * nv0 + triW[t * 4 + 1], ci * nv0 + triW[t * 4 + 2], triW[t * 4 + 3]], k * 4);
          k++;
        }
      });
    }
    const skinner = await createSoftSkinner(device, bodies, { bindings, rest, triangles: tris, stride: 4 });
    if (gen !== generation) { skinner.destroy(); return; }
    r.skinner?.destroy();
    bufs.forEach((b) => b.destroy()); bufs = [];
    r.skinner = skinner;
    r.verts = nv; r.tris = nt;
    r.opaqueTris = (opaquePerCar < 0 ? nt0 : opaquePerCar) * cars.length;
    const S = GPUBufferUsage.STORAGE;
    const vmetaB = buf(vmeta, S), prevB = buf(new Float32Array(nt), S);
    r.countsB = buf(Uint32Array.of(nv, nt, 0, 0), GPUBufferUsage.UNIFORM);
    const T = skinner.triangles, F = skinner.faces, P = skinner.positions, N = skinner.normals;
    r.groups = {
      spawn: bg(pSpawn, [[0, bodies], [2, T], [3, F], [4, P], [5, prevB], [6, vmetaB], [7, r.countsB], [8, fxBuf]]),
      fx: bg(pFx, [[7, r.countsB], [8, fxBuf]]),
      fxDraw: bg(fxPipe, [[1, frameBuf], [7, fxBuf]]),
      opaque: bg(opaque, [[1, frameBuf], [2, T], [3, F], [4, P], [5, N], [6, vmetaB], [9, shadowView], [10, shadowSampler]]),
      glass: bg(glass, [[1, frameBuf], [2, T], [3, F], [4, P], [5, N], [6, vmetaB], [9, shadowView], [10, shadowSampler]]),
      shadow: bg(shadow, [[1, frameBuf], [2, T], [3, F], [4, P], [5, N], [6, vmetaB]]),
    };
  };
  let seed = 0;
  const tick = new ArrayBuffer(8);
  /** Skin (engine kernel), then effects. `dt` = simulated seconds since the last
   * frame (0 = effects paused). */
  r.compute = (enc, dt = 0) => {
    if (!r.groups) return;
    r.skinner.encode(enc);
    new Uint32Array(tick)[0] = ++seed;
    new Float32Array(tick)[1] = dt;
    device.queue.writeBuffer(r.countsB, 8, tick);
    const pass = enc.beginComputePass();
    const run = (p, g, n) => { pass.setPipeline(p); pass.setBindGroup(0, g); pass.dispatchWorkgroups(Math.ceil(n / 64)); };
    run(pSpawn, r.groups.spawn, r.tris);
    run(pFx, r.groups.fx, FX_N);
    pass.end();
  };
  r.drawOpaque = (rp) => { if (r.groups && r.opaqueTris) { rp.setPipeline(opaque); rp.setBindGroup(0, r.groups.opaque); rp.draw(r.opaqueTris * 3); } };
  r.drawGlass = (rp) => {
    if (!r.groups || r.tris === r.opaqueTris) return;
    rp.setPipeline(glass); rp.setBindGroup(0, r.groups.glass); rp.draw((r.tris - r.opaqueTris) * 3, 1, r.opaqueTris * 3);
  };
  r.drawFx = (rp) => { if (r.groups) { rp.setPipeline(fxPipe); rp.setBindGroup(0, r.groups.fxDraw); rp.draw(6, FX_N); } };
  r.drawShadow = (rp) => { if (r.groups && r.opaqueTris) { rp.setPipeline(shadow); rp.setBindGroup(0, r.groups.shadow); rp.draw(r.opaqueTris * 3); } };
  return r;
}
