// Crash-test lab renderer for the explicit node–beam solver (softbody.html
// ?solver=explicit): beams as round shaded tubes, nodes as sphere impostors, a
// translucent ghost (or solid) collision skin, the grab line, and a dark studio
// background and floor. Rendering only: every number comes from the solver's GPU
// buffers (ExplicitSim nodesBuffer / miscBuffer / beamsBuffer, its render records
// and labBeamsBuffer) or, in replay, from frame states the page recorded
// (ExplicitSim.readFrameState).
//
// Each frame `gather(enc)` copies the live state into two compact buffers (node
// world positions, beam rest lengths negated when broken); a replay frame
// `upload(state)` writes them instead. Every draw reads only those, so live and
// replay render the same way.
//
// Beam colours (colour modes):
//   strain  the beam's elastic force estimate k·|length − rest| over its break
//           strength, or its permanent length change over 5 % (capped short of a
//           break): blue-grey at rest, yellow, orange, red near breaking;
//   damage  intact blue-grey, bent yellow to red (rest off its spawn length by
//           0.1 % to 8 %), broken magenta;
//   part    the beam's part (labParts: front, rear, roof, cabin, tyres, suspension).

export const STRAIN_STOPS = [[0.0, [0.46, 0.53, 0.63]], [0.12, [0.62, 0.66, 0.70]], [0.4, [1.0, 0.84, 0.18]], [0.7, [1.0, 0.48, 0.08]], [1.0, [0.93, 0.08, 0.07]]];
export const PART_COLORS = [[0.15, 0.85, 0.95], [0.62, 0.42, 0.98], [1.0, 0.84, 0.2], [1.0, 0.36, 0.7], [0.32, 0.34, 0.38], [0.7, 0.74, 0.8]];
export const COLOR_MODES = ['strain', 'damage', 'part'];
export const SKIN_MODES = ['off', 'ghost', 'solid'];

const css = (c) => `rgb(${c.map((x) => Math.round(x * 255)).join(',')})`;
/** CSS linear-gradient of the strain ramp (the legend). */
export const strainGradientCss = () => `linear-gradient(90deg, ${STRAIN_STOPS.map(([t, c]) => `${css(c)} ${(t * 100).toFixed(0)}%`).join(', ')})`;
export const partCss = (i) => css(PART_COLORS[i % PART_COLORS.length]);

const wgslVec = (c) => `vec3f(${c.map((x) => x.toFixed(3)).join(', ')})`;

const SHADER = (sides) => /* wgsl */`
struct Frame { viewProj: mat4x4f, eye: vec4f, light: vec4f, opts: vec4f, lightVP: mat4x4f, invViewProj: mat4x4f };
struct Lab {
  colorMode: u32, skinMode: u32, flags: u32, grabNode: u32,
  tubeR: f32, nodeR: f32, beams: u32, nodes: u32,
  grabTarget: vec4f,
  floorY: f32, focusX: f32, focusZ: f32, pad: f32,
};
@group(0) @binding(1) var<uniform> frame: Frame;
@group(0) @binding(2) var<uniform> lab: Lab;
@group(0) @binding(3) var<storage, read> pos: array<f32>;
@group(0) @binding(4) var<storage, read> rest: array<f32>;
@group(0) @binding(5) var<storage, read> recs: array<vec4u>;
@group(0) @binding(6) var<storage, read> info: array<vec4u>;
@group(0) @binding(7) var<storage, read> tris: array<vec4u>;
@group(0) @binding(8) var<storage, read> nodeVeh: array<u32>;
@group(0) @binding(9) var shadowTex: texture_depth_2d;
@group(0) @binding(10) var shadowSamp: sampler_comparison;

const SIDES = ${sides}u;
const HIDE = vec4f(0.0, 0.0, 2.0, 1.0);
const F_SKELETON = 1u;
const F_NODES = 2u;
const F_STUBS = 4u;
const F_STUDIO = 8u;
const F_TYRES = 16u;

fn P(i: u32) -> vec3f { return vec3f(pos[i * 3u], pos[i * 3u + 1u], pos[i * 3u + 2u]); }
fn toLinear(c: vec3f) -> vec3f { return pow(max(c, vec3f(0.0)), vec3f(2.2)); }
fn tonemap(c: vec3f) -> vec3f {
  let x = c * 0.9;
  let m = clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
  return pow(m, vec3f(1.0 / 2.2));
}
fn shadowAt(w: vec3f) -> f32 {
  let p = frame.lightVP * vec4f(w + vec3f(0.0, 0.02, 0.0), 1.0);
  let uv = p.xy * vec2f(0.5, -0.5) + 0.5;
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || p.z > 1.0) { return 1.0; }
  let texel = 1.0 / 2048.0;
  var s = 0.0;
  for (var x = -2; x <= 2; x++) {
    for (var y = -2; y <= 2; y++) {
      s += textureSampleCompareLevel(shadowTex, shadowSamp, uv + vec2f(f32(x), f32(y)) * texel * 2.5, p.z - 0.002);
    }
  }
  return s / 25.0;
}
// Studio lighting: a key light (the sun's direction), a cool rim from behind and a
// soft hemisphere fill; linear albedo in, display colour out.
fn studio(albedo: vec3f, n0: vec3f, w: vec3f, gloss: f32) -> vec3f {
  let v = normalize(frame.eye.xyz - w);
  let n = select(-n0, n0, dot(n0, v) >= 0.0);
  let key = normalize(frame.light.xyz);
  let rimL = normalize(vec3f(-key.x, 0.35, -key.z));
  let ndl = max(dot(n, key), 0.0);
  let h = normalize(v + key);
  let spec = pow(max(dot(n, h), 0.0), mix(16.0, 90.0, gloss)) * mix(0.15, 0.7, gloss);
  let fill = mix(vec3f(0.10, 0.12, 0.18), vec3f(0.42, 0.47, 0.58), n.y * 0.5 + 0.5);
  let rim = pow(1.0 - max(dot(n, v), 0.0), 3.0) * (0.35 + 0.65 * max(dot(n, rimL), 0.0));
  let c = albedo * (fill + vec3f(1.0, 0.97, 0.92) * ndl * 1.25) + vec3f(spec) + vec3f(0.45, 0.6, 0.9) * rim * 0.45;
  return tonemap(c);
}

fn ramp(t: f32) -> vec3f {
${(() => {
    let s = '  var c = ' + wgslVec(STRAIN_STOPS[0][1]) + ';\n';
    for (let k = 1; k < STRAIN_STOPS.length; k++) {
      const [t0, c0] = STRAIN_STOPS[k - 1], [t1, c1] = STRAIN_STOPS[k];
      s += `  c = select(c, mix(${wgslVec(c0)}, ${wgslVec(c1)}, clamp((t - ${t0.toFixed(3)}) / ${(t1 - t0).toFixed(3)}, 0.0, 1.0)), t > ${t0.toFixed(3)});\n`;
    }
    return s + '  return c;';
  })()}
}
fn partColor(p: u32) -> vec3f {
  var cols = array<vec3f, ${PART_COLORS.length}>(${PART_COLORS.map(wgslVec).join(', ')});
  return cols[min(p, ${PART_COLORS.length - 1}u)];
}

struct TubeOut { @builtin(position) clip: vec4f, @location(0) n: vec3f, @location(1) c: vec3f, @location(2) w: vec3f, @location(3) gloss: f32 };

// A tube between a and b: vertex k of SIDES·6 (two rings, no caps; the node
// spheres cover the ends).
fn tubeVertex(a: vec3f, b: vec3f, r: f32, k: u32, o: ptr<function, TubeOut>) {
  let quad = k / 6u;
  var corners = array<vec2u, 6>(vec2u(0u, 0u), vec2u(1u, 0u), vec2u(1u, 1u), vec2u(0u, 0u), vec2u(1u, 1u), vec2u(0u, 1u));
  let corner = corners[k % 6u];
  let side = (quad + corner.y) % SIDES;
  let along = f32(corner.x);
  let d0 = b - a;
  let len = max(length(d0), 1e-6);
  let d = d0 / len;
  let up0 = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(d.y) > 0.9);
  let u = normalize(cross(d, up0));
  let v = cross(d, u);
  let ang = f32(side) / f32(SIDES) * 6.2831853;
  let n = u * cos(ang) + v * sin(ang);
  let w = a + d0 * along + n * r;
  (*o).n = n;
  (*o).w = w;
  (*o).clip = frame.viewProj * vec4f(w, 1.0);
}

// Beams: two tubes of SIDES·6 vertices per beam: an intact beam draws the first
// (a to b), a broken one (with stubs on) a short stub at each end.
@vertex fn vsTube(@builtin(vertex_index) vall: u32) -> TubeOut {
  let per = SIDES * 6u;
  let beam = vall / (2u * per);
  let hf = (vall / per) & 1u;
  let k = vall % per;
  var o: TubeOut;
  o.gloss = 0.55;
  o.clip = HIDE; o.n = vec3f(0.0, 1.0, 0.0); o.c = vec3f(0.0); o.w = vec3f(0.0);
  if ((lab.flags & F_SKELETON) == 0u) { return o; }
  let rec = recs[beam];
  let inf = info[beam];
  let tyre = (rec.z >> 16u) != 0u;
  if (tyre && (lab.flags & F_TYRES) == 0u) { return o; }
  let pa = P(rec.x);
  let pb = P(rec.y);
  let r = rest[beam];
  let broken = r < 0.0;
  let L = abs(r);
  let L0 = bitcast<f32>(rec.w);
  if (broken && (lab.flags & F_STUBS) == 0u) { return o; }
  if (!broken && hf == 1u) { return o; }
  var a = pa;
  var b = pb;
  if (hf == 1u) { a = pb; b = pa; }
  if (broken) {
    let stub = min(0.3 * L0, 0.07);
    b = a + normalize(b - a + vec3f(1e-7, 0.0, 0.0)) * stub;
  }
  var radius = lab.tubeR * select(1.0, 0.55, tyre);
  tubeVertex(a, b, radius, k, &o);
  // Colour.
  let k_ = bitcast<f32>(inf.x);
  let strength = bitcast<f32>(inf.y);
  let len = length(pb - pa);
  let stress = k_ * abs(len - L) / max(strength, 1.0);
  let plastic = abs(L - L0) / max(L0, 1e-6);
  var c = vec3f(0.0);
  if (lab.colorMode == 0u) {
    let t = max(min(stress, 1.0), min(plastic / 0.05, 0.9));
    c = select(ramp(t), vec3f(0.93, 0.08, 0.07), broken);
    if (tyre) { c = vec3f(0.22, 0.23, 0.26); }
  } else if (lab.colorMode == 1u) {
    c = vec3f(0.46, 0.53, 0.63);
    if (plastic > 1e-3) { c = mix(vec3f(1.0, 0.85, 0.1), vec3f(1.0, 0.1, 0.05), clamp(plastic / 0.08, 0.0, 1.0)); }
    if (broken) { c = vec3f(0.95, 0.1, 0.9); }
    if (tyre) { c = vec3f(0.22, 0.23, 0.26); }
  } else {
    c = partColor(inf.w);
    if (broken) { c = c * 0.45; }
  }
  o.c = toLinear(c);
  return o;
}
@fragment fn fsTube(i: TubeOut) -> @location(0) vec4f {
  return vec4f(studio(i.c, normalize(i.n), i.w, i.gloss), 1.0);
}

// The grab line: a thin tube from the grabbed node to the target.
@vertex fn vsGrab(@builtin(vertex_index) k: u32) -> TubeOut {
  var o: TubeOut;
  o.gloss = 0.9;
  o.clip = HIDE; o.n = vec3f(0.0, 1.0, 0.0); o.w = vec3f(0.0);
  o.c = toLinear(vec3f(0.2, 0.95, 1.0));
  if (lab.grabNode == 0xffffffffu) { return o; }
  tubeVertex(P(lab.grabNode), lab.grabTarget.xyz, lab.tubeR * 0.6, k, &o);
  return o;
}

struct DotOut { @builtin(position) clip: vec4f, @location(0) uv: vec2f, @location(1) center: vec3f, @location(2) r: f32, @location(3) c: vec3f };

fn impostor(center: vec3f, r: f32, k: u32, c: vec3f) -> DotOut {
  var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let corner = corners[k];
  let f = normalize(center - frame.eye.xyz);
  let right = normalize(cross(f, vec3f(0.0, 1.0, 0.0)) + vec3f(1e-6, 0.0, 0.0));
  let up = cross(right, f);
  // Pulled toward the camera by r so the quad is not clipped by the sphere's own depth.
  let w = center + (right * corner.x + up * corner.y) * r * 1.05 - f * r;
  var o: DotOut;
  o.clip = frame.viewProj * vec4f(w, 1.0);
  o.uv = corner * 1.05;
  o.center = center;
  o.r = r;
  o.c = c;
  return o;
}
// Nodes: a camera-facing quad per node shaded as a sphere.
@vertex fn vsNode(@builtin(vertex_index) vall: u32) -> DotOut {
  let node = vall / 6u;
  let tyre = (nodeVeh[node] >> 16u) != 0u;
  var o = impostor(P(node), lab.nodeR * select(1.0, 0.6, tyre), vall % 6u, select(vec3f(0.96, 0.97, 1.0), vec3f(0.55, 0.57, 0.6), tyre));
  if ((lab.flags & F_NODES) == 0u || (tyre && (lab.flags & F_TYRES) == 0u)) { o.clip = HIDE; }
  return o;
}
@vertex fn vsGrabDot(@builtin(vertex_index) k: u32) -> DotOut {
  var o = impostor(lab.grabTarget.xyz, lab.nodeR * 1.4, k % 6u, vec3f(0.2, 0.95, 1.0));
  if (lab.grabNode == 0xffffffffu) { o.clip = HIDE; }
  return o;
}
struct DotFrag { @location(0) c: vec4f, @builtin(frag_depth) depth: f32 };
@fragment fn fsDot(i: DotOut) -> DotFrag {
  let q = dot(i.uv, i.uv);
  if (q > 1.0) { discard; }
  let f = normalize(i.center - frame.eye.xyz);
  let right = normalize(cross(f, vec3f(0.0, 1.0, 0.0)) + vec3f(1e-6, 0.0, 0.0));
  let up = cross(right, f);
  let n = normalize(right * i.uv.x + up * i.uv.y - f * sqrt(1.0 - q));
  let w = i.center + n * i.r;
  let clip = frame.viewProj * vec4f(w, 1.0);
  var o: DotFrag;
  o.c = vec4f(studio(toLinear(i.c), n, w, 0.8), 1.0);
  o.depth = clip.z / clip.w;
  return o;
}

// The collision skin: ghost (translucent grey, brighter at grazing angles) or
// solid (light grey paint); a triangle past 1.8× its spawn edges reads as torn and
// is dropped.
struct SkinOut { @builtin(position) clip: vec4f, @location(0) n: vec3f, @location(1) w: vec3f };
@vertex fn vsSkin(@builtin(vertex_index) vi: u32) -> SkinOut {
  let t = tris[(vi / 3u) * 2u];
  let r = bitcast<vec4f>(tris[(vi / 3u) * 2u + 1u]);
  let pa = P(t.x);
  let pb = P(t.y);
  let pc = P(t.z);
  let k = vi % 3u;
  let p = select(select(pc, pb, k == 1u), pa, k == 0u);
  let la = length(pa - pb) / max(r.x, 1e-4);
  let lb = length(pb - pc) / max(r.y, 1e-4);
  let lc = length(pc - pa) / max(r.z, 1e-4);
  let torn = max(max(la, lb), lc) > 1.8;
  var o: SkinOut;
  o.w = p;
  o.n = normalize(cross(pb - pa, pc - pa));
  o.clip = select(frame.viewProj * vec4f(p, 1.0), HIDE, torn || lab.skinMode == 0u);
  return o;
}
@fragment fn fsGhost(i: SkinOut) -> @location(0) vec4f {
  let v = normalize(frame.eye.xyz - i.w);
  let n = normalize(i.n);
  let fres = pow(1.0 - abs(dot(n, v)), 2.0);
  let c = studio(toLinear(vec3f(0.78, 0.8, 0.84)), n, i.w, 0.3);
  return vec4f(c, 0.1 + 0.32 * fres);
}
@fragment fn fsSolid(i: SkinOut) -> @location(0) vec4f {
  return vec4f(studio(toLinear(vec3f(0.74, 0.76, 0.8)), normalize(i.n), i.w, 0.6), 1.0);
}
@vertex fn vsSkinShadow(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let t = tris[(vi / 3u) * 2u];
  let k = vi % 3u;
  let p = P(select(select(t.z, t.y, k == 1u), t.x, k == 0u));
  return frame.lightVP * vec4f(p, 1.0);
}

// Studio: a navy gradient behind everything and a floor at the ground height with
// a faint metre grid, fading out with distance, shadowed by the car.
struct BgOut { @builtin(position) clip: vec4f, @location(0) ndc: vec2f };
@vertex fn vsBg(@builtin(vertex_index) vi: u32) -> BgOut {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u)) * 2.0 - 1.0;
  var o: BgOut; o.clip = vec4f(p, 0.99999, 1.0); o.ndc = p; return o;
}
fn backdrop(ndc: vec2f) -> vec3f {
  let top = vec3f(0.035, 0.055, 0.1);
  let mid = vec3f(0.07, 0.1, 0.17);
  let t = clamp(ndc.y * 0.5 + 0.5, 0.0, 1.0);
  let c = mix(mid, top, t);
  let vig = 1.0 - 0.35 * dot(ndc * vec2f(0.6, 0.8), ndc * vec2f(0.6, 0.8));
  return c * vig;
}
struct BgFrag { @location(0) c: vec4f, @builtin(frag_depth) depth: f32 };
@fragment fn fsBg(i: BgOut) -> BgFrag {
  let nearP = frame.invViewProj * vec4f(i.ndc, 0.0, 1.0);
  let farP = frame.invViewProj * vec4f(i.ndc, 1.0, 1.0);
  let a = nearP.xyz / nearP.w;
  let b = farP.xyz / farP.w;
  let d = b - a;
  // The floor plane (computed everywhere: the grid's derivatives need uniform flow).
  let hit = d.y < 0.0 && a.y > lab.floorY;
  let t = select(0.0, (lab.floorY - a.y) / min(d.y, -1e-6), hit);
  let w = a + d * t;
  let focus = vec2f(lab.focusX, lab.focusZ);
  let dist = length(w.xz - focus);
  let fade = exp(-dist * dist / 500.0);
  let fw = max(fwidth(w.xz), vec2f(1e-4));
  let g1 = abs(fract(w.xz) - 0.5) / fw;
  let g5 = abs(fract(w.xz / 5.0) - 0.5) / (fw / 5.0);
  let line1 = 1.0 - min(min(g1.x, g1.y) * 0.5, 1.0);
  let line5 = 1.0 - min(min(g5.x, g5.y) * 0.5, 1.0);
  var fl = vec3f(0.075, 0.1, 0.16) + vec3f(0.05, 0.07, 0.12) * exp(-dist * dist / 60.0);
  fl = fl + vec3f(0.03, 0.045, 0.075) * clamp(line1 * 0.45 + line5 * 0.8, 0.0, 1.0) * fade;
  var o: BgFrag;
  o.c = vec4f(backdrop(i.ndc), 1.0);
  o.depth = 0.99999;
  if (hit) {
    let clip = frame.viewProj * vec4f(w, 1.0);
    fl = fl * (0.35 + 0.65 * shadowAt(w));
    o.c = vec4f(mix(backdrop(i.ndc), fl, fade * 0.9 + 0.1), 1.0);
    o.depth = clamp(clip.z / clip.w, 0.0, 0.99999);
  }
  return o;
}

// Live state into the compact buffers: node world positions, beam rest lengths
// (negated when broken).
@group(0) @binding(0) var<storage, read> xnodes: array<vec4f>;
@group(0) @binding(11) var<storage, read> xmisc: array<vec4u>;
@group(0) @binding(12) var<storage, read> xbeams: array<vec4f>;
@group(0) @binding(13) var<storage, read_write> outPos: array<f32>;
@group(0) @binding(14) var<storage, read_write> outRest: array<f32>;
@group(0) @binding(15) var<uniform> counts: vec4u;
@compute @workgroup_size(256) fn gather(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i < counts.x) {
    let v = nodeVeh[i] & 0xffffu;
    let p = xnodes[i * 3u].xyz + bitcast<vec3f>(xmisc[v].xyz);
    outPos[i * 3u] = p.x; outPos[i * 3u + 1u] = p.y; outPos[i * 3u + 2u] = p.z;
    return;
  }
  let j = i - counts.x;
  if (j < counts.y) {
    let s0 = xbeams[j * 2u];
    let s1 = xbeams[j * 2u + 1u];
    outRest[j] = select(s0.x, -max(abs(s0.x), 1e-6), s1.z != 0.0);
  }
}
`;

/**
 * @param {object} o
 * @param {GPUDevice} o.device
 * @param {GPUTextureFormat} o.format
 * @param {number} o.sampleCount
 * @param {GPUBuffer} o.frameBuf the page's frame uniform (viewProj, eye, light, opts, lightVP, invViewProj)
 * @param {GPUTextureView} o.shadowView
 * @param {GPUSampler} o.shadowSampler
 */
export function createCrashLabRenderer({ device, format, sampleCount, frameBuf, shadowView, shadowSampler }) {
  let sim = null, layout = null, mod = null, pipes = null, bufs = null, groups = null, sides = 0;
  const labBuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const countsBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const opts = { colorMode: 'strain', skin: 'ghost', skeleton: true, nodes: true, stubs: true, tyres: true, studio: true, tubeR: 0.009, nodeR: 0.017, grab: null, floorY: 0, focus: [0, 0] };
  const depthStencil = { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' };

  const build = (n) => {
    sides = n;
    mod = device.createShaderModule({ label: 'crash-lab', code: SHADER(n) });
    const target = (blend) => [{ format, ...(blend ? { blend: { color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } } } : {}) }];
    const rp = (vs, fs, { blend = false, write = true, cull = 'none' } = {}) => device.createRenderPipeline({
      layout: 'auto', vertex: { module: mod, entryPoint: vs }, fragment: { module: mod, entryPoint: fs, targets: target(blend) },
      primitive: { topology: 'triangle-list', cullMode: cull }, depthStencil: { ...depthStencil, depthWriteEnabled: write }, multisample: { count: sampleCount },
    });
    pipes = {
      tube: rp('vsTube', 'fsTube'),
      grab: rp('vsGrab', 'fsTube'),
      node: rp('vsNode', 'fsDot'),
      grabDot: rp('vsGrabDot', 'fsDot'),
      solid: rp('vsSkin', 'fsSolid'),
      ghost: rp('vsSkin', 'fsGhost', { blend: true, write: false }),
      bg: device.createRenderPipeline({
        layout: 'auto', vertex: { module: mod, entryPoint: 'vsBg' }, fragment: { module: mod, entryPoint: 'fsBg', targets: target(false) },
        primitive: { cullMode: 'none' }, depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less-equal' }, multisample: { count: sampleCount },
      }),
      shadow: device.createRenderPipeline({
        layout: 'auto', vertex: { module: mod, entryPoint: 'vsSkinShadow' }, primitive: { cullMode: 'none' },
        depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBias: 2, depthBiasSlopeScale: 2.5 },
      }),
      gather: device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'gather' } }),
    };
  };

  const bindFor = (p, list) => device.createBindGroup({
    layout: p.getBindGroupLayout(0),
    entries: list.map(([b, r]) => ({ binding: b, resource: r instanceof GPUBuffer ? { buffer: r } : r })),
  });

  /** Bind a (re)built ExplicitSim: its layout, buffers and lab records. */
  function setSim(s, l) {
    sim = s; layout = l;
    const want = l.beams > 40000 ? 4 : 8;
    if (want !== sides) build(want);
    bufs?.pos.destroy(); bufs?.rest.destroy(); bufs?.nodeVeh.destroy();
    const nodeVeh = s.labNodeVehicles();
    bufs = {
      pos: device.createBuffer({ size: Math.max(16, l.nodes * 12), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
      rest: device.createBuffer({ size: Math.max(16, l.beams * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
      nodeVeh: device.createBuffer({ size: Math.max(16, nodeVeh.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
    };
    device.queue.writeBuffer(bufs.nodeVeh, 0, nodeVeh);
    device.queue.writeBuffer(countsBuf, 0, new Uint32Array([l.nodes, l.beams, 0, 0]));
    const recs = s.beamRecordsBuffer(), info = s.labBeamsBuffer(), tris = s.trisBuffer();
    const common = [[1, frameBuf], [2, labBuf], [3, bufs.pos]];
    groups = {
      tube: bindFor(pipes.tube, [...common, [4, bufs.rest], [5, recs], [6, info]]),
      grab: bindFor(pipes.grab, common),
      node: bindFor(pipes.node, [...common, [8, bufs.nodeVeh]]),
      grabDot: bindFor(pipes.grabDot, [[1, frameBuf], [2, labBuf]]),
      skin: bindFor(pipes.solid, [...common, [7, tris]]),
      ghost: bindFor(pipes.ghost, [...common, [7, tris]]),
      bg: bindFor(pipes.bg, [[1, frameBuf], [2, labBuf], [9, shadowView], [10, shadowSampler]]),
      shadow: bindFor(pipes.shadow, [[1, frameBuf], [3, bufs.pos], [7, tris]]),
      gather: bindFor(pipes.gather, [[0, s.nodesBuffer()], [8, bufs.nodeVeh], [11, s.miscBuffer()], [12, s.beamsBuffer()], [13, bufs.pos], [14, bufs.rest], [15, countsBuf]]),
    };
    writeOpts();
  }

  function writeOpts() {
    const u = new ArrayBuffer(64), w = new Uint32Array(u), f = new Float32Array(u);
    const flags = (opts.skeleton ? 1 : 0) | (opts.nodes ? 2 : 0) | (opts.stubs ? 4 : 0) | (opts.studio ? 8 : 0) | (opts.tyres ? 16 : 0);
    w[0] = Math.max(0, COLOR_MODES.indexOf(opts.colorMode)); w[1] = Math.max(0, SKIN_MODES.indexOf(opts.skin)); w[2] = flags;
    w[3] = opts.grab ? opts.grab.node : 0xffffffff;
    f[4] = opts.tubeR; f[5] = opts.nodeR; w[6] = layout?.beams ?? 0; w[7] = layout?.nodes ?? 0;
    if (opts.grab) f.set([...opts.grab.target, 1], 8);
    f[12] = opts.floorY; f[13] = opts.focus[0]; f[14] = opts.focus[1];
    device.queue.writeBuffer(labBuf, 0, u);
  }

  return {
    opts,
    setSim,
    /** Change options (see `opts`) and upload them. */
    set(o) { Object.assign(opts, o); writeOpts(); },
    /** The compact node positions (`array<f32>`, world x, y, z per node, global
     * order; live or replayed): what a textured body skinned to the nodes reads
     * (the showcase car renderer's input), so it follows replays too. */
    positionsBuffer() { return bufs?.pos ?? null; },
    /** Live frame: copy the solver's state into the compact buffers. */
    gather(enc) {
      if (!sim) return;
      const p = enc.beginComputePass({ label: 'crash-lab-gather' });
      p.setPipeline(pipes.gather); p.setBindGroup(0, groups.gather);
      p.dispatchWorkgroups(Math.ceil((layout.nodes + layout.beams) / 256));
      p.end();
    },
    /** Replay frame: a recorded state ({ pos, rest } from readFrameState). */
    upload(state) {
      device.queue.writeBuffer(bufs.pos, 0, state.pos);
      device.queue.writeBuffer(bufs.rest, 0, state.rest);
    },
    drawShadow(sp) {
      if (!sim || !layout.triangles) return;
      sp.setPipeline(pipes.shadow); sp.setBindGroup(0, groups.shadow); sp.draw(layout.triangles * 3);
    },
    /** The studio backdrop and floor (first, in place of the sky and ground). */
    drawBackground(rp) {
      rp.setPipeline(pipes.bg); rp.setBindGroup(0, groups.bg); rp.draw(3);
    },
    drawOpaque(rp) {
      if (!sim) return;
      if (opts.skin === 'solid' && layout.triangles) { rp.setPipeline(pipes.solid); rp.setBindGroup(0, groups.skin); rp.draw(layout.triangles * 3); }
      if (opts.skeleton) { rp.setPipeline(pipes.tube); rp.setBindGroup(0, groups.tube); rp.draw(layout.beams * 2 * sides * 6); }
      if (opts.nodes) { rp.setPipeline(pipes.node); rp.setBindGroup(0, groups.node); rp.draw(layout.nodes * 6); }
      if (opts.grab) {
        rp.setPipeline(pipes.grab); rp.setBindGroup(0, groups.grab); rp.draw(sides * 6);
        rp.setPipeline(pipes.grabDot); rp.setBindGroup(0, groups.grabDot); rp.draw(6);
      }
    },
    drawTransparent(rp) {
      if (!sim || opts.skin !== 'ghost' || !layout.triangles) return;
      rp.setPipeline(pipes.ghost); rp.setBindGroup(0, groups.ghost); rp.draw(layout.triangles * 3);
    },
  };
}
