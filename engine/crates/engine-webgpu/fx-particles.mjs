// GPU debris particles shared by the car renderers (car-render.mjs: the
// procedural soft-body car; showcase-car-render.mjs: the authored showcase
// vehicles): one ring buffer of particles, a spawn function the host's own
// event kernels call, the step (gravity, drag, a bouncing floor) and the
// camera-facing draw. Visual only: nothing feeds back into the physics.
//
// The host declares the buffer and its bindings, and supplies:
//   compute: `var<storage, read_write> fx: FxParticles;` and a seed (u32) for fxRnd;
//   draw:    `var<storage, read> fx: FxParticlesRead;`, `fn tonemap(c: vec3f) -> vec3f`
//            and `fn fxChip(kind: u32, h: f32) -> vec3f` (a chip's display colour from
//            the kind word's upper bits; h a per-particle random in 0..1).
//
// Kinds (low byte of `kind`; the upper 24 bits are the host's):
//   0 spark: a hot streak along its velocity, short-lived, emissive;
//   1 glass: a bright square (car-render.mjs's crumple shards);
//   2 chip: a tumbling flake of paint or plastic (fxChip colours it);
//   3 dust: a soft puff that drifts up and fades;
//   4 shard: a tumbling glass or lens granule with a glint; its tint in bits
//     8..15 (0 clear glass, 1 privacy glass, 2 red lens, 3 clear lens);
//   5 fleck: a small chip (4-12 mm) of paint or plastic (fxChip colours it).

/** The particle record and ring buffer (`n` particles). */
export const FX_STRUCT_WGSL = (n) => /* wgsl */`
struct FxParticle { pos: vec3f, life: f32, vel: vec3f, kind: u32 };
struct FxParticles { head: atomic<u32>, pad0: u32, pad1: u32, pad2: u32, p: array<FxParticle, ${n}> };
struct FxParticlesRead { head: u32, pad0: u32, pad1: u32, pad2: u32, p: array<FxParticle, ${n}> };
`;

/** Bytes of the ring buffer for `n` particles. */
export const fxBufferSize = (n) => 16 + n * 32;

/**
 * Compute-side helpers: fxRnd(a, b) (hash of two words and `seed`, in 0..1),
 * fxRndDir(a, b) (a unit vector), fxSpawn(pos, vel, life, kind) (writes the
 * ring buffer's next slot), and fxStep(i, dt, floorY) (one particle's step).
 */
export const FX_COMPUTE_WGSL = (n, seed) => /* wgsl */`
fn fxRnd(a: u32, b: u32) -> f32 {
  var x = a * 747796405u + b * 2891336453u + (${seed}) * 1664525u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return f32((x >> 22u) ^ x) / 4294967295.0;
}
fn fxRndDir(a: u32, b: u32) -> vec3f {
  let z = fxRnd(a, b) * 2.0 - 1.0;
  let ph = fxRnd(a, b + 7u) * 6.28318;
  let r = sqrt(max(1.0 - z * z, 0.0));
  return vec3f(r * cos(ph), z, r * sin(ph));
}
fn fxSpawn(pos: vec3f, vel: vec3f, life: f32, kind: u32) {
  let i = atomicAdd(&fx.head, 1u) % ${n}u;
  fx.p[i] = FxParticle(pos, life, vel, kind);
}
// Gravity, drag, a bouncing floor at floorY.
fn fxStep(i: u32, dt: f32, floorY: f32) {
  if (i >= ${n}u) { return; }
  var q = fx.p[i];
  if (q.life <= 0.0 || dt <= 0.0) { return; }
  let kind = q.kind & 255u;
  let drag = select(select(0.4, 3.0, kind == 3u), 1.2, kind == 0u);
  q.vel = q.vel * exp(-drag * dt) + vec3f(0.0, select(-9.81, 0.4, kind == 3u), 0.0) * dt;
  q.pos += q.vel * dt;
  if (q.pos.y < floorY + 0.01) {
    q.pos.y = floorY + 0.01;
    q.vel = vec3f(q.vel.x * 0.55, abs(q.vel.y) * select(0.25, 0.4, kind == 1u || kind == 4u), q.vel.z * 0.55);
    if (kind == 0u) { q.life -= 0.1; }
  }
  q.life -= dt;
  fx.p[i] = q;
}
`;

/**
 * Draw-side: fxVertex(vi, inst, eye, viewProj) (6 vertices per particle,
 * instanced; a dead particle collapses outside the view) and fxFragment.
 */
export const FX_DRAW_WGSL = /* wgsl */`
struct FxOut { @builtin(position) clip: vec4f, @location(0) c: vec4f, @location(1) uv: vec2f };
fn fxVertex(vi: u32, inst: u32, eye: vec3f, viewProj: mat4x4f) -> FxOut {
  let q = fx.p[inst];
  let kind = q.kind & 255u;
  var o: FxOut;
  let corner = array<vec2f, 6>(vec2f(-1, -1), vec2f(1, -1), vec2f(1, 1), vec2f(-1, -1), vec2f(1, 1), vec2f(-1, 1))[vi];
  o.uv = corner;
  let toEye = normalize(eye - q.pos);
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
  } else if (kind == 2u || kind == 5u) {
    let s = select(0.025 + 0.04 * h, 0.004 + 0.008 * h, kind == 5u);
    let ang = h * 6.28 + q.life * 4.0;
    let r2 = right * cos(ang) + up * sin(ang);
    let u2 = cross(r2, toEye);
    p = q.pos + (r2 * corner.x + u2 * corner.y * 0.6) * s;
    o.c = vec4f(fxChip(q.kind, h), 1.0);
  } else if (kind == 4u) {
    // A granule: a small tumbling chip of glass; it glints as it turns
    // (a sharp flash when its face crosses the eye), dull in between.
    let s = 0.006 + 0.012 * h;
    let ang = h * 6.28 + q.life * 9.0 * (0.5 + h);
    let r2 = right * cos(ang) + up * sin(ang);
    let u2 = cross(r2, toEye);
    p = q.pos + (r2 * corner.x + u2 * corner.y * (0.5 + 0.5 * h)) * s;
    let tint = (q.kind >> 8u) & 255u;
    var base = vec3f(0.55, 0.62, 0.6);
    if (tint == 1u) { base = vec3f(0.12, 0.13, 0.14); }
    if (tint == 2u) { base = vec3f(0.55, 0.03, 0.02); }
    if (tint == 3u) { base = vec3f(0.8, 0.82, 0.85); }
    let glint = pow(abs(sin(q.life * 7.0 * (1.0 + h) + h * 40.0)), 24.0) * 2.5;
    let fade = clamp(q.life * 0.5, 0.0, 1.0);
    o.c = vec4f(tonemap(base * (0.5 + glint)) * fade, fade);
  } else {
    let s = 0.1 + (1.0 - q.life) * 0.3;
    p = q.pos + (right * corner.x + up * corner.y) * s;
    let a = 0.1 * clamp(q.life, 0.0, 1.0);
    o.c = vec4f(vec3f(0.45, 0.43, 0.4) * a, a);
  }
  o.clip = select(viewProj * vec4f(p, 1.0), vec4f(0.0, 0.0, 2.0, 1.0), q.life <= 0.0);
  return o;
}
fn fxFragment(i: FxOut) -> vec4f {
  let r = length(i.uv);
  if (i.c.a > 0.0 && i.c.a < 0.5 && r > 1.0) { discard; } // round dust
  return i.c;
}
`;

/** The draw's blend (premultiplied: sparks add, chips and shards cover). */
export const FX_BLEND = {
  color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
};
