//! f64 vector helpers for scene building (JS numbers were f64; values are
//! rounded to f32 only when packed into GPU records).

pub type V3 = [f64; 3];
pub type Q = [f64; 4];

pub const IDENTITY: Q = [0.0, 0.0, 0.0, 1.0];

pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
pub fn length(a: V3) -> f64 {
    dot(a, a).sqrt()
}
pub fn norm(a: V3) -> V3 {
    let l = length(a);
    scale(a, 1.0 / if l == 0.0 { 1.0 } else { l })
}
pub fn rotate(q: Q, v: V3) -> V3 {
    let u = [q[0], q[1], q[2]];
    add(v, scale(cross(u, add(cross(u, v), scale(v, q[3]))), 2.0))
}
pub fn inverse_rotate(q: Q, v: V3) -> V3 {
    rotate(conjugate(q), v)
}
pub fn conjugate(q: Q) -> Q {
    [-q[0], -q[1], -q[2], q[3]]
}
pub fn quat_mul(a: Q, b: Q) -> Q {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}
pub fn f64x3(a: [f32; 3]) -> V3 {
    [a[0] as f64, a[1] as f64, a[2] as f64]
}
pub fn f32x3(a: V3) -> [f32; 3] {
    [a[0] as f32, a[1] as f32, a[2] as f32]
}
pub fn f32x4(a: [f64; 4]) -> [f32; 4] {
    [a[0] as f32, a[1] as f32, a[2] as f32, a[3] as f32]
}
