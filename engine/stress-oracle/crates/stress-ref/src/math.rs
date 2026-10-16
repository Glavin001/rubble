//! Small double-precision linear algebra: 3-vectors, 3x3 matrices, unit quaternions.
//!
//! Everything in the reference solver is `f64`. The types are plain `Copy` values
//! with operator overloads; nothing here allocates.

use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };
    pub const X: Vec3 = Vec3 { x: 1.0, y: 0.0, z: 0.0 };
    pub const Y: Vec3 = Vec3 { x: 0.0, y: 1.0, z: 0.0 };
    pub const Z: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 1.0 };

    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }
    pub fn splat(v: f64) -> Self {
        Vec3::new(v, v, v)
    }
    pub fn from_array(a: [f64; 3]) -> Self {
        Vec3::new(a[0], a[1], a[2])
    }
    pub fn to_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
    pub fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    pub fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z, self.x * o.y - self.y * o.x)
    }
    pub fn norm2(self) -> f64 {
        self.dot(self)
    }
    pub fn norm(self) -> f64 {
        self.norm2().sqrt()
    }
    /// Unit vector, or zero for a (near-)zero input.
    pub fn normalized(self) -> Vec3 {
        let n = self.norm();
        if n > 1e-300 {
            self / n
        } else {
            Vec3::ZERO
        }
    }
    pub fn mul_elem(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x * o.x, self.y * o.y, self.z * o.z)
    }
    pub fn abs(self) -> Vec3 {
        Vec3::new(self.x.abs(), self.y.abs(), self.z.abs())
    }
    pub fn max_elem(self) -> f64 {
        self.x.max(self.y).max(self.z)
    }
    pub fn min_elem(self) -> f64 {
        self.x.min(self.y).min(self.z)
    }
    pub fn component_min(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x.min(o.x), self.y.min(o.y), self.z.min(o.z))
    }
    pub fn component_max(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x.max(o.x), self.y.max(o.y), self.z.max(o.z))
    }
    /// Some unit vector perpendicular to `self` (assumed unit), chosen deterministically.
    pub fn any_perpendicular(self) -> Vec3 {
        let a = self.abs();
        let helper = if a.x <= a.y && a.x <= a.z {
            Vec3::X
        } else if a.y <= a.z {
            Vec3::Y
        } else {
            Vec3::Z
        };
        self.cross(helper).normalized()
    }
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}
impl Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
}
impl Mul<Vec3> for f64 {
    type Output = Vec3;
    fn mul(self, v: Vec3) -> Vec3 {
        v * self
    }
}
impl Div<f64> for Vec3 {
    type Output = Vec3;
    fn div(self, s: f64) -> Vec3 {
        Vec3::new(self.x / s, self.y / s, self.z / s)
    }
}
impl AddAssign for Vec3 {
    fn add_assign(&mut self, o: Vec3) {
        *self = *self + o;
    }
}
impl SubAssign for Vec3 {
    fn sub_assign(&mut self, o: Vec3) {
        *self = *self - o;
    }
}
impl MulAssign<f64> for Vec3 {
    fn mul_assign(&mut self, s: f64) {
        *self = *self * s;
    }
}
impl Index<usize> for Vec3 {
    type Output = f64;
    fn index(&self, i: usize) -> &f64 {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Vec3 index {i} out of range"),
        }
    }
}
impl IndexMut<usize> for Vec3 {
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Vec3 index {i} out of range"),
        }
    }
}

/// Row-major 3x3 matrix.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Mat3 {
    pub m: [[f64; 3]; 3],
}

impl Mat3 {
    pub const ZERO: Mat3 = Mat3 { m: [[0.0; 3]; 3] };
    pub const IDENTITY: Mat3 = Mat3 { m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] };

    pub fn diag(d: Vec3) -> Mat3 {
        Mat3 { m: [[d.x, 0.0, 0.0], [0.0, d.y, 0.0], [0.0, 0.0, d.z]] }
    }
    /// Matrix whose columns are `a`, `b`, `c`.
    pub fn from_cols(a: Vec3, b: Vec3, c: Vec3) -> Mat3 {
        Mat3 { m: [[a.x, b.x, c.x], [a.y, b.y, c.y], [a.z, b.z, c.z]] }
    }
    pub fn col(&self, j: usize) -> Vec3 {
        Vec3::new(self.m[0][j], self.m[1][j], self.m[2][j])
    }
    pub fn transpose(&self) -> Mat3 {
        let m = &self.m;
        Mat3 { m: [[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]] }
    }
    /// The cross-product matrix `[v]x` with `[v]x w = v x w`.
    pub fn skew(v: Vec3) -> Mat3 {
        Mat3 { m: [[0.0, -v.z, v.y], [v.z, 0.0, -v.x], [-v.y, v.x, 0.0]] }
    }
    /// `a b^T`.
    pub fn outer(a: Vec3, b: Vec3) -> Mat3 {
        Mat3 {
            m: [
                [a.x * b.x, a.x * b.y, a.x * b.z],
                [a.y * b.x, a.y * b.y, a.y * b.z],
                [a.z * b.x, a.z * b.y, a.z * b.z],
            ],
        }
    }
    pub fn det(&self) -> f64 {
        let m = &self.m;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }
    pub fn inverse(&self) -> Option<Mat3> {
        let d = self.det();
        if d.abs() < 1e-300 {
            return None;
        }
        let m = &self.m;
        let inv = [
            [
                m[1][1] * m[2][2] - m[1][2] * m[2][1],
                m[0][2] * m[2][1] - m[0][1] * m[2][2],
                m[0][1] * m[1][2] - m[0][2] * m[1][1],
            ],
            [
                m[1][2] * m[2][0] - m[1][0] * m[2][2],
                m[0][0] * m[2][2] - m[0][2] * m[2][0],
                m[0][2] * m[1][0] - m[0][0] * m[1][2],
            ],
            [
                m[1][0] * m[2][1] - m[1][1] * m[2][0],
                m[0][1] * m[2][0] - m[0][0] * m[2][1],
                m[0][0] * m[1][1] - m[0][1] * m[1][0],
            ],
        ];
        let mut out = Mat3 { m: inv };
        for r in 0..3 {
            for c in 0..3 {
                out.m[r][c] /= d;
            }
        }
        Some(out)
    }
    pub fn trace(&self) -> f64 {
        self.m[0][0] + self.m[1][1] + self.m[2][2]
    }
    pub fn is_finite(&self) -> bool {
        self.m.iter().all(|r| r.iter().all(|v| v.is_finite()))
    }
}

impl Mul<Vec3> for Mat3 {
    type Output = Vec3;
    fn mul(self, v: Vec3) -> Vec3 {
        let m = &self.m;
        Vec3::new(
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
        )
    }
}
impl Mul for Mat3 {
    type Output = Mat3;
    fn mul(self, o: Mat3) -> Mat3 {
        let mut r = Mat3::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                r.m[i][j] = (0..3).map(|k| self.m[i][k] * o.m[k][j]).sum();
            }
        }
        r
    }
}
impl Mul<f64> for Mat3 {
    type Output = Mat3;
    fn mul(self, s: f64) -> Mat3 {
        let mut r = self;
        r.m.iter_mut().flatten().for_each(|v| *v *= s);
        r
    }
}
impl Add for Mat3 {
    type Output = Mat3;
    fn add(self, o: Mat3) -> Mat3 {
        let mut r = self;
        for i in 0..3 {
            for j in 0..3 {
                r.m[i][j] += o.m[i][j];
            }
        }
        r
    }
}
impl Sub for Mat3 {
    type Output = Mat3;
    fn sub(self, o: Mat3) -> Mat3 {
        self + o * -1.0
    }
}
impl AddAssign for Mat3 {
    fn add_assign(&mut self, o: Mat3) {
        *self = *self + o;
    }
}

/// Unit quaternion `w + xi + yj + zk` representing a rotation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Default for Quat {
    fn default() -> Self {
        Quat::IDENTITY
    }
}

impl Quat {
    pub const IDENTITY: Quat = Quat { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };

    /// From `[w, x, y, z]` (the scene-file order).
    pub fn from_wxyz(a: [f64; 4]) -> Quat {
        Quat { w: a[0], x: a[1], y: a[2], z: a[3] }.normalized()
    }
    pub fn to_wxyz(self) -> [f64; 4] {
        [self.w, self.x, self.y, self.z]
    }
    pub fn from_axis_angle(axis: Vec3, angle: f64) -> Quat {
        let a = axis.normalized();
        let (s, c) = (0.5 * angle).sin_cos();
        Quat { w: c, x: a.x * s, y: a.y * s, z: a.z * s }
    }
    /// The squares summed in pairs, `(w^2 + x^2) + (y^2 + z^2)`: a half turn about a
    /// coordinate axis permutes the components with signs and keeps or swaps the pairs,
    /// so the sum rounds identically (DECISIONS.md 17).
    /// The rotation by the vector `v` (axis `v / |v|`, angle `|v|`); the identity for zero.
    pub fn from_rotation_vector(v: Vec3) -> Quat {
        let angle = v.norm();
        if angle == 0.0 {
            return Quat::IDENTITY;
        }
        Quat::from_axis_angle(v, angle)
    }
    /// The rotation vector (axis times angle, the angle in `[0, pi]`) of this rotation.
    pub fn to_rotation_vector(self) -> Vec3 {
        let q = if self.w < 0.0 { Quat { w: -self.w, x: -self.x, y: -self.y, z: -self.z } } else { self };
        let v = Vec3::new(q.x, q.y, q.z);
        let s = v.norm();
        if s == 0.0 {
            return Vec3::ZERO;
        }
        v * (2.0 * s.atan2(q.w) / s)
    }
    pub fn normalized(self) -> Quat {
        let n = ((self.w * self.w + self.x * self.x) + (self.y * self.y + self.z * self.z)).sqrt();
        Quat { w: self.w / n, x: self.x / n, y: self.y / n, z: self.z / n }
    }
    pub fn conjugate(self) -> Quat {
        Quat { w: self.w, x: -self.x, y: -self.y, z: -self.z }
    }
    /// The rotation matrix, its entries grouped so that a half turn about a coordinate
    /// axis (which permutes the quaternion's components with signs) changes them only by
    /// the exact sign flips of `D R`, `D = diag(+-1)`: the diagonal as differences and
    /// sums of the pairs `(w, x)`, `(y, z)`, the rest as two products.
    pub fn to_mat3(self) -> Mat3 {
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);
        let (ww, xx, yy, zz) = (w * w, x * x, y * y, z * z);
        let (xy, xz, yz, wx, wy, wz) = (x * y, x * z, y * z, w * x, w * y, w * z);
        Mat3 {
            m: [
                [(ww + xx) - (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), (ww - xx) + (yy - zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), (ww - xx) - (yy - zz)],
            ],
        }
    }
    pub fn rotate(self, v: Vec3) -> Vec3 {
        self.to_mat3() * v
    }
    /// Advance by angular velocity `omega` (world frame) over `dt` using the exact
    /// rotation for that constant velocity.
    pub fn integrate(self, omega: Vec3, dt: f64) -> Quat {
        let angle = omega.norm() * dt;
        if angle < 1e-300 {
            return self;
        }
        (Quat::from_axis_angle(omega, angle) * self).normalized()
    }
}

/// Hamilton product, each component's four terms summed in pairs: then a half turn
/// about a coordinate axis applied to either factor (left: a permutation of components
/// with signs; right: a body-frame factor, unchanged) maps every pair onto a pair, so the
/// product rounds identically up to sign (checked exhaustively on random factors; the
/// sequential sums reversed their order and did not).
impl Mul for Quat {
    type Output = Quat;
    fn mul(self, o: Quat) -> Quat {
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);
        Quat {
            w: (w * o.w - x * o.x) - (y * o.y + z * o.z),
            x: (w * o.x + x * o.w) + (y * o.z - z * o.y),
            y: (w * o.y - x * o.z) + (y * o.w + z * o.x),
            z: (w * o.z + x * o.y) - (y * o.x - z * o.w),
        }
    }
}

/// Rigid pose: `world = position + rotation * local`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pose {
    pub position: Vec3,
    pub rotation: Quat,
}

impl Pose {
    pub fn new(position: Vec3, rotation: Quat) -> Pose {
        Pose { position, rotation }
    }
    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        self.position + self.rotation.rotate(p)
    }
    pub fn transform_vector(&self, v: Vec3) -> Vec3 {
        self.rotation.rotate(v)
    }
    pub fn inverse_transform_point(&self, p: Vec3) -> Vec3 {
        self.rotation.conjugate().rotate(p - self.position)
    }
    pub fn inverse_transform_vector(&self, v: Vec3) -> Vec3 {
        self.rotation.conjugate().rotate(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_and_rotation_round_trip() {
        let m = Mat3 { m: [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]] };
        let i = m.inverse().unwrap();
        let p = m * i;
        for r in 0..3 {
            for c in 0..3 {
                assert!((p.m[r][c] - if r == c { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
        let q = Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0), 0.7);
        let v = Vec3::new(0.3, -1.0, 2.0);
        let back = q.conjugate().rotate(q.rotate(v));
        assert!((back - v).norm() < 1e-12);
        assert!(((q.to_mat3() * v) - q.rotate(v)).norm() < 1e-12);
    }
}
