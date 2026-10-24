//! Transforms between the user's body frame (the body origin, as in Box3D)
//! and the solver's frame (the centre of mass on principal axes, which the GPU
//! body record stores), and the principal decomposition of an inertia tensor.

use glam::{DMat3, DQuat, DVec3, Mat3, Quat, Vec3};

/// A rigid transform: position and rotation (x, y, z, w).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub position: [f32; 3],
    pub rotation: [f32; 4],
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    pub const IDENTITY: Self = Self {
        position: [0.0; 3],
        rotation: [0.0, 0.0, 0.0, 1.0],
    };

    pub fn new(position: [f32; 3], rotation: [f32; 4]) -> Self {
        Self { position, rotation }
    }

    pub(crate) fn p(&self) -> Vec3 {
        Vec3::from(self.position)
    }

    pub(crate) fn q(&self) -> Quat {
        Quat::from_array(self.rotation)
    }
}

/// The solver frame in the body frame: the centre of mass and the principal
/// axes. Identity for statics and for bodies whose shapes are centred and
/// aligned; then no conversion touches a value, bit for bit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SolverFrame {
    pub center: Vec3,
    pub axes: Quat,
}

impl Default for SolverFrame {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl SolverFrame {
    pub const IDENTITY: Self = Self {
        center: Vec3::ZERO,
        axes: Quat::IDENTITY,
    };

    pub fn is_identity(self) -> bool {
        self == Self::IDENTITY
    }

    /// Body origin pose -> solver pose.
    pub fn to_solver(self, origin: Transform) -> Transform {
        if self.is_identity() {
            return origin;
        }
        let q = origin.q();
        Transform {
            position: (origin.p() + q * self.center).to_array(),
            rotation: (q * self.axes).normalize().to_array(),
        }
    }

    /// Solver pose -> body origin pose.
    pub fn to_origin(self, solver: Transform) -> Transform {
        if self.is_identity() {
            return solver;
        }
        let q = (solver.q() * self.axes.conjugate()).normalize();
        Transform {
            position: (solver.p() - q * self.center).to_array(),
            rotation: q.to_array(),
        }
    }

    /// A pose given in the body frame (a shape, a joint frame) -> solver frame.
    pub fn local_to_solver(self, local: Transform) -> Transform {
        if self.is_identity() {
            return local;
        }
        let inv = self.axes.conjugate();
        Transform {
            position: (inv * (local.p() - self.center)).to_array(),
            rotation: (inv * local.q()).normalize().to_array(),
        }
    }

    /// The command that moves a body's solver frame from `self` to `next`:
    /// offset of the new centre in the old solver frame, relative rotation.
    pub fn delta_to(&self, next: &SolverFrame) -> (Vec3, Quat) {
        let inv = self.axes.conjugate();
        (
            inv * (next.center - self.center),
            (inv * next.axes).normalize(),
        )
    }
}

/// Principal moments (ascending axis order as found) and the rotation whose
/// columns are the principal axes: `tensor = R diag(m) R^T`. A diagonal tensor
/// keeps the identity rotation exactly.
pub(crate) fn principal(tensor: Mat3) -> (Vec3, Quat) {
    let off = [tensor.y_axis.x, tensor.z_axis.x, tensor.z_axis.y];
    let diag = Vec3::new(tensor.x_axis.x, tensor.y_axis.y, tensor.z_axis.z);
    if off.iter().all(|&v| v == 0.0) {
        return (diag, Quat::IDENTITY);
    }
    // Cyclic Jacobi in f64.
    let mut a = tensor.as_dmat3();
    let mut v = DMat3::IDENTITY;
    for _ in 0..32 {
        let (p, q) = largest_off_diagonal(&a);
        let apq = a.col(q)[p];
        if apq.abs() < 1e-15 * (a.col(p)[p].abs() + a.col(q)[q].abs()).max(1e-300) {
            break;
        }
        let theta = (a.col(q)[q] - a.col(p)[p]) / (2.0 * apq);
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let t = if theta == 0.0 { 1.0 } else { t };
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;
        let mut j = DMat3::IDENTITY;
        set(&mut j, p, p, c);
        set(&mut j, q, q, c);
        set(&mut j, p, q, s);
        set(&mut j, q, p, -s);
        a = j.transpose() * a * j;
        v *= j;
    }
    if v.determinant() < 0.0 {
        v.z_axis = -v.z_axis;
    }
    let moments = DVec3::new(a.x_axis.x, a.y_axis.y, a.z_axis.z).as_vec3();
    (moments, DQuat::from_mat3(&v).normalize().as_quat())
}

fn largest_off_diagonal(a: &DMat3) -> (usize, usize) {
    let mut best = (0, 1);
    let mut value = -1.0;
    for (p, q) in [(0, 1), (0, 2), (1, 2)] {
        let x = a.col(q)[p].abs();
        if x > value {
            value = x;
            best = (p, q);
        }
    }
    best
}

fn set(m: &mut DMat3, row: usize, col: usize, value: f64) {
    let mut c = m.col(col);
    c[row] = value;
    match col {
        0 => m.x_axis = c,
        1 => m.y_axis = c,
        _ => m.z_axis = c,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn principal_recovers_a_rotated_diagonal() {
        let r = Quat::from_euler(glam::EulerRot::XYZ, 0.3, -0.7, 1.1);
        let m = Mat3::from_quat(r);
        let d = Vec3::new(1.0, 2.0, 5.0);
        let tensor = m * Mat3::from_diagonal(d) * m.transpose();
        let (moments, axes) = principal(tensor);
        let back = Mat3::from_quat(axes)
            * Mat3::from_diagonal(moments)
            * Mat3::from_quat(axes).transpose();
        for c in 0..3 {
            assert!((back.col(c) - tensor.col(c)).abs().max_element() < 1e-4);
        }
        let mut sorted = moments.to_array();
        sorted.sort_by(f32::total_cmp);
        assert!((Vec3::from(sorted) - d).abs().max_element() < 1e-4);
    }

    #[test]
    fn diagonal_keeps_identity_exactly() {
        let (m, q) = principal(Mat3::from_diagonal(Vec3::new(1.0, 2.0, 3.0)));
        assert_eq!(q, Quat::IDENTITY);
        assert_eq!(m, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn frames_round_trip() {
        let f = SolverFrame {
            center: Vec3::new(0.3, -0.2, 0.5),
            axes: Quat::from_rotation_z(0.4),
        };
        let o = Transform::new([1.0, 2.0, 3.0], Quat::from_rotation_y(0.7).to_array());
        let back = f.to_origin(f.to_solver(o));
        assert!((back.p() - o.p()).length() < 1e-5);
        assert!(back.q().dot(o.q()).abs() > 1.0 - 1e-6);
    }
}
