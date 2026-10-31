//! Exact conversion between Rubble's frame (Y up, right-handed, metres) and a
//! host's, done once at the ABI boundary so the kernels never change.
//!
//! Unreal's frame is Z up, X forward, Y right, left-handed, centimetres. The
//! axis map is a reflection (determinant -1): Unreal X = Rubble -Z (forward),
//! Unreal Y = Rubble X (right), Unreal Z = Rubble Y (up). Under a reflection:
//!
//! - **polar vectors** (positions, points, velocities, forces, impulses,
//!   normals, gravity) map by `M v`, scaled by their units;
//! - **pseudovectors** (angular velocity, torque, angular impulse, rotation
//!   axes) map by `det(M) M v`: an extra sign flip;
//! - **quaternions** map as rotations: `(det(M) M q.xyz, q.w)`, so
//!   `R(q_host) = M R(q_rubble) M^T`;
//! - **joint frames** keep their axes' meaning: the host frame's z (hinge,
//!   cone and twist axis) and x (slide axis) stay the same physical axes, but
//!   a left-handed host measures angles about them with the opposite sign, so
//!   joint angle limits, target angles and motor speeds are negated;
//! - **triangle winding** flips (a reflection turns counter-clockwise faces
//!   clockwise), so exported meshes swap two indices per triangle.
//!
//! Units in centimetres: lengths and speeds ×100, accelerations ×100, forces
//! and impulses ×100 (kg·cm/s², kg·cm/s), torques and angular impulses ×10⁴,
//! inertia ×10⁴, density ×10⁻⁶. Mass, time and angles are unchanged.

use glam::{Mat3, Quat, Vec3};

/// The host's coordinate convention.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RbCoordinates {
    /// Rubble's own: Y up, right-handed, metres (no conversion).
    #[default]
    RB_COORDINATES_Y_UP_METRES = 0,
    /// Unreal Engine: Z up, X forward, Y right, left-handed, centimetres.
    RB_COORDINATES_UNREAL = 1,
}

/// Conversion for one world.
#[derive(Clone, Copy, Debug)]
pub struct Coords {
    kind: RbCoordinates,
}

/// Rubble -> Unreal axis map (rows: Unreal X, Y, Z in Rubble terms).
fn m() -> Mat3 {
    Mat3::from_cols(
        Vec3::new(0.0, 1.0, 0.0),  // Rubble X -> Unreal Y
        Vec3::new(0.0, 0.0, 1.0),  // Rubble Y -> Unreal Z
        Vec3::new(-1.0, 0.0, 0.0), // Rubble Z -> Unreal -X
    )
}

impl Coords {
    pub fn new(kind: RbCoordinates) -> Self {
        Self { kind }
    }

    pub fn unreal(&self) -> bool {
        self.kind == RbCoordinates::RB_COORDINATES_UNREAL
    }

    fn scale(&self, power: i32) -> f32 {
        if self.unreal() {
            100f32.powi(power)
        } else {
            1.0
        }
    }

    fn polar_out(&self, v: [f32; 3], power: i32) -> [f32; 3] {
        if !self.unreal() {
            return v;
        }
        (m() * Vec3::from(v) * self.scale(power)).to_array()
    }

    fn polar_in(&self, v: [f32; 3], power: i32) -> [f32; 3] {
        if !self.unreal() {
            return v;
        }
        (m().transpose() * Vec3::from(v) / self.scale(power)).to_array()
    }

    fn pseudo_out(&self, v: [f32; 3], power: i32) -> [f32; 3] {
        if !self.unreal() {
            return v;
        }
        (-(m() * Vec3::from(v)) * self.scale(power)).to_array()
    }

    fn pseudo_in(&self, v: [f32; 3], power: i32) -> [f32; 3] {
        if !self.unreal() {
            return v;
        }
        (-(m().transpose() * Vec3::from(v)) / self.scale(power)).to_array()
    }

    // Polar vectors, by units.
    pub fn position_out(&self, v: [f32; 3]) -> [f32; 3] {
        self.polar_out(v, 1)
    }
    pub fn position_in(&self, v: [f32; 3]) -> [f32; 3] {
        self.polar_in(v, 1)
    }
    /// Velocities, accelerations (gravity), forces and impulses all carry one
    /// length unit.
    pub fn linear_out(&self, v: [f32; 3]) -> [f32; 3] {
        self.polar_out(v, 1)
    }
    pub fn linear_in(&self, v: [f32; 3]) -> [f32; 3] {
        self.polar_in(v, 1)
    }
    /// Unit directions (normals, ray directions).
    pub fn direction_out(&self, v: [f32; 3]) -> [f32; 3] {
        self.polar_out(v, 0)
    }
    pub fn direction_in(&self, v: [f32; 3]) -> [f32; 3] {
        self.polar_in(v, 0)
    }
    /// Half extents and other lengths given as vectors of sizes (no sign): the
    /// axes permute, magnitudes scale.
    pub fn extents_in(&self, v: [f32; 3]) -> [f32; 3] {
        let p = self.polar_in(v, 1);
        [p[0].abs(), p[1].abs(), p[2].abs()]
    }

    // Pseudovectors.
    pub fn angular_velocity_out(&self, v: [f32; 3]) -> [f32; 3] {
        self.pseudo_out(v, 0)
    }
    pub fn angular_velocity_in(&self, v: [f32; 3]) -> [f32; 3] {
        self.pseudo_in(v, 0)
    }
    /// Torques and angular impulses (two length units).
    pub fn angular_in(&self, v: [f32; 3]) -> [f32; 3] {
        self.pseudo_in(v, 2)
    }

    pub fn rotation_out(&self, q: [f32; 4]) -> [f32; 4] {
        if !self.unreal() {
            return q;
        }
        let a = -(m() * Vec3::new(q[0], q[1], q[2]));
        [a.x, a.y, a.z, q[3]]
    }
    pub fn rotation_in(&self, q: [f32; 4]) -> [f32; 4] {
        if !self.unreal() {
            return q;
        }
        let a = -(m().transpose() * Vec3::new(q[0], q[1], q[2]));
        [a.x, a.y, a.z, q[3]]
    }

    /// A joint frame (body-local): position as a point, rotation such that the
    /// frame's z and x axes keep their physical directions.
    pub fn joint_frame_in(&self, position: [f32; 3], rotation: [f32; 4]) -> ([f32; 3], [f32; 4]) {
        if !self.unreal() {
            return (position, rotation);
        }
        // Q maps Rubble frame z -> the Rubble image of host z (Rubble y), and
        // x -> the image of host x (Rubble -z); a proper rotation.
        let q = Quat::from_mat3(&Mat3::from_cols(
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ));
        let r = Quat::from_array(self.rotation_in(rotation)) * q;
        (self.position_in(position), r.normalize().to_array())
    }

    /// An angle about a joint axis (limits, targets, motor speeds).
    pub fn joint_angle_in(&self, a: f32) -> f32 {
        if self.unreal() {
            -a
        } else {
            a
        }
    }

    // Scalars.
    pub fn length_in(&self, x: f32) -> f32 {
        x / self.scale(1)
    }
    pub fn length_out(&self, x: f32) -> f32 {
        x * self.scale(1)
    }
    pub fn density_in(&self, x: f32) -> f32 {
        x * self.scale(3)
    }
    /// Torque magnitudes (break torques, motor torques).
    pub fn torque_in(&self, x: f32) -> f32 {
        x / self.scale(2)
    }
    /// Force magnitudes (break forces, motor forces).
    pub fn force_in(&self, x: f32) -> f32 {
        x / self.scale(1)
    }
    pub fn force_out(&self, x: f32) -> f32 {
        x * self.scale(1)
    }
    pub fn torque_out(&self, x: f32) -> f32 {
        x * self.scale(2)
    }
    /// Energy (J: kg·m²/s²).
    pub fn energy_out(&self, x: f32) -> f32 {
        x * self.scale(2)
    }
    pub fn extents_out(&self, v: [f32; 3]) -> [f32; 3] {
        let p = self.polar_out(v, 1);
        [p[0].abs(), p[1].abs(), p[2].abs()]
    }

    /// An axis-aligned box's corners in: the map reflects axes, so the
    /// corners are re-sorted per component.
    pub fn box_in(&self, min: [f32; 3], max: [f32; 3]) -> ([f32; 3], [f32; 3]) {
        let (a, b) = (self.position_in(min), self.position_in(max));
        (
            [a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2])],
            [a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2])],
        )
    }

    /// Impulse per area (N s / m², or kg cm / s per cm²): ×100 in from cm.
    pub fn impulse_per_area_in(&self, x: f32) -> f32 {
        x / self.scale(-1)
    }
    pub fn impulse_per_area_out(&self, x: f32) -> f32 {
        x * self.scale(-1)
    }

    /// Inertia tensors (rows; kg·m² or kg·cm²): `I = Mᵀ I_host M` in, the
    /// reverse out. Quadratic in the axis map, so the reflection's sign cancels.
    pub fn inertia_in(&self, t: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
        if !self.unreal() {
            return t;
        }
        let host = Mat3::from_cols_array_2d(&t).transpose();
        let engine = m().transpose() * host * m() / self.scale(2);
        engine.transpose().to_cols_array_2d()
    }

    pub fn inertia_out(&self, t: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
        if !self.unreal() {
            return t;
        }
        let engine = Mat3::from_cols_array_2d(&t).transpose();
        let host = m() * engine * m().transpose() * self.scale(2);
        host.transpose().to_cols_array_2d()
    }

    /// Motion-lock bits (linear x, y, z, then angular x, y, z, in host axes)
    /// to Rubble's axes: a permutation, so signs do not matter.
    pub fn locks_in(&self, bits: u32) -> u32 {
        self.locks_map(bits, false)
    }

    pub fn locks_out(&self, bits: u32) -> u32 {
        self.locks_map(bits, true)
    }

    fn locks_map(&self, bits: u32, out: bool) -> u32 {
        if !self.unreal() {
            return bits;
        }
        let map = m();
        let mut result = 0;
        for engine in 0..3 {
            // The host axis that Rubble axis `engine` maps to.
            let col = map.col(engine);
            let host = (0..3)
                .max_by(|&a, &b| col[a].abs().total_cmp(&col[b].abs()))
                .unwrap();
            for block in [0, 3] {
                let (from, to) = if out { (engine, host) } else { (host, engine) };
                if bits & (1 << (block + from)) != 0 {
                    result |= 1 << (block + to);
                }
            }
        }
        result
    }

    /// Triangle winding for exported meshes: unchanged. Rubble's front faces wind
    /// counter-clockwise ((b - a) x (c - a) points out); Unreal's axis map is a
    /// mirror, which turns that into Unreal's own rule ((c - a) x (b - a) points
    /// out) without reordering the corners.
    pub fn triangle_out(&self, t: [u32; 3]) -> [u32; 3] {
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
        (Vec3::from(a) - Vec3::from(b)).length() <= tol
    }

    #[test]
    fn the_map_is_a_reflection_with_unreal_axes() {
        assert!((m().determinant() + 1.0).abs() < 1e-6);
        let c = Coords::new(RbCoordinates::RB_COORDINATES_UNREAL);
        // Rubble up (Y) is Unreal up (Z); 1 m is 100 cm.
        assert_eq!(c.position_out([0.0, 1.0, 0.0]), [0.0, 0.0, 100.0]);
        // Rubble forward (-Z) is Unreal forward (+X).
        assert_eq!(c.direction_out([0.0, 0.0, -1.0]), [1.0, 0.0, 0.0]);
        assert_eq!(c.direction_out([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]);
    }

    #[test]
    fn every_class_round_trips() {
        let c = Coords::new(RbCoordinates::RB_COORDINATES_UNREAL);
        let v = [0.3, -1.7, 2.9];
        assert!(close(c.position_in(c.position_out(v)), v, 1e-5));
        assert!(close(c.linear_in(c.linear_out(v)), v, 1e-5));
        assert!(close(c.direction_in(c.direction_out(v)), v, 1e-6));
        assert!(close(
            c.angular_velocity_in(c.angular_velocity_out(v)),
            v,
            1e-6
        ));
        let q = Quat::from_euler(glam::EulerRot::XYZ, 0.3, -0.9, 1.4).to_array();
        let back = c.rotation_in(c.rotation_out(q));
        assert!(Quat::from_array(back).dot(Quat::from_array(q)) > 1.0 - 1e-6);
        let rubble = Coords::new(RbCoordinates::RB_COORDINATES_Y_UP_METRES);
        assert_eq!(rubble.position_out(v), v);
        assert_eq!(rubble.rotation_out(q), q);
    }

    /// The rules agree with physics in the host frame: rotating a point,
    /// cross products (torque = r x F) and angular velocity (v = w x r).
    #[test]
    fn rotations_and_cross_products_agree_in_the_host_frame() {
        let c = Coords::new(RbCoordinates::RB_COORDINATES_UNREAL);
        let q = Quat::from_axis_angle(Vec3::new(1.0, 2.0, -0.5).normalize(), 0.8);
        let p = [0.4, 1.1, -2.0];
        // Rotate in Rubble then convert == convert then rotate in the host.
        let rotated = c.position_out((q * Vec3::from(p)).to_array());
        let host_q = Quat::from_array(c.rotation_out(q.to_array()));
        let host = (host_q * Vec3::from(c.position_out(p))).to_array();
        assert!(close(rotated, host, 1e-3), "{rotated:?} vs {host:?}");
        // v = w x r in both frames (w a pseudovector, r and v polar).
        let w = [0.2, -0.7, 1.3];
        let v = (Vec3::from(w).cross(Vec3::from(p))).to_array();
        let host_v = Vec3::from(c.angular_velocity_out(w))
            .cross(Vec3::from(c.position_out(p)))
            .to_array();
        assert!(close(c.linear_out(v), host_v, 1e-3), "{host_v:?}");
    }

    /// A joint frame keeps its z and x axes physically: identity in Unreal
    /// (z up, x forward) becomes Rubble y (up) and -z (forward).
    #[test]
    fn joint_frames_keep_their_axes() {
        let c = Coords::new(RbCoordinates::RB_COORDINATES_UNREAL);
        let (_, r) = c.joint_frame_in([0.0; 3], [0.0, 0.0, 0.0, 1.0]);
        let r = Quat::from_array(r);
        assert!(close((r * Vec3::Z).to_array(), [0.0, 1.0, 0.0], 1e-6));
        assert!(close((r * Vec3::X).to_array(), [0.0, 0.0, -1.0], 1e-6));
        let host = Quat::from_axis_angle(Vec3::new(0.3, -0.4, 0.8).normalize(), 1.1);
        let (_, r) = c.joint_frame_in([0.0; 3], host.to_array());
        let r = Quat::from_array(r);
        let z = c.direction_out((r * Vec3::Z).to_array());
        assert!(close(z, (host * Vec3::Z).to_array(), 1e-5), "{z:?}");
        let x = c.direction_out((r * Vec3::X).to_array());
        assert!(close(x, (host * Vec3::X).to_array(), 1e-5), "{x:?}");
    }

    #[test]
    fn the_mirror_keeps_front_faces_front() {
        // A Rubble triangle facing +Y (up): (b - a) x (c - a) = +Y.
        let (a, b, d) = ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
        let c = Coords::new(RbCoordinates::RB_COORDINATES_UNREAL);
        let t = c.triangle_out([0, 1, 2]);
        let p = [a, b, d].map(|v| Vec3::from(c.position_out(v)));
        let (a, b, d) = (p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]);
        // Unreal's front: (c - a) x (b - a) points out, here up (+Z).
        let n = (d - a).cross(b - a);
        assert!(n.z > 0.0 && n.x.abs() < 1e-6 && n.y.abs() < 1e-6, "{n:?}");
    }

    #[test]
    fn inertia_round_trips_and_follows_the_axes() {
        let c = Coords::new(RbCoordinates::RB_COORDINATES_UNREAL);
        // Rubble diag(1, 2, 3) kg m²: Rubble X is Unreal Y, Y is Z, Z is X.
        let i = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];
        let host = c.inertia_out(i);
        assert!(close(host[0], [3e4, 0.0, 0.0], 1e-2));
        assert!(close(host[1], [0.0, 1e4, 0.0], 1e-2));
        assert!(close(host[2], [0.0, 0.0, 2e4], 1e-2));
        let skew = [[2.0, 0.3, 0.1], [0.3, 1.5, -0.2], [0.1, -0.2, 1.0]];
        let back = c.inertia_in(c.inertia_out(skew));
        for r in 0..3 {
            assert!(close(back[r], skew[r], 1e-5), "{back:?}");
        }
    }

    #[test]
    fn locks_follow_the_axes() {
        let c = Coords::new(RbCoordinates::RB_COORDINATES_UNREAL);
        // Unreal Z (up) is Rubble Y; Unreal X is Rubble Z; Unreal Y is Rubble X.
        assert_eq!(c.locks_in(4), 2);
        assert_eq!(c.locks_in(1), 4);
        assert_eq!(c.locks_in(2), 1);
        assert_eq!(c.locks_in(32), 16);
        for bits in 0..64 {
            assert_eq!(c.locks_out(c.locks_in(bits)), bits);
        }
        let y_up = Coords::new(RbCoordinates::RB_COORDINATES_Y_UP_METRES);
        assert_eq!(y_up.locks_in(37), 37);
    }
}
