//! Mass properties of the primitive shapes, uniform density, about their centre
//! and principal axes (capsules along local +Y, as the kernels collide them).

use glam::Vec3;
use std::f32::consts::PI;

/// Mass (kg) and principal moments of inertia (kg m²) about the centre of mass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MassProperties {
    pub mass: f32,
    pub inertia: Vec3,
}

impl MassProperties {
    /// Solid sphere: I = 2/5 m r².
    pub fn sphere(radius: f32, density: f32) -> Self {
        let mass = density * (4.0 / 3.0) * PI * radius.powi(3);
        Self {
            mass,
            inertia: Vec3::splat(0.4 * mass * radius * radius),
        }
    }

    /// Solid box of half extents `half`: I_x = m (h_y² + h_z²) / 3 with half extents.
    pub fn cuboid(half: Vec3, density: f32) -> Self {
        let mass = density * 8.0 * half.x * half.y * half.z;
        let (x, y, z) = (half.x * half.x, half.y * half.y, half.z * half.z);
        Self {
            mass,
            inertia: Vec3::new(y + z, x + z, x + y) * (mass / 3.0),
        }
    }

    /// Solid capsule along +Y: a cylinder of half height `half_len` with two
    /// hemispherical caps of `radius`.
    pub fn capsule(radius: f32, half_len: f32, density: f32) -> Self {
        let (r, h) = (radius, 2.0 * half_len);
        let m_cyl = density * PI * r * r * h;
        let m_caps = density * (4.0 / 3.0) * PI * r.powi(3);
        let axial = m_cyl * r * r / 2.0 + m_caps * 0.4 * r * r;
        // Each cap: 2/5 m_cap r² about its own centre plus the shift to the body
        // centre (centroid 3r/8 from the flat face, which sits h/2 out).
        let cap = m_caps / 2.0;
        let cap_offset = h / 2.0 + 3.0 * r / 8.0;
        let caps_lateral =
            2.0 * (cap * (0.4 * r * r - (3.0 * r / 8.0).powi(2)) + cap * cap_offset * cap_offset);
        let lateral = m_cyl * (3.0 * r * r + h * h) / 12.0 + caps_lateral;
        Self {
            mass: m_cyl + m_caps,
            inertia: Vec3::new(lateral, axial, lateral),
        }
    }

    /// Scale to a total mass of `mass` (same shape and mass distribution).
    pub fn with_mass(self, mass: f32) -> Self {
        let k = if self.mass > 0.0 {
            mass / self.mass
        } else {
            0.0
        };
        Self {
            mass,
            inertia: self.inertia * k,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_cube_and_sphere() {
        let c = MassProperties::cuboid(Vec3::splat(0.5), 1.0);
        assert!((c.mass - 1.0).abs() < 1e-6);
        assert!((c.inertia.x - 1.0 / 6.0).abs() < 1e-6);
        let s = MassProperties::sphere(1.0, 3.0 / (4.0 * PI));
        assert!((s.mass - 1.0).abs() < 1e-6 && (s.inertia.y - 0.4).abs() < 1e-6);
    }

    #[test]
    fn capsule_tends_to_sphere_and_cylinder() {
        let sphere = MassProperties::sphere(0.5, 2.0);
        let cap = MassProperties::capsule(0.5, 0.0, 2.0);
        assert!((cap.mass - sphere.mass).abs() < 1e-5);
        assert!((cap.inertia - sphere.inertia).abs().max_element() < 1e-5);
        // A long capsule approaches a cylinder's lateral moment.
        let long = MassProperties::capsule(0.01, 50.0, 1.0);
        let cyl_lateral = long.mass * (100.0f32 * 100.0) / 12.0;
        assert!((long.inertia.x / cyl_lateral - 1.0).abs() < 1e-3);
    }
}
