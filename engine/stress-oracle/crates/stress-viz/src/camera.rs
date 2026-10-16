//! Perspective camera: orbit parameters (yaw, pitch, distance, target), named presets,
//! auto-fit to the scene bounds, and the world -> view -> screen projection.
//!
//! The world is z-up. Yaw 0 / pitch 0 looks along +y (the direction impactors usually
//! travel), from the -y side; positive yaw orbits the eye towards +x, positive pitch
//! raises it. View space has x right, y up and z the depth in front of the eye.

use stress_ref::math::Vec3;

/// Axis-aligned bounding box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub lo: Vec3,
    pub hi: Vec3,
}

impl Aabb {
    pub const EMPTY: Aabb =
        Aabb { lo: Vec3 { x: f64::INFINITY, y: f64::INFINITY, z: f64::INFINITY }, hi: Vec3 { x: f64::NEG_INFINITY, y: f64::NEG_INFINITY, z: f64::NEG_INFINITY } };

    pub fn include(&mut self, p: Vec3) {
        self.lo = self.lo.component_min(p);
        self.hi = self.hi.component_max(p);
    }

    pub fn is_empty(&self) -> bool {
        self.lo.x > self.hi.x
    }

    pub fn center(&self) -> Vec3 {
        (self.lo + self.hi) * 0.5
    }

    pub fn size(&self) -> Vec3 {
        self.hi - self.lo
    }

    pub fn corners(&self) -> [Vec3; 8] {
        let (l, h) = (self.lo, self.hi);
        [
            Vec3::new(l.x, l.y, l.z),
            Vec3::new(h.x, l.y, l.z),
            Vec3::new(l.x, h.y, l.z),
            Vec3::new(h.x, h.y, l.z),
            Vec3::new(l.x, l.y, h.z),
            Vec3::new(h.x, l.y, h.z),
            Vec3::new(l.x, h.y, h.z),
            Vec3::new(h.x, h.y, h.z),
        ]
    }
}

/// User-facing camera parameters; unset distance/target are fitted to the scene.
#[derive(Clone, Debug, PartialEq)]
pub struct CameraSpec {
    pub yaw_deg: f64,
    pub pitch_deg: f64,
    pub distance: Option<f64>,
    pub target: Option<Vec3>,
    pub fov_deg: f64,
}

impl Default for CameraSpec {
    fn default() -> Self {
        CameraSpec { yaw_deg: -35.0, pitch_deg: 25.0, distance: None, target: None, fov_deg: 30.0 }
    }
}

impl CameraSpec {
    /// Orientation of a named preset: `front`, `side`, `top`, `iso` (or the default `3/4`).
    pub fn preset(name: &str) -> Option<(f64, f64)> {
        Some(match name {
            "front" => (0.0, 6.0),
            "side" => (-90.0, 6.0),
            "top" => (0.0, 89.0),
            "iso" => (-45.0, 35.264),
            "3/4" | "default" => (-35.0, 25.0),
            _ => return None,
        })
    }

    /// Unit vector from the target to the eye.
    fn eye_direction(&self) -> Vec3 {
        let (yaw, pitch) = (self.yaw_deg.to_radians(), self.pitch_deg.to_radians());
        Vec3::new(pitch.cos() * yaw.sin(), -pitch.cos() * yaw.cos(), pitch.sin())
    }

    /// The camera for a viewport of `aspect = width / height`, fitting `bounds` (with a
    /// small margin) unless distance and target are given.
    pub fn build(&self, bounds: &Aabb, aspect: f64) -> Camera {
        let fov = self.fov_deg.clamp(1.0, 150.0).to_radians();
        let dir = self.eye_direction();
        let mut target = self.target.unwrap_or_else(|| bounds.center());
        let mut cam = Camera::looking(target + dir * self.distance.unwrap_or(1.0), target, fov);
        if self.distance.is_some() || bounds.is_empty() {
            return cam;
        }
        // Two passes: fit the distance, re-centre the projected bounds, fit again.
        let (tan_y, tan_x) = ((0.5 * fov).tan() * 0.88, (0.5 * fov).tan() * aspect * 0.92);
        for pass in 0..2 {
            let mut d: f64 = 1e-3;
            for c in bounds.corners() {
                let r = c - target;
                let (x, y, along) = (r.dot(cam.right), r.dot(cam.up), r.dot(cam.forward));
                d = d.max(x.abs() / tan_x - along).max(y.abs() / tan_y - along);
            }
            cam = Camera::looking(target + dir * d, target, fov);
            if pass == 0 && self.target.is_none() {
                // Shift the target so the projected box is centred in the viewport.
                let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
                for c in bounds.corners() {
                    let v = cam.world_to_view(c);
                    let s = [v.x / v.z, v.y / v.z];
                    for k in 0..2 {
                        lo[k] = lo[k].min(s[k]);
                        hi[k] = hi[k].max(s[k]);
                    }
                }
                let shift = cam.right * (0.5 * (lo[0] + hi[0]) * d) + cam.up * (0.5 * (lo[1] + hi[1]) * d);
                target += shift;
            }
        }
        cam
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub eye: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    /// Vertical field of view (radians).
    pub fov_y: f64,
}

impl Camera {
    /// Eye at `eye` looking at `target`, world z up (y up when looking straight down).
    pub fn looking(eye: Vec3, target: Vec3, fov_y: f64) -> Camera {
        let forward = (target - eye).normalized();
        let world_up = if forward.z.abs() > 0.999 { Vec3::new(0.0, 1.0, 0.0) } else { Vec3::new(0.0, 0.0, 1.0) };
        let right = forward.cross(world_up).normalized();
        let up = right.cross(forward);
        Camera { eye, forward, right, up, fov_y }
    }

    /// World point to view space (x right, y up, z depth along `forward`).
    pub fn world_to_view(&self, p: Vec3) -> Vec3 {
        let r = p - self.eye;
        Vec3::new(r.dot(self.right), r.dot(self.up), r.dot(self.forward))
    }

    /// The pixel projection for a `width x height` viewport.
    pub fn projection(&self, width: usize, height: usize) -> Projection {
        Projection {
            focal: (0.5 * height as f64 / (0.5 * self.fov_y).tan()) as f32,
            cx: 0.5 * width as f32,
            cy: 0.5 * height as f32,
        }
    }
}

/// Pinhole projection of view-space points to pixels (y down).
#[derive(Clone, Copy, Debug)]
pub struct Projection {
    pub focal: f32,
    pub cx: f32,
    pub cy: f32,
}

/// Points closer than this to the eye are clipped (m).
pub const NEAR: f32 = 1e-3;

impl Projection {
    /// Pixel coordinates of a view-space point in front of the eye.
    pub fn project(&self, v: [f32; 3]) -> [f32; 2] {
        [self.cx + self.focal * v[0] / v[2], self.cy - self.focal * v[1] / v[2]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wall() -> Aabb {
        Aabb { lo: Vec3::new(-1.0, 0.0, 0.0), hi: Vec3::new(1.0, 0.2, 2.0) }
    }

    #[test]
    fn default_view_is_front_left_and_above() {
        let cam = CameraSpec::default().build(&wall(), 16.0 / 9.0);
        assert!(cam.eye.x < 0.0 && cam.eye.y < 0.0 && cam.eye.z > 1.0, "{:?}", cam.eye);
        assert!((cam.right.norm() - 1.0).abs() < 1e-12 && cam.up.z > 0.0);
        assert!(cam.right.dot(cam.forward).abs() < 1e-12 && cam.up.dot(cam.forward).abs() < 1e-12);
    }

    #[test]
    fn projection_maps_the_axis_to_the_centre_and_right_to_right() {
        let cam = Camera::looking(Vec3::new(0.0, -5.0, 0.0), Vec3::ZERO, 60f64.to_radians());
        let proj = cam.projection(200, 100);
        let to = |p: Vec3| {
            let v = cam.world_to_view(p);
            proj.project([v.x as f32, v.y as f32, v.z as f32])
        };
        let c = to(Vec3::ZERO);
        assert!((c[0] - 100.0).abs() < 1e-4 && (c[1] - 50.0).abs() < 1e-4);
        assert!(to(Vec3::new(1.0, 0.0, 0.0))[0] > 100.0, "+x is to the right looking along +y");
        assert!(to(Vec3::new(0.0, 0.0, 1.0))[1] < 50.0, "+z is up (smaller row)");
        // Half the vertical fov at the target depth reaches the top edge.
        let top = to(Vec3::new(0.0, 0.0, 5.0 * 30f64.to_radians().tan()));
        assert!(top[1].abs() < 1e-3, "{top:?}");
    }

    #[test]
    fn auto_fit_keeps_every_corner_inside_the_viewport() {
        for preset in ["front", "side", "top", "iso", "3/4"] {
            let (yaw, pitch) = CameraSpec::preset(preset).unwrap();
            let spec = CameraSpec { yaw_deg: yaw, pitch_deg: pitch, ..Default::default() };
            let (w, h) = (320, 180);
            let cam = spec.build(&wall(), w as f64 / h as f64);
            let proj = cam.projection(w, h);
            let mut lo = [f32::INFINITY; 2];
            let mut hi = [f32::NEG_INFINITY; 2];
            for c in wall().corners() {
                let v = cam.world_to_view(c);
                assert!(v.z > 0.0, "{preset}: in front of the eye");
                let p = proj.project([v.x as f32, v.y as f32, v.z as f32]);
                for k in 0..2 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
            assert!(lo[0] >= 0.0 && lo[1] >= 0.0 && hi[0] <= w as f32 && hi[1] <= h as f32, "{preset}: {lo:?} {hi:?}");
            // And it is not tiny: the box fills a good part of one dimension.
            assert!(hi[0] - lo[0] > 0.6 * w as f32 || hi[1] - lo[1] > 0.6 * h as f32, "{preset}: {lo:?} {hi:?}");
        }
    }
}
