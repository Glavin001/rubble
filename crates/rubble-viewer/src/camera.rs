use glam::{Mat4, Vec3};

/// Orbit camera for 3D scenes. Rotates around a target point.
pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    /// Horizontal angle in radians.
    pub yaw: f32,
    /// Vertical angle in radians, clamped to avoid gimbal lock.
    pub pitch: f32,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::new(0.0, 3.0, 0.0),
            distance: 25.0,
            yaw: 0.4,
            pitch: 0.5,
            fov_y: std::f32::consts::FRAC_PI_4,
            near: 0.1,
            far: 200.0,
        }
    }
}

impl OrbitCamera {
    pub fn eye(&self) -> Vec3 {
        let x = self.distance * self.pitch.cos() * self.yaw.sin();
        let y = self.distance * self.pitch.sin();
        let z = self.distance * self.pitch.cos() * self.yaw.cos();
        self.target + Vec3::new(x, y, z)
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, Vec3::Y)
    }

    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, aspect, self.near, self.far)
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        self.projection_matrix(aspect) * self.view_matrix()
    }

    pub fn rotate(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.005;
        self.pitch = (self.pitch + dy * 0.005).clamp(-1.5, 1.5);
    }

    pub fn zoom(&mut self, delta: f32) {
        let factor = (-delta * 0.15).exp();
        self.distance = (self.distance * factor).clamp(0.5, 200.0);
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        let right = Vec3::new(self.yaw.cos(), 0.0, -self.yaw.sin());
        let up = Vec3::Y;
        let scale = self.distance * 0.002;
        self.target += right * (-dx * scale) + up * (dy * scale);
    }
}

/// Orthographic pan-zoom camera for 2D scenes.
pub struct Camera2D {
    pub center: glam::Vec2,
    pub zoom: f32,
}

impl Default for Camera2D {
    fn default() -> Self {
        Self {
            center: glam::Vec2::new(15.0, 10.0),
            zoom: 0.04,
        }
    }
}

impl Camera2D {
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let hw = 1.0 / self.zoom * aspect;
        let hh = 1.0 / self.zoom;
        Mat4::orthographic_rh(
            self.center.x - hw,
            self.center.x + hw,
            self.center.y - hh,
            self.center.y + hh,
            -1.0,
            1.0,
        )
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        let scale = 1.0 / self.zoom * 0.002;
        self.center.x -= dx * scale;
        self.center.y += dy * scale;
    }

    pub fn zoom_by(&mut self, delta: f32) {
        self.zoom *= (1.0 + delta * 0.05).clamp(0.5, 2.0);
        self.zoom = self.zoom.clamp(0.001, 1.0);
    }
}
