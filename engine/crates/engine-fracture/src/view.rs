//! What both hosts draw besides the GPU shader (`slang/render/fracture_draw.slang`):
//! the camera (orbit, first-person walk and fly view modes) and click-to-shoot
//! ray, the floor quad, and the indexed house mesh.

use crate::math::*;
use crate::scene::{vertex, Ray, Scene};
use std::collections::HashMap;

/// Vertical field of view (radians).
pub const FOV_Y: f64 = 0.7;
pub const NEAR: f64 = 0.02;
/// First-person look: radians per CSS pixel of drag and of mouse-look.
const LOOK_DRAG: f64 = 0.004;
const LOOK_MOUSE: f64 = 0.0025;
/// First-person look pitch limit (radians), short of straight up or down.
const MAX_LOOK_PITCH: f64 = 1.5;

/// Eye height above the floor of the first-person walk mode (m).
pub const EYE_HEIGHT: f64 = 1.7;
/// Lowest eye height of the fly mode (m): it stays above the floor.
pub const FLY_FLOOR: f64 = 0.1;
/// Travel speed multiple while sprinting.
pub const SPRINT: f64 = 3.0;

/// How the camera moves (the lab's Camera menu).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ViewMode {
    /// Around the scene centre: drag to orbit, scroll to zoom.
    #[default]
    Orbit,
    /// First person on foot at [`EYE_HEIGHT`]: look around, move across the floor.
    Walk,
    /// First person, free flight: move along the look direction, up and down.
    Fly,
}

impl ViewMode {
    /// The Camera menu's index: 0 orbit, 1 walk, 2 fly (anything else orbits).
    pub fn from_index(index: u32) -> Self {
        match index {
            1 => Self::Walk,
            2 => Self::Fly,
            _ => Self::Orbit,
        }
    }
    pub fn index(self) -> u32 {
        self as u32
    }
    pub fn first_person(self) -> bool {
        self != Self::Orbit
    }
}

/// The lab's camera. Orbit mode circles the scene centre (drag to orbit, scroll to
/// zoom); the first-person modes ([`ViewMode::Walk`], [`ViewMode::Fly`]) look from
/// `position` along `look_yaw` / `look_pitch` (drag or mouse-look to turn, keys to
/// [`travel`](Self::travel), scroll to change speed). The orbit state is kept while
/// in first person, so switching back returns to the same view.
#[derive(Clone, Copy, Debug)]
pub struct Orbit {
    pub yaw: f64,
    pub pitch: f64,
    pub distance: f64,
    pub target: [f64; 3],
    pub extent: f64,
    pub min_distance: f64,
    pub mode: ViewMode,
    /// First-person eye.
    pub position: [f64; 3],
    /// First-person look angles, as the orbit's: the eye looks along
    /// `-(sin yaw cos pitch, sin pitch, cos yaw cos pitch)` (pitch > 0 looks down).
    pub look_yaw: f64,
    pub look_pitch: f64,
    /// First-person travel speed (m/s), before sprinting.
    pub speed: f64,
}

impl Orbit {
    pub fn new(scene: &Scene) -> Self {
        let c = scene.camera;
        Self {
            yaw: c.map_or(0.25, |c| c.yaw),
            pitch: c.map_or(0.22, |c| c.pitch),
            distance: c.map_or(scene.extent * 1.9, |c| c.distance),
            target: scene.center,
            extent: scene.extent,
            min_distance: c.map_or(scene.extent * 0.35, |c| c.min_distance),
            mode: ViewMode::Orbit,
            position: [0.0; 3],
            look_yaw: 0.0,
            look_pitch: 0.0,
            speed: (scene.extent * 0.4).clamp(4.0, 60.0),
        }
    }

    /// Switch view mode. Entering first person from orbit starts at the orbit's eye
    /// (on the floor at [`EYE_HEIGHT`] when walking) looking at the scene centre;
    /// walk and fly hand over the eye; leaving to orbit restores the orbit view.
    pub fn set_mode(&mut self, mode: ViewMode, aspect: f64) {
        if mode == self.mode {
            return;
        }
        if mode.first_person() {
            if !self.mode.first_person() {
                self.position = self.orbit_eye(aspect);
            }
            if mode == ViewMode::Walk {
                self.position[1] = EYE_HEIGHT;
            }
            if !self.mode.first_person() {
                self.look_at(self.target);
            }
            self.look_pitch = self.look_pitch.clamp(-MAX_LOOK_PITCH, MAX_LOOK_PITCH);
        }
        self.mode = mode;
    }

    /// Point the first-person look at `point`.
    fn look_at(&mut self, point: [f64; 3]) {
        let back = sub(self.position, point);
        if length(back) < 1e-9 {
            return;
        }
        let b = norm(back);
        self.look_pitch = b[1].clamp(-1.0, 1.0).asin();
        self.look_yaw = b[0].atan2(b[2]);
    }

    /// Unit view direction.
    pub fn forward(&self, aspect: f64) -> [f64; 3] {
        match self.mode {
            ViewMode::Orbit => norm(sub(self.target, self.orbit_eye(aspect))),
            _ => {
                let (sy, cy, sp, cp) = (
                    self.look_yaw.sin(),
                    self.look_yaw.cos(),
                    self.look_pitch.sin(),
                    self.look_pitch.cos(),
                );
                [-sy * cp, -sp, -cy * cp]
            }
        }
    }

    /// Eye position; in orbit, narrow viewports back off so the scene still fits.
    pub fn eye(&self, aspect: f64) -> [f64; 3] {
        match self.mode {
            ViewMode::Orbit => self.orbit_eye(aspect),
            _ => self.position,
        }
    }

    fn orbit_eye(&self, aspect: f64) -> [f64; 3] {
        let fitted = self.distance / aspect.min(1.0);
        let (sy, cy, sp, cp) = (
            self.yaw.sin(),
            self.yaw.cos(),
            self.pitch.sin(),
            self.pitch.cos(),
        );
        add(self.target, scale([sy * cp, sp, cy * cp], fitted))
    }

    /// Column-major view-projection (right-handed, depth 0..1).
    pub fn view_proj(&self, aspect: f64) -> [f32; 16] {
        let (eye, far) = (self.eye(aspect), (self.extent * 50.0).max(500.0));
        let z = scale(self.forward(aspect), -1.0);
        let x = norm(cross([0.0, 1.0, 0.0], z));
        let y = cross(z, x);
        let view = [
            x[0],
            y[0],
            z[0],
            0.0,
            x[1],
            y[1],
            z[1],
            0.0,
            x[2],
            y[2],
            z[2],
            0.0,
            -dot(x, eye),
            -dot(y, eye),
            -dot(z, eye),
            1.0,
        ];
        let f = 1.0 / (FOV_Y / 2.0).tan();
        let proj = [
            f / aspect,
            0.0,
            0.0,
            0.0,
            0.0,
            f,
            0.0,
            0.0,
            0.0,
            0.0,
            far / (NEAR - far),
            -1.0,
            0.0,
            0.0,
            NEAR * far / (NEAR - far),
            0.0,
        ];
        std::array::from_fn(|i| {
            (0..4)
                .map(|k| proj[k * 4 + i % 4] * view[(i / 4) * 4 + k])
                .sum::<f64>() as f32
        })
    }

    /// A pointer drag (CSS pixels): orbit, or turn the first-person look (the view
    /// follows the pointer, as dragging the scene).
    pub fn drag(&mut self, dx: f64, dy: f64) {
        match self.mode {
            ViewMode::Orbit => {
                self.yaw -= dx * 0.006;
                self.pitch = (self.pitch + dy * 0.006).clamp(-0.1, 1.3);
            }
            _ => self.look(dx * LOOK_DRAG, -dy * LOOK_DRAG),
        }
    }

    /// Mouse-look (pointer-lock movement, CSS pixels): turn the first-person look
    /// toward the motion. Orbit mode orbits, as a drag.
    pub fn mouse_look(&mut self, dx: f64, dy: f64) {
        match self.mode {
            ViewMode::Orbit => self.drag(dx, dy),
            _ => self.look(-dx * LOOK_MOUSE, dy * LOOK_MOUSE),
        }
    }

    fn look(&mut self, yaw: f64, pitch: f64) {
        self.look_yaw += yaw;
        self.look_pitch = (self.look_pitch + pitch).clamp(-MAX_LOOK_PITCH, MAX_LOOK_PITCH);
    }

    /// A wheel delta: zoom the orbit, or scale the first-person travel speed.
    pub fn zoom(&mut self, delta_y: f64) {
        match self.mode {
            ViewMode::Orbit => {
                self.distance = (self.distance * (delta_y * 0.001).exp())
                    .min(self.extent * 8.0)
                    .max(self.min_distance)
            }
            _ => {
                let fastest = (self.extent * 4.0).max(60.0);
                self.speed = (self.speed * (-delta_y * 0.001).exp()).clamp(0.5, fastest)
            }
        }
    }

    /// Place the orbit camera (scripted takes and replays): yaw and pitch in radians, as
    /// a drag turns them, the distance in metres and, if given, the target. Switches to
    /// orbit mode. Pitch keeps the drag's range and distance the zoom's.
    pub fn set_orbit(&mut self, yaw: f64, pitch: f64, distance: f64, target: Option<[f64; 3]>) {
        self.mode = ViewMode::Orbit;
        self.yaw = yaw;
        self.pitch = pitch.clamp(-0.1, 1.3);
        self.distance = distance.min(self.extent * 8.0).max(self.min_distance);
        if let Some(target) = target {
            self.target = target;
        }
    }

    /// The orbit camera as [`set_orbit`](Self::set_orbit) takes it: yaw, pitch, distance
    /// and the target's x, y, z.
    pub fn orbit_state(&self) -> [f64; 6] {
        let [x, y, z] = self.target;
        [self.yaw, self.pitch, self.distance, x, y, z]
    }

    /// First-person travel for `dt` seconds, each axis in -1..1 (keys held):
    /// `forward` along the view, `right` across it, `up` along world +Y. Walking
    /// stays on the floor at [`EYE_HEIGHT`] (forward is horizontal, up is ignored);
    /// flying stays above [`FLY_FLOOR`]. Orbit mode does not travel.
    pub fn travel(&mut self, forward: f64, right: f64, up: f64, sprint: bool, dt: f64) {
        if !self.mode.first_person() || !dt.is_finite() || dt <= 0.0 {
            return;
        }
        let (sy, cy) = (self.look_yaw.sin(), self.look_yaw.cos());
        let side = [cy, 0.0, -sy];
        let (ahead, lift) = match self.mode {
            ViewMode::Walk => ([-sy, 0.0, -cy], 0.0),
            _ => (self.forward(1.0), up),
        };
        let wish = add(
            add(scale(ahead, forward), scale(side, right)),
            [0.0, lift, 0.0],
        );
        let n = length(wish);
        if n < 1e-9 {
            return;
        }
        let step = self.speed * if sprint { SPRINT } else { 1.0 } * dt;
        // Diagonals are no faster than one axis.
        self.position = add(self.position, scale(wish, step / n.max(1.0)));
        self.position[1] = match self.mode {
            ViewMode::Walk => EYE_HEIGHT,
            _ => self.position[1].max(FLY_FLOOR),
        };
    }

    /// Shot through a viewport point (u, v in 0..1 from the top left), starting
    /// just in front of the eye.
    pub fn aim(&self, u: f64, v: f64, aspect: f64, projectile_radius: f64) -> Ray {
        let eye = self.eye(aspect);
        let forward = self.forward(aspect);
        let right = norm(cross(forward, [0.0, 1.0, 0.0]));
        let up = cross(right, forward);
        let t = (FOV_Y / 2.0).tan();
        let (x, y) = ((2.0 * u - 1.0) * aspect * t, (1.0 - 2.0 * v) * t);
        let direction = norm(add(forward, add(scale(right, x), scale(up, y))));
        Ray {
            origin: add(eye, scale(direction, projectile_radius * 2.0)),
            direction,
        }
    }
}

/// The ground quad (body 0), sized to the scene.
pub fn floor_vertices(extent: f64) -> Vec<f32> {
    let size = (extent * 8.0).max(20.0);
    let mut v = Vec::new();
    for [x, z] in [
        [-size, -size],
        [-size, size],
        [size, size],
        [-size, -size],
        [size, size],
        [size, -size],
    ] {
        vertex(&mut v, [x, 0.0, z], [0.0, 1.0, 0.0], [0.1, 0.15, 0.18], 0);
    }
    v
}

/// Indexed render mesh: corners repeated by the triangle list are stored once
/// (only exact matches of all ten attributes merge, body id included), so each
/// instanced copy runs far fewer vertex shaders for the same image.
pub fn index_mesh(vertices: &[f32]) -> (Vec<f32>, Vec<u32>) {
    let mut seen: HashMap<[u32; 10], u32> = HashMap::new();
    let (mut unique, mut index) = (Vec::new(), Vec::with_capacity(vertices.len() / 10));
    for v in vertices.chunks_exact(10) {
        let key: [u32; 10] = std::array::from_fn(|k| v[k].to_bits());
        let next = (unique.len() / 10) as u32;
        let k = *seen.entry(key).or_insert_with(|| {
            unique.extend_from_slice(v);
            next
        });
        index.push(k);
    }
    (unique, index)
}
