//! What a page (or the native viewer) plays: a [`Scene`] (Broadside or the timber yard)
//! with the render state around it: the pose buffer it draws (the draw shader's `BodyPose`,
//! [`POSE_FLOATS`] floats a body: the render scene's bodies (the ground, every chunk in
//! order, its parked projectiles), then the extras drawn as a dynamic mesh
//! ([`Play::extra_vertices`]): a captain, [`MAX_BALLS`] balls, and the crush splinters so
//! far), interpolated between steps. `lib.rs` binds it for the browser;
//! `engine-viewer`'s `world_viewer` draws it natively.

use super::broadside::{self, Broadside, Director, Take};
use super::range::{Range, Splinter, BALL_RADIUS, BALL_SPEED, MAX_BALLS};
use super::timber_yard::{TimberYard, YardTake};
use crate::{GpuContext, Transform};

/// Floats a body takes in the draw shader's pose buffer (`fracture_draw.slang` `BodyPose`:
/// seven float4; it reads the position and the rotation).
pub const POSE_FLOATS: usize = 28;
/// Floats a render vertex takes (position, normal, colour, body id).
pub const VERTEX_FLOATS: usize = 10;
/// Where a body not in play is drawn (far under the ground).
const PARKED: [f32; 3] = [0.0, -1.0e4, 0.0];
const COAT: [f32; 3] = [0.10, 0.12, 0.22];
const SKIN: [f32; 3] = [0.86, 0.68, 0.56];
const HAT: [f32; 3] = [0.05, 0.05, 0.06];
const BALL: [f32; 3] = [0.16, 0.16, 0.17];
/// Fresh-split oak, a crushed chunk's splinters.
const SPLINTER: [f32; 3] = [0.80, 0.64, 0.42];

/// Vertices each piece of debris takes ([`splinter_vertices`]): a box's 36.
pub const SPLINTER_VERTS: u32 = 36;

/// The scene a page plays.
pub enum Scene {
    Broadside(Broadside, Director),
    Yard(TimberYard),
}

impl Scene {
    pub fn range(&self) -> &Range {
        match self {
            Scene::Broadside(b, _) => b,
            Scene::Yard(y) => y,
        }
    }

    pub fn range_mut(&mut self) -> &mut Range {
        match self {
            Scene::Broadside(b, _) => b,
            Scene::Yard(y) => y,
        }
    }

    pub fn step(&mut self) -> anyhow::Result<()> {
        match self {
            Scene::Broadside(b, _) => b.step(),
            Scene::Yard(y) => y.step(),
        }
    }

    pub fn set_scripted(&mut self, on: bool) {
        match self {
            Scene::Broadside(b, _) => b.scripted = on,
            Scene::Yard(y) => y.scripted = on,
        }
    }

    /// The take's camera `wall` seconds after the last frame (none: the page's own).
    pub fn camera(&mut self, wall: f32) -> Option<([f32; 3], [f32; 3])> {
        match self {
            Scene::Broadside(b, d) => Some(d.camera(b, wall)),
            Scene::Yard(y) => y.camera(),
        }
    }

    pub fn time_scale(&self) -> f32 {
        match self {
            Scene::Broadside(b, _) => Director::time_scale(b),
            Scene::Yard(y) => y.time_scale(),
        }
    }

    /// (scripted, finale fired, over, intact share).
    pub fn script_state(&self) -> (bool, bool, bool, f32) {
        match self {
            Scene::Broadside(b, _) => (
                b.scripted,
                b.finale.is_some(),
                b.over(),
                b.razing.intact_share,
            ),
            Scene::Yard(y) => (
                y.scripted,
                y.finale.is_some(),
                y.over(),
                y.razing.intact_share,
            ),
        }
    }
}

pub struct Play {
    pub scene: Scene,
    /// The render scene's body count (`build_scene`'s bodies): the extras follow.
    pub render_bodies: usize,
    curr: Vec<Transform>,
    prev: Option<Vec<Transform>>,
}

impl Play {
    /// Build scene `kind` (0 Broadside, 1 the timber yard) from its baked GLB at fracture
    /// `level`, on `gpu`; `raze` runs the raze take (else Broadside's walk, the yard's free
    /// play). Waits for the GPU.
    pub fn load(
        gpu: GpuContext,
        glb: &[u8],
        kind: u8,
        level: u8,
        raze: bool,
    ) -> anyhow::Result<Self> {
        let scene = if kind == 1 {
            let take = if raze { YardTake::Raze } else { YardTake::Free };
            Scene::Yard(TimberYard::new(gpu, glb, level, take)?)
        } else {
            let take = if raze { Take::Raze } else { Take::Walk };
            Scene::Broadside(
                Broadside::with_take(gpu, glb, level, take)?,
                Director::default(),
            )
        };
        let render_bodies = scene.range().render_bodies;
        let mut l = Play {
            scene,
            render_bodies,
            curr: Vec::new(),
            prev: None,
        };
        l.curr = render_poses(&l);
        Ok(l)
    }

    /// The extras' render vertices (body-local, [`VERTEX_FLOATS`] floats each): the
    /// captain, the balls, the splinters so far. Redraw them when
    /// [`Play::extras_version`] changes.
    pub fn extra_vertices(&self) -> Vec<f32> {
        let mut out = Vec::new();
        let mut id = self.render_bodies as f32;
        captain_vertices(id, &mut out);
        id += 1.0;
        for _ in 0..MAX_BALLS {
            sphere_vertices(BALL_RADIUS, 8, BALL, id, &mut out);
            id += 1.0;
        }
        let first = id;
        for (k, s) in self.scene.range().splinters.iter().enumerate() {
            splinter_vertices(k as u32, s, first + k as f32, &mut out);
        }
        out
    }

    /// The vertex where debris `k`'s vertices start in [`Play::extra_vertices`] (the
    /// captain's and the balls' come first): new debris is appended there.
    pub fn splinter_vertex_offset(k: u32) -> u32 {
        let (mut captain, mut ball) = (Vec::new(), Vec::new());
        captain_vertices(0.0, &mut captain);
        sphere_vertices(BALL_RADIUS, 8, BALL, 0.0, &mut ball);
        let verts = |v: &Vec<f32>| (v.len() / VERTEX_FLOATS) as u32;
        verts(&captain) + verts(&ball) * MAX_BALLS as u32 + k * SPLINTER_VERTS
    }

    /// The render vertices of debris `from` onward (append them at
    /// [`Play::splinter_vertex_offset`]`(from)`).
    pub fn splinter_vertices(&self, from: u32) -> Vec<f32> {
        let first = (self.render_bodies + 1 + MAX_BALLS) as f32;
        let mut out = Vec::new();
        for (k, s) in self
            .scene
            .range()
            .splinters
            .iter()
            .enumerate()
            .skip(from as usize)
        {
            splinter_vertices(k as u32, s, first + k as f32, &mut out);
        }
        out
    }

    /// Debris so far (crush particles and fragments).
    pub fn splinter_count(&self) -> u32 {
        self.scene.range().splinters.len() as u32
    }

    /// Changes whenever [`Play::extra_vertices`] would.
    pub fn extras_version(&self) -> u32 {
        self.scene.range().splinters_version
    }

    /// Advance `steps` steps, then wait for their readbacks.
    pub fn frame(&mut self, steps: u32) -> anyhow::Result<()> {
        for i in 0..steps {
            self.scene.step()?;
            // The poses a step before the last, to draw between (several steps a frame).
            if i + 2 == steps {
                self.scene.range_mut().poll_blocking()?;
                self.curr = render_poses(self);
            }
        }
        if steps > 0 {
            self.scene.range_mut().poll_blocking()?;
            let now = render_poses(self);
            self.prev = Some(std::mem::replace(&mut self.curr, now));
        }
        Ok(())
    }

    /// The pose buffer at `alpha` (0..1) of the way from the step before the last polled
    /// one to it: a frame between two steps (slow motion) draws the bodies between them.
    /// Render only: the physics never sees it.
    pub fn poses(&self, alpha: f32) -> Vec<f32> {
        let a = alpha.clamp(0.0, 1.0);
        let mut out = vec![0.0f32; POSE_FLOATS * self.curr.len()];
        for (k, t) in self.curr.iter().enumerate() {
            let t = match self.prev.as_ref().and_then(|p| p.get(k)) {
                Some(p) if p.position != PARKED && t.position != PARKED => between(p, t, a),
                _ => *t,
            };
            out[POSE_FLOATS * k..POSE_FLOATS * k + 3].copy_from_slice(&t.position);
            out[POSE_FLOATS * k + 4..POSE_FLOATS * k + 8].copy_from_slice(&t.rotation);
        }
        out
    }

    /// Fire a ball from `origin` along `direction` (world, the camera's aim) at the take's
    /// speed, before the next step.
    pub fn shoot_ray(&mut self, origin: [f32; 3], direction: [f32; 3]) {
        let d = glam::Vec3::from(direction).normalize_or_zero() * BALL_SPEED;
        let from = glam::Vec3::from(origin) + d.normalize_or_zero() * 1.0;
        self.scene.range_mut().shoot(from.to_array(), d.to_array());
    }

    /// The scene's state as JSON: steps, simulated time, the step rate, shots, bonds
    /// broken, splits, chunks crushed, pieces, whether the take is over.
    pub fn stats(&self) -> serde_json::Value {
        let b = self.scene.range();
        let d = b.world.destructible_stats(b.structure).ok();
        let (scripted, finale, over, intact) = self.scene.script_state();
        serde_json::json!({
            "frames": b.frame,
            "simSeconds": b.seconds(),
            "frameHz": 1.0 / b.dt,
            "shots": b.total.shots,
            "brokenBonds": b.total.broken,
            "splits": b.total.splits,
            "crushed": b.total.crushed,
            "chunks": d.map_or(0, |d| d.chunks),
            "bonds": d.map_or(0, |d| d.bonds),
            "pieces": d.map_or(0, |d| d.actors),
            "splinters": b.splinters.len(),
            "intact": intact,
            "scripted": scripted,
            "finale": finale,
            "over": over,
        })
    }
}

/// Debris `k`'s vertices, all [`SPLINTER_VERTS`] of them: a crush particle as one thin
/// pointed sliver of fresh-split wood centred on its body (length, thickness and shade of
/// its own), a box fragment as its box (padded to the same count with degenerate triangles).
fn splinter_vertices(k: u32, s: &Splinter, id: f32, out: &mut Vec<f32>) {
    let h = |i: u32| {
        let x = k.wrapping_mul(0x9E37_79B1) ^ i.wrapping_mul(0x85EB_CA77);
        let x = (x ^ (x >> 15)).wrapping_mul(0x2C1B_3C6D);
        ((x ^ (x >> 13)) >> 8) as f32 / (1u32 << 24) as f32
    };
    let start = out.len();
    if s.particle {
        let r = s.half[0];
        let shade = 0.78 + 0.34 * h(0);
        let color = SPLINTER.map(|c| (c * shade).min(1.0));
        let (len, t) = (r * (2.2 + 2.0 * h(1)), r * (0.22 + 0.16 * h(2)));
        // A wedge: a point at one end, a small triangle at the other.
        let p = [
            glam::Vec3::new(0.5 * len, 0.0, 0.0),
            glam::Vec3::new(-0.5 * len, t, 0.0),
            glam::Vec3::new(-0.5 * len, -0.5 * t, 0.87 * t),
            glam::Vec3::new(-0.5 * len, -0.5 * t, -0.87 * t),
        ];
        let centre = (p[0] + p[1] + p[2] + p[3]) * 0.25;
        for [a, b, c] in [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]] {
            let mut n = (p[b] - p[a]).cross(p[c] - p[a]).normalize_or_zero();
            let (mut b, mut c) = (b, c);
            if n.dot(p[a] - centre) < 0.0 {
                n = -n;
                std::mem::swap(&mut b, &mut c);
            }
            for v in [a, b, c] {
                vertex((p[v] - centre).to_array(), n.to_array(), color, id, out);
            }
        }
    } else {
        let shade = 0.82 + 0.3 * h(0);
        box_vertices(s.half, SPLINTER.map(|c| (c * shade).min(1.0)), id, out);
    }
    // The same count for every piece (the page appends by index).
    let want = start + SPLINTER_VERTS as usize * VERTEX_FLOATS;
    while out.len() < want {
        vertex([0.0; 3], [0.0, 1.0, 0.0], SPLINTER, id, out);
    }
    out.truncate(want);
}

fn between(p: &Transform, q: &Transform, a: f32) -> Transform {
    let x = glam::Vec3::from(p.position).lerp(glam::Vec3::from(q.position), a);
    let r = glam::Quat::from_array(p.rotation).slerp(glam::Quat::from_array(q.rotation), a);
    Transform::new(x.to_array(), r.normalize().to_array())
}

fn parked() -> Transform {
    Transform::new(PARKED, [0.0, 0.0, 0.0, 1.0])
}

/// Every drawn body's pose as of the last polled step (the module notes' order).
fn render_poses(l: &Play) -> Vec<Transform> {
    let b = l.scene.range();
    let mut poses = vec![Transform::IDENTITY];
    for (c, t) in b.chunk_poses().into_iter().enumerate() {
        poses.push(if b.crushed[c] { parked() } else { t });
    }
    poses.resize(l.render_bodies, parked());
    // The captain is a scripted point for now (no figure drawn).
    poses.push(parked());
    let balls = b.ball_poses();
    for k in 0..MAX_BALLS {
        poses.push(balls.get(k).copied().unwrap_or_else(parked));
    }
    for (k, s) in b.splinters.iter().enumerate() {
        let Some(t) = b.world.body_transform(s.body).ok().flatten() else {
            poses.push(parked());
            continue;
        };
        poses.push(Transform::new(
            t.position,
            tumble(k as u32, s, t).to_array(),
        ));
    }
    poses
}

/// A splinter's drawn turn: a crush particle never rotates, so the page tumbles its
/// splinter about an axis of its own by how far it has travelled (render only; it
/// settles when the particle does). A box fragment keeps its body's rotation.
fn tumble(k: u32, s: &Splinter, t: Transform) -> glam::Quat {
    let body = glam::Quat::from_array(t.rotation);
    if !s.particle {
        return body;
    }
    let (half, at) = (s.half, s.at);
    let h = |s: u32| {
        let x = k.wrapping_mul(0x9E37_79B1) ^ s.wrapping_mul(0x85EB_CA77);
        ((x ^ (x >> 15)).wrapping_mul(0x2C1B_3C6D) >> 8) as f32 / (1u32 << 24) as f32
    };
    let axis = glam::Vec3::new(h(1) - 0.5, h(2) - 0.5, h(3) - 0.5).normalize_or(glam::Vec3::Y);
    let start = glam::Quat::from_euler(glam::EulerRot::XYZ, 6.28 * h(4), 6.28 * h(5), 6.28 * h(6));
    let travelled = (glam::Vec3::from(t.position) - glam::Vec3::from(at)).length();
    glam::Quat::from_axis_angle(axis, travelled / half[0].max(0.005) * 0.3) * start
}

/// A simple figure standing on its origin's... the capsule centre: coat, head, hat.
fn captain_vertices(id: f32, out: &mut Vec<f32>) {
    use broadside::{CAPTAIN_HALF, CAPTAIN_RADIUS};
    let foot = -(CAPTAIN_HALF + CAPTAIN_RADIUS);
    // Coat (a tapered box would do; two boxes read as coat and legs).
    box_vertices_at(
        [0.0, foot + 0.45, 0.0],
        [0.13, 0.45, 0.17],
        [0.07, 0.07, 0.08],
        id,
        out,
    );
    box_vertices_at([0.0, foot + 1.2, 0.0], [0.15, 0.32, 0.24], COAT, id, out);
    let mut head = Vec::new();
    sphere_vertices(0.12, 8, SKIN, id, &mut head);
    offset(&mut head, [0.0, foot + 1.66, 0.0]);
    out.extend(head);
    // A tricorn's crown and brim.
    box_vertices_at([0.0, foot + 1.82, 0.0], [0.11, 0.06, 0.11], HAT, id, out);
    box_vertices_at([0.0, foot + 1.77, 0.0], [0.22, 0.015, 0.22], HAT, id, out);
}

fn offset(v: &mut [f32], by: [f32; 3]) {
    for p in v.chunks_exact_mut(VERTEX_FLOATS) {
        for k in 0..3 {
            p[k] += by[k];
        }
    }
}

fn box_vertices_at(at: [f32; 3], half: [f32; 3], color: [f32; 3], id: f32, out: &mut Vec<f32>) {
    let mut v = Vec::new();
    box_vertices(half, color, id, &mut v);
    offset(&mut v, at);
    out.extend(v);
}

/// A box's faces (`half` extents), its body `id`.
fn box_vertices(half: [f32; 3], color: [f32; 3], id: f32, out: &mut Vec<f32>) {
    for axis in 0..3 {
        for sign in [-1.0f32, 1.0] {
            let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
            let corner = |a: f32, b: f32| {
                let mut p = [0.0f32; 3];
                p[axis] = sign * half[axis];
                p[u] = a * half[u];
                p[v] = b * half[v];
                p
            };
            let mut n = [0.0f32; 3];
            n[axis] = sign;
            let quad = [
                corner(-1.0, -1.0),
                corner(1.0, -1.0),
                corner(1.0, 1.0),
                corner(-1.0, 1.0),
            ];
            let tri = if sign > 0.0 {
                [0, 1, 2, 0, 2, 3]
            } else {
                [0, 2, 1, 0, 3, 2]
            };
            for k in tri {
                vertex(quad[k], n, color, id, out);
            }
        }
    }
}

/// A sphere of `radius`, `rings` rings of twice as many segments.
fn sphere_vertices(radius: f32, rings: usize, color: [f32; 3], id: f32, out: &mut Vec<f32>) {
    let segs = 2 * rings;
    let n = |i: usize, j: usize| {
        let (th, ph) = (
            std::f32::consts::PI * i as f32 / rings as f32,
            std::f32::consts::TAU * j as f32 / segs as f32,
        );
        [th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin()]
    };
    for i in 0..rings {
        for j in 0..segs {
            let q = [n(i, j), n(i + 1, j), n(i + 1, j + 1), n(i, j + 1)];
            for k in [0, 1, 2, 0, 2, 3] {
                vertex(q[k].map(|x| x * radius), q[k], color, id, out);
            }
        }
    }
}

fn vertex(p: [f32; 3], n: [f32; 3], color: [f32; 3], id: f32, out: &mut Vec<f32>) {
    out.extend_from_slice(&p);
    out.extend_from_slice(&n);
    out.extend_from_slice(&color);
    out.push(id);
}

/// A range scene as a [`super::SceneDef`]: Broadside or the timber yard, its fracture
/// level and its take, as the range page reads them (`?scene=`, `?level=`, `?take=raze`).
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct RangeSceneDef {
    /// `broadside` or `timber`.
    pub scene: String,
    pub level: u8,
    pub raze: bool,
}

impl Default for RangeSceneDef {
    fn default() -> Self {
        RangeSceneDef {
            scene: "broadside".into(),
            level: 2,
            raze: false,
        }
    }
}

impl RangeSceneDef {
    pub(crate) fn from_query(q: &super::scene::Query) -> anyhow::Result<Self> {
        let scene = q.get("scene").unwrap_or("broadside");
        anyhow::ensure!(
            scene == "broadside" || scene == "timber",
            "range scene {scene}: broadside or timber"
        );
        // The page's default on a desktop (fine pointer): level 2.
        let level = match q.get("level") {
            Some(l) => super::scene::js_number(l) as u8,
            None => 2,
        };
        Ok(RangeSceneDef {
            scene: scene.into(),
            level,
            raze: q.get("take") == Some("raze"),
        })
    }

    /// 0 Broadside, 1 the timber yard ([`Play::load`]).
    pub fn kind(&self) -> u8 {
        if self.scene == "timber" {
            1
        } else {
            0
        }
    }

    /// Its baked GLB (`engine-webgpu/fracture/assets`).
    pub fn asset(&self) -> &'static str {
        if self.scene == "timber" {
            "timber-yard.glb"
        } else {
            "galleon-stern.glb"
        }
    }
}
