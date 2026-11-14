//! gen3: the crash physics rigged onto the generated sedan's render model
//! ([`super::model`]), the way an artist's `.glb` to the vehicle art spec would
//! be rigged. Everything is placed from the model's public data, so a restyle
//! of the surface flows through:
//!
//! * the body surface's public API ([`super::surface`]: the section rays, the
//!   belt and shoulder lines, the arches, the greenhouse corner);
//! * each part's mesh extents (where its skin is, read from its vertices);
//! * the spec's markers (wheel centres, hinges, latches, bolts, H-points).
//!
//! The lattice follows the BeamNG census (`beamng-authoring-census.md`):
//! structural nodes about 0.5 m apart on feature lines and panel edges (2–3 cm
//! in from them), panels with real depth (an inner frame off the skin, so no
//! panel is a flat sheet of in-plane beams), the body as members (rails, crush
//! cans, bumper beams, sills, pillars, roof rails and bows, tunnel, floor
//! cross members, firewall), light contact-only nodes in the pole lanes, and
//! hardware from the markers (hinges as break groups, latches as couplers,
//! bolts as break groups). Beams are a neighbourhood graph within each part
//! (every node to its part's nodes within a reach, none across an opening),
//! which gives the census's 12–18 spring ends per node; the builder's rules
//! ([`super::build`]) set the masses, springs, damping and yield.
//!
//! Frames: the model is glTF (+Y up, +Z front, +X left, origin between the
//! axles); the JBeam car is +X left, +Y rear, +Z up, the front axle at y = 0
//! ([`to_jbeam`]).

use super::build::*;
use super::model::Model;
use super::surface::{self as sf, Param, ARCH_R, AXLE_F, AXLE_R, TRACK_HALF, TYRE_R};
use crate::authoring::sedan::{
    ACCORD_BIW, ACCORD_DASH_BEHIND, ACCORD_DECKLID, ACCORD_DOOR_F, ACCORD_DOOR_R, ACCORD_ENGINE_KG,
    ACCORD_FENDER, ACCORD_HOOD, ACCORD_PT_AHEAD, ACCORD_PT_BEHIND, ACCORD_TRANSAXLE_KG,
    POWERTRAIN_CLEARANCE,
};
use glam::Vec3;
use std::collections::HashMap;

/// Kerb mass with the driver (kg).
pub const KERB: f32 = 1450.0;
/// Mass each raycast wheel adds to its two hub nodes (the loader's 20 kg).
pub const WHEEL_KG: f32 = 20.0;
/// Bumper beam assemblies and fascias (kg), DOT HS 811 666 §5.10.7.
pub const BUMPER_F_KG: f32 = 7.96 + 2.5;
pub const BUMPER_R_KG: f32 = 7.84 + 1.3;
/// Wheel proxy rim nodes per wheel, their radius and mass.
pub const RIM_NODES: usize = 6;
pub const RIM_R: f32 = 0.25;
pub const RIM_NODE_KG: f32 = 2.0;
/// Unsprung mass per corner on the hub nodes besides the wheel (kg).
pub const UNSPRUNG_KG: f32 = 30.0 - RIM_NODES as f32 * RIM_NODE_KG;
/// Plastic compression limit of the crush zones' beams (deformLimit).
pub const STACK_LIMIT: f32 = 0.7;
/// Bolt strengths per beam (N): fender bolts and its front bracket, the
/// bumper covers' bolts and their side mounts (a 56 km/h wall unbolted
/// both fenders and both covers at 9 / 6 / 4 kN).
pub const FENDER_BOLT: f32 = 25e3;
pub const BUMPER_BOLT: f32 = 20e3;
pub const BUMPER_SIDE_BOLT: f32 = 15e3;
/// A fender's flange bolts down the apron rail and onto the hinge pillar: an
/// M6 property class 8.8 bolt's ultimate tensile load (20.1 mm² stress area
/// × 800 MPa = 16 kN; ISO 898-1).
pub const FENDER_ROW_BOLT: f32 = 16e3;
/// An engine mount beam's strength (N): the round-3 value per beam, now in
/// four break groups (one per block corner).
pub const ENGINE_MOUNT: f32 = 60e3;
/// The cooling module and radiator support crushed solid between the block
/// and the bumper beam (m): a single-row aluminium radiator core (about
/// 27 mm), a condenser (about 16 mm) and the tie bar's, shroud's and fan's
/// flattened sheet and hub. An estimate from those parts' depths, not a
/// measured Accord value; `GEN3_COOLING_STACK` sweeps it.
pub const COOLING_STACK: f32 = 0.08;
/// A front wheel's hub centre to the hinge pillar once its tyre is crushed
/// onto the rim (m): the 16-inch rim's 0.20 m radius plus 3 cm of flattened
/// tyre and brake hardware.
pub const WHEEL_STACK: f32 = 0.23;
/// Neighbourhood reach of the beam graph (m): the body's members and each
/// closure's frame.
pub const REACH_BODY: f32 = 0.90;
pub const REACH_PANEL: f32 = 0.66;

/// glTF (model) to JBeam (car) coordinates.
pub fn to_jbeam(g: Vec3) -> Vec3 {
    Vec3::new(g.x, AXLE_F - g.z, g.y)
}

/// JBeam to glTF.
pub fn to_gltf(j: Vec3) -> Vec3 {
    Vec3::new(j.x, j.z, AXLE_F - j.y)
}

// ---------------------------------------------------------------------------
// Surface helpers (glTF frame, left side x > 0).
// ---------------------------------------------------------------------------

/// The surface on section `z` at height `y` (left side).
pub fn side(z: f32, y: f32) -> Vec3 {
    sf::hit(Param::Section, z, sf::theta_at_y(z, y))
}

/// The top surface on section `z` at half width `x`.
pub fn top(z: f32, x: f32) -> Vec3 {
    sf::hit(Param::Section, z, sf::theta_at_x(z, x))
}

/// The greenhouse's corner (roof rail, A/C-pillar edge) on section `z`.
pub fn corner(z: f32) -> Vec3 {
    sf::hit(Param::Section, z, sf::theta_corner(z))
}

/// A point `d` metres inside the surface from the surface point `p`.
pub fn inward(p: Vec3, d: f32) -> Vec3 {
    p - sf::normal(p) * d
}

/// The nose (`front`) or tail surface at half width `x`, height `y`.
pub fn end_pt(front: bool, x: f32, y: f32) -> Vec3 {
    let param = if front { Param::Nose } else { Param::Tail };
    let (mut lo, mut hi) = (0.0f32, std::f32::consts::FRAC_PI_2);
    for _ in 0..40 {
        let m = 0.5 * (lo + hi);
        if sf::hit(param, m, y).x < x {
            lo = m;
        } else {
            hi = m;
        }
    }
    sf::hit(param, 0.5 * (lo + hi), y)
}

fn mirror(p: Vec3, s: f32) -> Vec3 {
    Vec3::new(p.x * s, p.y, p.z)
}

fn lr(s: f32) -> &'static str {
    if s > 0.0 {
        "L"
    } else {
        "R"
    }
}

/// A model part's skin extents (glTF), read from its vertices.
struct Skin {
    pos: Vec<Vec3>,
    nrm: Vec<Vec3>,
}

impl Skin {
    fn of(m: &Model, name: &str) -> Skin {
        let p = m
            .parts
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("the model has no part {name}"));
        Skin {
            pos: p.pos.clone(),
            nrm: p.nrm.clone(),
        }
    }
    /// Outward-facing vertices only (the skin, not its inner panel or hems).
    fn outer(&self, out: impl Fn(Vec3, Vec3) -> bool) -> Vec<Vec3> {
        self.pos
            .iter()
            .zip(&self.nrm)
            .filter(|(p, n)| out(**p, **n))
            .map(|(p, _)| *p)
            .collect()
    }
}

fn bounds(v: &[Vec3]) -> (Vec3, Vec3) {
    v.iter().fold(
        (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
        |(a, b), p| (a.min(*p), b.max(*p)),
    )
}

/// Lowest y of `v` within `dz` of section `z` (None if none).
fn low_at(v: &[Vec3], z: f32, dz: f32) -> Option<f32> {
    v.iter()
        .filter(|p| (p.z - z).abs() < dz)
        .map(|p| p.y)
        .reduce(f32::min)
}

/// Largest |x| of `v` within `dz` of section `z`.
fn half_w_at(v: &[Vec3], z: f32, dz: f32) -> Option<f32> {
    v.iter()
        .filter(|p| (p.z - z).abs() < dz)
        .map(|p| p.x.abs())
        .reduce(f32::max)
}

/// Smallest |x| of `v` within `dz` of section `z`.
fn inner_x_at(v: &[Vec3], z: f32, dz: f32) -> Option<f32> {
    v.iter()
        .filter(|p| (p.z - z).abs() < dz)
        .map(|p| p.x.abs())
        .reduce(f32::min)
}

fn marker(m: &Model, name: &str) -> Vec3 {
    m.markers
        .iter()
        .find(|k| k.name == name)
        .unwrap_or_else(|| panic!("the model has no marker {name}"))
        .pos
}

/// A number as JSON: an integer when it is one (as the rig's literals were).
fn json_num(x: f32) -> serde_json::Value {
    if x.fract() == 0.0 && x.abs() < 1e9 {
        serde_json::json!(x as i64)
    } else {
        serde_json::json!(x)
    }
}

/// Evenly spaced values from `a` to `b` (both included), `n` of them.
fn span(a: f32, b: f32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| a + (b - a) * k as f32 / (n - 1).max(1) as f32)
        .collect()
}

// ---------------------------------------------------------------------------
// The rigger's input: the render model, its body surface and the physics
// block. Two implementations: [`Gen3Input`] (the Rust render model on its
// analytic surface, [`super::surface`]) and [`super::vg_input::VgInput`] (a
// vehicle-gen design's `.glb` and `rig.json`).
// ---------------------------------------------------------------------------

/// The running gear's dimensions (glTF metres).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dims {
    /// Front and rear axle z.
    pub axle_f: f32,
    pub axle_r: f32,
    /// Tyre radius (the wheel centre's height at design).
    pub tyre_r: f32,
    /// Half the track.
    pub track_half: f32,
    /// The wheel-arch opening's radius.
    pub arch_r: f32,
}

pub use engine_vehiclegen::physics::Physics;

/// What the rigger reads: the render model (part meshes and markers), the
/// body surface (a signed field, the belt and shoulder lines, the arches and
/// the end faces) and the physics block. The section queries have default
/// implementations on the field (ray marching, as the Rust surface does);
/// [`Gen3Input`] forwards them to the analytic surface's own functions, so
/// gen3 is unchanged bit for bit.
pub trait RigInput {
    fn model(&self) -> &Model;
    fn dims(&self) -> Dims;
    fn physics(&self) -> &Physics;
    /// The physics part a mesh rides (None: a wheel's piece).
    fn part_of(&self, mesh: &str) -> Option<Part> {
        super::gen3::part_of(mesh)
    }
    /// The body's signed field (negative inside, about metres near the skin).
    fn sdf(&self, q: Vec3) -> f32;
    /// Height of the belt (glass base) and shoulder lines at z.
    fn belt(&self, z: f32) -> f32;
    fn shoulder_y(&self, z: f32) -> f32;
    /// Height of the wheel-arch opening's edge at z (None outside it).
    fn arch_y(&self, z: f32) -> Option<f32>;
    /// The nose (`front`) or tail surface at half width `x`, height `y`.
    fn end_pt(&self, front: bool, x: f32, y: f32) -> Vec3;
    /// The section rays' origin height.
    fn ray_y(&self) -> f32;

    fn grad(&self, q: Vec3) -> Vec3 {
        let e = 2e-4;
        let f = |d: Vec3| self.sdf(q + d);
        Vec3::new(
            f(Vec3::X * e) - f(-Vec3::X * e),
            f(Vec3::Y * e) - f(-Vec3::Y * e),
            f(Vec3::Z * e) - f(-Vec3::Z * e),
        ) / (2.0 * e)
    }
    fn normal(&self, q: Vec3) -> Vec3 {
        self.grad(q).normalize_or(Vec3::Y)
    }
    /// Pull a point near the surface onto it (Newton along the gradient).
    fn project(&self, mut q: Vec3) -> Vec3 {
        for _ in 0..6 {
            let f = self.sdf(q);
            let g = self.grad(q);
            let l2 = g.length_squared().max(1e-8);
            q -= g * (f / l2);
            if f.abs() < 1e-6 {
                break;
            }
        }
        q
    }
    /// The section ray on plane z at angle θ from straight up (toward +X):
    /// its first crossing out of the body.
    fn hit(&self, z: f32, th: f32) -> Vec3 {
        let o = Vec3::new(0.0, self.ray_y(), z);
        let d = Vec3::new(th.sin(), th.cos(), 0.0);
        let f = |t: f32| self.sdf(o + d * t);
        let mut t0 = 0.0;
        let mut f0 = f(0.0);
        if f0 >= 0.0 {
            return o;
        }
        let mut t1;
        loop {
            t1 = t0 + (-f0 * 0.8).max(0.002);
            let f1 = f(t1);
            if f1 >= 0.0 || t1 > 4.0 {
                break;
            }
            t0 = t1;
            f0 = f1;
        }
        for _ in 0..30 {
            let m = 0.5 * (t0 + t1);
            if f(m) < 0.0 {
                t0 = m;
            } else {
                t1 = m;
            }
        }
        o + d * (0.5 * (t0 + t1))
    }
    /// θ on section `z` where the surface (left side) is at height `y`.
    fn theta_at_y(&self, z: f32, y: f32) -> f32 {
        let (mut lo, mut hi) = (0.02f32, 3.12f32);
        for _ in 0..40 {
            let m = 0.5 * (lo + hi);
            if self.hit(z, m).y > y {
                lo = m;
            } else {
                hi = m;
            }
        }
        0.5 * (lo + hi)
    }
    /// θ on section `z` where the top (left) is at half width `x`.
    fn theta_at_x(&self, z: f32, x: f32) -> f32 {
        let (mut lo, mut hi) = (0.0f32, 1.55f32);
        for _ in 0..40 {
            let m = 0.5 * (lo + hi);
            if self.hit(z, m).x < x {
                lo = m;
            } else {
                hi = m;
            }
        }
        0.5 * (lo + hi)
    }
    /// θ of the greenhouse's corner: where the normal leans 50° from up.
    fn theta_corner(&self, z: f32) -> f32 {
        let (mut lo, mut hi) = (0.05f32, 1.5f32);
        for _ in 0..40 {
            let m = 0.5 * (lo + hi);
            let n = self.normal(self.hit(z, m));
            if n.y > 50f32.to_radians().cos() {
                lo = m;
            } else {
                hi = m;
            }
        }
        0.5 * (lo + hi)
    }
    /// The surface on section `z` at height `y` (left side).
    fn side(&self, z: f32, y: f32) -> Vec3 {
        self.hit(z, self.theta_at_y(z, y))
    }
    /// The top surface on section `z` at half width `x`.
    fn top(&self, z: f32, x: f32) -> Vec3 {
        self.hit(z, self.theta_at_x(z, x))
    }
    /// The greenhouse's corner (roof rail, A/C-pillar edge) on section `z`.
    fn corner(&self, z: f32) -> Vec3 {
        self.hit(z, self.theta_corner(z))
    }
    /// A point `d` metres inside the surface from the surface point `p`.
    fn inward(&self, p: Vec3, d: f32) -> Vec3 {
        p - self.normal(p) * d
    }
    /// The rear closure's mesh: the boot lid, or a tailgate (its markers
    /// are the input's `hinge_trunk_*` and `latch_trunk` then).
    fn rear_closure(&self) -> String {
        "trunk_lid".into()
    }
    /// The collision skin from the render skin (`skin_hull`) instead of the
    /// panel grids' triangles; gen3's lattice predates it.
    fn hull_triangles(&self) -> bool {
        false
    }
    /// The shut lines' points, when the input knows them: (point on mesh
    /// `k`, the matching point on mesh `k2`, k, k2), model part indices.
    /// None: the rigger pairs the meshes' boundary vertices within 1 cm.
    fn seam_points(&self) -> Option<Vec<(Vec3, Vec3, usize, usize)>> {
        None
    }
    /// The design's body-in-white and mechanical parts, each defined once
    /// (`engine_vehiclegen::{structure, mechanical}`), when it has them: the
    /// rigger then builds the body, the powertrain, the running gear and the
    /// interior from those definitions ([`super::body2`]) instead of laying
    /// its own members on the skin.
    fn definitions(&self) -> Option<super::body2::Defs<'_>> {
        None
    }
}

/// gen3's own input: the Rust render model ([`super::model`]) on its analytic
/// surface ([`super::surface`]), with the census / Accord physics.
pub struct Gen3Input<'a> {
    pub model: &'a Model,
    pub physics: Physics,
}

impl<'a> Gen3Input<'a> {
    pub fn new(model: &'a Model) -> Self {
        Self {
            model,
            physics: Physics::default(),
        }
    }
}

impl RigInput for Gen3Input<'_> {
    fn model(&self) -> &Model {
        self.model
    }
    fn dims(&self) -> Dims {
        Dims {
            axle_f: AXLE_F,
            axle_r: AXLE_R,
            tyre_r: TYRE_R,
            track_half: TRACK_HALF,
            arch_r: ARCH_R,
        }
    }
    fn physics(&self) -> &Physics {
        &self.physics
    }
    fn sdf(&self, q: Vec3) -> f32 {
        sf::sdf(q)
    }
    fn belt(&self, z: f32) -> f32 {
        sf::with_profiles(|p| p.belt.at(z))
    }
    fn shoulder_y(&self, z: f32) -> f32 {
        sf::shoulder_y(z)
    }
    fn arch_y(&self, z: f32) -> Option<f32> {
        sf::arch_y(z)
    }
    fn end_pt(&self, front: bool, x: f32, y: f32) -> Vec3 {
        let param = if front { Param::Nose } else { Param::Tail };
        let (mut lo, mut hi) = (0.0f32, std::f32::consts::FRAC_PI_2);
        for _ in 0..40 {
            let m = 0.5 * (lo + hi);
            if sf::hit(param, m, y).x < x {
                lo = m;
            } else {
                hi = m;
            }
        }
        sf::hit(param, 0.5 * (lo + hi), y)
    }
    fn ray_y(&self) -> f32 {
        sf::YC
    }
    fn grad(&self, q: Vec3) -> Vec3 {
        sf::grad(q)
    }
    fn normal(&self, q: Vec3) -> Vec3 {
        sf::normal(q)
    }
    fn project(&self, q: Vec3) -> Vec3 {
        sf::project(q)
    }
    fn hit(&self, z: f32, th: f32) -> Vec3 {
        sf::hit(Param::Section, z, th)
    }
    fn theta_at_y(&self, z: f32, y: f32) -> f32 {
        sf::theta_at_y(z, y)
    }
    fn theta_at_x(&self, z: f32, x: f32) -> f32 {
        sf::theta_at_x(z, x)
    }
    fn theta_corner(&self, z: f32) -> f32 {
        sf::theta_corner(z)
    }
}

/// The builder's car plus the rig's per-node member tags.
pub struct Rig {
    pub car: Car,
    /// Per node its member (the beam graph's role between two nodes of one member).
    pub member: Vec<Option<Role>>,
    /// Per node: an outer skin node (on the panel surface).
    pub skin: Vec<bool>,
    /// The front axle's glTF z (the JBeam frame's origin along the car).
    pub axle_f: f32,
}

impl Rig {
    pub(super) fn node(&mut self, name: &str, g: Vec3, part: Part, member: Option<Role>, skin: bool) -> usize {
        let i = self.car.node(name, self.to_jbeam(g), part);
        self.member.push(member);
        self.skin.push(skin);
        i
    }
    pub(super) fn g(&self, i: usize) -> Vec3 {
        self.to_gltf(self.car.p(i))
    }
    /// glTF (model) to JBeam (car) coordinates, about this car's front axle.
    pub fn to_jbeam(&self, g: Vec3) -> Vec3 {
        Vec3::new(g.x, self.axle_f - g.z, g.y)
    }
    /// JBeam to glTF.
    pub fn to_gltf(&self, j: Vec3) -> Vec3 {
        Vec3::new(j.x, j.z, self.axle_f - j.y)
    }
    /// Nodes of `part` (structural) nearest a glTF point.
    pub(super) fn nearest(&self, part: Part, g: Vec3, k: usize, filter: impl Fn(usize) -> bool) -> Vec<usize> {
        let mut v: Vec<(f32, usize)> = (0..self.car.nodes.len())
            .filter(|&i| self.car.nodes[i].part == part && !self.car.nodes[i].contact && filter(i))
            .map(|i| (self.g(i).distance(g), i))
            .collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        v.into_iter().take(k).map(|x| x.1).collect()
    }
}

/// A door's aperture in the body side, from the door's mesh: per section
/// its lowest and highest skin point (binned every 2 cm along z).
struct Aperture {
    z0: f32,
    z1: f32,
    lo: Vec<f32>,
    hi: Vec<f32>,
}

impl Aperture {
    fn of(pts: &[Vec3]) -> Aperture {
        let (lo3, hi3) = bounds(pts);
        let n = (((hi3.z - lo3.z) / 0.02).ceil() as usize).max(1);
        let mut lo = vec![f32::MAX; n + 1];
        let mut hi = vec![f32::MIN; n + 1];
        for p in pts {
            let k = (((p.z - lo3.z) / 0.02).round() as usize).min(n);
            lo[k] = lo[k].min(p.y);
            hi[k] = hi[k].max(p.y);
        }
        // Fill empty bins from their neighbours.
        for k in 1..=n {
            if lo[k] == f32::MAX {
                lo[k] = lo[k - 1];
                hi[k] = hi[k - 1];
            }
        }
        for k in (0..n).rev() {
            if lo[k] == f32::MAX {
                lo[k] = lo[k + 1];
                hi[k] = hi[k + 1];
            }
        }
        Aperture {
            z0: lo3.z,
            z1: hi3.z,
            lo,
            hi,
        }
    }
    /// Inside the opening (a margin `m` in from its edges), on the side.
    fn inside(&self, p: Vec3, m: f32) -> bool {
        if p.x.abs() < 0.45 || p.z < self.z0 + m || p.z > self.z1 - m {
            return false;
        }
        let k = (((p.z - self.z0) / 0.02).round() as usize).min(self.lo.len() - 1);
        p.y > self.lo[k] + m && p.y < self.hi[k] - m
    }
}

/// Regions a body beam may not cross (glTF): the door apertures, the cabin,
/// the engine bay's top and the trunk opening.
fn body_blocked(inp: &dyn RigInput, p: Vec3, doors: &[Aperture], trunk_z: (f32, f32)) -> bool {
    let x = p.x.abs();
    if doors.iter().any(|d| d.inside(p, 0.055)) {
        return true;
    }
    let cabin = in_cabin(inp, p);
    let bay = x < 0.48 && p.y > 0.72 && p.z > 0.98 && p.z < 1.98;
    // The trunk opening (under the lid, from its mesh), clear of the shelf
    // and tail panel at its ends.
    let trunk = x < 0.58 && p.y > 0.86 && p.y < 1.06 && p.z > trunk_z.0 && p.z < trunk_z.1;
    cabin || bay || trunk
}

/// The collision skin's `dragCoef` for a car drag coefficient `cd`. A
/// triangle's drag is ½ρ·1.28·(dragCoef/100)·A·|u|²·|cos α|
/// (`explicit_jbeam_ext::aero_tri_force`, both faces), so a skin of
/// triangles drags ½ρ|u|²·1.28·(dragCoef/100)·Σ A|n·ẑ| head on; the car
/// should drag ½ρ|u|²·cd·A_f, A_f its frontal area (the render model's
/// silhouette on the x-y plane, rasterised at 1 cm).
fn hull_drag_coef(r: &Rig, inp: &dyn RigInput, cd: f32) -> f32 {
    let s: f32 = r
        .car
        .tris
        .iter()
        .map(|t| {
            let [a, b, c] = t.n.map(|i| r.g(i));
            let n = (b - a).cross(c - a);
            0.5 * n.z.abs()
        })
        .sum();
    let cell = 0.01f32;
    let mut on: std::collections::HashSet<(i32, i32)> = Default::default();
    for m in &inp.model().parts {
        for t in &m.tris {
            let [a, b, c] = t.map(|k| m.pos[k as usize]);
            let span = (a - b).length().max((b - c).length()).max((c - a).length());
            let k = ((span / cell).ceil() as usize).clamp(1, 64);
            for i in 0..=k {
                for j in 0..=k - i {
                    let (u, v) = (i as f32 / k as f32, j as f32 / k as f32);
                    let p = a + (b - a) * u + (c - a) * v;
                    on.insert(((p.x / cell).floor() as i32, (p.y / cell).floor() as i32));
                }
            }
        }
    }
    let af = on.len() as f32 * cell * cell;
    100.0 * cd * af / (crate::explicit_jbeam_ext::FLAT_PLATE_CD * s.max(1e-3))
}

/// Render meshes on the hull (outer skin, glass and lenses), by name.
fn hull_mesh(name: &str) -> bool {
    name == "body_shell"
        || name == "hood"
        || name == "trunk_lid"
        || name == "tailgate"
        || name.starts_with("glass_")
        || name.starts_with("fender_")
        || name.starts_with("bumper_")
        || (name.starts_with("door_") && !name.starts_with("door_trim"))
        || name.ends_with("_lens")
}

/// The collision skin as the render skin's restricted Delaunay triangulation
/// over each part's skin nodes (see the call). Replaces the panel grids'
/// triangles.
fn skin_hull(r: &mut Rig, inp: &dyn RigInput) {
    let model = inp.model();
    // Each part's nodes at its skin: structural, within 10 cm of the surface.
    let mut pool: HashMap<Part, Vec<usize>> = HashMap::new();
    for i in 0..r.car.nodes.len() {
        let n = &r.car.nodes[i];
        if n.contact
            || n.no_group
            || n.sub.is_some()
            || n.name.starts_with("hub")
            || n.name.starts_with("rim")
        {
            continue;
        }
        let within = std::env::var("RIG_HULL_POOL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.03f32);
        if inp.sdf(r.g(i)).abs() < within {
            if std::env::var("RIG_HULL_DEBUG").is_ok() {
                eprintln!(
                    "pool {:?} {} sdf {:.3} skin {}",
                    n.part,
                    n.name,
                    inp.sdf(r.g(i)),
                    r.skin[i]
                );
            }
            pool.entry(n.part).or_default().push(i);
        }
    }
    let mut tris: Vec<Tri> = Vec::new();
    let mut seen: std::collections::HashSet<[usize; 3]> = Default::default();
    for m in &model.parts {
        if !hull_mesh(&m.name) {
            continue;
        }
        let Some(part) = inp.part_of(&m.name) else {
            continue;
        };
        let Some(ids) = pool.get(&part) else { continue };
        let gs: Vec<Vec3> = ids.iter().map(|&i| r.g(i)).collect();
        // The outer surface's vertices only (inner panels and hems are
        // under it).
        let near: Vec<Option<usize>> = m
            .pos
            .iter()
            .map(|&p| {
                (inp.sdf(p).abs() < 0.02).then(|| {
                    let k = (0..gs.len())
                        .min_by(|&a, &b| {
                            gs[a]
                                .distance_squared(p)
                                .total_cmp(&gs[b].distance_squared(p))
                        })
                        .unwrap();
                    ids[k]
                })
            })
            .collect();
        if std::env::var("RIG_HULL_DEBUG").is_ok() {
            let outer = near.iter().filter(|x| x.is_some()).count();
            let mut sd: Vec<f32> = m.pos.iter().map(|&p| inp.sdf(p)).collect();
            sd.sort_by(|a, b| a.total_cmp(b));
            let q = |f: f32| sd[((sd.len() - 1) as f32 * f) as usize];
            eprintln!(
                "skin_hull {:<20} {:?}: {} skin nodes, {} of {} vertices outer (|sdf| < 2 cm); sdf p10 {:.3} p50 {:.3} p90 {:.3}",
                m.name,
                part,
                ids.len(),
                outer,
                m.pos.len(),
                q(0.1),
                q(0.5),
                q(0.9)
            );
        }
        // Each node's cell normal on this mesh: the area-weighted normal of
        // the render surface whose vertices went to it (a third of each
        // triangle to each corner's node). A node triangle stands for its
        // three cells, so their sum is the surface it must face (a single
        // render triangle at the cells' junction can lie on a fold: the
        // bumper face's triangles were judged by the lip's).
        let mut cell: HashMap<usize, Vec3> = HashMap::new();
        for t in &m.tris {
            let [a, b, c] = t.map(|k| near[k as usize]);
            let (Some(a), Some(b), Some(c)) = (a, b, c) else {
                continue;
            };
            let [pa, pb, pc] = t.map(|k| m.pos[k as usize]);
            let nr = (pb - pa).cross(pc - pa) / 3.0;
            for i in [a, b, c] {
                *cell.entry(i).or_default() += nr;
            }
        }
        // The candidates: every triple of nodes whose cells meet pairwise on
        // this mesh (a render edge between each two: the restricted Delaunay
        // triangles). A render triangle on three cells is one; a four-cell
        // junction (a grid of nodes) gives a quad's triangles only through
        // the pairwise rule, as a render triangle there may touch three of
        // the four cells or none. A quad whose four cells all meet keeps the
        // triangulation on its shorter diagonal.
        let mut adj: HashMap<usize, std::collections::BTreeSet<usize>> = HashMap::new();
        let mut cands: std::collections::BTreeSet<[usize; 3]> = Default::default();
        for t in &m.tris {
            let [a, b, c] = t.map(|k| near[k as usize]);
            let (Some(a), Some(b), Some(c)) = (a, b, c) else {
                continue;
            };
            for (x, y) in [(a, b), (b, c), (c, a)] {
                if x != y {
                    adj.entry(x).or_default().insert(y);
                    adj.entry(y).or_default().insert(x);
                }
            }
            if a != b && b != c && a != c {
                let mut key = [a, b, c];
                key.sort();
                cands.insert(key);
            }
        }
        let pairwise = std::env::var("RIG_HULL_JUNCTIONS_ONLY").is_err();
        if pairwise {
            for (&a, na) in &adj {
                for &b in na.range(a + 1..) {
                    for &c in adj[&b].range(b + 1..) {
                        if na.contains(&c) {
                            cands.insert([a, b, c]);
                        }
                    }
                }
            }
            let mut drop: std::collections::HashSet<(usize, usize)> = Default::default();
            for &[a, b, c] in &cands {
                for &d in adj[&c].range(c + 1..) {
                    if !(adj[&a].contains(&d) && adj[&b].contains(&d)) {
                        continue;
                    }
                    let q = [a, b, c, d];
                    let dd = |i: usize, j: usize| r.g(q[i]).distance(r.g(q[j]));
                    // The diagonals: the disjoint pairing of largest sum.
                    let pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))];
                    let ((i0, j0), (i1, j1)) = *pairings
                        .iter()
                        .max_by(|x, y| {
                            (dd(x.0 .0, x.0 .1) + dd(x.1 .0, x.1 .1))
                                .total_cmp(&(dd(y.0 .0, y.0 .1) + dd(y.1 .0, y.1 .1)))
                        })
                        .unwrap();
                    let long = if dd(i0, j0) > dd(i1, j1) {
                        (q[i0], q[j0])
                    } else {
                        (q[i1], q[j1])
                    };
                    drop.insert((long.0.min(long.1), long.0.max(long.1)));
                }
            }
            cands.retain(|t| !drop.iter().any(|&(p, q)| t.contains(&p) && t.contains(&q)));
        }
        for key in cands {
            let [a, b, c] = key;
            if !seen.insert(key) {
                continue;
            }
            let nr = cell[&a] + cell[&b] + cell[&c];
            let (qa, qb, qc) = (r.g(a), r.g(b), r.g(c));
            let nn = (qb - qa).cross(qc - qa);
            if nn.length() < 1e-6 {
                continue;
            }
            // No slivers (height under a tenth of the longest edge) and no
            // triangle leaning far off the skin it stands for (|cos| < 0.7):
            // both come of nodes the nearest-node map strings along a fold,
            // and such a triangle lies across its neighbours.
            let l = (qa - qb)
                .length()
                .max((qb - qc).length())
                .max((qc - qa).length());
            if nn.length() / l < 0.1 * l || nn.normalize().dot(nr.normalize_or_zero()).abs() < 0.7 {
                if std::env::var("RIG_HULL_DEBUG").is_ok() {
                    eprintln!(
                        "  dropped {} {:?}: sliver {} lean {:.2}",
                        m.name,
                        [a, b, c].map(|i| r.car.nodes[i].name.clone()),
                        nn.length() / l < 0.1 * l,
                        nn.normalize().dot(nr.normalize_or_zero()).abs()
                    );
                }
                continue;
            }
            if std::env::var("RIG_HULL_DEBUG").is_ok() {
                eprintln!(
                    "  kept {} {:?}",
                    m.name,
                    [a, b, c].map(|i| r.car.nodes[i].name.clone())
                );
            }
            // (glTF to JBeam is a rotation: the winding carries over.)
            tris.push(Tri {
                n: if nn.dot(nr) < 0.0 {
                    [a, c, b]
                } else {
                    [a, b, c]
                },
                part,
                group: None,
                sub: None,
                inner: false,
            });
        }
    }
    if std::env::var("RIG_HULL_NOBRIDGE").is_err() {
        let n = bridge_seams(r, inp, &pool, &mut tris);
        if std::env::var("RIG_HULL_DEBUG").is_ok() {
            eprintln!("skin_hull: {n} triangles bridge seams");
        }
    }
    // The wheels: a cone over each rim ring (six rim nodes at 0.25 m) to its
    // outer hub node. Raycast wheels have no collision geometry of their
    // own, so a head-on partner's wheel and arch went through this wheel;
    // now its nodes meet these triangles. (The inner cone too,
    // `RIG_WHEEL_BOTH`, closes the wheel but moved the parked car's rear
    // rails past the rest-drift rule: 7.4 mm, parked 1.12 mm.)
    if std::env::var("RIG_NO_WHEEL_TRIS").is_err() {
        let id = |n: &str| r.car.nodes.iter().position(|x| x.name == n);
        for w in ["F", "R"] {
            for x in ["L", "R"] {
                // (A defined wheel part: its two axle nodes and its three
                // frame nodes.)
                let c2 = format!("{}{}", w.to_lowercase(), x.to_lowercase());
                let (hi, ho, ring): (usize, usize, Vec<usize>) = match (
                    id(&format!("hub{w}i{x}")),
                    id(&format!("hub{w}o{x}")),
                    id(&format!("whe{c2}_ai")),
                    id(&format!("whe{c2}_ao")),
                ) {
                    (Some(hi), Some(ho), _, _) => (
                        hi,
                        ho,
                        (0..6)
                            .filter_map(|k| id(&format!("rim{w}{k}{x}")))
                            .collect(),
                    ),
                    (_, _, Some(hi), Some(ho)) => (
                        hi,
                        ho,
                        (1..=3)
                            .filter_map(|k| id(&format!("whe{c2}_f{k}")))
                            .collect(),
                    ),
                    _ => continue,
                };
                let nr = ring.len();
                if nr != 6 && nr != 3 {
                    continue;
                }
                let c = 0.5 * (r.g(hi) + r.g(ho));
                for k in 0..nr {
                    let (a, b) = (ring[k], ring[(k + 1) % nr]);
                    let hubs: &[usize] = if std::env::var("RIG_WHEEL_BOTH").is_ok() {
                        &[hi, ho]
                    } else {
                        &[ho]
                    };
                    for &h in hubs {
                        let (pa, pb, ph) = (r.g(a), r.g(b), r.g(h));
                        let n = (pb - pa).cross(ph - pa);
                        let out = (pa + pb + ph) / 3.0 - c;
                        tris.push(Tri {
                            n: if n.dot(out) < 0.0 {
                                [a, h, b]
                            } else {
                                [a, b, h]
                            },
                            part: r.car.nodes[hi].part,
                            group: None,
                sub: None,
                inner: false,
                        });
                    }
                }
            }
        }
    }
    r.car.tris = tris;
}

/// Seams between the parts' collision patches. Each part's patch ends half
/// a node spacing inside its outline, so the skin had a strip open along
/// every seam, 10–20 cm wide where the parts meet at the nose (cover to
/// grille to hood to lamps to fender): a head-on partner's nodes went in
/// there. The bridges are the restricted Delaunay triangles of all the
/// parts' skin nodes together (the grille's surface included) that join two
/// or more parts: each node's cell is the outer render surface nearest it
/// on any part, cells meet along render edges and, across a seam between
/// two meshes, where their surfaces come within 3 cm. Kept: bridges off the
/// patches (no bridge edge is a patch's inner edge), at most 0.5 m a side,
/// no sliver, facing their cells (|cos| ≥ 0.7), within 0.9 m of one of the
/// moving part's main mounts (bolts, hinges). A bridge belongs to its moving
/// part (bumper, then fender, then hood). It has no break group: one JBeam
/// triangle takes one group, and with the nearest main mount's
/// (`RIG_BRIDGE_GROUPS=main`; `nearest` takes any fastening) a 50 km/h
/// head-on broke the left front bumper bolt at 40 ms and its 19 bridges
/// died with it while the cover stayed on, reopening the corner. A part that
/// comes off whole would drag its bridges (an engine question: a triangle
/// that dies when its part separates, `handoff/known-bugs.md` S2). Doors and
/// the boot lid get none (they open; their shut lines are their patches'
/// edges).
fn bridge_seams(
    r: &Rig,
    inp: &dyn RigInput,
    pool: &HashMap<Part, Vec<usize>>,
    tris: &mut Vec<Tri>,
) -> usize {
    let model = inp.model();
    let mut nodes: Vec<usize> = pool.values().flatten().copied().collect();
    nodes.sort();
    let gs: Vec<Vec3> = nodes.iter().map(|&i| r.g(i)).collect();
    // Outer vertices of every hull mesh and the grille, each to its nearest
    // skin node of any part.
    let mut verts: Vec<(Vec3, usize, usize)> = Vec::new(); // (position, mesh, node)
    let mut adj: HashMap<usize, std::collections::BTreeSet<usize>> = HashMap::new();
    let mut cell: HashMap<usize, Vec3> = HashMap::new();
    let link = |adj: &mut HashMap<usize, std::collections::BTreeSet<usize>>, x: usize, y: usize| {
        if x != y {
            adj.entry(x).or_default().insert(y);
            adj.entry(y).or_default().insert(x);
        }
    };
    for (mi, m) in model.parts.iter().enumerate() {
        if !(hull_mesh(&m.name) || m.name == "grille") || inp.part_of(&m.name).is_none() {
            continue;
        }
        let near: Vec<Option<usize>> = m
            .pos
            .iter()
            .map(|&p| {
                (inp.sdf(p).abs() < 0.02).then(|| {
                    let k = (0..gs.len())
                        .min_by(|&a, &b| {
                            gs[a]
                                .distance_squared(p)
                                .total_cmp(&gs[b].distance_squared(p))
                        })
                        .unwrap();
                    nodes[k]
                })
            })
            .collect();
        for (k, p) in m.pos.iter().enumerate() {
            if let Some(n) = near[k] {
                verts.push((*p, mi, n));
            }
        }
        for t in &m.tris {
            let [a, b, c] = t.map(|k| near[k as usize]);
            let (Some(a), Some(b), Some(c)) = (a, b, c) else {
                continue;
            };
            link(&mut adj, a, b);
            link(&mut adj, b, c);
            link(&mut adj, c, a);
            let [pa, pb, pc] = t.map(|k| m.pos[k as usize]);
            let nr = (pb - pa).cross(pc - pa) / 3.0;
            for i in [a, b, c] {
                *cell.entry(i).or_default() += nr;
            }
        }
    }
    // Across seams: cells on two meshes whose surfaces come within 3 cm.
    let h = std::env::var("RIG_HULL_SEAM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.03f32);
    let key = |p: Vec3| {
        (
            (p.x / h).floor() as i32,
            (p.y / h).floor() as i32,
            (p.z / h).floor() as i32,
        )
    };
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (k, v) in verts.iter().enumerate() {
        grid.entry(key(v.0)).or_default().push(k);
    }
    for (p, mi, n) in &verts {
        let (x, y, z) = key(*p);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(l) = grid.get(&(x + dx, y + dy, z + dz)) {
                        for &j in l {
                            let (q, mj, nj) = verts[j];
                            if mj != *mi && nj != *n && q.distance(*p) < h {
                                link(&mut adj, *n, nj);
                            }
                        }
                    }
                }
            }
        }
    }
    let part = |i: usize| r.car.nodes[i].part;
    // Pairwise-adjacent triples joining two or more parts; a quad whose four
    // cells all meet keeps its shorter diagonal.
    let mut cands: std::collections::BTreeSet<[usize; 3]> = Default::default();
    for (&a, na) in &adj {
        for &b in na.range(a + 1..) {
            for &c in adj[&b].range(b + 1..) {
                if na.contains(&c) && !(part(a) == part(b) && part(b) == part(c)) {
                    cands.insert([a, b, c]);
                }
            }
        }
    }
    let mut drop: std::collections::HashSet<(usize, usize)> = Default::default();
    for &[a, b, c] in &cands {
        for &d in adj[&c].range(c + 1..) {
            if !(adj[&a].contains(&d) && adj[&b].contains(&d)) {
                continue;
            }
            let q = [a, b, c, d];
            let dd = |i: usize, j: usize| r.g(q[i]).distance(r.g(q[j]));
            let pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))];
            let ((i0, j0), (i1, j1)) = *pairings
                .iter()
                .max_by(|x, y| {
                    (dd(x.0 .0, x.0 .1) + dd(x.1 .0, x.1 .1))
                        .total_cmp(&(dd(y.0 .0, y.0 .1) + dd(y.1 .0, y.1 .1)))
                })
                .unwrap();
            let long = if dd(i0, j0) > dd(i1, j1) {
                (q[i0], q[j0])
            } else {
                (q[i1], q[j1])
            };
            drop.insert((long.0.min(long.1), long.0.max(long.1)));
        }
    }
    cands.retain(|t| !drop.iter().any(|&(p, q)| t.contains(&p) && t.contains(&q)));
    // The fastenings: each break group's beams with an end on a part.
    let mut fast: Vec<(Part, Vec3, String)> = Vec::new();
    for b in &r.car.beams {
        if let Some(g) = &b.break_group {
            for e in [b.a, b.b] {
                fast.push((part(e), r.g(e), g.clone()));
            }
        }
    }
    let mut uses: HashMap<(usize, usize), usize> = HashMap::new();
    for t in tris.iter() {
        for e in 0..3 {
            let (a, b) = (t.n[e], t.n[(e + 1) % 3]);
            *uses.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let per = |t: &[usize; 3]| {
        let [a, b, c] = t.map(|i| r.g(i));
        a.distance(b) + b.distance(c) + c.distance(a)
    };
    let mut order: Vec<[usize; 3]> = cands.into_iter().collect();
    order.sort_by(|x, y| per(x).total_cmp(&per(y)).then(x.cmp(y)));
    let debug = std::env::var("RIG_HULL_DEBUG").is_ok();
    // Adds bridge t when it passes the rules; true when added.
    let try_add =
        |t: [usize; 3], tris: &mut Vec<Tri>, uses: &mut HashMap<(usize, usize), usize>| -> bool {
            let parts: Vec<Part> = t.iter().map(|&i| part(i)).collect();
            if parts.iter().any(|p| {
                matches!(
                    p,
                    Part::DoorFL
                        | Part::DoorFR
                        | Part::DoorRL
                        | Part::DoorRR
                        | Part::Trunk
                        | Part::Engine
                )
            }) || (parts[0] == parts[1] && parts[1] == parts[2])
                || (std::env::var("RIG_BRIDGE_NO_BODY").is_ok() && parts.contains(&Part::Body))
            {
                return false;
            }
            let Some(owner) = [
                Part::BumperF,
                Part::BumperR,
                Part::FenderL,
                Part::FenderR,
                Part::Hood,
            ]
            .into_iter()
            .find(|p| parts.contains(p)) else {
                return false;
            };
            let [qa, qb, qc] = t.map(|i| r.g(i));
            let l = (qa - qb)
                .length()
                .max((qb - qc).length())
                .max((qc - qa).length());
            let nn = (qb - qa).cross(qc - qa);
            let nr = cell.get(&t[0]).copied().unwrap_or_default()
                + cell.get(&t[1]).copied().unwrap_or_default()
                + cell.get(&t[2]).copied().unwrap_or_default();
            if l > 0.5
                || nn.length() / l < 0.1 * l
                || nn.normalize().dot(nr.normalize_or_zero()).abs() < 0.7
            {
                return false;
            }
            let edges: Vec<(usize, usize)> = (0..3)
                .map(|e| (t[e].min(t[(e + 1) % 3]), t[e].max(t[(e + 1) % 3])))
                .collect();
            if edges.iter().any(|e| uses.get(e).copied().unwrap_or(0) >= 2) {
                return false;
            }
            let cen = (qa + qb + qc) / 3.0;
            // The part's main mounts (its bolts and hinges, not the clips and
            // row screws that let go while the part stays on in a crash).
            let main = |g: &str| {
                let g = g.rsplit('_').next().unwrap_or(g);
                g.starts_with("bolt") || g.starts_with("hinge") || g == "L" || g == "R"
            };
            let mode = std::env::var("RIG_BRIDGE_GROUPS").unwrap_or_default();
            let Some(group) = fast
                .iter()
                .filter(|f| {
                    f.0 == owner && (mode == "nearest" || main(&f.2)) && f.1.distance(cen) < 0.9
                })
                .min_by(|x, y| {
                    x.1.distance(cen)
                        .total_cmp(&y.1.distance(cen))
                        .then(x.2.cmp(&y.2))
                })
                .map(|f| f.2.clone())
            else {
                return false;
            };
            for e in edges {
                *uses.entry(e).or_default() += 1;
            }
            if debug {
                eprintln!(
                    "  bridge {owner:?} {:?} ({group})",
                    t.map(|i| r.car.nodes[i].name.clone())
                );
            }
            let [a, b, c] = t;
            tris.push(Tri {
                n: if nn.dot(nr) < 0.0 {
                    [a, c, b]
                } else {
                    [a, b, c]
                },
                part: owner,
                group: (mode == "main" || mode == "nearest").then_some(group),
                sub: None,
                inner: false,
            });
            true
        };
    let mut added = 0;
    for t in order {
        if try_add(t, tris, &mut uses) {
            added += 1;
        }
    }
    // The zipper: a seam the cells do not reach across (surfaces more than
    // 3 cm apart, as the grille's top under the hood's front edge) closes
    // from the open edges. Each open edge of the skin takes the nearest open
    // node of another part across it (on the far side of the edge from its
    // triangle, within 0.3 m of its middle) whose triangle passes the
    // bridges' rules; repeated while any closes.
    for _ in 0..6 {
        let mut open: Vec<(usize, usize, usize)> = Vec::new();
        for t in tris.iter() {
            for e in 0..3 {
                let (a, b, o) = (t.n[e], t.n[(e + 1) % 3], t.n[(e + 2) % 3]);
                if uses.get(&(a.min(b), a.max(b))).copied().unwrap_or(0) == 1 {
                    open.push((a, b, o));
                }
            }
        }
        let mut rim: Vec<usize> = open.iter().flat_map(|&(a, b, _)| [a, b]).collect();
        rim.sort();
        rim.dedup();
        open.sort_by(|x, y| {
            r.g(x.0)
                .distance(r.g(x.1))
                .total_cmp(&r.g(y.0).distance(r.g(y.1)))
                .then(x.cmp(y))
        });
        let mut any = false;
        for (a, b, o) in open {
            if uses.get(&(a.min(b), a.max(b))).copied().unwrap_or(0) != 1 {
                continue;
            }
            let (pa, pb, po) = (r.g(a), r.g(b), r.g(o));
            let tn = (pb - pa).cross(po - pa);
            let across = tn.cross(pb - pa).normalize_or_zero();
            let so = across.dot(po - pa);
            let mid = 0.5 * (pa + pb);
            let mut cs: Vec<(f32, usize)> = rim
                .iter()
                .copied()
                .filter(|&c| c != a && c != b && c != o)
                .filter(|&c| {
                    let s = across.dot(r.g(c) - pa);
                    r.g(c).distance(mid) < 0.3 && s * so < 0.0 && s.abs() > 0.01
                })
                .map(|c| (pa.distance(r.g(c)) + pb.distance(r.g(c)), c))
                .collect();
            cs.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
            for (_, c) in cs {
                if try_add([a, b, c], tris, &mut uses) {
                    added += 1;
                    any = true;
                    break;
                }
            }
        }
        if !any {
            break;
        }
    }
    added
}

/// The cabin's volume (8 cm in from the skin: beams along the bonded glass
/// and the roof are the shell's, not across the cabin).
fn in_cabin(inp: &dyn RigInput, p: Vec3) -> bool {
    p.x.abs() < 0.62 && p.y > 0.46 && p.y < 1.28 && p.z > -1.72 && p.z < 0.78 && inp.sdf(p) < -0.08
}

fn segment_blocked(a: Vec3, b: Vec3, blocked: &dyn Fn(Vec3) -> bool) -> bool {
    (1..12).any(|k| blocked(a.lerp(b, k as f32 / 12.0)))
}

/// Every node of `ids` to each other within `reach`, unless the segment
/// crosses a blocked region, with the role of their shared member (or the
/// web role).
fn graph(
    r: &mut Rig,
    ids: &[usize],
    reach: f32,
    part: Part,
    web: Role,
    blocked: &dyn Fn(Vec3) -> bool,
) {
    graph_pairs(r, ids, reach, part, web, blocked, &|_, _, _| true);
}

/// [`graph`] with a rule on the pair (`pair_ok(rig, a, b)`).
fn graph_pairs(
    r: &mut Rig,
    ids: &[usize],
    reach: f32,
    part: Part,
    web: Role,
    blocked: &dyn Fn(Vec3) -> bool,
    pair_ok: &dyn Fn(&Rig, usize, usize) -> bool,
) {
    for (ka, &a) in ids.iter().enumerate() {
        for &b in &ids[ka + 1..] {
            let (pa, pb) = (r.g(a), r.g(b));
            let d = pa.distance(pb);
            if d > reach || d < 0.03 || !pair_ok(r, a, b) {
                continue;
            }
            if segment_blocked(pa, pb, blocked) {
                continue;
            }
            let role = match (r.member[a], r.member[b]) {
                (Some(x), Some(y)) if x == y => x,
                _ => web,
            };
            r.car.beam(a, b, role, part);
        }
    }
}

/// The rigged sedan from the render model.
pub fn rig(model: &Model) -> Rig {
    rig_with(&Gen3Input::new(model))
}

/// The rig of any input ([`RigInput`]): a render model, its body surface and
/// its physics block.
pub fn rig_with(inp: &dyn RigInput) -> Rig {
    let model = inp.model();
    let dims = inp.dims();
    let ph = inp.physics();
    // The surface queries and the running gear, from the input (the local
    // closures shadow the module's gen3 surface functions in this body).
    let (axle_f, axle_r, tyre_r, track_half, arch_r) = (
        dims.axle_f,
        dims.axle_r,
        dims.tyre_r,
        dims.track_half,
        dims.arch_r,
    );
    let side = |z: f32, y: f32| inp.side(z, y);
    let top = |z: f32, x: f32| inp.top(z, x);
    let corner = |z: f32| inp.corner(z);
    let inward = |p: Vec3, d: f32| inp.inward(p, d);
    let end_pt = |front: bool, x: f32, y: f32| inp.end_pt(front, x, y);
    let belt = |z: f32| inp.belt(z);
    let to_jbeam = |g: Vec3| Vec3::new(g.x, axle_f - g.z, g.y);
    let to_gltf = |j: Vec3| Vec3::new(j.x, j.z, axle_f - j.y);
    // The physics block (the module's constants are its defaults).
    let (kerb, accord_biw, accord_door_f, accord_door_r, accord_fender) =
        (ph.kerb, ph.biw, ph.door_f, ph.door_r, ph.fender);
    let (accord_hood, accord_decklid, accord_engine_kg, accord_transaxle_kg) =
        (ph.hood, ph.decklid, ph.engine, ph.transaxle);
    let (bumper_f_kg, bumper_r_kg) = (ph.bumper_f, ph.bumper_r);
    let (accord_pt_ahead, accord_pt_behind, accord_dash_behind, powertrain_clearance) =
        (ph.pt_ahead, ph.pt_behind, ph.dash_behind, ph.pt_clearance);
    let (crush_yield, tower_yield, footwell_yield, greenhouse_yield, apillar_yield) = (
        ph.crush_yield,
        ph.tower_yield,
        ph.footwell_yield,
        ph.greenhouse_yield,
        ph.apillar_yield,
    );
    let (cross_member_yield, floor_pan_yield, bpillar_upper_yield, bpillar_lower_yield) = (
        ph.cross_member_yield,
        ph.floor_pan_yield,
        ph.bpillar_upper_yield,
        ph.bpillar_lower_yield,
    );
    let intrusion_yield = ph.intrusion_yield;
    let (fender_bolt, fender_row_bolt, bumper_bolt, bumper_side_bolt, engine_mount) = (
        ph.fender_bolt,
        ph.fender_row_bolt,
        ph.bumper_bolt,
        ph.bumper_side_bolt,
        ph.engine_mount,
    );
    let (cooling_stack, wheel_stack, reach_body, reach_panel) = (
        ph.cooling_stack,
        ph.wheel_stack,
        ph.reach_body,
        ph.reach_panel,
    );
    let mut r = Rig {
        car: Car::default(),
        member: Vec::new(),
        skin: Vec::new(),
        axle_f: axle_f,
    };
    let body = Part::Body;
    // The census lattice (BeamNG's node counts and masses per part: doors
    // about 13 nodes, hood 23, lid 15, no light edge nodes on the closures);
    // `RIG_OLD_CENSUS=1` builds the earlier, denser closures.
    let census = std::env::var("RIG_OLD_CENSUS").is_err();

    // Part extents from the meshes.
    let door_f = Skin::of(model, "door_FL");
    let door_r = Skin::of(model, "door_RL");
    let hood_m = Skin::of(model, "hood");
    let trunk_m = Skin::of(model, &inp.rear_closure());
    let fender_m = Skin::of(model, "fender_FL");
    let bf_m = Skin::of(model, "bumper_front");
    let br_m = Skin::of(model, "bumper_rear");
    let side_out = |_: Vec3, n: Vec3| n.x > 0.45;
    let up_out = |_: Vec3, n: Vec3| n.y > 0.45;
    let df = door_f.outer(side_out);
    let dr = door_r.outer(side_out);
    let (df_lo, df_hi) = bounds(&df);
    let (dr_lo, dr_hi) = bounds(&dr);
    let hood_top = hood_m.outer(up_out);
    let (hd_lo, hd_hi) = bounds(&hood_top);
    let trunk_top = trunk_m.outer(|_, n| n.y > 0.3 || n.z < -0.6);
    let (tk_lo, tk_hi) = bounds(&trunk_top);
    let fen = fender_m.outer(|_, n| n.x > 0.2 || n.y > 0.45);
    let (fe_lo, fe_hi) = bounds(&fen);
    let (bf_lo, bf_hi) = bounds(&bf_m.pos);
    let (br_lo, br_hi) = bounds(&br_m.pos);
    let y_rocker = df_lo.y; // door bottom = top of the rocker
    let z_a = df_hi.z; // door front shut line (hinge pillar)
    let z_b = 0.5 * (df_lo.z + dr_hi.z); // B-pillar
    let z_c = dr_lo.z; // rear door's rear edge
    let z_hood_r = hd_lo.z;
    let z_trunk_f = tk_hi.z;
    let z_tail = br_lo.z;

    // The body, powertrain, running gear and interior from the design's own
    // definitions when it has them (`RIG_OLD_BODY=1`: the members below, the
    // rigger's own, for comparison).
    let defs = inp
        .definitions()
        .filter(|_| std::env::var("RIG_OLD_BODY").map_or(true, |v| v != "1"));
    let mut named_ids: HashMap<String, usize> = HashMap::new();
    let defined = defs.as_ref().map(|d| super::body2::build(&mut r, inp, d));
    let apertures = [Aperture::of(&door_f.pos), Aperture::of(&door_r.pos)];
    let mut eng: Vec<usize> = Vec::new();
    let old_pt = std::env::var("GEN3_OLD_POWERTRAIN").is_ok();
    let mut pt_z = [0.0f32; 3];
    // (The rigger's own body and engine: not re-indented, so the definitions'
    // path reads as a diff beside it.)
    if defined.is_none() {
    // ------------------------------------------------------------------
    // Body: members per side.
    // ------------------------------------------------------------------
    // Sill stations: the toe board, the front seat's front cross member,
    // the B-pillar, the seat's rear cross member and the rear seat's kick-up
    // ([`SEAT_CM_F`] ahead of and [`SEAT_CM_R`] behind the B-pillar).
    let sill_z = if std::env::var("GEN3_OLD_SILL").is_ok() {
        span(z_a - 0.05, z_c + 0.10, 5)
    } else {
        vec![
            z_a - 0.05,
            z_b + SEAT_CM_F,
            z_b,
            z_b - SEAT_CM_R,
            z_c + 0.10,
        ]
    };
    let roof_z = [-0.08f32, z_b, 0.5 * (z_b + -1.05), -1.05];
    let mut sill_o = [Vec::new(), Vec::new()];
    let mut sill_t = [Vec::new(), Vec::new()];
    let mut sill_b = [Vec::new(), Vec::new()];
    let mut floor_r = [Vec::new(), Vec::new()];
    let mut rail_lo = [Vec::new(), Vec::new()];
    let mut rail_hi = [Vec::new(), Vec::new()];
    let mut apron = [Vec::new(), Vec::new()];
    let mut roof_rail = [Vec::new(), Vec::new()];
    let mut roof_mid = [Vec::new(), Vec::new()];
    let mut quarter: [Vec<Vec<Option<usize>>>; 2] = [Vec::new(), Vec::new()];
    let mut rrail = [Vec::new(), Vec::new()];
    let rail_z = [0.95f32, 1.32, 1.70, 2.05];
    let rrail_z = [-1.60f32, -1.95, -2.25];
    let qz = [z_c - 0.03, -1.85, -2.18, z_tail + 0.12];
    for (si, s) in [1.0f32, -1.0].into_iter().enumerate() {
        let x = lr(s);
        let mut put =
            |r: &mut Rig, name: String, g: Vec3, member: Option<Role>, skin: bool| -> usize {
                let i = r.node(&name, mirror(g, s), body, member, skin);
                named_ids.insert(name, i);
                i
            };
        // Sills: an outer rocker face node, the box's top (behind the door's
        // bottom edge) and its bottom.
        for (k, &z) in sill_z.iter().enumerate() {
            let o = side(z, 0.5 * (y_rocker + 0.17));
            sill_o[si].push(put(&mut r, format!("slo{k}{x}"), o, Some(Role::Sill), true));
            let t = inward(side(z, y_rocker + 0.02), 0.07);
            sill_t[si].push(put(
                &mut r,
                format!("slt{k}{x}"),
                t,
                Some(Role::Sill),
                false,
            ));
            let b = Vec3::new(o.x - 0.11, 0.17, z);
            sill_b[si].push(put(
                &mut r,
                format!("slb{k}{x}"),
                b,
                Some(Role::Sill),
                false,
            ));
            let f = Vec3::new(0.42, 0.19, z);
            floor_r[si].push(put(
                &mut r,
                format!("flr{k}{x}"),
                f,
                Some(Role::Floor),
                false,
            ));
        }
        // Hinge pillar (behind the fender, ahead of the front door) and the
        // A-pillar up to the windshield header.
        let hp_z = z_a + 0.04;
        let hp0 = put(
            &mut r,
            format!("hp0{x}"),
            inward(side(hp_z, 0.42), 0.04),
            Some(Role::PillarA),
            false,
        );
        let hp1 = put(
            &mut r,
            format!("hp1{x}"),
            inward(side(hp_z, 0.78), 0.04),
            Some(Role::PillarA),
            false,
        );
        let hp2 = put(
            &mut r,
            format!("hp2{x}"),
            inward(side(hp_z, belt(hp_z) - 0.02), 0.03),
            Some(Role::PillarA),
            true,
        );
        let hpi0 = put(
            &mut r,
            format!("hpi0{x}"),
            inward(side(hp_z, 0.50), 0.16),
            Some(Role::PillarA),
            false,
        );
        let hpi1 = put(
            &mut r,
            format!("hpi1{x}"),
            inward(side(hp_z, 0.88), 0.14),
            Some(Role::PillarA),
            false,
        );
        // The hinge pillar's front wall (the A-pillar lower's depth): behind
        // the front wheel arch, under the fender, so the pillar is a box
        // about 20 cm deep fore and aft, not a line of nodes at one station
        // (it rotated back about the sill under the upper load path).
        if std::env::var("GEN3_NO_HPF").is_err() {
            let zf = (axle_f - arch_r - 0.08).max(hp_z + 0.12);
            for (k, &y) in [0.45f32, 0.80].iter().enumerate() {
                put(
                    &mut r,
                    format!("hpf{k}{x}"),
                    inward(side(zf, y), 0.08),
                    Some(Role::PillarA),
                    false,
                );
            }
        }
        let _ = (hp0, hp1, hp2, hpi0, hpi1);
        // The A-pillar's middle (its top is the roof rail's front station,
        // `rr0`/`rri0`, at the windshield header: one node, not two
        // coincident ones the graph would leave unjoined).
        for (k, &z) in [0.33f32].iter().enumerate() {
            let c = corner(z);
            put(
                &mut r,
                format!("ap{}{x}", k + 1),
                inward(c, 0.015),
                Some(Role::PillarA),
                true,
            );
            put(
                &mut r,
                format!("api{}{x}", k + 1),
                inward(c, 0.07),
                Some(Role::PillarA),
                false,
            );
        }
        // Roof rails and the roof between them.
        for (k, &z) in roof_z.iter().enumerate() {
            let c = corner(z);
            roof_rail[si].push(put(
                &mut r,
                format!("rr{k}{x}"),
                inward(c, 0.015),
                Some(Role::RoofRail),
                true,
            ));
            put(
                &mut r,
                format!("rri{k}{x}"),
                inward(c, 0.08),
                Some(Role::RoofRail),
                false,
            );
            let mx = 0.48 * c.x;
            roof_mid[si].push(put(
                &mut r,
                format!("rm{k}{x}"),
                inward(top(z, mx), 0.01),
                Some(Role::RoofBow),
                true,
            ));
            // The bow under the roof skin (its depth).
            put(
                &mut r,
                format!("rmi{k}{x}"),
                inward(top(z, mx), 0.07),
                Some(Role::RoofBow),
                false,
            );
        }
        // B-pillar: outer face (between the doors) and inner.
        // A box section: front and rear outer flanges (the doors' shut
        // lines either side) and the inner panel, at four heights up to the
        // roof rail.
        let bpy = [
            0.45f32,
            0.72,
            belt(z_b) - 0.01,
            0.5 * (belt(z_b) + corner(z_b).y),
        ];
        for (k, y) in bpy.iter().enumerate() {
            for (tag, dz) in [("F", 0.045f32), ("R", -0.045)] {
                let o = side(z_b + dz, *y);
                put(
                    &mut r,
                    format!("bp{tag}{k}{x}"),
                    inward(o, 0.03),
                    Some(Role::PillarB),
                    k == 2,
                );
            }
            let o = side(z_b, *y);
            // 15 cm in, but outside the cabin's volume (a node inside it has
            // every beam blocked: a greenhouse with more tumblehome than
            // gen3's puts the upper one there).
            let mut din = 0.15f32;
            while din > 0.06 && in_cabin(inp, inward(o, din)) {
                din -= 0.005;
            }
            put(
                &mut r,
                format!("bpi{k}{x}"),
                inward(o, din),
                Some(Role::PillarB),
                false,
            );
        }
        // C-pillar: from the rear roof rail down to the quarter's belt.
        {
            let z = 0.5 * (-1.05 + z_c);
            let c = corner(z);
            let p = inward(inp.hit(z, inp.theta_corner(z) + 0.12), 0.015);
            let _ = c;
            put(&mut r, format!("cp1{x}"), p, Some(Role::PillarC), true);
            put(
                &mut r,
                format!("cpi1{x}"),
                inward(p, 0.08),
                Some(Role::PillarC),
                false,
            );
            // Its lower half, down to the quarter's belt behind the door.
            let z2 = 0.5 * (z + z_c) - 0.10;
            let q = inward(side(z2, belt(z2) + 0.10), 0.015);
            put(&mut r, format!("cp2{x}"), q, Some(Role::PillarC), true);
            put(
                &mut r,
                format!("cpi2{x}"),
                inward(q, 0.09),
                Some(Role::PillarC),
                false,
            );
        }
        // Quarter panel: rows (the arch or bumper line, the shoulder, the
        // belt/deck edge) at stations behind the rear door.
        for &z in &qz {
            let bottom = inp
                .arch_y(z)
                .map(|a| a + 0.03)
                .unwrap_or(br_hi.y.min(0.80) + 0.03)
                .max(y_rocker + 0.02);
            let top_y = belt(z) - 0.015;
            let mut col = vec![Some(put(
                &mut r,
                format!("qb{}{x}", (z * -100.0) as i32),
                side(z, bottom),
                Some(Role::Sheet),
                true,
            ))];
            let sh = inp.shoulder_y(z);
            if sh > bottom + 0.06 && sh < top_y - 0.06 {
                col.push(Some(put(
                    &mut r,
                    format!("qs{}{x}", (z * -100.0) as i32),
                    side(z, sh),
                    Some(Role::Sheet),
                    true,
                )));
            } else {
                col.push(None);
            }
            col.push(Some(put(
                &mut r,
                format!("qt{}{x}", (z * -100.0) as i32),
                inward(side(z, top_y), 0.01),
                Some(Role::Sheet),
                true,
            )));
            // The quarter's inner (wheelhouse and gutter) behind it.
            put(
                &mut r,
                format!("qi{}{x}", (z * -100.0) as i32),
                inward(side(z, 0.5 * (bottom + top_y)), 0.14),
                Some(Role::Web),
                false,
            );
            quarter[si].push(col);
        }
        // Rear wheelhouse and the rear suspension's top.
        put(
            &mut r,
            format!("wh{x}"),
            Vec3::new(0.60, 0.72, axle_r),
            Some(Role::RearRail),
            false,
        );
        // Front rails (low, high) and the aprons (upper rails).
        for (k, &z) in rail_z.iter().enumerate() {
            rail_lo[si].push(put(
                &mut r,
                format!("rl{k}{x}"),
                Vec3::new(0.44, 0.34, z),
                Some(Role::Rail),
                false,
            ));
            rail_hi[si].push(put(
                &mut r,
                format!("rh{k}{x}"),
                Vec3::new(0.44, 0.54, z),
                Some(Role::Rail),
                false,
            ));
            let fx = inner_x_at(&fen, z, 0.06).unwrap_or(0.70);
            let ay = top(z, fx).y - 0.10;
            apron[si].push(put(
                &mut r,
                format!("ar{k}{x}"),
                Vec3::new(fx - 0.03, ay, z),
                Some(Role::Apron),
                false,
            ));
        }
        // Strut tower top.
        put(
            &mut r,
            format!("tw{x}"),
            Vec3::new(0.56, 0.80, axle_f),
            Some(Role::Apron),
            false,
        );
        // Rear rails.
        for (k, &z) in rrail_z.iter().enumerate() {
            rrail[si].push(put(
                &mut r,
                format!("rrl{k}{x}"),
                Vec3::new(0.50, 0.30, z),
                Some(Role::RearRail),
                false,
            ));
            put(
                &mut r,
                format!("rrh{k}{x}"),
                Vec3::new(0.50, 0.48, z),
                Some(Role::RearRail),
                false,
            );
        }
        // Firewall and cowl.
        for (k, &y) in [0.34f32, 0.64].iter().enumerate() {
            put(
                &mut r,
                format!("fw{k}{x}"),
                Vec3::new(0.50, y, 0.88),
                Some(Role::Firewall),
                false,
            );
        }
        put(
            &mut r,
            format!("cw{x}"),
            inward(top(z_hood_r - 0.04, 0.50), 0.02),
            Some(Role::Firewall),
            true,
        );
        // Radiator support (upper tie bar end, lower).
        put(
            &mut r,
            format!("rsu{x}"),
            Vec3::new(0.50, 0.74, 2.08),
            Some(Role::Cross),
            false,
        );
        // Rear shelf and the tail panel (under the trunk lid's rear edge).
        put(
            &mut r,
            format!("sh{x}"),
            Vec3::new(0.50, belt(z_trunk_f) - 0.04, z_trunk_f - 0.02),
            Some(Role::Cross),
            false,
        );
        put(
            &mut r,
            format!("tp{x}"),
            Vec3::new(0.55, 0.78, z_tail + 0.10),
            Some(Role::Cross),
            false,
        );
        // The tail panel's face (between the bumper cover and the lid), on
        // the skin: the skin hull's triangles need nodes there (the tail
        // panel's member is 10 cm in).
        if inp.hull_triangles() {
            let y = 0.5 * (br_hi.y.min(0.85) + tk_lo.y);
            put(
                &mut r,
                format!("tfa{x}"),
                inward(end_pt(false, 0.45, y), 0.01),
                Some(Role::Cross),
                true,
            );
        }
        put(
            &mut r,
            format!("tf{x}"),
            Vec3::new(0.40, 0.30, -2.10),
            Some(Role::Floor),
            false,
        );
    }
    // Centre line.
    let mut cput = |r: &mut Rig, name: &str, g: Vec3, member: Option<Role>, skin: bool| -> usize {
        let i = r.node(name, g, body, member, skin);
        named_ids.insert(name.to_string(), i);
        i
    };
    let tunnel: Vec<usize> = sill_z
        .iter()
        .enumerate()
        .map(|(k, &z)| {
            cput(
                &mut r,
                &format!("tn{k}"),
                Vec3::new(0.0, 0.36, z),
                Some(Role::Tunnel),
                false,
            )
        })
        .collect();
    let roof_c: Vec<usize> = roof_z
        .iter()
        .enumerate()
        .map(|(k, &z)| {
            cput(
                &mut r,
                &format!("rc{k}"),
                inward(top(z, 0.0), 0.01),
                Some(Role::RoofBow),
                true,
            )
        })
        .collect();
    for (k, &z) in roof_z.iter().enumerate() {
        cput(
            &mut r,
            &format!("rci{k}"),
            inward(top(z, 0.0), 0.07),
            Some(Role::RoofBow),
            false,
        );
    }
    for (k, &y) in [0.34f32, 0.64, 0.90].iter().enumerate() {
        cput(
            &mut r,
            &format!("fwC{k}"),
            Vec3::new(0.0, y, 0.88),
            Some(Role::Firewall),
            false,
        );
    }
    // The bonded windshield and backlight (part of the shell's shear path):
    // a row across each glass's middle.
    let ws = model
        .parts
        .iter()
        .find(|p| p.name == "glass_windshield")
        .map(|p| bounds(&p.pos));
    let bl = model
        .parts
        .iter()
        .find(|p| p.name == "glass_rear")
        .map(|p| bounds(&p.pos));
    for (tag, b) in [("ws", ws), ("bl", bl)] {
        let Some((lo, hi)) = b else { continue };
        let z = 0.5 * (lo.z + hi.z);
        for (k, xx) in [-0.42f32, 0.0, 0.42].iter().enumerate() {
            let p = top(z, xx.abs());
            let p = mirror(p, if *xx < 0.0 { -1.0 } else { 1.0 });
            cput(
                &mut r,
                &format!("{tag}{k}"),
                inward(p, 0.005),
                Some(Role::Sheet),
                false,
            );
        }
    }
    cput(
        &mut r,
        "cwC",
        inward(top(z_hood_r - 0.04, 0.0), 0.02),
        Some(Role::Firewall),
        true,
    );
    cput(
        &mut r,
        "rsuC",
        Vec3::new(0.0, 0.76, 2.10),
        Some(Role::Cross),
        false,
    );
    cput(
        &mut r,
        "rslC",
        Vec3::new(0.0, 0.30, 2.08),
        Some(Role::Cross),
        false,
    );
    cput(
        &mut r,
        "shC",
        Vec3::new(0.0, belt(z_trunk_f) - 0.05, z_trunk_f - 0.02),
        Some(Role::Cross),
        false,
    );
    cput(
        &mut r,
        "tpC",
        Vec3::new(0.0, 0.78, z_tail + 0.12),
        Some(Role::Cross),
        false,
    );
    if inp.hull_triangles() {
        let y = 0.5 * (br_hi.y.min(0.85) + tk_lo.y);
        cput(
            &mut r,
            "tfaC",
            inward(end_pt(false, 0.0, y), 0.01),
            Some(Role::Cross),
            true,
        );
    }
    cput(
        &mut r,
        "tfC",
        Vec3::new(0.0, 0.30, -2.10),
        Some(Role::Floor),
        false,
    );
    cput(
        &mut r,
        "rsC",
        Vec3::new(0.0, 0.80, -1.76),
        Some(Role::Cross),
        false,
    );
    cput(
        &mut r,
        "rsLoC",
        Vec3::new(0.0, 0.30, -1.76),
        Some(Role::Floor),
        false,
    );

    // Body skins (collision triangles and sheet): sills, roof, quarters.
    for si in 0..2 {
        let rows: Vec<Vec<Option<usize>>> = vec![sill_o[si].iter().map(|&i| Some(i)).collect()];
        let _ = rows;
        // Rocker strip: outer face to the box top.
        let a: Vec<Option<usize>> = sill_o[si].iter().map(|&i| Some(i)).collect();
        let b: Vec<Option<usize>> = sill_t[si].iter().map(|&i| Some(i)).collect();
        r.car
            .panel(&[a, b], body, &[Role::Sill, Role::Sill], Role::Sill, false);
        // Quarter grid (columns are stations; rows bottom, shoulder, top).
        let q = &quarter[si];
        let rows: Vec<Vec<Option<usize>>> = (0..3)
            .map(|row| q.iter().map(|c| c[row]).collect())
            .collect();
        r.car
            .panel(&rows, body, &[Role::Sheet; 3], Role::Sheet, false);
    }
    // Roof: rows rail L, mid L, centre, mid R, rail R (each along z).
    {
        let rows: Vec<Vec<Option<usize>>> = vec![
            roof_rail[0].iter().map(|&i| Some(i)).collect(),
            roof_mid[0].iter().map(|&i| Some(i)).collect(),
            roof_c.iter().map(|&i| Some(i)).collect(),
            roof_mid[1].iter().map(|&i| Some(i)).collect(),
            roof_rail[1].iter().map(|&i| Some(i)).collect(),
        ];
        r.car.panel(
            &rows,
            body,
            &[
                Role::RoofRail,
                Role::Sheet,
                Role::Sheet,
                Role::Sheet,
                Role::RoofRail,
            ],
            Role::RoofBow,
            false,
        );
    }
    let _ = (
        &rail_lo, &rail_hi, &apron, &floor_r, &sill_b, &rrail, &tunnel,
    );
    // The body's member graph.
    let body_ids: Vec<usize> = (0..r.car.nodes.len()).collect();
    // The front structure's upper rails and towers meet the cabin at the
    // A-pillar's base (the hinge pillar and cowl), not the greenhouse: no
    // beam from them to a node above the belt (they were struts from the
    // aprons up to the A-pillar's middle and the windshield, under the cowl,
    // and pushed the greenhouse up and back in the wall).
    let split = std::env::var("GEN3_APRON_TO_GREENHOUSE").is_err();
    let front_member = |r: &Rig, i: usize| {
        matches!(
            r.member[i],
            Some(Role::Apron) | Some(Role::Rail) | Some(Role::Cross)
        ) && r.g(i).z > z_a + 0.08
    };
    let greenhouse = |r: &Rig, i: usize| {
        let g = r.g(i);
        g.y > belt(g.z.min(z_a)) + 0.05
    };
    graph_pairs(
        &mut r,
        &body_ids,
        reach_body,
        body,
        Role::Web,
        &|p| body_blocked(inp, p, &apertures, (tk_lo.z + 0.06, tk_hi.z - 0.06)),
        &|r, a, b| {
            !split
                || !((front_member(r, a) && greenhouse(r, b))
                    || (front_member(r, b) && greenhouse(r, a)))
        },
    );
    // The A-pillars' chords: the hinge pillar's top straight to the header
    // (the roof rail's front station), outer and inner walls. The pillar is
    // a closed section about 8 cm deep; as a chain of two bays through its
    // middle node it had no depth against the cabin's inertia pushing the
    // header forward and buckled out (the greenhouse racked 28 cm in the
    // 56 km/h wall, the ETK's 7 cm). The chords run inside the section (the
    // pillar's curve leaves its straight line by about 4 cm).
    if std::env::var("GEN3_NO_APILLAR_CHORD").is_err() {
        for x in ["L", "R"] {
            let id = |n: &str| named_ids[&format!("{n}{x}")];
            for (a, b) in [
                ("hp2", "rr0"),
                ("hpi1", "rri0"),
                ("hp2", "rri0"),
                ("hpi1", "rr0"),
            ] {
                r.car.beam(id(a), id(b), Role::PillarA, body);
            }
        }
    }

    // Braces (census): a body member node whose beams leave one direction
    // nearly free (their directions' second moment has a weak axis: the rear
    // rail's upper front node hung on three near-coplanar beams and swung at
    // 3.8 Hz, 30 mm at rest) takes a web to the member node most along that
    // axis, until its beams span all three.
    if census {
        let n0 = r.car.nodes.len();
        for i in 0..n0 {
            if r.car.nodes[i].part != body || r.member[i].is_none() {
                continue;
            }
            let g = r.g(i);
            for _ in 0..3 {
                let tied: Vec<usize> = r
                    .car
                    .beams
                    .iter()
                    .filter(|b| b.a == i || b.b == i)
                    .map(|b| if b.a == i { b.b } else { b.a })
                    .collect();
                let m = tied.iter().fold(glam::Mat3::ZERO, |m, &j| {
                    let d = (r.g(j) - g).normalize_or_zero();
                    m + glam::Mat3::from_cols(d * d.x, d * d.y, d * d.z)
                });
                let (lam, axis) = weakest_axis(m);
                if tied.len() >= 8 || (tied.len() >= 4 && lam > 0.45) {
                    break;
                }
                let best = r
                    .nearest(body, g, 24, |j| j != i && r.member[j].is_some() && !tied.contains(&j))
                    .into_iter()
                    .filter(|&j| {
                        let d = r.g(j).distance(g);
                        d > 0.05 && d < 0.7
                    })
                    .max_by(|&a, &b| {
                        let f = |j: usize| (r.g(j) - g).normalize_or_zero().dot(axis).abs();
                        f(a).total_cmp(&f(b))
                    });
                let Some(j) = best else { break };
                if std::env::var("RIG_LOG").is_ok() {
                    eprintln!(
                        "brace {} ({} beams, weak {:.2}) to {}",
                        r.car.nodes[i].name,
                        tied.len(),
                        lam,
                        r.car.nodes[j].name
                    );
                }
                r.car.beam(i, j, Role::Web, body);
            }
        }
    }

    // ------------------------------------------------------------------
    // Engine: a 12-node block on mounts.
    // ------------------------------------------------------------------
    // The powertrain's faces where the 2011 Accord's are: 0.39 m ahead of
    // and 0.12 m behind the front axle (DOT HS 811 666 fig. 87). Round 3
    // had its rear face at the axle.
    pt_z = if old_pt {
        [1.40, 1.61, 1.82]
    } else {
        let (r0, f0) = (axle_f - accord_pt_behind, axle_f + accord_pt_ahead);
        [r0, 0.5 * (r0 + f0), f0]
    };
    eng = {
        let mut ids = Vec::new();
        for (kz, &z) in pt_z.iter().enumerate() {
            for (ky, &y) in [0.38f32, 0.78].iter().enumerate() {
                for (kx, &xx) in [0.26f32, -0.26].iter().enumerate() {
                    ids.push(r.node(
                        &format!("e{kz}{ky}{kx}"),
                        Vec3::new(xx, y, z),
                        Part::Engine,
                        Some(Role::Engine),
                        false,
                    ));
                }
            }
        }
        graph(&mut r, &ids, 0.9, Part::Engine, Role::Engine, &|_| false);
        ids
    };
    // Mounts: each block node to its two nearest rail or apron nodes. They
    // are four break groups (the block's corners: left and right, front and
    // rear, as a transverse engine's side mounts and torque rods), so once
    // the block is stopped against the barrier and the cabin drives on, the
    // mounts tear and the block stacks back onto the dash panel instead of
    // dragging the rails beside it.
    for &e in &eng {
        let g = r.g(e);
        let mounts = r.nearest(body, g, 2, |i| {
            matches!(r.member[i], Some(Role::Rail) | Some(Role::Apron))
        });
        let group = format!(
            "engine_mount_{}{}",
            if g.x > 0.0 { "L" } else { "R" },
            if g.z > pt_z[1] { "F" } else { "R" }
        );
        for b in mounts {
            if old_pt {
                r.car.beam(e, b, Role::Mount, body);
            } else {
                let st = std::env::var("GEN3_MOUNT")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(engine_mount);
                r.car.hardware(e, b, Role::Mount, body, &group, st);
            }
        }
    }
    // The dash panel behind the block (BOUNDED stops from its rear face to
    // the firewall's nodes): the Accord's dash line is 0.27-0.30 m behind
    // the front axle, so the block's rear face meets it after closing
    // 0.15-0.18 m, less what stays crushed between them (manifold shield
    // and hoses, powertrain_clearance).
    let dash_gap = 0.5 * (accord_dash_behind[0] + accord_dash_behind[1])
        - accord_pt_behind
        - powertrain_clearance;
    let rear_eng: Vec<usize> = eng
        .iter()
        .copied()
        .filter(|&e| r.g(e).z < pt_z[0] + 0.05)
        .collect();
    for &e in &rear_eng {
        let g = r.g(e);
        for t in r.nearest(body, g, 2, |i| r.member[i] == Some(Role::Firewall)) {
            let j = r.car.beam(e, t, Role::Stop, body);
            r.car.beams[j].role = Role::Stop;
            if !old_pt {
                r.car.stop_close.insert(j, dash_gap);
            }
        }
    }
    } // (the rigger's own body and engine)

    // ------------------------------------------------------------------
    // Closures.
    // ------------------------------------------------------------------
    // Rigidifiers under the hood, the lid and the fenders on the defined
    // body (`RIG_NO_RIGIDIFIER=1`: none).
    let rigid_panels =
        defined.is_some() && std::env::var("RIG_NO_RIGIDIFIER").map_or(true, |v| v != "1");
    let mut door_nodes: HashMap<Part, Vec<usize>> = HashMap::new();
    for (front, lo, hi, pts) in [(true, df_lo, df_hi, &df), (false, dr_lo, dr_hi, &dr)] {
        for s in [1.0f32, -1.0] {
            let part = Part::door(front, s > 0.0);
            let dn = part.name().trim_start_matches("gen2_").to_string();
            let cols = [hi.z - 0.03, 0.5 * (lo.z + hi.z), lo.z + 0.03];
            let mut grid: Vec<Vec<Option<usize>>> = vec![Vec::new(), Vec::new(), Vec::new()];
            let mut inner: Vec<usize> = Vec::new();
            for (k, &z) in cols.iter().enumerate() {
                let bottom = low_at(pts, z, 0.04).unwrap_or(lo.y) + 0.03;
                let bt = belt(z) - 0.02;
                let sh = inp.shoulder_y(z);
                let rows = [bottom, sh, bt];
                for (row, &y) in rows.iter().enumerate() {
                    if row == 1 && (y < bottom + 0.06 || y > bt - 0.05) {
                        grid[row].push(None);
                        continue;
                    }
                    let p = mirror(side(z, y), s);
                    let i = r.node(&format!("{dn}_o{row}{k}"), p, part, Some(Role::Sheet), true);
                    grid[row].push(Some(i));
                }
                // The inner panel: bottom and belt, 8 cm in, and its middle
                // between them. The census door is a closed box of 17 nodes
                // (a BeamNG door has 13): the skin's three rows and the inner
                // panel's bottom and belt rows, [`DOOR_DEPTH`] in, so it
                // resists twist and bending by its section, not by the skin
                // alone.
                let rows_in = [bottom + 0.02, bt - 0.03];
                let depth = if census { DOOR_DEPTH } else { 0.08 };
                for (row, &y) in rows_in.iter().enumerate() {
                    let p = mirror(inward(side(z, y), depth), s);
                    inner.push(r.node(
                        &format!("{dn}_i{row}{k}"),
                        p,
                        part,
                        Some(Role::DoorFrame),
                        false,
                    ));
                }
                if census {
                    continue;
                }
                let ym = 0.5 * (bottom + bt);
                let p = mirror(inward(side(z, ym), 0.08), s);
                inner.push(r.node(
                    &format!("{dn}_im{k}"),
                    p,
                    part,
                    Some(Role::DoorFrame),
                    false,
                ));
            }
            // The side intrusion beam's stations between the columns, so the
            // tube can form a plastic hinge where a narrow object strikes it
            // (a pole between two columns otherwise met a 0.5 m straight
            // span). On the inner panel's middle row, between its columns.
            let split_beam = std::env::var("GEN3_OLD_INTRUSION").is_err() && !census;
            // The tube's outer chord, 2 cm under the skin at every station
            // (the inner chord is the middle row, 8 cm in): a 6 cm deep
            // truss, so the beam resists a load across the door by bending.
            // A straight chain of nodes carries none until it has sagged
            // into a catenary.
            let mut outer_chord: Vec<usize> = Vec::new();
            if split_beam {
                let mut zs: Vec<f32> = cols.to_vec();
                for k in 0..cols.len() - 1 {
                    zs.push(0.5 * (cols[k] + cols[k + 1]));
                }
                for (k, &z) in zs.iter().enumerate() {
                    let bottom = low_at(pts, z, 0.04).unwrap_or(lo.y) + 0.03;
                    let ym = 0.5 * (bottom + belt(z) - 0.02);
                    let out = std::env::var("GEN3_INTRUSION_OUT")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(INTRUSION_OUT);
                    let p = mirror(inward(side(z, ym), out), s);
                    let i = r.node(
                        &format!("{dn}_io{k}"),
                        p,
                        part,
                        Some(Role::DoorFrame),
                        false,
                    );
                    inner.push(i);
                    outer_chord.push(i);
                }
            }
            if split_beam {
                for k in 0..cols.len() - 1 {
                    let z = 0.5 * (cols[k] + cols[k + 1]);
                    let bottom = low_at(pts, z, 0.04).unwrap_or(lo.y) + 0.03;
                    let ym = 0.5 * (bottom + belt(z) - 0.02);
                    let p = mirror(inward(side(z, ym), 0.08), s);
                    inner.push(r.node(
                        &format!("{dn}_imh{k}"),
                        p,
                        part,
                        Some(Role::DoorFrame),
                        false,
                    ));
                }
            }
            // The window frame: its top at the front and rear.
            let (fz0, fz1) = if front {
                (0.20f32, lo.z + 0.04)
            } else {
                (hi.z - 0.04, -0.95)
            };
            let mut frame = Vec::new();
            for (k, &z) in [fz0, fz1].iter().enumerate() {
                let c = corner(z);
                let p = mirror(side(z, c.y - 0.05), s);
                frame.push(r.node(&format!("{dn}_f{k}"), p, part, Some(Role::DoorFrame), true));
            }
            // The frame's section: an inner node under each top corner.
            let frame_in: Vec<usize> = frame
                .iter()
                .filter(|_| !census)
                .enumerate()
                .map(|(k, &f)| {
                    let g = r.g(f);
                    let gi = mirror(inward(mirror(g, s), 0.06), s);
                    r.node(
                        &format!("{dn}_fi{k}"),
                        gi,
                        part,
                        Some(Role::DoorFrame),
                        false,
                    )
                })
                .collect();
            r.car
                .panel(&grid, part, &vec![Role::Sheet; grid.len()], Role::Sheet, true);
            let mut below: Vec<usize> = grid.iter().flatten().flatten().copied().collect();
            below.extend(&inner);
            // Below the belt: everything to everything in reach.
            graph(&mut r, &below, reach_panel, part, Role::DoorFrame, &|_| {
                false
            });
            // The frame: front and rear legs from the belt, and the top.
            let belt_row: Vec<usize> = grid[grid.len() - 1].iter().flatten().copied().collect();
            let ib: Vec<usize> = inner
                .iter()
                .copied()
                .filter(|&i| r.car.nodes[i].name.contains("_i1"))
                .collect();
            let (bf, br) = (belt_row[0], *belt_row.last().unwrap());
            let (ibf, ibr) = (ib[0], *ib.last().unwrap());
            let (f0, f1) = (frame[0], frame[1]);
            let mut legs = vec![
                (bf, f0),
                (ibf, f0),
                (br, f1),
                (ibr, f1),
                (f0, f1),
                (belt_row[1], f0),
                (belt_row[1], f1),
            ];
            if let [fi0, fi1] = frame_in[..] {
                legs.extend([
                    (f0, fi0),
                    (f1, fi1),
                    (fi0, fi1),
                    (f0, fi1),
                    (f1, fi0),
                    (ibf, fi0),
                    (ibr, fi1),
                    (bf, fi0),
                    (br, fi1),
                ]);
            }
            for (a, b) in legs {
                r.car.beam(a, b, Role::DoorFrame, part);
            }
            // The side intrusion beam: a tube along the door at its middle,
            // hinge pillar to latch (the inner middle row), tied to the skin's
            // crease row it backs.
            let mut mids: Vec<usize> = inner
                .iter()
                .copied()
                .filter(|&i| r.car.nodes[i].name.contains("_im"))
                .collect();
            // (The outer chord's nodes are `_io`: not in the inner chord.)
            // Hinge side to latch side (along the door).
            mids.sort_by(|&a, &b| r.g(b).z.total_cmp(&r.g(a).z));
            for w in mids.windows(2) {
                let j = r.car.beam(w[0], w[1], Role::Intrusion, part);
                r.car.beams[j].role = Role::Intrusion;
            }
            if census {
                // The belt reinforcement and the lower intrusion beam: the
                // inner panel's two rows and the diagonals between them.
                let row = |r: &Rig, tag: &str| -> Vec<usize> {
                    let mut v: Vec<usize> = inner
                        .iter()
                        .copied()
                        .filter(|&i| r.car.nodes[i].name.contains(tag))
                        .collect();
                    v.sort_by(|&a, &b| r.g(b).z.total_cmp(&r.g(a).z));
                    v
                };
                let (lo_row, hi_row) = (row(&r, "_i0"), row(&r, "_i1"));
                let mut pairs: Vec<(usize, usize)> = Vec::new();
                // The skin's crease row is the truss's outer chord (a load at
                // mid-door meets it and goes to both inner rows): without it
                // the static door test's initial resistance fell to 7.7 kN.
                let mut mid: Vec<usize> = grid[1].iter().flatten().copied().collect();
                mid.sort_by(|&a, &b| r.g(b).z.total_cmp(&r.g(a).z));
                for k in 0..mid.len() {
                    if k + 1 < mid.len() {
                        pairs.push((mid[k], mid[k + 1]));
                    }
                    for row in [&lo_row, &hi_row] {
                        for kk in k.saturating_sub(1)..(k + 2).min(row.len()) {
                            pairs.push((mid[k], row[kk]));
                        }
                    }
                }
                for k in 0..lo_row.len().min(hi_row.len()).saturating_sub(1) {
                    pairs.push((lo_row[k], lo_row[k + 1]));
                    pairs.push((hi_row[k], hi_row[k + 1]));
                    pairs.push((lo_row[k], hi_row[k + 1]));
                    pairs.push((hi_row[k], lo_row[k + 1]));
                }
                for (a, b) in pairs {
                    let j = r.car.beam(a, b, Role::Intrusion, part);
                    r.car.beams[j].role = Role::Intrusion;
                }
            }
            if split_beam {
                outer_chord.sort_by(|&a, &b| r.g(b).z.total_cmp(&r.g(a).z));
                let mut pairs: Vec<(usize, usize)> = Vec::new();
                for w in outer_chord.windows(2) {
                    pairs.push((w[0], w[1]));
                }
                for k in 0..mids.len().min(outer_chord.len()) {
                    pairs.push((mids[k], outer_chord[k]));
                    if k + 1 < mids.len() {
                        pairs.push((mids[k], outer_chord[k + 1]));
                        pairs.push((outer_chord[k], mids[k + 1]));
                    }
                }
                for (a, b) in pairs {
                    let j = r.car.beam(a, b, Role::Intrusion, part);
                    r.car.beams[j].role = Role::Intrusion;
                }
            } else if let (Some(&a), Some(&b)) = (mids.first(), mids.last()) {
                let j = r.car.beam(a, b, Role::Intrusion, part);
                r.car.beams[j].role = Role::Intrusion;
            }
            // The frame's legs run down inside the door to its inner panel
            // (the frame's base is the door's depth over its height, not the
            // belt's 8 cm).
            let deep: Vec<usize> = inner
                .iter()
                .copied()
                .filter(|&i| {
                    let n = &r.car.nodes[i].name;
                    n.contains("_im") || n.contains("_i0")
                })
                .collect();
            for &f in frame.iter().chain(frame_in.iter()) {
                let g = r.g(f);
                let mut v: Vec<(f32, usize)> =
                    deep.iter().map(|&i| (r.g(i).distance(g), i)).collect();
                v.sort_by(|a, b| a.0.total_cmp(&b.0));
                for &(_, i) in v.iter().take(3) {
                    r.car.beam(f, i, Role::DoorFrame, part);
                }
            }
            r.car.tri(bf, f0, belt_row[1], part);
            r.car.tri(belt_row[1], f0, f1, part);
            r.car.tri(belt_row[1], f1, br, part);
            let mut all: Vec<usize> = below
                .iter()
                .copied()
                .chain(frame.iter().copied())
                .chain(frame_in.iter().copied())
                .collect();
            if rigid_panels {
                // The door's rigidifier: inboard of its middle (the armrest's
                // place on the trim panel), beamed to every node of the door,
                // the window frame's corners included. The frame then stands
                // on a base as deep as the rigidifier, not on the belt's
                // 10 cm (it swung in and out at 15 Hz), and the lower corners
                // no longer twist about the hinge-latch line.
                let z = cols[1];
                let bottom = low_at(pts, z, 0.04).unwrap_or(lo.y) + 0.03;
                let ym = 0.5 * (bottom + belt(z) - 0.02);
                let at = mirror(inward(side(z, ym), RIGIDIFIER_DEPTH), s);
                let own = all.clone();
                all.push(rigidifier(&mut r, &format!("{dn}_r0"), at, part, &own));
            }
            door_nodes.insert(part, all);
        }
    }

    // Hood: a 5 × 4 skin grid and an inner frame 5 cm under it.
    let hood_ids = {
        let part = Part::Hood;
        let rows_z = span(
            hd_lo.z + 0.03,
            hd_hi.z - 0.04,
            std::env::var("GEN3_HOOD_ROWS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4),
        );
        let mut grid = Vec::new();
        let mut inner = Vec::new();
        for (kr, &z) in rows_z.iter().enumerate() {
            let w = half_w_at(&hood_top, z, 0.04).unwrap_or(0.75) - 0.025;
            let mut row = Vec::new();
            for (kc, f) in [-1.0f32, -0.5, 0.0, 0.5, 1.0].iter().enumerate() {
                let xx = f * w;
                let p = mirror(top(z, xx.abs()), if xx < 0.0 { -1.0 } else { 1.0 });
                let i = r.node(&format!("hd_o{kr}{kc}"), p, part, Some(Role::Sheet), true);
                row.push(Some(i));
                // The inner frame: under the edges and the centre line (the
                // old lattice), or the centre spine alone (census: 24
                // nodes; an inner row under each side edge instead measured
                // lower, 6.7 Hz against 8.8: the bump stops hold the edges).
                if if census { kc == 2 } else { kc % 2 == 0 } {
                    inner.push(r.node(
                        &format!("hd_i{kr}{kc}"),
                        inward(p, 0.06),
                        part,
                        Some(Role::DoorFrame),
                        false,
                    ));
                }
            }
            grid.push(row);
        }
        r.car.panel(
            &grid,
            part,
            &vec![Role::Sheet; grid.len()],
            Role::Sheet,
            true,
        );
        let mut all: Vec<usize> = grid.iter().flatten().flatten().copied().collect();
        all.extend(&inner);
        graph(&mut r, &all, 0.85, part, Role::Sheet, &|_| false);
        if rigid_panels {
            // Two rigidifiers on the centre line, a third of the hood's
            // length from each end.
            let (z0, z1) = (rows_z[0], *rows_z.last().unwrap());
            for (k, t) in [1.0f32 / 3.0, 2.0 / 3.0].iter().enumerate() {
                let at = inward(top(z0 + t * (z1 - z0), 0.0), RIGIDIFIER_DEPTH);
                let own = all.clone();
                all.push(rigidifier(&mut r, &format!("hd_r{k}"), at, part, &own));
            }
        }
        all
    };

    // Trunk lid: deck rows and the rear face, 5 columns.
    let trunk_ids = {
        let part = Part::Trunk;
        let mut grid = Vec::new();
        // Census: two deck rows and the rear face (18 nodes with the inner row).
        let deck = span(tk_hi.z - 0.03, tk_lo.z + 0.07, if census { 2 } else { 3 });
        for (kr, &z) in deck.iter().enumerate() {
            let w = half_w_at(&trunk_top, z, 0.04).unwrap_or(0.72) - 0.025;
            let mut row = Vec::new();
            for (kc, f) in [-1.0f32, -0.5, 0.0, 0.5, 1.0].iter().enumerate() {
                let xx = f * w;
                let p = mirror(top(z, xx.abs()), if xx < 0.0 { -1.0 } else { 1.0 });
                row.push(Some(r.node(
                    &format!("tk_o{kr}{kc}"),
                    p,
                    part,
                    Some(Role::Sheet),
                    true,
                )));
            }
            grid.push(row);
        }
        // The rear face's bottom edge.
        let yb = tk_lo.y + 0.03;
        let w = 0.95 * half_w_at(&trunk_top, tk_lo.z + 0.02, 0.05).unwrap_or(0.65);
        let mut row = Vec::new();
        for (kc, f) in [-1.0f32, -0.5, 0.0, 0.5, 1.0].iter().enumerate() {
            let xx = f * (w - 0.03);
            let p = mirror(
                end_pt(false, xx.abs(), yb),
                if xx < 0.0 { -1.0 } else { 1.0 },
            );
            row.push(Some(r.node(
                &format!("tk_o3{kc}"),
                p,
                part,
                Some(Role::Sheet),
                true,
            )));
        }
        grid.push(row);
        let mut inner = Vec::new();
        let inner_z: Vec<f32> = if census {
            vec![0.5 * (deck[0] + deck[1])]
        } else {
            vec![deck[0], deck[2]]
        };
        for (kr, &z) in inner_z.iter().enumerate() {
            for (k, &xx) in [-0.45f32, 0.0, 0.45].iter().enumerate() {
                let p = top(z, xx.abs()) * Vec3::new(if xx < 0.0 { -1.0 } else { 1.0 }, 1.0, 1.0);
                inner.push(r.node(
                    &format!("tk_i{kr}{k}"),
                    inward(p, 0.06),
                    part,
                    Some(Role::DoorFrame),
                    false,
                ));
            }
        }
        r.car
            .panel(&grid, part, &vec![Role::Sheet; grid.len()], Role::Sheet, true);
        let mut all: Vec<usize> = grid.iter().flatten().flatten().copied().collect();
        all.extend(&inner);
        graph(&mut r, &all, 0.85, part, Role::Sheet, &|_| false);
        if rigid_panels {
            let at = inward(top(0.5 * (deck[0] + deck[deck.len() - 1]), 0.0), RIGIDIFIER_DEPTH);
            let own = all.clone();
            all.push(rigidifier(&mut r, "tk_r0", at, part, &own));
        }
        all
    };

    // Fenders: columns along z; rows top (the hood line), shoulder, bottom
    // (the arch, the rocker or the bumper line).
    let mut fender_ids: HashMap<Part, Vec<usize>> = HashMap::new();
    for s in [1.0f32, -1.0] {
        let part = if s > 0.0 {
            Part::FenderL
        } else {
            Part::FenderR
        };
        let fx = lr(s);
        let cols = span(fe_lo.z + 0.03, fe_hi.z - 0.05, 5);
        let mut grid: Vec<Vec<Option<usize>>> = vec![Vec::new(), Vec::new(), Vec::new()];
        for (k, &z) in cols.iter().enumerate() {
            let xi = inner_x_at(&fen, z, 0.04).unwrap_or(0.6) + 0.025;
            let topp = top(z, xi);
            let mut bottom = low_at(&fen, z, 0.04).unwrap_or(0.4) + 0.025;
            if let Some(a) = inp.arch_y(z) {
                bottom = bottom.max(a + 0.025);
            }
            let sh = inp.shoulder_y(z);
            let pts = [
                Some(side(z, bottom)),
                (sh > bottom + 0.06 && sh < topp.y - 0.05).then(|| side(z, sh)),
                Some(topp),
            ];
            for (row, p) in pts.into_iter().enumerate() {
                grid[row].push(p.map(|p| {
                    r.node(
                        &format!("fe{fx}_{row}{k}"),
                        mirror(p, s),
                        part,
                        Some(Role::Sheet),
                        true,
                    )
                }));
            }
        }
        r.car
            .panel(&grid, part, &[Role::Sheet; 3], Role::Sheet, true);
        let mut all: Vec<usize> = grid.iter().flatten().flatten().copied().collect();
        graph(&mut r, &all, reach_panel, part, Role::Sheet, &|_| false);
        if rigid_panels {
            // Inboard of the shoulder at the fender's middle (over the
            // wheelhouse, where its inner flange runs to the apron).
            let z = cols[cols.len() / 2];
            let y = 0.5 * (inp.shoulder_y(z) + top(z, inner_x_at(&fen, z, 0.04).unwrap_or(0.6)).y);
            let at = mirror(inward(side(z, y), RIGIDIFIER_DEPTH), s);
            let own = all.clone();
            all.push(rigidifier(&mut r, &format!("fe{fx}_r0"), at, part, &own));
        }
        fender_ids.insert(part, all);
    }

    // Bumpers: the cover (rows low and high; columns over the face and round
    // the corner to the side) and a bumper beam behind it.
    let mut bumper_ids: HashMap<Part, (Vec<usize>, Vec<usize>)> = HashMap::new();
    for front in [true, false] {
        let part = if front { Part::BumperF } else { Part::BumperR };
        let tag = if front { "bf" } else { "bb" };
        let (lo, hi) = if front {
            (bf_lo, bf_hi)
        } else {
            (br_lo, br_hi)
        };
        // Census: a third row across the face at the bumper beam's height
        // ([`BUMPER_BEAM_Y`]), so the cover's contact with a car ahead is at
        // the beam and the rails, below the centre of mass (the two rows
        // alone met it at the cover's lower lip and at its top, level with
        // the centre of mass: both noses rose, known-bugs S2).
        let ys: Vec<(usize, f32, bool)> = if census {
            vec![(0, lo.y + 0.06, true), (2, BUMPER_BEAM_Y, false), (1, hi.y - 0.06, true)]
        } else {
            vec![(0, lo.y + 0.06, true), (1, hi.y - 0.06, true)]
        };
        let side_z = if front { lo.z + 0.05 } else { hi.z - 0.05 };
        let mut grid: Vec<Vec<Option<usize>>> = vec![Vec::new(); ys.len()];
        for (gi, &(row, y, wrap)) in ys.iter().enumerate() {
            let mut cols: Vec<Option<Vec3>> = Vec::new();
            cols.push(wrap.then(|| mirror(side(side_z, y), -1.0)));
            for f in [-0.80f32, -0.42, 0.0, 0.42, 0.80] {
                let p = end_pt(front, f.abs() * 0.92, y);
                cols.push(Some(mirror(p, if f < 0.0 { -1.0 } else { 1.0 })));
            }
            cols.push(wrap.then(|| side(side_z, y)));
            for (k, p) in cols.into_iter().enumerate() {
                grid[gi].push(p.map(|p| {
                    r.node(&format!("{tag}_o{row}{k}"), p, part, Some(Role::Sheet), true)
                }));
            }
        }
        r.car
            .panel(&grid, part, &vec![Role::Sheet; grid.len()], Role::Sheet, true);
        let cover: Vec<usize> = grid.iter().flatten().flatten().copied().collect();
        // The beam: 0.13 m behind the face, five stations, two heights.
        let face = if front { hi.z } else { lo.z };
        let bz =
            |x: f32| face - (0.13 + 0.10 * (x / 0.75).powi(2)) * if front { 1.0 } else { -1.0 };
        let mut beam = Vec::new();
        // (Defined: the impact beam is its own part, already built; the
        // cover's foam bears on it.)
        if let Some(b) = &defined {
            beam = b.impact[if front { 0 } else { 1 }].clone();
        }
        let own_beam = defined.is_none();
        for (k, &xx) in [-0.72f32, -0.40, 0.0, 0.40, 0.72]
            .iter()
            .enumerate()
            .filter(|_| own_beam)
        {
            for (kh, &y) in [0.40f32, 0.56].iter().enumerate() {
                beam.push(r.node(
                    &format!("{tag}_b{kh}{k}"),
                    Vec3::new(xx, y, bz(xx)),
                    part,
                    Some(Role::BumperBeam),
                    false,
                ));
            }
            // Its back flange (the section's depth).
            let back = bz(xx) - 0.07 * if front { 1.0 } else { -1.0 };
            beam.push(r.node(
                &format!("{tag}_bb{k}"),
                Vec3::new(xx, 0.48, back),
                part,
                Some(Role::BumperBeam),
                false,
            ));
        }
        if own_beam {
            graph(&mut r, &beam, 0.50, part, Role::BumperBeam, &|_| false);
        }
        let mut all = cover.clone();
        all.extend(&beam);
        // The beam's nodes nearest a point (its own part's when defined).
        let beam_near = |r: &Rig, g: Vec3, k: usize| -> Vec<usize> {
            let mut v: Vec<(f32, usize)> = beam.iter().map(|&i| (r.g(i).distance(g), i)).collect();
            v.sort_by(|a, b| a.0.total_cmp(&b.0));
            v.into_iter().take(k).map(|q| q.1).collect()
        };
        // Cover to beam: the foam (sheet role, soft).
        for &c in &cover {
            let g = r.g(c);
            let near = if own_beam {
                r.nearest(part, g, 3, |i| beam.contains(&i))
            } else {
                beam_near(&r, g, 3)
            };
            for b in near {
                r.car.beam(c, b, Role::Sheet, part);
            }
        }
        // Census: the foam crushed solid. Each face node of the cover meets
        // the beam (a BOUNDED stop to its nearest beam node) once the gap
        // between them has closed to [`FOAM_SOLID`], so a load on the cover
        // goes into the bumper beam, the crush cans and the rails, at bumper
        // height, not round them.
        if census {
            let face_z = if front { hi.z } else { lo.z };
            for &c in &cover {
                let g = r.g(c);
                if (g.z - face_z).abs() > 0.25 {
                    continue;
                }
                let near = if own_beam {
                    r.nearest(part, g, 1, |i| beam.contains(&i))
                } else {
                    beam_near(&r, g, 1)
                };
                if let Some(&b) = near.first() {
                    let l = r.g(b).distance(g);
                    if l > FOAM_SOLID + 0.02 {
                        let j = r.car.beam(c, b, Role::Stop, part);
                        r.car.beams[j].role = Role::Stop;
                        r.car.stop_close.insert(j, l - FOAM_SOLID);
                    }
                }
            }
        }
        graph(&mut r, &cover, 0.62, part, Role::Sheet, &|_| false);
        for &c in &cover {
            let g = r.g(c);
            if let Some(&b) = r.nearest(body, g, 1, |_| true).first() {
                if r.g(b).distance(g) < 0.30 {
                    r.car.beam(c, b, Role::Bearing, part);
                }
            }
        }
        // The grille's top edge (front, when the design has a grille): a row
        // of the cover's nodes on the grille's upper edge, so the collision
        // skin reaches from the cover's top row to the hood's front edge
        // (without it the band between them, the grille and the cover's
        // upper lip, 0.19 m tall across the car, had no collision triangle
        // within 0.6 m behind it, and a head-on partner's hood and lamps
        // went into the engine bay). Tied to the cover (sheet) and to the
        // radiator support's upper tie bar behind it, which carries a real
        // grille.
        if front && std::env::var("RIG_NO_GRILLE_ROW").is_err() {
            if let Some(gm) = model.parts.iter().find(|p| p.name == "grille") {
                let tie: Vec<usize> = (0..r.car.nodes.len())
                    .filter(|&i| {
                        r.car.nodes[i].part == body
                            && r.member[i] == Some(Role::Cross)
                            && r.g(i).z > hi.z - 0.5
                            && r.g(i).y > 0.6
                    })
                    .collect();
                let mut row = Vec::new();
                for (k, xx) in [-0.58f32, -0.386, 0.0, 0.386, 0.58].into_iter().enumerate() {
                    let near: Vec<&Vec3> =
                        gm.pos.iter().filter(|q| (q.x - xx).abs() < 0.03).collect();
                    let Some(top) = near.iter().map(|q| q.y).reduce(f32::max) else {
                        continue;
                    };
                    let y = top - 0.02;
                    let Some(z) = near
                        .iter()
                        .filter(|q| (q.y - y).abs() < 0.03)
                        .map(|q| q.z)
                        .reduce(f32::max)
                    else {
                        continue;
                    };
                    row.push(r.node(
                        &format!("{tag}_g{k}"),
                        Vec3::new(xx, y, z),
                        part,
                        Some(Role::Sheet),
                        true,
                    ));
                }
                for &gn in &row {
                    let g = r.g(gn);
                    let mut v: Vec<(f32, usize)> = cover
                        .iter()
                        .chain(&row)
                        .filter(|&&i| i != gn)
                        .map(|&i| (r.g(i).distance(g), i))
                        .collect();
                    v.sort_by(|a, b| a.0.total_cmp(&b.0));
                    for &(_, c) in v.iter().take(4) {
                        r.car.beam(gn, c, Role::Sheet, part);
                    }
                    let mut v: Vec<(f32, usize)> =
                        tie.iter().map(|&i| (r.g(i).distance(g), i)).collect();
                    v.sort_by(|a, b| a.0.total_cmp(&b.0));
                    for &(_, b) in v.iter().take(2) {
                        r.car.beam(gn, b, Role::Sheet, part);
                    }
                }
            }
        }
        bumper_ids.insert(part, (cover, beam));
        let _ = all;
    }

    // ------------------------------------------------------------------
    // Crush cans and rear rails onto the bumper beams.
    // ------------------------------------------------------------------
    let mut rims = Vec::new();
    if defined.is_none() {
    for (si, s) in [1.0f32, -1.0].into_iter().enumerate() {
        let x = lr(s);
        let (fb, rb) = (&bumper_ids[&Part::BumperF].1, &bumper_ids[&Part::BumperR].1);
        let tip_lo = named_ids[&format!("rl3{x}")];
        let tip_hi = named_ids[&format!("rh3{x}")];
        let near = |r: &Rig, set: &[usize], g: Vec3, k: usize| -> Vec<usize> {
            let mut v: Vec<(f32, usize)> = set.iter().map(|&i| (r.g(i).distance(g), i)).collect();
            v.sort_by(|a, b| a.0.total_cmp(&b.0));
            v.into_iter().take(k).map(|q| q.1).collect()
        };
        for t in [tip_lo, tip_hi] {
            for b in near(&r, fb, r.g(t), 3) {
                r.car.beam(t, b, Role::CrushCan, body);
            }
        }
        let rr_end = [
            named_ids[&format!("rrl2{x}")],
            named_ids[&format!("rrh2{x}")],
        ];
        for t in rr_end {
            for b in near(&r, rb, r.g(t), 3) {
                r.car.beam(t, b, Role::CrushCan, body);
            }
        }
        let _ = si;
    }
    // The front stack (round 5): the block's front face meets the bumper
    // beam through the cooling module once what lies between is crushed
    // solid (cooling_stack), as a frontal pulse's late peak is the engine
    // stopped against the barrier through the radiator support. BOUNDED
    // stops from the block's front nodes to their two nearest beam nodes.
    if !old_pt && std::env::var("GEN3_NO_STACK").is_err() {
        let stack = env_f("GEN3_COOLING_STACK", cooling_stack);
        let front_eng: Vec<usize> = eng
            .iter()
            .copied()
            .filter(|&e| r.g(e).z > pt_z[2] - 0.05)
            .collect();
        for &e in &front_eng {
            let g = r.g(e);
            for t in r.nearest(Part::BumperF, g, 2, |i| {
                r.member[i] == Some(Role::BumperBeam)
            }) {
                let l = r.g(t).distance(g);
                if l <= stack + 0.02 {
                    continue;
                }
                let j = r.car.beam(e, t, Role::Stop, body);
                r.car.beams[j].role = Role::Stop;
                r.car.stop_close.insert(j, l - stack);
            }
        }
    }

    // ------------------------------------------------------------------
    // Suspension: hubs on the body, and rim proxies.
    // ------------------------------------------------------------------
    for s in [1.0f32, -1.0] {
        let x = lr(s);
        for (w, za) in [("F", axle_f), ("R", axle_r)] {
            let wc = marker(model, &format!("wheel_center_{w}{x}"));
            let hi = r.node(
                &format!("hub{w}i{x}"),
                Vec3::new(s * (wc.x.abs() - 0.10), wc.y, wc.z),
                body,
                None,
                false,
            );
            let ho = r.node(
                &format!("hub{w}o{x}"),
                Vec3::new(s * (wc.x.abs() + 0.10), wc.y, wc.z),
                body,
                None,
                false,
            );
            r.car.beam(hi, ho, Role::Susp, body);
            let tops: Vec<usize> = if w == "F" {
                vec![
                    named_ids[&format!("tw{x}")],
                    named_ids[&format!("rl1{x}")],
                    named_ids[&format!("rh1{x}")],
                    named_ids[&format!("ar1{x}")],
                ]
            } else {
                vec![
                    named_ids[&format!("wh{x}")],
                    named_ids[&format!("rrl0{x}")],
                    named_ids[&format!("rrh0{x}")],
                    named_ids[&format!("slb4{x}")],
                ]
            };
            for &t in &tops {
                r.car.beam(hi, t, Role::Susp, body);
                r.car.beam(ho, t, Role::Susp, body);
            }
            let arm = tops[0];
            for k in 0..RIM_NODES {
                let a = (k as f32 + 0.5) * std::f32::consts::TAU / RIM_NODES as f32;
                let q = Vec3::new(
                    s * track_half,
                    tyre_r + RIM_R * a.sin(),
                    za + RIM_R * a.cos(),
                );
                let rn = r.node(&format!("rim{w}{k}{x}"), q, body, None, false);
                r.car.nodes[rn].no_group = true;
                r.car.nodes[rn].fixed_mass = RIM_NODE_KG;
                rims.push((rn, hi, ho, arm));
            }
        }
    }
    for k in 0..rims.len() {
        let (rn, hi, ho, arm) = rims[k];
        r.car.fan(rn, &[hi, ho, arm], Role::Susp, body);
        let base = k - k % RIM_NODES;
        r.car.beam(
            rn,
            rims[base + (k % RIM_NODES + 1) % RIM_NODES].0,
            Role::Susp,
            body,
        );
        r.car.beam(
            rn,
            rims[base + (k % RIM_NODES + 2) % RIM_NODES].0,
            Role::Susp,
            body,
        );
    }
    } // (the rigger's own crush cans, hubs and rim proxies)

    // ------------------------------------------------------------------
    // Hardware from the markers.
    // ------------------------------------------------------------------
    // A hinge's beams: 2 × 2 nodes (the old lattice) or, census, 3 × 3 (a
    // BeamNG hinge group is 8-12 beams: they hold the leaf in every
    // direction, where four near-parallel ones left it to swing on its latch).
    let hinge_n = if census { 3 } else { 2 };
    // On the defined body a latch and its striker carry their hardware's
    // mass ([`hardware_mass`]; `RIG_LIGHT_LATCH=1`: the part's share, as
    // before).
    let latch_mass =
        defined.is_some() && std::env::var("RIG_LIGHT_LATCH").map_or(true, |v| v != "1");
    // On the defined body a mount's body nodes are picked to span all three
    // directions ([`spread_pick`]; `RIG_NO_SPREAD=1`: the nearest, as before).
    let spread = defined.is_some() && std::env::var("RIG_NO_SPREAD").map_or(true, |v| v != "1");
    let shell_nodes = |r: &Rig| -> Vec<usize> {
        (0..r.car.nodes.len())
            .filter(|&i| {
                let n = &r.car.nodes[i];
                n.part == Part::Body
                    && r.member[i].is_some()
                    && !n.no_group
                    && !n.contact
                    && !n.name.starts_with("hub")
                    && !n.name.starts_with("rim")
            })
            .collect()
    };
    let pillar = |i: usize, r: &Rig| {
        matches!(
            r.member[i],
            Some(Role::PillarA)
                | Some(Role::PillarB)
                | Some(Role::Sill)
                | Some(Role::PillarC)
                | Some(Role::Sheet)
                | Some(Role::Web)
        )
    };
    for (front, tagf) in [(true, "F"), (false, "R")] {
        for s in [1.0f32, -1.0] {
            let x = lr(s);
            let part = Part::door(front, s > 0.0);
            let dn = part.name().trim_start_matches("gen2_").to_string();
            let own = door_nodes[&part].clone();
            for which in ["upper", "lower"] {
                let mk = marker(model, &format!("hinge_door_{tagf}{x}_{which}"));
                let mut dv: Vec<(f32, usize)> =
                    own.iter().map(|&i| (r.g(i).distance(mk), i)).collect();
                dv.sort_by(|a, b| a.0.total_cmp(&b.0));
                let bv = if spread {
                    spread_pick(&r, mk, &shell_nodes(&r), hinge_n)
                } else {
                    r.nearest(body, mk, hinge_n, |i| pillar(i, &r))
                };
                for &(_, d) in dv.iter().take(hinge_n) {
                    for &b in &bv {
                        r.car.hardware(
                            d,
                            b,
                            Role::Hinge,
                            part,
                            &format!("{dn}_hinge_{which}"),
                            ph.door_hinge,
                        );
                    }
                }
            }
            // Latch: a coupler at the marker (door node to a tag node on the
            // body), on the door's rear shut line.
            let mk = marker(model, &format!("latch_door_{tagf}{x}"));
            let latch = r.node(&format!("{dn}_latch"), mk, part, None, false);
            let strike = r.node(&format!("{dn}_strike"), mk, body, None, false);
            r.car.nodes[latch].no_group = true;
            r.car.nodes[strike].no_group = true;
            let ltag = format!("{dn}_latch");
            r.car.nodes[latch].opts = vec![
                ("couplerTag".into(), serde_json::json!(ltag)),
                ("couplerStrength".into(), json_num(ph.door_latch)),
                ("couplerStartRadius".into(), serde_json::json!(0.02)),
                (
                    "breakGroup".into(),
                    serde_json::json!(format!("{dn}_latch")),
                ),
            ];
            r.car.nodes[strike].opts = vec![("tag".into(), serde_json::json!(ltag))];
            let mut dv: Vec<(f32, usize)> = own.iter().map(|&i| (r.g(i).distance(mk), i)).collect();
            dv.sort_by(|a, b| a.0.total_cmp(&b.0));
            let inner_own: Vec<usize> = own.iter().copied().filter(|&i| !r.skin[i]).collect();
            let mut di: Vec<(f32, usize)> = inner_own
                .iter()
                .map(|&i| (r.g(i).distance(mk), i))
                .collect();
            di.sort_by(|a, b| a.0.total_cmp(&b.0));
            for &(_, d) in di.iter().take(2) {
                r.car.beam(latch, d, Role::DoorFrame, part);
            }
            for &(_, d) in dv.iter().take(6) {
                r.car.beam(latch, d, Role::DoorFrame, part);
            }
            // The latch is bolted through the door's rear wall into its inner
            // panel: it holds the door's rear edge across the car too. Its
            // nearest nodes all lie in the rear edge's plane; the door's
            // rigidifier, inboard, gives it the third direction (without it
            // the rear edge swung in and out on the latch at 17 Hz).
            if spread {
                if let Some(&rg) = own.iter().find(|&&i| r.car.nodes[i].name.ends_with("_r0")) {
                    if !r.car.has_beam(latch, rg) {
                        r.car.beam(latch, rg, Role::DoorFrame, part);
                    }
                }
            }
            let strike_to = if spread {
                spread_pick(&r, mk, &shell_nodes(&r), 5)
            } else {
                r.nearest(body, mk, 5, |i| pillar(i, &r))
            };
            if latch_mass {
                hardware_mass(&mut r, latch, strike, DOOR_LATCH_KG, &strike_to);
            }
            for b in strike_to {
                r.car.beam(strike, b, Role::Web, body);
            }
            // The window frame seats inward on the roof rail's and pillars'
            // inner flanges (SUPPORT, inboard of it: the seal it closes on).
            let frames: Vec<usize> = own
                .iter()
                .copied()
                .filter(|&i| r.car.nodes[i].name.contains("_f"))
                .collect();
            for d in frames {
                let g = r.g(d);
                let inb = r.nearest(body, g, 1, |i| {
                    let q = r.g(i);
                    q.x.abs() < g.x.abs() - 0.03
                        && matches!(
                            r.member[i],
                            Some(Role::RoofRail)
                                | Some(Role::PillarA)
                                | Some(Role::PillarB)
                                | Some(Role::PillarC)
                        )
                });
                for b in inb {
                    if r.g(b).distance(g) < 0.25 {
                        if std::env::var("GEN3_NOSEAL").is_err() {
                            r.car.beam(d, b, Role::Bearing, part);
                        }
                    }
                }
            }
            // Seals: the door's skin perimeter on its opening (SUPPORT).
            for &d in own.iter().filter(|&&i| r.skin[i]) {
                let g = r.g(d);
                if let Some(&b) = r
                    .nearest(body, g, 1, |i| {
                        pillar(i, &r) || r.member[i] == Some(Role::RoofRail)
                    })
                    .first()
                {
                    if r.g(b).distance(g) < 0.25 {
                        if std::env::var("GEN3_NOSEAL").is_err() {
                            r.car.beam(d, b, Role::Bearing, part);
                        }
                    }
                }
            }
        }
    }
    // Hood hinges, latch (two couplers at 30 kN) and rests.
    for s in [1.0f32, -1.0] {
        let x = lr(s);
        let mk = marker(model, &format!("hinge_hood_{x}"));
        let mut dv: Vec<(f32, usize)> =
            hood_ids.iter().map(|&i| (r.g(i).distance(mk), i)).collect();
        dv.sort_by(|a, b| a.0.total_cmp(&b.0));
        let bv = if spread {
            spread_pick(&r, mk, &shell_nodes(&r), hinge_n)
        } else {
            r.nearest(body, mk, hinge_n, |i| {
                matches!(
                    r.member[i],
                    Some(Role::Apron) | Some(Role::Firewall) | Some(Role::PillarA)
                )
            })
        };
        for &(_, d) in dv.iter().take(hinge_n) {
            for &b in &bv {
                r.car.hardware(
                    d,
                    b,
                    Role::Hinge,
                    Part::Hood,
                    &format!("hood_hinge_{x}"),
                    ph.hood_hinge,
                );
            }
        }
    }
    let latch_pair = |r: &mut Rig,
                      name: &str,
                      at: Vec3,
                      part: Part,
                      own: &[usize],
                      strength: f32,
                      to: &dyn Fn(usize, &Rig) -> bool| {
        let l = r.node(&format!("{name}_latch"), at, part, None, false);
        let t = r.node(&format!("{name}_strike"), at, Part::Body, None, false);
        r.car.nodes[l].no_group = true;
        r.car.nodes[t].no_group = true;
        r.car.nodes[l].opts = vec![
            (
                "couplerTag".into(),
                serde_json::json!(format!("{name}_latch")),
            ),
            ("couplerStrength".into(), serde_json::json!(strength)),
            ("couplerStartRadius".into(), serde_json::json!(0.02)),
            (
                "breakGroup".into(),
                serde_json::json!(format!("{name}_latch")),
            ),
        ];
        r.car.nodes[t].opts = vec![("tag".into(), serde_json::json!(format!("{name}_latch")))];
        let mut dv: Vec<(f32, usize)> = own.iter().map(|&i| (r.g(i).distance(at), i)).collect();
        dv.sort_by(|a, b| a.0.total_cmp(&b.0));
        for &(_, d) in dv.iter().take(6) {
            r.car.beam(l, d, Role::DoorFrame, part);
        }
        let _ = to;
        let ties = r.nearest(Part::Body, at, 5, |i| r.member[i].is_some() && !r.skin[i]);
        if latch_mass {
            let kg = if part == Part::Hood { HOOD_LATCH_KG } else { TRUNK_LATCH_KG };
            hardware_mass(r, l, t, kg, &ties);
        }
        for b in ties {
            r.car.beam(t, b, Role::Web, Part::Body);
        }
    };
    {
        let mk = marker(model, "latch_hood");
        for (k, dx) in [-0.12f32, 0.12].iter().enumerate() {
            latch_pair(
                &mut r,
                &format!("hood{k}"),
                mk + Vec3::X * *dx,
                Part::Hood,
                &hood_ids,
                ph.hood_latch,
                &|i, r| r.member[i] == Some(Role::Cross),
            );
        }
        let mk = marker(model, "latch_trunk");
        latch_pair(
            &mut r,
            "trunk",
            mk,
            Part::Trunk,
            &trunk_ids,
            ph.trunk_latch,
            &|i, r| r.member[i] == Some(Role::Cross),
        );
    }
    // Hood rests on the aprons and cowl; trunk rests on the gutters (SUPPORT).
    for (ids, part) in [(&hood_ids, Part::Hood), (&trunk_ids, Part::Trunk)] {
        for &d in ids.iter().filter(|&&i| r.skin[i]) {
            let g = r.g(d);
            if let Some(&b) = r.nearest(body, g, 1, |_| true).first() {
                if r.g(b).distance(g) < 0.30 {
                    r.car.beam(d, b, Role::Bearing, part);
                }
            }
        }
    }
    // Trunk hinges.
    for s in [1.0f32, -1.0] {
        let x = lr(s);
        let mk = marker(model, &format!("hinge_trunk_{x}"));
        let mut dv: Vec<(f32, usize)> = trunk_ids
            .iter()
            .map(|&i| (r.g(i).distance(mk), i))
            .collect();
        dv.sort_by(|a, b| a.0.total_cmp(&b.0));
        let bv = if spread {
            spread_pick(&r, mk, &shell_nodes(&r), hinge_n)
        } else {
            r.nearest(body, mk, hinge_n, |i| r.member[i].is_some())
        };
        for &(_, d) in dv.iter().take(hinge_n) {
            for &b in &bv {
                r.car.hardware(
                    d,
                    b,
                    Role::Hinge,
                    Part::Trunk,
                    &format!("trunk_hinge_{x}"),
                    ph.trunk_hinge,
                );
            }
        }
    }
    // Fender bolts (markers 1–4 per side) to the apron, hinge pillar, rail.
    for s in [1.0f32, -1.0] {
        let x = lr(s);
        let part = if s > 0.0 {
            Part::FenderL
        } else {
            Part::FenderR
        };
        let own = fender_ids[&part].clone();
        for q in 1..=4 {
            let mk = marker(model, &format!("bolt_fender_F{x}_{q}"));
            let mut dv: Vec<(f32, usize)> = own.iter().map(|&i| (r.g(i).distance(mk), i)).collect();
            dv.sort_by(|a, b| a.0.total_cmp(&b.0));
            let bv = r.nearest(body, mk, 2, |_| true);
            for &(_, d) in dv.iter().take(2) {
                for &b in &bv {
                    r.car.hardware(
                        d,
                        b,
                        Role::Bolt,
                        part,
                        &format!("fender_{x}_bolt{q}"),
                        fender_bolt,
                    );
                }
            }
        }
    }
    // Fender front brackets: the front column's top and bottom to the
    // radiator support and the rail tip (the fender's nose is otherwise held
    // by its skin alone).
    for s in [1.0f32, -1.0] {
        let x = lr(s);
        let part = if s > 0.0 {
            Part::FenderL
        } else {
            Part::FenderR
        };
        let own = fender_ids[&part].clone();
        let zmax = own.iter().map(|&i| r.g(i).z).fold(f32::MIN, f32::max);
        let front: Vec<usize> = own
            .iter()
            .copied()
            .filter(|&i| r.g(i).z > zmax - 0.02)
            .collect();
        for d in front {
            let g = r.g(d);
            let bv = r.nearest(body, g, 2, |i| {
                matches!(
                    r.member[i],
                    Some(Role::Apron) | Some(Role::Cross) | Some(Role::Rail)
                )
            });
            for b in bv {
                r.car.hardware(
                    d,
                    b,
                    Role::Bolt,
                    part,
                    &format!("fender_{x}_bracket"),
                    fender_bolt,
                );
            }
        }
    }
    // Bumper covers: clipped, not bolted through (a cover pushed onto its
    // beam slides on its clips; pulled off them, they hold to their
    // strength). Every mount is a [`Role::Clip`]: soft in compression, the
    // bolt's spring in tension. Mounts go to the body's structure only (the
    // hubs and wheel proxies are body nodes too: the side mounts had taken
    // them, so the wheels tore the covers' corners off at 11 ms in the wall).
    let structural = |i: usize, r: &Rig| {
        r.member[i].is_some()
            && !r.car.nodes[i].no_group
            && !r.car.nodes[i].name.starts_with("hub")
            && !r.car.nodes[i].name.starts_with("rim")
    };
    let old_mounts = std::env::var("GEN3_OLD_BUMPER").is_ok();
    for (front, part, nm) in [
        (true, Part::BumperF, "front"),
        (false, Part::BumperR, "rear"),
    ] {
        let (cover, _) = bumper_ids[&part].clone();
        let mount = if old_mounts { Role::Bolt } else { Role::Clip };
        for q in 1..=4 {
            let mk = marker(model, &format!("bolt_bumper_{nm}_{q}"));
            let mut dv: Vec<(f32, usize)> =
                cover.iter().map(|&i| (r.g(i).distance(mk), i)).collect();
            dv.sort_by(|a, b| a.0.total_cmp(&b.0));
            let bv = if old_mounts {
                r.nearest(body, mk, 2, |_| true)
            } else {
                r.nearest(body, mk, 2, |i| structural(i, &r))
            };
            for &(_, d) in dv.iter().take(1) {
                for &b in &bv {
                    r.car.hardware(
                        d,
                        b,
                        mount,
                        part,
                        &format!("bumper_{nm}_bolt{q}"),
                        bumper_bolt,
                    );
                }
            }
        }
        // The cover's side ends: to the body's arch front (rear: the
        // quarter's) and, at the front, to the fender's front edge (a real
        // cover's side bracket is screwed to the fender).
        let end_z = if front { bf_lo.z } else { br_hi.z };
        for &c in cover.iter() {
            let g = r.g(c);
            if (g.z - end_z).abs() >= 0.10 {
                continue;
            }
            let mut to = if old_mounts {
                r.nearest(body, g, 3, |_| true)
            } else {
                r.nearest(body, g, 2, |i| structural(i, &r))
            };
            if front && !old_mounts {
                let fp = if g.x > 0.0 {
                    Part::FenderL
                } else {
                    Part::FenderR
                };
                to.extend(r.nearest(fp, g, 2, |_| true));
            }
            for b in to {
                if r.g(b).distance(g) > 0.35 && !old_mounts {
                    continue;
                }
                r.car.hardware(
                    c,
                    b,
                    mount,
                    part,
                    &format!("bumper_{nm}_side"),
                    bumper_side_bolt,
                );
            }
        }
        // The cover's top edge: screwed to the radiator support's upper tie
        // bar (front) or the tail panel (rear) every column across the face.
        if !old_mounts {
            let top_y = cover.iter().map(|&i| r.g(i).y).fold(f32::MIN, f32::max);
            let tie: Vec<usize> = (0..r.car.nodes.len())
                .filter(|&i| {
                    r.car.nodes[i].part == body
                        && r.member[i] == Some(Role::Cross)
                        && (r.g(i).z - end_z).abs() < 0.5
                        && r.g(i).y > 0.6
                        && if front {
                            r.g(i).z > 1.9
                        } else {
                            r.g(i).z < -2.0
                        }
                })
                .collect();
            let tops: Vec<usize> = cover
                .iter()
                .copied()
                .filter(|&i| r.g(i).y > top_y - 0.02 && (r.g(i).z - end_z).abs() > 0.10)
                .collect();
            for (k, c) in tops.into_iter().enumerate() {
                let g = r.g(c);
                let mut v: Vec<(f32, usize)> =
                    tie.iter().map(|&i| (r.g(i).distance(g), i)).collect();
                v.sort_by(|a, b| a.0.total_cmp(&b.0));
                for &(d, b) in v.iter().take(1) {
                    if d < 0.45 {
                        r.car.hardware(
                            c,
                            b,
                            Role::Clip,
                            part,
                            &format!("bumper_{nm}_top{k}"),
                            bumper_side_bolt,
                        );
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Fender bolt rows: a real fender's top flange is bolted down the apron
    // rail every 15-25 cm and its rear edge to the hinge pillar (two or
    // three bolts), so it moves with the apron and the pillar rather than
    // with its own skin. Each top-flange node (the fender's nodes on the
    // hood's shut line) is bolted to the nearest apron or cowl node; each
    // rear-edge node to the nearest hinge-pillar node. One M6 8.8 bolt each
    // ([`FENDER_ROW_BOLT`]).
    // (Before the seals: a seal never takes a pair a bolt holds. With the
    // seals first the fender's rear lower leg had a SUPPORT beam to the sill
    // where its bolt belonged, and swung on it at 8.6 Hz.)
    // ------------------------------------------------------------------
    if std::env::var("GEN3_NO_FENDER_ROWS").is_err() {
        for (s, part) in [(1.0f32, Part::FenderL), (-1.0, Part::FenderR)] {
            let x = lr(s);
            let own: Vec<usize> = (0..r.car.nodes.len())
                .filter(|&i| r.car.nodes[i].part == part && !r.car.nodes[i].contact)
                .collect();
            let hood_n: Vec<usize> = (0..r.car.nodes.len())
                .filter(|&i| r.car.nodes[i].part == Part::Hood && !r.car.nodes[i].no_group)
                .collect();
            let z_rear = own.iter().map(|&i| r.g(i).z).fold(f32::MAX, f32::min);
            let mut k = 0;
            for &f in &own {
                let g = r.g(f);
                let on_hood_line = hood_n.iter().any(|&h| r.g(h).distance(g) < 0.12);
                let rear_edge = g.z < z_rear + 0.06;
                let to: Vec<usize> = if on_hood_line {
                    r.nearest(body, g, 1, |i| {
                        matches!(r.member[i], Some(Role::Apron) | Some(Role::Firewall))
                    })
                } else if rear_edge && defined.is_some() {
                    // The defined shell's hinge pillar stands on the sill's
                    // first station: its base nodes are the sill's.
                    r.nearest(body, g, 2, |i| {
                        matches!(r.member[i], Some(Role::PillarA) | Some(Role::Sill))
                    })
                } else if rear_edge {
                    r.nearest(body, g, 1, |i| r.member[i] == Some(Role::PillarA))
                } else {
                    continue;
                };
                for b in to {
                    if r.g(b).distance(g) > 0.30 || r.car.has_beam(f, b) {
                        continue;
                    }
                    k += 1;
                    r.car.hardware(
                        f,
                        b,
                        Role::Bolt,
                        part,
                        &format!("fender_{x}_row{k}"),
                        fender_row_bolt,
                    );
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Seals and stops (census): as a BeamNG car's 7-11 % of SUPPORT beams.
    // Every closure and bolted panel node bears on its [`SEAL_K`] nearest
    // body members within [`SEAL_REACH`] (compression only: the door's seal
    // on its aperture, the lids' bump stops, a fender on its apron), where
    // the old lattice had one. Each hung closure's two nodes farthest from
    // its hinges (the latch side: seal squeeze, the bump stops' preload) are
    // precompressed by [`SEAL_PRELOAD`] a beam, so the panel sits loaded
    // against its latch and a small motion meets the stop at once.
    // ------------------------------------------------------------------
    if census {
        let n0 = r.car.nodes.len();
        let body_ok = |j: usize, r: &Rig| {
            r.member[j].is_some()
                && !r.car.nodes[j].no_group
                && !r.car.nodes[j].name.starts_with("hub")
                && !r.car.nodes[j].name.starts_with("rim")
        };
        let hung = [
            (Part::DoorFL, "hinge_door_FL_upper", "hinge_door_FL_lower"),
            (Part::DoorFR, "hinge_door_FR_upper", "hinge_door_FR_lower"),
            (Part::DoorRL, "hinge_door_RL_upper", "hinge_door_RL_lower"),
            (Part::DoorRR, "hinge_door_RR_upper", "hinge_door_RR_lower"),
            (Part::Hood, "hinge_hood_L", "hinge_hood_R"),
            (Part::Trunk, "hinge_trunk_L", "hinge_trunk_R"),
        ];
        for part in [
            Part::DoorFL,
            Part::DoorFR,
            Part::DoorRL,
            Part::DoorRR,
            Part::Hood,
            Part::Trunk,
            Part::FenderL,
            Part::FenderR,
            Part::BumperF,
            Part::BumperR,
        ] {
            let own: Vec<usize> = (0..n0)
                .filter(|&i| {
                    let n = &r.car.nodes[i];
                    n.part == part && !n.no_group && !n.contact
                })
                .collect();
            // The preloaded nodes: the two farthest from the hinge line.
            let pre: Vec<usize> = hung
                .iter()
                .find(|h| h.0 == part)
                .map(|&(_, a, b)| {
                    let (ha, hb) = (marker(model, a), marker(model, b));
                    let mut v: Vec<(f32, usize)> = own
                        .iter()
                        .filter(|&&i| r.skin[i])
                        .map(|&i| {
                            let g = r.g(i);
                            (g.distance(ha).min(g.distance(hb)), i)
                        })
                        .collect();
                    v.sort_by(|x, y| y.0.total_cmp(&x.0));
                    v.into_iter().take(2).map(|x| x.1).collect()
                })
                .unwrap_or_default();
            let k = if matches!(part, Part::BumperF | Part::BumperR) { 1 } else { SEAL_K };
            for &d in &own {
                let g = r.g(d);
                for b in r.nearest(body, g, k, |j| body_ok(j, &r)) {
                    let l = r.g(b).distance(g);
                    if l > SEAL_REACH || l < 0.03 {
                        continue;
                    }
                    if r.car.has_beam(d, b) {
                        continue;
                    }
                    let j = r.car.beam(d, b, Role::Bearing, part);
                    r.car.beams[j].role = Role::Bearing;
                    if pre.contains(&d) {
                        let kk = Role::Bearing.spec(part).k_fixed;
                        r.car.precomp.insert(j, 1.0 + SEAL_PRELOAD / (kk * l));
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Shut lines: structural edge nodes along both sides of every shut line.
    // ------------------------------------------------------------------
    if std::env::var("GEN3_NO_EDGES").is_err() {
        edge_nodes(
            &mut r,
            inp,
            &apertures,
            (tk_lo.z + 0.06, tk_hi.z - 0.06),
            ph.edge_spacing,
            census,
            // (The body keeps its edge nodes: they are the corners of its
            // collision skin along every aperture. `RIG_SHELL_STOPS=1` puts
            // the stops on the shell's own nodes instead: 126 fewer collision
            // triangles, the skin's coverage p99 0.25 m against 0.16.)
            defined.is_some() && std::env::var("RIG_SHELL_STOPS").is_ok_and(|v| v == "1"),
            defined.is_some(),
        );
    }

    // ------------------------------------------------------------------
    // Push links (round 5): in the ETK the bumper fascia, lamps, fenders
    // and hood carry SUPPORT beams to the part behind them along the car
    // (measured in its JBeam: fascia to fender and hood, lamp to fascia,
    // hood and fender), so a part driven back pushes its neighbour back
    // instead of sliding past it, and the two can still part (no tension).
    // gen3's own: from each front-end part's nodes ahead of the front axle
    // to the nearest node of the part behind it, 4-30 cm back and within
    // 35° of the car's axis. SUPPORT ([`Role::Bearing`]).
    // ------------------------------------------------------------------
    if std::env::var("GEN3_NO_PUSH").is_err() {
        push_links(&mut r, census);
    }

    // ------------------------------------------------------------------
    // The collision skin from the render skin (an input that opts in,
    // `RigInput::hull_triangles`): every part's triangles are the render
    // surface's restricted Delaunay triangulation over the part's nodes at
    // its skin. Each outer render vertex goes to its nearest such node; each
    // render triangle whose corners went to three nodes gives that triangle,
    // facing as the render triangle does. The collision skin then covers the
    // render skin wherever the part has one (glass, pillars, floor, tail
    // face) and nowhere else (no triangle across an opening).
    // ------------------------------------------------------------------
    if inp.hull_triangles() {
        skin_hull(&mut r, inp);
        r.car.tri_aero = Some((hull_drag_coef(&r, inp, ph.drag_cd), 0.0));
    }
    // ------------------------------------------------------------------
    // Collision faces inside the car (the defined body): the parts of one
    // car meet each other as its outside meets the world.
    // ------------------------------------------------------------------
    let inner_skins =
        defined.is_some() && std::env::var("RIG_NO_INNER_SKINS").map_or(true, |v| v != "1");
    if inner_skins {
        interior_skins(&mut r);
    }
    // ------------------------------------------------------------------
    // Pole lanes: light contact-only nodes on the side-facing skin edges.
    // ------------------------------------------------------------------
    let surf = |p: Vec3| -> Vec3 { to_jbeam(inp.project(to_gltf(p))) };
    let side_edge = |a: Vec3, b: Vec3| {
        let m = to_gltf(0.5 * (a + b));
        // On the outer skin only (an edge behind another part's skin, such
        // as the sill's top under a door, would put lane nodes in that
        // part's contact band).
        inp.normal(inp.project(m)).x.abs() > 0.55 && inp.sdf(m) > -0.02
    };
    // No lanes on a skin hull: its triangles are already 0.24 m apart at the
    // median, and with surface contact the lanes' light nodes (0.15 kg) as
    // triangle corners made the side pole inject energy (3.7e29 J; without
    // lanes 0 J). `generated-vehicles.md`, follow-ups.
    let lane_parts: Vec<Part> = if std::env::var("GEN3_NOLANES").is_ok() || inp.hull_triangles() {
        vec![]
    } else {
        vec![
            Part::DoorFL,
            Part::DoorFR,
            Part::DoorRL,
            Part::DoorRR,
            Part::FenderL,
            Part::FenderR,
            Part::Body,
        ]
    };
    for part in lane_parts {
        let backing: Vec<usize> = (0..r.car.nodes.len())
            .filter(|&i| {
                r.car.nodes[i].part == part
                    && !r.skin[i]
                    && !r.car.nodes[i].no_group
                    && !r.car.nodes[i].contact
                    && !r.car.nodes[i].edge
            })
            .collect();
        // A part with no inner nodes (the fenders) backs its lanes on its
        // skin and edge nodes: with none, a lane node on a flat triangle had
        // its ties in one plane and hung free across it (the skin-hull's
        // fender triangles; gen3's grids happen to fold).
        let backing = if backing.is_empty() && inp.hull_triangles() {
            (0..r.car.nodes.len())
                .filter(|&i| {
                    r.car.nodes[i].part == part
                        && !r.car.nodes[i].no_group
                        && !r.car.nodes[i].contact
                })
                .collect()
        } else {
            backing
        };
        let before = r.car.nodes.len();
        r.car.lanes(part, &backing, &surf, &side_edge);
        for _ in before..r.car.nodes.len() {
            r.member.push(None);
            r.skin.push(true);
        }
    }

    // Self-collision: skin nodes, engine, rims.
    let mut on_tri = vec![false; r.car.nodes.len()];
    for t in &r.car.tris {
        for &i in &t.n {
            on_tri[i] = true;
        }
    }
    for (i, nd) in r.car.nodes.iter_mut().enumerate() {
        // Lane nodes meet the world (poles, walls, other cars) but not the
        // car's own skin: on light ties beside a shut line they would ride
        // the neighbouring part's contact band at rest.
        // (The hub nodes stay out: they are corners of the wheels' collision
        // cones only, and inside the arch beside the rails.)
        let block = nd.part == Part::Engine && nd.sub.as_deref().map_or(true, |s| s == "engine");
        // Every node of the defined shell is a contacter (as a BeamNG
        // shell's: the study's §10.3), so a door driven into its opening
        // meets the sill's and the pillars' nodes, not only the outer skin's.
        let shell = inner_skins && nd.name.starts_with("sh_");
        nd.self_collision = (on_tri[i] || block || shell || nd.name.starts_with("rim"))
            && !nd.contact
            && !nd.name.starts_with("hub")
            && std::env::var("GEN3_NOSELF").is_err()
            && std::env::var("GEN3_NOSELF_PART").map_or(true, |p| !nd.name.contains(&p));
    }

    // ------------------------------------------------------------------
    // Masses (JBeam frame positions, the 2011 Accord's).
    // ------------------------------------------------------------------
    let hpf = marker(model, "seat_H_point_FL");
    let hpr = marker(model, "seat_H_point_FR");
    let j = to_jbeam;
    if let Some(b) = &defined {
        // Every mass is its part's own (the shell's nodes, the mechanical
        // and interior parts'); the closures keep the design's part masses.
        super::body2::masses(&mut r, inp, b);
    } else {
    let named = vec![
        ("driver", 75.0, j(hpf + Vec3::new(0.0, 0.05, 0.0)), 0.6),
        ("front_seats", 35.0, j(0.5 * (hpf + hpr)), 0.75),
        ("rear_seat", 17.5, j(Vec3::new(0.0, 0.40, -1.45)), 0.65),
        (
            "dashboard_steering",
            40.0,
            j(Vec3::new(0.0, 0.80, 0.55)),
            0.6,
        ),
        ("fuel", 45.0, j(Vec3::new(0.0, 0.30, -1.20)), 0.6),
        ("interior", 50.0, j(Vec3::new(0.0, 0.60, -0.55)), 1.1),
        ("exhaust", 15.0, j(Vec3::new(0.0, 0.25, -0.40)), 1.5),
        ("spare_wheel", 15.0, j(Vec3::new(0.0, 0.35, -2.15)), 0.45),
        ("battery", 12.4, j(Vec3::new(0.55, 0.70, 1.85)), 0.45),
        (
            "radiator_cooling",
            15.0,
            j(Vec3::new(0.0, 0.55, 2.05)),
            0.45,
        ),
        ("windshield", 13.0, j(Vec3::new(0.0, 1.20, 0.25)), 0.8),
        ("rear_glass", 9.0, j(Vec3::new(0.0, 1.25, -1.45)), 0.7),
        (
            "front_subframe_steering",
            45.0,
            j(Vec3::new(0.0, 0.36, 1.40)),
            0.6,
        ),
        ("rear_subframe", 22.0, j(Vec3::new(0.0, 0.35, -1.45)), 0.7),
        ("hvac_wiring", 40.0, j(Vec3::new(0.0, 0.65, 0.60)), 0.6),
    ];
    for (name, kg, at, radius) in named {
        let kg = ph.named.get(name).copied().unwrap_or(kg);
        r.car.named.push(Named {
            name,
            kg,
            at,
            radius,
            part: body,
        });
    }
    // The raycast wheels' arms (the strut and shock tops) carry the spring
    // and damper tops: a mass of their own, so the hub's rigidity beams to
    // them (the loader's) stay inside the step's bound.
    for (s, x) in [(1.0f32, "L"), (-1.0, "R")] {
        for (nm, kg) in [(format!("tw{x}"), 6.0f32), (format!("wh{x}"), 4.0)] {
            let at = r.car.p(r.car.id(&nm));
            let _ = s;
            r.car.named.push(Named {
                name: "strut_top",
                kg,
                at,
                radius: 0.02,
                part: body,
            });
        }
    }
    for s in [1.0f32, -1.0] {
        for (za, kg) in [(axle_f, UNSPRUNG_KG + 6.0), (axle_r, UNSPRUNG_KG - 6.0)] {
            r.car.named.push(Named {
                name: "unsprung",
                kg,
                at: j(Vec3::new(track_half * s, tyre_r, za)),
                radius: 0.15,
                part: body,
            });
        }
    }
    let fixed: f32 = bumper_f_kg
        + bumper_r_kg
        + 2.0 * (accord_door_f + accord_door_r + accord_fender)
        + accord_hood
        + accord_decklid
        + accord_engine_kg
        + accord_transaxle_kg
        + accord_biw
        + 4.0 * RIM_NODES as f32 * RIM_NODE_KG
        + r.car.named.iter().map(|m| m.kg).sum::<f32>()
        + 4.0 * WHEEL_KG;
    let module = kerb - fixed;
    assert!(module >= 0.0, "masses exceed the kerb by {} kg", -module);
    r.car.named.push(Named {
        name: "front_end_module",
        kg: module,
        at: j(Vec3::new(0.0, 0.62, 1.80)),
        radius: 0.60,
        part: body,
    });
    r.car.part_mass = [
        (Part::Body, accord_biw),
        (Part::Engine, accord_engine_kg + accord_transaxle_kg),
        (Part::DoorFL, accord_door_f),
        (Part::DoorFR, accord_door_f),
        (Part::DoorRL, accord_door_r),
        (Part::DoorRR, accord_door_r),
        (Part::Hood, accord_hood),
        (Part::Trunk, accord_decklid),
        (Part::FenderL, accord_fender),
        (Part::FenderR, accord_fender),
        (Part::BumperF, bumper_f_kg),
        (Part::BumperR, bumper_r_kg),
    ]
    .into_iter()
    .collect();
    } // (the rigger's own masses)
    r.car.sheet_area = Some(voronoi_area(inp, &r.car));
    r.car.mass_floor = Some(0.7);
    r.car.row_cap_contact_only = false;
    r.car.centred_damping = true;
    // (`RIG_BOUNDED_STOPS=1`: the stops as BOUNDED beams, as before.)
    r.car.stops_in_bound = std::env::var("RIG_BOUNDED_STOPS").map_or(true, |v| v != "1");
    if census {
        r.car.bumper_zeta = Some(BUMPER_ZETA);
    }
    r.car.assign_masses();

    // The wheel stack (round 5): a front wheel pushed back meets the hinge
    // pillar's base once its tyre is crushed onto the rim (wheel_stack from
    // the hub's centre), and loads the sill and A-pillar. BOUNDED stops from
    // both hub nodes to the pillar's base nodes.
    if let Some(b) = &defined {
        super::body2::wheels(&mut r, b, env_f("GEN3_WHEEL_STACK", wheel_stack));
    } else {
    if std::env::var("GEN3_NO_WHEEL_STACK").is_err() {
        let stack = env_f("GEN3_WHEEL_STACK", wheel_stack);
        for s in [1.0f32, -1.0] {
            let x = lr(s);
            for hub in [format!("hubFi{x}"), format!("hubFo{x}")] {
                let h = r.car.id(&hub);
                for p in [format!("hp0{x}"), format!("hpi0{x}")] {
                    let t = r.car.id(&p);
                    let l = r.g(t).distance(r.g(h));
                    if l <= stack + 0.02 {
                        continue;
                    }
                    let j = r.car.beam(h, t, Role::Stop, body);
                    r.car.beams[j].role = Role::Stop;
                    r.car.stop_close.insert(j, l - stack);
                }
            }
        }
    }

    // Wheels and reference nodes.
    for (name, w, s) in [
        ("FL", "F", 1.0f32),
        ("FR", "F", -1.0),
        ("RL", "R", 1.0),
        ("RR", "R", -1.0),
    ] {
        let x = lr(s);
        let (n1, n2) = (
            r.car.id(&format!("hub{w}i{x}")),
            r.car.id(&format!("hub{w}o{x}")),
        );
        let arm = if w == "F" {
            r.car.id(&format!("tw{x}"))
        } else {
            r.car.id(&format!("wh{x}"))
        };
        r.car.wheels.push(Wheel {
            name,
            n1,
            n2,
            arm,
            dir: s,
        });
    }
    r.car.refs = [
        r.car.id("tn1"),
        r.car.id("tn2"),
        r.car.id("slt1L"),
        r.car.id("rc1"),
    ];
    } // (the rigger's own wheels and reference nodes)
    // (Defined wheels: the bound without the raycast wheels' mass, so the
    // car is stable on pressure wheels too, where the axle nodes are bare.)
    let hub = if defined.is_some() {
        HashMap::new()
    } else {
        hub_extra(&r.car)
    };
    r.car.assign_springs(&hub);
    // Zones: the crush zones' members stack (deformLimit) and carry the crush
    // yield multiple; the cabin's the cabin multiple.
    let fw_y = to_jbeam(Vec3::new(0.0, 0.0, 0.88)).y;
    let rear_y = to_jbeam(Vec3::new(0.0, 0.0, -1.75)).y;
    for b in &mut r.car.beams {
        if !matches!(b.part, Part::Body | Part::BumperF | Part::BumperR) || !b.deform.is_finite() {
            continue;
        }
        let mid = 0.5 * (r.car.nodes[b.a].p + r.car.nodes[b.b].p);
        if (mid.y < fw_y - 0.05 || mid.y > rear_y)
            && !matches!(
                b.role,
                Role::Sheet
                    | Role::Bend
                    | Role::Tie
                    | Role::Backing
                    | Role::EdgeTie
                    | Role::EdgeStop
            )
        {
            b.deform_limit = STACK_LIMIT;
        }
        if b.part != body {
            continue;
        }
        let web = matches!(
            b.role,
            Role::Web
                | Role::Cross
                | Role::Firewall
                | Role::Floor
                | Role::Tunnel
                | Role::RoofBow
                | Role::Sheet
        );
        if !web {
            continue;
        }
        if mid.y < fw_y - 0.05 || mid.y > rear_y {
            b.deform *= crush_mult();
        } else {
            b.deform *= cabin_mult();
        }
    }
    // The cabin's yield by height (the floor, sills and lower pillars, and
    // the greenhouse above the belt) and the doors': `GEN3_FLOOR_Y`,
    // `GEN3_ROOF_Y`, `GEN3_DOOR_Y` (multiples on the census yield).
    let env = |k: &str, d: f32| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(d)
    };
    let (fy, ry, dy) = (
        env("GEN3_FLOOR_Y", footwell_yield),
        env("GEN3_ROOF_Y", greenhouse_yield),
        env("GEN3_DOOR_Y", ph.door_yield),
    );
    let (sy, cy) = (
        env("GEN3_SIDE_Y", ph.side_yield),
        env("GEN3_CRUSH_Y", crush_yield),
    );
    let belt_h = to_jbeam(Vec3::new(0.0, 0.95, 0.0)).z;
    // The door above its crease (the upper inner panel and the belt
    // reinforcement): `GEN3_DOOR_UP_Y` times the door's yield.
    let door_up_h = to_jbeam(Vec3::new(0.0, 0.75, 0.0)).z;
    let duy = env("GEN3_DOOR_UP_Y", 1.0);
    // The footwell: the floor, sills and hinge pillars ahead of the front
    // seats (the side tests strike behind it).
    let foot_y = to_jbeam(Vec3::new(0.0, 0.0, 0.40)).y;
    // The front crush zone's yield graded along the car (round 5): a rail's
    // section and gauge grow from its triggered tip to the dash (tailored
    // blanks), so the crush runs soft first and stiffens as it nears the
    // cabin. CRUSH_GRADE[0] at the car's nose, [1] at the firewall, linear
    // between (`GEN3_GRADE_FRONT`, `GEN3_GRADE_REAR`).
    let nose_y = r.car.nodes.iter().map(|n| n.p.y).fold(f32::MAX, f32::min);
    let (gf, gr) = (
        env("GEN3_GRADE_FRONT", CRUSH_GRADE[0]),
        env("GEN3_GRADE_REAR", CRUSH_GRADE[1]),
    );
    let front_grade = |y: f32| {
        let t = ((fw_y - y) / (fw_y - nose_y).max(1e-3)).clamp(0.0, 1.0);
        gr + (gf - gr) * t
    };
    for b in &mut r.car.beams {
        // Census: the body's edge nodes carry real mass, so their ties keep
        // the rule's spring where the light nodes' were softened to about a
        // third. Ahead of the strut towers (the hood's and lamps' apertures,
        // the front's upper band) they yield at [`EDGE_TIE_YIELD`] of the
        // sheet's, a flange's: at the full yield they carried the head-on
        // load high (front hubs to 0.61 m against 0.51) and stiffened the
        // wall (crush 564 mm, b1 42). Behind the towers they keep the full
        // yield (at 0.35 two bent in the parked car at the B-pillar and six
        // on a kerb; the price is the roof, 6.2 × weight against 4.9).
        let ahead = 0.5 * (r.car.nodes[b.a].p.y + r.car.nodes[b.b].p.y) < -0.25;
        if census && ahead && b.part == body && b.role == Role::EdgeTie && b.deform.is_finite() {
            b.deform *= env("RIG_EDGE_TIE_Y", EDGE_TIE_YIELD);
        }
        if !b.deform.is_finite()
            || b.break_group.is_some()
            || matches!(
                b.role,
                Role::EdgeTie
                    | Role::EdgeStop
                    | Role::Bearing
                    | Role::Susp
                    | Role::Mount
                    | Role::Stop
                    | Role::Engine
                    | Role::Tie
                    | Role::Backing
            )
        {
            continue;
        }
        let mid = 0.5 * (r.car.nodes[b.a].p + r.car.nodes[b.b].p);
        let crush_zone = mid.y < fw_y - 0.05 || mid.y > rear_y;
        if census && b.part == Part::Hood {
            // The hood buckles: its sheet and inner frame yield in their
            // plane at [`HOOD_YIELD`] of the census's (a beam at 9.9-10.9 kN
            // against a rail beam's 2.8 kN made the hood the stiffer path).
            b.deform *= env("RIG_HOOD_Y", HOOD_YIELD);
        } else if census
            && b.part == body
            && crush_zone
            && mid.y < fw_y - 0.05
            && matches!(b.role, Role::Rail | Role::CrushCan)
        {
            // The primary frontal load path: the front rails and crush cans
            // at [`RAIL_YIELD`] times the crush zone's yield.
            b.deform *= cy * front_grade(mid.y) * env("RIG_RAIL_Y", RAIL_YIELD);
        } else if b.part.is_door() {
            b.deform *= dy;
            if mid.z > door_up_h {
                b.deform *= duy;
            }
            // Census: the window frame is a thin channel, not door structure
            // (its legs at the door frame's yield propped the roof rail
            // through their seals: FMVSS 216a 7.05 × weight).
            let frame = |i: usize| {
                let n = &r.car.nodes[i].name;
                n.ends_with("_f0") || n.ends_with("_f1")
            };
            if census && (frame(b.a) || frame(b.b)) {
                b.deform *= env("RIG_WINDOW_FRAME_Y", WINDOW_FRAME_YIELD);
            }
        } else if b.part == body && crush_zone && b.role != Role::Sheet {
            // The suspension towers (front strut, rear wheelhouse) carry the
            // car's weight into the rails: at the crush zone's 0.28 two of
            // their beams yielded under the parked car; they keep round 3's
            // 0.4, which none did ([`TOWER_YIELD`]). Round 5: so do the
            // apron's joints into the hinge pillar's front (`hpf`), the
            // tower's load path to the cabin: at 0.25 one (hpf0R-ar1R)
            // yielded parked.
            let tower = |i: usize| {
                let n = &r.car.nodes[i].name;
                n.starts_with("tw") || n.starts_with("wh") || n.starts_with("hpf")
            };
            // Census: the apron (upper rail) from the strut tower back to the
            // cowl and hinge pillar is the suspension's load path, not crush
            // structure: at the crush yield a 5 cm kerb at 30 km/h bent 42 of
            // its beams (2.3 mm, the whole front clip left 2-3 mm off the
            // body); it takes [`APRON_REAR_YIELD`].
            let apron_rear = |i: usize| census && r.car.nodes[i].name.starts_with("ar") && r.car.nodes[i].p.y > -0.15;
            // Census: the front's upper band (the apron tips, the radiator
            // support's upper tie bar, the lamp pockets, above
            // [`UPPER_BAND_Y`]) at [`UPPER_YIELD`] of the crush zone's yield:
            // sheet-metal brackets that fold, where the rails carry.
            let upper = census
                && mid.y < -0.15
                && mid.z > to_jbeam(Vec3::new(0.0, UPPER_BAND_Y, 0.0)).z;
            // Census: the crush zone's anchorage on the cabin (the rails'
            // kick-down onto the floor, the sills and the firewall's base:
            // a front beam with an end on a floor, sill or tunnel node) at
            // [`KICKDOWN_YIELD`]. At the crush yield it folded first: head
            // on, the rails drove the toe board's floor and sill nodes down
            // onto the ground by 60 ms, and the ground threw both cars up
            // (1.2 m/s, every wheel in the air: known-bugs S2's nose lift).
            let cabin_floor = |i: usize| {
                let n = &r.car.nodes[i].name;
                ["slo", "slt", "slb", "flr", "tn"].iter().any(|p| n.starts_with(p))
            };
            let kick = census && mid.y < fw_y - 0.05 && (cabin_floor(b.a) || cabin_floor(b.b));
            b.deform *= if kick {
                env("RIG_KICKDOWN_Y", KICKDOWN_YIELD)
            } else if apron_rear(b.a) || apron_rear(b.b) {
                env("RIG_APRON_REAR_Y", APRON_REAR_YIELD)
            } else if tower(b.a) || tower(b.b) {
                cy.max(env("GEN3_TOWER_Y", tower_yield))
            } else if upper {
                cy * front_grade(mid.y) * env("RIG_UPPER_Y", UPPER_YIELD)
            } else if mid.y < fw_y - 0.05 {
                // The front zone's beams across the car's axis (the rails'
                // and aprons' webs and diagonals) against the beams along it
                // ([`CROSS_SECTION_YIELD`], 1: alike). Measured head on: a
                // stiffer section (3, 6) passes more of the contact's moment
                // to the cabin and lifts the nose more (front hubs to 0.60,
                // 0.96 m against 0.47 at 1).
                let d = r.car.nodes[b.a].p - r.car.nodes[b.b].p;
                let along = d.y.abs() / d.length().max(1e-6);
                let sec = if census && along < 0.8 {
                    env("RIG_SECTION_Y", CROSS_SECTION_YIELD)
                } else {
                    1.0
                };
                cy * front_grade(mid.y) * sec
            } else {
                cy
            };
        } else if b.part == body && mid.y >= fw_y - 0.05 && mid.y <= rear_y {
            b.deform *= if mid.z > belt_h {
                ry
            } else if mid.y < foot_y {
                fy
            } else {
                sy
            };
        }
    }
    // The doors' side intrusion beams: `GEN3_INTRUSION_Y` times their yield.
    let iy = env("GEN3_INTRUSION_Y", intrusion_yield);
    for b in &mut r.car.beams {
        if b.role == Role::Intrusion && b.deform.is_finite() {
            b.deform *= iy;
        }
    }
    // The A-pillar's chords (hinge pillar top to the header, outer and inner
    // walls): a hot-stamped section, its yield `GEN3_APILLAR_Y` times the
    // greenhouse's.
    let ay = env("GEN3_APILLAR_Y", apillar_yield);
    let chord_ends = ["hp2", "hpi1", "rr0", "rri0"];
    for b in &mut r.car.beams {
        let na = &r.car.nodes[b.a].name;
        let nb = &r.car.nodes[b.b].name;
        let is = |n: &str| {
            chord_ends
                .iter()
                .any(|c| n.len() == c.len() + 1 && n.starts_with(c))
        };
        let low = |n: &str| n.starts_with("hp");
        if b.role == Role::PillarA && is(na) && is(nb) && low(na) != low(nb) && b.deform.is_finite()
        {
            b.deform *= ay;
        }
    }
    // The floor: cross members at the stations that carry one (sill to the
    // floor rail and the tunnel), the floor pan elsewhere. A pole at the
    // B-pillar's station meets the sill between two cross members and bends
    // it; a wide barrier pushes the sill at every station and loads them
    // end on.
    if std::env::var("GEN3_OLD_SILL").is_err() {
        let (cmy, pany) = (
            env("GEN3_CM_Y", cross_member_yield),
            env("GEN3_PAN_Y", floor_pan_yield),
        );
        // (station, kind): kind 0 sill, 1 floor rail or tunnel.
        let station = |n: &str| -> Option<(usize, u8)> {
            let t = n.trim_end_matches(['L', 'R']);
            for (pfx, kind) in [("slo", 0u8), ("slt", 0), ("slb", 0), ("flr", 1), ("tn", 1)] {
                if let Some(k) = t.strip_prefix(pfx) {
                    if let Ok(k) = k.parse::<usize>() {
                        return Some((k, kind));
                    }
                }
            }
            None
        };
        let cm = |k: usize| k != 2;
        for b in &mut r.car.beams {
            if b.part != body || !b.deform.is_finite() || b.break_group.is_some() {
                continue;
            }
            let (Some((ka, ta)), Some((kb, tb))) = (
                station(&r.car.nodes[b.a].name),
                station(&r.car.nodes[b.b].name),
            ) else {
                continue;
            };
            // The sill's own section at a station (its outer wall onto its
            // top and bottom): `GEN3_SILL_SEC_Y`.
            if ta == 0 && tb == 0 {
                if ka == kb {
                    b.deform *= env("GEN3_SILL_SEC_Y", 1.0);
                } else {
                    b.deform *= env("GEN3_LONG_Y", 1.0);
                }
                continue;
            }
            if ta == 1 && tb == 1 && ka != kb {
                b.deform *= env("GEN3_LONG_Y", 1.0);
                continue;
            }
            // Across the floor: sill to floor, or floor rail to tunnel.
            let across = ta != tb
                || (ta == 1
                    && (r.car.nodes[b.a].name.starts_with("tn")
                        != r.car.nodes[b.b].name.starts_with("tn")));
            if !across {
                continue;
            }
            // The footwell's floor (toe board to the seat's front cross
            // member) keeps its own yield: the frontal's feet intrusion.
            b.deform *= if ka == kb && cm(ka) {
                cmy
            } else if ka.max(kb) <= 1 {
                1.0
            } else {
                pany
            };
        }
        // The B-pillar: hot-stamped above the door's crease, the tailored
        // blank's soft zone below it.
        let (bu, bl) = (
            env("GEN3_BP_UP", bpillar_upper_yield),
            env("GEN3_BP_LO", bpillar_lower_yield),
        );
        let split = to_jbeam(Vec3::new(0.0, 0.60, 0.0)).z;
        for b in &mut r.car.beams {
            if b.role != Role::PillarB || !b.deform.is_finite() || b.break_group.is_some() {
                continue;
            }
            let mid = 0.5 * (r.car.nodes[b.a].p + r.car.nodes[b.b].p);
            b.deform *= if mid.z > split { bu } else { bl };
        }
    }
    let _ = (arch_r, &eng);
    // Defined beams keep their definitions' yield forces (the zone rules
    // above are the rigger's own members').
    for (&j, f) in &r.car.fixed {
        r.car.beams[j].deform = f.deform;
        r.car.beams[j].deform_limit = 0.0;
    }
    r
}

/// The yield of the crush zones' body members (rails, cans, aprons, rear
/// rails and their webs) as a multiple of the census's 1.15 %: 0.32 %, in
/// the census's body band (p10 0.2-0.3 %, p50 1.0-1.3 %). Measured on the
/// wall (surface contact on), round 4's structure: dynamic crush 581 mm at
/// 0.4, 619 at 0.3, 631 at 0.28 (620-750); t_m 66 / 71 / 72 ms (63-76);
/// b1 42.1 / 39.8 / 39.2 (<= 39.2).
/// Round 5 (push links, stacks, the hinge pillar's joints at the towers'
/// yield): 620 mm at 0.25, 625 at 0.24, 629 at 0.23; b1 39.4 / 38.6 / 38.6;
/// restitution 0.201 / 0.207 / 0.202 (each ±0.005 run to run of the yield).
pub const CRUSH_YIELD: f32 = 0.23;
/// The suspension towers' beams in the crush zone (see the zone pass).
pub const TOWER_YIELD: f32 = 0.4;
/// The footwell's (floor, sills, hinge pillars and firewall ahead of the
/// front seats, below the belt): 4.1 %, under the census's body p90
/// (4.9-9.3 %): high-strength steel there. Feet intrusion 94 to 53 mm (<= 67).
pub const FOOTWELL_YIELD: f32 = 3.0;
/// The greenhouse's (above the belt, firewall to the rear shelf): 0.52 %.
/// FMVSS 216a 5.94 to 5.02 x weight at 0.45 in round 3 (4.48-5.32); with
/// round 4's hot-stamped A-pillar and B-pillar upper, 5.57 at 0.45, 5.32 at
/// 0.38.
pub const GREENHOUSE_YIELD: f32 = 0.38;
/// The A-pillar chords' yield over the greenhouse's: a hot-stamped section
/// (boron steel, about 1,000-1,200 MPa against 300-400 MPa for the HSLA the
/// greenhouse's yield stands for).
pub const APILLAR_YIELD: f32 = 3.0;
/// The floor's cross members ahead of and behind the B-pillar (m): the
/// footwell's rear (the front seat's front mounts; the old evenly spaced
/// station, which the frontal's feet intrusion needs) and the front seat's
/// rear mounts.
pub const SEAT_CM_F: f32 = 0.53;
pub const SEAT_CM_R: f32 = 0.35;
/// The floor cross members' yield over the census's (dash, seat front, seat
/// rear, kick-up): high-strength sections a wide side load reaches through
/// the sill (`GEN3_CM_Y`).
pub const CROSS_MEMBER_YIELD: f32 = 3.0;
/// The front crush zone's yield grade, nose and firewall, on its yield
/// (1, 1: ungraded; calibrated in round 5 on the pulse's bands).
pub const CRUSH_GRADE: [f32; 2] = [1.0, 1.0];
/// The floor pan's (the floor between the cross members and its diagonals):
/// 0.7 mm sheet, which buckles in plane (`GEN3_PAN_Y`).
pub const FLOOR_PAN_YIELD: f32 = 0.3;
/// The B-pillar's upper section over the greenhouse's yield (hot-stamped)
/// and its lower section over the side's (the tailored blank's soft zone):
/// `GEN3_BP_UP`, `GEN3_BP_LO`.
pub const BPILLAR_UPPER_YIELD: f32 = 3.0;
pub const BPILLAR_LOWER_YIELD: f32 = 0.5;
/// How far the intrusion beam's outer chord sits under the door skin (m).
pub const INTRUSION_OUT: f32 = 0.02;
/// The door intrusion beams' yield over their role's.
pub const INTRUSION_YIELD: f32 = 1.0;

/// Spacing of the shut lines' edge nodes (m): a real closure's flange is
/// bolted, hinged or stopped every 15-25 cm along it.
pub const EDGE_SPACING: f32 = 0.22;
/// The front crush zone's beams across the car's axis: their yield against
/// the beams along it.
pub const CROSS_SECTION_YIELD: f32 = 1.0;
/// The bumpers' damping ratio (plastic covers on foam: the census's bumper
/// beams run at ζ 0.07-0.33, the ETK's at 0.16).
pub const BUMPER_ZETA: f32 = 0.16;
/// The rails' anchorage on the cabin floor: its yield (× the census 1.15 %).
pub const KICKDOWN_YIELD: f32 = 1.0;
/// The bumper beam's centre height (m, model): the cover's middle row.
pub const BUMPER_BEAM_Y: f32 = 0.48;
/// The front's upper band starts here (m, model) and yields at this share
/// of the crush zone's yield.
pub const UPPER_BAND_Y: f32 = 0.62;
pub const UPPER_YIELD: f32 = 0.7;
/// The bumper foam's solid thickness (m): a cover node's stop on the beam.
pub const FOAM_SOLID: f32 = 0.03;
/// The hood's yield against the census sheet's (it buckles in a frontal).
pub const HOOD_YIELD: f32 = 0.4;
/// The front rails' and crush cans' yield against the crush zone's.
pub const RAIL_YIELD: f32 = 4.0;
/// The body's shut-line edge ties ahead of the strut towers: their yield
/// against the sheet's.
pub const EDGE_TIE_YIELD: f32 = 0.35;
/// A door's window frame: its yield against the door frame's.
pub const WINDOW_FRAME_YIELD: f32 = 0.3;
/// The apron's yield from the strut tower back (× the census 1.15 %).
pub const APRON_REAR_YIELD: f32 = 1.0;
/// Seals and stops: body members a closure node bears on, their reach (m)
/// and the preload of a precompressed one (N).
pub const SEAL_K: usize = 2;
pub const SEAL_REACH: f32 = 0.30;
pub const SEAL_PRELOAD: f32 = 150.0;
/// The census door's depth: its inner panel behind its skin (m).
pub const DOOR_DEPTH: f32 = 0.10;
/// How far an edge node sits in from its panel's edge, and under its skin (m).
pub const EDGE_IN: f32 = 0.025;
pub const EDGE_UNDER: f32 = 0.01;

/// A mesh's boundary vertices (positions welded, so a split vertex counts
/// once): one vertex index per welded boundary point.
fn boundary_points(p: &super::model::Part) -> Vec<usize> {
    let key = |q: Vec3| {
        (
            (q.x * 2e4).round() as i64,
            (q.y * 2e4).round() as i64,
            (q.z * 2e4).round() as i64,
        )
    };
    let mut weld: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let wid: Vec<usize> = p
        .pos
        .iter()
        .map(|&q| {
            let l = weld.len();
            *weld.entry(key(q)).or_insert(l)
        })
        .collect();
    let mut edges: HashMap<(usize, usize), u32> = HashMap::new();
    for t in &p.tris {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let (wa, wb) = (wid[a as usize], wid[b as usize]);
            if wa != wb {
                *edges.entry((wa.min(wb), wa.max(wb))).or_insert(0) += 1;
            }
        }
    }
    let mut on = vec![false; weld.len()];
    for (&(a, b), &n) in &edges {
        if n == 1 {
            on[a] = true;
            on[b] = true;
        }
    }
    let mut seen = vec![false; weld.len()];
    (0..p.pos.len())
        .filter(|&i| {
            let w = wid[i];
            let keep = on[w] && !seen[w];
            seen[w] = true;
            keep
        })
        .collect()
}

/// Shut-line edge nodes: where two parts' render meshes meet (boundary
/// vertices of different parts within 1 cm, as the harness finds seams), a
/// structural node on each side every [`EDGE_SPACING`], [`EDGE_IN`] in from
/// its panel's edge and [`EDGE_UNDER`] under its skin, each tied into its
/// own part's frame (its nearest structural nodes, an inner one among them)
/// and to its neighbours along the edge, and a SUPPORT stop to the node
/// across the line. They are in their part's binding pool, so a border
/// vertex rides nodes within about 10 cm instead of extrapolating past the
/// part's lattice; the stops carry a closure's edge onto its neighbour as a
/// flange does. Returns the edge nodes.
/// The push links (see the call): (front part, part behind) pairs.
fn push_links(r: &mut Rig, census: bool) {
    let pairs = [
        (Part::BumperF, Part::Body),
        (Part::BumperF, Part::FenderL),
        (Part::BumperF, Part::FenderR),
        (Part::BumperF, Part::Hood),
        (Part::FenderL, Part::Body),
        (Part::FenderR, Part::Body),
        (Part::Hood, Part::Body),
    ];
    let axle = r.axle_f;
    let reach = env_f("GEN3_PUSH_REACH", PUSH_REACH);
    let cone = (35f32).to_radians().tan();
    for (pa, pb) in pairs {
        // Census: the cover does not push the hood (an unyielding stop from
        // the cover's top row into the hood's front edge put the head-on
        // load through the hood, above the centre of mass: known-bugs S2).
        if census && pa == Part::BumperF && pb == Part::Hood {
            continue;
        }
        let fronts: Vec<usize> = (0..r.car.nodes.len())
            .filter(|&i| {
                let n = &r.car.nodes[i];
                n.part == pa && !n.contact && r.g(i).z > axle
            })
            .collect();
        for a in fronts {
            let ga = r.g(a);
            let best = (0..r.car.nodes.len())
                .filter(|&j| {
                    let n = &r.car.nodes[j];
                    n.part == pb
                        && !n.contact
                        && !n.name.starts_with("hub")
                        && !n.name.starts_with("rim")
                })
                .filter_map(|j| {
                    let d = r.g(j) - ga;
                    let back = -d.z;
                    let side = (d.x * d.x + d.y * d.y).sqrt();
                    (back >= PUSH_MIN && back <= reach && side <= cone * back)
                        .then_some((d.length(), j))
                })
                .min_by(|x, y| x.0.total_cmp(&y.0));
            if let Some((_, b)) = best {
                if !r.car.has_beam(a, b) {
                    let part = if pa == Part::Body { pb } else { pa };
                    let j = r.car.beam(a, b, Role::Bearing, part);
                    r.car.beams[j].role = Role::Bearing;
                }
            }
        }
    }
}

/// Push links reach 4 cm to [`PUSH_REACH`] behind a node (shorter ones are
/// the shut line's own edge stops).
pub const PUSH_MIN: f32 = 0.04;
pub const PUSH_REACH: f32 = 0.30;

/// A latch's mass on its closure (kg): the latch or striker, its
/// reinforcement in the panel's wall and what is bolted beside it (a door's
/// handle carrier and lock; a lid's lock barrel). A door latch alone weighs
/// about 0.9 kg.
pub const DOOR_LATCH_KG: f32 = 2.5;
pub const HOOD_LATCH_KG: f32 = 1.0;
pub const TRUNK_LATCH_KG: f32 = 1.5;
/// The body's side of a latch (kg): the striker or lock and the reinforcement
/// it bolts through, mass that belongs to the member it stands on and is
/// taken from that member's nodes.
pub const STRIKE_KG: f32 = 2.5;

/// Gives a latch node and its striker node their hardware's mass. The
/// solver holds a coupled pair with a spring sized by the pair's reduced mass
/// (`explicit_beams::coupler_rates`: k = 0.16 mu / h^2), so two 0.5 kg tag
/// nodes are a 2e5 N/m latch: more than half of each door's first mode's
/// strain energy sat in it (18 Hz), a third of the lid's. The striker's mass
/// is the member's own at that place: it comes out of the nodes the striker
/// is tied to, equally, so the body's mass and its distribution stay.
fn hardware_mass(r: &mut Rig, latch: usize, strike: usize, latch_kg: f32, ties: &[usize]) {
    r.car.nodes[latch].fixed_mass = latch_kg;
    let givers: Vec<usize> = ties
        .iter()
        .copied()
        .filter(|&j| r.car.nodes[j].fixed_mass > 2.0 * STRIKE_KG / ties.len() as f32)
        .collect();
    if givers.is_empty() {
        return;
    }
    let each = STRIKE_KG / givers.len() as f32;
    for j in givers {
        r.car.nodes[j].fixed_mass -= each;
    }
    r.car.nodes[strike].fixed_mass = STRIKE_KG;
}

/// How far a mount's beams may reach for a body node that gives them a
/// direction they lack ([`spread_pick`]).
pub const MOUNT_SPREAD_REACH: f32 = 0.75;
/// A rigidifier's depth behind its panel's skin (the study's rule 20:
/// 0.2-0.4 m), and how far its beams reach.
pub const RIGIDIFIER_DEPTH: f32 = 0.22;
pub const RIGIDIFIER_REACH: f32 = 0.95;

/// `n` body nodes for a mount at `at`: the nearest of `cands`, then each time
/// the candidate (within [`MOUNT_SPREAD_REACH`]) that most raises the
/// weakest direction of the beams so far, so the mount's beams span all
/// three axes. A hinge leaf bolted to a pillar holds in every direction;
/// the nearest nodes alone can all lie in the pillar's plane (the front
/// doors slid fore and aft on theirs at 12.6 Hz).
fn spread_pick(r: &Rig, at: Vec3, cands: &[usize], n: usize) -> Vec<usize> {
    let mut v: Vec<(f32, usize)> = cands
        .iter()
        .map(|&i| (r.g(i).distance(at), i))
        .filter(|x| x.0 > 0.02)
        .collect();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out: Vec<usize> = v.iter().take(1).map(|x| x.1).collect();
    let second = |d: Vec3| glam::Mat3::from_cols(d * d.x, d * d.y, d * d.z);
    while out.len() < n {
        let m = out.iter().fold(glam::Mat3::ZERO, |m, &j| {
            m + second((r.g(j) - at).normalize_or_zero())
        });
        let best = v
            .iter()
            .filter(|x| x.0 < MOUNT_SPREAD_REACH && !out.contains(&x.1))
            .map(|&(d, j)| {
                let mm = m + second((r.g(j) - at).normalize_or_zero());
                // Two beams leave one axis empty whatever they are: judge
                // them by their second-weakest direction.
                let lam = if out.len() == 1 {
                    let d0 = (r.g(out[0]) - at).normalize_or_zero();
                    1.0 - d0.dot((r.g(j) - at).normalize_or_zero()).abs()
                } else {
                    weakest_axis(mm).0
                };
                (lam - 0.15 * d, j)
            })
            .max_by(|a, b| a.0.total_cmp(&b.0));
        match best {
            Some((_, j)) => out.push(j),
            None => break,
        }
    }
    out
}

/// A panel's rigidifier ([`RIGIDIFIER_DEPTH`] under its skin at `at`, a point
/// of its inner frame): beamed to the panel's nodes within
/// [`RIGIDIFIER_REACH`], so the panel bends as a section of that depth and
/// not as its skin alone (bending stiffness goes as k d²; the study's §5.1
/// and rule 20). Not a collision node.
fn rigidifier(r: &mut Rig, name: &str, at: Vec3, part: Part, own: &[usize]) -> usize {
    let i = r.node(name, at, part, Some(Role::DoorFrame), false);
    for &j in own {
        let d = r.g(j).distance(at);
        if d > 0.03 && d < RIGIDIFIER_REACH && !r.car.has_beam(i, j) {
            r.car.beam(i, j, Role::DoorFrame, part);
        }
    }
    i
}

/// The parts whose own solid is a collision skin: the convex hull of their
/// nodes ([`interior_skins`]).
pub const HULL_PARTS: [&str; 8] = [
    "seat_FL",
    "seat_FR",
    "seat_rear",
    "dash",
    "console",
    "engine",
    "fuel_tank",
    "radiator",
];

/// The convex hull of `p` as polygons (indices into `p`, in order around
/// each face): every plane through three points with all the others on one
/// side, its coplanar points taken together. A flat set is one polygon.
fn hull_faces(p: &[Vec3]) -> Vec<Vec<usize>> {
    let eps = 2e-3;
    let mut seen: std::collections::BTreeSet<Vec<usize>> = Default::default();
    let mut out = Vec::new();
    for a in 0..p.len() {
        for b in a + 1..p.len() {
            for c in b + 1..p.len() {
                let n = (p[b] - p[a]).cross(p[c] - p[a]);
                if n.length() < 1e-5 {
                    continue;
                }
                let n = n.normalize();
                let d: Vec<f32> = p.iter().map(|q| n.dot(*q - p[a])).collect();
                if !(d.iter().all(|&x| x <= eps) || d.iter().all(|&x| x >= -eps)) {
                    continue;
                }
                let mut on: Vec<usize> = (0..p.len()).filter(|&i| d[i].abs() <= eps).collect();
                if !seen.insert(on.clone()) {
                    continue;
                }
                let cen = on.iter().map(|&i| p[i]).sum::<Vec3>() / on.len() as f32;
                let u = (p[on[0]] - cen).normalize_or_zero();
                let v = n.cross(u);
                on.sort_by(|&i, &j| {
                    let f = |k: usize| (p[k] - cen).dot(v).atan2((p[k] - cen).dot(u));
                    f(i).total_cmp(&f(j))
                });
                out.push(on);
            }
        }
    }
    out
}

/// The collision faces inside the defined car. Each is a triangle on its
/// part's own nodes, marked [`Tri::inner`] (no air on it); its nodes become
/// contacters with the rest of the skin's.
///
/// - **The cabin's structure** (the shell): each sill's outer face (bottom
///   to top), the floor pan, the tunnel's walls, and the firewall from the
///   cowl to the toe board.
/// - **Each door's box:** its inner panel, its bottom, its belt and its two
///   end walls, so a door is a solid on both faces.
/// - **Interior and engine-bay parts** ([`HULL_PARTS`]): the convex hull of
///   the part's nodes, the solid they are the corners of.
///
/// Before this only the outer skin had triangles: a door T-boned at 50 km/h
/// went through the sill's top edge and the A-pillar, and nothing in the
/// cabin could stop it or be pushed by it.
fn interior_skins(r: &mut Rig) {
    let ix: HashMap<String, usize> = r
        .car
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.name.clone(), i))
        .collect();
    let mut have: std::collections::BTreeSet<[usize; 3]> = r
        .car
        .tris
        .iter()
        .map(|t| {
            let mut k = t.n;
            k.sort_unstable();
            k
        })
        .collect();
    let mut add: Vec<Tri> = Vec::new();
    let pos: Vec<Vec3> = (0..r.car.nodes.len()).map(|i| r.g(i)).collect();
    let mut poly = |ids: &[usize], part: Part, sub: Option<&str>| {
        for k in 1..ids.len().saturating_sub(1) {
            let n = [ids[0], ids[k], ids[k + 1]];
            let area = (pos[n[1]] - pos[n[0]]).cross(pos[n[2]] - pos[n[0]]).length();
            let mut key = n;
            key.sort_unstable();
            if area < 1e-5 || !have.insert(key) {
                continue;
            }
            add.push(Tri {
                n,
                part,
                group: None,
                sub: sub.map(str::to_string),
                inner: true,
            });
        }
    };
    let named = |names: &[String]| -> Option<Vec<usize>> {
        names.iter().map(|n| ix.get(n).copied()).collect()
    };
    // The shell.
    for s in ["l", "r"] {
        for row in 1..=4 {
            let (a, b) = (row, row + 1);
            for q in [
                // the sill's outer face
                [format!("sh_c1_{a}o{s}"), format!("sh_c1_{b}o{s}"), format!("sh_c2_{b}{s}"), format!("sh_c2_{a}{s}")],
                // the floor pan
                [format!("sh_c1_{a}o{s}"), format!("sh_c1_{b}o{s}"), format!("sh_c1_{b}t{s}"), format!("sh_c1_{a}t{s}")],
                // the tunnel's wall
                [format!("sh_c1_{a}t{s}"), format!("sh_c1_{b}t{s}"), format!("sh_c3_{b}"), format!("sh_c3_{a}")],
            ] {
                if let Some(ids) = named(&q) {
                    poly(&ids, Part::Body, None);
                }
            }
        }
    }
    let across = ["ol", "il", "ir", "or"];
    let floor1 = ["ol", "tl", "tr", "or"];
    for k in 0..3 {
        for q in [
            [format!("sh_b2_{}", across[k]), format!("sh_b2_{}", across[k + 1]), format!("sh_b1_{}", across[k + 1]), format!("sh_b1_{}", across[k])],
            [format!("sh_b1_{}", across[k]), format!("sh_b1_{}", across[k + 1]), format!("sh_c1_1{}", floor1[k + 1]), format!("sh_c1_1{}", floor1[k])],
        ] {
            if let Some(ids) = named(&q) {
                poly(&ids, Part::Body, None);
            }
        }
    }
    // The doors.
    for part in [Part::DoorFL, Part::DoorFR, Part::DoorRL, Part::DoorRR] {
        let dn = part.name().trim_start_matches("gen2_").to_string();
        let o = |row: usize, k: usize| format!("{dn}_o{row}{k}");
        let i = |row: usize, k: usize| format!("{dn}_i{row}{k}");
        for k in 0..2 {
            for q in [
                [i(0, k), i(0, k + 1), i(1, k + 1), i(1, k)],
                [o(0, k), o(0, k + 1), i(0, k + 1), i(0, k)],
                [o(2, k), o(2, k + 1), i(1, k + 1), i(1, k)],
            ] {
                if let Some(ids) = named(&q) {
                    poly(&ids, part, None);
                }
            }
        }
        for k in [0, 2] {
            if let Some(ids) = named(&[o(0, k), i(0, k), i(1, k), o(2, k)]) {
                poly(&ids, part, None);
            }
        }
    }
    // The parts that are solids of their own.
    for sub in HULL_PARTS {
        let ids: Vec<usize> = (0..r.car.nodes.len())
            .filter(|&i| r.car.nodes[i].sub.as_deref() == Some(sub))
            .collect();
        if ids.len() < 3 {
            continue;
        }
        let pts: Vec<Vec3> = ids.iter().map(|&i| pos[i]).collect();
        for face in hull_faces(&pts) {
            let f: Vec<usize> = face.iter().map(|&k| ids[k]).collect();
            poly(&f, Part::Engine, Some(sub));
        }
    }
    r.car.tris.extend(add);
}

/// The least share of an edge node's ties' directions along their weakest
/// axis (the smallest eigenvalue of the unit directions' second moment over
/// their number; a third is isotropic) for the node to be made.
pub const EDGE_SPREAD: f32 = 0.05;

/// The spread the ties of a body edge node at `g` could reach: its four
/// nearest structural nodes, then up to three more within 0.65 m, each the
/// one most along the weakest axis so far (as `edge_nodes` ties them).
fn edge_tie_spread(r: &Rig, g: Vec3, edge: &[usize]) -> f32 {
    let cands: Vec<usize> = r
        .nearest(Part::Body, g, 24, |j| {
            !edge.contains(&j)
                && !r.car.nodes[j].no_group
                && !r.car.nodes[j].name.starts_with("hub")
                && !r.car.nodes[j].name.starts_with("rim")
        })
        .into_iter()
        .filter(|&j| r.g(j).distance(g) < 0.65 && r.g(j).distance(g) > 0.03)
        .collect();
    let mut tied: Vec<usize> = cands.iter().copied().take(4).collect();
    let moment = |tied: &[usize]| {
        tied.iter().fold(glam::Mat3::ZERO, |m, &j| {
            let d = (r.g(j) - g).normalize_or_zero();
            m + glam::Mat3::from_cols(d * d.x, d * d.y, d * d.z)
        })
    };
    for _ in 0..3 {
        let (lam, axis) = weakest_axis(moment(&tied));
        if lam > 0.12 * tied.len() as f32 {
            break;
        }
        let best = cands.iter().copied().filter(|j| !tied.contains(j)).max_by(|&a, &b| {
            let f = |j: usize| (r.g(j) - g).normalize_or_zero().dot(axis).abs();
            f(a).total_cmp(&f(b))
        });
        match best {
            Some(j) => tied.push(j),
            None => break,
        }
    }
    if tied.is_empty() {
        return 0.0;
    }
    weakest_axis(moment(&tied)).0 / tied.len() as f32
}

fn edge_nodes(
    r: &mut Rig,
    inp: &dyn RigInput,
    apertures: &[Aperture],
    trunk_z: (f32, f32),
    edge_spacing: f32,
    census: bool,
    shell_stops: bool,
    budget_body: bool,
) -> Vec<usize> {
    let model = inp.model();
    // Skinned and rigid pieces of the parts (glass excluded: it sits in its
    // own part's frame), their boundary vertices.
    let meshes: Vec<(usize, Part, bool)> = model
        .parts
        .iter()
        .enumerate()
        .filter(|(_, p)| !p.name.starts_with("glass_"))
        .filter_map(|(k, p)| {
            inp.part_of(&p.name)
                .map(|part| (k, part, super::gen3::rigid(&p.name)))
        })
        .collect();
    let cell = 0.01f32;
    let ck = |q: Vec3| {
        (
            (q.x / cell).floor() as i32,
            (q.y / cell).floor() as i32,
            (q.z / cell).floor() as i32,
        )
    };
    let mut grid: HashMap<(i32, i32, i32), Vec<(usize, usize)>> = HashMap::new();
    let mut bnd: Vec<Vec<usize>> = vec![Vec::new(); model.parts.len()];
    for &(k, _, _) in &meshes {
        bnd[k] = boundary_points(&model.parts[k]);
        for &v in &bnd[k] {
            grid.entry(ck(model.parts[k].pos[v]))
                .or_default()
                .push((k, v));
        }
    }
    let info = |k: usize| meshes.iter().find(|m| m.0 == k).map(|m| (m.1, m.2));
    // Seam points per part pair (lower part first): each side's point and mesh.
    type Pts = Vec<(Vec3, Vec3, usize, usize)>;
    let mut seams: std::collections::BTreeMap<(Part, Part), Pts> = Default::default();
    // An input that knows its shut lines (vehicle-gen: the welded skins'
    // shared boundary points, before the shut-line inset and hems) gives
    // them directly.
    if let Some(pairs) = inp.seam_points() {
        for (q, q2, k, k2) in pairs {
            let (Some((pa, _)), Some((pb, _))) = (info(k), info(k2)) else {
                continue;
            };
            if pa < pb {
                seams.entry((pa, pb)).or_default().push((q, q2, k, k2));
            }
        }
    }
    let derive = inp.seam_points().is_none();
    for &(k, pa, _) in meshes.iter().filter(|_| derive) {
        for &v in &bnd[k] {
            let q = model.parts[k].pos[v];
            let (x, y, z) = ck(q);
            let mut best: Option<(f32, usize, usize)> = None;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let Some(cands) = grid.get(&(x + dx, y + dy, z + dz)) else {
                            continue;
                        };
                        for &(k2, v2) in cands {
                            let Some((pb, _)) = info(k2) else { continue };
                            if pb == pa {
                                continue;
                            }
                            let d = q.distance(model.parts[k2].pos[v2]);
                            if d < cell && best.is_none_or(|b| d < b.0) {
                                best = Some((d, k2, v2));
                            }
                        }
                    }
                }
            }
            if let Some((_, k2, v2)) = best {
                let pb = info(k2).unwrap().0;
                if pa < pb {
                    seams
                        .entry((pa, pb))
                        .or_default()
                        .push((q, model.parts[k2].pos[v2], k, k2));
                }
            }
        }
    }
    // A side's node: in from the edge (toward the panel's vertices within
    // 8 cm, in its tangent plane) and under the skin.
    let place = |k: usize, q: Vec3| -> Option<Vec3> {
        let m = &model.parts[k];
        let (mut c, mut n, mut w) = (Vec3::ZERO, Vec3::ZERO, 0.0f32);
        for (p, nn) in m.pos.iter().zip(&m.nrm) {
            if p.distance(q) < 0.08 {
                c += *p;
                n += *nn;
                w += 1.0;
            }
        }
        let n = n.normalize_or_zero();
        if w < 3.0 || n == Vec3::ZERO {
            return None;
        }
        let d = c / w - q;
        let t = d - n * n.dot(d);
        if t.length() < 1e-4 {
            return None;
        }
        Some(q + t.normalize() * EDGE_IN - n * EDGE_UNDER)
    };
    let mut serial = 0usize;
    let mut made: Vec<(usize, Part)> = Vec::new();
    for (&(pa, pb), pts) in &seams {
        // Greedy samples along the line, in a fixed order.
        let mut pts = pts.clone();
        pts.sort_by(|a, b| {
            (a.0.z, a.0.y, a.0.x)
                .partial_cmp(&(b.0.z, b.0.y, b.0.x))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut picked: Pts = Vec::new();
        for s in &pts {
            if picked.iter().all(|p| p.0.distance(s.0) >= edge_spacing) {
                picked.push(*s);
            }
        }
        for (a, b, ka, kb) in picked {
            let mut pair = [None, None];
            for (side, (part, k, q)) in [(pa, ka, a), (pb, kb, b)].into_iter().enumerate() {
                // A rigid piece (a lamp) rides one frame: no edge node for it.
                if info(k).is_some_and(|m| m.1) {
                    continue;
                }
                let Some(g) = place(k, q) else { continue };
                // Census: a closure has no edge nodes of its own (a BeamNG
                // door's perimeter nodes are its structure); the stop goes to
                // its nearest skin node. The body's are structural nodes with
                // their share of its mass (no 0.15 kg nodes on the step).
                if census && part != Part::Body {
                    pair[side] = r
                        .nearest(part, g, 1, |j| r.skin[j] && !r.car.nodes[j].no_group)
                        .first()
                        .copied()
                        .filter(|&j| r.g(j).distance(g) < 0.35);
                    continue;
                }
                // A shell on its members' nodes takes the stop on the nearest
                // of them (the aperture's flange is part of that member), as
                // a BeamNG shell does: no light edge nodes hung on soft ties
                // (they rang at 6-20 Hz and never settled).
                // The same for an edge node its part could not hold in three
                // directions ([`EDGE_SPREAD`]): on a pillar every node in
                // reach lies in the pillar's plane, and a node tied in one
                // plane swings across it (two A-pillar edge nodes moved
                // 1.2 mm at rest and never stopped).
                let flat = budget_body
                    && part == Part::Body
                    && edge_tie_spread(r, g, &made.iter().map(|m| m.0).collect::<Vec<_>>())
                        < EDGE_SPREAD;
                if (shell_stops || flat) && part == Part::Body {
                    pair[side] = r
                        .nearest(part, g, 1, |j| {
                            r.member[j].is_some()
                                && !r.car.nodes[j].no_group
                                && !r.car.nodes[j].name.starts_with("hub")
                                && !r.car.nodes[j].name.starts_with("rim")
                        })
                        .first()
                        .copied()
                        .filter(|&j| r.g(j).distance(g) < 0.35);
                    continue;
                }
                // An edge node of the part already within half a spacing serves.
                if let Some(&(near, _)) = made
                    .iter()
                    .filter(|m| m.1 == part)
                    .min_by(|x, y| r.g(x.0).distance(g).total_cmp(&r.g(y.0).distance(g)))
                {
                    if r.g(near).distance(g) < 0.5 * edge_spacing {
                        pair[side] = Some(near);
                        continue;
                    }
                }
                if r.nearest(part, g, 1, |_| true)
                    .first()
                    .is_some_and(|&j| r.g(j).distance(g) < 0.03)
                {
                    continue;
                }
                serial += 1;
                let nm = format!("e{}{serial}", part.name().trim_start_matches("gen2_"));
                let i = r.node(&nm, g, part, Some(Role::Sheet), false);
                r.car.nodes[i].edge = !census;
                made.push((i, part));
                pair[side] = Some(i);
            }
            if let [Some(x), Some(y)] = pair {
                let part = if pa == Part::Body { pb } else { pa };
                if x != y
                    && !r.car.has_beam(x, y)
                    && r.g(x).distance(r.g(y)) > 0.03
                    && std::env::var("GEN3_NO_EDGESTOP").is_err()
                {
                    let j = r.car.beam(x, y, Role::EdgeStop, part);
                    r.car.beams[j].role = Role::EdgeStop;
                }
            }
        }
    }
    // Ties: each edge node to its part's nearest structural nodes (none
    // across an opening), an inner one among them, and its neighbours along
    // the edge.
    let edge: Vec<usize> = made.iter().map(|m| m.0).collect();
    for &(i, part) in &made {
        let g = r.g(i);
        let ok = |j: usize, r: &Rig| {
            j != i
                && !r.car.nodes[j].no_group
                && !r.car.nodes[j].name.starts_with("hub")
                && !r.car.nodes[j].name.starts_with("rim")
                && (part != Part::Body
                    || !segment_blocked(g, r.g(j), &|p| body_blocked(inp, p, apertures, trunk_z)))
        };
        let mut frame: Vec<usize> = r
            .nearest(part, g, 12, |j| !edge.contains(&j) && ok(j, r))
            .into_iter()
            .filter(|&j| r.g(j).distance(g) < 0.65)
            .collect();
        // At an opening's very edge (the trunk's, a door's) every segment
        // may graze the opening: the nearest frame nodes then, at most 0.4 m.
        if frame.len() < 4 {
            let more: Vec<usize> = r
                .nearest(part, g, 6, |j| {
                    !edge.contains(&j)
                        && j != i
                        && !r.car.nodes[j].no_group
                        && !r.car.nodes[j].name.starts_with("hub")
                        && !r.car.nodes[j].name.starts_with("rim")
                })
                .into_iter()
                .filter(|&j| r.g(j).distance(g) < 0.4 && !frame.contains(&j))
                .collect();
            frame.extend(more);
        }
        let mut tie = |r: &mut Rig, j: usize| {
            if !r.car.has_beam(i, j) && r.g(i).distance(r.g(j)) > 0.03 {
                r.car.beam(i, j, Role::EdgeTie, part);
            }
        };
        let mut tied: Vec<usize> = Vec::new();
        for &j in frame.iter().take(4) {
            tie(r, j);
            tied.push(j);
        }
        if let Some(&j) = frame.iter().find(|&&j| !r.skin[j] && !tied.contains(&j)) {
            tie(r, j);
            tied.push(j);
        }
        // Ties that span all three directions (no free mode out of their
        // plane): while their directions' second moment has a weak axis, the
        // frame node most along it joins them.
        for _ in 0..3 {
            let m = tied.iter().fold(glam::Mat3::ZERO, |m, &j| {
                let d = (r.g(j) - g).normalize_or_zero();
                m + glam::Mat3::from_cols(d * d.x, d * d.y, d * d.z)
            });
            let (lam, axis) = weakest_axis(m);
            if lam > 0.12 * tied.len() as f32 {
                break;
            }
            let best = frame
                .iter()
                .copied()
                .chain(r.nearest(part, g, 20, |j| {
                    !edge.contains(&j)
                        && j != i
                        && !r.car.nodes[j].no_group
                        && !r.car.nodes[j].name.starts_with("hub")
                        && !r.car.nodes[j].name.starts_with("rim")
                }))
                .filter(|j| !tied.contains(j) && r.g(*j).distance(g) < 0.65)
                .max_by(|&a, &b| {
                    let f = |j: usize| (r.g(j) - g).normalize_or_zero().dot(axis).abs();
                    f(a).total_cmp(&f(b))
                });
            match best {
                Some(j) => {
                    tie(r, j);
                    tied.push(j);
                }
                None => break,
            }
        }
        for j in r.nearest(part, g, 2, |j| edge.contains(&j) && ok(j, r)) {
            if r.g(j).distance(g) < 1.6 * edge_spacing {
                tie(r, j);
            }
        }
    }
    edge
}

/// The raycast wheels' mass on their hub nodes (half a wheel each).
pub fn hub_extra(c: &Car) -> HashMap<usize, f32> {
    let mut h = HashMap::new();
    let kg = c.raycast_wheel_kg.unwrap_or(WHEEL_KG);
    for w in &c.wheels {
        *h.entry(w.n1).or_insert(0.0) += 0.5 * kg;
        *h.entry(w.n2).or_insert(0.0) += 0.5 * kg;
    }
    h
}

/// Each structural node's share of its part's render surface (m²): every
/// mesh triangle's area, a third to each corner, to the nearest structural
/// node of the part the mesh rides (wheels' pieces excluded).
pub fn voronoi_area(inp: &dyn RigInput, car: &Car) -> Vec<f32> {
    let model = inp.model();
    let axle_f = inp.dims().axle_f;
    let mut area = vec![0.0f32; car.nodes.len()];
    let mut pools: HashMap<Part, Vec<usize>> = HashMap::new();
    for (i, n) in car.nodes.iter().enumerate() {
        if !n.contact && !n.no_group && !n.edge && !n.name.starts_with("hub") {
            pools.entry(n.part).or_default().push(i);
        }
    }
    for p in &model.parts {
        let Some(part) = inp.part_of(&p.name) else {
            continue;
        };
        let Some(pool) = pools.get(&part) else {
            continue;
        };
        let mut va = vec![0.0f32; p.pos.len()];
        for t in &p.tris {
            let [a, b, c] = t.map(|k| p.pos[k as usize]);
            let ar = 0.5 * (b - a).cross(c - a).length() / 3.0;
            for &k in t {
                va[k as usize] += ar;
            }
        }
        for (k, &a) in va.iter().enumerate() {
            if a == 0.0 {
                continue;
            }
            let q = Vec3::new(p.pos[k].x, axle_f - p.pos[k].z, p.pos[k].y);
            let best = pool
                .iter()
                .min_by(|&&x, &&y| {
                    car.nodes[x]
                        .p
                        .distance_squared(q)
                        .total_cmp(&car.nodes[y].p.distance_squared(q))
                })
                .copied();
            if let Some(b) = best {
                area[b] += a;
            }
        }
    }
    area
}
