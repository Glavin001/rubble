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

fn belt(z: f32) -> f32 {
    sf::with_profiles(|p| p.belt.at(z))
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

/// Evenly spaced values from `a` to `b` (both included), `n` of them.
fn span(a: f32, b: f32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| a + (b - a) * k as f32 / (n - 1).max(1) as f32)
        .collect()
}

/// The builder's car plus the rig's per-node member tags.
pub struct Rig {
    pub car: Car,
    /// Per node its member (the beam graph's role between two nodes of one member).
    pub member: Vec<Option<Role>>,
    /// Per node: an outer skin node (on the panel surface).
    pub skin: Vec<bool>,
}

impl Rig {
    fn node(&mut self, name: &str, g: Vec3, part: Part, member: Option<Role>, skin: bool) -> usize {
        let i = self.car.node(name, to_jbeam(g), part);
        self.member.push(member);
        self.skin.push(skin);
        i
    }
    fn g(&self, i: usize) -> Vec3 {
        to_gltf(self.car.p(i))
    }
    /// Nodes of `part` (structural) nearest a glTF point.
    fn nearest(&self, part: Part, g: Vec3, k: usize, filter: impl Fn(usize) -> bool) -> Vec<usize> {
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
fn body_blocked(p: Vec3, doors: &[Aperture], trunk_z: (f32, f32)) -> bool {
    let x = p.x.abs();
    if doors.iter().any(|d| d.inside(p, 0.055)) {
        return true;
    }
    // The cabin's volume (8 cm in from the skin: beams along the bonded
    // glass and the roof are the shell's, not across the cabin).
    let cabin =
        x < 0.62 && p.y > 0.46 && p.y < 1.28 && p.z > -1.72 && p.z < 0.78 && sf::sdf(p) < -0.08;
    let bay = x < 0.48 && p.y > 0.72 && p.z > 0.98 && p.z < 1.98;
    // The trunk opening (under the lid, from its mesh), clear of the shelf
    // and tail panel at its ends.
    let trunk = x < 0.58 && p.y > 0.86 && p.y < 1.06 && p.z > trunk_z.0 && p.z < trunk_z.1;
    cabin || bay || trunk
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
    let mut r = Rig {
        car: Car::default(),
        member: Vec::new(),
        skin: Vec::new(),
    };
    let body = Part::Body;

    // Part extents from the meshes.
    let door_f = Skin::of(model, "door_FL");
    let door_r = Skin::of(model, "door_RL");
    let hood_m = Skin::of(model, "hood");
    let trunk_m = Skin::of(model, "trunk_lid");
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
    let mut named_ids: HashMap<String, usize> = HashMap::new();
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
            let zf = (AXLE_F - ARCH_R - 0.08).max(hp_z + 0.12);
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
            put(
                &mut r,
                format!("bpi{k}{x}"),
                inward(o, 0.15),
                Some(Role::PillarB),
                false,
            );
        }
        // C-pillar: from the rear roof rail down to the quarter's belt.
        {
            let z = 0.5 * (-1.05 + z_c);
            let c = corner(z);
            let p = inward(
                sf::hit(Param::Section, z, sf::theta_corner(z) + 0.12),
                0.015,
            );
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
            let bottom = sf::arch_y(z)
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
            let sh = sf::shoulder_y(z);
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
            Vec3::new(0.60, 0.72, AXLE_R),
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
            Vec3::new(0.56, 0.80, AXLE_F),
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
    let apertures = [Aperture::of(&door_f.pos), Aperture::of(&door_r.pos)];
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
        REACH_BODY,
        body,
        Role::Web,
        &|p| body_blocked(p, &apertures, (tk_lo.z + 0.06, tk_hi.z - 0.06)),
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

    // ------------------------------------------------------------------
    // Engine: a 12-node block on mounts.
    // ------------------------------------------------------------------
    // The powertrain's faces where the 2011 Accord's are: 0.39 m ahead of
    // and 0.12 m behind the front axle (DOT HS 811 666 fig. 87). Round 3
    // had its rear face at the axle.
    let old_pt = std::env::var("GEN3_OLD_POWERTRAIN").is_ok();
    let pt_z: [f32; 3] = if old_pt {
        [1.40, 1.61, 1.82]
    } else {
        let (r0, f0) = (AXLE_F - ACCORD_PT_BEHIND, AXLE_F + ACCORD_PT_AHEAD);
        [r0, 0.5 * (r0 + f0), f0]
    };
    let eng = {
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
                    .unwrap_or(ENGINE_MOUNT);
                r.car.hardware(e, b, Role::Mount, body, &group, st);
            }
        }
    }
    // The dash panel behind the block (BOUNDED stops from its rear face to
    // the firewall's nodes): the Accord's dash line is 0.27-0.30 m behind
    // the front axle, so the block's rear face meets it after closing
    // 0.15-0.18 m, less what stays crushed between them (manifold shield
    // and hoses, POWERTRAIN_CLEARANCE).
    let dash_gap = 0.5 * (ACCORD_DASH_BEHIND[0] + ACCORD_DASH_BEHIND[1])
        - ACCORD_PT_BEHIND
        - POWERTRAIN_CLEARANCE;
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

    // ------------------------------------------------------------------
    // Closures.
    // ------------------------------------------------------------------
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
                let sh = sf::shoulder_y(z);
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
                // The inner frame: bottom and belt, 8 cm in, and the inner
                // panel's middle between them.
                for (row, &y) in [bottom + 0.02, bt - 0.03].iter().enumerate() {
                    let p = mirror(inward(side(z, y), 0.08), s);
                    inner.push(r.node(
                        &format!("{dn}_i{row}{k}"),
                        p,
                        part,
                        Some(Role::DoorFrame),
                        false,
                    ));
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
            let split_beam = std::env::var("GEN3_OLD_INTRUSION").is_err();
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
                .panel(&grid, part, &[Role::Sheet; 3], Role::Sheet, true);
            let mut below: Vec<usize> = grid.iter().flatten().flatten().copied().collect();
            below.extend(&inner);
            // Below the belt: everything to everything in reach.
            graph(&mut r, &below, REACH_PANEL, part, Role::DoorFrame, &|_| {
                false
            });
            // The frame: front and rear legs from the belt, and the top.
            let belt_row: Vec<usize> = grid[2].iter().flatten().copied().collect();
            let ib: Vec<usize> = inner
                .iter()
                .copied()
                .filter(|&i| r.car.nodes[i].name.contains("_i1"))
                .collect();
            let (bf, br) = (belt_row[0], *belt_row.last().unwrap());
            let (ibf, ibr) = (ib[0], *ib.last().unwrap());
            let (f0, f1) = (frame[0], frame[1]);
            let (fi0, fi1) = (frame_in[0], frame_in[1]);
            for (a, b) in [
                (bf, f0),
                (ibf, f0),
                (br, f1),
                (ibr, f1),
                (f0, f1),
                (belt_row[1], f0),
                (belt_row[1], f1),
                (f0, fi0),
                (f1, fi1),
                (fi0, fi1),
                (f0, fi1),
                (f1, fi0),
                (ibf, fi0),
                (ibr, fi1),
                (bf, fi0),
                (br, fi1),
            ] {
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
            let all: Vec<usize> = below
                .iter()
                .copied()
                .chain(frame.iter().copied())
                .chain(frame_in.iter().copied())
                .collect();
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
                if kc % 2 == 0 {
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
        all
    };

    // Trunk lid: deck rows and the rear face, 5 columns.
    let trunk_ids = {
        let part = Part::Trunk;
        let mut grid = Vec::new();
        let deck = span(tk_hi.z - 0.03, tk_lo.z + 0.07, 3);
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
        for (kr, &z) in [deck[0], deck[2]].iter().enumerate() {
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
            .panel(&grid, part, &[Role::Sheet; 4], Role::Sheet, true);
        let mut all: Vec<usize> = grid.iter().flatten().flatten().copied().collect();
        all.extend(&inner);
        graph(&mut r, &all, 0.85, part, Role::Sheet, &|_| false);
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
            if let Some(a) = sf::arch_y(z) {
                bottom = bottom.max(a + 0.025);
            }
            let sh = sf::shoulder_y(z);
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
        let all: Vec<usize> = grid.iter().flatten().flatten().copied().collect();
        graph(&mut r, &all, REACH_PANEL, part, Role::Sheet, &|_| false);
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
        let ys = [lo.y + 0.06, hi.y - 0.06];
        let side_z = if front { lo.z + 0.05 } else { hi.z - 0.05 };
        let mut grid: Vec<Vec<Option<usize>>> = vec![Vec::new(), Vec::new()];
        for (row, &y) in ys.iter().enumerate() {
            let mut cols: Vec<Vec3> = Vec::new();
            cols.push(mirror(side(side_z, y), -1.0));
            for f in [-0.80f32, -0.42, 0.0, 0.42, 0.80] {
                let p = end_pt(front, f.abs() * 0.92, y);
                cols.push(mirror(p, if f < 0.0 { -1.0 } else { 1.0 }));
            }
            cols.push(side(side_z, y));
            for (k, p) in cols.into_iter().enumerate() {
                grid[row].push(Some(r.node(
                    &format!("{tag}_o{row}{k}"),
                    p,
                    part,
                    Some(Role::Sheet),
                    true,
                )));
            }
        }
        r.car
            .panel(&grid, part, &[Role::Sheet; 2], Role::Sheet, true);
        let cover: Vec<usize> = grid.iter().flatten().flatten().copied().collect();
        // The beam: 0.13 m behind the face, five stations, two heights.
        let face = if front { hi.z } else { lo.z };
        let bz =
            |x: f32| face - (0.13 + 0.10 * (x / 0.75).powi(2)) * if front { 1.0 } else { -1.0 };
        let mut beam = Vec::new();
        for (k, &xx) in [-0.72f32, -0.40, 0.0, 0.40, 0.72].iter().enumerate() {
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
        graph(&mut r, &beam, 0.50, part, Role::BumperBeam, &|_| false);
        let mut all = cover.clone();
        all.extend(&beam);
        // Cover to beam: the foam (sheet role, soft).
        for &c in &cover {
            let g = r.g(c);
            for b in r.nearest(part, g, 3, |i| beam.contains(&i)) {
                r.car.beam(c, b, Role::Sheet, part);
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
        bumper_ids.insert(part, (cover, beam));
        let _ = all;
    }

    // ------------------------------------------------------------------
    // Crush cans and rear rails onto the bumper beams.
    // ------------------------------------------------------------------
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
    // solid (COOLING_STACK), as a frontal pulse's late peak is the engine
    // stopped against the barrier through the radiator support. BOUNDED
    // stops from the block's front nodes to their two nearest beam nodes.
    if !old_pt && std::env::var("GEN3_NO_STACK").is_err() {
        let stack = env_f("GEN3_COOLING_STACK", COOLING_STACK);
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
    let mut rims = Vec::new();
    for s in [1.0f32, -1.0] {
        let x = lr(s);
        for (w, za) in [("F", AXLE_F), ("R", AXLE_R)] {
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
                    s * TRACK_HALF,
                    TYRE_R + RIM_R * a.sin(),
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

    // ------------------------------------------------------------------
    // Hardware from the markers.
    // ------------------------------------------------------------------
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
                let bv = r.nearest(body, mk, 2, |i| pillar(i, &r));
                for &(_, d) in dv.iter().take(2) {
                    for &b in &bv {
                        r.car.hardware(
                            d,
                            b,
                            Role::Hinge,
                            part,
                            &format!("{dn}_hinge_{which}"),
                            40e3,
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
                ("couplerStrength".into(), serde_json::json!(50000)),
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
            for b in r.nearest(body, mk, 5, |i| pillar(i, &r)) {
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
        let bv = r.nearest(body, mk, 2, |i| {
            matches!(
                r.member[i],
                Some(Role::Apron) | Some(Role::Firewall) | Some(Role::PillarA)
            )
        });
        for &(_, d) in dv.iter().take(2) {
            for &b in &bv {
                r.car.hardware(
                    d,
                    b,
                    Role::Hinge,
                    Part::Hood,
                    &format!("hood_hinge_{x}"),
                    40e3,
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
        for b in r.nearest(Part::Body, at, 5, |i| r.member[i].is_some() && !r.skin[i]) {
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
                30000.0,
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
            45000.0,
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
        let bv = r.nearest(body, mk, 2, |_| true);
        for &(_, d) in dv.iter().take(2) {
            for &b in &bv {
                r.car.hardware(
                    d,
                    b,
                    Role::Hinge,
                    Part::Trunk,
                    &format!("trunk_hinge_{x}"),
                    25e3,
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
                        FENDER_BOLT,
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
                    FENDER_BOLT,
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
                        BUMPER_BOLT,
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
                    BUMPER_SIDE_BOLT,
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
                            BUMPER_SIDE_BOLT,
                        );
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Shut lines: structural edge nodes along both sides of every shut line.
    // ------------------------------------------------------------------
    if std::env::var("GEN3_NO_EDGES").is_err() {
        edge_nodes(&mut r, model, &apertures, (tk_lo.z + 0.06, tk_hi.z - 0.06));
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
        push_links(&mut r);
    }

    // ------------------------------------------------------------------
    // Fender bolt rows: a real fender's top flange is bolted down the apron
    // rail every 15-25 cm and its rear edge to the hinge pillar (two or
    // three bolts), so it moves with the apron and the pillar rather than
    // with its own skin. Each top-flange node (the fender's nodes on the
    // hood's shut line) is bolted to the nearest apron or cowl node; each
    // rear-edge node to the nearest hinge-pillar node. One M6 8.8 bolt each
    // ([`FENDER_ROW_BOLT`]).
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
                        FENDER_ROW_BOLT,
                    );
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Pole lanes: light contact-only nodes on the side-facing skin edges.
    // ------------------------------------------------------------------
    let surf = |p: Vec3| -> Vec3 { to_jbeam(sf::project(to_gltf(p))) };
    let side_edge = |a: Vec3, b: Vec3| {
        let m = to_gltf(0.5 * (a + b));
        // On the outer skin only (an edge behind another part's skin, such
        // as the sill's top under a door, would put lane nodes in that
        // part's contact band).
        sf::normal(sf::project(m)).x.abs() > 0.55 && sf::sdf(m) > -0.02
    };
    let lane_parts: Vec<Part> = if std::env::var("GEN3_NOLANES").is_ok() {
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
        nd.self_collision = (on_tri[i] || nd.part == Part::Engine || nd.name.starts_with("rim"))
            && !nd.contact
            && std::env::var("GEN3_NOSELF").is_err()
            && std::env::var("GEN3_NOSELF_PART").map_or(true, |p| !nd.name.contains(&p));
    }

    // ------------------------------------------------------------------
    // Masses (JBeam frame positions, the 2011 Accord's).
    // ------------------------------------------------------------------
    let hpf = marker(model, "seat_H_point_FL");
    let hpr = marker(model, "seat_H_point_FR");
    let j = to_jbeam;
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
        for (za, kg) in [(AXLE_F, UNSPRUNG_KG + 6.0), (AXLE_R, UNSPRUNG_KG - 6.0)] {
            r.car.named.push(Named {
                name: "unsprung",
                kg,
                at: j(Vec3::new(TRACK_HALF * s, TYRE_R, za)),
                radius: 0.15,
                part: body,
            });
        }
    }
    let fixed: f32 = BUMPER_F_KG
        + BUMPER_R_KG
        + 2.0 * (ACCORD_DOOR_F + ACCORD_DOOR_R + ACCORD_FENDER)
        + ACCORD_HOOD
        + ACCORD_DECKLID
        + ACCORD_ENGINE_KG
        + ACCORD_TRANSAXLE_KG
        + ACCORD_BIW
        + 4.0 * RIM_NODES as f32 * RIM_NODE_KG
        + r.car.named.iter().map(|m| m.kg).sum::<f32>()
        + 4.0 * WHEEL_KG;
    let module = KERB - fixed;
    assert!(module >= 0.0, "masses exceed the kerb by {} kg", -module);
    r.car.named.push(Named {
        name: "front_end_module",
        kg: module,
        at: j(Vec3::new(0.0, 0.62, 1.80)),
        radius: 0.60,
        part: body,
    });
    r.car.part_mass = [
        (Part::Body, ACCORD_BIW),
        (Part::Engine, ACCORD_ENGINE_KG + ACCORD_TRANSAXLE_KG),
        (Part::DoorFL, ACCORD_DOOR_F),
        (Part::DoorFR, ACCORD_DOOR_F),
        (Part::DoorRL, ACCORD_DOOR_R),
        (Part::DoorRR, ACCORD_DOOR_R),
        (Part::Hood, ACCORD_HOOD),
        (Part::Trunk, ACCORD_DECKLID),
        (Part::FenderL, ACCORD_FENDER),
        (Part::FenderR, ACCORD_FENDER),
        (Part::BumperF, BUMPER_F_KG),
        (Part::BumperR, BUMPER_R_KG),
    ]
    .into_iter()
    .collect();
    r.car.sheet_area = Some(voronoi_area(model, &r.car));
    r.car.mass_floor = Some(0.7);
    r.car.row_cap_contact_only = false;
    r.car.centred_damping = true;
    r.car.assign_masses();

    // The wheel stack (round 5): a front wheel pushed back meets the hinge
    // pillar's base once its tyre is crushed onto the rim (WHEEL_STACK from
    // the hub's centre), and loads the sill and A-pillar. BOUNDED stops from
    // both hub nodes to the pillar's base nodes.
    if std::env::var("GEN3_NO_WHEEL_STACK").is_err() {
        let stack = env_f("GEN3_WHEEL_STACK", WHEEL_STACK);
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
    let hub = hub_extra(&r.car);
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
        env("GEN3_FLOOR_Y", FOOTWELL_YIELD),
        env("GEN3_ROOF_Y", GREENHOUSE_YIELD),
        env("GEN3_DOOR_Y", 1.0),
    );
    let (sy, cy) = (env("GEN3_SIDE_Y", 1.0), env("GEN3_CRUSH_Y", CRUSH_YIELD));
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
    let nose_y = r
        .car
        .nodes
        .iter()
        .map(|n| n.p.y)
        .fold(f32::MAX, f32::min);
    let (gf, gr) = (
        env("GEN3_GRADE_FRONT", CRUSH_GRADE[0]),
        env("GEN3_GRADE_REAR", CRUSH_GRADE[1]),
    );
    let front_grade = |y: f32| {
        let t = ((fw_y - y) / (fw_y - nose_y).max(1e-3)).clamp(0.0, 1.0);
        gr + (gf - gr) * t
    };
    for b in &mut r.car.beams {
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
        if b.part.is_door() {
            b.deform *= dy;
            if mid.z > door_up_h {
                b.deform *= duy;
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
            b.deform *= if tower(b.a) || tower(b.b) {
                cy.max(env("GEN3_TOWER_Y", TOWER_YIELD))
            } else if mid.y < fw_y - 0.05 {
                cy * front_grade(mid.y)
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
    let iy = env("GEN3_INTRUSION_Y", INTRUSION_YIELD);
    for b in &mut r.car.beams {
        if b.role == Role::Intrusion && b.deform.is_finite() {
            b.deform *= iy;
        }
    }
    // The A-pillar's chords (hinge pillar top to the header, outer and inner
    // walls): a hot-stamped section, its yield `GEN3_APILLAR_Y` times the
    // greenhouse's.
    let ay = env("GEN3_APILLAR_Y", APILLAR_YIELD);
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
            env("GEN3_CM_Y", CROSS_MEMBER_YIELD),
            env("GEN3_PAN_Y", FLOOR_PAN_YIELD),
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
            env("GEN3_BP_UP", BPILLAR_UPPER_YIELD),
            env("GEN3_BP_LO", BPILLAR_LOWER_YIELD),
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
    let _ = (ARCH_R, &eng);
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
fn push_links(r: &mut Rig) {
    let pairs = [
        (Part::BumperF, Part::Body),
        (Part::BumperF, Part::FenderL),
        (Part::BumperF, Part::FenderR),
        (Part::BumperF, Part::Hood),
        (Part::FenderL, Part::Body),
        (Part::FenderR, Part::Body),
        (Part::Hood, Part::Body),
    ];
    let axle = AXLE_F;
    let reach = env_f("GEN3_PUSH_REACH", PUSH_REACH);
    let cone = (35f32).to_radians().tan();
    for (pa, pb) in pairs {
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

fn edge_nodes(
    r: &mut Rig,
    model: &Model,
    apertures: &[Aperture],
    trunk_z: (f32, f32),
) -> Vec<usize> {
    // Skinned and rigid pieces of the parts (glass excluded: it sits in its
    // own part's frame), their boundary vertices.
    let meshes: Vec<(usize, Part, bool)> = model
        .parts
        .iter()
        .enumerate()
        .filter(|(_, p)| !p.name.starts_with("glass_"))
        .filter_map(|(k, p)| {
            super::gen3::part_of(&p.name).map(|part| (k, part, super::gen3::rigid(&p.name)))
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
    for &(k, pa, _) in &meshes {
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
            if picked.iter().all(|p| p.0.distance(s.0) >= EDGE_SPACING) {
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
                // An edge node of the part already within half a spacing serves.
                if let Some(&(near, _)) = made
                    .iter()
                    .filter(|m| m.1 == part)
                    .min_by(|x, y| r.g(x.0).distance(g).total_cmp(&r.g(y.0).distance(g)))
                {
                    if r.g(near).distance(g) < 0.5 * EDGE_SPACING {
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
                r.car.nodes[i].edge = true;
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
                    || !segment_blocked(g, r.g(j), &|p| body_blocked(p, apertures, trunk_z)))
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
            if r.g(j).distance(g) < 1.6 * EDGE_SPACING {
                tie(r, j);
            }
        }
    }
    edge
}

/// The raycast wheels' mass on their hub nodes (half a wheel each).
pub fn hub_extra(c: &Car) -> HashMap<usize, f32> {
    let mut h = HashMap::new();
    for w in &c.wheels {
        *h.entry(w.n1).or_insert(0.0) += 0.5 * WHEEL_KG;
        *h.entry(w.n2).or_insert(0.0) += 0.5 * WHEEL_KG;
    }
    h
}

/// Each structural node's share of its part's render surface (m²): every
/// mesh triangle's area, a third to each corner, to the nearest structural
/// node of the part the mesh rides (wheels' pieces excluded).
pub fn voronoi_area(model: &Model, car: &Car) -> Vec<f32> {
    let mut area = vec![0.0f32; car.nodes.len()];
    let mut pools: HashMap<Part, Vec<usize>> = HashMap::new();
    for (i, n) in car.nodes.iter().enumerate() {
        if !n.contact && !n.no_group && !n.edge && !n.name.starts_with("hub") {
            pools.entry(n.part).or_default().push(i);
        }
    }
    for p in &model.parts {
        let Some(part) = super::gen3::part_of(&p.name) else {
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
            let q = to_jbeam(p.pos[k]);
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
