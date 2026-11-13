//! The mechanical and interior parts: each defined once.
//!
//! One definition per part (`engine/docs/beamng-structure-study.md` §10.5):
//! its nodes with their masses, its internal beams, its mounts (named break
//! groups with spring, damping, yield and break force) and the beam-only
//! members that hang on it (arms, struts, tie rods, shafts), and the closed
//! solid drawn through those same points. The rigger turns the definition
//! into JBeam; nothing here is invented again downstream.
//!
//! Layout: a longitudinal engine on the front subframe, the transmission in
//! the tunnel's mouth, a driveshaft down the tunnel to a differential on the
//! rear subframe. MacPherson struts in front (the strut and the lower arm
//! hold the hub; the tie rod steers it), an upper and a lower arm at the
//! rear. The wheels stay raycast wheels: a wheel part is two axle nodes and
//! three frame nodes that carry the rim's and tyre's mass and meshes.

use glam::DVec3;

use crate::frame::Frame;
use crate::mesh::{box_, closed_lathe, rbox, rot_x, Mat, Mesh};
use crate::structure::{box_along, box_stations, cyl_along, Structure};

pub(crate) const INF: f64 = f64::INFINITY;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BeamKind {
    Normal,
    /// a rubber mount: free travel, then stiff
    Bounded,
    /// pushes only: a stop or a rest
    Support,
}

/// Beam values: spring N/m (None: from the node budget, Σk/m), damping
/// N·s/m, yield force N, break force N (infinite: does not break).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vals {
    pub k: Option<f64>,
    pub c: f64,
    pub deform: f64,
    pub strength: f64,
}

/// `k` 0: from the node budget.
pub fn vals(k: f64, c: f64, deform: f64, strength: f64) -> Vals {
    Vals { k: if k > 0.0 { Some(k) } else { None }, c, deform, strength }
}

#[derive(Clone, Debug)]
pub struct PartNode {
    pub name: String,
    pub pos: DVec3,
    pub mass: f64,
    pub collide: bool,
}

/// Beams inside a part, between its own nodes (indices).
#[derive(Clone, Debug)]
pub struct Internal {
    pub pairs: Vec<(usize, usize)>,
    pub v: Vals,
    /// (deform group, trigger ratio): the part reports damage (a seized
    /// engine, a leaking radiator)
    pub deform_group: Option<(String, f64)>,
}

/// A mount: `beams` beams between the nearest pairs of `from` (the part's
/// nodes) and `to` (nodes of the shell or of another part), in one named
/// break group.
#[derive(Clone, Debug)]
pub struct Mount {
    pub group: String,
    pub from: Vec<String>,
    pub to: Vec<String>,
    pub beams: usize,
    pub kind: BeamKind,
    pub v: Vals,
    /// rest length over spawn length (SUPPORT rests: 0.85–0.95)
    pub precomp: f64,
    /// The far ends are nodes of this body panel's own lattice (a door, the
    /// hood, the boot lid, a bumper cover: the rigger's nodes for it, which
    /// have no names here), the nearest ones to `from`; `to` is then empty.
    /// A hinge or a latch is a part mounted to the shell and to its panel.
    pub to_part: Option<String>,
}

/// A member that is one beam between two nodes of other parts (an arm's
/// leg, a strut, a tie rod, a shaft).
#[derive(Clone, Debug)]
pub struct Link {
    pub name: String,
    pub a: String,
    pub b: String,
    pub v: Vals,
    pub group: Option<String>,
}

#[derive(Clone, Debug)]
pub struct MechPart {
    /// also the render part of its own solid (when `mesh` is not empty)
    pub name: String,
    /// the study's class: bumper, suspension, wheel, engine, driveline,
    /// exhaust, fuel, cooling, interior
    pub class: &'static str,
    pub mesh: Mesh,
    pub nodes: Vec<PartNode>,
    pub internal: Vec<Internal>,
    pub mounts: Vec<Mount>,
    pub links: Vec<Link>,
    /// nodes of other parts its solid binds to as well (an arm has no nodes
    /// of its own: its solid follows its joints)
    pub binds: Vec<String>,
    /// other render parts that ride this part (a wheel's rim and tyre)
    pub meshes: Vec<String>,
    /// The body panel this part is fixed inside or onto (a door's intrusion
    /// beam, a number plate): it has no nodes, its solid follows that
    /// panel's own nodes.
    pub host: Option<String>,
    /// Mass (kg) of a part without nodes: carried by its host panel's or its
    /// bound part's listed mass (a caliper on its hub), not added to the car.
    pub carried_kg: f64,
    /// Its mass was part of the shell's fixed-trim allowance before it was a
    /// part ([`crate::structure::Structure::take_trim`]): the study's parts
    /// (§10.5) are not, the equipment and trim of [`crate::equipment`] are.
    pub from_trim: bool,
}

impl MechPart {
    pub fn mass(&self) -> f64 {
        self.nodes.iter().map(|n| n.mass).sum()
    }
    pub(crate) fn new(name: &str, class: &'static str) -> MechPart {
        MechPart { name: name.into(), class, mesh: Mesh::new(), nodes: Vec::new(), internal: Vec::new(), mounts: Vec::new(), links: Vec::new(), binds: Vec::new(), meshes: Vec::new(), host: None, carried_kg: 0.0, from_trim: false }
    }
    pub(crate) fn node(&mut self, tag: &str, pos: DVec3, mass: f64) -> String {
        let name = format!("{}_{tag}", self.prefix());
        self.nodes.push(PartNode { name: name.clone(), pos, mass, collide: true });
        name
    }
    /// Node names are short: the part's initials and its corner.
    pub(crate) fn prefix(&self) -> String {
        let mut it = self.name.split('_');
        let head: String = it.next().unwrap().chars().take(3).collect();
        let rest: String = it.map(|w| if w.len() <= 2 { w.to_lowercase() } else { w.chars().take(1).collect() }).collect();
        format!("{head}{rest}")
    }
    pub(crate) fn names(&self) -> Vec<String> {
        self.nodes.iter().map(|n| n.name.clone()).collect()
    }
    /// Every pair of the part's nodes.
    pub(crate) fn complete(&self) -> Vec<(usize, usize)> {
        let n = self.nodes.len();
        (0..n).flat_map(|i| (i + 1..n).map(move |j| (i, j))).collect()
    }
}

/// The running gear's dimensions (metres, as built).
#[derive(Clone)]
pub struct Gear {
    pub f: Frame,
    pub half_track: f64,
    pub wheel_y: f64,
    /// axle z, front and rear (as built)
    pub zf: f64,
    pub zr: f64,
    pub tyre_d: f64,
    pub tyre_w: f64,
    pub rim_d: f64,
    pub rim_w: f64,
    /// reference axle z (before the frame's mapping)
    pub axle_f: f64,
    pub axle_r: f64,
    /// the tail and nose faces (as built): z at (x, y)
    pub z_nose: std::rc::Rc<dyn Fn(f64, f64) -> f64>,
    pub z_tail: std::rc::Rc<dyn Fn(f64, f64) -> f64>,
    /// the body's surface
    pub tube: std::rc::Rc<crate::tube::Tube>,
}

pub(crate) fn cyl(a: DVec3, b: DVec3, r: f64) -> Mesh {
    cyl_along(&[a, b], r, 10, Mat::MetalBare)
}

pub(crate) fn bar(a: DVec3, b: DVec3, w: f64, h: f64) -> Mesh {
    let up = if (b - a).normalize().dot(DVec3::Y).abs() > 0.9 { DVec3::Z } else { DVec3::Y };
    box_along(&[a, b], up, w, h)
}

pub(crate) fn sh(st: &Structure, name: &str) -> DVec3 {
    st.node(name).unwrap_or_else(|| panic!("no shell node {name}")).pos
}

const CORNERS: [(&str, f64, bool); 4] = [("FL", 1.0, true), ("FR", -1.0, true), ("RL", 1.0, false), ("RR", -1.0, false)];

/// The sedan's mechanical and interior parts on the body-in-white `st`.
pub fn sedan_parts(g: &Gear, st: &Structure) -> Vec<MechPart> {
    let f = &g.f;
    let r = |x: f64, y: f64, z: f64| f.p(x, y, z);
    let mut out: Vec<MechPart> = Vec::new();
    let (t, wy) = (g.half_track, g.wheel_y);

    // ---- bumper beams: 5 across × 2 heights
    for (front, name, kg, yb, h, dz, dep, strength) in [(true, "impact_beam_front", 8.0, 0.45, 0.110, -0.105, 20e3, 60e3), (false, "impact_beam_rear", 6.0, 0.414, 0.090, 0.105, 15e3, 40e3)] {
        let mut p = MechPart::new(name, "bumper");
        let face = if front { &g.z_nose } else { &g.z_tail };
        let zc = |x: f64| face(f.x(x.abs()), f.y(yb)) + dz;
        let xs = [-0.62, -0.31, 0.0, 0.31, 0.62];
        let path: Vec<DVec3> = (0..=16).map(|i| -0.64 + 1.28 * i as f64 / 16.0).map(|x| DVec3::new(f.x(x), f.y(yb), zc(x))).collect();
        p.mesh = box_along(&path, DVec3::Y, 0.050, h);
        for (i, x) in xs.iter().enumerate() {
            for (tg, dy) in [("u", h / 2.0), ("d", -h / 2.0)] {
                p.node(&format!("{}{tg}", i + 1), DVec3::new(f.x(*x), f.y(yb) + dy, zc(*x)), kg / 10.0);
            }
        }
        // each rectangle with both diagonals, and a skip beam along each row
        let mut pairs = Vec::new();
        for i in 0..5 {
            pairs.push((2 * i, 2 * i + 1));
            if i < 4 {
                pairs.extend([(2 * i, 2 * i + 2), (2 * i + 1, 2 * i + 3), (2 * i, 2 * i + 3), (2 * i + 1, 2 * i + 2)]);
            }
            if i < 3 {
                pairs.extend([(2 * i, 2 * i + 4), (2 * i + 1, 2 * i + 5)]);
            }
        }
        p.internal.push(Internal { pairs, v: vals(0.0, 100.0, 8e3, INF), deform_group: None });
        for (s, lo) in [("l", 3), ("r", 0)] {
            let from: Vec<String> = (lo..lo + 2).flat_map(|i| [p.nodes[2 * i].name.clone(), p.nodes[2 * i + 1].name.clone()]).collect();
            let to = if front { vec![format!("sh_a1_u{s}"), format!("sh_a1_d{s}")] } else { vec![format!("sh_e1_3u{s}"), format!("sh_e1_3d{s}")] };
            p.mounts.push(Mount { group: format!("{name}_{s}"), from, to, beams: 8, kind: BeamKind::Normal, v: vals(1e6, 100.0, dep, strength), precomp: 1.0, to_part: None });
        }
        out.push(p);
    }

    // ---- front subframe: 4 corners, 2 rack mounts, a centre node
    let (sx, sy) = (0.36, 0.25);
    let (zs_f, zs_r, z_rack) = (g.axle_f + 0.225, g.axle_f - 0.225, g.axle_f + 0.135);
    let tube = |a: DVec3, b: DVec3| bar(a, b, 0.050, 0.050);
    {
        let mut p = MechPart::new("subframe_front", "suspension");
        for (tg, x, z) in [("fl", sx, zs_f), ("fr", -sx, zs_f), ("rl", sx, zs_r), ("rr", -sx, zs_r)] {
            p.node(tg, r(x, sy, z), 7.0);
        }
        p.node("kl", r(0.25, sy, z_rack), 7.0);
        p.node("kr", r(-0.25, sy, z_rack), 7.0);
        p.node("c", r(0.0, sy, zs_r), 4.5);
        for (a, b) in [((sx + 0.025, zs_f), (-sx - 0.025, zs_f)), ((sx + 0.025, zs_r), (-sx - 0.025, zs_r)), ((sx - 0.03, z_rack), (-sx + 0.03, z_rack))] {
            p.mesh.add(&tube(r(a.0, sy, a.1), r(b.0, sy, b.1)));
        }
        for s in [1.0, -1.0] {
            p.mesh.add(&tube(r(s * sx, sy, zs_f - 0.03), r(s * sx, sy, z_rack + 0.03)));
            p.mesh.add(&tube(r(s * sx, sy, z_rack - 0.03), r(s * sx, sy, zs_r + 0.03)));
        }
        p.internal.push(Internal { pairs: p.complete(), v: vals(5.5e6, 150.0, 88e3, INF), deform_group: None });
        for (s, sg) in [("l", 1.0), ("r", -1.0)] {
            let from: Vec<String> = p.nodes.iter().filter(|n| n.pos.x * sg > -0.01).map(|n| n.name.clone()).collect();
            let to = vec![format!("sh_a4_{s}"), format!("sh_a5_{s}"), format!("sh_a2_d{s}"), format!("sh_a2_u{s}"), format!("sh_a7_{s}"), format!("sh_c1_1t{s}"), format!("sh_b1_i{s}"), format!("sh_a3_dc")];
            p.mounts.push(Mount { group: format!("subframe_front_{s}"), from, to, beams: 16, kind: BeamKind::Normal, v: vals(1.5e6, 180.0, 157e3, 400e3), precomp: 1.0, to_part: None });
        }
        out.push(p);
    }
    // ---- rear subframe: 4 corners, 2 centre nodes
    let (qy, zq_f, zq_r) = (0.20, g.axle_r + 0.215, g.axle_r - 0.205);
    {
        let mut p = MechPart::new("subframe_rear", "suspension");
        for (tg, x, z) in [("fl", sx, zq_f), ("fr", -sx, zq_f), ("rl", sx, zq_r), ("rr", -sx, zq_r), ("cf", 0.0, zq_f), ("cr", 0.0, zq_r)] {
            p.node(tg, r(x, qy, z), 7.0);
        }
        for z in [zq_f, zq_r] {
            p.mesh.add(&tube(r(sx + 0.025, qy, z), r(-sx - 0.025, qy, z)));
        }
        for s in [1.0, -1.0] {
            p.mesh.add(&tube(r(s * sx, qy, zq_f - 0.03), r(s * sx, qy, zq_r + 0.03)));
        }
        p.internal.push(Internal { pairs: p.complete(), v: vals(5.5e6, 150.0, 88e3, INF), deform_group: None });
        for (s, sg) in [("l", 1.0), ("r", -1.0)] {
            for (e, k, zc) in [("f", 1, zq_f), ("r", 2, zq_r)] {
                let from: Vec<String> = p.nodes.iter().filter(|n| n.pos.x * sg > -0.01 && (n.pos.z - f.z(zc)).abs() < 0.05).map(|n| n.name.clone()).collect();
                let to = vec![format!("sh_e1_{k}d{s}"), format!("sh_e1_{k}u{s}"), format!("sh_e1_{}d{s}", 3 - k), format!("sh_e3_{k}"), format!("sh_e2_{s}")];
                p.mounts.push(Mount { group: format!("subframe_rear_{e}{s}"), from, to, beams: 10, kind: BeamKind::Normal, v: vals(2.1e6, 500.0, 22e3, 150e3), precomp: 1.0, to_part: None });
            }
        }
        out.push(p);
    }

    // ---- steering rack on the front subframe
    let y_rack = sy + 0.055;
    {
        let mut p = MechPart::new("steering_rack", "suspension");
        p.node("l", r(0.28, y_rack, z_rack), 5.0);
        p.node("r", r(-0.28, y_rack, z_rack), 5.0);
        p.mesh = cyl(r(-0.30, y_rack, z_rack), r(0.30, y_rack, z_rack), 0.024);
        p.internal.push(Internal { pairs: vec![(0, 1)], v: vals(0.0, 100.0, 85e3, INF), deform_group: None });
        p.mounts.push(Mount { group: "steering_rack".into(), from: p.names(), to: vec!["subf_kl".into(), "subf_kr".into()], beams: 4, kind: BeamKind::Normal, v: vals(0.0, 100.0, 85e3, INF), precomp: 1.0, to_part: None });
        out.push(p);
    }

    // ---- per wheel: hub, arms, strut, toe link, wheel, hub cap
    for (cn, sg, front) in CORNERS {
        let s = if sg > 0.0 { "l" } else { "r" };
        let c = cn.to_lowercase();
        let z0 = if front { g.zf } else { g.zr };
        let fwd = if front { 1.0 } else { -1.0 };
        let at = |dx: f64, dy: f64, dz: f64| DVec3::new(sg * (t + dx), wy + dy, z0 + dz);
        let (lo, up, toe, n4) = (at(-0.16, -0.14, 0.0), at(-0.17, 0.14, 0.0), at(-0.15, -0.035, fwd * 0.135), at(-0.14, 0.02, -fwd * 0.12));
        let mut hub = MechPart::new(&format!("hub_{cn}"), "suspension");
        let kg = if front { 17.0 } else { 24.0 };
        let (nlo, nup, ntoe, nn4) = (hub.node("lo", lo, kg / 4.0), hub.node("up", up, kg / 4.0), hub.node("toe", toe, kg / 4.0), hub.node("n4", n4, kg / 4.0));
        hub.internal.push(Internal { pairs: hub.complete(), v: vals(6e6, 100.0, 129e3, INF), deform_group: None });
        // the knuckle: an upright with a steering arm and the caliper's bracket
        hub.mesh.add(&bar(lo, up, 0.045, 0.060));
        hub.mesh.add(&bar(at(-0.165, -0.035, fwd * 0.034), toe + DVec3::Z * fwd * 0.012, 0.030, 0.025));
        hub.mesh.add(&bar(at(-0.165, 0.02, -fwd * 0.034), n4 - DVec3::Z * fwd * 0.012, 0.030, 0.030));
        // the wheel bearing and its flange, through the disc's hat
        hub.mesh.add(&crate::equipment::drum(at(-0.143, 0.0, 0.0), at(-0.004, 0.0, 0.0), 0.036, 14, Mat::MetalBare));
        hub.meshes.push(format!("brake_disc_{cn}"));
        // lower arm: two legs from the subframe's corners to the lower joint
        let sub = if front { "subf" } else { "subr" };
        let arm_v = vals(1.2e7, 2000.0, 155e3, 300e3);
        let mut arm = MechPart::new(&format!("arm_lower_{cn}"), "suspension");
        let pos_of = |n: &str| out.iter().flat_map(|p| p.nodes.iter()).find(|x| x.name == n).unwrap_or_else(|| panic!("no node {n}")).pos;
        for e in ["f", "r"] {
            let a = format!("{sub}_{e}{s}");
            let pa = pos_of(&a);
            let d = (lo - pa).normalize();
            arm.mesh.add(&bar(pa + d * 0.075, lo - d * 0.065, 0.040, 0.025));
            arm.links.push(Link { name: format!("arm_lower_{c}_{e}"), a: a.clone(), b: nlo.clone(), v: arm_v, group: Some(format!("arm_lower_{c}")) });
            arm.binds.push(a);
        }
        arm.binds.extend([nlo.clone(), nup.clone()]);
        // strut: the tube stands on the knuckle's top; its beam carries the
        // lower joint (front: with its clamp pair). At the rear, the upper
        // arm from the rail
        let tower = if front { format!("sh_a6_{s}") } else { format!("sh_e2_{s}") };
        let pt = sh(st, &tower);
        let mut strut = MechPart::new(&format!("strut_{cn}"), "suspension");
        let foot = up;
        // (the tube stops under the wheel-well liner; the tower is above it)
        let dir = (pt - foot).normalize();
        let len = ((wy + 0.375 + (g.tyre_d - 0.683) / 2.0 - 0.045 - foot.y) / dir.y).min((pt - foot).length() - 0.02);
        strut.mesh = cyl(foot + dir * 0.03, foot + dir * len, 0.032);
        // the coil spring round it, between its seats (inboard of the tyre)
        strut.mesh.add(&crate::equipment::coil_spring(foot + dir * 0.075, foot + dir * (len - 0.075), 0.040, 0.005, 4.0));
        strut.links.push(Link { name: format!("strut_{c}"), a: nlo.clone(), b: tower.clone(), v: vals(0.0, 200.0, 17e3, 180e3), group: Some(format!("strut_{c}")) });
        strut.binds.extend([tower.clone(), nlo.clone(), nup.clone(), nn4.clone()]);
        let mut upper = MechPart::new(&format!("arm_upper_{cn}"), "suspension");
        if front {
            // (a MacPherson strut is clamped to the knuckle at two points)
            for (e, n) in [("a", &nup), ("b", &nn4)] {
                strut.links.push(Link { name: format!("strut_clamp_{c}_{e}"), a: tower.clone(), b: n.clone(), v: arm_v, group: Some(format!("strut_clamp_{c}")) });
            }
        } else {
            for (e, k) in [("f", 1), ("r", 2)] {
                let a = format!("sh_e1_{k}u{s}");
                let pa = sh(st, &a);
                // (from the rail's outer face, clear of the floor above it)
                let p0 = pa + DVec3::new(sg * 0.064, -0.035, 0.0);
                let d = (up - p0).normalize();
                upper.mesh.add(&bar(p0, up - d * 0.068, 0.036, 0.022));
                upper.links.push(Link { name: format!("arm_upper_{c}_{e}"), a: a.clone(), b: nup.clone(), v: arm_v, group: Some(format!("arm_upper_{c}")) });
                upper.binds.push(a);
            }
            upper.binds.extend([nup.clone(), nlo.clone()]);
        }
        // toe link: the rack's end (front) or the subframe's centre (rear)
        let inner = if front { format!("ster_{s}") } else { format!("subr_r{s}") };
        let pi = pos_of(&inner);
        let mut tie = MechPart::new(&format!("toe_link_{cn}"), "suspension");
        let a0 = if front { pi + DVec3::X * sg * 0.03 } else { pi + (toe - pi).normalize() * 0.075 + DVec3::Y * 0.03 };
        tie.mesh = cyl(a0, toe - (toe - a0).normalize() * 0.03, 0.012);
        tie.links.push(Link { name: format!("toe_link_{c}"), a: inner.clone(), b: ntoe.clone(), v: vals(1.7e7, 100.0, 85e3, 250e3), group: Some(format!("toe_link_{c}")) });
        tie.binds.extend([inner, ntoe.clone(), nlo.clone(), nup.clone()]);
        // wheel: two axle nodes on the spin axis and three frame nodes
        let mut wh = MechPart::new(&format!("wheel_{cn}"), "wheel");
        wh.node("ai", at(-0.125, 0.0, 0.0), 5.0);
        wh.node("ao", at(0.125, 0.0, 0.0), 5.0);
        let rr = 0.42 * g.rim_d;
        for k in 0..3 {
            let a = std::f64::consts::TAU * (k as f64 / 3.0 + 0.25);
            wh.node(&format!("f{}", k + 1), at(0.0, rr * a.sin(), rr * a.cos()), 3.0);
        }
        wh.internal.push(Internal { pairs: wh.complete(), v: vals(0.0, 100.0, 168e3, INF), deform_group: None });
        wh.mounts.push(Mount { group: format!("wheel_{c}"), from: vec![wh.nodes[0].name.clone(), wh.nodes[1].name.clone()], to: hub.names(), beams: 8, kind: BeamKind::Normal, v: vals(6e6, 100.0, 168e3, 168e3), precomp: 1.0, to_part: None });
        wh.meshes.extend([format!("rim_{cn}"), format!("tyre_{cn}")]);
        // hub cap: a pressed disc clipped into the rim's centre
        let mut cap = MechPart::new(&format!("hub_cap_{cn}"), "wheel");
        let xc = 0.5 * g.rim_w - 0.012;
        for k in 0..3 {
            let a = std::f64::consts::TAU * (k as f64 / 3.0 + 0.25);
            cap.node(&format!("{}", k + 1), at(xc, 0.055 * a.sin(), 0.055 * a.cos()), 0.2);
        }
        let mut disc = closed_lathe(&[(0.0, 0.0), (0.068, 0.0), (0.068, 0.006), (0.040, 0.014), (0.0, 0.016)], 24, Mat::Chrome);
        if sg < 0.0 {
            disc.mirror_x();
        }
        disc.transform(None, at(xc, 0.0, 0.0));
        cap.mesh = disc;
        cap.internal.push(Internal { pairs: cap.complete(), v: vals(0.0, 20.0, 300.0, INF), deform_group: None });
        cap.mounts.push(Mount { group: format!("hub_cap_{c}"), from: cap.names(), to: wh.names(), beams: 6, kind: BeamKind::Normal, v: vals(0.0, 20.0, 300.0, 600.0), precomp: 1.0, to_part: None });
        out.extend([hub, arm, strut]);
        if !upper.links.is_empty() {
            out.push(upper);
        }
        out.extend([tie, wh, cap]);
    }

    // ---- engine: 8 block corners and 2 mount ears, on the front subframe
    let (ex, ey0, ey1, ez0, ez1) = (0.20, 0.335, 0.735, g.axle_f - 0.215, g.axle_f + 0.385);
    {
        let mut p = MechPart::new("engine", "engine");
        for (i, (x, y, z)) in [(ex, ey0, ez1), (-ex, ey0, ez1), (ex, ey1, ez1), (-ex, ey1, ez1), (ex, ey0, ez0), (-ex, ey0, ez0), (ex, ey1, ez0), (-ex, ey1, ez0)].into_iter().enumerate() {
            p.node(&format!("{}", i + 1), r(x, y, z), 19.0);
        }
        let zm = 0.5 * (ez0 + ez1);
        p.node("ml", r(0.275, 0.40, zm), 9.0);
        p.node("mr", r(-0.275, 0.40, zm), 9.0);
        let mut pairs: Vec<(usize, usize)> = (0..8).flat_map(|i| (i + 1..8).map(move |j| (i, j))).collect();
        for (ear, sgn) in [(8, 1.0), (9, -1.0)] {
            // an ear to its side's four corners and the two low ones across
            pairs.extend((0..8).filter(|&i| p.nodes[i].pos.x * sgn > 0.0 || p.nodes[i].pos.y < f.y(0.5)).map(|i| (i, ear)));
        }
        p.internal.push(Internal { pairs, v: vals(2e7, 0.0, 250e3, INF), deform_group: Some(("engine".into(), 0.001)) });
        for (s, ear) in [("l", "eng_ml"), ("r", "eng_mr")] {
            p.mounts.push(Mount { group: format!("engine_mount_{s}"), from: vec![ear.into()], to: vec![format!("subf_f{s}"), format!("subf_r{s}"), format!("subf_k{s}")], beams: 3, kind: BeamKind::Bounded, v: vals(32_000.0, 600.0, 20e3, 90e3), precomp: 1.0, to_part: None });
        }
        // block, head and cover, sump, intake, manifold, accessory drive, ears
        p.mesh = crate::equipment::engine_mesh(f, ex, ez0, ez1);
        out.push(p);
    }
    // ---- transmission behind it, into the tunnel's mouth
    let (z_tt, y_tt) = (g.axle_f - 0.93, 0.305);
    {
        let mut p = MechPart::new("transmission", "driveline");
        p.node("t", r(0.0, y_tt, z_tt), 20.0);
        p.node("b", r(0.0, 0.40, ez0 - 0.10), 25.0);
        p.internal.push(Internal { pairs: vec![(0, 1)], v: vals(4.8e6, 150.0, 27e3, INF), deform_group: None });
        p.mounts.push(Mount { group: "transmission_engine".into(), from: p.names(), to: (5..=8).map(|i| format!("eng_{i}")).collect(), beams: 8, kind: BeamKind::Normal, v: vals(4.8e6, 150.0, 27e3, INF), precomp: 1.0, to_part: None });
        p.mounts.push(Mount { group: "transmission_mount".into(), from: vec![p.nodes[0].name.clone()], to: ["c1_1tl", "c1_1tr", "c1_2tl", "c1_2tr", "c3_1", "c3_2", "c1_3tl", "c1_3tr", "c3_3"].iter().map(|n| format!("sh_{n}")).collect(), beams: 9, kind: BeamKind::Bounded, v: vals(1.4e5, 2700.0, 5000.0, 160e3), precomp: 1.0, to_part: None });
        // bellhousing, then the case tapering into the tunnel
        let stn = [(ez0 - 0.004, 0.44, 0.34, 0.30), (ez0 - 0.20, 0.42, 0.30, 0.26), (ez0 - 0.24, 0.315, 0.17, 0.11), (z_tt - 0.06, 0.305, 0.13, 0.115)];
        let rings: Vec<(DVec3, DVec3, DVec3, f64, f64)> = stn.iter().map(|&(z, y, w, h)| (r(0.0, y, z), DVec3::X, DVec3::Y, w, h)).collect();
        p.mesh = box_stations(&rings);
        out.push(p);
    }
    // ---- driveshaft, differential, half-shafts
    let (z_dn, y_d) = (g.axle_r + 0.17, 0.31);
    {
        let mut p = MechPart::new("driveshaft", "driveline");
        let (a, b) = (r(0.0, y_tt, z_tt - 0.07), r(0.0, y_d - 0.012, z_dn + 0.018));
        p.node("f", a, 0.5);
        p.node("r", b, 0.5);
        p.mesh = cyl(a - DVec3::Z * 0.01, b + DVec3::Z * 0.01, 0.030);
        p.internal.push(Internal { pairs: vec![(0, 1)], v: vals(0.0, 50.0, 1500.0, INF), deform_group: None });
        p.mounts.push(Mount { group: "driveshaft_front".into(), from: vec!["dri_f".into()], to: vec!["tra_t".into(), "tra_b".into()], beams: 2, kind: BeamKind::Normal, v: vals(0.0, 50.0, 1500.0, 3500.0), precomp: 1.0, to_part: None });
        p.mounts.push(Mount { group: "driveshaft_rear".into(), from: vec!["dri_r".into()], to: vec!["dif_n".into(), "dif_l".into(), "dif_r".into()], beams: 3, kind: BeamKind::Normal, v: vals(0.0, 50.0, 1500.0, 3500.0), precomp: 1.0, to_part: None });
        out.push(p);
    }
    {
        let mut p = MechPart::new("differential", "driveline");
        p.node("n", r(0.0, y_d, z_dn), 14.0);
        p.node("l", r(0.11, y_d, g.axle_r - 0.02), 14.0);
        p.node("r", r(-0.11, y_d, g.axle_r - 0.02), 14.0);
        p.mesh.add(&rbox(r(0.0, y_d, g.axle_r - 0.02), DVec3::new(0.105, 0.065, 0.07), 0.05, 3, Mat::MetalBare));
        p.mesh.add(&cyl(r(0.0, y_d, g.axle_r + 0.054), r(0.0, y_d, z_dn + 0.008), 0.024));
        p.internal.push(Internal { pairs: p.complete(), v: vals(4.8e6, 150.0, 27e3, INF), deform_group: None });
        p.mounts.push(Mount { group: "differential".into(), from: p.names(), to: ["fl", "fr", "rl", "rr", "cf", "cr"].iter().map(|n| format!("subr_{n}")).collect(), beams: 18, kind: BeamKind::Normal, v: vals(4.8e6, 150.0, 27e3, 200e3), precomp: 1.0, to_part: None });
        for (cn, sg) in [("RL", 1.0), ("RR", -1.0)] {
            let c = cn.to_lowercase();
            let s = if sg > 0.0 { "l" } else { "r" };
            let mut hs = MechPart::new(&format!("halfshaft_{cn}"), "driveline");
            let (a, b) = (r(sg * 0.112, y_d, g.axle_r), DVec3::new(sg * (t - 0.215), wy, g.zr));
            hs.mesh = cyl(a, b, 0.018);
            hs.links.push(Link { name: format!("halfshaft_{c}"), a: format!("dif_{s}"), b: format!("whe{c}_ai"), v: vals(0.0, 20.0, 3600.0, 9500.0), group: Some(format!("halfshaft_{c}")) });
            hs.binds.extend([format!("dif_{s}"), format!("dif_n"), format!("whe{c}_ai"), format!("hub{c}_up")]);
            out.push(hs);
        }
        out.insert(out.len() - 2, p);
    }

    // ---- exhaust: a chain of 13 from the manifold to the tail pipe
    {
        let mut p = MechPart::new("exhaust", "exhaust");
        let zr = g.axle_r;
        let pts = [
            (0.26, 0.47, g.axle_f + 0.05),
            (0.27, 0.33, g.axle_f - 0.25),
            (0.25, 0.166, g.axle_f - 0.52),
            (0.24, 0.166, g.axle_f - 0.80),
            (0.24, 0.166, 0.25),
            (0.24, 0.166, -0.15),
            (0.24, 0.166, -0.55),
            (0.24, 0.166, zr + 0.375),
            (0.245, 0.262, zr + 0.26),
            (0.25, 0.262, zr - 0.02),
            (0.27, 0.262, zr - 0.25),
            (0.30, 0.285, zr - 0.60),
            (0.31, 0.285, f.zi((g.z_tail)(f.x(0.31), f.y(0.30))) + 0.05),
        ];
        let path: Vec<DVec3> = pts.iter().map(|&(x, y, z)| r(x, y, z)).collect();
        for (i, q) in path.iter().enumerate() {
            p.node(&format!("{}", i + 1), *q, 1.4);
        }
        // the sections, their flanges and the hangers
        p.mesh = crate::equipment::exhaust_mesh(&path, &[4, 6, 8, 11]);
        let n = p.nodes.len();
        let mut pairs: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        pairs.extend((0..n - 2).map(|i| (i, i + 2)));
        pairs.extend((0..n - 3).map(|i| (i, i + 3)));
        p.internal.push(Internal { pairs, v: vals(1.5e6, 50.0, 12e3, INF), deform_group: None });
        p.mounts.push(Mount { group: "exhaust_manifold".into(), from: vec!["exh_1".into(), "exh_2".into()], to: vec!["eng_1".into(), "eng_3".into(), "eng_5".into(), "eng_7".into()], beams: 4, kind: BeamKind::Normal, v: vals(1.5e6, 50.0, 12e3, 65e3), precomp: 1.0, to_part: None });
        for (grp, from, to) in [
            ("exhaust_hanger_1", vec![4, 5], vec!["c1_1tl", "c1_2tl", "c1_2ol", "c3_2"]),
            ("exhaust_hanger_2", vec![6, 7], vec!["c1_3tl", "c1_4tl", "c1_3ol", "c1_4ol"]),
            ("exhaust_hanger_3", vec![9, 10], vec!["e1_1dl", "e1_1ul", "e3_1", "c1_5tl"]),
            ("exhaust_hanger_4", vec![12, 13], vec!["e1_3dl", "e1_3ul", "e3_3", "e1_2dl"]),
        ] {
            p.mounts.push(Mount { group: grp.into(), from: from.iter().map(|i| format!("exh_{i}")).collect(), to: to.iter().map(|n| format!("sh_{n}")).collect(), beams: 8, kind: BeamKind::Normal, v: vals(47_000.0, 370.0, 4000.0, 15e3), precomp: 1.0, to_part: None });
        }
        out.push(p);
    }
    // ---- fuel tank: a soft box of 8 under the boot floor, on straps
    {
        let mut p = MechPart::new("fuel_tank", "fuel");
        let zr = g.axle_r;
        let (x0, x1, y0, y1, z0, z1) = (-0.40, 0.16, 0.335, 0.440, zr - 0.62, zr - 0.25);
        for (i, (x, y, z)) in [(x0, y0, z0), (x1, y0, z0), (x0, y1, z0), (x1, y1, z0), (x0, y0, z1), (x1, y0, z1), (x0, y1, z1), (x1, y1, z1)].into_iter().enumerate() {
            p.node(&format!("{}", i + 1), r(x, y, z), 5.0);
        }
        p.mesh = rbox(r(0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1)), DVec3::new(0.5 * (f.x(x1) - f.x(x0)), 0.5 * (y1 - y0), 0.5 * (z1 - z0)), 0.03, 3, Mat::TrimBlack);
        p.internal.push(Internal { pairs: p.complete(), v: vals(0.0, 50.0, 5000.0, INF), deform_group: None });
        p.mounts.push(Mount { group: "fuel_tank".into(), from: p.names(), to: ["e1_2dl", "e1_2ul", "e1_2dr", "e1_2ur", "e1_3dl", "e1_3dr", "e3_2", "e3_3"].iter().map(|n| format!("sh_{n}")).collect(), beams: 28, kind: BeamKind::Normal, v: vals(30_000.0, 500.0, 5000.0, 35e3), precomp: 1.0, to_part: None });
        out.push(p);
    }
    // ---- radiator on the radiator support
    {
        let mut p = MechPart::new("radiator", "cooling");
        let a3 = sh(st, "sh_a3_uc");
        let z = a3.z + 0.062;
        let (hx, y0, y1) = (f.x(0.36), sh(st, "sh_a3_dc").y + 0.035, a3.y - 0.045);
        for (tg, x, y) in [("ul", hx, y1), ("ur", -hx, y1), ("dl", hx, y0), ("dr", -hx, y0)] {
            p.node(tg, DVec3::new(x, y, z), 1.4);
        }
        p.mesh = box_(DVec3::new(0.0, 0.5 * (y0 + y1), z), DVec3::new(hx, 0.5 * (y1 - y0), 0.018), Mat::TrimBlack, None);
        p.internal.push(Internal { pairs: p.complete(), v: vals(0.0, 50.0, 7500.0, INF), deform_group: Some(("radiator".into(), 0.05)) });
        p.mounts.push(Mount { group: "radiator".into(), from: p.names(), to: ["a3_ul", "a3_uc", "a3_ur", "a3_dc", "a2_ul", "a2_ur"].iter().map(|n| format!("sh_{n}")).collect(), beams: 12, kind: BeamKind::Normal, v: vals(1e6, 150.0, 7500.0, 20e3), precomp: 1.0, to_part: None });
        out.push(p);
    }

    // ---- interior: seats on four feet, the rear bench, the instrument
    // panel on its cross-car beam, the console
    let y_pan = 0.222;
    let rows = {
        // (the floor rows the seat cross members stand on)
        let c3 = st.node("sh_c3_3").unwrap().pos.z;
        let c4 = st.node("sh_c3_4").unwrap().pos.z;
        (c3, c4)
    };
    let y_xm = f.y(y_pan) + 0.060 + 0.006;
    for (cn, sg) in [("FL", 1.0), ("FR", -1.0)] {
        let s = if sg > 0.0 { "l" } else { "r" };
        let mut p = MechPart::new(&format!("seat_{cn}"), "interior");
        let (x, zc) = (sg * f.x(0.38), 0.5 * (rows.0 + rows.1));
        let h = y_xm + 0.020;
        p.mesh = crate::parts::bucket_seat(x, zc, h, Mat::InteriorFabric, true);
        let mut feet = Vec::new();
        for (tg, dx, z) in [("fo", 0.16, rows.0), ("fi", -0.16, rows.0), ("ro", 0.16, rows.1), ("ri", -0.16, rows.1)] {
            let q = DVec3::new(x + sg * dx, y_xm, z);
            p.mesh.add(&box_(q + DVec3::Y * 0.012, DVec3::new(0.018, 0.011, 0.020), Mat::MetalBare, None));
            feet.push(p.node(tg, q + DVec3::Y * 0.03, 2.7));
        }
        // the cushion's top and the back's top corners
        p.node("c", DVec3::new(x, h + 0.17, zc), 2.7);
        p.node("bo", DVec3::new(x + sg * 0.20, h + 0.66, zc - 0.42), 2.7);
        p.node("bi", DVec3::new(x - sg * 0.20, h + 0.66, zc - 0.42), 2.7);
        p.internal.push(Internal { pairs: p.complete(), v: vals(0.0, 38.0, 6000.0, INF), deform_group: None });
        for (foot, (row, lane)) in feet.iter().zip([(3, "o"), (3, "t"), (4, "o"), (4, "t")]) {
            let to = vec![format!("sh_c1_{row}{lane}{s}"), format!("sh_c1_{row}{}{s}", if lane == "o" { "t" } else { "o" }), format!("sh_c2_{row}{s}"), format!("sh_c3_{row}")];
            let from = vec![foot.clone(), p.nodes[4].name.clone()];
            p.mounts.push(Mount { group: format!("seat_{}_{}", cn.to_lowercase(), &foot[foot.len() - 2..]), from, to, beams: 6, kind: BeamKind::Normal, v: vals(2.8e5, 38.0, 6000.0, 30e3), precomp: 1.0, to_part: None });
        }
        out.push(p);
    }
    {
        let mut p = MechPart::new("seat_rear", "interior");
        let e3 = sh(st, "sh_e3_1");
        let (hx, yc, zc) = (f.x(0.47), e3.y + 0.042 + 0.07, e3.z + 0.185);
        p.mesh.add(&rbox(DVec3::new(0.0, yc, zc), DVec3::new(hx, 0.07, 0.20), 0.05, 3, Mat::InteriorFabric));
        let mut back = rbox(DVec3::ZERO, DVec3::new(hx, 0.25, 0.05), 0.045, 3, Mat::InteriorFabric);
        let (yb, zb) = (yc + 0.335, zc - 0.25);
        back.transform(Some(rot_x((-8.0f64).to_radians())), DVec3::new(0.0, yb, zb));
        p.mesh.add(&back);
        for (tg, x, y, z) in [("fl", hx - 0.06, yc, zc + 0.16), ("fr", -hx + 0.06, yc, zc + 0.16), ("rl", hx - 0.06, yc, zc - 0.16), ("rr", -hx + 0.06, yc, zc - 0.16), ("c", 0.0, yc, zc), ("bl", hx - 0.06, yb + 0.20, zb - 0.03), ("br", -hx + 0.06, yb + 0.20, zb - 0.03), ("bc", 0.0, yb + 0.20, zb - 0.03)] {
            p.node(tg, DVec3::new(x, y, z), 2.3);
        }
        p.internal.push(Internal { pairs: p.complete(), v: vals(0.0, 38.0, 6000.0, INF), deform_group: None });
        p.mounts.push(Mount { group: "seat_rear_front".into(), from: vec!["sear_fl".into(), "sear_fr".into(), "sear_c".into(), "sear_rl".into(), "sear_rr".into()], to: ["c1_5tl", "c1_5tr", "c1_5ol", "c1_5or", "c3_5", "e3_1", "e1_1ul", "e1_1ur"].iter().map(|n| format!("sh_{n}")).collect(), beams: 14, kind: BeamKind::Normal, v: vals(2.8e5, 38.0, 6000.0, 30e3), precomp: 1.0, to_part: None });
        p.mounts.push(Mount { group: "seat_rear_back".into(), from: vec!["sear_bl".into(), "sear_br".into(), "sear_bc".into()], to: ["e7_l", "e7_r", "e2_l", "e2_r", "e6_ol", "e6_or"].iter().map(|n| format!("sh_{n}")).collect(), beams: 10, kind: BeamKind::Normal, v: vals(2.8e5, 38.0, 6000.0, 30e3), precomp: 1.0, to_part: None });
        out.push(p);
    }
    {
        let mut p = MechPart::new("dash", "interior");
        let b2 = sh(st, "sh_b2_ol");
        // the cross-car beam between the hinge pillars, under the cowl; the
        // panel hangs behind it on two brackets
        let (hx, yb, zb) = (b2.x + 0.012, b2.y - 0.135, b2.z + 0.040);
        let (yp, zp) = (b2.y - 0.075, sh(st, "sh_c3_1").z - 0.35);
        p.mesh.add(&cyl(DVec3::new(-hx, yb, zb), DVec3::new(hx, yb, zb), 0.026));
        p.mesh.add(&rbox(DVec3::new(0.0, yp, zp), DVec3::new(hx - 0.03, 0.105, 0.15), 0.05, 3, Mat::InteriorPlastic));
        p.mesh.add(&rbox(DVec3::new(f.x(0.38), yp + 0.125, zp - 0.03), DVec3::new(0.17, 0.028, 0.075), 0.022, 2, Mat::InteriorPlastic));
        for sgn in [1.0, -1.0] {
            p.mesh.add(&bar(DVec3::new(sgn * 0.30, yb, zb - 0.03), DVec3::new(sgn * 0.30, yp - 0.02, zp + 0.155), 0.03, 0.03));
        }
        // the cluster's dials, the centre screen and vents, the glovebox
        p.mesh.add(&crate::equipment::dash_features(f, yp, zp - 0.15));
        for (tg, x, y, z, kg) in [("bl", hx, yb, zb, 6.0), ("br", -hx, yb, zb, 6.0), ("ml", 0.30, yb, zb, 7.0), ("mr", -0.30, yb, zb, 7.0), ("tl", 0.45, yp + 0.09, zp, 6.0), ("tr", -0.45, yp + 0.09, zp, 6.0), ("c", 0.0, yp - 0.07, zp - 0.10, 7.0)] {
            p.node(tg, DVec3::new(x, y, z), kg);
        }
        p.internal.push(Internal { pairs: p.complete(), v: vals(0.0, 60.0, 32e3, INF), deform_group: None });
        for (grp, from, to) in [
            ("dash_l", vec!["das_bl", "das_ml", "das_tl"], vec!["b2_ol", "b2_il", "b1_ol", "b1_il", "c2_1l"]),
            ("dash_r", vec!["das_br", "das_mr", "das_tr"], vec!["b2_or", "b2_ir", "b1_or", "b1_ir", "c2_1r"]),
            ("dash_c", vec!["das_c", "das_ml", "das_mr"], vec!["c3_1", "c3_2", "b1_il", "b1_ir"]),
        ] {
            p.mounts.push(Mount { group: grp.into(), from: from.iter().map(|n| n.to_string()).collect(), to: to.iter().map(|n| format!("sh_{n}")).collect(), beams: 10, kind: BeamKind::Normal, v: vals(4e6, 60.0, 32e3, 60e3), precomp: 1.0, to_part: None });
        }
        p.meshes.push("steering_wheel".into());
        out.push(p);
    }
    {
        let mut p = MechPart::new("console", "interior");
        let (c2, c4) = (sh(st, "sh_c3_2"), sh(st, "sh_c3_4"));
        let (z0, z1, y0) = (c2.z - 0.10, c4.z + 0.05, c2.y + 0.008);
        p.mesh = rbox(DVec3::new(0.0, y0 + 0.075, 0.5 * (z0 + z1)), DVec3::new(0.085, 0.075, 0.5 * (z0 - z1)), 0.03, 2, Mat::InteriorPlastic);
        for (tg, y, z) in [("f", y0 + 0.02, z0 - 0.03), ("m", y0 + 0.13, 0.5 * (z0 + z1)), ("r", y0 + 0.02, z1 + 0.03), ("t", y0 + 0.13, z0 - 0.10)] {
            p.node(tg, DVec3::new(0.0, y, z), 1.25);
        }
        p.internal.push(Internal { pairs: p.complete(), v: vals(0.0, 30.0, 3000.0, INF), deform_group: None });
        p.mounts.push(Mount { group: "console".into(), from: p.names(), to: ["c3_2", "c3_3", "c3_4", "c1_2tl", "c1_2tr", "c1_3tl", "c1_3tr", "c1_4tl", "c1_4tr"].iter().map(|n| format!("sh_{n}")).collect(), beams: 12, kind: BeamKind::Normal, v: vals(0.0, 30.0, 3000.0, 10e3), precomp: 1.0, to_part: None });
        out.push(p);
    }
    crate::equipment::sedan_equipment(g, st, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_prefixes_are_short_and_distinct() {
        let names = ["impact_beam_front", "impact_beam_rear", "subframe_front", "subframe_rear", "steering_rack", "hub_FL", "hub_RR", "wheel_FL", "hub_cap_FL", "engine", "transmission", "driveshaft", "differential", "exhaust", "fuel_tank", "radiator", "seat_FL", "seat_rear", "dash", "console"];
        let mut p: Vec<String> = names.iter().map(|n| MechPart::new(n, "x").prefix()).collect();
        assert_eq!(MechPart::new("subframe_front", "x").prefix(), "subf");
        assert_eq!(MechPart::new("wheel_RL", "x").prefix(), "wherl");
        assert_eq!(MechPart::new("hub_FL", "x").prefix(), "hubfl");
        p.sort();
        p.dedup();
        assert_eq!(p.len(), names.len());
    }
}
