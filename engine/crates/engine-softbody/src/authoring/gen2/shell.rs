//! The body shell's lattice on the body-in-white's own nodes.
//!
//! `engine_vehiclegen::structure` defines each member once and places the
//! shell's 131 nodes on them. This module braces those nodes the way the
//! measured BeamNG shells are braced (`engine/docs/beamng-structure-study.md`
//! §3.1, §3.5, §10.4): edges along and across every member, skip beams along
//! the rails, sills and roof rails, cross-car beams where a cross member is,
//! then nearest-neighbour bracing until every node is a corner of a
//! tetrahedron. Springs come from the zone table, capped by each node's
//! stiffness budget (Σk/m); yield forces from the zone table, with the
//! longitudinal beams of a boxed member sharing that member's own mean
//! crush force.
//!
//! It also measures the lattice against the study's shell rules (§10.7,
//! rules 1–9, 12, 13) and its highest mode against the step (ω·h by power
//! iteration on the solver's operator, k + 2c/h).
//!
//! glTF frame (x left, y up, z forward), metres, SI.

use std::collections::{BTreeSet, HashMap};

use engine_vehiclegen::mechanical::MechPart;

use engine_vehiclegen::structure::Structure;
use engine_vehiclegen::physics::CrashFeel;
use glam::{DMat3, DVec3};

/// The fixed step (s).
pub const H: f64 = 0.0005;
/// Per-node stiffness budget: Σk/m (s⁻²). The cars sit at 7–10e6.
pub const NODE_BUDGET: f64 = 8.5e6;
/// Whole-lattice bound: ω·h by power iteration (the cars: 1.80–1.90; the
/// builder's own limit, `build::OMEGA_H`).
pub const OMEGA_H: f64 = 1.85;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// along or across a member
    Edge,
    /// past one node along a member
    Skip,
    /// bracing (a rectangle's diagonal, a tetrahedron's edge)
    Brace,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Zone {
    CrashBox,
    FrontRail,
    Tower,
    Cabin,
    /// sill and door-frame primaries
    Primary,
    Pillar,
    Bow,
    RearRail,
    Tail,
}

impl Zone {
    /// (k N/m, c N·s/m, deform N, deformLimitExpansion): §10.4.
    pub fn values(self) -> (f64, f64, f64, f64) {
        match self {
            Zone::CrashBox => (5.0e6, 115.0, 5_000.0, 1.2),
            Zone::FrontRail => (5.0e6, 115.0, 10_500.0, 1.2),
            Zone::Tower => (5.0e6, 140.0, 27_500.0, 1.2),
            Zone::Cabin => (1.75e6, 128.0, 22_000.0, 1.1),
            Zone::Primary => (2.1e6, 400.0, 57_500.0, 1.1),
            Zone::Pillar => (2.0e6, 110.0, 19_500.0, 1.1),
            Zone::Bow => (2.0e6, 110.0, 10_000.0, 1.1),
            Zone::RearRail => (3.0e6, 140.0, 10_500.0, 1.2),
            Zone::Tail => (1.5e6, 125.0, 7_500.0, 1.2),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ShellBeam {
    pub a: usize,
    pub b: usize,
    pub class: Class,
    pub zone: Zone,
    pub k: f64,
    pub c: f64,
    pub deform: f64,
    pub limit_expansion: f64,
}

/// A four-node torsion spring along the floor (the pan's bending).
#[derive(Clone, Copy, Debug)]
pub struct Torsion {
    pub nodes: [usize; 4],
    /// N·m/rad
    pub k: f64,
    pub deform: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Lattice {
    pub beams: Vec<ShellBeam>,
    pub torsion: Vec<Torsion>,
}

fn zone_of(ga: &str, gb: &str, primary: bool) -> Zone {
    let crash = |g: &str| matches!(g, "A1" | "A2");
    let mount = |g: &str| matches!(g, "A4" | "A5" | "A6" | "A7");
    let front = |g: &str| g.starts_with('A');
    let cabin = |g: &str| g.starts_with('B') || g.starts_with('C');
    let rail_r = |g: &str| matches!(g, "E1" | "E2" | "E3");
    let tail = |g: &str| matches!(g, "E4" | "E5" | "E6" | "E7");
    if crash(ga) && crash(gb) {
        Zone::CrashBox
    } else if mount(ga) && mount(gb) {
        Zone::Tower
    } else if (front(ga) && cabin(gb)) || (front(gb) && cabin(ga)) {
        // the front structure's roots on the cabin (toe board, hinge pillar,
        // cowl): the firewall station, stronger than the rails ahead of it
        Zone::Cabin
    } else if front(ga) || front(gb) {
        Zone::FrontRail
    } else if ga == "D4" || gb == "D4" {
        Zone::Bow
    } else if primary {
        Zone::Primary
    } else if ga.starts_with('D') || gb.starts_with('D') {
        Zone::Pillar
    } else if tail(ga) || tail(gb) {
        Zone::Tail
    } else if rail_r(ga) || rail_r(gb) {
        Zone::RearRail
    } else {
        debug_assert!(cabin(ga) && cabin(gb));
        Zone::Cabin
    }
}

/// The members' node chains (front to rear, or bottom to top): `{s}` is the
/// side. `true`: a primary (sill, door frame).
const CHAINS: [(&[&str], bool); 16] = [
    (&["a1_d{s}", "a2_d{s}", "a4_{s}", "a5_{s}"], false),
    (&["a1_u{s}", "a2_u{s}"], false),
    (&["a3_u{s}", "a8_f{s}", "a8_m{s}", "b2_o{s}"], false),
    (&["c1_4o{s}", "c1_5o{s}", "e1_1d{s}"], false),
    (&["c2_4{s}", "c2_5{s}", "e5_1d{s}"], false),
    (&["a7_{s}", "a6_{s}"], false),
    (
        &["c1_1o{s}", "c1_2o{s}", "c1_3o{s}", "c1_4o{s}", "c1_5o{s}"],
        true,
    ),
    (
        &["c2_1{s}", "c2_2{s}", "c2_3{s}", "c2_4{s}", "c2_5{s}"],
        true,
    ),
    (
        &["c1_1t{s}", "c1_2t{s}", "c1_3t{s}", "c1_4t{s}", "c1_5t{s}"],
        false,
    ),
    (&["e1_1d{s}", "e1_2d{s}", "e1_3d{s}"], false),
    (&["e1_1u{s}", "e1_2u{s}", "e1_3u{s}"], false),
    (
        &["e5_1d{s}", "e5_2d{s}", "e5_3d{s}", "e5_4d{s}", "e5_5d{s}"],
        false,
    ),
    (
        &["e5_1u{s}", "e5_2u{s}", "e5_3u{s}", "e5_4u{s}", "e5_5u{s}"],
        false,
    ),
    (
        &[
            "b2_o{s}", "d1_m{s}", "d1_t{s}", "d2_t{s}", "d3_t{s}", "d3_b{s}",
        ],
        true,
    ),
    (&["c2_3{s}", "d2_b{s}", "d2_t{s}"], true),
    (&["c2_1{s}", "b2_o{s}"], true),
];

/// Chains on the centre line.
const CENTRE: [&[&str]; 3] = [
    &["c3_1", "c3_2", "c3_3", "c3_4", "c3_5"],
    &["e3_1", "e3_2", "e3_3"],
    &["d4_h", "d4_b", "d4_r"],
];

/// Rows across the car, where a cross member or a panel is (left to right).
fn rows_across() -> Vec<Vec<String>> {
    let mut r: Vec<Vec<String>> = Vec::new();
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    r.push(s(&["a3_ul", "a3_uc", "a3_ur"]));
    r.push(s(&["a2_dl", "a3_dc", "a2_dr"]));
    r.push(s(&["b1_ol", "b1_il", "b1_ir", "b1_or"]));
    r.push(s(&["b2_ol", "b2_il", "b2_ir", "b2_or"]));
    for n in 1..=5 {
        r.push(vec![
            format!("c1_{n}ol"),
            format!("c1_{n}tl"),
            format!("c1_{n}tr"),
            format!("c1_{n}or"),
        ]);
        r.push(vec![
            format!("c1_{n}tl"),
            format!("c3_{n}"),
            format!("c1_{n}tr"),
        ]);
    }
    r.push(s(&["d1_tl", "d4_h", "d1_tr"]));
    r.push(s(&["d2_tl", "d4_b", "d2_tr"]));
    r.push(s(&["d3_tl", "d4_r", "d3_tr"]));
    r.push(s(&["e7_l", "e7_r"]));
    r.push(s(&["e6_ol", "e6_rl", "e6_rr", "e6_or"]));
    r.push(s(&["e4_dl", "e4_dc", "e4_dr"]));
    r.push(s(&["e4_ul", "e4_uc", "e4_ur"]));
    for k in 1..=3 {
        r.push(vec![
            format!("e1_{k}ul"),
            format!("e3_{k}"),
            format!("e1_{k}ur"),
        ]);
    }
    r
}

/// Brace the structure's shell nodes: the beams with their zone values under
/// the node budget, and the floor's torsion bars.
pub fn lattice(st: &Structure) -> Lattice {
    lattice_with(st, &CrashFeel::real_crash())
}

/// A zone's multiple in a crash character.
fn feel_of(zone: Zone, f: &CrashFeel) -> f64 {
    f64::from(match zone {
        Zone::CrashBox | Zone::FrontRail => f.front_crush,
        Zone::Tower => f.tower,
        Zone::Cabin | Zone::Primary | Zone::Pillar | Zone::Bow => f.cabin,
        Zone::RearRail | Zone::Tail => f.rear_crush,
    })
}

/// [`lattice`] in a crash character: every beam's yield is its zone's (or its
/// member's mean crush force) times the character's multiple for that zone.
/// Springs, damping and topology do not change.
pub fn lattice_with(st: &Structure, feel: &CrashFeel) -> Lattice {
    let mut l = lattice_defined(st);
    for b in &mut l.beams {
        b.deform *= feel_of(b.zone, feel);
    }
    l
}

fn lattice_defined(st: &Structure) -> Lattice {
    let n = st.nodes.len();
    let id: HashMap<&str, usize> = st
        .nodes
        .iter()
        .enumerate()
        .map(|(i, x)| (x.name.as_str(), i))
        .collect();
    let at = |name: &str| {
        *id.get(format!("sh_{name}").as_str())
            .unwrap_or_else(|| panic!("no shell node sh_{name}"))
    };
    let pos: Vec<DVec3> = st.nodes.iter().map(|x| x.pos).collect();
    let mirror: Vec<usize> = (0..n)
        .map(|i| {
            let m = DVec3::new(-pos[i].x, pos[i].y, pos[i].z);
            (0..n)
                .min_by(|&a, &b| (pos[a] - m).length().total_cmp(&(pos[b] - m).length()))
                .unwrap()
        })
        .collect();
    let mut edges: HashMap<(usize, usize), (Class, bool)> = HashMap::new();
    let mut add = |a: usize, b: usize, class: Class, primary: bool| {
        if a == b {
            return;
        }
        for (a, b) in [(a, b), (mirror[a], mirror[b])] {
            let key = (a.min(b), a.max(b));
            let e = edges.entry(key).or_insert((class, primary));
            // an edge outranks a skip beam outranks a brace
            if class == Class::Edge || (class == Class::Skip && e.0 == Class::Brace) {
                e.0 = class;
            }
            e.1 |= primary;
        }
    };
    let mut chains: Vec<(Vec<usize>, bool)> = Vec::new();
    for (c, primary) in CHAINS {
        chains.push((
            c.iter().map(|x| at(&x.replace("{s}", "l"))).collect(),
            primary,
        ));
    }
    for c in CENTRE {
        chains.push((c.iter().map(|x| at(x)).collect(), false));
    }
    for r in rows_across() {
        chains.push((r.iter().map(|x| at(x)).collect(), false));
    }
    for (c, primary) in &chains {
        for w in c.windows(2) {
            add(w[0], w[1], Class::Edge, *primary);
        }
        for w in c.windows(3) {
            add(w[0], w[2], Class::Skip, *primary);
        }
    }
    // a boxed member's two faces, and each sill's top over its bottom
    for s in ["l", "r"] {
        for (a, b) in [("a1_u", "a1_d"), ("a2_u", "a2_d")] {
            add(
                at(&format!("{a}{s}")),
                at(&format!("{b}{s}")),
                Class::Edge,
                false,
            );
        }
        for k in 1..=3 {
            add(
                at(&format!("e1_{k}u{s}")),
                at(&format!("e1_{k}d{s}")),
                Class::Edge,
                false,
            );
        }
        for k in 1..=5 {
            add(
                at(&format!("c2_{k}{s}")),
                at(&format!("c1_{k}o{s}")),
                Class::Edge,
                true,
            );
            add(
                at(&format!("e5_{k}u{s}")),
                at(&format!("e5_{k}d{s}")),
                Class::Edge,
                false,
            );
        }
    }
    for t in ["l", "c", "r"] {
        add(
            at(&format!("e4_u{t}")),
            at(&format!("e4_d{t}")),
            Class::Edge,
            false,
        );
    }
    // Bracing: every node takes its nearest nodes until it has `DEGREE`
    // neighbours, nearest first, none over `REACH` long.
    const DEGREE: usize = 10;
    const REACH: f64 = 0.80;
    let degree = |edges: &HashMap<(usize, usize), (Class, bool)>, i: usize| {
        edges.keys().filter(|k| k.0 == i || k.1 == i).count()
    };
    for i in 0..n {
        let mut near: Vec<usize> = (0..n)
            .filter(|&j| j != i && (pos[i] - pos[j]).length() <= REACH)
            .collect();
        near.sort_by(|&a, &b| {
            (pos[i] - pos[a])
                .length()
                .total_cmp(&(pos[i] - pos[b]).length())
        });
        for j in near {
            if degree(&edges, i) >= DEGREE {
                break;
            }
            if !edges.contains_key(&(i.min(j), i.max(j))) {
                let mut add = |a: usize, b: usize| {
                    for (a, b) in [(a, b), (mirror[a], mirror[b])] {
                        if a != b {
                            edges
                                .entry((a.min(b), a.max(b)))
                                .or_insert((Class::Brace, false));
                        }
                    }
                };
                add(i, j);
            }
        }
    }
    // Every node a corner of a tetrahedron, and its beams spread in all
    // three directions: where the bracing left a node flat (a roof bow's
    // centre, a pillar's middle), it takes a beam along its weakest axis
    // (through the cabin if it must: the study allows six over 0.9 m).
    let mut long = 0usize;
    let pair_cost = |a: usize, b: usize| {
        if (pos[a] - pos[b]).length() <= 0.9 {
            0
        } else if mirror[a] == a && mirror[b] == b {
            1
        } else {
            2
        }
    };
    for _ in 0..6 {
        let nb = neighbours(n, edges.keys());
        // (the flattest nodes first: the long braces go where they are needed)
        let mut order: Vec<usize> = (0..n)
            .filter(|&i| spread_of(&pos, &nb[i], i).0 < 0.03 || !tetrahedron_at(&pos, &nb, i))
            .collect();
        if order.is_empty() {
            break;
        }
        order.sort_by(|&a, &b| {
            spread_of(&pos, &nb[a], a)
                .0
                .total_cmp(&spread_of(&pos, &nb[b], b).0)
        });
        let mut done: BTreeSet<usize> = BTreeSet::new();
        for i in order {
            if done.contains(&i) {
                continue;
            }
            let axis = spread_of(&pos, &nb[i], i).1;
            // A fourth corner j for a triangle (i, k, l) the node already
            // has: the one most along the weak axis for its length, which
            // makes the least flat tetrahedron with the fewest long beams.
            let mut best: Option<(f64, usize, usize, usize, usize)> = None;
            let v: Vec<usize> = nb[i].iter().copied().collect();
            for j in (0..n).filter(|&j| j != i && (pos[j] - pos[i]).length() <= 1.25) {
                let dj = pos[j] - pos[i];
                for (x, &k) in v.iter().enumerate() {
                    for &l in &v[x + 1..] {
                        if k == j || l == j || !nb[k].contains(&l) {
                            continue;
                        }
                        let vol = dj
                            .normalize()
                            .dot(
                                (pos[k] - pos[i])
                                    .normalize()
                                    .cross((pos[l] - pos[i]).normalize()),
                            )
                            .abs();
                        if vol <= 0.12
                            || (pos[j] - pos[k]).length() > 1.25
                            || (pos[j] - pos[l]).length() > 1.25
                        {
                            continue;
                        }
                        let new: Vec<(usize, usize)> = [(i, j), (j, k), (j, l)]
                            .into_iter()
                            .filter(|&(a, b)| !nb[a].contains(&b))
                            .collect();
                        let cost: usize = new.iter().map(|&(a, b)| pair_cost(a, b)).sum();
                        if long + cost > 6 {
                            continue;
                        }
                        let score = vol * dj.normalize().dot(axis).abs().max(0.2)
                            / (dj.length().max(0.2)
                                * (1.0 + cost as f64)
                                * (1.0 + new.len() as f64));
                        if best.is_none_or(|b| score > b.0) {
                            best = Some((score, j, k, l, cost));
                        }
                    }
                }
            }
            if let Some((_, j, k, l, cost)) = best {
                long += cost;
                for (a, b) in [(i, j), (j, k), (j, l)] {
                    for (a, b) in [(a, b), (mirror[a], mirror[b])] {
                        if a != b {
                            edges
                                .entry((a.min(b), a.max(b)))
                                .or_insert((Class::Brace, false));
                        }
                    }
                }
                done.insert(mirror[i]);
            }
        }
    }
    let mut keys: Vec<(usize, usize)> = edges.keys().copied().collect();
    keys.sort();
    let mut beams: Vec<ShellBeam> = keys
        .iter()
        .map(|&(a, b)| {
            let (class, primary) = edges[&(a, b)];
            let zone = zone_of(
                st.nodes[a].group,
                st.nodes[b].group,
                primary && class != Class::Brace,
            );
            let (k, c, deform, lim) = zone.values();
            // a skip beam yields below its zone's edges (§10.4: 0.5–0.9 ×)
            let deform = if class == Class::Skip {
                0.7 * deform
            } else {
                deform
            };
            ShellBeam {
                a,
                b,
                class,
                zone,
                k,
                c,
                deform,
                limit_expansion: lim,
            }
        })
        .collect();
    // The longitudinal edges of a boxed member share its mean crush force
    // (the member's own section, gauge and grade): the crash boxes, the
    // front rails and the rear rails.
    for (kind, groups) in [
        ("crash_box", &["A1", "A2"][..]),
        ("front_rail", &["A2", "A4", "A5"][..]),
        ("rear_rail", &["E1"][..]),
    ] {
        let Some(m) = st.members.iter().find(|m| m.kind == kind) else {
            continue;
        };
        let crush = m.crush_n();
        for s in [1.0, -1.0] {
            let of: Vec<usize> = (0..beams.len())
                .filter(|&j| {
                    let b = &beams[j];
                    let d = (pos[b.a] - pos[b.b]).normalize();
                    b.class == Class::Edge
                        && d.z.abs() > 0.707
                        && pos[b.a].x * s > 0.0
                        && pos[b.b].x * s > 0.0
                        && groups.contains(&st.nodes[b.a].group)
                        && groups.contains(&st.nodes[b.b].group)
                })
                .collect();
            // beams in parallel across one section of the member
            let par = if kind == "front_rail" { 1.0 } else { 2.0 };
            for j in of {
                beams[j].deform = crush / par;
            }
        }
    }
    budget(st, &mut beams);
    // Torsion bars over the floor's cells: sill to tunnel side, and across
    // the tunnel, between consecutive rows.
    let mut torsion = Vec::new();
    for k in 1..5 {
        for s in ["l", "r"] {
            torsion.push([
                at(&format!("c1_{k}o{s}")),
                at(&format!("c1_{k}t{s}")),
                at(&format!("c1_{}t{s}", k + 1)),
                at(&format!("c1_{}o{s}", k + 1)),
            ]);
        }
        torsion.push([
            at(&format!("c1_{k}tl")),
            at(&format!("c1_{k}tr")),
            at(&format!("c1_{}tr", k + 1)),
            at(&format!("c1_{}tl", k + 1)),
        ]);
    }
    Lattice {
        beams,
        torsion: torsion
            .into_iter()
            .map(|nodes| Torsion {
                nodes,
                k: 6.5e5,
                deform: 7_500.0,
            })
            .collect(),
    }
}

fn neighbours<'a>(
    n: usize,
    edges: impl Iterator<Item = &'a (usize, usize)>,
) -> Vec<BTreeSet<usize>> {
    let mut nb = vec![BTreeSet::new(); n];
    for &(a, b) in edges {
        nb[a].insert(b);
        nb[b].insert(a);
    }
    nb
}

/// Beam-direction spread at a node: the smallest eigenvalue of the mean of
/// d dᵀ over its beams (0: all in a plane), and that axis.
fn spread_of(pos: &[DVec3], nb: &BTreeSet<usize>, i: usize) -> (f64, DVec3) {
    let mut m = DMat3::ZERO;
    for &j in nb {
        let d = (pos[j] - pos[i]).normalize();
        m += DMat3::from_cols(d * d.x, d * d.y, d * d.z);
    }
    m *= 1.0 / nb.len().max(1) as f64;
    let tr = m.x_axis.x + m.y_axis.y + m.z_axis.z;
    let shifted = DMat3::from_diagonal(DVec3::splat(tr + 1e-6)) - m;
    let mut v = DVec3::new(0.577, 0.577, 0.577);
    for _ in 0..200 {
        v = (shifted * v).normalize_or(DVec3::X);
    }
    (v.dot(m * v), v)
}

/// Is node `i` a corner of a non-flat tetrahedron of beams?
fn tetrahedron_at(pos: &[DVec3], nb: &[BTreeSet<usize>], i: usize) -> bool {
    let v: Vec<usize> = nb[i].iter().copied().collect();
    for a in 0..v.len() {
        for b in a + 1..v.len() {
            if !nb[v[a]].contains(&v[b]) {
                continue;
            }
            for c in b + 1..v.len() {
                if nb[v[a]].contains(&v[c]) && nb[v[b]].contains(&v[c]) {
                    let (p, q, s) = (pos[v[a]] - pos[i], pos[v[b]] - pos[i], pos[v[c]] - pos[i]);
                    // (not a sliver: the three edges at the node span a solid
                    // angle; 0.003 is about 10 degrees out of plane)
                    if p.normalize().dot(q.normalize().cross(s.normalize())).abs() > 0.003 * 30.0 {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Σk/m at every node (s⁻²).
pub fn sum_k_over_m(st: &Structure, beams: &[ShellBeam]) -> Vec<f64> {
    let mut s = vec![0.0; st.nodes.len()];
    for b in beams {
        s[b.a] += b.k;
        s[b.b] += b.k;
    }
    s.iter().zip(&st.nodes).map(|(k, n)| k / n.mass).collect()
}

/// Fit the springs to the node budget: no node over [`NODE_BUDGET`], every
/// beam as near its zone's spring as its two nodes allow, then the whole
/// lattice under [`OMEGA_H`].
fn budget(st: &Structure, beams: &mut [ShellBeam]) {
    // The zone table gives the springs their proportions; the budget, not
    // the table, is the rule (§10.4): a beam rises to the table's greatest
    // spring where both its nodes have room, and falls where one has none.
    const K_MAX: f64 = 7.0e6;
    for _ in 0..400 {
        let r = sum_k_over_m(st, beams);
        let mut moved = 0.0f64;
        for b in beams.iter_mut() {
            // room at its fuller end
            let room = (NODE_BUDGET / r[b.a]).min(NODE_BUDGET / r[b.b]);
            let k = (b.k * room.clamp(0.5, 1.05)).min(K_MAX);
            moved = moved.max((k / b.k - 1.0).abs());
            b.k = k;
        }
        if moved < 1e-4 {
            break;
        }
    }
    for _ in 0..200 {
        if omega_h(st, beams) <= OMEGA_H {
            break;
        }
        for b in beams.iter_mut() {
            b.k *= 0.98;
        }
    }
}

/// The lattice's highest mode against the step: ω·h, by power iteration on
/// the solver's operator (each beam k + 2c/h along its axis, nodes free).
pub fn omega_h(st: &Structure, beams: &[ShellBeam]) -> f64 {
    let n = st.nodes.len();
    let dirs: Vec<DVec3> = beams
        .iter()
        .map(|b| (st.nodes[b.a].pos - st.nodes[b.b].pos).normalize())
        .collect();
    let mut x: Vec<DVec3> = (0..n)
        .map(|i| {
            DVec3::new(
                ((i * 37 % 11) as f64 - 5.0) / 5.0,
                ((i * 53 % 7) as f64 - 3.0) / 3.0,
                ((i * 29 % 13) as f64 - 6.0) / 6.0,
            )
        })
        .collect();
    let mut lam = 0.0;
    for _ in 0..600 {
        let mut y = vec![DVec3::ZERO; n];
        for (j, b) in beams.iter().enumerate() {
            let f = dirs[j] * ((b.k + 2.0 * b.c / H) * dirs[j].dot(x[b.a] - x[b.b]));
            y[b.a] += f;
            y[b.b] -= f;
        }
        for i in 0..n {
            y[i] /= st.nodes[i].mass;
        }
        let ny = y.iter().map(|v| v.length_squared()).sum::<f64>().sqrt();
        let nx = x.iter().map(|v| v.length_squared()).sum::<f64>().sqrt();
        lam = ny / nx.max(1e-300);
        x = y.into_iter().map(|v| v / ny.max(1e-300)).collect();
    }
    lam.sqrt() * H
}

/// The study's shell rules measured on a lattice (§10.7).
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub nodes: usize,
    pub beams: usize,
    pub beams_per_node: f64,
    pub neighbours_p50: usize,
    pub neighbours_min: usize,
    /// nodes that are no corner of a non-flat tetrahedron of beams
    pub not_in_tetrahedron: Vec<String>,
    /// least beam-direction spread at a node (smallest eigenvalue of the
    /// mean of d dᵀ), and where
    pub spread_min: (f64, String),
    pub skip_share: f64,
    pub cross_share: f64,
    /// beams over 0.9 m with their midpoint inside the cabin
    pub long_cabin: usize,
    /// Σk/m: median, greatest
    pub sum_k_m_p50: f64,
    pub sum_k_m_max: f64,
    pub omega_h: f64,
    /// Σ deform of the longitudinal beams by station (N): nose, front axle,
    /// firewall, cabin front, cabin rear, rear axle, tail
    pub crush: [f64; 7],
    /// the front's longitudinal yield by height: share under 0.45 m, share
    /// over 0.75 m (beams ahead of s 0.2)
    pub low_share: f64,
    pub high_share: f64,
}

pub const STATIONS: [&str; 7] = [
    "nose",
    "front axle",
    "firewall",
    "cabin front",
    "cabin rear",
    "rear axle",
    "tail",
];

fn station(s: f64) -> usize {
    [-0.18, 0.08, 0.30, 0.62, 0.90, 1.12]
        .iter()
        .filter(|&&e| s >= e)
        .count()
}

/// Measure a lattice. `axle_f`, `axle_r`: the axles' z. `others`: the other
/// parts' beams (ends and yield force), which count in the front's load
/// path by height (rule 13 is over every part).
pub fn report(
    st: &Structure,
    lat: &Lattice,
    axle_f: f64,
    axle_r: f64,
    others: &[(DVec3, DVec3, f64)],
) -> Report {
    let n = st.nodes.len();
    let pos: Vec<DVec3> = st.nodes.iter().map(|x| x.pos).collect();
    let mut nb: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    for b in &lat.beams {
        nb[b.a].insert(b.b);
        nb[b.b].insert(b.a);
    }
    let mut r = Report {
        nodes: n,
        beams: lat.beams.len(),
        beams_per_node: lat.beams.len() as f64 / n as f64,
        ..Default::default()
    };
    let mut deg: Vec<usize> = nb.iter().map(|s| s.len()).collect();
    r.neighbours_min = deg.iter().copied().min().unwrap_or(0);
    deg.sort();
    r.neighbours_p50 = deg[n / 2];
    r.spread_min = (f64::INFINITY, String::new());
    for i in 0..n {
        if !tetrahedron_at(&pos, &nb, i) {
            r.not_in_tetrahedron.push(st.nodes[i].name.clone());
        }
        let sp = spread_of(&pos, &nb[i], i).0;
        if sp < r.spread_min.0 {
            r.spread_min = (sp, st.nodes[i].name.clone());
        }
    }
    let nbeams = lat.beams.len().max(1) as f64;
    r.skip_share = lat.beams.iter().filter(|b| b.class == Class::Skip).count() as f64 / nbeams;
    r.cross_share = lat
        .beams
        .iter()
        .filter(|b| pos[b.a].x * pos[b.b].x < -1e-4)
        .count() as f64
        / nbeams;
    let wb = axle_f - axle_r;
    let s_of = |z: f64| (axle_f - z) / wb;
    r.long_cabin = lat
        .beams
        .iter()
        .filter(|b| {
            let mid = 0.5 * (pos[b.a] + pos[b.b]);
            (pos[b.a] - pos[b.b]).length() > 0.9
                && (0.30..0.90).contains(&s_of(mid.z))
                && b.class == Class::Brace
        })
        .count();
    let mut skm = sum_k_over_m(st, &lat.beams);
    r.sum_k_m_max = skm.iter().copied().fold(0.0, f64::max);
    skm.sort_by(|a, b| a.total_cmp(b));
    r.sum_k_m_p50 = skm[n / 2];
    r.omega_h = omega_h(st, &lat.beams);
    let (mut low, mut high, mut all) = (0.0, 0.0, 0.0);
    for b in &lat.beams {
        let d = (pos[b.a] - pos[b.b]).normalize();
        if d.z.abs() < 0.707 {
            continue;
        }
        let mid = 0.5 * (pos[b.a] + pos[b.b]);
        let s = s_of(mid.z);
        r.crush[station(s)] += b.deform;
        if s < 0.2 {
            all += b.deform;
            if mid.y < 0.45 {
                low += b.deform;
            } else if mid.y > 0.75 {
                high += b.deform;
            }
        }
    }
    for &(a, b, deform) in others {
        let mid = 0.5 * (a + b);
        if (a - b).normalize().z.abs() >= 0.707 && s_of(mid.z) < 0.2 && deform.is_finite() {
            all += deform;
            if mid.y < 0.45 {
                low += deform;
            } else if mid.y > 0.75 {
                high += deform;
            }
        }
    }
    r.low_share = low / all.max(1e-9);
    r.high_share = high / all.max(1e-9);
    r
}

impl Report {
    /// The hard rules this lattice breaks (empty: it passes). Rule numbers
    /// are the study's (§10.7).
    pub fn failures(&self) -> Vec<String> {
        let mut f = Vec::new();
        if !(110..=140).contains(&self.nodes) {
            f.push(format!("rule 1: {} shell nodes (110-140)", self.nodes));
        }
        if !self.not_in_tetrahedron.is_empty() {
            f.push(format!(
                "rule 3: no tetrahedron at {}",
                self.not_in_tetrahedron.join(", ")
            ));
        }
        if !(5.0..=5.8).contains(&self.beams_per_node) {
            f.push(format!(
                "rule 4: {:.2} beams per node (5.0-5.8)",
                self.beams_per_node
            ));
        }
        if self.neighbours_min < 5 {
            f.push(format!(
                "rule 4: a node with {} neighbours (5 at least)",
                self.neighbours_min
            ));
        }
        if self.spread_min.0 < 0.02 {
            f.push(format!(
                "rule 5: beam-direction spread {:.3} at {} (0.02 at least)",
                self.spread_min.0, self.spread_min.1
            ));
        }
        if self.long_cabin > 6 {
            f.push(format!(
                "rule 7: {} braces over 0.9 m through the cabin (6 at most)",
                self.long_cabin
            ));
        }
        if self.sum_k_m_max > NODE_BUDGET * 1.001 {
            f.push(format!(
                "rule 9: a node at Σk/m {:.2e} (budget {:.2e})",
                self.sum_k_m_max, NODE_BUDGET
            ));
        }
        if self.omega_h > 1.95 {
            f.push(format!("rule 10: ω·h {:.3} (1.95 at most)", self.omega_h));
        }
        // rule 12: the crush force rises from the nose to the firewall, and
        // the cabin holds at least twice the front's peak
        let c = &self.crush;
        if !(c[0] <= c[1] && c[1] <= c[2]) {
            f.push(format!(
                "rule 12: crush force does not rise nose to firewall ({:.0}, {:.0}, {:.0} kN)",
                c[0] / 1e3,
                c[1] / 1e3,
                c[2] / 1e3
            ));
        }
        if c[3].min(c[4]) < 2.0 * c[0].max(c[1]) {
            f.push(format!(
                "rule 12: the cabin ({:.0} kN) holds under twice the front's peak ({:.0} kN)",
                c[3].min(c[4]) / 1e3,
                c[0].max(c[1]) / 1e3
            ));
        }
        if self.low_share < 0.55 {
            f.push(format!(
                "rule 13: the low band carries {:.0} % of the front's yield (55 % at least)",
                self.low_share * 100.0
            ));
        }
        if self.high_share > 0.06 {
            f.push(format!(
                "rule 13: the hood band carries {:.0} % of the front's yield (6 % at most)",
                self.high_share * 100.0
            ));
        }
        f
    }
}

/// The mechanical parts' internal beams and links as (end, end, yield
/// force), for [`report`]'s load path by height.
pub fn other_beams(st: &Structure, parts: &[MechPart]) -> Vec<(DVec3, DVec3, f64)> {
    let mut pos: HashMap<&str, DVec3> = st.nodes.iter().map(|n| (n.name.as_str(), n.pos)).collect();
    pos.extend(
        parts
            .iter()
            .flat_map(|p| &p.nodes)
            .map(|n| (n.name.as_str(), n.pos)),
    );
    let mut out = Vec::new();
    for p in parts {
        for i in &p.internal {
            out.extend(
                i.pairs
                    .iter()
                    .map(|&(a, b)| (p.nodes[a].pos, p.nodes[b].pos, i.v.deform)),
            );
        }
        out.extend(
            p.links
                .iter()
                .map(|l| (pos[l.a.as_str()], pos[l.b.as_str()], l.v.deform)),
        );
    }
    out
}
