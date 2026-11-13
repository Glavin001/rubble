//! The car's body, powertrain, running gear and interior from the design's
//! own definitions.
//!
//! `engine_vehiclegen::structure` defines the body-in-white once (members,
//! the shell's named nodes and their masses) and
//! `engine_vehiclegen::mechanical` every mechanical and interior part (nodes,
//! masses, internal beams, mounts as named break groups, beam-only members).
//! [`super::shell`] braces the shell's nodes. This module turns those into
//! the rigger's car: nothing here invents a member, a mass or a strength.
//! What it adds is listed, with its reason:
//!
//! * **Engine stops and exhaust rests** (SUPPORT, with a gap): the block
//!   against the firewall, rails, towers, radiator support and subframe; the
//!   exhaust against the floor. They stand for contact between solids that
//!   the node-against-triangle contact does not see inside the car
//!   (`beamng-structure-study.md` §10.5).
//! * **Glass trigger beams**: a centre node with the pane's mass and spokes
//!   to its aperture; the aperture's own shell beams carry the pane's deform
//!   group (§10.5, glass is visual only).
//! * **Mount preload**: a soft-mounted part's mount beams are precompressed
//!   to carry its weight at the design position (an engine sits on its
//!   mounts loaded; without it the block sags a centimetre at spawn).
//! * **Wheel stops**: a front wheel driven back meets the sill and the
//!   A-pillar's base once its tyre is crushed (BOUNDED, as before).
//!
//! The raycast wheel (the solver's) puts each wheel's 20 kg on its two axle
//! nodes, so a wheel part's own five nodes carry less than their definition
//! ([`WHEEL_AXLE_KG`], [`WHEEL_FRAME_KG`]), and the step's bound is taken
//! without the solver's 20 kg, so the car is also stable on pressure wheels.
//!
//! Every defined beam keeps its spring, damping, yield and break force
//! ([`super::build::Fixed`]); only the explicit step's bound may lower a
//! spring, and [`report`] prints what was asked for beside what was built.
//!
//! glTF frame in the definitions (x left, y up, z forward); the rigger's
//! nodes are JBeam frame ([`Rig::to_jbeam`]).

use super::build::{Bind, Fixed, FixedKind, Part, Role, Torsion, Wheel, EDGE_MASS};
use super::rig::{Rig, RigInput};
use super::shell::{self, Zone};
use engine_vehiclegen::mechanical::{BeamKind, MechPart, Vals};
use engine_vehiclegen::structure::Structure;
use glam::{DVec3, Vec3};
use std::collections::{BTreeMap, HashMap};

/// A wheel part's axle node (kg). The definition gives the wheel 19 kg, but
/// the raycast wheel (the solver's, a fixed 20 kg) puts a whole wheel on the
/// two axle nodes itself, so the part's own nodes carry only what the step
/// needs under the wheel's mount beams when the car runs without raycast
/// wheels (pressure wheels: the crash scenes).
pub const WHEEL_AXLE_KG: f32 = 4.0;
/// A wheel part's frame node (kg): three carry the rim's and tyre's meshes.
pub const WHEEL_FRAME_KG: f32 = 0.5;
/// Engine stops: (spring N/m, damping, yield N, rest length over spawn
/// length): §10.5's band (1–3e6, 150, 20–50 kN, 0.85–0.95).
pub const ENGINE_STOP: (f32, f32, f32, f32) = (1.0e6, 150.0, 35e3, 0.90);
/// Stops per block corner.
pub const ENGINE_STOPS_EACH: usize = 5;
/// Exhaust rests on the floor (two per chain node).
pub const EXHAUST_REST: (f32, f32, f32, f32) = (1.5e5, 50.0, 4e3, 0.90);
/// A rubber mount's free travel as a share of its beam's length, and its
/// limit spring (N/m).
pub const MOUNT_TRAVEL: f32 = 0.05;
pub const MOUNT_LIMIT_K: f32 = 2.0e6;
/// Mounts softer than this (N/m) are preloaded to carry their part.
pub const PRELOAD_BELOW: f64 = 3.0e5;
/// Windscreen and rear screen (kg, trigger ratio, spoke yield N, break N).
pub const SCREENS: [(&str, f32, f32, f32, f32); 2] = [
    ("glass_windshield", 12.0, 0.02, 9_500.0, 19e3),
    ("glass_rear", 8.0, 0.02, 9_500.0, 19e3),
];
/// A torsion bar's equivalent spring on an arm node (k / arm²) as a share of
/// that node's stiffness budget.
pub const TORSION_SHARE: f32 = 0.10;
/// A skin's nodes must span this much (m, root mean square) in their
/// thinnest direction for its fit (`flexskin`: 12 mm at every vertex).
pub const THICK: f32 = 0.04;
/// The whole car's bound on ω·h (the power iteration of
/// [`super::build::Car::assign_springs`]; the measured cars: 1.80–1.90).
pub const OMEGA_H: f32 = 1.85;
/// A body node with no defined mass (a latch striker, a hinge's body leaf):
/// kg, taken from the shell's total.
pub const BODY_HARDWARE_KG: f32 = 0.6;

/// The design's definitions.
pub struct Defs<'a> {
    pub structure: &'a Structure,
    pub parts: &'a [MechPart],
}

/// What was built from them.
pub struct Built {
    /// Every defined node by name.
    pub ids: HashMap<String, usize>,
    /// The front and rear impact beams' nodes.
    pub impact: [Vec<usize>; 2],
    /// The shell's nodes (the structure's order).
    pub shell: Vec<usize>,
}

fn v3(p: DVec3) -> Vec3 {
    p.as_vec3()
}

/// The member tag the closures' hardware looks for on a shell node.
fn node_role(group: &str, name: &str) -> Role {
    let tail = name.rsplit('_').next().unwrap_or("");
    match group {
        "A1" | "A2" | "A4" | "A5" => Role::Rail,
        "A3" => Role::Cross,
        "A6" | "A7" | "A8" => Role::Apron,
        "B1" => Role::Firewall,
        // the cowl's outer two are the A-pillars' bases
        "B2" if tail.starts_with('o') => Role::PillarA,
        "B2" => Role::Firewall,
        // the floor rows' outer lane is the sill
        "C1" if tail.len() >= 2 && tail.as_bytes()[1] == b'o' => Role::Sill,
        "C1" | "E3" => Role::Floor,
        "C2" => Role::Sill,
        "C3" => Role::Tunnel,
        "D1" => Role::PillarA,
        "D2" => Role::PillarB,
        "D3" => Role::PillarC,
        "D4" => Role::RoofBow,
        "E1" | "E2" => Role::RearRail,
        "E5" => Role::Sheet,
        _ => Role::Cross,
    }
}

fn beam_role(z: Zone) -> Role {
    match z {
        Zone::CrashBox => Role::CrushCan,
        Zone::FrontRail => Role::Rail,
        Zone::Tower => Role::Apron,
        Zone::Cabin => Role::Floor,
        Zone::Primary => Role::Sill,
        Zone::Pillar => Role::PillarB,
        Zone::Bow => Role::RoofBow,
        Zone::RearRail => Role::RearRail,
        Zone::Tail => Role::Cross,
    }
}

fn fixed_of(v: &Vals, kind: FixedKind) -> Fixed {
    Fixed {
        k: v.k.map(|k| k as f32),
        c: v.c as f32,
        deform: v.deform as f32,
        kind,
        limit_expansion: 0.0,
        deform_group: None,
    }
}

/// A defined beam of `sub` (a second request for a pair keeps the first).
#[allow(clippy::too_many_arguments)]
fn put(
    r: &mut Rig,
    a: usize,
    b: usize,
    role: Role,
    sub: &str,
    f: Fixed,
    strength: f32,
    group: Option<&str>,
) -> Option<usize> {
    if a == b || r.car.has_beam(a, b) {
        return None;
    }
    let j = r.car.beam(a, b, role, Part::Engine);
    let bm = &mut r.car.beams[j];
    bm.sub = Some(sub.to_string());
    bm.strength = strength;
    if strength.is_finite() {
        bm.break_group = group.map(str::to_string);
    }
    r.car.fixed.insert(j, f);
    Some(j)
}

/// `n` pairs between `from` and `to`: each `from` node's nearest `to` node,
/// then each one's second nearest, and so on (so every `from` node is held
/// before any is held twice).
fn mount_pairs(r: &Rig, from: &[usize], to: &[usize], n: usize) -> Vec<(usize, usize)> {
    let order: Vec<Vec<usize>> = from
        .iter()
        .map(|&f| {
            let g = r.g(f);
            let mut v: Vec<(f32, usize)> = to
                .iter()
                .filter(|&&t| t != f)
                .map(|&t| (r.g(t).distance(g), t))
                .collect();
            v.sort_by(|a, b| a.0.total_cmp(&b.0));
            v.into_iter().map(|x| x.1).collect()
        })
        .collect();
    let mut out = Vec::new();
    for round in 0..to.len() {
        for (fi, &f) in from.iter().enumerate() {
            if out.len() == n {
                return out;
            }
            if let Some(&t) = order[fi].get(round) {
                out.push((f, t));
            }
        }
    }
    out
}

/// Solve the 6 × 6 system `a x = b` (Gaussian elimination, partial pivot).
fn solve6(mut a: [[f64; 6]; 6], mut b: [f64; 6]) -> Option<[f64; 6]> {
    for i in 0..6 {
        let p = (i..6).max_by(|&x, &y| a[x][i].abs().total_cmp(&a[y][i].abs()))?;
        if a[p][i].abs() < 1e-9 {
            return None;
        }
        a.swap(i, p);
        b.swap(i, p);
        for k in i + 1..6 {
            let f = a[k][i] / a[i][i];
            for c in i..6 {
                a[k][c] -= f * a[i][c];
            }
            b[k] -= f * b[i];
        }
    }
    let mut x = [0.0; 6];
    for i in (0..6).rev() {
        let s: f64 = (i + 1..6).map(|c| a[i][c] * x[c]).sum();
        x[i] = (b[i] - s) / a[i][i];
    }
    Some(x)
}

/// Precompress `beams` (each with its spring, from a node of `body` to a
/// node outside it) so they carry `body`'s weight at the design position:
/// the least strain energy that balances its weight and leaves no moment.
fn preload(r: &mut Rig, body: &[usize], beams: &[(usize, f64)], mass_of: &dyn Fn(usize) -> f64) {
    if beams.len() < 6 {
        return;
    }
    let m: f64 = body.iter().map(|&i| mass_of(i)).sum();
    let com = body
        .iter()
        .fold(DVec3::ZERO, |s, &i| s + r.car.p(i).as_dvec3() * mass_of(i))
        / m;
    // JBeam frame: z up. Per beam its force on the body per metre of rest
    // length added: k along the beam, from the outside end to the body's.
    let mut rows: Vec<([f64; 6], f64)> = Vec::new();
    for &(j, k) in beams {
        let b = &r.car.beams[j];
        let (inn, out) = if body.contains(&b.a) { (b.a, b.b) } else { (b.b, b.a) };
        let (pi, po) = (r.car.p(inn).as_dvec3(), r.car.p(out).as_dvec3());
        let u = (pi - po).normalize_or_zero();
        let t = (pi - com).cross(u);
        rows.push(([u.x * k, u.y * k, u.z * k, t.x * k, t.y * k, t.z * k], k));
    }
    let mut a = [[0.0; 6]; 6];
    for (row, k) in &rows {
        for x in 0..6 {
            for y in 0..6 {
                a[x][y] += row[x] * row[y] / k;
            }
        }
    }
    let rhs = [0.0, 0.0, m * 9.81, 0.0, 0.0, 0.0];
    let Some(lam) = solve6(a, rhs) else {
        return;
    };
    for (&(j, _), (row, k)) in beams.iter().zip(&rows) {
        let delta: f64 = (0..6).map(|x| row[x] * lam[x]).sum::<f64>() / k;
        let b = &r.car.beams[j];
        let l = r.car.p(b.a).distance(r.car.p(b.b)) as f64;
        let pc = ((l + delta) / l).clamp(0.8, 1.25);
        let old = r.car.precomp.get(&j).copied().unwrap_or(1.0);
        r.car.precomp.insert(j, old * pc as f32);
        // (a rubber mount's free travel is about its loaded position)
        if let Some(Fixed {
            kind: FixedKind::Bounded { travel, .. },
            ..
        }) = r.car.fixed.get_mut(&j)
        {
            *travel = travel.max((pc as f32 - 1.0).abs() + MOUNT_TRAVEL);
        }
    }
}

/// The shell, then every mechanical and interior part, on `r`.
pub fn build(r: &mut Rig, inp: &dyn RigInput, d: &Defs) -> Built {
    let st = d.structure;
    let log = std::env::var("RIG_LOG").is_ok();
    let mut ids: HashMap<String, usize> = HashMap::new();
    // ---- the shell's nodes, each with its defined mass
    let mut shell_ids = Vec::new();
    for n in &st.nodes {
        let g = v3(n.pos);
        let role = node_role(n.group, &n.name);
        let on_skin = inp.sdf(g).abs() < 0.03
            && matches!(
                role,
                Role::Sheet
                    | Role::PillarA
                    | Role::PillarB
                    | Role::PillarC
                    | Role::RoofBow
                    | Role::Sill
            );
        let i = r.node(&n.name, g, Part::Body, Some(role), on_skin);
        r.car.nodes[i].fixed_mass = n.mass as f32;
        ids.insert(n.name.clone(), i);
        shell_ids.push(i);
        if log {
            eprintln!(
                "shell {} {} ({:.2} {:.2} {:.2}) {:.2} kg {:?} sdf {:.3} part {}",
                n.name,
                n.group,
                g.x,
                g.y,
                g.z,
                n.mass,
                role,
                inp.sdf(g),
                st.members[n.member].part
            );
        }
    }
    // ---- its lattice: springs, damping and yield as built by `shell`
    let lat = shell::lattice(st);
    for b in &lat.beams {
        let j = r
            .car
            .beam(shell_ids[b.a], shell_ids[b.b], beam_role(b.zone), Part::Body);
        r.car.fixed.insert(
            j,
            Fixed {
                k: Some(b.k as f32),
                c: b.c as f32,
                deform: b.deform as f32,
                kind: FixedKind::Normal,
                limit_expansion: b.limit_expansion as f32,
                deform_group: None,
            },
        );
    }
    // The floor's torsion bars. A bar of spring k (N·m/rad) acts on its two
    // arm nodes like a spring k / arm² each: the bars across the tunnel have
    // 0.24 m arms, where the defined 6.5e5 is 1.1e7 N/m on a 4 kg node, more
    // than its whole budget (the floor beams there yielded at rest). Each
    // bar is held to [`TORSION_SHARE`] of its lightest node's budget.
    for t in &lat.torsion {
        let n = t.nodes.map(|i| shell_ids[i]);
        let p = n.map(|i| r.g(i));
        let axis = (p[2] - p[1]).normalize_or_zero();
        let arm = [0usize, 3]
            .map(|k| {
                let e = p[k] - p[1];
                (e - axis * e.dot(axis)).length()
            })
            .into_iter()
            .fold(f32::MAX, f32::min);
        let m = t
            .nodes
            .iter()
            .map(|&i| st.nodes[i].mass as f32)
            .fold(f32::MAX, f32::min);
        let cap = TORSION_SHARE * shell::NODE_BUDGET as f32 * m * arm * arm;
        if log {
            eprintln!(
                "torsion {}-{}-{}-{}: arm {arm:.2} m, {:.2e} of {:.2e} N·m/rad",
                r.car.nodes[n[0]].name,
                r.car.nodes[n[1]].name,
                r.car.nodes[n[2]].name,
                r.car.nodes[n[3]].name,
                (t.k as f32).min(cap),
                t.k
            );
        }
        r.car.torsion.push(Torsion {
            n,
            k: (t.k as f32).min(cap),
            c: 0.0,
            deform: t.deform as f32,
        });
    }
    // ---- the mechanical and interior parts' nodes
    for p in d.parts {
        for n in &p.nodes {
            let i = r.node(&n.name, v3(n.pos), Part::Engine, None, false);
            let nd = &mut r.car.nodes[i];
            nd.sub = Some(p.name.clone());
            nd.fixed_mass = if p.class == "wheel" && p.name.starts_with("wheel_") {
                if n.name.ends_with("_ai") || n.name.ends_with("_ao") {
                    WHEEL_AXLE_KG
                } else {
                    WHEEL_FRAME_KG
                }
            } else {
                n.mass as f32
            };
            ids.insert(n.name.clone(), i);
        }
    }
    let id = |ids: &HashMap<String, usize>, n: &str| -> usize {
        *ids.get(n).unwrap_or_else(|| panic!("no defined node {n}"))
    };
    // The driver (the design's kerb mass is with one): on the driver's seat,
    // half on the cushion, a quarter on each of the back's corners.
    let driver = inp.physics().named.get("driver").copied().unwrap_or(75.0);
    for (node, share) in [("seafl_c", 0.5), ("seafl_bo", 0.25), ("seafl_bi", 0.25)] {
        if let Some(&i) = ids.get(node) {
            r.car.nodes[i].fixed_mass += driver * share;
        }
    }
    // ---- their internal beams, mounts and links
    let mut soft: BTreeMap<String, Vec<(usize, f64)>> = BTreeMap::new();
    for p in d.parts {
        let own: Vec<usize> = p.nodes.iter().map(|n| id(&ids, &n.name)).collect();
        for it in &p.internal {
            for &(a, b) in &it.pairs {
                let mut f = fixed_of(&it.v, FixedKind::Normal);
                f.deform_group = it.deform_group.clone().map(|(g, t)| (g, t as f32));
                put(r, own[a], own[b], Role::Engine, &p.name, f, it.v.strength as f32, None);
            }
        }
        for m in &p.mounts {
            let from: Vec<usize> = m.from.iter().map(|n| id(&ids, n)).collect();
            let to: Vec<usize> = m.to.iter().map(|n| id(&ids, n)).collect();
            let kind = match m.kind {
                BeamKind::Normal => FixedKind::Normal,
                BeamKind::Support => FixedKind::Support,
                BeamKind::Bounded => FixedKind::Bounded {
                    travel: MOUNT_TRAVEL,
                    limit_k: MOUNT_LIMIT_K,
                    limit_c: m.v.c as f32,
                },
            };
            let role = if m.kind == BeamKind::Support {
                Role::Bearing
            } else {
                Role::Mount
            };
            let pairs = mount_pairs(r, &from, &to, m.beams);
            if pairs.len() != m.beams {
                panic!("mount {}: {} of {} beams", m.group, pairs.len(), m.beams);
            }
            for (a, b) in pairs {
                let Some(j) = put(
                    r,
                    a,
                    b,
                    role,
                    &p.name,
                    fixed_of(&m.v, kind),
                    m.v.strength as f32,
                    Some(&m.group),
                ) else {
                    continue;
                };
                if m.precomp != 1.0 {
                    r.car.precomp.insert(j, m.precomp as f32);
                }
                if let Some(k) = m.v.k.filter(|&k| k < PRELOAD_BELOW) {
                    soft.entry(p.name.clone()).or_default().push((j, k));
                }
            }
        }
        for l in &p.links {
            put(
                r,
                id(&ids, &l.a),
                id(&ids, &l.b),
                Role::Susp,
                &p.name,
                fixed_of(&l.v, FixedKind::Normal),
                l.v.strength as f32,
                l.group.as_deref(),
            );
        }
    }
    // ---- preload: the soft-mounted parts carry their weight at the design
    // position (the engine and the transmission bolted to it are one body)
    let nodes_of = |r: &Rig, names: &[&str]| -> Vec<usize> {
        (0..r.car.nodes.len())
            .filter(|&i| {
                r.car.nodes[i]
                    .sub
                    .as_deref()
                    .is_some_and(|s| names.contains(&s))
            })
            .collect()
    };
    let mut done: Vec<&str> = Vec::new();
    for group in [vec!["engine", "transmission"]] {
        let body = nodes_of(r, &group);
        let beams: Vec<(usize, f64)> = group
            .iter()
            .flat_map(|g| soft.get(*g).cloned().unwrap_or_default())
            .collect();
        let mass = |i: usize| r.car.nodes[i].fixed_mass as f64;
        let masses: HashMap<usize, f64> = body.iter().map(|&i| (i, mass(i))).collect();
        preload(r, &body, &beams, &|i| masses[&i]);
        done.extend(group);
    }
    for (name, beams) in &soft {
        // (not the exhaust: a chain is not one body; its hangers carry
        // 18 kg and it sags 4 mm)
        if done.contains(&name.as_str()) || name == "exhaust" {
            continue;
        }
        let body = nodes_of(r, &[name.as_str()]);
        let masses: HashMap<usize, f64> = body
            .iter()
            .map(|&i| (i, r.car.nodes[i].fixed_mass as f64))
            .collect();
        preload(r, &body, beams, &|i| masses[&i]);
    }
    // ---- engine stops: each block corner against the structure around it
    if let Some(eng) = d.parts.iter().find(|p| p.name == "engine") {
        let around: Vec<usize> = (0..r.car.nodes.len())
            .filter(|&i| {
                let n = &r.car.nodes[i];
                (n.part == Part::Body
                    && matches!(
                        r.member[i],
                        Some(Role::Firewall)
                            | Some(Role::Rail)
                            | Some(Role::Apron)
                            | Some(Role::Cross)
                    ))
                    || n.sub.as_deref() == Some("subframe_front")
            })
            .filter(|&i| r.g(i).z > v3(eng.nodes[0].pos).z - 1.2)
            .collect();
        let (k, c, deform, pc) = ENGINE_STOP;
        for n in eng.nodes.iter().take(8) {
            let e = id(&ids, &n.name);
            for (a, b) in mount_pairs(r, &[e], &around, ENGINE_STOPS_EACH) {
                let f = Fixed {
                    k: Some(k),
                    c,
                    deform,
                    kind: FixedKind::Support,
                    limit_expansion: 0.0,
                    deform_group: None,
                };
                if let Some(j) = put(r, a, b, Role::Bearing, "engine", f, f32::INFINITY, None) {
                    r.car.precomp.insert(j, pc);
                }
            }
        }
    }
    // ---- exhaust rests: each chain node on the floor above it
    if let Some(ex) = d.parts.iter().find(|p| p.name == "exhaust") {
        let floor: Vec<usize> = (0..r.car.nodes.len())
            .filter(|&i| {
                r.car.nodes[i].part == Part::Body
                    && matches!(
                        r.member[i],
                        Some(Role::Floor)
                            | Some(Role::Tunnel)
                            | Some(Role::Sill)
                            | Some(Role::RearRail)
                            | Some(Role::Firewall)
                    )
            })
            .collect();
        let (k, c, deform, pc) = EXHAUST_REST;
        for n in &ex.nodes {
            let e = id(&ids, &n.name);
            for (a, b) in mount_pairs(r, &[e], &floor, 2) {
                let f = Fixed {
                    k: Some(k),
                    c,
                    deform,
                    kind: FixedKind::Support,
                    limit_expansion: 0.0,
                    deform_group: None,
                };
                if let Some(j) = put(r, a, b, Role::Bearing, "exhaust", f, f32::INFINITY, None) {
                    r.car.precomp.insert(j, pc);
                }
            }
        }
    }
    // ---- the screens: a centre node with the pane's mass on spokes to its
    // aperture; the aperture's own shell beams trigger the pane too
    for (mesh, kg, ratio, deform, strength) in SCREENS {
        let Some(m) = inp.model().parts.iter().find(|p| p.name == mesh) else {
            continue;
        };
        let c = m.pos.iter().fold(Vec3::ZERO, |s, &p| s + p) / m.pos.len().max(1) as f32;
        // (on the pane: the vertex nearest its centroid)
        let c = *m
            .pos
            .iter()
            .min_by(|a, b| a.distance(c).total_cmp(&b.distance(c)))
            .unwrap();
        let mut ring: Vec<(f32, usize)> = shell_ids
            .iter()
            .map(|&i| {
                let g = r.g(i);
                let dmin = m
                    .pos
                    .iter()
                    .step_by(7)
                    .map(|q| q.distance(g))
                    .fold(f32::MAX, f32::min);
                (dmin, i)
            })
            .filter(|x| x.0 < 0.16)
            .collect();
        ring.sort_by(|a, b| a.0.total_cmp(&b.0));
        let ring: Vec<usize> = ring.into_iter().map(|x| x.1).collect();
        if ring.len() < 4 {
            panic!("{mesh}: {} shell nodes round its aperture", ring.len());
        }
        let name = format!("gl_{}", mesh.trim_start_matches("glass_"));
        let ci = r.node(&name, c, Part::Engine, None, false);
        r.car.nodes[ci].sub = Some(mesh.to_string());
        r.car.nodes[ci].fixed_mass = kg;
        ids.insert(name, ci);
        for (k, &t) in ring.iter().enumerate() {
            let f = Fixed {
                k: None,
                c: 120.0,
                deform,
                kind: FixedKind::Normal,
                limit_expansion: 0.0,
                deform_group: Some((mesh.to_string(), ratio)),
            };
            // two spokes hold the pane until they break
            let (st, grp) = if k < 2 {
                (strength, Some(mesh))
            } else {
                (f32::INFINITY, None)
            };
            put(r, ci, t, Role::Sheet, mesh, f, st, grp);
        }
        for (ka, &a) in ring.iter().enumerate() {
            for &b in &ring[ka + 1..] {
                if !r.car.has_beam(a, b) {
                    continue;
                }
                let j = r.car.beam(a, b, Role::Web, Part::Body);
                if let Some(f) = r.car.fixed.get_mut(&j) {
                    if f.deform_group.is_none() {
                        f.deform_group = Some((mesh.to_string(), ratio));
                    }
                }
            }
        }
    }
    // ---- which nodes each solid follows
    // The body-in-white is one welded part: each of its solids follows the
    // shell's own 131 nodes (not the body's shut-line edge nodes or latch
    // strikers). Its render parts (front, floor, cabin, rear) are a split of
    // the drawing, not of the structure: the cabin's eight nodes alone do
    // not span a fit.
    for (name, _) in st.parts() {
        r.car.bind.insert(
            name,
            Bind {
                nodes: shell_ids.clone(),
                owner: None,
                rigid: false,
            },
        );
    }
    for p in d.parts {
        let mut nodes: Vec<usize> = p.nodes.iter().map(|n| id(&ids, &n.name)).collect();
        let add = |nodes: &mut Vec<usize>, i: usize| {
            if !nodes.contains(&i) {
                nodes.push(i);
            }
        };
        for b in &p.binds {
            add(&mut nodes, id(&ids, b));
        }
        // A skin's fit needs six nodes around it that span all three
        // directions ([`THICK`]: a flat ladder or a chain alone does not).
        // A part short of that takes, in order and only until it has them:
        // the nodes its mounts land on, its links' ends, the nodes of the
        // parts mounted on it, then the rest of the parts those belong to.
        const NEED: usize = 6;
        let short = |r: &Rig, nodes: &[usize]| {
            if nodes.len() < NEED {
                return true;
            }
            let c = nodes.iter().fold(Vec3::ZERO, |s, &i| s + r.g(i)) / nodes.len() as f32;
            let m = nodes.iter().fold(glam::Mat3::ZERO, |m, &i| {
                let e = r.g(i) - c;
                m + glam::Mat3::from_cols(e * e.x, e * e.y, e * e.z)
            });
            let (lam, _) = super::build::weakest_axis(m);
            (lam.max(0.0) / nodes.len() as f32).sqrt() < THICK
        };
        'grow: {
            for m in &p.mounts {
                for t in &m.to {
                    if !short(r, &nodes) {
                        break 'grow;
                    }
                    add(&mut nodes, id(&ids, t));
                }
            }
            for l in &p.links {
                if !short(r, &nodes) {
                    break 'grow;
                }
                add(&mut nodes, id(&ids, &l.a));
                add(&mut nodes, id(&ids, &l.b));
            }
            for q in d.parts {
                for m in &q.mounts {
                    if m.to.iter().any(|t| p.nodes.iter().any(|n| &n.name == t)) {
                        for f in &m.from {
                            if !short(r, &nodes) {
                                break 'grow;
                            }
                            add(&mut nodes, id(&ids, f));
                        }
                    }
                }
            }
            let subs: Vec<String> = nodes
                .iter()
                .filter_map(|&i| r.car.nodes[i].sub.clone())
                .collect();
            for i in 0..r.car.nodes.len() {
                if !short(r, &nodes) {
                    break 'grow;
                }
                if r.car.nodes[i].sub.as_ref().is_some_and(|s| subs.contains(s)) {
                    add(&mut nodes, i);
                }
            }
            // (last: the shell beside the shell nodes it already follows)
            let on_shell: Vec<usize> = nodes
                .iter()
                .copied()
                .filter(|i| shell_ids.contains(i))
                .collect();
            for i in on_shell {
                for (_, t) in mount_pairs(r, &[i], &shell_ids, 3) {
                    if !short(r, &nodes) {
                        break 'grow;
                    }
                    add(&mut nodes, t);
                }
            }
        }
        if log {
            eprintln!(
                "bind {}: {} nodes ({} its own)",
                p.name,
                nodes.len(),
                p.nodes.len()
            );
        }
        // One frame for the pieces that are rigid in the definition (a
        // casting, a wheel); the pressed and tubular parts follow their
        // nodes one by one.
        let rigid = matches!(p.class, "engine" | "wheel")
            || ["transmission", "differential", "steering_rack", "hub_"]
                .iter()
                .any(|x| p.name.starts_with(x));
        for mesh in std::iter::once(&p.name).chain(&p.meshes) {
            if mesh == &p.name && p.mesh.v.is_empty() {
                continue;
            }
            r.car.bind.insert(
                mesh.clone(),
                Bind {
                    nodes: nodes.clone(),
                    owner: Some(p.name.clone()),
                    rigid: rigid || mesh != &p.name,
                },
            );
        }
    }
    let impact = ["impact_beam_front", "impact_beam_rear"].map(|n| {
        d.parts
            .iter()
            .find(|p| p.name == n)
            .map(|p| p.nodes.iter().map(|x| id(&ids, &x.name)).collect())
            .unwrap_or_default()
    });
    Built {
        ids,
        impact,
        shell: shell_ids,
    }
}

/// Masses: every defined node has its own; the closures keep the design's
/// part masses; the body's undefined nodes (shut-line edge nodes, latch
/// strikers) come out of the shell's total. (The driver is on the seat's
/// nodes already: [`build`].)
pub fn masses(r: &mut Rig, inp: &dyn RigInput, b: &Built) {
    let ph = inp.physics();
    let n = r.car.nodes.len();
    let free = |i: usize| {
        let nd = &r.car.nodes[i];
        nd.part == Part::Body && nd.fixed_mass == 0.0
    };
    let edges = (0..n).filter(|&i| free(i) && r.car.nodes[i].edge).count();
    let others = (0..n).filter(|&i| free(i) && !r.car.nodes[i].edge).count();
    let hardware = edges as f32 * EDGE_MASS + others as f32 * BODY_HARDWARE_KG;
    let shell: f32 = b.shell.iter().map(|&i| r.car.nodes[i].fixed_mass).sum();
    let f = (shell - hardware) / shell;
    assert!(f > 0.8, "the body's hardware nodes take {hardware} kg of the shell's {shell}");
    for &i in &b.shell {
        r.car.nodes[i].fixed_mass *= f;
    }
    r.car.part_mass = [
        (Part::Body, hardware),
        (Part::DoorFL, ph.door_f),
        (Part::DoorFR, ph.door_f),
        (Part::DoorRL, ph.door_r),
        (Part::DoorRR, ph.door_r),
        (Part::Hood, ph.hood),
        (Part::Trunk, ph.decklid),
        (Part::FenderL, ph.fender),
        (Part::FenderR, ph.fender),
        (Part::BumperF, ph.bumper_f),
        (Part::BumperR, ph.bumper_r),
    ]
    .into_iter()
    .filter(|x| x.1 > 0.0)
    .collect();
    r.car.omega_limit = Some(super::build::env_f("RIG_OMEGA_H", OMEGA_H));
    r.car.node_budget = shell::NODE_BUDGET as f32;
}

/// The raycast wheels on the wheel parts' axle nodes, the wheel stops and
/// the reference nodes.
pub fn wheels(r: &mut Rig, b: &Built, stack: f32) {
    let id = |n: &str| *b.ids.get(n).unwrap_or_else(|| panic!("no defined node {n}"));
    for (name, c, s) in [
        ("FL", "fl", 1.0f32),
        ("FR", "fr", -1.0),
        ("RL", "rl", 1.0),
        ("RR", "rr", -1.0),
    ] {
        let (n1, n2) = (id(&format!("whe{c}_ai")), id(&format!("whe{c}_ao")));
        // (The solver ties a raycast wheel's arm node to its axle nodes
        // with beams that never yield: the arm is the wheel's own inner
        // axle node, so the wheel leaves when its mount group breaks.)
        r.car.wheels.push(Wheel {
            name,
            n1,
            n2,
            arm: n1,
            dir: s,
        });
        if !name.starts_with('F') || std::env::var("GEN3_NO_WHEEL_STACK").is_ok() {
            continue;
        }
        // A front wheel driven back meets the sill's front and the
        // A-pillar's base once its tyre is crushed onto the rim.
        let x = if s > 0.0 { "l" } else { "r" };
        for h in [n1, n2] {
            for t in [format!("sh_c2_1{x}"), format!("sh_c1_1o{x}"), format!("sh_b2_o{x}")] {
                let Some(&t) = b.ids.get(&t) else { continue };
                let l = r.g(t).distance(r.g(h));
                if l <= stack + 0.02 {
                    continue;
                }
                let j = r.car.beam(h, t, Role::Stop, Part::Body);
                r.car.beams[j].role = Role::Stop;
                r.car.stop_close.insert(j, l - stack);
            }
        }
    }
    r.car.refs = [id("sh_c3_2"), id("sh_c3_3"), id("sh_c2_2l"), id("sh_d4_b")];
}

/// What each defined part asked for and what the step's bound left it:
/// (part, beams, Σ rule spring, Σ built spring), the shell first.
pub fn report(car: &super::build::Car) -> Vec<(String, usize, f64, f64)> {
    let mut rows: BTreeMap<String, (usize, f64, f64)> = BTreeMap::new();
    for (&j, _) in &car.fixed {
        let b = &car.beams[j];
        let name = b.sub.clone().unwrap_or_else(|| "shell".into());
        let e = rows.entry(name).or_default();
        e.0 += 1;
        e.1 += car.k_rule.get(j).copied().unwrap_or(b.k) as f64;
        e.2 += b.k as f64;
    }
    let mut out: Vec<(String, usize, f64, f64)> =
        rows.into_iter().map(|(n, v)| (n, v.0, v.1, v.2)).collect();
    out.sort_by_key(|r| r.0 != "shell");
    out
}
