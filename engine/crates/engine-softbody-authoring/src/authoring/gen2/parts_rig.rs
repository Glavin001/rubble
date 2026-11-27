//! The parts-only rigger: a design made only of defined parts
//! (`engine_vehiclegen::mechanical::MechPart`, the heavy vehicles of
//! `engine_vehiclegen::heavy`: a ladder frame, axles, a cab, tanks), with no
//! unibody shell (its body-in-white is empty). Nothing is invented here:
//!
//! * every node, mass, internal beam, mount (a named break group) and link is
//!   the definition's ([`super::build::Fixed`]); a beam with no spring of its
//!   own takes its nodes' budget (Σk/m, (`build::NODE_BUDGET`)) and the explicit
//!   step's bound (ω·h ≤ [`OMEGA_H`], the power iteration of
//!   [`super::build::Car::assign_springs`]) is the only thing that may lower
//!   a spring;
//! * a mount's beams are its nearest pairs, every `from` node held before any
//!   twice, none shorter than [`MOUNT_MIN_LEN`] (a shorter beam cannot yield in
//!   compression: the solver keeps a rest length of 0.1 m or more);
//! * the collision skin: each colliding part's closed convex hull over its own
//!   colliding nodes, facing out, every hull vertex a node (no T-junctions:
//!   the hull's planar faces are ear-clipped over all their boundary nodes).
//!   A part that collides is therefore convex in its nodes (the definitions'
//!   rule, `heavy` module docs). Wheels and axles do not collide (raycast
//!   wheels, as on the cars);
//! * the raycast wheels on the wheel parts' `_ai` / `_ao` axle nodes
//!   (parts of class `wheel` named `wheel_<corner>`), each the solver's
//!   [`WHEEL_KG`] (the definitions carry the rest of a wheel on its nodes);
//! * the reference nodes on the frame (class `frame`): the lowest right
//!   node at the front axle, the one behind it, the one left of it, the one
//!   over it;
//! * the visual binding: each mesh over its part's nodes and the nodes it
//!   names (`binds`), grown by its mounts' targets where its own nodes do not
//!   span a fit; wheels and engines one rigid frame;
//! * triangle aerodynamics sized to the design's C_d, as the cars'.

use super::build::{Bind, Fixed, FixedKind, Part, Role, Tri, Wheel};
use super::rig::{Rig, RigInput, WHEEL_KG};
use super::vg_input::VgInput;
use engine_vehiclegen::mechanical::{BeamKind, MechPart, Vals};
use glam::{DVec3, Vec3};
use std::collections::HashMap;

/// The whole vehicle's bound on ω·h (as the defined cars', `body2::OMEGA_H`).
pub const OMEGA_H: f32 = 1.85;
/// The spring a defined beam with none of its own asks for: Σk/m per node
/// (s⁻²), shared by the node's beams. The cars' census value
/// ((`build::NODE_BUDGET`), 8.5e6) leaves the 7.5 t truck at ω·h 1.50, a third of
/// the step's room unused, and its frame sagging 6 mm parked; the step's
/// bound ([`OMEGA_H`], the power iteration) is the hard limit, so a parts-only
/// rig asks for more and lets the bound take back what the step cannot carry
/// (the truck: ω·h 1.84, 15 % of the asked springs taken back).
pub const HEAVY_BUDGET: f32 = 2.0e7;
/// A mount's beams are at least this long (m).
pub const MOUNT_MIN_LEN: f32 = 0.12;
/// A skin's nodes must span this much (m, root mean square) in their thinnest
/// direction for a fit (as `body2::THICK`).
const THICK: f32 = 0.04;

/// Whether `p` lies on the segment a–b (within `tol`), strictly between its ends.
fn on_segment(p: DVec3, a: DVec3, b: DVec3, tol: f64) -> bool {
    let ab = b - a;
    let t = (p - a).dot(ab) / ab.length_squared().max(1e-18);
    t > 1e-9 && t < 1.0 - 1e-9 && (a + ab * t).distance(p) <= tol
}

/// A design rigged by this path: an empty body-in-white and defined parts.
pub fn applies(inp: &VgInput) -> bool {
    inp.structure
        .as_ref()
        .is_some_and(|s| s.nodes.is_empty() && s.members.is_empty())
        && !inp.mechanical.is_empty()
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

/// Up to `n` pairs, each `from` node's nearest `to` node first, then each
/// one's second nearest, …; none shorter than [`MOUNT_MIN_LEN`], none twice.
fn mount_pairs(r: &Rig, from: &[usize], to: &[usize], n: usize) -> Vec<(usize, usize)> {
    let order: Vec<Vec<usize>> = from
        .iter()
        .map(|&f| {
            let g = r.g(f);
            let mut v: Vec<(f32, usize)> = to
                .iter()
                .filter(|&&t| t != f)
                .map(|&t| (r.g(t).distance(g), t))
                .filter(|x| x.0 >= MOUNT_MIN_LEN)
                .collect();
            v.sort_by(|a, b| a.0.total_cmp(&b.0));
            v.into_iter().map(|x| x.1).collect()
        })
        .collect();
    let mut out: Vec<(usize, usize)> = Vec::new();
    for round in 0..to.len() {
        for (fi, &f) in from.iter().enumerate() {
            if out.len() == n {
                return out;
            }
            if let Some(&t) = order[fi].get(round) {
                if !out
                    .iter()
                    .any(|&(a, b)| (a, b) == (f, t) || (a, b) == (t, f))
                    && !r.car.has_beam(f, t)
                {
                    out.push((f, t));
                }
            }
        }
    }
    out
}

/// The closed convex hull of `pts` as triangles over the points themselves,
/// facing out: each planar face is ear-clipped over every point on its
/// boundary (no T-junction, no sliver of zero area).
pub fn hull(pts: &[DVec3]) -> Vec<[usize; 3]> {
    let n = pts.len();
    if n < 4 {
        return Vec::new();
    }
    let c = pts.iter().fold(DVec3::ZERO, |s, p| s + *p) / n as f64;
    let scale = pts
        .iter()
        .map(|p| (*p - c).length())
        .fold(0.0, f64::max)
        .max(1e-6);
    let tol = 1e-6 * scale.max(1.0);
    let mut faces: Vec<(DVec3, Vec<usize>)> = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            for k in j + 1..n {
                let nr = (pts[j] - pts[i]).cross(pts[k] - pts[i]);
                if nr.length() < 1e-9 {
                    continue;
                }
                let mut nrm = nr.normalize();
                let d = |q: DVec3| (q - pts[i]).dot(nrm);
                let (mut pos, mut neg) = (false, false);
                for q in pts {
                    let x = d(*q);
                    if x > tol {
                        pos = true;
                    } else if x < -tol {
                        neg = true;
                    }
                }
                if pos && neg {
                    continue;
                }
                if pos {
                    nrm = -nrm;
                }
                let on: Vec<usize> = (0..n)
                    .filter(|&q| ((pts[q] - pts[i]).dot(nrm)).abs() <= tol)
                    .collect();
                if !faces.iter().any(|f| f.1 == on) {
                    faces.push((nrm, on));
                }
            }
        }
    }
    let mut out = Vec::new();
    for (nrm, on) in faces {
        // the face's points in order round its centre
        let fc = on.iter().fold(DVec3::ZERO, |s, &q| s + pts[q]) / on.len() as f64;
        let u = (pts[on[0]] - fc).normalize();
        let v = nrm.cross(u);
        let mut ring = on.clone();
        ring.sort_by(|&a, &b| {
            let (pa, pb) = (pts[a] - fc, pts[b] - fc);
            pa.dot(v)
                .atan2(pa.dot(u))
                .total_cmp(&pb.dot(v).atan2(pb.dot(u)))
        });
        // ear clipping: the best-shaped ear of non-zero area each time
        while ring.len() > 3 {
            let m = ring.len();
            let mut best: Option<(f64, usize)> = None;
            for e in 0..m {
                let (a, b, cc) = (ring[(e + m - 1) % m], ring[e], ring[(e + 1) % m]);
                let area = (pts[b] - pts[a]).cross(pts[cc] - pts[a]).dot(nrm);
                if area <= 1e-9 {
                    continue;
                }
                let l = [
                    (pts[a] - pts[b]).length(),
                    (pts[b] - pts[cc]).length(),
                    (pts[cc] - pts[a]).length(),
                ];
                let q = area / (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]);
                // An ear whose new edge a–c runs over another point of the ring would
                // leave that point on a triangle's edge (a T-junction) and the last
                // three points collinear, a triangle short: points along a long face's
                // edges (the trailer's box, 17 stations) did, by the greedy order.
                if ring.iter().any(|&o| o != a && o != cc && on_segment(pts[o], pts[a], pts[cc], tol)) {
                    continue;
                }
                if best.is_none_or(|x| q > x.0) {
                    best = Some((q, e));
                }
            }
            let Some((_, e)) = best else { break };
            out.push([ring[(e + m - 1) % m], ring[e], ring[(e + 1) % m]]);
            ring.remove(e);
        }
        if ring.len() == 3 {
            let (a, b, cc) = (ring[0], ring[1], ring[2]);
            if (pts[b] - pts[a]).cross(pts[cc] - pts[a]).dot(nrm) > 1e-9 {
                out.push([a, b, cc]);
            }
        }
    }
    out
}

/// The frontal drag coefficient of the collision skin's triangles (JBeam
/// `dragCoef`), sized so it drags C_d·A head on, A the render model's frontal
/// area (as `rig::hull_drag_coef`).
fn drag_coef(r: &Rig, inp: &dyn RigInput, cd: f32) -> f32 {
    let s: f32 = r
        .car
        .tris
        .iter()
        .filter(|t| !t.inner)
        .map(|t| {
            let [a, b, c] = t.n.map(|i| r.g(i));
            0.5 * (b - a).cross(c - a).z.abs()
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
    100.0 * cd * af / (engine_softbody::explicit_jbeam_ext::FLAT_PLATE_CD * s.max(1e-3))
}

/// Rig `inp` from its defined parts alone.
pub fn rig(inp: &VgInput) -> Rig {
    let parts: &[MechPart] = &inp.mechanical;
    let mut r = Rig {
        car: Default::default(),
        member: Vec::new(),
        skin: Vec::new(),
        axle_f: inp.rig.dims.axle_f,
    };
    let mut ids: HashMap<String, usize> = HashMap::new();
    for p in parts {
        for n in &p.nodes {
            let i = r.node(&n.name, n.pos.as_vec3(), Part::Engine, None, n.collide);
            let nd = &mut r.car.nodes[i];
            nd.sub = Some(p.name.clone());
            nd.fixed_mass = n.mass as f32;
            nd.self_collision = n.collide;
            ids.insert(n.name.clone(), i);
        }
    }
    let id = |n: &str| -> usize { *ids.get(n).unwrap_or_else(|| panic!("no defined node {n}")) };
    for p in parts {
        let own: Vec<usize> = p.nodes.iter().map(|n| id(&n.name)).collect();
        for it in &p.internal {
            for &(a, b) in &it.pairs {
                let mut f = fixed_of(&it.v, FixedKind::Normal);
                f.deform_group = it.deform_group.clone().map(|(g, t)| (g, t as f32));
                put(
                    &mut r,
                    own[a],
                    own[b],
                    Role::Engine,
                    &p.name,
                    f,
                    it.v.strength as f32,
                    None,
                );
            }
        }
        for m in &p.mounts {
            let from: Vec<usize> = m.from.iter().map(|n| id(n)).collect();
            let to: Vec<usize> = m.to.iter().map(|n| id(n)).collect();
            let kind = match m.kind {
                BeamKind::Normal => FixedKind::Normal,
                BeamKind::Support => FixedKind::Support,
                BeamKind::Bounded => panic!(
                    "mount {}: a parts-only rig defines no bounded mounts",
                    m.group
                ),
            };
            let pairs = mount_pairs(&r, &from, &to, m.beams);
            assert!(
                !pairs.is_empty(),
                "mount {}: no pair of its nodes at least {MOUNT_MIN_LEN} m apart",
                m.group
            );
            for (a, b) in pairs {
                if let Some(j) = put(
                    &mut r,
                    a,
                    b,
                    Role::Mount,
                    &p.name,
                    fixed_of(&m.v, kind),
                    m.v.strength as f32,
                    Some(&m.group),
                ) {
                    if m.precomp != 1.0 {
                        r.car.precomp.insert(j, m.precomp as f32);
                    }
                }
            }
        }
        for l in &p.links {
            put(
                &mut r,
                id(&l.a),
                id(&l.b),
                Role::Susp,
                &p.name,
                fixed_of(&l.v, FixedKind::Normal),
                l.v.strength as f32,
                l.group.as_deref(),
            );
        }
    }
    // ---- the collision skin, the wheels and reference nodes, the meshes' nodes
    collision_hulls(&mut r, parts, &ids);
    wheels_and_refs(&mut r, inp, parts, &ids);
    bind_meshes(&mut r, parts, &ids);
    // ---- masses and springs: the definitions', under the step's bound
    let ph = inp.physics();
    r.car.tri_aero = Some((drag_coef(&r, inp, ph.drag_cd), 0.0));
    r.car.centred_damping = true;
    r.car.stops_in_bound = true;
    r.car.row_cap_contact_only = false;
    r.car.node_budget = super::build::env_f("PARTS_RIG_BUDGET", HEAVY_BUDGET);
    r.car.omega_limit = Some(super::build::env_f("RIG_OMEGA_H", OMEGA_H));
    r.car.assign_masses();
    r.car.assign_springs(&HashMap::new());
    mirror_springs(&mut r);
    r
}

/// The collision skin: each colliding part's hull (wheels and axles do not collide).
fn collision_hulls(r: &mut Rig, parts: &[MechPart], ids: &HashMap<String, usize>) {
    let id = |n: &str| -> usize { *ids.get(n).unwrap_or_else(|| panic!("no defined node {n}")) };
    for p in parts
        .iter()
        .filter(|p| !matches!(p.class, "wheel" | "suspension"))
    {
        let own: Vec<usize> = p
            .nodes
            .iter()
            .filter(|n| n.collide)
            .map(|n| id(&n.name))
            .collect();
        let pts: Vec<DVec3> = own.iter().map(|&i| r.g(i).as_dvec3()).collect();
        let inner = p.class == "engine";
        for t in hull(&pts) {
            // (hull() faces out in glTF; the JBeam frame is a rotation of it)
            let n = [own[t[0]], own[t[1]], own[t[2]]];
            r.car.tris.push(Tri {
                n,
                part: Part::Engine,
                group: None,
                sub: Some(p.name.clone()),
                inner,
            });
        }
    }
}

/// The raycast wheels on the wheel parts' axle nodes, and the reference nodes on the frame.
fn wheels_and_refs(r: &mut Rig, inp: &VgInput, parts: &[MechPart], ids: &HashMap<String, usize>) {
    let id = |n: &str| -> usize { *ids.get(n).unwrap_or_else(|| panic!("no defined node {n}")) };
    let sizes = inp.wheel_sizes();
    let dims = inp.dims();
    for p in parts.iter().filter(|p| p.class == "wheel") {
        let corner = p
            .name
            .strip_prefix("wheel_")
            .unwrap_or_else(|| panic!("wheel part {} is not wheel_<corner>", p.name));
        let name: &'static str = match corner {
            "FL" => "FL",
            "FR" => "FR",
            "RL" => "RL",
            "RR" => "RR",
            _ => panic!("wheel corner {corner}"),
        };
        let end = |s: &str| {
            p.nodes
                .iter()
                .find(|n| n.name.ends_with(s))
                .map(|n| id(&n.name))
                .unwrap_or_else(|| panic!("{}: no {s} node", p.name))
        };
        let (n1, n2) = (end("_ai"), end("_ao"));
        let size = sizes
            .map(|sz| sz[if name.starts_with('F') { 0 } else { 1 }])
            .unwrap_or((dims.tyre_r, 0.215));
        r.car.wheels.push(Wheel {
            name,
            n1,
            n2,
            arm: n1,
            dir: if name.ends_with('L') { 1.0 } else { -1.0 },
            size: Some(size),
        });
    }
    r.car.raycast_wheel_kg = Some(WHEEL_KG);
    {
        let frame: Vec<usize> = parts
            .iter()
            .filter(|p| p.class == "frame")
            .flat_map(|p| p.nodes.iter().map(|n| id(&n.name)))
            .collect();
        assert!(
            !frame.is_empty(),
            "a parts-only rig needs a frame (class \"frame\") for its reference nodes"
        );
        let g = |i: usize| r.g(i);
        let lo = frame
            .iter()
            .fold(Vec3::splat(f32::MAX), |a, &i| a.min(g(i)));
        let hi = frame
            .iter()
            .fold(Vec3::splat(f32::MIN), |a, &i| a.max(g(i)));
        let near = |t: Vec3, not: &[usize]| -> usize {
            *frame
                .iter()
                .filter(|i| !not.contains(i))
                .min_by(|&&a, &&b| g(a).distance(t).total_cmp(&g(b).distance(t)))
                .unwrap()
        };
        let rf = near(Vec3::new(lo.x, lo.y, dims.axle_f), &[]);
        let p0 = g(rf);
        let back = near(p0 - Vec3::Z * 0.5, &[rf]);
        let left = near(Vec3::new(hi.x, p0.y, p0.z), &[rf, back]);
        let up = near(Vec3::new(p0.x, hi.y, p0.z), &[rf, back, left]);
        r.car.refs = [rf, back, left, up];
    }
}

/// Which nodes each mesh follows: its part's and the nodes it names, grown by its
/// mounts' targets where they do not span a fit; wheels and engines one rigid frame.
fn bind_meshes(r: &mut Rig, parts: &[MechPart], ids: &HashMap<String, usize>) {
    let id = |n: &str| -> usize { *ids.get(n).unwrap_or_else(|| panic!("no defined node {n}")) };
    for p in parts {
        let mut nodes: Vec<usize> = p.nodes.iter().map(|n| id(&n.name)).collect();
        for b in &p.binds {
            if !nodes.contains(&id(b)) {
                nodes.push(id(b));
            }
        }
        let short = |r: &Rig, nodes: &[usize]| {
            if nodes.len() < 6 {
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
        let rigid = matches!(p.class, "engine" | "wheel");
        if !rigid {
            'grow: for m in &p.mounts {
                for t in &m.to {
                    if !short(r, &nodes) {
                        break 'grow;
                    }
                    if !nodes.contains(&id(t)) {
                        nodes.push(id(t));
                    }
                }
            }
        }
        for mesh in std::iter::once(&p.name).chain(&p.meshes) {
            if mesh == &p.name && p.mesh.v.is_empty() {
                continue;
            }
            r.car.bind.insert(
                mesh.clone(),
                Bind {
                    nodes: nodes.clone(),
                    owner: Some(p.name.clone()),
                    rigid,
                },
            );
        }
    }
}

/// A mirror-symmetric lattice keeps mirror-symmetric springs. The step's bound
/// (`Car::assign_springs`' power iteration) softens the beams that carry the top mode,
/// and a left-right pair of equal modes is found as a combination of the two, so it
/// softened one side more: 184 of the truck's mirrored beams differed and a 30 km/h
/// wall broke the left cab mount alone. Each beam whose mirror image (x → −x, within
/// 1 mm, the design's positions) is a beam takes the lower of the two springs and that
/// one's damping: only ever lower, so the bound still holds. Returns the beams changed.
fn mirror_springs(r: &mut Rig) -> usize {
    let key = |g: Vec3| {
        (
            (g.x * 1e3).round() as i64,
            (g.y * 1e3).round() as i64,
            (g.z * 1e3).round() as i64,
        )
    };
    let at: HashMap<(i64, i64, i64), usize> =
        (0..r.car.nodes.len()).map(|i| (key(r.g(i)), i)).collect();
    let mirror = |i: usize| -> Option<usize> {
        let g = r.g(i);
        at.get(&key(Vec3::new(-g.x, g.y, g.z))).copied()
    };
    let beam_of: HashMap<(usize, usize), usize> = r
        .car
        .beams
        .iter()
        .enumerate()
        .map(|(j, b)| ((b.a.min(b.b), b.a.max(b.b)), j))
        .collect();
    let pairs: Vec<(usize, usize)> = (0..r.car.beams.len())
        .filter_map(|j| {
            let b = &r.car.beams[j];
            let (ma, mb) = (mirror(b.a)?, mirror(b.b)?);
            beam_of
                .get(&(ma.min(mb), ma.max(mb)))
                .copied()
                .filter(|&o| o > j)
                .map(|o| (j, o))
        })
        .collect();
    let mut changed = 0;
    for (j, o) in pairs {
        let (lo, hi) = if r.car.beams[j].k <= r.car.beams[o].k {
            (j, o)
        } else {
            (o, j)
        };
        if r.car.beams[hi].k != r.car.beams[lo].k || r.car.beams[hi].c != r.car.beams[lo].c {
            changed += 1;
            r.car.beams[hi].k = r.car.beams[lo].k;
            r.car.beams[hi].c = r.car.beams[lo].c;
        }
    }
    changed
}

/// The crash reference's gauges of a heavy vehicle (`crash.json`; a car's are
/// `crash_zones::gauges`), node pairs picked from the parts' geometry: each side's
/// wheelbase (the wheels' inner axle nodes), the cab's width at mid height and its height
/// on the left at mid length, the left rail's length, and the front overhang (the
/// bumper's foremost node on the centre plane to the left rail's node nearest the front
/// axle's line). The car's frame: +Z forward, +Y up, +X left.
pub fn gauges(inp: &VgInput, r: &Rig) -> Vec<engine_softbody::crash_reference::Gauge> {
    use engine_softbody::crash_reference::Gauge;
    let id: HashMap<&str, usize> = r
        .car
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.name.as_str(), i))
        .collect();
    let of_class = |class: &str| -> Vec<(Vec3, String)> {
        inp.mechanical
            .iter()
            .filter(|p| p.class == class)
            .flat_map(|p| {
                p.nodes
                    .iter()
                    .filter_map(|n| id.get(n.name.as_str()).map(|&i| (r.g(i), n.name.clone())))
            })
            .collect()
    };
    let nearest = |of: &[(Vec3, String)], t: Vec3| -> Option<String> {
        of.iter()
            .min_by(|a, b| a.0.distance(t).total_cmp(&b.0.distance(t)))
            .map(|x| x.1.clone())
    };
    let bounds = |of: &[(Vec3, String)]| {
        of.iter().fold(
            (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
            |(lo, hi), x| (lo.min(x.0), hi.max(x.0)),
        )
    };
    let gauge = |name: &str, a: Option<String>, b: Option<String>| -> Option<Gauge> {
        Some(Gauge {
            name: name.into(),
            a: a?,
            b: b?,
        })
    };
    // each wheel's centre and its axle node nearest the centre plane
    let wheels: Vec<(Vec3, String)> = inp
        .mechanical
        .iter()
        .filter(|p| p.class == "wheel")
        .filter_map(|p| {
            let pts: Vec<(Vec3, &str)> = p
                .nodes
                .iter()
                .filter_map(|n| id.get(n.name.as_str()).map(|&i| (r.g(i), n.name.as_str())))
                .collect();
            let c = pts.iter().map(|x| x.0).sum::<Vec3>() / pts.len().max(1) as f32;
            let inner = pts
                .iter()
                .min_by(|a, b| a.0.x.abs().total_cmp(&b.0.x.abs()))?;
            Some((c, inner.1.to_string()))
        })
        .collect();
    let wheel = |side: f32, front: bool| -> Option<String> {
        wheels
            .iter()
            .filter(|w| w.0.x * side > 0.0)
            .max_by(|a, b| {
                if front {
                    a.0.z.total_cmp(&b.0.z)
                } else {
                    b.0.z.total_cmp(&a.0.z)
                }
            })
            .map(|w| w.1.clone())
    };
    let axle_z = wheels.iter().map(|w| w.0.z).fold(f32::MIN, f32::max);
    let cab = of_class("cab");
    let (clo, chi) = bounds(&cab);
    let cm = 0.5 * (clo + chi);
    let left = |v: &[(Vec3, String)]| -> Vec<(Vec3, String)> {
        v.iter().filter(|x| x.0.x > 0.0).cloned().collect()
    };
    let cab_l = left(&cab);
    let rail = left(&of_class("frame"));
    let (flo, fhi) = bounds(&rail);
    let bumper = of_class("bumper");
    let xmin = bumper.iter().map(|x| x.0.x.abs()).fold(f32::MAX, f32::min);
    let front = bumper
        .iter()
        .filter(|x| x.0.x.abs() <= xmin + 0.01)
        .max_by(|a, b| a.0.z.total_cmp(&b.0.z))
        .map(|x| x.1.clone());
    [
        gauge("wheelbase_left", wheel(1.0, true), wheel(1.0, false)),
        gauge("wheelbase_right", wheel(-1.0, true), wheel(-1.0, false)),
        gauge(
            "cab_width",
            nearest(&cab, Vec3::new(chi.x, cm.y, cm.z)),
            nearest(&cab, Vec3::new(clo.x, cm.y, cm.z)),
        ),
        gauge(
            "cab_height",
            nearest(&cab_l, Vec3::new(chi.x, clo.y, cm.z)),
            nearest(&cab_l, Vec3::new(chi.x, chi.y, cm.z)),
        ),
        gauge(
            "frame_length",
            nearest(&rail, Vec3::new(fhi.x, flo.y, fhi.z)),
            nearest(&rail, Vec3::new(fhi.x, flo.y, flo.z)),
        ),
        gauge(
            "front_overhang",
            front,
            nearest(&rail, Vec3::new(fhi.x, flo.y, axle_z)),
        ),
    ]
    .into_iter()
    .flatten()
    .chain(trailer_gauges(&of_class, &nearest, &bounds, wheel(1.0, false)))
    .collect()
}

type ClassNodes = Vec<(Vec3, String)>;

/// A semi-trailer's gauges (none on a vehicle without its parts): the kingpin plate's
/// left rear lower node to the rear axle's left inner node (`kingpin_to_axle`: the
/// kingpin-to-rear-axle distance less the plate's half length), the cargo box's floor
/// width and its height on the left at mid length (`box_width`, `box_height`), and the
/// rear sill's lowest left rear node down to the guard bar's (`guard_drop`: how far the
/// guard hangs under the sill; its change in a crash is the guard pushed against the
/// trailer).
fn trailer_gauges(
    of_class: &dyn Fn(&str) -> ClassNodes,
    nearest: &dyn Fn(&[(Vec3, String)], Vec3) -> Option<String>,
    bounds: &dyn Fn(&[(Vec3, String)]) -> (Vec3, Vec3),
    rear_axle_left: Option<String>,
) -> Vec<engine_softbody::crash_reference::Gauge> {
    use engine_softbody::crash_reference::Gauge;
    let pair = |name: &str, a: Option<String>, b: Option<String>| -> Option<Gauge> {
        Some(Gauge {
            name: name.into(),
            a: a?,
            b: b?,
        })
    };
    let low_left_rear = |of: &[(Vec3, String)]| {
        let (lo, hi) = bounds(of);
        nearest(of, Vec3::new(hi.x, lo.y, lo.z))
    };
    let king = of_class("kingpin");
    let bx = of_class("cargo_box");
    let (blo, bhi) = bounds(&bx);
    let mid = 0.5 * (blo.z + bhi.z);
    [
        pair("kingpin_to_axle", low_left_rear(&king), rear_axle_left),
        pair(
            "box_width",
            nearest(&bx, Vec3::new(bhi.x, blo.y, mid)),
            nearest(&bx, Vec3::new(blo.x, blo.y, mid)),
        ),
        pair(
            "box_height",
            nearest(&bx, Vec3::new(bhi.x, blo.y, mid)),
            nearest(&bx, Vec3::new(bhi.x, bhi.y, mid)),
        ),
        pair(
            "guard_drop",
            low_left_rear(&of_class("rear_frame")),
            low_left_rear(&of_class("guard")),
        ),
    ]
    .into_iter()
    .flatten()
    .collect()
}

#[cfg(test)]
mod tests {
    use super::hull;
    use glam::{DVec3, Vec3};

    /// A long box with nodes along its edges (a trailer's cargo box: 17 stations over
    /// 16 m, positions through f32 as the rig holds them): its hull is closed, every node a
    /// vertex, 2V − 4 triangles.
    #[test]
    fn a_long_box_with_collinear_edge_nodes_has_a_closed_hull() {
        let b = engine_vehiclegen::designs::box_trailer::BoxTrailer::new(Default::default()).cargo();
        let mut pts = Vec::new();
        for z in b.stations() {
            for (x, y) in [(1.2914, 1.155), (1.2914, 4.1148), (-1.2914, 1.155), (-1.2914, 4.1148)] {
                // (through the JBeam frame and back, as `Rig::g` gives them)
                let af = engine_vehiclegen::designs::box_trailer::Z_FA as f32;
                let jy = af - z as f32;
                pts.push(Vec3::new(x as f32, y as f32, af - jy).as_dvec3());
            }
        }
        let t = hull(&pts);
        let mut used = vec![false; pts.len()];
        for tri in &t {
            for &i in tri {
                used[i] = true;
            }
        }
        let missing: Vec<usize> = (0..pts.len()).filter(|&i| !used[i]).collect();
        assert!(missing.is_empty(), "points in no triangle: {missing:?}");
        assert_eq!(t.len(), 2 * pts.len() - 4);
        let _ = DVec3::ZERO;
    }
}
