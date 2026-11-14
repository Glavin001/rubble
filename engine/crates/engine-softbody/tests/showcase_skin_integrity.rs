//! The showcase sedan's visual skin stays sound under any abuse: crashes into
//! a wall (once and repeatedly), a car into the rear door, random impacts,
//! and the page's node grab pulling panels 0.5, 2 and 10 m and letting go.
//! The car and its skin are the page's (`softbody.html?solver=explicit&car=
//! showcase`, drive yard): the medium sedan on raycast wheels
//! (`skin::raycast_visual`), the explicit solver at its defaults with drag.
//! What is drawn is the renderer's: [`deform_beams`] positions, triangles cut
//! where the tear ([`vertex_tears`]) passes ½. Measured
//! (engine/docs/measurements/skin-shards.md).
//!
//! Invariants, each from the physics, not tuned:
//! - **Rest and rigid motion are exact** (0.1 mm: f32 sums at car scale are
//!   ~10⁻⁶ m; a tenth of a millimetre is far under a pixel).
//! - **No shards**: no drawn triangle has an edge, as drawn, longer than
//!   [`SHARD_STRETCH`] × its rest length and [`SHARD_MIN`] longer than at rest.
//!   Steel parts at 20–40 % elongation and the tear field cuts by 57 %
//!   (½ on its 1.35–1.8 ramp), so intact sheet stretches at most ~1.6×;
//!   bending a panel's thickness of skin (its offset from the nodes, under
//!   10 cm) round a fold adds at most π·h. 3× is past both.
//! - **No triangle spans a break**: every node that drives a drawn
//!   triangle's vertices is in one piece of the structure (connected by
//!   live beams), or the triangle is cut.
//! - **No saw-tooth**: no drawn vertex inside the drawn skin (its whole
//!   ring drawn, in three triangles or more: one lone triangle is the shard
//!   test's) moves more than
//!   [`JUMP_EDGES`] of its mean rest edge away from the mean of its mesh
//!   neighbours' motion (an intact sheet cannot without tearing).
//! - **Flips only where the structure folded**: drawn triangles facing away
//!   from their rest orientation are at most the share of the skeleton's own
//!   skin triangles that face away from theirs, plus [`FLIP_SLACK`].
//! - **Cut only where the structure broke** (coverage): a triangle is cut
//!   (any of its vertices torn past ½ or dropped) only where a beam has
//!   actually broken: a broken beam joins two of the nodes that carry one of
//!   its vertices (its bound nodes at rest), or its glass shattered (its
//!   damage unit's beams broke). Never for stretch, fold or disagreement:
//!   dented sheet metal stays a continuous dented panel (BeamNG deforms the
//!   mesh continuously and removes polygons only where a beam breaks). The
//!   bound: cut triangles without a broken beam under them = 0; reported per
//!   part as drawn / rest triangles.
//! - **No holes without a tear**: a part's drawn mesh has no boundary edge
//!   it did not have at rest, except beside a cut that a broken beam
//!   explains.

use engine_softbody::authoring::skin::{
    deform_beams, raycast_visual, showcase_visual_lod, unit_states, vertex_tears, BeamState,
    ShowcaseVisual,
};
use engine_softbody::authoring::{build, style, BuildParams, Built, Resolution};
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, Grab, StaticBox, STEP};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use glam::{Quat, Vec3};
use std::sync::OnceLock;

const SHARD_STRETCH: f32 = 3.0;
const SHARD_MIN: f32 = 0.05;
const JUMP_EDGES: f32 = 3.0;
const FLIP_SLACK: f32 = 0.005;
const EXACT: f32 = 1e-4;

struct Car {
    b: Built,
    v: ShowcaseVisual,
}

fn car() -> &'static Car {
    static CAR: OnceLock<Car> = OnceLock::new();
    CAR.get_or_init(|| {
        let b = build(
            &style("sedan").unwrap(),
            &BuildParams {
                resolution: Resolution::Medium,
                ..Default::default()
            },
        )
        .unwrap();
        let mut v = showcase_visual_lod(&b, 1.0).unwrap();
        raycast_visual(&mut v, &b.truck).unwrap();
        Car { b, v }
    })
}

/// The page's settings (drag and turbulence on).
fn settings() -> ExplicitBeamsSettings {
    ExplicitBeamsSettings::default()
}

/// A raycast spawn centred at `at` (x, z), turned `yaw` degrees, moving at
/// `vel`, its lowest wheel 2 cm up (`ExplicitSim::spawn`), wheels rolling.
fn spawn(at: Vec3, yaw: f32, vel: Vec3) -> ExplicitBeams {
    let b = &car().b;
    let (p, _) =
        ExplicitBeams::from_truck_raycast(&b.truck, None, Vec3::ZERO, yaw, Vec3::ZERO, settings()).unwrap();
    let d = p.drive.as_ref().unwrap();
    let low = d
        .config
        .wheels
        .iter()
        .map(|w| 0.5 * (p.nodes[w.node1 as usize].pos.y + p.nodes[w.node2 as usize].pos.y) - w.radius)
        .fold(f32::MAX, f32::min);
    let (lo, hi) = p
        .nodes
        .iter()
        .fold((Vec3::MAX, Vec3::MIN), |(a, c), n| (a.min(n.pos), c.max(n.pos)));
    let mid = 0.5 * (lo + hi);
    let pos = Vec3::new(at.x - mid.x, 0.02 - low, at.z - mid.z);
    let (mut s, _) = ExplicitBeams::from_truck_raycast(&b.truck, None, pos, yaw, vel, settings()).unwrap();
    let sp = vel.length();
    if let Some(d) = s.drive.as_mut() {
        let r: Vec<f32> = d.config.wheels.iter().map(|w| w.radius).collect();
        for (w, r) in d.wheels.iter_mut().zip(r) {
            w.omega = sp / r;
        }
    }
    s
}

fn beam_states(s: &ExplicitBeams) -> Vec<BeamState> {
    s.beams
        .iter()
        .map(|b| BeamState {
            broken: b.broken,
            rest: b.rest,
        })
        .collect()
}

/// Live-beam components of the nodes.
fn components(s: &ExplicitBeams) -> Vec<u32> {
    let mut p: Vec<u32> = (0..s.nodes.len() as u32).collect();
    fn find(p: &mut [u32], mut i: u32) -> u32 {
        while p[i as usize] != i {
            p[i as usize] = p[p[i as usize] as usize];
            i = p[i as usize];
        }
        i
    }
    for b in s.beams.iter().filter(|b| !b.broken) {
        let (x, y) = (find(&mut p, b.a), find(&mut p, b.b));
        if x != y {
            p[x as usize] = y;
        }
    }
    (0..s.nodes.len() as u32).map(|i| find(&mut p, i)).collect()
}

/// The nodes that drive vertex `i` now.
fn drivers(v: &ShowcaseVisual, i: usize, tears: &[f32]) -> Vec<u32> {
    if let Some(ids) = v.bl_node.get(i).filter(|b| b[0] != u32::MAX) {
        return ids
            .iter()
            .zip(&v.bl_w[i])
            .filter(|(&j, &w)| j != u32::MAX && w > 0.0 && tears[j as usize] < 1.0)
            .map(|(&j, _)| j)
            .collect();
    }
    if v.rigid[i] >= 2 {
        return Vec::new();
    }
    v.bind_nodes[i].to_vec()
}

/// The drawn length of an edge: its part where the tear, linear along it, is
/// under ½.
fn drawn(len: f32, ta: f32, tb: f32) -> f32 {
    match (ta < 0.5, tb < 0.5) {
        (true, true) => len,
        (false, false) => 0.0,
        _ => {
            let (lo, hi) = (ta.min(tb), ta.max(tb));
            len * (0.5 - lo) / (hi - lo).max(1e-9)
        }
    }
}

#[derive(Debug, Default)]
struct Report {
    drawn: usize,
    shards: usize,
    longest: f32,
    spans: usize,
    jumpy: usize,
    flipped: usize,
    lattice_flip: f32,
    nan_nodes: usize,
    /// Cut triangles (any vertex torn past ½ or dropped).
    cut: usize,
    /// Of them with no broken beam on the nodes carrying them (and no
    /// shattered glass): must be 0.
    cut_unexplained: usize,
    /// Boundary edges of the drawn mesh that were not boundary at rest and
    /// whose cut neighbour no broken beam explains: must be 0.
    boundary_unexplained: usize,
    /// Per part (name): rest triangles, drawn (uncut) triangles, cut without
    /// a broken beam, boundary edges at rest and now.
    parts: std::collections::BTreeMap<String, [usize; 5]>,
}

impl Report {
    fn check(&self, tag: &str) {
        let Report { parts, .. } = self;
        eprintln!(
            "{tag}: drawn {} shards {} longest {:.2} spans {} jumpy {} flipped {} lattice_flip {:.4} nan {} cut {} cut_unexplained {} boundary_unexplained {}",
            self.drawn, self.shards, self.longest, self.spans, self.jumpy, self.flipped, self.lattice_flip,
            self.nan_nodes, self.cut, self.cut_unexplained, self.boundary_unexplained
        );
        for (name, [all, drawn, bad, b0, b1]) in parts {
            if drawn < all {
                eprintln!(
                    "    {name}: {drawn}/{all} drawn ({:.1} %), {bad} cut without a break, boundary edges {b0} → {b1}",
                    100.0 * *drawn as f32 / *all as f32
                );
            }
        }
        assert_eq!(self.nan_nodes, 0, "{tag}: non-finite nodes");
        assert_eq!(
            self.cut_unexplained, 0,
            "{tag}: {} triangles cut with no broken beam under them",
            self.cut_unexplained
        );
        assert_eq!(
            self.boundary_unexplained, 0,
            "{tag}: {} new boundary edges (holes) with no broken beam under them",
            self.boundary_unexplained
        );
        assert_eq!(self.shards, 0, "{tag}: {} shards (longest {:.2} m)", self.shards, self.longest);
        assert_eq!(self.spans, 0, "{tag}: {} drawn triangles span a break", self.spans);
        assert_eq!(self.jumpy, 0, "{tag}: {} saw-tooth vertices", self.jumpy);
        let share = self.flipped as f32 / self.drawn.max(1) as f32;
        assert!(
            share <= self.lattice_flip + FLIP_SLACK,
            "{tag}: {:.3} % of drawn triangles flipped, the skeleton's skin {:.3} %",
            100.0 * share,
            100.0 * self.lattice_flip
        );
    }
}

fn measure(s: &ExplicitBeams) -> Report {
    let v = &car().v;
    let nodes = s.positions();
    let mut r = Report {
        nan_nodes: nodes.iter().filter(|p| !p.is_finite()).count(),
        ..Default::default()
    };
    if r.nan_nodes > 0 {
        return r;
    }
    let bs = beam_states(s);
    let us = unit_states(v, &bs);
    let (pos, nrm) = deform_beams(v, &nodes, Some(&us), Some(&bs));
    let tear = vertex_tears(v, &nodes, Some(&bs), &pos);
    let ntears = engine_softbody::authoring::node_frames::node_tears(v, &nodes);
    let comp = components(s);
    let rp = |i: u32| Vec3::from_array(v.positions[i as usize]);
    let n = v.positions.len();
    let (mut nb, mut cnt, mut edge) = (vec![Vec3::ZERO; n], vec![0u32; n], vec![0f32; n]);
    // Each vertex's edges in the mesh: the saw-tooth test takes only vertices
    // whose whole ring is drawn (a cut's edge has a one-sided ring).
    let mut ring = vec![0u32; n];
    for t in v.indices.chunks_exact(3) {
        for &i in t {
            ring[i as usize] += 2;
        }
    }
    for t in v.indices.chunks_exact(3) {
        let ids = [t[0] as usize, t[1] as usize, t[2] as usize];
        let p = ids.map(|i| pos[i]);
        if p.iter().any(|q| !q.x.is_finite()) {
            continue;
        }
        let tv = ids.map(|i| tear[i]);
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            if tv[a] < 0.5 && tv[b] < 0.5 {
                nb[ids[a]] += p[b] - rp(t[b]);
                nb[ids[b]] += p[a] - rp(t[a]);
                cnt[ids[a]] += 1;
                cnt[ids[b]] += 1;
                let l0 = rp(t[a]).distance(rp(t[b]));
                edge[ids[a]] += l0;
                edge[ids[b]] += l0;
            }
        }
        if tv.iter().all(|&x| x >= 0.5) {
            continue;
        }
        r.drawn += 1;
        let mut shard = false;
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            let l0 = rp(t[a]).distance(rp(t[b]));
            let l = drawn(p[a].distance(p[b]), tv[a], tv[b]);
            if l > SHARD_STRETCH * l0 && l > l0 + SHARD_MIN {
                shard = true;
                r.longest = r.longest.max(l);
            }
        }
        r.shards += shard as usize;
        if shard && std::env::var("DUMP").is_ok() {
            let piece = v.pieces.iter().find(|pc| (pc.1 as usize..(pc.1 + pc.2) as usize).contains(&ids[0])).map_or("?", |pc| pc.0.as_str());
            eprintln!("  shard on {piece}: rest {:?} now {:?} tears {:?}",
                [(0, 1), (1, 2), (2, 0)].map(|(a, b)| rp(t[a]).distance(rp(t[b]))),
                [(0, 1), (1, 2), (2, 0)].map(|(a, b)| p[a].distance(p[b])), tv);
            for &i in &ids {
                eprintln!("    v{i} rigid {} part {} nodes {:?} w {:?} node tears {:?} moved {:?}", v.rigid[i], v.part[i], v.bl_node[i], v.bl_w[i],
                    v.bl_node[i].map(|j| if j == u32::MAX { -1.0 } else { ntears[j as usize] }), pos[i] - rp(i as u32));
            }
        }
        let mut c: Vec<u32> = ids.iter().flat_map(|&i| drivers(v, i, &ntears)).map(|j| comp[j as usize]).collect();
        c.sort_unstable();
        c.dedup();
        r.spans += (c.len() > 1) as usize;
        let r0 = ids.map(|i| rp(i as u32));
        let f0 = (r0[1] - r0[0]).cross(r0[2] - r0[0]);
        let f = (p[1] - p[0]).cross(p[2] - p[0]);
        let n0: Vec3 = ids.iter().map(|&i| Vec3::from_array(v.normals[i])).sum();
        let n1: Vec3 = ids.iter().map(|&i| nrm[i]).sum();
        if f0.dot(n0) > 0.0 && f.length_squared() > 1e-18 && f.dot(n1) < 0.0 {
            r.flipped += 1;
        }
    }
    for i in 0..n {
        if ring[i] < 6 || cnt[i] < ring[i] || tear[i] >= 0.5 || !pos[i].x.is_finite() {
            continue;
        }
        let d = pos[i] - rp(i as u32);
        let mean = nb[i] / cnt[i] as f32;
        let e = (edge[i] / cnt[i] as f32).max(0.005);
        if (d - mean).length() > JUMP_EDGES * e {
            r.jumpy += 1;
            if std::env::var("DUMP").is_ok() {
                let piece = v.pieces.iter().find(|pc| (pc.1 as usize..(pc.1 + pc.2) as usize).contains(&i)).map_or("?", |pc| pc.0.as_str());
                eprintln!("  jumpy v{i} on {piece}: off {:.3} m vs mean edge {:.3}, {} neighbours, tear {:.2}, nodes {:?} w {:?}", (d - mean).length(), e, cnt[i], tear[i], v.bl_node[i], v.bl_w[i]);
            }
        }
    }
    // Coverage: which triangles are cut, and whether a broken beam explains
    // each (an end of a broken beam among the nodes carrying it at rest, or
    // its glass shattered with its damage unit).
    let broken: std::collections::HashSet<(u32, u32)> = s
        .beams
        .iter()
        .filter(|b| b.broken)
        .map(|b| (b.a.min(b.b), b.a.max(b.b)))
        .collect();
    let zero = vec![0.0f32; nodes.len()];
    // A vertex is at a break when a broken beam joins two of the nodes that
    // carry it.
    let at_break = |i: usize| {
        let d = drivers(v, i, &zero);
        d.iter().any(|&a| d.iter().any(|&b| a < b && broken.contains(&(a, b))))
    };
    let shattered = |i: usize| {
        let u = v.unit.get(i).copied().unwrap_or(u32::MAX);
        u != u32::MAX && us[u as usize].broken
    };
    let explained = |i: usize| shattered(i) || at_break(i);
    let part_name = |i: usize| match v.part.get(i).copied().unwrap_or(u16::MAX) {
        u16::MAX => "structure".to_string(),
        p => car().b.skeleton.parts[p as usize].clone(),
    };
    let mut edges: std::collections::HashMap<(u32, u32), [u32; 3]> = Default::default();
    for t in v.indices.chunks_exact(3) {
        let ids = [t[0] as usize, t[1] as usize, t[2] as usize];
        let name = part_name(ids[0]);
        let cut = ids.iter().any(|&i| !pos[i].x.is_finite() || tear[i] >= 0.5);
        let ok = !cut || ids.iter().any(|&i| explained(i));
        let e = r.parts.entry(name).or_default();
        e[0] += 1;
        e[1] += !cut as usize;
        e[2] += !ok as usize;
        r.cut += cut as usize;
        r.cut_unexplained += !ok as usize;
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            let key = (t[a].min(t[b]), t[a].max(t[b]));
            let c = edges.entry(key).or_default();
            c[0] += 1; // triangles on the edge (all drawn at rest)
            c[1] += !cut as u32; // drawn now
            c[2] += !ok as u32; // cut with no broken beam under it
        }
    }
    for (&(a, _), &[n0, d, u]) in &edges {
        let name = part_name(a as usize);
        let e = r.parts.get_mut(&name).unwrap();
        // A boundary edge: exactly one drawn triangle on it.
        e[3] += (n0 == 1) as usize;
        e[4] += (d == 1) as usize;
        // New, and a hole no break explains: it was inside the mesh, and a
        // triangle beside it was cut without a broken beam.
        r.boundary_unexplained += (n0 >= 2 && d == 1 && u > 0) as usize;
    }
    // The skeleton's own skin: triangles (of one part) facing away from their
    // rest orientation, by count.
    let rest = |i: u32| Vec3::from_array(v.node_rest[i as usize]);
    let (mut flip, mut all) = (0usize, 0usize);
    for t in &v.normal_tris {
        let f0 = (rest(t[1]) - rest(t[0])).cross(rest(t[2]) - rest(t[0]));
        let q = t.map(|k| nodes[k as usize]);
        let f = (q[1] - q[0]).cross(q[2] - q[0]);
        all += 1;
        // Rigid motion aside: against the mean rotation of the car is not
        // known here, so compare with the part's own neighbours' normals:
        // a triangle facing away from the mean normal of its nodes' other
        // triangles has folded over.
        let mut m = Vec3::ZERO;
        for &k in t {
            let k = k as usize;
            for &tj in &v.node_tri_list[v.node_tri_start[k] as usize..v.node_tri_start[k + 1] as usize] {
                let u = v.normal_tris[tj as usize];
                let qu = u.map(|x| nodes[x as usize]);
                let ru = u.map(rest);
                let fu = (qu[1] - qu[0]).cross(qu[2] - qu[0]);
                let fu0 = (ru[1] - ru[0]).cross(ru[2] - ru[0]);
                m += fu.normalize_or_zero() * fu0.dot(f0).signum();
            }
        }
        if f.dot(m) < 0.0 {
            flip += 1;
        }
    }
    r.lattice_flip = flip as f32 / all.max(1) as f32;
    r
}

// ---- Rest and rigid motion ---------------------------------------------------

#[test]
fn at_rest_and_under_rigid_motion_the_skin_is_exact() {
    let v = &car().v;
    let s = spawn(Vec3::ZERO, 0.0, Vec3::ZERO);
    let base = s.positions();
    // The spawn is the rest pose translated.
    let shift = base[0] - Vec3::from_array(v.node_rest[0]);
    for (motion, q) in [
        ("rest", Quat::IDENTITY),
        ("turned", Quat::from_euler(glam::EulerRot::YXZ, 1.1, 0.4, -2.3)),
    ] {
        let nodes: Vec<Vec3> = v
            .node_rest
            .iter()
            .map(|p| q * Vec3::from_array(*p) + shift + Vec3::new(3.0, 1.0, -7.0))
            .collect();
        let (pos, nrm) = deform_beams(v, &nodes, None, None);
        let tear = vertex_tears(v, &nodes, None, &pos);
        let mut worst = 0.0f32;
        let mut worst_n = 0.0f32;
        for i in 0..pos.len() {
            if v.rigid[i] >= 2 {
                continue; // wheels: drawn from the drive state
            }
            let want = q * Vec3::from_array(v.positions[i]) + shift + Vec3::new(3.0, 1.0, -7.0);
            worst = worst.max(pos[i].distance(want));
            worst_n = worst_n.max((nrm[i] - q * Vec3::from_array(v.normals[i]).normalize()).length());
            assert!(tear[i] < 0.5, "{motion}: vertex {i} torn");
        }
        eprintln!("{motion}: worst {worst:.2e} m, normal {worst_n:.2e}");
        assert!(worst < EXACT, "{motion}: off by {worst} m");
        assert!(worst_n < 1e-3, "{motion}: normal off by {worst_n}");
    }
}

// ---- Crashes -------------------------------------------------------------------

/// A wall square to the axis the car mostly moves along, 0.6 m ahead of
/// its front; the car given `kmh` again `hits` times (backed off to 1 m from
/// the wall between), measured during and after each.
fn wall(kmh: f32, yaw: f32, hits: u32, tag: &str) {
    let vel = Quat::from_rotation_y(yaw.to_radians()) * Vec3::new(0.0, 0.0, kmh / 3.6);
    let mut s = spawn(Vec3::ZERO, yaw, vel);
    // The axis (x or z) and its sign.
    let (axis, sign) = if vel.z.abs() >= vel.x.abs() {
        (Vec3::Z, vel.z.signum())
    } else {
        (Vec3::X, vel.x.signum())
    };
    let reach = |s: &ExplicitBeams| s.positions().iter().map(|q| q.dot(axis) * sign).fold(f32::MIN, f32::max);
    let face = reach(&s) + 0.6;
    let centre = axis * sign * (face + 0.3) + Vec3::Y * 1.5;
    let half = Vec3::splat(8.0) - axis * 7.7 + Vec3::Y * (1.5 - 8.0);
    s.boxes.push(StaticBox::centered(centre, half));
    for hit in 0..hits {
        if hit > 0 {
            let back = axis * sign * (reach(&s) - (face - 1.0));
            for n in &mut s.nodes {
                n.pos -= back;
                n.vel = vel;
            }
        }
        for f in 0..90 {
            s.frame();
            if f == 20 {
                measure(&s).check(&format!("{tag} hit {} t 0.33 s", hit + 1));
            }
        }
        measure(&s).check(&format!("{tag} hit {} t 1.5 s", hit + 1));
    }
}

#[test]
fn wall_at_70_kmh() {
    wall(70.0, 0.0, 1, "wall 70");
}

#[test]
fn wall_at_90_kmh_three_times() {
    wall(90.0, 0.0, 3, "wall 90 ×3");
}

/// The struck car parked, a second car square into its left rear door.
#[test]
fn car_into_the_rear_door_at_60_kmh() {
    let b = &car().b;
    let struck = spawn(Vec3::ZERO, 0.0, Vec3::ZERO);
    let p0 = struck.positions();
    let rear = b.skeleton.parts.iter().position(|p| p == "door_r_l").unwrap() as u16;
    let door: Vec<Vec3> = (0..b.skeleton.nodes.len())
        .filter(|&i| b.skeleton.nodes[i].part == rear)
        .map(|i| p0[i])
        .collect();
    let zd = door.iter().map(|p| p.z).sum::<f32>() / door.len() as f32;
    let hw = p0.iter().map(|p| p.x).fold(f32::MIN, f32::max);
    let len = b.def.profile.length;
    let striker = spawn(Vec3::new(hw + 0.5 * len + 0.3, 0.0, zd), -90.0, Vec3::new(-60.0 / 3.6, 0.0, 0.0));
    let mut w = ExplicitBeamsWorld::new(vec![struck, striker]);
    for f in 0..120u32 {
        w.frame_60();
        if f == 15 || f == 119 {
            measure(&w.vehicles[0]).check(&format!("side 60 struck f {f}"));
            measure(&w.vehicles[1]).check(&format!("side 60 striker f {f}"));
        }
    }
}

/// Random impacts (seeded): 50–110 km/h into a wall, the car turned up to
/// ±35° from square.
#[test]
fn random_impacts() {
    let mut seed = 0x2545_f491_u32;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed as f32 / u32::MAX as f32
    };
    for k in 0..3 {
        let kmh = 50.0 + 60.0 * rnd();
        let yaw = -35.0 + 70.0 * rnd();
        wall(kmh, yaw, 1, &format!("random {k}: {kmh:.0} km/h at {yaw:.0}°"));
    }
}

// ---- The node grab -------------------------------------------------------------

/// The page's grab (crash-lab.mjs: 50 kN/m default; damping as `setGrab`):
/// the part's outermost node dragged `dist` m outward over 0.5 s, held 1 s,
/// released, 1 s more. Measured held and after.
fn grab(part: &str, dist: f32, k: f32) {
    let b = &car().b;
    let mut s = spawn(Vec3::ZERO, 0.0, Vec3::ZERO);
    s.run(0.5);
    let pi = b.skeleton.parts.iter().position(|p| p == part).unwrap() as u16;
    let p = s.positions();
    let c = p.iter().copied().sum::<Vec3>() / p.len() as f32;
    let node = (0..b.skeleton.nodes.len())
        .filter(|&i| b.skeleton.nodes[i].part == pi)
        .max_by(|&x, &y| (p[x] - c).length().total_cmp(&(p[y] - c).length()))
        .unwrap();
    let dir = ((p[node] - c) * Vec3::new(1.0, 0.3, 1.0)).normalize();
    let m = s.nodes[node].mass;
    let cd = (2.0 * (k * m).sqrt()).min(0.5 * m / STEP);
    for f in 0..150u32 {
        s.grab = (f < 90).then(|| Grab {
            node: node as u32,
            target: p[node] + dir * dist * (f as f32 / 30.0).min(1.0),
            k,
            c: cd,
        });
        s.frame();
        if f == 89 {
            measure(&s).check(&format!("grab {part} {dist} m held"));
        }
    }
    measure(&s).check(&format!("grab {part} {dist} m released"));
}

fn grab_parts(parts: &[&str]) {
    for part in parts {
        for dist in [0.5, 2.0, 10.0] {
            grab(part, dist, 5.0e4);
        }
    }
}

#[test]
fn grab_doors() {
    grab_parts(&["door_f_l", "door_r_r"]);
}

#[test]
fn grab_front() {
    grab_parts(&["bonnet", "fender_l", "bumper_f"]);
}

#[test]
fn grab_rear_and_roof() {
    grab_parts(&["bootlid", "bumper_r", "fender_r", "body"]);
}

// ---- Physics: the grab at the page's strongest setting --------------------------

/// The lab's grab slider reaches 10^6.5 ≈ 3.16 MN/m (crash-lab.mjs
/// `labGrabK`). An explicit step is stable only while h·√(k/m) < 2 for the
/// grab's own spring on its node; the bumper's outermost node (0.23 kg) at
/// 3.16 MN/m has h·√(k/m) = 1.86 (h = 0.5 ms) before its beams add theirs. Pulling a
/// light front-bumper node 0.5 m at that setting must not blow the car up:
/// every node finite, no node faster than the 5 m/s the drag implies plus the
/// solver's own crash speeds (50 m/s bounds both by far).
#[test]
#[ignore = "known failure (engine/docs/measurements/skin-shards.md): at 3.16 MN/m the grabbed bumper node is past the explicit step's stability limit and the car goes non-finite"]
fn grab_at_the_strongest_setting_keeps_the_car_finite() {
    let b = &car().b;
    let mut s = spawn(Vec3::ZERO, 0.0, Vec3::ZERO);
    s.run(0.5);
    let pi = b.skeleton.parts.iter().position(|p| p == "bumper_f").unwrap() as u16;
    let p = s.positions();
    let c = p.iter().copied().sum::<Vec3>() / p.len() as f32;
    let node = (0..b.skeleton.nodes.len())
        .filter(|&i| b.skeleton.nodes[i].part == pi)
        .max_by(|&x, &y| (p[x] - c).length().total_cmp(&(p[y] - c).length()))
        .unwrap();
    let k = 10f32.powf(6.5);
    let m = s.nodes[node].mass;
    eprintln!("bumper node {node}: {m:.3} kg; h·√(k/m) = {:.2}", STEP * (k / m).sqrt());
    let dir = ((p[node] - c) * Vec3::new(1.0, 0.3, 1.0)).normalize();
    let cd = (2.0 * (k * m).sqrt()).min(0.5 * m / STEP);
    let mut fastest = 0.0f32;
    for f in 0..60u32 {
        s.grab = Some(Grab {
            node: node as u32,
            target: p[node] + dir * 0.5 * (f as f32 / 30.0).min(1.0),
            k,
            c: cd,
        });
        s.frame();
        fastest = s.nodes.iter().map(|n| n.vel.length()).fold(fastest, f32::max);
        assert!(s.nodes.iter().all(|n| n.pos.is_finite() && n.vel.is_finite()), "frame {f}: non-finite nodes");
    }
    assert!(fastest < 50.0, "a node reached {fastest} m/s");
}
