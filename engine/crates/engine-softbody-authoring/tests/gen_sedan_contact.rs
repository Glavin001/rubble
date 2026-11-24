//! Solids never share space: contact invariants of the **generated sedan** (the car the
//! lab drives, `rig::rig_vehicle("sedan")`, CPU reference, surface contact on as the
//! lab's `surface=1`). Layer 2 of `engine/docs/vehicle-test-strategy.md`, for the
//! failures the user saw in the lab (known-bugs S4):
//! * two cars after a head-on: one's nose inside the other, the pair fused (pushed
//!   apart, contact holds them);
//! * one car into a pole: its own panels through each other (a door bent back
//!   through the door behind it), and a crash that ends in an instability stop.
//!
//! Each scene runs once per test binary (cached); every test asserts one invariant
//! and prints a row `| name | scene | measured | bound | pass/FAIL |`. Tests written
//! ahead of their fix fail today: each is listed in `engine/tests-expected-failures.txt`
//! (CI runs it every time with its bar and fails when it starts passing, so it is
//! taken off the list with the fix), its measured value in its doc comment.
//!
//! Run: `cargo test --release -p engine-softbody-authoring --test gen_sedan_contact -- --nocapture`

use engine_softbody::desc::SoftVehicleDesc;
use engine_softbody::explicit_beams::{ExplicitBeams, StaticShape, STEP};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use engine_softbody::explicit_drive::DriveInput;
use engine_softbody::explicit_rigid::{ProxyGeom, ProxySet, ProxyShape, RigidProxy};
use glam::Quat;
use engine_softbody_authoring::authoring::rig;
use engine_softbody_authoring::vehicle_compare::settings;
use glam::Vec3;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

// --- The car ---------------------------------------------------------------------

fn desc() -> &'static SoftVehicleDesc {
    static D: OnceLock<SoftVehicleDesc> = OnceLock::new();
    D.get_or_init(|| {
        let built = rig::rig_vehicle("sedan", None, None).expect("rig the generated sedan");
        let files = rig::mod_files(&built);
        let refs: Vec<(String, String)> = files
            .iter()
            .filter(|(p, _)| p.ends_with(".jbeam"))
            .map(|(p, b)| (p.clone(), String::from_utf8_lossy(b).into_owned()))
            .collect();
        let pc = files
            .iter()
            .find(|(p, _)| p.ends_with(".pc"))
            .map(|(_, b)| String::from_utf8_lossy(b).into_owned());
        let r: Vec<(&str, &str)> = refs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
        engine_softbody::jbeam::load_jbeam(&r, pc.as_deref()).expect("load the sedan's JBeam")
    })
}

/// The car on its wheels 2 cm over the ground at `at`, moving at `vel`; surface
/// contact on (the lab's `surface=1`).
fn spawn(d: &SoftVehicleDesc, at: Vec3, vel: Vec3) -> ExplicitBeams {
    let lift = 0.02 - d.nodes.iter().map(|n| n.position[1]).fold(f32::MAX, f32::min);
    let (mut s, _) = ExplicitBeams::from_desc_jbeam_raycast(d, at + Vec3::Y * lift, vel, settings())
        .expect("spawn the sedan");
    s.surface_contact = true;
    // Measurement switches for the fixes under way (off: today's rule).
    s.edge_contact = std::env::var("EDGE_CONTACT").is_ok_and(|v| v == "1");
    s.outward_skin = std::env::var("OUTWARD").is_ok_and(|v| v == "1");
    s.finish();
    s
}

// --- Geometry --------------------------------------------------------------------

/// A car's live collision triangles (not broken, not anti-clip only), world.
fn hull(s: &ExplicitBeams) -> Vec<[Vec3; 3]> {
    live(s)
        .into_iter()
        .map(|t| s.triangles[t].nodes.map(|i| s.origin + s.nodes[i as usize].pos))
        .collect()
}

fn live(s: &ExplicitBeams) -> Vec<usize> {
    (0..s.triangles.len())
        .filter(|&t| !s.tri_dead.get(t).copied().unwrap_or(false) && !s.tri_anti_only(t))
        .collect()
}

/// The generalised winding number of the triangles about `p` (> 0.5: inside).
fn winding(p: Vec3, tris: &[[Vec3; 3]]) -> f32 {
    let mut w = 0.0f64;
    for t in tris {
        let (a, b, c) = (t[0] - p, t[1] - p, t[2] - p);
        let (la, lb, lc) = (a.length(), b.length(), c.length());
        let num = a.dot(b.cross(c)) as f64;
        let den = (la * lb * lc + a.dot(b) * lc + a.dot(c) * lb + b.dot(c) * la) as f64;
        w += 2.0 * num.atan2(den);
    }
    (w / (4.0 * std::f64::consts::PI)) as f32
}

fn dist(p: Vec3, tris: &[[Vec3; 3]]) -> f32 {
    tris.iter()
        .map(|t| engine_softbody::explicit_skin::closest_on_triangle(p, t[0], t[1], t[2]).0.distance_squared(p))
        .fold(f32::MAX, f32::min)
        .sqrt()
}

/// Whether segment p→q passes through triangle t (Möller–Trumbore, ends included).
fn seg_tri(p: Vec3, q: Vec3, t: &[Vec3; 3]) -> bool {
    let d = q - p;
    let (e1, e2) = (t[1] - t[0], t[2] - t[0]);
    let h = d.cross(e2);
    let a = e1.dot(h);
    if a.abs() < 1e-12 {
        return false;
    }
    let f = 1.0 / a;
    let s = p - t[0];
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return false;
    }
    let qv = s.cross(e1);
    let v = f * d.dot(qv);
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    let tt = f * e2.dot(qv);
    (0.0..=1.0).contains(&tt)
}

/// The live collision skin's edges (node pairs), each once.
fn edges(s: &ExplicitBeams) -> Vec<(u32, u32)> {
    let mut e: Vec<(u32, u32)> = live(s)
        .into_iter()
        .flat_map(|t| {
            let n = s.triangles[t].nodes;
            (0..3).map(move |k| (n[k].min(n[(k + 1) % 3]), n[k].max(n[(k + 1) % 3])))
        })
        .collect();
    e.sort();
    e.dedup();
    e
}

/// `a`'s skin edges through `b`'s live triangles (two cars).
fn edges_through(a: &ExplicitBeams, b: &ExplicitBeams) -> usize {
    let tris = hull(b);
    let w = |s: &ExplicitBeams, i: u32| s.origin + s.nodes[i as usize].pos;
    edges(a)
        .into_iter()
        .map(|(i, j)| tris.iter().filter(|t| seg_tri(w(a, i), w(a, j), t)).count())
        .sum()
}

/// One car's skin edges through its own live triangles, as (edge, triangle) pairs
/// that share no node: panels of one car through each other.
fn self_crossings(s: &ExplicitBeams) -> Vec<((u32, u32), usize)> {
    let w = |i: u32| s.origin + s.nodes[i as usize].pos;
    let tris: Vec<(usize, [u32; 3], [Vec3; 3])> = live(s)
        .into_iter()
        .map(|t| {
            let n = s.triangles[t].nodes;
            (t, n, n.map(w))
        })
        .collect();
    let mut out = Vec::new();
    for (i, j) in edges(s) {
        let (p, q) = (w(i), w(j));
        let (lo, hi) = (p.min(q), p.max(q));
        for (t, n, x) in &tris {
            if n.contains(&i) || n.contains(&j) {
                continue;
            }
            let (tl, th) = (x[0].min(x[1]).min(x[2]), x[0].max(x[1]).max(x[2]));
            if hi.cmplt(tl).any() || lo.cmpgt(th).any() {
                continue;
            }
            if seg_tri(p, q, x) {
                out.push(((i, j), *t));
            }
        }
    }
    out
}

fn centre(s: &ExplicitBeams) -> (Vec3, Vec3) {
    let (mut c, mut v, mut m) = (Vec3::ZERO, Vec3::ZERO, 0.0f32);
    for n in &s.nodes {
        c += n.pos * n.mass;
        v += n.vel * n.mass;
        m += n.mass;
    }
    (s.origin + c / m, v / m)
}

fn finite(s: &ExplicitBeams) -> bool {
    s.nodes.iter().all(|n| n.pos.is_finite() && n.vel.is_finite())
}

fn extent(s: &ExplicitBeams) -> f32 {
    let (lo, hi) = s.nodes.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(l, h), n| {
        (l.min(n.pos), h.max(n.pos))
    });
    (hi - lo).max_element()
}

fn row(name: &str, scene: &str, measured: String, bound: String, pass: bool) {
    println!(
        "| {name} | {scene} | {measured} | {bound} | {} |",
        if pass { "pass" } else { "FAIL" }
    );
}

// --- Head on, then backing apart -------------------------------------------------

/// Two generated sedans head on, full overlap, 60 + 60 km/h, 0.6 s. Then, from that one
/// state, the same push apart twice: with vehicle contact and without it (the user's
/// "fused together": not whether a crushed car can still drive, but whether contact
/// holds two cars that should simply part).
#[derive(Clone, Debug, Default)]
struct HeadOn {
    /// The most over the crash (sampled every 20 ms to 0.6 s): per car, its contact
    /// nodes inside the other's live hull deeper than the contact band, and the
    /// deepest (m).
    inside: [usize; 2],
    deepest: [f32; 2],
    /// The most over the crash: per car, its skin edges through the other's live
    /// triangles.
    edges: [usize; 2],
    /// How much further apart the centres get in the second after the push (m): with
    /// vehicle contact, and with it off.
    parted: f32,
    parted_free: f32,
    /// At the end: per car, its nearest contact node outside the other's hull (m).
    gap_end: [f32; 2],
    finite: bool,
}

/// Every node of car 0 gets `PUSH` m/s backwards (−z) and car 1's forwards (+z): the
/// cars move apart at 2·PUSH, as two cars pulled apart by a tow. **Choice**: 3 m/s, a
/// walking push that a ground's friction (about 0.75 g sliding) stops in 0.4 s.
const PUSH: f32 = 3.0;

fn parted_after_push(mut w: ExplicitBeamsWorld) -> (f32, bool) {
    for (c, s) in w.vehicles.iter_mut().enumerate() {
        let dv = if c == 0 { -Vec3::Z * PUSH } else { Vec3::Z * PUSH };
        for n in &mut s.nodes {
            n.vel += dv;
        }
    }
    let gap = |w: &ExplicitBeamsWorld| centre(&w.vehicles[0]).0.distance(centre(&w.vehicles[1]).0);
    let g0 = gap(&w);
    for _ in 0..(1.0 / STEP).round() as usize {
        w.step();
    }
    (gap(&w) - g0, w.vehicles.iter().all(finite))
}

fn head_on() -> HeadOn {
    pair_crash(60, 60, 0, 600, true)
}

/// Two generated sedans nose to nose (5 cm apart), car A at `kmh_a` towards car B, car B
/// at `kmh_b` towards car A, car B moved `offset_mm` sideways; run `ms`, sampled every 20
/// ms; `push`: then the fused check (`parted_after_push`). Cached per scene.
fn pair_crash(kmh_a: u32, kmh_b: u32, offset_mm: i32, ms: u32, push: bool) -> HeadOn {
    static R: OnceLock<Mutex<HashMap<(u32, u32, i32, u32, bool), HeadOn>>> = OnceLock::new();
    let m = R.get_or_init(Default::default);
    let key = (kmh_a, kmh_b, offset_mm, ms, push);
    if let Some(r) = m.lock().unwrap().get(&key) {
        return r.clone();
    }
    let d = desc();
    let zmax = d.nodes.iter().map(|n| n.position[2]).fold(f32::MIN, f32::max);
    let mut turned = d.clone();
    for n in &mut turned.nodes {
        n.position = [-n.position[0], n.position[1], -n.position[2]];
    }
    let a = spawn(d, Vec3::new(0.0, 0.0, -0.025 - zmax), Vec3::Z * (kmh_a as f32 / 3.6));
    let b = spawn(
        &turned,
        Vec3::new(offset_mm as f32 * 1e-3, 0.0, 0.025 + zmax),
        -Vec3::Z * (kmh_b as f32 / 3.6),
    );
    let mut w = ExplicitBeamsWorld::new(vec![a, b]);
    if kmh_b == 0 && ms > 1000 {
        // The slow push: car B braked (handbrake), car A pressing on at a light throttle.
        w.vehicles[0].set_drive_input(DriveInput { forward: 0.3, ..Default::default() });
        w.vehicles[1].set_drive_input(DriveInput { brake: 1.0, handbrake: 1.0, ..Default::default() });
    }
    let steps = (ms as f32 * 1e-3 / STEP).round() as usize;
    let every = (0.02 / STEP).round() as usize;
    let mut r = HeadOn::default();
    for k in 1..=steps {
        w.step();
        if k % every != 0 && k != steps {
            continue;
        }
        let hulls = [hull(&w.vehicles[0]), hull(&w.vehicles[1])];
        for c in 0..2 {
            let (x, o) = (&w.vehicles[c], &hulls[1 - c]);
            let band = w.vehicles[1 - c].collision_range;
            let mut inside = 0;
            let mut nearest = f32::MAX;
            for &i in &x.contact_points {
                let p = x.origin + x.nodes[i as usize].pos;
                let dd = dist(p, o);
                if winding(p, o) > 0.5 {
                    if dd > band {
                        inside += 1;
                    }
                    r.deepest[c] = r.deepest[c].max(dd);
                } else {
                    nearest = nearest.min(dd);
                }
            }
            r.inside[c] = r.inside[c].max(inside);
            r.edges[c] = r.edges[c].max(edges_through(&w.vehicles[c], &w.vehicles[1 - c]));
            if k == steps {
                r.gap_end[c] = nearest;
            }
        }
    }
    r.finite = w.vehicles.iter().all(finite);
    if push {
        let mut free = w.clone();
        free.contact = false;
        let (on, f1) = parted_after_push(w);
        let (off, f2) = parted_after_push(free);
        r.parted = on;
        r.parted_free = off;
        r.finite &= f1 && f2;
    }
    println!("pair {kmh_a} into {kmh_b} km/h, offset {offset_mm} mm, {ms} ms: {r:?}");
    m.lock().unwrap().insert(key, r.clone());
    r
}

/// The overlap invariants of one pair scene: no node inside the other car past the band
/// and no skin edge through it, at the worst sample.
fn no_overlap(name: &str, scene: &str, r: &HeadOn) -> bool {
    let pass = r.finite && r.inside == [0, 0] && r.edges == [0, 0];
    row(
        name,
        scene,
        format!(
            "{} / {} nodes past the band (deepest {:.0} / {:.0} mm), {} / {} edges through",
            r.inside[0],
            r.inside[1],
            r.deepest[0] * 1e3,
            r.deepest[1] * 1e3,
            r.edges[0],
            r.edges[1]
        ),
        "0, 0".into(),
        pass,
    );
    pass
}

/// Faster: 100 + 100 km/h head on (each car moves 28 mm a step, more than the 25 mm band).
/// Bound: exact. FAILS today (2026-11-24): see the printed row.
#[test]
fn fast_head_on_no_overlap() {
    assert!(no_overlap("no overlap, fast head-on", "head on 100 + 100, worst to 0.6 s", &pair_crash(100, 100, 0, 600, false)));
}

/// Unequal: 100 km/h into a car doing 10 towards it (the relative speed sets the contact,
/// not either car's own). Bound: exact. FAILS today (2026-11-24): see the printed row.
#[test]
fn unequal_head_on_no_overlap() {
    assert!(no_overlap("no overlap, 100 into 10", "head on 100 + 10, worst to 0.6 s", &pair_crash(100, 10, 0, 600, false)));
}

/// Offset: 60 + 60 km/h, car B 0.8 m to the side (front corners, wheels). Bound: exact.
/// FAILS today (2026-11-24): see the printed row.
#[test]
fn offset_head_on_no_overlap() {
    assert!(no_overlap("no overlap, offset head-on", "head on 60 + 60, 0.8 m offset, worst to 0.6 s", &pair_crash(60, 60, 800, 600, false)));
}

/// A slow push leaves no gap: car A at 4 km/h (1.1 m/s) into car B at rest, A at a light
/// throttle (0.3) and B braked, 2 s. They end touching:
/// each car's nearest node no further from the other's collision skin than the band plus
/// 10 mm, and nothing inside. Bound: the band is where contact acts (the dissipative law
/// holds a pair at its edge, not beyond); 10 mm a **Choice** (a contact settling under
/// friction). This keeps any contact rule from stopping cars short of each other.
#[test]
fn slow_push_ends_touching() {
    let r = pair_crash(4, 0, 0, 2000, false);
    let band = 0.025;
    let gap = r.gap_end[0].min(r.gap_end[1]);
    let pass = r.finite && r.inside == [0, 0] && gap <= band + 0.01;
    row(
        "a slow push ends touching",
        "car at 4 km/h into a parked car, 2 s",
        format!("nearest node {:.0} mm from the other's skin; {} / {} inside", gap * 1e3, r.inside[0], r.inside[1]),
        "≤ band + 10 mm, 0 inside".into(),
        pass,
    );
    assert!(pass);
}

/// Two cars that hit each other are not fused: pushed apart after the head-on, they part
/// at least 80 % as far as the same two crushed cars with vehicle contact switched off
/// (the same state and push; only contact differs, so a drivetrain the crash broke does
/// not enter). Bound: **Choice** (80 %: contact may only slow the parting where crushed
/// fronts still touch as they slide out, not hold them).
/// FAILS today (2026-11-24, S4): see the printed row (driven in reverse instead, they
/// moved 0.14 m in 2 s against 17 m with contact off).
#[test]
fn head_on_cars_are_not_fused() {
    let r = head_on();
    let pass = r.finite && r.parted >= 0.8 * r.parted_free;
    row(
        "cars are not fused after a head-on",
        "head on 60 + 60, pushed apart 2 × 3 m/s",
        format!("parted {:.2} m in 1 s; {:.2} m with contact off", r.parted, r.parted_free),
        "≥ 80 % of contact off".into(),
        pass,
    );
    assert!(pass);
}

/// No node of one car inside the other: at any sample of the head-on (every 20 ms to
/// 0.6 s), a contact node inside the other car's live collision hull (winding number)
/// deeper than the contact band. Bound: exact (two solids share no space; the band is
/// where contact acts).
/// FAILS today (2026-11-24, S4): see the printed row (at 0.6 s alone: 23 / 35 nodes,
/// deepest 142 / 241 mm; most entered through gaps in the collision skin).
#[test]
fn head_on_no_node_inside_the_other_car() {
    let r = head_on();
    let pass = r.finite && r.inside == [0, 0];
    row(
        "no node inside the other car",
        "head on 60 + 60, worst to 0.6 s",
        format!(
            "{} / {} nodes past the band, deepest {:.0} / {:.0} mm",
            r.inside[0],
            r.inside[1],
            r.deepest[0] * 1e3,
            r.deepest[1] * 1e3
        ),
        "0".into(),
        pass,
    );
    assert!(pass);
}

/// No panel of one car through the other's: at any sample of the head-on, a skin edge
/// of one car through a live collision triangle of the other. Bound: exact (with
/// node-only contact an edge can pass between nodes; edge contact is the fix).
/// FAILS today (2026-11-24, S4): see the printed row (at 0.6 s alone: 62 / 54 edges).
#[test]
fn head_on_no_edge_through_the_other_car() {
    let r = head_on();
    let pass = r.finite && r.edges == [0, 0];
    row(
        "no skin edge through the other car",
        "head on 60 + 60, worst to 0.6 s",
        format!("{} / {} edges", r.edges[0], r.edges[1]),
        "0".into(),
        pass,
    );
    assert!(pass);
}

// --- One car into a pole ---------------------------------------------------------

/// The generated sedan coasting into the lab's pole (r 0.15 m, steel, fixed) with its
/// front left corner (the pole 0.45 m left of the centre line), `kmh`, 1.5 s.
#[derive(Clone, Debug, Default)]
struct Pole {
    /// Own skin edges through own live triangles (sharing no node) at spawn, and the
    /// most after that were not there at spawn; the parts named at that worst.
    crossings_spawn: usize,
    crossings_worst: usize,
    /// The spawn's crossings by part (the collision skin through itself at rest).
    spawn_parts: Vec<(String, usize)>,
    worst_parts: Vec<(String, usize)>,
    /// The mechanical energy's largest rise over the start, and the start (J).
    rise: f64,
    start: f64,
    /// The largest extent over the extent at spawn.
    growth: f32,
    finite: bool,
}

fn pole(kmh: u32) -> Pole {
    static R: OnceLock<Mutex<HashMap<u32, Pole>>> = OnceLock::new();
    let m = R.get_or_init(Default::default);
    if let Some(r) = m.lock().unwrap().get(&kmh) {
        return r.clone();
    }
    let d = desc();
    let zmax = d.nodes.iter().map(|n| n.position[2]).fold(f32::MIN, f32::max);
    let mut s = spawn(d, Vec3::ZERO, Vec3::Z * (kmh as f32 / 3.6));
    // The car's left is +x (its forward +z, up +y).
    s.shapes.push(StaticShape::cylinder(Vec3::new(0.45, 2.0, zmax + 0.3 + 0.15), 0.15, 2.0));
    let ext0 = extent(&s);
    let part_of = |i: u32| -> String {
        let n = &d.nodes[i as usize].name;
        n.trim_end_matches(|c: char| c.is_ascii_digit() || c == '_').to_string()
    };
    let by_part = |s: &ExplicitBeams, x: &[((u32, u32), usize)]| -> Vec<(String, usize)> {
        let mut by: HashMap<String, usize> = HashMap::new();
        for ((i, j), t) in x {
            let tn = s.triangles[*t].nodes[0];
            *by.entry(format!("{}-{} through {}", part_of(*i), part_of(*j), part_of(tn))).or_default() += 1;
        }
        let mut v: Vec<(String, usize)> = by.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v.truncate(12);
        v
    };
    let at_spawn = self_crossings(&s);
    let spawn_set: std::collections::HashSet<((u32, u32), usize)> = at_spawn.iter().copied().collect();
    let mut r = Pole {
        crossings_spawn: at_spawn.len(),
        spawn_parts: by_part(&s, &at_spawn),
        finite: true,
        ..Default::default()
    };
    let mut w = ExplicitBeamsWorld::new(vec![s]);
    w.enable_ledger();
    let start = w.ledger.as_ref().unwrap().start;
    r.start = start;
    let steps = (1.5 / STEP).round() as usize;
    for k in 1..=steps {
        w.step();
        let l = w.ledger.as_ref().unwrap();
        r.rise = r.rise.max(l.mechanical() - start);
        if k % 40 == 0 {
            let s = &w.vehicles[0];
            r.finite &= finite(s);
            if !r.finite {
                break;
            }
            r.growth = r.growth.max(extent(s) / ext0);
            let x: Vec<((u32, u32), usize)> =
                self_crossings(s).into_iter().filter(|c| !spawn_set.contains(c)).collect();
            if x.len() > r.crossings_worst {
                r.crossings_worst = x.len();
                r.worst_parts = by_part(s, &x);
            }
        }
    }
    println!("pole {kmh} km/h: {r:?}");
    m.lock().unwrap().insert(kmh, r.clone());
    r
}

/// A car's own panels never pass through each other: one car into a pole at 64 km/h,
/// its collision skin's edges through its own live triangles (pairs sharing no node)
/// that were not there at spawn. Bound: exact (one part of a car is a solid against
/// another; self contact is node against triangle, so an edge can still pass between
/// nodes). The user's lab showed the front door bent back through the rear door.
/// FAILS today (2026-11-24): 52 new crossings, most the front left door's edges through the
/// left fender and the fender's through the door (5 + 5), the engine through the
/// radiator, the rear bumper through the body.
#[test]
fn pole_own_panels_stay_apart() {
    let r = pole(64);
    let pass = r.finite && r.crossings_worst == 0;
    row(
        "own panels stay apart",
        "pole 64 km/h, front left corner",
        format!("{} new edges through own triangles at worst; {:?}", r.crossings_worst, r.worst_parts),
        "0".into(),
        pass,
    );
    assert!(pass);
}

/// The collision skin does not pass through itself at rest (a measurement of the
/// generated sedan's authoring): its skin edges through its own live triangles at
/// spawn. Bound: exact (a skin through itself has no outside there).
/// FAILS today (2026-11-24): 89 at spawn (the rear bumper through the body, the sills'
/// rows through each other, the fenders' corners through the hood, the rear doors).
#[test]
fn collision_skin_does_not_cross_itself_at_rest() {
    let r = pole(64);
    let pass = r.crossings_spawn == 0;
    row(
        "collision skin does not cross itself at rest",
        "spawn",
        format!("{} edges through own triangles; {:?}", r.crossings_spawn, r.spawn_parts),
        "0".into(),
        pass,
    );
    assert!(pass);
}

/// A pole crash stays finite and creates no energy: the mechanical energy (kinetic,
/// gravitational, elastic) never rises past 0.5 % of the start (the symplectic step's
/// bounded oscillation, `crash_invariants.rs`'s rule) and nothing flies off (the car's
/// extent stays within 3× its spawn extent: the lab's instability stop is at 10× + 50 m).
/// The user's lab stopped one pole crash with the car 104 m across and its kinetic
/// energy up from 0 to 302 kJ (generated sedan, `surface=1`, 2026-11-24).
#[test]
fn pole_crashes_stay_finite_and_create_no_energy() {
    let mut ok = true;
    for kmh in [40, 64, 90] {
        let r = pole(kmh);
        let pass = r.finite && r.rise <= 5e-3 * r.start && r.growth <= 3.0;
        row(
            "finite, energy never rises, nothing flies off",
            &format!("pole {kmh} km/h, front left corner"),
            format!(
                "rise {:.1} J of {:.0} kJ, extent ×{:.2}",
                r.rise,
                r.start * 1e-3,
                r.growth
            ),
            "≤ 0.5 % of start, ×3".into(),
            pass,
        );
        ok &= pass;
    }
    assert!(ok);
}

// --- A cannonball into the door ------------------------------------------------------

/// A steel ball (radius `radius_mm`) fired square at the parked car's front left door at
/// `speed` m/s, 0.3 s: the most car contact nodes inside the ball deeper than the band at
/// any step, the deepest, and the steps its centre's path crossed a live collision
/// triangle (tunnelling: a panel the ball tears open dies first, so a live one crossed was
/// never met).
#[derive(Clone, Debug, Default)]
struct Shot {
    inside: usize,
    deepest: f32,
    crossed: usize,
    /// The ball's speed at the end and the share of its momentum the car took.
    speed_end: f32,
    taken: f32,
    finite: bool,
}

fn shot(speed: u32, radius_mm: u32) -> Shot {
    static R: OnceLock<Mutex<HashMap<(u32, u32), Shot>>> = OnceLock::new();
    let m = R.get_or_init(Default::default);
    if let Some(r) = m.lock().unwrap().get(&(speed, radius_mm)) {
        return r.clone();
    }
    let d = desc();
    let mut s = spawn(d, Vec3::ZERO, Vec3::ZERO);
    let door: Vec<Vec3> = (0..s.nodes.len())
        .filter(|&i| d.nodes[i].name.starts_with("door_FL"))
        .map(|i| s.origin + s.nodes[i].pos)
        .collect();
    let aim = door.iter().copied().sum::<Vec3>() / door.len() as f32;
    let side = door.iter().map(|p| p.x).fold(f32::MIN, f32::max);
    let radius = radius_mm as f32 * 1e-3;
    let mass = 7850.0 * 4.0 / 3.0 * std::f32::consts::PI * radius.powi(3);
    let mut set = ProxySet {
        bodies: vec![RigidProxy {
            body: 0,
            pos: Vec3::new(side + radius + 0.3, aim.y, aim.z),
            rot: Quat::IDENTITY,
            vel: Vec3::new(-(speed as f32), 0.0, 0.0),
            ang_vel: Vec3::ZERO,
            inv_mass: 1.0 / mass,
            inv_inertia: Vec3::splat(1.0 / (0.4 * mass * radius * radius)),
            share: 1.0,
            count: 0,
        }],
        shapes: vec![ProxyShape::new(0, Vec3::ZERO, Quat::IDENTITY, ProxyGeom::Sphere { radius }, 0, None)],
        frozen: false,
    };
    set.begin_window();
    s.rigid = Some(Box::new(set));
    let band = s.collision_range;
    let mut r = Shot { finite: true, ..Default::default() };
    let mut prev = s.rigid.as_ref().unwrap().bodies[0].pos;
    for _ in 0..(0.3 / STEP).round() as usize {
        s.step();
        let c = s.rigid.as_ref().unwrap().bodies[0].pos;
        r.finite &= finite(&s) && c.is_finite();
        if !r.finite {
            break;
        }
        let mut inside = 0;
        for &i in &s.contact_points {
            let dd = radius - (s.origin + s.nodes[i as usize].pos).distance(c);
            if dd > band {
                inside += 1;
            }
            r.deepest = r.deepest.max(dd);
        }
        r.inside = r.inside.max(inside);
        if hull(&s).iter().any(|t| seg_tri(prev, c, t)) {
            r.crossed += 1;
        }
        prev = c;
    }
    let b = &s.rigid.as_ref().unwrap().bodies[0];
    r.speed_end = b.vel.length();
    let pc: Vec3 = s.nodes.iter().map(|n| n.vel * n.mass).sum();
    r.taken = pc.x.abs() / (mass * speed as f32);
    println!("ball r {radius_mm} mm at {speed} m/s into the door: {r:?}; ball at {:?}", b.pos);
    m.lock().unwrap().insert((speed, radius_mm), r.clone());
    r
}

/// A cannonball never shares space with the car: a 0.1 m steel ball at 30 and 150 m/s
/// into the front door, no car node inside the ball past the band and the ball never
/// through a live panel. Bound: exact (two solids; a torn panel is dead, so crossing a live
/// one is tunnelling). FAILS today (2026-11-24): see the printed rows (node contact with
/// rigid bodies has no push-out, known-bugs B1).
#[test]
fn cannonball_never_inside_the_car() {
    let mut ok = true;
    for speed in [30, 150] {
        let r = shot(speed, 100);
        let pass = r.finite && r.inside == 0 && r.crossed == 0;
        row(
            "a cannonball never shares space with the car",
            &format!("steel ball r 0.1 m at {speed} m/s into the front door"),
            format!("{} nodes inside it past the band (deepest {:.0} mm), {} steps through a live panel", r.inside, r.deepest * 1e3, r.crossed),
            "0, 0".into(),
            pass,
        );
        ok &= pass;
    }
    assert!(ok);
}
