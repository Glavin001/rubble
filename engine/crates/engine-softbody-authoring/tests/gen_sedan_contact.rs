//! Solids never share space: contact invariants of the **generated sedan** (the car the
//! lab drives, `rig::rig_vehicle("sedan")`, CPU reference, surface contact on as the
//! lab's `surface=1`). Layer 2 of `engine/docs/vehicle-test-strategy.md`, for the
//! failures the user saw in the lab (known-bugs S4):
//! * two cars after a head-on: one's nose inside the other, the pair locked so they
//!   cannot back apart;
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

/// Two generated sedans head on, full overlap, 60 + 60 km/h, 0.6 s; then both drivers
/// reverse at full throttle for 2 s (the user's "I can't reverse out").
#[derive(Clone, Debug, Default)]
struct HeadOn {
    /// At 0.6 s: per car, its contact nodes inside the other's live hull deeper
    /// than the contact band, and the deepest (m).
    inside: [usize; 2],
    deepest: [f32; 2],
    /// At 0.6 s: per car, its skin edges through the other's live triangles.
    edges: [usize; 2],
    /// The centres' distance at 0.6 s and after the 2 s of reverse (m).
    gap_crash: f32,
    gap_end: f32,
    /// Their separating speed at the end (m/s).
    separating: f32,
    finite: bool,
}

fn head_on() -> HeadOn {
    static R: OnceLock<HeadOn> = OnceLock::new();
    R.get_or_init(|| {
        let d = desc();
        let zmax = d.nodes.iter().map(|n| n.position[2]).fold(f32::MIN, f32::max);
        let mut turned = d.clone();
        for n in &mut turned.nodes {
            n.position = [-n.position[0], n.position[1], -n.position[2]];
        }
        let v = 60.0 / 3.6;
        let a = spawn(d, Vec3::new(0.0, 0.0, -0.025 - zmax), Vec3::Z * v);
        let b = spawn(&turned, Vec3::new(0.0, 0.0, 0.025 + zmax), -Vec3::Z * v);
        let mut w = ExplicitBeamsWorld::new(vec![a, b]);
        let crash = (0.6 / STEP).round() as usize;
        for _ in 0..crash {
            w.step();
        }
        let mut r = HeadOn::default();
        let hulls = [hull(&w.vehicles[0]), hull(&w.vehicles[1])];
        for c in 0..2 {
            let (x, o) = (&w.vehicles[c], &hulls[1 - c]);
            let band = w.vehicles[1 - c].collision_range;
            for &i in &x.contact_points {
                let p = x.origin + x.nodes[i as usize].pos;
                if winding(p, o) > 0.5 {
                    let dd = dist(p, o);
                    if dd > band {
                        r.inside[c] += 1;
                    }
                    r.deepest[c] = r.deepest[c].max(dd);
                }
            }
            r.edges[c] = edges_through(&w.vehicles[c], &w.vehicles[1 - c]);
        }
        r.gap_crash = centre(&w.vehicles[0]).0.distance(centre(&w.vehicles[1]).0);
        for s in &mut w.vehicles {
            s.set_drive_input(DriveInput { forward: -1.0, ..Default::default() });
        }
        for _ in 0..(2.0 / STEP).round() as usize {
            w.step();
        }
        let ((ca, va), (cb, vb)) = (centre(&w.vehicles[0]), centre(&w.vehicles[1]));
        r.gap_end = ca.distance(cb);
        r.separating = (vb - va).dot((cb - ca).normalize_or_zero());
        r.finite = w.vehicles.iter().all(finite);
        println!("head on 60 + 60 km/h, then reverse: {r:?}");
        r
    })
    .clone()
}

/// Two cars that hit each other can back apart: after the head-on both drivers reverse
/// for 2 s and the centres must move at least 1 m further apart. Bound: **Choice**
/// (with vehicle contact switched off at the same moment they move 17 m apart, 18 m/s
/// at the end, so 1 m only asks that contact does not hold them).
/// FAILS today (2026-11-24, S4): 0.15 m, 0.00 m/s at the end: contact holds them.
#[test]
fn head_on_cars_back_apart() {
    let r = head_on();
    let moved = r.gap_end - r.gap_crash;
    let pass = r.finite && moved >= 1.0;
    row(
        "cars back apart after a head-on",
        "head on 60 + 60, reverse 2 s",
        format!("{moved:.2} m further apart, separating {:.2} m/s", r.separating),
        "≥ 1 m".into(),
        pass,
    );
    assert!(pass);
}

/// No node of one car inside the other at rest: 0.6 s after the head-on, a contact node
/// inside the other car's live collision hull (winding number) deeper than the contact
/// band. Bound: exact (two solids share no space; the band is where contact acts).
/// FAILS today (2026-11-24, S4): 28 / 41 nodes inside, deepest 142 / 241 mm (most entered
/// through gaps in the collision skin).
#[test]
fn head_on_no_node_inside_the_other_car() {
    let r = head_on();
    let pass = r.finite && r.inside == [0, 0];
    row(
        "no node inside the other car",
        "head on 60 + 60, at 0.6 s",
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

/// No panel of one car through the other's: 0.6 s after the head-on, a skin edge of
/// one car through a live collision triangle of the other. Bound: exact (with node-only
/// contact an edge can pass between nodes; edge contact is the fix).
/// FAILS today (2026-11-24, S4): 62 / 54 edges.
#[test]
fn head_on_no_edge_through_the_other_car() {
    let r = head_on();
    let pass = r.finite && r.edges == [0, 0];
    row(
        "no skin edge through the other car",
        "head on 60 + 60, at 0.6 s",
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
