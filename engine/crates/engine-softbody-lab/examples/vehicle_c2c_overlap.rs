//! How far two rigged cars get inside each other head on (CPU reference): the
//! `vehicle_c2c` / `vehicle_film c2c` scene (two cars of a design, full
//! overlap, `km/h` each, raycast wheels, vehicle contact, surface contact on),
//! measured every 4 ms. "Inside" is the generalised winding number (> 0.5)
//! of the other car's live collision triangles (its closed-or-not collision
//! hull); depth is the distance to the nearest of those triangles.
//!
//! Per direction (A in B, B in A):
//! * `nodes`: the car's contact points inside the other's hull (count, deepest);
//!   each node that enters is classified by its path since the last sample:
//!   through a live triangle ("face") or not ("gap": between triangles, a hole
//!   in the hull, or a dead triangle);
//! * `paint` / `tyre`: render vertices (paint, tyre) inside the other's hull
//!   (count, p99 and deepest);
//! * `edges`: the car's collision-skin edges through the other's triangles;
//! * `slab`: the render surfaces' overlap along the cars' axis (5 cm bins
//!   across the car: A's frontmost vertex beyond B's frontmost), the deepest bin
//!   and the overlapping volume.
//!
//! At spawn it prints the collision hull's open edges (edges on one triangle)
//! and how well the front's render skin is covered by collision triangles.
//!
//! ```bash
//! cargo run --release -p engine-softbody-lab --example vehicle_c2c_overlap -- [design] [km/h] [seconds]
//! ```
use engine_softbody_authoring::authoring::rig;
use engine_softbody_authoring::authoring::skin::{deform, mat, ShowcaseVisual};
use engine_softbody::desc::SoftVehicleDesc;
use engine_softbody::explicit_beams::ExplicitBeams;
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use engine_softbody_authoring::vehicle_compare::settings;
use glam::Vec3;
use std::collections::HashMap;

struct Loaded {
    desc: SoftVehicleDesc,
    refs: Vec<(String, String)>,
    pc: Option<String>,
    built: rig::RiggedVehicle,
}

fn load(design: &str) -> Result<Loaded, String> {
    let built = rig::rig_vehicle(design, None, None)?;
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
    let desc = engine_softbody::jbeam::load_jbeam(&r, pc.as_deref())?;
    Ok(Loaded {
        desc,
        refs,
        pc,
        built,
    })
}

fn visual(
    l: &Loaded,
    desc: &SoftVehicleDesc,
    sim: &ExplicitBeams,
) -> Result<ShowcaseVisual, String> {
    let r: Vec<(&str, &str)> = l
        .refs
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let flex = engine_softbody::jbeam::assembled_rows(&r, l.pc.as_deref(), "flexbodies")?;
    let g = &l.built.input.glb;
    let meshes: Vec<_> = g
        .meshes
        .iter()
        .map(|m| engine_softbody_authoring::flexskin::glb_mesh(g, m))
        .collect();
    let paint = engine_softbody_authoring::glb::paint_of(g).unwrap_or([0.5, 0.05, 0.04]);
    let (v, _) = engine_softbody_authoring::flexmesh::jbeam_visual_skinned(
        &flex,
        &[],
        desc,
        sim,
        &meshes,
        &Default::default(),
        paint,
        Some(&l.built.skin),
    )?;
    Ok(v)
}

fn spawn(desc: &SoftVehicleDesc, at: Vec3, vel: Vec3) -> Result<ExplicitBeams, String> {
    let lift = 0.02
        - desc
            .nodes
            .iter()
            .map(|n| n.position[1])
            .fold(f32::MAX, f32::min);
    let (mut s, _) =
        ExplicitBeams::from_desc_jbeam_raycast(desc, at + Vec3::Y * lift, vel, settings())?;
    s.surface_contact = std::env::var("SURFACE_CONTACT").map_or(true, |v| v != "0");
    // OUTWARD=1: vehicle contact takes the triangle's outward side (outward_skin).
    s.outward_skin = std::env::var("OUTWARD").is_ok_and(|v| v == "1" || v == "2");
    s.edge_contact = std::env::var("EDGE_CONTACT").is_ok_and(|v| v == "1");
    // COLLISION_RANGE=m: the contact band (a measurement).
    if let Some(r) = std::env::var("COLLISION_RANGE").ok().and_then(|v| v.parse().ok()) {
        s.collision_range = r;
    }
    if std::env::var("SHOW_RANGE").is_ok() {
        eprintln!("collision range {} m", s.collision_range);
    }
    if std::env::var("OUTWARD").is_ok_and(|v| v == "2") {
        // OUTWARD=2: only the outer skin's triangles (in front of each, 1 cm out,
        // the winding number of the whole collision skin is below 0.5).
        let h = hull(&s);
        s.outward_tris = (0..s.triangles.len())
            .map(|t| {
                let [a, b, c] = s.triangles[t].nodes.map(|i| s.origin + s.nodes[i as usize].pos);
                let n = (b - a).cross(c - a).normalize_or_zero();
                winding((a + b + c) / 3.0 + n * 0.01, &h) < 0.5
            })
            .collect();
        eprintln!(
            "  outer-skin triangles: {} of {}",
            s.outward_tris.iter().filter(|&&x| x).count(),
            s.triangles.len()
        );
    }
    s.finish();
    Ok(s)
}

/// A car's live collision triangles (not broken, not anti-clip only), world.
fn hull(s: &ExplicitBeams) -> Vec<[Vec3; 3]> {
    (0..s.triangles.len())
        .filter(|&t| !s.tri_dead.get(t).copied().unwrap_or(false) && !s.tri_anti_only(t))
        .map(|t| {
            s.triangles[t]
                .nodes
                .map(|i| s.origin + s.nodes[i as usize].pos)
        })
        .collect()
}

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

fn closest_on_tri(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    // Ericson, Real-Time Collision Detection 5.1.5.
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let den = 1.0 / (va + vb + vc);
    a + ab * (vb * den) + ac * (vc * den)
}

fn dist(p: Vec3, tris: &[[Vec3; 3]]) -> f32 {
    tris.iter()
        .map(|t| closest_on_tri(p, t[0], t[1], t[2]).distance_squared(p))
        .fold(f32::MAX, f32::min)
        .sqrt()
}

/// Whether segment p→q crosses triangle t (Möller–Trumbore, both ends).
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

/// Closest points' parameters of segments p1–q1 and p2–q2 (Ericson §5.1.9).
fn seg_seg(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> (f32, f32) {
    let (d1, d2, r) = (q1 - p1, q2 - p2, p1 - p2);
    let (a, e, f) = (d1.dot(d1), d2.dot(d2), d2.dot(r));
    if a <= 1e-12 && e <= 1e-12 {
        return (0.0, 0.0);
    }
    if a <= 1e-12 {
        return (0.0, (f / e).clamp(0.0, 1.0));
    }
    let c = d1.dot(r);
    if e <= 1e-12 {
        return ((-c / a).clamp(0.0, 1.0), 0.0);
    }
    let b = d1.dot(d2);
    let den = a * e - b * b;
    let mut s = if den > 1e-12 { ((b * f - c * e) / den).clamp(0.0, 1.0) } else { 0.0 };
    let mut t = (b * s + f) / e;
    if t < 0.0 {
        t = 0.0;
        s = (-c / a).clamp(0.0, 1.0);
    } else if t > 1.0 {
        t = 1.0;
        s = ((b - c) / a).clamp(0.0, 1.0);
    }
    (s, t)
}

fn aabb(tris: &[[Vec3; 3]]) -> (Vec3, Vec3) {
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    for t in tris {
        for p in t {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
    }
    (lo, hi)
}

fn inside_box(p: Vec3, (lo, hi): (Vec3, Vec3)) -> bool {
    p.cmpge(lo).all() && p.cmple(hi).all()
}

fn pct(v: &mut [f32], q: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f32 * q).round() as usize]
}

#[derive(Default, Clone, Copy)]
struct Dir {
    nodes: usize,
    node_depth: f32,
    face: usize,
    gap: usize,
    paint: usize,
    paint_p99: f32,
    paint_depth: f32,
    tyre: usize,
    tyre_depth: f32,
    edges: usize,
    /// The inside nodes by their anti-clip memory for the other car's triangles
    /// (LOCK): holding a side they are on (held inside: the lock), holding a side
    /// they are behind (pushed back out), or no memory of the other car (free).
    held_in: usize,
    push_out: usize,
    free: usize,
}

#[path = "vehicle_c2c_overlap/reports.rs"]
mod reports;
use reports::*;

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let design = a.first().cloned().unwrap_or_else(|| "sedan".into());
    let kmh: f32 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(50.0);
    let secs: f32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.6);
    let l = load(&design)?;
    let d = &l.desc;
    let zmax = d
        .nodes
        .iter()
        .map(|n| n.position[2])
        .fold(f32::MIN, f32::max);
    let v = kmh / 3.6;
    let mut tb = d.clone();
    for n in &mut tb.nodes {
        n.position = [-n.position[0], n.position[1], -n.position[2]];
    }
    let sa = spawn(d, Vec3::new(0.0, 0.0, -0.025 - zmax), Vec3::Z * v)?;
    // OFFSET=m: car B moved sideways (an offset hit; 0 full overlap).
    let offset: f32 = std::env::var("OFFSET").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let sb = spawn(&tb, Vec3::new(offset, 0.0, 0.025 + zmax), -Vec3::Z * v)?;
    let vis: Vec<ShowcaseVisual> = [d, &tb]
        .iter()
        .map(|dd| {
            let (probe, _) =
                ExplicitBeams::from_desc_jbeam_raycast(dd, Vec3::ZERO, Vec3::ZERO, settings())?;
            visual(&l, dd, &probe)
        })
        .collect::<Result<_, String>>()?;

    near_report(d, &sa)?;
    // ---- Spawn diagnostics (car A) ----
    if std::env::var("MESHES").is_ok() {
        // The render meshes at the front (glTF: +Y up, +Z forward): each mesh's
        // vertex count and extent in the front 0.4 m.
        let m = &l.built.input.model;
        let zf = m
            .parts
            .iter()
            .flat_map(|p| p.pos.iter().map(|q| q.z))
            .fold(f32::MIN, f32::max);
        for p in &m.parts {
            let v: Vec<&Vec3> = p.pos.iter().filter(|q| q.z > zf - 0.4).collect();
            if v.is_empty() {
                continue;
            }
            let lo = v.iter().fold(Vec3::splat(f32::MAX), |a, q| a.min(**q));
            let hi = v.iter().fold(Vec3::splat(f32::MIN), |a, q| a.max(**q));
            println!(
                "  mesh {:<22} {:6} vertices in the front 0.4 m: x {:.2}..{:.2} y {:.2}..{:.2} z {:.2}..{:.2} (nose {:.2})",
                p.name, v.len(), lo.x, hi.x, lo.y, hi.y, lo.z - zf, hi.z - zf, zf
            );
        }
    }
    skin_report(&design, d, &sa);
    cover_report(d, &sa, &vis)?;
    tyre_report(d, &sa, &vis)?;

    let mut w = ExplicitBeamsWorld::new(vec![sa, sb]);
    // NO_THROTTLE=1: vehicle contact tests every triangle every step (a measurement).
    if std::env::var("NO_THROTTLE").is_ok_and(|v| v == "1") {
        w.throttle = false;
    }
    let edges_of = |s: &ExplicitBeams| -> Vec<(u32, u32)> {
        let mut e: Vec<(u32, u32)> = (0..s.triangles.len())
            .filter(|&t| !s.tri_dead.get(t).copied().unwrap_or(false) && !s.tri_anti_only(t))
            .flat_map(|t| {
                let n = s.triangles[t].nodes;
                (0..3).map(move |k| (n[k].min(n[(k + 1) % 3]), n[k].max(n[(k + 1) % 3])))
            })
            .collect();
        e.sort();
        e.dedup();
        e
    };
    let mut prev_in: [Vec<bool>; 2] = [
        vec![false; w.vehicles[0].nodes.len()],
        vec![false; w.vehicles[1].nodes.len()],
    ];
    let mut prev_pos: [Vec<Vec3>; 2] = [
        w.vehicles[0]
            .nodes
            .iter()
            .map(|n| w.vehicles[0].origin + n.pos)
            .collect(),
        w.vehicles[1]
            .nodes
            .iter()
            .map(|n| w.vehicles[1].origin + n.pos)
            .collect(),
    ];
    let mut worst = [Dir::default(); 2];
    // Per direction, each gap entry's path distance to the other car's nearest open edge.
    let mut gap_rim: [Vec<f32>; 2] = Default::default();
    let mut gap_rim_parts: [HashMap<String, usize>; 2] = Default::default();
    let (mut slab_max, mut slab_vol_max, mut slab_t) = (0.0f32, 0.0f32, 0.0f32);
    let mut entered: [HashMap<String, usize>; 2] = Default::default();
    // REVERSE=s: after the crash, both drivers reverse at full throttle for s
    // seconds (backing apart); the cars' separation speed shows a lock.
    let reverse: f32 = std::env::var("REVERSE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let crash_steps = (secs / 0.0005).round() as usize;
    let steps = ((secs + reverse) / 0.0005).round() as usize;
    let com = |s: &ExplicitBeams| {
        let (mut c, mut m, mut v) = (Vec3::ZERO, 0.0f32, Vec3::ZERO);
        for n in &s.nodes {
            c += n.pos * n.mass;
            v += n.vel * n.mass;
            m += n.mass;
        }
        (s.origin + c / m, v / m)
    };
    let every = 8; // 4 ms
    println!("  t ms | A in B: nodes (deepest mm) face/gap, paint n p99/max mm, tyre n max, edges | B in A: same | slab max mm, volume L | dead tris A/B");
    // FORCES=1: the vehicle-contact force on each car by height band and by
    // part (rearward push along the car, vertical), its nose-up moment about
    // the car's centre of mass, per 50 ms and integrated (impulse).
    let forces = std::env::var("FORCES").is_ok();
    let trace_face = std::env::var("TRACE_FACE").is_ok_and(|v| v == "1");
    let mut hist: std::collections::VecDeque<(usize, Vec<(Vec<Vec3>, Vec<Vec3>, Vec<Vec3>)>)> = Default::default();
    let mut traced = 0usize;
    if forces || std::env::var("LEDGER").is_ok() || trace_face {
        w.enable_ledger();
    }
    let part_of: HashMap<String, String> = l
        .built
        .rig
        .car
        .nodes
        .iter()
        .map(|n| (n.name.clone(), n.part.name().trim_start_matches("gen2_").to_string()))
        .collect();
    const BANDS: [f32; 5] = [0.30, 0.45, 0.60, 0.80, 9.0];
    let band = |y: f32| BANDS.iter().position(|&b| y < b).unwrap_or(4);
    // [car][band] (push, up), [car] part → (push, up), [car] nose-up moment.
    let mut win_band = [[[0.0f64; 2]; 5]; 2];
    let mut tot_band = [[[0.0f64; 2]; 5]; 2];
    let mut win_part: [std::collections::BTreeMap<String, [f64; 2]>; 2] = Default::default();
    let mut tot_part: [std::collections::BTreeMap<String, [f64; 2]>; 2] = Default::default();
    let mut win_m = [0.0f64; 2];
    let mut node_imp: HashMap<String, (f64, f32)> = HashMap::new();
    // The load path: beams crossing a transverse section SECTION m behind the
    // nose at spawn (default 0.55), their compression's component along the
    // car, by the beam's spawn height band and by member (node-name prefix).
    let section: f32 = std::env::var("SECTION").ok().and_then(|v| v.parse().ok()).unwrap_or(0.55);
    let member = |n: &str| -> String {
        let t = n.trim_end_matches(['L', 'R']);
        t.trim_end_matches(|c: char| c.is_ascii_digit()).trim_end_matches('_').to_string()
    };
    let cross: Vec<Vec<(usize, usize, String)>> = (0..2)
        .map(|c| {
            let s = &w.vehicles[c];
            let fwd = if c == 0 { 1.0f32 } else { -1.0 };
            let p0: Vec<Vec3> = s.nodes.iter().map(|n| s.origin + n.pos).collect();
            let nose = p0.iter().map(|p| fwd * p.z).fold(f32::MIN, f32::max);
            let behind = |p: Vec3| nose - fwd * p.z;
            s.beams
                .iter()
                .enumerate()
                .filter(|(_, b)| !b.broken && b.k > 0.0)
                .filter(|(_, b)| {
                    let (ba, bb) = (behind(p0[b.a as usize]), behind(p0[b.b as usize]));
                    (ba - section) * (bb - section) < 0.0
                })
                .map(|(j, b)| {
                    let y = 0.5 * (p0[b.a as usize].y + p0[b.b as usize].y);
                    let (na, nb) = (&l.desc.nodes[b.a as usize].name, &l.desc.nodes[b.b as usize].name);
                    let (ma, mb) = (member(na), member(nb));
                    let m = if ma <= mb { format!("{ma}-{mb}") } else { format!("{mb}-{ma}") };
                    (j, band(y), m)
                })
                .collect()
        })
        .collect();
    let mut sec_band = [[0.0f64; 5]; 2];
    let mut sec_mem: [std::collections::BTreeMap<String, f64>; 2] = Default::default();
    let mut sec_band_tot = [[0.0f64; 5]; 2];
    let mut sec_mem_tot: [std::collections::BTreeMap<String, f64>; 2] = Default::default();
    if forces {
        println!("  section {section} m behind the nose: {} / {} beams cross it", cross[0].len(), cross[1].len());
    }
    if forces {
        println!("  FORCES: per 50 ms window, mean kN pushing each car back / up, by node height band {BANDS:?} m; nose-up moment kN·m about its centre of mass");
    }
    for k in 0..=steps {
        if k > 0 {
            if k == crash_steps + 1 && reverse > 0.0 {
                for s in &mut w.vehicles {
                    s.set_drive_input(engine_softbody::explicit_drive::DriveInput {
                        forward: -1.0,
                        ..Default::default()
                    });
                }
                // UNCONTACT=1: vehicle contact off from here (the control: do
                // the cars back apart when nothing holds them?).
                if std::env::var("UNCONTACT").is_ok() {
                    w.contact = false;
                }
                let wheels: Vec<String> = w
                    .vehicles
                    .iter()
                    .map(|s| {
                        s.drive.as_ref().map_or("no drive".into(), |d| {
                            format!("{} of {} wheels on the ground", d.wheels.iter().filter(|x| x.contact).count(), d.wheels.len())
                        })
                    })
                    .collect();
                println!("  ---- {:.0} ms: both reverse ({wheels:?}) ----", k as f32 * 0.5);
            }
            w.step();
            if trace_face {
                // TRACE_FACE=1: the last 24 steps' positions, velocities and vehicle-contact
                // forces of both cars, for the face entries' traces.
                let snap: Vec<(Vec<Vec3>, Vec<Vec3>, Vec<Vec3>)> = (0..2)
                    .map(|c| {
                        let s = &w.vehicles[c];
                        let lg = w.ledger.as_ref();
                        (
                            s.nodes.iter().map(|n| s.origin + n.pos).collect(),
                            s.nodes.iter().map(|n| n.vel).collect(),
                            (0..s.nodes.len()).map(|i| lg.map_or(Vec3::ZERO, |l| l.vehicle_contact(c, i))).collect(),
                        )
                    })
                    .collect();
                hist.push_back((k, snap));
                if hist.len() > 24 {
                    hist.pop_front();
                }
            }
            if forces {
                for c in 0..2 {
                    let s = &w.vehicles[c];
                    let fwd = if c == 0 { 1.0f32 } else { -1.0 };
                    let lg = w.ledger.as_ref().unwrap();
                    let (mut cm, mut mm) = (Vec3::ZERO, 0.0f32);
                    for n in &s.nodes {
                        cm += n.pos * n.mass;
                        mm += n.mass;
                    }
                    let cm = s.origin + cm / mm;
                    for i in 0..s.nodes.len() {
                        let f = lg.vehicle_contact(c, i);
                        if f == Vec3::ZERO {
                            continue;
                        }
                        let p = s.origin + s.nodes[i].pos;
                        let push = (-fwd * f.z) as f64 * 0.0005;
                        let up = f.y as f64 * 0.0005;
                        let bi = band(p.y);
                        for (b, x) in [(&mut win_band[c][bi], 0), (&mut tot_band[c][bi], 0)] {
                            let _ = x;
                            b[0] += push;
                            b[1] += up;
                        }
                        let pn = part_of
                            .get(&l.desc.nodes[i].name)
                            .cloned()
                            .unwrap_or_else(|| "?".into());
                        for m in [&mut win_part[c], &mut tot_part[c]] {
                            let e = m.entry(pn.clone()).or_default();
                            e[0] += push;
                            e[1] += up;
                        }
                        if c == 0 {
                            let e = node_imp.entry(l.desc.nodes[i].name.clone()).or_insert((0.0, p.y));
                            e.0 += push;
                        }
                        let mx = (p - cm).cross(f).x;
                        win_m[c] += (-fwd * mx) as f64 * 0.0005;
                    }
                    for (j, bi, m) in &cross[c] {
                        let b = &s.beams[*j];
                        if b.broken {
                            continue;
                        }
                        let d = s.nodes[b.a as usize].pos - s.nodes[b.b as usize].pos;
                        let ax = (b.force * d.z.abs() / d.length().max(1e-6)) as f64 * 0.0005;
                        sec_band[c][*bi] += ax;
                        sec_band_tot[c][*bi] += ax;
                        *sec_mem[c].entry(m.clone()).or_default() += ax;
                        *sec_mem_tot[c].entry(m.clone()).or_default() += ax;
                    }
                }
            }
            if forces && k % 100 == 0 {
                let t = k as f32 * 0.0005;
                for c in 0..2 {
                    let bands: Vec<String> = win_band[c]
                        .iter()
                        .map(|b| format!("{:5.0}/{:4.0}", b[0] / 0.05 / 1e3, b[1] / 0.05 / 1e3))
                        .collect();
                    let parts: Vec<String> = win_part[c]
                        .iter()
                        .filter(|(_, v)| v[0].abs() > 0.05 * 0.05 * 1e3)
                        .map(|(p, v)| format!("{p} {:.0}/{:.0}", v[0] / 0.05 / 1e3, v[1] / 0.05 / 1e3))
                        .collect();
                    println!(
                        "  F {:3.0} ms car {} bands [{}] nose-up {:6.1} kN·m | {}",
                        t * 1e3,
                        ["A", "B"][c],
                        bands.join(" "),
                        win_m[c] / 0.05 / 1e3,
                        parts.join(", ")
                    );
                    let mut mem: Vec<(&String, &f64)> = sec_mem[c].iter().collect();
                    mem.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
                    println!(
                        "      section: compression kN by band [{}] | {}",
                        sec_band[c].iter().map(|v| format!("{:.0}", v / 0.05 / 1e3)).collect::<Vec<_>>().join(" "),
                        mem.iter().take(10).map(|(m, v)| format!("{m} {:.0}", *v / 0.05 / 1e3)).collect::<Vec<_>>().join(", ")
                    );
                    sec_band[c] = [0.0; 5];
                    sec_mem[c].clear();
                    win_band[c] = [[0.0; 2]; 5];
                    win_part[c].clear();
                    win_m[c] = 0.0;
                }
            }
            if forces && k == steps {
                let mut v: Vec<(&String, &(f64, f32))> = node_imp.iter().collect();
                v.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
                println!(
                    "  car A nodes by push impulse (kN·s, first contact height m): {}",
                    v.iter().take(30).map(|(n, (i, y))| format!("{n} {:.2} @{y:.2}", i / 1e3)).collect::<Vec<_>>().join(", ")
                );
                for c in 0..2 {
                    println!(
                        "  impulse car {} (kN·s push/up) by band: {:?}; by part: {:?}",
                        ["A", "B"][c],
                        tot_band[c].iter().map(|b| format!("{:.2}/{:.2}", b[0] / 1e3, b[1] / 1e3)).collect::<Vec<_>>(),
                        tot_part[c].iter().map(|(p, v)| format!("{p} {:.2}/{:.2}", v[0] / 1e3, v[1] / 1e3)).collect::<Vec<_>>()
                    );
                    println!(
                        "  section impulse car {} (kN·s compression along the car) by band: {:?}",
                        ["A", "B"][c],
                        sec_band_tot[c].iter().map(|v| format!("{:.2}", v / 1e3)).collect::<Vec<_>>()
                    );
                    let mut mem: Vec<(&String, &f64)> = sec_mem_tot[c].iter().collect();
                    mem.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
                    println!(
                        "    by member: {}",
                        mem.iter().take(24).map(|(m, v)| format!("{m} {:.2}", *v / 1e3)).collect::<Vec<_>>().join(", ")
                    );
                }
            }
        }
        if k % every != 0 {
            continue;
        }
        let full = k % (every * 5) == 0 || k == steps;
        let hulls = [hull(&w.vehicles[0]), hull(&w.vehicles[1])];
        let boxes = [aabb(&hulls[0]), aabb(&hulls[1])];
        let mut now = [Dir::default(); 2];
        let mut rend: [Vec<Vec3>; 2] = [Vec::new(), Vec::new()];
        if full {
            for c in 0..2 {
                let s = &w.vehicles[c];
                let n: Vec<Vec3> = s.nodes.iter().map(|n| n.pos).collect();
                rend[c] = deform(&vis[c], &n)
                    .0
                    .into_iter()
                    .map(|q| q + s.origin)
                    .collect();
            }
        }
        for c in 0..2 {
            let o = 1 - c;
            let (x, y) = (&w.vehicles[c], &hulls[o]);
            // The other car's open skin edges (on one live triangle) now: a hole's rim.
            let rims: Vec<(Vec3, Vec3, u32, u32)> = {
                let s = &w.vehicles[o];
                let mut cnt: HashMap<(u32, u32), usize> = HashMap::new();
                for t in 0..s.triangles.len() {
                    if s.tri_dead.get(t).copied().unwrap_or(false) || s.tri_anti_only(t) {
                        continue;
                    }
                    let n = s.triangles[t].nodes;
                    for k in 0..3 {
                        let (a, b) = (n[k], n[(k + 1) % 3]);
                        *cnt.entry((a.min(b), a.max(b))).or_default() += 1;
                    }
                }
                cnt.into_iter()
                    .filter(|(_, c)| *c == 1)
                    .map(|((a, b), _)| (s.origin + s.nodes[a as usize].pos, s.origin + s.nodes[b as usize].pos, a, b))
                    .collect()
            };
            let mut r = Dir::default();
            let mut ins = vec![false; x.nodes.len()];
            for &i in &x.contact_points {
                let p = x.origin + x.nodes[i as usize].pos;
                if !inside_box(p, boxes[o]) || winding(p, y) < 0.5 {
                    continue;
                }
                ins[i as usize] = true;
                {
                    // LOCK: this node's held sides for the other car's triangles.
                    let ys = &w.vehicles[o];
                    let mut kind = 0; // 0 free, 1 pushed out, 2 held inside
                    if let Some(cw) = x.clip_sides.get(i as usize) {
                        for &word in &cw[..2] {
                            if word == engine_softbody::explicit_contact::CLIP_EMPTY
                                || word >> 21 != o as u32
                            {
                                continue;
                            }
                            let t = ((word >> 1) & engine_softbody::explicit_contact::CLIP_MAX_TRIANGLES) as usize;
                            let side = if word & 1 != 0 { 1.0 } else { -1.0 };
                            let [i0, i1, i2] = ys.triangles[t].nodes.map(|q| ys.origin + ys.nodes[q as usize].pos);
                            let fr = engine_softbody::explicit_contact::ContactFrame::new(i0, i1, i2);
                            let (_, _, _, delta) = fr.local(p);
                            kind = kind.max(if side * delta >= 0.0 { 2 } else { 1 });
                        }
                    }
                    match kind {
                        2 => r.held_in += 1,
                        1 => r.push_out += 1,
                        _ => r.free += 1,
                    }
                }
                r.nodes += 1;
                r.node_depth = r.node_depth.max(dist(p, y));
                if !prev_in[c][i as usize] {
                    let q = prev_pos[c][i as usize];
                    if y.iter().any(|t| seg_tri(q, p, t)) {
                        r.face += 1;
                        if trace_face && traced < 4 {
                            traced += 1;
                            let ys = &w.vehicles[o];
                            let crossed = (0..ys.triangles.len()).find(|&t| {
                                !ys.tri_dead.get(t).copied().unwrap_or(false)
                                    && seg_tri(q, p, &ys.triangles[t].nodes.map(|z| ys.origin + ys.nodes[z as usize].pos))
                            });
                            if let Some(t) = crossed {
                                let tn = ys.triangles[t].nodes;
                                println!(
                                    "  TRACE {} node {} of car {c} entered car {o} through triangle {t} {:?} by {:.0} ms:",
                                    if c == 0 { "A" } else { "B" },
                                    l.desc.nodes[i as usize].name,
                                    tn.map(|z| l.desc.nodes[z as usize].name.clone()),
                                    k as f32 * 0.5
                                );
                                println!("    step | signed distance (mm, + outside) | approach speed m/s (+ closing) | contact force on node along n (N) | on the triangle's nodes (N)");
                                for (ks, snap) in &hist {
                                    let (pa, va, fa) = (&snap[c].0, &snap[c].1, &snap[c].2);
                                    let (pb, vb, fb) = (&snap[o].0, &snap[o].1, &snap[o].2);
                                    let [x0, x1, x2] = tn.map(|z| pb[z as usize]);
                                    let nrm = (x1 - x0).cross(x2 - x0).normalize_or_zero();
                                    let d = nrm.dot(pa[i as usize] - x0);
                                    let vt = (vb[tn[0] as usize] + vb[tn[1] as usize] + vb[tn[2] as usize]) / 3.0;
                                    let vn = -(va[i as usize] - vt).dot(nrm);
                                    let ft = fb[tn[0] as usize] + fb[tn[1] as usize] + fb[tn[2] as usize];
                                    println!(
                                        "    {:5} | {:8.1} | {:7.2} | {:9.0} | {:9.0}",
                                        ks,
                                        d * 1e3,
                                        vn,
                                        fa[i as usize].dot(nrm),
                                        ft.dot(nrm)
                                    );
                                }
                            }
                        }
                    } else {
                        r.gap += 1;
                        // Its path's nearest approach to a rim (m).
                        let (q0, p1) = (prev_pos[c][i as usize], p);
                        let (dmin, rim) = rims
                            .iter()
                            .map(|&(ra, rb, ia, ib)| {
                                let (s1, t1) = seg_seg(q0, p1, ra, rb);
                                ((q0 + (p1 - q0) * s1).distance(ra + (rb - ra) * t1), (ia, ib))
                            })
                            .fold((f32::MAX, (0, 0)), |m, x| if x.0 < m.0 { x } else { m });
                        gap_rim[c].push(dmin);
                        if dmin <= 0.2 {
                            let part = |i: u32| {
                                let n = &l.desc.nodes[i as usize].name;
                                n.trim_end_matches(|ch: char| ch.is_ascii_digit() || ch == '_').to_string()
                            };
                            let (pa, pb) = (part(rim.0), part(rim.1));
                            let key = if pa <= pb { format!("{pa}-{pb}") } else { format!("{pb}-{pa}") };
                            *gap_rim_parts[c].entry(key).or_default() += 1;
                        }
                    }
                    *entered[c]
                        .entry(l.desc.nodes[i as usize].name.clone())
                        .or_default() += 1;
                }
            }
            prev_in[c] = ins;
            prev_pos[c] = x.nodes.iter().map(|n| x.origin + n.pos).collect();
            // Skin edges through the other's triangles.
            for (i, j) in edges_of(x) {
                let (p, q) = (
                    x.origin + x.nodes[i as usize].pos,
                    x.origin + x.nodes[j as usize].pos,
                );
                if !inside_box(p, boxes[o]) && !inside_box(q, boxes[o]) {
                    continue;
                }
                r.edges += y.iter().filter(|t| seg_tri(p, q, t)).count();
            }
            if full {
                let mut dp: Vec<f32> = Vec::new();
                for (q, &m) in rend[c].iter().zip(&vis[c].material) {
                    if (m != mat::PAINT && m != mat::TYRE) || !inside_box(*q, boxes[o]) {
                        continue;
                    }
                    if winding(*q, y) < 0.5 {
                        continue;
                    }
                    let dd = dist(*q, y);
                    if m == mat::TYRE {
                        r.tyre += 1;
                        r.tyre_depth = r.tyre_depth.max(dd);
                    } else {
                        r.paint += 1;
                        dp.push(dd);
                    }
                }
                r.paint_p99 = pct(&mut dp, 0.99);
                r.paint_depth = pct(&mut dp, 1.0);
            }
            now[c] = r;
            let wv = &mut worst[c];
            wv.nodes = wv.nodes.max(r.nodes);
            wv.node_depth = wv.node_depth.max(r.node_depth);
            wv.face += r.face;
            wv.gap += r.gap;
            wv.paint = wv.paint.max(r.paint);
            wv.paint_p99 = wv.paint_p99.max(r.paint_p99);
            wv.paint_depth = wv.paint_depth.max(r.paint_depth);
            wv.tyre = wv.tyre.max(r.tyre);
            wv.tyre_depth = wv.tyre_depth.max(r.tyre_depth);
            wv.edges = wv.edges.max(r.edges);
            wv.held_in = wv.held_in.max(r.held_in);
            wv.push_out = wv.push_out.max(r.push_out);
            wv.free = wv.free.max(r.free);
        }
        let (mut smax, mut svol) = (0.0f32, 0.0f32);
        if full {
            // Slab overlap: per 5 cm (x, y) bin, A's frontmost (max z) render
            // vertex against B's frontmost (min z).
            let bin = 0.05f32;
            let key = |q: Vec3| ((q.x / bin).floor() as i32, (q.y / bin).floor() as i32);
            let mut af: HashMap<(i32, i32), f32> = HashMap::new();
            let mut bf: HashMap<(i32, i32), f32> = HashMap::new();
            for (q, &m) in rend[0].iter().zip(&vis[0].material) {
                if m == mat::PAINT || m == mat::TYRE {
                    let e = af.entry(key(*q)).or_insert(f32::MIN);
                    *e = e.max(q.z);
                }
            }
            for (q, &m) in rend[1].iter().zip(&vis[1].material) {
                if m == mat::PAINT || m == mat::TYRE {
                    let e = bf.entry(key(*q)).or_insert(f32::MAX);
                    *e = e.min(q.z);
                }
            }
            for (kk, za) in &af {
                if let Some(zb) = bf.get(kk) {
                    let o = za - zb;
                    if o > 0.0 {
                        smax = smax.max(o);
                        svol += o * bin * bin;
                    }
                }
            }
            if smax > slab_max {
                slab_max = smax;
                slab_t = k as f32 * 0.0005;
            }
            slab_vol_max = slab_vol_max.max(svol);
        }
        let dead = |s: &ExplicitBeams| s.tri_dead.iter().filter(|&&x| x).count();
        if std::env::var("PITCH").is_ok() && full {
            // The front hubs' height (car A, B): a nose lifting or diving.
            let hy = |s: &ExplicitBeams| {
                ["hubFiL", "hubFiR"]
                    .iter()
                    .filter_map(|n| l.desc.nodes.iter().position(|x| x.name == *n))
                    .map(|i| s.origin.y + s.nodes[i].pos.y)
                    .sum::<f32>()
                    / 2.0
            };
            // Car A's rear hubs and its cabin's pitch (the sill's front
            // station above its rear one over their distance; nose up > 0).
            let at = |s: &ExplicitBeams, n: &str| {
                l.desc.nodes.iter().position(|x| x.name == n).map(|i| s.origin + s.nodes[i].pos)
            };
            let a = &w.vehicles[0];
            let rear = ["hubRiL", "hubRiR"].iter().filter_map(|n| at(a, n)).map(|p| p.y).sum::<f32>() / 2.0;
            let pitch = match (at(a, "slt0L"), at(a, "slt4L")) {
                (Some(f), Some(r)) => ((f.y - r.y) / (f.z - r.z).abs().max(1e-3)).atan().to_degrees(),
                _ => f32::NAN,
            };
            // Car A's nodes on the ground (within 3 cm) and its vertical speed (centre of mass).
            let mut low: Vec<(f32, &str)> = (0..a.nodes.len())
                .map(|i| (a.origin.y + a.nodes[i].pos.y, l.desc.nodes[i].name.as_str()))
                .filter(|x| x.0 < 0.03)
                .collect();
            low.sort_by(|x, y| x.0.total_cmp(&y.0));
            let (mut py, mut mm) = (0.0f32, 0.0f32);
            for n in &a.nodes {
                py += n.vel.y * n.mass;
                mm += n.mass;
            }
            if let Ok(list) = std::env::var("HEIGHTS") {
                // HEIGHTS=a,b,c: those nodes' heights (car A), m.
                println!(
                    "       heights: {}",
                    list.split(',')
                        .filter_map(|n| at(a, n).map(|p| format!("{n} {:.3}", p.y)))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            println!(
                "       A vy {:.2} m/s; on the ground {}: {}",
                py / mm,
                low.len(),
                low.iter().take(8).map(|x| format!("{} {:.3}", x.1, x.0)).collect::<Vec<_>>().join(", ")
            );
            println!(
                "  {:4.0} ms front hubs y A {:.3} B {:.3} m; A rear hubs {:.3} m, sill pitch {:.2} deg",
                k as f32 * 0.5,
                hy(&w.vehicles[0]),
                hy(&w.vehicles[1]),
                rear,
                pitch
            );
        }
        if full {
            let f = |r: &Dir| {
                format!(
                    "{:3} ({:4.0}) {:2}/{:2} in/out/free {:3}/{:3}/{:3}, {:5} {:4.0}/{:4.0}, {:4} {:4.0}, {:3}",
                    r.nodes,
                    r.node_depth * 1e3,
                    r.face,
                    r.gap,
                    r.held_in,
                    r.push_out,
                    r.free,
                    r.paint,
                    r.paint_p99 * 1e3,
                    r.paint_depth * 1e3,
                    r.tyre,
                    r.tyre_depth * 1e3,
                    r.edges
                )
            };
            let ((ca, va), (cb, vb)) = (com(&w.vehicles[0]), com(&w.vehicles[1]));
            let ax = (cb - ca).normalize_or_zero();
            let ke: f32 = w
                .vehicles
                .iter()
                .flat_map(|s| s.nodes.iter())
                .map(|n| 0.5 * n.mass * n.vel.length_squared())
                .sum();
            println!(
                "       centres {:.3} m apart, separating at {:.2} m/s; kinetic energy {:.1} kJ",
                ca.distance(cb),
                (vb - va).dot(ax),
                ke * 1e-3
            );
            if w.edge_stats[0] > 0 {
                println!("       edge pass so far: tested {} at-end {} parallel {} not-facing {} deeper {} hits {} crossed-hits {}", w.edge_stats[0], w.edge_stats[1], w.edge_stats[2], w.edge_stats[3], w.edge_stats[4], w.edge_stats[5], w.edge_stats[6]);
                let f = w.edge_first;
                if f.iter().any(|&x| x > 0) {
                    println!("       hits first seen crossed, by the step before: at-end {} parallel {} not-facing {} outside-band {} deeper {} untested {}; held pairs overrun past the depth {}", f[0], f[1], f[2], f[3], f[4], f[6], f[5]);
                }
            }
            if let Some(lg) = w.ledger.as_ref() {
                // LEDGER=1: vehicle contact's work so far (normal, friction, position
                // steps) and self contact's, kJ; contacts must not do positive work.
                println!(
                    "       ledger kJ: vehicle contact normal {:.2} friction {:.2} steps {:.2}; self contact {:.2}; mechanical {:.1} (start {:.1})",
                    lg.w_contact_normal * 1e-3,
                    lg.w_contact_friction * 1e-3,
                    lg.w_contact_shift * 1e-3,
                    lg.w_self_contact * 1e-3,
                    lg.mechanical() * 1e-3,
                    lg.start * 1e-3
                );
            }
            println!(
                "  {:4.0} | {} | {} | {:4.0}, {:5.1} | {}/{}",
                k as f32 * 0.5,
                f(&now[0]),
                f(&now[1]),
                smax * 1e3,
                svol * 1e3,
                dead(&w.vehicles[0]),
                dead(&w.vehicles[1])
            );
        }
    }
    {
        let s = &w.vehicles[0];
        let dead: Vec<[String; 3]> = (0..s.triangles.len())
            .filter(|&t| s.tri_dead.get(t).copied().unwrap_or(false))
            .map(|t| {
                s.triangles[t]
                    .nodes
                    .map(|k| l.desc.nodes[k as usize].name.clone())
            })
            .collect();
        println!("  car A's dead triangles: {dead:?}");
    }
    for (c, name) in ["A in B", "B in A"].iter().enumerate() {
        let r = worst[c];
        println!(
            "  worst {name}: nodes {} (deepest {:.0} mm; entries through a face {}, through a gap {}; held inside {}, pushed out {}, free {}), paint vertices {} (p99 {:.0}, deepest {:.0} mm), tyre vertices {} (deepest {:.0} mm), skin edges through triangles {}",
            r.nodes,
            r.node_depth * 1e3,
            r.face,
            r.gap,
            r.held_in,
            r.push_out,
            r.free,
            r.paint,
            r.paint_p99 * 1e3,
            r.paint_depth * 1e3,
            r.tyre,
            r.tyre_depth * 1e3,
            r.edges
        );
        let mut e: Vec<(&String, &usize)> = entered[c].iter().collect();
        e.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        println!(
            "    nodes that entered: {:?}",
            e.iter().take(40).collect::<Vec<_>>()
        );
    }
    for (c, name) in ["A in B", "B in A"].iter().enumerate() {
        let v = &gap_rim[c];
        let within = |d: f32| v.iter().filter(|&&x| x <= d).count();
        println!(
            "  {name}: gap entries {} by their path's distance to the other's nearest open skin edge: <= 5 cm {}, <= 10 cm {}, <= 20 cm {}, farther {}",
            v.len(),
            within(0.05),
            within(0.10),
            within(0.20),
            v.len() - within(0.20)
        );
        let mut by: Vec<(&String, &usize)> = gap_rim_parts[c].iter().collect();
        by.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        println!("    the open edge each passed nearest, by its nodes' part: {:?}", by);
    }
    println!(
        "  slab overlap: deepest {:.0} mm at {:.0} ms, largest volume {:.1} L",
        slab_max * 1e3,
        slab_t * 1e3,
        slab_vol_max * 1e3
    );
    Ok(())
}
