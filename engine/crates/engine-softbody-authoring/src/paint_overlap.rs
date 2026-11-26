//! What is seen of two cars inside each other: the length of the lines along which one
//! car's painted render triangles cut the other's (m). It needs no closed surface and no
//! collision skin. Torn triangles (an edge past 3 times rest: not drawn) are left out.

use crate::authoring::rig;
use crate::authoring::skin::{deform, mat, ShowcaseVisual};
use crate::vehicle_compare::settings;
use engine_softbody::desc::SoftVehicleDesc;
use engine_softbody::explicit_beams::{ExplicitBeams, StaticShape};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use engine_softbody::explicit_drive::DriveInput;
use glam::Vec3;
use std::collections::HashMap;

pub type Tri = [Vec3; 3];
thread_local!(static FREQ: std::cell::RefCell<(Vec3, bool, u32, f32)> = const { std::cell::RefCell::new((Vec3::ZERO, false, 0, 0.0)) });



/// (A measurement, PAINT_ON_SKIN=1) The paint embedded in its part's collision skin: per render
/// vertex the closest point on its own part's triangles as built (clamped, so never
/// extrapolated), as the barycentric blend of the triangle's nodes, each carrying the vertex's
/// rest offset from it by its own rotation (its skin neighbours', `rotations`). Continuous
/// across triangle edges.
struct OnSkin {
    at: Vec<Option<([u32; 3], [f32; 3])>>,
    rest_p: Vec<Vec3>,
    rest_n: Vec<Vec3>,
    near: Vec<Vec<u32>>,
}
static ON_SKIN: std::sync::OnceLock<OnSkin> = std::sync::OnceLock::new();

/// Each node's rotation from rest: the one best fitting its skin neighbours (the neighbours'
/// area-free normals stand in for the third direction of a flat patch).
fn rotations(o: &OnSkin, x: &[Vec3]) -> Vec<glam::Quat> {
    (0..x.len())
        .map(|i| {
            let nb = &o.near[i];
            if nb.len() < 2 {
                return glam::Quat::IDENTITY;
            }
            let pairs: Vec<(Vec3, Vec3)> = nb.iter().map(|&j| (o.rest_n[j as usize] - o.rest_n[i], x[j as usize] - x[i])).collect();
            let mut m = glam::DMat3::ZERO;
            let outer = |a: Vec3, b: Vec3, w: f64| { let (a, b) = (a.as_dvec3(), b.as_dvec3()); glam::DMat3::from_cols(b * a.x, b * a.y, b * a.z) * w };
            for (k, (r, d)) in pairs.iter().enumerate() {
                m += outer(*r, *d, 1.0);
                let (r2, d2) = pairs[(k + 1) % pairs.len()];
                let (nr, nd) = (r.cross(r2), d.cross(d2));
                let l = r.length().max(1e-6);
                if nr.length() > 1e-8 && nd.length() > 1e-8 {
                    m += outer(nr.normalize() * l, nd.normalize() * l, 1.0);
                }
            }
            // The rotation of m (Müller et al. 2016, iterated from the identity).
            let mut q = glam::DQuat::IDENTITY;
            for _ in 0..60 {
                let r = glam::DMat3::from_quat(q);
                let c = r.x_axis.cross(m.x_axis) + r.y_axis.cross(m.y_axis) + r.z_axis.cross(m.z_axis);
                let d = r.x_axis.dot(m.x_axis) + r.y_axis.dot(m.y_axis) + r.z_axis.dot(m.z_axis);
                let w = c / (d.abs() + 1e-9);
                let a = w.length();
                if a < 1e-9 { break; }
                q = (glam::DQuat::from_axis_angle(w / a, a) * q).normalize();
            }
            q.as_quat()
        })
        .collect()
}

fn on_skin(v: &ShowcaseVisual, s: &ExplicitBeams, part: impl Fn(u32) -> String) -> OnSkin {
    let r: Vec<Vec3> = s.nodes.iter().map(|n| n.pos).collect();
    let p = deform(v, &r).0;
    let mut near: Vec<Vec<u32>> = vec![Vec::new(); r.len()];
    let mut tri: HashMap<String, Vec<[u32; 3]>> = HashMap::new();
    for t in (0..s.triangles.len()).filter(|&t| !s.tri_anti_only(t)) {
        let n = s.triangles[t].nodes;
        let name = part(n[0]);
        // (A seam bridge, its nodes of two parts: none's.)
        if n.iter().any(|&k| part(k) != name) { continue; }
        for k in 0..3 {
            for m in 0..3 {
                if m != k && !near[n[k] as usize].contains(&n[m]) { near[n[k] as usize].push(n[m]); }
            }
        }
        tri.entry(name).or_default().push(n);
    }
    let at = (0..p.len())
        .map(|i| {
            if v.material[i] != mat::PAINT { return None; }
            // Its part: of the nodes it follows (one part's pool), the nearest as built.
            let (a, b) = (v.wt_start[i] as usize, v.wt_start[i + 1] as usize);
            let top = (a..b).map(|k| v.wt_node[k]).min_by(|&x, &y| r[x as usize].distance(p[i]).total_cmp(&r[y as usize].distance(p[i]))).unwrap_or(v.bind_nodes[i][0]);
            let mut best = (f32::MAX, [0u32; 3], [0.0f32; 3]);
            for &n in tri.get(&part(top))? {
                let q = n.map(|k| r[k as usize]);
                let w = closest(p[i], &q);
                let d = p[i].distance(q[0] * w[0] + q[1] * w[1] + q[2] * w[2]);
                if d < best.0 { best = (d, n, w); }
            }
            (best.0 < f32::MAX).then_some((best.1, best.2))
        })
        .collect();
    OnSkin { at, rest_p: p, rest_n: r, near }
}

/// The closest point of a triangle to p, as weights (Ericson 5.1.5).
fn closest(p: Vec3, t: &Tri) -> [f32; 3] {
    let (ab, ac, ap) = (t[1] - t[0], t[2] - t[0], p - t[0]);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 { return [1.0, 0.0, 0.0]; }
    let bp = p - t[1];
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 { return [0.0, 1.0, 0.0]; }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 { let v = d1 / (d1 - d3); return [1.0 - v, v, 0.0]; }
    let cp = p - t[2];
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 { return [0.0, 0.0, 1.0]; }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 { let w = d2 / (d2 - d6); return [1.0 - w, 0.0, w]; }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 { let w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); return [0.0, 1.0 - w, w]; }
    let k = 1.0 / (va + vb + vc);
    [1.0 - vb * k - vc * k, vb * k, vc * k]
}

/// The crossing segment's length of two triangles (0: none, or coplanar).
fn cut(a: &Tri, b: &Tri) -> f32 {
    let mut pts: Vec<Vec3> = Vec::new();
    for (s, t) in [(a, b), (b, a)] {
        let n = (t[1] - t[0]).cross(t[2] - t[0]);
        for k in 0..3 {
            let (p, q) = (s[k], s[(k + 1) % 3]);
            let (dp, dq) = ((p - t[0]).dot(n), (q - t[0]).dot(n));
            if dp * dq >= 0.0 {
                continue;
            }
            let x = p + (q - p) * (dp / (dp - dq));
            if (0..3).all(|e| (t[(e + 1) % 3] - t[e]).cross(x - t[e]).dot(n) >= 0.0) {
                pts.push(x);
            }
        }
    }
    let mut d = 0.0f32;
    for i in 0..pts.len() {
        for j in 0..i {
            d = d.max(pts[i].distance(pts[j]));
        }
    }
    d
}

/// Every pair of `a`'s and `b`'s triangles with a grid cell in common.
fn near_pairs(a: &[Tri], b: &[Tri], mut f: impl FnMut(usize, usize)) {
    const CELL: f32 = 0.08;
    let cells = |t: &Tri| {
        let (lo, hi) = (t[0].min(t[1]).min(t[2]) / CELL, t[0].max(t[1]).max(t[2]) / CELL);
        (lo.floor().as_ivec3(), hi.floor().as_ivec3())
    };
    let each = |t: &Tri, g: &mut dyn FnMut((i32, i32, i32))| {
        let (lo, hi) = cells(t);
        for x in lo.x..=hi.x {
            for y in lo.y..=hi.y {
                for z in lo.z..=hi.z {
                    g((x, y, z));
                }
            }
        }
    };
    let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
    for (k, t) in b.iter().enumerate() {
        each(t, &mut |c| grid.entry(c).or_default().push(k as u32));
    }
    let mut near: Vec<u32> = Vec::new();
    for (i, t) in a.iter().enumerate() {
        near.clear();
        each(t, &mut |c| near.extend(grid.get(&c).into_iter().flatten()));
        near.sort_unstable();
        near.dedup();
        near.iter().for_each(|&k| f(i, k as usize));
    }
}

/// Total length of the lines where `a`'s triangles cut `b`'s.
pub fn crossing(a: &[Tri], b: &[Tri]) -> f32 {
    let mut sum = 0.0;
    near_pairs(a, b, |i, j| sum += cut(&a[i], &b[j]));
    sum
}

/// The car's painted triangles now (torn ones left out), each with its index among the
/// painted triangles.
fn paint_ids(v: &ShowcaseVisual, rest: &[Vec3], s: &ExplicitBeams) -> Vec<(u32, Tri)> {
    let n: Vec<Vec3> = s.nodes.iter().map(|n| s.origin + n.pos).collect();
    let mut p = deform(v, &n).0;
    // (PAINT_STRETCHED=1: triangles stretched past 3 times rest too; the page draws weighted paint so.)
    let stretched = std::env::var("PAINT_STRETCHED").is_ok_and(|x| x == "1");
    if let Some(on) = ON_SKIN.get() {
        let rot = rotations(on, &n);
        for (i, b) in on.at.iter().enumerate() {
            let Some((t, w)) = b else { continue };
            p[i] = (0..3).map(|k| { let j = t[k] as usize; (n[j] + rot[j] * (on.rest_p[i] - on.rest_n[j])) * w[k] }).sum();
        }
    }
    v.indices
        .chunks_exact(3)
        .filter(|t| t.iter().all(|&i| v.material[i as usize] == mat::PAINT))
        .map(|t| [t[0] as usize, t[1] as usize, t[2] as usize])
        .enumerate()
        .filter(|(_, t)| (0..3).all(|k| {
            let (i, j) = (t[k], t[(k + 1) % 3]);
            p[i].is_finite() && (stretched || p[i].distance(p[j]) <= 3.0 * rest[i].distance(rest[j]) + 0.01)
        }))
        .map(|(k, t)| (k as u32, t.map(|i| p[i])))
        .collect()
}

/// A solid as vehicle contact sees it: the live outer collision triangles at their contact
/// offsets, and per corner how far that surface stands outside the paint (m, signed; [`gaps`]).
pub struct Solid { pub tris: Vec<Tri>, gap: Vec<[f32; 3]> }

/// How deep `a`'s points lie inside `b`'s paint (m; 0: none): the largest distance to `b`'s
/// surface, less the surface's own gap to the paint there, of a point where `b`'s generalized
/// winding number is over a half (it needs no closed surface; Jacobson et al. 2013).
/// (Well over a half: at an open wheel arch, real empty space, the winding is about a half.)
const INSIDE: f32 = 0.75;
pub fn depth(a: &[Tri], b: &Solid) -> f32 { deepest(a, b).0 }

/// [`depth`] with its point.
pub fn deepest(a: &[Tri], b: &Solid) -> (f32, Vec3) {
    let (lo, hi) = b.tris.iter().flatten().fold((Vec3::MAX, Vec3::MIN), |(l, h), p| (l.min(*p), h.max(*p)));
    let mut worst = (0.0f32, Vec3::ZERO);
    for t in a {
        for p in [t[0], (t[0] + t[1] + t[2]) / 3.0] {
            if p.cmplt(lo).any() || p.cmpgt(hi).any() {
                continue;
            }
            let (mut d, mut g) = (f32::MAX, 0.0);
            for (q, gap) in b.tris.iter().zip(&b.gap) {
                let w = closest(p, q);
                let x = p.distance(q[0] * w[0] + q[1] * w[1] + q[2] * w[2]);
                if x < d {
                    (d, g) = (x, gap[0] * w[0] + gap[1] * w[1] + gap[2] * w[2]);
                }
            }
            if d - g > worst.0 && winding(p, &b.tris) > INSIDE {
                worst = (d - g, p);
            }
        }
    }
    worst
}

/// Per node of the car as built: how far its contact surface stands outside the nearest of the
/// visible mesh (`drawn`: every triangle), along the surface's normal there.
fn gaps(s: &ExplicitBeams, drawn: &[Tri]) -> Vec<f32> {
    let p = engine_softbody::explicit_contact::contact_surface_positions(s);
    let mut n = vec![Vec3::ZERO; p.len()];
    for t in live_outer(s) {
        let [a, b, c] = s.triangles[t].nodes.map(|i| p[i as usize]);
        s.triangles[t].nodes.iter().for_each(|&i| n[i as usize] += (b - a).cross(c - a).normalize_or_zero());
    }
    (0..p.len())
        .map(|i| {
            if n[i] == Vec3::ZERO { return 0.0; }
            let x = s.origin + p[i];
            let c = drawn.iter().map(|t| { let w = closest(x, t); t[0] * w[0] + t[1] * w[1] + t[2] * w[2] }).min_by(|a, b| x.distance(*a).total_cmp(&x.distance(*b))).unwrap_or(x);
            x.distance(c).copysign((x - c).dot(n[i]))
        })
        .collect()
}

fn live_outer(s: &ExplicitBeams) -> impl Iterator<Item = usize> + '_ {
    (0..s.triangles.len()).filter(|&t| !s.tri_dead.get(t).copied().unwrap_or(false) && !s.tri_anti_only(t) && s.tri_outer.get(t).copied().unwrap_or(true))
}

/// The car's closed interior solids (dash, console, engine, fuel tank, seats, radiator: their
/// collision hulls, by node name) and how deep the car's own paint lies inside them (m): per
/// painted vertex not bound to the solid, its depth past the solid's surface where the solid's
/// winding number is over [`INSIDE`]; the largest, over vertices not inside one as built (`base`).
pub const SOLIDS: [&str; 8] = ["das_", "con_", "eng_", "fuet_", "seafl_", "seafr_", "sear_", "rad_"];
fn solid_depths(v: &ShowcaseVisual, rest: &[Vec3], s: &ExplicitBeams, names: &[String]) -> Vec<f32> {
    let np: Vec<Vec3> = s.nodes.iter().map(|n| s.origin + n.pos).collect();
    let p = deform(v, &np).0;
    let mut out = vec![0.0f32; p.len()];
    for pre in SOLIDS {
        let of = |i: u32| names[i as usize].starts_with(pre);
        let tris: Vec<Tri> = s.triangles.iter().filter(|t| t.nodes.iter().all(|&i| of(i))).map(|t| t.nodes.map(|i| np[i as usize])).collect();
        if std::env::var("SOLID_LOG").is_ok() { eprintln!("solid {pre}: {} triangles", tris.len()); }
        if tris.len() < 4 {
            continue;
        }
        let (lo, hi) = tris.iter().flatten().fold((Vec3::MAX, Vec3::MIN), |(l, h), q| (l.min(*q), h.max(*q)));
        for (i, x) in p.iter().enumerate() {
            if v.material[i] != mat::PAINT || v.bind_nodes[i].iter().any(|&n| of(n)) || x.cmplt(lo).any() || x.cmpgt(hi).any() {
                continue;
            }
            let _ = rest;
            if winding(*x, &tris) > INSIDE {
                out[i] = out[i].max(tris.iter().map(|t| distance(*x, t)).fold(f32::MAX, f32::min));
            }
        }
    }
    out
}

/// The distance from `p` to the triangle.
fn distance(p: Vec3, t: &Tri) -> f32 {
    let w = closest(p, t);
    p.distance(t[0] * w[0] + t[1] * w[1] + t[2] * w[2])
}

/// `t`'s generalized winding number at `p`: 1 inside, 0 outside a closed outward surface.
pub fn winding(p: Vec3, t: &[Tri]) -> f32 {
    let mut sum = 0.0f64;
    for q in t {
        let [a, b, c] = q.map(|x| (x - p).as_dvec3());
        let (la, lb, lc) = (a.length(), b.length(), c.length());
        let den = la * lb * lc + a.dot(b) * lc + b.dot(c) * la + c.dot(a) * lb;
        sum += a.dot(b.cross(c)).atan2(den);
    }
    (sum / (2.0 * std::f64::consts::PI)) as f32
}

fn surface(s: &ExplicitBeams, gap: &[f32]) -> Solid {
    let p = engine_softbody::explicit_contact::contact_surface_positions(s);
    let tri = |t: usize| s.triangles[t].nodes;
    Solid {
        tris: live_outer(s).map(|t| tri(t).map(|i| s.origin + p[i as usize])).collect(),
        gap: live_outer(s).map(|t| tri(t).map(|i| gap[i as usize])).collect(),
    }
}

pub fn paint(v: &ShowcaseVisual, rest: &[Vec3], s: &ExplicitBeams) -> Vec<Tri> {
    paint_ids(v, rest, s).into_iter().map(|x| x.1).collect()
}

/// A car's own painted triangles cutting each other: the pairs and their line's length.
/// Pairs with a corner in common (within 1 mm now) are left out.
fn self_pairs(t: &[(u32, Tri)]) -> Vec<((u32, u32), f32)> {
    let only: Vec<Tri> = t.iter().map(|x| x.1).collect();
    let mut out = Vec::new();
    near_pairs(&only, &only, |i, j| {
        if j <= i {
            return;
        }
        let (a, b) = (&only[i], &only[j]);
        if a.iter().any(|p| b.iter().any(|q| p.distance_squared(*q) < 1e-6)) {
            return;
        }
        let l = cut(a, b);
        if l > 0.0 {
            out.push(((t[i].0, t[j].0), l));
        }
    });
    out
}

/// Two generated sedans (CPU reference, raycast wheels, surface contact on), head on
/// (`scene` "head") or car A into the middle of B's side ("side"), B moved 0 .. runs - 1 mm;
/// `press`: A at full throttle throughout, B's handbrake on; `contact` off: the control.
/// Per run: the paint crossing (m) at spawn, at the worst sample (every 20 ms) and after
/// `secs`, the centres' distance then (m), and the
/// larger of the cars' own paint cutting itself (pairs that did not as built), worst and last;
/// then one car's paint inside the other ([`depth`], m): at the worst sample and at the end, and
/// the largest from 0.3 s after the hit (the 25 cm between the cars closed) on, and the seconds
/// past 1 cm before that.
#[allow(clippy::too_many_arguments)]
pub fn crash(scene: &str, kmh: f32, secs: f32, runs: usize, edge: bool, press: bool, contact: bool) -> Result<Vec<[f32; 13]>, String> {
    crash_measuring(scene, kmh, secs, runs, edge, press, contact, true, true)
}

/// [`crash`] with only some measures: a car's own paint crossing itself (`own`, the slowest by
/// far) and the length of one car's paint cutting the other's (`line`); those off read 0.
#[allow(clippy::too_many_arguments)]
pub fn crash_measuring(scene: &str, kmh: f32, secs: f32, runs: usize, edge: bool, press: bool, contact: bool, own_on: bool, line_on: bool) -> Result<Vec<[f32; 13]>, String> {
    Ok(crash_mode(scene, kmh, secs, runs, edge, press, contact, own_on, line_on, None)?.into_iter().map(|x| x.0).collect())
}

/// [`crash_measuring`] in a contact mode (`continuous`: Some(true) continuous, Some(false)
/// discrete, None as the environment's `CONTACT_MODE`), with each run's most edges of one car
/// through the other's collision triangles at any 20 ms check (checked in continuous mode or
/// with `DAT_CHECK`; 0 otherwise).
#[allow(clippy::too_many_arguments)]
pub fn crash_mode(scene: &str, kmh: f32, secs: f32, runs: usize, edge: bool, press: bool, contact: bool, own_on: bool, line_on: bool, continuous: Option<bool>) -> Result<Vec<([f32; 13], usize)>, String> {
    let built = rig::rig_vehicle("sedan", None, None)?;
    let files = rig::mod_files(&built);
    let txt: Vec<(String, String)> =
        files.iter().filter(|(p, _)| p.ends_with(".jbeam")).map(|(p, b)| (p.clone(), String::from_utf8_lossy(b).into_owned())).collect();
    let pc = files.iter().find(|(p, _)| p.ends_with(".pc")).map(|(_, b)| String::from_utf8_lossy(b).into_owned());
    let refs: Vec<(&str, &str)> = txt.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let d = engine_softbody::jbeam::load_jbeam(&refs, pc.as_deref())?;
    let flex = engine_softbody::jbeam::assembled_rows(&refs, pc.as_deref(), "flexbodies")?;
    let meshes: Vec<_> =
        built.input.glb.meshes.iter().map(|m| crate::flexskin::glb_mesh(&built.input.glb, m)).collect();
    let (probe, _) = ExplicitBeams::from_desc_jbeam_raycast(&d, Vec3::ZERO, Vec3::ZERO, settings())?;
    let (v, _) = crate::flexmesh::jbeam_visual_skinned(
        &flex, &[], &d, &probe, &meshes, &Default::default(), [0.5; 3], Some(&built.skin),
    )?;
    let mut turned: SoftVehicleDesc = d.clone();
    for n in &mut turned.nodes {
        n.position = [-n.position[0], n.position[1], -n.position[2]];
    }
    let mut sideways = d.clone();
    for n in &mut sideways.nodes {
        n.position = [n.position[2], n.position[1], -n.position[0]];
    }
    // (The visual binds to the nodes, so a turned car's is the same mesh.)
    let ext = |d: &SoftVehicleDesc, k: usize| d.nodes.iter().map(|n| n.position[k]).fold((f32::MAX, f32::MIN), |(a, b), x| (a.min(x), b.max(x)));
    let spawn = |d: &SoftVehicleDesc, at: Vec3, vel: Vec3| -> ExplicitBeams {
        let lift = 0.02 - ext(d, 1).0;
        let (mut s, _) = ExplicitBeams::from_desc_jbeam_raycast(d, at + Vec3::Y * lift, vel, settings()).expect("spawn");
        s.surface_contact = true;
        s.edge_contact = edge;
        // (THROTTLE_OFF=1, a measurement: no contact throttle, every triangle tested every step.)
        if std::env::var("THROTTLE_OFF").is_ok_and(|x| x == "1") {
            s.settings.triangle_throttle = false;
            s.fixed_contact = true;
        }
        // (SELF_OFF=1, the control of a car's own measures: one self group, so no self contact.)
        if std::env::var("SELF_OFF").is_ok_and(|x| x == "1") {
            s.self_groups = vec![1; s.nodes.len()];
        }
        s.finish();
        // (SELF_EXCL=0: the pairs excluded from self contact for starting in range take part;
        // PAINT_PARTS=1 counts them.)
        if std::env::var("PAINT_PARTS").is_ok_and(|x| x == "1") {
            let mut by: HashMap<(String, String), usize> = HashMap::new();
            let part = |n: u32| d.nodes.get(n as usize).map_or("?".into(), |n| n.name.trim_end_matches(|c: char| c.is_ascii_digit() || c == '_').to_string());
            for (t, e) in s.tri_excl.iter().enumerate() {
                for &h in e { *by.entry((part(h), part(s.triangles[t].nodes[0]))).or_default() += 1; }
            }
            let mut o: Vec<_> = by.into_iter().collect();
            o.sort_by(|a, b| b.1.cmp(&a.1));
            println!("excluded node-triangle pairs {}: {:?}", o.iter().map(|x| x.1).sum::<usize>(), &o[..o.len().min(16)]);
        }
        if std::env::var("SELF_EXCL").is_ok_and(|x| x == "0") {
            s.tri_excl.iter_mut().for_each(|e| e.clear());
        }
        s
    };
    let rest: Vec<Vec3> = v.positions.iter().map(|&p| Vec3::from(p)).collect();
    if std::env::var("PAINT_ON_SKIN").is_ok_and(|x| x == "1") {
        // The rig's parts, joined to the body where a part has no collision triangle of its own
        // (groups: the file's self-collision groups; the car's own table of node parts).
        let parts: HashMap<String, String> = built.rig.car.nodes.iter().map(|n| (n.name.clone(), n.part.name().to_string())).collect();
        let own: std::collections::HashSet<String> = (0..probe.triangles.len()).filter_map(|t| d.nodes.get(probe.triangles[t].nodes[0] as usize)).filter_map(|n| parts.get(&n.name).cloned()).collect();
        println!("parts with collision triangles: {}", own.len());
        let part = |n: u32| d.nodes.get(n as usize).and_then(|n| parts.get(&n.name)).filter(|p| own.contains(*p)).cloned().unwrap_or("?".into());
        let on = ON_SKIN.get_or_init(|| on_skin(&v, &probe, part));
        let n = v.material.iter().filter(|&&m| m == mat::PAINT).count();
        println!("paint carried by its collision panel: {} of {n} vertices", on.at.iter().filter(|x| x.is_some()).count());
    }
    let run = |k: usize| -> ([f32; 13], usize) {
        let (za, speed) = (ext(&d, 2).1, kmh / 3.6);
        // (25 cm apart: the paint reaches past the nodes.)
        let aa = spawn(&d, Vec3::new(0.0, 0.0, -0.125 - za), Vec3::Z * speed);
        let off = k as f32 * 1e-3;
        let bb = if scene == "side" {
            spawn(&sideways, Vec3::new(off - 0.5 * (ext(&sideways, 0).0 + ext(&sideways, 0).1), 0.0, 0.125 - ext(&sideways, 2).0), Vec3::ZERO)
        } else {
            spawn(&turned, Vec3::new(off, 0.0, 0.125 - ext(&turned, 2).0), -Vec3::Z * speed)
        };
        // ("pole": car A alone into a fixed 0.15 m pole at its front left corner.)
        // ("drive": car A alone, no pole: the control for its own paint.)
        let pole = scene == "pole" || scene == "drive";
        let mut w = if pole {
            let mut aa = aa;
            if scene == "pole" {
                aa.shapes.push(StaticShape::cylinder(Vec3::new(0.45 + off, 2.0, 0.3), 0.15, 2.0));
            }
            ExplicitBeamsWorld::new(vec![aa])
        } else {
            ExplicitBeamsWorld::new(vec![aa, bb])
        };
        w.contact = contact;
        if let Some(c) = continuous {
            w.continuous = c.then(|| engine_softbody::explicit_contact::Dat::new(0.08, 0.85));
        }
        if press {
            w.vehicles[0].set_drive_input(DriveInput { forward: 1.0, ..Default::default() });
            if let Some(b) = w.vehicles.get_mut(1) {
                b.set_drive_input(DriveInput { brake: 1.0, handbrake: 1.0, ..Default::default() });
            }
        }
        let now = |w: &ExplicitBeamsWorld| if pole || !line_on { 0.0 } else { crossing(&paint(&v, &rest, &w.vehicles[0]), &paint(&v, &rest, &w.vehicles[1])) };
        // (As built: the pairs of a car's own paint that cut then do not count.)
        // The collision skin's own triangles, at the nodes or at the contact offsets.
        let skin = |s: &ExplicitBeams, off: bool| -> Vec<(u32, Tri)> {
            let p: Vec<Vec3> = if off { engine_softbody::explicit_contact::contact_surface_positions(s) } else { s.nodes.iter().map(|n| n.pos).collect() };
            (0..s.triangles.len())
                .filter(|&t| !s.tri_dead.get(t).copied().unwrap_or(false) && !s.tri_anti_only(t))
                .map(|t| (t as u32, s.triangles[t].nodes.map(|i| p[i as usize])))
                .collect()
        };
        let skin_built: [std::collections::HashSet<(u32, u32)>; 2] = [false, true].map(|o| self_pairs(&skin(&w.vehicles[0], o)).into_iter().map(|x| x.0).collect());
        let built: std::collections::HashSet<(u32, u32)> = if !own_on { Default::default() } else { self_pairs(&paint_ids(&v, &rest, &w.vehicles[0])).into_iter().map(|x| x.0).collect() };
        let own = |w: &ExplicitBeamsWorld| -> f32 {
            w.vehicles.iter().map(|s| self_pairs(&paint_ids(&v, &rest, s)).iter().filter(|x| !built.contains(&x.0)).map(|x| x.1).sum::<f32>()).fold(0.0, f32::max)
        };
        let (mut own_worst, mut own_last) = (0.0f32, 0.0f32);
        let gap = {
            let n: Vec<Vec3> = w.vehicles[0].nodes.iter().map(|n| w.vehicles[0].origin + n.pos).collect();
            let p = deform(&v, &n).0;
            gaps(&w.vehicles[0], &v.indices.chunks_exact(3).map(|t| [0, 1, 2].map(|k| p[t[k] as usize])).collect::<Vec<_>>())
        };
        // (The depth's resolution: a car's own paint against its own contact surface, as built.)
        if k == 0 && std::env::var("PAINT_CALIB").is_ok_and(|x| x == "1") {
            let (pt, sf) = (paint(&v, &rest, &w.vehicles[0]), surface(&w.vehicles[0], &gap));
            let mut d: Vec<f32> = gap.iter().copied().filter(|g| *g != 0.0).collect();
            d.sort_by(f32::total_cmp);
            let q = |f: f32| d[((d.len() - 1) as f32 * f) as usize] * 1e3;
            println!("the contact surface outside the drawn mesh, per node (mm): p1 {:.0} p10 {:.0} p50 {:.0} p90 {:.0} p99 {:.0} max {:.0}; {} painted, {} surface triangles", q(0.01), q(0.1), q(0.5), q(0.9), q(0.99), q(1.0), pt.len(), sf.tris.len());
            // Painted points outside the car's own contact surface (as built), by the node they bind to.
            let mut by: HashMap<String, (usize, f32)> = HashMap::new();
            let np: Vec<Vec3> = w.vehicles[0].nodes.iter().map(|n| w.vehicles[0].origin + n.pos).collect();
            let pp = deform(&v, &np).0;
            for i in (0..pp.len()).filter(|&i| v.material[i] == mat::PAINT) {
                let x = pp[i];
                if winding(x, &sf.tris) > 0.5 { continue; }
                let dd = sf.tris.iter().map(|t| distance(x, t)).fold(f32::MAX, f32::min);
                if dd < 0.02 { continue; }
                let n = turned.nodes[v.bind_nodes[i][0] as usize].name.trim_end_matches(|c: char| c.is_ascii_digit() || c == '_').to_string();
                let e = by.entry(n.clone()).or_default(); e.0 += 1; e.1 = e.1.max(dd);
                if std::env::var("PAINT_OUT_PART").is_ok_and(|p| n == p) { println!("out {n} at {:.2} by {:.0} mm, bound to {:?} piece {}", x - w.vehicles[0].origin, dd * 1e3, v.bind_nodes[i].map(|k| turned.nodes[k as usize].name.clone()), v.pieces.iter().find(|q| (q.1..q.1 + q.2).contains(&(i as u32))).map_or("?", |q| q.0.as_str())); }
            }
            let mut o: Vec<_> = by.into_iter().collect(); o.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
            println!("painted vertices over 2 cm outside the contact surface, by part (count, max mm): {:?}", o.iter().take(20).map(|(n, (c, m))| format!("{n} {c} {:.0}", m * 1e3)).collect::<Vec<_>>());
        }
        // (PAINT_CARRY=1: painted area by the part it is bound to, and how much is carried by no
        // collision triangle's nodes or by nodes over 0.4 m apart.)
        if k == 0 && std::env::var("PAINT_CARRY").is_ok_and(|x| x == "1") {
            let a = &w.vehicles[0];
            let hull: std::collections::HashSet<u32> = (0..a.triangles.len()).filter(|&t| !a.tri_anti_only(t)).flat_map(|t| a.triangles[t].nodes).collect();
            let np: Vec<Vec3> = a.nodes.iter().map(|n| n.pos).collect();
            let mut by: std::collections::BTreeMap<String, [f32; 4]> = Default::default();
            for t in v.indices.chunks_exact(3).filter(|t| t.iter().all(|&i| v.material[i as usize] == mat::PAINT)) {
                let [x, y, z] = [0, 1, 2].map(|k| rest[t[k] as usize]);
                let area = (y - x).cross(z - x).length() * 0.5;
                let b = v.bind_nodes[t[0] as usize];
                let span = (0..3).map(|e| np[b[e] as usize].distance(np[b[(e + 1) % 3] as usize])).fold(0.0, f32::max);
                let off = b.iter().filter(|n| !hull.contains(n)).count();
                let name: String = d.nodes[b[0] as usize].name.split('_').take(2).collect::<Vec<_>>().join("_").trim_end_matches(|c: char| c.is_ascii_digit()).into();
                for key in [name, "ALL".into()] { let e = by.entry(key).or_default(); e[0] += area; if off > 0 { e[1] += area; } if off == 3 { e[2] += area; } if span > 0.4 { e[3] += area; } }
            }
            let mut bare: std::collections::BTreeMap<u32, f32> = Default::default();
            for t in v.indices.chunks_exact(3).filter(|t| t.iter().all(|&i| v.material[i as usize] == mat::PAINT)) {
                let [x, y, z] = [0, 1, 2].map(|k| rest[t[k] as usize]);
                for n in v.bind_nodes[t[0] as usize] { if !hull.contains(&n) { *bare.entry(n).or_default() += (y - x).cross(z - x).length() * 0.5; } }
            }
            let out = surface(a, &gap).tris;
            let mut l: Vec<_> = bare.into_iter().collect();
            l.sort_by(|a, b| b.1.total_cmp(&a.1));
            for (n, area) in l {
                let x = a.origin + np[n as usize];
                let (dd, t, w) = out.iter().enumerate().map(|(i, q)| { let w = closest(x, q); (distance(x, q), i, w) }).min_by(|a, b| a.0.total_cmp(&b.0)).unwrap();
                let ids: Vec<usize> = live_outer(a).collect();
                println!("bare {:14} at {:.2} carries {area:.2} m2, {:.0} mm from the outer surface (winding {:.2}), triangle {:?} weights {:.2?}", d.nodes[n as usize].name, d.nodes[n as usize].position.map(|x| (x * 100.0).round() / 100.0)[0], dd * 1e3, winding(x, &out), a.triangles[ids[t]].nodes.map(|i| d.nodes[i as usize].name.clone()), w);
            }
            let mut o: Vec<_> = by.into_iter().collect();
            o.sort_by(|a, b| b.1[0].total_cmp(&a.1[0]));
            for (n, e) in o.into_iter().take(26) { println!("carry {n:12} painted {:6.2} m2; some bound node without a triangle {:3.0} %, none with one {:3.0} %, nodes over 0.4 m apart {:3.0} %", e[0], 100.0 * e[1] / e[0], 100.0 * e[2] / e[0], 100.0 * e[3] / e[0]); }
        }
        let names: Vec<String> = d.nodes.iter().map(|n| n.name.clone()).collect();
        let solid_base: Vec<Vec<f32>> = w.vehicles.iter().map(|x| solid_depths(&v, &rest, x, &names)).collect();
        let start = now(&w);
        let steps = (secs * 2000.0) as usize;
        let (mut worst, mut last) = (0.0f32, 0.0f32);
        // Jitter: each car's nodes' rms speed about its centre of mass (m/s), the largest over the last 0.25 s.
        // (About its rigid motion: the centre's velocity and the angular velocity of its angular
        // momentum, so a car spinning or pitching whole after a hit does not count as shaking.)
        let shake = |w: &ExplicitBeamsWorld| w.vehicles.iter().map(|x| {
            let m: f64 = x.nodes.iter().map(|n| n.mass as f64).sum();
            let c = x.nodes.iter().map(|n| n.pos.as_dvec3() * n.mass as f64).sum::<glam::DVec3>() / m;
            let vc = x.nodes.iter().map(|n| n.vel.as_dvec3() * n.mass as f64).sum::<glam::DVec3>() / m;
            let (mut l, mut i) = (glam::DVec3::ZERO, glam::DMat3::ZERO);
            for n in &x.nodes {
                let (r, mm) = (n.pos.as_dvec3() - c, n.mass as f64);
                l += r.cross(n.vel.as_dvec3() - vc) * mm;
                i += (glam::DMat3::IDENTITY * r.length_squared() - glam::DMat3::from_cols(r * r.x, r * r.y, r * r.z)) * mm;
            }
            let om = if i.determinant().abs() > 1e-12 { i.inverse() * l } else { glam::DVec3::ZERO };
            (x.nodes.iter().map(|n| { let r = n.pos.as_dvec3() - c; n.mass as f64 * (n.vel.as_dvec3() - vc - om.cross(r)).length_squared() }).sum::<f64>() / m).sqrt() as f32
        }).fold(0.0f32, f32::max);
        let mut jitter = 0.0f32;
        let (mut deep, mut deep_last, mut lasting, mut brief) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let settled = 0.3 + 0.25 / (speed * if scene == "side" { 1.0 } else { 2.0 });
        // (DAT_CHECK=N, run 0: every N steps the edges of one car crossing the other's
        // collision triangles, and at the end the continuous mode's counters.)
        let check: usize = std::env::var("DAT_CHECK").ok().and_then(|x| x.parse().ok()).unwrap_or(if w.continuous.is_some() { 40 } else { 0 });
        let (mut crossed_max, mut crossed_steps) = (0usize, 0usize);
        for s in 1..=steps {
            w.step();
            if check > 0 && !pole && s % check == 0 {
                let c = w.mesh_crossings();
                crossed_max = crossed_max.max(c);
                crossed_steps += (c > 0) as usize;
                if k == 0 && s == steps - steps % check && std::env::var("DAT_CHECK").is_ok() { println!("DAT {scene} {kmh}: crossings max {crossed_max}, checks with any {crossed_steps} of {}; {:?}", steps / check, w.continuous.as_ref().map(|d| d.total));
                    if let Some(who) = w.continuous.as_ref().and_then(|d| d.who.as_ref()) {
                        let mut l: Vec<_> = who.iter().collect();
                        l.sort_by_key(|x| std::cmp::Reverse(x.1[0] + x.1[1]));
                        println!("DAT_WHO (car node: over, frozen) {:?}", l.iter().take(24).map(|((c, i), n)| format!("{c} {} {}/{}", d.nodes[*i as usize].name, n[0], n[1])).collect::<Vec<_>>());
                    }
                }
            }
            if s >= steps.saturating_sub(500) && s % 20 == 0 {
                jitter = jitter.max(shake(&w));
            }
            // (JITTER_FREQ=car:node: that node's velocity about its car's centre along its first value,
            // its reversals over the last 0.25 s: the frequency of the shake.)
            if k == 0 && s >= steps.saturating_sub(500) {
                if let Some((c, name)) = std::env::var("JITTER_FREQ").ok().and_then(|x| x.split_once(':').map(|(a, b)| (a.parse::<usize>().unwrap_or(0), b.to_string()))) {
                    let x = &w.vehicles[c];
                    let i = d.nodes.iter().position(|n| n.name == name).unwrap_or(0);
                    if s == steps.saturating_sub(500) || s == steps { println!("JITTER_FREQ node {name} index {i} node pos {:.3?}", x.origin + x.nodes[i].pos); for y in &w.vehicles { let m: f32 = y.nodes.iter().map(|n| n.mass).sum(); println!("  car com {:.3?} vel {:.3?}", y.nodes.iter().map(|n| (y.origin + n.pos) * n.mass).sum::<Vec3>() / m, y.nodes.iter().map(|n| n.vel * n.mass).sum::<Vec3>() / m); } }
                    let m: f32 = x.nodes.iter().map(|n| n.mass).sum();
                    let vc = x.nodes.iter().map(|n| n.vel * n.mass).sum::<Vec3>() / m;
                    let v = x.nodes[i].vel - vc;
                    if std::env::var("JITTER_TRACE").is_ok() && s > steps - 80 { let dir = FREQ.with(|f| f.borrow().0); println!("step {s} v {:+.2} edge hits {} node hits {}", v.dot(dir), w.edge_stats[5], w.total.hits); }
                    FREQ.with(|f| { let mut f = f.borrow_mut(); if f.0 == Vec3::ZERO { f.0 = v.normalize_or_zero(); } let sgn = v.dot(f.0) > 0.0; if s > steps - 500 && sgn != f.1 { f.2 += 1; } f.1 = sgn; f.3 = f.3.max(v.length()); });
                    if s == steps { FREQ.with(|f| { let f = f.borrow(); println!("{name}: {} reversals in 0.25 s ({:.0} Hz), peak {:.2} m/s", f.2, f.2 as f32 / 2.0 / 0.25, f.3); }); }
                }
            }
            if k == 0 && s == steps && std::env::var("JITTER_NODES").is_ok() {
                for (c, x) in w.vehicles.iter().enumerate() {
                    let m: f32 = x.nodes.iter().map(|n| n.mass).sum();
                    let vc = x.nodes.iter().map(|n| n.vel * n.mass).sum::<Vec3>() / m;
                    let mut l: Vec<(f32, usize)> = x.nodes.iter().enumerate().map(|(i, n)| ((n.vel - vc).length(), i)).collect();
                    l.sort_by(|a, b| b.0.total_cmp(&a.0));
                    println!("car {c} fastest nodes about its centre: {:?}", l.iter().take(12).map(|&(v, i)| format!("{} {:.1}", d.nodes[i].name, v)).collect::<Vec<_>>());
                }
            }
            if s % 40 == 0 || s == steps {
                last = now(&w);
                worst = worst.max(last);
                if !pole {
                    let [a, b] = [0, 1].map(|k| &w.vehicles[k]);
                    deep_last = depth(&paint(&v, &rest, a), &surface(b, &gap)).max(depth(&paint(&v, &rest, b), &surface(a, &gap)));
                    deep = deep.max(deep_last);
                    // (PAINT_WHERE=1, run 0: the deepest paint of each car in the other, and the parts.)
                    let at_mm = std::env::var("PAINT_WHERE_MM").ok().and_then(|x| x.parse::<f32>().ok()).unwrap_or(45.0) * 1e-3;
                    if deep_last > at_mm && s as f32 / 2000.0 >= settled && std::env::var("PAINT_WHERE").is_ok_and(|x| x == "1") {
                        for (n, x, y) in [("A in B", a, b), ("B in A", b, a)] {
                            let (dd, at) = deepest(&paint(&v, &rest, x), &surface(y, &gap));
                            if dd < at_mm { continue; }
                            let near = |s: &ExplicitBeams| { let i = (0..s.nodes.len()).min_by(|&i, &j| (s.origin + s.nodes[i].pos).distance(at).total_cmp(&(s.origin + s.nodes[j].pos).distance(at))).unwrap(); d.nodes[i].name.clone() };
                            let cs = |s: &ExplicitBeams, at: Vec3| { let sf = surface(s, &gap); let ids: Vec<usize> = live_outer(s).collect();
                                let m = (0..ids.len()).min_by(|&i, &j| { let h = |t: &Tri| { let w = closest(at, t); at.distance(t[0] * w[0] + t[1] * w[1] + t[2] * w[2]) }; h(&sf.tris[i]).total_cmp(&h(&sf.tris[j])) }).unwrap();
                                (s.triangles[ids[m]].nodes.map(|i| d.nodes[i as usize].name.clone()), sf.gap[m], sf.tris[m], (sf.tris[m][1] - sf.tris[m][0]).cross(sf.tris[m][2] - sf.tris[m][0]).normalize()) };
                            { let np: Vec<Vec3> = x.nodes.iter().map(|n| x.origin + n.pos).collect(); let pp = deform(&v, &np).0;
                              if let Some(i) = (0..pp.len()).filter(|&i| v.material[i] == mat::PAINT).min_by(|&i, &j| pp[i].distance(at).total_cmp(&pp[j].distance(at))) {
                                let names: Vec<_> = v.bind_nodes[i].iter().map(|&n| { let n = n as usize; format!("{} at {:.2} (contact {}, triangles {})", d.nodes[n].name, np[n], x.contact_points.contains(&(n as u32)), live_outer(x).filter(|&t| x.triangles[t].nodes.contains(&(n as u32))).count()) }).collect();
                                println!("    paint vertex, bound to {names:?}");
                              } }
                            println!("    its surface there {:?}", cs(y, at));
                            println!("    own surface there {:?}", cs(x, at));
                            for (i, n) in x.nodes.iter().enumerate() { let q = x.origin + n.pos; let dq = deepest(&[[q; 3]], &surface(y, &gap)).0; if dq > 0.0 { println!("    node {} inside {:.0} mm, contact node {}", d.nodes[i].name, dq * 1e3, x.contact_points.contains(&(i as u32))); } }
                            println!("  t {:.2} {n}: {:.0} mm at {at:.2}, nearest nodes {} / {}, winding {:.2}", s as f32 / 2000.0, dd * 1e3, near(x), near(y), winding(at, &surface(y, &gap).tris));
                        }
                    }
                    let t = s as f32 / 2000.0;
                    if t >= settled {
                        lasting = lasting.max(deep_last);
                    } else if deep_last > 0.01 {
                        brief += 0.02;
                    }
                }
                if s % 80 == 0 || s == steps {
                    own_last = if own_on { own(&w) } else { 0.0 };
                    own_worst = own_worst.max(own_last);
                }
            }
        }
        // (PAINT_PARTS=1, run 0: the parts whose paint cuts, and whether the car's own collision
        // skin crosses there too.)
        if k == 0 && std::env::var("PAINT_PARTS").is_ok_and(|x| x == "1") {
            let ids: Vec<[u32; 3]> = v.indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).filter(|t| t.iter().all(|&i| v.material[i as usize] == mat::PAINT)).collect();
            let part = |t: u32| {
                let n = &d.nodes[v.bind_nodes[ids[t as usize][0] as usize][0] as usize];
                n.name.trim_end_matches(|c: char| c.is_ascii_digit() || c == '_').to_string()
            };
            for s in &w.vehicles {
                let of = |o: bool| self_pairs(&skin(s, o)).into_iter().filter(|x| !skin_built[o as usize].contains(&x.0)).collect::<Vec<_>>();
                let name = |t: u32| { let n = &d.nodes[s.triangles[t as usize].nodes[0] as usize].name; n.trim_end_matches(|c: char| c.is_ascii_digit() || c == '_').to_string() };
                // (The panels you see: the parts whose paint shows above.)
                let seen = |t: u32| ["hd", "fe", "door", "bf", "bb", "ebody", "tk"].iter().any(|p| name(t).starts_with(p));
                let sum = |v: &[((u32, u32), f32)], only: bool| v.iter().filter(|x| !only || (seen(x.0 .0) && seen(x.0 .1))).map(|x| x.1).sum::<f32>();
                let (raw, off) = (of(false), of(true));
                println!("{scene} {kmh}: collision skin through itself {:.2} m (visible panels {:.2}), at the offsets {:.2} ({:.2})", sum(&raw, false), sum(&raw, true), sum(&off, false), sum(&off, true));
                // Each crossing painted triangle against its own part's collision skin: over it (the
                // closest point inside a triangle: how far) or past its edge.
                // (Its part's dead triangles counted too, PAINT_DEAD=1.)
                let with_dead = std::env::var("PAINT_DEAD").is_ok_and(|x| x == "1");
                let sk: Vec<(u32, Tri)> = (0..s.triangles.len())
                    .filter(|&t| (with_dead || !s.tri_dead.get(t).copied().unwrap_or(false)) && !s.tri_anti_only(t))
                    .map(|t| (t as u32, s.triangles[t].nodes.map(|i| s.nodes[i as usize].pos)))
                    .collect();
                let (mut total, mut dead): (HashMap<String, usize>, HashMap<String, usize>) = Default::default();
                for t in 0..s.triangles.len() {
                    *total.entry(name(t as u32)).or_default() += 1;
                    if s.tri_dead.get(t).copied().unwrap_or(false) { *dead.entry(name(t as u32)).or_default() += 1; }
                }
                let mut dl: Vec<_> = dead.iter().map(|(p, n)| (*n, total[p], p.clone())).collect();
                dl.sort(); dl.reverse();
                println!("{scene} {kmh}: dead triangles {} of {}: {:?}", dl.iter().map(|x| x.0).sum::<usize>(), s.triangles.len(), &dl[..dl.len().min(9)]);
                let (mut over, mut past): (Vec<f32>, Vec<f32>) = Default::default();
                let pt = paint_ids(&v, &rest, s);
                let at: HashMap<u32, Tri> = pt.iter().copied().collect();
                for ((i, j), _) in self_pairs(&pt) {
                    if built.contains(&(i, j)) { continue; }
                    for k in [i, j] {
                        let c = (at[&k][0] + at[&k][1] + at[&k][2]) / 3.0 - s.origin;
                        let mine = part(k);
                        let mut best = (f32::MAX, false);
                        for (t, q) in sk.iter().filter(|(t, _)| name(*t) == mine) {
                            let _ = t;
                            let n = (q[1] - q[0]).cross(q[2] - q[0]);
                            let x = c - n * ((c - q[0]).dot(n) / n.length_squared().max(1e-12));
                            if (0..3).all(|e| (q[(e + 1) % 3] - q[e]).cross(x - q[e]).dot(n) >= 0.0) {
                                if c.distance(x) < best.0 { best = (c.distance(x), true); }
                            } else {
                                for e in 0..3 {
                                    let (u, w) = (q[e], q[(e + 1) % 3] - q[e]);
                                    let y = u + w * ((c - u).dot(w) / w.length_squared().max(1e-12)).clamp(0.0, 1.0);
                                    if c.distance(y) < best.0 { best = (c.distance(y), false); }
                                }
                            }
                        }
                        if best.0 < f32::MAX { if best.1 { over.push(best.0) } else { past.push(best.0) } }
                    }
                }
                let q = |v: &mut Vec<f32>| { v.sort_by(f32::total_cmp); if v.is_empty() { "none".into() } else { format!("{} (p50 {:.0}, p90 {:.0} mm)", v.len(), v[v.len() / 2] * 1e3, v[v.len() * 9 / 10] * 1e3) } };
                println!("{scene} {kmh}: crossing paint over its part's skin {}, past its edge {}", q(&mut over), q(&mut past));
                let mut by: HashMap<(String, String), f32> = HashMap::new();
                for ((i, j), l) in self_pairs(&paint_ids(&v, &rest, s)) {
                    if !built.contains(&(i, j)) {
                        let (a, b) = (part(i), part(j));
                        *by.entry(if a <= b { (a, b) } else { (b, a) }).or_default() += l;
                    }
                }
                let mut o: Vec<_> = by.into_iter().collect();
                o.sort_by(|a, b| b.1.total_cmp(&a.1));
                let same: f32 = o.iter().filter(|x| x.0 .0 == x.0 .1).map(|x| x.1).sum();
                println!("{scene} {kmh}: own paint {:.2} m, of it a part through itself {same:.2}: {:?}", o.iter().map(|x| x.1).sum::<f32>(), &o[..o.len().min(10)]);
            }
        }
        // (PAINT_BIND=1, run 0: is each painted vertex bound to a collision triangle, and how far is
        // its paint from where the collision triangle it lies on as built carries it?)
        if k == 0 && std::env::var("PAINT_BIND").is_ok_and(|x| x == "1") {
            for s in &w.vehicles {
                let key = |mut t: [u32; 3]| { t.sort_unstable(); t };
                let tris: std::collections::HashSet<[u32; 3]> = (0..s.triangles.len()).filter(|&t| !s.tri_anti_only(t)).map(|t| key(s.triangles[t].nodes)).collect();
                let used: std::collections::BTreeSet<usize> = v.indices.iter().map(|&i| i as usize).filter(|&i| v.material[i] == mat::PAINT).collect();
                let on = used.iter().filter(|&&i| tris.contains(&key(v.bind_nodes[i]))).count();
                println!("{scene} {kmh}: painted vertices {} bound to a collision triangle {} ({:.0} %)", used.len(), on, 100.0 * on as f32 / used.len() as f32);
                let r0: Vec<Vec3> = probe.nodes.iter().map(|n| n.pos).collect();
                let r1: Vec<Vec3> = s.nodes.iter().map(|n| n.pos).collect();
                let p1 = deform(&v, &r1).0;
                let p0 = deform(&v, &r0).0;
                let (mut dev_on, mut dev_off): (Vec<f32>, Vec<f32>) = Default::default();
                for &i in used.iter().step_by(7) {
                    let c = p0[i];
                    let mut best = (f32::MAX, 0usize, [0.0f32; 3]);
                    for t in 0..s.triangles.len() {
                        if s.tri_anti_only(t) { continue; }
                        let q = s.triangles[t].nodes.map(|n| r0[n as usize]);
                        let (e1, e2) = (q[1] - q[0], q[2] - q[0]);
                        let n = e1.cross(e2).normalize_or_zero();
                        let h = (c - q[0]).dot(n);
                        let x = c - q[0] - n * h;
                        let (d11, d12, d22) = (e1.dot(e1), e1.dot(e2), e2.dot(e2));
                        let det = d11 * d22 - d12 * d12;
                        if det <= 1e-12 { continue; }
                        let (b1, b2) = ((d22 * x.dot(e1) - d12 * x.dot(e2)) / det, (d11 * x.dot(e2) - d12 * x.dot(e1)) / det);
                        if b1 < -0.02 || b2 < -0.02 || b1 + b2 > 1.02 { continue; }
                        if h.abs() < best.0 { best = (h.abs(), t, [b1, b2, h]); }
                    }
                    if best.0 > 0.15 { continue; }
                    let q = s.triangles[best.1].nodes.map(|n| r1[n as usize]);
                    let (e1, e2) = (q[1] - q[0], q[2] - q[0]);
                    let y = q[0] + e1 * best.2[0] + e2 * best.2[1] + e1.cross(e2).normalize_or_zero() * best.2[2];
                    if tris.contains(&key(v.bind_nodes[i])) { dev_on.push(y.distance(p1[i])) } else { dev_off.push(y.distance(p1[i])) }
                }
                let q = |v: &mut Vec<f32>| { v.sort_by(f32::total_cmp); if v.is_empty() { "none".into() } else { format!("{}: p50 {:.0} p90 {:.0} p99 {:.0} mm", v.len(), v[v.len() / 2] * 1e3, v[v.len() * 9 / 10] * 1e3, v[v.len() * 99 / 100] * 1e3) } };
                println!("{scene} {kmh}: paint from where its collision panel carries it: bound to one {}; not bound to one {}", q(&mut dev_on), q(&mut dev_off));
            }
        }
        // (PAINT_DEPTH=1, run 0: how deep a car's own crossings go. Crossing triangles are joined
        // into patches (pairs within 5 cm); a patch's depth is the farthest any painted vertex
        // within 25 cm of it, of the part that pokes, lies behind the other part's crossing plane.)
        if k == 0 && std::env::var("PAINT_DEPTH").is_ok_and(|x| x == "1") {
            let ids: Vec<u32> = v.indices.chunks_exact(3).filter(|t| t.iter().all(|&i| v.material[i as usize] == mat::PAINT)).map(|t| t[0]).collect();
            let part = |t: u32| { let n = &d.nodes[v.bind_nodes[ids[t as usize] as usize][0] as usize].name; n.trim_end_matches(|c: char| c.is_ascii_digit() || c == '_').to_string() };
            for s in &w.vehicles {
                let pt = paint_ids(&v, &rest, s);
                let at: HashMap<u32, Tri> = pt.iter().copied().collect();
                // Per pair of parts the crossing points with the other's normal there.
                let mut by: HashMap<(String, String), Vec<(Vec3, f32, f32)>> = HashMap::new();
                for ((i, j), l) in self_pairs(&pt) {
                    if built.contains(&(i, j)) { continue; }
                    let (a, b) = (part(i), part(j));
                    let c = (at[&i][0] + at[&i][1] + at[&i][2]) / 3.0;
                    // How far each triangle reaches through the other's plane.
                    let through = |x: &Tri, y: &Tri| {
                        let n = (y[1] - y[0]).cross(y[2] - y[0]).normalize_or_zero();
                        let e = x.map(|p| (p - y[0]).dot(n));
                        e[0].max(e[1]).max(e[2]).min(-(e[0].min(e[1]).min(e[2])))
                    };
                    let t = through(&at[&i], &at[&j]).min(through(&at[&j], &at[&i]));
                    by.entry(if a <= b { (a, b) } else { (b, a) }).or_default().push((c, l, t));
                }
                let mut o: Vec<_> = by.into_iter().map(|(k, v)| {
                    let lo = v.iter().fold(Vec3::splat(f32::MAX), |m, x| m.min(x.0));
                    let hi = v.iter().fold(Vec3::splat(f32::MIN), |m, x| m.max(x.0));
                    (v.iter().map(|x| x.1).sum::<f32>(), k, v.len(), (hi - lo).length(), v.iter().map(|x| x.2).fold(0.0f32, f32::max))
                }).collect();
                o.sort_by(|a, b| b.0.total_cmp(&a.0));
                println!("{scene} {kmh}: own crossings by parts (line m, pairs, extent m, deepest one triangle reaches mm):");
                for (l, k, n, e, t) in o.iter().take(9) { println!("    {:>11} x {:<11} {l:5.2} {n:5} {e:5.2} {:5.1}", k.0, k.1, t * 1e3); }
                // The depth proper: per painted vertex of a crossing part, its distance inside the other
                // part's own closed neighbourhood: signed against the other part's nearest painted triangle.
                for (_, kk, _, _, _) in o.iter().take(5) {
                    let of = |p: &str| -> Vec<Tri> { pt.iter().filter(|x| part(x.0) == p).map(|x| x.1).collect() };
                    let (x, y) = (of(&kk.0), of(&kk.1));
                    let deep = |x: &[Tri], y: &[Tri]| -> (usize, f32) {
                        let mut r: Vec<f32> = Vec::new();
                        for t in x.iter().step_by(3) {
                            let c = t[0];
                            let mut best = (f32::MAX, 0.0f32);
                            for q in y {
                                let n = (q[1] - q[0]).cross(q[2] - q[0]).normalize_or_zero();
                                let h = (c - q[0]).dot(n);
                                if h.abs() > 0.3 { continue; }
                                let z = c - n * h;
                                if (0..3).all(|e| (q[(e + 1) % 3] - q[e]).cross(z - q[e]).dot(n) >= 0.0) && h.abs() < best.0 { best = (h.abs(), h); }
                            }
                            if best.0 < f32::MAX && best.1 < 0.0 { r.push(best.0); }
                        }
                        r.sort_by(f32::total_cmp);
                        (r.len(), r.get(r.len() * 9 / 10).copied().unwrap_or(0.0))
                    };
                    let (a, b) = (deep(&x, &y), deep(&y, &x));
                    println!("    behind: {} of {} under {} p90 {:.0} mm; {} of {} under {} p90 {:.0} mm", a.0, x.len() / 3, kk.1, a.1 * 1e3, b.0, y.len() / 3, kk.0, b.1 * 1e3);
                }
            }
        }
        // (PAINT_FIT=1, run 0, at the end; `drive 0 0.02`: as built.) Per part the painted area and
        // how much of it lies past the edge of the whole car's collision skin, by how far.
        if k == 0 && std::env::var("PAINT_FIT").is_ok_and(|x| x == "1") {
            let s = &w.vehicles[0];
            let ids: Vec<u32> = v.indices.chunks_exact(3).filter(|t| t.iter().all(|&i| v.material[i as usize] == mat::PAINT)).map(|t| t[0]).collect();
            let sk = skin(s, false);
            let mut by: HashMap<String, (f32, f32, Vec<f32>)> = HashMap::new();
            for (id, t) in paint_ids(&v, &rest, s) {
                let name = &d.nodes[v.bind_nodes[ids[id as usize] as usize][0] as usize].name;
                let part = name.split('_').next().unwrap_or(name).trim_end_matches(|c: char| c.is_ascii_digit()).to_string();
                let c = (t[0] + t[1] + t[2]) / 3.0 - s.origin;
                let area = 0.5 * (t[1] - t[0]).cross(t[2] - t[0]).length();
                // The nearest collision triangle: over it, or past its edge by the in-plane distance.
                let mut best = (f32::MAX, 0.0f32);
                for (_, q) in &sk {
                    let n = (q[1] - q[0]).cross(q[2] - q[0]).normalize_or_zero();
                    let h = (c - q[0]).dot(n);
                    let x = c - n * h;
                    let inside = (0..3).all(|e| (q[(e + 1) % 3] - q[e]).cross(x - q[e]).dot(n) >= 0.0);
                    let lat = if inside { 0.0 } else {
                        (0..3).map(|e| { let (u, w) = (q[e], q[(e + 1) % 3] - q[e]); x.distance(u + w * ((x - u).dot(w) / w.length_squared().max(1e-12)).clamp(0.0, 1.0)) }).fold(f32::MAX, f32::min)
                    };
                    let dist = h.hypot(lat);
                    if dist < best.0 { best = (dist, lat); }
                }
                let e = by.entry(part).or_default();
                e.0 += area;
                if best.1 > 0.02 { e.1 += area; }
                e.2.push(best.1);
            }
            let mut o: Vec<_> = by.into_iter().collect();
            o.sort_by(|a, b| b.1 .1.total_cmp(&a.1 .1));
            for (p, (a, past, mut l)) in o.into_iter().take(14) {
                l.sort_by(f32::total_cmp);
                println!("fit {p:10} painted {a:6.2} m2, past an edge by > 2 cm {past:5.2} m2 ({:2.0} %), p90 {:4.0} max {:4.0} mm", 100.0 * past / a.max(1e-6), l[l.len() * 9 / 10] * 1e3, l[l.len() - 1] * 1e3);
            }
        }
        let centre = |s: &ExplicitBeams| s.nodes.iter().map(|n| (s.origin + n.pos) * n.mass).sum::<Vec3>() / s.nodes.iter().map(|n| n.mass).sum::<f32>();
        // A car's own paint inside its interior solids at the end (new since spawn): the deepest, and how many vertices past 1 cm.
        let (mut solid, mut solid_n) = (0.0f32, 0usize);
        for (x, base) in w.vehicles.iter().zip(&solid_base) {
            for (dd, b) in solid_depths(&v, &rest, x, &names).into_iter().zip(base) {
                if dd > b + 0.01 { solid = solid.max(dd); solid_n += 1; }
            }
        }
        ([start, worst, last, if pole { 0.0 } else { centre(&w.vehicles[0]).distance(centre(&w.vehicles[1])) }, own_worst, own_last, deep, deep_last, lasting, brief, solid, solid_n as f32, jitter], crossed_max)
    };
    let res: Vec<([f32; 13], usize)> = std::thread::scope(|t| {
        let run = &run;
        let js: Vec<_> = (0..runs).map(|k| t.spawn(move || run(k))).collect();
        js.into_iter().map(|j| j.join().expect("run")).collect()
    });
    Ok(res)
}
