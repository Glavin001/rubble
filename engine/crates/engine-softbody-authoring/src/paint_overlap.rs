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
/// larger of the cars' own paint cutting itself (pairs that did not as built), worst and last.
#[allow(clippy::too_many_arguments)]
pub fn crash(scene: &str, kmh: f32, secs: f32, runs: usize, edge: bool, press: bool, contact: bool) -> Result<Vec<[f32; 6]>, String> {
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
    let run = |k: usize| -> [f32; 6] {
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
        if press {
            w.vehicles[0].set_drive_input(DriveInput { forward: 1.0, ..Default::default() });
            if let Some(b) = w.vehicles.get_mut(1) {
                b.set_drive_input(DriveInput { brake: 1.0, handbrake: 1.0, ..Default::default() });
            }
        }
        let now = |w: &ExplicitBeamsWorld| if pole { 0.0 } else { crossing(&paint(&v, &rest, &w.vehicles[0]), &paint(&v, &rest, &w.vehicles[1])) };
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
        let built: std::collections::HashSet<(u32, u32)> = self_pairs(&paint_ids(&v, &rest, &w.vehicles[0])).into_iter().map(|x| x.0).collect();
        let own = |w: &ExplicitBeamsWorld| -> f32 {
            w.vehicles.iter().map(|s| self_pairs(&paint_ids(&v, &rest, s)).iter().filter(|x| !built.contains(&x.0)).map(|x| x.1).sum::<f32>()).fold(0.0, f32::max)
        };
        let (mut own_worst, mut own_last) = (0.0f32, 0.0f32);
        let start = now(&w);
        let steps = (secs * 2000.0) as usize;
        let (mut worst, mut last) = (0.0f32, 0.0f32);
        for s in 1..=steps {
            w.step();
            if s % 40 == 0 || s == steps {
                last = now(&w);
                worst = worst.max(last);
                if s % 80 == 0 || s == steps {
                    own_last = own(&w);
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
        [start, worst, last, if pole { 0.0 } else { centre(&w.vehicles[0]).distance(centre(&w.vehicles[1])) }, own_worst, own_last]
    };
    let res: Vec<[f32; 6]> = std::thread::scope(|t| {
        let run = &run;
        let js: Vec<_> = (0..runs).map(|k| t.spawn(move || run(k))).collect();
        js.into_iter().map(|j| j.join().expect("run")).collect()
    });
    Ok(res)
}
