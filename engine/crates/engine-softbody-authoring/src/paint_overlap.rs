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
    let p = deform(v, &n).0;
    v.indices
        .chunks_exact(3)
        .filter(|t| t.iter().all(|&i| v.material[i as usize] == mat::PAINT))
        .map(|t| [t[0] as usize, t[1] as usize, t[2] as usize])
        .enumerate()
        .filter(|(_, t)| (0..3).all(|k| {
            let (i, j) = (t[k], t[(k + 1) % 3]);
            p[i].is_finite() && p[i].distance(p[j]) <= 3.0 * rest[i].distance(rest[j]) + 0.01
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
        s
    };
    let rest: Vec<Vec3> = v.positions.iter().map(|&p| Vec3::from(p)).collect();
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
