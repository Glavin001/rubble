//! Per-node contact offsets (`ExplicitBeams::contact_offset`): how far each skin node's
//! contact surface sits out from the node, so the collision skin is the car's paint and not
//! the straight triangles between its nodes (the paint bulges up to 0.2 m past them on the
//! generated sedan's bumpers and body sides: the lint's `skin_coverage`). Two cars then meet
//! paint to paint, not with one car's paint inside the other's.
//!
//! For each outer render vertex (the coverage rule's: paint and glass facing out, not the
//! wheels, interior or mirrors) its nearest collision triangle on the outer skin and its
//! height above that triangle's plane (outward); per triangle the most; per node the most of
//! its outer triangles', at most `cap`. Linear across a triangle, the offset surface is then
//! over every point of the paint the triangle covers.

use crate::authoring::lint::tri_dist_pub as tri_dist;
use crate::authoring::skin::{mat, ShowcaseVisual};
use engine_softbody::desc::SoftVehicleDesc;
use glam::Vec3;
use std::collections::HashMap;

/// The offsets (m) per node of `d` from its visual `v` (its rest pose), at most `cap`.
pub fn contact_offsets(d: &SoftVehicleDesc, v: &ShowcaseVisual, cap: f32) -> Vec<f32> {
    let nr: Vec<Vec3> = v.node_rest.iter().map(|p| Vec3::from_array(*p)).collect();
    let tris: Vec<(usize, [u32; 3])> = d
        .triangles
        .iter()
        .enumerate()
        .filter(|(_, t)| t.collision && t.aero[0] > 0.0 && t.nodes.iter().all(|&i| (i as usize) < nr.len()))
        .map(|(k, t)| (k, t.nodes))
        .collect();
    let cell = 0.1f32;
    let key = |p: Vec3| ((p.x / cell).floor() as i32, (p.y / cell).floor() as i32, (p.z / cell).floor() as i32);
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (k, (_, t)) in tris.iter().enumerate() {
        let [a, b, c] = t.map(|i| nr[i as usize]);
        let (lo, hi) = (key(a.min(b).min(c)), key(a.max(b).max(c)));
        for x in lo.0..=hi.0 {
            for y in lo.1..=hi.1 {
                for z in lo.2..=hi.2 {
                    grid.entry((x, y, z)).or_default().push(k);
                }
            }
        }
    }
    let (lo, hi) = nr.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(a, b), p| (a.min(*p), b.max(*p)));
    let c = 0.5 * (lo + hi);
    let half = ((hi - lo) * 0.5).max(Vec3::splat(1e-3));
    let outer = |p: Vec3, n: Vec3| n.dot((p - c) / (half * half)) > 0.0;
    let piece_of = |i: u32| v.pieces.iter().find(|q| (q.1..q.1 + q.2).contains(&i)).map_or("?", |q| q.0.as_str());
    let skip = [
        "rim", "tyre", "brake", "engine", "wheel", "seat", "dash", "steering", "console", "headliner", "door_trim",
        "interior", "mirror",
    ];
    let mut bulge = vec![0.0f32; tris.len()];
    for i in 0..v.positions.len() {
        let m = v.material.get(i).copied().unwrap_or(255);
        if !matches!(m, mat::PAINT | mat::GLASS | mat::SCREEN | mat::PRIVACY) {
            continue;
        }
        let pc = piece_of(i as u32);
        let bare = pc.rsplit(':').next().unwrap_or(pc);
        if skip.iter().any(|w| bare.starts_with(w)) {
            continue;
        }
        let p = Vec3::from_array(v.positions[i]);
        if !outer(p, Vec3::from_array(v.normals[i])) {
            continue;
        }
        let (x, y, z) = key(p);
        let mut best = (f32::MAX, usize::MAX);
        for dx in -2..=2 {
            for dy in -2..=2 {
                for dz in -2..=2 {
                    for &k in grid.get(&(x + dx, y + dy, z + dz)).map_or(&[][..], |l| l.as_slice()) {
                        let [a, b, c] = tris[k].1.map(|j| nr[j as usize]);
                        let dd = tri_dist(p, a, b, c);
                        if dd < best.0 {
                            best = (dd, k);
                        }
                    }
                }
            }
        }
        if best.1 == usize::MAX {
            continue;
        }
        let [a, b, c] = tris[best.1].1.map(|j| nr[j as usize]);
        let n = (b - a).cross(c - a).normalize_or_zero();
        let h = (p - a).dot(n);
        if h > 0.0 {
            bulge[best.1] = bulge[best.1].max(h);
        }
    }
    // (CONTACT_OFFSET_LOG=N: the N triangles the paint stands furthest out from, by node.)
    if let Some(n) = std::env::var("CONTACT_OFFSET_LOG").ok().and_then(|x| x.parse::<usize>().ok()) {
        let mut by: Vec<usize> = (0..tris.len()).collect();
        by.sort_by(|&a, &b| bulge[b].total_cmp(&bulge[a]));
        for &k in by.iter().take(n) {
            let names = tris[k].1.map(|j| d.nodes[j as usize].name.clone());
            eprintln!("bulge {:.0} mm over {names:?}", bulge[k] * 1e3);
        }
    }
    let mut off = vec![0.0f32; d.nodes.len()];
    for (k, (_, t)) in tris.iter().enumerate() {
        for &i in t {
            off[i as usize] = off[i as usize].max(bulge[k].min(cap));
        }
    }
    off
}
