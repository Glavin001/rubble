//! The generated sedan's collision skin may not open further (an authoring ratchet,
//! 2026-11-28). An open edge is an edge on one collision triangle only: a node passing
//! through the gap there meets no triangle, so continuous contact cannot stop it, at any
//! step size. `vehicle_c2c_overlap` traced most of the cars' fusion entries to such gaps
//! (bumper top and bottom, hood-fender seams; `docs/handoff/known-bugs.md` S4). `lint_skin`'s
//! `skin_holes` fails closed loops under 2.5 m round; this guards the total, and its limit
//! only comes down: lower [`OPEN_EDGES_MAX`] when the generator closes some.

use engine_softbody_authoring::authoring::rig;
use std::collections::HashMap;

/// Open collision edges on the generated sedan (measured 2026-11-28).
const OPEN_EDGES_MAX: usize = 172;

#[test]
fn the_generated_sedans_collision_skin_does_not_open_further() {
    let built = rig::rig_vehicle("sedan", None, None).expect("the generated sedan");
    let files = rig::mod_files(&built);
    let refs: Vec<(String, String)> = files.iter().filter(|(p, _)| p.ends_with(".jbeam")).map(|(p, b)| (p.clone(), String::from_utf8_lossy(b).into_owned())).collect();
    let pc = files.iter().find(|(p, _)| p.ends_with(".pc")).map(|(_, b)| String::from_utf8_lossy(b).into_owned());
    let r: Vec<(&str, &str)> = refs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let d = engine_softbody::jbeam::load_jbeam(&r, pc.as_deref()).expect("the sedan's jbeam");
    let mut by: HashMap<(u32, u32), usize> = HashMap::new();
    for t in d.triangles.iter().filter(|t| t.collision) {
        for k in 0..3 {
            let (a, b) = (t.nodes[k], t.nodes[(k + 1) % 3]);
            *by.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let mut open: Vec<String> = by.iter().filter(|(_, &n)| n == 1).map(|(&(a, b), _)| format!("{}-{}", d.nodes[a as usize].name, d.nodes[b as usize].name)).collect();
    open.sort();
    println!("| open collision edges | {} of {} (limit {OPEN_EDGES_MAX}) | first: {:?} |", open.len(), by.len(), &open[..open.len().min(12)]);
    assert!(open.len() <= OPEN_EDGES_MAX, "the collision skin opened: {} open edges, at most {OPEN_EDGES_MAX}", open.len());
}
