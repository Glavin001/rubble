//! What is seen of two cars inside each other: the length of the lines along which one
//! car's painted render triangles cut the other's (m):
//! it needs no closed surface and no collision skin. Two rigged cars (CPU reference, raycast
//! wheels, surface contact on), head on (`head`) or car A into B's side (`side`), car B
//! moved 0 .. N-1 mm: the worst sample and the last one (at rest or near it), min /
//! median / max over the runs. Per run also the crossing at spawn (must be 0).
//!
//! ```bash
//! EDGE_CONTACT=1 cargo run --release -p engine-softbody-lab --example vehicle_paint_overlap -- head 60 1.5 8
//! ```
//! The measure: `engine_softbody_authoring::paint_overlap`; the tests: gen_sedan_paint.rs.

//! Environment: EDGE_CONTACT=1, PRESS=1 (A at full throttle, B's handbrake on), CONTACT=0
//! (the control: the cars go through each other), DEPTH_ONLY=1 (only the depth: a few times faster).

use engine_softbody_authoring::paint_overlap::crash_measuring;

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let scene = a.get(1).cloned().unwrap_or("head".into());
    let kmh: f32 = a.get(2).and_then(|x| x.parse().ok()).unwrap_or(60.0);
    let secs: f32 = a.get(3).and_then(|x| x.parse().ok()).unwrap_or(1.5);
    let runs: usize = a.get(4).and_then(|x| x.parse().ok()).unwrap_or(8);
    let on = |k: &str| std::env::var(k).is_ok_and(|x| x == "1");
    let edge = on("EDGE_CONTACT");
    let res = crash_measuring(&scene, kmh, secs, runs, edge, on("PRESS"), std::env::var("CONTACT").map_or(true, |x| x != "0"), !on("DEPTH_ONLY"), !on("DEPTH_ONLY"))?;
    let col = |c: usize| {
        let mut x: Vec<f32> = res.iter().map(|r| r[c]).collect();
        x.sort_by(f32::total_cmp);
        format!("{:.2} / {:.2} / {:.2}", x[0], x[x.len() / 2], x[x.len() - 1])
    };
    println!("{scene} {kmh} km/h, edge {edge}, {runs} runs, paint cutting paint (m): at spawn {} | worst {} | at {secs} s {} | centres apart then (m) {} | own paint: worst {} | at the end {}", col(0), col(1), col(2), col(3), col(4), col(5));
    let mm = |c: usize| { let mut x: Vec<f32> = res.iter().map(|r| r[c] * 1e3).collect(); x.sort_by(f32::total_cmp); format!("{:.0} / {:.0} / {:.0}", x[0], x[x.len() / 2], x[x.len() - 1]) };
    let pass = res.iter().all(|r| r[6] <= 0.05 && r[8] <= 0.01 && r[9] <= 0.2);
    println!("  DEPTH {} (5 cm, 1 cm from 0.3 s, 0.2 s) {scene} {kmh} edge {edge} press {}: paint inside the other car (mm): worst {} | at the end {} | from 0.3 s after the hit {} | s past 1 cm before {}", if pass { "pass" } else { "FAIL" }, on("PRESS"), mm(6), mm(7), mm(8), col(9));
    Ok(())
}
