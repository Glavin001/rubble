//! Scratch: which visual pieces move or tear at idle (not committed).
use engine_softbody::authoring::skin::{deform, showcase_visual};
use engine_softbody::authoring::{build, style, BuildParams, Resolution};
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use glam::Vec3;
fn main() {
    let b = build(&style("sedan").unwrap(), &BuildParams { resolution: Resolution::Medium, ..Default::default() }).unwrap();
    let t0 = std::time::Instant::now();
    let v = showcase_visual(&b).unwrap();
    eprintln!("visual build {:?}", t0.elapsed());
    if std::env::var("COUNTS").is_ok() {
        let mut per: std::collections::BTreeMap<String, usize> = Default::default();
        for t in v.indices.chunks(3) {
            let i = t[0];
            let name = v.pieces.iter().find(|(_, f, c)| i >= *f && i < f + c).map(|p| p.0.clone()).unwrap_or_default();
            *per.entry(name).or_default() += 1;
        }
        for (k, n) in per { println!("{k}: {n}"); }
        return;
    }
    let settings = ExplicitBeamsSettings { turbulence: false, ..Default::default() };
    let probe = ExplicitBeams::from_truck(&b.truck, Vec3::ZERO, Vec3::ZERO, settings).unwrap();
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let mut s = ExplicitBeams::from_truck(&b.truck, Vec3::new(0.0, 0.02 - low, 0.0), Vec3::ZERO, settings).unwrap();
    let rest = s.positions();
    let frames: u32 = std::env::args().nth(1).map(|a| a.parse().unwrap()).unwrap_or(90);
    for _ in 0..frames { s.frame(); }
    let (p, _) = deform(&v, &s.positions());
    let (p0, _) = deform(&v, &rest);
    for (name, first, count) in &v.pieces {
        let mut torn = 0; let mut big = 0; let mut maxd = 0.0f32; let mut at = Vec3::ZERO;
        for i in *first..first + count {
            let i = i as usize;
            if !p[i].is_finite() { torn += 1; continue; }
            let d = p[i].distance(p0[i]);
            if d > 0.05 { big += 1; }
            if d > maxd { maxd = d; at = p0[i]; }
        }
        if torn > 0 || big > 0 { println!("{name}: {count} verts, torn {torn}, >5cm {big}, max {maxd:.3} at {at:.2}"); }
        if name == "bumper_f" {
            let mut best = (0.0f32, 0usize);
            for i in *first as usize..(first + count) as usize { let d = p[i].distance(p0[i]); if d > best.0 { best = (d, i); } }
            let i = best.1;
            let t = v.bind_nodes[i];
            println!("  bind {:?} coords {:?} pos {:?}", t, v.bind_coords[i], v.positions[i]);
            for tri in v.indices.chunks(3) { if tri.contains(&(i as u32)) { println!("   tri {:?}", tri.iter().map(|&k| v.positions[k as usize]).collect::<Vec<_>>()); } }
            for &n in &t { let n = n as usize; println!("   node {n} part {} rest {:.3} moved {:.3}", b.skeleton.parts[b.skeleton.nodes.get(n).map(|q| q.part as usize).unwrap_or(0)], rest[n], s.positions()[n].distance(rest[n])); }
        }
    }
}
