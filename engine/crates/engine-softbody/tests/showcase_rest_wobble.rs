//! Diagnostic: how still is a car standing at rest (the "cloth" look)? After 3 s of
//! settling, 1 s of samples: the RMS and max deviation of each node from its own
//! mean position, and the max node speed. Prints the showcase sedan beside a
//! real Rigs of Rods car when one is in engine/local-vehicles (gitignored).
//! `cargo test --release -p engine-softbody --test showcase_rest_wobble -- --ignored --nocapture`

use engine_softbody::authoring::{build, style, BuildParams, Resolution};
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use glam::Vec3;

fn pct(mut v: Vec<f32>) -> String {
    v.retain(|x| x.is_finite());
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |f: f32| v[((v.len() - 1) as f32 * f) as usize];
    format!("{:.3e} / {:.3e} / {:.3e}", q(0.1), q(0.5), q(0.9))
}

/// The car's materials as loaded: node mass, beam k and c, the damping ratio per
/// beam (c / 2√(k μ), μ the pair's reduced mass) and the per-node rate √(Σk/m).
fn materials(label: &str, s: &ExplicitBeams) {
    let m = |i: u32| s.nodes[i as usize].mass;
    let live: Vec<_> = s.beams.iter().filter(|b| b.k > 0.0).collect();
    let mut sumk = vec![0.0f32; s.nodes.len()];
    for b in &live { sumk[b.a as usize] += b.k; sumk[b.b as usize] += b.k; }
    let total: f32 = s.nodes.iter().map(|n| n.mass).sum();
    let len: f32 = live.iter().map(|b| b.rest0).sum();
    eprintln!("{label}: {} nodes, {} beams, {:.0} kg, {:.0} m of beam, {:.1} beams/node (p10 / p50 / p90 below)", s.nodes.len(), live.len(), total, len, 2.0 * live.len() as f32 / s.nodes.len() as f32);
    eprintln!("    node mass kg        {}", pct(s.nodes.iter().map(|n| n.mass).collect()));
    eprintln!("    beam k N/m          {}", pct(live.iter().map(|b| b.k).collect()));
    eprintln!("    beam k·L N          {}", pct(live.iter().map(|b| b.k * b.rest0).collect()));
    eprintln!("    beam c N·s/m        {}", pct(live.iter().map(|b| b.c).collect()));
    eprintln!("    damping ratio ζ     {}", pct(live.iter().map(|b| { let mu = m(b.a) * m(b.b) / (m(b.a) + m(b.b)); b.c / (2.0 * (b.k * mu).sqrt()) }).collect()));
    eprintln!("    node rate √(Σk/m)Hz {}", pct((0..s.nodes.len()).map(|i| (sumk[i] / s.nodes[i].mass).sqrt() / std::f32::consts::TAU).collect()));
    eprintln!("    break strength N    {}", pct(live.iter().map(|b| b.strength).collect()));
    eprintln!("    yield (deform) N    {}", pct(live.iter().map(|b| b.tt).collect()));
}

fn settle_and_measure(label: &str, truck: &str) {
    let settings = ExplicitBeamsSettings { turbulence: false, ..Default::default() };
    let probe = ExplicitBeams::from_truck(truck, Vec3::ZERO, Vec3::ZERO, settings.clone()).unwrap();
    materials(label, &probe);
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let mut s = ExplicitBeams::from_truck(truck, Vec3::new(0.0, 0.02 - low, 0.0), Vec3::ZERO, settings).unwrap();
    for _ in 0..180 { s.frame(); }
    let n = s.nodes.len();
    let frames = 60;
    let mut sum = vec![Vec3::ZERO; n];
    let mut samples: Vec<Vec<Vec3>> = Vec::with_capacity(frames);
    let mut vmax = 0.0f32;
    for _ in 0..frames {
        s.frame();
        let p: Vec<Vec3> = s.nodes.iter().map(|x| x.pos).collect();
        for i in 0..n { sum[i] += p[i]; }
        vmax = s.nodes.iter().map(|x| x.vel.length()).fold(vmax, f32::max);
        samples.push(p);
    }
    let mean: Vec<Vec3> = sum.iter().map(|v| *v / frames as f32).collect();
    let (mut sq, mut dmax) = (0.0f64, 0.0f32);
    for p in &samples { for i in 0..n { let d = p[i].distance(mean[i]); sq += (d * d) as f64; dmax = dmax.max(d); } }
    let rms = (sq / (n * frames) as f64).sqrt();
    let vrms = (s.nodes.iter().map(|x| x.vel.length_squared() as f64).sum::<f64>() / n as f64).sqrt();
    eprintln!("{label:>28}: {n:5} nodes · node wobble RMS {:.2} mm, max {:.1} mm · speed RMS {:.1} mm/s, max {:.2} m/s", rms * 1e3, dmax * 1e3, vrms * 1e3, vmax);
}

#[test]
#[ignore]
fn rest_wobble() {
    for res in [Resolution::Low, Resolution::Medium] {
        let b = build(&style("sedan").unwrap(), &BuildParams { resolution: res, ..Default::default() }).unwrap();
        settle_and_measure(&format!("showcase {res:?}"), &b.truck);
    }
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../local-vehicles");
    for f in ["E36Sedan.truck", "1982-Ford-Mustang-GT-351W.truck", "95bbUID-agoras.truck", "ViperGTS.car"] {
        let p = format!("/Users/glavin/Development/rubble-showcase/engine/local-vehicles/{f}");
        let p = if std::path::Path::new(&format!("{dir}/{f}")).exists() { format!("{dir}/{f}") } else { p };
        if let Ok(t) = std::fs::read_to_string(&p) { settle_and_measure(f, &t); }
    }
}
