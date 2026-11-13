//! Development probe: where a part's surface is inside a box.
//! `PROBE="part x0 x1 y0 y1 z0 z1[; part ...]" cargo test --release -p engine-vehiclegen --test probe -- --nocapture --ignored`

use engine_vehiclegen::{build, design_from_json, BuildOpts};

#[test]
#[ignore]
fn probe() {
    let d = design_from_json("sedan", "{}").unwrap();
    let b = build(d.as_ref(), BuildOpts { uv: false, ..Default::default() });
    let q = std::env::var("PROBE").unwrap_or_default();
    for item in q.split(';').filter(|s| !s.trim().is_empty()) {
        let w: Vec<&str> = item.split_whitespace().collect();
        let v: Vec<f64> = w[1..].iter().map(|x| x.parse().unwrap()).collect();
        for (name, m) in &b.parts.0 {
            if !(name == w[0] || (w[0].ends_with('*') && name.starts_with(&w[0][..w[0].len() - 1]))) {
                continue;
            }
            let pts: Vec<_> = m.v.iter().filter(|p| p.x >= v[0] && p.x <= v[1] && p.y >= v[2] && p.y <= v[3] && p.z >= v[4] && p.z <= v[5]).collect();
            if pts.is_empty() {
                println!("{name}: nothing in the box");
                continue;
            }
            let f = |k: usize, max: bool| pts.iter().map(|p| p[k]).fold(if max { f64::NEG_INFINITY } else { f64::INFINITY }, |a, b| if max { a.max(b) } else { a.min(b) });
            println!("{name}: {} verts  x {:.3}..{:.3}  y {:.3}..{:.3}  z {:.3}..{:.3}", pts.len(), f(0, false), f(0, true), f(1, false), f(1, true), f(2, false), f(2, true));
        }
    }
}
