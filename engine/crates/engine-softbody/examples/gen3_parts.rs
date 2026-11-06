use engine_softbody::authoring::gen2::model;
use glam::Vec3;
fn main() {
    let m = model::build();
    let mut tot = 0;
    for p in &m.parts {
        let lo = p.pos.iter().fold(Vec3::splat(9.0), |a, b| a.min(*b));
        let hi = p.pos.iter().fold(Vec3::splat(-9.0), |a, b| a.max(*b));
        tot += p.pos.len();
        println!(
            "{:<22} v {:6} t {:6} ch {:3} lo ({:.3},{:.3},{:.3}) hi ({:.3},{:.3},{:.3})",
            p.name,
            p.pos.len(),
            p.tris.len(),
            p.charts.len(),
            lo.x,
            lo.y,
            lo.z,
            hi.x,
            hi.y,
            hi.z
        );
    }
    println!("total verts {tot}");
    // `GEN3_EDGE=<part>`: its side-facing skin's front and rear edge (z) per
    // height, every 5 cm.
    if let Ok(name) = std::env::var("GEN3_EDGE") {
        let p = m.parts.iter().find(|p| p.name == name).expect("part");
        let side: Vec<Vec3> = p
            .pos
            .iter()
            .zip(&p.nrm)
            .filter(|(_, n)| n.x.abs() > 0.45)
            .map(|(q, _)| *q)
            .collect();
        for k in 0..30 {
            let y = 0.25 + 0.05 * k as f32;
            let zs: Vec<f32> = side
                .iter()
                .filter(|q| (q.y - y).abs() < 0.025)
                .map(|q| q.z)
                .collect();
            if zs.is_empty() {
                continue;
            }
            let (lo, hi) = zs
                .iter()
                .fold((f32::MAX, f32::MIN), |a, &z| (a.0.min(z), a.1.max(z)));
            println!("{name} y {y:.2}: z {lo:.3} .. {hi:.3}");
        }
    }
    for k in &m.markers {
        println!("{} {:.3} {:.3} {:.3}", k.name, k.pos.x, k.pos.y, k.pos.z);
    }
}
