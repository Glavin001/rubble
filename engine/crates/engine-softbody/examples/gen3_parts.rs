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
    for k in &m.markers {
        println!("{} {:.3} {:.3} {:.3}", k.name, k.pos.x, k.pos.y, k.pos.z);
    }
}
