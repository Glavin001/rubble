//! PR #105 audit S6: two sedans head-on, spawned `offset` from the World origin, against
//! the same crash at the origin (the reference: every position there is within a few
//! metres of zero, where f32 resolves 2.4e-7 m). Per frame of 33 steps: the largest
//! distance of any node from where the origin run has it, positions rebuilt in f64 as
//! origin + node − offset. Builds unchanged on main (30ab0e1d) and on the PR.
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use glam::Vec3;

const SEDAN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
));

fn world(at: Vec3) -> ExplicitBeamsWorld {
    let mut s = ExplicitBeamsSettings::default();
    s.fast_inv_sqrt = true;
    let a = ExplicitBeams::from_truck(SEDAN, at + Vec3::new(-4.0, 0.3, 0.0), Vec3::new(10.0, 0.0, 0.0), s).unwrap();
    let b = ExplicitBeams::from_truck(SEDAN, at + Vec3::new(4.0, 0.3, 0.4), Vec3::new(-10.0, 0.0, 0.0), s).unwrap();
    ExplicitBeamsWorld::new(vec![a, b])
}

fn positions(w: &ExplicitBeamsWorld, at: Vec3) -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    for v in &w.vehicles {
        for n in &v.nodes {
            let mut p = [0.0f64; 3];
            for k in 0..3 {
                p[k] = v.origin.to_array()[k] as f64 + n.pos.to_array()[k] as f64 - at.to_array()[k] as f64;
            }
            out.push(p);
        }
    }
    out
}

fn main() {
    let frames: usize = std::env::var("FRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(25);
    let mut reference = world(Vec3::ZERO);
    let mut refs = Vec::new();
    for _ in 0..frames {
        reference.frame(33);
        refs.push(positions(&reference, Vec3::ZERO));
    }
    let offsets: Vec<(&str, Vec3)> = vec![
        ("100m", Vec3::new(100.0, 0.0, 100.0)),
        ("101m", Vec3::new(71.3, 0.0, 71.9)),
        ("99m", Vec3::new(-64.0, 0.0, 75.5)),
        ("103m", Vec3::new(98.7, 0.0, -29.3)),
        ("360m", Vec3::new(300.0, 0.0, -200.0)),
        ("361m", Vec3::new(255.1, 0.0, 255.3)),
        ("359m", Vec3::new(-180.4, 0.0, 310.2)),
        ("362m", Vec3::new(-350.9, 0.0, -88.6)),
    ];
    println!("offset\tframe\tmax_err_m\tfirst_hit_step");
    for (name, at) in offsets {
        let mut w = world(at);
        for f in 0..frames {
            w.frame(33);
            let p = positions(&w, at);
            let e = p
                .iter()
                .zip(&refs[f])
                .map(|(a, b)| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt())
                .fold(0.0f64, f64::max);
            println!("{name}\t{}\t{e:.3e}\t{:?}", f + 1, w.first_hit_step);
        }
    }
}
