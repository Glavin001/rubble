//! The side's geometry at the oblique pole's line, for comparison with the
//! 2018 Accord's (NHTSA v10841, data sheets 7 and 10): the rig's nodes as
//! CSV on stdout (name, part, x left, y aft of the front axle, z up, mass,
//! skin) and the render model's side parts' extents along the car by height
//! band on stderr.
//!
//! ```bash
//! cargo run --release -p engine-softbody --example gen3_geom > nodes.csv
//! ```
use engine_softbody::authoring::gen2::{model, rig};

fn main() {
    let m = model::build();
    let r = rig::rig(&m);
    println!("name,part,x,y,z,mass,skin");
    for (i, n) in r.car.nodes.iter().enumerate() {
        println!(
            "{},{:?},{:.4},{:.4},{:.4},{:.3},{}",
            n.name, n.part, n.p.x, n.p.y, n.p.z, n.mass, r.skin[i] as u8
        );
    }
    for p in &m.parts {
        let aft: Vec<f32> = p.pos.iter().map(|&q| rig::to_jbeam(q).y).collect();
        eprintln!(
            "part {}: {:.3} .. {:.3} m aft of the front axle",
            p.name,
            aft.iter().copied().fold(f32::MAX, f32::min),
            aft.iter().copied().fold(f32::MIN, f32::max)
        );
    }
    for mk in &m.markers {
        let j = rig::to_jbeam(mk.pos);
        eprintln!(
            "marker {}: {:.3} aft, {:.3} up, {:+.3} left",
            mk.name, j.y, j.z, j.x
        );
    }
    for p in &m.parts {
        if !["door_FL", "door_RL", "body_shell", "glass_door_FL"].contains(&p.name.as_str()) {
            continue;
        }
        for band in [
            (0.3f32, 0.5f32),
            (0.5, 0.8),
            (0.8, 1.0),
            (1.0, 1.15),
            (1.15, 1.3),
            (1.3, 1.45),
        ] {
            let ys: Vec<f32> = p
                .pos
                .iter()
                .filter(|q| q.x > 0.3 && q.y >= band.0 && q.y < band.1)
                .map(|&q| rig::to_jbeam(q).y)
                .collect();
            if ys.is_empty() {
                continue;
            }
            let lo = ys.iter().copied().fold(f32::MAX, f32::min);
            let hi = ys.iter().copied().fold(f32::MIN, f32::max);
            eprintln!(
                "{} height {:.2}-{:.2} m: {lo:.3} .. {hi:.3} m aft of the front axle",
                p.name, band.0, band.1
            );
        }
    }
}
