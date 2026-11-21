//! What surface contact changes, on the CPU reference (no GPU): the car at `--speed` into
//! a fixed 12 cm pole that passes between the nose's nodes, with node contact only and
//! with surface contact. Per frame: the car's mean forward speed and how far the pole's
//! near face is inside the car's outline (the front-most node's z less the face's).
//!   surfdemo [--speed V] [--frames F]
use engine_softbody::explicit_beams::{JbeamWheels, StaticBox};
use gpu_floor::car;

fn arg(k: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1).cloned())
}

fn main() -> anyhow::Result<()> {
    let speed: f32 = arg("--speed").map_or(11.0, |x| x.parse().unwrap());
    let frames: usize = arg("--frames").map_or(40, |x| x.parse().unwrap());
    let t = car::load_template(&car::default_car_dir(), JbeamWheels::Raycast)?;
    let mut rows: Vec<Vec<(f32, f32, u32, u32)>> = Vec::new();
    let mut face = 0.0;
    for surface in [false, true] {
        let mut c = car::place(&t, 0.05, speed);
        c.drive = None;
        let front = c.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
        face = front + 0.3;
        c.boxes.push(StaticBox { lo: glam::Vec3::new(0.07, -1.0, face), hi: glam::Vec3::new(0.19, 3.0, face + 0.12) });
        c.surface_contact = surface;
        let mut r = Vec::new();
        for _ in 0..frames {
            let (mut inside, mut hits) = (0u32, 0u32);
            for _ in 0..34 {
                c.step();
                hits += c.surface_last;
                let b = &c.boxes[0];
                inside += c.nodes.iter().filter(|n| n.pos.cmpgt(b.lo).all() && n.pos.cmplt(b.hi).all()).count() as u32;
            }
            let vz = c.nodes.iter().map(|n| n.vel.z * n.mass).sum::<f32>() / c.nodes.iter().map(|n| n.mass).sum::<f32>();
            let nose = c.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
            r.push((vz, nose - face, inside, hits));
        }
        rows.push(r);
    }
    println!("the car at {speed} m/s into a fixed 12 cm pole (near face at z {face:.2}), 0.3 m ahead");
    println!("frame | node contact only: speed m/s, pole inside the outline m, node-in-pole steps | with surface contact: speed, inside, node-in-pole steps, surface hits");
    for f in 0..frames {
        if (f + 1) % 4 == 0 || f + 1 == frames {
            let (a, b) = (rows[0][f], rows[1][f]);
            println!("{:5} | {:7.2} {:7.3} {:5} | {:7.2} {:7.3} {:5} {:6}", f + 1, a.0, a.1, a.2, b.0, b.1, b.2, b.3);
        }
    }
    Ok(())
}
