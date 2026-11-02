//! How a vehicle settles on the explicit solver (CPU reference): idle, or a wall
//! impact, run for a long time, printing every 2.5 s the vehicle's internal motion
//! (node speed RMS and max relative to its mean velocity), bent / broken beams and
//! node-distance drift of the shape relative to 2.5 s earlier.
//!
//! cargo run --release -p engine-softbody --example explicit_settle -- <file.truck>
//!     [scene=idle|wall] [secs=45] [drag=1] [speed=27.78]
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, StaticBox};
use glam::Vec3;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let file = args.iter().find(|a| !a.contains('=')).expect("a .truck file");
    let get = |k: &str, d: &str| {
        args.iter()
            .find_map(|a| a.strip_prefix(&format!("{k}=")).map(str::to_string))
            .unwrap_or_else(|| d.to_string())
    };
    let scene = get("scene", "wall");
    let secs: f32 = get("secs", "45").parse().unwrap();
    let drag = get("drag", "1") != "0";
    let speed: f32 = get("speed", "27.78").parse().unwrap();
    let text = std::fs::read_to_string(file).unwrap();
    let settings = ExplicitBeamsSettings { drag, turbulence: drag, ..Default::default() };
    let vel = if scene == "idle" { Vec3::ZERO } else { Vec3::Z * speed };
    let mut sim = ExplicitBeams::from_truck(&text, Vec3::ZERO, vel, settings).unwrap();
    let lowest = sim.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut sim.nodes {
        n.pos.y += 0.02 - lowest;
    }
    sim.roll_tyres(vel);
    if scene == "wall" {
        let front = sim.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
        sim.boxes.push(StaticBox::centered(
            Vec3::new(0.0, 2.0, front + 1.0 + 0.5),
            Vec3::new(4.0, 2.0, 0.5),
        ));
    }
    println!("{file} scene={scene} drag={drag} nodes={} beams={}", sim.nodes.len(), sim.beams.len());
    println!("   t(s)  speed RMS / max (m/s, internal)  bent  broken  shape change over 2.5 s RMS / max (mm)");
    let frame = 5000u64; // 2.5 s
    let total = (secs * 2000.0) as u64;
    let shape = |s: &ExplicitBeams| -> Vec<Vec3> {
        let c = s.nodes.iter().map(|n| n.pos).sum::<Vec3>() / s.nodes.len() as f32;
        s.nodes.iter().map(|n| n.pos - c).collect()
    };
    let mut prev = shape(&sim);
    for k in 1..=total / frame {
        for _ in 0..frame {
            sim.step();
        }
        let n = sim.nodes.len() as f32;
        let mean = sim.nodes.iter().map(|n| n.vel).sum::<Vec3>() / n;
        let rel: Vec<f32> = sim.nodes.iter().map(|x| (x.vel - mean).length()).collect();
        let rms = (rel.iter().map(|v| v * v).sum::<f32>() / n).sqrt();
        let max = rel.iter().copied().fold(0.0, f32::max);
        let (broken, bent) = sim.damage();
        let now = shape(&sim);
        let d: Vec<f32> = now.iter().zip(&prev).map(|(a, b)| (*a - *b).length() * 1000.0).collect();
        let drms = (d.iter().map(|v| v * v).sum::<f32>() / n).sqrt();
        let dmax = d.iter().copied().fold(0.0, f32::max);
        println!(
            "{:7.1}  {:8.4} / {:7.3}                   {:5} {:6}   {:9.2} / {:8.1}",
            k as f32 * 2.5, rms, max, bent, broken, drms, dmax
        );
        prev = now;
    }
}
