//! Which beams break while a vehicle stands still (CPU reference, drag off): spawns
//! the `.truck` 2 cm above the ground at rest, runs `secs`, and lists the broken
//! beams by kind and type with their stiffness, strength, rest length, node masses
//! and the step at which each broke.
//!
//! cargo run --release -p engine-softbody --example explicit_idle_breaks -- <file.truck> [secs=2]
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use glam::Vec3;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let file = &args[0];
    let secs: f32 = args
        .iter()
        .find_map(|a| a.strip_prefix("secs="))
        .map_or(2.0, |s| s.parse().unwrap());
    let text = std::fs::read_to_string(file).unwrap();
    let settings = ExplicitBeamsSettings { drag: false, turbulence: false, ..Default::default() };
    let mut sim = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, settings).unwrap();
    let lowest = sim.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut sim.nodes {
        n.pos.y += 0.02 - lowest;
    }
    let mut when = vec![u64::MAX; sim.beams.len()];
    for s in 0..(secs * 2000.0) as u64 {
        sim.step();
        for (i, b) in sim.beams.iter().enumerate() {
            if b.broken && when[i] == u64::MAX {
                when[i] = s;
            }
        }
    }
    let broken: Vec<usize> = (0..sim.beams.len()).filter(|&i| sim.beams[i].broken).collect();
    println!("{file}: {} of {} beams broken after {secs} s at rest", broken.len(), sim.beams.len());
    let mut by: std::collections::BTreeMap<String, usize> = Default::default();
    for &i in &broken {
        let b = &sim.beams[i];
        *by.entry(format!("{:?}/{:?}", b.kind, b.ty)).or_default() += 1;
    }
    println!("by kind/type: {by:?}");
    let mut list: Vec<usize> = broken.clone();
    list.sort_by_key(|&i| when[i]);
    println!("first 40 to break: step  a  b  kind/type  k  c  strength  deform(tc)  rest0  mass_a mass_b  group");
    for &i in list.iter().take(40) {
        let b = &sim.beams[i];
        println!(
            "{:6} {:4} {:4} {:?}/{:?} k={:.0} c={:.0} S={:.0} D={:.0} L0={:.3} m=({:.2},{:.2}) g={}",
            when[i], b.a, b.b, b.kind, b.ty, b.k, b.c, b.strength, b.tc, b.rest0,
            sim.nodes[b.a as usize].mass, sim.nodes[b.b as usize].mass, b.group
        );
    }
}
