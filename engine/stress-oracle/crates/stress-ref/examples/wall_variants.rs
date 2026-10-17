//! b5's ram-on-wall scene with other walls, for calibrating the punch-through speed:
//! a wall 2 m wide and 2.1 m tall of 0.1 m chunks, `layers` chunks thick (0.1 m each),
//! of `material`, struck at 1 m height by the 1000 kg steel box ram at `speed`. Writes
//! the scene as JSON.
//!
//! ```text
//! cargo run --release -p stress-ref --example wall_variants -- OUT.json concrete|brick LAYERS SPEED
//! ```

use stress_ref::builders::{auto_bonds, body, grid, grid_index, new_scene};
use stress_ref::material::{preset, Material};
use stress_ref::math::Vec3;
use stress_ref::scene::{ImpactorDesc, ImpactorShape, Support};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (out, material, layers, speed) = (&args[0], args[1].as_str(), args[2].parse::<usize>().expect("layers"), args[3].parse::<f64>().expect("speed"));
    let mut s = new_scene(&format!("wall_{material}_{layers}_v{speed}"), "Wall struck by a rigid ram (punch-through calibration).");
    let mat: Material = if material == "concrete" { stress_ref::builders::oracle_concrete(None) } else { preset(material).expect("material") };
    s.materials.insert(material.into(), mat);
    s.materials.insert("steel".into(), stress_ref::builders::elastic_steel());
    let counts = [20, layers, 21];
    let mut chunks = grid(Vec3::new(-1.0, 0.0, -0.1), counts, Vec3::splat(0.1), material);
    for i in 0..counts[0] {
        for j in 0..counts[1] {
            chunks[grid_index(counts, i, j, 0)].support = Support::Fixed;
        }
    }
    let joint = material.to_string();
    let bonds = auto_bonds(&chunks, |_, _| joint.clone());
    s.bodies.push(body("wall", chunks, bonds));
    s.impactors.push(ImpactorDesc {
        name: "ram".into(),
        shape: ImpactorShape::Box { half_extents: [0.2, 0.25, 0.2] },
        mass: 1000.0,
        position: [0.0, -0.251, 1.0],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [0.0, speed, 0.0],
        angular_velocity: [0.0; 3],
        material: "steel".into(),
        crush: None,
    });
    s.sim.duration = if speed >= 20.0 { 0.04 } else if speed >= 8.0 { 0.1 } else { 0.3 };
    std::fs::write(out, serde_json::to_string(&s).expect("json")).expect("write");
}
