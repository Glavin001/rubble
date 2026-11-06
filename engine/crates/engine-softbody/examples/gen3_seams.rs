//! Opened seams against crush: the wall at several speeds for each car of a
//! manifest (surface contact on), printing residual crush and the mesh's
//! seams opened past 1 and 5 cm, so cars are compared at equal crush.
//!
//! ```bash
//! cargo run --release -p engine-softbody --example gen3_seams -- <manifest.json> [kmh...]
//! ```
use engine_softbody::vehicle_compare::{scene_wall, Car, Manifest};

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let m: Manifest =
        serde_json::from_str(&std::fs::read_to_string(&a[0]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let speeds: Vec<f32> = a.iter().skip(1).filter_map(|s| s.parse().ok()).collect();
    let speeds = if speeds.is_empty() {
        vec![32.0, 40.0, 48.0, 56.0]
    } else {
        speeds
    };
    std::env::set_var("SURFACE_CONTACT", "1");
    for spec in &m.cars {
        let car = Car::load(spec)?;
        for &k in &speeds {
            let w = scene_wall(&car, k, 1.0);
            let v = &w["visual"];
            println!(
                "{} {k} km/h: residual max {} mm, dynamic {} mm; seams > 1 cm {}, > 5 cm {} of {}; stretch p99 {}, fold p99 {}",
                spec.name,
                w["residual_crush_max_mm"],
                w["dynamic_crush_mm"],
                v["seam_open_1cm"],
                v["seam_open_5cm"],
                v["seam_pairs"],
                v["stretch_p99"],
                v["dihedral_p99_deg"]
            );
        }
    }
    Ok(())
}
