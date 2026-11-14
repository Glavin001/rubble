//! Seam evidence (vehicle-gen3.md round 5): per piece pair that shares a seam
//! at rest, the groups each side binds to, whether they share skinning nodes,
//! and in the wall the widest skin gap and node separation; the hood's tent.
//!
//! ```bash
//! SURFACE_CONTACT=1 cargo run --release -p engine-softbody --example gen3_seam_evidence -- \
//!     [--cars etkk,gen2] [--kmh 56] [--cell 0.01] [--seconds 0.5]
//! ```
use engine_softbody::vehicle_compare::{available, default_manifest, seam_evidence, Car, Manifest};

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let opt = |k: &str| {
        a.iter()
            .position(|x| x == k)
            .and_then(|i| a.get(i + 1).cloned())
    };
    let cars: Vec<String> = opt("--cars")
        .unwrap_or_else(|| "etkk,gen2".into())
        .split(',')
        .map(str::to_string)
        .collect();
    let kmh: f32 = opt("--kmh").and_then(|x| x.parse().ok()).unwrap_or(56.0);
    let cell: f32 = opt("--cell").and_then(|x| x.parse().ok()).unwrap_or(0.01);
    let secs: f32 = opt("--seconds").and_then(|x| x.parse().ok()).unwrap_or(0.5);
    let m: Manifest = serde_json::from_str(
        &std::fs::read_to_string(default_manifest()).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    for spec in m
        .cars
        .iter()
        .filter(|c| cars.contains(&c.name) && available(c))
    {
        let car = Car::load(spec)?;
        let e = seam_evidence(&car, kmh, secs, cell);
        println!("{}", serde_json::to_string_pretty(&e).unwrap());
    }
    Ok(())
}
