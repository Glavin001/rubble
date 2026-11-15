//! The side tests for one car of the comparison manifest: the oblique pole at
//! several offsets along the car (how much of its residual is the lattice's
//! place against the pole) and the MDB, on threads.
//!
//! ```bash
//! SURFACE_CONTACT=1 cargo run --release -p engine-softbody --example gen3_side -- [car] [offset_m ...]
//! ```
use engine_softbody::vehicle_compare::{default_manifest, scene_mdb, scene_pole_at, Car, Manifest};

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let name = a.first().cloned().unwrap_or_else(|| "gen2".into());
    let mut dz: Vec<f32> = a.iter().skip(1).filter_map(|s| s.parse().ok()).collect();
    if dz.is_empty() {
        dz = vec![-0.10, -0.05, 0.0, 0.05, 0.10];
    }
    let m: Manifest = serde_json::from_str(
        &std::fs::read_to_string(default_manifest()).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let spec = m
        .cars
        .iter()
        .find(|c| c.name == name)
        .ok_or("no such car")?;
    let car = Car::load(spec)?;
    let (poles, mdb) = std::thread::scope(|sc| {
        let hs: Vec<_> = dz
            .iter()
            .map(|&d| {
                let car = &car;
                sc.spawn(move || scene_pole_at(car, 32.0, 1.5, d))
            })
            .collect();
        let mdb = sc.spawn(|| scene_mdb(&car, 1.5));
        (
            hs.into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Vec<_>>(),
            mdb.join().unwrap(),
        )
    });
    let mut sum = 0.0;
    for (d, p) in dz.iter().zip(&poles) {
        let r = p["residual_crush_mm"].as_f64().unwrap_or(f64::NAN);
        sum += r;
        println!(
            "pole {d:+.2} m: residual {r} mm, dynamic {} mm, slide {} mm",
            p["dynamic_crush_mm"], p["pole_slide_mm"]
        );
    }
    println!(
        "pole mean residual {:.0} mm; MDB residual {} mm, dynamic {} mm",
        sum / dz.len() as f64,
        mdb["residual_crush_mm"],
        mdb["dynamic_crush_mm"]
    );
    Ok(())
}
