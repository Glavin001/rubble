//! The pack scenes `sim.methods.multirate` is gated on with `scripts/fidelity.py
//! --scenes DIR`: each scene pack rammed (`scenes::rammed`, a 400 kg box at 12 m/s, as
//! stress-gpu's tests), a Shardwright export rammed and hit by frac-sim's projectile
//! (`shardwright::destruction_case`):
//!
//!   cargo run --release --example multirate_scenes -- OUT_DIR DURATION <pack.json | shardwright.json> ...

use stress_ref::scene::Scene;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (out, duration) = (std::path::Path::new(&args[0]), args[1].parse::<f64>().expect("duration (s)"));
    std::fs::create_dir_all(out).unwrap();
    for path in &args[2..] {
        let p = std::path::Path::new(path);
        let name = p.file_stem().unwrap().to_string_lossy().replace('-', "_");
        let base = stress_ref::scene_pack::import(p, &name, "structure").unwrap_or_else(|e| panic!("{path}: {e}"));
        let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
        let scenes: Vec<Scene> = if json["format"] == "shardwright-stress/1" {
            ["ram", "projectile"].iter().map(|c| stress_ref::shardwright::destruction_case(&base, c, "").unwrap()).collect()
        } else {
            let mut s = stress_ref::scenes::rammed(base);
            s.name = format!("{name}_rammed");
            vec![s]
        };
        for mut s in scenes {
            s.sim.duration = duration;
            let file = out.join(format!("{}.json", s.name));
            std::fs::write(&file, serde_json::to_string(&s).unwrap()).unwrap();
            println!("{} ({} chunks, {} impactors)", file.display(), s.bodies.iter().map(|b| b.chunks.len()).sum::<usize>(), s.impactors.len());
        }
    }
}
