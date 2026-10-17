//! Import a scene pack and write it as a stress-oracle scene: `import_pack <pack.json> <out.json>`.

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (src, out) = (std::path::Path::new(&args[1]), &args[2]);
    let name = src.file_stem().unwrap().to_string_lossy().to_string();
    let start = std::time::Instant::now();
    let scene = stress_ref::scene_pack::import(src, &name, "structure").unwrap_or_else(|e| panic!("{e}"));
    eprintln!("imported in {:.2} s: {}; {:?}", start.elapsed().as_secs_f64(), scene.description, stress_ref::scene_pack::summary(&scene));
    std::fs::write(out, serde_json::to_string(&scene).unwrap()).unwrap();
}
