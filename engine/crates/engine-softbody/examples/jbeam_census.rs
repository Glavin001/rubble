//! Which JBeam features a vehicle configuration uses ([`engine_softbody::jbeam::feature_census`]).
//!
//! ```bash
//! cargo run --release -p engine-softbody --example jbeam_census -- <vehicles-dir> <name> <config.pc>
//! ```
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let [dir, name, pc] = &a[..] else {
        panic!("usage: <vehicles-dir> <name> <config.pc>");
    };
    let dir = std::path::Path::new(dir);
    let texts = engine_softbody::jbeam::read_vehicle_dir(dir, name).unwrap();
    let files: Vec<(&str, &str)> = texts
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let pc = std::fs::read_to_string(dir.join(name).join(pc)).unwrap();
    for (k, n) in engine_softbody::jbeam::feature_census(&files, Some(&pc)).unwrap() {
        println!("{n:6} {k}");
    }
}
