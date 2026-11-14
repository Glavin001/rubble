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
    let census = engine_softbody::jbeam::feature_census(&files, Some(&pc)).unwrap();
    for (k, n) in &census {
        println!("{n:6} {k}");
    }
    // What JBeam mode does not simulate (engine/docs/jbeam-semantics-spec.md §11,
    // Part II): empty means every physics feature the configuration uses runs.
    let unsupported = engine_softbody::jbeam::unsupported_features(&census);
    if unsupported.is_empty() {
        println!("JBeam mode simulates every physics feature this configuration uses");
    } else {
        println!("not simulated by JBeam mode:");
        for (k, n) in unsupported {
            println!("{n:6} {k}");
        }
    }
    for (k, n, why) in engine_softbody::jbeam::out_of_scope_features(&census) {
        println!("out of scope: {n:6} {k}: {why}");
    }
}
