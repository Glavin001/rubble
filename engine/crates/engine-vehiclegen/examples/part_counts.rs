//! Every part's triangle count of a design's build (a check that a change touches only its part).
fn main() {
    let name = std::env::args().nth(1).unwrap_or("sedan".into());
    let d = engine_vehiclegen::design_from_json(&name, "{}").unwrap();
    let b = engine_vehiclegen::build(d.as_ref(), engine_vehiclegen::BuildOpts::default());
    for (n, m) in &b.parts.0 {
        println!("{n} {}", m.ntris());
    }
}
