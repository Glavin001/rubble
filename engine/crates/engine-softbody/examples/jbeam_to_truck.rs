//! Converts a JBeam vehicle (BeamNG layout: `<vehicles>/<name>/**.jbeam`, the shared
//! `<vehicles>/common/**.jbeam`, and a `.pc` part configuration) to Rigs of Rods
//! `.truck` text through Rubble's importer (`jbeam::load_jbeam`, then
//! `truck_export::desc_to_truck`), as `ExplicitBeams::from_jbeam` does. Both engines
//! then load the same file.
//!
//! ```bash
//! cargo run --release -p engine-softbody --example jbeam_to_truck -- <vehicles-dir> <name> <config.pc> <out.truck>
//! ```
use std::path::{Path, PathBuf};

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let p = e.unwrap().path();
        if p.is_dir() {
            walk(&p, out);
        } else if p.extension().is_some_and(|x| x == "jbeam") {
            out.push(p);
        }
    }
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let [dir, name, pc, out] = &a[..] else {
        panic!("usage: <vehicles-dir> <name> <config.pc> <out.truck>");
    };
    let dir = Path::new(dir);
    let mut paths = Vec::new();
    walk(&dir.join(name), &mut paths);
    walk(&dir.join("common"), &mut paths);
    paths.sort();
    let texts: Vec<(String, String)> = paths
        .iter()
        .map(|p| {
            let rel = p.strip_prefix(dir).unwrap().to_string_lossy().into_owned();
            (rel, String::from_utf8_lossy(&std::fs::read(p).unwrap()).into_owned())
        })
        .collect();
    let files: Vec<(&str, &str)> = texts.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let pc = std::fs::read_to_string(dir.join(name).join(pc)).unwrap();
    let desc = engine_softbody::jbeam::load_jbeam(&files, Some(&pc)).unwrap();
    let (text, warnings) = engine_softbody::truck_export::desc_to_truck(
        &desc,
        engine_softbody::truck_export::TyreDefaults::for_vehicle(&desc),
    );
    std::fs::write(out, &text).unwrap();
    println!(
        "{} jbeam files -> {out}: {} nodes, {} beams, {} triangles, {} wheels; {} warnings",
        files.len(),
        desc.nodes.len(),
        desc.beams.len(),
        desc.triangles.len(),
        desc.wheels.len(),
        warnings.len()
    );
    for w in warnings.iter().take(15) {
        println!("  warning: {w}");
    }
}
