//! The solo-car kernel's feasibility census (perf doc §8.43): what one car needs in a
//! workgroup's memory and on its lanes, from the car itself. Prints the counts the
//! layout is computed from; no GPU.
//!
//! `cargo run --profile fast -p engine-world --example solo_census -- [CAR_DIR]`
use engine_softbody::explicit_beams::{ExplicitBeamsSettings, ExplicitKind as K, JbeamWheels};
use std::collections::BTreeMap;

fn main() -> anyhow::Result<()> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| {
        format!("{}/Development/rubble-artifacts/sedan-live/mods/sedan_new", std::env::var("HOME").unwrap())
    });
    let dir = std::path::Path::new(&dir);
    let list: Vec<String> = serde_json::from_str(&std::fs::read_to_string(dir.join("files.json"))?)?;
    let mut texts = Vec::new();
    let mut pc = None;
    for f in &list {
        if f.ends_with(".jbeam") {
            texts.push((f.clone(), std::fs::read_to_string(dir.join(f))?));
        } else if f.ends_with(".pc") {
            pc = Some(std::fs::read_to_string(dir.join(f))?);
        }
    }
    let files: Vec<(&str, &str)> = texts.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let settings = ExplicitBeamsSettings { fast_inv_sqrt: true, ..Default::default() };
    let (s, _) = rubble::showcase_scene::jbeam_vehicle_on(&files, pc.as_deref(), settings, JbeamWheels::Raycast)?;
    let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
    for b in &s.beams {
        *kinds.entry(format!("{:?}", b.kind)).or_default() += 1;
    }
    println!("nodes {}", s.nodes.len());
    println!("beam records {}", s.beams.len());
    for (k, n) in &kinds {
        println!("  {n:5} {k}");
    }
    // Records whose force on an end is a scalar along the beam (one f32 hands both ends
    // their force): every kind but the vector elements.
    let vector = |k: K| matches!(k, K::JbeamAeroTri | K::JbeamAeroTri2 | K::JbeamCoupler | K::JbeamTorsion | K::JbeamTorsionPair);
    let nscalar = s.beams.iter().filter(|b| !vector(b.kind)).count();
    let aero = s.beams.iter().filter(|b| b.kind == K::JbeamAeroTri).count();
    let cpl = s.beams.iter().filter(|b| b.kind == K::JbeamCoupler).count();
    let tor = s.beams.iter().filter(|b| b.kind == K::JbeamTorsion).count();
    println!("scalar records {nscalar}; aero triangles {aero}; couplers {cpl}; torsion bars {tor}");
    let inc: Vec<usize> = s.incidence.iter().map(|v| v.len()).collect();
    println!(
        "incidence max {} mean {:.2} total {}",
        inc.iter().max().unwrap(),
        inc.iter().sum::<usize>() as f32 / inc.len() as f32,
        inc.iter().sum::<usize>()
    );
    println!("collision triangles {} contacters {} boxes {} shapes {}", s.triangles.len(), s.candidates.len(), s.boxes.len(), s.shapes.len());
    println!(
        "tyres {} brakes {} wings {} slides {} shocks2 {} triggers {} commands {} centering {} contact_points {} self_groups {}",
        s.tyres.len(),
        s.brakes.len(),
        s.wings.len(),
        s.slides.len(),
        s.shocks2.len(),
        s.triggers.len(),
        s.commands.len(),
        s.centering.len(),
        s.contact_points.len(),
        s.self_groups.len()
    );
    let groups: std::collections::BTreeSet<i32> = s.beams.iter().map(|b| b.group.abs()).filter(|&g| g != 0).collect();
    println!("break groups {}", groups.len());
    Ok(())
}
