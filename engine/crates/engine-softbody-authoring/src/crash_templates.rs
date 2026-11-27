//! The crash reference suite's vehicles (`engine_softbody::crash_reference`): each design a
//! scenario names, generated from the current generator code and ready to spawn.

use crate::authoring::rig;
use engine_softbody::crash_reference::{CrashScenario, Template, Templates, ZoneMap};

/// The FMVSS 214 moving deformable barrier's design name.
pub const MDB214: &str = "mdb214";

/// A generated design as a template: its JBeam as the solver loads it, the lab's solver
/// settings, and the generator's crash zones (`crash.json`, when the design writes one).
///
/// [`MDB214`] is the FMVSS 214 moving deformable barrier
/// (`engine_softbody::crash_rigs::mdb214_jbeam`), loaded as the web crash lab's `mdb214`
/// scene loads it (`engine-world-wasm` `explicit.rs`, `"native": true`): JBeam mode, i.e.
/// `load_jbeam` and the JBeam spawn (`from_desc_jbeam_with`, the BeamNG profile), not the
/// `.truck` export. It has no wheels (frictionless skids), so the scenario's raycast spawn
/// is the same spawn as the lab's pressure-wheel one, and it never rides a carrier (the
/// lab's `on_carrier` skips it). It has no crash zones.
pub fn template(design: &str) -> Result<Template, String> {
    if design == MDB214 {
        let text = engine_softbody::crash_rigs::mdb214_jbeam();
        let desc = engine_softbody::jbeam::load_jbeam(&[("mdb214.jbeam", text.as_str())], None)?;
        return Ok(Template { desc, settings: crate::vehicle_compare::settings(), zones: ZoneMap::default() });
    }
    let built = rig::rig_vehicle(design, None, None)?;
    template_from_files(design, &rig::mod_files(&built))
}

/// Whether [`template`] builds `design` from code alone, with no mod folder (the barrier).
pub fn from_code(design: &str) -> bool {
    design == MDB214
}

/// A generated design's template from its mod folder's files (path relative to the folder,
/// bytes), as [`template`] makes it from the files the rigger writes: the `.jbeam` parts and
/// the `.pc` loaded as the solver loads them, the lab's solver settings, and `crash.json`'s
/// zones. The web crash lab builds the suite's templates with it from the folder the site
/// build generates (`__mods/<design>_new`, the same rigger on the same code), whose render
/// model and skins it draws.
pub fn template_from_files(design: &str, files: &[(String, Vec<u8>)]) -> Result<Template, String> {
    let txt: Vec<(String, String)> = files.iter().filter(|(p, _)| p.ends_with(".jbeam")).map(|(p, b)| (p.clone(), String::from_utf8_lossy(b).into_owned())).collect();
    let pc = files.iter().find(|(p, _)| p.ends_with(".pc")).map(|(_, b)| String::from_utf8_lossy(b).into_owned());
    let refs: Vec<(&str, &str)> = txt.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let desc = engine_softbody::jbeam::load_jbeam(&refs, pc.as_deref())?;
    let zones = match files.iter().find(|(p, _)| p.ends_with("crash.json")) {
        Some((_, b)) => serde_json::from_slice(b).map_err(|e| format!("{design}: crash.json: {e}"))?,
        None => ZoneMap::default(),
    };
    Ok(Template { desc, settings: crate::vehicle_compare::settings(), zones })
}

/// Every template a scenario needs.
pub fn templates_for(s: &CrashScenario) -> Result<Templates, String> {
    s.designs.iter().map(|d| Ok((d.to_string(), template(d)?))).collect()
}
