//! The crash reference suite's vehicles (`engine_softbody::crash_reference`): each design a
//! scenario names, generated from the current generator code and ready to spawn.

use crate::authoring::rig;
use engine_softbody::crash_reference::{CrashScenario, Template, Templates, ZoneMap};

/// A generated design as a template: its JBeam as the solver loads it, the lab's solver
/// settings, and the generator's crash zones (`crash.json`, when the design writes one).
pub fn template(design: &str) -> Result<Template, String> {
    let built = rig::rig_vehicle(design, None, None)?;
    let files = rig::mod_files(&built);
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
