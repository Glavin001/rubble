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
    template_with(design, None)
}

/// [`template`] from the design's parameters (`params`: JSON, any subset, the design's
/// defaults for the rest; `vehicle_gen_rig --params`), e.g. a physics block with other front
/// rules, to compare a variant against the default car in the same scenario.
pub fn template_with(design: &str, params: Option<&str>) -> Result<Template, String> {
    if design == MDB214 {
        let text = engine_softbody::crash_rigs::mdb214_jbeam();
        let desc = engine_softbody::jbeam::load_jbeam(&[("mdb214.jbeam", text.as_str())], None)?;
        return Ok(Template { desc, settings: crate::vehicle_compare::settings(), zones: ZoneMap::default() });
    }
    if let Some(local) = design.strip_prefix(engine_softbody::crash_reference::family::LOCAL) {
        return local_template(design, local);
    }
    let built = rig::rig_vehicle(design, params, None)?;
    template_from_files(design, &rig::mod_files(&built))
}

/// A local imported vehicle (`local:<folder>/<pc>`, `crash_reference::family::LOCAL`): the
/// folder's `.jbeam` files and that `.pc` under `RUBBLE_LOCAL_MODS` (the directory of converted
/// mods, e.g. a checkout's `engine/crates/engine-webgpu/__mods`). Local content only: nothing
/// in the repo's tests or builds names one.
fn local_template(design: &str, local: &str) -> Result<Template, String> {
    let root = std::env::var("RUBBLE_LOCAL_MODS").map_err(|_| format!("{design}: set RUBBLE_LOCAL_MODS to the directory of converted mods (it holds {})", local.split('/').next().unwrap_or(local)))?;
    let (folder, pc) = local.split_once('/').ok_or_else(|| format!("{design}: local:<folder>/<path of its .pc>"))?;
    let base = std::path::Path::new(&root).join(folder);
    let list: Vec<String> = serde_json::from_slice(&std::fs::read(base.join("files.json")).map_err(|e| format!("{design}: {}: {e}", base.join("files.json").display()))?).map_err(|e| format!("{design}: files.json: {e}"))?;
    let mut files = Vec::new();
    for f in list.iter().filter(|f| f.ends_with(".jbeam") || f.as_str() == pc) {
        files.push((f.clone(), std::fs::read(base.join(f)).map_err(|e| format!("{design}: {f}: {e}"))?));
    }
    if !files.iter().any(|(f, _)| f == pc) {
        return Err(format!("{design}: {folder} has no {pc}"));
    }
    template_from_files(design, &files)
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
    templates_for_with(s, &[])
}

/// [`templates_for`] with parameters for some designs (`(design, JSON)`; the others their
/// defaults).
pub fn templates_for_with(s: &CrashScenario, params: &[(String, String)]) -> Result<Templates, String> {
    let of = |d: &str| params.iter().find(|(n, _)| n == d).map(|(_, p)| p.as_str());
    s.designs.iter().map(|d| Ok((d.to_string(), template_with(d, of(d))?))).collect()
}

/// Design parameters from `RUBBLE_DESIGN_PARAMS` (`design=FILE[,design=FILE]`), for the
/// diagnostics that build their own templates.
pub fn params_from_env() -> Result<Vec<(String, String)>, String> {
    let Ok(v) = std::env::var("RUBBLE_DESIGN_PARAMS") else { return Ok(Vec::new()) };
    params_from_list(&v)
}

/// `design=FILE[,design=FILE]` read into (design, the file's JSON).
pub fn params_from_list(v: &str) -> Result<Vec<(String, String)>, String> {
    v.split(',')
        .filter(|x| !x.is_empty())
        .map(|kv| {
            let (d, f) = kv.split_once('=').ok_or(format!("params {kv}: want design=FILE"))?;
            Ok((d.to_string(), std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?))
        })
        .collect()
}
