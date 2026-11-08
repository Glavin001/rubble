//! BeamNG's own cars as calibration references (`vehicle-authoring.md`,
//! "BeamNG's content on the reference scenes"). Every path is given by the
//! caller (the tests read `RUBBLE_JBEAM_BX` / `RUBBLE_BEAMNG_MODS`); nothing
//! here reads or embeds content, and nothing in the repository depends on it.

use crate::desc::SoftVehicleDesc;
use crate::jbeam::{load_jbeam, read_vehicle_dir};
use std::path::Path;

/// A vehicle's assembled configuration: `name`'s files under `vehicles`, with
/// the `.pc` text `pc` (slot choices; `None` = every slot's default).
pub fn load(vehicles: &Path, name: &str, pc: Option<&str>) -> Result<SoftVehicleDesc, String> {
    let files = read_vehicle_dir(vehicles, name)?;
    let refs: Vec<(&str, &str)> = files
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    load_jbeam(&refs, pc)
}

/// The tyre a donor car's first tyred pressure wheel carries: its values and
/// flags other than the hub's, the brake's, the ray count and the offset.
fn is_tyre_key(k: &str) -> bool {
    !k.starts_with("hub") && !k.starts_with("brake") && k != "numRays" && k != "wheelOffset"
}

/// Put `donor`'s tyre on every pressure wheel of `desc` that has none
/// (`hasTire` false: a car whose tyres are in BeamNG's shared `common`
/// content, which is not local). The hubs keep their own values. Returns the
/// donor tyre's (radius, width, pressure psi), or `None` when `desc` already
/// has tyres or the donor has none.
pub fn fit_donor_tyre(desc: &mut SoftVehicleDesc, donor: &SoftVehicleDesc) -> Option<[f32; 3]> {
    if desc
        .pressure_wheels
        .iter()
        .all(|w| w.flags.get("hasTire") != Some(&false))
    {
        return None;
    }
    let t = donor
        .pressure_wheels
        .iter()
        .find(|w| w.flags.get("hasTire") != Some(&false) && w.values.contains_key("radius"))?;
    for w in &mut desc.pressure_wheels {
        for (k, v) in t.values.iter().filter(|(k, _)| is_tyre_key(k)) {
            w.values.insert(k.clone(), *v);
        }
        for (k, v) in t.flags.iter().filter(|(k, _)| is_tyre_key(k)) {
            w.flags.insert(k.clone(), *v);
        }
        w.flags.insert("hasTire".into(), true);
    }
    let get = |k: &str| t.values.get(k).copied().unwrap_or(f32::NAN);
    Some([get("radius"), get("tireWidth"), get("pressurePSI")])
}

/// The only passenger-car pressure tyre in the local community content: a
/// 205/65 R15 (`qlourie_astral_r`'s front, radius 0.324–0.325 m), under
/// `mods` (`RUBBLE_BEAMNG_MODS`). `None` when it is absent.
pub fn donor_tyre_car(mods: &Path) -> Option<SoftVehicleDesc> {
    let dir = mods.join("qlourie_astral_r/vehicles");
    let pc = std::fs::read_to_string(dir.join("qlourie_astral_r/wrc17_tarmac_balanced.pc")).ok()?;
    load(&dir, "qlourie_astral_r", Some(&pc)).ok()
}
