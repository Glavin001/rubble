//! The explicit modules' WGSL as the host creates pipelines from it: every lookup of a
//! module's text by name goes through here ([`module_wgsl`]), the build's embedded
//! modules, a measurement's override directory, or a test-only module compiled on
//! demand ([`crate::test_kernels_dir`]).

use crate::kernel::{kernel, Kernel};
use crate::{
    EXPLICIT_BEAMS_AERO_WGSL, EXPLICIT_BEAMS_AERO_WIDE_WGSL, EXPLICIT_BEAMS_CONTACT_AERO_WGSL,
    EXPLICIT_BEAMS_CONTACT_AERO_WIDE_WGSL, EXPLICIT_BEAMS_CONTACT_WGSL,
    EXPLICIT_BEAMS_CONTACT_WIDE_WGSL, EXPLICIT_BEAMS_SOLO_AERO_WGSL, EXPLICIT_BEAMS_SOLO_AERO_WIDE_WGSL,
    EXPLICIT_BEAMS_SOLO_WGSL, EXPLICIT_BEAMS_SOLO_WIDE_WGSL, EXPLICIT_BEAMS_VV_PASSES, EXPLICIT_BEAMS_WGSL,
    EXPLICIT_BEAMS_WIDE_PASSES, EXPLICIT_BEAMS_WIDE_WGSL,
};

/// The split group path (`EB_VV_SPLIT`, engine/docs/measurements/contact-groups-design.md):
/// per frame-module variant (plain, wide, aero, aero wide) the members' step
/// (`explicit_beams_split_member`), the groups' pass (`explicit_beams_split_group`) and
/// its triangle tests across workgroups (`explicit_beams_split_tests`); and the pass's
/// front as fleet-wide dispatches (`explicit_beams_fleet_front`, `explicit_beams_fleet_plan`:
/// explicit_beams_fleet.slang, contact-groups-design.md §8); and the members' raycast
/// wheels on each other (`explicit_beams_split_wheels`).
/// WebGPU only (native wgpu and browsers).
pub const EXPLICIT_BEAMS_SPLIT_GROUPS: [(&str, Kernel); 48] = [
    ("explicit_beams_split_edges", kernel!("explicit_beams_split_edges")),
    ("explicit_beams_split_edges_wide", kernel!("explicit_beams_split_edges_wide")),
    ("explicit_beams_split_edges_aero", kernel!("explicit_beams_split_edges_aero")),
    ("explicit_beams_split_edges_aero_wide", kernel!("explicit_beams_split_edges_aero_wide")),
    ("explicit_beams_split_groupbegin", kernel!("explicit_beams_split_groupbegin")),
    ("explicit_beams_split_groupbegin_wide", kernel!("explicit_beams_split_groupbegin_wide")),
    ("explicit_beams_split_groupbegin_aero", kernel!("explicit_beams_split_groupbegin_aero")),
    ("explicit_beams_split_groupbegin_aero_wide", kernel!("explicit_beams_split_groupbegin_aero_wide")),
    ("explicit_beams_split_grouplevels", kernel!("explicit_beams_split_grouplevels")),
    ("explicit_beams_split_grouplevels_wide", kernel!("explicit_beams_split_grouplevels_wide")),
    ("explicit_beams_split_grouplevels_aero", kernel!("explicit_beams_split_grouplevels_aero")),
    ("explicit_beams_split_grouplevels_aero_wide", kernel!("explicit_beams_split_grouplevels_aero_wide")),
    ("explicit_beams_split_grouplv", kernel!("explicit_beams_split_grouplv")),
    ("explicit_beams_split_grouplv_wide", kernel!("explicit_beams_split_grouplv_wide")),
    ("explicit_beams_split_grouplv_aero", kernel!("explicit_beams_split_grouplv_aero")),
    ("explicit_beams_split_grouplv_aero_wide", kernel!("explicit_beams_split_grouplv_aero_wide")),
    ("explicit_beams_split_wheels", kernel!("explicit_beams_split_wheels")),
    ("explicit_beams_split_wheels_wide", kernel!("explicit_beams_split_wheels_wide")),
    ("explicit_beams_split_wheels_aero", kernel!("explicit_beams_split_wheels_aero")),
    ("explicit_beams_split_wheels_aero_wide", kernel!("explicit_beams_split_wheels_aero_wide")),
    ("explicit_beams_split_fleetlevels", kernel!("explicit_beams_split_fleetlevels")),
    ("explicit_beams_split_fleetlevels_wide", kernel!("explicit_beams_split_fleetlevels_wide")),
    ("explicit_beams_split_fleetlevels_aero", kernel!("explicit_beams_split_fleetlevels_aero")),
    ("explicit_beams_split_fleetlevels_aero_wide", kernel!("explicit_beams_split_fleetlevels_aero_wide")),
    ("explicit_beams_split_fleetpost", kernel!("explicit_beams_split_fleetpost")),
    ("explicit_beams_split_fleetpost_wide", kernel!("explicit_beams_split_fleetpost_wide")),
    ("explicit_beams_split_fleetpost_aero", kernel!("explicit_beams_split_fleetpost_aero")),
    ("explicit_beams_split_fleetpost_aero_wide", kernel!("explicit_beams_split_fleetpost_aero_wide")),
    ("explicit_beams_split_fleetfront", kernel!("explicit_beams_split_fleetfront")),
    ("explicit_beams_split_fleetplan", kernel!("explicit_beams_split_fleetplan")),
    ("explicit_beams_split_fleetfront_wide", kernel!("explicit_beams_split_fleetfront_wide")),
    ("explicit_beams_split_fleetplan_wide", kernel!("explicit_beams_split_fleetplan_wide")),
    ("explicit_beams_split_fleetfront_aero", kernel!("explicit_beams_split_fleetfront_aero")),
    ("explicit_beams_split_fleetplan_aero", kernel!("explicit_beams_split_fleetplan_aero")),
    ("explicit_beams_split_fleetfront_aero_wide", kernel!("explicit_beams_split_fleetfront_aero_wide")),
    ("explicit_beams_split_fleetplan_aero_wide", kernel!("explicit_beams_split_fleetplan_aero_wide")),
    ("explicit_beams_split_member", kernel!("explicit_beams_split_member")),
    ("explicit_beams_split_group", kernel!("explicit_beams_split_group")),
    ("explicit_beams_split_tests", kernel!("explicit_beams_split_tests")),
    ("explicit_beams_split_member_wide", kernel!("explicit_beams_split_member_wide")),
    ("explicit_beams_split_group_wide", kernel!("explicit_beams_split_group_wide")),
    ("explicit_beams_split_tests_wide", kernel!("explicit_beams_split_tests_wide")),
    ("explicit_beams_split_member_aero", kernel!("explicit_beams_split_member_aero")),
    ("explicit_beams_split_group_aero", kernel!("explicit_beams_split_group_aero")),
    ("explicit_beams_split_tests_aero", kernel!("explicit_beams_split_tests_aero")),
    ("explicit_beams_split_member_aero_wide", kernel!("explicit_beams_split_member_aero_wide")),
    ("explicit_beams_split_group_aero_wide", kernel!("explicit_beams_split_group_aero_wide")),
    ("explicit_beams_split_tests_aero_wide", kernel!("explicit_beams_split_tests_aero_wide")),
];

/// Measurement only: the explicit modules' WGSL read from another build's generated
/// directory (`NAME.wgsl`, e.g. a copy of an older build's `OUT_DIR/wgsl`) instead of
/// this build's, for the pipelines created after the call (`None`: this build's). With
/// it, one process runs an older kernel against the current one from one snapshot
/// (`vehicle_lab --ab-switch kernel`): bit for bit and time. Process-wide.
#[doc(hidden)]
pub fn set_explicit_module_override(dir: Option<std::path::PathBuf>) {
    *MODULE_OVERRIDE.lock().unwrap() = dir;
}

static MODULE_OVERRIDE: std::sync::Mutex<Option<std::path::PathBuf>> = std::sync::Mutex::new(None);

fn module_override(name: &str) -> Option<&'static str> {
    type Cache = std::sync::Mutex<std::collections::HashMap<std::path::PathBuf, &'static str>>;
    static READ: std::sync::OnceLock<Cache> = std::sync::OnceLock::new();
    let dir = MODULE_OVERRIDE.lock().unwrap().clone()?;
    let path = dir.join(format!("{name}.wgsl"));
    let mut cache = READ.get_or_init(Default::default).lock().unwrap();
    let text = cache.entry(path.clone()).or_insert_with(|| {
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        Box::leak(text.into_boxed_str())
    });
    Some(*text)
}

/// A module's WGSL as the host creates its pipeline: the uniform-load rewrite
/// ([`crate::uniformize_workgroup_lanes`]), then with bit-exact results off the strict
/// arithmetic's rewrite ([`crate::fast_float_wgsl`]; `bit_exact`: the build's text as
/// before).
pub(crate) fn host_wgsl(src: &str, bit_exact: bool) -> String {
    let u = crate::uniformize_workgroup_lanes(src);
    if bit_exact {
        u
    } else {
        crate::fast_float_wgsl(&u).into_owned()
    }
}

/// The WGSL of a plan module ([`engine_exec::explicit_plan::MODULES`]).
pub(crate) fn module_wgsl(name: &str) -> &'static str {
    if let Some(text) = module_override(name) {
        return text;
    }
    match name {
        "explicit_beams" => EXPLICIT_BEAMS_WGSL.text(),
        "explicit_beams_wide" => EXPLICIT_BEAMS_WIDE_WGSL.text(),
        "explicit_beams_aero" => EXPLICIT_BEAMS_AERO_WGSL.text(),
        "explicit_beams_aero_wide" => EXPLICIT_BEAMS_AERO_WIDE_WGSL.text(),
        "explicit_beams_solo" => EXPLICIT_BEAMS_SOLO_WGSL.text(),
        "explicit_beams_solo_wide" => EXPLICIT_BEAMS_SOLO_WIDE_WGSL.text(),
        "explicit_beams_solo_aero" => EXPLICIT_BEAMS_SOLO_AERO_WGSL.text(),
        "explicit_beams_solo_aero_wide" => EXPLICIT_BEAMS_SOLO_AERO_WIDE_WGSL.text(),
        "explicit_beams_contact" => EXPLICIT_BEAMS_CONTACT_WGSL.text(),
        "explicit_beams_contact_wide" => EXPLICIT_BEAMS_CONTACT_WIDE_WGSL.text(),
        "explicit_beams_contact_aero" => EXPLICIT_BEAMS_CONTACT_AERO_WGSL.text(),
        "explicit_beams_contact_aero_wide" => EXPLICIT_BEAMS_CONTACT_AERO_WIDE_WGSL.text(),
        "explicit_beams_sleep_list" => crate::EXPLICIT_BEAMS_SLEEP_LIST_WGSL.text(),
        _ => EXPLICIT_BEAMS_VV_PASSES
            .iter()
            .chain(EXPLICIT_BEAMS_WIDE_PASSES.iter())
            .chain(crate::EXPLICIT_BEAMS_NOSURF.iter())
            .chain(crate::EXPLICIT_BEAMS_NOCPL.iter())
            .chain(crate::EXPLICIT_BEAMS_CONTACT_BIG.iter())
            .chain(crate::EXPLICIT_BEAMS_SURF_SPLIT.iter())
            .chain(crate::EXPLICIT_BEAMS_SPLIT_GROUPS.iter())
            .chain(crate::EXPLICIT_BEAMS_SELF_SPLIT.iter())
            .chain(crate::EXPLICIT_BEAMS_WIDE_LIST.iter())
            .chain(crate::EXPLICIT_BEAMS_VV_WIDE.iter())
            .find(|(n, _)| *n == name)
            .map(|(_, w)| w.text())
            .unwrap_or_else(|| test_module(name)),
    }
}

/// A test-only module's WGSL (measurement: the self split's ablation builds), from
/// [`crate::test_kernels_dir`] on first use, kept for the process.
fn test_module(name: &str) -> &'static str {
    #[cfg(not(target_arch = "wasm32"))]
    {
        static KEPT: std::sync::Mutex<Vec<(String, &'static str)>> = std::sync::Mutex::new(Vec::new());
        let mut kept = KEPT.lock().unwrap();
        if let Some((_, w)) = kept.iter().find(|(n, _)| n == name) {
            return w;
        }
        let path = crate::test_kernels_dir().join(format!("{name}.wgsl"));
        let w: &'static str = match std::fs::read_to_string(&path) {
            Ok(w) => Box::leak(w.into_boxed_str()),
            Err(_) => panic!("explicit beams: no WGSL module {name}"),
        };
        kept.push((name.to_string(), w));
        w
    }
    #[cfg(target_arch = "wasm32")]
    panic!("explicit beams: no WGSL module {name} (the test-only modules are native only)")
}

/// A kernel variant's WGSL by module name (the bitwise suites' baseline, reference and
/// deterministic-hits builds: the build's modules or the test-only ones). Native only.
pub(crate) fn variant_wgsl(name: &str) -> String {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let path = crate::explicit_wgsl_path(name);
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }
    #[cfg(target_arch = "wasm32")]
    panic!("{name}: the test-only kernel variants are native only")
}
