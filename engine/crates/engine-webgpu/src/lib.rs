//! WebGPU backend. Default: Slang → WGSL directly at build time (`-DENGINE_WGSL`, see
//! `slang/physics/atomics.slang`). `WG_NAGA=1` selects the older Slang → SPIR-V → naga path.
//!
//! **Fair speed bar vs three-avbd:** same scenes in **Chrome WebGPU** on this machine
//! (not native Metal `scene_fps` vs their browser `gpuTotal`). Metal remains the M3
//! interactive gate; this crate is the apples-to-apples oracle path.

#[cfg(feature = "gpu")]
mod compute_spec;
#[cfg(feature = "gpu")]
mod explicit_beams_gpu;
pub mod kernel;
#[cfg(kernels_at_runtime)]
pub mod kernel_store;
/// The kernels' compile (src/kernel_build.rs, as build.rs runs it), for the run-time
/// store and the test that holds the two the same.
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
mod kernel_compile {
    use crate::explicit_post_text;
    use anyhow::{Context, Result};
    use engine_slang_build::{compile, compile_entries_to, compile_many, compile_with_defines, par_map, Compile, SlangTarget};
    use std::path::{Path, PathBuf};
    include!("kernel_build.rs");
}
pub use kernel::Kernel;
use kernel::kernel;

/// On wasm, how a build with its kernels compiled at run time (a fast web build) reads
/// them: `load(rel)` returns the store's file `rel` (`wgsl/<name>.wgsl`, `wgsl_ok.txt`,
/// `spirv_entries.txt`). Nothing to do anywhere else (embedded kernels, or native, which
/// reads its own store).
pub fn set_kernel_loader(load: fn(&str) -> Result<String, String>) {
    #[cfg(all(kernels_at_runtime, target_arch = "wasm32"))]
    kernel_store::set_loader(load);
    #[cfg(not(all(kernels_at_runtime, target_arch = "wasm32")))]
    let _ = load;
}
#[cfg(feature = "gpu")]
mod explicit_modules;
#[cfg(feature = "gpu")]
pub mod explicit_coupling;
#[cfg(feature = "gpu")]
mod fast_float;
#[cfg(feature = "gpu")]
mod features;
#[cfg(feature = "gpu")]
mod gpu;
#[cfg(feature = "gpu")]
pub mod gpu_labels;
pub mod host_profile;
#[cfg(feature = "gpu")]
pub mod gpu_marks;
#[cfg(feature = "gpu")]
mod poses;
#[cfg(all(feature = "gpu", target_vendor = "apple"))]
pub mod metal_interop;
#[cfg(feature = "gpu")]
mod soft_skin;
#[cfg(feature = "gpu")]
pub use compute_spec::{mark_pipeline_compiled, pipeline_ready, set_pipeline_gate, ComputeSpec};
#[cfg(feature = "gpu")]
pub use explicit_modules::set_explicit_module_override;
#[cfg(feature = "gpu")]
pub use explicit_beams_gpu::{
    explicit_beams_shipped_wgsl, group_split_default, set_explicit_shader_checks_off, explicit_shader_module, explicit_shader_checks_off, explicit_compilation_options, SHADER_CHECKS_OFF_METAL, render_scene_json, CouplingCapacity, CouplingLayout, ExplicitBeamsGpu,
    GpuBeam, GpuSnapshot, KernelVariant, StagedSnapshot, StagedState, state_differences, COUPLING_FOLD_OFF, COUPLING_LIST_OFF, COUPLING_LIST_SKIN_CEILING, COUPLING_LIST_SKIN_NODES_ONLY,
};
#[cfg(feature = "gpu")]
pub use fast_float::{bit_exact_default, explicit_wgsl, fast_float_wgsl};
#[cfg(feature = "gpu")]
pub use features::{
    ContactStorage, EngineFeatures, GpuTiming, GridLayout, KernelGroups, ManifoldPersistence,
    RestingContacts, SimOptions, WarmHash,
};
#[cfg(feature = "gpu")]
pub use gpu::creation_errors;
/// The explicit kernels' loop-cap error (every target writes the flag): see
/// `engine_exec::explicit_plan::EXPLICIT_LOOP_CAP`.
pub use engine_exec::explicit_plan::{explicit_loop_cap_hit, loop_budget_override};
#[cfg(feature = "gpu")]
pub use gpu::{
    default_device, default_device_with, hold_gpu_lock, instance_flags, replay, CharacterHandle, KernelReplayOptions, KernelReplayReport, ReplayedDispatch, MotionEvent, Prewarm, RingEvents, SimSnapshot, SoftEvents, StateMeasure, TerrainStepReport, TerrainStrike, TerrainWatch, WgpuSim, MOVED_BINDING, RestView,
    POSES_BINDING, POSE_HEADER_BINDING, POSE_RECORD_BYTES,
};
#[cfg(feature = "gpu")]
pub use poses::PoseRecord;
#[cfg(feature = "gpu")]
pub use soft_skin::SoftSkinner;

/// The explicit vehicles' World coupling (`slang/kernels/vehicle_coupling.slang`).
pub const VEHICLE_COUPLING_WGSL: Kernel = kernel!("vehicle_coupling");
/// The skinning kernels as one WGSL module (entries `soft_skin_vertices`,
/// `soft_skin_faces`, `soft_skin_normals`), as `generated/wgsl/soft_skin.wgsl`.
pub const SOFT_SKIN_WGSL: Kernel = kernel!("soft_skin");

/// The explicit-beams solver (`slang/kernels/explicit_beams.slang`, entry
/// `explicit_beams_frame`): one workgroup per vehicle runs a whole frame of steps
/// ([`ExplicitBeamsGpu`]), as `generated/wgsl/explicit_beams.wgsl`.
pub const EXPLICIT_BEAMS_WGSL: Kernel = kernel!("explicit_beams");
/// The larger contact groups' modules (`EB_VV_BIG_ONLY`, entry
/// `explicit_beams_contact`; `engine_exec::explicit_plan::Kernel::ContactBig`), at the
/// frame module's four variants.
pub const EXPLICIT_BEAMS_CONTACT_BIG: [(&str, Kernel); 4] = [
    ("explicit_beams_contact_big", kernel!("explicit_beams_contact_big")),
    ("explicit_beams_contact_big_wide", kernel!("explicit_beams_contact_big_wide")),
    ("explicit_beams_contact_big_aero", kernel!("explicit_beams_contact_big_aero")),
    ("explicit_beams_contact_big_aero_wide", kernel!("explicit_beams_contact_big_aero_wide")),
];
/// The twelve frame and contact modules (not the large groups' four, which a fleet of
/// more than four without coupling falls back to as shipped) with surface contact but
/// without the World's coupling (`EB_NO_COUPLING`, name `<module>_nocpl`): the surface pass without its
/// proxy paths (the per-element proxy loops, the casts, the pair path), which no
/// upload without coupling room can reach. The host runs them for such an upload
/// with surface contact (`explicit_beams_gpu::frame_module`); what executes is the
/// same code, and a browser compiles a kernel a third the size.
pub const EXPLICIT_BEAMS_NOCPL: [(&str, Kernel); 12] = [
    ("explicit_beams_nocpl", kernel!("explicit_beams_nocpl")),
    ("explicit_beams_wide_nocpl", kernel!("explicit_beams_wide_nocpl")),
    ("explicit_beams_aero_nocpl", kernel!("explicit_beams_aero_nocpl")),
    ("explicit_beams_aero_wide_nocpl", kernel!("explicit_beams_aero_wide_nocpl")),
    ("explicit_beams_solo_nocpl", kernel!("explicit_beams_solo_nocpl")),
    ("explicit_beams_solo_wide_nocpl", kernel!("explicit_beams_solo_wide_nocpl")),
    ("explicit_beams_solo_aero_nocpl", kernel!("explicit_beams_solo_aero_nocpl")),
    ("explicit_beams_solo_aero_wide_nocpl", kernel!("explicit_beams_solo_aero_wide_nocpl")),
    ("explicit_beams_contact_nocpl", kernel!("explicit_beams_contact_nocpl")),
    ("explicit_beams_contact_wide_nocpl", kernel!("explicit_beams_contact_wide_nocpl")),
    ("explicit_beams_contact_aero_nocpl", kernel!("explicit_beams_contact_aero_nocpl")),
    ("explicit_beams_contact_aero_wide_nocpl", kernel!("explicit_beams_contact_aero_wide_nocpl")),
];
/// The sixteen frame and contact modules without surface contact (`EB_NO_SURFACE`,
/// name `<module>_nosurf`; `engine_exec::explicit_plan::Kernel::module_for`): the host
/// runs them when no vehicle of the upload has surface contact.
pub const EXPLICIT_BEAMS_NOSURF: [(&str, Kernel); 16] = [
    ("explicit_beams_nosurf", kernel!("explicit_beams_nosurf")),
    ("explicit_beams_wide_nosurf", kernel!("explicit_beams_wide_nosurf")),
    ("explicit_beams_aero_nosurf", kernel!("explicit_beams_aero_nosurf")),
    ("explicit_beams_aero_wide_nosurf", kernel!("explicit_beams_aero_wide_nosurf")),
    ("explicit_beams_solo_nosurf", kernel!("explicit_beams_solo_nosurf")),
    ("explicit_beams_solo_wide_nosurf", kernel!("explicit_beams_solo_wide_nosurf")),
    ("explicit_beams_solo_aero_nosurf", kernel!("explicit_beams_solo_aero_nosurf")),
    ("explicit_beams_solo_aero_wide_nosurf", kernel!("explicit_beams_solo_aero_wide_nosurf")),
    ("explicit_beams_contact_nosurf", kernel!("explicit_beams_contact_nosurf")),
    ("explicit_beams_contact_wide_nosurf", kernel!("explicit_beams_contact_wide_nosurf")),
    ("explicit_beams_contact_aero_nosurf", kernel!("explicit_beams_contact_aero_nosurf")),
    ("explicit_beams_contact_aero_wide_nosurf", kernel!("explicit_beams_contact_aero_wide_nosurf")),
    ("explicit_beams_contact_big_nosurf", kernel!("explicit_beams_contact_big_nosurf")),
    ("explicit_beams_contact_big_wide_nosurf", kernel!("explicit_beams_contact_big_wide_nosurf")),
    ("explicit_beams_contact_big_aero_nosurf", kernel!("explicit_beams_contact_big_aero_nosurf")),
    ("explicit_beams_contact_big_aero_wide_nosurf", kernel!("explicit_beams_contact_big_aero_wide_nosurf")),
];
/// Surface contact split from the frame (`engine_exec::explicit_plan::Kernel::FrameSeg`,
/// `Kernel::Surf`): the frame in segments at its four variants (`EB_FRAME_SEG`), then the
/// surface pass's pipelines (`EB_SURF_PIPE`, `SURF_PASSES` order).
pub const EXPLICIT_BEAMS_SURF_SPLIT: [(&str, Kernel); 13] = [
    ("explicit_beams_seg", kernel!("explicit_beams_seg")),
    ("explicit_beams_wide_seg", kernel!("explicit_beams_wide_seg")),
    ("explicit_beams_aero_seg", kernel!("explicit_beams_aero_seg")),
    ("explicit_beams_aero_wide_seg", kernel!("explicit_beams_aero_wide_seg")),
    ("explicit_surface_prep", kernel!("explicit_surface_prep")),
    ("explicit_surface_find_tri", kernel!("explicit_surface_find_tri")),
    ("explicit_surface_find_seg", kernel!("explicit_surface_find_seg")),
    ("explicit_surface_find_point", kernel!("explicit_surface_find_point")),
    ("explicit_surface_find_pair", kernel!("explicit_surface_find_pair")),
    ("explicit_surface_apply_tri", kernel!("explicit_surface_apply_tri")),
    ("explicit_surface_apply_seg", kernel!("explicit_surface_apply_seg")),
    ("explicit_surface_apply_point", kernel!("explicit_surface_apply_point")),
    ("explicit_surface_apply_pair", kernel!("explicit_surface_apply_pair")),
];
/// The split group path (`EB_VV_SPLIT`, engine/docs/measurements/contact-groups-design.md):
/// per frame-module variant (plain, wide, aero, aero wide) the members' step
/// (`explicit_beams_split_member`), the groups' pass (`explicit_beams_split_group`) and
/// its triangle tests across workgroups (`explicit_beams_split_tests`); and the pass's
/// front as fleet-wide dispatches (`explicit_beams_fleet_front`, `explicit_beams_fleet_plan`:
/// explicit_beams_fleet.slang, contact-groups-design.md §8); and the members' raycast
/// wheels on each other (`explicit_beams_split_wheels`).
/// WebGPU only (native wgpu and browsers).
pub const EXPLICIT_BEAMS_SPLIT_GROUPS: [(&str, Kernel); 44] = [
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
/// Self-collision's listed pairs across workgroups (`EB_SELF_SPLIT`,
/// engine/docs/measurements/selfcol-design.md): per frame-module variant the solo frame's
/// step cut at its tests (`explicit_beams_self_step`), the walk dispatch
/// (`explicit_beams_self_walk`) and the pairs dispatch (`explicit_beams_self_pairs`);
/// with vehicle contact (`_vv`) the vehicles' dispatch of the split group path
/// (`explicit_beams_split_member`, a solo vehicle's step cut at its tests) and the walk
/// and pairs that skip a contact group's members. WebGPU only (native wgpu and browsers).
/// The wide passes' reused pair list (`explicit_beams_wide_list`, the self split's family
/// and bindings): the list's part of a step, a workgroup a vehicle, at 256 and 512 lanes.
pub const EXPLICIT_BEAMS_WIDE_LIST: [(&str, Kernel); 2] = [
    ("explicit_beams_selfsplit_list", kernel!("explicit_beams_selfsplit_list")),
    ("explicit_beams_selfsplit_list_wide", kernel!("explicit_beams_selfsplit_list_wide")),
];
pub const EXPLICIT_BEAMS_SELF_SPLIT: [(&str, Kernel); 24] = [
    ("explicit_beams_selfsplit_step", kernel!("explicit_beams_selfsplit_step")),
    ("explicit_beams_selfsplit_walk", kernel!("explicit_beams_selfsplit_walk")),
    ("explicit_beams_selfsplit_pairs", kernel!("explicit_beams_selfsplit_pairs")),
    ("explicit_beams_selfsplit_step_wide", kernel!("explicit_beams_selfsplit_step_wide")),
    ("explicit_beams_selfsplit_walk_wide", kernel!("explicit_beams_selfsplit_walk_wide")),
    ("explicit_beams_selfsplit_pairs_wide", kernel!("explicit_beams_selfsplit_pairs_wide")),
    ("explicit_beams_selfsplit_step_aero", kernel!("explicit_beams_selfsplit_step_aero")),
    ("explicit_beams_selfsplit_walk_aero", kernel!("explicit_beams_selfsplit_walk_aero")),
    ("explicit_beams_selfsplit_pairs_aero", kernel!("explicit_beams_selfsplit_pairs_aero")),
    ("explicit_beams_selfsplit_step_aero_wide", kernel!("explicit_beams_selfsplit_step_aero_wide")),
    ("explicit_beams_selfsplit_walk_aero_wide", kernel!("explicit_beams_selfsplit_walk_aero_wide")),
    ("explicit_beams_selfsplit_pairs_aero_wide", kernel!("explicit_beams_selfsplit_pairs_aero_wide")),

    ("explicit_beams_selfsplit_member_vv", kernel!("explicit_beams_selfsplit_member_vv")),
    ("explicit_beams_selfsplit_walk_vv", kernel!("explicit_beams_selfsplit_walk_vv")),
    ("explicit_beams_selfsplit_pairs_vv", kernel!("explicit_beams_selfsplit_pairs_vv")),
    ("explicit_beams_selfsplit_member_vv_wide", kernel!("explicit_beams_selfsplit_member_vv_wide")),
    ("explicit_beams_selfsplit_walk_vv_wide", kernel!("explicit_beams_selfsplit_walk_vv_wide")),
    ("explicit_beams_selfsplit_pairs_vv_wide", kernel!("explicit_beams_selfsplit_pairs_vv_wide")),
    ("explicit_beams_selfsplit_member_vv_aero", kernel!("explicit_beams_selfsplit_member_vv_aero")),
    ("explicit_beams_selfsplit_walk_vv_aero", kernel!("explicit_beams_selfsplit_walk_vv_aero")),
    ("explicit_beams_selfsplit_pairs_vv_aero", kernel!("explicit_beams_selfsplit_pairs_vv_aero")),
    ("explicit_beams_selfsplit_member_vv_aero_wide", kernel!("explicit_beams_selfsplit_member_vv_aero_wide")),
    ("explicit_beams_selfsplit_walk_vv_aero_wide", kernel!("explicit_beams_selfsplit_walk_vv_aero_wide")),
    ("explicit_beams_selfsplit_pairs_vv_aero_wide", kernel!("explicit_beams_selfsplit_pairs_vv_aero_wide")),
];
/// The build's WGSL directory: OUT_DIR's, or with the kernels compiled at run time the
/// store's ([`kernel_store`]). Native tests and tools read modules from it.
#[doc(hidden)]
#[cfg(not(target_arch = "wasm32"))]
pub fn explicit_beams_wgsl_dir() -> std::path::PathBuf {
    #[cfg(kernels_at_runtime)]
    {
        kernel_store::dir().join("wgsl")
    }
    #[cfg(not(kernels_at_runtime))]
    {
        std::path::PathBuf::from(concat!(env!("OUT_DIR"), "/wgsl"))
    }
}

/// The explicit-beams wide passes (entry, WGSL), one module each: one thread per
/// element over every vehicle, a dispatch per pass ([`ExplicitBeamsGpu::set_wide`]).
pub const EXPLICIT_BEAMS_WIDE_PASSES: [(&str, Kernel); 11] = [
    ("explicit_beams_wide_begin", kernel!("explicit_beams_wide_begin")),
    ("explicit_beams_wide_rebase", kernel!("explicit_beams_wide_rebase")),
    ("explicit_beams_wide_nodes", kernel!("explicit_beams_wide_nodes")),
    ("explicit_beams_wide_beams", kernel!("explicit_beams_wide_beams")),
    ("explicit_beams_wide_tris", kernel!("explicit_beams_wide_tris")),
    ("explicit_beams_wide_end", kernel!("explicit_beams_wide_end")),
    ("explicit_beams_wide_close", kernel!("explicit_beams_wide_close")),
    ("explicit_beams_wide_hits", kernel!("explicit_beams_wide_hits")),
    ("explicit_beams_wide_centred_beams", kernel!("explicit_beams_wide_centred_beams")),
    ("explicit_beams_wide_centred_nodes", kernel!("explicit_beams_wide_centred_nodes")),
    ("explicit_beams_wide_aux", kernel!("explicit_beams_wide_aux")),
];

/// [`EXPLICIT_BEAMS_WGSL`] with 512 lanes per vehicle, for large vehicles.
pub const EXPLICIT_BEAMS_WIDE_WGSL: Kernel = kernel!("explicit_beams_wide");
/// [`EXPLICIT_BEAMS_WGSL`] with wings, the fuselage drag and command beams
/// (`EB_AERO`), for fleets that have them; 256 and 512 lanes.
pub const EXPLICIT_BEAMS_AERO_WGSL: Kernel = kernel!("explicit_beams_aero");
pub const EXPLICIT_BEAMS_AERO_WIDE_WGSL: Kernel = kernel!("explicit_beams_aero_wide");
/// Vehicle–vehicle contact of the explicit-beams solver: the per-frame plan passes
/// (entry, WGSL) and the contact groups' entry `explicit_beams_contact` at the four
/// frame module variants (256 lanes, 512, with `EB_AERO`, both).
pub const EXPLICIT_BEAMS_VV_PASSES: [(&str, Kernel); 3] = [
    ("explicit_beams_vv_bounds", kernel!("explicit_beams_vv_bounds")),
    ("explicit_beams_vv_pairs", kernel!("explicit_beams_vv_pairs")),
    ("explicit_beams_vv_group", kernel!("explicit_beams_vv_group")),
];
/// Vehicle sleep's awake list (`engine_exec::explicit_plan::SLEEP_LIST`).
pub const EXPLICIT_BEAMS_SLEEP_LIST_WGSL: Kernel = kernel!("explicit_beams_sleep_list");
pub const EXPLICIT_BEAMS_CONTACT_WGSL: Kernel = kernel!("explicit_beams_contact");
pub const EXPLICIT_BEAMS_CONTACT_WIDE_WGSL: Kernel = kernel!("explicit_beams_contact_wide");
pub const EXPLICIT_BEAMS_CONTACT_AERO_WGSL: Kernel = kernel!("explicit_beams_contact_aero");
pub const EXPLICIT_BEAMS_CONTACT_AERO_WIDE_WGSL: Kernel = kernel!("explicit_beams_contact_aero_wide");
/// The frame entry for the vehicles stepped alone while contact is active
/// (`EB_VV_SOLO`: the vehicle from the solo list, the frame bounds check), at the
/// four variants; [`EXPLICIT_BEAMS_WGSL`] and the others stay the plain frame.
pub const EXPLICIT_BEAMS_SOLO_WGSL: Kernel = kernel!("explicit_beams_solo");
pub const EXPLICIT_BEAMS_SOLO_WIDE_WGSL: Kernel = kernel!("explicit_beams_solo_wide");
pub const EXPLICIT_BEAMS_SOLO_AERO_WGSL: Kernel = kernel!("explicit_beams_solo_aero");
pub const EXPLICIT_BEAMS_SOLO_AERO_WIDE_WGSL: Kernel = kernel!("explicit_beams_solo_aero_wide");
/// A buffer's extent for compact snapshots (`slang/kernels/state_extent.slang`,
/// entry `state_extent`; `WgpuSim::measure_state`).
pub const STATE_EXTENT_WGSL: Kernel = kernel!("state_extent");

/// The MPM material probe (`slang/kernels/mpm_material_probe.slang`, entry
/// `mpm_material_probe`): runs `physics/mpm_material.slang` on test cases so
/// `tests/mpm_material.rs` can compare it with the engine-terrain CPU mirror.
pub const MPM_MATERIAL_PROBE_WGSL: Kernel = kernel!("mpm_material_probe");

use engine_exec::{ExecError, PhaseTimingsMs, PhysicsBackend};
use engine_schema::{AvbdParams, StepCounters, WorldCapacity};
#[cfg(feature = "gpu")]
use std::borrow::Cow;

pub struct WebGpuWorld {
    capacity: WorldCapacity,
    params: AvbdParams,
    last_counters: StepCounters,
    last_timings: PhaseTimingsMs,
    pub wgsl_shaders: Vec<(String, String)>,
}

#[cfg(not(kernels_at_runtime))]
mod direct {
    include!(concat!(env!("OUT_DIR"), "/direct_wgsl.rs"));
}

fn spirv_entry_names() -> Vec<&'static str> {
    // Generated at build time (or by the run-time compile, kernel_store).
    #[cfg(not(kernels_at_runtime))]
    let entries = include_str!(concat!(env!("OUT_DIR"), "/spirv_entries.txt"));
    #[cfg(kernels_at_runtime)]
    let entries = kernel_store::spirv_entries();
    entries
        .lines()
        .filter(|l| !l.is_empty())
        .map(|s| {
            // Leak is fine — process lifetime, one-time at first call via OnceLock below.
            Box::leak(s.to_string().into_boxed_str()) as &'static str
        })
        .collect()
}

fn entry_list() -> &'static [&'static str] {
    use std::sync::OnceLock;
    static LIST: OnceLock<Vec<&'static str>> = OnceLock::new();
    LIST.get_or_init(spirv_entry_names).as_slice()
}

impl WebGpuWorld {
    pub fn spirv_bytes(name: &str) -> anyhow::Result<Vec<u8>> {
        #[cfg(any(not(kernels_at_runtime), target_arch = "wasm32"))]
        let path = format!("{}/spirv/{name}.spv", env!("OUT_DIR"));
        #[cfg(all(kernels_at_runtime, not(target_arch = "wasm32")))]
        let path = format!("{}/spirv/{name}.spv", kernel_store::dir().display());
        Ok(std::fs::read(&path).map_err(|e| anyhow::anyhow!("{path}: {e}"))?)
    }

    pub fn entry_names() -> &'static [&'static str] {
        entry_list()
    }

    /// WGSL for every entry: `(name, wgsl)` or `(name, err)`.
    pub fn compile_shaders_report() -> Vec<(String, Result<String, String>)> {
        if std::env::var_os("WG_NAGA").is_some() {
            return Self::naga_shaders_report();
        }
        Self::direct_shaders()
            .iter()
            .map(|(n, w)| {
                (
                    (*n).to_string(),
                    // No entry uses storage textures. Slang over-lowers
                    // DeviceMemoryBarrierWithGroupSync to a texture barrier
                    // unavailable in the supported WGSL/Naga version, and ends it
                    // with an empty statement (`;;`) that Tint accepts but
                    // Safari's WGSL parser rejects.
                    Ok(uniformize_workgroup_lanes(&cap_loop_iterations(
                        &w.replace(
                            "storageBarrier(); textureBarrier(); workgroupBarrier();;",
                            "storageBarrier(); workgroupBarrier();",
                        )
                        .replace(
                            "storageBarrier(); textureBarrier(); workgroupBarrier();",
                            "storageBarrier(); workgroupBarrier();",
                        ),
                    ))),
                )
            })
            .collect()
    }

    /// Build-time Slang → WGSL modules (embedded; works on wasm32).
    pub fn direct_shaders() -> &'static [(&'static str, &'static str)] {
        #[cfg(not(kernels_at_runtime))]
        {
            direct::DIRECT_WGSL
        }
        #[cfg(kernels_at_runtime)]
        {
            kernel_store::direct()
        }
    }

    /// Translate every SPIR-V entry to WGSL through naga.
    pub fn naga_shaders_report() -> Vec<(String, Result<String, String>)> {
        let mut out = Vec::new();
        for name in entry_list() {
            let result = (|| -> anyhow::Result<String> {
                let bytes = Self::spirv_bytes(name)?;
                let module = naga::front::spv::parse_u8_slice(
                    &bytes,
                    &naga::front::spv::Options::default(),
                )?;
                let mut validator = naga::valid::Validator::new(
                    naga::valid::ValidationFlags::empty(),
                    naga::valid::Capabilities::all(),
                );
                let info = validator.validate(&module)?;
                let wgsl = naga::back::wgsl::write_string(
                    &module,
                    &info,
                    naga::back::wgsl::WriterFlags::empty(),
                )?;
                // naga puts the for-loop test `break` inside `switch 0i`, so it
                // exits the switch and the loop increments forever (GPU hang).
                Ok(cap_loop_iterations(&fix_naga_loop_exits(&wgsl)))
            })();
            out.push(((*name).to_string(), result.map_err(|e| format!("{e:#}"))));
        }
        out
    }

    pub fn compile_shaders() -> anyhow::Result<Vec<(String, String)>> {
        let mut ok = Vec::new();
        let mut errs = Vec::new();
        for (name, r) in Self::compile_shaders_report() {
            match r {
                Ok(wgsl) => ok.push((name, wgsl)),
                Err(e) => errs.push(format!("{name}: {e}")),
            }
        }
        if !errs.is_empty() {
            anyhow::bail!(
                "SPIR-V→WGSL failed for {}/{} entries:\n{}",
                errs.len(),
                ok.len() + errs.len(),
                errs.join("\n")
            );
        }
        Ok(ok)
    }

    pub fn new(capacity: WorldCapacity) -> anyhow::Result<Self> {
        let wgsl_shaders = Self::compile_shaders()?;
        Ok(Self {
            capacity,
            params: AvbdParams::default(),
            last_counters: StepCounters::default(),
            last_timings: PhaseTimingsMs::default(),
            wgsl_shaders,
        })
    }

    #[cfg(feature = "gpu")]
    /// Create a wgpu device and load every successfully translated WGSL module.
    /// Returns adapter label + counts (ok / fail).
    pub fn smoke_device() -> anyhow::Result<String> {
        let (info, device, _queue) = crate::gpu::default_device()?;

        let report = Self::compile_shaders_report();
        let mut loaded = 0u32;
        let mut failed = 0u32;
        let mut fail_names = Vec::new();
        for (name, r) in &report {
            match r {
                Ok(wgsl) => {
                    let _ = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(name),
                        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl)),
                    });
                    loaded += 1;
                }
                Err(_) => {
                    failed += 1;
                    fail_names.push(name.clone());
                }
            }
        }
        Ok(format!(
            "{} / {} — WGSL loaded={loaded} failed={failed}{}",
            info.name,
            info.backend,
            if fail_names.is_empty() {
                String::new()
            } else {
                format!(" ({})", fail_names.join(", "))
            }
        ))
    }
}

impl PhysicsBackend for WebGpuWorld {
    fn name(&self) -> &'static str {
        "webgpu"
    }
    fn capacity(&self) -> &WorldCapacity {
        &self.capacity
    }
    fn params(&self) -> &AvbdParams {
        &self.params
    }
    fn set_params(&mut self, params: AvbdParams) {
        self.params = params;
    }
    fn step(&mut self) -> Result<StepCounters, ExecError> {
        Err(ExecError::Backend(
            "WebGPU full step: Slang→WGSL modules ready; Chrome bench harness next \
             (fair bar vs three-avbd WebGPU — not native Metal scene_fps)"
                .into(),
        ))
    }
    fn last_timings(&self) -> PhaseTimingsMs {
        self.last_timings.clone()
    }
    fn last_counters(&self) -> StepCounters {
        self.last_counters
    }
}

/// `loop { switch 0i { default { if (i < n) {} else { break; } ... } } i++; continue; }`
/// never terminates: that `break` leaves the switch. Turn the first else-break into a
/// real loop exit.
fn fix_naga_loop_exits(src: &str) -> String {
    let mut sites: Vec<(usize, usize, usize)> = Vec::new();
    let mut cursor = 0;
    while let Some(rel) = src[cursor..].find("switch 0i") {
        let switch_at = cursor + rel;
        let before = src[..switch_at].trim_end();
        if !before.ends_with("loop {") {
            cursor = switch_at + "switch 0i".len();
            continue;
        }
        let Some(open_rel) = src[switch_at..].find('{') else {
            break;
        };
        let switch_open = switch_at + open_rel;
        let Some(switch_close) = matching_brace(src, switch_open) else {
            break;
        };
        let default_at = match src[switch_at..switch_close].find("default:") {
            Some(p) => switch_at + p,
            None => {
                cursor = switch_close;
                continue;
            }
        };
        let Some(def_rel) = src[default_at..].find('{') else {
            break;
        };
        let def_open = default_at + def_rel;
        let Some(def_close) = matching_brace(src, def_open) else {
            break;
        };
        let body = &src[def_open + 1..def_close];
        let Some(br) = first_else_break(body) else {
            cursor = switch_close;
            continue;
        };
        let loop_open = before.len() - 1; // the '{' of `loop {`
        sites.push((loop_open, def_open + 1 + br, switch_close));
        cursor = def_open + 1;
    }
    let mut edits: Vec<(usize, String)> = Vec::new();
    for (i, &(loop_open, break_at, switch_close)) in sites.iter().enumerate() {
        let flag = format!("nagaExit{i}");
        edits.push((switch_close + 1, format!("\n if ({flag}) {{ break; }}\n")));
        edits.push((break_at, format!("{flag} = true; ")));
        edits.push((loop_open + 1, format!("\n var {flag} = false;\n")));
    }
    edits.sort_by(|a, b| b.0.cmp(&a.0));
    let mut s = src.to_string();
    for (at, text) in edits {
        s.insert_str(at, &text);
    }
    s
}

include!("wgsl_loop_cap.rs");
include!("explicit_post.rs");

/// The test-only explicit modules' WGSL (the frozen baseline and reference, the
/// deterministic-hits and ablation builds: `engine_kernel_manifest::explicit::test_modules`),
/// compiled on the first call in a process (each compile cached by what it depends on,
/// engine-slang-build) and post-processed as the shipped ones are, into
/// `OUT_DIR/test-wgsl`. No build compiles them: only the tests and measurements that run
/// them. Native only (it runs slangc).
#[cfg(not(target_arch = "wasm32"))]
pub fn test_kernels_dir() -> &'static std::path::Path {
    static DIR: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        compile_test_kernels().unwrap_or_else(|e| panic!("the test-only explicit kernels: {e}"))
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn compile_test_kernels() -> Result<std::path::PathBuf, String> {
    use engine_kernel_manifest::explicit::{test_modules, TestSource};
    use std::path::{Path, PathBuf};
    let slang = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../slang/kernels");
    let dir = PathBuf::from(concat!(env!("OUT_DIR"), "/test-wgsl"));
    // (Compiled beside it and renamed into it, one file at a time: another process may
    // read or write the same directory.)
    let staging = dir.join(format!(".staging-{}", std::process::id()));
    std::fs::create_dir_all(&staging).map_err(|e| e.to_string())?;
    let modules = test_modules();
    // The shipped source (a measurement build may name another, as build.rs does).
    let shipped = match std::env::var("RUBBLE_EB_SRC") {
        Ok(f) => PathBuf::from(f),
        Err(_) if std::env::var("RUBBLE_EB_REF").is_ok_and(|v| v == "1") => slang.join("explicit_beams_ref.slang"),
        Err(_) => slang.join("explicit_beams.slang"),
    };
    let sources: Vec<PathBuf> = modules
        .iter()
        .map(|m| match m.source {
            TestSource::File(f) => slang.join(f),
            TestSource::Shipped => shipped.clone(),
        })
        .collect();
    let outs: Vec<PathBuf> = modules.iter().map(|m| staging.join(format!("{}.wgsl", m.name))).collect();
    let jobs: Vec<engine_slang_build::Compile> = modules
        .iter()
        .enumerate()
        .map(|(k, m)| engine_slang_build::Compile {
            source: &sources[k],
            entry: &m.entry,
            stage: "compute",
            target: engine_slang_build::SlangTarget::Wgsl,
            out: &outs[k],
            include_dirs: &[],
            defines: &m.defines,
        })
        .collect();
    for (m, r) in modules.iter().zip(engine_slang_build::compile_many(&jobs)) {
        r.map_err(|e| format!("{}: {e}", m.name))?;
    }
    for (m, out) in modules.iter().zip(&outs) {
        let name = format!("{}.wgsl", m.name);
        let text = std::fs::read_to_string(out).map_err(|e| format!("{name}: {e}"))?;
        std::fs::write(out, explicit_post_text(&name, &text)?).map_err(|e| e.to_string())?;
        std::fs::rename(out, dir.join(&name)).map_err(|e| e.to_string())?;
    }
    let _ = std::fs::remove_dir_all(&staging);
    Ok(dir)
}

/// An explicit module's WGSL file: the build's ([`explicit_beams_wgsl_dir`]), else a
/// test-only module's ([`test_kernels_dir`]).
#[cfg(not(target_arch = "wasm32"))]
pub fn explicit_wgsl_path(name: &str) -> std::path::PathBuf {
    let built = explicit_beams_wgsl_dir().join(format!("{name}.wgsl"));
    if built.is_file() {
        return built;
    }
    test_kernels_dir().join(format!("{name}.wgsl"))
}

/// Chrome treats a storage/workgroup load as non-uniform, so a later barrier
/// inside `if (lanes == 1)` is rejected. `workgroupUniformLoad` makes it uniform.
pub fn uniformize_workgroup_lanes(src: &str) -> String {
    const NAMES: [&str; 8] = ["gsUniformLanes_0", "gsUniformHits_0", "gsUniformOpen_0", "gsUniformCouple_0", "gsUniformDrive_0", "gsUniformSurf_0", "gsUniformCpl_0", "gsUniformCplKeep_0"];
    let mut out = src.to_string();
    for name in NAMES {
        if !out.contains(name) {
            continue;
        }
        let mut next = String::with_capacity(out.len() + 256);
        for line in out.lines() {
            if line.contains("var<workgroup>") && line.contains(name) {
                next.push_str(line);
            } else {
                next.push_str(
                    &line
                        .replace(name, &format!("workgroupUniformLoad(&{name})"))
                        // Restore stores only: `= ` must not also match a `==` read.
                        .replace(
                            &format!("workgroupUniformLoad(&{name}) = "),
                            &format!("{name} = "),
                        ),
                );
            }
            next.push('\n');
        }
        out = next;
    }
    out
}

fn first_else_break(body: &str) -> Option<usize> {
    let mut i = 0;
    let b = body.as_bytes();
    while i + 4 < b.len() {
        if b[i..].starts_with(b"else") && (i == 0 || !b[i - 1].is_ascii_alphanumeric()) {
            let mut j = i + 4;
            while j < b.len() && b[j].is_ascii_whitespace() {
                j += 1;
            }
            if j < b.len() && b[j] == b'{' {
                j += 1;
                while j < b.len() && b[j].is_ascii_whitespace() {
                    j += 1;
                }
                if b[j..].starts_with(b"break") {
                    return Some(j);
                }
            }
        }
        i += 1;
    }
    None
}

fn matching_brace(src: &str, open: usize) -> Option<usize> {
    let b = src.as_bytes();
    let mut depth = 0i32;
    let mut i = open;
    while i < b.len() {
        match b[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// `loop` statements in WGSL (the keyword before a block, as `cap_loop_iterations` finds them).
#[cfg(test)]
fn loop_statements(src: &str) -> usize {
    let b = src.as_bytes();
    let mut n = 0;
    let mut i = 0;
    while let Some(rel) = src[i..].find("loop") {
        let at = i + rel;
        let after = at + 4;
        let word = (at == 0 || !(b[at - 1].is_ascii_alphanumeric() || b[at - 1] == b'_'))
            && (after >= b.len() || !(b[after].is_ascii_alphanumeric() || b[after] == b'_'));
        let mut j = after;
        while j < b.len() && b[j].is_ascii_whitespace() {
            j += 1;
        }
        if word && j < b.len() && b[j] == b'{' {
            n += 1;
        }
        i = after;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_wgsl_loop_is_capped_and_valid() {
        for (name, res) in WebGpuWorld::naga_shaders_report() {
            let wgsl = res.unwrap_or_else(|e| panic!("{name}: {e}"));
            let loops = wgsl.matches("loop {").count();
            let guards = wgsl.matches("loopGuard").count() / 3;
            assert_eq!(loops, guards, "{name}: uncapped loop");
            let module = naga::front::wgsl::parse_str(&wgsl)
                .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&wgsl)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        }
    }

    #[test]
    fn direct_wgsl_covers_every_entry_and_validates() {
        let direct = WebGpuWorld::direct_shaders();
        for name in WebGpuWorld::entry_names()
            .iter()
            .filter(|n| **n != "abi_roundtrip")
        {
            assert!(
                direct.iter().any(|(n, _)| n == name),
                "{name}: missing direct WGSL"
            );
        }
        for (name, res) in WebGpuWorld::compile_shaders_report() {
            let wgsl = &res.unwrap();
            let loops = direct.iter().find(|(n, _)| *n == name).unwrap().1;
            // `loop` statements, not the word inside names (solve_primal_sparse_loop).
            let raw = loop_statements(loops);
            assert_eq!(
                raw,
                wgsl.matches("loopGuard").count() / 3,
                "{name}: uncapped loop"
            );
            let module = naga::front::wgsl::parse_str(wgsl)
                .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(wgsl)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        }
    }

    /// Chrome exposes 10 storage buffers per compute stage. Every entry must fit,
    /// which is why convex work is split out of `narrowphase`; the one exception is the
    /// solve loop on canonical and compact contacts (11), which `WgpuSim` builds only on
    /// a device that allows that many (the step keeps its per-colour dispatches
    /// otherwise). Its persistent-contact variant, the web demos' contacts, fits.
    #[test]
    fn every_entry_fits_the_webgpu_storage_buffer_budget() {
        for (name, res) in WebGpuWorld::compile_shaders_report() {
            let wgsl = res.unwrap();
            let storage = wgsl.matches("var<storage").count();
            let guarded = name == "solve_primal_sparse_loop" || name == "solve_primal_sparse_loop_compact";
            if guarded {
                assert!(storage <= 11, "{name}: {storage} storage buffers > 11");
                continue;
            }
            assert!(storage <= 10, "{name}: {storage} storage buffers > 10");
        }
    }

    #[test]
    fn compact_primal_reads_64_byte_points_without_canonical_contact_binding() {
        for name in ["solve_primal_color_compact", "solve_primal_one_compact"] {
            let source = WebGpuWorld::compile_shaders_report()
                .into_iter()
                .find(|(n, _)| n == name)
                .expect("compact entry must be exported")
                .1
                .unwrap();
            let module = naga::front::wgsl::parse_str(&source).unwrap();
            let globals: Vec<_> = module.global_variables.iter().map(|(_, g)| g).collect();
            assert!(!globals.iter().any(|g| g.binding.as_ref().is_some_and(|b| b.binding == 4)), "{name}: compact primal still reads canonical 128-byte contacts; hot-path traffic reduction was not compiled");
            let points = globals
                .iter()
                .find(|g| g.binding.as_ref().is_some_and(|b| b.binding == 27))
                .expect("compact point binding missing");
            let naga::TypeInner::Array { base, stride, .. } = module.types[points.ty].inner else {
                panic!("compact points must be an array");
            };
            assert_eq!(stride,64,"{name}: solver point stride changed; host allocations and bandwidth contract require 64 bytes");
            let naga::TypeInner::Struct { ref members, span } = module.types[base].inner else {
                panic!("compact points must be structs");
            };
            assert_eq!(span, 64);
            assert_eq!(
                members.iter().map(|m| m.offset).collect::<Vec<_>>(),
                vec![0, 12, 16, 28, 32, 44, 48, 60],
                "{name}: compact field offsets differ from the packing ABI"
            );
            assert!(
                globals
                    .iter()
                    .filter(|g| matches!(g.space, naga::AddressSpace::Storage { .. }))
                    .count()
                    <= 10,
                "{name}: exceeds browser storage-buffer budget"
            );
        }
    }

    #[test]
    fn compact_solvers_use_cached_body_pairs_without_collider_indirection() {
        for name in [
            "solve_primal_color_compact",
            "solve_primal_one_compact",
            "solve_dual_compact",
        ] {
            let source = WebGpuWorld::compile_shaders_report()
                .into_iter()
                .find(|(n, _)| n == name)
                .expect("compact entry must be exported")
                .1
                .unwrap();
            let module = naga::front::wgsl::parse_str(&source).unwrap();
            let globals: Vec<_> = module.global_variables.iter().map(|(_, g)| g).collect();
            for (binding, resource) in [(2, "colliders"), (25, "canonical manifolds")] {
                assert!(!globals.iter().any(|g| g.binding.as_ref().is_some_and(|b| b.binding == binding)),
                    "{name}: still reads {resource}; cached solver pairs must remove collider-to-body lookups from every solver iteration");
            }
            let pairs = globals
                .iter()
                .find(|g| g.binding.as_ref().is_some_and(|b| b.binding == 28))
                .expect("compact solver pair binding missing");
            let naga::TypeInner::Array { base, stride, .. } = module.types[pairs.ty].inner else {
                panic!("{name}: solver pairs must be an array");
            };
            assert_eq!(
                stride, 32,
                "{name}: solver pair stride differs from host allocation"
            );
            let naga::TypeInner::Struct { ref members, span } = module.types[base].inner else {
                panic!("{name}: solver pairs must be structs");
            };
            assert_eq!(span, 32);
            assert_eq!(members.iter().map(|m| m.offset).collect::<Vec<_>>(), vec![0,4,8,12,16,28],
                "{name}: solver pair body IDs, point range, sphere flag or normal/friction offsets changed");
        }
    }

    #[test]
    fn persistent_contact_kernels_keep_hot_cold_layouts_and_omit_copyback() {
        for name in [
            "narrowphase_persistent",
            "contact_hash_insert_persistent",
            "contact_hash_insert_manifolds_persistent",
            "persist_contacts_persistent",
            "persist_manifolds_persistent",
            "solve_primal_color_persistent",
            "solve_dual_persistent",
        ] {
            let source = WebGpuWorld::compile_shaders_report()
                .into_iter()
                .find(|(n, _)| n == name)
                .expect("persistent entry must be exported")
                .1
                .unwrap();
            let module = naga::front::wgsl::parse_str(&source).unwrap();
            let globals: Vec<_> = module.global_variables.iter().map(|(_, g)| g).collect();
            let solver = name.starts_with("solve_");
            for (binding, expected_stride) in [(4, 32), (27, 64)] {
                let view = globals
                    .iter()
                    .find(|g| g.binding.as_ref().is_some_and(|b| b.binding == binding));
                if solver && binding == 4 {
                    assert!(view.is_none(), "{name}: solver still accesses cold contact metadata/canonical copyback; persistent forces must remain in hot storage");
                    continue;
                }
                let view =
                    view.unwrap_or_else(|| panic!("{name}: missing contact binding {binding}"));
                let naga::TypeInner::Array { base, stride, .. } = module.types[view.ty].inner
                else {
                    panic!("{name}: contact storage must be an array")
                };
                assert_eq!(
                    stride, expected_stride,
                    "{name}: binding {binding} stride disagrees with host allocation"
                );
                if binding == 4 {
                    let naga::TypeInner::Struct { ref members, .. } = module.types[base].inner
                    else {
                        panic!("metadata must be a struct")
                    };
                    assert_eq!(
                        members.iter().map(|m| m.offset).collect::<Vec<_>>(),
                        vec![0, 4, 8, 12, 16, 28]
                    );
                    for field in [2, 3] {
                        assert!(matches!(module.types[members[field].ty].inner,naga::TypeInner::Scalar(naga::Scalar {kind:naga::ScalarKind::Uint,width:4})),
                            "{name}: feature/generation field {field} must retain all 32 unsigned bits independently of owner/reuse/stick flags");
                    }
                }
            }
            assert!(
                globals
                    .iter()
                    .filter(|g| matches!(g.space, naga::AddressSpace::Storage { .. }))
                    .count()
                    <= 10,
                "{name}: persistent layout exceeds the browser's ten storage-buffer budget"
            );
        }
    }

    #[test]
    fn collision_clipping_uses_pointer_arguments_and_eight_vertex_scratch() {
        for name in ["narrowphase"] {
            let source = WebGpuWorld::compile_shaders_report()
                .into_iter()
                .find(|(n, _)| n == name)
                .expect("collision entry must be exported")
                .1
                .unwrap();
            let module = naga::front::wgsl::parse_str(&source).expect("valid collision WGSL");
            let clip = module
                .functions
                .iter()
                .find(|(_, f)| {
                    f.name
                        .as_deref()
                        .is_some_and(|n| n.starts_with("boxClipPoly"))
                })
                .expect("clipping function must be present")
                .1;
            for index in [0, 4] {
                let naga::TypeInner::Pointer {
                    base,
                    space: naga::AddressSpace::Function,
                } = module.types[clip.arguments[index].ty].inner
                else {
                    panic!("{name} clip argument {index}: array passed by value; four clipping planes must share local scratch without copying whole polygons");
                };
                let naga::TypeInner::Array {
                    size: naga::ArraySize::Constant(size),
                    ..
                } = module.types[base].inner
                else {
                    panic!("{name}: clipping scratch must be a fixed vertex array");
                };
                assert_eq!(size.get(), 8,
                    "{name}: four half-plane clips of a convex quad need exactly eight slots; smaller drops support points, larger increases shader scratch");
            }
        }
    }

    #[test]
    fn production_contact_reuse_has_no_shared_diagnostic_counter_buffer() {
        for name in ["update_refs", "narrowphase"] {
            let shaders = WebGpuWorld::compile_shaders_report();
            let source = shaders
                .iter()
                .find(|(n, _)| n == name)
                .unwrap()
                .1
                .as_ref()
                .unwrap();
            let module = naga::front::wgsl::parse_str(source).expect("valid reuse shader");
            assert!(!module.global_variables.iter().any(|(_, global)|
                global.binding.as_ref().is_some_and(|b| b.group == 0 && b.binding == 21)),
                "{name}: production contact reuse binds the shared scratch/statistics buffer. The removed hit/miss/generation diagnostics serialized every body/pair on six global atomic counters; keep optional instrumentation out of the default performance path.");
        }
    }

    #[test]
    fn exported_wgsl_has_no_empty_statements() {
        // Safari's WGSL parser rejects `;;` ("Not a valid statement"); Tint accepts it.
        for (name, result) in WebGpuWorld::compile_shaders_report() {
            let source = result.unwrap();
            assert!(
                !source.contains(";;"),
                "{name}: empty statement `;;` breaks Safari"
            );
        }
    }

    #[test]
    fn sparse_color_barrier_lowering_preserves_buffer_visibility_and_has_no_textures() {
        // Slang emits a texture barrier alongside storage/workgroup barriers.
        // Removing it is legal ONLY while these entry points access no textures.
        for (name, result) in WebGpuWorld::compile_shaders_report() {
            if !name.starts_with("solve_primal_sparse") {
                continue;
            }
            let source = result.unwrap();
            let module = naga::front::wgsl::parse_str(&source).expect("valid sparse shader");
            assert!(!module.types.iter().any(|(_,ty)| matches!(ty.inner,
                naga::TypeInner::Image{..} | naga::TypeInner::Sampler{..})),
                "{name}: texture resources invalidate the buffer-only barrier lowering; retain the appropriate texture synchronization before adding textures");
            assert!(source.contains("storageBarrier(); workgroupBarrier();"),
                "{name}: later colors depend on earlier pose writes; preserve storage visibility AND workgroup execution synchronization");
            assert!(
                !source.contains("textureBarrier"),
                "{name}: unsupported Slang texture barrier escaped lowering"
            );
        }
    }

    #[test]
    fn primal_topology_uses_plain_loads_and_writes_only_pose_fields() {
        fn stores(block: &naga::Block, out: &mut Vec<naga::Handle<naga::Expression>>) {
            for statement in block.iter() {
                match statement {
                    naga::Statement::Store { pointer, .. } => out.push(*pointer),
                    naga::Statement::Block(b) => stores(b, out),
                    naga::Statement::If { accept, reject, .. } => {
                        stores(accept, out);
                        stores(reject, out);
                    }
                    naga::Statement::Switch { cases, .. } => {
                        for case in cases {
                            stores(&case.body, out);
                        }
                    }
                    naga::Statement::Loop {
                        body, continuing, ..
                    } => {
                        stores(body, out);
                        stores(continuing, out);
                    }
                    _ => {}
                }
            }
        }
        for (name, source) in WebGpuWorld::compile_shaders_report()
            .into_iter()
            .filter(|(name, _)| name.starts_with("solve_primal_"))
        {
            let source = source.unwrap();
            let module = naga::front::wgsl::parse_str(&source).expect("valid primal WGSL");
            for binding in if name.starts_with("solve_primal_color") {
                &[14, 17][..]
            } else {
                &[14][..]
            } {
                let views: Vec<_> = module
                    .global_variables
                    .iter()
                    .filter(|(_, g)| {
                        g.binding
                            .as_ref()
                            .is_some_and(|b| b.group == 0 && b.binding == *binding)
                    })
                    .collect();
                assert_eq!(views.len(), 1, "{name}: binding {binding} must retain exactly one topology view after entry-point elimination");
                let naga::TypeInner::Array { base, .. } = module.types[views[0].1.ty].inner else {
                    panic!("topology must remain a u32 array");
                };
                assert!(matches!(module.types[base].inner, naga::TypeInner::Scalar(_)),
                    "{name}: binding {binding} still uses atomic topology loads inside every solver iteration; construction has finished before primal dispatch");
            }
            let mut pose_stores = 0;
            for function in module
                .functions
                .iter()
                .map(|(_, f)| f)
                .chain(module.entry_points.iter().map(|e| &e.function))
            {
                let mut pointers = Vec::new();
                stores(&function.body, &mut pointers);
                for mut pointer in pointers {
                    let mut member = None;
                    loop {
                        pointer = match function.expressions[pointer] {
                            naga::Expression::AccessIndex { base, index } => {
                                member = Some(index);
                                base
                            }
                            naga::Expression::Access { base, .. } => base,
                            _ => break,
                        };
                    }
                    let naga::Expression::GlobalVariable(g) = function.expressions[pointer] else {
                        continue;
                    };
                    let global = &module.global_variables[g];
                    if !global
                        .binding
                        .as_ref()
                        .is_some_and(|b| b.group == 0 && b.binding == 0)
                    {
                        continue;
                    }
                    let naga::TypeInner::Array { base, .. } = module.types[global.ty].inner else {
                        panic!("body array ABI");
                    };
                    let naga::TypeInner::Struct { ref members, .. } = module.types[base].inner
                    else {
                        panic!("body struct ABI");
                    };
                    let field = member.map(|i| &members[i as usize]);
                    assert!(field.is_some_and(|f| f.offset == 0 || f.offset == 16),
                        "{name}: primal writes non-pose body field {:?}; full 112-byte BodyState stores waste bandwidth and risk overwriting unrelated state", field.and_then(|f| f.name.as_ref()));
                    pose_stores += 1;
                }
            }
            assert_eq!(
                pose_stores, 2,
                "{name}: publish exactly position and orientation after the Newton solve"
            );
        }
    }

    #[test]
    fn count_only_kernels_do_not_load_unrelated_atomic_counters() {
        // These kernels need immutable body/collider counts; reparent commands
        // additionally read the contact count in their invalidation branch, and the
        // awake-list per-body passes (PASS_U == 1) bound threads by the awake count
        // in place of bodyCount: one read of a counter no thread of that dispatch
        // writes, the same as every other awake-list kernel. Loading a
        // whole StepCounters snapshot emits ten atomic reads for every body on
        // WGSL, including counters being incremented by neighbouring threads.
        // Check parsed IR rather than generated identifiers or text formatting.
        let entries = [
            "apply_commands",
            "build_aabbs",
            "broadphase_insert",
            "broadphase_query",
            "build_awake_list",
            "predict_bodies",
            "extract_velocity_sleep",
            "record_stab_velocity",
            "sleep_island_commit",
            "recompute_inertia",
        ];
        let shaders = WebGpuWorld::compile_shaders_report();
        for name in entries {
            let source = shaders
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("missing count-access regression kernel {name}"))
                .1
                .as_ref()
                .unwrap();
            let module = naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(source)));
            let mut unexpected = Vec::new();
            let mut atomic_offsets = Vec::new();
            for function in module
                .functions
                .iter()
                .map(|(_, f)| f)
                .chain(module.entry_points.iter().map(|e| &e.function))
            {
                for (_, expr) in function.expressions.iter() {
                    let naga::Expression::Load { pointer } = expr else {
                        continue;
                    };
                    let naga::Expression::AccessIndex { index: member, .. } =
                        function.expressions[*pointer]
                    else {
                        continue;
                    };
                    let mut root = *pointer;
                    loop {
                        root = match function.expressions[root] {
                            naga::Expression::AccessIndex { base, .. }
                            | naga::Expression::Access { base, .. } => base,
                            _ => break,
                        };
                    }
                    let naga::Expression::GlobalVariable(global) = function.expressions[root]
                    else {
                        continue;
                    };
                    let global = &module.global_variables[global];
                    if !global
                        .binding
                        .as_ref()
                        .is_some_and(|b| b.group == 0 && b.binding == 7)
                    {
                        continue;
                    }
                    let naga::TypeInner::Array { base, .. } = module.types[global.ty].inner else {
                        panic!("{name}: counter buffer ABI is no longer an array");
                    };
                    let naga::TypeInner::Struct { ref members, .. } = module.types[base].inner
                    else {
                        panic!("{name}: counter buffer element is no longer a struct");
                    };
                    let field = &members[member as usize];
                    if matches!(module.types[field.ty].inner, naga::TypeInner::Atomic(_)) {
                        atomic_offsets.push(field.offset);
                        unexpected.push(format!(
                            "{} at byte {} in {}",
                            field.name.as_deref().unwrap_or("unnamed counter"),
                            field.offset,
                            function.name.as_deref().unwrap_or("entry point")
                        ));
                    }
                }
            }
            atomic_offsets.sort_unstable();
            let expected = match name {
                "apply_commands" => vec![8],
                "predict_bodies"
                | "extract_velocity_sleep"
                | "record_stab_velocity"
                | "sleep_island_commit" => vec![20],
                _ => vec![],
            };
            assert_eq!(atomic_offsets, expected,
                "{name}: unrelated atomic counter reads per invocation: {unexpected:?}. Read bodyCount/colliderCount directly; a full counter snapshot can cause GPU contention at 100k–506k scale. Atomic increments remain allowed.");
        }
    }
}

/// The kernels the build embeds are the run-time compile's (the fast profile's store),
/// byte for byte: one compile, two places it runs.
#[cfg(all(test, not(kernels_at_runtime), not(target_arch = "wasm32")))]
mod embedded_is_runtime {
    #[test]
    fn the_embedded_kernels_are_the_run_time_compile() {
        let built = std::path::Path::new(env!("OUT_DIR"));
        let dir = built.join("runtime-compile-check");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../slang");
        let compiled = crate::kernel_compile::compile_all(&root, &dir).unwrap();
        let mut n = 0;
        for sub in ["wgsl", "spirv"] {
            let mut names: Vec<_> = std::fs::read_dir(built.join(sub)).unwrap().map(|e| e.unwrap().file_name()).collect();
            names.sort();
            let mut theirs: Vec<_> = std::fs::read_dir(dir.join(sub)).unwrap().map(|e| e.unwrap().file_name()).collect();
            theirs.sort();
            assert_eq!(names, theirs, "{sub}: the same modules");
            for name in names {
                let a = std::fs::read(built.join(sub).join(&name)).unwrap();
                let b = std::fs::read(dir.join(sub).join(&name)).unwrap();
                assert!(a == b, "{sub}/{}: the embedded module is not the run-time compile's", name.to_string_lossy());
                n += 1;
            }
        }
        assert_eq!(
            std::fs::read_to_string(built.join("spirv_entries.txt")).unwrap(),
            std::fs::read_to_string(dir.join("spirv_entries.txt")).unwrap()
        );
        let embedded: Vec<&str> = super::WebGpuWorld::direct_shaders().iter().map(|(n, _)| *n).collect();
        assert_eq!(embedded, compiled.wgsl_ok.iter().map(String::as_str).collect::<Vec<_>>());
        assert!(n > 500, "{n} modules compared");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
