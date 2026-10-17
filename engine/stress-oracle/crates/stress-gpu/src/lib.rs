//! Single-precision GPU stress solver, diffed against the f64 reference (`stress-ref`).
//!
//! Kernels are written in Slang (`shaders/*.slang`) and compiled by
//! `scripts/compile_shaders.sh` (pinned Slang in Docker) to Metal Shading Language and
//! WGSL, both checked in. wgpu runs them: on macOS the Metal library goes straight to
//! the driver (`build.rs` builds it with Xcode), in a browser the WGSL goes to WebGPU.
//! Everything on the GPU is f32; accuracy against the reference is measured, not assumed.

pub mod gpu;
pub mod gate;
pub mod contacts;
pub mod coupled;
pub mod coverage;
pub use stress_layout::joint;
pub mod loads;
pub mod methods;
pub mod micro;

/// The generated shaders of one Slang file, in every form the runtime can load.
#[derive(Clone, Copy, Debug)]
pub struct ShaderSet {
    pub name: &'static str,
    pub wgsl: &'static str,
    pub msl: &'static str,
    /// Built from `msl` by `build.rs` (empty when Xcode was unavailable): without fast
    /// math (parity), and with it (production, `STRESS_GPU_MATH=fast`).
    pub metallib: &'static [u8],
    pub metallib_fast: &'static [u8],
    /// Entry points and their workgroup sizes (render stages: `(1, 1, 1)`).
    pub entries: &'static [(&'static str, (u32, u32, u32))],
}

macro_rules! shader_set {
    ($name:literal, [$(($entry:literal, $size:expr)),* $(,)?]) => {
        ShaderSet {
            name: $name,
            wgsl: include_str!(concat!("../shaders/generated/", $name, ".wgsl")),
            msl: include_str!(concat!("../shaders/generated/", $name, ".metal")),
            metallib: include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".metallib")),
            metallib_fast: include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".fast.metallib")),
            entries: &[$(($entry, $size)),*],
        }
    };
}

/// Every Slang file's generated shaders.
pub mod shaders {
    use super::ShaderSet;
    pub const SMOKE: ShaderSet = shader_set!("smoke", [("smoke", (64, 1, 1))]);
    pub const STRESS_STEP: ShaderSet = shader_set!("stress_step", [("bond_forces", (64, 1, 1)), ("chunk_integrate", (64, 1, 1)), ("island_substeps", (256, 1, 1)), ("island_shared_substeps", (256, 1, 1))]);
    pub const JOINT_TEST: ShaderSet = shader_set!("joint_test", [("joint_eval_test", (64, 1, 1))]);
    pub const WORLD: ShaderSet =
        shader_set!("world", [
            ("contact_forces", (64, 1, 1)),
            ("contact_sums", (64, 1, 1)),
            ("impactor_shares", (64, 1, 1)),
            ("impactor_crush", (256, 1, 1)),
            ("island_frame", (256, 1, 1)),
            ("island_frame_bonds_wide", (512, 1, 1)),
            ("impactor_integrate", (256, 1, 1)),
            ("wide_bonds", (256, 1, 1)),
            ("wide_chunks", (256, 1, 1)),
            ("wide_drift", (256, 1, 1)),
            ("wide_rigid", (256, 1, 1)),
            ("wide_end", (256, 1, 1)),
            ("wide_wake", (256, 1, 1)),
            ("island_statics", (256, 1, 1)),
            ("settled_fatigue", (256, 1, 1)),
            ("sums_and_impactors", (256, 1, 1)),
            ("islands_and_wide_chunks", (256, 1, 1)),
            ("sums_impactors_bonds", (256, 1, 1)),
            ("contact_islands", (256, 1, 1)),
            ("contact_islands_and_wide_chunks", (256, 1, 1)),
            ("forces_and_bonds", (256, 1, 1)),
            ("islands_sums_impactors", (256, 1, 1)),
            ("islands_sums_impactors_wake", (256, 1, 1)),
            ("travel_start", (64, 1, 1)),
        ]);
    pub const STRESS_RENDER: ShaderSet = shader_set!("stress_render", [("stress_vs", (1, 1, 1)), ("stress_fs", (1, 1, 1))]);
    pub const DESTRUCTION_RENDER: ShaderSet = shader_set!(
        "destruction_render",
        [("chunk_vs", (1, 1, 1)), ("chunk_hull_vs", (1, 1, 1)), ("host_chunk_vs", (1, 1, 1)), ("host_hull_vs", (1, 1, 1)), ("impactor_vs", (1, 1, 1)), ("ground_vs", (1, 1, 1)), ("lit_fs", (1, 1, 1)), ("ground_fs", (1, 1, 1))]
    );
    pub const HUD_OVERLAY: ShaderSet = shader_set!("hud_overlay", [("hud_vs", (1, 1, 1)), ("hud_fs", (1, 1, 1))]);
}

pub use stress_ref::scenes;
pub mod solver;
pub mod statics;
pub mod static_inputs;
pub mod stability;
pub mod world;
pub mod stress;
