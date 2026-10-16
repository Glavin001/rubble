//! Shared phase graph. Backends only encode submission; physics equations live in Slang.

pub mod rigid_plan;

use engine_schema::{AvbdParams, StepCounters, WorldCapacity};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ExecError {
    #[error("GPU overflow flags=0x{0:x}")]
    Overflow(u32),
    #[error("backend: {0}")]
    Backend(String),
    #[error("capacity: {0}")]
    Capacity(String),
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PhaseTimingsMs {
    pub commands: f64,
    pub predict: f64,
    pub broadphase: f64,
    pub narrowphase: f64,
    pub graph: f64,
    pub solve: f64,
    pub sleep: f64,
    pub total_gpu: f64,
    pub cpu_submit: f64,
}

/// Canonical phase order. CCD is optional and does not reorder primal/dual.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    ApplyCommands,
    RecomputeInertia,
    Predict,
    BuildAabbs,
    Broadphase,
    Narrowphase,
    PersistContacts,
    BuildAwakeList,
    InitColors,
    ColorBodies,
    PackColorLists,
    SolvePrimalColor,
    SolveDual,
    ExtractVelocitySleep,
    CcdClassify,
    CcdSweep,
}

pub const PHASE_GRAPH: &[Phase] = &[
    Phase::ApplyCommands,
    Phase::RecomputeInertia,
    Phase::Predict,
    Phase::BuildAabbs,
    Phase::Broadphase,
    Phase::Narrowphase,
    Phase::PersistContacts,
    Phase::BuildAwakeList,
    Phase::InitColors,
    Phase::ColorBodies,
    Phase::PackColorLists,
    Phase::SolvePrimalColor,
    Phase::SolveDual,
    Phase::ExtractVelocitySleep,
];

pub const CCD_PHASES: &[Phase] = &[Phase::CcdClassify, Phase::CcdSweep];

pub trait PhysicsBackend {
    fn name(&self) -> &'static str;
    fn capacity(&self) -> &WorldCapacity;
    fn params(&self) -> &AvbdParams;
    fn set_params(&mut self, params: AvbdParams);
    /// Submit one step. Must not read back body/contact buffers on the hot path.
    fn step(&mut self) -> Result<StepCounters, ExecError>;
    /// Sample GPU timestamps / counters after the frame (async OK).
    fn last_timings(&self) -> PhaseTimingsMs;
    fn last_counters(&self) -> StepCounters;
}

pub fn entry_name(phase: Phase) -> &'static str {
    match phase {
        Phase::ApplyCommands => "apply_commands",
        Phase::RecomputeInertia => "recompute_inertia",
        Phase::Predict => "predict_bodies",
        Phase::BuildAabbs => "build_aabbs",
        Phase::Broadphase => "broadphase_grid",
        Phase::Narrowphase => "narrowphase",
        Phase::PersistContacts => "persist_contacts",
        Phase::BuildAwakeList => "build_awake_list",
        Phase::InitColors => "init_colors",
        Phase::ColorBodies => "color_bodies",
        Phase::PackColorLists => "pack_color_lists",
        Phase::SolvePrimalColor => "solve_primal_color",
        Phase::SolveDual => "solve_dual",
        Phase::ExtractVelocitySleep => "extract_velocity_sleep",
        Phase::CcdClassify => "ccd_classify",
        Phase::CcdSweep => "ccd_sweep",
    }
}
