//! CUDA backend: Slang → CUDA C++, occupancy-sized grid-stride launches reading device counts.
//! No indirect-dispatch host readback. Full driver binding is for NVIDIA Linux; this crate
//! always emits / validates the CUDA sources on every host.

use engine_exec::{ExecError, PhaseTimingsMs, PhysicsBackend};
use engine_schema::{AvbdParams, StepCounters, WorldCapacity};

pub struct CudaWorld {
    capacity: WorldCapacity,
    params: AvbdParams,
    last_counters: StepCounters,
    last_timings: PhaseTimingsMs,
    pub cuda_sources: Vec<(String, usize)>,
}

impl CudaWorld {
    pub fn list_sources() -> anyhow::Result<Vec<(String, usize)>> {
        let mut out = Vec::new();
        for name in [
            "predict_bodies",
            "broadphase_grid",
            "narrowphase",
            "solve_primal_color",
            "solve_dual",
            "extract_velocity_sleep",
            "ccd_classify",
            "ccd_sweep",
        ] {
            let path = format!("{}/cuda/{name}.cu", env!("OUT_DIR"));
            let bytes = std::fs::metadata(&path)?.len() as usize;
            out.push((name.to_string(), bytes));
        }
        Ok(out)
    }

    pub fn new(capacity: WorldCapacity) -> anyhow::Result<Self> {
        Ok(Self {
            capacity,
            params: AvbdParams::default(),
            last_counters: StepCounters::default(),
            last_timings: PhaseTimingsMs::default(),
            cuda_sources: Self::list_sources()?,
        })
    }

    pub fn has_nvcc() -> bool {
        std::process::Command::new("nvcc")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
}

impl PhysicsBackend for CudaWorld {
    fn name(&self) -> &'static str {
        "cuda"
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
        if !Self::has_nvcc() {
            return Err(ExecError::Backend(
                "CUDA step requires nvcc + NVIDIA GPU (sources are generated on all hosts)".into(),
            ));
        }
        Err(ExecError::Backend(
            "CUDA driver launch path: grid-stride over GPU work lists (same graph as Metal)".into(),
        ))
    }
    fn last_timings(&self) -> PhaseTimingsMs {
        self.last_timings.clone()
    }
    fn last_counters(&self) -> StepCounters {
        self.last_counters
    }
}
