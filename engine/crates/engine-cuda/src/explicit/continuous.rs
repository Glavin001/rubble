//! Continuous contact on CUDA (`docs/continuous-contact-design.md`, "The GPU port"): the
//! host's switch, as engine-webgpu's `explicit_continuous_gpu.rs`. The shared plan cuts each
//! step's members in two with the continuous passes between (`engine_exec::explicit_plan::
//! continuous`); it runs on the split group path ([`CudaExplicitBeams::set_group_split`]).

use super::CudaExplicitBeams;

impl CudaExplicitBeams {
    /// Continuous contact on or off (off, the default). Refused for a fleet with a vehicle
    /// whose own surfaces are closer than the own clearance at spawn (no pair is left out).
    pub fn set_continuous(&mut self, on: bool) -> Result<(), String> {
        if let (true, Some(e)) = (on, &self.layout.cc_refused) {
            return Err(e.clone());
        }
        self.planner.continuous = on;
        Ok(())
    }
}
