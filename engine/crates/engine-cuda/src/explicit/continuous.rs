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

    /// Whether continuous contact has run on any frame so far: it is on, and a frame took
    /// the split group path without self-collision split (`explicit_plan::continuous::on`).
    /// The split group path needs vehicle contact between two vehicles or more
    /// (`FramePlanner::contact_active`), so for one vehicle against static boxes the setting
    /// is taken and never runs: such a run is an edge-contact run.
    pub fn continuous_ran(&self) -> bool {
        self.planner.continuous && !self.self_split && self.split_frames > 0
    }
}
