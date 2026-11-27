//! Continuous contact on CUDA (`docs/continuous-contact-design.md`, "The GPU port"): the
//! host's switch, as engine-webgpu's `explicit_continuous_gpu.rs`. The shared plan cuts each
//! step's members in two with the continuous passes between (`engine_exec::explicit_plan::
//! continuous`); it runs on the split group path ([`CudaExplicitBeams::set_group_split`]).

use super::CudaExplicitBeams;
use engine_exec::explicit_plan::{self as plan, continuous};
use engine_exec::ExecError;

impl CudaExplicitBeams {
    /// Continuous contact on or off (off, the default). Refused for a fleet with a vehicle
    /// whose own surfaces are closer than the own clearance at spawn (no pair is left out).
    pub fn set_continuous(&mut self, on: bool) -> Result<(), String> {
        if let (true, Some(e)) = (on, &self.layout.cc_refused) {
            return Err(e.clone());
        }
        self.planner.continuous.on = on;
        self.write_cc_mode().map_err(|e| e.to_string())
    }

    /// Velocity contact as continuous contact's response (engine-webgpu's
    /// `set_continuous_velocity`; off: the position response).
    pub fn set_continuous_velocity(&mut self, on: bool) -> Result<(), ExecError> {
        if on && !plan::SPLIT_WHICH.contains(&"ccvel") {
            return Err(ExecError::Backend("velocity contact's GPU module (ccvel) is not built".into()));
        }
        self.planner.continuous.velocity = on;
        self.write_cc_mode()
    }

    /// The response's mode into the tables' statics ([`continuous::cc_mode_words`]).
    fn write_cc_mode(&self) -> Result<(), ExecError> {
        for (rec, w) in continuous::cc_mode_words(&self.planner, &self.layout) {
            self.upload(plan::BIND_STATIC, rec as u64 * 16 + 4, bytemuck::cast_slice(&w))?;
        }
        Ok(())
    }
}
