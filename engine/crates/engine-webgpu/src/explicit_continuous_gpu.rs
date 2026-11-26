//! Continuous contact on the GPU (`docs/continuous-contact-design.md`, "The GPU port"): the
//! host's switch. The split group path's members then run each step in two halves (the node
//! update, the rest) with the continuous passes between (`engine_exec::explicit_plan::continuous`).

use super::ExplicitBeamsGpu;

impl ExplicitBeamsGpu {
    /// Continuous contact on or off (off, the default). It runs on the split group path
    /// ([`ExplicitBeamsGpu::set_group_split`]); without it, frames are as before.
    pub fn set_continuous(&mut self, on: bool) {
        self.planner.continuous = on;
    }
}
