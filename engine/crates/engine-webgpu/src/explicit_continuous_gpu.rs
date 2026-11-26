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

    /// Continuous contact's counts after the last frame (None without its tables): each
    /// vehicle's listed pairs at its last pass, then the counters (overflow, impacts, zones,
    /// unresolved, zones past 1 rad).
    pub fn continuous_counts(&self) -> Option<(Vec<u32>, [i32; 5])> {
        let at = self.layout.words[engine_exec::explicit_plan::continuous::W_CC + 1] as usize;
        if self.layout.words[engine_exec::explicit_plan::continuous::W_CC] == 0 {
            return None;
        }
        let acc: Vec<i32> = bytemuck::cast_slice(&self.read(&self.acc)).to_vec();
        let nv = self.layout.spans.len();
        let pairs = acc[at..at + nv].iter().map(|&x| x as u32).collect();
        let mut c = [0; 5];
        c.copy_from_slice(&acc[at + nv..at + nv + 5]);
        Some((pairs, c))
    }
}
