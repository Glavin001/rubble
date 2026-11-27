//! Continuous contact on the GPU (`docs/continuous-contact-design.md`, "The GPU port"): the
//! host's switch. The split group path's members then run each step in two halves (the node
//! update, the rest) with the continuous passes between (`engine_exec::explicit_plan::continuous`).

use super::ExplicitBeamsGpu;

impl ExplicitBeamsGpu {
    /// Continuous contact on or off (off, the default). It runs on the split group path
    /// ([`ExplicitBeamsGpu::set_group_split`]); without it, frames are as before. Refused for
    /// a fleet with a vehicle whose own surfaces are closer than the own clearance at spawn
    /// (no pair is left out).
    pub fn set_continuous(&mut self, on: bool) -> Result<(), String> {
        if let (true, Some(e)) = (on, &self.layout.cc_refused) {
            return Err(e.clone());
        }
        self.planner.continuous = on;
        Ok(())
    }

    /// Continuous contact's counts after the last frame (None without its tables): each
    /// vehicle's listed pairs at its last pass, then the counters (overflow, impacts, zones,
    /// unresolved, zones past 1 rad, nodes past the listing's reach).
    pub fn continuous_counts(&self) -> Option<(Vec<u32>, [i32; 6])> {
        let at = self.layout.words[engine_exec::explicit_plan::continuous::W_CC + 1] as usize;
        if self.layout.words[engine_exec::explicit_plan::continuous::W_CC] == 0 {
            return None;
        }
        let acc: Vec<i32> = bytemuck::cast_slice(&self.read(&self.acc)).to_vec();
        let nv = self.layout.spans.len();
        let w = engine_exec::explicit_plan::continuous::CC_VEH_WORDS;
        let pairs = (0..nv).map(|v| acc[at + w * v] as u32).collect();
        let mut c = [0; 6];
        c.copy_from_slice(&acc[at + w * nv..at + w * nv + 6]);
        Some((pairs, c))
    }

    /// Every listed pair's four global node keys after the last frame (a diagnostic; empty
    /// without the tables).
    pub fn continuous_pair_keys(&self) -> Vec<[u32; 4]> {
        use engine_exec::explicit_plan::continuous::{CC_PAIR_WORDS, CC_VEH_WORDS, W_CC};
        let (base, at) = (self.layout.words[W_CC] as usize, self.layout.words[W_CC + 1] as usize);
        if base == 0 {
            return Vec::new();
        }
        let acc: Vec<i32> = bytemuck::cast_slice(&self.read(&self.acc)).to_vec();
        let next: Vec<u32> = bytemuck::cast_slice(&self.read(&self.bufs[8])).to_vec();
        let statics: Vec<[u32; 4]> = bytemuck::cast_slice(&self.read(&self.bufs[1])).to_vec();
        let mut out = Vec::new();
        for v in 0..self.layout.spans.len() {
            let rec = statics[base + 2 + 3 * v];
            let n = (acc[at + CC_VEH_WORDS * v] as u32).min(rec[3]) as usize;
            for p in 0..n {
                let w = rec[2] as usize + CC_PAIR_WORDS * p;
                out.push([next[w], next[w + 1], next[w + 2], next[w + 3]]);
            }
        }
        out
    }
}
