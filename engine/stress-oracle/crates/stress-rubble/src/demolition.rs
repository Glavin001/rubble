//! Demolition charges on the GPU worlds: a pack's `defaults.charges`
//! (`stress_ref::wrecking::charges_of`) set off by name, each cut between two frames at
//! its delay ([`Demolition`]), by `ThresholdWorld::cut_bonds` (the threshold profile) or
//! `GpuWorld::cut_bonds` (`blast_parity`, the resident world).

use stress_ref::wrecking::PackCharge;

/// A structure's charges and the ones set off, waiting for their frame.
#[derive(Clone, Debug, Default)]
pub struct Demolition {
    pub charges: Vec<PackCharge>,
    /// (charge, the frame before which it is cut), in firing order.
    armed: Vec<(usize, u64)>,
}

impl Demolition {
    pub fn new(charges: Vec<PackCharge>) -> Demolition {
        Demolition { charges, armed: Vec::new() }
    }

    /// The charges of a pack (`defaults.charges`).
    pub fn of_pack(pack: &serde_json::Value) -> Demolition {
        Demolition::new(stress_ref::wrecking::charges_of(pack))
    }

    /// Set off the charge `name` before frame `frame` (the frames stepped so far), or with
    /// an empty name every charge, each its delay later (rounded to whole frames of
    /// `frame_dt`). False when no charge has that name.
    pub fn arm(&mut self, name: &str, frame: u64, frame_dt: f64) -> bool {
        let picked: Vec<usize> = (0..self.charges.len()).filter(|&k| name.is_empty() || self.charges[k].name == name).collect();
        for &k in &picked {
            let delay = if name.is_empty() { (self.charges[k].delay / frame_dt).round() as u64 } else { 0 };
            self.armed.push((k, frame + delay));
        }
        self.armed.sort_by_key(|&(k, f)| (f, k));
        !picked.is_empty()
    }

    /// The charges due before frame `frame`, taken off the list.
    pub fn due(&mut self, frame: u64) -> Vec<PackCharge> {
        let (now, later): (Vec<_>, Vec<_>) = std::mem::take(&mut self.armed).into_iter().partition(|&(_, f)| f <= frame);
        self.armed = later;
        now.into_iter().map(|(k, _)| self.charges[k].clone()).collect()
    }

    /// Charges set off and not yet cut.
    pub fn pending(&self) -> usize {
        self.armed.len()
    }

    pub fn names(&self) -> Vec<String> {
        self.charges.iter().map(|c| c.name.clone()).collect()
    }
}
