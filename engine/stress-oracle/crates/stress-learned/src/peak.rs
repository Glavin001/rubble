//! Per-bond peak utilization within a frame (oracle recorder helper).
//!
//! Mirrors `RtBond.peak_utilization` when `SimDesc.record_peak_utilization` is on;
//! this type is also usable standalone in tests without a full world.

#[derive(Clone, Debug, Default)]
pub struct PeakUtilization {
    peaks: Vec<f64>,
}

impl PeakUtilization {
    pub fn new(n_bonds: usize) -> Self {
        Self { peaks: vec![0.0; n_bonds] }
    }

    pub fn reset(&mut self) {
        for p in &mut self.peaks {
            *p = 0.0;
        }
    }

    pub fn observe(&mut self, bond: usize, utilization: f64) {
        if let Some(p) = self.peaks.get_mut(bond) {
            *p = p.max(utilization);
        }
    }

    pub fn peaks(&self) -> &[f64] {
        &self.peaks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn running_max_resets() {
        let mut p = PeakUtilization::new(3);
        p.observe(1, 0.4);
        p.observe(1, 1.2);
        p.observe(1, 0.9);
        assert!((p.peaks()[1] - 1.2).abs() < 1e-12);
        p.reset();
        assert_eq!(p.peaks()[1], 0.0);
        p.observe(1, 0.3);
        assert!((p.peaks()[1] - 0.3).abs() < 1e-12);
    }
}
