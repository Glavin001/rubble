//! Launched bonded bodies on the GPU worlds: a pack's `defaults.launches`
//! (`stress_ref::launch::launches_of`) asked for by name, each set moving between two frames
//! at its delay ([`Launcher`]), by `ThresholdWorld::launch_island` (the threshold profile) or
//! `GpuWorld::launch_island` (`blast_parity`, the resident world).

use stress_ref::launch::Launch;

/// A structure's launches and the ones asked for, waiting for their frame.
#[derive(Clone, Debug, Default)]
pub struct Launcher {
    pub launches: Vec<Launch>,
    /// (launch, the frame before which it is released), in release order.
    armed: Vec<(usize, u64)>,
}

impl Launcher {
    pub fn new(launches: Vec<Launch>) -> Launcher {
        Launcher { launches, armed: Vec::new() }
    }

    /// The launches of a pack (`defaults.launches`); an error names a malformed one.
    pub fn of_pack(pack: &serde_json::Value) -> Result<Launcher, String> {
        Ok(Launcher::new(stress_ref::launch::launches_of(pack)?))
    }

    /// Release the launch `name` before frame `frame` (the frames stepped so far), or with an
    /// empty name every launch, each its delay later (rounded to whole frames of
    /// `frame_dt`). False when no launch has that name.
    pub fn arm(&mut self, name: &str, frame: u64, frame_dt: f64) -> bool {
        let picked: Vec<usize> = (0..self.launches.len()).filter(|&k| name.is_empty() || self.launches[k].name == name).collect();
        for &k in &picked {
            self.armed.push((k, frame + (self.launches[k].delay / frame_dt).round() as u64));
        }
        self.armed.sort_by_key(|&(k, f)| (f, k));
        !picked.is_empty()
    }

    /// The launches due before frame `frame`, taken off the list.
    pub fn due(&mut self, frame: u64) -> Vec<Launch> {
        let (now, later): (Vec<_>, Vec<_>) = std::mem::take(&mut self.armed).into_iter().partition(|&(_, f)| f <= frame);
        self.armed = later;
        now.into_iter().map(|(k, _)| self.launches[k].clone()).collect()
    }

    /// Launches asked for and not yet released.
    pub fn pending(&self) -> usize {
        self.armed.len()
    }

    pub fn names(&self) -> Vec<String> {
        self.launches.iter().map(|l| l.name.clone()).collect()
    }
}
