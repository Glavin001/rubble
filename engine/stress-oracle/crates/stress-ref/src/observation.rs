//! The observation format (`stress-observation/1`): what any solver reports about a run.
//!
//! Ours and every oracle's post-processor write this same JSON, and `metrics.rs`
//! derives every benchmark quantity from it with one implementation, so a metric
//! can never be computed differently for the two sides of a comparison.
//!
//! * `probes`: sampled time series. `lo`/`hi` are the min/max of the value since the
//!   previous sample (an envelope), so peaks between samples are not lost; oracles
//!   that cannot provide them may leave both empty.
//! * `bodies`: final state of every chunk of every scene body, indexed like the scene's
//!   `chunks`. A chunk is *detached* when it is no longer connected to the body's main
//!   piece (the supported piece, or the heaviest piece of an unsupported body).
//!   Continuum oracles map their elements to chunks (majority vote).
//! * `flags` and `values`: named booleans and scalars (e.g. `collapse`).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub const OBSERVATION_FORMAT: &str = "stress-observation/1";

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ProbeSeries {
    pub t: Vec<f64>,
    pub v: Vec<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lo: Vec<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hi: Vec<f64>,
}

impl ProbeSeries {
    pub fn push(&mut self, t: f64, v: f64, lo: f64, hi: f64) {
        self.t.push(t);
        self.v.push(v);
        self.lo.push(lo);
        self.hi.push(hi);
    }

    pub fn has_envelope(&self) -> bool {
        self.lo.len() == self.v.len() && self.hi.len() == self.v.len() && !self.v.is_empty()
    }

    /// Linear interpolation at `time` (clamped to the sampled range).
    pub fn at(&self, time: f64) -> Option<f64> {
        if self.t.is_empty() {
            return None;
        }
        if time <= self.t[0] {
            return Some(self.v[0]);
        }
        for i in 1..self.t.len() {
            if time <= self.t[i] {
                let f = (time - self.t[i - 1]) / (self.t[i] - self.t[i - 1]).max(1e-300);
                return Some(self.v[i - 1] + f * (self.v[i] - self.v[i - 1]));
            }
        }
        self.v.last().copied()
    }

    /// Min and max over samples with `t >= from` (using the envelope when present).
    pub fn range_after(&self, from: f64) -> Option<(f64, f64)> {
        let env = self.has_envelope();
        let mut out: Option<(f64, f64)> = None;
        for i in 0..self.t.len() {
            if self.t[i] < from {
                continue;
            }
            let (lo, hi) = if env && i > 0 && self.t[i - 1] >= from {
                (self.lo[i].min(self.v[i]), self.hi[i].max(self.v[i]))
            } else {
                (self.v[i], self.v[i])
            };
            out = Some(match out {
                None => (lo, hi),
                Some((a, b)) => (a.min(lo), b.max(hi)),
            });
        }
        out
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ChunkObservation {
    pub detached: bool,
    /// Removed by a scripted event (not counted as detached debris).
    #[serde(default)]
    pub removed: bool,
    /// Fragment (piece) id; chunks sharing an id moved as one body. -1 if unknown.
    #[serde(default = "minus_one")]
    pub fragment: i64,
    pub velocity: [f64; 3],
    pub displacement: [f64; 3],
}

fn minus_one() -> i64 {
    -1
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Observation {
    pub format: String,
    pub scene: String,
    pub solver: String,
    #[serde(default)]
    pub solver_version: String,
    #[serde(default)]
    pub seed: u64,
    pub end_time: f64,
    #[serde(default)]
    pub probes: BTreeMap<String, ProbeSeries>,
    #[serde(default)]
    pub bodies: BTreeMap<String, Vec<ChunkObservation>>,
    #[serde(default)]
    pub flags: BTreeMap<String, bool>,
    #[serde(default)]
    pub values: BTreeMap<String, f64>,
    #[serde(default)]
    pub notes: Vec<String>,
}

impl Observation {
    pub fn new(scene: &str, solver: &str, version: &str, seed: u64) -> Observation {
        Observation {
            format: OBSERVATION_FORMAT.into(),
            scene: scene.into(),
            solver: solver.into(),
            solver_version: version.into(),
            seed,
            end_time: 0.0,
            probes: BTreeMap::new(),
            bodies: BTreeMap::new(),
            flags: BTreeMap::new(),
            values: BTreeMap::new(),
            notes: Vec::new(),
        }
    }

    pub fn load(path: &std::path::Path) -> Result<Observation, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let o: Observation = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        if o.format != OBSERVATION_FORMAT {
            return Err(format!("{}: unsupported observation format '{}'", path.display(), o.format));
        }
        Ok(o)
    }

    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).expect("observation serializes")
    }
}
