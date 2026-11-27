//! A run recorded for playback (the Reference Studio's Simulate): every sample's node
//! positions (every [`SAMPLE_STEPS`] steps, 4 ms) and when each beam broke, with the run's
//! report. Playback, scrubbing, Rubble's outline at an image's pinned time and the
//! comparison all read the recording; none of them re-simulates.
//!
//! The binary form ([`Recording::to_bytes`]) is what the studio caches in its workspace
//! (`recordings/<scenario hash>.rrec`): `RREC1\0`, a little-endian `u32` header length, the
//! header as JSON ([`Header`]), then every sample's positions (`f32` x, y, z per node, bodies in
//! order) and per beam the first sample it is broken in (`u32`, `u32::MAX`: never).

use super::*;

/// What a recording is of.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, ts_rs::TS)]
pub struct RecBody {
    pub name: String,
    pub design: String,
    pub nodes: u32,
    pub beams: u32,
}

/// A recording's header (its binary form's JSON part).
#[derive(Clone, Debug, Serialize, Deserialize, ts_rs::TS)]
pub struct Header {
    pub scenario: String,
    /// What it was made from (`ScenarioDesc::digest`, or `builtin-<id>-<contact>`).
    #[serde(default)]
    pub hash: String,
    /// What ran it: `webgpu` (the page's GPU solver, what production runs) or `cpu-fallback`
    /// (the wasm CPU reference: the automated smoke test only).
    pub executor: String,
    pub contact: Contact,
    pub run: u32,
    /// First contact (s from the start).
    pub impact_s: f32,
    /// Between samples (s).
    pub sample_s: f32,
    pub samples: u32,
    pub bodies: Vec<RecBody>,
    /// The run's report (metrics and criteria), once complete.
    #[ts(optional)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub report: Option<CrashReport>,
}

#[derive(Clone, Debug)]
pub struct Recording {
    pub scenario: String,
    pub hash: String,
    pub executor: String,
    pub contact: Contact,
    pub run: u32,
    pub impact_s: f32,
    pub sample_s: f32,
    pub bodies: Vec<RecBody>,
    /// Per sample, every body's node positions (x, y, z), bodies in order.
    pub pos: Vec<Vec<f32>>,
    /// Per beam (bodies in order), the first sample it is broken in.
    pub broken_at: Vec<u32>,
    pub report: Option<CrashReport>,
}

const MAGIC: &[u8; 6] = b"RREC1\0";

impl Recording {
    pub fn new(s: &CrashScenario, built: &Built, executor: &str) -> Self {
        let bodies: Vec<RecBody> = built.layout.bodies.iter().zip(&built.fleet).map(|(b, f)| RecBody { name: b.name.clone(), design: b.design.clone(), nodes: f.nodes.len() as u32, beams: f.beams.len() as u32 }).collect();
        let beams = bodies.iter().map(|b| b.beams as usize).sum();
        Self {
            scenario: s.id.clone(),
            hash: String::new(),
            executor: executor.into(),
            contact: built.contact,
            run: 0,
            impact_s: built.layout.impact_s,
            sample_s: DT * SAMPLE_STEPS as f32,
            bodies,
            pos: Vec::new(),
            broken_at: vec![u32::MAX; beams],
            report: None,
        }
    }

    /// The next sample (samples arrive in order).
    pub fn push(&mut self, x: &Sample) {
        let k = self.pos.len() as u32;
        self.pos.push(x.pos.iter().flatten().flat_map(|p| p.to_array()).collect());
        let mut at = 0;
        for (b, st) in self.bodies.iter().zip(x.beams.iter().chain(std::iter::repeat(&Vec::new()))) {
            for (j, s) in st.iter().enumerate().take(b.beams as usize) {
                if s.broken && self.broken_at[at + j] == u32::MAX {
                    self.broken_at[at + j] = k;
                }
            }
            at += b.beams as usize;
        }
    }

    pub fn samples(&self) -> usize {
        self.pos.len()
    }

    /// The run's simulated time (s).
    pub fn duration_s(&self) -> f32 {
        self.samples().saturating_sub(1) as f32 * self.sample_s
    }

    /// Sample `k`'s time (ms after first contact).
    pub fn ms_of(&self, k: usize) -> f32 {
        (k as f32 * self.sample_s - self.impact_s) * 1000.0
    }

    /// The (fractional) sample at `ms` after first contact, within the run.
    pub fn index_at_ms(&self, ms: f32) -> f32 {
        let k = (self.impact_s + ms / 1000.0) / self.sample_s;
        k.clamp(0.0, self.samples().saturating_sub(1) as f32)
    }

    /// Every body's node positions at `ms` after first contact: between the two samples
    /// either side of it, linearly (a sample's own positions exactly at its time).
    pub fn positions_at_ms(&self, ms: f32) -> Vec<Vec<Vec3>> {
        let k = self.index_at_ms(ms);
        let (k0, f) = (k.floor() as usize, k.fract());
        let k1 = (k0 + 1).min(self.samples().saturating_sub(1));
        let (a, b) = (&self.pos[k0], &self.pos[k1]);
        let flat: Vec<Vec3> = (0..a.len() / 3).map(|i| {
            let p = Vec3::new(a[3 * i], a[3 * i + 1], a[3 * i + 2]);
            if f == 0.0 { p } else { p.lerp(Vec3::new(b[3 * i], b[3 * i + 1], b[3 * i + 2]), f) }
        }).collect();
        let mut out = Vec::new();
        let mut at = 0;
        for b in &self.bodies {
            out.push(flat[at..at + b.nodes as usize].to_vec());
            at += b.nodes as usize;
        }
        out
    }

    pub fn header(&self) -> Header {
        Header { scenario: self.scenario.clone(), hash: self.hash.clone(), executor: self.executor.clone(), contact: self.contact, run: self.run, impact_s: self.impact_s, sample_s: self.sample_s, samples: self.samples() as u32, bodies: self.bodies.clone(), report: self.report.clone() }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut h = self.header();
        // The report's time histories are left out: the studio shows metrics and criteria.
        if let Some(r) = h.report.as_mut() {
            r.series = None;
        }
        let head = serde_json::to_vec(&h).unwrap_or_default();
        let mut out = Vec::with_capacity(16 + head.len() + self.pos.iter().map(|p| 4 * p.len()).sum::<usize>() + 4 * self.broken_at.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(head.len() as u32).to_le_bytes());
        out.extend_from_slice(&head);
        for p in &self.pos {
            for x in p {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        for b in &self.broken_at {
            out.extend_from_slice(&b.to_le_bytes());
        }
        out
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self, String> {
        if b.len() < 10 || &b[..6] != MAGIC {
            return Err("not a Rubble recording (RREC1)".into());
        }
        let n = u32::from_le_bytes([b[6], b[7], b[8], b[9]]) as usize;
        let h: Header = serde_json::from_slice(b.get(10..10 + n).ok_or("recording header cut short")?).map_err(|e| format!("recording header: {e}"))?;
        let nodes: usize = h.bodies.iter().map(|x| x.nodes as usize).sum();
        let beams: usize = h.bodies.iter().map(|x| x.beams as usize).sum();
        let mut at = 10 + n;
        let want = at + 4 * (3 * nodes * h.samples as usize + beams);
        if b.len() != want {
            return Err(format!("recording is {} bytes, its header says {want}", b.len()));
        }
        let word = |i: usize| [b[i], b[i + 1], b[i + 2], b[i + 3]];
        let mut pos = Vec::with_capacity(h.samples as usize);
        for _ in 0..h.samples {
            pos.push((0..3 * nodes).map(|i| f32::from_le_bytes(word(at + 4 * i))).collect());
            at += 12 * nodes;
        }
        let broken_at = (0..beams).map(|i| u32::from_le_bytes(word(at + 4 * i))).collect();
        Ok(Self { scenario: h.scenario, hash: h.hash, executor: h.executor, contact: h.contact, run: h.run, impact_s: h.impact_s, sample_s: h.sample_s, bodies: h.bodies, pos, broken_at, report: h.report })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_round_trip_and_interpolate() {
        let mut r = Recording { scenario: "x".into(), hash: String::new(), executor: "cpu-fallback".into(), contact: Contact::Edge, run: 0, impact_s: 0.004, sample_s: 0.004, bodies: vec![RecBody { name: "car".into(), design: "d".into(), nodes: 2, beams: 1 }], pos: Vec::new(), broken_at: vec![u32::MAX], report: None };
        let s = |y: f32, broken: bool| Sample { t: 0.0, pos: vec![vec![Vec3::new(0.0, y, 0.0), Vec3::new(1.0, y, 0.0)]], vel: vec![], beams: vec![vec![BeamState { rest: 1.0, broken }]], wheel_spin: vec![] };
        r.push(&s(0.0, false));
        r.push(&s(1.0, true));
        r.push(&s(2.0, true));
        assert_eq!(r.broken_at, vec![1]);
        let b = Recording::from_bytes(&r.to_bytes()).unwrap();
        assert_eq!(b.pos, r.pos);
        assert_eq!(b.broken_at, r.broken_at);
        // 0 ms after a first contact at 4 ms is sample 1; 2 ms after it, half way to sample 2.
        assert_eq!(b.positions_at_ms(0.0)[0][0].y, 1.0);
        assert!((b.positions_at_ms(2.0)[0][1].y - 1.5).abs() < 1e-6);
        assert_eq!(b.positions_at_ms(1e6)[0][0].y, 2.0);
        assert_eq!(b.duration_s(), 0.008);
    }
}
