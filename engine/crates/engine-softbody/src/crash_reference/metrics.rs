//! What a crash run is measured by, computed on the host from sampled state, so every
//! executor is measured by the same code.
//!
//! Metric names: `<body>.<measure>` for a body, `all.<measure>` for the scene.
//!
//! | measure | meaning |
//! |---|---|
//! | `mass` | kg |
//! | `speed_before` | the centre of mass's speed at the start (m/s) |
//! | `v_along_end` | its velocity along its first heading at the end (m/s) |
//! | `rebound_ratio` | −`v_along_end` / `speed_before` (positive: bounced back) |
//! | `length0`, `crush_peak`, `crush_residual` | length along its first heading at the start; the most it lost; what it has lost at the end (m) |
//! | `bent_frac`, `broken_frac` | share of its beams yielded for good; broken |
//! | `plastic_work_j` | Σ yield threshold × permanent length change over its beams |
//! | `all.finite` | 1 when every sampled value was finite |
//! | `all.ke_start_j`, `all.ke_end_j` | kinetic energy |

use super::{Built, Sample, Templates};
use glam::Vec3;
use std::collections::BTreeMap;

struct BodyRec {
    name: &'static str,
    mass: Vec<f32>,
    heading: Vec3,
    /// Per beam: spawn rest length and yield thresholds in compression and tension (N).
    beams: Vec<(f32, f32, f32)>,
    length0: f32,
    crush_peak: f32,
}

pub struct Recorder {
    bodies: Vec<BodyRec>,
    first: Option<Sample>,
    last: Option<Sample>,
    finite: bool,
}

fn length_along(pos: &[Vec3], h: Vec3) -> f32 {
    let (lo, hi) = pos.iter().map(|p| p.dot(h)).fold((f32::MAX, f32::MIN), |(a, b), x| (a.min(x), b.max(x)));
    hi - lo
}

impl Recorder {
    pub fn new(_templates: &Templates, built: &Built) -> Self {
        let bodies = built
            .layout
            .bodies
            .iter()
            .zip(&built.fleet)
            .map(|(b, s)| BodyRec {
                name: b.name,
                mass: s.nodes.iter().map(|n| n.mass).collect(),
                heading: b.heading(),
                beams: s.beams.iter().map(|x| (x.rest0, x.tc, x.tt)).collect(),
                length0: 0.0,
                crush_peak: 0.0,
            })
            .collect();
        Self { bodies, first: None, last: None, finite: true }
    }

    pub fn push(&mut self, x: &Sample) {
        self.finite &= x.pos.iter().flatten().chain(x.vel.iter().flatten()).all(|p| p.is_finite());
        for (k, b) in self.bodies.iter_mut().enumerate() {
            let l = length_along(&x.pos[k], b.heading);
            if self.first.is_none() {
                b.length0 = l;
            }
            b.crush_peak = b.crush_peak.max(b.length0 - l);
        }
        if self.first.is_none() {
            self.first = Some(x.clone());
        }
        self.last = Some(x.clone());
    }

    pub fn finish(self) -> BTreeMap<String, f64> {
        let mut m = BTreeMap::new();
        let (Some(first), Some(last)) = (&self.first, &self.last) else { return m };
        let com_v = |x: &Sample, k: usize, mass: &[f32]| -> Vec3 {
            x.vel[k].iter().zip(mass).map(|(v, m)| *v * *m).sum::<Vec3>() / mass.iter().sum::<f32>()
        };
        let ke = |x: &Sample| -> f64 {
            self.bodies.iter().enumerate().map(|(k, b)| x.vel[k].iter().zip(&b.mass).map(|(v, m)| 0.5 * f64::from(*m) * f64::from(v.length_squared())).sum::<f64>()).sum()
        };
        for (k, b) in self.bodies.iter().enumerate() {
            let mut put = |n: &str, v: f64| {
                m.insert(format!("{}.{n}", b.name), v);
            };
            let v0 = com_v(first, k, &b.mass).length();
            let v1 = com_v(last, k, &b.mass).dot(b.heading);
            put("mass", f64::from(b.mass.iter().sum::<f32>()));
            put("speed_before", f64::from(v0));
            put("v_along_end", f64::from(v1));
            put("rebound_ratio", if v0 > 0.0 { f64::from(-v1 / v0) } else { 0.0 });
            put("length0", f64::from(b.length0));
            put("crush_peak", f64::from(b.crush_peak));
            put("crush_residual", f64::from(b.length0 - length_along(&last.pos[k], b.heading)));
            let n = b.beams.len().max(1) as f64;
            let (mut bent, mut broken, mut work) = (0usize, 0usize, 0.0f64);
            for (s, &(rest0, tc, tt)) in last.beams[k].iter().zip(&b.beams) {
                if s.broken {
                    broken += 1;
                }
                let d = s.rest - rest0;
                if d.abs() > 1e-3 * rest0.abs() {
                    bent += usize::from(!s.broken);
                    let t = if d < 0.0 { tc } else { tt };
                    if t.is_finite() {
                        work += f64::from(t.abs() * d.abs());
                    }
                }
            }
            put("bent_frac", bent as f64 / n);
            put("broken_frac", broken as f64 / n);
            put("plastic_work_j", work);
        }
        m.insert("all.finite".into(), f64::from(u8::from(self.finite)));
        m.insert("all.ke_start_j".into(), ke(first));
        m.insert("all.ke_end_j".into(), ke(last));
        m
    }
}
