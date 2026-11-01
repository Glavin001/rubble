//! Crash metrics shared by the explicit oracle, the GPU harnesses
//! (`soft_calibrate`, `soft_crash`) and the tests: penetration and
//! self-intersection counts, cabin intrusion, the crush profile across the
//! front, engine-block travel, time to stop and peak deceleration.
//!
//! Everything works on positions sampled once per frame, so the GPU (read back
//! each step) and the oracle (sampled every N substeps) are measured the same way.

use crate::calibrate::rigid_fit;
use crate::desc::SoftVehicleDesc;
use glam::{DVec3, Mat3, Vec3};
use std::collections::HashMap;

/// One vehicle's collision triangles and the nodes that test against them, as the
/// penetration tracker sees them.
#[derive(Clone, Debug, Default)]
pub struct TrackedBody {
    /// First node in the sampled position array.
    pub first: u32,
    pub nodes: u32,
    /// Triangles (node indices relative to `first`).
    pub tris: Vec<[u32; 3]>,
    /// Rest triangle edge lengths, for the tear test.
    pub tri_rest: Vec<[f32; 3]>,
    /// Nodes that collide with triangles (contacters).
    pub contacter: Vec<bool>,
    /// Per triangle, the vehicle's beams along its edges (beam index in the
    /// desc's packed order), when any: a triangle loses collision once one breaks.
    pub edge_beams: Vec<Vec<u32>>,
    /// Self pairs (node, triangle) excluded by the rest-pose rule
    /// ([`crate::explicit_ref::self_excluded`]) or because the node is a corner.
    pub excluded: std::collections::HashSet<(u32, u32)>,
}

impl TrackedBody {
    pub fn from_desc(desc: &SoftVehicleDesc, first: u32) -> Self {
        let rest: Vec<Vec3> = desc
            .nodes
            .iter()
            .map(|n| Vec3::from_array(n.position))
            .collect();
        let tris: Vec<[u32; 3]> = desc
            .triangles
            .iter()
            .filter(|t| t.collision)
            .map(|t| t.nodes)
            .collect();
        let tri_rest = tris
            .iter()
            .map(|t| {
                let p = t.map(|k| rest[k as usize]);
                [
                    (p[0] - p[1]).length(),
                    (p[1] - p[2]).length(),
                    (p[2] - p[0]).length(),
                ]
            })
            .collect();
        let mut by_edge: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        for (k, b) in desc.beams.iter().enumerate() {
            by_edge
                .entry((b.a.min(b.b), b.a.max(b.b)))
                .or_default()
                .push(k as u32);
        }
        let edge_beams = tris
            .iter()
            .map(|t| {
                (0..3)
                    .flat_map(|k| {
                        let (p, q) = (t[k], t[(k + 1) % 3]);
                        by_edge
                            .get(&(p.min(q), p.max(q)))
                            .cloned()
                            .unwrap_or_default()
                    })
                    .collect()
            })
            .collect();
        let mut excluded = std::collections::HashSet::new();
        for (ti, t) in tris.iter().enumerate() {
            let [a, b, c] = t.map(|k| rest[k as usize].as_dvec3());
            for (n, p) in rest.iter().enumerate() {
                if t.contains(&(n as u32))
                    || crate::explicit_ref::self_excluded(p.as_dvec3(), a, b, c)
                {
                    excluded.insert((n as u32, ti as u32));
                }
            }
        }
        Self {
            first,
            nodes: desc.nodes.len() as u32,
            tris,
            tri_rest,
            contacter: desc.nodes.iter().map(|n| n.contacter).collect(),
            edge_beams,
            excluded,
        }
    }
}

/// Counts of nodes on the wrong side of a triangle they were in front of.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PenetrationCount {
    /// Against another vehicle's triangles.
    pub external: u32,
    /// Against the node's own vehicle's triangles.
    pub self_: u32,
}

impl PenetrationCount {
    pub fn total(&self) -> u32 {
        self.external + self.self_
    }
}

/// Persistent (node, triangle) side cache, the metric twin of BeamNG's anti-clip:
/// a pair enters when a node comes within `range` of a triangle's face inside its
/// prism (or crosses it between samples), with the side it was on at the
/// previous sample. A pair whose node is now deeper than `tolerance` on the other
/// side, still inside the prism, is a penetration.
pub struct PenetrationTracker {
    pub bodies: Vec<TrackedBody>,
    pub range: f32,
    pub tolerance: f32,
    pairs: HashMap<(u32, u32, u32), f32>,
    prev: Vec<Vec3>,
    /// Largest count seen so far.
    pub max: PenetrationCount,
}

impl PenetrationTracker {
    pub fn new(bodies: Vec<TrackedBody>) -> Self {
        Self {
            bodies,
            range: 0.125,
            tolerance: 0.005,
            pairs: HashMap::new(),
            prev: Vec::new(),
            max: PenetrationCount::default(),
        }
    }

    /// Updates the pair cache from this sample and returns its counts. `broken`
    /// holds each body's broken-beam flags (desc order), when known: triangles
    /// along a broken beam no longer collide and are not counted.
    pub fn update(&mut self, x: &[Vec3], broken: Option<&[Vec<bool>]>) -> PenetrationCount {
        let mut count = PenetrationCount::default();
        if self.prev.len() != x.len() {
            self.prev = x.to_vec();
        }
        let mut live = HashMap::with_capacity(self.pairs.len());
        let range = self.range;
        // Triangles by body: (body, tri) → plane, bounds.
        for (tb, body) in self.bodies.iter().enumerate() {
            for (ti, t) in body.tris.iter().enumerate() {
                let idx = t.map(|k| (body.first + k) as usize);
                let [a, b, c] = idx.map(|k| x[k]);
                // Torn: stretched past 2.5× its rest edges (as the GPU).
                let r = body.tri_rest[ti];
                if let Some(br) = broken {
                    if body.edge_beams[ti].iter().any(|&e| br[tb][e as usize]) {
                        continue;
                    }
                }
                if (a - b).length() > 2.5 * r[0]
                    || (b - c).length() > 2.5 * r[1]
                    || (c - a).length() > 2.5 * r[2]
                {
                    continue;
                }
                let nr = (b - a).cross(c - a);
                let area = nr.length();
                if area < 1e-9 {
                    continue;
                }
                let n = nr / area;
                let [pa, pb, pc] = idx.map(|k| self.prev[k]);
                let pnr = (pb - pa).cross(pc - pa);
                let pn = pnr.normalize_or_zero();
                let lo = a.min(b).min(c) - Vec3::splat(range);
                let hi = a.max(b).max(c) + Vec3::splat(range);
                for (nb, other) in self.bodies.iter().enumerate() {
                    let same = nb == tb;
                    for k in 0..other.nodes {
                        if !other.contacter[k as usize]
                            || (same && body.excluded.contains(&(k, ti as u32)))
                        {
                            continue;
                        }
                        let gi = (other.first + k) as usize;
                        let p = x[gi];
                        if p.cmplt(lo).any() || p.cmpgt(hi).any() {
                            continue;
                        }
                        let d = (p - a).dot(n);
                        let foot = p - n * d;
                        let inside = in_prism(foot, a, b, c);
                        let key = (tb as u32, ti as u32, gi as u32);
                        let side = match self.pairs.get(&key) {
                            Some(&s) => s,
                            None => {
                                let dp = (self.prev[gi] - pa).dot(pn);
                                let crossed = dp.signum() != d.signum();
                                if !inside || (d.abs() > range && !crossed) {
                                    continue;
                                }
                                if crossed && pn != Vec3::ZERO {
                                    dp.signum()
                                } else {
                                    d.signum()
                                }
                            }
                        };
                        if d.abs() > range + 0.05 {
                            continue;
                        }
                        live.insert(key, side);
                        if inside && side * d < -self.tolerance {
                            if same {
                                count.self_ += 1;
                            } else {
                                count.external += 1;
                            }
                        }
                    }
                }
            }
        }
        self.pairs = live;
        self.prev.copy_from_slice(x);
        self.max.external = self.max.external.max(count.external);
        self.max.self_ = self.max.self_.max(count.self_);
        count
    }
}

/// Whether `p` (on the plane of abc) lies inside the triangle.
fn in_prism(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> bool {
    let (q, _) = crate::explicit_ref::closest_on_triangle(
        p.as_dvec3(),
        a.as_dvec3(),
        b.as_dvec3(),
        c.as_dvec3(),
    );
    (q - p.as_dvec3()).length_squared() < 1e-8
}

/// The vehicle frame of a sample: the rigid motion taking the rest positions of
/// `frame` nodes onto their current ones.
pub fn vehicle_frame(rest: &[Vec3], now: &[Vec3], frame: &[u32]) -> (Mat3, Vec3) {
    let a: Vec<Vec3> = frame.iter().map(|&i| rest[i as usize]).collect();
    let b: Vec<Vec3> = frame.iter().map(|&i| now[i as usize]).collect();
    rigid_fit(&a, &b)
}

/// Largest inward displacement (m) of `probes` toward the centroid of `cabin`, in
/// the frame of `cabin` (positive = into the cabin).
pub fn intrusion(rest: &[Vec3], now: &[Vec3], cabin: &[u32], probes: &[u32]) -> f32 {
    let (r, t) = vehicle_frame(rest, now, cabin);
    let centre = cabin.iter().map(|&i| rest[i as usize]).sum::<Vec3>() / cabin.len().max(1) as f32;
    probes
        .iter()
        .map(|&i| {
            let p0 = rest[i as usize];
            let inward = (centre - p0).normalize_or_zero();
            let fitted = r * p0 + t;
            (now[i as usize] - fitted).dot(r * inward)
        })
        .fold(0.0, f32::max)
}

/// Residual crush across the front: `bins` lateral strips over the rest width of
/// `nodes`; per strip, how far its most forward node (along `forward`) moved
/// back, in the frame of `frame` nodes (the rear of the car, which the crash
/// leaves whole). Strips with no node report NaN.
pub fn crush_profile(
    rest: &[Vec3],
    now: &[Vec3],
    nodes: &[u32],
    frame: &[u32],
    forward: Vec3,
    lateral: Vec3,
    bins: usize,
) -> Vec<f32> {
    let (r, t) = vehicle_frame(rest, now, frame);
    // Back to the rest frame: x_rest = Rᵀ (x − t).
    let rt = r.transpose();
    let (lo, hi) = nodes.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &i| {
        let s = rest[i as usize].dot(lateral);
        (lo.min(s), hi.max(s))
    });
    let width = (hi - lo).max(1e-6);
    let mut front0 = vec![f32::MIN; bins];
    let mut front1 = vec![f32::MIN; bins];
    for &i in nodes {
        let p0 = rest[i as usize];
        let bin = (((p0.dot(lateral) - lo) / width * bins as f32) as usize).min(bins - 1);
        front0[bin] = front0[bin].max(p0.dot(forward));
        let p1 = rt * (now[i as usize] - t);
        front1[bin] = front1[bin].max(p1.dot(forward));
    }
    front0
        .iter()
        .zip(&front1)
        .map(|(&a, &b)| {
            if a == f32::MIN {
                f32::NAN
            } else {
                (a - b).max(0.0)
            }
        })
        .collect()
}

/// Displacement (m) of the centroid of `block` relative to `cabin`, and its
/// component along −`forward` (positive = pushed back toward the cabin).
pub fn block_travel(
    rest: &[Vec3],
    now: &[Vec3],
    cabin: &[u32],
    block: &[u32],
    forward: Vec3,
) -> (f32, f32) {
    let (r, t) = vehicle_frame(rest, now, cabin);
    let c0 = block.iter().map(|&i| rest[i as usize]).sum::<Vec3>() / block.len().max(1) as f32;
    let c1 = block.iter().map(|&i| now[i as usize]).sum::<Vec3>() / block.len().max(1) as f32;
    let d = r.transpose() * (c1 - t) - c0;
    (d.length(), -d.dot(forward))
}

/// Velocity history of a cabin centroid: time to stop and peak deceleration.
#[derive(Clone, Debug, Default)]
pub struct DecelTracker {
    samples: Vec<(f64, DVec3)>,
}

impl DecelTracker {
    pub fn push(&mut self, t: f64, v: Vec3) {
        self.samples.push((t, v.as_dvec3()));
    }

    /// First time the speed falls below `slow` (m/s) after it was above it, from
    /// `t0` (the impact); `None` if it never stops.
    pub fn time_to_stop(&self, t0: f64, slow: f64) -> Option<f64> {
        self.samples
            .iter()
            .find(|(t, v)| *t >= t0 && v.length() < slow)
            .map(|(t, _)| t - t0)
    }

    /// Peak deceleration (in g) of the velocity averaged over `window` seconds
    /// (crash pulses are filtered like this; 10 ms is about a CFC-60 filter).
    pub fn peak_decel_g(&self, window: f64) -> f64 {
        let mut best: f64 = 0.0;
        for (i, &(t, v)) in self.samples.iter().enumerate() {
            let Some(&(t1, v1)) = self.samples[i..].iter().find(|(s, _)| *s >= t + window) else {
                break;
            };
            best = best.max((v - v1).length() / (t1 - t));
        }
        best / 9.81
    }
}

/// Mass-weighted centroid velocity of `nodes`.
pub fn centroid_velocity(v: &[Vec3], mass: &[f32], nodes: &[u32]) -> Vec3 {
    let (mut s, mut m) = (Vec3::ZERO, 0.0);
    for &i in nodes {
        s += v[i as usize] * mass[i as usize];
        m += mass[i as usize];
    }
    s / m.max(1e-9)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_tri() -> TrackedBody {
        TrackedBody {
            first: 0,
            nodes: 3,
            tris: vec![[0, 1, 2]],
            tri_rest: vec![[1.0, 1.5, 1.0]],
            contacter: vec![false; 3],
            edge_beams: vec![vec![]],
            excluded: Default::default(),
        }
    }

    fn probe(first: u32) -> TrackedBody {
        TrackedBody {
            first,
            nodes: 1,
            tris: vec![],
            tri_rest: vec![],
            contacter: vec![true],
            edge_beams: vec![],
            excluded: Default::default(),
        }
    }

    #[test]
    fn crossing_between_samples_counts_as_penetration() {
        let mut tr = PenetrationTracker::new(vec![one_tri(), probe(3)]);
        let tri = [Vec3::ZERO, Vec3::X, Vec3::Z];
        // Normal of (0, X, Z) = X × Z = −Y: the front face looks down.
        let mut x = tri.to_vec();
        x.push(Vec3::new(0.2, -0.5, 0.2));
        assert_eq!(tr.update(&x, None).total(), 0);
        x[3].y = -0.05; // approaches from the front: tracked, still fine
        assert_eq!(tr.update(&x, None).total(), 0);
        x[3].y = 0.04; // through to the back
        assert_eq!(tr.update(&x, None).external, 1);
        // A node arriving from the back in one jump is caught too.
        let mut tr = PenetrationTracker::new(vec![one_tri(), probe(3)]);
        x[3].y = -0.3;
        tr.update(&x, None);
        x[3].y = 0.03;
        assert_eq!(tr.update(&x, None).external, 1);
        assert_eq!(tr.max.external, 1);
    }

    #[test]
    fn crush_profile_reads_a_dented_corner() {
        // A 2 × 2 grid of "front" nodes plus a rear frame.
        let rest = vec![
            Vec3::new(-1.0, 0.0, 2.0),
            Vec3::new(1.0, 0.0, 2.0),
            Vec3::new(-1.0, 1.0, 2.0),
            Vec3::new(1.0, 1.0, 2.0),
            Vec3::new(-1.0, 0.0, -2.0),
            Vec3::new(1.0, 0.0, -2.0),
            Vec3::new(0.0, 1.0, -2.0),
            Vec3::new(0.0, 0.0, 0.0),
        ];
        let mut now: Vec<Vec3> = rest.iter().map(|p| *p + Vec3::new(5.0, 0.0, 3.0)).collect();
        now[1].z -= 0.3;
        now[3].z -= 0.3;
        let p = crush_profile(
            &rest,
            &now,
            &[0, 1, 2, 3],
            &[4, 5, 6, 7],
            Vec3::Z,
            Vec3::X,
            2,
        );
        assert!(p[0].abs() < 1e-4 && (p[1] - 0.3).abs() < 1e-4, "{p:?}");
    }

    #[test]
    fn decel_of_a_linear_stop() {
        let mut d = DecelTracker::default();
        for k in 0..=200 {
            let t = k as f64 * 0.001;
            // 15 m/s to 0 in 0.1 s: 150 m/s² ≈ 15.3 g.
            let v = (15.0 - 150.0 * t).max(0.0) as f32;
            d.push(t, Vec3::new(0.0, 0.0, v));
        }
        assert!((d.peak_decel_g(0.01) - 15.29).abs() < 0.1);
        assert!((d.time_to_stop(0.0, 0.5).unwrap() - 0.097).abs() < 0.002);
    }
}
