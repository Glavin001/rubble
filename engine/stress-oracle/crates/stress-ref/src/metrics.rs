//! Benchmark metrics, computed identically from any solver's [`Observation`].

use serde::Serialize;

use crate::math::{Quat, Vec3};
use crate::observation::{ChunkObservation, Observation};
use crate::scene::{BodyDesc, MetricDesc, MetricKind, Region, Scene, Tolerance};

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum MetricValue {
    Number(f64),
    Bool(bool),
    Category(String),
}

impl MetricValue {
    pub fn from_json(v: &serde_json::Value) -> Option<MetricValue> {
        match v {
            serde_json::Value::Number(n) => n.as_f64().map(MetricValue::Number),
            serde_json::Value::Bool(b) => Some(MetricValue::Bool(*b)),
            serde_json::Value::String(s) => Some(MetricValue::Category(s.clone())),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            MetricValue::Number(x) => Some(*x),
            MetricValue::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            MetricValue::Category(_) => None,
        }
    }
}

impl std::fmt::Display for MetricValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetricValue::Number(x) => write!(f, "{x:.6e}"),
            MetricValue::Bool(b) => write!(f, "{b}"),
            MetricValue::Category(s) => write!(f, "{s}"),
        }
    }
}

fn probe<'a>(obs: &'a Observation, name: &str) -> Result<&'a crate::observation::ProbeSeries, String> {
    obs.probes.get(name).filter(|p| !p.t.is_empty()).ok_or_else(|| format!("probe '{name}' missing from observation"))
}

fn body_states<'a>(scene: &'a Scene, obs: &'a Observation, body: &str) -> Result<(&'a BodyDesc, &'a [ChunkObservation]), String> {
    let (_, desc) = scene.body(body).ok_or_else(|| format!("unknown body '{body}'"))?;
    let states = obs.bodies.get(body).ok_or_else(|| format!("body '{body}' missing from observation"))?;
    if states.len() != desc.chunks.len() {
        return Err(format!("body '{body}': observation has {} chunks, scene has {}", states.len(), desc.chunks.len()));
    }
    Ok((desc, states))
}

fn chunk_mass(scene: &Scene, body: &BodyDesc, i: usize) -> f64 {
    let c = &body.chunks[i];
    8.0 * c.half_extents.iter().product::<f64>() * scene.material(&c.material).density
}

/// Coarse (level-0) chunks only: oracles see the structural level.
fn coarse(body: &BodyDesc) -> impl Iterator<Item = usize> + '_ {
    (0..body.chunks.len()).filter(|&i| body.chunks[i].level == 0)
}

fn in_region(body: &BodyDesc, i: usize, region: &Option<Region>) -> bool {
    region.map(|r| r.contains(Vec3::from_array(body.chunks[i].center))).unwrap_or(true)
}

/// Mass-weighted mean velocity along `axis` of the chunks that started in `region`.
fn region_speed(scene: &Scene, desc: &BodyDesc, st: &[ChunkObservation], region: &Region, axis: [f64; 3]) -> f64 {
    let a = Vec3::from_array(axis).normalized();
    let (mut p, mut m) = (0.0, 0.0);
    for i in coarse(desc).filter(|&i| !st[i].removed && region.contains(Vec3::from_array(desc.chunks[i].center))) {
        let mass = chunk_mass(scene, desc, i);
        p += mass * Vec3::from_array(st[i].velocity).dot(a);
        m += mass;
    }
    if m > 0.0 {
        p / m
    } else {
        f64::NAN
    }
}

/// Area of a box chunk projected on the plane perpendicular to `axis` (body frame).
pub fn projected_area(body: &BodyDesc, i: usize, axis: Vec3) -> f64 {
    let c = &body.chunks[i];
    let r = Quat::from_wxyz(c.orientation).to_mat3();
    let h = c.half_extents;
    let a = axis.normalized();
    let face = [4.0 * h[1] * h[2], 4.0 * h[0] * h[2], 4.0 * h[0] * h[1]];
    (0..3).map(|k| a.dot(r.col(k)).abs() * face[k]).sum()
}

/// "hole": the supported piece keeps at least half the mass but chunks in the impact
/// region came off; "push_over": the supported piece kept less than half the mass;
/// "intact": nothing in the impact region detached.
pub fn failure_mode(scene: &Scene, body: &BodyDesc, states: &[ChunkObservation], impact: &Region) -> String {
    let mut total = 0.0;
    let mut attached = 0.0;
    let mut hole = false;
    for i in coarse(body) {
        if states[i].removed {
            continue;
        }
        let m = chunk_mass(scene, body, i);
        total += m;
        if states[i].detached {
            if impact.contains(Vec3::from_array(body.chunks[i].center)) {
                hole = true;
            }
        } else {
            attached += m;
        }
    }
    if total > 0.0 && attached < 0.5 * total {
        "push_over".into()
    } else if hole {
        "hole".into()
    } else {
        "intact".into()
    }
}

/// Dominant frequency (Hz) from zero crossings of `v - mean` after `after`.
pub fn zero_crossing_frequency(t: &[f64], v: &[f64], after: f64) -> Option<f64> {
    let idx: Vec<usize> = (0..t.len()).filter(|&i| t[i] >= after).collect();
    if idx.len() < 4 {
        return None;
    }
    let mean = idx.iter().map(|&i| v[i]).sum::<f64>() / idx.len() as f64;
    let mut crossings = Vec::new();
    for w in idx.windows(2) {
        let (a, b) = (v[w[0]] - mean, v[w[1]] - mean);
        if a == 0.0 || a * b < 0.0 {
            let f = if a == b { 0.0 } else { a / (a - b) };
            crossings.push(t[w[0]] + f * (t[w[1]] - t[w[0]]));
        }
    }
    if crossings.len() < 3 {
        return None;
    }
    let span = crossings[crossings.len() - 1] - crossings[0];
    Some((crossings.len() - 1) as f64 / (2.0 * span))
}

/// First time |value| reaches `threshold` (linear interpolation between samples).
pub fn arrival(s: &crate::observation::ProbeSeries, threshold: f64) -> Option<f64> {
    for i in 0..s.t.len() {
        let mag = if s.has_envelope() { s.v[i].abs().max(s.lo[i].abs()).max(s.hi[i].abs()) } else { s.v[i].abs() };
        if mag >= threshold {
            if i == 0 {
                return Some(s.t[0]);
            }
            let (a, b) = (s.v[i - 1].abs(), s.v[i].abs());
            let f = if b > a { ((threshold - a) / (b - a)).clamp(0.0, 1.0) } else { 1.0 };
            return Some(s.t[i - 1] + f * (s.t[i] - s.t[i - 1]));
        }
    }
    None
}

/// Evaluate one metric on an observation.
pub fn evaluate(scene: &Scene, metric: &MetricDesc, obs: &Observation) -> Result<MetricValue, String> {
    Ok(match &metric.kind {
        MetricKind::ProbeFinal { probe: p } => MetricValue::Number(*probe(obs, p)?.v.last().unwrap()),
        MetricKind::ProbePeak { probe: p } => {
            let s = probe(obs, p)?;
            let (lo, hi) = s.range_after(f64::NEG_INFINITY).unwrap();
            MetricValue::Number(lo.abs().max(hi.abs()))
        }
        MetricKind::ProbeMax { probe: p } => MetricValue::Number(probe(obs, p)?.range_after(f64::NEG_INFINITY).unwrap().1),
        MetricKind::ProbeMin { probe: p } => MetricValue::Number(probe(obs, p)?.range_after(f64::NEG_INFINITY).unwrap().0),
        MetricKind::ProbePeakRatioOf { probe: p, reference } => {
            let peak = |name: &str| -> Result<f64, String> {
                let (lo, hi) = probe(obs, name)?.range_after(f64::NEG_INFINITY).unwrap();
                Ok(lo.abs().max(hi.abs()))
            };
            MetricValue::Number(peak(p)? / peak(reference)?)
        }
        MetricKind::WaveSpeed { probe_a, probe_b, distance, threshold } => {
            let ta = arrival(probe(obs, probe_a)?, *threshold).ok_or_else(|| format!("probe '{probe_a}' never reached {threshold}"))?;
            let tb = arrival(probe(obs, probe_b)?, *threshold).ok_or_else(|| format!("probe '{probe_b}' never reached {threshold}"))?;
            MetricValue::Number(distance / (tb - ta))
        }
        MetricKind::ProbePeakRatio { probe: p, reference_time } => {
            let s = probe(obs, p)?;
            let (lo, hi) = s.range_after(f64::NEG_INFINITY).unwrap();
            let r = s.at(*reference_time).unwrap().abs();
            MetricValue::Number(lo.abs().max(hi.abs()) / r)
        }
        MetricKind::ProbeAt { probe: p, time } => MetricValue::Number(probe(obs, p)?.at(*time).unwrap()),
        MetricKind::ProbeArrival { probe: p, threshold } => MetricValue::Number(
            arrival(probe(obs, p)?, *threshold).ok_or_else(|| format!("probe '{p}' never reached {threshold}"))?,
        ),
        MetricKind::ProbeFrequency { probe: p, after } => {
            let s = probe(obs, p)?;
            MetricValue::Number(
                zero_crossing_frequency(&s.t, &s.v, *after).ok_or_else(|| format!("probe '{p}': no oscillation"))?,
            )
        }
        MetricKind::ProbeDrop { probe: p } => {
            let s = probe(obs, p)?;
            MetricValue::Number(s.v[0] - s.v[s.v.len() - 1])
        }
        MetricKind::DynamicAmplification { probe: p, before } => {
            let s = probe(obs, p)?;
            let v0 = s.at(*before).unwrap();
            let v1 = *s.v.last().unwrap();
            let (lo, hi) = s.range_after(*before).ok_or("no samples after the load change")?;
            let peak = if v1 >= v0 { hi } else { lo };
            MetricValue::Number((peak - v0) / (v1 - v0))
        }
        MetricKind::DetachedArea { body, region, axis } => {
            let (desc, st) = body_states(scene, obs, body)?;
            let a = Vec3::from_array(*axis);
            MetricValue::Number(
                coarse(desc)
                    .filter(|&i| st[i].detached && !st[i].removed && in_region(desc, i, region))
                    .map(|i| projected_area(desc, i, a))
                    .sum(),
            )
        }
        MetricKind::DetachedMass { body, region } => {
            let (desc, st) = body_states(scene, obs, body)?;
            MetricValue::Number(
                coarse(desc)
                    .filter(|&i| st[i].detached && !st[i].removed && in_region(desc, i, region))
                    .map(|i| chunk_mass(scene, desc, i))
                    .sum(),
            )
        }
        MetricKind::DetachedAny { body, region } => {
            let (desc, st) = body_states(scene, obs, body)?;
            MetricValue::Bool(coarse(desc).any(|i| st[i].detached && !st[i].removed && in_region(desc, i, region)))
        }
        MetricKind::DetachedSpeed { body, region, axis, max } => {
            // Mass-weighted mean (momentum over mass) of the detached chunks, or the max.
            let (desc, st) = body_states(scene, obs, body)?;
            let a = Vec3::from_array(*axis).normalized();
            let pieces: Vec<(f64, f64)> = coarse(desc)
                .filter(|&i| st[i].detached && !st[i].removed && in_region(desc, i, region))
                .map(|i| (chunk_mass(scene, desc, i), Vec3::from_array(st[i].velocity).dot(a)))
                .collect();
            let mass: f64 = pieces.iter().map(|p| p.0).sum();
            MetricValue::Number(if pieces.is_empty() {
                0.0
            } else if *max {
                pieces.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max)
            } else {
                pieces.iter().map(|p| p.0 * p.1).sum::<f64>() / mass
            })
        }
        MetricKind::RegionSpeed { body, region, axis } => {
            let (desc, st) = body_states(scene, obs, body)?;
            MetricValue::Number(region_speed(scene, desc, st, region, *axis))
        }
        MetricKind::Separating { body, outer, inner, axis, threshold } => {
            let (desc, st) = body_states(scene, obs, body)?;
            let gap = region_speed(scene, desc, st, outer, *axis) - region_speed(scene, desc, st, inner, *axis);
            MetricValue::Bool(gap > *threshold)
        }
        MetricKind::FailureMode { body, impact_region } => {
            let (desc, st) = body_states(scene, obs, body)?;
            MetricValue::Category(failure_mode(scene, desc, st, impact_region))
        }
        MetricKind::Value { key } => {
            MetricValue::Number(*obs.values.get(key).ok_or_else(|| format!("value '{key}' missing from observation"))?)
        }
        MetricKind::Flag { flag } => {
            MetricValue::Bool(*obs.flags.get(flag).ok_or_else(|| format!("flag '{flag}' missing from observation"))?)
        }
    })
}

/// Outcome of comparing our metric with a reference value.
#[derive(Clone, Debug, Serialize)]
pub struct Comparison {
    pub metric: String,
    pub reference_source: String,
    pub ours: Option<MetricValue>,
    pub reference: Option<MetricValue>,
    /// Relative difference for numbers.
    pub relative_error: Option<f64>,
    /// None when the tolerance is `Report` or a value is missing.
    pub pass: Option<bool>,
    pub note: String,
}

pub fn compare_values(tol: &Tolerance, ours: &MetricValue, reference: &MetricValue) -> (Option<bool>, Option<f64>) {
    match (ours, reference) {
        (MetricValue::Number(a), MetricValue::Number(b)) => {
            let rel = if *b != 0.0 { (a - b).abs() / b.abs() } else { (a - b).abs() };
            let pass = match tol {
                Tolerance::Relative(t) => Some(rel <= *t),
                Tolerance::Absolute(t) => Some((a - b).abs() <= *t),
                Tolerance::Exact => Some(a == b),
                Tolerance::Report => None,
            };
            (pass, Some(rel))
        }
        (a, b) => match tol {
            Tolerance::Report => (None, None),
            _ => (Some(a == b), None),
        },
    }
}

/// Comparison against several independent oracles that disagree among themselves
/// (different formulations: continuum, rigid blocks, DEM): ours passes if it lies in
/// their spread widened by the tolerance (numbers), or equals one of their outcomes
/// (booleans, categories; all of them when they agree). Returns (pass, low, high).
pub fn ensemble(tol: &Tolerance, ours: &MetricValue, oracles: &[MetricValue]) -> (Option<bool>, Option<MetricValue>, Option<MetricValue>) {
    let numbers: Vec<f64> = oracles.iter().filter_map(|v| if let MetricValue::Number(x) = v { Some(*x) } else { None }).collect();
    if let (MetricValue::Number(x), true) = (ours, numbers.len() == oracles.len()) {
        let x = *x;
        let lo = numbers.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = numbers.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let (lo_w, hi_w) = match tol {
            Tolerance::Relative(r) => (lo - r * lo.abs(), hi + r * hi.abs()),
            Tolerance::Absolute(a) => (lo - a, hi + a),
            Tolerance::Exact | Tolerance::Report => (lo, hi),
        };
        let pass = (*tol != Tolerance::Report).then_some(x >= lo_w && x <= hi_w);
        return (pass, Some(MetricValue::Number(lo)), Some(MetricValue::Number(hi)));
    }
    (Some(oracles.contains(ours)), oracles.first().cloned(), None)
}

/// Compare every metric of `scene` for `ours` against analytic expectations and the
/// given oracle observations.
pub fn compare(scene: &Scene, ours: &Observation, oracles: &[Observation]) -> Vec<Comparison> {
    let mut out = Vec::new();
    for m in &scene.metrics {
        let our_val = evaluate(scene, m, ours);
        // The analytic row (if any) comes first, then one row per oracle.
        let first_oracle_row = out.len() + usize::from(m.expected.is_some());
        let mut push = |src: &str, reference: Result<MetricValue, String>| {
            let (pass, rel, note) = match (&our_val, &reference) {
                (Ok(a), Ok(b)) => {
                    let (p, r) = compare_values(&m.tolerance_for(src), a, b);
                    (p, r, String::new())
                }
                (Err(e), _) => (Some(false), None, format!("ours: {e}")),
                (_, Err(e)) => (None, None, format!("reference: {e}")),
            };
            out.push(Comparison {
                metric: m.name.clone(),
                reference_source: src.into(),
                ours: our_val.clone().ok(),
                reference: reference.ok(),
                relative_error: rel,
                pass,
                note,
            });
        };
        if let Some(exp) = &m.expected {
            push("analytic", MetricValue::from_json(exp).ok_or_else(|| "bad expected value".to_string()));
        }
        let mut compared = m.expected.is_some();
        let mut oracle_values = Vec::new();
        for o in oracles {
            if m.oracles.is_empty() || m.oracles.iter().any(|x| x == &o.solver) {
                let v = evaluate(scene, m, o);
                if let Ok(v) = &v {
                    oracle_values.push(v.clone());
                }
                push(&o.solver, v);
                compared = true;
            }
        }
        // Several independent oracles: their spread is the reference (see `ensemble`);
        // the single-oracle rows stay as information.
        if oracle_values.len() >= 2 && m.tolerance_for("oracles") != Tolerance::Report {
            if let Ok(ours) = &our_val {
                let (pass, lo, hi) = ensemble(&m.tolerance_for("oracles"), ours, &oracle_values);
                for row in &mut out[first_oracle_row..] {
                    if row.pass.is_some() {
                        row.pass = None;
                        row.note = "(see ensemble)".into();
                    }
                }
                out.push(Comparison {
                    metric: m.name.clone(),
                    reference_source: format!("{} oracles", oracle_values.len()),
                    ours: Some(ours.clone()),
                    reference: lo,
                    relative_error: None,
                    pass,
                    note: hi.map(|h| format!("range up to {h}")).unwrap_or_default(),
                });
            }
        }
        if !compared {
            // No reference available: still show our value.
            out.push(Comparison {
                metric: m.name.clone(),
                reference_source: "none".into(),
                ours: our_val.clone().ok(),
                reference: None,
                relative_error: None,
                pass: None,
                note: our_val.as_ref().err().cloned().unwrap_or_default(),
            });
        }
    }
    out
}

/// Compare metric distributions over seeds (fracture is chaotic: compare statistics,
/// not single crack patterns). Numbers: the mean is compared with the metric's
/// tolerance and the standard deviations are reported. Booleans and categories: the
/// most frequent outcome must agree and its frequency may differ by at most 0.25.
pub fn compare_distributions(scene: &Scene, ours: &[Observation], oracle: &[Observation], source: &str) -> Vec<Comparison> {
    let mut out = Vec::new();
    for m in &scene.metrics {
        if !m.oracles.is_empty() && !m.oracles.iter().any(|x| x == source) {
            continue;
        }
        let a: Vec<MetricValue> = ours.iter().filter_map(|o| evaluate(scene, m, o).ok()).collect();
        let b: Vec<MetricValue> = oracle.iter().filter_map(|o| evaluate(scene, m, o).ok()).collect();
        if a.is_empty() || b.is_empty() {
            continue;
        }
        let numeric = |v: &[MetricValue]| -> Option<Vec<f64>> {
            v.iter().map(|x| if let MetricValue::Number(n) = x { Some(*n) } else { None }).collect()
        };
        let (pass, rel, ours_v, ref_v, note) = match (numeric(&a), numeric(&b)) {
            (Some(x), Some(y)) => {
                let stats = |v: &[f64]| {
                    let n = v.len() as f64;
                    let mean = v.iter().sum::<f64>() / n;
                    (mean, (v.iter().map(|q| (q - mean).powi(2)).sum::<f64>() / n).sqrt())
                };
                let ((ma, sa), (mb, sb)) = (stats(&x), stats(&y));
                let (p, r) = compare_values(&m.tolerance_for(source), &MetricValue::Number(ma), &MetricValue::Number(mb));
                (p, r, MetricValue::Number(ma), MetricValue::Number(mb), format!("std {sa:.3e} vs {sb:.3e}, n {} vs {}", x.len(), y.len()))
            }
            _ => {
                let mode = |v: &[MetricValue]| {
                    let mut counts: Vec<(String, usize)> = Vec::new();
                    for x in v {
                        let k = x.to_string();
                        match counts.iter_mut().find(|c| c.0 == k) {
                            Some(c) => c.1 += 1,
                            None => counts.push((k, 1)),
                        }
                    }
                    counts.sort_by_key(|p| std::cmp::Reverse(p.1));
                    (counts[0].0.clone(), counts[0].1 as f64 / v.len() as f64)
                };
                let ((ka, fa), (kb, fb)) = (mode(&a), mode(&b));
                let pass = match m.tolerance {
                    Tolerance::Report => None,
                    _ => Some(ka == kb && (fa - fb).abs() <= 0.25),
                };
                (pass, None, MetricValue::Category(format!("{ka} ({:.0}%)", 100.0 * fa)), MetricValue::Category(format!("{kb} ({:.0}%)", 100.0 * fb)), String::new())
            }
        };
        out.push(Comparison {
            metric: m.name.clone(),
            reference_source: format!("{source} x{}", b.len()),
            ours: Some(ours_v),
            reference: Some(ref_v),
            relative_error: rel,
            pass,
            note,
        });
    }
    out
}

/// Pretty table of comparisons.
pub fn format_table(scene: &str, rows: &[Comparison]) -> String {
    let mut s = format!("scene {scene}\n");
    for r in rows {
        let status = match r.pass {
            Some(true) => "PASS",
            Some(false) => "FAIL",
            None => "----",
        };
        let fmt = |v: &Option<MetricValue>| v.as_ref().map(|x| x.to_string()).unwrap_or_else(|| "-".into());
        let rel = r.relative_error.map(|e| format!("{:6.2}%", 100.0 * e)).unwrap_or_else(|| "     -".into());
        s += &format!(
            "  [{status}] {:<28} vs {:<14} ours {:<14} ref {:<14} diff {rel} {}\n",
            r.metric,
            r.reference_source,
            fmt(&r.ours),
            fmt(&r.reference),
            r.note
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensemble_accepts_the_oracle_spread_widened_by_the_tolerance() {
        let oracles = [MetricValue::Number(2.0), MetricValue::Number(4.0)];
        let tol = Tolerance::Relative(0.25);
        assert_eq!(ensemble(&tol, &MetricValue::Number(1.6), &oracles).0, Some(true));
        assert_eq!(ensemble(&tol, &MetricValue::Number(4.9), &oracles).0, Some(true));
        assert_eq!(ensemble(&tol, &MetricValue::Number(1.4), &oracles).0, Some(false));
        assert_eq!(ensemble(&tol, &MetricValue::Number(5.1), &oracles).0, Some(false));
        let modes = [MetricValue::Category("hole".into()), MetricValue::Category("push_over".into())];
        assert_eq!(ensemble(&Tolerance::Exact, &MetricValue::Category("hole".into()), &modes).0, Some(true));
        assert_eq!(ensemble(&Tolerance::Exact, &MetricValue::Category("intact".into()), &modes).0, Some(false));
    }

    #[test]
    fn frequency_of_a_sine() {
        let f = 3.7;
        let t: Vec<f64> = (0..5000).map(|i| i as f64 * 1e-3).collect();
        let v: Vec<f64> = t.iter().map(|t| 0.2 + (2.0 * std::f64::consts::PI * f * t).sin()).collect();
        let est = zero_crossing_frequency(&t, &v, 0.0).unwrap();
        assert!((est - f).abs() / f < 2e-3, "{est}");
    }
}
