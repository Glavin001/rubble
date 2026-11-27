//! Choosing a generated design from a traced side view: the tracing's proportions in
//! wheelbase units (front and rear overhang, height, wheel radius) against each design's
//! own, measured on its template (its gauges where `crash.json` has them, its rest shape
//! otherwise), ranked by a visibility-weighted distance. Ranks the existing designs only; it
//! does not propose generator parameters.

use super::{dist, Shape, Vis, P};
use crate::crash_reference::{Extent, Template};
use glam::Vec3;
use serde::Serialize;
use ts_rs::TS;

/// A design's proportions (m), from its template.
#[derive(Clone, Debug, Serialize, TS)]
pub struct DesignDims {
    pub design: String,
    pub wheelbase_m: f64,
    pub front_overhang_m: f64,
    pub rear_overhang_m: f64,
    /// Its top above the ground (the wheels' lowest point).
    pub height_m: f64,
    pub wheel_radius_m: f64,
    pub length_m: f64,
}

/// A gauge's rest length on the template (`crash.json`'s tape measure between two nodes).
fn gauge(t: &Template, name: &str) -> Option<f64> {
    let g = t.zones.gauges.iter().find(|g| g.name == name)?;
    let at = |n: &str| t.desc.nodes.iter().find(|x| x.name == n).map(|x| Vec3::from(x.position));
    Some(f64::from(at(&g.a)?.distance(at(&g.b)?)))
}

impl DesignDims {
    /// From the template: the wheelbase gauge (`wheelbase_left`) or the axles' spacing, the
    /// front overhang gauge or the front axle to the foremost node, the rearmost node to the
    /// rear axle, the topmost node above the wheels' lowest point, the mean wheel radius.
    /// None for a design without two axles (the barrier).
    pub fn of(design: &str, t: &Template) -> Option<Self> {
        let e = Extent::of(&t.desc);
        let (front, rear) = e.axles?;
        if front - rear < 0.1 {
            return None;
        }
        let wheelbase = gauge(t, "wheelbase_left").unwrap_or(f64::from(front - rear));
        let ground = e.wheel_low.unwrap_or(e.lo.y);
        let r = t.desc.wheels.iter().map(|w| f64::from(w.radius)).sum::<f64>() / t.desc.wheels.len().max(1) as f64;
        Some(Self {
            design: design.into(),
            wheelbase_m: wheelbase,
            front_overhang_m: gauge(t, "front_overhang").unwrap_or(f64::from(e.hi.z - front)),
            rear_overhang_m: f64::from(rear - e.lo.z),
            height_m: f64::from(e.hi.y - ground),
            wheel_radius_m: r,
            length_m: f64::from(e.length()),
        })
    }

    fn ratios(&self) -> [f64; 4] {
        let w = self.wheelbase_m;
        [self.front_overhang_m / w, self.rear_overhang_m / w, self.height_m / w, self.wheel_radius_m / w]
    }
}

/// The proportions compared, in order.
pub const RATIOS: [&str; 4] = ["front overhang / wheelbase", "rear overhang / wheelbase", "height / wheelbase", "wheel radius / wheelbase"];

/// One proportion read off a tracing, with how much it counts (its elements' worst
/// visibility: clear 1, blurred 0.5, hidden or guessed 0.25).
#[derive(Clone, Debug, Serialize, TS)]
pub struct TracedRatio {
    pub name: String,
    pub value: f64,
    pub weight: f64,
}

fn weight(v: Vis) -> f64 {
    match v {
        Vis::Clear => 1.0,
        Vis::Blurred => 0.5,
        Vis::Hidden => 0.25,
    }
}

/// A side view's proportions (structure axes: x toward the front, y up), from its wheels
/// (`front_wheel`, `rear_wheel` circles: the wheelbase is their centres' distance), its
/// front-most point (`bumper_front`, `hood`, `fender_top`), its rear-most (`rear_face`,
/// `roofline`, `sill`), its top (`roofline`, `windscreen`) above the ground (the wheels'
/// lowest point). Empty without both wheels.
pub fn traced_ratios(s: &Shape) -> Vec<TracedRatio> {
    let (Some(f), Some(r)) = (s.circles.get("front_wheel"), s.circles.get("rear_wheel")) else { return Vec::new() };
    let wb = dist(f.c, r.c);
    if wb < 1e-9 {
        return Vec::new();
    }
    let worst = |names: &[&str]| names.iter().filter(|n| s.has(n)).map(|n| s.vis_of(n)).max().unwrap_or_default();
    let pts = |names: &[&str]| -> Vec<P> { names.iter().flat_map(|n| s.lines.get(*n).into_iter().flatten().copied().chain(s.points.get(*n).copied())).collect() };
    let wheels = worst(&["front_wheel", "rear_wheel"]);
    let ground = (f.c[1] - f.r).min(r.c[1] - r.r);
    let mut out = Vec::new();
    let mut put = |i: usize, v: f64, vis: Vis| out.push(TracedRatio { name: RATIOS[i].into(), value: v / wb, weight: weight(vis.max(wheels)) });
    let front = ["bumper_front", "hood", "fender_top"];
    if let Some(p) = pts(&front).into_iter().max_by(|a, b| a[0].total_cmp(&b[0])) {
        put(0, p[0] - f.c[0], worst(&front));
    }
    let rear = ["rear_face", "roofline", "sill"];
    if let Some(p) = pts(&rear).into_iter().min_by(|a, b| a[0].total_cmp(&b[0])) {
        put(1, r.c[0] - p[0], worst(&rear));
    }
    let top = ["roofline", "windscreen"];
    if let Some(p) = pts(&top).into_iter().max_by(|a, b| a[1].total_cmp(&b[1])) {
        put(2, p[1] - ground, worst(&top));
    }
    put(3, 0.5 * (f.r + r.r), wheels);
    out
}

/// One design against the tracing: its proportions beside the traced ones, and the
/// distance (root of the weighted mean squared difference).
#[derive(Clone, Debug, Serialize, TS)]
pub struct DesignMatch {
    pub design: String,
    pub distance: f64,
    pub dims: DesignDims,
    /// Per traced proportion: its name, the traced value, the design's, the weight.
    pub rows: Vec<MatchRow>,
}

#[derive(Clone, Debug, Serialize, TS)]
pub struct MatchRow {
    pub name: String,
    pub traced: f64,
    pub design: f64,
    pub weight: f64,
}

/// Every design ranked against the traced proportions, nearest first.
pub fn rank(traced: &[TracedRatio], designs: &[DesignDims]) -> Vec<DesignMatch> {
    let mut out: Vec<DesignMatch> = designs
        .iter()
        .map(|d| {
            let r = d.ratios();
            let rows: Vec<MatchRow> = traced.iter().map(|t| MatchRow { name: t.name.clone(), traced: t.value, design: r[RATIOS.iter().position(|n| *n == t.name).unwrap_or(0)], weight: t.weight }).collect();
            let w: f64 = rows.iter().map(|x| x.weight).sum();
            let distance = if w > 0.0 { (rows.iter().map(|x| x.weight * (x.traced - x.design).powi(2)).sum::<f64>() / w).sqrt() } else { f64::INFINITY };
            DesignMatch { design: d.design.clone(), distance, dims: d.clone(), rows }
        })
        .collect();
    out.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    out
}

#[cfg(test)]
mod tests {
    use super::super::Circle;
    use super::*;

    fn dims(name: &str, f: f64, r: f64, h: f64) -> DesignDims {
        DesignDims { design: name.into(), wheelbase_m: 2.8, front_overhang_m: f * 2.8, rear_overhang_m: r * 2.8, height_m: h * 2.8, wheel_radius_m: 0.12 * 2.8, length_m: 0.0 }
    }

    /// A side view drawn to a design's proportions ranks that design first, with distance 0;
    /// a blurred element counts half.
    #[test]
    fn a_tracing_of_a_design_ranks_it_first() {
        let mut s = Shape::default();
        let (wb, rad) = (280.0, 0.12 * 280.0);
        s.circles.insert("rear_wheel".into(), Circle { c: [0.0, rad], r: rad });
        s.circles.insert("front_wheel".into(), Circle { c: [wb, rad], r: rad });
        s.lines.insert("bumper_front".into(), vec![[wb + 0.35 * wb, 0.5 * rad], [wb + 0.33 * wb, 2.0 * rad]]);
        s.lines.insert("rear_face".into(), vec![[-0.4 * wb, 1.0 * rad], [-0.38 * wb, 2.5 * rad]]);
        s.lines.insert("roofline".into(), vec![[0.0, 0.5 * wb], [0.6 * wb, 0.52 * wb]]);
        s.vis.insert("roofline".into(), Vis::Blurred);
        let t = traced_ratios(&s);
        assert_eq!(t.len(), 4);
        assert!((t[0].value - 0.35).abs() < 1e-9 && (t[1].value - 0.4).abs() < 1e-9 && (t[2].value - 0.52).abs() < 1e-9);
        assert_eq!(t[2].weight, 0.5);
        let ranked = rank(&t, &[dims("long", 0.6, 0.7, 0.5), dims("this", 0.35, 0.4, 0.52), dims("tall", 0.35, 0.4, 0.8)]);
        assert_eq!(ranked[0].design, "this");
        assert!(ranked[0].distance < 1e-9);
        assert_eq!(ranked[1].design, "tall");
    }
}
