//! A schema's elements read off Rubble's part outlines (`outlines.json` from the
//! `crash_outlines` example, `super::super::outline`): each element names its rule in the
//! schema (`"rubble": {...}`), so a new structure kind needs no code here unless it needs a
//! new kind of rule.
//!
//! A body's outlines come in two views: `car_frame` (the car's own side frame, x forward,
//! y up, metres) and `camera:<name>` (pixels of that camera's projection, y down). Rules run
//! in structure axes (x forward, y up); a camera view is turned into them by flipping y and,
//! when the front wheel is left of the rear one, x.

use super::{Circle, Kind, Schema, Shape, Vis, P};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Deserialize)]
pub struct Envelope {
    pub groups: Vec<String>,
    /// "x": per bin along x, the point furthest in y; "y": per bin along y, furthest in x.
    pub along: String,
    /// "max" or "min" of the other coordinate.
    pub take: String,
    pub bins: usize,
    /// Only the front half of the points (by x): a windscreen without the roof behind it.
    #[serde(default)]
    pub front_half: bool,
}

/// How one element is read off the outlines.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rule {
    /// A wheel's centre and radius (circle).
    Wheel(String),
    /// The hood's centreline profile (polyline).
    Centreline(bool),
    /// The envelope of part groups on one side (polyline).
    Envelope(Envelope),
    /// The lowest front corner of a group as the cabin carries it (its ghost: the panel
    /// itself can bulge, the corner where it hangs does not move) (point).
    GhostCorner(String),
    /// The convex hull of groups' points, closed (polyline).
    Hull(Vec<String>),
}

/// A body's outlines in structure axes.
pub struct BodyView {
    now: BTreeMap<String, Vec<Vec<P>>>,
    ghost: BTreeMap<String, Vec<Vec<P>>>,
    centreline: Vec<P>,
    wheels: BTreeMap<String, Circle>,
    fit: BTreeMap<String, f64>,
    /// Structure axes from the view's: x × sx, y × sy.
    pub sx: f64,
    pub sy: f64,
}

fn pt(v: &Value) -> Option<P> {
    Some([v.get(0)?.as_f64()?, v.get(1)?.as_f64()?])
}

fn lines(v: &Value) -> Vec<Vec<P>> {
    v.as_array()
        .map(|ls| {
            ls.iter()
                .map(|l| {
                    l.as_array()
                        .map_or(vec![], |l| l.iter().filter_map(pt).collect())
                })
                .filter(|l: &Vec<P>| !l.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

impl BodyView {
    /// `body`: one entry of a moment's `bodies`; `view`: "car_frame" or "camera:<name>".
    pub fn new(body: &Value, view: &str) -> Result<Self, String> {
        let (src, camera) = match view.strip_prefix("camera:") {
            Some(c) => (body.pointer(&format!("/cameras/{c}")), true),
            None if view == "car_frame" => (body.get("car_frame"), false),
            None => return Err(format!("unknown view {view}")),
        };
        let src = src.ok_or(format!("no {view} in the outlines"))?;
        let mut wheels = BTreeMap::new();
        for w in src["wheels"].as_array().into_iter().flatten() {
            let name = w["name"].as_str().unwrap_or_default().to_string();
            let c = if camera {
                pt(&w["now"]["c"]).zip(w["now"]["r"].as_f64())
            } else {
                pt(&w["now"]).zip(w["r"].as_f64())
            };
            if let Some((c, r)) = c {
                wheels.insert(name, Circle { c, r });
            }
        }
        let fx = |pre: &str| {
            wheels
                .iter()
                .find(|(n, _)| n.starts_with(pre))
                .map(|(_, c)| c.c[0])
        };
        let sx = match (camera, fx("wheel_f"), fx("wheel_r")) {
            (true, Some(f), Some(r)) if f < r => -1.0,
            _ => 1.0,
        };
        let sy = if camera { -1.0 } else { 1.0 };
        let mut v = BodyView {
            now: BTreeMap::new(),
            ghost: BTreeMap::new(),
            centreline: vec![],
            wheels: BTreeMap::new(),
            fit: BTreeMap::new(),
            sx,
            sy,
        };
        let f = |p: P| [sx * p[0], sy * p[1]];
        for (g, x) in src["parts"].as_object().into_iter().flatten() {
            let map = |l: Vec<Vec<P>>| {
                l.into_iter()
                    .map(|l| l.into_iter().map(f).collect())
                    .collect()
            };
            v.now.insert(g.clone(), map(lines(&x["now"]["lines"])));
            v.ghost.insert(g.clone(), map(lines(&x["ghost"]["lines"])));
        }
        v.centreline = src["hood_centreline"]["now"]
            .as_array()
            .map_or(vec![], |l| l.iter().filter_map(pt).map(f).collect());
        v.wheels = wheels
            .into_iter()
            .map(|(n, c)| (n, Circle { c: f(c.c), r: c.r }))
            .collect();
        for (k, key) in [("rear_pitch_deg", "pitch"), ("rear_yaw_deg", "yaw")] {
            if let Some(x) = body[k].as_f64() {
                v.fit.insert(key.into(), x);
            }
        }
        Ok(v)
    }

    fn points(&self, groups: &[String], ghost: bool) -> Vec<P> {
        let src = if ghost { &self.ghost } else { &self.now };
        groups
            .iter()
            .filter_map(|g| src.get(g))
            .flat_map(|ls| ls.iter().flatten().copied())
            .collect()
    }

    /// Structure axes back to the view's coordinates.
    pub fn to_view(&self, p: P) -> P {
        [self.sx * p[0], self.sy * p[1]]
    }
}

/// Per bin of coordinate `along` (0: x, 1: y), the point with the largest (`take` > 0)
/// or smallest other coordinate, in order of `along` (`outlines.py`'s envelope).
pub fn envelope(points: &[P], along: usize, take: f64, bins: usize) -> Vec<P> {
    let other = 1 - along;
    let lo = points.iter().map(|p| p[along]).fold(f64::MAX, f64::min);
    let hi = points.iter().map(|p| p[along]).fold(f64::MIN, f64::max);
    if points.is_empty() || hi - lo < 1e-9 {
        return vec![];
    }
    (0..bins)
        .filter_map(|k| {
            let a = lo + (hi - lo) * k as f64 / bins as f64;
            let b = lo + (hi - lo) * (k + 1) as f64 / bins as f64;
            points
                .iter()
                .filter(|p| a - 1e-9 <= p[along] && p[along] <= b + 1e-9)
                .copied()
                .reduce(|x, y| {
                    if take * y[other] > take * x[other] {
                        y
                    } else {
                        x
                    }
                })
        })
        .collect()
}

/// Convex hull (Andrew's monotone chain), counter-clockwise, closed (first point last).
pub fn hull(points: &[P]) -> Vec<P> {
    let mut p = points.to_vec();
    p.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    p.dedup();
    if p.len() < 3 {
        return p;
    }
    let cross = |o: P, a: P, b: P| (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    let mut h: Vec<P> = Vec::new();
    for pass in 0..2 {
        let start = h.len();
        let it: Box<dyn Iterator<Item = &P>> = if pass == 0 {
            Box::new(p.iter())
        } else {
            Box::new(p.iter().rev())
        };
        for &q in it {
            while h.len() >= start + 2 && cross(h[h.len() - 2], h[h.len() - 1], q) <= 0.0 {
                h.pop();
            }
            h.push(q);
        }
        h.pop();
    }
    h.push(h[0]);
    h
}

fn apply(rule: &Rule, v: &BodyView, s: &mut Shape, name: &str) {
    match rule {
        Rule::Wheel(w) => {
            if let Some(c) = v.wheels.get(w) {
                s.circles.insert(name.into(), *c);
            }
        }
        Rule::Centreline(_) => {
            if !v.centreline.is_empty() {
                s.lines.insert(name.into(), v.centreline.clone());
            }
        }
        Rule::Envelope(e) => {
            let mut p = v.points(&e.groups, false);
            if e.front_half && !p.is_empty() {
                let mut xs: Vec<f64> = p.iter().map(|q| q[0]).collect();
                xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mid = xs[xs.len() / 2];
                p.retain(|q| q[0] >= mid);
            }
            let along = usize::from(e.along == "y");
            let take = if e.take == "min" { -1.0 } else { 1.0 };
            let mut l = envelope(&p, along, take, e.bins);
            if along == 0 {
                l.sort_by(|a, b| b[0].partial_cmp(&a[0]).unwrap_or(std::cmp::Ordering::Equal));
            }
            if !l.is_empty() {
                s.lines.insert(name.into(), l);
            }
        }
        Rule::GhostCorner(g) => {
            let p = v.points(std::slice::from_ref(g), true);
            let lo = p.iter().map(|q| q[1]).fold(f64::MAX, f64::min);
            let hi = p.iter().map(|q| q[1]).fold(f64::MIN, f64::max);
            let tol = 0.1 * (hi - lo);
            let c = p
                .iter()
                .filter(|q| q[1] <= lo + tol)
                .copied()
                .reduce(|a, b| if b[0] > a[0] { b } else { a });
            if let Some(c) = c {
                s.points.insert(name.into(), c);
            }
        }
        Rule::Hull(groups) => {
            let h = hull(&v.points(groups, false));
            if h.len() >= 3 {
                s.lines.insert(name.into(), h);
            }
        }
    }
}

/// The schema's elements on Rubble's body, in structure axes, with its part outlines as
/// loops and its fitted cabin pitch and yaw.
pub fn derive(schema: &Schema, v: &BodyView) -> Shape {
    let mut s = Shape::default();
    for e in &schema.elements {
        if let Some(r) = &e.rubble {
            apply(r, v, &mut s, &e.name);
            if s.has(&e.name) {
                s.vis.insert(e.name.clone(), Vis::Clear);
            }
        }
    }
    s.loops = v.now.clone();
    s.fit = v.fit.clone();
    s
}

/// A derived shape as a tracing instance over a `width` × `height` still of the view (the
/// studio draws it in the same schema as a hand tracing).
pub fn to_instance(schema: &Schema, s: &Shape, v: &BodyView, width: f64, height: f64) -> Value {
    let n = |p: P| {
        let q = v.to_view(p);
        json!([q[0] / width, q[1] / height])
    };
    let mut el = serde_json::Map::new();
    for e in &schema.elements {
        let val = match e.kind {
            Kind::Point => s
                .points
                .get(&e.name)
                .map(|&p| json!({ "p": n(p), "vis": "clear" })),
            Kind::Polyline => s.lines.get(&e.name).map(
                |l| json!({ "pts": l.iter().map(|&p| n(p)).collect::<Vec<_>>(), "vis": "clear" }),
            ),
            Kind::Circle => s.circles.get(&e.name).map(
                |c| json!({ "c": n(c.c), "rim": n([c.c[0] + c.r * v.sx, c.c[1]]), "vis": "clear" }),
            ),
        };
        if let Some(x) = val {
            el.insert(e.name.clone(), x);
        }
    }
    let loops: serde_json::Map<String, Value> = s
        .loops
        .iter()
        .map(|(g, ls)| {
            (
                g.clone(),
                json!(ls
                    .iter()
                    .map(|l| l.iter().map(|&p| n(p)).collect::<Vec<_>>())
                    .collect::<Vec<_>>()),
            )
        })
        .collect();
    let front = if v.sx < 0.0 { "left" } else { "right" };
    json!({ "schema": schema.id, "front": front, "elements": el, "loops": loops })
}
