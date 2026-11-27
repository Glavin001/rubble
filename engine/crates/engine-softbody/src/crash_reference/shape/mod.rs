//! Shapes traced on reference images and derived from Rubble's outlines, put in one frame
//! and measured by the same code for every source (the Reference Studio,
//! `engine/tools/reference-studio/`; `engine/tools/crash-reference/outlines.py`).
//!
//! A structure is described by data, not code: a [`Schema`] (one JSON file in
//! `engine/tools/reference-studio/structures/`) names its elements (keypoints, polylines and
//! circles), the frames that normalise a shape (an origin and a unit, first one that every
//! source has wins), the measures that apply and how each element is read off Rubble's part
//! outlines ([`rubble`]). A tracing is an [`Instance`]: per element its points in the
//! region's normalised coordinates (x / width, y / height, from the top-left corner) and a
//! visibility flag.
//!
//! [`Shape`] is a source at one moment in structure axes: x toward the structure's front
//! (or image right when it has none), y up. [`compare`] normalises each source by its own
//! "before" shape and evaluates the schema's measures ([`measures`]).

pub mod link;
pub mod measures;
pub mod rubble;
#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub type P = [f64; 2];

/// How well an element can be seen; a measure reports the worst of its inputs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Vis {
    #[default]
    Clear,
    Blurred,
    /// Hidden or guessed.
    Hidden,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Point,
    Polyline,
    Circle,
}

/// One element of a schema.
#[derive(Clone, Debug, Deserialize)]
pub struct ElementDef {
    pub name: String,
    #[serde(rename = "type")]
    pub kind: Kind,
    /// How Rubble's outlines give it ([`rubble::Rule`]); absent: Rubble has no such element.
    #[serde(default)]
    pub rubble: Option<rubble::Rule>,
}

/// A point of a shape that a frame or a measure refers to.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Anchor {
    /// A circle's centre.
    Centre(String),
    /// A keypoint.
    Point(String),
    /// A polyline's lowest point (smallest y).
    Lowest(String),
    /// A polyline's rearmost point (smallest x).
    Rearmost(String),
    /// A polyline's frontmost point (largest x).
    Frontmost(String),
    /// A polyline's mean point.
    Centroid(String),
}

impl Anchor {
    pub fn element(&self) -> &str {
        match self {
            Anchor::Centre(n)
            | Anchor::Point(n)
            | Anchor::Lowest(n)
            | Anchor::Rearmost(n)
            | Anchor::Frontmost(n)
            | Anchor::Centroid(n) => n,
        }
    }

    pub fn of(&self, s: &Shape) -> Option<P> {
        let line = |n: &str| s.lines.get(n).filter(|l| !l.is_empty());
        let by = |n: &str, f: &dyn Fn(&P) -> f64, max: bool| {
            line(n).and_then(|l| {
                l.iter().copied().reduce(|a, b| {
                    if (f(&b) > f(&a)) == max && f(&b) != f(&a) {
                        b
                    } else {
                        a
                    }
                })
            })
        };
        match self {
            Anchor::Centre(n) => s.circles.get(n).map(|c| c.c),
            Anchor::Point(n) => s.points.get(n).copied(),
            Anchor::Lowest(n) => by(n, &|p| p[1], false),
            Anchor::Rearmost(n) => by(n, &|p| p[0], false),
            Anchor::Frontmost(n) => by(n, &|p| p[0], true),
            Anchor::Centroid(n) => line(n).map(|l| {
                let k = l.len() as f64;
                [
                    l.iter().map(|p| p[0]).sum::<f64>() / k,
                    l.iter().map(|p| p[1]).sum::<f64>() / k,
                ]
            }),
        }
    }
}

/// A frame's unit, from a source's before shape.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Unit {
    /// The distance between two anchors' elements (circle centres or keypoints).
    Distance([String; 2]),
    /// A circle's radius.
    Radius(String),
    /// A polyline's chord (first to last point).
    Chord(String),
}

/// Origin and unit of the common frame.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FrameDef {
    pub id: String,
    #[serde(default)]
    pub label: String,
    pub origin: Anchor,
    pub unit: Unit,
}

/// A structure kind as data (`structures/<id>.json`).
#[derive(Clone, Debug, Deserialize)]
pub struct Schema {
    pub id: String,
    #[serde(default)]
    pub elements: Vec<ElementDef>,
    #[serde(default)]
    pub frames: Vec<FrameDef>,
    #[serde(default)]
    pub measures: Vec<measures::MeasureDef>,
}

impl Schema {
    pub fn element(&self, name: &str) -> Option<&ElementDef> {
        self.elements.iter().find(|e| e.name == name)
    }
}

/// One element's traced value. Points are in the region's normalised coordinates.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ElementValue {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p: Option<P>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pts: Option<Vec<P>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub c: Option<P>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rim: Option<P>,
    #[serde(default)]
    pub vis: Vis,
}

/// A traced instance of a structure in one region.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct Instance {
    /// "left" or "right": where the structure's front is in the image (default right).
    #[serde(default)]
    pub front: Option<String>,
    #[serde(default)]
    pub elements: BTreeMap<String, ElementValue>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Circle {
    pub c: P,
    pub r: f64,
}

/// A source at one moment in structure axes (x forward, y up).
#[derive(Clone, Debug, Default, Serialize)]
pub struct Shape {
    pub points: BTreeMap<String, P>,
    pub lines: BTreeMap<String, Vec<P>>,
    pub circles: BTreeMap<String, Circle>,
    /// Outlines drawn but not measured (Rubble's part groups).
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub loops: BTreeMap<String, Vec<Vec<P>>>,
    pub vis: BTreeMap<String, Vis>,
    /// Values a source reports itself (Rubble's fitted cabin pitch and yaw, degrees).
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub fit: BTreeMap<String, f64>,
}

impl Shape {
    pub fn has(&self, name: &str) -> bool {
        self.points.contains_key(name)
            || self.circles.contains_key(name)
            || self.lines.get(name).is_some_and(|l| !l.is_empty())
    }

    pub fn vis_of(&self, name: &str) -> Vis {
        self.vis.get(name).copied().unwrap_or_default()
    }

    /// Every coordinate through `f` (radii scaled by `k`).
    pub fn map(&self, f: &dyn Fn(P) -> P, k: f64) -> Shape {
        let ml = |l: &Vec<P>| l.iter().map(|&p| f(p)).collect::<Vec<_>>();
        Shape {
            points: self
                .points
                .iter()
                .map(|(n, &p)| (n.clone(), f(p)))
                .collect(),
            lines: self.lines.iter().map(|(n, l)| (n.clone(), ml(l))).collect(),
            circles: self
                .circles
                .iter()
                .map(|(n, c)| {
                    (
                        n.clone(),
                        Circle {
                            c: f(c.c),
                            r: c.r * k,
                        },
                    )
                })
                .collect(),
            loops: self
                .loops
                .iter()
                .map(|(n, v)| (n.clone(), v.iter().map(ml).collect()))
                .collect(),
            vis: self.vis.clone(),
            fit: self.fit.clone(),
        }
    }
}

/// A traced instance in structure axes: region coordinates to pixels of the region
/// (`width` × `height`), y flipped up, x flipped when the front is at the left.
pub fn from_tracing(inst: &Instance, schema: &Schema, width: f64, height: f64) -> Shape {
    let sx = if inst.front.as_deref() == Some("left") {
        -1.0
    } else {
        1.0
    };
    let f = |p: P| [sx * p[0] * width, -p[1] * height];
    let mut s = Shape::default();
    for (name, v) in &inst.elements {
        let kind = schema.element(name).map(|e| e.kind);
        let ok = match kind {
            Some(Kind::Point) => v.p.map(|p| s.points.insert(name.clone(), f(p))).is_some(),
            Some(Kind::Polyline) => match &v.pts {
                Some(l) if !l.is_empty() => {
                    s.lines
                        .insert(name.clone(), l.iter().map(|&p| f(p)).collect());
                    true
                }
                _ => false,
            },
            Some(Kind::Circle) => match (v.c, v.rim) {
                (Some(c), Some(rim)) => {
                    let (c, rim) = (f(c), f(rim));
                    s.circles
                        .insert(name.clone(), Circle { c, r: dist(c, rim) });
                    true
                }
                _ => false,
            },
            None => false,
        };
        if ok {
            s.vis.insert(name.clone(), v.vis);
        }
    }
    s
}

pub fn dist(a: P, b: P) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// The unit of `u` on shape `s`.
pub fn unit_of(u: &Unit, s: &Shape) -> Option<f64> {
    let at = |n: &str| {
        s.circles
            .get(n)
            .map(|c| c.c)
            .or_else(|| s.points.get(n).copied())
    };
    let v = match u {
        Unit::Distance([a, b]) => dist(at(a)?, at(b)?),
        Unit::Radius(n) => s.circles.get(n)?.r,
        Unit::Chord(n) => {
            let l = s.lines.get(n)?;
            dist(*l.first()?, *l.last()?)
        }
    };
    (v > 1e-12).then_some(v)
}

/// A source's before and now shapes.
pub struct Source {
    pub name: String,
    pub before: Shape,
    pub now: Shape,
}

/// The first of the schema's frames that every source can be normalised by.
pub fn choose_frame<'a>(schema: &'a Schema, sources: &[Source]) -> Option<&'a FrameDef> {
    schema.frames.iter().find(|f| {
        sources.iter().all(|s| {
            f.origin.of(&s.before).is_some()
                && f.origin.of(&s.now).is_some()
                && unit_of(&f.unit, &s.before).is_some()
        })
    })
}

/// `s` with each shape's own origin at 0 and lengths over `unit`.
pub fn normalise(s: &Shape, origin: &Anchor, unit: f64) -> Option<Shape> {
    let o = origin.of(s)?;
    Some(s.map(
        &|p| [(p[0] - o[0]) / unit, (p[1] - o[1]) / unit],
        1.0 / unit,
    ))
}

/// The comparison of every source: the frame used, the normalised shapes and the
/// measures' table (JSON, as the studio's server and `outlines.py` read it).
pub fn compare(schema: &Schema, sources: &[Source]) -> Value {
    let frame = choose_frame(schema, sources);
    let mut shapes = serde_json::Map::new();
    let mut norm: Vec<(String, Shape, Shape)> = Vec::new();
    for s in sources {
        let (b, n) = match frame.and_then(|f| Some((f, unit_of(&f.unit, &s.before)?))) {
            Some((f, u)) => (
                normalise(&s.before, &f.origin, u).unwrap_or_default(),
                normalise(&s.now, &f.origin, u).unwrap_or_default(),
            ),
            None => (s.before.clone(), s.now.clone()),
        };
        shapes.insert(s.name.clone(), json!({ "before": b, "now": n }));
        norm.push((s.name.clone(), b, n));
    }
    let rows: Vec<Value> = schema
        .measures
        .iter()
        .filter(|m| m.applies(frame.map(|f| f.id.as_str())))
        .map(|m| {
            let mut vals = serde_json::Map::new();
            for (name, b, n) in &norm {
                vals.insert(name.clone(), m.eval(b, n).map_or(Value::Null, |r| json!(r)));
            }
            json!({ "id": m.id, "label": m.label, "values": vals })
        })
        .collect();
    json!({
        "schema": schema.id,
        "frame": frame.map(|f| json!({ "id": f.id, "label": f.label })),
        "shapes": shapes,
        "rows": rows,
    })
}
