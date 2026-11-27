//! The measures a schema can name (its `measures` list): each compares one source's
//! normalised now shape against its normalised before shape. Reported comparisons, not
//! criteria. The definitions are `outlines.py`'s first ones (hood fold height, chord
//! shortening, creases, wheel setback and drop, windscreen angle, ride height, tilt),
//! made generic over element names.

use super::{dist, Anchor, Shape, Vis, P};
use serde::{Deserialize, Serialize};

/// A measure: an id, a label, its kind, and the frames it needs (empty: any, or none).
#[derive(Clone, Debug, Deserialize)]
pub struct MeasureDef {
    pub id: String,
    #[serde(default)]
    pub label: String,
    #[serde(flatten)]
    pub kind: MeasureKind,
    /// Only when the common frame is one of these (lengths that mean something only in
    /// that frame, such as a height over the rear wheel).
    #[serde(default)]
    pub frames: Vec<String>,
}

fn default_crease_deg() -> f64 {
    20.0
}

fn one() -> f64 {
    1.0
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MeasureKind {
    /// Largest signed distance of the now line from its chord (+ up), over the before
    /// line's chord.
    FoldHeight { line: String },
    /// Now chord over before chord.
    ChordRatio { line: String },
    /// Turns of the now line sharper than `deg`.
    Creases {
        line: String,
        #[serde(default = "default_crease_deg")]
        deg: f64,
    },
    /// Sign changes between consecutive sharp turns.
    CreaseSignChanges {
        line: String,
        #[serde(default = "default_crease_deg")]
        deg: f64,
    },
    /// Points on the now line.
    PointCount { line: String },
    /// `sign` × (now − before) of an anchor's x (along the structure, + forward).
    ShiftX {
        of: Anchor,
        #[serde(default = "one")]
        sign: f64,
    },
    /// `sign` × (now − before) of an anchor's y (+ up).
    ShiftY {
        of: Anchor,
        #[serde(default = "one")]
        sign: f64,
    },
    /// The before line's angle from horizontal (end to end, degrees, 0..90).
    Angle { line: String },
    /// Now angle − before angle (degrees).
    AngleChange { line: String },
    /// Change of a line's tilt from its front end to its rear end (+ rear up, degrees).
    TiltChange { line: String },
    /// Change of a line's heading, first to last point (degrees, + counter-clockwise).
    HeadingChange { line: String },
    /// Change of a value the source reports itself (Rubble's fitted cabin pitch or yaw).
    FitChange { key: String },
}

/// A measured value and the worst visibility of what it was measured from.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Value {
    pub v: f64,
    pub vis: Vis,
    /// A count (drawn without decimals).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub int: bool,
}

pub fn chord(l: &[P]) -> f64 {
    dist(l[0], l[l.len() - 1])
}

/// Largest signed distance of `l` from its own chord, normal pointing up, over `chord0`.
pub fn fold_height(l: &[P], chord0: f64) -> f64 {
    let ([x0, y0], [x1, y1]) = (l[0], l[l.len() - 1]);
    let len = (x1 - x0).hypot(y1 - y0).max(1e-9);
    let (mut nx, mut ny) = (-(y1 - y0) / len, (x1 - x0) / len);
    if ny < 0.0 {
        (nx, ny) = (-nx, -ny);
    }
    let best = l
        .iter()
        .map(|p| (p[0] - x0) * nx + (p[1] - y0) * ny)
        .fold(0.0f64, |a, d| if d.abs() > a.abs() { d } else { a });
    best / chord0
}

/// The turns (degrees, + counter-clockwise) at each interior point.
pub fn turns(l: &[P]) -> Vec<f64> {
    l.windows(3)
        .map(|w| {
            let a = (w[1][1] - w[0][1]).atan2(w[1][0] - w[0][0]);
            let b = (w[2][1] - w[1][1]).atan2(w[2][0] - w[1][0]);
            wrap_deg((b - a).to_degrees())
        })
        .collect()
}

pub fn wrap_deg(t: f64) -> f64 {
    (t + 180.0).rem_euclid(360.0) - 180.0
}

/// (sharp turns, sign changes between consecutive sharp turns).
pub fn creases(l: &[P], deg: f64) -> (usize, usize) {
    let sharp: Vec<f64> = turns(l).into_iter().filter(|t| t.abs() > deg).collect();
    let changes = sharp
        .windows(2)
        .filter(|w| (w[0] > 0.0) != (w[1] > 0.0))
        .count();
    (sharp.len(), changes)
}

/// End-to-end angle from horizontal, 0..90 degrees.
pub fn angle(l: &[P]) -> f64 {
    let (a, b) = (l[0], l[l.len() - 1]);
    (b[1] - a[1]).abs().atan2((b[0] - a[0]).abs()).to_degrees()
}

/// Tilt from the frontmost to the rearmost point, + when the rear is higher.
pub fn rear_up(l: &[P]) -> f64 {
    let f = l
        .iter()
        .copied()
        .fold(l[0], |a, p| if p[0] > a[0] { p } else { a });
    let b = l
        .iter()
        .copied()
        .fold(l[0], |a, p| if p[0] < a[0] { p } else { a });
    (b[1] - f[1]).atan2(f[0] - b[0]).to_degrees()
}

pub fn heading(l: &[P]) -> f64 {
    let (a, b) = (l[0], l[l.len() - 1]);
    (b[1] - a[1]).atan2(b[0] - a[0]).to_degrees()
}

impl MeasureDef {
    pub fn applies(&self, frame: Option<&str>) -> bool {
        self.frames.is_empty() || frame.is_some_and(|f| self.frames.iter().any(|x| x == f))
    }

    /// The measure on one source (normalised before and now), or None where an input is
    /// missing.
    pub fn eval(&self, before: &Shape, now: &Shape) -> Option<Value> {
        let line =
            |s: &Shape, n: &str, min: usize| s.lines.get(n).filter(|l| l.len() >= min).cloned();
        let vis = |n: &str| before.vis_of(n).max(now.vis_of(n));
        let val = |v: f64, n: &str| {
            Some(Value {
                v,
                vis: vis(n),
                int: false,
            })
        };
        let count = |v: usize, n: &str| {
            Some(Value {
                v: v as f64,
                vis: vis(n),
                int: true,
            })
        };
        match &self.kind {
            MeasureKind::FoldHeight { line: n } => {
                let (b, a) = (line(before, n, 2)?, line(now, n, 2)?);
                let c0 = chord(&b);
                (c0 > 1e-12)
                    .then(|| fold_height(&a, c0))
                    .and_then(|v| val(v, n))
            }
            MeasureKind::ChordRatio { line: n } => {
                let (b, a) = (line(before, n, 2)?, line(now, n, 2)?);
                let c0 = chord(&b);
                (c0 > 1e-12).then(|| chord(&a) / c0).and_then(|v| val(v, n))
            }
            MeasureKind::Creases { line: n, deg } => {
                line(before, n, 2)?;
                count(creases(&line(now, n, 2)?, *deg).0, n)
            }
            MeasureKind::CreaseSignChanges { line: n, deg } => {
                line(before, n, 2)?;
                count(creases(&line(now, n, 2)?, *deg).1, n)
            }
            MeasureKind::PointCount { line: n } => {
                line(before, n, 1)?;
                count(line(now, n, 1)?.len(), n)
            }
            MeasureKind::ShiftX { of, sign } => {
                val(sign * (of.of(now)?[0] - of.of(before)?[0]), of.element())
            }
            MeasureKind::ShiftY { of, sign } => {
                val(sign * (of.of(now)?[1] - of.of(before)?[1]), of.element())
            }
            MeasureKind::Angle { line: n } => {
                line(now, n, 2)?;
                val(angle(&line(before, n, 2)?), n)
            }
            MeasureKind::AngleChange { line: n } => {
                val(angle(&line(now, n, 2)?) - angle(&line(before, n, 2)?), n)
            }
            MeasureKind::TiltChange { line: n } => val(
                rear_up(&line(now, n, 2)?) - rear_up(&line(before, n, 2)?),
                n,
            ),
            MeasureKind::HeadingChange { line: n } => val(
                wrap_deg(heading(&line(now, n, 2)?) - heading(&line(before, n, 2)?)),
                n,
            ),
            MeasureKind::FitChange { key } => Some(Value {
                v: now.fit.get(key)? - before.fit.get(key)?,
                vis: Vis::Clear,
                int: false,
            }),
        }
    }
}
