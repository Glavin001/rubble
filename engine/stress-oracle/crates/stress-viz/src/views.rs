//! View modes: what each panel colours chunks by, the colour scales (fixed over a whole
//! recording, and shared by compared variants), and the legend each panel shows.

use std::str::FromStr;

use crate::colormap::{categorical, rgb8, Colormap, Rgb};
use crate::record::{ChunkRec, Frame, Recording};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewMode {
    /// Chunk failure index (largest bond utilization, 1 = at strength).
    Utilization,
    /// Dominant (largest-magnitude, signed) principal stress, diverging around zero.
    Stress,
    /// Von Mises equivalent stress.
    VonMises,
    /// Neutral chunks with cracked (yellow to red) and broken (dark) bond patches.
    Damage,
    /// One colour per rigid cluster; anchored clusters grey.
    Fragments,
    /// Speed heat map.
    Velocity,
    /// Hidden deformation, exaggerated, coloured by magnitude.
    Deformation,
}

impl FromStr for ViewMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        Ok(match s {
            "utilization" | "util" => ViewMode::Utilization,
            "stress" | "principal" => ViewMode::Stress,
            "von_mises" | "vm" => ViewMode::VonMises,
            "damage" => ViewMode::Damage,
            "fragments" | "clusters" => ViewMode::Fragments,
            "velocity" | "speed" => ViewMode::Velocity,
            "deformation" | "deform" => ViewMode::Deformation,
            _ => {
                return Err(format!(
                    "unknown view '{s}' (utilization, stress, von_mises, damage, fragments, velocity, deformation)"
                ))
            }
        })
    }
}

/// Neutral chunk colour of the damage view and anchored clusters.
pub const NEUTRAL: Rgb = rgb8(0xd6, 0xd8, 0xdc);
pub const ANCHORED: Rgb = rgb8(0xb4, 0xb8, 0xc0);
/// Utilization at or above 1 (failed).
pub const FAILED: Rgb = rgb8(0xb0, 0x00, 0x1c);
/// Broken bond patches.
pub const BROKEN: Rgb = rgb8(0x24, 0x26, 0x2c);

/// Colour scales of one rendering (fixed for every frame and variant).
#[derive(Clone, Copy, Debug)]
pub struct Scales {
    /// Principal stress mapped to the ends of the diverging map (Pa, symmetric).
    pub stress: f64,
    /// Von Mises stress mapped to the top of the map (Pa).
    pub von_mises: f64,
    /// Speed mapped to the top of the map (m/s).
    pub speed: f64,
    /// Deformation magnitude mapped to the top of the map (m).
    pub deformation: f64,
    /// Display exaggeration of the hidden deformation.
    pub exaggeration: f64,
}

/// User overrides of the automatic scales.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScaleOverrides {
    pub stress: Option<f64>,
    pub exaggeration: Option<f64>,
}

/// `q`-quantile of the values (0 for none).
fn quantile(mut v: Vec<f64>, q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let k = ((v.len() - 1) as f64 * q).round() as usize;
    *v.select_nth_unstable_by(k, f64::total_cmp).1
}

/// Rounds up to 1, 2 or 5 times a power of ten.
pub fn nice_ceil(x: f64) -> f64 {
    if x <= 0.0 || !x.is_finite() {
        return 1.0;
    }
    let p = 10f64.powf(x.log10().floor());
    let m = x / p;
    p * if m <= 1.0 + 1e-9 {
        1.0
    } else if m <= 2.0 + 1e-9 {
        2.0
    } else if m <= 5.0 + 1e-9 {
        5.0
    } else {
        10.0
    }
}

/// Rounds down to 1, 2 or 5 times a power of ten.
fn nice_floor(x: f64) -> f64 {
    if x <= 0.0 || !x.is_finite() {
        return 1.0;
    }
    let p = 10f64.powf(x.log10().floor());
    let m = x / p;
    p * if m >= 5.0 {
        5.0
    } else if m >= 2.0 {
        2.0
    } else {
        1.0
    }
}

fn norm(v: [f32; 3]) -> f64 {
    (v[0] as f64).hypot(v[1] as f64).hypot(v[2] as f64)
}

impl Scales {
    /// Robust (high-percentile) scales over every frame of every recording, rounded
    /// to readable values.
    pub fn fit(recs: &[&Recording], over: ScaleOverrides) -> Scales {
        let frames: Vec<&Frame> = recs.iter().flat_map(|r| r.frames.iter()).collect();
        let chunk = recs.iter().map(|r| r.typical_chunk).fold(f64::INFINITY, f64::min);
        Scales::fit_frames(&frames, chunk, over)
    }

    /// [`Scales::fit`] over `frames` whose median chunk size is `chunk`.
    pub fn fit_frames(frames: &[&Frame], chunk: f64, over: ScaleOverrides) -> Scales {
        let chunks = || frames.iter().flat_map(|f| f.chunks.iter());
        let stress = quantile(chunks().map(|c| (c.principal as f64).abs()).collect(), 0.99);
        let von_mises = quantile(chunks().map(|c| c.von_mises as f64).collect(), 0.99);
        let speed = quantile(chunks().map(|c| norm(c.velocity)).collect(), 0.995);
        let deformation = quantile(chunks().map(|c| norm(c.deformation)).collect(), 0.999);
        let exaggeration = over.exaggeration.unwrap_or_else(|| {
            if deformation > 0.0 && chunk.is_finite() {
                nice_floor(0.2 * chunk / deformation).max(1.0)
            } else {
                1.0
            }
        });
        Scales {
            stress: over.stress.unwrap_or_else(|| nice_ceil(stress.max(1.0))),
            von_mises: nice_ceil(von_mises.max(1.0)),
            speed: nice_ceil(speed.max(1e-3)),
            deformation: nice_ceil(deformation.max(1e-9)),
            exaggeration,
        }
    }
}

impl ViewMode {
    pub fn title(self, scales: &Scales) -> String {
        match self {
            ViewMode::Utilization => "Utilization (failure index)".into(),
            ViewMode::Stress => "Principal stress (tension +)".into(),
            ViewMode::VonMises => "Von Mises stress".into(),
            ViewMode::Damage => "Bond damage".into(),
            ViewMode::Fragments => "Fragments".into(),
            ViewMode::Velocity => "Speed".into(),
            ViewMode::Deformation => format!("Deformation ×{}", fmt_number(scales.exaggeration)),
        }
    }

    /// Fill colour of a chunk.
    pub fn chunk_color(self, c: &ChunkRec, s: &Scales) -> Rgb {
        match self {
            ViewMode::Utilization => {
                if c.utilization >= 1.0 {
                    FAILED
                } else {
                    Colormap::Turbo.sample(c.utilization as f64)
                }
            }
            ViewMode::Stress => Colormap::Diverging.sample(0.5 + 0.5 * c.principal as f64 / s.stress),
            ViewMode::VonMises => Colormap::Turbo.sample(c.von_mises as f64 / s.von_mises),
            ViewMode::Damage => NEUTRAL,
            ViewMode::Fragments => {
                if c.anchored {
                    ANCHORED
                } else {
                    categorical(c.cluster)
                }
            }
            ViewMode::Velocity => Colormap::Turbo.sample(norm(c.velocity) / s.speed),
            ViewMode::Deformation => Colormap::Turbo.sample(norm(c.deformation) / s.deformation),
        }
    }

    /// Displacement added to the drawn chunk centre (the exaggerated deformation).
    pub fn displacement(self, c: &ChunkRec, s: &Scales) -> [f32; 3] {
        if self == ViewMode::Deformation {
            let k = (s.exaggeration - 1.0) as f32;
            [c.deformation[0] * k, c.deformation[1] * k, c.deformation[2] * k]
        } else {
            [0.0; 3]
        }
    }

    pub fn legend(self, s: &Scales) -> Legend {
        let gradient = |map, lo: f64, hi: f64, unit: &str, divisor: f64| Legend::Gradient {
            map,
            lo: lo / divisor,
            hi: hi / divisor,
            unit: unit.to_string(),
            top_swatch: None,
        };
        match self {
            ViewMode::Utilization => Legend::Gradient {
                map: Colormap::Turbo,
                lo: 0.0,
                hi: 1.0,
                unit: "index".into(),
                top_swatch: Some((FAILED, "≥1 failed".into())),
            },
            ViewMode::Stress => {
                let (unit, div) = stress_unit(s.stress);
                gradient(Colormap::Diverging, -s.stress, s.stress, unit, div)
            }
            ViewMode::VonMises => {
                let (unit, div) = stress_unit(s.von_mises);
                gradient(Colormap::Turbo, 0.0, s.von_mises, unit, div)
            }
            ViewMode::Velocity => gradient(Colormap::Turbo, 0.0, s.speed, "m/s", 1.0),
            ViewMode::Deformation => {
                let (unit, div) = if s.deformation < 0.1 { ("mm", 1e-3) } else { ("m", 1.0) };
                gradient(Colormap::Turbo, 0.0, s.deformation, unit, div)
            }
            ViewMode::Damage => Legend::Gradient {
                map: Colormap::Heat,
                lo: 0.0,
                hi: 1.0,
                unit: "damage".into(),
                top_swatch: Some((BROKEN, "broken".into())),
            },
            ViewMode::Fragments => Legend::Fragments,
        }
    }
}

/// Unit and divisor for a stress magnitude.
fn stress_unit(pa: f64) -> (&'static str, f64) {
    if pa >= 1e8 {
        ("GPa", 1e9)
    } else if pa >= 1e5 {
        ("MPa", 1e6)
    } else {
        ("kPa", 1e3)
    }
}

/// Short human formatting (`120`, `2.5`, `0.04`, `1e-05`).
pub fn fmt_number(x: f64) -> String {
    if x == 0.0 {
        return "0".into();
    }
    let a = x.abs();
    if !(1e-3..1e6).contains(&a) {
        return format!("{x:.0e}");
    }
    let s = if a >= 100.0 {
        format!("{x:.0}")
    } else if a >= 10.0 {
        format!("{x:.1}")
    } else if a >= 1.0 {
        format!("{x:.2}")
    } else {
        format!("{x:.3}")
    };
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

/// What a panel's colour bar shows.
#[derive(Clone, Debug)]
pub enum Legend {
    Gradient { map: Colormap, lo: f64, hi: f64, unit: String, top_swatch: Option<(Rgb, String)> },
    /// Swatches for the fragment view (the count is added per frame).
    Fragments,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nice_rounding() {
        assert_eq!(nice_ceil(3.2e6), 5e6);
        assert_eq!(nice_ceil(1.0), 1.0);
        assert_eq!(nice_ceil(0.012), 0.02);
        assert_eq!(nice_floor(370.0), 200.0);
        assert_eq!(nice_floor(9.9), 5.0);
        assert_eq!(quantile(vec![5.0, 1.0, 3.0, 2.0, 4.0], 0.5), 3.0);
        assert_eq!(fmt_number(2.50), "2.5");
        assert_eq!(fmt_number(-120.0), "-120");
        assert_eq!(fmt_number(0.04), "0.04");
    }

    #[test]
    fn view_names_parse() {
        assert_eq!("util".parse::<ViewMode>().unwrap(), ViewMode::Utilization);
        assert_eq!("von_mises".parse::<ViewMode>().unwrap(), ViewMode::VonMises);
        assert!("nope".parse::<ViewMode>().is_err());
    }

    #[test]
    fn utilization_marks_failure() {
        let mut c = ChunkRec {
            center: [0.0; 3],
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            half_extents: [0.1; 3],
            cluster: 1,
            anchored: true,
            velocity: [0.0; 3],
            deformation: [0.0; 3],
            von_mises: 0.0,
            principal: -2e6,
            utilization: 1.3,
            hull: None,
        };
        let s = Scales { stress: 4e6, von_mises: 1.0, speed: 1.0, deformation: 1.0, exaggeration: 1.0 };
        assert_eq!(ViewMode::Utilization.chunk_color(&c, &s), FAILED);
        c.utilization = 0.2;
        assert_ne!(ViewMode::Utilization.chunk_color(&c, &s), FAILED);
        // Compression is on the blue side.
        let col = ViewMode::Stress.chunk_color(&c, &s);
        assert!(col[2] > col[0]);
        assert_eq!(ViewMode::Fragments.chunk_color(&c, &s), ANCHORED);
    }
}
