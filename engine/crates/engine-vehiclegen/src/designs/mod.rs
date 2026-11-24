//! Vehicle designs and the registry every consumer reads.
//!
//! Adding a vehicle = one file here plus one [`REGISTRY`] line. A *variant*
//! (police, wagon) wraps another design and overrides only what differs;
//! nothing else (CLI, wasm designer, web build, tests, lab) names a design.

pub mod mach1;
pub mod mach1_gasser;
pub mod police;
pub mod sedan;
pub mod suv4x4;
pub mod wagon;

use crate::assemble::Design;

/// One registered design.
pub struct Entry {
    pub name: &'static str,
    /// One line for UIs.
    pub title: &'static str,
    /// Default parameters as JSON.
    pub defaults: fn() -> String,
    /// Build from JSON parameters (missing fields take the defaults).
    pub build: fn(&str) -> Result<Box<dyn Design>, String>,
    /// Numeric parameters' ranges (the ones `validate` holds), by parameter name; a
    /// 3-number parameter (paint) takes its name's range per channel.
    pub ranges: &'static [(&'static str, f64, f64)],
    /// The values of the `rim_style` parameter.
    pub rim_styles: &'static [&'static str],
}

fn sedan_params(json: &str) -> Result<sedan::SedanParams, String> {
    let p: sedan::SedanParams = serde_json::from_str(json).map_err(|e| format!("parameters: {e}"))?;
    p.validate()?;
    Ok(p)
}

fn sedan_defaults() -> String {
    serde_json::to_string_pretty(&sedan::SedanParams::default()).unwrap()
}

fn wagon_defaults() -> String {
    serde_json::to_string_pretty(&wagon::livery(sedan::SedanParams::default())).unwrap()
}

fn police_defaults() -> String {
    serde_json::to_string_pretty(&police::livery(sedan::SedanParams::default())).unwrap()
}

fn mach1_params(json: &str) -> Result<mach1::Mach1Params, String> {
    let p: mach1::Mach1Params = serde_json::from_str(json).map_err(|e| format!("parameters: {e}"))?;
    p.validate()?;
    Ok(p)
}

fn mach1_defaults() -> String {
    serde_json::to_string_pretty(&mach1::Mach1Params::default()).unwrap()
}

fn gasser_defaults() -> String {
    serde_json::to_string_pretty(&mach1_gasser::livery(mach1::Mach1Params::default())).unwrap()
}

fn suv_params(json: &str) -> Result<suv4x4::Suv4x4Params, String> {
    let p: suv4x4::Suv4x4Params = serde_json::from_str(json).map_err(|e| format!("parameters: {e}"))?;
    p.validate()?;
    Ok(p)
}

fn suv_defaults() -> String {
    serde_json::to_string_pretty(&suv4x4::Suv4x4Params::default()).unwrap()
}

pub static REGISTRY: &[Entry] = &[
    Entry {
        name: "sedan",
        title: "Mid-size four-door sedan",
        defaults: sedan_defaults,
        build: |j| Ok(Box::new(sedan::Sedan::new(sedan_params(j)?))),
        ranges: sedan::RANGES,
        rim_styles: sedan::RIM_STYLES,
    },
    Entry {
        name: "police_sedan",
        title: "Police interceptor (sedan variant)",
        defaults: police_defaults,
        build: |j| Ok(Box::new(police::Police::new(sedan_params(j)?))),
        ranges: sedan::RANGES,
        rim_styles: sedan::RIM_STYLES,
    },
    Entry {
        name: "wagon",
        title: "Estate wagon (sedan variant)",
        defaults: wagon_defaults,
        build: |j| Ok(Box::new(wagon::Wagon::new(sedan_params(j)?))),
        ranges: sedan::RANGES,
        rim_styles: sedan::RIM_STYLES,
    },
    Entry {
        name: "mach1_1969",
        title: "1969 fastback Mach 1 (two-door)",
        defaults: mach1_defaults,
        build: |j| Ok(Box::new(mach1::Mach1::new(mach1_params(j)?))),
        ranges: mach1::RANGES,
        rim_styles: mach1::RIM_STYLES,
    },
    Entry {
        name: "mach1_gasser",
        title: "Drag-strip gasser (Mach 1 variant)",
        defaults: gasser_defaults,
        build: |j| Ok(Box::new(mach1_gasser::Gasser::new(mach1_params(j)?))),
        ranges: mach1::RANGES,
        rim_styles: mach1::RIM_STYLES,
    },
    Entry {
        name: "suv_4x4",
        title: "Off-road 4x4 (tailgate, flares, roof rack)",
        defaults: suv_defaults,
        build: |j| Ok(Box::new(suv4x4::Suv4x4::new(suv_params(j)?))),
        ranges: suv4x4::RANGES,
        rim_styles: suv4x4::RIM_STYLES,
    },
];

pub fn find(name: &str) -> Result<&'static Entry, String> {
    REGISTRY.iter().find(|e| e.name == name).ok_or_else(|| {
        format!("unknown design '{name}' (have: {})", REGISTRY.iter().map(|e| e.name).collect::<Vec<_>>().join(", "))
    })
}

/// Every registered design's name.
pub fn names() -> Vec<&'static str> {
    REGISTRY.iter().map(|e| e.name).collect()
}

/// The controls a configurator draws for a design, as JSON, built from its default
/// parameters and its [`Entry::ranges`]:
/// `[{"key", "kind": "number"|"bool"|"choice"|"color", "value", "min", "max", "step",
/// "options"}]`. Nested blocks (the crash physics) are not controls: they pass through
/// unchanged from the defaults.
pub fn schema_json(name: &str) -> Result<String, String> {
    let e = find(name)?;
    let defaults: serde_json::Value = serde_json::from_str(&(e.defaults)()).map_err(|x| x.to_string())?;
    let obj = defaults.as_object().ok_or("defaults are not an object")?;
    let range = |k: &str| e.ranges.iter().find(|r| r.0 == k).map(|r| (r.1, r.2));
    let mut rows = Vec::new();
    for (k, v) in obj {
        let row = match v {
            serde_json::Value::Number(n) => {
                let (lo, hi) = range(k).ok_or_else(|| format!("{name}: no range for '{k}'"))?;
                let step = ((hi - lo) / 200.0).max(1e-9);
                // a round step: 1, 2 or 5 times a power of ten at or below span / 200
                let mag = 10f64.powf(step.log10().floor());
                let step = [5.0, 2.0, 1.0].iter().map(|m| m * mag).find(|s| *s <= step * 1.0001).unwrap_or(mag);
                serde_json::json!({"key": k, "kind": "number", "value": n, "min": lo, "max": hi, "step": step})
            }
            serde_json::Value::Bool(b) => serde_json::json!({"key": k, "kind": "bool", "value": b}),
            serde_json::Value::String(_) if k == "rim_style" => {
                serde_json::json!({"key": k, "kind": "choice", "value": v, "options": e.rim_styles})
            }
            serde_json::Value::Array(a) if a.len() == 3 && a.iter().all(|x| x.is_number()) => {
                serde_json::json!({"key": k, "kind": "color", "value": v})
            }
            serde_json::Value::Object(_) => continue,
            _ => return Err(format!("{name}: parameter '{k}' has no control kind")),
        };
        rows.push(row);
    }
    serde_json::to_string(&rows).map_err(|x| x.to_string())
}
