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
    },
    Entry {
        name: "police_sedan",
        title: "Police interceptor (sedan variant)",
        defaults: police_defaults,
        build: |j| Ok(Box::new(police::Police::new(sedan_params(j)?))),
    },
    Entry {
        name: "wagon",
        title: "Estate wagon (sedan variant)",
        defaults: wagon_defaults,
        build: |j| Ok(Box::new(wagon::Wagon::new(sedan_params(j)?))),
    },
    Entry {
        name: "mach1_1969",
        title: "1969 fastback Mach 1 (two-door)",
        defaults: mach1_defaults,
        build: |j| Ok(Box::new(mach1::Mach1::new(mach1_params(j)?))),
    },
    Entry {
        name: "mach1_gasser",
        title: "Drag-strip gasser (Mach 1 variant)",
        defaults: gasser_defaults,
        build: |j| Ok(Box::new(mach1_gasser::Gasser::new(mach1_params(j)?))),
    },
    Entry {
        name: "suv_4x4",
        title: "Off-road 4x4 (tailgate, flares, roof rack)",
        defaults: suv_defaults,
        build: |j| Ok(Box::new(suv4x4::Suv4x4::new(suv_params(j)?))),
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
