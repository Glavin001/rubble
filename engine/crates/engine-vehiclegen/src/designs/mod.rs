//! Vehicle designs and the registry every consumer reads.
//!
//! Adding a vehicle = one file here plus one [`REGISTRY`] line. A *variant*
//! (police, wagon) wraps another design and overrides only what differs;
//! nothing else (CLI, wasm designer, web build, tests, lab) names a design.

pub mod police;
pub mod sedan;

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

fn police_defaults() -> String {
    serde_json::to_string_pretty(&police::livery(sedan::SedanParams::default())).unwrap()
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
