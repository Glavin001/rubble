//! Procedural part-based vehicle models to the vehicle art spec
//! (`docs/vehicle-art-spec.md`): the Rust generator behind the CLI
//! (`vehiclegen`) and the browser configurator (`engine-vehiclegen-wasm`).
//!
//! A port of the Python generator in `engine/tools/vehicle-gen` (which keeps
//! the photo-matching and offline-render tools): one parametric skin surface
//! ([`tube`]), a block-structured panel mesher ([`layout`], [`carbody`]),
//! panel finishing ([`panels`]), a parts library ([`parts`]), assembly
//! ([`assemble`]), UV packing ([`uvpack`]) and glTF export ([`export`]).

pub mod curves;
pub mod fx;
pub mod mesh;
pub mod tube;
pub mod layout;
pub mod carbody;
pub mod panels;
pub mod parts;
pub mod frame;
pub mod uvpack;
pub mod assemble;
pub mod export;
pub mod designs;

pub use assemble::{build, BuildOpts, Built, Design};

/// Build a design by name from JSON parameters (missing fields take the
/// design's defaults).
pub fn design_from_json(name: &str, params: &str) -> Result<Box<dyn Design>, String> {
    let params = if params.trim().is_empty() { "{}" } else { params };
    match name {
        "sedan" => {
            let p: designs::sedan::SedanParams = serde_json::from_str(params).map_err(|e| format!("sedan parameters: {e}"))?;
            p.validate()?;
            Ok(Box::new(designs::sedan::Sedan::new(p)))
        }
        _ => Err(format!("unknown design '{name}' (have: sedan)")),
    }
}

/// A design's default parameters as JSON.
pub fn default_params_json(name: &str) -> Result<String, String> {
    match name {
        "sedan" => Ok(serde_json::to_string_pretty(&designs::sedan::SedanParams::default()).unwrap()),
        _ => Err(format!("unknown design '{name}' (have: sedan)")),
    }
}
