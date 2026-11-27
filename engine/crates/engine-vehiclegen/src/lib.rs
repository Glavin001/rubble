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
pub mod solid;
pub mod structure;
pub mod mechanical;
pub mod parts;
pub mod frame;
pub mod uvpack;
pub mod assemble;
pub mod export;
pub mod designs;
pub mod physics;
pub mod crash;

pub use assemble::{build, build_with_body, AxleWheel, BuildOpts, Built, Design, Pose, RigSpec};

/// Build a design by name from JSON parameters (missing fields take the
/// design's defaults). The names come from [`designs::REGISTRY`].
pub fn design_from_json(name: &str, params: &str) -> Result<Box<dyn Design>, String> {
    let params = if params.trim().is_empty() { "{}" } else { params };
    (designs::find(name)?.build)(params).map_err(|e| format!("{name} {e}"))
}

/// A design's default parameters as JSON.
pub fn default_params_json(name: &str) -> Result<String, String> {
    Ok((designs::find(name)?.defaults)())
}

/// Every registered design's name.
pub fn design_names() -> Vec<&'static str> {
    designs::names()
}

/// Every registered design's (name, one-line title), in registry order.
pub fn design_titles() -> Vec<(&'static str, &'static str)> {
    designs::REGISTRY.iter().map(|e| (e.name, e.title)).collect()
}

/// The configurator's controls for a design (numbers with their validated ranges, choices,
/// switches, colours) as JSON; see [`designs::schema_json`].
pub fn param_schema_json(name: &str) -> Result<String, String> {
    designs::schema_json(name)
}

/// Check a design's parameters (the `validate()` of its parameter struct) without building it.
pub fn validate_params(name: &str, params: &str) -> Result<(), String> {
    design_from_json(name, params).map(|_| ())
}
