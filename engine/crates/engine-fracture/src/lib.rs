//! Shardwright fracture GLB import for native hosts.
//!
//! Rust port of `engine-webgpu/fracture/asset.mjs` + `geometry.mjs`: a bounds-checked
//! GLB / FRAC (FlatBuffers schema 1) reader and the AVBD scene builder that packs
//! baked convex hulls into the shared hull ABI consumed by `fracture_step.slang`.
//! No GPU or windowing dependencies, so it can also back a wasm host later.

pub mod asset;
pub mod scene;

pub use asset::{parse_fracture_glb, FractureAsset};
pub use scene::{build_scene, collider_bounds, grid_config, Body, Collider, FractureScene};
