//! Fracture lab host logic, shared by both hosts: the browser lab
//! (`engine-webgpu/fracture/`, through the `engine-fracture-wasm` bindings) and
//! the native Metal viewer (`engine-viewer`, bin `fracture_viewer`).
//!
//! - `asset`: bounds-checked SHARDWRIGHT_fracture v1 GLB / FRAC reader.
//! - `scene`, `cuboid`, `pairs`, `bonds`, `samples`: scene building. Hulls are
//!   cooked by `engine-schema` into the packed hull ABI v2.
//! - `plan`: buffer sizes, host words and the ordered per-tick dispatch
//!   schedule, which each host replays on its GPU API.
//!
//! No GPU or windowing dependencies.

pub mod asset;
pub mod bonds;
pub mod cook;
pub mod cuboid;
pub mod math;
pub mod pack;
pub mod pairs;
pub mod plan;
pub mod samples;
pub mod scene;
pub mod stress_export;
pub mod view;

pub use asset::{parse_fracture_glb, FractureAsset};
pub use plan::{Command, Counters, Limits, Planner};
pub use samples::{Layout, Sample, SAMPLES};
pub use scene::{build_grid, build_row, build_scene, build_suburb, Body, Collider, Joint, Scene};
