//! Every rubble demo as a preset of one library (engine/docs/playground-plan.md): a scene,
//! its actions, its settings and its camera and controls, played on one [`World`]. The web
//! pages (engine-world-wasm), the C ABI and the native viewer play the same presets; none of
//! them decides physics.
//!
//! - [`presets`]: the registry, every demo by name, with the page that hosts it today.
//! - [`balls`]: the balls every scene fires (the fracture lab's stress models, the range).
//! - [`range`]: a baked Shardwright GLB as a threshold destructible on a ground slab, the
//!   balls fired at it and its crush debris; [`broadside`] and [`timber_yard`] are its
//!   scripted takes.
//! - [`stage`]: what a host draws for a range scene (its poses interpolated between steps,
//!   its extra meshes) and its stats.
//! - The fracture lab ([`crate::lab`]): every lab scene under every bond model.

pub mod balls;
pub mod broadside;
pub mod presets;
pub mod range;
pub mod stage;
pub mod timber_yard;

pub use presets::{preset, presets, Family, Preset};
