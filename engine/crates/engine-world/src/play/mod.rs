//! Every rubble demo as a preset of one library (engine/docs/playground-plan.md): a scene,
//! its actions, its settings and its camera and controls, played on one [`World`]. The web
//! pages (engine-world-wasm), the C ABI and the native viewer play the same presets; none of
//! them decides physics.
//!
//! - [`presets`]: the registry, every demo by name, with the page that hosts it today.
//! - [`rigid`]: the rigid benchmark scenes (wall, ring, gables, pyramid, pile, columns).
//! - [`hills`]: the terrain demo's heightfield and its bodies.
//! - [`motion`]: the motion demo's scenes (the tunnelling gallery, the character
//!   playground, bullets through walls).
//! - [`vehicles`]: the vehicle demo's scenes (playground, traffic, the coupling labs, the
//!   buggies, the meteor range, the gauntlet, the meteor run).
//! - [`balls`]: the balls every scene fires (the fracture lab's stress models, the range).
//! - [`range`]: a baked Shardwright GLB as a threshold destructible on a ground slab, the
//!   balls fired at it and its crush debris; [`broadside`] and [`timber_yard`] are its
//!   scripted takes.
//! - [`stage`]: what a host draws for a range scene (its poses interpolated between steps,
//!   its extra meshes) and its stats.
//! - The fracture lab ([`crate::lab`]): every lab scene under every bond model.

pub mod balls;
pub mod broadside;
pub mod hills;
pub mod motion;
pub mod presets;
pub mod range;
pub mod rigid;
pub mod stage;
pub mod timber_yard;
pub mod vehicles;

pub use presets::{preset, presets, Family, Preset};
