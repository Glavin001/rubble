//! The vehicle demo's scene scripts: the logic that shells, steers and judges
//! the streets and range of `crate::streets`. Pure Rust, so a scene host
//! (`engine-world`) runs them directly and `engine-vehicle-wasm` wraps them for
//! the pages.

pub mod gauntlet;
pub mod meteors;
pub mod run;
