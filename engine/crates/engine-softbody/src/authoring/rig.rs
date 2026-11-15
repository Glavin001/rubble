//! The vehicle rigger: a vehicle design in, a deformable node–beam car out.
//!
//! ```ignore
//! use engine_softbody::authoring::rig;
//! let car = rig::rig_vehicle("sedan", None, None)?;   // design, parameters (JSON), physics
//! for (path, bytes) in rig::mod_files(&car) { /* a BeamNG mod folder */ }
//! ```
//!
//! The design comes from the vehicle generator: the Rust one
//! (`engine-vehiclegen`, built in memory from its parameters, the designer
//! page's) or, for designs not yet ported, a Python build's `.glb` and
//! `<design>.rig.json` (`engine/tools/vehicle-gen`). The rigger lays a crash
//! lattice on the render model (members along the body's structure, panels
//! with depth, shut-line edge nodes), sizes it by the census rules and the
//! design's [`Physics`] block (masses, zone yields, hardware), puts hardware
//! at the art spec's markers (hinges and bolts as break groups, latches as
//! couplers), checks the explicit step's stability bound, and binds every
//! render vertex to the lattice by construction (weighted `.rskin` skins).
//!
//! The output is plain JBeam that the loader, the GPU solver and the World
//! pages read like community content, with Rubble's features: weighted skins,
//! a collision skin made from the render skin (for surface contact),
//! triangle aerodynamics sized to the car's drag coefficient (no per-node
//! drag), break and deform groups. `engine/docs/generated-vehicles.md` is
//! the guide; `authoring::lint` checks the result.
//!
//! (The lattice code lives in `authoring::gen2`, which also still builds the
//! earlier Rust render model's car for comparison.)

use super::gen2::vg_input;
use std::path::Path;

pub use super::gen2::rig::{Dims, Physics, Rig, RigInput};
pub use super::gen2::vg_input::{Built as RiggedVehicle, VgInput as DesignInput};

/// Rig a design by name: the Rust generator's when it has it (`params` its
/// JSON parameters, any subset), else a Python build's from its default
/// folder. `physics` replaces the design's own block.
pub fn rig_vehicle(
    design: &str,
    params: Option<&str>,
    physics: Option<Physics>,
) -> Result<RiggedVehicle, String> {
    let mut inp = DesignInput::find(design, params, None)?;
    if let Some(p) = physics {
        inp.physics = p;
    }
    vg_input::build(inp)
}

/// Rig a Python build's design from a folder holding `<design>.glb` and
/// `<design>.rig.json`.
pub fn rig_vehicle_from(
    dir: &Path,
    design: &str,
    physics: Option<Physics>,
) -> Result<RiggedVehicle, String> {
    let mut inp = DesignInput::find(design, None, Some(dir))?;
    if let Some(p) = physics {
        inp.physics = p;
    }
    vg_input::build(inp)
}

/// The rigged car as a mod folder's files (paths relative to it):
/// `<design>/<design>.jbeam` and its parts, `<design>.pc`, the render
/// model `<design>.glb`, its skins `<design>.rskin`, then `files.json`.
pub fn mod_files(car: &RiggedVehicle) -> Vec<(String, Vec<u8>)> {
    vg_input::files(car)
}
