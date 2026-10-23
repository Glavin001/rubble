//! Soft-body (node–beam) vehicles for Rubble's AVBD solver: Rigs of Rods and
//! BeamNG-style vehicles simulated as 3-DOF particles joined by plastic,
//! breakable beams (`slang/kernels/soft_step.slang`), with a rigid core that
//! carries Rubble's raycast wheels and drivetrain (stage 1).
//!
//! Pipeline: parse (`truck`, `jbeam`) → lower into [`SoftVehicleDesc`] →
//! validate and repair → pack GPU records ([`pack`]) → upload through a
//! [`SoftBackend`] ([`SoftHost`]). [`cpu`] is a reference solver of the same
//! particle and beam model for tests and calibration.

pub mod buggy;
pub mod calibrate;
pub mod color;
pub mod cpu;
pub mod damage;
pub mod desc;
pub mod flexbody;
pub mod host;
pub mod jbeam;
pub mod layout;
pub mod model;
pub mod pack;
pub mod procedural;
pub mod truck;
pub mod visual;

pub use desc::*;
pub use host::{
    decode_beam_breaks, BeamBreak, PieceDetached, SkinBinding, SoftBackend, SoftCounts,
    SoftEntry, SoftHandle, SoftHost,
};
pub use damage::{damage_summary, SoftDamage};
pub use layout::*;
pub use pack::SpawnOptions;
