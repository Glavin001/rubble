//! Vehicle authoring: a small library for writing node–beam vehicles the way real
//! ones are designed, plus declarative definitions built with it.
//!
//! * [`curve`]: editable control-point curves and outlines;
//! * [`body`]: the body's shape ([`BodyProfile`]) and its lofted, capped shell;
//! * [`material`]: steel grades and the rest as beam properties;
//! * [`members`]: structural trusses (rails, pillars, sills, cross members),
//!   lattice blocks (engine), plates (radiator) and suspension corners;
//! * [`vehicle`]: the definition ([`VehicleDef`]: profile, panels, zones, crumple
//!   gradient, members, masses, suspension, look) and [`build`];
//! * [`skin`]: the high-resolution visual mesh bound to the skeleton;
//! * [`sedan`], [`hatchback`]: definitions.
//!
//! Guide: `engine/docs/vehicle-authoring.md`.

pub mod body;
pub mod curve;
pub mod hatchback;
pub mod material;
pub mod members;
pub mod parts;
pub mod sedan;
pub mod skeleton;
pub mod skin;
pub mod vehicle;

pub use body::{BodyProfile, Seg, Spacing, Well};
pub use curve::{Curve, Outline};
pub use material::{Grades, Material};
pub use members::{Corner, MemberSpec};
pub use skeleton::Skeleton;
pub use vehicle::{
    build, BraceSpec, BuildParams, Built, DamageKind, DamageUnit, Resolution, ShearSpec, Stats,
    VehicleDef,
};

/// A definition by body style name (`sedan`, `hatchback`).
pub fn style(name: &str) -> Option<VehicleDef> {
    match name {
        "sedan" => Some(sedan::sedan()),
        // One tier (BeamNG's construction): `sedan::sedan_one_tier`.
        "sedan1" | "showcase1" => Some(sedan::sedan_one_tier()),
        "hatchback" | "hatch" => Some(hatchback::hatchback()),
        _ => None,
    }
}
