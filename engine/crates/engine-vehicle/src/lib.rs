//! Vehicle controller host side for the Rubble AVBD engine.
//!
//! Wheels are raycast "virtual wheels" solved inside the AVBD primal of their
//! chassis (`slang/kernels/vehicle_step.slang`). This crate owns everything the
//! hosts share: serialisable descriptors and presets, packing into the GPU
//! records, pool allocation, control mirrors and output decoding.

pub mod desc;
pub mod host;
pub mod pack;
pub mod presets;

pub use desc::*;
pub use host::*;
pub use pack::{pack, requirements, PackedVehicle, Requirements, VehicleCapacity, VehicleLayout, VehicleSlots};
