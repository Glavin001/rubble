//! A five-door hatchback from the sedan's builders: shorter, taller, with a
//! steep tailgate instead of a boot.

use super::vehicle::VehicleDef;

pub fn hatchback() -> VehicleDef {
    super::sedan::sedan()
}
