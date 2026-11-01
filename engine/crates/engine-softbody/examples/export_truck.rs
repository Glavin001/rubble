//! Writes the procedural cars' Rigs of Rods `.truck` text for external tools (the Rigs
//! of Rods reference harness): `<name>.truck` unchanged, and `<name>_bare.truck`, the
//! node–beam body alone ([`car_truck_bare`]). Both sides of a comparison read the same
//! file.
//!
//! ```bash
//! cargo run -p engine-softbody --example export_truck -- <out-dir>
//! ```
use engine_softbody::procedural::{car_truck, car_truck_bare, CarParams};

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| ".".into());
    let ror_sedan = CarParams {
        minimass: 10.0,
        ..CarParams::sedan()
    };
    for (name, p) in [
        ("rubble_sedan", CarParams::sedan()),
        ("rubble_car", CarParams::default()),
        ("rubble_sedan_ror", ror_sedan),
    ] {
        for (suffix, t) in [("", car_truck(&p)), ("_bare", car_truck_bare(&p))] {
            let path = format!("{dir}/{name}{suffix}.truck");
            std::fs::write(&path, t).expect("write .truck");
            println!("{path}");
        }
    }
}
