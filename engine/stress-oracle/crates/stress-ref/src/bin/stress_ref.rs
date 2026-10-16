//! `stress-ref`: the command line (see `cli.rs`) with scenes in the standalone world.

use std::process::ExitCode;

use stress_ref::world::World;

fn main() -> ExitCode {
    stress_ref::cli::main(&World::new, None, std::env::args().skip(1).collect())
}
