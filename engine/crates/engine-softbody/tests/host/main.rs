//! engine-softbody's host-only integration tests in one binary (each file a module): one
//! link instead of one per file (a release test build links each with thin LTO).
//! Run one file's: `cargo test -p engine-softbody --test host <file>::`.

mod centred_damping_objective;
mod contact_breaks;
mod cpu_reference;
mod elements_oracle;
mod explicit_aero;
mod explicit_beams;
mod explicit_contact;
mod explicit_grab;
mod explicit_jbeam;
mod explicit_jbeam_ext;
mod explicit_oracle;
mod explicit_rigid;
mod explicit_shapes;
mod explicit_surface;
mod explicit_surface_cast;
mod explicit_surface_cull;
mod explicit_surface_friction;
mod explicit_thin_box;
mod explicit_triggers;
mod explicit_wheels2_shocks2;
mod parking_brake;
mod tension_yield;
mod truck_export;
mod vehicle_contact_work;
mod wheels_hit_bodies;
