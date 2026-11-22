//! Which kernels exist: every Slang module's name, entry point, defines and source
//! file, as data and nothing else.
//!
//! The build scripts that compile Slang (engine-webgpu, engine-metal, engine-cuda,
//! engine-viewer) read their lists here, not from the plans that schedule the kernels.
//! A build script reruns whenever one of its build-dependencies changes; with the lists
//! in the plan crates, an edit to the explicit plan or to the soft-body model (which
//! the plan depends on) reran every kernel build script and compiled those crates a
//! second time for the build scripts. This crate depends on nothing: keep it that way
//! (engine/docs/build-architecture.md).

pub mod explicit;
pub mod fracture;
pub mod rigid;
pub mod terrain;
