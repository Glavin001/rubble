//! Fault injection — the "test the tests" layer.
//!
//! A test suite is only trustworthy if it actually fails when the engine is
//! wrong. These faults inject *known* bugs into a scenario's configuration and
//! body descriptors; the detection-matrix test then asserts that the suite
//! catches them. A fault that no scenario catches is a **blind spot** in the
//! suite (and, equally important, proves a passing scenario's tolerances are not
//! merely loose enough to pass anything).
//!
//! These are Rust-side faults injected via `SimConfig` / `RigidBodyDesc` only —
//! no engine recompile and nothing that could ship in a release build (gated
//! behind the `faults` feature). WGSL-level faults (e.g. flipping a contact
//! normal) require a small engine hook and are tracked as future work.

use glam::Vec3;
use rubble3d::{RigidBodyDesc, SimConfig};

/// A deliberate, known bug to inject.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FaultKind {
    /// Flip gravity's sign — bodies should "fall up".
    NegateGravity,
    /// Remove gravity entirely.
    ZeroGravity,
    /// Scale gravity (wrong fall distance / velocity).
    ScaleGravity(f32),
    /// Collapse the constraint solver to a single iteration (under-resolved
    /// contacts → penetration / jitter).
    DropSolverIterations,
    /// Remove all friction (config default and every body).
    ZeroFriction,
}

impl FaultKind {
    pub fn label(&self) -> String {
        format!("{self:?}")
    }
}

/// The catalog exercised by the detection matrix.
pub fn all_faults() -> Vec<FaultKind> {
    vec![
        FaultKind::NegateGravity,
        FaultKind::ZeroGravity,
        FaultKind::ScaleGravity(1.5),
        FaultKind::DropSolverIterations,
        FaultKind::ZeroFriction,
    ]
}

/// Apply a fault in place to a scenario's config and body descriptors.
pub fn apply(fault: FaultKind, cfg: &mut SimConfig, bodies: &mut [RigidBodyDesc]) {
    match fault {
        FaultKind::NegateGravity => cfg.gravity = -cfg.gravity,
        FaultKind::ZeroGravity => cfg.gravity = Vec3::ZERO,
        FaultKind::ScaleGravity(s) => cfg.gravity *= s,
        FaultKind::DropSolverIterations => cfg.solver_iterations = 1,
        FaultKind::ZeroFriction => {
            cfg.friction_default = 0.0;
            for b in bodies.iter_mut() {
                b.friction = 0.0;
            }
        }
    }
}
