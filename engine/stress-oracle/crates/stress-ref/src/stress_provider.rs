//! Supplied stress responses at the native pre-integration phase.
//!
//! The engine owns force projection, support reactions, rigid-frame transfer,
//! splits, contacts and bookkeeping. A provider owns the constitutive response
//! and may also replace the internal chunk-state transition. The default retains
//! numerical internal integration. This interface is not itself a trained model.
use crate::{bond::Local6, joint::JointResponse, solver::{ChunkLoads,ReferenceSolver}};

/// Internal physical state after a transition, before the engine transfers
/// rigid-frame drift or splits a cluster. Rotations use the native rotation
/// vector convention; velocities have the native explicit half-step convention.
/// This is the wave-carrying chunk state, not the rigid body's world pose.
#[derive(Clone, Copy, Debug)]
pub struct ChunkTransition {
    pub key: (usize, usize),
    pub u: crate::math::Vec3,
    pub th: crate::math::Vec3,
    pub v: crate::math::Vec3,
    pub w: crate::math::Vec3,
}

/// Causal mechanical context for a graph transition. Known external/inertial
/// forcing and the mass dashpot are separated from internal bond forces, which
/// the learned graph must model. No target bond response is included here.
#[derive(Clone, Copy, Debug)]
pub struct ChunkTransitionInput {
    pub state: ChunkTransition,
    pub frame_force: crate::math::Vec3,
    pub frame_torque: crate::math::Vec3,
    pub mass_damping_force: crate::math::Vec3,
    pub mass_damping_torque: crate::math::Vec3,
    pub inverse_mass: f64,
    pub inverse_inertia: crate::math::Mat3,
    pub support: crate::scene::Support,
}

impl ChunkTransition {
    pub fn capture(solver: &ReferenceSolver, key: (usize, usize)) -> Self {
        let st = &solver.chunks[key.0][key.1];
        Self { key, u: st.u, th: st.th, v: st.v, w: st.w }
    }

    pub(crate) fn validate(&self, solver: &ReferenceSolver, key: (usize, usize)) -> Result<(), String> {
        if self.key != key { return Err("chunk transition key/order mismatch".into()); }
        if [self.u, self.th, self.v, self.w].iter().flat_map(|v| v.to_array()).any(|v| !v.is_finite()) {
            return Err("nonfinite chunk transition".into());
        }
        let old = &solver.chunks[key.0][key.1];
        let zero = crate::math::Vec3::ZERO;
        match solver.structures[key.0].chunks[key.1].support {
            crate::scene::Support::Fixed if self.u != old.u || self.th != old.th || self.v != zero || self.w != zero =>
                Err("chunk transition moves a fixed support".into()),
            crate::scene::Support::Pinned if self.u != old.u || self.v != zero =>
                Err("chunk transition translates a pinned support".into()),
            _ => Ok(()),
        }
    }
}

/// (structure index, bond index, native multirate weight).
pub type BondJob = (usize, usize, f64);

#[derive(Clone,Debug)]
pub struct SuppliedStress {
    pub key:(usize,usize),
    pub joint:JointResponse,
    /// Elastic/material response plus the bond dashpot, in its generalized frame.
    pub total_force:Local6,
    pub damping_dissipated:f64,
}

pub trait StressProvider {
    /// Return exactly one response per job, in the given order. Inputs are the
    /// current native state and the actual current external loads, never future
    /// oracle frames. Empty jobs must return an empty vector.
    fn responses(&mut self,solver:&ReferenceSolver,loads:&ChunkLoads,jobs:&[BondJob],dt:f64,fracture:bool)
        ->Result<Vec<SuppliedStress>,String>;

    /// Optional full-graph mechanical transition. Called on the SAME initial
    /// state as `responses`, before any bond response is committed. Some means
    /// exactly one state per job and replaces the numerical internal-state kick
    /// and drift. None explicitly retains numerical internal integration.
    /// Current contact loads are causal inputs; future oracle phases are not.
    fn chunk_transitions(&mut self, _solver: &ReferenceSolver, _loads: &ChunkLoads,
        _jobs: &[(usize, usize)], _dt: f64) -> Result<Option<Vec<ChunkTransition>>, String> {
        Ok(None)
    }

    /// Training-only observation of pre-transfer targets. Disabled by default.
    fn records_chunk_transitions(&self) -> bool { false }
    fn observe_chunk_transitions(&mut self, _states: &[ChunkTransition], _dt: f64) -> Result<(), String> { Ok(()) }
}

impl SuppliedStress {
    pub(crate) fn validate(&self,solver:&ReferenceSolver,job:BondJob)->Result<(),String> {
        if self.key!=(job.0,job.1) {return Err("supplied stress key/order mismatch".into())}
        let old=&solver.bonds[job.0][job.1];let j=&self.joint;let s=&j.state;
        let fields=[s.damage,s.crush,s.kappa,s.kappa_c,s.fatigue,s.rebar_plastic,s.rebar_work,
            s.strain_rate,s.governing_stress,s.dissipated,s.utilization];
        if fields.iter().chain(s.plastic.lin.iter()).chain(s.plastic.ang.iter()).chain(s.rebar_slip.iter()).any(|x|!x.is_finite())
            || [s.ductility,s.ductility_c].iter().any(|v|!(v.is_finite()||*v==f64::INFINITY))
            || [s.damage,s.crush,s.fatigue].iter().any(|v|!(0. ..=1.).contains(v)) {
            return Err("invalid supplied joint history".into())
        }
        let connected=s.connected(old.rebar.is_some());
        if (!old.connected() && connected) || j.disconnected!=(old.connected()&&!connected) {
            return Err("supplied stress connectivity contradiction/healing".into())
        }
        // Match the native law's irreversible histories, including partial
        // damage before a bond disconnects. Slip and strain rate are reversible
        // or filtered quantities and must not receive this monotonicity guard.
        if [s.damage<old.joint.damage,s.crush<old.joint.crush,
            s.kappa<old.joint.kappa,s.kappa_c<old.joint.kappa_c,
            s.fatigue<old.joint.fatigue,s.rebar_work<old.joint.rebar_work,
            old.joint.rebar_broken&&!s.rebar_broken].into_iter().any(|v|v) {
            return Err("supplied stress reverses irreversible history".into())
        }
        if !solver.config.fracture && (s.damage!=old.joint.damage || s.crush!=old.joint.crush
            || s.kappa!=old.joint.kappa || s.kappa_c!=old.joint.kappa_c
            || s.ductility!=old.joint.ductility || s.ductility_c!=old.joint.ductility_c
            || s.fatigue!=old.joint.fatigue || s.rebar_broken!=old.joint.rebar_broken) {
            return Err("supplied stress changes fracture history while fracture is disabled".into())
        }
        let m=&j.measures;
        // overshoot is the native signed bookkeeping residual `released - law`.
        // Even an oracle response can be negative from frozen-minus-stored
        // cancellation (house04: -1 ULP at 8.8044480744765 J), and current
        // mixed-mode reference updates also have larger negative residuals.
        // Preserve it exactly, without clamping; physical dissipated work
        // remains >= 0. This validator is not proof of energy conservation.
        let values=[j.dissipated,j.overshoot,j.stored,self.damping_dissipated,m.tension,m.shear,m.normal_compression,m.compression,m.compressive_force];
        if values.iter().chain(j.force.to_array().iter()).chain(self.total_force.to_array().iter()).any(|v|!v.is_finite())
            || [j.dissipated,j.stored,self.damping_dissipated,m.shear,m.normal_compression,m.compressive_force].iter().any(|v|*v<0.) {
            return Err(format!("invalid supplied force/work response at {:?}, time {}: work/measures={values:?}, elastic={:?}, total={:?}",self.key,solver.time,j.force.to_array(),self.total_force.to_array()))
        }
        // Match the reference accumulation operation rather than subtracting two
        // large cumulative histories, which can lose a small positive increment.
        if s.dissipated!=old.joint.dissipated+j.dissipated {
            return Err("supplied dissipated increment/history mismatch".into())
        }
        Ok(())
    }
}
