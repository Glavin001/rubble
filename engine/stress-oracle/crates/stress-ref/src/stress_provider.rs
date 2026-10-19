//! Supplied stress responses at the native pre-integration phase.
//!
//! The engine owns chunk force projection, integration, support reactions,
//! rigid-frame transfer, splits, contacts and bookkeeping. A provider owns the
//! constitutive response and its work/history increments. This interface is not
//! itself a learned model or a speed improvement.
use crate::{bond::Local6, joint::JointResponse, solver::{ChunkLoads,ReferenceSolver}};

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
