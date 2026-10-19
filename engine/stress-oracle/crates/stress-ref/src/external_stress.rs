//! Experimental independent endpoint integration, never selected by normal stepping.
//! Contacts keep the original substeps. Learned hidden-state increments are spread
//! over those substeps; all learned joint responses and splits commit at the endpoint.
//! This approximation does not model within-interval fracture or damping work.
use std::collections::HashSet;
use crate::{joint::{JointState, StressMeasures}, bond::Local6, math::Vec3, scene::Support,
            solver::{ChunkLoads,ReferenceSolver}};

#[derive(Clone,Debug)]
pub struct ChunkEndpoint { pub key:(usize,usize), pub delta:[Vec3;4] }
#[derive(Clone,Debug)]
pub struct BondEndpoint {
    pub key:(usize,usize), pub state:JointState, pub force:Local6, pub elastic:Local6,
    pub stored:f64, pub measures:StressMeasures, pub connected:bool,
}
#[derive(Clone,Debug)]
pub struct StressEndpoint {
    pub start_time:f64, pub dt:f64, pub chunks:Vec<ChunkEndpoint>, pub bonds:Vec<BondEndpoint>,
}
#[derive(Debug,serde::Serialize)]
pub struct EndpointReport {
    pub contact_substeps:u64, pub analytic_bond_evaluations:u64, pub new_breaks:usize,
    pub supported_dofs_projected:usize,
    pub energy_change_at_joint_commit_j:f64,
    pub scope:&'static str,
}
impl StressEndpoint {
    /// Validate the entire update before any world mutation. Stable keys must cover
    /// every active chunk and alive interface exactly once, including cracked ones.
    pub fn validate(&self,s:&ReferenceSolver)->Result<(),String> {
        if self.start_time!=s.time || !self.dt.is_finite() || self.dt<=0. {return Err("stale/invalid external stress time".into())}
        let chunks:HashSet<_>=s.chunks.iter().enumerate().flat_map(|(i,cs)|cs.iter().enumerate().filter(|(_,c)|c.active).map(move |(j,_)|(i,j))).collect();
        let bonds:HashSet<_>=s.bonds.iter().enumerate().flat_map(|(i,bs)|bs.iter().enumerate().filter(|(_,b)|b.alive).map(move |(j,_)|(i,j))).collect();
        if self.chunks.len()!=chunks.len() || self.chunks.iter().map(|x|x.key).collect::<HashSet<_>>()!=chunks {return Err("external stress chunk coverage mismatch".into())}
        if self.bonds.len()!=bonds.len() || self.bonds.iter().map(|x|x.key).collect::<HashSet<_>>()!=bonds {return Err("external stress bond coverage mismatch".into())}
        for x in &self.chunks {
            let old=&s.chunks[x.key.0][x.key.1];
            for (v,d) in [old.u,old.th,old.v,old.w].into_iter().zip(x.delta) {
                if d.to_array().iter().chain((v+d).to_array().iter()).any(|x|!x.is_finite()) {return Err("nonfinite external chunk update".into())}
            }
        }
        for x in &self.bonds {
            let old=&s.bonds[x.key.0][x.key.1];let j=&x.state;
            // JointState itself serializes unset ductility as null; validate every
            // other field explicitly, rather than accepting arbitrary NaN/null.
            let scalars=[j.damage,j.crush,j.kappa,j.kappa_c,j.fatigue,j.rebar_plastic,j.rebar_work,j.strain_rate,j.governing_stress,j.dissipated,j.utilization];
            if scalars.iter().chain(j.plastic.lin.iter()).chain(j.plastic.ang.iter()).chain(j.rebar_slip.iter()).any(|v|!v.is_finite())
                || [j.ductility,j.ductility_c].iter().any(|v|!(v.is_finite() || *v==f64::INFINITY))
                || [j.damage,j.crush,j.fatigue].iter().any(|v|!(0. ..=1.).contains(v)) {
                return Err("invalid external joint state".into())
            }
            if j.connected(old.rebar.is_some())!=x.connected || (!old.connected() && x.connected) {return Err("external connectivity contradiction/healing".into())}
            if j.dissipated<old.joint.dissipated {return Err("external cumulative dissipation decreased".into())}
            let m=&x.measures;
            if x.force.to_array().iter().chain(x.elastic.to_array().iter()).chain([x.stored,m.tension,m.shear,m.normal_compression,m.compression,m.compressive_force].iter()).any(|v|!v.is_finite())
                || [x.stored,m.shear,m.normal_compression,m.compressive_force].iter().any(|v|*v<0.) {return Err("invalid external cached response".into())}
        }
        Ok(())
    }
    pub fn projected_dofs(&self,s:&ReferenceSolver)->usize {
        self.chunks.iter().map(|x|{
            let support=s.structures[x.key.0].chunks[x.key.1].support;
            x.delta.iter().enumerate().filter(|(i,_)|support==Support::Fixed || (support==Support::Pinned && (*i==0 || *i==2)))
                .map(|(_,v)|v.to_array().iter().filter(|&&v|v!=0.).count()).sum::<usize>()
        }).sum()
    }
}
impl ReferenceSolver {
    /// Contact-driven rigid integration plus supplied hidden-state increments.
    /// No bond constitutive evaluation, implicit solve, or analytic fracture.
    pub(crate) fn external_stress_substep(&mut self,dt:f64,loads:&ChunkLoads,endpoint:&StressEndpoint,fraction:f64) {
        for ci in 0..self.clusters.len() {self.advance_rigid(ci,dt,loads)}
        for x in &endpoint.chunks {
            let (s,c)=x.key;let support=self.structures[s].chunks[c].support;let st=&mut self.chunks[s][c];
            if support==Support::None {st.u+=x.delta[0]*fraction;st.v+=x.delta[2]*fraction}
            else {st.v=Vec3::ZERO}
            if support!=Support::Fixed {st.th+=x.delta[1]*fraction;st.w+=x.delta[3]*fraction}
            else {st.w=Vec3::ZERO}
        }
        // Book actual external work; do not invent an internal damping term or
        // adjust energy to make an unstable learned result appear conservative.
        for x in &endpoint.chunks {
            let (s,c)=x.key;let (v,w)=self.chunk_velocity(s,c);
            self.energy.external_work+=(loads.force[s][c].dot(v)+loads.torque[s][c].dot(w))*dt;
        }
        self.time+=dt;self.substeps+=1;
    }
    pub(crate) fn commit_external_stress(&mut self,endpoint:&StressEndpoint)->usize {
        let mut split=HashSet::new();let mut broken=0;
        for x in &endpoint.bonds {
            let (s,bi)=x.key;let ci=self.chunks[s][self.bonds[s][bi].geometry.a].cluster;
            let was=self.bonds[s][bi].connected();let b=&mut self.bonds[s][bi];
            let previous=std::mem::replace(&mut b.joint,x.state.clone());
            self.energy.bond_dissipation+=b.joint.dissipated-previous.dissipated;
            b.force=x.force;b.elastic=x.elastic;b.stored=x.stored;b.measures=x.measures;
            b.observe_utilization();self.max_utilization=self.max_utilization.max(b.joint.utilization);
            let disconnected=was&&!x.connected;
            if self.record_bond_events(ci,bi,&previous,disconnected,self.time) {split.insert(ci);broken+=1;}
        }
        let mut split:Vec<_>=split.into_iter().collect();split.sort_unstable();
        for ci in split.into_iter().rev() {self.split_cluster(ci)}
        // Same physical frame-transfer operation used by the explicit solver.
        for ci in 0..self.clusters.len() {if !self.clusters[ci].anchored {self.remove_rigid_drift(ci)}}
        broken
    }
}
