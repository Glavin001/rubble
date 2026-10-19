//! Complete interval forecasts queried at the native stress phase. No oracle
//! queries, endpoint pose injection, or hidden change to native stepping.
use std::collections::HashMap;
use serde::{Deserialize,Serialize};
use stress_ref::{bond::Local6,joint::{JointResponse,StressMeasures},math::Vec3,
    solver::{ChunkLoads,ReferenceSolver},stress_provider::{BondJob,StressProvider,SuppliedStress},world::World};
use crate::{joint_history,runtime_snapshot::RuntimeGraph};

pub const FORMAT:&str="learned-native-trajectory/2";
pub const TRANSIENT_FORMAT:&str="learned-native-trajectory/3";
#[derive(Clone,Copy,Serialize,Deserialize,PartialEq)]
#[serde(rename_all="snake_case")]
pub enum HistoryEncoding { Physical, SignedLog }
const HISTORY_SCALE:[f64;21]=[1.,1.,1.,1.,1.,1.,1.,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,1000.,1.,1e6,1000.,1.];

#[derive(Clone,Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BondForecast {
    pub slot:usize,
    /// Uniform normalized-time knots. Null is allowed only for unset ductility.
    pub history:Vec<[Option<f64>;21]>,
    pub response:Vec<[f64;18]>,
    /// Joules per second: physical dissipation, signed residual, damping work.
    pub work_rate:Vec<[f64;3]>,
    /// v3: total joules on the bond's first native evaluation this interval.
    /// Replaces (does not add to) the rate integral for that evaluation.
    #[serde(default,skip_serializing_if="Option::is_none")]
    pub initial_work:Option<[f64;3]>,
    pub mode:Vec<i32>,
    pub damage_complete_time:Option<f64>,
    pub rebar_break_time:Option<f64>,
    pub history_origin_encoded:Option<[f64;21]>,
    /// In signed-log mode, continuous forecast times for initialization of
    /// previously unset ductility; None means it remains unset this interval.
    pub ductility_set_time:[Option<f64>;2],
}
#[derive(Clone,Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prediction {
    pub format:String,pub start_time:f64,pub dt:f64,pub history_encoding:HistoryEncoding,pub bonds:Vec<BondForecast>,
}
pub struct TrajectoryProvider {
    prediction:Prediction,keys:HashMap<(usize,usize),usize>,origins:Vec<[f64;21]>,
    initial_work_applied:Vec<bool>,
    pub calls:u64,pub responses:u64,
}

fn location(t:f64,n:usize)->(usize,f64){
    let x=t.clamp(0.,1.)*(n-1) as f64;let i=(x.floor() as usize).min(n-2);(i,x-i as f64)
}
fn sample<const N:usize>(rows:&[[f64;N]],t:f64)->[f64;N]{
    let (i,a)=location(t,rows.len());std::array::from_fn(|j|rows[i][j]+(rows[i+1][j]-rows[i][j])*a)
}
fn integrate(rows:&[[f64;3]],lo:f64,hi:f64,dt:f64)->[f64;3]{
    let mut result=[0.;3];let mut t=lo.clamp(0.,1.);let end=hi.clamp(0.,1.);
    while t<end {
        let (i,_)=location(t,rows.len());let next=(((i+1) as f64/(rows.len()-1) as f64).max(t+f64::EPSILON)).min(end);
        let a=sample(rows,t);let b=sample(rows,next);
        for j in 0..3{result[j]+=(a[j]+b[j])*0.5*(next-t)*dt;}
        t=next;
    }
    result
}
impl Prediction {
    pub fn decode(self,graph:&RuntimeGraph,world:&World)->Result<TrajectoryProvider,String>{
        if (self.format!=FORMAT && self.format!=TRANSIENT_FORMAT) || self.start_time!=world.solver.time || self.dt!=world.scene.sim.frame_dt || !self.dt.is_finite() || self.dt<=0. {
            return Err("trajectory format/start clock mismatch".into())
        }
        let mut keys=HashMap::new();let mut origins=Vec::new();
        for (index,b) in self.bonds.iter().enumerate(){
            if (self.format==TRANSIENT_FORMAT)!=b.initial_work.is_some() {return Err("initial-work field/format mismatch".into())}
            if let Some(w)=b.initial_work {if w.iter().any(|x|!x.is_finite())||w[0]<0.||w[2]<0. {return Err("invalid initial trajectory work".into())}}
            let key=*graph.bond_keys().get(b.slot).ok_or("unknown trajectory bond slot")?;
            if !world.solver.bonds[key.0][key.1].alive || keys.insert(key,index).is_some(){return Err("retired or duplicate trajectory bond slot".into())}
            origins.push(joint_history::pack(&world.solver.bonds[key.0][key.1].joint));
            if self.history_encoding==HistoryEncoding::SignedLog && b.history_origin_encoded.is_none_or(|a|a.iter().any(|x|!x.is_finite())) {
                return Err("signed-log history requires a finite frame-start anchor".into())
            }
            let n=b.history.len();
            if n<2 || [b.response.len(),b.work_rate.len(),b.mode.len()].iter().any(|&v|v!=n){return Err("trajectory knot shape mismatch".into())}
            for event in [b.damage_complete_time,b.rebar_break_time,b.ductility_set_time[0],b.ductility_set_time[1]].into_iter().flatten(){
                if !event.is_finite() || !(0. ..=1.).contains(&event){return Err("invalid trajectory event time".into())}
            }
            for i in 0..n {
                if !(0..=4).contains(&b.mode[i]){return Err("invalid trajectory failure mode".into())}
                for (j,x) in b.history[i].iter().enumerate(){
                    match x {Some(v) if v.is_finite()=>{},None if (j==4||j==5)&&self.history_encoding==HistoryEncoding::Physical=>{},_=>return Err("invalid trajectory history".into())}
                }
                if self.history_encoding==HistoryEncoding::Physical {for j in [0,1,6]{if !(0. ..=1.).contains(&b.history[i][j].unwrap()){return Err("trajectory bounded history outside range".into())}}}
                if b.response[i].iter().chain(b.work_rate[i].iter()).any(|x|!x.is_finite())
                    || [12,14,15,17].iter().any(|&j|b.response[i][j]<0.) || b.work_rate[i][0]<0. || b.work_rate[i][2]<0. {
                    return Err("invalid trajectory force/work".into())
                }
            }
        }
        for &key in graph.bond_keys(){if world.solver.bonds[key.0][key.1].alive&&!keys.contains_key(&key){return Err("missing alive trajectory bond".into())}}
        let initial_work_applied=vec![false;self.bonds.len()];
        Ok(TrajectoryProvider{prediction:self,keys,origins,initial_work_applied,calls:0,responses:0})
    }
}
impl StressProvider for TrajectoryProvider {
    fn responses(&mut self,s:&ReferenceSolver,_:&ChunkLoads,jobs:&[BondJob],dt:f64,fracture:bool)->Result<Vec<SuppliedStress>,String>{
        let p=&self.prediction;let a=(s.time-p.start_time)/p.dt;let z=(s.time+dt-p.start_time)/p.dt;
        if !dt.is_finite()||dt<=0.||a< -1e-8||z>1.+1e-8{return Err("native phase outside trajectory interval".into())}
        let mut result=Vec::with_capacity(jobs.len());let mut applied=Vec::new();
        for &(structure,bond,weight) in jobs {
            if weight!=1.{return Err("trajectory requires single-rate native jobs".into())}
            let key=(structure,bond);let index=*self.keys.get(&key).ok_or("native job absent from trajectory")?;
            let b=&p.bonds[index];let old=&s.bonds[structure][bond];let previous=joint_history::pack(&old.joint);
            let (i,alpha)=location(z,b.history.len());
            let mut h=std::array::from_fn(|j|match (b.history[i][j],b.history[i+1][j]) {
                (Some(x),Some(y))=>x+(y-x)*alpha,
                // Unset ductility is a discrete state, never interpolate infinity.
                _=>if alpha<0.5{b.history[i][j]}else{b.history[i+1][j]}.unwrap_or(f64::INFINITY),
            });
            if p.history_encoding==HistoryEncoding::SignedLog {
                let anchor=b.history_origin_encoded.unwrap();
                for j in 0..21 {
                    h[j]=if h[j]==anchor[j]&&self.origins[index][j].is_finite(){self.origins[index][j]}
                        else {h[j].signum()*h[j].abs().exp_m1()*HISTORY_SCALE[j]};
                    if !h[j].is_finite(){return Err("trajectory history decode overflow".into())}
                }
                for j in [0,1,6]{h[j]=h[j].clamp(0.,1.);}
                for j in 0..2 {if previous[j+4]==f64::INFINITY && !b.ductility_set_time[j].is_some_and(|t|z>=t){h[j+4]=f64::INFINITY;}}
            }
            // Explicit learned decoder constraints: preserve irreversible history,
            // separate complete damage from continuous partial damage, and let
            // native state determine connectivity. No reference values are read.
            for j in [0,1,2,3,6,16]{h[j]=h[j].max(previous[j]);}
            let mut rebar=old.joint.rebar_broken || b.rebar_break_time.is_some_and(|t|z>=t);
            if old.joint.damage>=1. || b.damage_complete_time.is_some_and(|t|z>=t){h[0]=1.;}
            else{h[0]=h[0].min(1.-0.5*f64::EPSILON).max(previous[0]);}
            if !fracture||!s.config.fracture {
                for j in [0,1,2,3,4,5,6]{h[j]=previous[j];}rebar=old.joint.rebar_broken;
            }
            let work=match b.initial_work {
                Some(w) if !self.initial_work_applied[index]=>{applied.push(index);w},
                _=>integrate(&b.work_rate,a,z,p.dt),
            };h[19]=old.joint.dissipated+work[0];
            let state=joint_history::unpack(h,rebar,b.mode[if alpha<0.5{i}else{i+1}])?;
            let connected=state.connected(old.rebar.is_some());let r=sample(&b.response,a);
            let local=|offset|Local6{lin:Vec3::new(r[offset],r[offset+1],r[offset+2]),ang:Vec3::new(r[offset+3],r[offset+4],r[offset+5])};
            result.push(SuppliedStress{key,total_force:local(0),damping_dissipated:work[2],joint:JointResponse{
                state,force:local(6),stored:r[12],dissipated:work[0],overshoot:work[1],
                disconnected:old.connected()&&!connected,
                measures:StressMeasures{tension:r[13],shear:r[14],normal_compression:r[15],compression:r[16],compressive_force:r[17]},
            }});
        }
        for index in applied {self.initial_work_applied[index]=true;}
        self.calls+=1;self.responses+=result.len() as u64;Ok(result)
    }
}
