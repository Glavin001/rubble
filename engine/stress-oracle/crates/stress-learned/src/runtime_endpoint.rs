//! Strict wire adapter for the experimental endpoint rollout, not model inference.
use serde::Deserialize;
use stress_ref::{bond::Local6,external_stress::{StressEndpoint,ChunkEndpoint,BondEndpoint},joint::StressMeasures,math::Vec3,world::World};
use crate::{runtime_snapshot::RuntimeGraph,joint_history};
pub const FORMAT:&str="learned-external-endpoint/1";
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prediction {
    pub format:String, pub start_time:f64, pub dt:f64,
    pub chunk_delta:Vec<[f64;12]>,pub bond_ids:Vec<usize>,
    pub joint_values:Vec<[Option<f64>;21]>, pub rebar_broken:Vec<bool>,pub mode:Vec<i32>,
    pub connected:Vec<bool>,pub response:Vec<[f64;18]>,
}
impl Prediction {
    pub fn decode(self,graph:&RuntimeGraph,world:&World)->Result<StressEndpoint,String> {
        let n=self.bond_ids.len();
        if self.format!=FORMAT || self.chunk_delta.len()!=graph.chunk_keys().len()
            || [self.joint_values.len(),self.rebar_broken.len(),self.mode.len(),self.connected.len(),self.response.len()].iter().any(|&x|x!=n) {
            return Err("external prediction format/shape mismatch".into())
        }
        let v=|a:&[f64],i|Vec3::new(a[i],a[i+1],a[i+2]);
        let mut chunks=Vec::new();
        for (&key,d) in graph.chunk_keys().iter().zip(self.chunk_delta) {
            if d.iter().any(|v|!v.is_finite()) {return Err("nonfinite chunk prediction".into())}
            if world.solver.chunks[key.0][key.1].active {chunks.push(ChunkEndpoint{key,delta:[v(&d,0),v(&d,3),v(&d,6),v(&d,9)]})}
        }
        let mut bonds=Vec::new();
        for i in 0..n {
            let key=*graph.bond_keys().get(self.bond_ids[i]).ok_or("unknown external bond slot")?;
            let mut values=[0.;21];
            for (j,x) in self.joint_values[i].iter().enumerate() {
                values[j]=match x {Some(x)=>*x,None if j==4||j==5=>f64::INFINITY,_=>return Err("null outside unset ductility".into())};
            }
            let r=self.response[i];
            bonds.push(BondEndpoint{key,state:joint_history::unpack(values,self.rebar_broken[i],self.mode[i])?,
                force:Local6{lin:v(&r,0),ang:v(&r,3)},elastic:Local6{lin:v(&r,6),ang:v(&r,9)},stored:r[12],
                measures:StressMeasures{tension:r[13],shear:r[14],normal_compression:r[15],compression:r[16],compressive_force:r[17]},connected:self.connected[i]});
        }
        let endpoint=StressEndpoint{start_time:self.start_time,dt:self.dt,chunks,bonds};endpoint.validate(&world.solver)?;Ok(endpoint)
    }
}
