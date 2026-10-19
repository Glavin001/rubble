//! Experimental, causal material-state network. Native motion/contact integration
//! remains unchanged. Every feature is from the current state, never a future frame.
use std::{fs::File,io::{BufWriter,Write},path::Path};
use serde::{Serialize,Deserialize};
use stress_ref::{bond::Local6,math::Vec3,joint::{JointResponse,stress_measures},
    solver::{ReferenceSolver,ChunkLoads},stress_provider::{StressProvider,SuppliedStress,BondJob}};
use crate::{joint_history,phase_history};

pub const FORMAT:&str="causal-material-state/1";
pub const NY:usize=42;
pub const NG:usize=29; // 21 changed histories, complete damage, two finite ductilities, five modes
fn local(a:&[f64])->Local6{Local6{lin:Vec3::new(a[0],a[1],a[2]),ang:Vec3::new(a[3],a[4],a[5])}}
fn finite(x:f64)->f64{if x.is_finite(){x}else{0.}}

/// Baseline is a cheap elastic prior, not an oracle constitutive evaluation.
/// Infinity in unset ductility is represented by explicit presence bits.
pub fn features(s:&ReferenceSolver,job:BondJob,dt:f64,fracture:bool)->Result<(Vec<f64>,[f64;NY],Local6,Local6),String>{
    let b=&s.bonds[job.0][job.1];let g=&b.geometry;let a=s.chunks[job.0][g.a];let c=s.chunks[job.0][g.b];
    if b.rebar.is_some(){return Err("causal material v1 has no rebar training contract".into())}
    let (d,rate)=if s.config.methods.corotational_bonds{let cr=g.corotation(a.u,a.th,c.u,c.th);(cr.deformation(),cr.variation(a.v,a.w,c.v,c.w))}
        else{(g.kinematics(a.u,a.th,c.u,c.th),g.kinematics(a.v,a.w,c.v,c.w))};
    let k=b.stiffness.as_local();let q=d.mul_elem(&k);let m=stress_measures(g,&q);
    let h=joint_history::pack(&b.joint);let st=&b.strength;
    let mut x=Vec::new();x.extend(d.to_array());x.extend(rate.to_array());x.extend(h.map(finite));
    x.extend(k.to_array());x.extend(b.damping.to_array());
    x.extend([dt,fracture as u8 as f64,h[4].is_finite() as u8 as f64,h[5].is_finite() as u8 as f64,
        st.tensile,finite(st.compressive),st.cohesion,st.friction,finite(st.shear_cap),st.g_tension,st.g_shear,st.g_compression,
        st.youngs_modulus,st.rate_filter_time,b.weibull,g.area,g.length,g.width[0],g.width[1],
        m.tension,m.shear,m.normal_compression,m.compression,m.compressive_force]);
    let mut base=[0.;NY];base[..21].copy_from_slice(&h.map(finite));
    let factors=b.model().secant_factors(&b.joint,&d);
    let material=q.mul_elem(&Local6{lin:Vec3::new(1.-h[0],1.-h[0],if d.lin.z<0.{1.-h[1]}else{1.-h[0]}),ang:Vec3::splat(1.-h[0])});
    let damp=rate.mul_elem(&b.damping).mul_elem(&factors);
    base[21..27].copy_from_slice(&material.add(&damp).to_array());base[27..33].copy_from_slice(&material.to_array());
    base[33]=0.5*d.dot(&material);base[34..39].copy_from_slice(&[m.tension,m.shear,m.normal_compression,m.compression,m.compressive_force]);
    base[41]=damp.dot(&rate)*dt;
    Ok((x,base,d,rate))
}
fn target(r:&SuppliedStress)->[f64;NY]{let mut y=[0.;NY];y[..21].copy_from_slice(&joint_history::pack(&r.joint.state).map(finite));
    y[21..39].copy_from_slice(&phase_history::response_values(r));y[39..].copy_from_slice(&[r.joint.dissipated,r.joint.overshoot,r.damping_dissipated]);y}
fn gates(base:&[f64;NY],r:&SuppliedStress)->[f64;NG]{let y=target(r);let mut out=[0.;NG];
    for j in 0..21{out[j]=(y[j]!=base[j]) as u8 as f64;}
    out[21]=(r.joint.state.damage>=1.) as u8 as f64;
    out[22]=r.joint.state.ductility.is_finite() as u8 as f64;out[23]=r.joint.state.ductility_c.is_finite() as u8 as f64;
    out[24+joint_history::mode_code(r.joint.state.mode) as usize]=1.;out}

pub struct Teacher {out:BufWriter<File>,pub rows:u64,pub input_width:usize,pub calls:u64}
impl Teacher {pub fn new(p:&Path)->Result<Self,String>{Ok(Self{out:BufWriter::new(File::create(p).map_err(|e|e.to_string())?),rows:0,input_width:0,calls:0})}
    pub fn flush(&mut self)->Result<(),String>{self.out.flush().map_err(|e|e.to_string())}}
impl StressProvider for Teacher{fn responses(&mut self,s:&ReferenceSolver,_:&ChunkLoads,jobs:&[BondJob],dt:f64,fracture:bool)->Result<Vec<SuppliedStress>,String>{
    let mut out=Vec::new();for &job in jobs{let (x,base,_,_)=features(s,job,dt,fracture)?;self.input_width=x.len();
        let r=s.oracle_stress_response(job,dt,fracture);let y=target(&r);let tags=gates(&base,&r);
        for v in x.iter().chain(&base).chain(&y).chain(&tags){self.out.write_all(&v.to_le_bytes()).map_err(|e|e.to_string())?;}
        self.rows+=1;out.push(r);}self.calls+=1;Ok(out)}}

#[derive(Serialize,Deserialize)]
pub struct Layer{pub weight:Vec<Vec<f64>>,pub bias:Vec<f64>}
#[derive(Serialize,Deserialize)]
pub struct Model{pub format:String,pub trained_steps:u64,pub input_scale:Vec<f64>,pub input_mean:Vec<f64>,pub input_std:Vec<f64>,pub output_scale:Vec<f64>,pub layers:Vec<Layer>}
impl Model{
    pub fn load(path:&Path)->Result<Self,String>{let m:Self=serde_json::from_slice(&std::fs::read(path).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        if m.format!=FORMAT||m.trained_steps==0||m.output_scale.len()!=NY||m.layers.is_empty(){return Err("invalid or untrained causal model".into())}
        let n=m.input_scale.len();if m.input_mean.len()!=n||m.input_std.len()!=n{return Err("normalizer shape".into())}
        if m.input_scale.iter().chain(&m.input_std).chain(&m.output_scale).any(|x|!x.is_finite()||*x<=0.){return Err("invalid scales".into())}
        let mut width=n;for layer in &m.layers{if layer.weight.len()!=layer.bias.len()||layer.weight.iter().any(|r|r.len()!=width||r.iter().any(|x|!x.is_finite()))||layer.bias.iter().any(|x|!x.is_finite()){return Err("invalid dense layer".into())}width=layer.bias.len();}
        if width!=NY+NG{return Err("output shape".into())}Ok(m)}
    pub fn predict(&self,x:&[f64])->Result<Vec<f64>,String>{if x.len()!=self.input_scale.len(){return Err("feature width mismatch".into())}
        let mut v:Vec<f64>=x.iter().enumerate().map(|(i,x)|((x/self.input_scale[i]).asinh()-self.input_mean[i])/self.input_std[i]).collect();
        for (i,l) in self.layers.iter().enumerate(){v=l.weight.iter().zip(&l.bias).map(|(w,b)|{let z=w.iter().zip(&v).fold(*b,|sum,(w,x)|sum+w*x);if i+1<self.layers.len(){z.max(0.)}else{z}}).collect();}
        if v.iter().any(|x|!x.is_finite()){return Err("nonfinite inference".into())}Ok(v)}
}
pub struct Provider{pub model:Model,pub calls:u64,pub rows:u64}
impl StressProvider for Provider{fn responses(&mut self,s:&ReferenceSolver,_:&ChunkLoads,jobs:&[BondJob],dt:f64,fracture:bool)->Result<Vec<SuppliedStress>,String>{
    let mut out=Vec::new();for &job in jobs{if job.2!=1.{return Err("single-rate jobs required".into())}
        let b=&s.bonds[job.0][job.1];let (x,base,d,rate)=features(s,job,dt,fracture)?;let pred=self.model.predict(&x)?;
        let y:[f64;NY]=std::array::from_fn(|j|base[j]+pred[j].sinh()*self.model.output_scale[j]);
        let previous=joint_history::pack(&b.joint);let mut h=previous;
        for j in 0..21{if pred[NY+j]>0.{h[j]=y[j];}}
        for j in [0,1,6]{h[j]=h[j].clamp(0.,1.);}
        for j in [0,1,2,3,6,16]{h[j]=h[j].max(previous[j]);}
        h[0]=if previous[0]>=1.||pred[NY+21]>0.{1.}else{h[0].min(1.-f64::EPSILON)};
        for j in 4..6{h[j]=if pred[NY+22+j-4]>0.{y[j].max(0.)}else{previous[j]};}
        if !fracture||!s.config.fracture{h[..7].copy_from_slice(&previous[..7]);}
        // Known stress-rate filter is physical state, not a learned latent vector.
        let m=stress_measures(&b.geometry,&d.mul_elem(&b.stiffness.as_local()));let governing=m.tension.max(m.shear).max(m.compression);
        let raw=((governing-previous[18])/dt).max(0.)/b.strength.youngs_modulus;
        let a=if b.strength.exact_rate_filter{-(-dt/b.strength.rate_filter_time).exp_m1()}else{(dt/b.strength.rate_filter_time).min(1.)};
        h[17]=previous[17]+(raw-previous[17])*a;h[18]=governing;
        let changed=[0,1,7,8,9,10,11,12].iter().any(|&j|h[j]!=previous[j]);
        let dissipated=if changed{y[39].max(0.)}else{0.};h[19]=previous[19]+dissipated;
        let mode=(0..5).max_by(|&a,&c|pred[NY+24+a].total_cmp(&pred[NY+24+c])).unwrap() as i32;
        let state=joint_history::unpack(h,b.joint.rebar_broken,mode)?;
        // Exact elastic rest branch; nonlinear response is learned when damaged.
        let nonlinear=h[0]>0.||h[1]>0.;let force=if nonlinear{local(&y[27..33])}else{d.mul_elem(&b.stiffness.as_local())};
        let stored=if nonlinear{y[33].max(0.)}else{0.5*d.dot(&force)};
        let damp=rate.mul_elem(&b.damping).mul_elem(&b.model().secant_factors(&state,&d));
        out.push(SuppliedStress{key:(job.0,job.1),total_force:force.add(&damp),damping_dissipated:damp.dot(&rate)*dt,joint:JointResponse{
            disconnected:b.connected()&&!state.connected(false),state,force,stored,dissipated,overshoot:if changed{y[40]}else{0.},measures:m}});
    }self.calls+=1;self.rows+=out.len() as u64;Ok(out)}}
