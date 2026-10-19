//! Lossless targets at every native stress phase, for interval-trajectory learning.
//! Phase-local observations are future information for a frame-start predictor.
//! Keep them out of that predictor's inputs; frame snapshots are stored separately.
use std::{collections::HashMap,fs::{self,File},io::{BufWriter,Write},path::{Path,PathBuf}};
use serde::Serialize;
use stress_ref::{solver::{ReferenceSolver,ChunkLoads},stress_provider::{StressProvider,SuppliedStress,BondJob}};
use crate::{joint_history,response_history,runtime_snapshot::RuntimeGraph};

pub const FORMAT:&str="learned-stress-phase/1";
pub const INPUT_WIDTH:usize=51; // deformation6, rate6, old history21, old response18
pub const TARGET_WIDTH:usize=42; // new history21, response18, work3
pub const CHUNK_WIDTH:usize=18; // local u/th/v/w, current world force/torque

#[derive(Serialize)]
pub struct Phase {
    pub frame:u32,pub substep:u64,pub time:f64,pub dt:f64,pub fracture:bool,
    pub row_start:u64,pub rows:usize,
}

pub struct Recorder {
    root:PathBuf,
    inputs:BufWriter<File>,targets:BufWriter<File>,keys:BufWriter<File>,
    input_flags:BufWriter<File>,target_flags:BufWriter<File>,
    chunks:BufWriter<File>,chunk_flags:BufWriter<File>,
    bond_keys:Vec<(usize,usize)>,chunk_keys:Vec<(usize,usize)>,slots:HashMap<(usize,usize),u32>,
    pub phases:Vec<Phase>,pub frame:u32,pub rows:u64,
}
fn writer(root:&Path,name:&str)->Result<BufWriter<File>,String>{
    File::create(root.join(name)).map(BufWriter::new).map_err(|e|e.to_string())
}
fn f64s(out:&mut BufWriter<File>,values:&[f64])->Result<(),String>{
    let bytes:Vec<u8>=values.iter().flat_map(|x|x.to_le_bytes()).collect();out.write_all(&bytes).map_err(|e|e.to_string())
}
fn i32s(out:&mut BufWriter<File>,values:&[i32])->Result<(),String>{
    let bytes:Vec<u8>=values.iter().flat_map(|x|x.to_le_bytes()).collect();out.write_all(&bytes).map_err(|e|e.to_string())
}
pub fn response_values(r:&SuppliedStress)->[f64;18] {
    let q=r.total_force.to_array();let f=r.joint.force.to_array();let m=r.joint.measures;
    [q[0],q[1],q[2],q[3],q[4],q[5],f[0],f[1],f[2],f[3],f[4],f[5],
     r.joint.stored,m.tension,m.shear,m.normal_compression,m.compression,m.compressive_force]
}
impl Recorder {
    pub fn new(root:&Path,graph:&RuntimeGraph)->Result<Self,String>{
        let root=root.join("stress_phases");fs::create_dir(&root).map_err(|e|e.to_string())?;
        let bond_keys=graph.bond_keys().to_vec();let chunk_keys=graph.chunk_keys().to_vec();
        let slots=bond_keys.iter().enumerate().map(|(i,&key)|Ok((key,u32::try_from(i).map_err(|_|"too many bond slots")?))).collect::<Result<_,String>>()?;
        Ok(Self{inputs:writer(&root,"inputs.f64")?,targets:writer(&root,"targets.f64")?,keys:writer(&root,"bond_slots.u32")?,
            input_flags:writer(&root,"input_flags.i32")?,target_flags:writer(&root,"target_flags.i32")?,
            chunks:writer(&root,"chunks.f64")?,chunk_flags:writer(&root,"chunk_flags.u64")?,
            root,bond_keys,chunk_keys,slots,phases:Vec::new(),frame:0,rows:0})
    }
    pub fn finish(mut self)->Result<(),String>{
        for out in [&mut self.inputs,&mut self.targets,&mut self.keys,&mut self.input_flags,&mut self.target_flags,&mut self.chunks,&mut self.chunk_flags] {
            out.flush().map_err(|e|e.to_string())?;
        }
        let manifest=serde_json::json!({"format":FORMAT,"status":"completed","rows":self.rows,"phases":self.phases,
            "bond_keys":self.bond_keys,"chunk_keys":self.chunk_keys,"input_width":INPUT_WIDTH,"target_width":TARGET_WIDTH,"chunk_width":CHUNK_WIDTH,
            "input_groups":[["deformation",6],["rate",6],["joint_values",21],["previous_response",18]],
            "target_groups":[["joint_values",21],["response",18],["work",3]],
            "joint_fields":joint_history::FIELDS,"response_fields":response_history::FIELDS,
            "work_fields":["dissipated","overshoot_signed","damping_dissipated"],
            "input_flags":["rebar_broken","mode","alive","connected"],
            "target_flags":["rebar_broken","mode","connected","disconnected_this_phase"],
            "chunk_fields":["u_xyz","theta_xyz","v_xyz","w_xyz","world_force_xyz","world_torque_xyz"],
            "chunk_flags":["active","cluster_id_or_u64_max"],
            "scope":"Full native-step oracle targets and phase-local observations. Interval predictors may use only their frame-start snapshot; future phase observations and loads are labels, never frame-start inputs. No temporal decimation. Not a full World checkpoint."});
        fs::write(self.root.join("manifest.json"),serde_json::to_vec_pretty(&manifest).unwrap()).map_err(|e|e.to_string())
    }
}
impl StressProvider for Recorder {
    fn responses(&mut self,s:&ReferenceSolver,loads:&ChunkLoads,jobs:&[BondJob],dt:f64,fracture:bool)->Result<Vec<SuppliedStress>,String>{
        let mut inputs=Vec::with_capacity(jobs.len()*INPUT_WIDTH);let mut targets=Vec::with_capacity(jobs.len()*TARGET_WIDTH);
        let mut input_flags=Vec::with_capacity(jobs.len()*4);let mut target_flags=Vec::with_capacity(jobs.len()*4);
        let mut key_bytes=Vec::with_capacity(jobs.len()*4);let mut responses=Vec::with_capacity(jobs.len());
        for &job in jobs {
            if job.2!=1. {return Err("stress-phase recorder requires single-rate jobs".into())}
            let slot=*self.slots.get(&(job.0,job.1)).ok_or("new bond outside phase graph")?;
            let b=&s.bonds[job.0][job.1];let g=&b.geometry;let a=s.chunks[job.0][g.a];let c=s.chunks[job.0][g.b];
            let (d,rate)=if s.config.methods.corotational_bonds {
                let cr=g.corotation(a.u,a.th,c.u,c.th);(cr.deformation(),cr.variation(a.v,a.w,c.v,c.w))
            } else {(g.kinematics(a.u,a.th,c.u,c.th),g.kinematics(a.v,a.w,c.v,c.w))};
            inputs.extend(d.to_array());inputs.extend(rate.to_array());inputs.extend(joint_history::pack(&b.joint));inputs.extend(response_history::pack(b));
            input_flags.extend([b.joint.rebar_broken as i32,joint_history::mode_code(b.joint.mode),b.alive as i32,b.connected() as i32]);
            let r=s.oracle_stress_response(job,dt,fracture);
            targets.extend(joint_history::pack(&r.joint.state));targets.extend(response_values(&r));
            targets.extend([r.joint.dissipated,r.joint.overshoot,r.damping_dissipated]);
            target_flags.extend([r.joint.state.rebar_broken as i32,joint_history::mode_code(r.joint.state.mode),
                r.joint.state.connected(b.rebar.is_some()) as i32,r.joint.disconnected as i32]);
            key_bytes.extend(slot.to_le_bytes());responses.push(r);
        }
        let mut chunks=Vec::with_capacity(self.chunk_keys.len()*CHUNK_WIDTH);let mut tags=Vec::with_capacity(self.chunk_keys.len()*16);
        for &(i,j) in &self.chunk_keys {
            let c=&s.chunks[i][j];
            for v in [c.u,c.th,c.v,c.w,loads.force[i][j],loads.torque[i][j]] {chunks.extend(v.to_array());}
            tags.extend((c.active as u64).to_le_bytes());tags.extend(if c.active{s.clusters[c.cluster].id}else{u64::MAX}.to_le_bytes());
        }
        if chunks.iter().any(|v|!v.is_finite()){return Err("nonfinite phase chunk observation".into())}
        f64s(&mut self.inputs,&inputs)?;f64s(&mut self.targets,&targets)?;f64s(&mut self.chunks,&chunks)?;
        i32s(&mut self.input_flags,&input_flags)?;i32s(&mut self.target_flags,&target_flags)?;
        self.keys.write_all(&key_bytes).map_err(|e|e.to_string())?;self.chunk_flags.write_all(&tags).map_err(|e|e.to_string())?;
        self.phases.push(Phase{frame:self.frame,substep:s.substeps,time:s.time,dt,fracture,row_start:self.rows,rows:jobs.len()});
        self.rows+=jobs.len() as u64;
        Ok(responses)
    }
}
