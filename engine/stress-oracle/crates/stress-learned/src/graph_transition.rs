//! Graph-state training contract at the native pre-integration boundary.
//! Each input is captured before ANY bond update; targets precede rigid-frame
//! transfer and splitting. Later contacts and later oracle states are not inputs.
use crate::causal_state;
use serde::Serialize;
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};
use stress_ref::{
    scene::Support,
    solver::{ChunkLoads, ReferenceSolver},
    stress_provider::{BondJob, ChunkTransition, StressProvider, SuppliedStress},
};

pub const NODE_WIDTH: usize = 39;
pub const EDGE_WIDTH: usize = 84;
pub fn state_values(st: &ChunkTransition) -> [f64; 12] {
    let mut out = [0.; 12];
    for (j, v) in [st.u, st.th, st.v, st.w].iter().enumerate() {
        out[j * 3..j * 3 + 3].copy_from_slice(&v.to_array());
    }
    out
}
pub fn node_inputs(
    s: &ReferenceSolver,
    loads: &ChunkLoads,
    keys: &[(usize, usize)],
    dt: f64,
) -> Result<Vec<f64>, String> {
    let mut out = Vec::with_capacity(keys.len() * NODE_WIDTH);
    for input in s.chunk_transition_inputs(loads, keys, dt)? {
        out.extend(state_values(&input.state));
        for v in [
            input.frame_force,
            input.frame_torque,
            input.mass_damping_force,
            input.mass_damping_torque,
        ] {
            out.extend(v.to_array());
        }
        out.push(input.inverse_mass);
        out.extend(input.inverse_inertia.m.into_iter().flatten());
        for support in [Support::None, Support::Pinned, Support::Fixed] {
            out.push((input.support == support) as u8 as f64);
        }
        out.extend([dt, s.config.methods.corotational_bonds as u8 as f64]);
    }
    if out.len() != keys.len() * NODE_WIDTH || out.iter().any(|v| !v.is_finite()) {
        return Err("invalid graph node features".into());
    }
    Ok(out)
}
pub fn edge_inputs(
    s: &ReferenceSolver,
    jobs: &[BondJob],
    dt: f64,
    fracture: bool,
) -> Result<Vec<f64>, String> {
    let mut out = Vec::with_capacity(jobs.len() * EDGE_WIDTH);
    for &job in jobs {
        if job.2 != 1. {
            return Err("graph transition requires single-rate bonds".into());
        }
        let (x, _, _, _) = causal_state::features(s, job, dt, fracture)?;
        out.extend(x);
        let g = &s.bonds[job.0][job.1].geometry;
        for v in [g.t1, g.t2, g.normal, g.ra, g.rb] {
            out.extend(v.to_array());
        }
    }
    if out.len() != jobs.len() * EDGE_WIDTH || out.iter().any(|v| !v.is_finite()) {
        return Err("invalid graph edge features".into());
    }
    Ok(out)
}
pub fn local_endpoints(
    s: &ReferenceSolver,
    jobs: &[BondJob],
    nodes: &[(usize, usize)],
) -> Result<Vec<[u32; 2]>, String> {
    let slots: HashMap<_, _> = nodes
        .iter()
        .enumerate()
        .map(|(i, &k)| (k, i as u32))
        .collect();
    jobs.iter()
        .map(|&(i, b, _)| {
            let g = &s.bonds[i][b].geometry;
            Ok([
                *slots
                    .get(&(i, g.a))
                    .ok_or("edge endpoint outside transition graph")?,
                *slots
                    .get(&(i, g.b))
                    .ok_or("edge endpoint outside transition graph")?,
            ])
        })
        .collect()
}
#[derive(Serialize)]
pub struct Phase {
    pub frame: u32,
    pub substep: u64,
    pub time: f64,
    pub dt: f64,
    pub node_start: u64,
    pub nodes: usize,
    pub edge_start: u64,
    pub edges: usize,
}
pub struct Recorder {
    root: PathBuf,
    node_x: BufWriter<File>,
    node_y: BufWriter<File>,
    node_keys: BufWriter<File>,
    edge_x: BufWriter<File>,
    edge_y: BufWriter<File>,
    edge_gates: BufWriter<File>,
    edge_keys: BufWriter<File>,
    endpoints: BufWriter<File>,
    pending_jobs: Option<Vec<BondJob>>,
    pending_nodes: Vec<(usize, usize)>,
    pending_phase: Option<Phase>,
    pub frame: u32,
    pub nodes: u64,
    pub edges: u64,
    pub phases: Vec<Phase>,
}
fn writer(root: &Path, name: &str) -> Result<BufWriter<File>, String> {
    File::create(root.join(name))
        .map(BufWriter::new)
        .map_err(|e| e.to_string())
}
fn doubles(w: &mut BufWriter<File>, values: &[f64]) -> Result<(), String> {
    for v in values {
        w.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
    }
    Ok(())
}
fn pairs(
    w: &mut BufWriter<File>,
    values: impl IntoIterator<Item = [u32; 2]>,
) -> Result<(), String> {
    for pair in values {
        for v in pair {
            w.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}
impl Recorder {
    pub fn new(root: &Path) -> Result<Self, String> {
        let root = root.join("graph-transitions");
        fs::create_dir(&root).map_err(|e| e.to_string())?;
        Ok(Self {
            node_x: writer(&root, "nodes.f64")?,
            node_y: writer(&root, "node-targets.f64")?,
            node_keys: writer(&root, "node-keys.u32")?,
            edge_x: writer(&root, "edges.f64")?,
            edge_y: writer(&root, "edge-targets.f64")?,
            edge_gates: writer(&root, "edge-gates.f64")?,
            edge_keys: writer(&root, "edge-keys.u32")?,
            endpoints: writer(&root, "endpoints.u32")?,
            root,
            pending_jobs: None,
            pending_nodes: Vec::new(),
            pending_phase: None,
            frame: 0,
            nodes: 0,
            edges: 0,
            phases: Vec::new(),
        })
    }
    pub fn finish(mut self) -> Result<(), String> {
        if self.pending_jobs.is_some() || self.pending_phase.is_some() {
            return Err("incomplete graph transition pair".into());
        }
        for w in [
            &mut self.node_x,
            &mut self.node_y,
            &mut self.node_keys,
            &mut self.edge_x,
            &mut self.edge_y,
            &mut self.edge_gates,
            &mut self.edge_keys,
            &mut self.endpoints,
        ] {
            w.flush().map_err(|e| e.to_string())?;
        }
        let manifest = serde_json::json!({"format":"native-graph-transition/1","status":"completed","phases":self.phases,
            "nodes":self.nodes,"edges":self.edges,"node_width":NODE_WIDTH,"node_target_width":12,"edge_width":EDGE_WIDTH,
            "edge_target_width":causal_state::NY,"edge_gate_width":causal_state::NG,
            "node_groups":[["u_theta_v_w",12],["current_frame_force_torque",6],["current_mass_damping_force_torque",6],
                ["inverse_effective_mass",1],["inverse_effective_inertia_row_major",9],["support_none_pinned_fixed",3],["dt",1],["corotational",1]],
            "edge_groups":[["causal_material_inputs_v2",69],["t1_t2_normal_ra_rb",15]],
            "target_boundary":"after internal state transition, before rigid-frame drift transfer and topology splitting",
            "clock_convention":"native central-difference half-step velocities, rotations are native rotation vectors",
            "scope":"Oracle training pairs, all steps retained. Not learned inference. Phase-local inputs must never be used as future inputs to an earlier predictor."});
        fs::write(
            self.root.join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .map_err(|e| e.to_string())
    }
}
impl StressProvider for Recorder {
    fn responses(
        &mut self,
        s: &ReferenceSolver,
        _: &ChunkLoads,
        jobs: &[BondJob],
        dt: f64,
        fracture: bool,
    ) -> Result<Vec<SuppliedStress>, String> {
        if self.pending_jobs.is_some() {
            return Err("previous graph phase not completed".into());
        }
        doubles(&mut self.edge_x, &edge_inputs(s, jobs, dt, fracture)?)?;
        let mut result = Vec::with_capacity(jobs.len());
        for &job in jobs {
            let (_, base, _, _) = causal_state::features(s, job, dt, fracture)?;
            let r = s.oracle_stress_response(job, dt, fracture);
            doubles(&mut self.edge_y, &causal_state::target(&r))?;
            doubles(&mut self.edge_gates, &causal_state::gates(&base, &r))?;
            result.push(r);
        }
        pairs(
            &mut self.edge_keys,
            jobs.iter().map(|&(s, b, _)| [s as u32, b as u32]),
        )?;
        self.pending_jobs = Some(jobs.to_vec());
        Ok(result)
    }
    fn chunk_transitions(
        &mut self,
        s: &ReferenceSolver,
        loads: &ChunkLoads,
        jobs: &[(usize, usize)],
        dt: f64,
    ) -> Result<Option<Vec<ChunkTransition>>, String> {
        let edges = self
            .pending_jobs
            .as_ref()
            .ok_or("missing graph bond inputs")?;
        doubles(&mut self.node_x, &node_inputs(s, loads, jobs, dt)?)?;
        pairs(
            &mut self.node_keys,
            jobs.iter().map(|&(s, c)| [s as u32, c as u32]),
        )?;
        pairs(&mut self.endpoints, local_endpoints(s, edges, jobs)?)?;
        self.pending_nodes = jobs.to_vec();
        self.pending_phase = Some(Phase {
            frame: self.frame,
            substep: s.substeps,
            time: s.time,
            dt,
            node_start: self.nodes,
            nodes: jobs.len(),
            edge_start: self.edges,
            edges: edges.len(),
        });
        Ok(None)
    }
    fn records_chunk_transitions(&self) -> bool {
        true
    }
    fn observe_chunk_transitions(
        &mut self,
        states: &[ChunkTransition],
        _: f64,
    ) -> Result<(), String> {
        if states.len() != self.pending_nodes.len()
            || states
                .iter()
                .zip(&self.pending_nodes)
                .any(|(s, &k)| s.key != k)
        {
            return Err("graph transition target key mismatch".into());
        }
        for st in states {
            doubles(&mut self.node_y, &state_values(st))?;
        }
        let phase = self
            .pending_phase
            .take()
            .ok_or("missing graph phase inputs")?;
        self.nodes += phase.nodes as u64;
        self.edges += phase.edges as u64;
        self.phases.push(phase);
        self.pending_jobs = None;
        self.pending_nodes.clear();
        Ok(())
    }
}
