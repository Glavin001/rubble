//! Accuracy-first native graph inference using a resident PyTorch worker.
//! The worker sees only current graph features; no oracle trajectory is opened.
//! This transport is an experiment, not the intended production GPU runtime.
use crate::{
    causal_state::{Model, Provider},
    graph_transition::{edge_inputs, local_endpoints, node_inputs},
};
use std::{
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
};
use stress_ref::{
    math::{Quat, Vec3},
    scene::Support,
    solver::{ChunkLoads, ReferenceSolver},
    stress_provider::{BondJob, ChunkTransition, StressProvider, SuppliedStress},
};

pub struct GraphProvider {
    material: Provider,
    child: Child,
    input: BufWriter<ChildStdin>,
    output: BufReader<ChildStdout>,
    pending: Option<(u64, f64, bool, Vec<BondJob>)>,
    pub calls: u64,
    pub chunk_predictions: u64,
}
impl GraphProvider {
    pub fn new(python: &Path, checkpoint: &Path, material: &Path) -> Result<Self, String> {
        let material = Provider {
            model: Model::load(material)?,
            calls: 0,
            rows: 0,
            labels: None,
            energy_decoder: true,
            frozen_history: false,
        };
        let mut child = Command::new(python)
            .args(["-m", "learned_impact.graph_state", "serve", "--checkpoint"])
            .arg(checkpoint)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| e.to_string())?;
        let input = BufWriter::new(child.stdin.take().ok_or("graph worker stdin")?);
        let output = BufReader::new(child.stdout.take().ok_or("graph worker stdout")?);
        Ok(Self {
            material,
            child,
            input,
            output,
            pending: None,
            calls: 0,
            chunk_predictions: 0,
        })
    }
    fn write_doubles(&mut self, values: &[f64]) -> Result<(), String> {
        for v in values {
            self.input
                .write_all(&v.to_le_bytes())
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}
impl Drop for GraphProvider {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
impl StressProvider for GraphProvider {
    fn responses(
        &mut self,
        s: &ReferenceSolver,
        loads: &ChunkLoads,
        jobs: &[BondJob],
        dt: f64,
        fracture: bool,
    ) -> Result<Vec<SuppliedStress>, String> {
        if self.pending.is_some() {
            return Err("unfinished graph prediction".into());
        }
        self.pending = Some((s.substeps, dt, fracture, jobs.to_vec()));
        self.material.responses(s, loads, jobs, dt, fracture)
    }
    fn chunk_transitions(
        &mut self,
        s: &ReferenceSolver,
        loads: &ChunkLoads,
        keys: &[(usize, usize)],
        dt: f64,
    ) -> Result<Option<Vec<ChunkTransition>>, String> {
        let (step, interval, fracture, jobs) =
            self.pending.take().ok_or("missing graph material phase")?;
        if step != s.substeps || interval != dt {
            return Err("graph phase mismatch".into());
        }
        let nodes = node_inputs(s, loads, keys, dt)?;
        let edges = edge_inputs(s, &jobs, dt, fracture)?;
        let endpoints = local_endpoints(s, &jobs, keys)?;
        for n in [keys.len(), jobs.len()] {
            let n = u32::try_from(n).map_err(|_| "graph exceeds worker index range")?;
            self.input
                .write_all(&n.to_le_bytes())
                .map_err(|e| e.to_string())?;
        }
        self.write_doubles(&nodes)?;
        self.write_doubles(&edges)?;
        for pair in endpoints {
            for value in pair {
                self.input
                    .write_all(&value.to_le_bytes())
                    .map_err(|e| e.to_string())?;
            }
        }
        self.input.flush().map_err(|e| e.to_string())?;
        let mut bytes = vec![0u8; keys.len() * 6 * 8];
        self.output
            .read_exact(&mut bytes)
            .map_err(|e| format!("graph worker response: {e}"))?;
        let values: Vec<f64> = bytes
            .chunks_exact(8)
            .map(|v| f64::from_le_bytes(v.try_into().unwrap()))
            .collect();
        if values.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite graph velocity increment".into());
        }
        let mut result = Vec::with_capacity(keys.len());
        for (&key, dv) in keys.iter().zip(values.chunks_exact(6)) {
            let mut next = ChunkTransition::capture(s, key);
            let support = s.structures[key.0].chunks[key.1].support;
            if support == Support::None {
                next.v += Vec3::new(dv[0], dv[1], dv[2]);
                next.u += next.v * dt;
            } else {
                next.v = Vec3::ZERO;
            }
            if support != Support::Fixed {
                next.w += Vec3::new(dv[3], dv[4], dv[5]);
                next.th = if s.config.methods.corotational_bonds {
                    (Quat::from_rotation_vector(next.w * dt) * Quat::from_rotation_vector(next.th))
                        .to_rotation_vector()
                } else {
                    next.th + next.w * dt
                };
            } else {
                next.w = Vec3::ZERO;
            }
            result.push(next);
        }
        self.calls += 1;
        self.chunk_predictions += result.len() as u64;
        Ok(Some(result))
    }
}
