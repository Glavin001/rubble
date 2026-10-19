//! Write `learned-impact/1` event directories.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::catalog::{self, Catalog};
use crate::constants::{
    BOND_REBAR_DIM, CHUNK_STATE_DIM, JOINT_STATE_DIM, K, LOCAL6_DIM, SCHEMA_VERSION,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub schema_version: String,
    pub event_id: String,
    pub structure_type: String,
    pub load_type: String,
    pub n_frames: u32,
    pub n_chunks: u32,
    pub n_bonds: u32,
    #[serde(rename = "K")]
    pub k: u32,
    pub dt: f64,
    pub seed: u64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub notes: String,
}

/// Static graph features for one structure (schema § static arrays).
#[derive(Clone, Debug, Default)]
pub struct StaticFeatures {
    pub chunk_mass: Vec<f32>,
    pub chunk_inertia: Vec<f32>,       // N*3
    pub chunk_half_extents: Vec<f32>,  // N*3
    pub chunk_material_id: Vec<i32>,
    pub chunk_support: Vec<i32>,
    pub chunk_level: Vec<i32>,
    pub chunk_coarse_parent: Vec<i32>,
    pub bond_area: Vec<f32>,
    pub bond_width: Vec<f32>,            // E*2
    pub bond_offset: Vec<f32>,           // E*3
    pub bond_stiffness: Vec<f32>,        // E*6
    pub bond_strength: Vec<f32>,         // E*3
    pub bond_fracture_energy: Vec<f32>,  // E*3
    pub bond_rebar: Vec<f32>,            // E*5
    pub bond_weibull: Vec<f32>,
    pub bond_buckling_length: Vec<f32>,
    pub bond_endpoints: Vec<i32>, // E*2
}

/// One frame of inputs + labels.
#[derive(Clone, Debug, Default)]
pub struct FrameRecord {
    pub chunk_state: Vec<f32>,
    pub chunk_load_bins: Vec<f32>,
    pub chunk_contact: Vec<i32>,
    pub chunk_gravity_local: Vec<f32>,
    pub bond_joint_state: Vec<f32>,
    pub bond_force_prev: Vec<f32>,
    pub bond_qs_force: Vec<f32>,
    pub bond_fail_class: Vec<i32>,
    pub bond_peak_util: Vec<f32>,
    pub bond_fail_mode: Vec<i32>,
    pub bond_joint_state_end: Vec<f32>,
    pub bond_force_end: Vec<f32>,
    pub chunk_state_delta: Vec<f32>,
}

pub struct EventRecorder {
    root: PathBuf,
    manifest: Manifest,
    n_chunks: usize,
    n_bonds: usize,
    frames_written: u32,
}

impl EventRecorder {
    pub fn begin(
        root: impl AsRef<Path>,
        mut manifest: Manifest,
        statics: &StaticFeatures,
    ) -> Result<Self, String> {
        if manifest.schema_version != SCHEMA_VERSION {
            return Err(format!("schema_version must be {SCHEMA_VERSION}"));
        }
        if manifest.k as usize != K {
            return Err(format!("K must be {K}"));
        }
        let n = manifest.n_chunks as usize;
        let e = manifest.n_bonds as usize;
        check_static(statics, n, e)?;
        let root = root.as_ref().to_path_buf();
        if root.exists() {
            fs::remove_dir_all(&root).map_err(|e| e.to_string())?;
        }
        fs::create_dir_all(root.join("static")).map_err(|e| e.to_string())?;
        write_static(&root, statics, n, e)?;
        manifest.n_frames = 0;
        let text = serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())?;
        fs::write(root.join("manifest.json"), text).map_err(|e| e.to_string())?;
        Ok(Self { root, manifest, n_chunks: n, n_bonds: e, frames_written: 0 })
    }

    pub fn record_frame(&mut self, frame: &FrameRecord) -> Result<(), String> {
        check_frame(frame, self.n_chunks, self.n_bonds)?;
        let dir = catalog::frame_dir(&self.root, self.frames_written);
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        write_frame(&dir, frame, self.n_chunks, self.n_bonds)?;
        self.frames_written += 1;
        self.manifest.n_frames = self.frames_written;
        let text = serde_json::to_string_pretty(&self.manifest).map_err(|e| e.to_string())?;
        fs::write(self.root.join("manifest.json"), text).map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn finish(self) -> Result<PathBuf, String> {
        if self.frames_written == 0 {
            return Err("event has no frames".into());
        }
        Ok(self.root)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

fn check_static(s: &StaticFeatures, n: usize, e: usize) -> Result<(), String> {
    req("chunk_mass", s.chunk_mass.len(), n)?;
    req("chunk_inertia", s.chunk_inertia.len(), n * 3)?;
    req("chunk_half_extents", s.chunk_half_extents.len(), n * 3)?;
    req("chunk_material_id", s.chunk_material_id.len(), n)?;
    req("chunk_support", s.chunk_support.len(), n)?;
    req("chunk_level", s.chunk_level.len(), n)?;
    req("chunk_coarse_parent", s.chunk_coarse_parent.len(), n)?;
    req("bond_area", s.bond_area.len(), e)?;
    req("bond_width", s.bond_width.len(), e * 2)?;
    req("bond_offset", s.bond_offset.len(), e * 3)?;
    req("bond_stiffness", s.bond_stiffness.len(), e * LOCAL6_DIM)?;
    req("bond_strength", s.bond_strength.len(), e * 3)?;
    req("bond_fracture_energy", s.bond_fracture_energy.len(), e * 3)?;
    req("bond_rebar", s.bond_rebar.len(), e * BOND_REBAR_DIM)?;
    req("bond_weibull", s.bond_weibull.len(), e)?;
    req("bond_buckling_length", s.bond_buckling_length.len(), e)?;
    req("bond_endpoints", s.bond_endpoints.len(), e * 2)?;
    for (k, pair) in s.bond_endpoints.chunks_exact(2).enumerate() {
        let (i, j) = (pair[0], pair[1]);
        if i < 0 || j < 0 || i as usize >= n || j as usize >= n || i == j {
            return Err(format!("bond {k} endpoints [{i},{j}] out of range for N={n}"));
        }
    }
    Ok(())
}

fn check_frame(f: &FrameRecord, n: usize, e: usize) -> Result<(), String> {
    req("chunk_state", f.chunk_state.len(), n * CHUNK_STATE_DIM)?;
    req("chunk_load_bins", f.chunk_load_bins.len(), n * K * LOCAL6_DIM)?;
    req("chunk_contact", f.chunk_contact.len(), n)?;
    req("chunk_gravity_local", f.chunk_gravity_local.len(), n * 3)?;
    req("bond_joint_state", f.bond_joint_state.len(), e * JOINT_STATE_DIM)?;
    req("bond_force_prev", f.bond_force_prev.len(), e * LOCAL6_DIM)?;
    req("bond_qs_force", f.bond_qs_force.len(), e * LOCAL6_DIM)?;
    req("bond_fail_class", f.bond_fail_class.len(), e)?;
    req("bond_peak_util", f.bond_peak_util.len(), e)?;
    req("bond_fail_mode", f.bond_fail_mode.len(), e)?;
    req("bond_joint_state_end", f.bond_joint_state_end.len(), e * JOINT_STATE_DIM)?;
    req("bond_force_end", f.bond_force_end.len(), e * LOCAL6_DIM)?;
    req("chunk_state_delta", f.chunk_state_delta.len(), n * CHUNK_STATE_DIM)?;
    for (i, &c) in f.bond_fail_class.iter().enumerate() {
        if c < 0 || c as usize > K {
            return Err(format!("bond_fail_class[{i}]={c} not in 0..={K}"));
        }
    }
    Ok(())
}

fn req(name: &str, got: usize, want: usize) -> Result<(), String> {
    if got != want {
        Err(format!("{name}: len {got}, expected {want}"))
    } else {
        Ok(())
    }
}

fn write_static(root: &Path, s: &StaticFeatures, n: usize, e: usize) -> Result<(), String> {
    let mut cat = Catalog::new();
    let put_f = |cat: &mut Catalog, name: &str, shape: &[usize], data: &[f32]| -> Result<(), String> {
        let rel = format!("static/{name}.bin");
        catalog::write_f32(root, &rel, shape, data)?;
        cat.insert(name, "f32", shape, &rel);
        Ok(())
    };
    let put_i = |cat: &mut Catalog, name: &str, shape: &[usize], data: &[i32]| -> Result<(), String> {
        let rel = format!("static/{name}.bin");
        catalog::write_i32(root, &rel, shape, data)?;
        cat.insert(name, "i32", shape, &rel);
        Ok(())
    };
    put_f(&mut cat, "chunk_mass", &[n], &s.chunk_mass)?;
    put_f(&mut cat, "chunk_inertia", &[n, 3], &s.chunk_inertia)?;
    put_f(&mut cat, "chunk_half_extents", &[n, 3], &s.chunk_half_extents)?;
    put_i(&mut cat, "chunk_material_id", &[n], &s.chunk_material_id)?;
    put_i(&mut cat, "chunk_support", &[n], &s.chunk_support)?;
    put_i(&mut cat, "chunk_level", &[n], &s.chunk_level)?;
    put_i(&mut cat, "chunk_coarse_parent", &[n], &s.chunk_coarse_parent)?;
    put_f(&mut cat, "bond_area", &[e], &s.bond_area)?;
    put_f(&mut cat, "bond_width", &[e, 2], &s.bond_width)?;
    put_f(&mut cat, "bond_offset", &[e, 3], &s.bond_offset)?;
    put_f(&mut cat, "bond_stiffness", &[e, LOCAL6_DIM], &s.bond_stiffness)?;
    put_f(&mut cat, "bond_strength", &[e, 3], &s.bond_strength)?;
    put_f(&mut cat, "bond_fracture_energy", &[e, 3], &s.bond_fracture_energy)?;
    put_f(&mut cat, "bond_rebar", &[e, BOND_REBAR_DIM], &s.bond_rebar)?;
    put_f(&mut cat, "bond_weibull", &[e], &s.bond_weibull)?;
    put_f(&mut cat, "bond_buckling_length", &[e], &s.bond_buckling_length)?;
    put_i(&mut cat, "bond_endpoints", &[e, 2], &s.bond_endpoints)?;
    cat.save(&root.join("catalog.json"))
}

fn write_frame(dir: &Path, f: &FrameRecord, n: usize, e: usize) -> Result<(), String> {
    let mut cat = Catalog::new();
    let put_f = |cat: &mut Catalog, name: &str, shape: &[usize], data: &[f32]| -> Result<(), String> {
        let rel = format!("{name}.bin");
        catalog::write_f32(dir, &rel, shape, data)?;
        cat.insert(name, "f32", shape, &rel);
        Ok(())
    };
    let put_i = |cat: &mut Catalog, name: &str, shape: &[usize], data: &[i32]| -> Result<(), String> {
        let rel = format!("{name}.bin");
        catalog::write_i32(dir, &rel, shape, data)?;
        cat.insert(name, "i32", shape, &rel);
        Ok(())
    };
    put_f(&mut cat, "chunk_state", &[n, CHUNK_STATE_DIM], &f.chunk_state)?;
    put_f(&mut cat, "chunk_load_bins", &[n, K, LOCAL6_DIM], &f.chunk_load_bins)?;
    put_i(&mut cat, "chunk_contact", &[n], &f.chunk_contact)?;
    put_f(&mut cat, "chunk_gravity_local", &[n, 3], &f.chunk_gravity_local)?;
    put_f(&mut cat, "bond_joint_state", &[e, JOINT_STATE_DIM], &f.bond_joint_state)?;
    put_f(&mut cat, "bond_force_prev", &[e, LOCAL6_DIM], &f.bond_force_prev)?;
    put_f(&mut cat, "bond_qs_force", &[e, LOCAL6_DIM], &f.bond_qs_force)?;
    put_i(&mut cat, "bond_fail_class", &[e], &f.bond_fail_class)?;
    put_f(&mut cat, "bond_peak_util", &[e], &f.bond_peak_util)?;
    put_i(&mut cat, "bond_fail_mode", &[e], &f.bond_fail_mode)?;
    put_f(&mut cat, "bond_joint_state_end", &[e, JOINT_STATE_DIM], &f.bond_joint_state_end)?;
    put_f(&mut cat, "bond_force_end", &[e, LOCAL6_DIM], &f.bond_force_end)?;
    put_f(&mut cat, "chunk_state_delta", &[n, CHUNK_STATE_DIM], &f.chunk_state_delta)?;
    cat.save(&dir.join("catalog.json"))
}

/// Build a tiny synthetic event (2 frames, 4 chunks, 3 bonds) for fixtures / smoke.
pub fn write_synthetic_tiny(root: impl AsRef<Path>) -> Result<PathBuf, String> {
    let n = 4usize;
    let e = 3usize;
    let mut statics = StaticFeatures {
        chunk_mass: vec![1.0; n],
        chunk_inertia: vec![0.1; n * 3],
        chunk_half_extents: vec![0.25; n * 3],
        chunk_material_id: vec![0; n],
        chunk_support: vec![1, 0, 0, 0],
        chunk_level: vec![0; n],
        chunk_coarse_parent: vec![-1; n],
        bond_area: vec![0.01; e],
        bond_width: vec![0.1; e * 2],
        bond_offset: vec![0.0; e * 3],
        bond_stiffness: vec![1.0e6; e * LOCAL6_DIM],
        bond_strength: {
            let mut v = vec![0.0; e * 3];
            for i in 0..e {
                v[i * 3] = 1.0e6;
                v[i * 3 + 1] = 0.5e6;
                v[i * 3 + 2] = 10.0e6;
            }
            v
        },
        bond_fracture_energy: vec![100.0; e * 3],
        bond_rebar: vec![0.0; e * BOND_REBAR_DIM],
        bond_weibull: vec![1.0; e],
        bond_buckling_length: vec![0.0; e],
        bond_endpoints: vec![0, 1, 1, 2, 2, 3],
    };
    // silence unused mut if we don't mutate later
    let _ = &mut statics;

    let manifest = Manifest {
        schema_version: SCHEMA_VERSION.into(),
        event_id: "tiny_event".into(),
        structure_type: "synthetic_chain".into(),
        load_type: "synthetic_pulse".into(),
        n_frames: 0,
        n_chunks: n as u32,
        n_bonds: e as u32,
        k: K as u32,
        dt: 1.0 / 60.0,
        seed: 1,
        notes: "synthetic fixture; not oracle output".into(),
    };
    let mut rec = EventRecorder::begin(root, manifest, &statics)?;
    for frame_i in 0..2u32 {
        let mut fr = FrameRecord {
            chunk_state: vec![0.0; n * CHUNK_STATE_DIM],
            chunk_load_bins: vec![0.0; n * K * LOCAL6_DIM],
            chunk_contact: vec![0; n],
            chunk_gravity_local: {
                let mut g = vec![0.0; n * 3];
                for i in 0..n {
                    g[i * 3 + 2] = -9.81;
                }
                g
            },
            bond_joint_state: vec![0.0; e * JOINT_STATE_DIM],
            bond_force_prev: vec![0.0; e * LOCAL6_DIM],
            bond_qs_force: vec![0.0; e * LOCAL6_DIM],
            bond_fail_class: vec![0; e],
            bond_peak_util: vec![0.1 * (frame_i as f32 + 1.0); e],
            bond_fail_mode: vec![0; e],
            bond_joint_state_end: vec![0.0; e * JOINT_STATE_DIM],
            bond_force_end: vec![0.0; e * LOCAL6_DIM],
            chunk_state_delta: vec![0.0; n * CHUNK_STATE_DIM],
        };
        // Mild motion + one near-failure on bond 1 in frame 1.
        for i in 0..n {
            fr.chunk_state[i * CHUNK_STATE_DIM + 2] = 0.001 * frame_i as f32;
        }
        fr.bond_qs_force[LOCAL6_DIM] = 1.0e3 * (frame_i as f32 + 1.0);
        if frame_i == 1 {
            fr.bond_fail_class[1] = 2;
            fr.bond_fail_mode[1] = 1;
            fr.bond_peak_util[1] = 1.05;
            fr.bond_joint_state_end[1 * JOINT_STATE_DIM] = 1.0; // damage
            fr.bond_joint_state_end[1 * JOINT_STATE_DIM + 21] = 1.05;
        }
        for b in 0..e {
            fr.bond_joint_state[b * JOINT_STATE_DIM + 21] = fr.bond_peak_util[b] * 0.5;
        }
        rec.record_frame(&fr)?;
    }
    rec.finish()
}
