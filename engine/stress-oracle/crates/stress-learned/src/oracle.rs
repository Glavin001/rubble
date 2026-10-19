//! Capture `learned-impact/1` events from a running `stress_ref::World`.
//!
//! Graph is frozen at [`OracleSession::begin`] (active chunks + alive bonds). Topology
//! changes mid-event keep those slots (failed bonds get fail labels; new bonds are ignored).
//!
//! `bond_qs_force` comes from a scratch static solve under gravity at frame start
//! (true QS probe). `chunk_load_bins` are external loads (scripted + contact) averaged
//! per time bin via [`World::learned_bins`]; gravity stays in `chunk_gravity_local`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use stress_ref::bond::Local6;
use stress_ref::joint::FailureMode;
use stress_ref::scene::Support;
use stress_ref::math::Vec3;
use stress_ref::scene::{Scene, SolveMode};
use stress_ref::solver::{ChunkLoads, ReferenceSolver, SolverEvent};
use stress_ref::statics::StaticOptions;
use stress_ref::world::{LearnedFrameBins, World};

use crate::validate::validate_event;

use crate::constants::{BOND_REBAR_DIM, CHUNK_STATE_DIM, JOINT_STATE_DIM, K, LOCAL6_DIM, SCHEMA_VERSION};
use crate::joint_pack::pack_joint_state;
use crate::recorder::{EventRecorder, FrameRecord, Manifest, StaticFeatures};

#[derive(Clone, Debug)]
pub struct RecordMeta {
    pub event_id: String,
    pub structure_type: String,
    pub load_type: String,
    pub notes: String,
}

struct Pending {
    t0: f64,
    event_cursor: usize,
    chunk_state: Vec<f32>,
    bond_joint_state: Vec<f32>,
    bond_force_prev: Vec<f32>,
    bond_qs_force: Vec<f32>,
    chunk_load_bins: Vec<f32>,
    chunk_contact: Vec<i32>,
    chunk_gravity_local: Vec<f32>,
}

/// Prepare a scene for recording (peak util on, explicit dynamics, duration = frames).
pub fn prepare_scene(mut scene: Scene, frames: u32) -> Scene {
    scene.sim.record_peak_utilization = true;
    if scene.sim.solve_mode == SolveMode::QuasiStatic {
        scene.sim.solve_mode = SolveMode::Explicit;
    }
    scene.sim.duration = scene.sim.frame_dt * frames as f64;
    scene
}

/// Record `frames` from a scene into `out` and validate.
pub fn record_event(
    scene: Scene,
    out: impl AsRef<Path>,
    meta: RecordMeta,
    frames: u32,
    qs_probe: bool,
) -> Result<PathBuf, String> {
    let scene = prepare_scene(scene, frames);
    let mut world = World::new(&scene);
    let mut session = OracleSession::begin(out, &world, meta, qs_probe)?;
    for _ in 0..frames {
        session.begin_frame(&mut world)?;
        world.step_frame();
        session.end_frame(&mut world)?;
    }
    let path = session.finish()?;
    validate_event(&path).map_err(|e| e.to_string())?;
    Ok(path)
}

/// Records frames from an oracle [`World`] into an event directory.
pub struct OracleSession {
    rec: EventRecorder,
    chunk_keys: Vec<(usize, usize)>,
    bond_keys: Vec<(usize, usize)>,
    pending: Option<Pending>,
    qs_probe: bool,
}

impl OracleSession {
    pub fn begin(out: impl AsRef<Path>, world: &World, meta: RecordMeta, qs_probe: bool) -> Result<Self, String> {
        let (chunk_keys, bond_keys, statics) = extract_static(world)?;
        let n = chunk_keys.len();
        let e = bond_keys.len();
        if n == 0 {
            return Err("no active chunks".into());
        }
        let manifest = Manifest {
            schema_version: SCHEMA_VERSION.into(),
            event_id: meta.event_id,
            structure_type: meta.structure_type,
            load_type: meta.load_type,
            n_frames: 0,
            n_chunks: n as u32,
            n_bonds: e as u32,
            k: K as u32,
            dt: world.scene.sim.frame_dt,
            seed: world.scene.sim.seed,
            notes: meta.notes,
        };
        let rec = EventRecorder::begin(out, manifest, &statics)?;
        Ok(Self { rec, chunk_keys, bond_keys, pending: None, qs_probe })
    }

    pub fn begin_frame(&mut self, world: &mut World) -> Result<(), String> {
        if self.pending.is_some() {
            return Err("begin_frame called twice without end_frame".into());
        }
        world.solver.reset_peak_utilizations();
        let n = self.chunk_keys.len();
        let e = self.bond_keys.len();
        let mut chunk_state = vec![0.0f32; n * CHUNK_STATE_DIM];
        let mut chunk_gravity_local = vec![0.0f32; n * 3];
        // Load bins filled at end_frame from World::learned_bins (contact + scripted).
        let chunk_load_bins = vec![0.0f32; n * K * LOCAL6_DIM];
        let chunk_contact = vec![0i32; n];
        let g = world.solver.config.gravity;
        for (ni, &(s, c)) in self.chunk_keys.iter().enumerate() {
            let st = &world.solver.chunks[s][c];
            write_chunk_state(&mut chunk_state[ni * CHUNK_STATE_DIM..], st);
            let cl = &world.solver.clusters[st.cluster];
            let gl = cl.pose.inverse_transform_vector(g);
            chunk_gravity_local[ni * 3] = gl.x as f32;
            chunk_gravity_local[ni * 3 + 1] = gl.y as f32;
            chunk_gravity_local[ni * 3 + 2] = gl.z as f32;
        }
        let mut bond_joint_state = vec![0.0f32; e * JOINT_STATE_DIM];
        let mut bond_force_prev = vec![0.0f32; e * LOCAL6_DIM];
        for (ei, &(s, b)) in self.bond_keys.iter().enumerate() {
            let bond = &world.solver.bonds[s][b];
            bond_joint_state[ei * JOINT_STATE_DIM..(ei + 1) * JOINT_STATE_DIM]
                .copy_from_slice(&pack_joint_state(&bond.joint));
            write_local6(&mut bond_force_prev[ei * LOCAL6_DIM..], &bond.force);
        }
        let bond_qs_force = if self.qs_probe {
            qs_probe_forces(world, &self.bond_keys)?
        } else {
            bond_force_prev.clone()
        };
        let t0 = world.solver.time;
        world.learned_bins = Some(LearnedFrameBins::new(&world.solver, K, t0, world.scene.sim.frame_dt));
        self.pending = Some(Pending {
            t0,
            event_cursor: world.solver.events.len(),
            chunk_state,
            bond_joint_state,
            bond_force_prev,
            bond_qs_force,
            chunk_load_bins,
            chunk_contact,
            chunk_gravity_local,
        });
        Ok(())
    }

    pub fn end_frame(&mut self, world: &mut World) -> Result<(), String> {
        let mut pending = self.pending.take().ok_or("end_frame without begin_frame")?;
        let bins = world.learned_bins.take();
        let n = self.chunk_keys.len();
        let e = self.bond_keys.len();
        let dt = world.scene.sim.frame_dt.max(1e-12);
        if let Some(bins) = bins {
            fill_load_bins(&mut pending, &self.chunk_keys, world, &bins);
        }
        let mut fail_class = vec![0i32; e];
        let mut fail_mode = vec![0i32; e];
        for ev in &world.solver.events[pending.event_cursor..] {
            if let SolverEvent::Broken { time, structure, bond, mode, .. } = ev {
                let Some(ei) = self.bond_keys.iter().position(|&k| k == (*structure, *bond)) else {
                    continue;
                };
                let frac = ((*time - pending.t0) / dt).clamp(0.0, 0.999999);
                let bin = (frac * K as f64).floor() as i32 + 1;
                fail_class[ei] = bin;
                fail_mode[ei] = mode_to_i32(*mode);
            }
        }
        let mut peak = vec![0.0f32; e];
        let mut joint_end = vec![0.0f32; e * JOINT_STATE_DIM];
        let mut force_end = vec![0.0f32; e * LOCAL6_DIM];
        for (ei, &(s, b)) in self.bond_keys.iter().enumerate() {
            let bond = &world.solver.bonds[s][b];
            peak[ei] = bond.peak_utilization as f32;
            joint_end[ei * JOINT_STATE_DIM..(ei + 1) * JOINT_STATE_DIM]
                .copy_from_slice(&pack_joint_state(&bond.joint));
            write_local6(&mut force_end[ei * LOCAL6_DIM..], &bond.force);
            if !bond.alive && fail_class[ei] == 0 {
                fail_class[ei] = K as i32;
            }
        }
        let mut chunk_end = vec![0.0f32; n * CHUNK_STATE_DIM];
        let mut delta = vec![0.0f32; n * CHUNK_STATE_DIM];
        for (ni, &(s, c)) in self.chunk_keys.iter().enumerate() {
            let st = &world.solver.chunks[s][c];
            write_chunk_state(&mut chunk_end[ni * CHUNK_STATE_DIM..], st);
            for k in 0..CHUNK_STATE_DIM {
                delta[ni * CHUNK_STATE_DIM + k] =
                    chunk_end[ni * CHUNK_STATE_DIM + k] - pending.chunk_state[ni * CHUNK_STATE_DIM + k];
            }
        }
        let frame = FrameRecord {
            chunk_state: pending.chunk_state,
            chunk_load_bins: pending.chunk_load_bins,
            chunk_contact: pending.chunk_contact,
            chunk_gravity_local: pending.chunk_gravity_local,
            bond_joint_state: pending.bond_joint_state,
            bond_force_prev: pending.bond_force_prev,
            bond_qs_force: pending.bond_qs_force,
            bond_fail_class: fail_class,
            bond_peak_util: peak,
            bond_fail_mode: fail_mode,
            bond_joint_state_end: joint_end,
            bond_force_end: force_end,
            chunk_state_delta: delta,
        };
        self.rec.record_frame(&frame)?;
        Ok(())
    }

    pub fn finish(self) -> Result<PathBuf, String> {
        self.rec.finish()
    }

    pub fn root(&self) -> &Path {
        self.rec.root()
    }
}

fn fill_load_bins(pending: &mut Pending, chunk_keys: &[(usize, usize)], world: &World, bins: &LearnedFrameBins) {
    for (ni, &(s, c)) in chunk_keys.iter().enumerate() {
        let st = &world.solver.chunks[s][c];
        if !st.active {
            continue;
        }
        let cl = &world.solver.clusters[st.cluster];
        let to_local = |v: Vec3| cl.pose.inverse_transform_vector(v);
        if s < bins.contacted.len() && c < bins.contacted[s].len() && bins.contacted[s][c] {
            pending.chunk_contact[ni] = 1;
        }
        for bin in 0..K {
            let f = to_local(bins.mean_force(bin, s, c));
            let tq = to_local(bins.mean_torque(bin, s, c));
            let o = (ni * K + bin) * LOCAL6_DIM;
            pending.chunk_load_bins[o] = f.x as f32;
            pending.chunk_load_bins[o + 1] = f.y as f32;
            pending.chunk_load_bins[o + 2] = f.z as f32;
            pending.chunk_load_bins[o + 3] = tq.x as f32;
            pending.chunk_load_bins[o + 4] = tq.y as f32;
            pending.chunk_load_bins[o + 5] = tq.z as f32;
        }
    }
}

fn mode_to_i32(mode: Option<FailureMode>) -> i32 {
    match mode {
        None => 0,
        Some(FailureMode::Tension) => 1,
        Some(FailureMode::Shear) => 2,
        Some(FailureMode::Compression) => 3,
        Some(FailureMode::Buckling) => 4,
    }
}

fn write_chunk_state(dst: &mut [f32], st: &stress_ref::solver::ChunkState) {
    let v = [
        st.u.x, st.u.y, st.u.z, st.th.x, st.th.y, st.th.z, st.v.x, st.v.y, st.v.z, st.w.x, st.w.y, st.w.z,
    ];
    for (i, x) in v.into_iter().enumerate() {
        dst[i] = x as f32;
    }
}

fn write_local6(dst: &mut [f32], f: &Local6) {
    for (i, x) in f.to_array().into_iter().enumerate() {
        dst[i] = x as f32;
    }
}

fn qs_probe_forces(world: &World, bond_keys: &[(usize, usize)]) -> Result<Vec<f32>, String> {
    let mut scratch: ReferenceSolver = world.solver.clone();
    let mut loads = ChunkLoads::new(&scratch);
    let g = scratch.config.gravity;
    for (s, chunks) in scratch.structures.iter().enumerate() {
        for (c, ch) in chunks.chunks.iter().enumerate() {
            if !scratch.chunks[s][c].active {
                continue;
            }
            loads.force[s][c] = g * ch.mass;
        }
    }
    let opts = StaticOptions { cascade: false, ..Default::default() };
    let _ = scratch.solve_static_all(&loads, &opts, world.scene.sim.frame_dt);
    let mut out = vec![0.0f32; bond_keys.len() * LOCAL6_DIM];
    for (ei, &(s, b)) in bond_keys.iter().enumerate() {
        write_local6(&mut out[ei * LOCAL6_DIM..], &scratch.bonds[s][b].force);
    }
    Ok(out)
}

fn extract_static(world: &World) -> Result<(Vec<(usize, usize)>, Vec<(usize, usize)>, StaticFeatures), String> {
    let mut chunk_keys = Vec::new();
    let mut chunk_of: Vec<Vec<Option<usize>>> = world
        .solver
        .structures
        .iter()
        .map(|s| vec![None; s.chunks.len()])
        .collect();
    let mut mat_ids: BTreeMap<String, i32> = BTreeMap::new();
    let mut next_mat = 0i32;
    for (s, st) in world.solver.structures.iter().enumerate() {
        for (c, ch) in st.chunks.iter().enumerate() {
            if !world.solver.chunks[s][c].active {
                continue;
            }
            let id = chunk_keys.len();
            chunk_of[s][c] = Some(id);
            chunk_keys.push((s, c));
            mat_ids.entry(ch.material.clone()).or_insert_with(|| {
                let id = next_mat;
                next_mat += 1;
                id
            });
        }
    }
    let n = chunk_keys.len();
    let mut bond_keys = Vec::new();
    for (s, bonds) in world.solver.bonds.iter().enumerate() {
        for (b, bond) in bonds.iter().enumerate() {
            if !bond.alive {
                continue;
            }
            let ga = bond.geometry.a;
            let gb = bond.geometry.b;
            if chunk_of[s][ga].is_none() || chunk_of[s][gb].is_none() {
                continue;
            }
            bond_keys.push((s, b));
        }
    }
    let e = bond_keys.len();
    let mut statics = StaticFeatures {
        chunk_mass: vec![0.0; n],
        chunk_inertia: vec![0.0; n * 3],
        chunk_half_extents: vec![0.0; n * 3],
        chunk_material_id: vec![0; n],
        chunk_support: vec![0; n],
        chunk_level: vec![0; n],
        chunk_coarse_parent: vec![-1; n],
        bond_area: vec![0.0; e],
        bond_width: vec![0.0; e * 2],
        bond_offset: vec![0.0; e * 3],
        bond_stiffness: vec![0.0; e * LOCAL6_DIM],
        bond_strength: vec![0.0; e * 3],
        bond_fracture_energy: vec![0.0; e * 3],
        bond_rebar: vec![0.0; e * BOND_REBAR_DIM],
        bond_weibull: vec![0.0; e],
        bond_buckling_length: vec![0.0; e],
        bond_endpoints: vec![0; e * 2],
    };
    for (ni, &(s, c)) in chunk_keys.iter().enumerate() {
        let ch = &world.solver.structures[s].chunks[c];
        statics.chunk_mass[ni] = ch.mass as f32;
        let inert = [
            ch.inertia.m[0][0] as f32,
            ch.inertia.m[1][1] as f32,
            ch.inertia.m[2][2] as f32,
        ];
        statics.chunk_inertia[ni * 3..ni * 3 + 3].copy_from_slice(&inert);
        statics.chunk_half_extents[ni * 3] = ch.half_extents.x as f32;
        statics.chunk_half_extents[ni * 3 + 1] = ch.half_extents.y as f32;
        statics.chunk_half_extents[ni * 3 + 2] = ch.half_extents.z as f32;
        statics.chunk_material_id[ni] = mat_ids[&ch.material];
        statics.chunk_support[ni] = match ch.support {
            Support::None => 0,
            Support::Fixed => 1,
            Support::Pinned => 2,
        };
        statics.chunk_level[ni] = ch.level as i32;
        statics.chunk_coarse_parent[ni] = match ch.parent {
            Some(p) => chunk_of[s][p].map(|x| x as i32).unwrap_or(-1),
            None => -1,
        };
    }
    for (ei, &(s, b)) in bond_keys.iter().enumerate() {
        let bond = &world.solver.bonds[s][b];
        let g = &bond.geometry;
        let ia = chunk_of[s][g.a].unwrap() as i32;
        let ib = chunk_of[s][g.b].unwrap() as i32;
        statics.bond_endpoints[ei * 2] = ia;
        statics.bond_endpoints[ei * 2 + 1] = ib;
        statics.bond_area[ei] = g.area as f32;
        statics.bond_width[ei * 2] = g.width[0] as f32;
        statics.bond_width[ei * 2 + 1] = g.width[1] as f32;
        let d = g.rb - g.ra;
        statics.bond_offset[ei * 3] = d.dot(g.t1) as f32;
        statics.bond_offset[ei * 3 + 1] = d.dot(g.t2) as f32;
        statics.bond_offset[ei * 3 + 2] = d.dot(g.normal) as f32;
        write_local6(&mut statics.bond_stiffness[ei * LOCAL6_DIM..], &bond.stiffness.as_local());
        let str = &bond.strength;
        statics.bond_strength[ei * 3] = str.tensile as f32;
        statics.bond_strength[ei * 3 + 1] = str.cohesion as f32;
        statics.bond_strength[ei * 3 + 2] = if str.compressive.is_finite() { str.compressive as f32 } else { 0.0 };
        statics.bond_fracture_energy[ei * 3] = str.g_tension as f32;
        statics.bond_fracture_energy[ei * 3 + 1] = str.g_shear as f32;
        statics.bond_fracture_energy[ei * 3 + 2] = str.g_compression as f32;
        if let Some(r) = bond.rebar {
            statics.bond_rebar[ei * BOND_REBAR_DIM] = r.k_axial as f32;
            statics.bond_rebar[ei * BOND_REBAR_DIM + 1] = r.k_dowel as f32;
            statics.bond_rebar[ei * BOND_REBAR_DIM + 2] = r.yield_force as f32;
            statics.bond_rebar[ei * BOND_REBAR_DIM + 3] = r.dowel_capacity as f32;
            statics.bond_rebar[ei * BOND_REBAR_DIM + 4] = r.rupture_work as f32;
        }
        statics.bond_weibull[ei] = bond.weibull as f32;
        let src = bond.source;
        statics.bond_buckling_length[ei] = world.solver.structures[s].bonds[src]
            .buckling_length
            .unwrap_or(0.0) as f32;
    }
    Ok((chunk_keys, bond_keys, statics))
}
