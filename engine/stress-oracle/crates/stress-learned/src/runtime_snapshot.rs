//! Targetless frame-start bridge for native Python models. This is a read-only
//! observation contract, not a World restart or a learned step implementation.
use crate::{joint_history, oracle, recorder::StaticFeatures, replay, response_history};
use serde::Serialize;
use std::collections::HashSet;
use stress_ref::world::World;

pub const FORMAT: &str = "learned-native-start/1";

/// Bind once at the same boundary as OracleSession::begin. Slot identity and the
/// initial displacement reference stay fixed across fragment coordinate resets.
#[derive(Serialize)]
pub struct RuntimeGraph {
    pub format: &'static str,
    pub scene: serde_json::Value,
    pub context: serde_json::Value,
    pub statics: StaticFeatures,
    pub dt: f64,
    pub initial_time: f64,
    chunk_keys: Vec<(usize, usize)>,
    bond_keys: Vec<(usize, usize)>,
    #[serde(skip)]
    chunk_slots: HashSet<(usize, usize)>,
    #[serde(skip)]
    bond_slots: HashSet<(usize, usize)>,
}

#[derive(Serialize)]
pub struct NativeStart {
    pub format: &'static str,
    pub world: replay::ReplayFrame,
    pub chunk_state: Vec<f32>,
    pub chunk_gravity_local: Vec<f32>,
    pub bond_joint_state: Vec<f32>,
    pub bond_force_prev: Vec<f32>,
    /// Only ductility fields 4/5 may be null, explicitly meaning +infinity.
    pub joint_values: Vec<[Option<f64>; 21]>,
    pub joint_flags: Vec<[i32; 4]>,
    pub response_values: Vec<[f64; 18]>,
}

/// Optional f64 features computed from the current native state only. Kept
/// separate so existing snapshot consumers do not pay extra kinematics work.
#[derive(Serialize)]
pub struct BoundaryKinematics {
    pub format: &'static str,
    pub time: f64,
    /// Generalized displacement (6), then its actual kinematic rate (6).
    /// Fixed RuntimeGraph bond order; retired bonds have zero placeholders.
    pub values: Vec<[f64; 12]>,
}

/// Exact physical chunk/contact history accompanying the legacy NativeStart.
/// Joint history and world poses remain in that companion snapshot. This still
/// is not a self-contained World restart (scheduling/coupling state is separate).
#[derive(Serialize)]
pub struct PhysicalHistory {
    pub format: &'static str,
    pub time: f64,
    pub chunks: Vec<PhysicalChunk>,
    pub contacts: stress_ref::world::contact_history::ContactHistory,
}

#[derive(Serialize)]
pub struct PhysicalChunk {
    pub key: (usize, usize),
    pub cluster_id: Option<u64>,
    pub active: bool,
    pub removed: bool,
    /// Raw cluster-frame u, theta, v, omega; unlike the legacy float32 feature
    /// these values contain neither an initial-state residual nor normalization.
    pub state: [[f64; 3]; 4],
    pub inertia_scale: f64,
    pub mass: f64,
    pub inertia: [[f64; 3]; 3],
    /// Translation and rotation constraints in the structure/cluster coordinates.
    pub fixed: [bool; 6],
}

impl RuntimeGraph {
    pub fn chunk_keys(&self)->&[(usize,usize)] { &self.chunk_keys }
    pub fn bond_keys(&self)->&[(usize,usize)] { &self.bond_keys }

    pub fn capture_with_physical_history(&self, world: &World) -> Result<(NativeStart, PhysicalHistory), String> {
        let snapshot = self.capture(world)?;
        let mut chunks = Vec::with_capacity(self.chunk_keys.len());
        for &(s, c) in &self.chunk_keys {
            let chunk = world.solver.chunks[s][c];
            let data = &world.solver.structures[s].chunks[c];
            let state = [chunk.u.to_array(), chunk.th.to_array(), chunk.v.to_array(), chunk.w.to_array()];
            if state.iter().flatten().chain([&chunk.inertia_scale, &data.mass]).chain(data.inertia.m.iter().flatten()).any(|x| !x.is_finite()) {
                return Err("nonfinite exact physical chunk history".into());
            }
            let fixed = std::array::from_fn(|d| match data.support {
                stress_ref::scene::Support::Fixed => true,
                stress_ref::scene::Support::Pinned => d < 3,
                stress_ref::scene::Support::None => false,
            });
            chunks.push(PhysicalChunk { key: (s, c),
                cluster_id: if chunk.active { Some(world.solver.clusters[chunk.cluster].id) } else { None },
                active: chunk.active, removed: chunk.removed, state, inertia_scale: chunk.inertia_scale,
                mass: data.mass, inertia: data.inertia.m, fixed });
        }
        Ok((snapshot, PhysicalHistory { format: "native-physical-history/1", time: world.solver.time,
            chunks, contacts: world.contact_history()? }))
    }

    pub fn capture_with_kinematics(&self, world: &World) -> Result<(NativeStart, BoundaryKinematics), String> {
        let snapshot = self.capture(world)?;
        let mut values = Vec::with_capacity(self.bond_keys.len());
        for &(s, b) in &self.bond_keys {
            let bond = &world.solver.bonds[s][b];
            let mut row = [0.; 12];
            if bond.alive {
                let g = &bond.geometry;
                let a = world.solver.chunks[s][g.a];
                let c = world.solver.chunks[s][g.b];
                let (d, rate) = if world.solver.config.methods.corotational_bonds {
                    let cr = g.corotation(a.u, a.th, c.u, c.th);
                    (cr.deformation(), cr.variation(a.v, a.w, c.v, c.w))
                } else {
                    (g.kinematics(a.u, a.th, c.u, c.th), g.kinematics(a.v, a.w, c.v, c.w))
                };
                row[..6].copy_from_slice(&d.to_array());
                row[6..].copy_from_slice(&rate.to_array());
                if row.iter().any(|x| !x.is_finite()) {return Err("nonfinite boundary kinematics".into())}
            }
            values.push(row);
        }
        Ok((snapshot, BoundaryKinematics {format:"native-boundary-kinematics/1",time:world.solver.time,values}))
    }

    pub fn new(world: &World) -> Result<Self, String> {
        if !world.scene.sim.frame_dt.is_finite()
            || world.scene.sim.frame_dt <= 0.
            || !world.solver.time.is_finite()
        {
            return Err("invalid native frame time".into());
        }
        let (chunk_keys, bond_keys, statics) = oracle::extract_static(world)?;
        if chunk_keys.is_empty() {
            return Err("no active chunks".into());
        }
        let mut scene = serde_json::to_value(&world.scene).map_err(|e| e.to_string())?;
        scene["sim"]["methods"] = serde_json::to_value(world.scene.sim.methods).unwrap();
        let centers: Vec<_> = chunk_keys
            .iter()
            .map(|&(s, c)| world.solver.chunk_position(s, c).to_array())
            .collect();
        let frames: Vec<_> = bond_keys
            .iter()
            .map(|&(s, b)| {
                let g = &world.solver.bonds[s][b].geometry;
                [g.t1.to_array(), g.t2.to_array(), g.normal.to_array()]
            })
            .collect();
        let context = serde_json::json!({"centers":centers, "chunk_keys":chunk_keys, "bond_keys":bond_keys,
            "bond_frames":frames, "source_scene":world.scene.name, "solver_methods":world.scene.sim.methods,
            "load_basis":"frame_start_cluster", "qs_probe":false});
        let chunk_slots = chunk_keys.iter().copied().collect();
        let bond_slots = bond_keys.iter().copied().collect();
        Ok(Self {
            format: FORMAT,
            scene,
            context,
            statics,
            dt: world.scene.sim.frame_dt,
            initial_time: world.solver.time,
            chunk_keys,
            bond_keys,
            chunk_slots,
            bond_slots,
        })
    }

    pub fn capture(&self, world: &World) -> Result<NativeStart, String> {
        let mut scene = serde_json::to_value(&world.scene).map_err(|e| e.to_string())?;
        scene["sim"]["methods"] = serde_json::to_value(world.scene.sim.methods).unwrap();
        if scene != self.scene {
            return Err("native snapshot source scene changed; rebind the graph".into());
        }
        let step = (world.solver.time - self.initial_time) / self.dt;
        if !step.is_finite() || step < 0. || (step - step.round()).abs() > 1e-6 {
            return Err("native snapshot must be captured at a frame boundary".into());
        }
        // No silent graph truncation when refinement or newly spawned objects
        // invalidate this frozen-graph model contract.
        for (s, chunks) in world.solver.chunks.iter().enumerate() {
            for (c, chunk) in chunks.iter().enumerate() {
                if chunk.active && !self.chunk_slots.contains(&(s, c)) {
                    return Err("native snapshot: newly active chunk outside graph".into());
                }
            }
        }
        for (s, bonds) in world.solver.bonds.iter().enumerate() {
            for (b, bond) in bonds.iter().enumerate() {
                if bond.alive && !self.bond_slots.contains(&(s, b)) {
                    return Err("native snapshot: newly alive bond outside graph".into());
                }
            }
        }
        if world.scene.name != self.context["source_scene"] || world.scene.sim.frame_dt != self.dt {
            return Err("native snapshot scene identity or timestep changed".into());
        }
        let local = oracle::capture_local_start(world, &self.chunk_keys, &self.bond_keys);
        let mut joint_values = Vec::new();
        let mut joint_flags = Vec::new();
        let mut response_values = Vec::new();
        for &(s, b) in &self.bond_keys {
            let bond = &world.solver.bonds[s][b];
            let row = joint_history::pack(&bond.joint);
            let mode = joint_history::mode_code(bond.joint.mode);
            joint_history::unpack(row, bond.joint.rebar_broken, mode)?;
            joint_values.push(row.map(|x| if x == f64::INFINITY { None } else { Some(x) }));
            joint_flags.push([
                bond.joint.rebar_broken as i32,
                mode,
                bond.alive as i32,
                bond.connected() as i32,
            ]);
            let response = response_history::pack(bond);
            if response.iter().any(|x| !x.is_finite()) {
                return Err("nonfinite native response".into());
            }
            response_values.push(response);
        }
        if local
            .chunk_state
            .iter()
            .chain(&local.chunk_gravity_local)
            .chain(&local.bond_joint_state)
            .chain(&local.bond_force_prev)
            .any(|x| !x.is_finite())
        {
            return Err("native local inputs exceed finite float32 range".into());
        }
        Ok(NativeStart {
            format: FORMAT,
            world: replay::capture(world, &self.chunk_keys),
            chunk_state: local.chunk_state,
            chunk_gravity_local: local.chunk_gravity_local,
            bond_joint_state: local.bond_joint_state,
            bond_force_prev: local.bond_force_prev,
            joint_values,
            joint_flags,
            response_values,
        })
    }
}
