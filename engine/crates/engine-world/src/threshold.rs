//! The fracture lab's `threshold` mode in `World` (D1b, [`StrengthModel::Threshold`]).
//!
//! The lab's `ThresholdWorld` (stress-rubble `threshold.rs`) runs on the world's own
//! device and `WgpuSim`: each bonded cluster is one body, rubble's contacts become chunk
//! loads (stress-gpu `EngineIngest`), and bonds break and bodies split in place on the GPU
//! (stress-gpu `GpuFracture`). Every body a split can make is reserved when the
//! destructible is placed, so a split allocates nothing on the GPU.
//!
//! The asset data comes from the stress oracle's import of the same source the lab
//! imports ([`StressSource`]): chunk materials, volumes and supports, bond capacities and
//! areas, strike geometry. Nothing is converted or simplified at load.
//!
//! Frames: the fracture works in each structure's Z-up stress frame and converts to
//! rubble's Y-up frame itself. The world keeps every piece body's origin at the instance
//! frame (the render contract) and its solver frame at the piece's centre of mass and
//! principal axes, as the GPU computed them: each step's readback carries the chunk
//! owners, the bond states and the body frames, and a step that split adopts the new
//! bodies as [`BodyId`]s and reports `BondsBroken` and `Split`.
//!
//! Breaks are forward-only: loads from the step just taken break bonds and split bodies
//! before the next step; nothing is rewound or solved again.
//!
//! [`StrengthModel::Threshold`]: crate::StrengthModel::Threshold
//! [`BodyId`]: crate::BodyId

use std::sync::Arc;

/// The stress oracle's input for an asset (the fracture lab's stress-solver modes): its
/// scene, imported from the same GLB level, cuboid houses or pack the asset's chunks come
/// from. Built by the asset loaders; present only with the `stress` feature.
#[derive(Clone)]
pub struct StressSource {
    /// The import, or why it failed (reported when a stress model is placed; the
    /// asset's other models do not need it).
    #[cfg(feature = "stress")]
    pub(crate) scene: Arc<Result<stress_ref::scene::Scene, String>>,
    #[cfg(not(feature = "stress"))]
    _scene: Arc<()>,
    /// The asset frame is `placement` plus the stress frame turned Y up (the lab's
    /// placement of the source).
    pub(crate) placement: [f64; 3],
    /// Per stress chunk (structure by structure, chunks in order): its glTF node.
    pub(crate) nodes: Vec<Option<u32>>,
}

impl std::fmt::Debug for StressSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StressSource")
            .field("placement", &self.placement)
            .field("nodes", &self.nodes.len())
            .finish_non_exhaustive()
    }
}

impl PartialEq for StressSource {
    fn eq(&self, other: &Self) -> bool {
        #[cfg(feature = "stress")]
        let same = Arc::ptr_eq(&self.scene, &other.scene);
        #[cfg(not(feature = "stress"))]
        let same = Arc::ptr_eq(&self._scene, &other._scene);
        same && self.placement == other.placement && self.nodes == other.nodes
    }
}

/// A body whose contacts load threshold destructibles as a projectile's do (the lab's
/// shots): its mass and its single sphere or box shape give the impact, and strike
/// prediction checks it each step (`World::body_set_projectile`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Projectile {
    /// Pa.
    pub youngs_modulus: f32,
    pub poisson_ratio: f32,
}

impl Default for Projectile {
    /// The lab's projectile material (stress-ref `elastic_steel`: steel's stiffness).
    fn default() -> Self {
        Self {
            youngs_modulus: 200e9,
            poisson_ratio: 0.3,
        }
    }
}

#[cfg(feature = "stress")]
pub(crate) use imp::*;

#[cfg(not(feature = "stress"))]
#[derive(Default)]
pub(crate) struct Fracture {
    pub fixed_plan: Option<(u32, u32)>,
}

#[cfg(not(feature = "stress"))]
impl Fracture {
    pub fn active(&self) -> bool {
        false
    }
    pub fn body_destroyed(&mut self, _body: u32) {}
}

#[cfg(feature = "stress")]
mod imp {
    use super::{Projectile, StressSource};
    use crate::defs::{BodyDef, BodyType, Geometry, MassData};
    use crate::destruction::{ActorInfo, DestructibleDef, StrengthModel};
    use crate::events::DestructionEvent;
    use crate::math::SolverFrame;
    use crate::{BodyId, DestructibleAssetDesc, DestructibleId, Owner, Transform, World};
    use engine_schema::convex::ConvexHull;
    use engine_schema::{AvbdParams, BodyState, FLAG_CONTACT_GAP, FLAG_WORLD_INERTIA};
    use glam::{DMat3, DQuat, DVec3, Vec3};
    use std::sync::Arc;
    use stress_gpu::fracture::{self, FractureTables, GpuFracture, INFO_SUPPORT};
    use stress_gpu::gpu::Gpu;
    use stress_gpu::ingest::{
        ColliderInfo, EngineBuffers, EngineIngest, KIND_BODY, KIND_STATIC, NONE,
    };
    use stress_ref::capability::Profile;
    use stress_ref::engine::{BodyGeometry, BodyKey};
    use stress_ref::scene::{Scene, Support};

    /// The words of a fracture counters block (`fracture.slang` `C_*`).
    const COUNTER_WORDS: u64 = 16;

    /// The rotation from the stress frame (Z up) to rubble's (Y up): -90 degrees about x.
    fn to_y_up() -> DQuat {
        let h = std::f64::consts::FRAC_1_SQRT_2;
        DQuat::from_xyzw(-h, 0.0, 0.0, h)
    }

    fn dv(v: stress_ref::math::Vec3) -> DVec3 {
        DVec3::new(v.x, v.y, v.z)
    }

    fn dq(q: stress_ref::math::Quat) -> DQuat {
        DQuat::from_xyzw(q.x, q.y, q.z, q.w)
    }

    fn f3(v: DVec3) -> [f32; 3] {
        [v.x as f32, v.y as f32, v.z as f32]
    }

    fn f4(q: DQuat) -> [f32; 4] {
        [q.x as f32, q.y as f32, q.z as f32, q.w as f32]
    }

    impl StressSource {
        /// A Shardwright GLB level as the lab's stress modes import it
        /// (`stress_export::shardwright_stress_json`, ground bonds kept).
        pub(crate) fn from_glb(
            asset: &engine_fracture::FractureAsset,
            level: u8,
            placement: [f64; 3],
        ) -> Self {
            let json = engine_fracture::stress_export::shardwright_stress_json(asset, level);
            let scene =
                stress_ref::shardwright::import_value_with(&json, &asset.physics.name, true);
            // The importer makes the level's fragments its first chunks, in the asset's
            // order; the ground anchors it adds follow them.
            let nodes = asset
                .physics
                .fragments
                .iter()
                .filter(|f| f.level == level)
                .map(|f| u32::try_from(f.node).ok())
                .collect();
            Self {
                scene: Arc::new(scene),
                placement,
                nodes,
            }
        }

        /// The cuboid house as the lab's stress modes import it.
        pub(crate) fn cuboid_house(
            options: &engine_fracture::cuboid::CuboidOptions,
            placement: [f64; 3],
        ) -> Self {
            let pieces = engine_fracture::cuboid::cuboid_pieces(options);
            let json = engine_fracture::cuboid::cuboid_stress_json(&pieces, &[[0.0; 3]]);
            let scene = stress_ref::shardwright::import_value_with(&json, "cuboid_houses", true);
            Self {
                scene: Arc::new(scene),
                placement,
                nodes: Vec::new(),
            }
        }

        /// A scene pack as the lab's stress modes import it (the stress profiles' import
        /// options).
        pub(crate) fn from_pack(bytes: &[u8], key: &str, placement: [f64; 3]) -> Self {
            let scene = serde_json::from_slice(bytes)
                .map_err(|e| format!("scene pack: {e}"))
                .and_then(|json| {
                    stress_ref::scene_pack::import_value_with(
                        &json,
                        key,
                        "structure",
                        key,
                        Profile::Threshold.import_options(),
                    )
                })
                .map(|(scene, _notes)| scene);
            Self {
                scene: Arc::new(scene),
                placement,
                nodes: Vec::new(),
            }
        }

        /// The stress scene, or why the import failed.
        pub fn scene(&self) -> anyhow::Result<&Scene> {
            self.scene
                .as_ref()
                .as_ref()
                .map_err(|e| anyhow::anyhow!("the asset's stress import failed: {e}"))
        }

        /// The height (asset frame, y) of the ground the stress scene stands on, if it
        /// has one: the lab's ground plane, which a host adds itself.
        pub fn ground_height(&self) -> Option<f32> {
            let scene = self.scene().ok()?;
            scene
                .ground
                .as_ref()
                .map(|g| (self.placement[1] + g.height) as f32)
        }

        /// The source's placement: the asset frame is this plus the stress frame turned
        /// Y up (the lab's placement of the source).
        pub fn placement(&self) -> [f64; 3] {
            self.placement
        }

        /// The stress scene as the threshold mode runs it (the lab's `stress_load`): bond
        /// strengths scaled, `dt` as its frame, the profile applied.
        pub fn threshold_scene(&self, strength_scale: f32, dt: f32) -> anyhow::Result<Scene> {
            let mut scene = self.scene()?.clone();
            let k = strength_scale as f64;
            if k != 1.0 {
                let used: std::collections::BTreeSet<String> = scene
                    .bodies
                    .iter()
                    .flat_map(|b| &b.bonds)
                    .map(|b| b.material.clone())
                    .collect();
                for name in used {
                    if let Some(m) = scene.materials.get_mut(&name) {
                        m.tensile_strength *= k;
                        m.compressive_strength *= k;
                        m.cohesion *= k;
                        m.shear_cap = m.shear_cap.map(|c| c * k);
                    }
                }
            }
            scene.sim.frame_dt = dt as f64;
            scene
                .validate()
                .map_err(|e| anyhow::anyhow!("stress scene: {e}"))?;
            let mut scene = scene.with_derived();
            scene.sim.methods = stress_ref::scene::Methods::established();
            Profile::Threshold.apply_scene(&mut scene);
            scene
                .materials
                .entry("projectile".into())
                .or_insert_with(stress_ref::builders::elastic_steel);
            Ok(scene)
        }

        /// The lab's engine settings for the threshold mode (`RubbleEngine` and
        /// `ThresholdWorld::step_frame`): never asleep, zero-gap contact, world inertia,
        /// the profile's iterations, the scene's frame and contact offset.
        pub(crate) fn threshold_solver(&self) -> Option<AvbdParams> {
            let scene = self.scene().ok()?;
            let (iterations, _) = Profile::Threshold.engine_policy();
            let mut p = AvbdParams {
                gravity_y: scene.gravity[2] as f32,
                flags: FLAG_CONTACT_GAP | FLAG_WORLD_INERTIA,
                sleep_lin: -1.0,
                sleep_ang: -1.0,
                dt: scene.sim.frame_dt as f32,
                contact_offset: 0.01,
                ..AvbdParams::default()
            };
            if let Some(n) = iterations {
                p.iterations = n;
            }
            Some(p)
        }
    }

    /// One placed threshold destructible's GPU state and what the host knows of it.
    pub(crate) struct ThresholdSlot {
        pub destructible: u32,
        ingest: EngineIngest,
        fracture: GpuFracture,
        tables: FractureTables,
        /// Per structure: its stress frame in the asset frame (rotation, translation).
        structures: Vec<(DQuat, DVec3)>,
        /// Per chunk: its shape slots.
        chunk_shapes: Vec<Vec<u32>>,
        /// Per chunk its body slot, per bond whether it holds, as last read back.
        chunk_body: Vec<u32>,
        intact: Vec<bool>,
        /// Reserved body slots no piece has taken yet.
        spare: Vec<u32>,
        /// Every body slot this destructible owns (pieces and spares).
        pub owned: Vec<u32>,
        /// The scene's stiffness scale (strike prediction's plane-strain modulus).
        stiffness_scale: f64,
        /// The ingest's default contact side for a collider (static geometry).
        static_info: ColliderInfo,
        /// Colliders the ingest now sees as projectiles.
        projectile_colliders: Vec<u32>,
        pub nodes: Vec<Option<u32>>,
        /// `DestructibleDef::support_loss`: what the pass reads.
        support: Option<SupportLoss>,
    }

    /// Support loss's inputs for one threshold destructible (`support_loss.rs`).
    struct SupportLoss {
        /// Per bond (the tables' order): its compressive capacity (N).
        compressive: Vec<f32>,
        /// Per chunk: its bonds.
        bonds_of: Vec<Vec<u32>>,
        /// Gravity in the stress frame (m/s²): the chunks' centres are in it.
        gravity: [f32; 3],
        /// Bonds the pass has broken that no readback has shown broken yet.
        sent: Vec<bool>,
        /// `DestructibleDef::support_loss_dynamic_factor`: on the increase of load in a
        /// piece whose supports changed this frame.
        dynamic_factor: f32,
        /// Per bond: its flow load (N) when the pass last ran on its piece (from the intact
        /// structure at placement), the `before` of a sudden loss's amplification.
        load: Vec<f32>,
    }

    /// The pass on one piece (`chunks`): the bonds (the tables' indices) it breaks, and its
    /// intact bonds' static flow loads now. `sudden`: the dynamic factor of a sudden loss of
    /// support (the bonds' `SupportLoss::load` as their loads before).
    fn evaluate_piece(
        tables: &FractureTables,
        intact: &[bool],
        sl: &SupportLoss,
        chunks: &[u32],
        sudden: Option<f32>,
    ) -> (Vec<u32>, Vec<(u32, f32)>) {
        let bonds = &tables.bonds;
        let mut local = std::collections::HashMap::with_capacity(chunks.len());
        for (j, &c) in chunks.iter().enumerate() {
            local.insert(c, j as u32);
        }
        let mut piece_bonds = Vec::new();
        let mut table_index = Vec::new();
        for &c in chunks.iter() {
            for &b in &sl.bonds_of[c as usize] {
                let [a, o, _, _] = bonds[b as usize];
                // Each bond once (from its first chunk), intact and inside the piece.
                if a != c || !intact[b as usize] || sl.sent[b as usize] {
                    continue;
                }
                if let (Some(&la), Some(&lo)) = (local.get(&a), local.get(&o)) {
                    piece_bonds.push((la, lo, sl.compressive[b as usize]));
                    table_index.push(b);
                }
            }
        }
        let mass: Vec<f32> = chunks
            .iter()
            .map(|&c| tables.chunks[c as usize][0][3])
            .collect();
        let support: Vec<bool> = chunks
            .iter()
            .map(|&c| tables.info[c as usize] & INFO_SUPPORT != 0)
            .collect();
        // Chunk centres in their structure's stress frame (a piece is in one structure).
        let centre: Vec<[f32; 3]> = chunks
            .iter()
            .map(|&c| {
                let r = tables.chunks[c as usize][0];
                [r[0], r[1], r[2]]
            })
            .collect();
        let piece = crate::support_loss::Piece {
            mass: &mass,
            support: &support,
            centre: &centre,
            bonds: &piece_bonds,
        };
        let before: Vec<f32> = table_index.iter().map(|&b| sl.load[b as usize]).collect();
        let sudden = sudden.map(|factor| crate::support_loss::Sudden {
            factor,
            before: &before,
        });
        let r = crate::support_loss::failing(&piece, sl.gravity, sudden.as_ref());
        let broken = r.bonds.iter().map(|&i| table_index[i as usize]).collect();
        let loads = table_index
            .iter()
            .zip(&r.load)
            .map(|(&b, &l)| (b, l as f32))
            .collect();
        (broken, loads)
    }

    /// The world's threshold destructibles.
    #[derive(Default)]
    pub(crate) struct Fracture {
        gpu: Option<Gpu>,
        pub slots: Vec<Option<ThresholdSlot>>,
        /// Registered projectiles: body slot and generation, material.
        projectiles: Vec<(u32, u32, Projectile)>,
        projectiles_dirty: bool,
        /// The plan hints the lab fixes (`set_plan_hints`): the contact capacity, every
        /// colour its own dispatch.
        pub fixed_plan: Option<(u32, u32)>,
    }

    impl Fracture {
        pub fn active(&self) -> bool {
            self.slots.iter().any(|s| s.is_some())
        }

        /// A destroyed body stops being a projectile (synced before the next step).
        pub fn body_destroyed(&mut self, body: u32) {
            let before = self.projectiles.len();
            self.projectiles.retain(|p| p.0 != body);
            self.projectiles_dirty |= self.projectiles.len() != before;
        }
    }

    /// One step's fracture readback for slot `slot`.
    #[derive(Clone, Copy, Debug)]
    pub(crate) enum FractureRead {
        Counters(u32),
        ChunkBodies(u32),
        Intact(u32),
        Frames(u32),
    }

    fn buffers(sim: &engine_webgpu::WgpuSim) -> EngineBuffers<'_> {
        EngineBuffers {
            bodies: sim.body_buffer(),
            colliders: sim.collider_buffer(),
            contacts: sim.contact_buffer(),
            counters: sim.counters_buffer(),
        }
    }

    impl World {
        /// Mark a body as a projectile for threshold destructibles (`None` clears it): its
        /// contacts load chunks as the lab's shots do, and strike prediction checks it
        /// each step. It needs mass and one sphere or box shape. At most 64.
        pub fn body_set_projectile(
            &mut self,
            body: BodyId,
            projectile: Option<Projectile>,
        ) -> anyhow::Result<()> {
            let b = self.live_body(body)?;
            if let Some(p) = &projectile {
                // Its mass and shape are checked now (they are read again each sync).
                self.projectile_shot(b as u32, p, 1.0)?;
            }
            let f = &mut self.fracture;
            f.projectiles.retain(|p| p.0 != b as u32);
            if let Some(p) = projectile {
                anyhow::ensure!(
                    f.projectiles.len() < fracture::MAX_SHOTS,
                    "at most {} projectiles (strike prediction's shot table)",
                    fracture::MAX_SHOTS
                );
                f.projectiles.push((b as u32, body.generation(), p));
            }
            f.projectiles_dirty = true;
            if self.fracture.active() {
                self.sync_projectiles()?;
            }
            Ok(())
        }

        /// A projectile's strike record and contact side, from its body as it is now.
        fn projectile_shot(
            &self,
            b: u32,
            p: &Projectile,
            stiffness_scale: f64,
        ) -> anyhow::Result<(fracture::Shot, u32, ColliderInfo)> {
            let slot = &self.bodies[b as usize];
            anyhow::ensure!(
                slot.body_type == BodyType::Dynamic,
                "projectile body {b} is not dynamic"
            );
            let mass = slot
                .mass
                .map(|m| m.mass)
                .filter(|m| *m > 0.0)
                .ok_or_else(|| anyhow::anyhow!("projectile body {b} has no mass"))?;
            let [s] = slot.shapes[..] else {
                anyhow::bail!("projectile body {b} needs exactly one shape");
            };
            let shape = &self.shapes[s as usize];
            let (face, bound) = match &shape.geometry {
                Geometry::Sphere { radius, .. } => (*radius, 1.01 * radius + 1e-3),
                Geometry::Box { half, .. } => (
                    half.iter().copied().fold(f32::INFINITY, f32::min),
                    1.01 * Vec3::from(*half).length() + 1e-3,
                ),
                _ => anyhow::bail!("projectile body {b}: only sphere and box projectiles"),
            };
            let nu = p.poisson_ratio as f64;
            let e_plane = p.youngs_modulus as f64 * stiffness_scale / (1.0 - nu * nu);
            let shot = fracture::Shot {
                body: b,
                bound,
                face,
                mass,
                modulus: e_plane as f32,
            };
            let info = ColliderInfo {
                a: [NONE, b, KIND_BODY, 0],
                b: [mass, p.youngs_modulus, face, 0.0],
            };
            Ok((shot, shape.collider, info))
        }

        /// Every threshold destructible's shot table and the projectiles' contact sides.
        pub(crate) fn sync_projectiles(&mut self) -> anyhow::Result<()> {
            self.fracture.projectiles_dirty = false;
            let live: Vec<(u32, Projectile)> = self
                .fracture
                .projectiles
                .iter()
                .filter(|p| {
                    self.bodies
                        .get(p.0 as usize)
                        .is_some_and(|s| s.alive && s.generation == p.1)
                })
                .map(|p| (p.0, p.2))
                .collect();
            let gpu = self
                .fracture
                .gpu
                .clone()
                .expect("a threshold destructible's GPU");
            for k in 0..self.fracture.slots.len() {
                let Some(scale) = self.fracture.slots[k].as_ref().map(|t| t.stiffness_scale) else {
                    continue;
                };
                let mut shots = Vec::with_capacity(live.len());
                let mut sides = Vec::with_capacity(live.len());
                for (b, p) in &live {
                    let (shot, collider, info) = self.projectile_shot(*b, p, scale)?;
                    shots.push(shot);
                    sides.push((collider, info));
                }
                let t = self.fracture.slots[k].as_mut().expect("live slot");
                let stride = std::mem::size_of::<ColliderInfo>() as u64;
                // A collider that stopped being a projectile (cleared, or its body
                // destroyed and the slot reused) is static geometry again.
                for &c in &t.projectile_colliders {
                    if !sides.iter().any(|s| s.0 == c) {
                        let info = t.static_info;
                        gpu.queue.write_buffer(
                            &t.ingest.collider_info,
                            stride * c as u64,
                            bytemuck::bytes_of(&info),
                        );
                    }
                }
                for (c, info) in &sides {
                    gpu.queue.write_buffer(
                        &t.ingest.collider_info,
                        stride * *c as u64,
                        bytemuck::bytes_of(info),
                    );
                }
                t.projectile_colliders = sides.iter().map(|s| s.0).collect();
                t.fracture
                    .set_shots(&gpu, &shots)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            }
            Ok(())
        }

        /// Place a threshold destructible: one body per bonded cluster (the lab's
        /// `ThresholdWorld::new`), a reserved slot for every other piece a split can make,
        /// the fracture tables and the contact ingest.
        pub(crate) fn create_threshold(
            &mut self,
            def: &DestructibleDef,
            asset: Arc<DestructibleAssetDesc>,
            index: usize,
        ) -> anyhow::Result<()> {
            let source = asset.stress.as_ref().ok_or_else(|| anyhow::anyhow!("StrengthModel::Threshold needs an asset with the stress oracle's input (from_glb, cuboid_house or from_pack)"))?;
            // The ingest reads rubble's canonical contact rows, as the lab's engine keeps them.
            anyhow::ensure!(
                self.contact_storage == crate::ContactStorage::Canonical,
                "StrengthModel::Threshold reads rubble's canonical contact rows: create the world with ContactStorage::Canonical (it uses {:?})",
                self.contact_storage
            );
            let params = self.sim.params();
            let scene = source.threshold_scene(def.strength_scale, params.dt)?;
            let world = stress_ref::world::World::new(&scene);
            let m = &world.solver;
            let chunk_count: usize = m.structures.iter().map(|s| s.chunks.len()).sum();
            let capacity = self.sim.capacity();
            let gpu = match &self.fracture.gpu {
                Some(g) => g.clone(),
                None => {
                    let g = Gpu::from_device(
                        self.adapter.clone(),
                        self.sim.device().clone(),
                        self.sim.queue().clone(),
                    );
                    self.fracture.gpu = Some(g.clone());
                    g
                }
            };
            let instance = def.transform;
            let (ip, iq) = (
                DVec3::from(instance.p().as_dvec3()),
                instance.q().as_dquat().normalize(),
            );
            let placement = DVec3::from(source.placement);
            let r = to_y_up();
            let mut bodies = world.initial_engine_bodies();
            bodies.retain(|b| matches!(b.key, BodyKey::Cluster(_)));
            // Each structure's stress frame in the asset frame (from its clusters' poses).
            let mut structures: Vec<Option<(DQuat, DVec3)>> = vec![None; m.structures.len()];
            for b in &bodies {
                let BodyKey::Cluster(id) = b.key else {
                    continue;
                };
                let cl = m
                    .clusters
                    .iter()
                    .find(|c| c.id == id)
                    .expect("a cluster of the world");
                let frame = (r * dq(b.pose.rotation), placement + r * dv(b.pose.position));
                structures[cl.structure].get_or_insert(frame);
            }
            let structures: Vec<(DQuat, DVec3)> = structures
                .into_iter()
                .map(|s| s.unwrap_or((r, placement)))
                .collect();
            // Global chunk numbering, structure by structure.
            let mut offset = Vec::with_capacity(m.structures.len());
            let mut chunk_of = Vec::with_capacity(chunk_count);
            for (s, st) in m.structures.iter().enumerate() {
                offset.push(chunk_of.len() as u32);
                chunk_of.extend((0..st.chunks.len()).map(|c| (s, c)));
            }
            let global = |s: usize, c: usize| offset[s] + c as u32;
            // The clusters' bodies, cooked as the lab cooks them.
            let mut cooker = stress_rubble::Cooker::default();
            let shape_def = crate::ShapeDef {
                update_body_mass: false,
                ..def.shape.clone()
            };
            let material = self.intern_material(&shape_def.material)?;
            let mut chunk_shapes: Vec<Vec<u32>> = vec![Vec::new(); chunk_count];
            let mut chunk_parts: Vec<Vec<stress_rubble::Part>> = vec![Vec::new(); chunk_count];
            let mut chunk_cols: Vec<Vec<u32>> = vec![Vec::new(); chunk_count];
            let mut chunk_body = vec![u32::MAX; chunk_count];
            let mut cluster_bodies = Vec::with_capacity(bodies.len());
            for b in &bodies {
                let BodyKey::Cluster(id) = b.key else {
                    continue;
                };
                let cl = m
                    .clusters
                    .iter()
                    .find(|c| c.id == id)
                    .expect("a cluster of the world");
                let (sq, st) = structures[cl.structure];
                let cooked = cooker.cook(b, &mut self.hulls);
                let BodyGeometry::Boxes(shapes) = &b.geometry else {
                    anyhow::bail!("cluster {id} is not made of chunk shapes")
                };
                let dynamic = !b.fixed && cooked.mass > 0.0;
                let id = self.create_body(&BodyDef {
                    body_type: if dynamic {
                        BodyType::Dynamic
                    } else {
                        BodyType::Static
                    },
                    position: instance.position,
                    rotation: instance.rotation,
                    is_awake: !def.asleep,
                    user_data: def.user_data,
                    ..BodyDef::default()
                })?;
                let bi = id.index() as usize;
                self.bodies[bi].owner = Owner::Piece;
                // The solver frame: the cooked centre and principal axes in the asset
                // frame (the body origin is the instance frame).
                let com = dv(cooked.com);
                let principal = dq(cooked.principal);
                let axes = (sq * principal).normalize();
                let frame = SolverFrame {
                    center: (st + sq * com).as_vec3(),
                    axes: axes.as_quat(),
                };
                let moments = dv(cooked.moments);
                let mass = dynamic.then(|| mass_data(cooked.mass, frame, moments));
                let slot = &mut self.bodies[bi];
                slot.mass = mass;
                slot.principal = moments.as_vec3();
                slot.frames = vec![(self.step, frame)];
                // The record as the lab writes it (in f64, then rounded once).
                let mut state = crate::body_state(slot);
                state.position = f3(ip + iq * (st + sq * com));
                state.orientation = f4((iq * axes).normalize());
                state.initial_pos = state.position;
                state.initial_rot = state.orientation;
                self.sim.write_bodies(bi as u32, &[state]);
                for ((s, col), part) in shapes.iter().zip(&cooked.colliders).zip(&cooked.parts) {
                    let (sc, cc) = s
                        .chunk
                        .ok_or_else(|| anyhow::anyhow!("cluster shape without a chunk"))?;
                    let k = global(sc, cc) as usize;
                    let local = Transform::new(
                        f3(st + sq * dv(s.local.position)),
                        f4((sq * dq(s.local.rotation)).normalize()),
                    );
                    let geometry = match &s.hull {
                        Some(points)
                            if col.shape == engine_schema::ShapeType::ConvexHull as u32 =>
                        {
                            let pts: Vec<Vec3> = points.iter().map(|p| dv(*p).as_vec3()).collect();
                            let (hull, shift) = ConvexHull::from_points(&pts)
                                .map_err(|e| anyhow::anyhow!("chunk {k} hull: {e:?}"))?
                                .recentered();
                            let at = local.q() * shift + local.p();
                            Geometry::Hull {
                                hull,
                                transform: Transform::new(at.to_array(), local.rotation),
                            }
                        }
                        _ => Geometry::Box {
                            half: f3(dv(s.half_extents)),
                            transform: local,
                        },
                    };
                    let mut c = engine_schema::Collider {
                        body: bi as u32,
                        material,
                        ..*col
                    };
                    shape_def.filter.apply(&mut c);
                    c.material |= crate::contacts::event_bits(&shape_def);
                    let (shape, collider) = self.add_shape_record(bi, &shape_def, geometry, c)?;
                    chunk_shapes[k].push(shape);
                    chunk_cols[k].push(collider);
                    chunk_parts[k].push(*part);
                    chunk_body[k] = bi as u32;
                }
                cluster_bodies.push((bi as u32, cl.id, cl.structure, com, principal));
            }
            if cooked_hulls(&chunk_parts) {
                self.sim.sync_hulls(&mut self.hulls);
            }
            // Any chunk can split off onto a dynamic body on the GPU, before (or without)
            // this host adopting the split: the broadphase grid is sized for the largest
            // chunk collider moving from the start.
            let reach = chunk_cols
                .iter()
                .flatten()
                .map(|&c| engine_schema::convex::collider_radius(&self.sim.collider(c)))
                .fold(0.0f32, f32::max);
            self.sim.reserve_movable_radius(reach);
            // Reserved slots for every other piece a split can make: static, parked, out
            // of every pair until a split re-parents a chunk's colliders onto one.
            let spare_count = chunk_count.saturating_sub(m.clusters.len());
            let spare_defs = vec![
                BodyDef {
                    user_data: def.user_data,
                    ..BodyDef::default()
                };
                spare_count
            ];
            let spare_ids = self.create_bodies(&spare_defs)?;
            let mut spare = Vec::with_capacity(spare_count);
            for id in spare_ids {
                let i = id.index();
                let mut parked = BodyState::static_plane_support();
                parked.position = crate::graveyard(i);
                parked.initial_pos = parked.position;
                self.sim.write_bodies(i, &[parked]);
                let slot = &mut self.bodies[i as usize];
                slot.owner = Owner::Piece;
                slot.alive = false;
                spare.push(i);
            }
            let owned: Vec<u32> = cluster_bodies
                .iter()
                .map(|c| c.0)
                .chain(spare.iter().copied())
                .collect();
            let body_count = owned.iter().copied().max().map_or(0, |b| b + 1);
            let mut t = FractureTables {
                max_bodies: capacity.max_bodies,
                pool_cap: stress_rubble::threshold::pool_cap(capacity.max_contacts),
                impact_peak: true,
                band: (scene.gravity[2].abs() * scene.sim.frame_dt) as f32,
                dt: scene.sim.frame_dt as f32,
                strike_speed: stress_ref::world::tiered_impact_speed() as f32,
                frames: vec![[[0.0; 4], [0.0, 0.0, 0.0, 1.0]]; body_count as usize],
                free: vec![0; body_count as usize],
                ..Default::default()
            };
            for &s in &spare {
                t.free[s as usize] = 1;
            }
            let f = |x: f64| x as f32;
            let mut compressive = Vec::new();
            for &(s, c) in &chunk_of {
                let k = global(s, c) as usize;
                let ch = &m.structures[s].chunks[c];
                let parts = &chunk_parts[k];
                anyhow::ensure!(!parts.is_empty(), "chunk {s}/{c}: no parts cooked");
                let mass: f64 = parts.iter().map(|p| p.mass).sum();
                let centre = parts
                    .iter()
                    .fold(stress_ref::math::Vec3::ZERO, |a, p| a + p.center * p.mass)
                    * (1.0 / mass.max(1e-300));
                let mut i = stress_ref::math::Mat3::ZERO;
                for p in parts {
                    let d = p.center - centre;
                    i += p.inertia
                        + (stress_ref::math::Mat3::diag(stress_ref::math::Vec3::splat(d.norm2()))
                            - stress_ref::math::Mat3::outer(d, d))
                            * p.mass;
                }
                let first = t.parts.len() as u32;
                for (p, &col) in parts.iter().zip(&chunk_cols[k]) {
                    let q = p.rotation;
                    t.parts.push([
                        [
                            p.origin.x as f32,
                            p.origin.y as f32,
                            p.origin.z as f32,
                            f32::from_bits(col),
                        ],
                        [q.x as f32, q.y as f32, q.z as f32, q.w as f32],
                    ]);
                }
                t.chunks.push([
                    [f(centre.x), f(centre.y), f(centre.z), f(mass)],
                    [f(i.m[0][0]), f(i.m[1][1]), f(i.m[2][2]), f(i.m[0][1])],
                    [
                        f(i.m[0][2]),
                        f(i.m[1][2]),
                        f32::from_bits(first),
                        f32::from_bits(parts.len() as u32),
                    ],
                    [
                        f(ch.youngs_modulus),
                        f(ch.poisson_ratio),
                        f(ch.volume().cbrt()),
                        f(ch.half_extents.min_elem()),
                    ],
                ]);
                let mat = scene.material(&ch.material);
                let e_plane = mat.youngs_modulus * scene.sim.stiffness_scale
                    / (1.0 - mat.poisson_ratio * mat.poisson_ratio);
                let (bc, h, bq) = (
                    ch.center,
                    ch.half_extents,
                    stress_rubble::threshold::rotation_quat(&ch.rotation),
                );
                t.geometry.push([
                    [f(bc.x), f(bc.y), f(bc.z), f(h.norm())],
                    [f(h.x), f(h.y), f(h.z), f(e_plane)],
                    [f(bq.x), f(bq.y), f(bq.z), f(bq.w)],
                ]);
                t.info.push(if ch.support != Support::None {
                    INFO_SUPPORT
                } else {
                    0
                });
                t.chunk_body.push(chunk_body[k]);
            }
            for &(b, id, structure, com, principal) in &cluster_bodies {
                let cl = m.clusters.iter().find(|c| c.id == id).expect("a cluster");
                let (c, p) = (com.as_vec3(), principal.as_quat());
                t.frames[b as usize] = [
                    [c.x, c.y, c.z, f32::from_bits(structure as u32)],
                    [p.x, p.y, p.z, p.w],
                ];
                for &bi in &cl.bonds {
                    let bond = &m.bonds[cl.structure][bi];
                    if !bond.connected() {
                        continue;
                    }
                    let capacity = bond.strength.tensile * bond.geometry.area * bond.weibull;
                    let source = &scene.bodies[cl.structure].bonds[bond.source];
                    compressive.push(
                        (scene.material(&source.material).compressive_strength * bond.geometry.area)
                            as f32,
                    );
                    t.bonds.push([
                        global(cl.structure, bond.geometry.a),
                        global(cl.structure, bond.geometry.b),
                        (capacity as f32).to_bits(),
                        (bond.geometry.area as f32).to_bits(),
                    ]);
                }
            }
            // The ingest's collider table: every collider the static side by default (the
            // lab's ground), the chunks' on their bodies, the projectiles' (synced below).
            let ground_modulus = scene
                .ground
                .as_ref()
                .map(|g| scene.material(&g.material).youngs_modulus)
                .unwrap_or(3e10);
            let static_info = ColliderInfo {
                a: [NONE, NONE, KIND_STATIC, 0],
                b: [f32::INFINITY, ground_modulus as f32, 1.0, 0.0],
            };
            let mut table = vec![static_info; capacity.max_colliders as usize];
            let mut root_of_body = std::collections::HashMap::new();
            for (k, &b) in t.chunk_body.iter().enumerate() {
                root_of_body.entry(b).or_insert(k);
            }
            for k in 0..chunk_count {
                let body = t.chunk_body[k];
                let root = root_of_body[&body];
                let mass = self.bodies[body as usize]
                    .mass
                    .map_or(f32::INFINITY, |m| m.mass);
                let rr = t.chunks[root][3];
                for &col in &chunk_cols[k] {
                    table[col as usize] = ColliderInfo {
                        a: [k as u32, body, KIND_BODY, 1],
                        b: [mass, rr[0], rr[3], 0.0],
                    };
                }
            }
            let ingest = EngineIngest::new(
                &gpu,
                capacity.max_contacts,
                1,
                capacity.max_bodies,
                chunk_count as u32,
                &table,
            );
            let fracture = GpuFracture::new(&gpu, &t, &buffers(&self.sim), &ingest)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let bond_count = t.bonds.len();
            let slot = ThresholdSlot {
                destructible: index as u32,
                ingest,
                fracture,
                chunk_body: t.chunk_body.clone(),
                tables: t,
                structures,
                chunk_shapes,
                intact: vec![true; bond_count],
                spare,
                owned,
                stiffness_scale: scene.sim.stiffness_scale,
                static_info,
                projectile_colliders: Vec::new(),
                nodes: source.nodes.clone(),
                support: None,
            };
            let mut slot = slot;
            if def.support_loss {
                let mut bonds_of = vec![Vec::new(); slot.tables.chunks.len()];
                for (i, b) in slot.tables.bonds.iter().enumerate() {
                    bonds_of[b[0] as usize].push(i as u32);
                    bonds_of[b[1] as usize].push(i as u32);
                }
                let g = scene.gravity;
                let mut sl = SupportLoss {
                    sent: vec![false; compressive.len()],
                    load: vec![0.0; compressive.len()],
                    compressive,
                    bonds_of,
                    gravity: g.map(|x| x as f32),
                    dynamic_factor: def.support_loss_dynamic_factor,
                };
                // The intact structure's flow: every bond's load before any loss.
                for &(b, ..) in &cluster_bodies {
                    let chunks: Vec<u32> = (0..slot.chunk_body.len() as u32)
                        .filter(|&c| slot.chunk_body[c as usize] == b)
                        .collect();
                    let (_, loads) = evaluate_piece(&slot.tables, &slot.intact, &sl, &chunks, None);
                    for (bond, load) in loads {
                        sl.load[bond as usize] = load;
                    }
                }
                slot.support = Some(sl);
            }
            let k = match self.fracture.slots.iter().position(|s| s.is_none()) {
                Some(k) => k,
                None => {
                    self.fracture.slots.push(None);
                    self.fracture.slots.len() - 1
                }
            };
            self.fracture.slots[k] = Some(slot);
            // A plan from counters read a frame or more late would differ run to run: the
            // lab fixes it (the hash at the contact capacity, every colour its own
            // dispatch).
            let plan = (capacity.max_contacts, capacity.max_colors.min(32));
            self.fracture.fixed_plan = Some(plan);
            self.sim.set_plan_hints(plan.0, plan.1);
            let d = &mut self.destructibles[index];
            d.model = StrengthModel::Threshold;
            d.threshold = Some(k as u32);
            d.chunk_bodies = self.fracture.slots[k]
                .as_ref()
                .expect("placed")
                .chunk_body
                .iter()
                .map(|&b| BodyId::new(b, self.bodies[b as usize].generation))
                .collect();
            d.bond_joints = vec![None; bond_count];
            d.bond_intact = vec![true; bond_count];
            self.sync_projectiles()?;
            Ok(())
        }

        /// The fracture's part of a step, between `ingest.begin_frame` (before the
        /// engine's step) and the readback: the ingest of the step's contacts, the loads,
        /// the breaks and the splits.
        pub(crate) fn encode_fracture_begin(&self, enc: &mut wgpu::CommandEncoder) {
            let e = buffers(&self.sim);
            for t in self.fracture.slots.iter().flatten() {
                t.ingest.begin_frame(enc, &e);
            }
        }

        pub(crate) fn encode_fracture(&mut self, enc: &mut wgpu::CommandEncoder) {
            let Some(gpu) = self.fracture.gpu.clone() else {
                return;
            };
            let e = buffers(&self.sim);
            let offset = self.sim.last_contact_offset();
            let dt = self.sim.params().dt as f64;
            let step = self.sim.steps_done();
            for t in self.fracture.slots.iter_mut().flatten() {
                t.ingest.substep(&gpu, enc, &e, offset, 0, dt);
                t.ingest.merge(&gpu, enc, &e, 1);
                t.fracture.encode_frame(&gpu, enc, step);
            }
        }
    }

    impl Fracture {
        /// The readback copies of every threshold destructible's state after a step:
        /// its counters, chunk owners, bond states and body frames.
        pub(crate) fn extras(&self) -> Vec<crate::readback::Extra<'_, crate::Extra>> {
            let mut out = Vec::new();
            for (k, t) in self.slots.iter().enumerate() {
                let Some(t) = t else { continue };
                let k = k as u32;
                let l = t.fracture.layout;
                let f = &t.fracture;
                for (read, buffer, offset, size) in [
                    (
                        FractureRead::Counters(k),
                        &f.atom,
                        4 * l.a_counters() as u64,
                        4 * COUNTER_WORDS,
                    ),
                    (
                        FractureRead::ChunkBodies(k),
                        &f.work,
                        4 * l.w_body() as u64,
                        4 * l.n as u64,
                    ),
                    (
                        FractureRead::Intact(k),
                        &f.work,
                        4 * l.w_alive() as u64,
                        4 * l.e as u64,
                    ),
                    (FractureRead::Frames(k), &f.frames, 0, 32 * l.b as u64),
                ] {
                    if size > 0 {
                        out.push(crate::readback::Extra {
                            kind: crate::Extra::Fracture(read),
                            buffer,
                            offset,
                            size,
                        });
                    }
                }
            }
            out
        }
    }

    impl World {
        /// A step's fracture readback: bonds broken and bodies split on the GPU become
        /// events, and the pieces' bodies take their new solver frames from the next step
        /// on. A fracture error (no free slot, a mass sum out of range, an unsettled union,
        /// a full impact pulse pool) fails the frame.
        pub(crate) fn fracture_frame(
            &mut self,
            step: u64,
            reads: &[(FractureRead, &[u8])],
        ) -> anyhow::Result<Vec<DestructionEvent>> {
            let mut out = Vec::new();
            for k in 0..self.fracture.slots.len() {
                let find = |want: fn(u32) -> FractureRead| {
                    reads
                        .iter()
                        .find(|(r, _)| {
                            std::mem::discriminant(r) == std::mem::discriminant(&want(k as u32))
                                && read_slot(*r) == k as u32
                        })
                        .map(|(_, b)| *b)
                };
                let (Some(counters), Some(chunks)) = (
                    find(FractureRead::Counters),
                    find(FractureRead::ChunkBodies),
                ) else {
                    continue;
                };
                let counters: &[u32] = bytemuck::cast_slice(counters);
                anyhow::ensure!(counters[3] == 0, "step {step}: threshold fracture error {:#x} (1: no free body slot, 2: mass sum out of range, 4: union did not settle, 8: impact pulse pool full)", counters[3]);
                let chunks: Vec<u32> = bytemuck::cast_slice(chunks).to_vec();
                let intact: Vec<bool> = find(FractureRead::Intact)
                    .map(|b| {
                        bytemuck::cast_slice::<u8, u32>(b)
                            .iter()
                            .map(|&x| x != 0)
                            .collect()
                    })
                    .unwrap_or_default();
                let frames: Vec<[f32; 4]> = find(FractureRead::Frames)
                    .map(|b| bytemuck::cast_slice(b).to_vec())
                    .unwrap_or_default();
                let events = self.apply_fracture(k, step, chunks, intact, &frames)?;
                let broken: Vec<u32> = events
                    .iter()
                    .filter_map(|e| match e {
                        DestructionEvent::BondsBroken { bonds, .. } => Some(bonds.clone()),
                        _ => None,
                    })
                    .flatten()
                    .collect();
                if !broken.is_empty() {
                    self.support_loss(k, &broken);
                }
                out.extend(events);
            }
            Ok(out)
        }

        fn apply_fracture(
            &mut self,
            k: usize,
            step: u64,
            chunks: Vec<u32>,
            intact: Vec<bool>,
            frames: &[[f32; 4]],
        ) -> anyhow::Result<Vec<DestructionEvent>> {
            let t = self.fracture.slots[k].as_mut().expect("live slot");
            let d = t.destructible as usize;
            let broken: Vec<u32> = (0..t.intact.len() as u32)
                .filter(|&b| {
                    t.intact[b as usize] && !intact.get(b as usize).copied().unwrap_or(true)
                })
                .collect();
            let moved: Vec<u32> = (0..chunks.len() as u32)
                .filter(|&c| chunks[c as usize] != t.chunk_body[c as usize])
                .collect();
            if broken.is_empty() && moved.is_empty() {
                return Ok(Vec::new());
            }
            for &b in &broken {
                t.intact[b as usize] = false;
            }
            let id = DestructibleId::new(d as u32, self.destructibles[d].generation);
            let mut out = Vec::new();
            if !broken.is_empty() {
                out.push(DestructionEvent::BondsBroken {
                    destructible: id,
                    bonds: broken.clone(),
                });
                for &b in &broken {
                    self.destructibles[d].bond_intact[b as usize] = false;
                }
            }
            if moved.is_empty() {
                return Ok(out);
            }
            let t = self.fracture.slots[k].as_mut().expect("live slot");
            let before = std::mem::replace(&mut t.chunk_body, chunks);
            // Bodies whose chunk set changed: the parents (where moved chunks were) and
            // the bodies they went to.
            let mut parents: Vec<u32> = moved.iter().map(|&c| before[c as usize]).collect();
            parents.sort_unstable();
            parents.dedup();
            let mut changed: Vec<u32> = moved
                .iter()
                .map(|&c| t.chunk_body[c as usize])
                .chain(parents.iter().copied())
                .collect();
            changed.sort_unstable();
            changed.dedup();
            let mut spare_taken = Vec::new();
            for &b in &changed {
                if let Some(pos) = t.spare.iter().position(|&s| s == b) {
                    t.spare.swap_remove(pos);
                    spare_taken.push(b);
                }
            }
            // Adopt the pieces' new bodies; re-frame every changed body from the step
            // after this one (this step's poses are from before the split).
            for &b in &spare_taken {
                let slot = &mut self.bodies[b as usize];
                slot.alive = true;
                slot.generation = slot.generation.wrapping_add(1).max(1);
                slot.shapes.clear();
                slot.last_transform = None;
                slot.last_solver = None;
                slot.last_asleep = false;
                slot.asleep = false;
            }
            for &b in &changed {
                self.reframe_piece(k, b, step + 1, frames)?;
            }
            // Splits: each parent's chunks before, grouped by where they are now.
            let t = self.fracture.slots[k].as_ref().expect("live slot");
            let chunk_body = t.chunk_body.clone();
            for &p in &parents {
                let members: Vec<u32> = (0..before.len() as u32)
                    .filter(|&c| before[c as usize] == p)
                    .collect();
                let mut groups: Vec<u32> =
                    members.iter().map(|&c| chunk_body[c as usize]).collect();
                groups.sort_unstable();
                groups.dedup();
                if groups.len() < 2 && groups.first() == Some(&p) {
                    continue;
                }
                let parent = self.piece_info(k, &members, p);
                let children = groups
                    .iter()
                    .map(|&g| {
                        let chunks: Vec<u32> = members
                            .iter()
                            .copied()
                            .filter(|&c| chunk_body[c as usize] == g)
                            .collect();
                        self.piece_info(k, &chunks, g)
                    })
                    .collect();
                out.push(DestructionEvent::Split {
                    destructible: id,
                    parent,
                    children,
                });
            }
            let dslot = &mut self.destructibles[d];
            for (c, &b) in chunk_body.iter().enumerate() {
                dslot.chunk_bodies[c] = BodyId::new(b, self.bodies[b as usize].generation);
            }
            dslot.actor = actor_labels(&chunk_body);
            Ok(out)
        }

        /// Support loss after a frame's breaks (`broken`): the pieces they touched carry their
        /// weight to their anchors, and the bonds it overloads break with the next frame.
        fn support_loss(&mut self, k: usize, broken: &[u32]) {
            let Some(gpu) = self.fracture.gpu.clone() else {
                return;
            };
            let t = self.fracture.slots[k].as_mut().expect("live slot");
            let Some(sl) = t.support.as_mut() else { return };
            for &b in broken {
                sl.sent[b as usize] = false;
            }
            let bonds = &t.tables.bonds;
            let mut bodies: Vec<u32> = broken
                .iter()
                .flat_map(|&b| {
                    [
                        t.chunk_body[bonds[b as usize][0] as usize],
                        t.chunk_body[bonds[b as usize][1] as usize],
                    ]
                })
                .collect();
            bodies.sort_unstable();
            bodies.dedup();
            // The touched pieces' chunks, in one sweep.
            let mut chunks_of: Vec<Vec<u32>> = vec![Vec::new(); bodies.len()];
            for (c, b) in t.chunk_body.iter().enumerate() {
                if let Ok(i) = bodies.binary_search(b) {
                    chunks_of[i].push(c as u32);
                }
            }
            let lost = crate::support_loss::lost_support(
                &bodies,
                &chunks_of,
                broken,
                |b| {
                    let [a, o, _, _] = bonds[b as usize];
                    (t.chunk_body[a as usize], t.chunk_body[o as usize])
                },
                |c| t.tables.info[c as usize] & INFO_SUPPORT != 0,
            );
            let mut over = Vec::new();
            for (i, chunks) in chunks_of.iter().enumerate() {
                // A sudden loss of support: the dynamic factor on the load's increase; else
                // the quasi-static flow.
                let sudden = (lost[i] && sl.dynamic_factor != 1.0).then_some(sl.dynamic_factor);
                let (broken, loads) = evaluate_piece(&t.tables, &t.intact, sl, chunks, sudden);
                over.extend(broken);
                for (bond, load) in loads {
                    sl.load[bond as usize] = load;
                }
            }
            for &b in &over {
                sl.sent[b as usize] = true;
            }
            t.fracture
                .break_bonds_sparse(&gpu, &over, &t.chunk_body, bonds);
        }

        /// A changed body's mass, solver frame (from the GPU's frame record), type and
        /// shapes, from `from_step` on.
        fn reframe_piece(
            &mut self,
            k: usize,
            b: u32,
            from_step: u64,
            frames: &[[f32; 4]],
        ) -> anyhow::Result<()> {
            let t = self.fracture.slots[k].as_ref().expect("live slot");
            let chunks: Vec<usize> = (0..t.chunk_body.len())
                .filter(|&c| t.chunk_body[c] == b)
                .collect();
            let shapes: Vec<u32> = chunks
                .iter()
                .flat_map(|&c| t.chunk_shapes[c].iter().copied())
                .collect();
            if chunks.is_empty() {
                // Every chunk left it (cannot happen: the largest piece keeps its body).
                return Ok(());
            }
            let rec = |i: usize| {
                frames
                    .get(i)
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("fracture frame {i} missing"))
            };
            let (c0, c1) = (rec(2 * b as usize)?, rec(2 * b as usize + 1)?);
            let structure = c0[3].to_bits() as usize;
            let (sq, st) = t.structures[structure];
            let com = DVec3::new(c0[0] as f64, c0[1] as f64, c0[2] as f64);
            let principal =
                DQuat::from_xyzw(c1[0] as f64, c1[1] as f64, c1[2] as f64, c1[3] as f64)
                    .normalize();
            let anchored = chunks.iter().any(|&c| t.tables.info[c] & INFO_SUPPORT != 0);
            // The piece's mass and inertia about its centre, summed from its chunks (f64;
            // the GPU sums the same records in fixed point), in the stress frame.
            let mut mass = 0.0;
            let mut first = DVec3::ZERO;
            for &c in &chunks {
                let r = t.tables.chunks[c];
                mass += r[0][3] as f64;
                first +=
                    DVec3::new(r[0][0] as f64, r[0][1] as f64, r[0][2] as f64) * r[0][3] as f64;
            }
            let centre = first / mass.max(1e-300);
            let mut inertia = DMat3::ZERO;
            for &c in &chunks {
                let r = t.tables.chunks[c];
                let i = DMat3::from_cols(
                    DVec3::new(r[1][0] as f64, r[1][3] as f64, r[2][0] as f64),
                    DVec3::new(r[1][3] as f64, r[1][1] as f64, r[2][1] as f64),
                    DVec3::new(r[2][0] as f64, r[2][1] as f64, r[1][2] as f64),
                );
                let dd = DVec3::new(r[0][0] as f64, r[0][1] as f64, r[0][2] as f64) - centre;
                inertia += i
                    + (DMat3::from_diagonal(DVec3::splat(dd.length_squared())) - outer(dd, dd))
                        * r[0][3] as f64;
            }
            let pm = DMat3::from_quat(principal);
            let local = pm.transpose() * inertia * pm;
            let moments = DVec3::new(local.x_axis.x, local.y_axis.y, local.z_axis.z);
            let frame = SolverFrame {
                center: (st + sq * com).as_vec3(),
                axes: (sq * principal).normalize().as_quat(),
            };
            let body_type = if anchored || mass <= 0.0 {
                BodyType::Static
            } else {
                BodyType::Dynamic
            };
            let slot = &mut self.bodies[b as usize];
            slot.body_type = body_type;
            slot.mass = (body_type == BodyType::Dynamic).then(|| mass_data(mass, frame, moments));
            slot.principal = moments.as_vec3();
            slot.frames.retain(|f| f.0 < from_step);
            slot.frames.push((from_step, frame));
            slot.shapes = shapes.clone();
            let state = crate::body_state(slot);
            for s in shapes {
                self.shapes[s as usize].body = b;
            }
            // The split was the GPU's (`fr_commit`, `fr_reparent`): the host's mirror of the
            // body and of its chunks' colliders follows it, with no upload. The step plan
            // reads that mirror: the hierarchical grid sizes its levels from the colliders of
            // moving bodies (`hierarchical_grid`), and a moving collider larger than the
            // levels a stale mirror gives goes to the oversized-static list, which never
            // queries, so it passed through the ground and every static piece.
            let t = self.fracture.slots[k].as_ref().expect("live slot");
            let com = Vec3::new(c0[0], c0[1], c0[2]);
            let to_body = glam::Quat::from_xyzw(c1[0], c1[1], c1[2], c1[3])
                .normalize()
                .conjugate();
            let mut cols = Vec::new();
            for &c in &chunks {
                let r = t.tables.chunks[c][2];
                let (first, count) = (r[2].to_bits() as usize, r[3].to_bits() as usize);
                for p in &t.tables.parts[first..first + count] {
                    let id = p[0][3].to_bits();
                    let mut col = self.sim.collider(id);
                    col.body = b;
                    col.local_pos =
                        (to_body * (Vec3::new(p[0][0], p[0][1], p[0][2]) - com)).to_array();
                    col.local_rot = (to_body * glam::Quat::from_array(p[1]))
                        .normalize()
                        .to_array();
                    cols.push((id, col));
                }
            }
            self.sim.mirror_bodies(b, &[state]);
            for (id, col) in cols {
                self.sim.mirror_colliders(id, &[col]);
            }
            Ok(())
        }

        fn piece_info(&self, k: usize, chunks: &[u32], body: u32) -> ActorInfo {
            let t = self.fracture.slots[k].as_ref().expect("live slot");
            // A parent keeps its slot and generation: its largest piece keeps the body.
            let generation = self.bodies[body as usize].generation;
            ActorInfo {
                chunks: chunks.to_vec(),
                bodies: vec![BodyId::new(body, generation)],
                anchored: chunks
                    .iter()
                    .any(|&c| t.tables.info[c as usize] & INFO_SUPPORT != 0),
                mass: chunks
                    .iter()
                    .map(|&c| t.tables.chunks[c as usize][0][3])
                    .sum(),
            }
        }

        /// The actor a threshold destructible's chunk is part of: every chunk on its body.
        pub(crate) fn threshold_actor(&self, k: usize, chunk: u32) -> anyhow::Result<ActorInfo> {
            let t = self.fracture.slots[k].as_ref().expect("live slot");
            let b = *t
                .chunk_body
                .get(chunk as usize)
                .ok_or_else(|| anyhow::anyhow!("chunk {chunk} out of range"))?;
            let chunks: Vec<u32> = (0..t.chunk_body.len() as u32)
                .filter(|&c| t.chunk_body[c as usize] == b)
                .collect();
            Ok(self.piece_info(k, &chunks, b))
        }

        pub(crate) fn threshold_chunk_node(
            &self,
            k: usize,
            chunk: u32,
        ) -> anyhow::Result<Option<u32>> {
            let t = self.fracture.slots[k].as_ref().expect("live slot");
            anyhow::ensure!(
                (chunk as usize) < t.chunk_body.len(),
                "chunk {chunk} out of range"
            );
            Ok(t.nodes.get(chunk as usize).copied().flatten())
        }

        /// Remove a threshold destructible's GPU state and every body it owns.
        pub(crate) fn destroy_threshold(&mut self, k: usize) -> anyhow::Result<()> {
            let t = self.fracture.slots[k].take().expect("live slot");
            for b in t.owned {
                let slot = &self.bodies[b as usize];
                if slot.alive {
                    self.bodies[b as usize].owner = Owner::User;
                    let id = BodyId::new(b, self.bodies[b as usize].generation);
                    self.destroy_body(id)?;
                } else {
                    // A reserved slot no piece took: free it.
                    let slot = &mut self.bodies[b as usize];
                    slot.owner = Owner::User;
                    slot.generation = slot.generation.wrapping_add(1).max(1);
                    slot.removed_at = Some(self.step);
                    self.free_bodies.push(b);
                }
            }
            if !self.fracture.active() {
                self.fracture.fixed_plan = None;
            }
            Ok(())
        }

        pub(crate) fn fracture_projectiles_dirty(&self) -> bool {
            self.fracture.projectiles_dirty
        }
    }

    fn read_slot(r: FractureRead) -> u32 {
        match r {
            FractureRead::Counters(k)
            | FractureRead::ChunkBodies(k)
            | FractureRead::Intact(k)
            | FractureRead::Frames(k) => k,
        }
    }

    fn cooked_hulls(parts: &[Vec<stress_rubble::Part>]) -> bool {
        parts
            .iter()
            .flatten()
            .any(|p| p.collider.shape == engine_schema::ShapeType::ConvexHull as u32)
    }

    fn outer(a: DVec3, b: DVec3) -> DMat3 {
        DMat3::from_cols(a * b.x, a * b.y, a * b.z)
    }

    /// Mass data in the body frame from a solver frame and principal moments.
    fn mass_data(mass: f64, frame: SolverFrame, moments: DVec3) -> MassData {
        let r = glam::Mat3::from_quat(frame.axes);
        let tensor = r * glam::Mat3::from_diagonal(moments.as_vec3()) * r.transpose();
        MassData {
            mass: mass as f32,
            center: frame.center.to_array(),
            rotational_inertia: tensor.to_cols_array_2d(),
        }
    }

    /// Each chunk's actor label: the lowest chunk on its body.
    pub(crate) fn actor_labels(chunk_body: &[u32]) -> Vec<u32> {
        let mut first = std::collections::HashMap::new();
        chunk_body
            .iter()
            .enumerate()
            .map(|(c, b)| *first.entry(*b).or_insert(c as u32))
            .collect()
    }
}
