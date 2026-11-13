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
    /// A scene pack's source, for the profiles that import it differently
    /// (`blast_gpu` keeps its Blast authoring).
    #[cfg(feature = "stress")]
    pub(crate) pack: Option<Arc<imp::PackSource>>,
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
        #[cfg(feature = "stress")]
        let same = same
            && match (&self.pack, &other.pack) {
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                (None, None) => true,
                _ => false,
            };
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
    /// Blast loading's threshold side (`crate::blast`): a child, so it reads the slots.
    mod blast {
        include!("blast_threshold.rs");
    }

    use super::{Projectile, StressSource};
    use crate::defs::{BodyDef, BodyType, Geometry, MassData};
    use crate::destruction::{ActorInfo, CrushMass, DestructibleDef, StrengthModel};
    use crate::events::{BreakCause, DestructionEvent};
    use crate::math::SolverFrame;
    use crate::{BodyId, DestructibleAssetDesc, DestructibleId, Owner, Transform, World};
    use engine_schema::convex::ConvexHull;
    use engine_schema::{AvbdParams, BodyState, FLAG_CONTACT_GAP, FLAG_WORLD_INERTIA};
    use glam::{DMat3, DQuat, DVec3, Vec3};
    use std::sync::Arc;
    use stress_gpu::fracture::{
        self, CrushEvent, CrushTables, FractureTables, GpuFracture, INFO_SUPPORT,
    };
    use stress_gpu::gpu::Gpu;
    use stress_gpu::ingest::{
        ColliderInfo, EngineBuffers, EngineIngest, KIND_BODY, KIND_STATIC, NONE,
    };
    use stress_ref::capability::Profile;
    use stress_ref::engine::{BodyGeometry, BodyKey};
    use stress_ref::scene::{Scene, Support};
    use stress_rubble::threshold::{BlastInputs, ProfileSetup};

    /// The stress oracle's profile a stress model runs.
    pub(crate) fn profile_of(model: StrengthModel) -> anyhow::Result<Profile> {
        Ok(match model {
            StrengthModel::Threshold => Profile::Threshold,
            StrengthModel::LoadPath => Profile::LoadPath,
            StrengthModel::ImpactLocal => Profile::ImpactLocal,
            StrengthModel::BlastGpu => Profile::BlastGpu,
            StrengthModel::Bonds => anyhow::bail!("the Bonds model has no stress profile"),
        })
    }

    /// A destructible's fracture set-up: its model's profile with its options
    /// (stress-rubble `ProfileSetup`, the one place both the lab and the world decide it).
    pub(crate) fn setup_of(def: &DestructibleDef) -> anyhow::Result<ProfileSetup> {
        let options = stress_rubble::threshold::FractureOptions {
            contact: def.fracture.contact_fracture,
            strike_prediction: def.fracture.strike_prediction,
            ..Default::default()
        };
        ProfileSetup::new(profile_of(def.model)?, &options).map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// A scene pack's JSON, kept to import it again under a profile whose import options
    /// differ from the threshold's (`blast_gpu`: its Blast authoring), once.
    pub(crate) struct PackSource {
        json: serde_json::Value,
        key: String,
        blast: std::sync::OnceLock<Result<Scene, String>>,
    }

    /// What a fracture's error bits mean (`fracture.slang` `C_ERROR`).
    pub(crate) fn fracture_error_text(bits: u32) -> String {
        const BITS: [(u32, &str); 8] = [
            (1, "no free body slot"),
            (2, "a mass sum out of range"),
            (4, "a union did not settle"),
            (8, "an impact pulse found no room in the pool"),
            (16, "a crush found the fragment pool spent"),
            (32, "more crushes than the crush events' room"),
            (64, "a load-path block was not positive definite"),
            (128, "an event list was full"),
        ];
        let named: Vec<&str> = BITS.iter().filter(|b| bits & b.0 != 0).map(|b| b.1).collect();
        if named.is_empty() {
            "unknown".into()
        } else {
            named.join("; ")
        }
    }

    /// The words of a fracture counters block (`fracture.slang` `C_*`).
    const COUNTER_WORDS: u64 = fracture::COUNTER_WORDS as u64;

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
                pack: None,
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
                pack: None,
            }
        }

        /// A scene pack as the lab's stress modes import it (the stress profiles' import
        /// options).
        pub(crate) fn from_pack(bytes: &[u8], key: &str, placement: [f64; 3]) -> Self {
            let mut node_count = 0;
            let mut pack = None;
            let scene = serde_json::from_slice::<serde_json::Value>(bytes)
                .map_err(|e| format!("scene pack: {e}"))
                .and_then(|json| {
                    node_count = json["scenario"]["nodes"].as_array().map_or(0, |n| n.len());
                    let scene = stress_ref::scene_pack::import_value_with(
                        &json,
                        key,
                        "structure",
                        key,
                        Profile::Threshold.import_options(),
                    );
                    pack = Some(Arc::new(PackSource {
                        json,
                        key: key.to_string(),
                        blast: std::sync::OnceLock::new(),
                    }));
                    scene
                })
                .map(|(scene, _notes)| scene);
            // The importer makes one chunk per pack node, in order; the ground anchors it
            // adds follow them.
            Self {
                scene: Arc::new(scene),
                placement,
                nodes: (0..node_count as u32).map(Some).collect(),
                pack,
            }
        }

        /// The stress scene as `profile` imports it: a pack's under `blast_gpu` keeps its
        /// Blast authoring (imported once, when first asked); every other import is the
        /// threshold's.
        pub fn scene_for(&self, profile: Profile) -> anyhow::Result<&Scene> {
            match (&self.pack, profile.import_options().blast_authoring) {
                (Some(p), true) => p
                    .blast
                    .get_or_init(|| {
                        stress_ref::scene_pack::import_value_with(
                            &p.json,
                            &p.key,
                            "structure",
                            &p.key,
                            profile.import_options(),
                        )
                        .map(|(scene, _notes)| scene)
                    })
                    .as_ref()
                    .map_err(|e| anyhow::anyhow!("the asset's {} import failed: {e}", profile.name())),
                _ => self.scene(),
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
            self.stress_scene(StrengthModel::Threshold, strength_scale, dt)
        }

        /// The stress scene as `model` runs it (the lab's `stress_load`): its profile's
        /// import, bond strengths scaled, `dt` as its frame, the profile applied.
        pub fn stress_scene(
            &self,
            model: StrengthModel,
            strength_scale: f32,
            dt: f32,
        ) -> anyhow::Result<Scene> {
            let profile = profile_of(model)?;
            let mut scene = self.scene_for(profile)?.clone();
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
            profile.apply_scene(&mut scene);
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

    /// One placed threshold destructible: where its records sit in the world's shared
    /// fracture ([`Shared`]) and what the host knows of it. Its tables, chunk owners and
    /// bond states are kept in its own (local) chunk and bond numbering; the shared
    /// tables hold them at `base`.
    pub(crate) struct ThresholdSlot {
        pub destructible: u32,
        /// Its first chunk, bond and part in the shared tables.
        base: fracture::Base,
        tables: FractureTables,
        /// Its tables' fixed-point mass-sum scales (the shared fracture uses the smallest).
        scales: [f32; 3],
        /// Bodies it was placed with (its clusters) plus the spare slots it added to the
        /// world's pool: what it returns when destroyed.
        clusters: u32,
        spares_added: u32,
        /// Per structure: its stress frame in the asset frame (rotation, translation).
        structures: Vec<(DQuat, DVec3)>,
        /// Per chunk: its shape slots.
        chunk_shapes: Vec<Vec<u32>>,
        /// Per chunk its body slot, per bond whether it holds, as last read back.
        chunk_body: Vec<u32>,
        intact: Vec<bool>,
        pub nodes: Vec<Option<u32>>,
        /// Blast loading's state (`DestructibleDef::blast_loading`).
        blast: Option<blast::BlastSlot>,
        /// `DestructibleDef::support_loss`: what the pass reads.
        support: Option<SupportLoss>,
        /// Chunk crushing (`DestructibleDef::crush`).
        crush: Option<CrushSlot>,
        /// Its own ingest and fracture (crushing, blast loading), else `None`: it is in the
        /// world's shared fracture at `base`.
        own: Option<Own>,
        /// Kept while its fracture is laid out anew at each placement (a shared fracture
        /// whose model's records cannot be appended, [`Shared::rebuild`]): its clusters'
        /// body frames, its colliders' contact sides (its own chunk numbering) and Blast's
        /// graph of it (`blast_gpu`).
        cluster_frames: Vec<(u32, fracture::FrameRecord)>,
        sides: Vec<(u32, ColliderInfo)>,
        blast_inputs: Option<BlastInputs>,
    }

    impl ThresholdSlot {
        /// Its own fracture (a crushing or blast-loaded destructible's).
        fn own(&self) -> &Own {
            self.own
                .as_ref()
                .expect("a destructible with its own fracture")
        }

        fn own_mut(&mut self) -> &mut Own {
            self.own
                .as_mut()
                .expect("a destructible with its own fracture")
        }
    }

    /// A threshold destructible with its own contact ingest and GPU fracture. Crushing's
    /// GPU state (its material laws, fragment pool, crush events) and blast loading's are
    /// per fracture (`fracture::CrushTables`), so those destructibles do not join the
    /// world's shared fracture; each costs its own ingest and fracture kernels per step.
    struct Own {
        ingest: EngineIngest,
        fracture: GpuFracture,
        /// Reserved body slots no piece has taken yet.
        spare: Vec<u32>,
        /// Every body slot it owns (pieces, spares, crush fragments).
        owned: Vec<u32>,
        /// The scene's stiffness scale (strike prediction's plane-strain modulus).
        stiffness_scale: f64,
        /// The ingest's default contact side for a collider (static geometry).
        static_info: ColliderInfo,
        /// Colliders the ingest now sees as projectiles.
        projectile_colliders: Vec<u32>,
        /// Its model's set-up.
        setup: ProfileSetup,
    }

    /// A frame's body frame records: an own fracture's every slot, or the shared
    /// fracture's changed bodies.
    #[derive(Clone, Copy)]
    enum Frames<'a> {
        Dense(&'a [[f32; 4]]),
        Sparse(&'a std::collections::HashMap<u32, Frame>),
    }

    impl Frames<'_> {
        fn get(&self, b: u32) -> Option<Frame> {
            match self {
                Frames::Dense(f) => Some([*f.get(2 * b as usize)?, *f.get(2 * b as usize + 1)?]),
                Frames::Sparse(m) => m.get(&b).copied(),
            }
        }
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
        /// With crushing: some chunk's flow stress alone is at its crush surface, so its
        /// damage grows every step: the world steps even when nothing is awake.
        overstressed: bool,
        /// Every bond the pass has broken, in order (a diagnostic record:
        /// `World::destructible_support_loss_breaks`).
        broke: Vec<u32>,
    }

    /// Whether the flow's loads alone (`SupportLoss::load`) put a crushable, uncrushed
    /// chunk at (or within 0.1% of) its crush surface: the virial `fracture.slang`
    /// `flow_virial` builds, in the structure frame (the invariants do not depend on it).
    /// Conservative (the GPU decides the damage): it only keeps the world stepping.
    fn flow_overstressed(t: &ThresholdSlot, sl: &SupportLoss, crushed: &[bool]) -> bool {
        let Some(ct) = t.tables.crush.as_ref().filter(|c| c.flow) else {
            return false;
        };
        (0..t.tables.chunks.len()).any(|c| {
            let [m0, m1] = ct.chunks[c];
            if crushed.get(c).copied().unwrap_or(false) || m1[2].to_bits() & 1 == 0 {
                return false;
            }
            let r0 = t.tables.chunks[c][0];
            let centre = DVec3::new(r0[0] as f64, r0[1] as f64, r0[2] as f64);
            let mut virial = [0.0f64; 6];
            for &b in &sl.bonds_of[c] {
                let l = sl.load[b as usize] as f64;
                if !t.intact[b as usize] || sl.sent[b as usize] || !(l > 0.0) {
                    continue;
                }
                let g = ct.bonds[b as usize];
                let n = ct.normals[b as usize];
                let r = DVec3::new(g[0] as f64, g[1] as f64, g[2] as f64) - centre;
                let mut n = DVec3::new(n[0] as f64, n[1] as f64, n[2] as f64);
                if n.dot(r) < 0.0 {
                    n = -n;
                }
                let s = stress_ref::crush::sym_outer(r.to_array(), (-l * n).to_array());
                for (v, x) in virial.iter_mut().zip(s) {
                    *v += x;
                }
            }
            let (p, q) = stress_ref::crush::invariants(virial, m1[1] as f64);
            let m = stress_ref::crush::CrushParams {
                cap_pressure: m0[0] as f64,
                cohesion: m0[1] as f64,
                friction_slope: m0[2] as f64,
                crush_energy: m0[3] as f64,
                crush_viscosity: m1[0] as f64,
            };
            m.excess(1.001 * p, 1.001 * q).is_some()
        })
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

    /// A crush particle's spot in its chunk's box, each coordinate in [-1, 1]
    /// (`fracture.slang` `crush_particle_spot`, the same hash).
    fn crush_particle_spot(k: u32, i: u32) -> Vec3 {
        let mut h = k.wrapping_mul(0x9E37_79B1) ^ i.wrapping_mul(0x85EB_CA77);
        h ^= h >> 15;
        h = h.wrapping_mul(0x2C1B_3C6D);
        h ^= h >> 12;
        h = h.wrapping_mul(0x297A_2D39);
        h ^= h >> 15;
        let h2 = h.wrapping_mul(0x9E37_79B1).wrapping_add(0x7F4A_7C15);
        Vec3::new(
            (h & 0x3FF) as f32,
            ((h >> 10) & 0x3FF) as f32,
            (h2 & 0x3FF) as f32,
        ) / 511.5
            - Vec3::ONE
    }

    /// A threshold destructible's crushing: the reserved fragment bodies (body, shape,
    /// collider; the GPU's `CrushTables::pool` in order) and how many a crush takes.
    struct CrushSlot {
        pool: Vec<(u32, u32, u32)>,
        fragments: u32,
        /// `CrushMass::Particles`: each chunk's particle count (0 where it cannot crush).
        particles: Option<Vec<u32>>,
        /// Its particles' continuous collision while fast.
        particle_ccd: bool,
        material: u32,
        shape_def: crate::ShapeDef,
    }

    /// First-fit ranges of a shared table (chunks, bonds or parts): `free` sorted and
    /// coalesced. More than the room is an error naming the table.
    struct Ranges {
        what: &'static str,
        room: u32,
        free: Vec<(u32, u32)>,
    }

    impl Ranges {
        fn new(what: &'static str, room: u32) -> Self {
            Self {
                what,
                room,
                free: vec![(0, room)],
            }
        }

        fn take(&mut self, n: u32) -> anyhow::Result<u32> {
            if n == 0 {
                return Ok(0);
            }
            let Some(i) = self.free.iter().position(|r| r.1 >= n) else {
                let left: u32 = self.free.iter().map(|r| r.1).sum();
                anyhow::bail!(
                    "threshold destructibles: {n} more {} do not fit (room for {} in this world, {left} free{}); raise WorldCapacity::max_bodies",
                    self.what,
                    self.room,
                    if left >= n { ", but not in one range" } else { "" }
                );
            };
            let at = self.free[i].0;
            self.free[i].0 += n;
            self.free[i].1 -= n;
            if self.free[i].1 == 0 {
                self.free.remove(i);
            }
            Ok(at)
        }

        fn give(&mut self, at: u32, n: u32) {
            if n == 0 {
                return;
            }
            let i = self.free.partition_point(|r| r.0 < at);
            self.free.insert(i, (at, n));
            // Coalesce with the neighbours.
            if i + 1 < self.free.len() && self.free[i].0 + self.free[i].1 == self.free[i + 1].0 {
                self.free[i].1 += self.free[i + 1].1;
                self.free.remove(i + 1);
            }
            if i > 0 && self.free[i - 1].0 + self.free[i - 1].1 == self.free[i].0 {
                self.free[i - 1].1 += self.free[i].1;
                self.free.remove(i);
            }
        }
    }

    impl CrushSlot {
        /// Pool entries chunk `c`'s crush takes.
        fn pieces(&self, c: usize) -> u32 {
            match &self.particles {
                Some(counts) => counts[c],
                None => self.fragments,
            }
        }
    }

    /// Every threshold destructible of a world in one contact ingest and one GPU fracture:
    /// each step reads the engine's contacts once and runs each fracture kernel once, over
    /// the chunks and bonds in use, however many destructibles there are. Tables are laid
    /// out for the world's capacity (a chunk per body slot: every piece a split can make
    /// has one) and destructibles are placed into free ranges of them.
    pub(crate) struct Shared {
        ingest: EngineIngest,
        fracture: GpuFracture,
        chunks: Ranges,
        bonds: Ranges,
        parts: Ranges,
        /// The ingest's default contact side for a collider (static geometry: the ground).
        static_info: ColliderInfo,
        /// The scenes' stiffness scale (strike prediction's plane-strain modulus).
        stiffness_scale: f64,
        /// The frame settings every destructible's scene must share: |g| dt and dt.
        band: f32,
        dt: f32,
        /// Colliders the ingest now sees as projectiles.
        projectile_colliders: Vec<u32>,
        /// The stress model every destructible in it runs, with its options.
        setup: ProfileSetup,
        /// Laid out anew at each placement ([`ProfileSetup::appendable`] false: the load
        /// path's and the analytic strike's records, the contact fracture's, Blast's graph),
        /// which only a fracture that has not stepped yet can be.
        rebuild: bool,
        /// It has run a step.
        stepped: bool,
    }

    /// The world's threshold destructibles.
    #[derive(Default)]
    pub(crate) struct Fracture {
        gpu: Option<Gpu>,
        pub slots: Vec<Option<ThresholdSlot>>,
        shared: Option<Shared>,
        /// Reserved body slots no piece has taken yet: one pool for every destructible
        /// (a split takes the lowest free slot, whoever reserved it).
        spare: std::collections::BTreeSet<u32>,
        /// Live slots by their first chunk and first bond (start, end, slot), sorted: an
        /// event's destructible.
        chunk_ranges: Vec<(u32, u32, usize)>,
        bond_ranges: Vec<(u32, u32, usize)>,
        /// Registered projectiles: body slot and generation, material, launch mark (the
        /// analytic strike's: a projectile does not strike its own debris again).
        projectiles: Vec<(u32, u32, Projectile, u32)>,
        /// Launch marks handed out.
        launches: u32,
        projectiles_dirty: bool,
        /// The plan hints the lab fixes (`set_plan_hints`): the contact capacity, every
        /// colour its own dispatch.
        pub fixed_plan: Option<(u32, u32)>,
        /// This frame's crushed bodies and their origins as the split left them (before
        /// parking): a crushed chunk that kept its parent's body leaves no record of where
        /// the parent was, and the split's other pieces start there (`World::frame`).
        pub crushed_origins: Vec<(u32, Transform)>,
        /// Each asset's instance-independent build, shared by its instances.
        templates: Vec<Arc<ThresholdTemplate>>,
    }

    /// What `create_threshold` derives from an asset alone, the same for every instance
    /// (the city's houses): the stress scene and world, the cluster bodies cooked, each
    /// cluster shape's geometry in the asset frame. Built for the first instance and
    /// reused, so placing a copy only places it.
    pub(crate) struct ThresholdTemplate {
        /// The asset it was built from (assets are replaced, never mutated: identity).
        asset: Arc<DestructibleAssetDesc>,
        model: StrengthModel,
        strength_scale: u32,
        dt: u32,
        scene: Scene,
        world: stress_ref::world::World,
        /// The cluster bodies.
        bodies: Vec<stress_ref::engine::EngineBody>,
        /// Each structure's stress frame in the asset frame.
        structures: Vec<(DQuat, DVec3)>,
        cooked: Vec<stress_rubble::Cooked>,
        geometry: Vec<Vec<Geometry>>,
    }

    impl Fracture {
        /// The buffers a step's fracture writes last (its atomics and work), for a GPU span
        /// that must end after them.
        pub(crate) fn state_buffers(&self) -> Vec<&wgpu::Buffer> {
            let mut out = Vec::new();
            if let Some(sh) = &self.shared {
                out.push(&sh.fracture.atom);
                out.push(&sh.fracture.work);
            }
            for o in self.slots.iter().flatten().filter_map(|t| t.own.as_ref()) {
                out.push(&o.fracture.atom);
                out.push(&o.fracture.work);
            }
            out
        }

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

    /// One step's fracture readback (the shared fracture's).
    #[derive(Clone, Copy, Debug)]
    pub(crate) enum FractureRead {
        /// The shared fracture's counters and this step's events: bonds broken, chunks
        /// moved (chunk, body), body frames.
        Counters,
        Broken,
        Moved,
        Frames,
        /// A destructible with its own fracture (crushing, blast loading): its counters,
        /// chunk owners, bond states, body frames and this frame's crush events
        /// (`GpuFracture::crush_events_range`).
        SlotCounters(u32),
        ChunkBodies(u32),
        Intact(u32),
        SlotFrames(u32),
        Crush(u32),
    }

    /// Each step's event lists' room (`fracture::Events`): a step that breaks, moves or
    /// re-frames more fails with an error naming the limit.
    pub(crate) const EVENT_ROOM: fracture::Events = fracture::Events {
        broken: 16_384,
        moved: 16_384,
        frames: 4_096,
    };

    /// A body's frame record, as the GPU wrote it.
    type Frame = [[f32; 4]; 2];

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
                self.projectile_shot(b as u32, p, 1.0, 0)?;
            }
            let f = &mut self.fracture;
            f.projectiles.retain(|p| p.0 != b as u32);
            if let Some(p) = projectile {
                anyhow::ensure!(
                    f.projectiles.len() < fracture::MAX_SHOTS,
                    "at most {} projectiles (strike prediction's shot table)",
                    fracture::MAX_SHOTS
                );
                // Each registration is a launch (the lab's `fired`): marks from 1.
                f.launches = f.launches.wrapping_add(1).max(1);
                f.projectiles.push((b as u32, body.generation(), p, f.launches));
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
            launch: u32,
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
                // The analytic strike's (`impact_local`) mark: its own debris is not struck
                // again (the other models write no marks).
                launch,
            };
            let info = ColliderInfo {
                a: [NONE, b, KIND_BODY, 0],
                b: [mass, p.youngs_modulus, face, f32::from_bits(launch)],
            };
            Ok((shot, shape.collider, info))
        }

        /// Every threshold destructible's shot table and the projectiles' contact sides:
        /// the shared fracture's and each own fracture's.
        pub(crate) fn sync_projectiles(&mut self) -> anyhow::Result<()> {
            self.fracture.projectiles_dirty = false;
            let live: Vec<(u32, Projectile, u32)> = self
                .fracture
                .projectiles
                .iter()
                .filter(|p| {
                    self.bodies
                        .get(p.0 as usize)
                        .is_some_and(|s| s.alive && s.generation == p.1)
                })
                .map(|p| (p.0, p.2, p.3))
                .collect();
            let gpu = self
                .fracture
                .gpu
                .clone()
                .expect("a threshold destructible's GPU");
            // The fractures' stiffness scales and whether they predict strikes: the shared
            // one's (`None` slot), each own's.
            let mut targets: Vec<(Option<usize>, f64, bool)> = Vec::new();
            if let Some(sh) = &self.fracture.shared {
                targets.push((None, sh.stiffness_scale, sh.setup.predict_strikes));
            }
            for (k, t) in self.fracture.slots.iter().enumerate() {
                if let Some(o) = t.as_ref().and_then(|t| t.own.as_ref()) {
                    targets.push((Some(k), o.stiffness_scale, o.setup.predict_strikes));
                }
            }
            let stride = std::mem::size_of::<ColliderInfo>() as u64;
            for (k, scale, predict) in targets {
                let mut shots = Vec::with_capacity(live.len());
                let mut sides = Vec::with_capacity(live.len());
                for (b, p, launch) in &live {
                    let (shot, collider, info) = self.projectile_shot(*b, p, scale, *launch)?;
                    shots.push(shot);
                    sides.push((collider, info));
                }
                let (ingest, fracture, static_info, colliders) = match k {
                    None => {
                        let sh = self.fracture.shared.as_mut().expect("shared fracture");
                        (
                            &sh.ingest,
                            &mut sh.fracture,
                            sh.static_info,
                            &mut sh.projectile_colliders,
                        )
                    }
                    Some(k) => {
                        let o = self.fracture.slots[k]
                            .as_mut()
                            .expect("live slot")
                            .own_mut();
                        (
                            &o.ingest,
                            &mut o.fracture,
                            o.static_info,
                            &mut o.projectile_colliders,
                        )
                    }
                };
                // A collider that stopped being a projectile (cleared, or its body
                // destroyed and the slot reused) is static geometry again.
                for &c in colliders.iter() {
                    if !sides.iter().any(|s| s.0 == c) {
                        gpu.queue.write_buffer(
                            &ingest.collider_info,
                            stride * c as u64,
                            bytemuck::bytes_of(&static_info),
                        );
                    }
                }
                for (c, info) in &sides {
                    gpu.queue.write_buffer(
                        &ingest.collider_info,
                        stride * *c as u64,
                        bytemuck::bytes_of(info),
                    );
                }
                *colliders = sides.iter().map(|s| s.0).collect();
                // Without strike prediction the projectiles still load what they touch
                // (their contact sides), but no strike is predicted.
                fracture
                    .set_shots(&gpu, if predict { &shots } else { &[] })
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            }
            Ok(())
        }

        /// Place a threshold destructible: one body per bonded cluster (the lab's
        /// `ThresholdWorld::new`), a reserved slot for every other piece a split can make,
        /// the fracture tables and the contact ingest.
        /// `asset`'s template for `def` ([`ThresholdTemplate`]): the stored one, or built
        /// (and stored, unless `def` crushes: its materials change the scene).
        fn threshold_template(
            &mut self,
            def: &DestructibleDef,
            asset: &Arc<DestructibleAssetDesc>,
            source: &StressSource,
        ) -> anyhow::Result<Arc<ThresholdTemplate>> {
            let dt = self.sim.params().dt;
            let (scale_bits, dt_bits) = (def.strength_scale.to_bits(), dt.to_bits());
            if def.crush.is_none() {
                if let Some(t) = self.fracture.templates.iter().find(|t| {
                    Arc::ptr_eq(&t.asset, asset)
                        && t.model == def.model
                        && t.strength_scale == scale_bits
                        && t.dt == dt_bits
                }) {
                    return Ok(t.clone());
                }
            }
            let mut scene = source.stress_scene(def.model, def.strength_scale, dt)?;
            if let Some(c) = &def.crush {
                apply_crush_materials(&mut scene, c)?;
            }
            let world = stress_ref::world::World::new(&scene);
            let m = &world.solver;
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
            let mut cooker = stress_rubble::Cooker::default();
            let mut cooked = Vec::with_capacity(bodies.len());
            let mut geometry = Vec::with_capacity(bodies.len());
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
                let c = cooker.cook(b, &mut self.hulls);
                let BodyGeometry::Boxes(shapes) = &b.geometry else {
                    anyhow::bail!("cluster {id} is not made of chunk shapes")
                };
                let mut g = Vec::with_capacity(shapes.len());
                for (s, col) in shapes.iter().zip(&c.colliders) {
                    let local = Transform::new(
                        f3(st + sq * dv(s.local.position)),
                        f4((sq * dq(s.local.rotation)).normalize()),
                    );
                    g.push(match &s.hull {
                        Some(points)
                            if col.shape == engine_schema::ShapeType::ConvexHull as u32 =>
                        {
                            let pts: Vec<Vec3> = points.iter().map(|p| dv(*p).as_vec3()).collect();
                            let (hull, shift) = ConvexHull::from_points(&pts)
                                .map_err(|e| anyhow::anyhow!("chunk {:?} hull: {e:?}", s.chunk))?
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
                    });
                }
                cooked.push(c);
                geometry.push(g);
            }
            let template = Arc::new(ThresholdTemplate {
                asset: asset.clone(),
                model: def.model,
                strength_scale: scale_bits,
                dt: dt_bits,
                scene,
                world,
                bodies,
                structures,
                cooked,
                geometry,
            });
            if def.crush.is_none() {
                self.fracture.templates.push(template.clone());
            }
            Ok(template)
        }

        pub(crate) fn create_threshold(
            &mut self,
            def: &DestructibleDef,
            asset: Arc<DestructibleAssetDesc>,
            index: usize,
        ) -> anyhow::Result<()> {
            let source = asset.stress.as_ref().ok_or_else(|| anyhow::anyhow!("StrengthModel::{:?} needs an asset with the stress oracle's input (from_glb, cuboid_house or from_pack)", def.model))?;
            // The ingest reads rubble's canonical contact rows, as the lab's engine keeps them.
            anyhow::ensure!(
                self.contact_storage == crate::ContactStorage::Canonical,
                "StrengthModel::{:?} reads rubble's canonical contact rows: create the world with ContactStorage::Canonical (it uses {:?})",
                def.model,
                self.contact_storage
            );
            let setup = setup_of(def)?;
            let template = self.threshold_template(def, &asset, source)?;
            let ThresholdTemplate {
                scene,
                world,
                bodies,
                structures,
                cooked: cooked_bodies,
                geometry: body_geometry,
                ..
            } = &*template;
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
                    // The fracture re-parents colliders, re-frames and retires bodies on
                    // the GPU between steps, with no host write: the awake-indexed
                    // broadphase's resting table and moved-only bounds would not see it
                    // (city: a struck house's static remainder is re-framed, its bounds
                    // rebuilt from the new frame differ in the last bits, and a pair was
                    // missed). The engine watches for re-framed bodies at rest and rebuilds
                    // both on the steps that have one (`StepModes::gpu_reframes`).
                    // RUBBLE_FRACTURE_FULL_BROADPHASE=1 (A/B measurement): the full pass
                    // on every step instead, as before.
                    self.sim.set_gpu_reframes(true);
                    if std::env::var("RUBBLE_FRACTURE_FULL_BROADPHASE").is_ok() {
                        self.sim.set_awake_broadphase(false);
                    }
                    g
                }
            };
            let instance = def.transform;
            let (ip, iq) = (
                DVec3::from(instance.p().as_dvec3()),
                instance.q().as_dquat().normalize(),
            );
            let structures = structures.clone();
            // Global chunk numbering, structure by structure.
            let mut offset = Vec::with_capacity(m.structures.len());
            let mut chunk_of = Vec::with_capacity(chunk_count);
            for (s, st) in m.structures.iter().enumerate() {
                offset.push(chunk_of.len() as u32);
                chunk_of.extend((0..st.chunks.len()).map(|c| (s, c)));
            }
            let global = |s: usize, c: usize| offset[s] + c as u32;
            // The clusters' bodies, cooked as the lab cooks them (once an asset: the template).
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
            for ((b, cooked), geometries) in bodies.iter().zip(cooked_bodies).zip(body_geometry) {
                let BodyKey::Cluster(id) = b.key else {
                    continue;
                };
                let cl = m
                    .clusters
                    .iter()
                    .find(|c| c.id == id)
                    .expect("a cluster of the world");
                let (sq, st) = structures[cl.structure];
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
                for (((s, col), part), geometry) in shapes
                    .iter()
                    .zip(&cooked.colliders)
                    .zip(&cooked.parts)
                    .zip(geometries)
                {
                    let (sc, cc) = s
                        .chunk
                        .ok_or_else(|| anyhow::anyhow!("cluster shape without a chunk"))?;
                    let k = global(sc, cc) as usize;
                    let geometry = geometry.clone();
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
            // With a world piece pool (`WorldDef::fracture_pieces`) the pool serves them.
            // Crushing and blast loading keep their own fracture (`Own`) and its spares.
            let own_fracture = def.crush.is_some() || def.blast_loading;
            let spare_count = if self.fracture_pieces.is_some() && !own_fracture {
                0
            } else {
                chunk_count.saturating_sub(m.clusters.len())
            };
            let spare_defs = vec![
                BodyDef {
                    user_data: def.user_data,
                    ..BodyDef::default()
                };
                spare_count
            ];
            let spare_ids = self.create_bodies(&spare_defs)?;
            let spare: Vec<u32> = spare_ids.iter().map(|id| id.index()).collect();
            self.park_spares(&spare);
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
            // Per GPU bond: its structure and bond (blast loading's mirror).
            let mut bond_source = Vec::new();
            let mut crush_bonds: Vec<[f32; 4]> = Vec::new();
            let mut crush_normals: Vec<[f32; 4]> = Vec::new();
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
                    bond_source.push((cl.structure, bi));
                    let source = &scene.bodies[cl.structure].bonds[bond.source];
                    compressive.push(
                        (scene.material(&source.material).compressive_strength * bond.geometry.area)
                            as f32,
                    );
                    let g = bond.geometry.centroid;
                    crush_bonds.push([
                        g.x as f32,
                        g.y as f32,
                        g.z as f32,
                        bond.geometry.area as f32,
                    ]);
                    let nn = bond.geometry.normal;
                    crush_normals.push([nn.x as f32, nn.y as f32, nn.z as f32, 0.0]);
                    t.bonds.push([
                        global(cl.structure, bond.geometry.a),
                        global(cl.structure, bond.geometry.b),
                        (capacity as f32).to_bits(),
                        (bond.geometry.area as f32).to_bits(),
                    ]);
                }
            }
            // The model's records (the load path's and analytic strike's tables, its warm
            // start, the contact fracture's sweeps, Blast's graph): stress-rubble's
            // `apply_profile`, as the lab's fracture world makes them.
            let blast_inputs = stress_rubble::threshold::apply_profile(
                &mut t,
                &setup,
                scene,
                m,
                &chunk_of,
                &bond_source,
                capacity.max_contacts,
            )
            .map_err(|e| anyhow::anyhow!("{}: {e}", def.model.name()))?;
            // Chunk crushing: each chunk's material law, the bonds' centroids, the fragment
            // pool (reserved bodies, one box collider each, parked as the spares are).
            let mut crush_slot = None;
            if let Some(c) = &def.crush {
                // Particles: about volume / size³ a chunk, within its bounds.
                let particle_count = |volume: f64| -> Result<u32, String> {
                    match c.mass {
                        CrushMass::Particles {
                            size,
                            max_per_chunk,
                            ..
                        } => {
                            if !(size > 0.0 && (1..=4096).contains(&max_per_chunk)) {
                                return Err(format!("CrushMass::Particles: size {size} m must be positive and max_per_chunk {max_per_chunk} from 1 to 4096"));
                            }
                            let n = (volume / (size as f64).powi(3)).round();
                            Ok((n as u32).clamp(1, max_per_chunk))
                        }
                        _ => Ok(0),
                    }
                };
                let (rows, counts, crushable, lacking) =
                    stress_rubble::threshold::crush_rows(&t, scene, m, &chunk_of, particle_count)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                anyhow::ensure!(
                    crushable > 0,
                    "DestructibleDef::crush: no chunk can crush: its materials {lacking:?} have no crush parameters (give them in Crush::materials, e.g. from stress_ref::crush::presets, or import a ScenePack v3 with materials[].crush)"
                );
                let (conserve, fragments, budget, particles) = match c.mass {
                    CrushMass::Conserve { fragments, budget } => {
                        anyhow::ensure!(
                            fragments.is_power_of_two() && fragments <= 64,
                            "CrushMass::Conserve: {fragments} fragments per chunk; a power of two up to 64"
                        );
                        (true, fragments, budget, false)
                    }
                    CrushMass::Particles { budget, .. } => (true, 0, budget, true),
                    CrushMass::Remove => (false, 0, 0, false),
                };
                let particle_ccd = matches!(
                    c.mass,
                    CrushMass::Particles {
                        continuous: true,
                        ..
                    }
                );
                anyhow::ensure!(
                    budget < stress_gpu::fracture::CRUSH_POOL_SPAN,
                    "DestructibleDef::crush: a budget of {budget} crush bodies; under {}",
                    stress_gpu::fracture::CRUSH_POOL_SPAN
                );
                // Particles share a negative group: they never collide with one another.
                let pool_def = if particles {
                    crate::ShapeDef {
                        filter: crate::Filter {
                            group: crate::CRUSH_PARTICLE_GROUP,
                            ..shape_def.filter
                        },
                        ..shape_def.clone()
                    }
                } else {
                    shape_def.clone()
                };
                let ids = self.create_bodies(&vec![
                    BodyDef {
                        user_data: def.user_data,
                        ..BodyDef::default()
                    };
                    budget as usize
                ])?;
                let mut pool = Vec::with_capacity(ids.len());
                for id in ids {
                    let i = id.index();
                    let mut parked = BodyState::static_plane_support();
                    parked.position = crate::graveyard(i);
                    parked.initial_pos = parked.position;
                    if particle_ccd {
                        // The particles' continuous collision: the passes are scheduled
                        // from placement (a parked static body is never a CCD body).
                        parked.flags |= engine_schema::BODY_CCD_FAST;
                    }
                    self.sim.write_bodies(i, &[parked]);
                    let geometry = if particles {
                        Geometry::Sphere {
                            center: [0.0; 3],
                            radius: 0.05,
                        }
                    } else {
                        Geometry::Box {
                            half: [0.05; 3],
                            transform: Transform::IDENTITY,
                        }
                    };
                    let mut col =
                        self.collider(i, SolverFrame::IDENTITY, &geometry, 0.0, material, None)?;
                    pool_def.filter.apply(&mut col);
                    col.material |= crate::contacts::event_bits(&pool_def);
                    let (shape, collider) =
                        self.add_shape_record(i as usize, &pool_def, geometry, col)?;
                    let slot = &mut self.bodies[i as usize];
                    slot.owner = Owner::Piece;
                    slot.alive = false;
                    pool.push((i, shape, collider));
                }
                let max_events = if c.max_per_step == 0 {
                    256
                } else {
                    c.max_per_step
                };
                // With support loss, its flow's bond loads enter the stress estimate
                // (`fracture.slang` `crush_loads_flow`: crushes with cause Load).
                let flow = def.support_loss;
                t.crush = Some(CrushTables {
                    chunks: rows,
                    bonds: crush_bonds,
                    normals: if flow { crush_normals } else { Vec::new() },
                    flow,
                    conserve,
                    fragments,
                    particles,
                    particle_ccd,
                    pool: pool.iter().map(|p| [p.0, p.2]).collect(),
                    max_events,
                    park_y: crate::graveyard(0)[1],
                    park_step: crate::graveyard(1)[0],
                });
                crush_slot = Some(CrushSlot {
                    pool,
                    fragments,
                    particles: particles.then_some(counts),
                    particle_ccd,
                    material,
                    shape_def: pool_def,
                });
            }
            let owned: Vec<u32> = owned
                .into_iter()
                .chain(crush_slot.iter().flat_map(|c| c.pool.iter().map(|p| p.0)))
                .collect();
            // The frames and free marks cover every slot the fracture may see.
            let body_count = owned.iter().copied().max().map_or(0, |b| b + 1) as usize;
            t.frames
                .resize(body_count, [[0.0; 4], [0.0, 0.0, 0.0, 1.0]]);
            t.free.resize(body_count, 0);
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
            let mut root_of_body = std::collections::HashMap::new();
            for (k, &b) in t.chunk_body.iter().enumerate() {
                root_of_body.entry(b).or_insert(k);
            }
            let mut sides: Vec<(u32, ColliderInfo)> = Vec::new();
            for k in 0..chunk_count {
                let body = t.chunk_body[k];
                let root = root_of_body[&body];
                let mass = self.bodies[body as usize]
                    .mass
                    .map_or(f32::INFINITY, |m| m.mass);
                let rr = t.chunks[root][3];
                for &col in &chunk_cols[k] {
                    // The chunk index is the shared tables' (set once placed below).
                    sides.push((
                        col,
                        ColliderInfo {
                            a: [k as u32, body, KIND_BODY, 1],
                            b: [mass, rr[0], rr[3], 0.0],
                        },
                    ));
                }
            }
            let bond_count = t.bonds.len();
            let local_sides = sides.clone();
            let blast = def.blast_loading.then(|| {
                blast::BlastSlot::new(
                    &gpu.device,
                    m.clone(),
                    &scene,
                    bond_source.clone(),
                    chunk_of.clone(),
                )
            });
            let scales = t.scales();
            let (base, own) = if own_fracture {
                // The ingest merges two records a contact row in 64-thread groups, one
                // dispatch dimension (WebGPU's 65,535 groups).
                anyhow::ensure!(
                    capacity.max_contacts as u64 * 2 <= 65_535 * 64,
                    "a threshold destructible needs max_contacts <= {} (the contact ingest's dispatch), not {}",
                    65_535 * 32,
                    capacity.max_contacts
                );
                let mut table = vec![static_info; capacity.max_colliders as usize];
                for (col, info) in &sides {
                    table[*col as usize] = *info;
                }
                for c in crush_slot.iter().flat_map(|c| &c.pool) {
                    table[c.2 as usize] = ColliderInfo {
                        a: [NONE, c.0, KIND_BODY, 0],
                        b: [f32::INFINITY, ground_modulus as f32, 1.0, 0.0],
                    };
                }
                let ingest = EngineIngest::new(
                    &gpu,
                    capacity.max_contacts,
                    1,
                    capacity.max_bodies,
                    chunk_count as u32,
                    &table,
                );
                let mut fracture = GpuFracture::new(&gpu, &t, &buffers(&self.sim), &ingest)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                if let Some(b) = &blast_inputs {
                    let frame = b.frame(&gpu, scene.gravity).map_err(|e| anyhow::anyhow!("{e}"))?;
                    fracture.set_blast(&gpu, frame).map_err(|e| anyhow::anyhow!("{e}"))?;
                }
                let own = Own {
                    ingest,
                    fracture,
                    spare: spare.clone(),
                    owned: owned.clone(),
                    stiffness_scale: scene.sim.stiffness_scale,
                    static_info,
                    projectile_colliders: Vec::new(),
                    setup,
                };
                (fracture::Base::default(), Some(own))
            } else {
                let band = t.band;
                let dt = t.dt;
                let stiffness_scale = scene.sim.stiffness_scale;
                self.threshold_shared(&gpu, static_info, stiffness_scale, band, dt, setup)?;
                if self.fracture.shared.as_ref().is_some_and(|sh| sh.rebuild) {
                    // Laid out with the others once the slot is in (`rebuild_shared`).
                    self.fracture.spare.extend(spare.iter().copied());
                    (fracture::Base::default(), None)
                } else {
                let sh = self.fracture.shared.as_mut().expect("shared fracture");
                let (n, e, p) = (chunk_count as u32, bond_count as u32, t.parts.len() as u32);
                let chunk_at = sh.chunks.take(n)?;
                let bond_at = match sh.bonds.take(e) {
                    Ok(at) => at,
                    Err(err) => {
                        sh.chunks.give(chunk_at, n);
                        return Err(err);
                    }
                };
                let part_at = match sh.parts.take(p) {
                    Ok(at) => at,
                    Err(err) => {
                        sh.chunks.give(chunk_at, n);
                        sh.bonds.give(bond_at, e);
                        return Err(err);
                    }
                };
                let base = fracture::Base {
                    chunk: chunk_at,
                    bond: bond_at,
                    part: part_at,
                };
                // The records as the shared tables number them.
                let mut g = FractureTables {
                    chunks: t.chunks.clone(),
                    info: t.info.clone(),
                    bonds: t.bonds.clone(),
                    parts: t.parts.clone(),
                    chunk_body: t.chunk_body.clone(),
                    geometry: t.geometry.clone(),
                    ..Default::default()
                };
                for c in &mut g.chunks {
                    c[2][2] = f32::from_bits(c[2][2].to_bits() + base.part);
                }
                for b in &mut g.bonds {
                    b[0] += base.chunk;
                    b[1] += base.chunk;
                }
                let frames: Vec<(u32, fracture::FrameRecord)> = cluster_bodies
                    .iter()
                    .map(|c| (c.0, t.frames[c.0 as usize]))
                    .collect();
                let all_scales = self
                    .fracture
                    .slots
                    .iter()
                    .flatten()
                    .filter(|s| s.own.is_none())
                    .map(|s| s.scales)
                    .fold(scales, |a, b| {
                        [a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2])]
                    });
                let sh = self.fracture.shared.as_mut().expect("shared fracture");
                sh.fracture
                    .append(&gpu, base, &g, &frames, &spare, Some(all_scales))
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                let stride = std::mem::size_of::<ColliderInfo>() as u64;
                let sides: Vec<(u32, ColliderInfo)> = sides
                    .into_iter()
                    .map(|(col, mut info)| {
                        info.a[0] += base.chunk;
                        (col, info)
                    })
                    .collect();
                for (first, infos) in runs(sides) {
                    gpu.queue.write_buffer(
                        &sh.ingest.collider_info,
                        stride * first as u64,
                        bytemuck::cast_slice(&infos),
                    );
                }
                self.fracture.spare.extend(spare.iter().copied());
                (base, None)
                }
            };
            let rebuild = own.is_none()
                && self.fracture.shared.as_ref().is_some_and(|sh| sh.rebuild);
            let cluster_frames: Vec<(u32, fracture::FrameRecord)> = if rebuild {
                cluster_bodies
                    .iter()
                    .map(|c| (c.0, t.frames[c.0 as usize]))
                    .collect()
            } else {
                Vec::new()
            };
            if own.is_none() {
                // Its body frames and free marks went to the shared fracture with it (and
                // nothing reads them here again): they cover every body slot below its own,
                // so keeping them costs a map O(houses × body slots) host memory (about
                // 2 GB at 1,024 houses, past wasm's 4 GiB heap with the rest).
                t.frames = Vec::new();
                t.free = Vec::new();
            }
            let slot = ThresholdSlot {
                destructible: index as u32,
                base,
                scales,
                clusters: cluster_bodies.len() as u32,
                spares_added: spare.len() as u32,
                chunk_body: t.chunk_body.clone(),
                tables: t,
                structures,
                chunk_shapes,
                intact: vec![true; bond_count],
                nodes: source.nodes.clone(),
                blast,
                support: None,
                crush: crush_slot,
                own,
                cluster_frames,
                sides: if rebuild { local_sides } else { Vec::new() },
                blast_inputs: if rebuild { blast_inputs } else { None },
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
                    broke: Vec::new(),
                    compressive,
                    bonds_of,
                    gravity: g.map(|x| x as f32),
                    dynamic_factor: def.support_loss_dynamic_factor,
                    overstressed: false,
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
                if slot.own.as_ref().is_some_and(|o| o.fracture.crush_flow()) {
                    slot.own()
                        .fracture
                        .set_crush_flow(&gpu, &sl.load)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    sl.overstressed = flow_overstressed(&slot, &sl, &[]);
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
            let (base, n, e, shared) = (
                slot.base,
                slot.chunk_body.len() as u32,
                slot.intact.len() as u32,
                slot.own.is_none(),
            );
            self.fracture.slots[k] = Some(slot);
            // The shared fracture's events find their destructible by these.
            for (ranges, start, len) in [
                (&mut self.fracture.chunk_ranges, base.chunk, n),
                (&mut self.fracture.bond_ranges, base.bond, e),
            ] {
                if shared && !rebuild {
                    let at = ranges.partition_point(|r| r.0 < start);
                    ranges.insert(at, (start, start + len, k));
                }
            }
            // A plan from counters read a frame or more late would differ run to run: the
            // lab fixes it (the hash at the contact capacity, every colour its own
            // dispatch).
            let plan = (capacity.max_contacts, capacity.max_colors.min(32));
            self.fracture.fixed_plan = Some(plan);
            self.sim.set_plan_hints(plan.0, plan.1);
            let d = &mut self.destructibles[index];
            d.model = def.model;
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
            d.crushed = vec![false; chunk_count];
            if rebuild {
                self.rebuild_shared(&gpu)?;
            }
            self.sync_projectiles()?;
            // Hand this destructible's uploads to the GPU and wait for them: staging memory
            // is reclaimed only once its copies finish, so placing a map's worth without
            // waiting piles up staging buffers until the driver blocks a buffer create
            // (with the GPU busy elsewhere, it never recovers). Native only: in the browser
            // `writeBuffer` stages in the browser, which frees it, and a poll cannot wait
            // there (WebGPU polls itself), so placement never blocks a page's frame.
            let index = self.sim.queue().submit(std::iter::empty());
            #[cfg(not(target_arch = "wasm32"))]
            let _ = self.sim.device().poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: None,
            });
            #[cfg(target_arch = "wasm32")]
            let _ = index;
            Ok(())
        }

        /// The world's shared fracture, made on first use, or why a scene cannot join it
        /// (every threshold destructible in a world shares its ground's contact side, its
        /// stiffness scale and its frame).
        fn threshold_shared(
            &mut self,
            gpu: &Gpu,
            static_info: ColliderInfo,
            stiffness_scale: f64,
            band: f32,
            dt: f32,
            setup: ProfileSetup,
        ) -> anyhow::Result<()> {
            if let Some(sh) = &self.fracture.shared {
                anyhow::ensure!(
                    sh.setup == setup,
                    "stress destructibles in one world share one fracture, which runs one model: this one's {} (contact fracture sweeps {}, strike prediction {}) is not the world's {} ({}, {}); give it crushing or blast loading for its own fracture, or place it in another world",
                    setup.profile.name(),
                    setup.contact_sweeps,
                    setup.predict_strikes,
                    sh.setup.profile.name(),
                    sh.setup.contact_sweeps,
                    sh.setup.predict_strikes
                );
                anyhow::ensure!(
                    !(sh.rebuild && sh.stepped),
                    "a {} world lays out its fracture once, before its first step (the model's per-bond tables and graph are not appended to a running fracture): place every {} destructible before stepping",
                    setup.profile.name(),
                    setup.profile.name()
                );
                anyhow::ensure!(
                    sh.static_info.b == static_info.b
                        && sh.stiffness_scale == stiffness_scale
                        && sh.band == band
                        && sh.dt == dt,
                    "threshold destructibles in one world share one fracture: this asset's ground modulus {}, stiffness scale {stiffness_scale}, |g| dt {band} and dt {dt} differ from the world's ({}, {}, {}, {})",
                    static_info.b[1],
                    sh.static_info.b[1],
                    sh.stiffness_scale,
                    sh.band,
                    sh.dt
                );
                return Ok(());
            }
            // Resting contacts survive sleep (a woken cluster carries its weight from its
            // first step, as one that never slept): before the ingest binds the contacts.
            let rows = self
                .sleep_archive_rows
                .unwrap_or(self.sim.capacity().max_bodies.saturating_mul(16));
            self.sim.enable_sleep_archive(rows)?;
            let capacity = self.sim.capacity();
            // The ingest merges two records a contact row in 64-thread groups, one dispatch
            // dimension (WebGPU's 65,535 groups).
            anyhow::ensure!(
                capacity.max_contacts as u64 * 2 <= 65_535 * 64,
                "a threshold destructible needs max_contacts <= {} (the contact ingest's dispatch), not {}",
                65_535 * 32,
                capacity.max_contacts
            );
            // A chunk per collider at most, bonds at four per chunk (a brick's six faces,
            // shared), a part per collider.
            let room = fracture::Capacity {
                // A chunk has at least one collider; the split's scans hold 2^20.
                chunks: capacity.max_colliders.min(1 << 20),
                bonds: capacity.max_colliders.min(1 << 20).saturating_mul(4),
                parts: capacity.max_colliders,
                events: EVENT_ROOM,
            };
            let table = vec![static_info; capacity.max_colliders as usize];
            let ingest = EngineIngest::new(
                gpu,
                capacity.max_contacts,
                1,
                capacity.max_bodies,
                room.chunks,
                &table,
            );
            let rebuild = !setup.appendable();
            // Laid out anew at each placement: a placeholder until the first.
            let made = if rebuild {
                fracture::Capacity {
                    chunks: 1,
                    bonds: 1,
                    parts: 1,
                    events: EVENT_ROOM,
                }
            } else {
                room
            };
            let empty = FractureTables {
                max_bodies: capacity.max_bodies,
                pool_cap: stress_rubble::threshold::pool_cap(capacity.max_contacts),
                impact_peak: true,
                band,
                dt,
                strike_speed: stress_ref::world::tiered_impact_speed() as f32,
                frames: vec![[[0.0; 4], [0.0, 0.0, 0.0, 1.0]]; capacity.max_bodies as usize],
                free: vec![0; capacity.max_bodies as usize],
                ..Default::default()
            };
            let fracture =
                GpuFracture::with_capacity(gpu, &empty, made, &buffers(&self.sim), &ingest)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            self.fracture.shared = Some(Shared {
                ingest,
                fracture,
                chunks: Ranges::new("chunks", room.chunks),
                bonds: Ranges::new("bonds", room.bonds),
                parts: Ranges::new("chunk parts", room.parts),
                static_info,
                stiffness_scale,
                band,
                dt,
                projectile_colliders: Vec::new(),
                setup,
                rebuild,
                stepped: false,
            });
            if let Some(n) = self.fracture_pieces {
                // The world's piece pool: parked static slots any split may take.
                self.add_to_pool(gpu, n)?;
            }
            Ok(())
        }

        /// Lay the shared fracture out anew over every destructible in it (a model whose
        /// records cannot be appended: [`ProfileSetup::appendable`]): their tables one after
        /// another (each chunk's first part, each bond's chunks, each per-bond record's
        /// material and each chunk's adjacency at its offset), the load path's warm starts,
        /// Blast's graphs joined, the colliders' contact sides renumbered. Only before the
        /// fracture's first step (`threshold_shared` says so otherwise): nothing has broken,
        /// split or loaded yet, so nothing is lost.
        fn rebuild_shared(&mut self, gpu: &Gpu) -> anyhow::Result<()> {
            let capacity = self.sim.capacity();
            let sh = self.fracture.shared.as_mut().expect("shared fracture");
            anyhow::ensure!(sh.rebuild && !sh.stepped, "rebuild_shared: a fracture that steps");
            let mut chunks = Ranges::new("chunks", sh.chunks.room);
            let mut bonds = Ranges::new("bonds", sh.bonds.room);
            let mut parts = Ranges::new("chunk parts", sh.parts.room);
            let mut t = FractureTables {
                max_bodies: capacity.max_bodies,
                pool_cap: stress_rubble::threshold::pool_cap(capacity.max_contacts),
                band: sh.band,
                dt: sh.dt,
                strike_speed: stress_ref::world::tiered_impact_speed() as f32,
                frames: vec![[[0.0; 4], [0.0, 0.0, 0.0, 1.0]]; capacity.max_bodies as usize],
                free: vec![0; capacity.max_bodies as usize],
                ..Default::default()
            };
            let mut blast: Option<BlastInputs> = None;
            let mut sides: Vec<(u32, ColliderInfo)> = Vec::new();
            let (mut chunk_ranges, mut bond_ranges) = (Vec::new(), Vec::new());
            let mut first = true;
            for (k, slot) in self.fracture.slots.iter_mut().enumerate() {
                let Some(slot) = slot.as_mut().filter(|s| s.own.is_none()) else {
                    continue;
                };
                let l = &slot.tables;
                if first {
                    // Every member runs the same model (`threshold_shared`).
                    t.load_model = l.load_model;
                    t.strike_model = l.strike_model;
                    t.contact_sweeps = l.contact_sweeps;
                    t.max_contacts = l.max_contacts;
                    t.impact_peak = l.impact_peak;
                    t.gravity = l.gravity;
                    first = false;
                }
                let (n, e, p) = (l.chunks.len() as u32, l.bonds.len() as u32, l.parts.len() as u32);
                let base = fracture::Base {
                    chunk: chunks.take(n)?,
                    bond: bonds.take(e)?,
                    part: parts.take(p)?,
                };
                // One after another from 0: the tables' lengths are the offsets.
                debug_assert_eq!(base.chunk as usize, t.chunks.len());
                debug_assert_eq!(base.bond as usize, t.bonds.len());
                t.chunks.extend(l.chunks.iter().map(|c| {
                    let mut c = *c;
                    c[2][2] = f32::from_bits(c[2][2].to_bits() + base.part);
                    c
                }));
                t.info.extend(&l.info);
                t.bonds.extend(l.bonds.iter().map(|b| [b[0] + base.chunk, b[1] + base.chunk, b[2], b[3]]));
                t.parts.extend(&l.parts);
                t.chunk_body.extend(&l.chunk_body);
                t.geometry.extend(&l.geometry);
                let (adj, mats) = (t.ext.adjacency.len() as u32, t.ext.materials.len() as u32);
                t.ext.bonds.extend(l.ext.bonds.iter().map(|b| {
                    let mut b = *b;
                    b[9][0] = f32::from_bits(b[9][0].to_bits() + mats);
                    b
                }));
                t.ext.chunks.extend(l.ext.chunks.iter().map(|c| {
                    let mut c = *c;
                    c[0][0] = f32::from_bits(c[0][0].to_bits() + adj);
                    c
                }));
                t.ext.adjacency.extend(l.ext.adjacency.iter().map(|&b| b + base.bond));
                t.ext.materials.extend(&l.ext.materials);
                t.lp_init.extend(&l.lp_init);
                for &(body, frame) in &slot.cluster_frames {
                    t.frames[body as usize] = frame;
                }
                if let Some(b) = &slot.blast_inputs {
                    blast.get_or_insert_with(BlastInputs::default).append(b);
                }
                sides.extend(slot.sides.iter().map(|&(col, mut info)| {
                    info.a[0] += base.chunk;
                    (col, info)
                }));
                chunk_ranges.push((base.chunk, base.chunk + n, k));
                bond_ranges.push((base.bond, base.bond + e, k));
                slot.base = base;
            }
            for &s in &self.fracture.spare {
                t.free[s as usize] = 1;
            }
            let room = fracture::Capacity {
                chunks: (t.chunks.len() as u32).max(1),
                bonds: (t.bonds.len() as u32).max(1),
                parts: (t.parts.len() as u32).max(1),
                events: EVENT_ROOM,
            };
            anyhow::ensure!(
                t.chunks.len() as u32 == room.chunks && t.bonds.len() as u32 == room.bonds,
                "a {} fracture needs at least one chunk and one bond",
                sh.setup.profile.name()
            );
            let engine = EngineBuffers {
                bodies: self.sim.body_buffer(),
                colliders: self.sim.collider_buffer(),
                contacts: self.sim.contact_buffer(),
                counters: self.sim.counters_buffer(),
            };
            // The ingest's per-chunk records are laid out for its chunk count, which the
            // fracture reads at its own: a new ingest for the fracture's (every collider the
            // static side but the members', whose sides follow; the projectiles' are synced).
            let mut table = vec![sh.static_info; capacity.max_colliders as usize];
            for &(col, info) in &sides {
                table[col as usize] = info;
            }
            sh.ingest = EngineIngest::new(
                gpu,
                capacity.max_contacts,
                1,
                capacity.max_bodies,
                room.chunks,
                &table,
            );
            let mut f = GpuFracture::with_capacity(gpu, &t, room, &engine, &sh.ingest)
                .map_err(|e| anyhow::anyhow!("{}: {e}", sh.setup.profile.name()))?;
            if let Some(b) = &blast {
                let g = [t.gravity[0] as f64, t.gravity[1] as f64, t.gravity[2] as f64];
                let frame = b.frame(gpu, g).map_err(|e| anyhow::anyhow!("{e}"))?;
                f.set_blast(gpu, frame).map_err(|e| anyhow::anyhow!("{e}"))?;
            }
            sh.fracture = f;
            sh.chunks = chunks;
            sh.bonds = bonds;
            sh.parts = parts;
            // The projectiles' sides and shots go to the new fracture (`sync_projectiles`).
            sh.projectile_colliders.clear();
            self.fracture.chunk_ranges = chunk_ranges;
            self.fracture.bond_ranges = bond_ranges;
            self.fracture.projectiles_dirty = true;
            Ok(())
        }

        /// `n` more parked slots in the world's piece pool.
        fn add_to_pool(&mut self, gpu: &Gpu, n: u32) -> anyhow::Result<()> {
            if n == 0 {
                return Ok(());
            }
            let ids = self.create_bodies(&vec![BodyDef::default(); n as usize])?;
            let pool: Vec<u32> = ids.iter().map(|id| id.index()).collect();
            self.park_spares(&pool);
            let sh = self.fracture.shared.as_mut().expect("shared fracture");
            let none = FractureTables::default();
            sh.fracture
                .append(gpu, fracture::Base::default(), &none, &[], &pool, None)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            self.fracture.spare.extend(pool);
            Ok(())
        }

        /// Park `slots` as spares: static, out of every pair, until a split takes one.
        fn park_spares(&mut self, slots: &[u32]) {
            let mut parked_records = Vec::with_capacity(slots.len());
            for &i in slots {
                let mut parked = BodyState::static_plane_support();
                parked.position = crate::graveyard(i);
                parked.initial_pos = parked.position;
                parked_records.push((i, parked));
                let slot = &mut self.bodies[i as usize];
                slot.owner = Owner::Piece;
                slot.alive = false;
            }
            // One upload per run of consecutive slots (a write each would stage thousands
            // of buffers before anything is submitted).
            for (first, records) in runs(parked_records) {
                self.sim.write_bodies(first, &records);
            }
        }

        /// The fracture's part of a step, between `ingest.begin_frame` (before the
        /// engine's step) and the readback: the ingest of the step's contacts, the loads,
        /// the breaks and the splits.
        pub(crate) fn encode_fracture_begin(&self, enc: &mut wgpu::CommandEncoder) {
            let e = buffers(&self.sim);
            if let Some(sh) = &self.fracture.shared {
                sh.ingest.begin_frame(enc, &e);
            }
            for o in self
                .fracture
                .slots
                .iter()
                .flatten()
                .filter_map(|t| t.own.as_ref())
            {
                o.ingest.begin_frame(enc, &e);
            }
        }

        /// Directed debris on what the stress models' splits make: every reserved body slot a
        /// split may move a piece onto (the shared pool and each own fracture's spares) armed
        /// with `tag_mask` (0: off), so a piece that breaks off is guided as a chunk body is.
        /// A parked slot is static and out of every pair; the record waits for its free flight.
        pub(crate) fn guide_fracture_spares(&mut self, tag_mask: u32) -> anyhow::Result<()> {
            let mut slots: Vec<u32> = self.fracture.spare.iter().copied().collect();
            for o in self.fracture.slots.iter().flatten().filter_map(|t| t.own.as_ref()) {
                slots.extend(&o.spare);
            }
            slots.sort_unstable();
            slots.dedup();
            let records: Vec<(u32, engine_schema::guidance::GuideBodyGpu)> = slots
                .into_iter()
                .map(|b| {
                    let r = if tag_mask == 0 {
                        engine_schema::guidance::GuideBodyGpu::default()
                    } else {
                        engine_schema::guidance::GuideBodyGpu::armed(self.bodies[b as usize].generation, tag_mask)
                    };
                    (b, r)
                })
                .collect();
            for (first, run) in runs(records) {
                self.sim.write_guide_bodies(first, &run)?;
            }
            Ok(())
        }

        /// Diagnostic (`blast_gpu`): the shared fracture's Blast graph, for a reference
        /// solve of it (stress-ref `BlastSolver`).
        #[doc(hidden)]
        pub fn stress_blast_graph(&self) -> Option<stress_ref::blast_bond::BlastGraph> {
            let sh = self.fracture.shared.as_ref()?;
            Some(sh.fracture.blast()?.solver.graph.clone())
        }

        /// Diagnostic (`blast_gpu`, waits on the GPU): the shared fracture's Blast bond
        /// impulses as its last solve left them, in the graph's bond order.
        #[doc(hidden)]
        pub fn stress_blast_impulses(&self) -> Option<Vec<stress_ref::blast_bond::AngLin>> {
            let sh = self.fracture.shared.as_ref()?;
            Some(sh.fracture.blast()?.solver.impulses())
        }

        /// Whether a fracture runs the in-step contact fracture
        /// (`FractureOptions::contact_fracture`): the rigid step is then encoded with it
        /// ([`Self::step_encode_contact`]).
        pub(crate) fn fracture_in_step(&self) -> bool {
            self.fracture
                .shared
                .as_ref()
                .is_some_and(|sh| sh.setup.contact_sweeps > 0)
                || self
                    .fracture
                    .slots
                    .iter()
                    .flatten()
                    .filter_map(|t| t.own.as_ref())
                    .any(|o| o.setup.contact_sweeps > 0)
        }

        /// The rigid step with the in-step contact fracture where its contacts are made and
        /// kept, before its solve (stress-rubble `CONTACT_PHASE`), as the lab's fracture
        /// world steps.
        pub(crate) fn step_encode_contact(
            &mut self,
            enc: &mut wgpu::CommandEncoder,
        ) -> anyhow::Result<()> {
            let gpu = self.fracture.gpu.clone().expect("a fracture's GPU");
            let rows = self.sim.next_contact_offset();
            let mut fractures: Vec<&mut GpuFracture> = Vec::new();
            if let Some(sh) = self.fracture.shared.as_mut() {
                if sh.setup.contact_sweeps > 0 {
                    fractures.push(&mut sh.fracture);
                }
            }
            for o in self
                .fracture
                .slots
                .iter_mut()
                .flatten()
                .filter_map(|t| t.own.as_mut())
            {
                if o.setup.contact_sweeps > 0 {
                    fractures.push(&mut o.fracture);
                }
            }
            for f in fractures.iter_mut() {
                f.set_contact_rows(&gpu, rows);
            }
            self.sim
                .step_encode_with(enc, stress_rubble::threshold::CONTACT_PHASE, &mut |e| {
                    for f in fractures.iter_mut() {
                        f.encode_contact_fracture(e);
                    }
                })
                .map_err(|e| anyhow::anyhow!("step: {e:?}"))?;
            Ok(())
        }

        pub(crate) fn encode_fracture(&mut self, enc: &mut wgpu::CommandEncoder) {
            let Some(gpu) = self.fracture.gpu.clone() else {
                return;
            };
            let e = buffers(&self.sim);
            let offset = self.sim.last_contact_offset();
            let dt = self.sim.params().dt as f64;
            let step = self.sim.steps_done();
            if let Some(sh) = self.fracture.shared.as_mut() {
                sh.ingest.substep(&gpu, enc, &e, offset, 0, dt);
                sh.ingest.merge(&gpu, enc, &e, 1);
                sh.fracture.encode_frame(&gpu, enc, step);
                sh.stepped = true;
            }
            for o in self
                .fracture
                .slots
                .iter_mut()
                .flatten()
                .filter_map(|t| t.own.as_mut())
            {
                o.ingest.substep(&gpu, enc, &e, offset, 0, dt);
                o.ingest.merge(&gpu, enc, &e, 1);
                o.fracture.encode_frame(&gpu, enc, step);
            }
        }
    }

    impl Fracture {
        /// The readback copies of every threshold destructible's state after a step: the
        /// shared fracture's counters and event lists (nothing that scales with the
        /// destructibles placed), and each own fracture's counters, chunk owners, bond
        /// states, body frames and crush events.
        pub(crate) fn extras(&self) -> Vec<crate::readback::Extra<'_, crate::Extra>> {
            let mut out = Vec::new();
            let mut push = |read, buffer, offset, size| {
                if size > 0 {
                    out.push(crate::readback::Extra {
                        kind: crate::Extra::Fracture(read),
                        buffer,
                        offset,
                        size,
                    });
                }
            };
            if let Some(sh) = &self.shared {
                let f = &sh.fracture;
                let l = f.layout;
                push(
                    FractureRead::Counters,
                    &f.atom,
                    4 * l.a_counters() as u64,
                    4 * COUNTER_WORDS,
                );
                push(
                    FractureRead::Broken,
                    &f.atom,
                    4 * l.a_events() as u64,
                    4 * l.events.broken as u64,
                );
                push(
                    FractureRead::Moved,
                    &f.atom,
                    4 * l.a_moved() as u64,
                    8 * l.events.moved as u64,
                );
                push(
                    FractureRead::Frames,
                    &f.frames,
                    16 * l.f_events() as u64,
                    48 * l.events.frames as u64,
                );
            }
            for (k, t) in self.slots.iter().enumerate() {
                let Some(o) = t.as_ref().and_then(|t| t.own.as_ref()) else {
                    continue;
                };
                let k = k as u32;
                let f = &o.fracture;
                let l = f.layout;
                push(
                    FractureRead::SlotCounters(k),
                    &f.atom,
                    4 * l.a_counters() as u64,
                    4 * COUNTER_WORDS,
                );
                push(
                    FractureRead::ChunkBodies(k),
                    &f.work,
                    4 * l.w_body() as u64,
                    4 * l.n as u64,
                );
                push(
                    FractureRead::Intact(k),
                    &f.work,
                    4 * l.w_alive() as u64,
                    4 * l.e as u64,
                );
                push(FractureRead::SlotFrames(k), &f.frames, 0, 32 * l.b as u64);
                push(
                    FractureRead::Crush(k),
                    &f.crush_rw,
                    f.crush_events_range().0,
                    if f.crushes() {
                        f.crush_events_range().1
                    } else {
                        0
                    },
                );
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
            // This frame's (a blast's crushes before the step report no adoption).
            self.fracture.crushed_origins.clear();
            let mut out = self.shared_frame(step, reads)?;
            out.extend(self.own_frames(step, reads)?);
            Ok(out)
        }

        /// The shared fracture's frame: its event lists, each to its destructible.
        fn shared_frame(
            &mut self,
            step: u64,
            reads: &[(FractureRead, &[u8])],
        ) -> anyhow::Result<Vec<DestructionEvent>> {
            let mut out = Vec::new();
            let find = |want: FractureRead| {
                reads
                    .iter()
                    .find(|(r, _)| std::mem::discriminant(r) == std::mem::discriminant(&want))
                    .map(|(_, b)| *b)
            };
            let Some(counters) = find(FractureRead::Counters) else {
                return Ok(out);
            };
            let counters: &[u32] = bytemuck::cast_slice(counters);
            let [nb, nm, nf] = fracture::Layout::EVENT_COUNTERS.map(|w| counters[w as usize]);
            anyhow::ensure!(
                counters[3] & fracture::ERROR_EVENTS == 0,
                "step {step}: threshold fracture events beyond their room: {nb} bonds broken (room {}), {nm} chunks moved (room {}), {nf} bodies re-framed (room {})",
                EVENT_ROOM.broken,
                EVENT_ROOM.moved,
                EVENT_ROOM.frames
            );
            anyhow::ensure!(counters[3] == 0, "step {step}: fracture error {:#x} ({})", counters[3], fracture_error_text(counters[3]));
            let words = |r: FractureRead, n: usize| -> &[u32] {
                find(r)
                    .map(|b| &bytemuck::cast_slice::<u8, u32>(b)[..n])
                    .unwrap_or_default()
            };
            let broken = words(FractureRead::Broken, nb as usize);
            let moved = words(FractureRead::Moved, 2 * nm as usize);
            let frame_words: &[[f32; 4]] = find(FractureRead::Frames)
                .map(|b| &bytemuck::cast_slice::<u8, [f32; 4]>(b)[..3 * nf as usize])
                .unwrap_or_default();
            if broken.is_empty() && moved.is_empty() {
                return Ok(out);
            }
            let frames: std::collections::HashMap<u32, Frame> = frame_words
                .chunks_exact(3)
                .map(|r| (r[0][0].to_bits(), [r[1], r[2]]))
                .collect();
            // Each event to its destructible, in its own numbering.
            let owner = |ranges: &[(u32, u32, usize)], at: u32| -> Option<(usize, u32)> {
                let i = ranges.partition_point(|r| r.0 <= at).checked_sub(1)?;
                let (start, end, k) = ranges[i];
                (at < end).then_some((k, at - start))
            };
            type SlotEvents = (Vec<u32>, Vec<(u32, u32)>);
            let mut per_slot: std::collections::BTreeMap<usize, SlotEvents> =
                std::collections::BTreeMap::new();
            for &b in broken {
                if let Some((k, local)) = owner(&self.fracture.bond_ranges, b) {
                    per_slot.entry(k).or_default().0.push(local);
                }
            }
            for pair in moved.chunks_exact(2) {
                if let Some((k, local)) = owner(&self.fracture.chunk_ranges, pair[0]) {
                    per_slot.entry(k).or_default().1.push((local, pair[1]));
                }
            }
            for (k, (bonds, chunks)) in per_slot {
                let t = self.fracture.slots[k].as_ref().expect("live slot");
                let mut own_chunks = t.chunk_body.clone();
                for (c, b) in chunks {
                    own_chunks[c as usize] = b;
                }
                let mut own_intact = t.intact.clone();
                for b in bonds {
                    own_intact[b as usize] = false;
                }
                let events = self.apply_fracture(
                    k,
                    step,
                    own_chunks,
                    own_intact,
                    Frames::Sparse(&frames),
                    &[],
                )?;
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

        /// Each own fracture's frame (crushing, blast loading): its whole chunk owners,
        /// bond states and frames, and its crushes.
        fn own_frames(
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
                    find(FractureRead::SlotCounters),
                    find(FractureRead::ChunkBodies),
                ) else {
                    continue;
                };
                let counters: &[u32] = bytemuck::cast_slice(counters);
                anyhow::ensure!(counters[3] & fracture::CRUSH_ERROR_FRAGMENTS == 0, "step {step}: a chunk crush found the fragment bodies spent: raise CrushMass::Conserve's budget");
                anyhow::ensure!(
                    counters[3] & fracture::CRUSH_ERROR_EVENTS == 0,
                    "step {step}: more chunks crushed in one step than Crush::max_per_step"
                );
                anyhow::ensure!(counters[3] == 0, "step {step}: fracture error {:#x} ({})", counters[3], fracture_error_text(counters[3]));
                let crushes: Vec<CrushEvent> = find(FractureRead::Crush)
                    .map(|b| {
                        CrushEvent::decode(
                            bytemuck::cast_slice(b),
                            counters[fracture::C_CRUSH_EVENTS],
                        )
                    })
                    .unwrap_or_default();
                let crushed_now: Vec<u32> = crushes.iter().map(|e| e.chunk).collect();
                let chunks: Vec<u32> = bytemuck::cast_slice(chunks).to_vec();
                let intact: Vec<bool> = find(FractureRead::Intact)
                    .map(|b| {
                        bytemuck::cast_slice::<u8, u32>(b)
                            .iter()
                            .map(|&x| x != 0)
                            .collect()
                    })
                    .unwrap_or_default();
                let frames: Vec<[f32; 4]> = find(FractureRead::SlotFrames)
                    .map(|b| bytemuck::cast_slice(b).to_vec())
                    .unwrap_or_default();
                let events = self.apply_fracture(
                    k,
                    step,
                    chunks,
                    intact,
                    Frames::Dense(&frames),
                    &crushed_now,
                )?;
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
                out.extend(self.apply_crush(k, step, &crushes, &frames)?);
            }
            Ok(out)
        }

        fn apply_fracture(
            &mut self,
            k: usize,
            step: u64,
            chunks: Vec<u32>,
            intact: Vec<bool>,
            frames: Frames<'_>,
            crushed_now: &[u32],
        ) -> anyhow::Result<Vec<DestructionEvent>> {
            // This step's poses are from before the split: re-frame from the next.
            self.apply_fracture_from(
                k,
                step + 1,
                chunks,
                intact,
                frames,
                crushed_now,
                BreakCause::Stress,
            )
        }

        /// A split's adoption, the changed bodies re-framed from `from_step` on. The
        /// broken bonds are reported by cause: a crushed chunk's (`crushed_now`) Crush, a
        /// bond support loss broke SupportLoss, the rest `cause`.
        #[allow(clippy::too_many_arguments)]
        fn apply_fracture_from(
            &mut self,
            k: usize,
            from_step: u64,
            chunks: Vec<u32>,
            intact: Vec<bool>,
            frames: Frames<'_>,
            crushed_now: &[u32],
            cause: BreakCause,
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
                let bonds = &t.tables.bonds;
                let sent = t.support.as_ref().map(|sl| &sl.sent);
                let cause_of = |b: u32| {
                    let [a, o, _, _] = bonds[b as usize];
                    if crushed_now.contains(&a) || crushed_now.contains(&o) {
                        BreakCause::Crush
                    } else if sent.is_some_and(|s| s[b as usize]) {
                        BreakCause::SupportLoss
                    } else {
                        cause
                    }
                };
                for c in [
                    BreakCause::Stress,
                    BreakCause::Blast,
                    BreakCause::SupportLoss,
                    BreakCause::Crush,
                ] {
                    let these: Vec<u32> = broken
                        .iter()
                        .copied()
                        .filter(|&b| cause_of(b) == c)
                        .collect();
                    if !these.is_empty() {
                        out.push(DestructionEvent::BondsBroken {
                            destructible: id,
                            bonds: these,
                            cause: c,
                        });
                    }
                }
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
                // An own fracture's spares are its own; the shared one's, the world's.
                let taken = match t.own.as_mut() {
                    Some(o) => match o.spare.iter().position(|&s| s == b) {
                        Some(i) => {
                            o.spare.swap_remove(i);
                            true
                        }
                        None => false,
                    },
                    None => self.fracture.spare.remove(&b),
                };
                if taken {
                    spare_taken.push(b);
                }
            }
            // Adopt the pieces' new bodies; re-frame every changed body from `from_step`.
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
                self.reframe_piece(k, b, from_step, frames)?;
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
                // A crushed chunk is no piece (`ChunkCrushed` reports it).
                let children = groups
                    .iter()
                    .map(|&g| {
                        members
                            .iter()
                            .copied()
                            .filter(|&c| chunk_body[c as usize] == g)
                            .collect::<Vec<u32>>()
                    })
                    .filter(|chunks| !chunks.iter().all(|c| crushed_now.contains(c)))
                    .map(|chunks| {
                        let g = chunk_body[chunks[0] as usize];
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
            sl.broke.extend_from_slice(&over);
            match t.own.as_mut() {
                Some(o) => {
                    // The GPU marks each bond's body from its own chunk to body map: a step
                    // still in flight may have split the piece since `chunk_body` was read.
                    o.fracture.break_bonds_sparse(&over);
                    // The flow's loads for the crush estimate from the next frame on.
                    if o.fracture.crush_flow() {
                        o.fracture
                            .set_crush_flow(&gpu, &sl.load)
                            .expect("a flow-crushing destructible's loads");
                        let t = self.fracture.slots[k].as_ref().expect("live slot");
                        let sl = t.support.as_ref().expect("support loss");
                        let crushed = &self.destructibles[t.destructible as usize].crushed;
                        let over = flow_overstressed(t, sl, crushed);
                        let t = self.fracture.slots[k].as_mut().expect("live slot");
                        t.support.as_mut().expect("support loss").overstressed = over;
                    }
                }
                None => {
                    // In the shared tables' numbering; the next frame's events report them.
                    let at: Vec<u32> = over.iter().map(|&b| b + t.base.bond).collect();
                    let sh = self.fracture.shared.as_mut().expect("shared fracture");
                    sh.fracture.break_bonds_sparse(&at);
                }
            }
        }

        /// Whether a destructible's flow keeps a chunk's crush damage growing
        /// (`SupportLoss::overstressed`): its steps must run with nothing awake.
        pub(crate) fn fracture_flow_needs_step(&self) -> bool {
            // The load path and Blast's solve iterate a warm-started equilibrium a fixed
            // budget a step, converging over steps: a world at rest still steps them, as
            // the lab's fracture world does (skipping would leave the first, unconverged
            // answer in place).
            let iterates = |s: &ProfileSetup| {
                s.profile == Profile::BlastGpu || s.load_model != stress_ref::capability::LoadModel::Threshold
            };
            self.fracture.shared.as_ref().is_some_and(|sh| iterates(&sh.setup))
                || self
                    .fracture
                    .slots
                    .iter()
                    .flatten()
                    .any(|t| {
                        t.support.as_ref().is_some_and(|s| s.overstressed)
                            || t.own.as_ref().is_some_and(|o| iterates(&o.setup))
                    })
        }

        /// A step's crushes: each chunk's body (parked on the GPU) retired, its fragments
        /// (made on the GPU) adopted as bodies, and the events. `frames` is the fracture's
        /// frame record after the step (the crushed bodies' principal frames).
        fn apply_crush(
            &mut self,
            k: usize,
            _step: u64,
            crushes: &[CrushEvent],
            frames: &[[f32; 4]],
        ) -> anyhow::Result<Vec<DestructionEvent>> {
            if crushes.is_empty() {
                return Ok(Vec::new());
            }
            let t = self.fracture.slots[k].as_ref().expect("live slot");
            let d = t.destructible as usize;
            let id = DestructibleId::new(d as u32, self.destructibles[d].generation);
            let mut out = Vec::with_capacity(crushes.len());
            // What slept on a crushed chunk must not be left in the air: the chunk is gone,
            // not moved, so no awake body is left where it was to wake them by touch
            // (`World::sleeping_pile`; a frame or two late, as the crush is read back).
            let gone: Vec<(Vec3, f32)> = crushes
                .iter()
                .map(|e| {
                    let b = self.fracture.slots[k].as_ref().expect("live slot").chunk_body[e.chunk as usize];
                    (Vec3::from(e.position), self.body_reach(b as usize).0)
                })
                .collect();
            let resting = self.sleeping_pile(None, &gone, false);
            if !resting.is_empty() {
                self.dirty = true;
                self.sim.push_body_commands(&resting)?;
            }
            for e in crushes {
                let t = self.fracture.slots[k].as_ref().expect("live slot");
                let c = e.chunk as usize;
                let b = t.chunk_body[c] as usize;
                let (pool, fragments, particles, particle_ccd, material, shape_def) = {
                    let cs = t.crush.as_ref().expect("a crushing destructible");
                    (
                        cs.pool.clone(),
                        cs.pieces(c),
                        cs.particles.is_some(),
                        cs.particle_ccd,
                        cs.material,
                        cs.shape_def.clone(),
                    )
                };
                // Its body (alone after the split, parked out of the world on the GPU).
                let solver = Transform::new(e.position, e.orientation);
                let origin = self.bodies[b].frame().to_origin(solver);
                let pose = Transform::new(e.position, origin.rotation);
                self.fracture.crushed_origins.push((b as u32, origin));
                let slot = &mut self.bodies[b];
                slot.alive = false;
                slot.generation = slot.generation.wrapping_add(1).max(1);
                slot.body_type = BodyType::Static;
                slot.mass = None;
                slot.last_transform = None;
                slot.last_solver = None;
                // Its fragments: boxes about its centre of mass, origin = solver frame.
                let mut ids = Vec::new();
                // Each fragment's pose and velocity as `fr_crush_apply` wrote them (rubble's
                // frame): the chunk's box (its rotation `g` and half extents in the
                // structure frame) about its centre of mass, the body's `orientation` less
                // its principal rotation `p` turning the structure frame into the world's.
                let t = self.fracture.slots[k].as_ref().expect("live slot");
                let fragment_poses = {
                    let rec = frames
                        .get(2 * b + 1)
                        .ok_or_else(|| anyhow::anyhow!("fracture frame {} missing", 2 * b + 1))?;
                    let p = glam::Quat::from_array(*rec).normalize();
                    let geo =
                        t.tables.geometry.get(c).ok_or_else(|| {
                            anyhow::anyhow!("crush: chunk {c} has no box geometry")
                        })?;
                    let g = glam::Quat::from_array(geo[2]).normalize();
                    let o = (glam::Quat::from_array(e.orientation) * p.conjugate() * g).normalize();
                    let h = Vec3::new(geo[1][0], geo[1][1], geo[1][2]);
                    let (mut dims, mut cell) = ([1u32; 3], h);
                    let mut n = fragments;
                    while n > 1 {
                        let ax = if cell.x >= cell.y && cell.x >= cell.z {
                            0
                        } else if cell.y >= cell.z {
                            1
                        } else {
                            2
                        };
                        dims[ax] *= 2;
                        cell[ax] *= 0.5;
                        n >>= 1;
                    }
                    let (b0, v, w) = (
                        Vec3::from(e.position),
                        Vec3::from(e.linear_velocity),
                        Vec3::from(e.angular_velocity),
                    );
                    if particles {
                        // As `fr_crush_apply` spreads them: through the box, a radius in,
                        // about the centre of mass.
                        let r = e.fragment_half[0];
                        let room = (h - Vec3::splat(r)).max(h * 0.25);
                        let mut mean = Vec3::ZERO;
                        for i in 0..fragments {
                            mean += crush_particle_spot(e.chunk, i);
                        }
                        mean /= fragments as f32;
                        (0..fragments)
                            .map(|i| {
                                let x =
                                    b0 + o * (room * (crush_particle_spot(e.chunk, i) - mean));
                                (
                                    Transform::new(x.to_array(), [0.0, 0.0, 0.0, 1.0]),
                                    (v + w.cross(x - b0)).to_array(),
                                    [0.0; 3],
                                )
                            })
                            .collect::<Vec<_>>()
                    } else {
                    (0..fragments)
                        .map(|i| {
                            let idx = Vec3::new(
                                (i % dims[0]) as f32,
                                ((i / dims[0]) % dims[1]) as f32,
                                (i / (dims[0] * dims[1])) as f32,
                            );
                            let x = b0 + o * (-h + cell * (2.0 * idx + Vec3::ONE));
                            (
                                Transform::new(x.to_array(), o.to_array()),
                                (v + w.cross(x - b0)).to_array(),
                                w.to_array(),
                            )
                        })
                        .collect::<Vec<_>>()
                    }
                };
                if let Some(first) = e.first_fragment {
                    let mf = e.mass / fragments as f32;
                    let h = Vec3::from(e.fragment_half);
                    // A particle never rotates: no inertia (its inverse is zero).
                    let moments = if particles {
                        Vec3::ZERO
                    } else {
                        Vec3::new(
                            h.y * h.y + h.z * h.z,
                            h.x * h.x + h.z * h.z,
                            h.x * h.x + h.y * h.y,
                        ) * (mf / 3.0)
                    };
                    let geometry = if particles {
                        Geometry::Sphere {
                            center: [0.0; 3],
                            radius: h.x,
                        }
                    } else {
                        Geometry::Box {
                            half: e.fragment_half,
                            transform: Transform::IDENTITY,
                        }
                    };
                    for i in 0..fragments {
                        let (fb, shape, collider) =
                            *pool.get((first + i) as usize).ok_or_else(|| {
                                anyhow::anyhow!("crush fragment {} outside the pool", first + i)
                            })?;
                        let slot = &mut self.bodies[fb as usize];
                        slot.alive = true;
                        slot.generation = slot.generation.wrapping_add(1).max(1);
                        slot.body_type = BodyType::Dynamic;
                        slot.mass = Some(MassData {
                            mass: mf,
                            center: [0.0; 3],
                            rotational_inertia: glam::Mat3::from_diagonal(moments)
                                .to_cols_array_2d(),
                        });
                        slot.principal = moments;
                        slot.ccd_when_fast = particle_ccd;
                        slot.frames = vec![(0, SolverFrame::IDENTITY)];
                        slot.last_transform = None;
                        slot.last_solver = None;
                        slot.last_asleep = false;
                        slot.asleep = false;
                        let (pose, v, w) = fragment_poses[i as usize];
                        slot.origin = pose;
                        slot.velocity = (v, w);
                        // The GPU wrote the body (dynamic, awake): the host's mirror follows,
                        // with no upload (the step plan reads which bodies move).
                        let state = crate::body_state(slot);
                        self.sim.mirror_bodies(fb, &[state]);
                        self.shapes[shape as usize].geometry = geometry.clone();
                        // The GPU rewrote the collider: the host's mirror follows.
                        let mut col = self.collider(
                            fb,
                            SolverFrame::IDENTITY,
                            &geometry,
                            0.0,
                            material,
                            None,
                        )?;
                        shape_def.filter.apply(&mut col);
                        col.material |= crate::contacts::event_bits(&shape_def);
                        col.generation = self.sim.collider_record(collider).generation;
                        self.sim.mirror_colliders(collider, &[col]);
                        ids.push(BodyId::new(fb, self.bodies[fb as usize].generation));
                    }
                }
                self.destructibles[d].crushed[c] = true;
                self.destructibles[d].chunk_bodies[c] =
                    BodyId::new(b as u32, self.bodies[b].generation);
                out.push(DestructionEvent::ChunkCrushed {
                    destructible: id,
                    chunk: e.chunk,
                    mass: e.mass,
                    volume: e.volume,
                    pose,
                    linear_velocity: e.linear_velocity,
                    angular_velocity: e.angular_velocity,
                    peak_pressure: e.peak_pressure,
                    peak_deviator: e.peak_deviator,
                    cause: match e.cause {
                        2 => crate::CrushCause::Impact,
                        3 => crate::CrushCause::Blast,
                        4 => crate::CrushCause::Load,
                        _ => crate::CrushCause::Contact,
                    },
                    fragments: ids,
                });
            }
            Ok(out)
        }

        /// The bonds support loss (`DestructibleDef::support_loss`) has broken so far, in the
        /// order it broke them (bond indices into the asset's threshold tables, as
        /// `BondsBroken` reports them). `BondsBroken` does not carry a cause; this tells a
        /// host which of its breaks were the weight's. Empty without support loss.
        pub fn destructible_support_loss_breaks(
            &self,
            id: DestructibleId,
        ) -> anyhow::Result<&[u32]> {
            let i = self.live_destructible(id)?;
            let k = self.destructibles[i].threshold.ok_or_else(|| {
                anyhow::anyhow!("destructible {} is not a threshold one", id.index())
            })? as usize;
            let t = self.fracture.slots[k].as_ref().expect("live slot");
            Ok(t.support.as_ref().map_or(&[][..], |s| &s.broke[..]))
        }

        /// A threshold destructible's chunks' crush state (blocking: for tools and
        /// tests): per chunk (damage, peak pressure, peak deviator (Pa), crushed). Empty
        /// without `DestructibleDef::crush`.
        pub fn destructible_crush_state(
            &self,
            id: DestructibleId,
        ) -> anyhow::Result<Vec<(f32, f32, f32, bool)>> {
            let i = self.live_destructible(id)?;
            let k = self.destructibles[i].threshold.ok_or_else(|| {
                anyhow::anyhow!("destructible {} is not a threshold one", id.index())
            })? as usize;
            let t = self.fracture.slots[k].as_ref().expect("live slot");
            let gpu = self
                .fracture
                .gpu
                .as_ref()
                .expect("a threshold destructible's GPU");
            let Some(o) = &t.own else {
                // Crushing has its own fracture: a shared one's destructible has none.
                return Ok(Vec::new());
            };
            Ok(o.fracture
                .crush_state(gpu)
                .iter()
                .map(|s| (s[0], s[1], s[2], s[3].to_bits() & 1 != 0))
                .collect())
        }

        /// A threshold destructible's fracture counters on the GPU (blocking: for tools and
        /// tests): bonds broken and new piece bodies made, in all, after every step
        /// submitted. Every broken bond is counted once, whatever broke it (contact loads,
        /// a strike, a blast, support loss, a crush), as `BondsBroken` reports it once.
        pub fn destructible_fracture_counts(
            &self,
            id: DestructibleId,
        ) -> anyhow::Result<(u32, u32)> {
            let i = self.live_destructible(id)?;
            let k = self.destructibles[i].threshold.ok_or_else(|| {
                anyhow::anyhow!("destructible {} is not a threshold one", id.index())
            })? as usize;
            let t = self.fracture.slots[k].as_ref().expect("live slot");
            let gpu = self
                .fracture
                .gpu
                .as_ref()
                .expect("a threshold destructible's GPU");
            match &t.own {
                Some(o) => {
                    let c = o.fracture.counters(gpu);
                    Ok((c.broken, c.pieces))
                }
                None => {
                    // Its own ranges of the shared tables: bonds no longer intact, and
                    // bodies its chunks are on beyond the clusters it was placed with.
                    let sh = self.fracture.shared.as_ref().expect("shared fracture");
                    let l = sh.fracture.layout;
                    let (n, e) = (t.chunk_body.len(), t.intact.len());
                    let alive: Vec<u32> =
                        gpu.read_range(&sh.fracture.work, (l.w_alive() + t.base.bond) as usize, e);
                    let mut bodies: Vec<u32> =
                        gpu.read_range(&sh.fracture.work, (l.w_body() + t.base.chunk) as usize, n);
                    bodies.sort_unstable();
                    bodies.dedup();
                    let broken = alive.iter().filter(|&&a| a == 0).count() as u32;
                    Ok((broken, (bodies.len() as u32).saturating_sub(t.clusters)))
                }
            }
        }

        /// A changed body's mass, solver frame (from the GPU's frame record), type and
        /// shapes, from `from_step` on.
        fn reframe_piece(
            &mut self,
            k: usize,
            b: u32,
            from_step: u64,
            frames: Frames<'_>,
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
            let [c0, c1] = frames
                .get(b)
                .ok_or_else(|| anyhow::anyhow!("fracture frame of body {b} missing"))?;
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
            let frame = piece_frame(t, c0, c1);
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

        /// A piece body's solver frame as the GPU holds it now, after every step submitted
        /// (polled or not), for `read_body_state`: a step still in flight may have split the
        /// body and re-framed it, and the host's frame changes only when its readback is
        /// polled. `None` for a body that is no threshold piece. Found by slot (every slot
        /// a destructible's pieces may take), not by the host's last read of the chunks'
        /// bodies, which a split in flight may have changed; the crush fragments' pool,
        /// whose frames the fracture does not keep, excluded.
        pub(crate) fn threshold_frame_now(&self, body: u32) -> Option<SolverFrame> {
            let gpu = self.fracture.gpu.as_ref()?;
            let t = self
                .fracture
                .slots
                .iter()
                .flatten()
                .find(|t| match &t.own {
                    Some(o) => {
                        o.owned.contains(&body)
                            && !t
                                .crush
                                .as_ref()
                                .is_some_and(|c| c.pool.iter().any(|p| p.0 == body))
                    }
                    // In the shared fracture a spare serves whichever destructible splits
                    // first: a piece the host has adopted (its chunks are on it).
                    None => t.chunk_body.contains(&body),
                })?;
            let frames = match &t.own {
                Some(o) => &o.fracture.frames,
                None => &self.fracture.shared.as_ref()?.fracture.frames,
            };
            let r: Vec<[f32; 4]> = gpu.read_range(frames, 2 * body as usize, 2);
            Some(piece_frame(t, r[0], r[1]))
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

        /// Remove a threshold destructible's records from the shared fracture and every
        /// body it owns: the bodies its chunks are on, and as many spare slots as it added
        /// that no piece took (a split takes the lowest spare, whoever reserved it).
        pub(crate) fn destroy_threshold(&mut self, k: usize) -> anyhow::Result<()> {
            let t = self.fracture.slots[k].take().expect("live slot");
            if let Some(o) = t.own {
                // Its own fracture goes with it: every slot it owns is freed.
                for b in o.owned {
                    let slot = &self.bodies[b as usize];
                    if slot.alive {
                        self.bodies[b as usize].owner = Owner::User;
                        let id = BodyId::new(b, self.bodies[b as usize].generation);
                        self.destroy_body(id)?;
                    } else {
                        // A reserved slot no piece took, a crushed chunk's parked body or
                        // an unused fragment: its shapes retired (a crushed chunk's, a
                        // fragment's box), then the slot freed.
                        for s in std::mem::take(&mut self.bodies[b as usize].shapes) {
                            self.release_shape(s);
                        }
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
                return Ok(());
            }
            self.fracture.chunk_ranges.retain(|r| r.2 != k);
            self.fracture.bond_ranges.retain(|r| r.2 != k);
            let gpu = self
                .fracture
                .gpu
                .clone()
                .expect("a threshold destructible's GPU");
            let colliders: Vec<u32> = t
                .chunk_shapes
                .iter()
                .flatten()
                .map(|&s| self.shapes[s as usize].collider)
                .collect();
            let mut bodies: Vec<u32> = t.chunk_body.clone();
            bodies.sort_unstable();
            bodies.dedup();
            // With a world piece pool, the slots its pieces took are replaced in the pool.
            let pool_refill = if self.fracture_pieces.is_some() {
                (bodies.len() as u32).saturating_sub(t.clusters)
            } else {
                0
            };
            // Pieces beyond its clusters took spares; the rest of its spares go back.
            let taken = (bodies.len() as u32).saturating_sub(t.clusters);
            let give_back = t.spares_added.saturating_sub(taken) as usize;
            let returned: Vec<u32> = self
                .fracture
                .spare
                .iter()
                .rev()
                .take(give_back)
                .copied()
                .collect();
            for b in &returned {
                self.fracture.spare.remove(b);
            }
            if let Some(sh) = self.fracture.shared.as_mut() {
                let (n, e, p) = (
                    t.chunk_body.len() as u32,
                    t.intact.len() as u32,
                    t.tables.parts.len() as u32,
                );
                sh.fracture.retire(&gpu, t.base, n, e, &returned);
                // Its colliders are static geometry again (their slots will be reused).
                let stride = std::mem::size_of::<ColliderInfo>() as u64;
                for c in colliders {
                    gpu.queue.write_buffer(
                        &sh.ingest.collider_info,
                        stride * c as u64,
                        bytemuck::bytes_of(&sh.static_info),
                    );
                }
                sh.chunks.give(t.base.chunk, n);
                sh.bonds.give(t.base.bond, e);
                sh.parts.give(t.base.part, p);
            }
            for b in bodies {
                if self.bodies[b as usize].alive {
                    self.bodies[b as usize].owner = Owner::User;
                    let id = BodyId::new(b, self.bodies[b as usize].generation);
                    self.destroy_body(id)?;
                } else {
                    // A reserved slot no piece took, a crushed chunk's parked body or an
                    // unused fragment: its shapes retired (a crushed chunk's, a fragment's
                    // box), then the slot freed.
                    for s in std::mem::take(&mut self.bodies[b as usize].shapes) {
                        self.release_shape(s);
                    }
                    let slot = &mut self.bodies[b as usize];
                    slot.owner = Owner::User;
                    slot.generation = slot.generation.wrapping_add(1).max(1);
                    slot.removed_at = Some(self.step);
                    self.free_bodies.push(b);
                }
            }
            for b in returned {
                // A reserved slot no piece took: free it.
                let slot = &mut self.bodies[b as usize];
                slot.owner = Owner::User;
                slot.generation = slot.generation.wrapping_add(1).max(1);
                slot.removed_at = Some(self.step);
                self.free_bodies.push(b);
            }
            self.add_to_pool(&gpu, pool_refill)?;
            if !self.fracture.active() {
                self.fracture.fixed_plan = None;
            }
            Ok(())
        }

        pub(crate) fn fracture_projectiles_dirty(&self) -> bool {
            self.fracture.projectiles_dirty
        }
    }

    /// `items` grouped into runs of consecutive indices (sorted first): one upload each.
    fn runs<T>(mut items: Vec<(u32, T)>) -> Vec<(u32, Vec<T>)> {
        items.sort_by_key(|i| i.0);
        let mut out: Vec<(u32, Vec<T>)> = Vec::new();
        for (i, item) in items {
            match out.last_mut() {
                Some((first, run)) if *first + run.len() as u32 == i => run.push(item),
                _ => out.push((i, vec![item])),
            }
        }
        out
    }

    /// `Crush::materials` onto the stress scene's chunk materials: a name matches its
    /// material, and a pack import's per-node copies of it (`brick#mass12`, `brick@crush#0`).
    fn apply_crush_materials(scene: &mut Scene, c: &crate::Crush) -> anyhow::Result<()> {
        for (name, m) in &c.materials {
            let params = stress_ref::crush::CrushParams {
                cap_pressure: m.cap_pressure,
                cohesion: m.cohesion,
                friction_slope: m.friction_slope,
                crush_energy: m.crush_energy,
                crush_viscosity: m.crush_viscosity,
            };
            params
                .validate(name)
                .map_err(|e| anyhow::anyhow!("Crush::materials: {e}"))?;
            let mut found = false;
            for (key, mat) in scene.materials.iter_mut() {
                let base = key.split(['#', '@']).next().unwrap_or(key);
                if key == name || base == name {
                    mat.crush = Some(params);
                    found = true;
                }
            }
            anyhow::ensure!(
                found,
                "Crush::materials: the stress scene has no material '{name}' (its materials: {:?})",
                scene.materials.keys().collect::<Vec<_>>()
            );
        }
        Ok(())
    }

    /// A body's solver frame from the GPU's frame record (`frames`: its centre of mass
    /// and principal rotation in its structure's stress frame, the structure as bits).
    fn piece_frame(t: &ThresholdSlot, c0: [f32; 4], c1: [f32; 4]) -> SolverFrame {
        let (sq, st) = t.structures[c0[3].to_bits() as usize];
        let com = DVec3::new(c0[0] as f64, c0[1] as f64, c0[2] as f64);
        let principal =
            DQuat::from_xyzw(c1[0] as f64, c1[1] as f64, c1[2] as f64, c1[3] as f64).normalize();
        SolverFrame {
            center: (st + sq * com).as_vec3(),
            axes: (sq * principal).normalize().as_quat(),
        }
    }

    fn read_slot(r: FractureRead) -> u32 {
        match r {
            FractureRead::SlotCounters(k)
            | FractureRead::ChunkBodies(k)
            | FractureRead::Intact(k)
            | FractureRead::SlotFrames(k)
            | FractureRead::Crush(k) => k,
            FractureRead::Counters
            | FractureRead::Broken
            | FractureRead::Moved
            | FractureRead::Frames => u32::MAX,
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

/// The host's mirror of a crush's fragment bodies (`apply_crush`): dynamic, where the GPU
/// put them. Reads the mirror, which no public call shows.
#[cfg(all(test, feature = "stress"))]
mod crush_mirror_tests {
    use crate::*;

    /// Two crush-wall blocks bonded front to back (`tests/crush.rs` `two_blocks_json`).
    fn two_blocks() -> DestructibleAssetDesc {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../engine-webgpu/fracture/packs/crush-wall.json"
        );
        let mut json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let sc = &mut json["scenario"];
        let nodes = sc["nodes"].as_array().unwrap().clone();
        let at = |x: f64| {
            nodes
                .iter()
                .position(|n| {
                    let c = &n["centroid"];
                    let near = |k: &str, v: f64| (c[k].as_f64().unwrap() - v).abs() < 1e-6;
                    near("x", x) && near("y", 1.8) && near("z", 0.2)
                })
                .unwrap()
        };
        let keep = [at(0.2), at(-0.2)];
        for key in ["nodes", "nodeTypes", "nodeSizes", "nodeColliders"] {
            if let Some(a) = sc[key].as_array() {
                let a: Vec<serde_json::Value> = keep.iter().map(|&i| a[i].clone()).collect();
                sc[key] = a.into();
            }
        }
        let bonds: Vec<serde_json::Value> = sc["bonds"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|b| {
                let a = keep
                    .iter()
                    .position(|&k| k as u64 == b["node0"].as_u64().unwrap())?;
                let c = keep
                    .iter()
                    .position(|&k| k as u64 == b["node1"].as_u64().unwrap())?;
                let mut b = b.clone();
                b["node0"] = a.into();
                b["node1"] = c.into();
                Some(b)
            })
            .collect();
        sc["bonds"] = bonds.into();
        DestructibleAssetDesc::from_pack(&serde_json::to_vec(&json).unwrap()).unwrap()
    }

    #[test]
    fn fragments_are_mirrored_dynamic_where_the_gpu_put_them() {
        let asset = two_blocks();
        let stress = asset.stress.as_ref().unwrap();
        let solver = asset.solver_for(StrengthModel::Threshold).unwrap();
        let scene = stress.threshold_scene(1.0, solver.dt).unwrap();
        let mut capacity = stress_rubble::capacity_for(&scene);
        capacity.max_bodies = capacity.max_bodies.max(64);
        capacity.max_colliders = capacity.max_colliders.max(capacity.max_bodies + 64);
        capacity.max_contacts =
            stress_rubble::threshold::contact_capacity(2, capacity.max_contacts, 1 << 30);
        capacity.max_candidates = capacity.max_candidates.max(capacity.max_contacts);
        let def = WorldDef {
            capacity,
            gravity: [0.0; 3],
            solver: engine_schema::AvbdParams {
                gravity_y: 0.0,
                ..solver
            },
            enable_sleep: false,
            contacts: ContactStorage::Canonical,
            ..WorldDef::default()
        };
        let mut w = World::new(GpuContext::Default, &def).unwrap();
        let offset = stress.placement().map(|x| -x as f32);
        let a = w.create_destructible_asset(&asset).unwrap();
        w.create_destructible(&DestructibleDef {
            model: StrengthModel::Threshold,
            transform: Transform::new(offset, [0.0, 0.0, 0.0, 1.0]),
            asleep: false,
            strength_scale: 1e6,
            crush: Some(Crush {
                mass: CrushMass::Conserve {
                    fragments: 8,
                    budget: 8,
                },
                ..Crush::default()
            }),
            ..DestructibleDef::new(a)
        })
        .unwrap();
        // A heavy fast ball at the front block (chunk 0), from +x.
        let p0 = glam::Vec3::from(asset.chunks[0].pose.position) + glam::Vec3::from(offset);
        let (r, m) = (0.45f32, 4800.0f32);
        let ball = w
            .create_body(&BodyDef {
                linear_velocity: [-100.0, 0.0, 0.0],
                ..BodyDef::dynamic((p0 + glam::Vec3::new(3.8, 0.0, 0.0)).to_array())
            })
            .unwrap();
        let shape = ShapeDef {
            update_body_mass: false,
            ..ShapeDef::default()
        };
        w.create_sphere_shape(ball, &shape, [0.0; 3], r).unwrap();
        let i = 0.4 * m * r * r;
        w.body_set_mass_data(
            ball,
            MassData {
                mass: m,
                center: [0.0; 3],
                rotational_inertia: [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, i]],
            },
        )
        .unwrap();
        w.body_set_projectile(ball, Some(Projectile::default()))
            .unwrap();
        let mut checked = 0;
        for _ in 0..20 {
            let f = w.step_sync().unwrap();
            for e in &f.destruction_events {
                let DestructionEvent::ChunkCrushed { fragments, .. } = e else {
                    continue;
                };
                assert_eq!(fragments.len(), 8);
                for b in fragments {
                    let mirror = engine_softbody::SoftBackend::host_body(&w.sim, b.index());
                    let gpu = w.sim.read_body(b.index());
                    assert!(mirror.inv_mass > 0.0, "fragment {b:?} mirrored static");
                    assert!((mirror.inv_mass - gpu.inv_mass).abs() <= 1e-4 * gpu.inv_mass);
                    let d = glam::Vec3::from(mirror.position) - glam::Vec3::from(gpu.position);
                    // No step since the crush: the GPU's record is `fr_crush_apply`'s.
                    assert!(
                        d.length() < 1e-4,
                        "fragment {b:?}: mirror {:?}, GPU {:?}",
                        mirror.position,
                        gpu.position
                    );
                    let dv = glam::Vec3::from(mirror.linear_vel) - glam::Vec3::from(gpu.linear_vel);
                    assert!(dv.length() < 1e-3, "fragment {b:?}: velocity");
                    // A fragment's solver frame is the identity (`threshold_frame_now`
                    // keeps the pool out of the fracture's frame records).
                    let read = w.read_body_state(*b).unwrap().transform;
                    assert_eq!(read.position, gpu.position, "fragment {b:?}: read pose");
                    let q = glam::Quat::from_array(mirror.orientation);
                    let qg = glam::Quat::from_array(gpu.orientation);
                    assert!(q.dot(qg).abs() > 0.999, "fragment {b:?}: {q:?} vs {qg:?}");
                    checked += 1;
                }
            }
            if checked > 0 {
                break;
            }
        }
        assert!(
            checked > 0,
            "the strike crushes the front block into fragments"
        );
    }
}
