//! The fracture lab's `stress` mode in `World` (D5, [`StrengthModel::Stress`]).
//!
//! The lab's stress mode is the blast_parity profile's stress world (stress-gpu
//! `GpuWorld`) coupled to the lab's engine (stress-rubble `RubbleEngine`). Here that
//! engine is *hosted* on the world's own sim: the world lends the engine its `WgpuSim`
//! and hull arena for each call, so the destructible's clusters, the lab's impactors and
//! every other body of the world are one simulation. Each world step is one stress-world
//! frame (`GpuWorld::step_frame`): resident frames step the sim with the contacts turned
//! into chunk loads on the GPU; frames with impact islands step it host-coupled, the
//! stress world simulating the island bodies itself and the engine holding them still.
//!
//! The engine makes and retires bodies as clusters split (on the GPU, or by the host
//! mirror); the world adopts each one it made as a [`BodyId`] (`Owner::Piece`), frees
//! each one it retired (the slot is reused after quarantine, so the engine never
//! compacts), and reports `BondsBroken` and `Split`. A piece's body origin is the
//! instance frame (the render contract); its solver frame is the engine's centre of mass
//! and principal axes. A split's children are new bodies: the parent's handle ends.
//!
//! Destruction is forward-only (AGENTS.md): no frame is redone (`redo_fractured_frames`
//! off) and the hosted engine refuses to save or restore; the correction pass stays the
//! measurement it is (`STRESS_CORRECTION_LIMIT`, off).
//!
//! Limits, each refused by name: one stress destructible per world (its frame steps the
//! whole world), no threshold destructibles beside it, canonical contact storage (the
//! ingest reads its rows), room in the world's capacity for the lab's
//! `capacity_for(scene)` headroom. The world's own bodies touching the structure load it
//! as static geometry; only impactors (`destructible_launch_impactor`, the lab's shots)
//! take part in impact islands.
//!
//! [`StrengthModel::Stress`]: crate::StrengthModel::Stress
//! [`BodyId`]: crate::BodyId

/// A body launched at a stress destructible (`World::destructible_launch_impactor`), as
/// the lab launches its shots: a sphere or a box of the lab's projectile material
/// (stress-ref `elastic_steel`), in world coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImpactorDef {
    pub shape: ImpactorShape,
    /// kg.
    pub mass: f32,
    pub position: [f32; 3],
    /// Quaternion `[x, y, z, w]`.
    pub rotation: [f32; 4],
    pub linear_velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
}

/// An impactor's shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ImpactorShape {
    Sphere { radius: f32 },
    Box { half_extents: [f32; 3] },
}

impl Default for ImpactorDef {
    fn default() -> Self {
        Self {
            shape: ImpactorShape::Sphere { radius: 0.25 },
            mass: 50.0,
            position: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
        }
    }
}

#[cfg(feature = "stress")]
pub(crate) use imp::*;

#[cfg(not(feature = "stress"))]
#[derive(Default)]
pub(crate) struct StressHost;

#[cfg(not(feature = "stress"))]
impl StressHost {
    pub fn active(&self) -> bool {
        false
    }
    pub fn note_collider(&mut self, _collider: u32) {}
}

#[cfg(feature = "stress")]
mod imp {
    use super::{ImpactorDef, ImpactorShape};
    use crate::defs::{BodyType, MassData};
    use crate::destruction::{ActorInfo, DestructibleDef, StrengthModel};
    use crate::events::DestructionEvent;
    use crate::math::SolverFrame;
    use crate::{BodyId, BodySlot, DestructibleAssetDesc, DestructibleId, Owner, Transform, World};
    use engine_schema::convex::HullArena;
    use glam::{DQuat, DVec3};
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::ops::Range;
    use std::rc::Rc;
    use std::sync::Arc;
    use stress_gpu::gpu::Gpu;
    use stress_gpu::world::GpuWorld;
    use stress_ref::capability::Profile;
    use stress_ref::engine::BodyKey;
    use stress_ref::scene::Support;
    use stress_rubble::{HostChange, RubbleEngine};

    /// The world's stress destructible, if any, and the colliders the world changed
    /// since the engine last saw them.
    #[derive(Default)]
    pub(crate) struct StressHost {
        pub state: Option<Box<StressState>>,
        touched: Vec<u32>,
    }

    impl StressHost {
        pub fn active(&self) -> bool {
            self.state.is_some()
        }

        /// The world wrote, retired or reused collider `collider` (the engine's foreign
        /// colliders follow at the next frame).
        pub fn note_collider(&mut self, collider: u32) {
            if self.state.is_some() {
                self.touched.push(collider);
            }
        }
    }

    /// A placed stress destructible: the lab's stress world and engine, and what the
    /// world knows of their bodies.
    pub(crate) struct StressState {
        gpu: Gpu,
        world: GpuWorld,
        engine: Rc<RefCell<RubbleEngine>>,
        pub destructible: u32,
        /// Per structure: its body frame in the instance frame (rotation, translation).
        structures: Vec<(DQuat, DVec3)>,
        /// Global chunk numbering: each structure's first (bonds number the same way).
        chunk_offset: Vec<u32>,
        /// Engine bodies by world slot: key, colliders, chunks (global).
        bodies: HashMap<u32, (BodyKey, Range<u32>, Vec<u32>)>,
        /// Per global chunk: its body slot, mass, whether it is supported.
        chunk_body: Vec<u32>,
        chunk_mass: Vec<f32>,
        chunk_support: Vec<bool>,
        bond_intact: Vec<bool>,
        /// Impactors launched, and the body slot each has once the engine made it.
        impactors: Vec<Option<(u32, u32)>>,
        pub nodes: Vec<Option<u32>>,
        /// Events by the step they happened in, for that step's frame.
        events: Vec<(u64, DestructionEvent)>,
    }

    /// The rotation from the stress frame (Z up) to the world's (Y up): -90 degrees
    /// about x (stress-rubble `to_rubble`).
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

    /// Mass data in the body frame from a solver frame and principal moments.
    fn mass_data(mass: f64, frame: SolverFrame, moments: glam::Vec3) -> MassData {
        let r = glam::Mat3::from_quat(frame.axes);
        let tensor = r * glam::Mat3::from_diagonal(moments) * r.transpose();
        MassData {
            mass: mass as f32,
            center: frame.center.to_array(),
            rotational_inertia: tensor.to_cols_array_2d(),
        }
    }

    impl World {
        /// Place a stress destructible (`StrengthModel::Stress`): the lab's stress
        /// world over the asset's stress scene, with its engine hosted on this world's
        /// sim.
        pub(crate) fn create_stress(
            &mut self,
            def: &DestructibleDef,
            asset: Arc<DestructibleAssetDesc>,
            index: usize,
        ) -> anyhow::Result<()> {
            let source = asset.stress.as_ref().ok_or_else(|| anyhow::anyhow!("StrengthModel::Stress needs an asset with the stress oracle's input (from_glb, cuboid_house or from_pack)"))?;
            anyhow::ensure!(
                self.contact_storage == crate::ContactStorage::Canonical,
                "StrengthModel::Stress reads rubble's canonical contact rows: create the world with ContactStorage::Canonical (it uses {:?})",
                self.contact_storage
            );
            anyhow::ensure!(
                !self.stress.active(),
                "a world holds one StrengthModel::Stress destructible (its stress world steps the whole world): destroy the other first"
            );
            anyhow::ensure!(
                !self.fracture.active(),
                "StrengthModel::Stress and StrengthModel::Threshold destructibles cannot share a world (each steps it its own way)"
            );
            let params = self.sim.params();
            let scene = source.stress_scene(
                def.strength_scale,
                params.dt,
                params.gravity_y,
                def.transform,
            )?;
            if let Some(why) = stress_gpu::world::unsupported_with(&scene, true) {
                anyhow::bail!("the stress world cannot run this scene: {why}");
            }
            // The lab's room for a scene (stress-rubble `capacity_for`): a split's children
            // take new slots while its parent's wait out their quarantine.
            let need = stress_rubble::capacity_for(&scene);
            let cap = self.sim.capacity();
            let (bodies_free, colliders_free) = (
                cap.max_bodies - self.sim.body_count(),
                cap.max_colliders - self.sim.collider_count(),
            );
            let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum();
            let parts: usize = scene
                .bodies
                .iter()
                .flat_map(|b| &b.chunks)
                .map(|c| c.compound.as_ref().map_or(1, |p| p.len().max(1)))
                .sum();
            let (want_bodies, want_colliders) = (3 * (chunks + 18), 3 * (parts + 18));
            anyhow::ensure!(
                bodies_free as usize >= want_bodies && colliders_free as usize >= want_colliders,
                "StrengthModel::Stress needs room for {want_bodies} bodies and {want_colliders} colliders (3x the {chunks} chunks and {parts} parts, as the lab sizes it); the world has {bodies_free} and {colliders_free} free (raise WorldCapacity::max_bodies / max_colliders)"
            );
            anyhow::ensure!(
                cap.max_contacts >= need.max_contacts.min(cap.max_colliders),
                "StrengthModel::Stress needs WorldCapacity::max_contacts >= {} (one per collider, as the lab sizes it)",
                need.max_contacts.min(cap.max_colliders)
            );
            let gpu = self.fracture_gpu();
            // Lend the sim to the hosted engine while the stress world is built (it makes
            // the clusters' bodies).
            let sim = self.sim.take();
            let arena = std::mem::replace(&mut self.hulls, HullArena::new());
            let engine = Rc::new(RefCell::new(RubbleEngine::hosted(sim, arena)));
            let (_, substeps) = Profile::BlastParity.engine_policy();
            engine.borrow_mut().set_policy(None, substeps);
            let mut built = GpuWorld::with_shared_resident_engine(&gpu, &scene, engine.clone())
                .map(|mut w| {
                    // The lab's stress mode (stress-web `load`): forward only, impact
                    // islands on, the profile's switches.
                    w.redo_fractured_frames = false;
                    w.set_impact_islands(true);
                    w.apply_profile(Profile::BlastParity);
                    w
                });
            // Numbering, chunk properties and bonds from the stress world's mirror (read
            // while the sim is lent: the view reads the engine's motion).
            let mirror = built.as_mut().ok().map(|world| {
                world.view_with(&gpu, |m, _| {
                    let (iq, ip) = (
                        def.transform.q().as_dquat().normalize(),
                        def.transform.p().as_dvec3(),
                    );
                    let r = to_y_up();
                    let mut frames: Vec<Option<(DQuat, DVec3)>> = vec![None; m.structures.len()];
                    for cl in &m.clusters {
                        // The cluster's body frame in the world, in the instance frame.
                        let (q, p) = (r * dq(cl.pose.rotation), r * dv(cl.pose.position));
                        frames[cl.structure].get_or_insert((
                            (iq.inverse() * q).normalize(),
                            iq.inverse() * (p - ip),
                        ));
                    }
                    let mut chunk_offset = Vec::new();
                    let mut nc = 0u32;
                    let mut mass = Vec::new();
                    let mut support = Vec::new();
                    let mut intact = Vec::new();
                    for (s, st) in m.structures.iter().enumerate() {
                        chunk_offset.push(nc);
                        nc += st.chunks.len() as u32;
                        mass.extend(st.chunks.iter().map(|c| c.mass as f32));
                        support.extend(st.chunks.iter().map(|c| c.support != Support::None));
                        intact.extend(m.bonds[s].iter().map(|b| b.connected()));
                    }
                    let frames = frames
                        .into_iter()
                        .map(|f| f.unwrap_or((DQuat::IDENTITY, DVec3::ZERO)))
                        .collect::<Vec<_>>();
                    (frames, chunk_offset, mass, support, intact)
                })
            });
            let log = engine.borrow_mut().take_host_log();
            let (sim, arena) = engine.borrow_mut().take_back();
            self.sim.put(sim);
            self.hulls = arena;
            let world = match built {
                Ok(w) => w,
                Err(e) => {
                    // Whatever the engine made before it failed goes back to the world.
                    self.release_engine_bodies(&log);
                    anyhow::bail!("stress world: {e}");
                }
            };
            let (structures, chunk_offset, chunk_mass, chunk_support, bond_intact) =
                mirror.expect("built");
            let chunk_count = chunk_mass.len();
            let mut st = Box::new(StressState {
                gpu,
                world,
                engine,
                destructible: index as u32,
                structures,
                chunk_offset,
                bodies: HashMap::new(),
                chunk_body: vec![u32::MAX; chunk_count],
                chunk_mass,
                chunk_support,
                bond_intact,
                impactors: Vec::new(),
                nodes: source.nodes.clone(),
                events: Vec::new(),
            });
            self.adopt_log(&mut st, &log, def.user_data);
            let d = &mut self.destructibles[index];
            d.model = StrengthModel::Stress;
            d.threshold = None;
            d.bond_joints = vec![None; st.bond_intact.len()];
            d.bond_intact = st.bond_intact.clone();
            self.stress.state = Some(st);
            self.stress_sync_destructible();
            Ok(())
        }

        /// Run `f` with the engine's sim lent: the colliders the world touched handed
        /// over first, the bodies the engine made and retired adopted after.
        fn stress_call<R>(&mut self, f: impl FnOnce(&mut StressState) -> R) -> R {
            let mut st = self.stress.state.take().expect("a stress destructible");
            let touched = std::mem::take(&mut self.stress.touched);
            {
                let sim = self.sim.take();
                let arena = std::mem::replace(&mut self.hulls, HullArena::new());
                let mut e = st.engine.borrow_mut();
                e.lend(sim, arena);
                e.host_colliders(&touched);
            }
            let r = f(&mut st);
            let log = st.engine.borrow_mut().take_host_log();
            let (sim, arena) = st.engine.borrow_mut().take_back();
            self.sim.put(sim);
            self.hulls = arena;
            let user_data = self.destructibles[st.destructible as usize].user_data;
            self.adopt_log(&mut st, &log, user_data);
            self.stress.state = Some(st);
            self.stress_sync_destructible();
            r
        }

        /// One stress-world frame in place of the step's own (`World::step`): the
        /// engine stepped on the lent sim, the bodies it made and retired adopted, the
        /// bonds that broke reported.
        pub(crate) fn stress_frame(&mut self) -> anyhow::Result<()> {
            let params = self.sim.params();
            let ran = self.stress_call(|st| {
                let gpu = st.gpu.clone();
                st.world.step_frame(&gpu).map(|()| {
                    // The frame's splits followed and the bonds synced (lossless: the
                    // view's read, which the lab takes every displayed frame).
                    st.world.view_with(&gpu, |m, _| {
                        m.bonds
                            .iter()
                            .flat_map(|b| b.iter().map(|b| b.connected()))
                            .collect::<Vec<bool>>()
                    })
                })
            });
            // The engine sets the frame's (sub)step and contact range; the world's own
            // settings stay as they were.
            let mut now = self.sim.params();
            now.dt = params.dt;
            now.contact_offset = params.contact_offset;
            self.sim.set_params(now);
            let intact = ran.map_err(|e| anyhow::anyhow!("stress frame {}: {e}", self.step))?;
            let step = self.step;
            let st = self.stress.state.as_mut().expect("a stress destructible");
            let broken: Vec<u32> = (0..st.bond_intact.len().min(intact.len()))
                .filter(|&b| st.bond_intact[b] && !intact[b])
                .map(|b| b as u32)
                .collect();
            if broken.is_empty() {
                return Ok(());
            }
            for &b in &broken {
                st.bond_intact[b as usize] = false;
            }
            let d = st.destructible as usize;
            let id = DestructibleId::new(d as u32, self.destructibles[d].generation);
            // Bonds before the splits they caused, as the threshold model reports.
            let at = st
                .events
                .iter()
                .position(|(s, _)| *s == step)
                .unwrap_or(st.events.len());
            st.events.insert(
                at,
                (
                    step,
                    DestructionEvent::BondsBroken {
                        destructible: id,
                        bonds: broken.clone(),
                    },
                ),
            );
            for &b in &broken {
                self.destructibles[d].bond_intact[b as usize] = false;
            }
            Ok(())
        }

        /// Adopt the bodies the engine made, free those it retired, and log the splits.
        fn adopt_log(&mut self, st: &mut StressState, log: &[HostChange], user_data: u64) {
            let step = self.step;
            let did = st.destructible;
            let id = DestructibleId::new(did, self.destructibles[did as usize].generation);
            // Retired bodies' chunks (global) and handles, and the bodies made, in order.
            let mut parents: Vec<(BodyId, Vec<u32>)> = Vec::new();
            let mut made: Vec<u32> = Vec::new();
            for change in log {
                match change {
                    HostChange::Retired {
                        slot, colliders, ..
                    } => {
                        let generation = self.bodies[*slot as usize].generation;
                        let chunks = st
                            .bodies
                            .remove(slot)
                            .map(|(_, _, chunks)| chunks)
                            .unwrap_or_default();
                        for &c in &chunks {
                            if st.chunk_body[c as usize] == *slot {
                                st.chunk_body[c as usize] = u32::MAX;
                            }
                        }
                        made.retain(|m| m != slot);
                        self.free_engine_body(*slot, colliders.clone());
                        if !chunks.is_empty() {
                            parents.push((BodyId::new(*slot, generation), chunks));
                        }
                    }
                    HostChange::Created {
                        key,
                        slot,
                        colliders,
                        chunks,
                    } => {
                        assert_eq!(
                            *slot as usize,
                            self.bodies.len(),
                            "the engine appends its bodies"
                        );
                        let global: Vec<u32> = chunks
                            .iter()
                            .flatten()
                            .map(|&(s, c)| st.chunk_offset[s] + c as u32)
                            .collect();
                        let body = st
                            .engine
                            .borrow()
                            .host_body(*key)
                            .filter(|b| b.slot == *slot);
                        let slot_state = match body {
                            Some(b) => {
                                // The body frame in the instance frame: a structure's
                                // clusters share their structure's; an impactor's is its own.
                                let (sq, sp) = chunks
                                    .iter()
                                    .flatten()
                                    .next()
                                    .map_or((DQuat::IDENTITY, DVec3::ZERO), |&(s, _)| {
                                        st.structures[s]
                                    });
                                let frame = SolverFrame {
                                    center: (sp + sq * dv(b.com)).as_vec3(),
                                    axes: (sq * dq(b.principal)).normalize().as_quat(),
                                };
                                let dynamic = !b.fixed && b.inv_mass > 0.0;
                                let inv = |x: f32| if x > 0.0 { 1.0 / x } else { 0.0 };
                                let moments = glam::Vec3::from(b.inv_inertia.map(inv));
                                let mass = dynamic
                                    .then(|| mass_data(inv(b.inv_mass) as f64, frame, moments));
                                self.piece_slot(
                                    if dynamic {
                                        BodyType::Dynamic
                                    } else {
                                        BodyType::Static
                                    },
                                    mass,
                                    moments,
                                    frame,
                                    user_data,
                                )
                            }
                            None => {
                                // Made and retired again within the call.
                                let mut s = self.piece_slot(
                                    BodyType::Static,
                                    None,
                                    glam::Vec3::ONE,
                                    SolverFrame::IDENTITY,
                                    user_data,
                                );
                                s.alive = false;
                                s
                            }
                        };
                        let alive = slot_state.alive;
                        self.bodies.push(slot_state);
                        let cols = self.sim.collider_count() as usize;
                        if self.collider_shape.len() < cols {
                            self.collider_shape.resize(cols, u32::MAX);
                        }
                        if !alive {
                            continue;
                        }
                        if let BodyKey::Impactor(i) = key {
                            if let Some(e) = st.impactors.get_mut(*i) {
                                *e = Some((*slot, self.bodies[*slot as usize].generation));
                            }
                        }
                        for &c in &global {
                            st.chunk_body[c as usize] = *slot;
                        }
                        st.bodies.insert(*slot, (*key, colliders.clone(), global));
                        made.push(*slot);
                    }
                }
            }
            debug_assert_eq!(self.bodies.len(), self.sim.body_count() as usize);
            // Splits: each retired cluster's chunks, grouped by the bodies they are on now.
            for (parent, chunks) in parents {
                let mut groups: Vec<u32> = chunks
                    .iter()
                    .map(|&c| st.chunk_body[c as usize])
                    .filter(|&b| b != u32::MAX)
                    .collect();
                groups.sort_unstable();
                groups.dedup();
                if groups.is_empty() {
                    continue;
                }
                let info = |chunks: Vec<u32>, body: BodyId| ActorInfo {
                    anchored: chunks.iter().any(|&c| st.chunk_support[c as usize]),
                    mass: chunks.iter().map(|&c| st.chunk_mass[c as usize]).sum(),
                    chunks,
                    bodies: vec![body],
                };
                let mut sorted = chunks.clone();
                sorted.sort_unstable();
                let children = groups
                    .iter()
                    .map(|&g| {
                        let mut mine: Vec<u32> = sorted
                            .iter()
                            .copied()
                            .filter(|&c| st.chunk_body[c as usize] == g)
                            .collect();
                        mine.sort_unstable();
                        info(mine, BodyId::new(g, self.bodies[g as usize].generation))
                    })
                    .collect();
                st.events.push((
                    step,
                    DestructionEvent::Split {
                        destructible: id,
                        parent: info(sorted, parent),
                        children,
                    },
                ));
            }
        }

        /// A world slot for a body the engine made at the next slot.
        fn piece_slot(
            &self,
            body_type: BodyType,
            mass: Option<MassData>,
            principal: glam::Vec3,
            frame: SolverFrame,
            user_data: u64,
        ) -> BodySlot {
            BodySlot {
                generation: 1,
                alive: true,
                owner: Owner::Piece,
                body_type,
                user_data,
                name: None,
                shapes: Vec::new(),
                joints: Vec::new(),
                removed_at: None,
                mass,
                principal,
                frames: vec![(self.step, frame)],
                written_step: self.step,
                origin: Transform::IDENTITY,
                velocity: ([0.0; 3], [0.0; 3]),
                asleep: false,
                last_transform: None,
                last_asleep: false,
                last_solver: None,
                damping: (0.0, 0.0),
                gravity_scale: 1.0,
                locks: 0,
                bullet: false,
            }
        }

        /// A body slot the engine retired (static, its colliders in no pair): free for
        /// the world's reuse after quarantine.
        fn free_engine_body(&mut self, slot: u32, colliders: Range<u32>) {
            let step = self.step;
            let b = &mut self.bodies[slot as usize];
            b.alive = false;
            b.owner = Owner::User;
            b.generation = b.generation.wrapping_add(1).max(1);
            b.removed_at = Some(step);
            b.mass = None;
            b.shapes.clear();
            self.free_bodies.push(slot);
            for c in colliders {
                self.free_colliders.push((c, step));
            }
        }

        /// Undo an engine's log after a failed build: its bodies parked and freed.
        fn release_engine_bodies(&mut self, log: &[HostChange]) {
            for change in log {
                if let HostChange::Created {
                    slot, colliders, ..
                } = change
                {
                    if *slot as usize == self.bodies.len() {
                        let s = self.piece_slot(
                            BodyType::Static,
                            None,
                            glam::Vec3::ONE,
                            SolverFrame::IDENTITY,
                            0,
                        );
                        self.bodies.push(s);
                    }
                    self.park_engine_body(*slot, colliders.clone());
                }
            }
            let cols = self.sim.collider_count() as usize;
            if self.collider_shape.len() < cols {
                self.collider_shape.resize(cols, u32::MAX);
            }
        }

        /// Retire an engine body the world takes back: static in the graveyard, its
        /// colliders in no pair, the slot freed.
        fn park_engine_body(&mut self, slot: u32, colliders: Range<u32>) {
            let mut parked = engine_schema::BodyState::static_plane_support();
            parked.position = crate::graveyard(slot);
            parked.initial_pos = parked.position;
            self.sim.write_bodies(slot, &[parked]);
            self.sim.forget_pose(slot);
            for c in colliders.clone() {
                self.sim.set_collider(c, crate::retired_collider(slot));
            }
            if self.bodies[slot as usize].alive {
                self.free_engine_body(slot, colliders);
            }
        }

        /// The destructible's chunk bodies and actor labels as the engine has them now.
        fn stress_sync_destructible(&mut self) {
            let Some(st) = self.stress.state.as_ref() else {
                return;
            };
            let d = st.destructible as usize;
            let bodies: Vec<BodyId> = st
                .chunk_body
                .iter()
                .map(|&b| {
                    if b == u32::MAX {
                        BodyId::NULL
                    } else {
                        BodyId::new(b, self.bodies[b as usize].generation)
                    }
                })
                .collect();
            let labels = crate::threshold::actor_labels(&st.chunk_body);
            let slot = &mut self.destructibles[d];
            slot.chunk_bodies = bodies;
            slot.actor = labels;
        }

        /// This step's destruction events of the stress destructible.
        pub(crate) fn stress_events(&mut self, step: u64) -> Vec<DestructionEvent> {
            let Some(st) = self.stress.state.as_mut() else {
                return Vec::new();
            };
            let mut out = Vec::new();
            st.events.retain(|(s, e)| {
                if *s == step {
                    out.push(e.clone());
                    false
                } else {
                    *s > step
                }
            });
            out
        }

        /// The actor a stress destructible's chunk is part of: every chunk on its body.
        pub(crate) fn stress_actor(&self, chunk: u32) -> anyhow::Result<ActorInfo> {
            let st = self.stress.state.as_ref().expect("a stress destructible");
            let b = *st
                .chunk_body
                .get(chunk as usize)
                .ok_or_else(|| anyhow::anyhow!("chunk {chunk} out of range"))?;
            let chunks: Vec<u32> = (0..st.chunk_body.len() as u32)
                .filter(|&c| st.chunk_body[c as usize] == b)
                .collect();
            Ok(ActorInfo {
                anchored: chunks.iter().any(|&c| st.chunk_support[c as usize]),
                mass: chunks.iter().map(|&c| st.chunk_mass[c as usize]).sum(),
                bodies: vec![BodyId::new(b, self.bodies[b as usize].generation)],
                chunks,
            })
        }

        pub(crate) fn stress_chunk_node(&self, chunk: u32) -> anyhow::Result<Option<u32>> {
            let st = self.stress.state.as_ref().expect("a stress destructible");
            anyhow::ensure!(
                (chunk as usize) < st.chunk_body.len(),
                "chunk {chunk} out of range"
            );
            Ok(st.nodes.get(chunk as usize).copied().flatten())
        }

        /// Remove the stress destructible: its stress world, its engine and every body
        /// the engine made.
        pub(crate) fn destroy_stress(&mut self) -> anyhow::Result<()> {
            let st = self.stress.state.take().expect("a stress destructible");
            self.stress.touched.clear();
            let mut slots: Vec<(u32, Range<u32>)> = st
                .bodies
                .iter()
                .map(|(s, (_, c, _))| (*s, c.clone()))
                .collect();
            slots.sort_by_key(|s| s.0);
            drop(st);
            for (slot, colliders) in slots {
                self.park_engine_body(slot, colliders);
            }
            Ok(())
        }

        fn live_stress(&self, id: DestructibleId) -> anyhow::Result<&StressState> {
            let i = self.live_destructible(id)?;
            self.stress
                .state
                .as_deref()
                .filter(|s| s.destructible as usize == i)
                .ok_or_else(|| {
                    anyhow::anyhow!("destructible {i} is not a StrengthModel::Stress one")
                })
        }

        /// Launch an impactor at a stress destructible, as the lab launches its shots
        /// (the lab's projectile material). It takes part in the next step; its body
        /// exists by the end of that step at the latest (`destructible_impactor_body`).
        /// Returns its index.
        pub fn destructible_launch_impactor(
            &mut self,
            id: DestructibleId,
            def: &ImpactorDef,
        ) -> anyhow::Result<u32> {
            self.live_stress(id)?;
            anyhow::ensure!(
                def.mass.is_finite() && def.mass > 0.0,
                "an impactor's mass must be positive"
            );
            let shape = match def.shape {
                ImpactorShape::Sphere { radius } => {
                    anyhow::ensure!(radius > 0.0, "an impactor's radius must be positive");
                    stress_ref::scene::ImpactorShape::Sphere {
                        radius: radius as f64,
                    }
                }
                ImpactorShape::Box { half_extents } => {
                    anyhow::ensure!(
                        half_extents.iter().all(|&h| h > 0.0),
                        "an impactor's half extents must be positive"
                    );
                    stress_ref::scene::ImpactorShape::Box {
                        half_extents: half_extents.map(|h| h as f64),
                    }
                }
            };
            // World (Y up) to the stress frame (Z up): exact for vectors (stress-rubble
            // `from_rubble`), the same turn for the orientation.
            let v = |a: [f32; 3]| [a[0] as f64, -a[2] as f64, a[1] as f64];
            let q = (to_y_up().inverse()
                * glam::Quat::from_array(def.rotation).normalize().as_dquat())
            .normalize();
            let st = self.stress.state.as_ref().expect("checked");
            let k = st.impactors.len();
            let desc = stress_ref::scene::ImpactorDesc {
                name: format!("impactor{k}"),
                shape,
                mass: def.mass as f64,
                position: v(def.position),
                orientation: [q.w, q.x, q.y, q.z],
                velocity: v(def.linear_velocity),
                angular_velocity: v(def.angular_velocity),
                material: "projectile".into(),
                crush: None,
            };
            let imp = stress_ref::world::Impactor::from_desc(&desc, &st.world.scene);
            // The stress world's tables may refresh (from the engine's).
            let i = self
                .stress_call(|st| {
                    let gpu = st.gpu.clone();
                    let i = st.world.add_impactor(&gpu, imp)?;
                    st.impactors.push(None);
                    Ok::<_, String>(i)
                })
                .map_err(|e| anyhow::anyhow!("launch: {e}"))?;
            debug_assert_eq!(i, k);
            self.dirty = true;
            Ok(k as u32)
        }

        /// The body of a launched impactor, once the engine has made it (the step after
        /// its launch).
        pub fn destructible_impactor_body(
            &self,
            id: DestructibleId,
            index: u32,
        ) -> anyhow::Result<Option<BodyId>> {
            let st = self.live_stress(id)?;
            let e = st
                .impactors
                .get(index as usize)
                .ok_or_else(|| anyhow::anyhow!("impactor {index} was not launched"))?;
            Ok(e.and_then(|(slot, generation)| {
                let id = BodyId::new(slot, generation);
                self.body_is_valid(id).then_some(id)
            }))
        }
    }
}
