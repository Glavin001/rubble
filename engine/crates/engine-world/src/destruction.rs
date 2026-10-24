//! Destructibles: an immutable asset (chunks, bonds) and placed instances.
//!
//! The asset is built from a [`DestructibleAssetDesc`], after Blast's
//! `NvBlastAssetDesc`; loaders turn every source into one (Shardwright GLB,
//! vibe-city / blast-stress packs, procedural cuboid houses, any
//! `engine_fracture::Scene`), so no input is special.
//!
//! **Render contract.** Every piece body uses the instance's frame as its
//! body origin; its centre of mass and principal axes are its solver frame.
//! A chunk's world transform is therefore `body_transform * chunk_local`
//! with `chunk_local` fixed per chunk (`World::destructible_chunk_local`), so
//! a renderer never rebuilds meshes when pieces split.
//!
//! Strength models: `Bonds` (breakable AVBD bond joints; every chunk is a
//! body; the fracture lab's model) ships now. `Threshold` (the GPU-resident
//! splitter, `GpuFracture`) is the next step and is refused until then.

use crate::defs::{BodyDef, BodyType, Geometry, MassData, ShapeDef};
use crate::error::unsupported;
use crate::events::DestructionEvent;
use crate::math::SolverFrame;
use crate::{BodyId, JointId, Transform, World};
use engine_fracture::Scene;
use engine_schema::convex::{convex_params_base, ConvexHull};
use engine_schema::{BondSpec, JointDesc, ShapeType};
use glam::{Mat3, Quat, Vec3};
use std::collections::HashMap;
use std::sync::Arc;

/// One chunk of a destructible asset.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkDesc {
    /// The chunk's principal frame (centre of mass and principal axes) in the
    /// asset frame.
    pub pose: Transform,
    /// kg; 0 makes the chunk static (part of the world).
    pub mass: f32,
    /// Principal moments (kg m²) about `pose`.
    pub inertia: [f32; 3],
    /// Collision geometry in the chunk's principal frame.
    pub shapes: Vec<Geometry>,
    /// The source fragment id (Shardwright), if any.
    pub fragment: Option<u32>,
    /// The glTF node that renders this chunk, if any.
    pub node: Option<u32>,
}

/// One bond between two chunks, or between a chunk and the world.
#[derive(Clone, Debug, PartialEq)]
pub struct BondDesc {
    pub a: u32,
    /// `None`: bonded to the world (an anchor).
    pub b: Option<u32>,
    /// Bond point in A's principal frame.
    pub anchor_a: [f32; 3],
    /// Bond point in B's principal frame (in the asset frame for world bonds).
    pub anchor_b: [f32; 3],
    /// Rest relative rotation `conj(qA) * qB`.
    pub rest: [f32; 4],
    /// Bond normal (A to B) and in-plane axis, in A's principal frame.
    pub normal: [f32; 3],
    pub frame_u: [f32; 3],
    /// Tension N, compression N, shear N, bending N m (<= 0: never breaks).
    pub strength: [f32; 4],
    /// Penalties: normal, u, v, about u, about v, about normal.
    pub stiffness: [f32; 6],
}

/// A destructible asset (Blast `NvBlastAssetDesc`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DestructibleAssetDesc {
    pub name: Option<String>,
    pub chunks: Vec<ChunkDesc>,
    pub bonds: Vec<BondDesc>,
    /// Settled under its own weight ([`World::preload_destructible_asset`]):
    /// instances start asleep at these poses.
    pub preload: Option<Preload>,
    /// The solver settings the source was tuned with (the fracture lab's:
    /// 120 Hz, 8 iterations, its contact and sleep flags). A world built with
    /// them (`WorldDef::solver`, and `gravity` from `gravity_y`) behaves as
    /// the lab does; the asset works in any world.
    pub solver: Option<engine_schema::AvbdParams>,
}

/// The fracture lab's scene parameters as world solver settings.
pub fn lab_solver(p: &engine_fracture::scene::SceneParams) -> engine_schema::AvbdParams {
    engine_schema::AvbdParams {
        dt: p.dt,
        gravity_y: p.gravity_y,
        iterations: p.iterations,
        flags: p.flags,
        alpha: p.alpha,
        gamma: p.gamma,
        beta_lin: p.beta_lin,
        beta_ang: p.beta_ang,
        k_start: p.k_start,
        k_max_n: p.k_max_n,
        k_max_t: p.k_max_t,
        contact_offset: p.contact_offset,
        sleep_lin: p.sleep_lin,
        sleep_ang: p.sleep_ang,
        sleep_time: p.sleep_time,
        residual_exit: p.residual_exit,
        trust_lin: p.trust_lin,
        trust_ang: p.trust_ang,
        pivot_floor: p.pivot_floor,
        grid_cell_size: p.grid_cell_size,
        ..engine_schema::AvbdParams::default()
    }
}

/// The fracture lab's bond preloading. Elastic bonds carry load only once
/// they deform, so a structure released at its baked rest pose drops onto
/// its springs and the transient overloads bonds that would hold it at rest.
/// A preloaded asset was settled once under gravity with breaking off; every
/// instance then starts asleep at the settled, statically loaded poses.
#[derive(Clone, Debug, PartialEq)]
pub struct Preload {
    /// Each chunk body's origin pose in the asset frame.
    pub body_poses: Vec<Transform>,
    pub steps: u32,
    /// It fell asleep (else it gave up after 600 steps).
    pub settled: bool,
}

impl DestructibleAssetDesc {
    /// One hierarchy level of a Shardwright fracture GLB.
    pub fn from_glb(bytes: &[u8], level: u8) -> anyhow::Result<Self> {
        let asset = engine_fracture::parse_fracture_glb(bytes)?;
        let scene = engine_fracture::build_scene(&asset, level, None)?;
        let mut desc = Self::from_scene(&scene)?;
        let nodes: HashMap<u32, (u32, Option<u32>)> = scene
            .body_of
            .iter()
            .map(|&(fragment, body)| {
                let node = asset
                    .physics
                    .fragments
                    .get(fragment as usize)
                    .and_then(|f| u32::try_from(f.node).ok());
                (body, (fragment, node))
            })
            .collect();
        // Chunks follow scene bodies in order, skipping the ground (body 0)
        // and the parked projectiles.
        let shots = scene.projectile_start..scene.projectile_start + scene.projectile_count;
        let bodies = (1..scene.bodies.len() as u32).filter(|b| !shots.contains(b));
        for (chunk, body) in desc.chunks.iter_mut().zip(bodies) {
            if let Some(&(fragment, node)) = nodes.get(&body) {
                chunk.fragment = Some(fragment);
                chunk.node = node;
            }
        }
        Ok(desc)
    }

    /// A vibe-city / blast-stress-solver JSON pack.
    pub fn from_pack(bytes: &[u8]) -> anyhow::Result<Self> {
        let (scene, info) = engine_fracture::pack::build_pack_scene(bytes)?;
        let mut desc = Self::from_scene(&scene)?;
        desc.name = Some(info.title);
        Ok(desc)
    }

    /// A procedural house of boxes (the fracture lab's cuboid scenes).
    pub fn cuboid_house(options: &engine_fracture::cuboid::CuboidOptions) -> anyhow::Result<Self> {
        Self::from_scene(&engine_fracture::cuboid::build_cuboid_house(options)?)
    }

    /// Any fracture-lab scene (`engine_fracture::Scene`): body 0 is its
    /// ground, the rest become chunks; joints become bonds.
    pub fn from_scene(scene: &Scene) -> anyhow::Result<Self> {
        let f3 = |v: [f64; 3]| [v[0] as f32, v[1] as f32, v[2] as f32];
        let f4 = |v: [f64; 4]| [v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32];
        // Scene body -> chunk: body 0 is the ground; the lab's parked
        // projectiles are not part of the structure.
        let shots = scene.projectile_start..scene.projectile_start + scene.projectile_count;
        let mut chunk_of = vec![None; scene.bodies.len()];
        let mut next = 0u32;
        for (i, slot) in chunk_of.iter_mut().enumerate().skip(1) {
            if !shots.contains(&(i as u32)) {
                *slot = Some(next);
                next += 1;
            }
        }
        let mut chunks: Vec<ChunkDesc> = scene
            .bodies
            .iter()
            .enumerate()
            .filter(|(i, _)| chunk_of[*i].is_some())
            .map(|(_, b)| {
                let inv = |x: f64| if x > 0.0 { (1.0 / x) as f32 } else { 0.0 };
                ChunkDesc {
                    pose: Transform::new(f3(b.p), f4(b.q)),
                    mass: inv(b.inv_mass),
                    inertia: [inv(b.inv_i[0]), inv(b.inv_i[1]), inv(b.inv_i[2])],
                    shapes: Vec::new(),
                    fragment: None,
                    node: None,
                }
            })
            .collect();
        for c in &scene.colliders {
            let Some(Some(k)) = chunk_of.get(c.body as usize) else {
                continue;
            };
            let chunk = &mut chunks[*k as usize];
            let local = Transform::new(f3(c.local_pos), f4(c.local_rot));
            let p = c.params;
            let geometry = match c.shape {
                s if s == ShapeType::Box as u32 => Geometry::Box {
                    half: [p[0] as f32, p[1] as f32, p[2] as f32],
                    transform: local,
                },
                s if s == ShapeType::Sphere as u32 => Geometry::Sphere {
                    center: local.position,
                    radius: p[0] as f32,
                },
                s if s == ShapeType::Capsule as u32 => Geometry::Capsule {
                    center: local.position,
                    radius: p[0] as f32,
                    half_len: p[1] as f32,
                },
                s if s == ShapeType::ConvexHull as u32 => {
                    let params = [p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32];
                    let base = convex_params_base(params);
                    let mut hull = ConvexHull::unpack(&scene.hull_data, base)
                        .map_err(|e| anyhow::anyhow!("hull at {base}: {e:?}"))?;
                    let scale = params[1];
                    if scale != 1.0 {
                        let pts: Vec<Vec3> = hull.vertices.iter().map(|v| *v * scale).collect();
                        hull =
                            ConvexHull::from_points(&pts).map_err(|e| anyhow::anyhow!("{e:?}"))?;
                    }
                    Geometry::Hull {
                        hull,
                        transform: local,
                    }
                }
                other => anyhow::bail!("unsupported chunk shape type {other}"),
            };
            chunk.shapes.push(geometry);
        }
        let bonds = scene
            .joints
            .iter()
            .map(|j| {
                let chunk = |body: u32| chunk_of.get(body as usize).copied().flatten();
                let a =
                    chunk(j.body_a).ok_or_else(|| anyhow::anyhow!("bond on body {}", j.body_a))?;
                let b = if j.world || j.body_b == 0 {
                    None
                } else {
                    Some(
                        chunk(j.body_b)
                            .ok_or_else(|| anyhow::anyhow!("bond on body {}", j.body_b))?,
                    )
                };
                Ok(BondDesc {
                    a,
                    b,
                    anchor_a: f3(j.anchor_a),
                    anchor_b: f3(j.anchor_b),
                    rest: f4(j.rest),
                    normal: f3(j.normal),
                    frame_u: f3(j.frame_u),
                    strength: f4(j.strength),
                    stiffness: j.k.map(|x| x as f32),
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let desc = Self {
            name: None,
            chunks,
            bonds,
            preload: None,
            solver: Some(lab_solver(&scene.params)),
        };
        desc.validate()?;
        Ok(desc)
    }

    /// Indices in range, shapes present, masses sane. Invalid assets are
    /// refused, never repaired.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(!self.chunks.is_empty(), "a destructible asset needs chunks");
        for (k, c) in self.chunks.iter().enumerate() {
            anyhow::ensure!(!c.shapes.is_empty(), "chunk {k} has no shapes");
            anyhow::ensure!(
                c.mass >= 0.0 && c.mass.is_finite(),
                "chunk {k} mass {}",
                c.mass
            );
            anyhow::ensure!(
                c.mass == 0.0 || c.inertia.iter().all(|&i| i > 0.0 && i.is_finite()),
                "chunk {k} inertia {:?}",
                c.inertia
            );
        }
        let n = self.chunks.len() as u32;
        for (k, b) in self.bonds.iter().enumerate() {
            anyhow::ensure!(
                b.a < n && b.b.is_none_or(|x| x < n),
                "bond {k} ends out of range"
            );
            anyhow::ensure!(Some(b.a) != b.b, "bond {k} joins a chunk to itself");
        }
        Ok(())
    }
}

/// An asset in a world (`World::create_destructible_asset`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DestructibleAssetId(pub u32);

/// How a destructible's bonds fail.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StrengthModel {
    /// Breakable AVBD bond joints in the rigid solve (the fracture lab's
    /// model): every chunk is a body; bonds break by their tension,
    /// compression, shear and bending capacities.
    #[default]
    Bonds,
    /// The GPU-resident threshold splitter (`GpuFracture`): bonded clusters
    /// are single bodies that split on the GPU. Not yet in `World`.
    Threshold,
}

/// How a destructible is anchored to the world.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Anchors {
    /// The asset's world bonds.
    #[default]
    AssetWorldBonds,
    /// None: the whole structure is free.
    None,
}

/// A placed destructible (`World::create_destructible`).
#[derive(Clone, Debug, PartialEq)]
pub struct DestructibleDef {
    pub asset: DestructibleAssetId,
    /// Where the asset frame goes in the world.
    pub transform: Transform,
    pub model: StrengthModel,
    /// Multiplies every bond capacity.
    pub strength_scale: f32,
    pub anchors: Anchors,
    /// Start asleep (woken by contact), as the lab does.
    pub asleep: bool,
    pub shape: ShapeDef,
    pub user_data: u64,
}

impl DestructibleDef {
    pub fn new(asset: DestructibleAssetId) -> Self {
        Self {
            asset,
            transform: Transform::IDENTITY,
            model: StrengthModel::Bonds,
            strength_scale: 1.0,
            anchors: Anchors::AssetWorldBonds,
            asleep: true,
            shape: ShapeDef {
                update_body_mass: false,
                ..ShapeDef::default()
            },
            user_data: 0,
        }
    }
}

crate::handles::handle!(
    /// A placed destructible.
    DestructibleId
);

/// A connected group of chunks: one rigid actor in Blast's terms.
#[derive(Clone, Debug, PartialEq)]
pub struct ActorInfo {
    /// Chunk indices, ascending.
    pub chunks: Vec<u32>,
    /// The bodies those chunks are (one per chunk in the `Bonds` model).
    pub bodies: Vec<BodyId>,
    /// Still bonded to the world.
    pub anchored: bool,
    pub mass: f32,
}

/// Counts for one destructible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DestructibleStats {
    pub chunks: u32,
    pub bonds: u32,
    pub bonds_intact: u32,
    pub actors: u32,
}

pub(crate) struct DestructibleSlot {
    pub generation: u32,
    pub alive: bool,
    pub asset: u32,
    pub user_data: u64,
    pub chunk_bodies: Vec<BodyId>,
    /// Static body world bonds attach to.
    pub anchor_body: Option<BodyId>,
    pub bond_joints: Vec<Option<JointId>>,
    pub bond_intact: Vec<bool>,
    /// Chunk -> actor label (the lowest chunk index in its component).
    pub actor: Vec<u32>,
}

impl World {
    /// The fracture lab's bond preloading for an asset: settle one copy on a
    /// ground plane (the asset frame's y = 0, as the lab's scenes stand) with
    /// this world's solver settings and breaking off, until it sleeps (checked
    /// every 60 steps, at most 600). Instances placed afterwards start asleep at
    /// the settled poses. Runs in a helper world on this world's device and
    /// blocks until done. Returns whether it settled.
    pub fn preload_destructible_asset(&mut self, id: DestructibleAssetId) -> anyhow::Result<bool> {
        let mut desc = self.destructible_asset(id)?.clone();
        if desc.bonds.is_empty() {
            return Ok(true);
        }
        desc.preload = None;
        let shapes: usize = desc.chunks.iter().map(|c| c.shapes.len()).sum();
        let hull_vertices: usize = desc
            .chunks
            .iter()
            .flat_map(|c| &c.shapes)
            .map(|g| match g {
                Geometry::Hull { hull, .. } => hull.vertices.len(),
                _ => 0,
            })
            .sum();
        let n = desc.chunks.len() as u32;
        let colliders = shapes as u32 + 8;
        let contacts = (colliders * 64).max(32_768);
        let def = crate::WorldDef {
            capacity: engine_schema::WorldCapacity {
                max_bodies: n + 8,
                max_colliders: colliders,
                max_contacts: contacts,
                max_joints: desc.bonds.len() as u32 + 8,
                max_candidates: contacts * 2,
                max_colors: 32,
                max_commands: 64,
                max_hull_vertices: (hull_vertices as u32 + 64).max(1024),
            },
            ..crate::WorldDef::default()
        };
        let gpu = crate::GpuContext::Shared {
            info: self.adapter.clone(),
            device: self.sim.device().clone(),
            queue: self.sim.queue().clone(),
        };
        let mut helper = World::new(gpu, &def)?;
        let mut params = self.sim.params();
        params.flags |= engine_fracture::scene::FLAG_BOND_PRELOAD;
        helper.sim.set_params(params);
        let ground = helper.create_body(&BodyDef::default())?;
        helper.create_shape(
            ground,
            &ShapeDef::default(),
            Geometry::Plane {
                normal: [0.0, 1.0, 0.0],
                offset: 0.0,
            },
        )?;
        let asset = helper.create_destructible_asset(&desc)?;
        let placed = helper.create_destructible(&DestructibleDef {
            asleep: false,
            ..DestructibleDef::new(asset)
        })?;
        let mut steps = 0u32;
        let awake = loop {
            for _ in 0..4 {
                helper.step()?;
            }
            steps += 4;
            let awake = helper
                .poll_blocking()?
                .last()
                .and_then(|f| f.counters)
                .map_or(0, |c| c.awake_count);
            if Scene::preload_done(steps, awake) {
                break awake;
            }
        };
        let body_poses = (0..n)
            .map(|k| {
                let body = helper.destructible_chunk_body(placed, k)?;
                Ok(helper.read_body_state(body)?.transform)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        desc.preload = Some(Preload {
            body_poses,
            steps,
            settled: awake == 0,
        });
        self.assets[id.0 as usize] = Arc::new(desc);
        Ok(awake == 0)
    }

    /// Register an asset (validated; shared by every instance of it).
    pub fn create_destructible_asset(
        &mut self,
        desc: &DestructibleAssetDesc,
    ) -> anyhow::Result<DestructibleAssetId> {
        desc.validate()?;
        self.assets.push(Arc::new(desc.clone()));
        Ok(DestructibleAssetId(self.assets.len() as u32 - 1))
    }

    pub fn destructible_asset(
        &self,
        id: DestructibleAssetId,
    ) -> anyhow::Result<&DestructibleAssetDesc> {
        self.assets
            .get(id.0 as usize)
            .map(|a| a.as_ref())
            .ok_or_else(|| anyhow::anyhow!("unknown destructible asset {}", id.0))
    }

    /// Place a destructible: one body per chunk (body origin = the instance
    /// frame), its shapes, and its bonds.
    pub fn create_destructible(&mut self, def: &DestructibleDef) -> anyhow::Result<DestructibleId> {
        if def.model == StrengthModel::Threshold {
            return unsupported("the Threshold strength model in World (D1b: GpuFracture)");
        }
        def.shape.validate()?;
        let asset = self
            .assets
            .get(def.asset.0 as usize)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("unknown destructible asset {}", def.asset.0))?;
        let instance = def.transform;
        let body_defs: Vec<BodyDef> = asset
            .chunks
            .iter()
            .enumerate()
            .map(|(k, c)| {
                let start = match &asset.preload {
                    Some(p) => compose(instance, p.body_poses[k]),
                    None => instance,
                };
                (c, start)
            })
            .map(|(c, start)| BodyDef {
                body_type: if c.mass > 0.0 {
                    BodyType::Dynamic
                } else {
                    BodyType::Static
                },
                position: start.position,
                rotation: start.rotation,
                is_awake: !def.asleep && asset.preload.is_none(),
                user_data: def.user_data,
                ..BodyDef::default()
            })
            .collect();
        let bodies = self.create_bodies(&body_defs)?;
        let shape = ShapeDef {
            update_body_mass: false,
            ..def.shape.clone()
        };
        for (c, &id) in asset.chunks.iter().zip(&bodies) {
            let b = id.index() as usize;
            if c.mass > 0.0 {
                let frame = SolverFrame {
                    center: Vec3::from(c.pose.position),
                    axes: Quat::from_array(c.pose.rotation).normalize(),
                };
                let r = Mat3::from_quat(frame.axes);
                let tensor = r * Mat3::from_diagonal(Vec3::from(c.inertia)) * r.transpose();
                let mass = MassData {
                    mass: c.mass,
                    center: c.pose.position,
                    rotational_inertia: tensor.to_cols_array_2d(),
                };
                self.set_mass_frame(b, Some(mass), frame, Vec3::from(c.inertia))?;
            }
            // Shapes are given in the chunk's principal frame; the body frame
            // is the asset frame.
            let pose = c.pose;
            for g in &c.shapes {
                let placed = match g {
                    Geometry::Sphere { center, radius } => Geometry::Sphere {
                        center: compose(pose, Transform::new(*center, [0.0, 0.0, 0.0, 1.0]))
                            .position,
                        radius: *radius,
                    },
                    Geometry::Capsule {
                        center,
                        radius,
                        half_len,
                    } if pose.rotation == [0.0, 0.0, 0.0, 1.0] => Geometry::Capsule {
                        center: compose(pose, Transform::new(*center, [0.0, 0.0, 0.0, 1.0]))
                            .position,
                        radius: *radius,
                        half_len: *half_len,
                    },
                    Geometry::Capsule { .. } => {
                        return unsupported("rotated capsule chunks");
                    }
                    Geometry::Plane { .. } => return unsupported("plane chunks"),
                    Geometry::Box { half, transform } => Geometry::Box {
                        half: *half,
                        transform: compose(pose, *transform),
                    },
                    Geometry::Hull { hull, transform } => Geometry::Hull {
                        hull: hull.clone(),
                        transform: compose(pose, *transform),
                    },
                };
                self.create_shape(id, &shape, placed)?;
            }
        }
        let anchor_body = if def.anchors == Anchors::AssetWorldBonds
            && asset.bonds.iter().any(|b| b.b.is_none())
        {
            Some(self.create_body(&BodyDef {
                position: instance.position,
                rotation: instance.rotation,
                user_data: def.user_data,
                ..BodyDef::default()
            })?)
        } else {
            None
        };
        let index = match self.destructibles.iter().position(|d| !d.alive) {
            Some(i) => i,
            None => {
                self.destructibles.push(DestructibleSlot {
                    generation: 0,
                    alive: false,
                    asset: 0,
                    user_data: 0,
                    chunk_bodies: Vec::new(),
                    anchor_body: None,
                    bond_joints: Vec::new(),
                    bond_intact: Vec::new(),
                    actor: Vec::new(),
                });
                self.destructibles.len() - 1
            }
        };
        let mut bond_joints = Vec::with_capacity(asset.bonds.len());
        for (k, bond) in asset.bonds.iter().enumerate() {
            let a = bodies[bond.a as usize].index() as usize;
            let b = match (bond.b, anchor_body) {
                (Some(x), _) => bodies[x as usize].index() as usize,
                (None, Some(anchor)) => anchor.index() as usize,
                (None, None) => {
                    bond_joints.push(None);
                    continue;
                }
            };
            let s = def.strength_scale;
            let spec = BondSpec {
                k: bond.stiffness,
                strength: bond.strength.map(|x| x * s),
            };
            let desc = JointDesc::bond(
                0,
                0,
                Vec3::from(bond.anchor_a),
                Vec3::from(bond.anchor_b),
                Vec3::from(bond.normal),
                Vec3::from(bond.frame_u),
                Quat::from_array(bond.rest),
                spec,
            );
            let joint = self.add_joint_solver(desc, a, b, def.user_data);
            self.bond_owner
                .insert(joint.index(), (index as u32, k as u32));
            bond_joints.push(Some(joint));
        }
        let n = asset.chunks.len();
        let slot = &mut self.destructibles[index];
        slot.generation = slot.generation.wrapping_add(1).max(1);
        slot.alive = true;
        slot.asset = def.asset.0;
        slot.user_data = def.user_data;
        slot.chunk_bodies = bodies;
        slot.anchor_body = anchor_body;
        slot.bond_intact = bond_joints.iter().map(|j| j.is_some()).collect();
        slot.bond_joints = bond_joints;
        slot.actor = vec![0; n];
        let id = DestructibleId::new(index as u32, slot.generation);
        let actors = self.actors(index);
        self.destructibles[index].actor = actors;
        Ok(id)
    }

    /// Remove a destructible: its bodies, shapes and bonds.
    pub fn destroy_destructible(&mut self, id: DestructibleId) -> anyhow::Result<()> {
        let i = self.live_destructible(id)?;
        let slot = &mut self.destructibles[i];
        slot.alive = false;
        let bodies = std::mem::take(&mut slot.chunk_bodies);
        let anchor = slot.anchor_body.take();
        for j in std::mem::take(&mut slot.bond_joints).into_iter().flatten() {
            self.bond_owner.remove(&j.index());
        }
        for b in bodies.into_iter().chain(anchor) {
            self.destroy_body(b)?;
        }
        Ok(())
    }

    pub fn destructible_is_valid(&self, id: DestructibleId) -> bool {
        self.live_destructible(id).is_ok()
    }

    /// The body chunk `chunk` currently belongs to.
    /// Every chunk's body, in chunk order.
    pub(crate) fn destructible_bodies(&self, id: DestructibleId) -> anyhow::Result<Vec<BodyId>> {
        Ok(self.destructibles[self.live_destructible(id)?].chunk_bodies.clone())
    }

    pub fn destructible_chunk_body(
        &self,
        id: DestructibleId,
        chunk: u32,
    ) -> anyhow::Result<BodyId> {
        let d = &self.destructibles[self.live_destructible(id)?];
        d.chunk_bodies
            .get(chunk as usize)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("chunk {chunk} out of range"))
    }

    /// The chunk's fixed transform relative to its body (render contract:
    /// chunk world transform = body transform * this). Identity in the
    /// `Bonds` model, where every body origin is the instance frame and chunk
    /// meshes are authored in the asset frame.
    pub fn destructible_chunk_local(
        &self,
        id: DestructibleId,
        chunk: u32,
    ) -> anyhow::Result<Transform> {
        let d = &self.destructibles[self.live_destructible(id)?];
        anyhow::ensure!(
            (chunk as usize) < d.chunk_bodies.len(),
            "chunk {chunk} out of range"
        );
        Ok(Transform::IDENTITY)
    }

    /// The glTF node that renders a chunk, if the asset has one.
    pub fn destructible_chunk_node(
        &self,
        id: DestructibleId,
        chunk: u32,
    ) -> anyhow::Result<Option<u32>> {
        let d = &self.destructibles[self.live_destructible(id)?];
        Ok(self.assets[d.asset as usize]
            .chunks
            .get(chunk as usize)
            .ok_or_else(|| anyhow::anyhow!("chunk {chunk} out of range"))?
            .node)
    }

    /// The actor (connected chunks) a chunk is part of now.
    pub fn destructible_actor_of(
        &self,
        id: DestructibleId,
        chunk: u32,
    ) -> anyhow::Result<ActorInfo> {
        let i = self.live_destructible(id)?;
        let d = &self.destructibles[i];
        let label = *d
            .actor
            .get(chunk as usize)
            .ok_or_else(|| anyhow::anyhow!("chunk {chunk} out of range"))?;
        let chunks: Vec<u32> = (0..d.actor.len() as u32)
            .filter(|&c| d.actor[c as usize] == label)
            .collect();
        Ok(self.actor_info(i, chunks))
    }

    pub fn destructible_stats(&self, id: DestructibleId) -> anyhow::Result<DestructibleStats> {
        let d = &self.destructibles[self.live_destructible(id)?];
        let mut labels = d.actor.clone();
        labels.sort_unstable();
        labels.dedup();
        Ok(DestructibleStats {
            chunks: d.chunk_bodies.len() as u32,
            bonds: d.bond_joints.len() as u32,
            bonds_intact: d.bond_intact.iter().filter(|&&b| b).count() as u32,
            actors: labels.len() as u32,
        })
    }

    pub(crate) fn live_destructible(&self, id: DestructibleId) -> anyhow::Result<usize> {
        let i = id.index() as usize;
        match self.destructibles.get(i) {
            Some(d) if d.alive && d.generation == id.generation() => Ok(i),
            _ => Err(crate::StaleHandle("destructible", id.to_bits()).into()),
        }
    }

    fn actor_info(&self, i: usize, chunks: Vec<u32>) -> ActorInfo {
        let d = &self.destructibles[i];
        let asset = &self.assets[d.asset as usize];
        let anchored = self.anchored(i);
        ActorInfo {
            bodies: chunks.iter().map(|&c| d.chunk_bodies[c as usize]).collect(),
            anchored: chunks.iter().any(|&c| anchored[c as usize]),
            mass: chunks.iter().map(|&c| asset.chunks[c as usize].mass).sum(),
            chunks,
        }
    }

    /// Per chunk: still connected to the world through intact bonds (or
    /// static itself).
    fn anchored(&self, i: usize) -> Vec<bool> {
        let d = &self.destructibles[i];
        let asset = &self.assets[d.asset as usize];
        let mut seed = vec![false; asset.chunks.len()];
        for (c, chunk) in asset.chunks.iter().enumerate() {
            seed[c] = chunk.mass == 0.0;
        }
        for (k, b) in asset.bonds.iter().enumerate() {
            if b.b.is_none() && d.bond_intact[k] {
                seed[b.a as usize] = true;
            }
        }
        let label = &d.actor;
        let mut anchored_labels: Vec<u32> = (0..seed.len())
            .filter(|&c| seed[c])
            .map(|c| label[c])
            .collect();
        anchored_labels.sort_unstable();
        (0..seed.len())
            .map(|c| anchored_labels.binary_search(&label[c]).is_ok())
            .collect()
    }

    /// Connected components over intact chunk-chunk bonds: each chunk's
    /// label is the lowest chunk index in its component.
    fn actors(&self, i: usize) -> Vec<u32> {
        let d = &self.destructibles[i];
        let asset = &self.assets[d.asset as usize];
        let n = asset.chunks.len();
        let mut parent: Vec<u32> = (0..n as u32).collect();
        fn find(p: &mut [u32], mut x: u32) -> u32 {
            while p[x as usize] != x {
                p[x as usize] = p[p[x as usize] as usize];
                x = p[x as usize];
            }
            x
        }
        for (k, b) in asset.bonds.iter().enumerate() {
            if let (true, Some(other)) = (d.bond_intact[k], b.b) {
                let (ra, rb) = (find(&mut parent, b.a), find(&mut parent, other));
                if ra != rb {
                    parent[ra.max(rb) as usize] = ra.min(rb);
                }
            }
        }
        (0..n as u32).map(|c| find(&mut parent, c)).collect()
    }

    /// Turn this frame's broken bond joints into destruction events: one
    /// `BondsBroken` per destructible, and a `Split` for every actor that came
    /// apart. Non-bond joint breaks pass through unchanged.
    pub(crate) fn destruction_events(
        &mut self,
        broken: Vec<DestructionEvent>,
    ) -> Vec<DestructionEvent> {
        let mut out = Vec::new();
        let mut by_destructible: Vec<(u32, Vec<u32>)> = Vec::new();
        for e in broken {
            if let DestructionEvent::JointBroke { joint, .. } = &e {
                if let Some(&(d, bond)) = self.bond_owner.get(&joint.index()) {
                    match by_destructible.iter_mut().find(|x| x.0 == d) {
                        Some(x) => x.1.push(bond),
                        None => by_destructible.push((d, vec![bond])),
                    }
                    continue;
                }
            }
            out.push(e);
        }
        for (d, mut bonds) in by_destructible {
            let i = d as usize;
            if !self.destructibles[i].alive {
                continue;
            }
            bonds.sort_unstable();
            bonds.dedup();
            for &b in &bonds {
                self.destructibles[i].bond_intact[b as usize] = false;
            }
            let id = DestructibleId::new(d, self.destructibles[i].generation);
            out.push(DestructionEvent::BondsBroken {
                destructible: id,
                bonds: bonds.clone(),
            });
            let before = self.destructibles[i].actor.clone();
            let anchored_before = self.anchored(i);
            let after = self.actors(i);
            self.destructibles[i].actor = after.clone();
            let mut parents: Vec<u32> = bonds
                .iter()
                .map(|&b| {
                    before[self.assets[self.destructibles[i].asset as usize].bonds[b as usize].a
                        as usize]
                })
                .collect();
            parents.sort_unstable();
            parents.dedup();
            for parent in parents {
                let members: Vec<u32> = (0..before.len() as u32)
                    .filter(|&c| before[c as usize] == parent)
                    .collect();
                let mut labels: Vec<u32> = members.iter().map(|&c| after[c as usize]).collect();
                labels.sort_unstable();
                labels.dedup();
                if labels.len() < 2 {
                    continue;
                }
                let asset = self.assets[self.destructibles[i].asset as usize].clone();
                let slot = &self.destructibles[i];
                let parent_info = ActorInfo {
                    bodies: members
                        .iter()
                        .map(|&c| slot.chunk_bodies[c as usize])
                        .collect(),
                    anchored: members.iter().any(|&c| anchored_before[c as usize]),
                    mass: members.iter().map(|&c| asset.chunks[c as usize].mass).sum(),
                    chunks: members.clone(),
                };
                let children = labels
                    .iter()
                    .map(|&l| {
                        let chunks: Vec<u32> = members
                            .iter()
                            .copied()
                            .filter(|&c| after[c as usize] == l)
                            .collect();
                        self.actor_info(i, chunks)
                    })
                    .collect();
                out.push(DestructionEvent::Split {
                    destructible: id,
                    parent: parent_info,
                    children,
                });
            }
        }
        out
    }
}

fn compose(a: Transform, b: Transform) -> Transform {
    let q = a.q();
    Transform::new(
        (a.p() + q * b.p()).to_array(),
        (q * b.q()).normalize().to_array(),
    )
}
