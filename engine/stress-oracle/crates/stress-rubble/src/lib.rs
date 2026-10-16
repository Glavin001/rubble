//! rubble's AVBD GPU engine (`engine-webgpu`'s `WgpuSim`) as the world's rigid-body
//! engine ([`RigidEngine`]), in the role PhysX played in `stress-physx`.
//!
//! One rubble body per world body (cluster or impactor). A cluster body carries one
//! collider per chunk shape (a box, or a convex hull for hull and compound chunks), so
//! contacts are reported on the chunk they touch. Supported clusters are static
//! (`inv_mass = 0`). Sphere impactors are true spheres.
//!
//! **Frames.** The oracle is Z-up SI, rubble Y-up (gravity along -Y only): every world
//! position, velocity, angular velocity and impulse crosses the boundary through the
//! proper rotation `(x, y, z) -> (x, z, -y)`. A rubble body's frame is its centre of mass
//! with principal axes (its ABI stores a diagonal inverse inertia, applied in the world
//! frame with `FLAG_WORLD_INERTIA`); the oracle body frame is recovered from the centre
//! of mass and the principal rotation kept per body.
//!
//! **Frozen bodies** (island bodies the world simulates for a frame) are held still by
//! making them static for the step: `inv_mass` and the inverse inertia 0, velocity 0
//! (`WgpuSim::write_bodies`). The solver then neither predicts nor integrates them, and
//! resolves contacts against them for the other bodies, as against any static body:
//! the semantics of PhysX's kinematic body without a target. `set_motion` writes the
//! world's end state back with the body's mass restored; a frozen body the world does
//! not hand back is released where it stands.
//!
//! **Impulses** of the scripted loads change the driven bodies' velocities before the
//! step (`dv = J / m`, `dw = I_world^-1 L`).
//!
//! **CCD.** rubble has no swept collision; an impactor flagged `ccd` makes the step
//! subdivide so that no CCD body moves more than half its smallest half extent per
//! substep (refining the step, never coarsening it). Contacts of every substep are
//! reported, each with its substep's impulse.
//!
//! **Contacts** are the solved rows of the last step (`WgpuSim::last_contacts`): the
//! force of each row times the (sub)step, at the midpoint of its two anchors, with the
//! closing speed along the normal from the bodies' velocities at the start of the
//! engine step (a pre-contact quantity).
//!
//! **Save / restore** snapshot the whole simulation on the GPU (bodies, contacts and
//! their warm-start forces, counters): a restored frame repeats bit for bit.
//!
//! **Body removal.** rubble's body and collider arrays are append-only: a body the
//! world no longer has (a cluster that split) is retired: made static and its
//! colliders filtered out of every pair. Capacity is therefore consumed by every
//! rebuilt body; `RubbleEngine::with_capacity` sizes it.

use std::collections::HashMap;
use std::ops::Range;

use engine_schema::convex::{convex_collider_params, principal_axes, CollisionFilter, ConvexHull, HullArena};
use engine_schema::{AvbdParams, BodyFlags, BodyState, Collider, ShapeType, WorldCapacity, FLAG_CONTACT_GAP, FLAG_WORLD_INERTIA};
use engine_webgpu::{SimSnapshot, WgpuSim};
use stress_ref::engine::{BodyGeometry, BodyKey, BodyMotion, EngineBody, EngineContact, RigidEngine};
use stress_ref::math::{Mat3, Pose, Quat, Vec3};

// ---- the Z-up / Y-up boundary -------------------------------------------------------

/// Oracle (Z-up) vector in rubble's frame (Y-up).
pub fn to_rubble(v: Vec3) -> [f32; 3] {
    [v.x as f32, v.z as f32, -v.y as f32]
}

/// rubble (Y-up) vector in the oracle's frame (Z-up).
pub fn from_rubble(v: [f32; 3]) -> Vec3 {
    Vec3::new(v[0] as f64, -v[2] as f64, v[1] as f64)
}

/// The rotation taking oracle coordinates to rubble's: -90 degrees about x.
fn axes() -> Quat {
    let h = std::f64::consts::FRAC_1_SQRT_2;
    Quat { w: h, x: -h, y: 0.0, z: 0.0 }
}

/// glam quaternion `[x, y, z, w]` of an oracle-frame rotation `q` (body to oracle
/// world), as a rotation from the same body frame to rubble's world.
fn rot_to_rubble(q: Quat) -> [f32; 4] {
    let r = (axes() * q).normalized();
    [r.x as f32, r.y as f32, r.z as f32, r.w as f32]
}

/// The inverse of [`rot_to_rubble`].
fn rot_from_rubble(q: [f32; 4]) -> Quat {
    let r = Quat { w: q[3] as f64, x: q[0] as f64, y: q[1] as f64, z: q[2] as f64 };
    (axes().conjugate() * r).normalized()
}

fn glam_quat(q: Quat) -> glam::Quat {
    glam::Quat::from_xyzw(q.x as f32, q.y as f32, q.z as f32, q.w as f32)
}

fn gv(v: Vec3) -> glam::Vec3 {
    glam::Vec3::new(v.x as f32, v.y as f32, v.z as f32)
}

fn rv(v: glam::Vec3) -> Vec3 {
    Vec3::new(v.x as f64, v.y as f64, v.z as f64)
}

// ---- bodies --------------------------------------------------------------------------

/// What a rubble collider stands for.
#[derive(Clone, Copy, Debug)]
enum Owner {
    Body { key: BodyKey, chunk: Option<(usize, usize)> },
    Ground,
    /// A body removed from the world (rubble cannot delete it).
    Retired,
}

#[derive(Clone, Debug)]
struct Entry {
    id: u32,
    signature: u64,
    fixed: bool,
    colliders: Range<u32>,
    /// Centre of mass in the oracle body frame.
    com: Vec3,
    /// Oracle body frame -> principal (rubble body) frame rotation.
    principal: Quat,
    inv_mass: f32,
    inv_inertia: [f32; 3],
    ccd: bool,
    /// Smallest half extent (sphere: radius) of its shapes (m).
    thin: f64,
}

/// A body's colliders and mass properties in its principal frame.
struct Cooked {
    mass: f64,
    moments: Vec3,
    com: Vec3,
    principal: Quat,
    colliders: Vec<Collider>,
    chunks: Vec<Option<(usize, usize)>>,
    thin: f64,
}

/// Inertia of a solid box of half extents `h` about its centre.
fn box_inertia(mass: f64, h: Vec3) -> Vec3 {
    Vec3::new(h.y * h.y + h.z * h.z, h.x * h.x + h.z * h.z, h.x * h.x + h.y * h.y) * (mass / 3.0)
}

/// Parallel-axis shift per unit mass for an offset `d`.
fn parallel_axis(d: Vec3) -> Mat3 {
    Mat3::diag(Vec3::splat(d.norm2())) - Mat3::outer(d, d)
}

fn cook(b: &EngineBody, arena: &mut HullArena) -> Cooked {
    match &b.geometry {
        BodyGeometry::Sphere { radius, mass } => Cooked {
            mass: *mass,
            moments: Vec3::splat(0.4 * mass * radius * radius),
            com: Vec3::ZERO,
            principal: Quat::IDENTITY,
            colliders: vec![Collider {
                body: 0,
                shape: ShapeType::Sphere as u32,
                material: 0,
                generation: 0,
                local_pos: [0.0; 3],
                density: (*mass / (4.0 / 3.0 * std::f64::consts::PI * radius.powi(3))) as f32,
                local_rot: [0.0, 0.0, 0.0, 1.0],
                params: [*radius as f32, 0.0, 0.0, 0.0],
            }],
            chunks: vec![None],
            thin: *radius,
        },
        BodyGeometry::Boxes(shapes) => {
            // Each shape's mass properties in the body frame: (mass, centre, inertia
            // about its centre), and its hull (if any) in the body frame.
            let mut parts: Vec<(f64, Vec3, Mat3)> = Vec::new();
            let mut thin = f64::INFINITY;
            for s in shapes {
                let r = s.local.rotation.to_mat3();
                match &s.hull {
                    None => {
                        thin = thin.min(s.half_extents.min_elem());
                        let i = Mat3::diag(box_inertia(s.mass, s.half_extents));
                        parts.push((s.mass, s.local.position, r * i * r.transpose()));
                    }
                    Some(points) => {
                        let hull = ConvexHull::from_points(&points.iter().map(|p| gv(*p)).collect::<Vec<_>>())
                            .unwrap_or_else(|e| panic!("chunk {:?}: hull not accepted by rubble: {e}", s.chunk));
                        let unit = hull.mass_properties(1.0);
                        let scale = s.mass / unit.volume as f64;
                        let c = s.local.transform_point(rv(unit.center_of_mass));
                        let m = unit.inertia;
                        let i = Mat3 {
                            m: [
                                [m.x_axis.x as f64, m.y_axis.x as f64, m.z_axis.x as f64],
                                [m.x_axis.y as f64, m.y_axis.y as f64, m.z_axis.y as f64],
                                [m.x_axis.z as f64, m.y_axis.z as f64, m.z_axis.z as f64],
                            ],
                        } * scale;
                        let (lo, hi) = hull.local_bounds();
                        thin = thin.min(rv((hi - lo) * 0.5).min_elem());
                        parts.push((s.mass, c, r * i * r.transpose()));
                    }
                }
            }
            let mass: f64 = parts.iter().map(|p| p.0).sum();
            let com = parts.iter().fold(Vec3::ZERO, |a, p| a + p.1 * p.0) * (1.0 / mass.max(1e-300));
            let mut inertia = Mat3::ZERO;
            for (m, c, i) in &parts {
                inertia += *i + parallel_axis(*c - com) * *m;
            }
            let t = &inertia.m;
            let (rot, moments) = principal_axes(glam::DMat3::from_cols(
                glam::DVec3::new(t[0][0], t[1][0], t[2][0]),
                glam::DVec3::new(t[0][1], t[1][1], t[2][1]),
                glam::DVec3::new(t[0][2], t[1][2], t[2][2]),
            ));
            let principal = Quat { w: rot.w, x: rot.x, y: rot.y, z: rot.z }.normalized();
            let to_body = principal.conjugate();
            let mut colliders = Vec::new();
            for s in shapes {
                let volume = 8.0 * s.half_extents.x * s.half_extents.y * s.half_extents.z;
                let density = (s.mass / volume.max(1e-300)) as f32;
                match &s.hull {
                    None => {
                        let q = (to_body * s.local.rotation).normalized();
                        colliders.push(Collider {
                            body: 0,
                            shape: ShapeType::Box as u32,
                            material: 0,
                            generation: 0,
                            local_pos: gv(to_body.rotate(s.local.position - com)).to_array(),
                            density,
                            local_rot: glam_quat(q).to_array(),
                            params: [s.half_extents.x as f32, s.half_extents.y as f32, s.half_extents.z as f32, 0.0],
                        });
                    }
                    Some(points) => {
                        let local: Vec<glam::Vec3> = points.iter().map(|p| gv(to_body.rotate(s.local.transform_point(*p) - com))).collect();
                        let hull = ConvexHull::from_points(&local).unwrap_or_else(|e| panic!("chunk {:?}: hull not accepted by rubble: {e}", s.chunk));
                        let (hull, offset) = hull.recentered();
                        let base = arena.insert(&hull).unwrap_or_else(|e| panic!("chunk {:?}: hull arena: {e}", s.chunk));
                        colliders.push(Collider {
                            body: 0,
                            shape: ShapeType::ConvexHull as u32,
                            material: 0,
                            generation: 0,
                            local_pos: offset.to_array(),
                            density,
                            local_rot: [0.0, 0.0, 0.0, 1.0],
                            params: convex_collider_params(base, 1.0, hull.bounding_radius()),
                        });
                    }
                }
            }
            Cooked {
                mass,
                moments: Vec3::new(moments.x, moments.y, moments.z),
                com,
                principal,
                colliders,
                chunks: shapes.iter().map(|s| s.chunk).collect(),
                thin,
            }
        }
    }
}

/// The engine's frame-start state saved for a redo.
struct Saved {
    snapshot: SimSnapshot,
    bodies: HashMap<BodyKey, Entry>,
    owners: Vec<Owner>,
    cols: Vec<Collider>,
    frozen: Vec<BodyKey>,
}

/// rubble's GPU AVBD engine behind the oracle's [`RigidEngine`].
pub struct RubbleEngine {
    pub sim: WgpuSim,
    arena: HullArena,
    bodies: HashMap<BodyKey, Entry>,
    /// What each collider stands for, by collider id.
    owners: Vec<Owner>,
    /// Host copy of every collider (for retiring).
    cols: Vec<Collider>,
    /// Bodies currently held static for an island.
    frozen: Vec<BodyKey>,
    saved: Option<Saved>,
    /// A spare snapshot whose GPU buffers the next `save` reuses.
    spare: Option<SimSnapshot>,
    /// Substeps per engine step before CCD refinement (1: one rubble step per frame).
    pub substeps: u32,
    /// Most substeps CCD may ask for in one engine step.
    pub max_substeps: u32,
    /// Substeps of the last engine step.
    pub last_substeps: u32,
}

/// The default capacity: room for a few thousand bodies (every rebuilt cluster takes
/// a new body; rubble cannot reuse a retired one's slot).
pub fn default_capacity() -> WorldCapacity {
    WorldCapacity {
        max_bodies: 16_384,
        max_colliders: 65_536,
        max_contacts: 65_536,
        max_joints: 64,
        max_candidates: 262_144,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 262_144,
    }
}

impl RubbleEngine {
    /// A rubble scene with the given gravity (oracle frame; along z only); `Err` if no
    /// wgpu adapter is available.
    pub fn new(gravity: Vec3) -> anyhow::Result<RubbleEngine> {
        Self::with_capacity(gravity, default_capacity())
    }

    pub fn with_capacity(gravity: Vec3, capacity: WorldCapacity) -> anyhow::Result<RubbleEngine> {
        anyhow::ensure!(gravity.x == 0.0 && gravity.y == 0.0, "rubble's gravity is along its up axis only; got {gravity:?} (Z-up)");
        let mut sim = WgpuSim::new_with_compact_contacts(capacity, false)?;
        let params = AvbdParams {
            gravity_y: gravity.z as f32,
            // Zero-gap rest (bodies do not sit `contact_offset` deep) and the world
            // inertia `R diag(I) R^T` (the principal moments are about body axes).
            flags: FLAG_CONTACT_GAP | FLAG_WORLD_INERTIA,
            // Never asleep: a sleeping body's support reactions are loads the stress
            // solve must keep seeing (PhysX: sleep thresholds 0).
            sleep_lin: -1.0,
            sleep_ang: -1.0,
            ..AvbdParams::default()
        };
        sim.set_params(params);
        Ok(RubbleEngine {
            sim,
            arena: HullArena::new(),
            bodies: HashMap::new(),
            owners: Vec::new(),
            cols: Vec::new(),
            frozen: Vec::new(),
            saved: None,
            spare: None,
            substeps: 1,
            max_substeps: 256,
            last_substeps: 0,
        })
    }

    /// The engine boxed for `World::with_engine`; `None` without a GPU adapter.
    pub fn boxed(gravity: Vec3) -> Option<Box<dyn RigidEngine>> {
        RubbleEngine::new(gravity).ok().map(|e| Box::new(e) as Box<dyn RigidEngine>)
    }

    /// Number of live bodies (ground and retired bodies excluded).
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// Rubble bodies created so far (retired ones included).
    pub fn rubble_bodies(&self) -> u32 {
        self.sim.body_count()
    }

    fn add_colliders(&mut self, cols: &[Collider], owners: impl Iterator<Item = Owner>) -> Range<u32> {
        let range = self.sim.add_colliders(cols);
        self.cols.extend_from_slice(cols);
        self.owners.extend(owners);
        debug_assert_eq!(self.owners.len(), self.sim.collider_count() as usize);
        range
    }

    fn create_body(&mut self, b: &EngineBody) {
        let c = cook(b, &mut self.arena);
        let position = b.pose.position + b.pose.rotation.rotate(c.com);
        let rotation = (b.pose.rotation * c.principal).normalized();
        let inv = |m: f64| if m > 0.0 && !b.fixed { (1.0 / m) as f32 } else { 0.0 };
        let mut state = BodyState::static_plane_support();
        state.position = to_rubble(position);
        state.orientation = rot_to_rubble(rotation);
        state.inv_mass = inv(c.mass);
        state.inv_inertia = [inv(c.moments.x), inv(c.moments.y), inv(c.moments.z)];
        if !b.fixed {
            state.flags = BodyFlags::Dynamic as u32 | BodyFlags::Awake as u32;
            state.linear_vel = to_rubble(b.velocity);
            state.angular_vel = to_rubble(b.angular_velocity);
        }
        let id = self.sim.add_body(state);
        let cols: Vec<Collider> = c.colliders.iter().map(|col| Collider { body: id, ..*col }).collect();
        let key = b.key;
        let colliders = self.add_colliders(&cols, c.chunks.iter().map(|&chunk| Owner::Body { key, chunk }));
        if c.colliders.iter().any(|col| col.shape == ShapeType::ConvexHull as u32) {
            self.sim.sync_hulls(&mut self.arena);
        }
        self.bodies.insert(
            b.key,
            Entry {
                id,
                signature: b.signature,
                fixed: b.fixed,
                colliders,
                com: c.com,
                principal: c.principal,
                inv_mass: state.inv_mass,
                inv_inertia: state.inv_inertia,
                ccd: b.ccd,
                thin: c.thin,
            },
        );
    }

    /// Retire a body: static where it stands, its colliders in no pair.
    fn remove_body(&mut self, key: BodyKey) {
        let Some(e) = self.bodies.remove(&key) else { return };
        let mut s = self.sim.read_body(e.id);
        s.inv_mass = 0.0;
        s.inv_inertia = [0.0; 3];
        s.linear_vel = [0.0; 3];
        s.angular_vel = [0.0; 3];
        s.flags = 0;
        self.sim.write_bodies(e.id, &[s]);
        // A category no collider's mask includes, and a mask that admits none.
        let never = CollisionFilter { categories: 0x8000, mask: 0, group: 0 };
        for id in e.colliders.clone() {
            let mut col = self.cols[id as usize];
            never.apply(&mut col);
            self.cols[id as usize] = col;
            self.sim.set_collider(id, col);
            self.owners[id as usize] = Owner::Retired;
        }
        self.frozen.retain(|k| *k != key);
    }

    /// Oracle motion of a body from its rubble state.
    fn motion_of(&self, key: BodyKey, e: &Entry, s: &BodyState) -> BodyMotion {
        let rotation = (rot_from_rubble(s.orientation) * e.principal.conjugate()).normalized();
        let com = from_rubble(s.position);
        BodyMotion {
            key,
            pose: Pose::new(com - rotation.rotate(e.com), rotation),
            velocity: from_rubble(s.linear_vel),
            angular_velocity: from_rubble(s.angular_vel),
        }
    }

    fn sorted_keys(&self) -> Vec<BodyKey> {
        let mut keys: Vec<BodyKey> = self.bodies.keys().copied().collect();
        keys.sort();
        keys
    }

    /// Release a frozen body: its mass back, at `state` (rubble frame).
    fn release(&mut self, key: BodyKey, mut s: BodyState) {
        let Some(e) = self.bodies.get(&key) else { return };
        s.inv_mass = e.inv_mass;
        s.inv_inertia = e.inv_inertia;
        s.flags = BodyFlags::Dynamic as u32 | BodyFlags::Awake as u32;
        s.sleep_timer = 0.0;
        self.sim.write_bodies(e.id, &[s]);
    }

    /// Total linear momentum of the dynamic bodies (oracle frame).
    pub fn momentum(&self) -> Vec3 {
        let all = self.sim.download_bodies();
        self.bodies
            .values()
            .filter(|e| e.inv_mass > 0.0)
            .map(|e| from_rubble(all[e.id as usize].linear_vel) * (1.0 / e.inv_mass as f64))
            .fold(Vec3::ZERO, |a, b| a + b)
    }

    /// Collect the solved contacts of the last rubble step (`h` its length) on chunks.
    fn contacts_of_step(&self, h: f64, start: &[BodyState], out: &mut Vec<EngineContact>) {
        let rows = self.sim.last_contacts().expect("contact readback");
        if rows.is_empty() {
            return;
        }
        let end = self.sim.download_bodies();
        // Velocity of a body's material point `p` (rubble frame) at the start of the
        // engine step.
        let point_velocity = |owner: Owner, body: u32, p: glam::Vec3| match owner {
            Owner::Body { .. } => {
                let s = &start[body as usize];
                glam::Vec3::from_array(s.linear_vel) + glam::Vec3::from_array(s.angular_vel).cross(p - glam::Vec3::from_array(s.position))
            }
            _ => glam::Vec3::ZERO,
        };
        for c in rows.iter().filter(|c| c.alive != 0) {
            let n = glam::Vec3::from_array(c.normal);
            let f = n * c.lambda_n + glam::Vec3::from_array(c.tangent1) * c.lambda_t1 + glam::Vec3::from_array(c.tangent2) * c.lambda_t2;
            // Impulse on B's body over the step (A's: the opposite).
            let jb = -f * h as f32;
            if jb == glam::Vec3::ZERO || !jb.is_finite() {
                continue;
            }
            let (ca, cb) = (self.cols[c.collider_a as usize], self.cols[c.collider_b as usize]);
            let anchor = |col: &Collider, a: [f32; 3]| {
                let s = &end[col.body as usize];
                glam::Vec3::from_array(s.position) + glam::Quat::from_array(s.orientation) * glam::Vec3::from_array(a)
            };
            let point = (anchor(&ca, c.anchor_a) + anchor(&cb, c.anchor_b)) * 0.5;
            let (oa, ob) = (self.owners[c.collider_a as usize], self.owners[c.collider_b as usize]);
            for (mine, mine_body, other, other_body, j, push) in
                [(oa, ca.body, ob, cb.body, -jb, -n), (ob, cb.body, oa, ca.body, jb, n)]
            {
                let Owner::Body { chunk: Some((structure, chunk)), .. } = mine else { continue };
                let other_key = match other {
                    Owner::Body { key, .. } => Some(key),
                    _ => None,
                };
                let closing = (point_velocity(other, other_body, point) - point_velocity(mine, mine_body, point)).dot(push).max(0.0);
                out.push(EngineContact {
                    structure,
                    chunk,
                    point: from_rubble(point.to_array()),
                    impulse: from_rubble(j.to_array()),
                    other: other_key,
                    approach_speed: closing as f64,
                });
            }
        }
    }
}

impl RigidEngine for RubbleEngine {
    fn name(&self) -> &str {
        "rubble-webgpu"
    }

    fn add_ground(&mut self, height: f64) {
        let id = self.sim.add_body(BodyState::static_plane_support());
        let plane = Collider {
            body: id,
            shape: ShapeType::Plane as u32,
            material: 0,
            generation: 0,
            local_pos: [0.0; 3],
            density: 0.0,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [0.0, 1.0, 0.0, height as f32],
        };
        self.add_colliders(&[plane], std::iter::once(Owner::Ground));
    }

    fn sync_bodies(&mut self, bodies: &[EngineBody]) {
        let wanted: HashMap<BodyKey, &EngineBody> = bodies.iter().map(|b| (b.key, b)).collect();
        let mut stale: Vec<BodyKey> = self
            .bodies
            .iter()
            .filter(|(k, e)| wanted.get(k).is_none_or(|b| b.signature != e.signature || b.fixed != e.fixed))
            .map(|(k, _)| *k)
            .collect();
        stale.sort();
        for k in stale {
            self.remove_body(k);
        }
        let mut new: Vec<&EngineBody> = bodies.iter().filter(|b| !self.bodies.contains_key(&b.key)).collect();
        new.sort_by_key(|b| b.key);
        for b in new {
            self.create_body(b);
        }
    }

    fn step(&mut self, dt: f64, frozen: &[BodyKey], impulses: &[(BodyKey, Vec3, Vec3)]) -> Vec<EngineContact> {
        let mut all = self.sim.download_bodies();
        // Hold island bodies still.
        for k in frozen {
            let Some(e) = self.bodies.get(k) else { continue };
            if e.fixed || self.frozen.contains(k) {
                continue;
            }
            let s = &mut all[e.id as usize];
            s.inv_mass = 0.0;
            s.inv_inertia = [0.0; 3];
            s.linear_vel = [0.0; 3];
            s.angular_vel = [0.0; 3];
            let s = *s;
            self.sim.write_bodies(e.id, &[s]);
            self.frozen.push(*k);
        }
        // The scripted loads' impulses into driven bodies.
        for (k, lin, ang) in impulses {
            let Some(e) = self.bodies.get(k) else { continue };
            if e.inv_mass <= 0.0 || self.frozen.contains(k) {
                continue;
            }
            let s = &mut all[e.id as usize];
            let q = glam::Quat::from_array(s.orientation);
            let inv_i = glam::Vec3::from_array(e.inv_inertia);
            let l = glam::Vec3::from_array(to_rubble(*ang));
            let dw = q * (inv_i * (q.conjugate() * l));
            let dv = glam::Vec3::from_array(to_rubble(*lin)) * e.inv_mass;
            s.linear_vel = (glam::Vec3::from_array(s.linear_vel) + dv).to_array();
            s.angular_vel = (glam::Vec3::from_array(s.angular_vel) + dw).to_array();
            let s = *s;
            self.sim.write_bodies(e.id, &[s]);
        }

        // Substeps: CCD bodies move at most half their thinnest extent per substep.
        let mut n = self.substeps.max(1);
        for (_, e) in self.bodies.iter().filter(|(k, e)| e.ccd && e.inv_mass > 0.0 && !self.frozen.contains(k)) {
            let s = &all[e.id as usize];
            let travel = glam::Vec3::from_array(s.linear_vel).length() as f64 * dt + glam::Vec3::from_array(s.angular_vel).length() as f64 * e.thin * dt;
            let need = (travel / (0.5 * e.thin)).ceil() as u32;
            n = n.max(need.min(self.max_substeps));
        }
        self.last_substeps = n;
        let any_moving = self.bodies.iter().any(|(k, e)| e.inv_mass > 0.0 && !self.frozen.contains(k));
        if !any_moving {
            return Vec::new();
        }
        let h = dt / n as f64;
        let mut params = self.sim.params();
        params.dt = h as f32;
        self.sim.set_params(params);
        let mut out = Vec::new();
        for _ in 0..n {
            if let Err(e) = self.sim.step() {
                panic!("rubble step failed: {e:?} (capacity overflow)");
            }
            self.contacts_of_step(h, &all, &mut out);
        }
        out
    }

    fn motion(&self) -> Vec<BodyMotion> {
        let all = self.sim.download_bodies();
        self.sorted_keys()
            .into_iter()
            .map(|k| {
                let e = &self.bodies[&k];
                self.motion_of(k, e, &all[e.id as usize])
            })
            .collect()
    }

    fn set_motion(&mut self, states: &[BodyMotion]) {
        let all = self.sim.download_bodies();
        for st in states {
            let Some(e) = self.bodies.get(&st.key) else { continue };
            if e.fixed {
                continue;
            }
            let mut s = all[e.id as usize];
            s.position = to_rubble(st.pose.position + st.pose.rotation.rotate(e.com));
            s.orientation = rot_to_rubble((st.pose.rotation * e.principal).normalized());
            s.linear_vel = to_rubble(st.velocity);
            s.angular_vel = to_rubble(st.angular_velocity);
            self.release(st.key, s);
        }
        // Any frozen body the world did not hand back (it no longer exists there) is
        // released where it stands; the next sync removes it.
        let released: Vec<BodyKey> = states.iter().map(|s| s.key).collect();
        for k in std::mem::take(&mut self.frozen) {
            if released.contains(&k) {
                continue;
            }
            if let Some(e) = self.bodies.get(&k) {
                let s = all[e.id as usize];
                self.release(k, s);
            }
        }
    }

    fn save(&mut self) {
        self.discard();
        let snapshot = match self.spare.take() {
            Some(old) => self.sim.save_state_reusing(old),
            None => self.sim.save_state(),
        };
        self.saved = Some(Saved {
            snapshot,
            bodies: self.bodies.clone(),
            owners: self.owners.clone(),
            cols: self.cols.clone(),
            frozen: self.frozen.clone(),
        });
    }

    fn restore(&mut self) {
        if let Some(saved) = self.saved.take() {
            self.sim.restore_state(&saved.snapshot);
            self.bodies = saved.bodies;
            self.owners = saved.owners;
            self.cols = saved.cols;
            self.frozen = saved.frozen;
            self.spare = Some(saved.snapshot);
        }
        // Release any body still frozen at the saved instant.
        if !self.frozen.is_empty() {
            let all = self.sim.download_bodies();
            for k in std::mem::take(&mut self.frozen) {
                if let Some(e) = self.bodies.get(&k) {
                    let mut s = all[e.id as usize];
                    s.linear_vel = [0.0; 3];
                    s.angular_vel = [0.0; 3];
                    self.release(k, s);
                }
            }
        }
    }

    fn discard(&mut self) {
        if let Some(saved) = self.saved.take() {
            self.spare = Some(saved.snapshot);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_round_trip() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let r = to_rubble(v);
        assert_eq!(r, [1.0, 3.0, -2.0]);
        assert_eq!(from_rubble(r), v);
        // Z-up maps to Y-up as a rotation (the same as the vector map).
        let up = axes().rotate(Vec3::Z);
        assert!((up - Vec3::Y).norm() < 1e-12, "{up:?}");
        let q = Quat::from_axis_angle(Vec3::new(0.3, -1.0, 0.5), 0.8);
        let back = rot_from_rubble(rot_to_rubble(q));
        assert!((back.to_mat3().col(0) - q.to_mat3().col(0)).norm() < 1e-6);
        // A body-frame vector maps through the rotation as its world image does.
        let local = Vec3::new(0.2, 0.7, -0.4);
        let world = q.rotate(local);
        let rq = rot_to_rubble(q);
        let rubble_world = glam::Quat::from_array(rq) * gv(local);
        let expect = to_rubble(world);
        assert!((rubble_world - glam::Vec3::from_array(expect)).length() < 1e-5);
    }

    #[test]
    fn compound_inertia_is_diagonal_in_the_principal_frame() {
        use stress_ref::engine::BoxShape;
        let shapes = vec![
            BoxShape { local: Pose::new(Vec3::new(0.0, 0.0, 0.0), Quat::IDENTITY), half_extents: Vec3::splat(0.05), mass: 2.0, chunk: Some((0, 0)), hull: None },
            BoxShape { local: Pose::new(Vec3::new(0.1, 0.1, 0.0), Quat::IDENTITY), half_extents: Vec3::splat(0.05), mass: 2.0, chunk: Some((0, 1)), hull: None },
        ];
        let b = EngineBody {
            key: BodyKey::Cluster(1),
            signature: 0,
            geometry: BodyGeometry::Boxes(shapes),
            fixed: false,
            ccd: false,
            pose: Pose::default(),
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
        };
        let c = cook(&b, &mut HullArena::new());
        assert!((c.com - Vec3::new(0.05, 0.05, 0.0)).norm() < 1e-12);
        // Two cubes along the diagonal: the principal axes are the diagonal ones.
        let moments = [c.moments.x, c.moments.y, c.moments.z];
        let cube = 2.0 * (2.0 / 3.0) * (2.0 * 0.05 * 0.05);
        assert!(moments.iter().all(|m| *m >= cube - 1e-12), "{moments:?}");
        let along = moments.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!((along - cube).abs() < 1e-9, "{moments:?} vs {cube}");
    }
}
