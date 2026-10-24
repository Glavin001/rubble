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
//! **Impulses** of the scripted loads change the driven bodies' velocities at the start
//! of the step (`dv = J / m`, `dw = I_world^-1 L`), as GPU impulse records
//! (`WgpuSim::apply_linear_angular_impulse`): no body upload, so manifold reuse stays on.
//!
//! **CCD.** An impactor flagged `ccd` makes the step subdivide so that no CCD body
//! moves more than half its smallest half extent per substep (refining the step, never
//! coarsening it). Contacts of every substep are reported, each with its substep's
//! impulse. `set_toi_ccd(true)` (or `STRESS_RUBBLE_TOI_CCD=1`) makes such an impactor
//! a `BODY_CCD` rubble body instead: time-of-impact contacts in one step
//! (engine/docs/ccd.md). Opt-in: `BODY_CCD` turns on look-ahead contact semantics for
//! the whole scene, and the stress coupling's breakage was measured against substeps
//! (punch_gpu: the ram leaves at 6.5 m/s instead of 28, reference 35.1).
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

pub mod scenes;
#[cfg(feature = "gpu")]
pub mod ingest_check;
#[cfg(feature = "gpu")]
pub mod resident;
#[cfg(feature = "gpu")]
pub mod render;
#[cfg(feature = "gpu")]
pub mod threshold;

use std::collections::HashMap;
use std::ops::Range;

use engine_schema::convex::{convex_collider_params, principal_axes, CollisionFilter, ConvexHull, HullArena};
use engine_schema::{AvbdParams, BodyFlags, BodyState, Collider, ShapeType, WorldCapacity, FLAG_CONTACT_GAP, FLAG_WORLD_INERTIA};

/// rubble's look-ahead contacts (world_step.slang `FLAG_SPEC_VELOCITY`).
const FLAG_SPEC_VELOCITY: u32 = 1 << 14;
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
pub(crate) struct Entry {
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
    /// Each collider's shape in the body's frame before the principal axes (for a
    /// cluster, the structure's frame), as the GPU needs it to make bodies of a split's
    /// children itself (`Part`).
    parts: Vec<Part>,
}

/// One collider's shape in the frame its body's shapes are given in (a cluster's: the
/// structure's): its mass, centre and inertia about the centre, its collider's origin and
/// rotation there, and the collider's shape record. A body of any set of a structure's
/// chunks is these combined: mass and centre summed, inertia shifted to the common
/// centre, principal axes, then each collider at `to_body (origin - com)` turned by
/// `to_body rotation` (`cook_with`).
#[derive(Clone, Copy, Debug)]
pub struct Part {
    pub mass: f64,
    pub center: Vec3,
    pub inertia: Mat3,
    pub origin: Vec3,
    pub rotation: Quat,
    pub collider: Collider,
}

/// Inertia of a solid box of half extents `h` about its centre.
fn box_inertia(mass: f64, h: Vec3) -> Vec3 {
    Vec3::new(h.y * h.y + h.z * h.z, h.x * h.x + h.z * h.z, h.x * h.x + h.y * h.y) * (mass / 3.0)
}

/// Parallel-axis shift per unit mass for an offset `d`.
fn parallel_axis(d: Vec3) -> Mat3 {
    Mat3::diag(Vec3::splat(d.norm2())) - Mat3::outer(d, d)
}

/// A shape's convex hull cooked once in its own (shape) frame: its arena entry, the
/// recentring offset (shape frame) and its bounding radius. Keyed by the exact bits of
/// its points, so every body the chunk later belongs to, and every identical chunk (the
/// copies of a building), reuses one cooked hull; its orientation goes on the collider.
type HullCache = HashMap<Vec<u64>, Option<HullEntry>>;

/// A cooked hull (`None` in the cache: rubble rejected it, and the shape is its box).
#[derive(Clone, Copy)]
struct HullEntry {
    /// Arena entry, recentring offset (shape frame), bounding radius.
    base: u32,
    offset: glam::Vec3,
    radius: f32,
    /// Unit-density mass properties (shape frame) and local bounds.
    volume: f32,
    com: glam::Vec3,
    inertia: glam::Mat3,
    lo: glam::Vec3,
    hi: glam::Vec3,
}

/// The cooked hull of `points` (shape frame), from the cache or built once.
fn hull_entry(points: &[Vec3], chunk: Option<(usize, usize)>, arena: &mut HullArena, hulls: &mut HullCache) -> Option<HullEntry> {
    let key: Vec<u64> = points.iter().flat_map(|p| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]).collect();
    if let Some(e) = hulls.get(&key) {
        return *e;
    }
    let pts: Vec<glam::Vec3> = points.iter().map(|p| gv(*p)).collect();
    let entry = match ConvexHull::from_points(&pts) {
        Ok(hull) => {
            let unit = hull.mass_properties(1.0);
            let (lo, hi) = hull.local_bounds();
            let (cooked, offset) = hull.recentered();
            let base = arena.insert(&cooked).unwrap_or_else(|e| panic!("chunk {chunk:?}: hull arena: {e}"));
            Some(HullEntry { base, offset, radius: cooked.bounding_radius(), volume: unit.volume, com: unit.center_of_mass, inertia: unit.inertia, lo, hi })
        }
        Err(e) => {
            // Never silent: reported, and the shape is its box (mass and collision).
            eprintln!("stress-rubble: chunk {chunk:?} hull rejected by rubble ({e}); its box stands in");
            HULL_FALLBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            None
        }
    };
    hulls.insert(key, entry);
    entry
}

/// Chunk shapes whose hull rubble rejected, colliding as boxes instead (reported).
pub static HULL_FALLBACKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(test)]
fn cook(b: &EngineBody, arena: &mut HullArena) -> Cooked {
    cook_with(b, arena, &mut HullCache::new())
}

fn cook_with(b: &EngineBody, arena: &mut HullArena, hulls: &mut HullCache) -> Cooked {
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
            parts: Vec::new(),
        },
        BodyGeometry::Boxes(shapes) => {
            // Each shape's mass properties in the body frame: (mass, centre, inertia
            // about its centre), and its hull (if any) in the body frame.
            let mut parts: Vec<(f64, Vec3, Mat3)> = Vec::new();
            let mut thin = f64::INFINITY;
            for s in shapes.iter() {
                let r = s.local.rotation.to_mat3();
                match &s.hull {
                    None => {
                        thin = thin.min(s.half_extents.min_elem());
                        let i = Mat3::diag(box_inertia(s.mass, s.half_extents));
                        parts.push((s.mass, s.local.position, r * i * r.transpose()));
                    }
                    Some(points) if hull_entry(points, s.chunk, arena, hulls).is_none() => {
                        thin = thin.min(s.half_extents.min_elem());
                        let i = Mat3::diag(box_inertia(s.mass, s.half_extents));
                        parts.push((s.mass, s.local.position, r * i * r.transpose()));
                    }
                    Some(points) => {
                        let h = hull_entry(points, s.chunk, arena, hulls).expect("cooked above");
                        let scale = s.mass / h.volume as f64;
                        let c = s.local.transform_point(rv(h.com));
                        let m = h.inertia;
                        let i = Mat3 {
                            m: [
                                [m.x_axis.x as f64, m.y_axis.x as f64, m.z_axis.x as f64],
                                [m.x_axis.y as f64, m.y_axis.y as f64, m.z_axis.y as f64],
                                [m.x_axis.z as f64, m.y_axis.z as f64, m.z_axis.z as f64],
                            ],
                        } * scale;
                        thin = thin.min(rv((h.hi - h.lo) * 0.5).min_elem());
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
            let mut cooked_parts = Vec::with_capacity(shapes.len());
            for (k, s) in shapes.iter().enumerate() {
                let before = colliders.len();
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
                        // Cooked in the shape's own frame (rubble accepts it there; rotated
                        // into a fragment's principal frame and rounded to f32, a convex
                        // hull can fail rubble's convexity tolerance), oriented by the
                        // collider's rotation.
                        let cooked = hull_entry(points, s.chunk, arena, hulls).map(|h| (h.base, h.offset, h.radius));
                        let q = (to_body * s.local.rotation).normalized();
                        match cooked {
                            Some((base, offset, radius)) => colliders.push(Collider {
                                body: 0,
                                shape: ShapeType::ConvexHull as u32,
                                material: 0,
                                generation: 0,
                                local_pos: gv(to_body.rotate(s.local.position + s.local.rotation.rotate(Vec3::new(offset.x as f64, offset.y as f64, offset.z as f64)) - com)).to_array(),
                                density,
                                local_rot: glam_quat(q).to_array(),
                                params: convex_collider_params(base, 1.0, radius),
                            }),
                            None => colliders.push(Collider {
                                body: 0,
                                shape: ShapeType::Box as u32,
                                material: 0,
                                generation: 0,
                                local_pos: gv(to_body.rotate(s.local.position - com)).to_array(),
                                density,
                                local_rot: glam_quat(q).to_array(),
                                params: [s.half_extents.x as f32, s.half_extents.y as f32, s.half_extents.z as f32, 0.0],
                            }),
                        }
                    }
                }
                // The part: this collider's shape in the shapes' frame.
                let col = colliders[before];
                let origin = match &s.hull {
                    Some(_) if col.shape == ShapeType::ConvexHull as u32 => {
                        let h = hull_entry(s.hull.as_ref().expect("hull"), s.chunk, arena, hulls).expect("cooked above");
                        s.local.position + s.local.rotation.rotate(Vec3::new(h.offset.x as f64, h.offset.y as f64, h.offset.z as f64))
                    }
                    _ => s.local.position,
                };
                let (mass, center, inertia) = parts[k];
                cooked_parts.push(Part { mass, center, inertia, origin, rotation: s.local.rotation, collider: col });
            }
            Cooked {
                mass,
                moments: Vec3::new(moments.x, moments.y, moments.z),
                com,
                principal,
                colliders,
                chunks: shapes.iter().map(|s| s.chunk).collect(),
                thin,
                parts: cooked_parts,
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
    /// Chunk hulls cooked so far (`HullCache`).
    hulls: HullCache,
    /// Each cooked chunk's parts (`Part`, the structure's frame), for the GPU's body
    /// creation (`ResidentEngine::part_table`).
    pub(crate) chunk_parts: HashMap<(usize, usize), Vec<Part>>,
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
    /// The GPU's body states as last read, while nothing has changed them since (a step,
    /// a body written or added, a restore): `bodies_now` answers from it.
    bodies_read: std::cell::RefCell<Option<Vec<BodyState>>>,
    /// The GPU contact ingest run and checked with every step (`attach_ingest_check`).
    #[cfg(feature = "gpu")]
    ingest_check: Option<ingest_check::IngestCheck>,
    /// Bumped whenever colliders are added or retired, or the body set is restored.
    layout_version: u64,
    /// The colliders each bump of `layout_version` changed, (version after, range), from
    /// `collider_log_floor` on (`collider_changes_since`).
    collider_log: Vec<(u64, Range<u32>)>,
    collider_log_floor: u64,
    /// Resident frames that read bodies back for the CCD bound.
    pub resident_reads: u64,
    /// The resident frame's speculative contact range (`resident_substeps`).
    resident_offset: f32,
    /// Look-ahead contacts instead of CCD substeps (`STRESS_RUBBLE_SPECULATIVE_CCD`).
    pub speculative_ccd: bool,
    /// CCD impactors as `BODY_CCD` bodies in one step, not substeps (`set_toi_ccd`).
    toi_ccd: bool,
}

/// Room for a scene: the default, or more when the scene is larger. A split's children
/// take new slots and its parent's are retired until `RubbleEngine::compact_slots`
/// reclaims them, which it does once half are retired (`compact_due`). So at a frame's
/// start fewer than twice the live slots are in use, and a frame adds at most one body
/// per chunk and one collider per part. Live bodies are at most one per chunk (plus the
/// impactors, the shots and the ground), so bodies and colliders get three times that.
pub fn capacity_for(scene: &stress_ref::scene::Scene) -> WorldCapacity {
    let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum();
    let parts: usize = scene.bodies.iter().flat_map(|b| &b.chunks).map(|c| c.compound.as_ref().map_or(1, |p| p.len().max(1))).sum();
    // The impactors, the lab's shots (16), the ground.
    let extra = scene.impactors.len() + 16 + 2;
    let mut c = default_capacity();
    c.max_colliders = c.max_colliders.max((3 * (parts + extra)).next_power_of_two() as u32);
    c.max_bodies = c.max_bodies.max((3 * (chunks + extra)).next_power_of_two() as u32);
    c.max_contacts = c.max_contacts.max(c.max_colliders);
    c.max_candidates = c.max_candidates.max(4 * c.max_colliders);
    c.max_hull_vertices = c.max_hull_vertices.max((32 * chunks).next_power_of_two() as u32);
    // More room on request (`STRESS_RUBBLE_MAX_BODIES`, `STRESS_RUBBLE_MAX_COLLIDERS`);
    // past it an upload fails, naming the capacity.
    let env = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<u32>().ok());
    if let Some(n) = env("STRESS_RUBBLE_MAX_BODIES") {
        c.max_bodies = c.max_bodies.max(n);
    }
    if let Some(n) = env("STRESS_RUBBLE_MAX_COLLIDERS") {
        c.max_colliders = c.max_colliders.max(n);
        c.max_contacts = c.max_contacts.max(n);
        c.max_candidates = c.max_candidates.max(4 * n);
    }
    c
}

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
        Self::with_sim(gravity, WgpuSim::new_with_compact_contacts(capacity, false)?)
    }

    /// The engine on an existing device: the GPU stress solver's, so rubble and the
    /// stress solve share one device and queue.
    pub fn with_device(gravity: Vec3, info: wgpu::AdapterInfo, device: wgpu::Device, queue: wgpu::Queue) -> anyhow::Result<RubbleEngine> {
        Self::with_device_capacity(gravity, info, device, queue, default_capacity())
    }

    /// `with_device` with the given capacity (`capacity_for` sizes it from a scene).
    pub fn with_device_capacity(gravity: Vec3, info: wgpu::AdapterInfo, device: wgpu::Device, queue: wgpu::Queue, capacity: WorldCapacity) -> anyhow::Result<RubbleEngine> {
        Self::with_sim(gravity, WgpuSim::with_device(capacity, info, device, queue)?)
    }

    fn with_sim(gravity: Vec3, mut sim: WgpuSim) -> anyhow::Result<RubbleEngine> {
        anyhow::ensure!(gravity.x == 0.0 && gravity.y == 0.0, "rubble's gravity is along its up axis only; got {gravity:?} (Z-up)");
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
        // `STRESS_RUBBLE_SLEEP=1` (a measurement, not adopted): rubble's own sleep thresholds,
        // so resting bodies move to the broadphase's rest table.
        let params = if std::env::var("STRESS_RUBBLE_SLEEP").is_ok_and(|v| v == "1") {
            let d = AvbdParams::default();
            AvbdParams { sleep_lin: d.sleep_lin, sleep_ang: d.sleep_ang, ..params }
        } else {
            params
        };
        // `STRESS_RUBBLE_ITERATIONS=n`: rubble's solver iterations per step (a PhysX-matched
        // setting for comparisons: vibe-land's PhysX runs 4 position iterations); default
        // rubble's own.
        let mut params = params;
        if let Some(n) = std::env::var("STRESS_RUBBLE_ITERATIONS").ok().and_then(|v| v.parse().ok()) {
            params.iterations = n;
        }
        // `STRESS_RUBBLE_SPECULATIVE_CCD=1` (opt-in, measured): rubble's look-ahead contacts
        // (FLAG_SPEC_VELOCITY: each moving collider's contacts found within its step's
        // motion, PhysX-style speculative CCD) with a reach cap of
        // `STRESS_RUBBLE_SPEC_REACH` m (default 3) instead of CCD substeps.
        let speculative_ccd = std::env::var("STRESS_RUBBLE_SPECULATIVE_CCD").is_ok_and(|v| v == "1");
        if speculative_ccd {
            params.flags |= FLAG_SPEC_VELOCITY;
            sim.set_spec_reach_max(std::env::var("STRESS_RUBBLE_SPEC_REACH").ok().and_then(|v| v.parse().ok()).unwrap_or(3.0));
        }
        sim.set_params(params);
        let _ = speculative_ccd;
        // Small coupling scenes dispatch only the colours in use by default (engine 8ev).
        Ok(RubbleEngine {
            sim,
            arena: HullArena::new(),
            hulls: HullCache::new(),
            chunk_parts: HashMap::new(),
            bodies: HashMap::new(),
            owners: Vec::new(),
            cols: Vec::new(),
            frozen: Vec::new(),
            saved: None,
            spare: None,
            substeps: 1,
            max_substeps: 256,
            last_substeps: 0,
            bodies_read: std::cell::RefCell::new(None),
            #[cfg(feature = "gpu")]
            ingest_check: None,
            layout_version: 0,
            collider_log: Vec::new(),
            collider_log_floor: 0,
            resident_reads: 0,
            resident_offset: 0.01,
            speculative_ccd: std::env::var("STRESS_RUBBLE_SPECULATIVE_CCD").is_ok_and(|v| v == "1"),
            toi_ccd: std::env::var("STRESS_RUBBLE_TOI_CCD").is_ok_and(|v| v == "1"),
        })
    }

    /// The engine boxed for `World::with_engine`; `None` without a GPU adapter.
    pub fn boxed(gravity: Vec3) -> Option<Box<dyn RigidEngine>> {
        RubbleEngine::new(gravity).ok().map(|e| Box::new(e) as Box<dyn RigidEngine>)
    }

    /// A capability profile's engine policy (`Profile::engine_policy`): solver iterations
    /// per step and substeps a frame at most (`None`: as they are).
    pub fn set_policy(&mut self, iterations: Option<u32>, max_substeps: Option<u32>) {
        if let Some(n) = iterations {
            let mut params = self.sim.params();
            params.iterations = n;
            self.sim.set_params(params);
        }
        if let Some(n) = max_substeps {
            self.max_substeps = n.max(1);
        }
    }

    /// CCD impactors as `BODY_CCD` rubble bodies with time-of-impact contacts in one
    /// step (true), or whole-world substeps (false, the default). Set before the
    /// impactors are created: it is their body flag.
    pub fn set_toi_ccd(&mut self, on: bool) {
        self.toi_ccd = on;
    }

    pub fn toi_ccd(&self) -> bool {
        self.toi_ccd
    }

    /// Number of live bodies (ground and retired bodies excluded).
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// Rubble bodies created so far (retired ones included).
    pub fn rubble_bodies(&self) -> u32 {
        self.sim.body_count()
    }

    /// Live (not retired) bodies and colliders, by old id, ascending.
    fn live_slots(&self) -> (Vec<u32>, Vec<u32>) {
        let cols: Vec<u32> = (0..self.owners.len() as u32).filter(|&c| !matches!(self.owners[c as usize], Owner::Retired)).collect();
        let mut live = vec![false; self.sim.body_count() as usize];
        for e in self.bodies.values() {
            live[e.id as usize] = true;
        }
        for &c in &cols {
            live[self.cols[c as usize].body as usize] = true;
        }
        ((0..live.len() as u32).filter(|&b| live[b as usize]).collect(), cols)
    }

    /// Whether the slots split-retired bodies left are worth reclaiming ([`Self::compact_slots`]):
    /// half or more of the bodies or of the colliders retired, or the colliders past
    /// three quarters of the capacity with any retired. Rubble appends a split's children
    /// and keeps the parent's slots, so without this a long fracture run fills the
    /// capacity, and every step walks the dead slots.
    pub fn compact_due(&self) -> bool {
        if self.saved.is_some() {
            return false;
        }
        // `STRESS_RUBBLE_COMPACT=always` (whenever anything is retired: a test of the
        // renumbering) or `never` (the capacity then bounds a run); default by the rule above.
        match std::env::var("STRESS_RUBBLE_COMPACT").as_deref() {
            Ok("never") => return false,
            Ok("always") => return self.owners.iter().any(|o| matches!(o, Owner::Retired)),
            _ => {}
        }
        let (bodies, cols) = (self.sim.body_count() as usize, self.sim.collider_count() as usize);
        let retired_cols = self.owners.iter().filter(|o| matches!(o, Owner::Retired)).count();
        let live_bodies = self.bodies.len() + 1;
        let cap = self.sim.capacity().max_colliders as usize;
        2 * retired_cols >= cols.max(1) || 2 * bodies.saturating_sub(live_bodies) >= bodies.max(1) || (4 * cols >= 3 * cap && retired_cols > 0) || (4 * bodies >= 3 * self.sim.capacity().max_bodies as usize && bodies > live_bodies)
    }

    /// Reclaim retired slots (`WgpuSim::compact`): every live body and collider moves down
    /// in order, its state and last step's contacts with it. Every id the engine handed
    /// out changes (`layout_version` bumped with no known changes): the caller rebuilds
    /// any table built from them. False when nothing was retired or a frame is saved.
    pub fn compact_slots(&mut self) -> bool {
        if self.saved.is_some() {
            return false;
        }
        let (bodies, cols) = self.live_slots();
        if bodies.len() == self.sim.body_count() as usize && cols.len() == self.sim.collider_count() as usize {
            return false;
        }
        let (body_map, col_map) = self.sim.compact(&bodies, &cols);
        for e in self.bodies.values_mut() {
            e.id = body_map[e.id as usize];
            let start = col_map[e.colliders.start as usize];
            debug_assert!(e.colliders.clone().all(|c| col_map[c as usize] == start + (c - e.colliders.start)));
            e.colliders = start..start + e.colliders.len() as u32;
        }
        self.owners = cols.iter().map(|&c| self.owners[c as usize]).collect();
        self.cols = cols
            .iter()
            .map(|&c| {
                let mut col = self.cols[c as usize];
                col.body = body_map[col.body as usize];
                col
            })
            .collect();
        *self.bodies_read.borrow_mut() = None;
        self.layout_version += 1;
        self.collider_log.clear();
        self.collider_log_floor = self.layout_version;
        true
    }

    /// A bump of `layout_version` for the colliders `range` (`collider_log`).
    pub(crate) fn colliders_changed(&mut self, range: Range<u32>) {
        self.layout_version += 1;
        self.collider_log.push((self.layout_version, range));
        if self.collider_log.len() > 256 {
            self.collider_log.drain(..128);
            self.collider_log_floor = self.collider_log[0].0 - 1;
        }
    }

    /// The colliders changed since `layout_version` was `version` (None: not known).
    pub(crate) fn collider_changes(&self, version: u64) -> Option<Vec<Range<u32>>> {
        if version < self.collider_log_floor || version > self.layout_version {
            return None;
        }
        Some(self.collider_log.iter().filter(|(v, _)| *v > version).map(|(_, r)| r.clone()).collect())
    }

    fn add_colliders(&mut self, cols: &[Collider], owners: impl Iterator<Item = Owner>) -> Range<u32> {
        let range = self.sim.add_colliders(cols);
        self.colliders_changed(range.clone());
        self.cols.extend_from_slice(cols);
        self.owners.extend(owners);
        debug_assert_eq!(self.owners.len(), self.sim.collider_count() as usize);
        range
    }

    fn create_body(&mut self, b: &EngineBody) {
        let c = cook_with(b, &mut self.arena, &mut self.hulls);
        // Each chunk's parts, once (they depend on the chunk only).
        let mut k = 0;
        while k < c.parts.len() {
            let chunk = c.chunks[k];
            let mut end = k + 1;
            while end < c.parts.len() && c.chunks[end] == chunk {
                end += 1;
            }
            if let Some(chunk) = chunk {
                self.chunk_parts.entry(chunk).or_insert_with(|| c.parts[k..end].to_vec());
            }
            k = end;
        }
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
        // TOI CCD: `BODY_CCD`, also on a parked (fixed) shot slot, where it is inert until
        // a launch makes the body dynamic, and tells the engine to schedule its CCD passes.
        if b.ccd && self.toi_ccd {
            state.flags |= engine_schema::BODY_CCD;
        }
        let id = self.sim.add_body(state);
        // The bodies' snapshot (if any) takes the record as uploaded (no download for the
        // next frame's CCD bound).
        {
            let mut read = self.bodies_read.borrow_mut();
            match read.as_mut() {
                Some(all) if all.len() == id as usize => all.push(state),
                _ => *read = None,
            }
        }
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
        let mut s = self.bodies_now()[e.id as usize];
        s.inv_mass = 0.0;
        s.inv_inertia = [0.0; 3];
        s.linear_vel = [0.0; 3];
        s.angular_vel = [0.0; 3];
        s.flags = 0;
        self.sim.write_bodies(e.id, &[s]);
        self.bodies_changed();
        // A category no collider's mask includes, and a mask that admits none.
        let never = CollisionFilter { categories: 0x8000, mask: 0, group: 0 };
        // The body's colliders are contiguous: retired in one upload.
        for id in e.colliders.clone() {
            never.apply(&mut self.cols[id as usize]);
            self.owners[id as usize] = Owner::Retired;
        }
        self.sim.set_colliders(e.colliders.start, &self.cols[e.colliders.start as usize..e.colliders.end as usize]);
        self.colliders_changed(e.colliders.clone());
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

    /// Every body's state as the GPU has it: read back once, then reused until a step,
    /// a write or a new body changes the buffer (`bodies_changed`).
    fn bodies_now(&self) -> Vec<BodyState> {
        let mut read = self.bodies_read.borrow_mut();
        read.get_or_insert_with(|| self.sim.download_bodies()).clone()
    }

    fn bodies_changed(&self) {
        *self.bodies_read.borrow_mut() = None;
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
        self.bodies_changed();
    }

    /// Total linear momentum of the dynamic bodies (oracle frame).
    pub fn momentum(&self) -> Vec3 {
        let all = self.bodies_now();
        self.bodies
            .values()
            .filter(|e| e.inv_mass > 0.0)
            .map(|e| from_rubble(all[e.id as usize].linear_vel) * (1.0 / e.inv_mass as f64))
            .fold(Vec3::ZERO, |a, b| a + b)
    }

    /// Collect the solved contacts of the last rubble step (`h` its length) on chunks.
    fn contacts_of_step(&self, h: f64, start: &[BodyState], out: &mut Vec<EngineContact>) {
        let t = web_time::Instant::now();
        let rows = self.sim.last_contacts().expect("contact readback");
        profile::add("step.contacts.rows", t);
        if rows.is_empty() {
            return;
        }
        let t = web_time::Instant::now();
        let end = self.bodies_now();
        profile::add("step.contacts.bodies", t);
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
        self.bodies_changed();
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
        let _timer = Timer(web_time::Instant::now(), "sync_bodies");
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
        let _timer = Timer(web_time::Instant::now(), "step");
        let mut all = self.bodies_now();
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
        self.bodies_changed();
            self.frozen.push(*k);
        }
        // The scripted loads' impulses into driven bodies: GPU impulse records,
        // applied at the start of the step (no body upload). The host mirror
        // `all` takes the same change for this step's CCD and contact bookkeeping.
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
            self.sim.apply_linear_angular_impulse(e.id, to_rubble(*lin), to_rubble(*ang));
            self.bodies_changed();
        }

        // Substeps: CCD bodies move at most half their thinnest extent per substep, unless
        // they are BODY_CCD (`toi_ccd`: time-of-impact contacts in the one step).
        let mut n = self.substeps.max(1);
        let ccd_substeps = !self.toi_ccd;
        for (_, e) in self.bodies.iter().filter(|(k, e)| ccd_substeps && !self.speculative_ccd && e.ccd && e.inv_mass > 0.0 && !self.frozen.contains(k)) {
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
        // `STRESS_RUBBLE_SPECULATIVE=1` (measurement only, off by default): instead of
        // sub-stepping the whole scene for a fast body, one step whose speculative
        // contact range (`contact_offset`, contacts still aim at zero gap) covers the
        // farthest a CCD body travels in it. A behaviour change, measured, not adopted.
        let mut offset = 0.01f32;
        if n > 1 && std::env::var("STRESS_RUBBLE_SPECULATIVE").is_ok_and(|v| v == "1") {
            let mut travel = 0.0f64;
            for (_, e) in self.bodies.iter().filter(|(k, e)| e.ccd && e.inv_mass > 0.0 && !self.frozen.contains(k)) {
                let s = &all[e.id as usize];
                travel = travel.max(glam::Vec3::from_array(s.linear_vel).length() as f64 * dt + glam::Vec3::from_array(s.angular_vel).length() as f64 * e.thin * dt);
            }
            offset = offset.max(travel as f32);
            n = 1;
            self.last_substeps = 1;
        }
        let h = dt / n as f64;
        let mut params = self.sim.params();
        params.dt = h as f32;
        params.contact_offset = offset;
        self.sim.set_params(params);
        let mut out = Vec::new();
        #[cfg(feature = "gpu")]
        self.ingest_begin(n);
        for _s in 0..n {
            let t = web_time::Instant::now();
            self.bodies_changed();
            if let Err(e) = self.sim.step() {
                panic!("rubble step failed: {e:?} (capacity overflow)");
            }
            profile::add("step.sim", t);
            #[cfg(feature = "gpu")]
            self.ingest_substep(_s, h);
            let t = web_time::Instant::now();
            self.contacts_of_step(h, &all, &mut out);
            profile::add("step.contacts", t);
        }
        #[cfg(feature = "gpu")]
        self.ingest_end(n, &out);
        out
    }

    fn motion(&self) -> Vec<BodyMotion> {
        let _timer = Timer(web_time::Instant::now(), "motion");
        let all = self.bodies_now();
        self.sorted_keys()
            .into_iter()
            .map(|k| {
                let e = &self.bodies[&k];
                self.motion_of(k, e, &all[e.id as usize])
            })
            .collect()
    }

    fn set_motion(&mut self, states: &[BodyMotion]) {
        let _timer = Timer(web_time::Instant::now(), "set_motion");
        // Nothing to write and nothing frozen: no read of the bodies (it would wait on the
        // device for whatever it runs).
        if states.is_empty() && self.frozen.is_empty() {
            return;
        }
        let all = self.bodies_now();
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
        let _timer = Timer(web_time::Instant::now(), "save");
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
        let _timer = Timer(web_time::Instant::now(), "restore");
        if let Some(saved) = self.saved.take() {
            self.sim.restore_state(&saved.snapshot);
        self.bodies_changed();
            self.bodies = saved.bodies;
            self.owners = saved.owners;
            self.layout_version += 1;
            // Anything may differ: no change before this version is known.
            self.collider_log.clear();
            self.collider_log_floor = self.layout_version;
            self.cols = saved.cols;
            self.frozen = saved.frozen;
            self.spare = Some(saved.snapshot);
        }
        // Release any body still frozen at the saved instant.
        if !self.frozen.is_empty() {
            let all = self.bodies_now();
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
        let _timer = Timer(web_time::Instant::now(), "discard");
        if let Some(saved) = self.saved.take() {
            self.spare = Some(saved.snapshot);
        }
    }
}


/// Wall time spent in each `RigidEngine` call (benchmarks): (calls, nanoseconds) per name.
pub mod profile {
    use std::sync::Mutex;
    static TIMES: Mutex<Vec<(&'static str, u64, u64)>> = Mutex::new(Vec::new());
    pub fn add(name: &'static str, t: web_time::Instant) {
        let ns = t.elapsed().as_nanos() as u64;
        let mut v = TIMES.lock().unwrap();
        match v.iter_mut().find(|e| e.0 == name) {
            Some(e) => {
                e.1 += 1;
                e.2 += ns;
            }
            None => v.push((name, 1, ns)),
        }
    }
    /// Take the accumulated (name, calls, milliseconds) and reset.
    pub fn take() -> Vec<(&'static str, u64, f64)> {
        std::mem::take(&mut *TIMES.lock().unwrap()).into_iter().map(|(n, c, ns)| (n, c, ns as f64 * 1e-6)).collect()
    }
}

struct Timer(web_time::Instant, &'static str);
impl Drop for Timer {
    fn drop(&mut self) {
        profile::add(self.1, self.0);
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
            geometry: BodyGeometry::Boxes(shapes.into()),
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

