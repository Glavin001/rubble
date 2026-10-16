//! Fracture lab scenes: one hierarchy level of a Shardwright asset as AVBD
//! bodies, compound convex colliders (packed hull ABI v2, `engine-schema`'s
//! cooker and arena), breakable bond joints and render streams; plus rows,
//! grids and suburbs of copies that share one hull arena and draw instanced.
//!
//! Scene data stays in f64 (as the JS lab computed it); GPU records round to
//! f32 when packed (`plan.rs`).

use crate::asset::FractureAsset;
use crate::bonds::{bond_capacities, bond_springs, scale_bond_springs, BOND_TICK};
use crate::math::*;
use anyhow::{anyhow, bail, Result};
use engine_schema::convex::{convex_collider_params, ConvexHull, HullArena, HullError, HULL_MAX_VERTICES};
use glam::{DMat3, DVec3, Vec3};
use serde::{Deserialize, Serialize};

pub const SHAPE_SPHERE: u32 = 0;
pub const SHAPE_CAPSULE: u32 = 1;
pub const SHAPE_BOX: u32 = 2;
pub const SHAPE_CONVEX: u32 = 3;
pub const SHAPE_PLANE: u32 = 4;
/// Body flags: dynamic, sleeping, awake.
pub const FLAG_DYNAMIC: u32 = 1;
pub const FLAG_SLEEPING: u32 = 4;
pub const FLAG_AWAKE: u32 = 8;
/// Projectile slots per scene ("Reset reloads 16 shots").
pub const PROJECTILES: u32 = 16;

fn identity() -> [f64; 4] {
    IDENTITY
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Body {
    pub p: [f64; 3],
    #[serde(default = "identity")]
    pub q: [f64; 4],
    #[serde(default)]
    pub v: [f64; 3],
    #[serde(default)]
    pub w: [f64; 3],
    pub inv_mass: f64,
    pub inv_i: [f64; 3],
    pub flags: u32,
}

impl Body {
    pub fn fixed(p: [f64; 3]) -> Self {
        Self { p, q: IDENTITY, v: [0.0; 3], w: [0.0; 3], inv_mass: 0.0, inv_i: [0.0; 3], flags: 0 }
    }
}

/// Collision filter (engine-schema `CollisionFilter`): 16-bit categories and
/// mask (0xffff = all) and a signed group.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Filter {
    pub categories: u16,
    pub mask: u16,
    pub group: i16,
}

impl Default for Filter {
    fn default() -> Self {
        Self { categories: 0xffff, mask: 0xffff, group: 0 }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Collider {
    pub body: u32,
    pub shape: u32,
    #[serde(default)]
    pub density: f64,
    /// Sphere radius; capsule radius, half length; box half extents; plane
    /// normal + offset; hull: tagged base bits, scale, 0, bound radius.
    pub params: [f64; 4],
    #[serde(default)]
    pub local_pos: [f64; 3],
    #[serde(default = "identity")]
    pub local_rot: [f64; 4],
    #[serde(default)]
    pub material: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<Filter>,
}

impl Collider {
    pub fn new(body: u32, shape: u32, density: f64, params: [f64; 4]) -> Self {
        Self { body, shape, density, params, local_pos: [0.0; 3], local_rot: IDENTITY, material: 0, filter: None }
    }
    pub fn ground() -> Self {
        Self::new(0, SHAPE_PLANE, 0.0, [0.0, 1.0, 0.0, 0.0])
    }
    /// A convex hull collider at arena `base` (uniform `scale`).
    pub fn hull(body: u32, density: f64, base: u32, scale: f32, radius: f32, local_pos: [f64; 3]) -> Self {
        let p = convex_collider_params(base, scale, radius);
        Self { local_pos, ..Self::new(body, SHAPE_CONVEX, density, p.map(|x| x as f64)) }
    }
    /// Arena base of a hull collider.
    pub fn hull_base(&self) -> u32 {
        engine_schema::convex::convex_params_base(self.params.map(|x| x as f32))
    }
    /// `(generation, material)` words with the filter applied.
    pub fn filter_words(&self) -> (u32, u32) {
        let mut c = engine_schema::Collider {
            body: 0,
            shape: 0,
            material: self.material,
            generation: 0,
            local_pos: [0.0; 3],
            density: 0.0,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [0.0; 4],
        };
        let f = self.filter.unwrap_or_default();
        engine_schema::convex::CollisionFilter { categories: f.categories, mask: f.mask, group: f.group }.apply(&mut c);
        (c.generation, c.material)
    }
    /// Carries a non-default filter (the GPU filter test is enabled).
    pub fn has_filter(&self) -> bool {
        let (generation, material) = self.filter_words();
        generation != 0 || material >> 16 != 0
    }
}

/// A breakable weld (JOINT_BOND) at a bond centroid: all six DOF about the
/// baked rest pose, breaking at the bond's capacities.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Joint {
    pub body_a: u32,
    pub body_b: u32,
    pub anchor_a: [f64; 3],
    pub anchor_b: [f64; 3],
    /// Rest relative rotation conj(qA) * qB.
    pub rest: [f64; 4],
    /// Bond normal in A's frame.
    pub normal: [f64; 3],
    /// Bond u axis in A's frame.
    #[serde(default)]
    pub frame_u: [f64; 3],
    /// Tension N, compression N, shear N, bending N m.
    pub strength: [f64; 4],
    /// Penalties: normal, u, v, about u, about v, about normal.
    pub k: [f64; 6],
    /// Anchored to the static ground (body 0) at `anchor_b` (world).
    #[serde(default)]
    pub world: bool,
}

/// Solver parameters, packed into the host uniform every step. Defaults are
/// the browser host's; scenes override what they need.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct SceneParams {
    pub dt: f32,
    pub gravity_y: f32,
    pub iterations: u32,
    pub flags: u32,
    pub alpha: f32,
    pub gamma: f32,
    pub beta_lin: f32,
    pub beta_ang: f32,
    pub k_start: f32,
    pub k_max_n: f32,
    pub k_max_t: f32,
    pub contact_offset: f32,
    pub sleep_lin: f32,
    pub sleep_ang: f32,
    pub sleep_time: f32,
    pub residual_exit: f32,
    pub trust_lin: f32,
    pub trust_ang: f32,
    pub pivot_floor: f32,
    pub grid_cell_size: f32,
    /// Contact friction of every material slot.
    pub friction: f32,
    /// Island sleep-gate rounds per step (0..=16).
    pub sleep_rounds: u32,
}

impl Default for SceneParams {
    fn default() -> Self {
        Self {
            dt: 1.0 / 60.0,
            gravity_y: -10.0,
            iterations: 8,
            flags: 0,
            alpha: 0.99,
            gamma: 0.999,
            beta_lin: 10000.0,
            beta_ang: 100.0,
            k_start: 1000.0,
            k_max_n: 1e10,
            k_max_t: 1e6,
            contact_offset: 0.01,
            sleep_lin: 0.15,
            sleep_ang: 0.15,
            sleep_time: 0.5,
            residual_exit: 1e-4,
            trust_lin: 0.35,
            trust_ang: 0.5,
            pivot_floor: 1e-6,
            grid_cell_size: 2.0,
            friction: 0.5,
            sleep_rounds: 8,
        }
    }
}

/// Params flags: contact reuse, bond preload (no breaking), deterministic SAT,
/// zero-gap contacts, world-frame inertia.
pub const FLAG_REUSE: u32 = 1;
pub const FLAG_BOND_PRELOAD: u32 = 2;
pub const FLAG_DETERMINISTIC: u32 = 8;
pub const FLAG_CONTACT_GAP: u32 = 32;
pub const FLAG_WORLD_INERTIA: u32 = 64;

impl SceneParams {
    /// The lab's settings for baked and procedural scenes (120 Hz, 8 iterations).
    pub fn lab(flags: u32) -> Self {
        Self {
            flags,
            iterations: 8,
            dt: 1.0 / 120.0,
            gravity_y: -9.81,
            contact_offset: 0.002,
            alpha: 0.95,
            sleep_lin: 0.03,
            sleep_ang: 0.05,
            sleep_time: 1.0,
            ..Self::default()
        }
    }
}

/// `Some(None)`: JSON `null` (dispatch every slot); `None`: absent (default).
fn headroom<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<Option<u32>>, D::Error> {
    Ok(Some(Option::deserialize(d)?))
}

/// GPU budgets and opt-in host strategies (the JS `scene.capacity`).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct Budget {
    pub contacts_per_collider: u32,
    pub candidates_per_collider: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_colors: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color_rounds: Option<u32>,
    #[serde(skip_serializing_if = "is_true")]
    pub awake_list_passes: bool,
    #[serde(skip_serializing_if = "is_false")]
    pub incremental_coloring: bool,
    #[serde(skip_serializing_if = "is_true")]
    pub tagged_warm_hash: bool,
    #[serde(skip_serializing_if = "is_false")]
    pub deterministic: bool,
    #[serde(skip_serializing_if = "is_false")]
    pub deterministic_sleep_gate: bool,
    #[serde(skip_serializing_if = "is_true")]
    pub awake_queries: bool,
    #[serde(skip_serializing_if = "is_true")]
    pub persistent_rest: bool,
    /// None: by awake count; Some: forced on or off.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub moved_only_bounds: Option<bool>,
    #[serde(deserialize_with = "headroom", skip_serializing_if = "Option::is_none")]
    pub color_round_headroom: Option<Option<u32>>,
    #[serde(deserialize_with = "headroom", skip_serializing_if = "Option::is_none")]
    pub color_headroom: Option<Option<u32>>,
}

fn is_true(b: &bool) -> bool {
    *b
}
fn is_false(b: &bool) -> bool {
    !*b
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            contacts_per_collider: 32,
            candidates_per_collider: 64,
            max_colors: None,
            color_rounds: None,
            awake_list_passes: true,
            incremental_coloring: false,
            tagged_warm_hash: true,
            deterministic: false,
            deterministic_sleep_gate: false,
            awake_queries: true,
            persistent_rest: true,
            moved_only_bounds: None,
            color_round_headroom: None,
            color_headroom: None,
        }
    }
}

impl Budget {
    /// Grids of houses: ~3 candidates and ~5 contacts per collider after impact
    /// on house level 2, so eight each; ~100k waking pieces can need more than
    /// 32 colouring rounds and colours (kernels allow 64).
    pub fn grid() -> Self {
        Self { candidates_per_collider: 8, contacts_per_collider: 8, max_colors: Some(64), color_rounds: Some(64), ..Self::default() }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Camera {
    pub yaw: f64,
    pub pitch: f64,
    pub distance: f64,
    pub min_distance: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Ray {
    pub origin: [f64; 3],
    pub direction: [f64; 3],
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct SkippedHull {
    pub fragment: u32,
    pub hull: u32,
}

/// Result of settling bonds under self-weight (`preload`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Preload {
    pub steps: u32,
    pub settled: bool,
}

/// A simulated scene: what both hosts upload and draw.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Scene {
    pub bodies: Vec<Body>,
    pub colliders: Vec<Collider>,
    #[serde(default)]
    pub joints: Vec<Joint>,
    /// Packed hull arena records (16 bytes each).
    #[serde(default)]
    pub hull_data: Vec<[u32; 4]>,
    #[serde(default)]
    pub params: SceneParams,
    #[serde(default)]
    pub capacity: Budget,
    #[serde(default)]
    pub skipped_hulls: Vec<SkippedHull>,
    /// Render streams: 10 floats per vertex (position, normal, colour, body id),
    /// body-local. Copies draw `vertices` and `lines` instanced.
    #[serde(default)]
    pub vertices: Vec<f32>,
    #[serde(default)]
    pub lines: Vec<f32>,
    #[serde(default)]
    pub projectile_vertices: Vec<f32>,
    /// Decorative road decals on the ground (body 0).
    #[serde(default)]
    pub road_vertices: Vec<f32>,
    #[serde(default)]
    pub projectile_start: u32,
    #[serde(default)]
    pub projectile_count: u32,
    #[serde(default)]
    pub projectile_radius: f64,
    #[serde(default)]
    pub projectile_mass: f64,
    #[serde(default)]
    pub projectile_speed: Option<f64>,
    /// Shots compensate for gravity (aim near the scene depth).
    #[serde(default = "yes")]
    pub projectile_arc: bool,
    /// Selected hierarchy level (None for procedural scenes).
    #[serde(default)]
    pub level: Option<u8>,
    /// Fragment ids of the level (or piece count for procedural scenes).
    #[serde(default)]
    pub selected: Vec<u32>,
    /// `(fragment id, body)` of each selected fragment in copy 0.
    #[serde(default)]
    pub body_of: Vec<(u32, u32)>,
    /// Baked bonds of the level (per house).
    #[serde(default)]
    pub bond_count: u32,
    #[serde(default)]
    pub extent: f64,
    #[serde(default)]
    pub extent_x: f64,
    #[serde(default)]
    pub extent_z: f64,
    #[serde(default)]
    pub height: f64,
    #[serde(default)]
    pub center: [f64; 3],
    #[serde(default = "one")]
    pub copies: u32,
    /// Bodies per copy (0: a single scene).
    #[serde(default)]
    pub instance_stride: u32,
    #[serde(default)]
    pub columns: u32,
    #[serde(default)]
    pub rows: u32,
    #[serde(default)]
    pub row_pitch: f64,
    #[serde(default)]
    pub column_pitch: f64,
    #[serde(default)]
    pub row_length: f64,
    #[serde(default)]
    pub grid_width: f64,
    #[serde(default)]
    pub grid_cell_size: f64,
    #[serde(default)]
    pub camera: Option<Camera>,
    /// Ready-made shot along the row (grids, rows, suburbs).
    #[serde(default)]
    pub row_shot: Option<Ray>,
    #[serde(default)]
    pub preload: Option<Preload>,
    /// Pieces of a procedural cuboid house (0 for baked assets).
    #[serde(default)]
    pub pieces: u32,
}

fn yes() -> bool {
    true
}
fn one() -> u32 {
    1
}

impl Scene {
    /// An empty scene with the given bodies, colliders and arena (tests, benches).
    pub fn from_parts(bodies: Vec<Body>, colliders: Vec<Collider>, arena: &HullArena, params: SceneParams) -> Self {
        let mut s = Self {
            bodies,
            colliders,
            joints: vec![],
            hull_data: arena.records().to_vec(),
            params,
            capacity: Budget::default(),
            skipped_hulls: vec![],
            vertices: vec![],
            lines: vec![],
            projectile_vertices: vec![],
            road_vertices: vec![],
            projectile_start: 0,
            projectile_count: 0,
            projectile_radius: 0.0,
            projectile_mass: 0.0,
            projectile_speed: None,
            projectile_arc: true,
            level: None,
            selected: vec![],
            body_of: vec![],
            bond_count: 0,
            extent: 1.0,
            extent_x: 1.0,
            extent_z: 1.0,
            height: 1.0,
            center: [0.0; 3],
            copies: 1,
            instance_stride: 0,
            columns: 0,
            rows: 0,
            row_pitch: 0.0,
            column_pitch: 0.0,
            row_length: 0.0,
            grid_width: 0.0,
            grid_cell_size: 0.0,
            camera: None,
            row_shot: None,
            preload: None,
            pieces: 0,
        };
        s.projectile_start = s.bodies.len() as u32;
        s.grid_cell_size = grid_config(&s.colliders).0;
        s
    }

    /// Hull colliders (baked or cooked hulls; boxes are not counted).
    pub fn hull_count(&self) -> usize {
        self.colliders.iter().filter(|c| c.shape == SHAPE_CONVEX).count()
    }

    /// Body of fragment `id` in copy 0.
    pub fn body_of_fragment(&self, id: u32) -> Option<u32> {
        self.body_of.iter().find(|(f, _)| *f == id).map(|(_, b)| *b)
    }

    /// Drop the bonds (the lab's default: bonds are inspection lines only).
    pub fn without_joints(mut self) -> Self {
        self.joints.clear();
        self
    }

    /// Dynamic (non-ground, non-projectile) body indices.
    fn dynamic(&self, i: usize) -> bool {
        i > 0 && i < self.projectile_start as usize && self.bodies[i].inv_mass > 0.0
    }

    /// Bond preloading, step 1. Elastic bonds carry load only once they deform,
    /// so a baked structure released at rest drops onto its springs and the
    /// transient overloads bonds that hold it at rest. Instead the host settles
    /// this copy (every piece awake, breaking off with FLAG_BOND_PRELOAD) until
    /// it sleeps, then calls `settled` with the body snapshot.
    pub fn preload_variant(&self) -> Scene {
        let mut s = self.clone();
        for i in 0..s.bodies.len() {
            if s.dynamic(i) {
                s.bodies[i].flags = FLAG_DYNAMIC | FLAG_AWAKE;
            }
        }
        s.params.flags |= FLAG_BOND_PRELOAD;
        s
    }

    /// Bond preloading: settled once the bonded island sleeps (sleep keeps its
    /// bond state), checked every 60 steps, giving up after 600.
    pub fn preload_done(steps: u32, awake: u32) -> bool {
        steps >= 600 || (steps % 60 == 0 && awake == 0)
    }

    /// Bond preloading, step 2: start every piece asleep at the settled poses
    /// (`snapshot`: 28 f32 words per body, as read back from the GPU).
    pub fn settled(&self, snapshot: &[f32], steps: u32, settled: bool) -> Result<Scene> {
        if snapshot.len() < self.bodies.len() * 28 {
            bail!("Body snapshot is shorter than the scene");
        }
        let mut s = self.clone();
        for i in 0..s.bodies.len() {
            if s.dynamic(i) {
                let o = i * 28;
                let f = |k: usize| snapshot[o + k] as f64;
                let b = &mut s.bodies[i];
                b.flags = FLAG_DYNAMIC | FLAG_SLEEPING;
                b.v = [0.0; 3];
                b.w = [0.0; 3];
                b.p = [f(0), f(1), f(2)];
                b.q = [f(4), f(5), f(6), f(7)];
            }
        }
        s.preload = Some(Preload { steps, settled });
        Ok(s)
    }

    /// Benchmark aid (headless `staticUnhit`): houses outside the row shot's
    /// column become static. Changes behaviour; only for estimates. Returns
    /// the number of frozen copies.
    pub fn freeze_unhit_copies(&mut self) -> u32 {
        let (Some(shot), stride) = (self.row_shot, self.instance_stride as usize) else { return 0 };
        if stride == 0 {
            return 0;
        }
        let (x, r, ground_flags) = (shot.origin[0], self.projectile_radius, self.bodies[0].flags);
        let mut frozen = 0;
        for k in 0..self.copies as usize {
            let range = 1 + k * stride..1 + (k + 1) * stride;
            let (lo, hi) = self.bodies[range.clone()]
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), b| (lo.min(b.p[0]), hi.max(b.p[0])));
            if x + r > lo - 0.5 && x - r < hi + 0.5 {
                continue;
            }
            for b in &mut self.bodies[range] {
                b.inv_mass = 0.0;
                b.inv_i = [0.0; 3];
                b.flags = ground_flags;
            }
            frozen += 1;
        }
        frozen
    }
}

// ---- hulls -----------------------------------------------------------------------

/// Cook baked face topology, recentre it on its bounds and insert it. Returns
/// the collider's local position, record base and bound radius.
pub fn append_hull(
    arena: &mut HullArena,
    vertices: &[V3],
    sizes: &[u32],
    indices: &[u32],
    source_scale: f64,
) -> Result<(u32, f32, [f64; 3]), HullError> {
    let v: Vec<Vec3> = vertices.iter().map(|p| Vec3::from(f32x3(*p))).collect();
    let hull = ConvexHull::from_faces_in_frame(&v, sizes, indices, source_scale as f32)?;
    let (hull, offset) = hull.recentered();
    let base = arena.insert(&hull)?;
    Ok((base, hull.bounding_radius(), f64x3(offset.to_array())))
}

/// Hull arena with an engine-wide vertex limit (8..=64; None = 64).
pub fn hull_arena(max_vertices: Option<usize>) -> Result<HullArena> {
    HullArena::with_max_vertices(max_vertices.unwrap_or(HULL_MAX_VERTICES)).map_err(|e| anyhow!("{e}"))
}

/// Principal inertia frame of a row-major tensor: rotation and moments.
pub fn principal_axes(inertia: &[f32; 9]) -> Result<([f64; 4], [f64; 3])> {
    let i = |r: usize, c: usize| inertia[r * 3 + c] as f64;
    let m = DMat3::from_cols(
        DVec3::new(i(0, 0), i(1, 0), i(2, 0)),
        DVec3::new(i(0, 1), i(1, 1), i(2, 1)),
        DVec3::new(i(0, 2), i(1, 2), i(2, 2)),
    );
    let (q, moments) = engine_schema::convex::principal_axes(m);
    let moments = moments.to_array();
    if moments.iter().any(|x| !(*x > 0.0) || !x.is_finite()) {
        bail!("Fragment inertia is not positive definite");
    }
    Ok((q.to_array(), moments))
}

/// World vertices of a convex collider for a body pose.
pub fn collider_vertices(hull_data: &[[u32; 4]], c: &Collider, p: [f64; 3], q: [f64; 4]) -> Vec<[f64; 3]> {
    let base = c.hull_base() as usize;
    let nv = (hull_data[base][0] & 255) as usize;
    let s = if c.params[1] != 0.0 { c.params[1] } else { 1.0 };
    (0..nv)
        .map(|i| {
            let r = hull_data[base + 4 + i];
            let v = [f32::from_bits(r[0]) as f64, f32::from_bits(r[1]) as f64, f32::from_bits(r[2]) as f64];
            add(p, rotate(q, add(c.local_pos, rotate(c.local_rot, scale(v, s)))))
        })
        .collect()
}

/// Conservative world AABB of a collider (+0.1 mm), as `build_aabbs`.
pub fn collider_bounds(hull_data: &[[u32; 4]], c: &Collider, p: [f64; 3], q: [f64; 4]) -> ([f64; 3], [f64; 3]) {
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    if c.shape == SHAPE_CONVEX {
        for v in collider_vertices(hull_data, c, p, q) {
            for k in 0..3 {
                lo[k] = lo[k].min(v[k]);
                hi[k] = hi[k].max(v[k]);
            }
        }
    } else {
        let r = if c.shape == SHAPE_PLANE { 1e6 } else { c.params[0] };
        let at = add(p, rotate(q, c.local_pos));
        for k in 0..3 {
            lo[k] = at[k] - r;
            hi[k] = at[k] + r;
        }
    }
    (lo.map(|x| x - 0.0001), hi.map(|x| x + 0.0001))
}

/// Hierarchical broadphase sizing (`hierLevel` in Slang): base cell = median
/// collider diameter; levels cover every collider. Returns (base, levels).
pub fn grid_config(colliders: &[Collider]) -> (f64, u32) {
    // Matches build_aabbs: capsules reach radius + half height.
    let radius = |c: &Collider| {
        if c.shape == SHAPE_CAPSULE {
            c.params[0] + c.params[1]
        } else if c.params[3] > 0.0 {
            c.params[3]
        } else {
            (c.params[0] * c.params[0] + c.params[1] * c.params[1] + c.params[2] * c.params[2]).sqrt()
        }
    };
    let mut d: Vec<f64> = colliders.iter().filter(|c| c.shape != SHAPE_PLANE).map(|c| 2.0 * radius(c) + 0.0004).collect();
    d.sort_by(f64::total_cmp);
    let median = d.get(d.len() / 2).copied().filter(|x| *x != 0.0).unwrap_or(1.0);
    let base = ((median.max(0.001) * (1.0 + 1e-5)) as f32) as f64;
    let top = d.last().copied().filter(|x| *x != 0.0).unwrap_or(base);
    let levels = 2.0 + (top / base).max(1.0).log2().ceil();
    (base, levels.clamp(2.0, 31.0) as u32)
}

// ---- render streams ------------------------------------------------------------

/// Push one 10-float render vertex.
pub fn vertex(out: &mut Vec<f32>, p: [f64; 3], n: [f64; 3], color: [f64; 3], body: u32) {
    out.extend(p.iter().chain(&n).chain(&color).map(|x| *x as f32));
    out.push(body as f32);
}

/// UV sphere for one projectile slot (12 x 24 quads).
fn sphere_mesh(out: &mut Vec<f32>, radius: f64, body: u32) {
    use std::f64::consts::PI;
    let point = |u: f64, v: f64| [v.sin() * u.cos(), v.cos(), v.sin() * u.sin()];
    for y in 0..12 {
        for x in 0..24 {
            let (x0, x1) = (x as f64 / 24.0 * PI * 2.0, (x + 1) as f64 / 24.0 * PI * 2.0);
            let (y0, y1) = (y as f64 / 12.0 * PI, (y + 1) as f64 / 12.0 * PI);
            let (a, b, c, d) = (point(x0, y0), point(x1, y0), point(x1, y1), point(x0, y1));
            for n in [a, b, c, a, c, d] {
                vertex(out, scale(n, radius), n, [1.0, 0.3, 0.08], body);
            }
        }
    }
}

/// Append `PROJECTILES` inactive projectile slots: static bodies below the
/// world, each with a sphere collider. Launching gives a slot real mass.
pub fn add_projectiles(s: &mut Scene, radius: f64, spacing: f64) {
    s.projectile_start = s.bodies.len() as u32;
    s.projectile_count = PROJECTILES;
    s.projectile_radius = radius;
    s.projectile_vertices.clear();
    for slot in 0..PROJECTILES {
        let id = s.bodies.len() as u32;
        s.bodies.push(Body::fixed([slot as f64 * spacing, -1000.0, 0.0]));
        s.colliders.push(Collider::new(id, SHAPE_SPHERE, 0.0, [radius, 0.0, 0.0, 0.0]));
        sphere_mesh(&mut s.projectile_vertices, radius, id);
    }
}

// ---- baked assets ----------------------------------------------------------------

/// One hierarchy level of a baked asset, standing on the floor at the origin.
/// Fragments start asleep; each becomes a body in its principal inertia frame
/// with one convex collider per baked hull. A zero-volume child (a flat sheet
/// baked into a compound) adds no volume or mass and is skipped while its
/// fragment keeps real hulls; nothing is ever thickened. Bonds become
/// breakable welds (dropped unless the host enables bonds).
pub fn build_scene(asset: &FractureAsset, level: u8, max_hull_vertices: Option<usize>) -> Result<Scene> {
    let physics = &asset.physics;
    let selected: Vec<_> = physics.fragments.iter().filter(|f| f.level == level).collect();
    if selected.is_empty() {
        bail!("No fragments at this level");
    }
    let mut bodies = vec![Body::fixed([0.0; 3])];
    let mut colliders = vec![Collider::ground()];
    let mut arena = hull_arena(max_hull_vertices)?;
    let (mut vertices, mut lines, mut skipped) = (vec![], vec![], vec![]);
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for f in &selected {
        for h in &physics.hulls[f.hull_begin as usize..(f.hull_begin + f.hull_count) as usize] {
            for v in &h.vertices {
                for k in 0..3 {
                    lo[k] = lo[k].min(v[k] as f64);
                    hi[k] = hi[k].max(v[k] as f64);
                }
            }
        }
    }
    if !lo.iter().all(|x| x.is_finite()) {
        bail!("Selected level has no collision hulls");
    }
    let center = scale(add(lo, hi), 0.5);
    let height = hi[1] - lo[1];
    let span = sub(hi, lo);
    let extent = span[0].max(span[1]).max(span[2]);
    let placement = [-center[0], -lo[1], -center[2]];
    let mut body_of: Vec<(u32, u32, [f64; 4])> = Vec::new();
    for f in &selected {
        if !(f.mass > 0.0) || f.hull_count == 0 {
            bail!("Fragment {} has no dynamic mass or hull", f.id);
        }
        let (q, moments) = principal_axes(&f.inertia)?;
        let id = bodies.len() as u32;
        let com = f64x3(f.com);
        bodies.push(Body {
            p: add(com, placement),
            q,
            v: [0.0; 3],
            w: [0.0; 3],
            inv_mass: 1.0 / f.mass as f64,
            inv_i: moments.map(|x| 1.0 / x),
            flags: FLAG_DYNAMIC | FLAG_SLEEPING,
        });
        body_of.push((f.id, id, q));
        let mut kept = 0;
        for h in f.hull_begin..f.hull_begin + f.hull_count {
            let source = &physics.hulls[h as usize].vertices;
            let verts: Vec<V3> = source.iter().map(|v| inverse_rotate(q, sub(f64x3(*v), com))).collect();
            // The asset stores hulls as f32 in its own frame; judge convexity at that precision.
            let source_scale = source.iter().flatten().fold(0.0f64, |m, x| m.max((*x as f64).abs()));
            let faces = &physics.hull_faces[h as usize];
            let sizes: Vec<u32> = faces.sizes.iter().map(|&s| s as u32).collect();
            match append_hull(&mut arena, &verts, &sizes, &faces.indices, source_scale) {
                Ok((base, radius, local_pos)) => {
                    let density = f.mass as f64 / (f.volume as f64).max(1e-12);
                    colliders.push(Collider::hull(id, density, base, 1.0, radius, local_pos));
                    kept += 1;
                }
                Err(HullError::ZeroVolume) => skipped.push(SkippedHull { fragment: f.id, hull: h }),
                Err(e) => bail!("Fragment {}, hull {h}: {e}", f.id),
            }
        }
        if kept == 0 {
            bail!("Fragment {}: every collision hull has zero volume; recook the source collision asset", f.id);
        }
        let node = &asset.json["nodes"][f.node.max(0) as usize];
        let mesh = if f.node >= 0 { node["mesh"].as_u64().map(|m| &asset.json["meshes"][m as usize]) } else { None };
        let Some(mesh) = mesh.filter(|m| m.is_object()) else {
            bail!("Fragment {} has no render mesh", f.id);
        };
        for prim in mesh["primitives"].as_array().map_or(&[][..], Vec::as_slice) {
            if prim.get("mode").and_then(|m| m.as_u64()).unwrap_or(4) != 4 {
                bail!("Demo supports triangle meshes");
            }
            let index = |v: &serde_json::Value| v.as_u64().map(|x| x as usize);
            let positions = asset.accessor(index(&prim["attributes"]["POSITION"]).ok_or_else(|| anyhow!("Missing or sparse accessor unsupported in this demo"))?)?;
            let normals = match index(&prim["attributes"]["NORMAL"]) {
                Some(i) => Some(asset.accessor(i)?),
                None => None,
            };
            let indices: Vec<usize> = match index(&prim["indices"]) {
                Some(i) => asset.accessor(i)?.iter().map(|v| v[0] as usize).collect(),
                None => (0..positions.len()).collect(),
            };
            let factor = &asset.json["materials"][prim["material"].as_u64().unwrap_or(u64::MAX) as usize]["pbrMetallicRoughness"]["baseColorFactor"];
            let color = match factor.as_array() {
                Some(c) if c.len() >= 3 => [0, 1, 2].map(|k| c[k].as_f64().unwrap_or(0.0)),
                _ => [0.65, 0.58, 0.47],
            };
            let at = |rows: &Vec<Vec<f64>>, i: usize| [rows[i][0], rows[i][1], rows[i][2]];
            for tri in indices.chunks(3) {
                if tri.len() != 3 || tri.iter().any(|&i| i >= positions.len()) {
                    bail!("Invalid triangle indices");
                }
                let p = [0, 1, 2].map(|k| at(&positions, tri[k]));
                let normal = norm(cross(sub(p[1], p[0]), sub(p[2], p[0])));
                for (k, &i) in tri.iter().enumerate() {
                    let n = normals.as_ref().filter(|n| i < n.len()).map_or(normal, |n| at(n, i));
                    vertex(&mut vertices, inverse_rotate(q, p[k]), inverse_rotate(q, n), color, id);
                }
            }
        }
    }
    let find = |frag: i64| body_of.iter().find(|(f, _, _)| *f as i64 == frag).copied();
    let bonds: Vec<_> = physics.bonds.iter().filter(|b| b.level == level).collect();
    let mut joints = Vec::new();
    for b in &bonds {
        let (Some(a), other) = (find(b.a as i64), if b.b >= 0 { find(b.b as i64) } else { None }) else { continue };
        if b.b >= 0 && other.is_none() {
            continue;
        }
        let c = add(f64x3(b.centroid), placement);
        let qa = a.2;
        let qb = other.map_or(IDENTITY, |o| o.2);
        joints.push(Joint {
            body_a: a.1,
            body_b: other.map_or(0, |o| o.1),
            anchor_a: inverse_rotate(qa, sub(c, bodies[a.1 as usize].p)),
            anchor_b: other.map_or(c, |o| inverse_rotate(qb, sub(c, bodies[o.1 as usize].p))),
            rest: quat_mul(conjugate(qa), qb),
            normal: inverse_rotate(qa, f64x3(b.normal)),
            frame_u: inverse_rotate(qa, f64x3(b.frame_u)),
            strength: bond_capacities(asset, b, 1.0),
            k: bond_springs(asset, b),
            world: other.is_none(),
        });
    }
    let joints = scale_bond_springs(&joints, &bodies, BOND_TICK);
    for b in &bonds {
        let Some(a) = find(b.a as i64) else { continue };
        let other = find(b.b as i64);
        let color = if b.anchor { [0.95, 0.6, 0.2] } else { [0.25, 0.9, 0.8] };
        vertex(&mut lines, [0.0; 3], [0.0, 1.0, 0.0], color, a.1);
        match other {
            Some(o) => vertex(&mut lines, [0.0; 3], [0.0, 1.0, 0.0], color, o.1),
            None => vertex(&mut lines, add(f64x3(b.centroid), placement), [0.0, 1.0, 0.0], color, 0),
        }
    }
    let mass: f64 = selected.iter().map(|f| f.mass as f64).sum();
    let mut s = Scene::from_parts(bodies, colliders, &arena, SceneParams::lab(FLAG_REUSE | FLAG_WORLD_INERTIA));
    add_projectiles(&mut s, (extent * 0.04).clamp(0.08, 0.4), 2.0);
    s.projectile_mass = mass / selected.len() as f64 * 1.5;
    s.joints = joints;
    s.skipped_hulls = skipped;
    s.vertices = vertices;
    s.lines = lines;
    s.level = Some(level);
    s.selected = selected.iter().map(|f| f.id).collect();
    s.body_of = body_of.iter().map(|(f, b, _)| (*f, *b)).collect();
    s.bond_count = bonds.len() as u32;
    s.extent = extent;
    s.extent_x = span[0];
    s.extent_z = span[2];
    s.height = height;
    s.center = [0.0, height * 0.5 + placement[1] + lo[1], 0.0];
    s.grid_cell_size = grid_config(&s.colliders).0;
    Ok(s)
}

// ---- layouts -----------------------------------------------------------------------

/// A road strip between blocks, drawn on the ground (decoration only: the
/// physics ground stays the plane y = 0).
#[derive(Clone, Copy, Debug)]
pub struct Road {
    pub x0: f64,
    pub x1: f64,
    pub z0: f64,
    pub z1: f64,
    pub along_z: bool,
    pub sidewalk: f64,
}

#[derive(Clone, Debug)]
pub struct GridOptions {
    pub columns: u32,
    pub rows: u32,
    pub gap: f64,
    pub projectile_radius: f64,
    pub projectile_mass: f64,
    pub projectile_speed: f64,
    /// `[dx, dz]` per copy, replacing the regular grid.
    pub positions: Option<Vec<[f64; 2]>>,
    /// x of the row shot (default: the middle column).
    pub shot_x: Option<f64>,
    pub roads: Vec<Road>,
}

impl Default for GridOptions {
    fn default() -> Self {
        Self {
            columns: 1,
            rows: 20,
            gap: 2.5,
            projectile_radius: 1.6,
            projectile_mass: 200000.0,
            projectile_speed: 40.0,
            positions: None,
            shot_x: None,
            roads: vec![],
        }
    }
}

/// A grid of identical houses (columns along +x, rows along +z). Copies share
/// the house's hull records and render mesh (drawn instanced), so GPU memory
/// grows only with bodies and colliders. The projectile is a heavy ball rolled
/// along the ground down the middle column. `one` is the (possibly
/// bond-preloaded) single house.
pub fn build_grid(one: &Scene, o: &GridOptions) -> Scene {
    let frags = one.projectile_start as usize - 1;
    let pitch_x = one.extent_x + o.gap;
    let pitch_z = one.extent_z + o.gap;
    let offset_x = |c: f64| (c - (o.columns as f64 - 1.0) / 2.0) * pitch_x;
    let offsets: Vec<[f64; 2]> = o.positions.clone().unwrap_or_else(|| {
        (0..o.columns * o.rows)
            .map(|k| [offset_x((k % o.columns) as f64), ((k / o.columns) as f64 - (o.rows as f64 - 1.0) / 2.0) * pitch_z])
            .collect()
    });
    let span_of = |k: usize| {
        let (lo, hi) = offsets.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| (lo.min(p[k]), hi.max(p[k])));
        hi - lo
    };
    let width = span_of(0) + one.extent_x;
    let length = span_of(1) + one.extent_z;
    let mut s = one.clone();
    s.bodies = vec![one.bodies[0]];
    s.colliders = vec![one.colliders[0]];
    s.joints = vec![];
    let house: Vec<&Collider> =
        one.colliders.iter().filter(|c| c.body >= 1 && c.body < one.projectile_start).collect();
    for (k, &[dx, dz]) in offsets.iter().enumerate() {
        let base = (k * frags) as u32;
        for b in &one.bodies[1..one.projectile_start as usize] {
            s.bodies.push(Body { p: [b.p[0] + dx, b.p[1], b.p[2] + dz], ..*b });
        }
        for c in &house {
            s.colliders.push(Collider { body: c.body + base, ..**c });
        }
        // World-anchored bonds keep body 0 and move their world anchor with the copy.
        for j in &one.joints {
            s.joints.push(if j.world {
                Joint { body_a: j.body_a + base, anchor_b: [j.anchor_b[0] + dx, j.anchor_b[1], j.anchor_b[2] + dz], ..*j }
            } else {
                Joint { body_a: j.body_a + base, body_b: j.body_b + base, ..*j }
            });
        }
    }
    s.projectile_start = s.bodies.len() as u32;
    s.projectile_vertices.clear();
    let stride = one.projectile_vertices.len() / one.projectile_count.max(1) as usize;
    let (r0, r) = (one.projectile_radius, o.projectile_radius);
    for slot in 0..one.projectile_count as usize {
        let id = s.bodies.len() as u32;
        s.bodies.push(Body::fixed([slot as f64 * 4.0, -1000.0, 0.0]));
        s.colliders.push(Collider::new(id, SHAPE_SPHERE, 0.0, [r, 0.0, 0.0, 0.0]));
        for v in (0..stride).step_by(10) {
            let src = &one.projectile_vertices[slot * stride + v..slot * stride + v + 10];
            for k in 0..3 {
                s.projectile_vertices.push((src[k] as f64 / r0 * r) as f32);
            }
            s.projectile_vertices.extend_from_slice(&src[3..9]);
            s.projectile_vertices.push(id as f32);
        }
    }
    let span = width.max(length);
    let shot_x = o.shot_x.unwrap_or_else(|| offset_x((o.columns / 2) as f64));
    s.projectile_radius = r;
    s.projectile_mass = o.projectile_mass;
    s.projectile_speed = Some(o.projectile_speed);
    s.projectile_arc = false;
    s.road_vertices = road_mesh(&o.roads);
    s.copies = offsets.len() as u32;
    s.columns = o.columns;
    s.rows = o.rows;
    s.instance_stride = frags as u32;
    s.row_pitch = pitch_z;
    s.column_pitch = pitch_x;
    s.row_length = length;
    s.grid_width = width;
    s.extent = one.extent.max(span);
    s.center = [0.0, one.height * 0.5, 0.0];
    s.grid_cell_size = grid_config(&s.colliders).0;
    s.capacity = Budget::grid();
    let wide = o.columns > 1;
    s.camera = Some(Camera {
        yaw: if wide { 0.75 } else { 1.15 },
        pitch: if wide { 0.5 } else { 0.32 },
        distance: span * 0.8,
        min_distance: 3.0,
    });
    // Ready-made shot: from beyond the first house, rolling along +z at ground level.
    s.row_shot = Some(Ray { origin: [shot_x, r + 0.02, -length / 2.0 - r - 8.0], direction: [0.0, 0.0, 1.0] });
    s
}

/// A single row of `copies` houses along +z (a one-column grid).
pub fn build_row(one: &Scene, copies: u32) -> Scene {
    build_grid(one, &GridOptions { columns: 1, rows: copies, ..GridOptions::default() })
}

#[derive(Clone, Debug)]
pub struct SuburbOptions {
    pub blocks_x: u32,
    pub blocks_z: u32,
    pub lots_x: u32,
    pub lots_z: u32,
    pub yard_x: f64,
    pub yard_z: f64,
    pub road: f64,
    pub sidewalk: f64,
    pub shot_block: Option<u32>,
}

impl Default for SuburbOptions {
    fn default() -> Self {
        Self { blocks_x: 4, blocks_z: 3, lots_x: 2, lots_z: 2, yard_x: 5.0, yard_z: 6.0, road: 8.0, sidewalk: 1.5, shot_block: None }
    }
}

/// A suburb: `blocks_x x blocks_z` blocks of `lots_x x lots_z` houses with
/// yards between houses and roads (asphalt, sidewalks, dashed centre lines)
/// between blocks. The row shot rolls down one column of lots.
pub fn build_suburb(one: &Scene, o: &SuburbOptions) -> Scene {
    let lot_x = one.extent_x + o.yard_x;
    let lot_z = one.extent_z + o.yard_z;
    let pitch_x = o.lots_x as f64 * lot_x + o.road;
    let pitch_z = o.lots_z as f64 * lot_z + o.road;
    let cx = |b: f64| (b - (o.blocks_x as f64 - 1.0) / 2.0) * pitch_x;
    let cz = |b: f64| (b - (o.blocks_z as f64 - 1.0) / 2.0) * pitch_z;
    let lot = |l: u32, n: u32, size: f64| (l as f64 - (n as f64 - 1.0) / 2.0) * size;
    let mut positions = Vec::new();
    for bz in 0..o.blocks_z {
        for bx in 0..o.blocks_x {
            for lz in 0..o.lots_z {
                for lx in 0..o.lots_x {
                    positions.push([cx(bx as f64) + lot(lx, o.lots_x, lot_x), cz(bz as f64) + lot(lz, o.lots_z, lot_z)]);
                }
            }
        }
    }
    let half_w = o.blocks_x as f64 * pitch_x / 2.0;
    let half_l = o.blocks_z as f64 * pitch_z / 2.0;
    let mut roads = Vec::new();
    for i in 0..=o.blocks_x {
        let x = -half_w + i as f64 * pitch_x;
        roads.push(Road { x0: x - o.road / 2.0, x1: x + o.road / 2.0, z0: -half_l - o.road / 2.0, z1: half_l + o.road / 2.0, along_z: true, sidewalk: o.sidewalk });
    }
    for i in 0..=o.blocks_z {
        let z = -half_l + i as f64 * pitch_z;
        roads.push(Road { x0: -half_w - o.road / 2.0, x1: half_w + o.road / 2.0, z0: z - o.road / 2.0, z1: z + o.road / 2.0, along_z: false, sidewalk: o.sidewalk });
    }
    let column = o.shot_block.unwrap_or((o.blocks_x - 1) / 2);
    let mut s = build_grid(
        one,
        &GridOptions { positions: Some(positions), shot_x: Some(cx(column as f64) + lot(0, o.lots_x, lot_x)), roads, ..GridOptions::default() },
    );
    s.columns = o.blocks_x * o.lots_x;
    s.rows = o.blocks_z * o.lots_z;
    s.extent = s.extent.max(2.0 * half_w).max(2.0 * half_l);
    s
}

/// Flat road surfaces on the static ground: asphalt, sidewalks and a dashed
/// centre line broken at intersections.
pub fn road_mesh(roads: &[Road]) -> Vec<f32> {
    let mut v = Vec::new();
    let mut quad = |x0: f64, z0: f64, x1: f64, z1: f64, y: f64, c: [f64; 3]| {
        for [x, z] in [[x0, z0], [x0, z1], [x1, z1], [x0, z0], [x1, z1], [x1, z0]] {
            vertex(&mut v, [x, y, z], [0.0, 1.0, 0.0], c, 0);
        }
    };
    let (asphalt, walk, paint) = ([0.17, 0.18, 0.2], [0.42, 0.42, 0.4], [0.86, 0.72, 0.25]);
    let crosses = |r: &Road, s: &Road| r.along_z != s.along_z && s.x0 <= r.x1 && s.x1 >= r.x0 && s.z0 <= r.z1 && s.z1 >= r.z0;
    for r in roads {
        quad(r.x0, r.z0, r.x1, r.z1, 0.004, asphalt);
        let w = r.sidewalk;
        if r.along_z {
            quad(r.x0 - w, r.z0, r.x0, r.z1, 0.003, walk);
            quad(r.x1, r.z0, r.x1 + w, r.z1, 0.003, walk);
        } else {
            quad(r.x0, r.z0 - w, r.x1, r.z0, 0.003, walk);
            quad(r.x0, r.z1, r.x1, r.z1 + w, 0.003, walk);
        }
        let others: Vec<&Road> = roads.iter().filter(|s| crosses(r, s)).collect();
        let mid = if r.along_z { (r.x0 + r.x1) / 2.0 } else { (r.z0 + r.z1) / 2.0 };
        let (a, b) = if r.along_z { (r.z0, r.z1) } else { (r.x0, r.x1) };
        let mut t = a + 1.0;
        while t + 2.0 <= b {
            let blocked = others.iter().any(|s| if r.along_z { t + 2.0 >= s.z0 && t <= s.z1 } else { t + 2.0 >= s.x0 && t <= s.x1 });
            if !blocked {
                if r.along_z {
                    quad(mid - 0.08, t, mid + 0.08, t + 2.0, 0.006, paint);
                } else {
                    quad(t, mid - 0.08, t + 2.0, mid + 0.08, 0.006, paint);
                }
            }
            t += 4.0;
        }
    }
    v
}
