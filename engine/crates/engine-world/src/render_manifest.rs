//! What a renderer draws for a [`World`], without saying how: every live collider of a
//! drawable body, grouped by geometry so a renderer can draw each group in one call.
//!
//! - Boxes, spheres and capsules are listed by collider index. A renderer instances one
//!   unit primitive over the list and scales it by the collider's `params` on the GPU
//!   (box half extents; sphere radius; capsule radius and half length).
//! - Hulls and heightfields are merged into one triangle mesh per tag, each vertex tagged
//!   with its collider. Vertices are in the collider's frame, before its record's
//!   `local_pos`/`local_rot`.
//!
//! Either way a vertex reaches the world as
//! `body.position + body.orientation · (collider.local_pos + collider.local_rot · v)`,
//! read from the World's own body and collider buffers. A collider record is already in
//! its body's solver frame, so this needs no host transform and no readback: a renderer
//! (three.js on the web, Unreal over the C API) reads poses where the solver writes them.
//!
//! Bodies drawn: user bodies, threshold-destructible pieces, vehicle-owned rigid parts
//! and characters. Soft-body nodes and terrain surface patches have their own outputs
//! (skins, the deformable surface) and are left out. Planes are listed apart (a renderer
//! draws its own ground). A collider the manifest cannot describe is counted in
//! `skipped` with the reason, never silently dropped.

use crate::*;
use engine_schema::ShapeType;
use serde::Serialize;
use std::collections::{BTreeMap, HashSet};

/// The look a renderer picks for a group. Tags are names, not materials: each client
/// maps them to its own (three.js PBR, Unreal Megascans).
pub type RenderTag = &'static str;

/// Static geometry (ground, walls, terrain).
pub const TAG_GROUND: RenderTag = "ground";
/// Moving bodies with no more specific look.
pub const TAG_DEFAULT: RenderTag = "default";
/// Vehicles' rigid parts.
pub const TAG_VEHICLE: RenderTag = "vehicle";
/// Character capsules.
pub const TAG_CHARACTER: RenderTag = "character";

/// One draw group of [`RenderManifest`].
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum RenderGroup {
    /// A unit box (−1..1) instanced over `colliders`.
    #[serde(rename_all = "camelCase")]
    Box { tag: RenderTag, colliders: Vec<u32> },
    /// A unit sphere instanced over `colliders`.
    #[serde(rename_all = "camelCase")]
    Sphere { tag: RenderTag, colliders: Vec<u32> },
    /// A capsule along local Y instanced over `colliders`.
    #[serde(rename_all = "camelCase")]
    Capsule { tag: RenderTag, colliders: Vec<u32> },
    /// Merged triangles in their colliders' frames: flat xyz `positions` and `normals`,
    /// `indices` (counter-clockwise from outside), one collider per vertex.
    #[serde(rename_all = "camelCase")]
    Mesh {
        tag: RenderTag,
        positions: Vec<f32>,
        normals: Vec<f32>,
        indices: Vec<u32>,
        vertex_colliders: Vec<u32>,
    },
}

/// A plane collider: `normal · x = offset`, in world space.
#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RenderPlane {
    pub collider: u32,
    pub normal: [f32; 3],
    pub offset: f32,
}

/// What [`World::render_manifest`] returns.
#[derive(Clone, Debug, Default, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RenderManifest {
    /// [`World::render_version`] when this was made.
    pub version: u64,
    pub groups: Vec<RenderGroup>,
    pub planes: Vec<RenderPlane>,
    /// Colliders not drawn, by reason (a reason a renderer may want to know).
    pub skipped: BTreeMap<String, u32>,
}

#[derive(Default)]
struct MeshAcc {
    positions: Vec<f32>,
    normals: Vec<f32>,
    indices: Vec<u32>,
    vertex_colliders: Vec<u32>,
}

impl MeshAcc {
    /// A hull, flat shaded: each face its own vertices and normal.
    fn hull(&mut self, collider: u32, hull: &ConvexHull) {
        for f in &hull.faces {
            let n = f.normal;
            let base = self.vertex_colliders.len() as u32;
            for &i in &f.indices {
                let v = hull.vertices[usize::from(i)];
                self.positions.extend_from_slice(&[v.x, v.y, v.z]);
                self.normals.extend_from_slice(&[n.x, n.y, n.z]);
                self.vertex_colliders.push(collider);
            }
            for k in 1..f.indices.len().saturating_sub(1) as u32 {
                self.indices.extend_from_slice(&[base, base + k, base + k + 1]);
            }
        }
    }

    /// A heightfield, smooth shaded (area-weighted vertex normals).
    fn height_field(&mut self, collider: u32, mesh: &ShapeMesh) {
        let base = self.vertex_colliders.len() as u32;
        let mut normals = vec![Vec3::ZERO; mesh.vertices.len()];
        for t in &mesh.triangles {
            let [a, b, c] = t.map(|i| Vec3::from(mesh.vertices[i as usize]));
            let n = (b - a).cross(c - a);
            for &i in t {
                normals[i as usize] += n;
            }
        }
        for (v, n) in mesh.vertices.iter().zip(&normals) {
            let n = n.normalize_or(Vec3::Y);
            self.positions.extend_from_slice(v);
            self.normals.extend_from_slice(&[n.x, n.y, n.z]);
            self.vertex_colliders.push(collider);
        }
        for t in &mesh.triangles {
            self.indices.extend(t.iter().map(|i| base + i));
        }
    }
}

impl World {
    /// Changes whenever a collider is added, removed or rewritten, or the body and
    /// collider buffers are replaced: a renderer rebuilds its groups when it does.
    pub fn render_version(&self) -> u64 {
        self.sim.collider_epoch()
    }

    /// Every drawable collider, grouped for drawing (see the module docs).
    pub fn render_manifest(&self) -> RenderManifest {
        let mut out = RenderManifest {
            version: self.render_version(),
            ..Default::default()
        };
        let free: HashSet<u32> = self.free_colliders.iter().map(|&(c, _)| c).collect();
        let mut prims: BTreeMap<(u32, RenderTag), Vec<u32>> = BTreeMap::new();
        let mut meshes: BTreeMap<RenderTag, MeshAcc> = BTreeMap::new();
        let mut skip = |why: &str| *out.skipped.entry(why.to_string()).or_default() += 1;
        for c in 0..self.sim.collider_count() {
            if free.contains(&c) {
                continue;
            }
            let rec = self.sim.collider_record(c);
            let Some(body) = self.bodies.get(rec.body as usize) else {
                skip("collider of no body");
                continue;
            };
            if !body.alive {
                continue;
            }
            let tag = match body.owner {
                Owner::SoftNode => {
                    skip("soft-body node (drawn by its skin)");
                    continue;
                }
                Owner::TerrainPatch => {
                    skip("terrain surface patch (drawn by the terrain surface)");
                    continue;
                }
                Owner::Vehicle => TAG_VEHICLE,
                Owner::Character => TAG_CHARACTER,
                Owner::User | Owner::Piece if body.body_type == BodyType::Static => TAG_GROUND,
                Owner::User | Owner::Piece => TAG_DEFAULT,
            };
            let slot = self
                .collider_shape
                .get(c as usize)
                .copied()
                .filter(|&s| s != u32::MAX)
                .map(|s| &self.shapes[s as usize]);
            match rec.shape {
                s if s == ShapeType::Box as u32 => prims.entry((s, tag)).or_default().push(c),
                s if s == ShapeType::Sphere as u32 => prims.entry((s, tag)).or_default().push(c),
                s if s == ShapeType::Capsule as u32 => prims.entry((s, tag)).or_default().push(c),
                s if s == ShapeType::Plane as u32 => out.planes.push(RenderPlane {
                    collider: c,
                    normal: [rec.params[0], rec.params[1], rec.params[2]],
                    offset: rec.params[3],
                }),
                s if s == ShapeType::ConvexHull as u32 => match slot.map(|s| &s.geometry) {
                    Some(Geometry::Hull { hull, .. }) => meshes.entry(tag).or_default().hull(c, hull),
                    _ => match ConvexHull::unpack(
                        self.hulls.records(),
                        engine_schema::convex::convex_params_base(rec.params),
                    ) {
                        Ok(hull) => meshes.entry(tag).or_default().hull(c, &hull),
                        Err(_) => skip("hull the arena cannot unpack"),
                    },
                },
                s if s == ShapeType::HeightField as u32 => match slot.map(|s| &s.geometry) {
                    Some(Geometry::HeightField { field, .. }) => meshes
                        .entry(tag)
                        .or_default()
                        .height_field(c, &height_field_mesh(field, Transform::IDENTITY)),
                    _ => skip("heightfield with no geometry"),
                },
                _ => skip("unknown shape type"),
            }
        }
        for ((shape, tag), colliders) in prims {
            out.groups.push(match shape {
                s if s == ShapeType::Box as u32 => RenderGroup::Box { tag, colliders },
                s if s == ShapeType::Sphere as u32 => RenderGroup::Sphere { tag, colliders },
                _ => RenderGroup::Capsule { tag, colliders },
            });
        }
        for (tag, m) in meshes {
            out.groups.push(RenderGroup::Mesh {
                tag,
                positions: m.positions,
                normals: m.normals,
                indices: m.indices,
                vertex_colliders: m.vertex_colliders,
            });
        }
        out
    }
}
