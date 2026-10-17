//! Convex hull cooking and the packed GPU hull ABI (version 2).
//!
//! Every hull lives in one `uint4` arena (Slang `gHulls`, binding 18). Floats are
//! stored as raw bits, so indices never pass through f32 and offsets are exact.
//! A record base `H` (in 16-byte records) addresses:
//!
//! | record / word            | contents                                                        |
//! |--------------------------|-----------------------------------------------------------------|
//! | `H+0`                    | `V | F<<8 | E<<16 | flags<<24`, loop word, edge word, adj word  |
//! | `H+1`                    | local AABB centre xyz, bounding radius about the local origin   |
//! | `H+2`                    | local AABB half extents xyz, inner radius about the interior    |
//! | `H+3`                    | interior point xyz, 0                                           |
//! | `H+4 .. H+4+V`           | vertex xyz, `w` = adjacency byte start `| degree << 16`         |
//! | next `F` records         | outward plane `(n.xyz, d)` with `n·x <= d` inside               |
//! | word `(4+V+F)*4 ..`      | `F` face words: loop byte start `| count << 16`                 |
//! | loop word                | packed `u8` vertex indices, counter-clockwise about `n`         |
//! | edge word                | `E` edge words: `a | b<<8 | faceL<<16 | faceR<<24`              |
//! | adj word                 | packed `u8` incident edge indices per vertex (closed hulls)     |
//!
//! Word offsets in `H+0` are relative to `H*4`. Face `faceL` traverses `a -> b`;
//! `faceR` traverses `b -> a` (255 when the hull is not certified closed).
//! `HULL_FLAG_CLOSED` certifies reverse-edge adjacency, which enables Gauss-map
//! pruning and hill-climbing supports. Uncertified hulls use exhaustive paths.

use glam::{DMat3, DVec3, Mat3, Vec3};
use std::collections::HashMap;

pub const HULL_MAX_VERTICES: usize = 64;
pub const HULL_MAX_FACES: usize = 128;
pub const HULL_MAX_EDGES: usize = 192;
pub const HULL_MAX_FACE_VERTICES: usize = 64;
pub const HULL_FLAG_CLOSED: u32 = 1;
/// `HullArena::new` always places a unit cube (half extent 1) at this base.
/// Box colliders reuse it with a per-axis scale for box-versus-hull pairs.
pub const UNIT_BOX_HULL_BASE: u32 = 0;
/// Collider `params.x` stores `base | HULL_PARAM_TAG` as raw float bits. The tag
/// keeps the value a normal float, never a subnormal that a GPU could flush.
pub const HULL_PARAM_TAG: u32 = 0x4000_0000;
pub const HULL_MAX_BASE: u32 = 0x1FFF_FFFF;
const NO_FACE: u8 = 255;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum HullError {
    #[error("convex hull needs 4 to 64 finite vertices (got {0})")]
    VertexCount(usize),
    #[error("convex hull has non-finite coordinates")]
    NonFinite,
    #[error("degenerate zero-volume hull: recook the source collision asset")]
    ZeroVolume,
    #[error("convex face size outside 3..64")]
    FaceSize,
    #[error("invalid convex face indices")]
    FaceIndices,
    #[error("degenerate convex face")]
    DegenerateFace,
    #[error("non-convex collision hull")]
    NonConvex,
    #[error("invalid convex topology ({faces} faces, {edges} edges)")]
    Topology { faces: usize, edges: usize },
    #[error("hull arena exceeds the 29-bit base range")]
    ArenaFull,
    #[error("convex hull has {got} vertices; the configured limit is {max} (hard maximum 64). Re-export the asset with fewer hull vertices or raise the limit")]
    TooManyVertices { got: usize, max: usize },
    #[error("hull vertex limit {0} is outside 8..=64 (the unit cube at record 0 has 8)")]
    VertexLimit(usize),
}

#[derive(Clone, Debug, PartialEq)]
pub struct HullFace {
    /// Outward unit normal.
    pub normal: Vec3,
    /// Plane offset: `normal · x = d` on the face.
    pub d: f32,
    /// Vertex indices, counter-clockwise when viewed from outside.
    pub indices: Vec<u8>,
}

/// A cooked convex hull in its collider's local frame.
#[derive(Clone, Debug, PartialEq)]
pub struct ConvexHull {
    pub vertices: Vec<Vec3>,
    pub faces: Vec<HullFace>,
    /// `[a, b, faceL, faceR]`; faceL walks `a -> b`.
    pub edges: Vec<[u8; 4]>,
    /// Reverse-edge adjacency certified (every edge shared by exactly two faces).
    pub closed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HullMass {
    pub volume: f32,
    pub mass: f32,
    pub center_of_mass: Vec3,
    /// Inertia tensor about the centre of mass.
    pub inertia: Mat3,
}

impl ConvexHull {
    /// Box with the given half extents, centred on the origin.
    pub fn cuboid(half: Vec3) -> Self {
        let mut v = Vec::with_capacity(8);
        for i in 0..8 {
            v.push(Vec3::new(
                if i & 1 != 0 { half.x } else { -half.x },
                if i & 2 != 0 { half.y } else { -half.y },
                if i & 4 != 0 { half.z } else { -half.z },
            ));
        }
        Self::from_points(&v).expect("cuboid hull")
    }

    /// Quickhull-style incremental hull of a point cloud. Coplanar triangles are
    /// merged into polygon faces. A hull with more than 64 vertices is an error,
    /// never silently reduced: simplify the source asset instead.
    pub fn from_points(points: &[Vec3]) -> Result<Self, HullError> {
        if points.iter().any(|p| !p.is_finite()) {
            return Err(HullError::NonFinite);
        }
        let mut pts: Vec<DVec3> = Vec::new();
        for p in points {
            let p = p.as_dvec3();
            if !pts.contains(&p) {
                pts.push(p);
            }
        }
        if pts.len() < 4 {
            return Err(HullError::VertexCount(pts.len()));
        }
        let tris = triangle_hull(&pts)?;
        let used = hull_vertex_ids(&tris);
        if used.len() > HULL_MAX_VERTICES {
            return Err(HullError::TooManyVertices { got: used.len(), max: HULL_MAX_VERTICES });
        }
        // Compact to hull vertices only, preserving first-seen order.
        let remap: HashMap<usize, usize> = used.iter().enumerate().map(|(k, &i)| (i, k)).collect();
        let vertices: Vec<Vec3> = used.iter().map(|&i| pts[i].as_vec3()).collect();
        let indices: Vec<u32> = tris
            .iter()
            .flat_map(|t| t.iter().map(|i| remap[i] as u32))
            .collect();
        let sizes = vec![3u32; tris.len()];
        Self::from_faces(&vertices, &sizes, &indices)
    }

    /// Normalise a baked face soup (Shardwright `hullFaces`). Exact duplicate
    /// positions are welded for adjacency only; original vertex coordinates and
    /// order are kept. Coplanar patches merge into planar convex loops. When that
    /// is impossible (`DegenerateFace`), the raw faces are kept and the hull is
    /// marked uncertified so the GPU never prunes with unproven adjacency.
    pub fn from_faces(vertices: &[Vec3], sizes: &[u32], indices: &[u32]) -> Result<Self, HullError> {
        match Self::pack_faces(vertices, sizes, indices, true) {
            Err(HullError::DegenerateFace) => Self::pack_faces(vertices, sizes, indices, false),
            other => other,
        }
    }

    fn pack_faces(
        vertices: &[Vec3],
        sizes: &[u32],
        indices: &[u32],
        normalize: bool,
    ) -> Result<Self, HullError> {
        if vertices.len() < 4 || vertices.len() > HULL_MAX_VERTICES {
            return Err(HullError::VertexCount(vertices.len()));
        }
        if vertices.iter().any(|v| !v.is_finite()) {
            return Err(HullError::NonFinite);
        }
        let v64: Vec<DVec3> = vertices.iter().map(|v| v.as_dvec3()).collect();
        let center = v64.iter().copied().sum::<DVec3>() / v64.len() as f64;
        let lo = v64.iter().copied().fold(DVec3::splat(f64::INFINITY), DVec3::min);
        let hi = v64.iter().copied().fold(DVec3::splat(f64::NEG_INFINITY), DVec3::max);
        // Weld exact duplicate positions (bitwise) for adjacency only.
        let mut first: HashMap<[u32; 3], usize> = HashMap::new();
        let ids: Vec<usize> = vertices
            .iter()
            .enumerate()
            .map(|(i, v)| *first.entry([v.x.to_bits(), v.y.to_bits(), v.z.to_bits()]).or_insert(i))
            .collect();
        let canonical: Vec<usize> = {
            let mut c: Vec<usize> = first.values().copied().collect();
            c.sort_unstable();
            c
        };
        let span = (hi - lo).length();
        let rel: Vec<DVec3> = v64.iter().map(|v| *v - center).collect();
        let axis = rel.iter().copied().fold(DVec3::ZERO, |a, b| if a.length_squared() >= b.length_squared() { a } else { b });
        let plane = rel
            .iter()
            .map(|v| axis.cross(*v))
            .fold(DVec3::ZERO, |a, b| if a.length_squared() >= b.length_squared() { a } else { b });
        let plane_len = plane.length();
        let thickness = if plane_len > 0.0 {
            rel.iter().map(|v| v.dot(plane).abs() / plane_len).fold(0.0, f64::max)
        } else {
            0.0
        };
        if plane_len == 0.0 || thickness <= (span * 1e-8).max(1e-10) {
            return Err(HullError::ZeroVolume);
        }
        let tol = (span * 1e-6).max(1e-9);
        // Convexity is judged against the input's own f32 precision: assets store
        // coordinates as f32 in their frame (metres from its origin), so plane
        // offsets carry ~|x|·2^-24 rounding per coordinate, amplified by small
        // faces. Only the check widens; face merging and packed bytes are unchanged.
        let max_abs = v64.iter().map(|p| p.abs().max_element()).fold(0.0f64, f64::max);
        let convex_tol = tol.max(max_abs * (2.0f64).powi(-20));

        struct Raw {
            ids: Vec<usize>,
            n: DVec3,
        }
        let mut planes: Vec<(DVec3, f64)> = Vec::new();
        let mut raws: Vec<Raw> = Vec::new();
        let mut offset = 0usize;
        for &count in sizes {
            let count = count as usize;
            if !(3..=HULL_MAX_FACE_VERTICES).contains(&count) {
                return Err(HullError::FaceSize);
            }
            let face = indices.get(offset..offset + count).ok_or(HullError::FaceIndices)?;
            offset += count;
            let mut f = Vec::with_capacity(count);
            for &i in face {
                let i = i as usize;
                if i >= vertices.len() {
                    return Err(HullError::FaceIndices);
                }
                f.push(ids[i]);
            }
            let mut sum = DVec3::ZERO;
            for k in 1..count - 1 {
                sum += (v64[f[k]] - v64[f[0]]).cross(v64[f[k + 1]] - v64[f[0]]);
            }
            if sum.length() < 1e-14 {
                continue;
            }
            let mut n = sum.normalize();
            if n.dot(v64[f[0]] - center) < 0.0 {
                n = -n;
            }
            let mut clean: Vec<usize> = Vec::new();
            for &i in &f {
                if !clean.contains(&i) {
                    clean.push(i);
                }
            }
            if sum.dot(n) < 0.0 {
                clean.reverse();
            }
            let d = n.dot(v64[f[0]]);
            if !planes.iter().any(|(pn, pd)| pn.dot(n) > 1.0 - 1e-7 && (pd - d).abs() < tol) {
                planes.push((n, d));
            }
            raws.push(Raw { ids: clean, n });
        }
        if offset != indices.len() {
            return Err(HullError::FaceIndices);
        }

        let mut faces: Vec<(Vec<usize>, DVec3)> = Vec::new();
        let entries: Vec<(DVec3, f64, Option<Vec<usize>>)> = if normalize {
            planes.iter().map(|&(n, d)| (n, d, None)).collect()
        } else {
            raws.iter().map(|r| (r.n, r.n.dot(v64[r.ids[0]]), Some(r.ids.clone()))).collect()
        };
        for (n, d, raw_ids) in entries {
            let loop_ids = match raw_ids {
                Some(ids) => ids,
                None => {
                    let u = n
                        .cross(if n.x.abs() < 0.8 { DVec3::X } else { DVec3::Y })
                        .normalize();
                    let v = n.cross(u);
                    let mut pts: Vec<(usize, f64, f64)> = canonical
                        .iter()
                        .filter(|&&i| (n.dot(v64[i]) - d).abs() < tol)
                        .map(|&i| (i, v64[i].dot(u), v64[i].dot(v)))
                        .collect();
                    pts.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.2.total_cmp(&b.2)));
                    planar_convex_loop(&pts)
                }
            };
            if loop_ids.len() < 3 {
                return Err(HullError::DegenerateFace);
            }
            if normalize && v64.iter().any(|p| n.dot(*p) - d > convex_tol * 2.0) {
                return Err(HullError::NonConvex);
            }
            faces.push((loop_ids, n));
        }
        if faces.len() < 4 {
            return Err(HullError::ZeroVolume);
        }

        // Directed edges: faceL walks a -> b, its twin must walk b -> a.
        let mut closed = normalize;
        let mut edge_map: HashMap<(usize, usize), usize> = HashMap::new();
        let mut edges: Vec<(usize, usize, usize, Option<usize>)> = Vec::new();
        for (fi, (ids, _)) in faces.iter().enumerate() {
            for k in 0..ids.len() {
                let a = ids[k];
                let b = ids[(k + 1) % ids.len()];
                let key = (a.min(b), a.max(b));
                match edge_map.get(&key) {
                    Some(&e) => {
                        let edge = &mut edges[e];
                        if edge.3.is_some() || edge.0 != b || edge.1 != a {
                            closed = false;
                        }
                        if edge.3.is_none() {
                            edge.3 = Some(fi);
                        }
                    }
                    None => {
                        edge_map.insert(key, edges.len());
                        edges.push((a, b, fi, None));
                    }
                }
            }
        }
        if faces.len() > HULL_MAX_FACES || edges.len() > HULL_MAX_EDGES {
            return Err(HullError::Topology { faces: faces.len(), edges: edges.len() });
        }
        if edges.iter().any(|e| e.3.is_none()) {
            closed = false;
        }
        Ok(Self {
            vertices: vertices.to_vec(),
            faces: faces
                .iter()
                .map(|(ids, n)| HullFace {
                    normal: n.as_vec3(),
                    d: n.dot(v64[ids[0]]) as f32,
                    indices: ids.iter().map(|&i| i as u8).collect(),
                })
                .collect(),
            edges: edges
                .iter()
                .map(|e| [e.0 as u8, e.1 as u8, e.2 as u8, e.3.map_or(NO_FACE, |f| f as u8)])
                .collect(),
            closed,
        })
    }

    pub fn local_bounds(&self) -> (Vec3, Vec3) {
        let lo = self.vertices.iter().copied().fold(Vec3::splat(f32::INFINITY), Vec3::min);
        let hi = self.vertices.iter().copied().fold(Vec3::splat(f32::NEG_INFINITY), Vec3::max);
        (lo, hi)
    }

    /// Mean of the vertices: strictly interior for a valid hull.
    pub fn interior(&self) -> Vec3 {
        self.vertices.iter().copied().sum::<Vec3>() / self.vertices.len() as f32
    }

    /// Largest vertex distance from the local origin.
    pub fn bounding_radius(&self) -> f32 {
        self.vertices.iter().map(|v| v.length()).fold(0.0, f32::max)
    }

    /// Shift vertices so the local AABB is centred on the origin. Returns the
    /// offset to add to the collider's `local_pos`. Compound children cooked in
    /// a body frame must be recentred, or their bound radius covers the body.
    pub fn recentered(&self) -> (Self, Vec3) {
        let (lo, hi) = self.local_bounds();
        let c = 0.5 * (lo + hi);
        let mut out = self.clone();
        for v in &mut out.vertices {
            *v -= c;
        }
        for f in &mut out.faces {
            f.d -= f.normal.dot(c);
        }
        (out, c)
    }

    /// Exact polyhedral volume, centre of mass and inertia about the COM.
    pub fn mass_properties(&self, density: f32) -> HullMass {
        let o = self.interior().as_dvec3();
        let canon = DMat3::from_cols(
            DVec3::new(2.0, 1.0, 1.0),
            DVec3::new(1.0, 2.0, 1.0),
            DVec3::new(1.0, 1.0, 2.0),
        ) * (1.0 / 120.0);
        let mut volume = 0.0f64;
        let mut first = DVec3::ZERO;
        let mut cov = DMat3::ZERO;
        for f in &self.faces {
            let p0 = self.vertices[f.indices[0] as usize].as_dvec3() - o;
            for k in 1..f.indices.len() - 1 {
                let p1 = self.vertices[f.indices[k] as usize].as_dvec3() - o;
                let p2 = self.vertices[f.indices[k + 1] as usize].as_dvec3() - o;
                let a = DMat3::from_cols(p0, p1, p2);
                let det = a.determinant();
                volume += det / 6.0;
                first += det / 24.0 * (p0 + p1 + p2);
                cov += a * canon * a.transpose() * det;
            }
        }
        let com_rel = if volume.abs() > 1e-30 { first / volume } else { DVec3::ZERO };
        // Shift the second moment from `o` to the centre of mass.
        let outer = DMat3::from_cols(com_rel * com_rel.x, com_rel * com_rel.y, com_rel * com_rel.z);
        let c = cov - outer * volume;
        let rho = density as f64;
        let trace = c.x_axis.x + c.y_axis.y + c.z_axis.z;
        let inertia = (DMat3::from_diagonal(DVec3::splat(trace)) - c) * rho;
        HullMass {
            volume: volume as f32,
            mass: (volume * rho) as f32,
            center_of_mass: (o + com_rel).as_vec3(),
            inertia: inertia.as_mat3(),
        }
    }

    /// Records this hull occupies in the arena.
    pub fn packed_records(&self) -> usize {
        let v = self.vertices.len();
        let f = self.faces.len();
        let loop_bytes: usize = self.faces.iter().map(|x| x.indices.len()).sum();
        let adj_bytes = if self.closed { 2 * self.edges.len() } else { 0 };
        let words = f + loop_bytes.div_ceil(4) + self.edges.len() + adj_bytes.div_ceil(4);
        4 + v + f + words.div_ceil(4)
    }

    /// Append the v2 GPU encoding. Returns the record base.
    pub fn pack(&self, out: &mut Vec<[u32; 4]>) -> Result<u32, HullError> {
        let base = out.len();
        let records = self.packed_records();
        if base + records > HULL_MAX_BASE as usize {
            return Err(HullError::ArenaFull);
        }
        let v = self.vertices.len();
        let f = self.faces.len();
        let e = self.edges.len();
        let f4 = |x: f32| x.to_bits();
        let (lo, hi) = self.local_bounds();
        let center = 0.5 * (lo + hi);
        let half = 0.5 * (hi - lo);
        let interior = self.interior();
        let inner = self
            .faces
            .iter()
            .map(|x| x.d - x.normal.dot(interior))
            .fold(f32::INFINITY, f32::min)
            .max(0.0);

        // Word sections, relative to base * 4.
        let face_word = (4 + v + f) * 4;
        let loop_word = face_word + f;
        let loop_bytes: usize = self.faces.iter().map(|x| x.indices.len()).sum();
        let edge_word = loop_word + loop_bytes.div_ceil(4);
        let adj_word = edge_word + e;
        let adj_bytes = if self.closed { 2 * e } else { 0 };
        let total_words = adj_word + adj_bytes.div_ceil(4);
        debug_assert_eq!(total_words.div_ceil(4), records);
        let mut words = vec![0u32; records * 4];

        let flags = if self.closed { HULL_FLAG_CLOSED } else { 0 };
        words[0] = v as u32 | (f as u32) << 8 | (e as u32) << 16 | flags << 24;
        words[1] = loop_word as u32;
        words[2] = edge_word as u32;
        words[3] = adj_word as u32;
        let put4 = |w: &mut [u32], at: usize, x: Vec3, last: u32| {
            w[at] = f4(x.x);
            w[at + 1] = f4(x.y);
            w[at + 2] = f4(x.z);
            w[at + 3] = last;
        };
        put4(&mut words, 4, center, f4(self.bounding_radius()));
        put4(&mut words, 8, half, f4(inner));
        put4(&mut words, 12, interior, 0);

        // Vertex adjacency (closed hulls): incident edge indices per vertex.
        let mut incident: Vec<Vec<u8>> = vec![Vec::new(); v];
        if self.closed {
            for (k, edge) in self.edges.iter().enumerate() {
                incident[edge[0] as usize].push(k as u8);
                incident[edge[1] as usize].push(k as u8);
            }
        }
        let mut adj_start = 0usize;
        for (i, p) in self.vertices.iter().enumerate() {
            let deg = incident[i].len();
            put4(&mut words, (4 + i) * 4, *p, (adj_start as u32) | (deg as u32) << 16);
            adj_start += deg;
        }
        for (k, face) in self.faces.iter().enumerate() {
            put4(&mut words, (4 + v + k) * 4, face.normal, f4(face.d));
        }
        let put_byte = |w: &mut [u32], word: usize, byte: usize, value: u8| {
            w[word + byte / 4] |= (value as u32) << ((byte % 4) * 8);
        };
        let mut start = 0usize;
        for (k, face) in self.faces.iter().enumerate() {
            words[face_word + k] = start as u32 | (face.indices.len() as u32) << 16;
            for &i in &face.indices {
                put_byte(&mut words, loop_word, start, i);
                start += 1;
            }
        }
        for (k, edge) in self.edges.iter().enumerate() {
            words[edge_word + k] = edge[0] as u32
                | (edge[1] as u32) << 8
                | (edge[2] as u32) << 16
                | (edge[3] as u32) << 24;
        }
        let mut at = 0usize;
        for list in &incident {
            for &k in list {
                put_byte(&mut words, adj_word, at, k);
                at += 1;
            }
        }
        for r in words.chunks_exact(4) {
            out.push([r[0], r[1], r[2], r[3]]);
        }
        Ok(base as u32)
    }
}

/// Collider `params` for a convex hull at `base` with a uniform `scale`.
/// `params.w` is the bounding radius used for grid sizing and sleep heuristics.
pub fn convex_collider_params(base: u32, scale: f32, bounding_radius: f32) -> [f32; 4] {
    assert!(base <= HULL_MAX_BASE, "hull base exceeds the tagged range");
    [f32::from_bits(base | HULL_PARAM_TAG), scale, 0.0, bounding_radius * scale]
}

/// Decode `convex_collider_params` (host-side mirror of the Slang accessor).
pub fn convex_params_base(params: [f32; 4]) -> u32 {
    params[0].to_bits() & HULL_MAX_BASE
}

/// Hull arena with runtime management: identical hulls are shared and
/// reference-counted, released blocks go to a first-fit free list (adjacent free
/// blocks merge), and every write is recorded as a dirty record range so hosts
/// upload only what changed. Hosts grow the GPU buffer when `len()` exceeds it.
/// The unit cube at `UNIT_BOX_HULL_BASE` is permanent (box/hull pairs use it).
#[derive(Clone, Debug)]
pub struct HullArena {
    records: Vec<[u32; 4]>,
    dedup: HashMap<Vec<u32>, u32>,
    blocks: HashMap<u32, HullBlock>,
    /// Sorted, non-adjacent free `(start, len)` record ranges.
    free: Vec<(u32, u32)>,
    dirty: Vec<std::ops::Range<u32>>,
    /// Engine-wide vertex limit (8..=64); `insert` rejects larger hulls.
    max_vertices: usize,
}

#[derive(Clone, Debug)]
struct HullBlock {
    len: u32,
    refs: u32,
    key: Vec<u32>,
}

impl Default for HullArena {
    fn default() -> Self {
        Self::new()
    }
}

impl HullArena {
    pub fn new() -> Self {
        Self::with_max_vertices(HULL_MAX_VERTICES).expect("hard maximum is a valid limit")
    }

    /// An arena whose hulls may have at most `max_vertices` vertices (8..=64).
    /// Scenes known to stay small (e.g. assets baked with a 16-vertex cap) set
    /// it so an oversized hull fails loudly at insert instead of running on a
    /// slower path; nothing is ever reduced to fit.
    pub fn with_max_vertices(max_vertices: usize) -> Result<Self, HullError> {
        if !(8..=HULL_MAX_VERTICES).contains(&max_vertices) {
            return Err(HullError::VertexLimit(max_vertices));
        }
        let mut arena = Self {
            records: Vec::new(),
            dedup: HashMap::new(),
            blocks: HashMap::new(),
            free: Vec::new(),
            dirty: Vec::new(),
            max_vertices,
        };
        let base = arena
            .insert(&ConvexHull::cuboid(Vec3::ONE))
            .expect("unit cube fits");
        debug_assert_eq!(base, UNIT_BOX_HULL_BASE);
        Ok(arena)
    }

    /// The configured vertex limit.
    pub fn max_vertices(&self) -> usize {
        self.max_vertices
    }

    /// Insert (or share an identical) hull; returns its record base. Each call
    /// takes one reference, released with `release`.
    pub fn insert(&mut self, hull: &ConvexHull) -> Result<u32, HullError> {
        if hull.vertices.len() > self.max_vertices {
            return Err(HullError::TooManyVertices { got: hull.vertices.len(), max: self.max_vertices });
        }
        let mut scratch = Vec::new();
        hull.pack(&mut scratch)?;
        let key: Vec<u32> = scratch.iter().flatten().copied().collect();
        if let Some(&base) = self.dedup.get(&key) {
            self.blocks.get_mut(&base).expect("live block").refs += 1;
            return Ok(base);
        }
        let len = scratch.len() as u32;
        let base = self.allocate(len)?;
        self.records[base as usize..(base + len) as usize].copy_from_slice(&scratch);
        self.mark_dirty(base..base + len);
        self.dedup.insert(key.clone(), base);
        self.blocks.insert(base, HullBlock { len, refs: 1, key });
        Ok(base)
    }

    /// Drop one reference. Returns true when the block was freed (its records may
    /// be reused by later inserts; colliders must no longer point at it).
    pub fn release(&mut self, base: u32) -> bool {
        if base == UNIT_BOX_HULL_BASE {
            return false;
        }
        let Some(block) = self.blocks.get_mut(&base) else {
            return false;
        };
        block.refs -= 1;
        if block.refs > 0 {
            return false;
        }
        let block = self.blocks.remove(&base).unwrap();
        self.dedup.remove(&block.key);
        self.free_range(base, block.len);
        true
    }

    /// References currently held on the hull at `base` (0 if none).
    pub fn refs(&self, base: u32) -> u32 {
        self.blocks.get(&base).map_or(0, |b| b.refs)
    }

    /// Record ranges written since the last call, merged and sorted.
    pub fn take_dirty(&mut self) -> Vec<std::ops::Range<u32>> {
        let mut d = std::mem::take(&mut self.dirty);
        d.sort_by_key(|r| r.start);
        let mut out: Vec<std::ops::Range<u32>> = Vec::new();
        for r in d {
            match out.last_mut() {
                Some(last) if r.start <= last.end => last.end = last.end.max(r.end),
                _ => out.push(r),
            }
        }
        out
    }

    /// Mark everything dirty (e.g. after a host recreated its GPU buffer).
    #[allow(clippy::single_range_in_vec_init)] // one dirty range covering the arena
    pub fn mark_all_dirty(&mut self) {
        self.dirty = vec![0..self.records.len() as u32];
    }

    pub fn records(&self) -> &[[u32; 4]] {
        &self.records
    }

    /// Records in use, including free holes (the GPU buffer must hold this many).
    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Free records inside the arena (fragmentation).
    pub fn free_records(&self) -> u32 {
        self.free.iter().map(|f| f.1).sum()
    }

    fn mark_dirty(&mut self, r: std::ops::Range<u32>) {
        self.dirty.push(r);
    }

    fn allocate(&mut self, len: u32) -> Result<u32, HullError> {
        if let Some(k) = self.free.iter().position(|&(_, n)| n >= len) {
            let (start, n) = self.free[k];
            if n == len {
                self.free.remove(k);
            } else {
                self.free[k] = (start + len, n - len);
            }
            return Ok(start);
        }
        let base = self.records.len() as u32;
        if base as u64 + len as u64 > HULL_MAX_BASE as u64 {
            return Err(HullError::ArenaFull);
        }
        self.records.resize(self.records.len() + len as usize, [0; 4]);
        Ok(base)
    }

    fn free_range(&mut self, start: u32, len: u32) {
        let k = self.free.partition_point(|&(s, _)| s < start);
        self.free.insert(k, (start, len));
        // Merge with the following and preceding neighbours.
        if k + 1 < self.free.len() && self.free[k].0 + self.free[k].1 == self.free[k + 1].0 {
            self.free[k].1 += self.free[k + 1].1;
            self.free.remove(k + 1);
        }
        if k > 0 && self.free[k - 1].0 + self.free[k - 1].1 == self.free[k].0 {
            self.free[k - 1].1 += self.free[k].1;
            self.free.remove(k);
        }
        // A free tail shrinks the arena instead of staying a hole.
        if let Some(&(s, n)) = self.free.last() {
            if (s + n) as usize == self.records.len() {
                self.free.pop();
                self.records.truncate(s as usize);
            }
        }
    }
}

/// Per-collider collision filter, packed into `Collider::generation` (categories
/// and mask, 16 bits each; 0 = every category, collides with everything) and the
/// upper 16 bits of `Collider::material` (signed group: a shared positive group
/// always collides, a shared negative group never does, overriding categories).
/// Hosts enable the GPU test when any collider carries a filter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CollisionFilter {
    pub categories: u16,
    pub mask: u16,
    pub group: i16,
}

impl Default for CollisionFilter {
    fn default() -> Self {
        Self { categories: 0xFFFF, mask: 0xFFFF, group: 0 }
    }
}

impl CollisionFilter {
    pub fn apply(self, c: &mut crate::Collider) {
        c.generation = if self.categories == 0xFFFF && self.mask == 0xFFFF {
            0
        } else {
            self.categories as u32 | (self.mask as u32) << 16
        };
        c.material = (c.material & 0xFFFF) | ((self.group as u16 as u32) << 16);
    }

    pub fn of(c: &crate::Collider) -> Self {
        let f = if c.generation == 0 { u32::MAX } else { c.generation };
        Self { categories: f as u16, mask: (f >> 16) as u16, group: (c.material >> 16) as u16 as i16 }
    }

    /// Host mirror of the GPU test: true when the pair may collide.
    pub fn allows(a: &crate::Collider, b: &crate::Collider) -> bool {
        let (fa, fb) = (Self::of(a), Self::of(b));
        if fa.group != 0 && fa.group == fb.group {
            return fa.group > 0;
        }
        (fa.categories & fb.mask) != 0 && (fb.categories & fa.mask) != 0
    }

    /// True when the collider carries any non-default filter.
    pub fn is_set(c: &crate::Collider) -> bool {
        c.generation != 0 || c.material >> 16 != 0
    }
}

// ---- rigid bodies from hulls --------------------------------------------------------

/// One collider of a cooked body: a hull recentred on its own bounds, placed at
/// `local_pos` in the body's principal frame (identity local rotation).
#[derive(Clone, Debug)]
pub struct CookedChild {
    pub hull: ConvexHull,
    pub local_pos: Vec3,
}

/// Mass properties and colliders for one rigid body made of convex hulls.
/// The body frame is the compound COM with principal axes, because the GPU body
/// ABI stores a diagonal inverse inertia.
#[derive(Clone, Debug)]
pub struct CookedBody {
    pub mass: f32,
    /// Principal moments about the COM.
    pub inertia: Vec3,
    /// Input-frame COM.
    pub center_of_mass: Vec3,
    /// Input frame -> principal frame rotation: `world = pose * (com + principal * p)`.
    pub principal: glam::Quat,
    pub children: Vec<CookedChild>,
}

impl CookedBody {
    /// World pose of the body for an input-frame pose `(position, rotation)`.
    pub fn body_pose(&self, position: Vec3, rotation: glam::Quat) -> (Vec3, glam::Quat) {
        (position + rotation * self.center_of_mass, (rotation * self.principal).normalize())
    }
}

/// Combine hulls (all in one input frame) into a single rigid body of uniform
/// density. Compound mass and inertia belong to the body, never to children.
pub fn cook_body(hulls: &[ConvexHull], density: f32) -> CookedBody {
    assert!(!hulls.is_empty(), "a body needs at least one hull");
    let parts: Vec<HullMass> = hulls.iter().map(|h| h.mass_properties(density)).collect();
    let mass: f64 = parts.iter().map(|m| m.mass as f64).sum();
    let com: DVec3 = parts
        .iter()
        .map(|m| m.center_of_mass.as_dvec3() * m.mass as f64)
        .sum::<DVec3>()
        / mass.max(1e-30);
    // Parallel-axis theorem with full tensors.
    let mut inertia = DMat3::ZERO;
    for m in &parts {
        let d = m.center_of_mass.as_dvec3() - com;
        let shift = DMat3::from_diagonal(DVec3::splat(d.length_squared()))
            - DMat3::from_cols(d * d.x, d * d.y, d * d.z);
        inertia += m.inertia.as_dmat3() + shift * m.mass as f64;
    }
    let (rotation, moments) = principal_axes(inertia);
    let to_body = rotation.inverse();
    let children = hulls
        .iter()
        .map(|h| {
            let mut local = h.clone();
            for v in &mut local.vertices {
                *v = (to_body * (v.as_dvec3() - com)).as_vec3();
            }
            for f in &mut local.faces {
                let n = (to_body * f.normal.as_dvec3()).normalize();
                let p = local.vertices[f.indices[0] as usize].as_dvec3();
                f.normal = n.as_vec3();
                f.d = n.dot(p) as f32;
            }
            let (hull, local_pos) = local.recentered();
            CookedChild { hull, local_pos }
        })
        .collect();
    CookedBody {
        mass: mass as f32,
        inertia: moments.as_vec3(),
        center_of_mass: com.as_vec3(),
        principal: rotation.as_quat().normalize(),
        children,
    }
}

/// Jacobi eigen-decomposition of a symmetric tensor: `I = R diag(m) Rᵀ`.
pub fn principal_axes(inertia: DMat3) -> (glam::DQuat, DVec3) {
    let mut a = [
        [inertia.x_axis.x, inertia.y_axis.x, inertia.z_axis.x],
        [inertia.x_axis.y, inertia.y_axis.y, inertia.z_axis.y],
        [inertia.x_axis.z, inertia.y_axis.z, inertia.z_axis.z],
    ];
    let mut v = [[1.0f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..32 {
        let (mut p, mut q) = (0usize, 1usize);
        for (i, j) in [(0usize, 2usize), (1, 2)] {
            if a[i][j].abs() > a[p][q].abs() {
                p = i;
                q = j;
            }
        }
        let scale = a[0][0].abs().max(a[1][1].abs()).max(a[2][2].abs()).max(1e-300);
        if a[p][q].abs() <= 1e-14 * scale {
            break;
        }
        let theta = 0.5 * (2.0 * a[p][q]).atan2(a[q][q] - a[p][p]);
        let (s, c) = theta.sin_cos();
        let (app, aqq, apq) = (a[p][p], a[q][q], a[p][q]);
        #[allow(clippy::needless_range_loop)] // symmetric updates touch a[k][*] and a[*][k]
        for k in 0..3 {
            if k != p && k != q {
                let (kp, kq) = (a[k][p], a[k][q]);
                a[k][p] = c * kp - s * kq;
                a[p][k] = a[k][p];
                a[k][q] = s * kp + c * kq;
                a[q][k] = a[k][q];
            }
        }
        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for row in &mut v {
            let (kp, kq) = (row[p], row[q]);
            row[p] = c * kp - s * kq;
            row[q] = s * kp + c * kq;
        }
    }
    // Columns of v are the eigenvectors; keep a proper rotation.
    let mut m = DMat3::from_cols(
        DVec3::new(v[0][0], v[1][0], v[2][0]),
        DVec3::new(v[0][1], v[1][1], v[2][1]),
        DVec3::new(v[0][2], v[1][2], v[2][2]),
    );
    if m.determinant() < 0.0 {
        m.z_axis = -m.z_axis;
    }
    (glam::DQuat::from_mat3(&m).normalize(), DVec3::new(a[0][0], a[1][1], a[2][2]))
}

// ---- broadphase sizing ------------------------------------------------------------

/// crate::Collider bound radius as the GPU stores it in `Aabb._pad0`.
pub fn collider_radius(c: &crate::Collider) -> f32 {
    // Matches build_aabbs: the grid needs every AABB half extent within it.
    if c.shape == crate::ShapeType::Capsule as u32 {
        c.params[0] + c.params[1]
    } else if c.params[3] > 0.0 {
        c.params[3]
    } else {
        (c.params[0] * c.params[0] + c.params[1] * c.params[1] + c.params[2] * c.params[2]).sqrt()
    }
}

/// Hierarchical grid: base cell = median collider diameter, enough levels to
/// cover every dynamic collider. Oversized statics beyond the top level use the
/// explicit 10-entry large list; when there are more, the levels cover them too.
/// Mirrors `hierLevel` in world_step.slang and `gridConfig` in fracture/geometry.mjs.
pub fn hierarchical_grid(cols: &[crate::Collider], bodies: &[crate::BodyState]) -> (f32, u32) {
    let mut diameters: Vec<f32> = cols
        .iter()
        .filter(|c| c.shape != crate::ShapeType::Plane as u32)
        .map(|c| 2.0 * collider_radius(c) + 0.0004)
        .collect();
    if diameters.is_empty() {
        return (1.0, 2);
    }
    // O(n) median (three-avbd uses Wirth selection for the same reason).
    let mid = diameters.len() / 2;
    let largest = diameters.iter().copied().fold(0.0f32, f32::max);
    let median = *diameters.select_nth_unstable_by(mid, f32::total_cmp).1;
    let base = (median.max(0.001) * (1.0 + 1e-5)).max(0.001);
    let level = |d: f32| ((d / base).max(1.0)).log2().ceil().max(0.0) as u32;
    let dynamic_max = cols
        .iter()
        .filter(|c| c.shape != crate::ShapeType::Plane as u32)
        .filter(|c| bodies.get(c.body as usize).is_some_and(|b| b.inv_mass > 0.0))
        .map(|c| level(2.0 * collider_radius(c) + 0.0004))
        .max()
        .unwrap_or(0);
    let mut levels = (dynamic_max + 2).max(2);
    let oversized = diameters.iter().filter(|&&d| level(d) >= levels).count();
    if oversized > 10 {
        levels = level(largest) + 1;
    }
    (base, levels.min(31))
}

// ---- triangle hull construction -------------------------------------------------

fn hull_vertex_ids(tris: &[[usize; 3]]) -> Vec<usize> {
    let mut used = Vec::new();
    for t in tris {
        for &i in t {
            if !used.contains(&i) {
                used.push(i);
            }
        }
    }
    used
}

/// Incremental 3D hull returning outward counter-clockwise triangles.
fn triangle_hull(pts: &[DVec3]) -> Result<Vec<[usize; 3]>, HullError> {
    let lo = pts.iter().copied().fold(DVec3::splat(f64::INFINITY), DVec3::min);
    let hi = pts.iter().copied().fold(DVec3::splat(f64::NEG_INFINITY), DVec3::max);
    let eps = (hi - lo).length() * 1e-9;
    // Initial simplex: extreme pair, farthest from line, farthest from plane.
    let a = (0..pts.len()).min_by(|&i, &j| pts[i].x.total_cmp(&pts[j].x)).unwrap();
    let b = (0..pts.len())
        .max_by(|&i, &j| (pts[i] - pts[a]).length_squared().total_cmp(&(pts[j] - pts[a]).length_squared()))
        .unwrap();
    let ab = pts[b] - pts[a];
    let c = (0..pts.len())
        .max_by(|&i, &j| ab.cross(pts[i] - pts[a]).length_squared().total_cmp(&ab.cross(pts[j] - pts[a]).length_squared()))
        .unwrap();
    let n = ab.cross(pts[c] - pts[a]);
    if n.length() <= eps * eps {
        return Err(HullError::ZeroVolume);
    }
    let d = (0..pts.len())
        .max_by(|&i, &j| n.dot(pts[i] - pts[a]).abs().total_cmp(&n.dot(pts[j] - pts[a]).abs()))
        .unwrap();
    if n.dot(pts[d] - pts[a]).abs() / n.length() <= eps {
        return Err(HullError::ZeroVolume);
    }
    let centroid = (pts[a] + pts[b] + pts[c] + pts[d]) * 0.25;
    let orient = |t: [usize; 3]| -> [usize; 3] {
        let n = (pts[t[1]] - pts[t[0]]).cross(pts[t[2]] - pts[t[0]]);
        if n.dot(pts[t[0]] - centroid) < 0.0 {
            [t[0], t[2], t[1]]
        } else {
            t
        }
    };
    let mut tris = vec![
        orient([a, b, c]),
        orient([a, b, d]),
        orient([a, c, d]),
        orient([b, c, d]),
    ];
    // Farthest-first insertion keeps intermediate hulls well conditioned.
    let mut order: Vec<usize> = (0..pts.len()).filter(|&i| ![a, b, c, d].contains(&i)).collect();
    order.sort_by(|&i, &j| (pts[j] - centroid).length_squared().total_cmp(&(pts[i] - centroid).length_squared()));
    for p in order {
        let visible: Vec<bool> = tris
            .iter()
            .map(|t| {
                let n = (pts[t[1]] - pts[t[0]]).cross(pts[t[2]] - pts[t[0]]);
                let len = n.length();
                len > 0.0 && n.dot(pts[p] - pts[t[0]]) / len > eps
            })
            .collect();
        if !visible.iter().any(|&v| v) {
            continue;
        }
        let mut directed: HashMap<(usize, usize), usize> = HashMap::new();
        for (k, t) in tris.iter().enumerate() {
            for e in 0..3 {
                directed.insert((t[e], t[(e + 1) % 3]), k);
            }
        }
        let mut next = Vec::new();
        for (k, t) in tris.iter().enumerate() {
            if !visible[k] {
                next.push(*t);
                continue;
            }
            for e in 0..3 {
                let (u, v) = (t[e], t[(e + 1) % 3]);
                let twin = directed.get(&(v, u)).copied();
                if twin.is_none_or(|o| !visible[o]) {
                    next.push([u, v, p]);
                }
            }
        }
        tris = next;
    }
    Ok(tris)
}

/// Andrew monotone chain over pre-sorted `(id, u, v)` points; counter-clockwise.
fn planar_convex_loop(points: &[(usize, f64, f64)]) -> Vec<usize> {
    let turn = |a: &(usize, f64, f64), b: &(usize, f64, f64), c: &(usize, f64, f64)| {
        (b.1 - a.1) * (c.2 - a.2) - (b.2 - a.2) * (c.1 - a.1)
    };
    const AREA_TOL: f64 = 1e-14;
    let mut lower: Vec<(usize, f64, f64)> = Vec::new();
    for p in points {
        while lower.len() > 1 && turn(&lower[lower.len() - 2], &lower[lower.len() - 1], p) <= AREA_TOL {
            lower.pop();
        }
        lower.push(*p);
    }
    let mut upper: Vec<(usize, f64, f64)> = Vec::new();
    for p in points.iter().rev() {
        while upper.len() > 1 && turn(&upper[upper.len() - 2], &upper[upper.len() - 1], p) <= AREA_TOL {
            upper.pop();
        }
        upper.push(*p);
    }
    lower.pop();
    upper.pop();
    lower.iter().chain(upper.iter()).map(|p| p.0).collect()
}

#[cfg(test)]
#[path = "convex_tests.rs"]
mod tests;

// ---- CPU reference (oracle) ------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub position: Vec3,
    pub rotation: glam::Quat,
}

/// World vertices, face normals and edge vectors of a scaled, posed hull.
pub struct WorldPolytope {
    pub vertices: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub edges: Vec<Vec3>,
}

impl WorldPolytope {
    pub fn new(h: &ConvexHull, scale: Vec3, pose: Pose) -> Self {
        let vertices: Vec<Vec3> = h.vertices.iter().map(|v| pose.position + pose.rotation * (*v * scale)).collect();
        let normals = h.faces.iter().map(|f| (pose.rotation * (f.normal / scale)).normalize()).collect();
        let edges = h.edges.iter().map(|e| vertices[e[1] as usize] - vertices[e[0] as usize]).collect();
        Self { vertices, normals, edges }
    }

    fn centroid(&self) -> Vec3 {
        self.vertices.iter().copied().sum::<Vec3>() / self.vertices.len() as f32
    }

    /// Separation of `other` from `self` along `n` (positive = gap).
    pub fn separation_along(&self, other: &WorldPolytope, n: Vec3) -> f32 {
        let project = |vs: &[Vec3]| {
            vs.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| {
                let d = n.dot(*v);
                (lo.min(d), hi.max(d))
            })
        };
        let (alo, ahi) = project(&self.vertices);
        let (blo, bhi) = project(&other.vertices);
        (blo - ahi).max(alo - bhi)
    }
}

/// Largest separation over every face normal and every edge-pair cross product,
/// using full vertex projections (no adjacency assumptions). Positive means a
/// separating axis exists. The axis points from A towards B. O(F·V + E²·V).
pub fn sat_separation(a: &WorldPolytope, b: &WorldPolytope) -> (Vec3, f32) {
    let (ca, cb) = (a.centroid(), b.centroid());
    let mut best = (Vec3::Y, f32::NEG_INFINITY);
    let mut test = |axis: Vec3| {
        let len = axis.length();
        if len < 1e-6 {
            return;
        }
        let mut n = axis / len;
        if n.dot(cb - ca) < 0.0 {
            n = -n;
        }
        let sep = a.separation_along(b, n);
        if sep > best.1 {
            best = (n, sep);
        }
    };
    for n in a.normals.iter().chain(&b.normals) {
        test(*n);
    }
    for da in &a.edges {
        for db in &b.edges {
            test(da.cross(*db));
        }
    }
    best
}

/// Conformance oracle: `Some((A→B normal, separation <= 0))` when the hulls overlap.
pub fn sat_oracle(
    a: &ConvexHull,
    scale_a: Vec3,
    pose_a: Pose,
    b: &ConvexHull,
    scale_b: Vec3,
    pose_b: Pose,
) -> Option<(Vec3, f32)> {
    let (n, sep) = sat_separation(&WorldPolytope::new(a, scale_a, pose_a), &WorldPolytope::new(b, scale_b, pose_b));
    (sep <= 0.0).then_some((n, sep))
}
