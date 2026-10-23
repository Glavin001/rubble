//! Flexbody skinning (design: "Visual mesh"): every render vertex is bound once,
//! at spawn, to a local frame of three nodes, and rebuilt every frame from the
//! particle positions (a vertex shader does this on the GPU; [`FlexBinding::skin`]
//! is the CPU twin).
//!
//! Binding, per vertex: the closest allowed node is the frame centre c; two more
//! nodes x and y within a distance limit are chosen so that c→x and c→y are as
//! close to perpendicular as possible. The frame is (c, eₓ = x − c, e_y = y − c,
//! e_z = eₓ × e_y) and the vertex is stored as its coordinates in that frame, so a
//! rigid motion reproduces the mesh exactly and a deformation carries it along.

use glam::{Mat3, Vec3};
use serde::{Deserialize, Serialize};

/// One vertex's frame: node indices (relative to the soft body) and coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FlexBinding {
    pub nodes: [u32; 3],
    pub coords: [f32; 3],
}

/// Frame of three node positions.
fn frame(c: Vec3, x: Vec3, y: Vec3) -> Mat3 {
    let ex = x - c;
    let ey = y - c;
    Mat3::from_cols(ex, ey, ex.cross(ey))
}

impl FlexBinding {
    /// Vertex position for the current node positions.
    pub fn skin(&self, nodes: &[Vec3]) -> Vec3 {
        let [c, x, y] = self.nodes.map(|i| nodes[i as usize]);
        c + frame(c, x, y) * Vec3::from_array(self.coords)
    }
}

/// Binding options.
#[derive(Clone, Copy, Debug)]
pub struct BindOptions {
    /// Frame nodes are searched among the `candidates` nodes closest to the vertex.
    pub candidates: usize,
    /// Frame edges longer than this (m) are not used.
    pub max_edge: f32,
}

impl Default for BindOptions {
    fn default() -> Self {
        Self { candidates: 12, max_edge: 1.5 }
    }
}

/// Bind `vertices` to frames of `nodes` (rest positions). `allowed` restricts the
/// nodes a mesh may bind to (a flexbody's node group); `None` allows all. Returns
/// `None` for a vertex with no valid frame (BeamNG drops such meshes).
pub fn bind(vertices: &[Vec3], nodes: &[Vec3], allowed: Option<&[u32]>, opts: BindOptions) -> Vec<Option<FlexBinding>> {
    let pool: Vec<u32> = match allowed {
        Some(a) => a.to_vec(),
        None => (0..nodes.len() as u32).collect(),
    };
    let mut near: Vec<(f32, u32)> = Vec::with_capacity(pool.len());
    vertices
        .iter()
        .map(|&v| {
            near.clear();
            near.extend(pool.iter().map(|&i| ((nodes[i as usize] - v).length_squared(), i)));
            let k = opts.candidates.min(near.len());
            if k < 3 {
                return None;
            }
            near.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));
            near[..k].sort_by(|a, b| a.0.total_cmp(&b.0));
            let c = near[0].1;
            let pc = nodes[c as usize];
            // Most perpendicular pair of edges from c (|sin θ|, ties to shorter edges).
            let mut best: Option<(f32, u32, u32)> = None;
            for i in 1..k {
                let ex = nodes[near[i].1 as usize] - pc;
                let lx = ex.length();
                if lx < 1e-4 || lx > opts.max_edge {
                    continue;
                }
                for j in i + 1..k {
                    let ey = nodes[near[j].1 as usize] - pc;
                    let ly = ey.length();
                    if ly < 1e-4 || ly > opts.max_edge {
                        continue;
                    }
                    let sin = ex.cross(ey).length() / (lx * ly);
                    let score = sin - 0.05 * (lx + ly) / opts.max_edge;
                    if best.is_none_or(|b| score > b.0) {
                        best = Some((score, near[i].1, near[j].1));
                    }
                }
            }
            let (score, x, y) = best?;
            if score < 0.2 {
                return None;
            }
            let f = frame(pc, nodes[x as usize], nodes[y as usize]);
            if f.determinant().abs() < 1e-9 {
                return None;
            }
            let coords = f.inverse() * (v - pc);
            Some(FlexBinding { nodes: [c, x, y], coords: coords.to_array() })
        })
        .collect()
}

/// A visual mesh made from a soft body's skin triangles: each triangle split in
/// four with its edge midpoints pushed out along the averaged vertex normals, so
/// flat node–beam panels read as rounded bodywork. Returns (vertices, triangles).
pub fn rounded_skin(nodes: &[Vec3], triangles: &[[u32; 3]], bulge: f32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let mut normals = vec![Vec3::ZERO; nodes.len()];
    for t in triangles {
        let [a, b, c] = t.map(|i| nodes[i as usize]);
        let n = (b - a).cross(c - a);
        for &i in t {
            normals[i as usize] += n;
        }
    }
    for n in &mut normals {
        *n = n.normalize_or_zero();
    }
    let mut verts: Vec<Vec3> = nodes.to_vec();
    let mut mids = std::collections::HashMap::new();
    let mut mid = |a: u32, b: u32, verts: &mut Vec<Vec3>| -> u32 {
        let key = (a.min(b), a.max(b));
        *mids.entry(key).or_insert_with(|| {
            let (pa, pb) = (nodes[a as usize], nodes[b as usize]);
            let n = (normals[a as usize] + normals[b as usize]).normalize_or_zero();
            verts.push(0.5 * (pa + pb) + n * (bulge * (pa - pb).length()));
            verts.len() as u32 - 1
        })
    };
    let mut tris = Vec::with_capacity(triangles.len() * 4);
    for t in triangles {
        let [a, b, c] = *t;
        let ab = mid(a, b, &mut verts);
        let bc = mid(b, c, &mut verts);
        let ca = mid(c, a, &mut verts);
        tris.extend_from_slice(&[[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]);
    }
    (verts, tris)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    fn lattice() -> Vec<Vec3> {
        let mut v = Vec::new();
        for z in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    v.push(Vec3::new(x as f32, y as f32, z as f32) * 0.5);
                }
            }
        }
        v
    }

    #[test]
    fn rigid_motion_reproduces_the_mesh() {
        let nodes = lattice();
        let verts: Vec<Vec3> = (0..50).map(|i| Vec3::new((i as f32 * 0.37).fract(), (i as f32 * 0.61).fract(), (i as f32 * 0.13).fract()) * 1.0).collect();
        let b = bind(&verts, &nodes, None, BindOptions::default());
        assert!(b.iter().all(Option::is_some));
        let q = Quat::from_rotation_y(0.7) * Quat::from_rotation_x(-0.3);
        let t = Vec3::new(3.0, -1.0, 2.0);
        let moved: Vec<Vec3> = nodes.iter().map(|p| q * *p + t).collect();
        for (v, bnd) in verts.iter().zip(&b) {
            let got = bnd.unwrap().skin(&moved);
            assert!((got - (q * *v + t)).length() < 1e-4, "{got:?}");
        }
    }

    #[test]
    fn deformation_carries_vertices_with_their_nodes() {
        let nodes = lattice();
        let verts = vec![Vec3::new(0.1, 0.1, 0.05), Vec3::new(0.9, 0.9, 0.95)];
        let b = bind(&verts, &nodes, None, BindOptions::default());
        // Crush the corner near the first vertex by 20 cm along +X.
        let mut crushed = nodes.clone();
        for p in &mut crushed {
            if p.length() < 0.6 {
                p.x += 0.2;
            }
        }
        let near = b[0].unwrap().skin(&crushed);
        let far = b[1].unwrap().skin(&crushed);
        assert!((near.x - verts[0].x - 0.2).abs() < 0.05, "near vertex follows its nodes: {near:?}");
        assert!((far - verts[1]).length() < 0.05, "far vertex stays: {far:?}");
    }

    #[test]
    fn allowed_set_and_degenerate_cases() {
        let nodes = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::new(2.0, 0.0, 0.0)];
        let v = vec![Vec3::new(0.2, 0.2, 0.0)];
        let b = bind(&v, &nodes, Some(&[0, 1, 2]), BindOptions::default());
        assert_eq!(b[0].unwrap().nodes[0], 0);
        // Collinear nodes give no frame.
        let line = vec![Vec3::ZERO, Vec3::X, Vec3::new(2.0, 0.0, 0.0)];
        assert!(bind(&v, &line, None, BindOptions::default())[0].is_none());
    }

    #[test]
    fn rounded_skin_quadruples_triangles() {
        let nodes = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
        let tris = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        let (v, t) = rounded_skin(&nodes, &tris, 0.05);
        assert_eq!(t.len(), 16);
        assert_eq!(v.len(), 4 + 6, "one midpoint per edge");
    }
}
