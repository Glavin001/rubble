//! Weighted flexbody skins (a **Rubble extension** to JBeam flexbodies): every
//! render vertex carries explicit node indices and weights, emitted with the
//! export (a `.rskin` sidecar a flexbody row names with `"rubbleSkin"`) and read
//! verbatim by the loader, so the mesh follows its part's nodes by construction
//! instead of by a nearest-node search at load time.
//!
//! **Weights** ([`bind`]): moving least squares (affine) over the vertex's own
//! part's nodes. Each node j has a smooth compact kernel ωⱼ(x) = (1 − r²/Rⱼ²)³
//! (C², zero with its derivatives at Rⱼ), Rⱼ from the node's local spacing. The
//! affine fit's point weights are
//!
//! Aⱼ(x) = ωⱼ/W + ωⱼ (x − p*)ᵀ M⁻¹ (pⱼ − p*),   W = Σω, p* = Σωp/W,
//! M = Σ ω (p − p*)(p − p*)ᵀ + λ W I,
//!
//! a partition of unity (Σ Aⱼ = 1) that reproduces any affine motion of the
//! nodes (so any rigid one) and is as smooth as the kernels. With them every
//! vertex stores (node, Aⱼ, ∂Aⱼ/∂T, ∂Aⱼ/∂B) for the nodes in reach (T, B its rest
//! tangents, by central differences in double precision), a detail offset `off`
//! (the rest vertex less Σ Aⱼpⱼ, in the frame below: λ's small bias and nothing
//! else) and its rest normal's coefficients `nrm` in that frame's basis.
//!
//! **Skinning** ([`eval`], the renderer's `weightedVertex()` is the same):
//!
//! P = Σ Aⱼqⱼ,  U = Σ aⱼqⱼ (= F·T),  V = Σ bⱼqⱼ (= F·B),
//! ŵ = (U × V)/|U × V|,  û = U/|U|,  v̂ = ŵ × û,
//! x = P + off₀ û + off₁ v̂ + off₂ ŵ,
//! n = normalize(nrm₀ (V × ŵ) + nrm₁ (ŵ × U) + nrm₂ (U × V)),
//!
//! the normal being F⁻ᵀ applied to the rest normal (the basis (U, V, ŵ)'s dual),
//! recomputed from the deformed field every frame. At rest both are exact; under
//! any rigid motion they move rigidly.

use glam::{DMat3, DVec3, Vec3};

/// The regularisation λ of the fit (m²): (5 mm)².
pub const LAMBDA: f64 = 2.5e-5;
/// The kernel radius of a node: this multiple of its distance to its
/// [`K_SPACING`]-th nearest neighbour in its pool, at least [`R_MIN`].
pub const R_SPACING: f64 = 1.45;
pub const K_SPACING: usize = 5;
pub const R_MIN: f64 = 0.42;
/// Each vertex needs this many nodes in reach and the fit's smallest
/// eigenvalue (without λ) above (`EIG_MIN` m)²·W; a pool that fails grows
/// every radius by 12 % (one field per pool: still smooth).
pub const MIN_NODES: usize = 5;
/// The frame's kernels: this multiple of the position's radii.
pub const FRAME_SCALE: f64 = 1.6;
pub const EIG_MIN: f64 = 0.012;

/// One vertex's skin.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Vert {
    /// (node, A, ∂A/∂T, ∂A/∂B).
    pub w: Vec<(u32, f32, f32, f32)>,
    pub off: [f32; 3],
    pub nrm: [f32; 3],
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeshSkin {
    pub name: String,
    pub verts: Vec<Vert>,
}

/// A vehicle's skins: node names (the entries index them) and per mesh its
/// vertices' skins, in the mesh's vertex order (a glTF mesh's primitives in
/// order, each primitive's vertices in order).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Skin {
    pub nodes: Vec<String>,
    pub meshes: Vec<MeshSkin>,
}

const MAGIC: &[u8; 4] = b"RSKN";

impl Skin {
    pub fn mesh(&self, name: &str) -> Option<&MeshSkin> {
        self.meshes.iter().find(|m| m.name == name)
    }

    /// The `.rskin` bytes (little endian): "RSKN", version 1, the node names,
    /// then per mesh its name, vertex and entry counts, per vertex its first
    /// entry, offset and normal coefficients (then the end), and its entries
    /// (node u32, A, a, b as f32).
    pub fn encode(&self) -> Vec<u8> {
        let mut o = Vec::new();
        o.extend_from_slice(MAGIC);
        o.extend_from_slice(&1u32.to_le_bytes());
        let s = |o: &mut Vec<u8>, t: &str| {
            o.extend_from_slice(&(t.len() as u16).to_le_bytes());
            o.extend_from_slice(t.as_bytes());
        };
        o.extend_from_slice(&(self.nodes.len() as u32).to_le_bytes());
        for n in &self.nodes {
            s(&mut o, n);
        }
        o.extend_from_slice(&(self.meshes.len() as u32).to_le_bytes());
        for m in &self.meshes {
            s(&mut o, &m.name);
            let ne: usize = m.verts.iter().map(|v| v.w.len()).sum();
            o.extend_from_slice(&(m.verts.len() as u32).to_le_bytes());
            o.extend_from_slice(&(ne as u32).to_le_bytes());
            let mut at = 0u32;
            for v in &m.verts {
                o.extend_from_slice(&at.to_le_bytes());
                for f in v.off.iter().chain(&v.nrm) {
                    o.extend_from_slice(&f.to_le_bytes());
                }
                at += v.w.len() as u32;
            }
            o.extend_from_slice(&at.to_le_bytes());
            for v in &m.verts {
                for &(n, a, b, c) in &v.w {
                    o.extend_from_slice(&n.to_le_bytes());
                    for f in [a, b, c] {
                        o.extend_from_slice(&f.to_le_bytes());
                    }
                }
            }
        }
        o
    }

    pub fn decode(b: &[u8]) -> Result<Skin, String> {
        let mut at = 0usize;
        let take = |at: &mut usize, n: usize| -> Result<&[u8], String> {
            let s = b.get(*at..*at + n).ok_or("rskin: truncated")?;
            *at += n;
            Ok(s)
        };
        let u32_ = |at: &mut usize| -> Result<u32, String> {
            Ok(u32::from_le_bytes(take(at, 4)?.try_into().unwrap()))
        };
        let f32_ = |at: &mut usize| -> Result<f32, String> {
            Ok(f32::from_le_bytes(take(at, 4)?.try_into().unwrap()))
        };
        let str_ = |at: &mut usize| -> Result<String, String> {
            let n = u16::from_le_bytes(take(at, 2)?.try_into().unwrap()) as usize;
            String::from_utf8(take(at, n)?.to_vec()).map_err(|e| e.to_string())
        };
        if take(&mut at, 4)? != MAGIC {
            return Err("rskin: not a Rubble skin".into());
        }
        let ver = u32_(&mut at)?;
        if ver != 1 {
            return Err(format!("rskin: version {ver}"));
        }
        let nn = u32_(&mut at)? as usize;
        let mut nodes = Vec::with_capacity(nn);
        for _ in 0..nn {
            nodes.push(str_(&mut at)?);
        }
        let nm = u32_(&mut at)? as usize;
        let mut meshes = Vec::with_capacity(nm);
        for _ in 0..nm {
            let name = str_(&mut at)?;
            let nv = u32_(&mut at)? as usize;
            let ne = u32_(&mut at)? as usize;
            let mut starts = Vec::with_capacity(nv + 1);
            let mut verts = Vec::with_capacity(nv);
            for _ in 0..nv {
                starts.push(u32_(&mut at)? as usize);
                let mut f = [0.0f32; 6];
                for x in &mut f {
                    *x = f32_(&mut at)?;
                }
                verts.push(Vert {
                    w: Vec::new(),
                    off: [f[0], f[1], f[2]],
                    nrm: [f[3], f[4], f[5]],
                });
            }
            starts.push(u32_(&mut at)? as usize);
            if starts[nv] != ne {
                return Err(format!(
                    "rskin: {name}: {} entries listed, {ne} stated",
                    starts[nv]
                ));
            }
            for (k, v) in verts.iter_mut().enumerate() {
                for _ in starts[k]..starts[k + 1] {
                    let n = u32_(&mut at)?;
                    if n as usize >= nn {
                        return Err(format!("rskin: {name}: node {n} of {nn}"));
                    }
                    let (a, bb, c) = (f32_(&mut at)?, f32_(&mut at)?, f32_(&mut at)?);
                    v.w.push((n, a, bb, c));
                }
            }
            meshes.push(MeshSkin { name, verts });
        }
        Ok(Skin { nodes, meshes })
    }
}

/// The skinned position and normal of one vertex from node positions `q`
/// (its entries' nodes already mapped to indices into `q`).
pub fn eval(w: &[(u32, f32, f32, f32)], off: [f32; 3], nrm: [f32; 3], q: &[Vec3]) -> (Vec3, Vec3) {
    let (mut p, mut u, mut v) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
    for &(n, a, da, db) in w {
        let x = q[n as usize];
        p += x * a;
        u += x * da;
        v += x * db;
    }
    let c = u.cross(v);
    let wn = c * (1.0 / c.length().max(1e-20));
    let un = u * (1.0 / u.length().max(1e-20));
    let vn = wn.cross(un);
    let pos = p + un * off[0] + vn * off[1] + wn * off[2];
    let n = v.cross(wn) * nrm[0] + wn.cross(u) * nrm[1] + c * nrm[2];
    (pos, n * (1.0 / n.length().max(1e-20)))
}

/// A pool of nodes one mesh binds to (rest positions, any frame) with each
/// node's kernel radius.
pub struct Pool {
    pub ids: Vec<u32>,
    pub pos: Vec<DVec3>,
    pub radius: Vec<f64>,
}

impl Pool {
    /// Radii from the local spacing (see [`R_SPACING`]), times `scale`.
    pub fn new(ids: Vec<u32>, pos: Vec<DVec3>, scale: f64) -> Pool {
        let n = pos.len();
        let radius = (0..n)
            .map(|i| {
                let mut d: Vec<f64> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| pos[i].distance(pos[j]))
                    .collect();
                d.sort_by(|a, b| a.total_cmp(b));
                let dk = d.get(K_SPACING - 1).or(d.last()).copied().unwrap_or(R_MIN);
                (R_SPACING * dk).max(R_MIN) * scale
            })
            .collect();
        Pool { ids, pos, radius }
    }

    /// The fit's point weights at `x`: (pool index, A) for nodes in reach,
    /// and whether the fit is well posed there.
    fn weights(&self, x: DVec3, near: &[usize]) -> (Vec<(usize, f64)>, bool) {
        let mut om: Vec<(usize, f64)> = Vec::with_capacity(near.len());
        for &j in near {
            let r2 = self.pos[j].distance_squared(x) / (self.radius[j] * self.radius[j]);
            if r2 < 1.0 {
                let t = 1.0 - r2;
                om.push((j, t * t * t));
            }
        }
        let w: f64 = om.iter().map(|o| o.1).sum();
        if om.len() < 3 || w <= 0.0 {
            return (Vec::new(), false);
        }
        let ps = om.iter().map(|&(j, o)| self.pos[j] * o).sum::<DVec3>() / w;
        let mut m0 = DMat3::ZERO;
        for &(j, o) in &om {
            let d = self.pos[j] - ps;
            m0 += DMat3::from_cols(d * d.x, d * d.y, d * d.z) * o;
        }
        let m = m0 + DMat3::IDENTITY * (LAMBDA * w);
        let mi = m.inverse();
        let e = x - ps;
        let g = mi * e;
        let out: Vec<(usize, f64)> = om
            .iter()
            .map(|&(j, o)| (j, o / w + o * g.dot(self.pos[j] - ps)))
            .collect();
        let ok = om.len() >= MIN_NODES && min_eig(m0) >= EIG_MIN * EIG_MIN * w;
        (out, ok)
    }

    fn near(&self, x: DVec3, pad: f64) -> Vec<usize> {
        (0..self.pos.len())
            .filter(|&j| self.pos[j].distance(x) < self.radius[j] + pad)
            .collect()
    }
}

/// Smallest eigenvalue of a symmetric 3 × 3 (the closed form).
fn min_eig(a: DMat3) -> f64 {
    let (a11, a22, a33) = (a.x_axis.x, a.y_axis.y, a.z_axis.z);
    let (a12, a13, a23) = (a.y_axis.x, a.z_axis.x, a.z_axis.y);
    let p1 = a12 * a12 + a13 * a13 + a23 * a23;
    let q = (a11 + a22 + a33) / 3.0;
    if p1 < 1e-30 {
        return a11.min(a22).min(a33);
    }
    let p2 = (a11 - q).powi(2) + (a22 - q).powi(2) + (a33 - q).powi(2) + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();
    let b = (a - DMat3::IDENTITY * q) * (1.0 / p);
    let r = (b.determinant() / 2.0).clamp(-1.0, 1.0);
    let phi = r.acos() / 3.0;
    q + 2.0 * p * (phi + 2.0 * std::f64::consts::PI / 3.0).cos()
}

/// A vertex's skin over `pool` (entries index the pool's `ids`), or None
/// where the fit is not well posed.
pub fn bind_vertex(pool: &Pool, frame: &Pool, x: DVec3, n0: DVec3) -> Option<Vert> {
    let n0 = n0.normalize_or(DVec3::Y);
    let t = n0.any_orthonormal_vector();
    let b = n0.cross(t);
    let eps = 2e-3;
    let near = frame.near(x, eps);
    let (w0, ok) = pool.weights(x, &near);
    if !ok {
        return None;
    }
    // The frame's gradients from the wider kernels (the position's own would
    // turn the frame with every few-millimetre difference between close
    // nodes; any affine-reproducing weights give an exact rigid motion).
    let grad = |d: DVec3| -> Vec<f64> {
        let (wp, _) = frame.weights(x + d * eps, &near);
        let (wm, _) = frame.weights(x - d * eps, &near);
        let get = |v: &[(usize, f64)], j: usize| v.iter().find(|q| q.0 == j).map_or(0.0, |q| q.1);
        near.iter()
            .map(|&j| (get(&wp, j) - get(&wm, j)) / (2.0 * eps))
            .collect()
    };
    let (gt, gb) = (grad(t), grad(b));
    let mut w = Vec::new();
    let (mut p, mut u, mut v) = (DVec3::ZERO, DVec3::ZERO, DVec3::ZERO);
    for (k, &j) in near.iter().enumerate() {
        let a = w0.iter().find(|q| q.0 == j).map_or(0.0, |q| q.1);
        if a == 0.0 && gt[k] == 0.0 && gb[k] == 0.0 {
            continue;
        }
        let (af, tf, bf) = (a as f32, gt[k] as f32, gb[k] as f32);
        // The stored (f32) weights are the ones the frame is measured with.
        let pj = pool.pos[j];
        p += pj * af as f64;
        u += pj * tf as f64;
        v += pj * bf as f64;
        w.push((pool.ids[j], af, tf, bf));
    }
    let c = u.cross(v);
    let wn = c.normalize();
    let un = u.normalize();
    let vn = wn.cross(un);
    let d = x - p;
    let e = DMat3::from_cols(u, v, wn);
    let cn = e.transpose() * n0;
    Some(Vert {
        w,
        off: [d.dot(un) as f32, d.dot(vn) as f32, d.dot(wn) as f32],
        nrm: [cn.x as f32, cn.y as f32, cn.z as f32],
    })
}

/// Bind every vertex (positions and normals in the pool's frame) to `ids`
/// (rest positions `pos`): the pool's radii grow until every vertex is well
/// posed. Returns the skins and the radius scale used.
pub fn bind(
    ids: &[u32],
    pos: &[Vec3],
    verts: &[Vec3],
    nrms: &[Vec3],
) -> Result<(Vec<Vert>, f64), String> {
    let dpos: Vec<DVec3> = pos.iter().map(|p| p.as_dvec3()).collect();
    let mut scale = 1.0;
    for _ in 0..12 {
        let pool = Pool::new(ids.to_vec(), dpos.clone(), scale);
        let frame = Pool::new(ids.to_vec(), dpos.clone(), scale * FRAME_SCALE);
        let mut out = Vec::with_capacity(verts.len());
        let mut failed = false;
        for (x, n) in verts.iter().zip(nrms) {
            match bind_vertex(&pool, &frame, x.as_dvec3(), n.as_dvec3()) {
                Some(v) => out.push(v),
                None => {
                    failed = true;
                    break;
                }
            }
        }
        if !failed {
            return Ok((out, scale));
        }
        scale *= 1.12;
    }
    Err(format!(
        "flexskin: no well-posed fit over {} nodes",
        ids.len()
    ))
}

/// Bind a rigid piece (a lamp, mirror, handle, seat, wheel): every vertex
/// takes the weights of the piece's centroid, so the frame (U, V, W) is one
/// for the whole piece and each vertex is its rest offset from Σ Aⱼqⱼ in
/// it: the piece moves rigidly with its nodes' local motion and never warps.
pub fn bind_rigid(
    ids: &[u32],
    pos: &[Vec3],
    verts: &[Vec3],
    nrms: &[Vec3],
) -> Result<(Vec<Vert>, f64), String> {
    let c = verts.iter().map(|v| v.as_dvec3()).sum::<DVec3>() / verts.len().max(1) as f64;
    let dpos: Vec<DVec3> = pos.iter().map(|p| p.as_dvec3()).collect();
    let mut scale = 1.0;
    for _ in 0..12 {
        let pool = Pool::new(ids.to_vec(), dpos.clone(), scale);
        // A rigid piece's frame is its centroid's nodes' own (its offsets
        // reach across the piece: a wider frame would turn it with the
        // structure around rather than the nodes that carry it).
        if let Some(cv) = bind_vertex(&pool, &pool, c, DVec3::Y) {
            let (mut p, mut u, mut v) = (DVec3::ZERO, DVec3::ZERO, DVec3::ZERO);
            let at = |id: u32| pool.pos[pool.ids.iter().position(|&x| x == id).unwrap()];
            for &(n, a, da, db) in &cv.w {
                p += at(n) * a as f64;
                u += at(n) * da as f64;
                v += at(n) * db as f64;
            }
            let wn = u.cross(v).normalize();
            let un = u.normalize();
            let vn = wn.cross(un);
            let e = DMat3::from_cols(u, v, wn);
            let out = verts
                .iter()
                .zip(nrms)
                .map(|(x, n)| {
                    let d = x.as_dvec3() - p;
                    let cn = e.transpose() * n.as_dvec3().normalize_or(DVec3::Y);
                    Vert {
                        w: cv.w.clone(),
                        off: [d.dot(un) as f32, d.dot(vn) as f32, d.dot(wn) as f32],
                        nrm: [cn.x as f32, cn.y as f32, cn.z as f32],
                    }
                })
                .collect();
            return Ok((out, scale));
        }
        scale *= 1.12;
    }
    Err(format!(
        "flexskin: no well-posed fit over {} nodes",
        ids.len()
    ))
}

/// A glTF mesh as a flexbody mesh in the JBeam frame (the loader's meshes):
/// its primitives' vertices in order, material names per triangle.
pub fn glb_mesh(g: &crate::glb::Glb, m: &crate::glb::Mesh) -> crate::flexmesh::DaeMesh {
    let to_j = |p: [f32; 3]| Vec3::new(p[0], -p[2], p[1]);
    let mut out = crate::flexmesh::DaeMesh {
        name: m.name.clone(),
        geometry: m.name.clone(),
        ..Default::default()
    };
    for p in &m.prims {
        let base = out.pos.len() as u32;
        let mname = g
            .materials
            .get(p.material)
            .map_or("", |x| x.name.as_str())
            .to_string();
        let mi = match out.materials.iter().position(|x| *x == mname) {
            Some(k) => k,
            None => {
                out.materials.push(mname);
                out.materials.len() - 1
            }
        };
        for k in 0..p.pos.len() {
            out.pos.push(to_j(p.pos[k]));
            out.nrm.push(to_j(p.nrm[k]));
            out.uv.push(p.uv.get(k).copied().unwrap_or([0.0; 2]));
        }
        for t in p.idx.chunks_exact(3) {
            out.tris.push([t[0] + base, t[1] + base, t[2] + base]);
            out.tri_mat.push(mi as u16);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    fn lattice() -> (Vec<u32>, Vec<Vec3>) {
        let mut p = Vec::new();
        for i in 0..5 {
            for j in 0..4 {
                for k in 0..2 {
                    p.push(Vec3::new(
                        i as f32 * 0.5,
                        j as f32 * 0.45,
                        k as f32 * 0.08 + 0.02 * (i as f32).sin(),
                    ));
                }
            }
        }
        ((0..p.len() as u32).collect(), p)
    }

    fn verts() -> (Vec<Vec3>, Vec<Vec3>) {
        let v: Vec<Vec3> = (0..200)
            .map(|k| {
                let f = k as f32;
                Vec3::new(
                    0.2 + 1.6 * (f * 0.37).fract(),
                    0.1 + 1.1 * (f * 0.61).fract(),
                    0.09 + 0.01 * (f * 0.13).fract(),
                )
            })
            .collect();
        let n: Vec<Vec3> = (0..200)
            .map(|k| Vec3::new(0.1 * (k as f32 * 0.7).sin(), 0.05, 1.0).normalize())
            .collect();
        (v, n)
    }

    #[test]
    fn rest_and_rigid_motion_are_exact() {
        let (ids, p) = lattice();
        let (v, n) = verts();
        let (sk, _) = bind(&ids, &p, &v, &n).unwrap();
        let q = Quat::from_rotation_y(0.7) * Quat::from_rotation_x(-0.3);
        let t = Vec3::new(3.0, -1.0, 2.0);
        let moved: Vec<Vec3> = p.iter().map(|x| q * *x + t).collect();
        for ((s, x), nn) in sk.iter().zip(&v).zip(&n) {
            let sum: f32 = s.w.iter().map(|e| e.1).sum();
            assert!((sum - 1.0).abs() < 1e-5, "partition of unity {sum}");
            let (a, an) = eval(&s.w, s.off, s.nrm, &p);
            assert!(
                a.distance(*x) < 2e-5 && an.angle_between(*nn) < 1e-3,
                "rest"
            );
            let (b, bn) = eval(&s.w, s.off, s.nrm, &moved);
            assert!(
                b.distance(q * *x + t) < 2e-5,
                "rigid {}",
                b.distance(q * *x + t)
            );
            assert!(bn.angle_between(q * *nn) < 1e-3);
        }
    }

    #[test]
    fn a_smooth_bend_moves_the_skin_smoothly() {
        let (ids, p) = lattice();
        let (v, n) = verts();
        let (sk, _) = bind(&ids, &p, &v, &n).unwrap();
        // Bend about the y axis: x → arc.
        let bend = |x: Vec3| {
            let r = 2.0;
            let a = x.x / r;
            Vec3::new((r - x.z) * a.sin(), x.y, r - (r - x.z) * a.cos())
        };
        let moved: Vec<Vec3> = p.iter().map(|x| bend(*x)).collect();
        for (s, x) in sk.iter().zip(&v) {
            let (b, _) = eval(&s.w, s.off, s.nrm, &moved);
            assert!(b.distance(bend(*x)) < 0.02, "{}", b.distance(bend(*x)));
        }
    }

    #[test]
    fn the_sidecar_round_trips() {
        let (ids, p) = lattice();
        let (v, n) = verts();
        let (sk, _) = bind(&ids, &p, &v, &n).unwrap();
        let s = Skin {
            nodes: ids.iter().map(|i| format!("n{i}")).collect(),
            meshes: vec![MeshSkin {
                name: "panel".into(),
                verts: sk,
            }],
        };
        assert_eq!(Skin::decode(&s.encode()).unwrap(), s);
    }
}
