//! The visual skin: the constructed car of [`super::parts`] (a dense body
//! surface cut into separate panels with real shut lines, and modelled lamps,
//! grille, glass, mirrors, handles, wheels and brakes), bound piece by piece to
//! the skeleton so it deforms with the crash, flexbody style.
//!
//! Binding: every vertex rides one triangle of nodes (n₀, n₁, n₂):
//!
//! ```text
//! e₁ = P₁ − P₀, e₂ = P₂ − P₀, N = normalize(e₁ × e₂)
//! position = P₀ + α·e₁ + β·e₂ + h·N
//! T = normalize(e₁), B = N × T, normal = nₜ·T + n_b·B + n_n·N
//! ```
//!
//! with (α, β, h) and (nₜ, n_b, n_n) fixed at rest. A piece binds to one
//! physics part: the part its vertices lie on (a majority vote of the nearest
//! skin triangles), then each vertex to that part's nearest triangle. So a
//! door's skin, its rolled edges and its handle ride the door's nodes and leave
//! with it; nothing stretches across a shut line. Nothing here depends on the
//! skeleton's node spacing: the same mesh rides a coarse or a dense skeleton.
//! Wheels ride their tyre's fan triangles (they turn); calipers the axle and a
//! knuckle node (they do not); seats and the dashboard their frames.
//! `showcase-car-render.mjs` evaluates this on the GPU from the explicit
//! solver's node buffer ([`deform`] is the same on the CPU).

use super::parts::design::{assemble, sedan_styling};
use super::parts::{Bind, Mesh, Piece};
use super::vehicle::Built;
use crate::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use glam::Vec3;
use std::collections::HashMap;

/// A binding triangle stretched past this × a rest edge has torn: its vertices
/// are dropped (their triangles vanish) instead of stretching across the tear.
pub const TEAR_STRETCH: f32 = 2.5;

/// Material ids of the visual mesh.
pub mod mat {
    pub const PAINT: u8 = 0;
    pub const GLASS: u8 = 1;
    pub const CHROME: u8 = 2;
    pub const TRIM: u8 = 3;
    pub const HEADLIGHT: u8 = 4;
    pub const TAILLIGHT: u8 = 5;
    pub const TYRE: u8 = 6;
    pub const RIM: u8 = 7;
    pub const GAP: u8 = 8;
    pub const UNDERBODY: u8 = 9;
    pub const PLATE: u8 = 10;
    pub const GRILLE: u8 = 11;
    pub const SEAT: u8 = 12;
    pub const DASH: u8 = 13;
    pub const DOORCARD: u8 = 14;
    pub const HEADLINER: u8 = 15;
    pub const CARPET: u8 = 16;
    /// Clear lamp lens (drawn with the glass).
    pub const LENS: u8 = 17;
    /// Daytime running light guide (emissive white).
    pub const DRL: u8 = 18;
    /// Lamp reflector (bright, slightly rough metal).
    pub const REFLECTOR: u8 = 19;
    /// Red tail-lamp lens (drawn with the glass).
    pub const TAIL_LENS: u8 = 20;
    /// Tail-lamp light guides (emissive red).
    pub const TAIL_LED: u8 = 21;
    /// Brake disc and wheel barrel (cast iron / dark metal).
    pub const BRAKE: u8 = 22;
    /// Brake caliper (painted).
    pub const CALIPER: u8 = 23;
    /// Tail-lamp housing back (dark red).
    pub const TAIL_BACK: u8 = 24;
    /// Laminated windscreen (cracks instead of shattering; drawn with the glass).
    pub const SCREEN: u8 = 25;
    pub const COUNT: usize = 26;
}

/// Material names, by id.
pub const MATERIAL_NAMES: [&str; mat::COUNT] = [
    "paint",
    "glass",
    "chrome",
    "trim",
    "headlight",
    "taillight",
    "tyre",
    "rim",
    "gap",
    "underbody",
    "plate",
    "grille",
    "seat",
    "dash",
    "doorcard",
    "headliner",
    "carpet",
    "lens",
    "drl",
    "reflector",
    "tail_lens",
    "tail_led",
    "brake",
    "caliper",
    "tail_back",
    "screen",
];

/// Whether a material is drawn blended (after the opaque ones).
pub fn transparent(m: u8) -> bool {
    matches!(m, mat::GLASS | mat::LENS | mat::TAIL_LENS | mat::SCREEN)
}

#[derive(Clone, Debug, Default)]
pub struct ShowcaseVisual {
    /// Rest positions in the `.truck`'s coordinates.
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub colors: Vec<[f32; 3]>,
    pub material: Vec<u8>,
    pub indices: Vec<u32>,
    /// Per vertex: the three solver nodes (indices into the explicit solver's
    /// node list, tyre nodes included) and (α, β, h).
    pub bind_nodes: Vec<[u32; 3]>,
    pub bind_coords: Vec<[f32; 3]>,
    /// The rest normal in the triangle's frame (T, B, N).
    pub bind_normal: Vec<[f32; 3]>,
    /// The binding triangle's rest edge lengths |e₁|, |e₂|: a vertex whose
    /// triangle has stretched past [`TEAR_STRETCH`] × either is torn (dropped).
    pub bind_rest: Vec<[f32; 2]>,
    /// Baked ambient occlusion (1 open; shut lines, recesses and lamp throats
    /// darker).
    pub ao: Vec<f32>,
    /// 1 when the vertex's normal follows the skin's node normals (smooth
    /// shading across the binding triangles): n = normalize(n_rest + Σ wᵢ (Nᵢ − Rᵢ))
    /// with wᵢ the barycentric weights (1 − α − β, α, β) of nodes n₀ n₁ n₂, Nᵢ a
    /// node's normal now (the normalised sum of its skin triangles' cross
    /// products) and Rᵢ at rest.
    pub smooth: Vec<u8>,
    /// The skin triangles node normals come from, and per node (CSR over the
    /// solver's nodes) the triangles it is in, and its rest normal.
    pub normal_tris: Vec<[u32; 3]>,
    pub node_tri_start: Vec<u32>,
    pub node_tri_list: Vec<u32>,
    pub node_rest_normal: Vec<[f32; 3]>,
    /// The solver's node count for this vehicle (tyre nodes included).
    pub node_count: u32,
    /// 1 for a rigid part's vertex (mirror, handle, lamp): (α, β, h) are its
    /// rest offset in the triangle's orthonormal frame (T, B, N), so the part
    /// moves rigidly with the triangle and never stretches.
    pub rigid: Vec<u8>,
    /// The pieces (name, first vertex, vertex count), in order.
    pub pieces: Vec<(String, u32, u32)>,
}

/// Candidate binding triangles of one part, with a coarse grid over their centroids.
struct Binder {
    tris: Vec<[u32; 3]>,
    pos: Vec<Vec3>,
    cell: f32,
    grid: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl Binder {
    fn new(tris: Vec<[u32; 3]>, pos: &[Vec3], cell: f32) -> Self {
        let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        for (ti, t) in tris.iter().enumerate() {
            let c = (pos[t[0] as usize] + pos[t[1] as usize] + pos[t[2] as usize]) / 3.0;
            grid.entry(key(c, cell)).or_default().push(ti);
        }
        Self {
            tris,
            pos: pos.to_vec(),
            cell,
            grid,
        }
    }

    /// The nearest triangle to `p`, and the binding coordinates.
    fn bind(&self, p: Vec3, normal: Vec3) -> Option<([u32; 3], [f32; 3], [f32; 3])> {
        self.bind_d(p, normal).map(|(t, c, n, _)| (t, c, n))
    }

    /// [`Self::bind`] and the distance to the triangle.
    fn bind_d(&self, p: Vec3, normal: Vec3) -> Option<([u32; 3], [f32; 3], [f32; 3], f32)> {
        if self.tris.is_empty() {
            return None;
        }
        let k = key(p, self.cell);
        let mut best: Option<(f32, usize)> = None;
        for r in 1..10 {
            for dx in -r..=r {
                for dy in -r..=r {
                    for dz in -r..=r {
                        if let Some(list) = self.grid.get(&(k.0 + dx, k.1 + dy, k.2 + dz)) {
                            for &ti in list {
                                let t = self.tris[ti];
                                let d = point_triangle(
                                    p,
                                    self.pos[t[0] as usize],
                                    self.pos[t[1] as usize],
                                    self.pos[t[2] as usize],
                                );
                                if best.is_none_or(|b| d < b.0) {
                                    best = Some((d, ti));
                                }
                            }
                        }
                    }
                }
            }
            // Everything within the searched block's inner distance is found.
            let lo = Vec3::new(
                (k.0 - r) as f32,
                (k.1 - r) as f32,
                (k.2 - r) as f32,
            ) * self.cell;
            let hi = lo + Vec3::splat((2 * r + 1) as f32 * self.cell);
            let reach = (p - lo).min(hi - p).min_element();
            if best.is_some_and(|b| b.0 <= reach) {
                break;
            }
        }
        let (dist, ti) = best?;
        let t = self.tris[ti];
        let (p0, p1, p2) = (
            self.pos[t[0] as usize],
            self.pos[t[1] as usize],
            self.pos[t[2] as usize],
        );
        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let n = e1.cross(e2).normalize_or_zero();
        let d = p - p0;
        let h = d.dot(n);
        let q = d - n * h;
        // Solve q = α e₁ + β e₂ in the plane.
        let (a11, a12, a22) = (e1.dot(e1), e1.dot(e2), e2.dot(e2));
        let (b1, b2) = (q.dot(e1), q.dot(e2));
        let det = a11 * a22 - a12 * a12;
        if det.abs() < 1e-12 {
            return None;
        }
        let alpha = (b1 * a22 - b2 * a12) / det;
        let beta = (a11 * b2 - a12 * b1) / det;
        let tt = e1.normalize();
        let bb = n.cross(tt);
        let nl = [normal.dot(tt), normal.dot(bb), normal.dot(n)];
        Some((t, [alpha, beta, h], nl, dist))
    }
}

fn key(p: Vec3, cell: f32) -> (i32, i32, i32) {
    (
        (p.x / cell).floor() as i32,
        (p.y / cell).floor() as i32,
        (p.z / cell).floor() as i32,
    )
}

/// Distance from `p` to triangle (a, b, c) (Ericson, Real-Time Collision
/// Detection, 5.1.5).
fn point_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return ap.length();
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return bp.length();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (p - (a + ab * v)).length();
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return cp.length();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (p - (a + ac * w)).length();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (p - (b + (c - b) * w)).length();
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (p - (a + ab * v + ac * w)).length()
}

/// Build the visual skin of a built vehicle.
pub fn showcase_visual(built: &Built) -> Result<ShowcaseVisual, String> {
    let def = &built.def;
    let sk = &built.skeleton;
    let paint = built.params.paint.unwrap_or(def.look.paint);
    // The solver's nodes (tyre nodes included), in definition coordinates.
    let sim = ExplicitBeams::from_truck(
        &built.truck,
        Vec3::ZERO,
        Vec3::ZERO,
        ExplicitBeamsSettings::default(),
    )?;
    let node_pos: Vec<Vec3> = sim.nodes.iter().map(|n| n.pos - built.shift).collect();
    let mut pieces = assemble(def, &sedan_styling(def));
    pieces.extend(interior(built));
    // Binders: per part, all skin, per group, per tyre and hub.
    let mut by_part: HashMap<u16, Vec<[u32; 3]>> = HashMap::new();
    for t in &sk.skin {
        let p = sk.nodes[t[0] as usize].part;
        if t.iter().all(|&i| sk.nodes[i as usize].part == p) {
            by_part.entry(p).or_default().push(*t);
        }
    }
    let fallback = Binder::new(sk.skin.clone(), &node_pos, 0.15);
    // Single-part skin triangles of every part (votes and reach checks).
    let pure = Binder::new(by_part.values().flatten().copied().collect(), &node_pos, 0.15);
    let mut binders: HashMap<u16, Binder> = HashMap::new();
    let groups: Vec<String> = {
        let mut g: Vec<String> = built.bind_tris.keys().cloned().collect();
        g.sort();
        g
    };
    let tyre_centre: Vec<Vec3> = sim
        .tyres
        .iter()
        .map(|t| 0.5 * (node_pos[t.axle[0] as usize] + node_pos[t.axle[1] as usize]))
        .collect();
    let susp: Vec<u32> = built.groups.get("suspension").cloned().unwrap_or_default();
    let mut out = ShowcaseVisual {
        node_count: node_pos.len() as u32,
        ..Default::default()
    };
    for piece in pieces.iter_mut() {
        piece.mesh.orient_by_normals();
    }
    for piece in &pieces {
        let m = &piece.mesh;
        if m.is_empty() {
            continue;
        }
        let first = out.positions.len() as u32;
        let centroid = m.pos.iter().copied().sum::<Vec3>() / m.len() as f32;
        let nearest_tyre = || {
            (0..tyre_centre.len())
                .min_by(|&a, &b| {
                    tyre_centre[a]
                        .distance(centroid)
                        .total_cmp(&tyre_centre[b].distance(centroid))
                })
                .unwrap_or(0)
        };
        // The binder for this piece.
        let own: Option<Binder> = match piece.bind {
            Bind::Wheel(_) => {
                let t = &sim.tyres[nearest_tyre()];
                let rays = (t.count / 2) as usize;
                let ring = |i: usize, outer: bool| t.first + 2 * (i % rays) as u32 + u32::from(!outer);
                let mut tris = Vec::new();
                for i in 0..rays {
                    tris.push([t.axle[0], ring(i, true), ring(i + 1, true)]);
                    tris.push([t.axle[1], ring(i, false), ring(i + 1, false)]);
                }
                Some(Binder::new(tris, &node_pos, 0.1))
            }
            Bind::Hub(_) => {
                let t = &sim.tyres[nearest_tyre()];
                let (a0, a1) = (t.axle[0], t.axle[1]);
                let axis = (node_pos[a1 as usize] - node_pos[a0 as usize]).normalize_or_zero();
                let c = tyre_centre[nearest_tyre()];
                let k = susp
                    .iter()
                    .copied()
                    .filter(|&i| i != a0 && i != a1)
                    .filter(|&i| {
                        let d = node_pos[i as usize] - c;
                        (d - axis * d.dot(axis)).length() > 0.03
                    })
                    .min_by(|&a, &b| {
                        node_pos[a as usize]
                            .distance(c)
                            .total_cmp(&node_pos[b as usize].distance(c))
                    });
                k.map(|k| Binder::new(vec![[a0, a1, k]], &node_pos, 0.5))
            }
            Bind::Group(g) => groups
                .get(g as usize)
                .and_then(|n| built.bind_tris.get(n))
                .map(|t| Binder::new(t.clone(), &node_pos, 0.25)),
            _ => None,
        };
        // A vote for the part.
        let rigid = piece.bind == Bind::Rigid;
        let part: Option<u16> = if piece.bind == Bind::Vote || rigid {
            let mut votes: HashMap<u16, usize> = HashMap::new();
            let step = (m.len() / 600).max(1);
            for k in (0..m.len()).step_by(step) {
                if let Some((t, _, _)) = pure.bind(m.pos[k], m.nrm[k]) {
                    *votes.entry(sk.nodes[t[0] as usize].part).or_default() += 1;
                }
            }
            votes
                .into_iter()
                .filter(|(p, _)| by_part.contains_key(p))
                .max_by_key(|&(p, n)| (n, std::cmp::Reverse(p)))
                .map(|(p, _)| p)
        } else {
            None
        };
        if rigid {
            // One triangle (the voted part's nearest to the centroid) carries
            // the whole part rigidly.
            let b = part
                .and_then(|pt| {
                    binders
                        .entry(pt)
                        .or_insert_with(|| Binder::new(by_part[&pt].clone(), &node_pos, 0.15))
                        .bind(centroid, Vec3::Y)
                })
                .or_else(|| pure.bind(centroid, Vec3::Y));
            let Some((t, _, _)) = b else {
                return Err(format!("no skin triangle for the rigid part {}", piece.name));
            };
            let (p0, p1, p2) = (node_pos[t[0] as usize], node_pos[t[1] as usize], node_pos[t[2] as usize]);
            let tt = (p1 - p0).normalize();
            let nn = (p1 - p0).cross(p2 - p0).normalize();
            let bb = nn.cross(tt);
            for k in 0..m.len() {
                let d = m.pos[k] - p0;
                let n = m.nrm[k];
                out.bind_nodes.push(t);
                out.bind_coords.push([d.dot(tt), d.dot(bb), d.dot(nn)]);
                out.bind_normal.push([n.dot(tt), n.dot(bb), n.dot(nn)]);
                out.bind_rest.push(rest_edges(&node_pos, t));
                out.positions.push((m.pos[k] + built.shift).to_array());
                out.normals.push(n.to_array());
                out.uvs.push([0.0, 0.0]);
                out.material.push(m.mat[k]);
                out.colors.push(color(m.mat[k], paint));
                out.ao.push(m.ao[k].clamp(0.0, 1.0));
                out.smooth.push(0);
                out.rigid.push(1);
            }
            out.indices.extend(m.idx.iter().map(|i| i + first));
            out.pieces.push((piece.name.clone(), first, m.len() as u32));
            continue;
        }
        // Per vertex: its binding and its part. A voted piece rides its part,
        // except where that part's skin is much farther than another's (the
        // piece reaches past the physics panel): those vertices ride the
        // nearer part, and triangles are split along that boundary (vertices
        // duplicated), so the piece parts there instead of stretching.
        type B = ([u32; 3], [f32; 3], [f32; 3]);
        let mut vb: Vec<(B, u16)> = Vec::with_capacity(m.len());
        for k in 0..m.len() {
            let (p, n) = (m.pos[k], m.nrm[k]);
            let r: Option<(B, u16)> = if let Some(o) = &own {
                o.bind(p, n).map(|b| (b, u16::MAX))
            } else if let Some(pt) = part {
                let near = pure.bind_d(p, n);
                let b = binders
                    .entry(pt)
                    .or_insert_with(|| Binder::new(by_part[&pt].clone(), &node_pos, 0.15));
                let mine = b.bind_d(p, n);
                match (mine, near) {
                    (Some(a), Some(z)) => {
                        let zp = sk.nodes[z.0[0] as usize].part;
                        if a.3 > z.3 + 0.08 && zp != pt {
                            Some(((z.0, z.1, z.2), zp))
                        } else {
                            Some(((a.0, a.1, a.2), pt))
                        }
                    }
                    (Some(a), None) => Some(((a.0, a.1, a.2), pt)),
                    (None, Some(z)) => Some(((z.0, z.1, z.2), u16::MAX)),
                    _ => None,
                }
            } else {
                None
            };
            let Some(r) = r.or_else(|| fallback.bind(p, n).map(|b| (b, u16::MAX))) else {
                return Err(format!("no skin triangle for a vertex at {p:?}"));
            };
            vb.push(r);
        }
        let mut idx: Vec<u32> = m.idx.clone();
        let mut extra: Vec<(usize, B, u16)> = Vec::new();
        if part.is_some() {
            let mut dup: HashMap<(u32, u16), u32> = HashMap::new();
            for t in idx.chunks_mut(3) {
                let parts = [vb[t[0] as usize].1, vb[t[1] as usize].1, vb[t[2] as usize].1];
                if parts[0] == parts[1] && parts[1] == parts[2] {
                    continue;
                }
                let tp = if parts[1] == parts[2] { parts[1] } else { parts[0] };
                let Some(bp) = binders.get(&tp).map(|_| ()).or_else(|| {
                    by_part.get(&tp).map(|tris| {
                        binders.insert(tp, Binder::new(tris.clone(), &node_pos, 0.15));
                    })
                }) else {
                    continue;
                };
                let _ = bp;
                for v in t.iter_mut() {
                    if vb[*v as usize].1 == tp {
                        continue;
                    }
                    let key = (*v, tp);
                    let nv = if let Some(&nv) = dup.get(&key) {
                        nv
                    } else {
                        let k = *v as usize;
                        let b = binders[&tp].bind(m.pos[k], m.nrm[k]).unwrap_or(vb[k].0);
                        let nv = (m.len() + extra.len()) as u32;
                        extra.push((k, b, tp));
                        dup.insert(key, nv);
                        nv
                    };
                    *v = nv;
                }
            }
        }
        let all = (0..m.len())
            .map(|k| (k, vb[k].0))
            .chain(extra.iter().map(|&(k, b, _)| (k, b)));
        for (k, (t, c, nl)) in all {
            out.bind_nodes.push(t);
            out.bind_coords.push(c);
            out.bind_normal.push(nl);
            out.bind_rest.push(rest_edges(&node_pos, t));
            out.positions.push((m.pos[k] + built.shift).to_array());
            out.normals.push(m.nrm[k].to_array());
            out.uvs.push([0.0, 0.0]);
            out.material.push(m.mat[k]);
            out.colors.push(color(m.mat[k], paint));
            out.ao.push(m.ao[k].clamp(0.0, 1.0));
            // Smooth shading from node normals only where the piece rides the
            // body's skin (not on wheels or frames).
            out.smooth.push(if own.is_none() { m.smooth[k] } else { 0 });
            out.rigid.push(0);
        }
        let count = out.positions.len() as u32 - first;
        out.indices.extend(idx.iter().map(|i| i + first));
        out.pieces.push((piece.name.clone(), first, count));
    }
    // Node normals for smooth shading.
    out.normal_tris = sk.skin.clone();
    let nn = node_pos.len();
    let mut count = vec![0u32; nn + 1];
    for t in &out.normal_tris {
        for &i in t {
            count[i as usize + 1] += 1;
        }
    }
    for i in 0..nn {
        count[i + 1] += count[i];
    }
    let mut fill = count.clone();
    let mut list = vec![0u32; *count.last().unwrap() as usize];
    for (ti, t) in out.normal_tris.iter().enumerate() {
        for &i in t {
            list[fill[i as usize] as usize] = ti as u32;
            fill[i as usize] += 1;
        }
    }
    out.node_tri_start = count;
    out.node_tri_list = list;
    out.node_rest_normal = node_normals(&out, &node_pos)
        .iter()
        .map(|n| n.to_array())
        .collect();
    Ok(out)
}

/// Seats, dashboard, console and steering wheel, riding their frames.
fn interior(built: &Built) -> Vec<Piece> {
    let mut groups: Vec<String> = built.bind_tris.keys().cloned().collect();
    groups.sort();
    let gi = |name: &str| groups.iter().position(|g| g == name);
    let mut out = Vec::new();
    let mut solid = |name: &str, parts: &[(Vec3, Vec3, f32, u8)]| {
        let Some(g) = gi(name) else { return };
        let mut m = Mesh::default();
        for &(c, h, tilt, mt) in parts {
            let rot = glam::Quat::from_rotation_x(tilt);
            m.block(c, rot * Vec3::X, rot * Vec3::Y, rot * Vec3::Z, h, mt, 0.3);
        }
        out.push(Piece {
            name: name.into(),
            mesh: m,
            bind: Bind::Group(g as u16),
        });
    };
    for (name, x) in [("seat_fl", 0.38f32), ("seat_fr", -0.38)] {
        solid(
            name,
            &[
                (Vec3::new(x, 0.53, 2.62), Vec3::new(0.24, 0.07, 0.25), 0.0, mat::SEAT),
                (Vec3::new(x, 0.86, 2.31), Vec3::new(0.23, 0.30, 0.07), -0.25, mat::SEAT),
                (Vec3::new(x, 1.23, 2.22), Vec3::new(0.12, 0.08, 0.05), -0.25, mat::SEAT),
            ],
        );
    }
    solid(
        "seat_rear",
        &[
            (Vec3::new(0.0, 0.53, 1.80), Vec3::new(0.64, 0.07, 0.24), 0.0, mat::SEAT),
            (Vec3::new(0.0, 0.86, 1.52), Vec3::new(0.64, 0.30, 0.07), -0.35, mat::SEAT),
        ],
    );
    solid(
        "dash",
        &[
            (Vec3::new(0.0, 0.86, 3.22), Vec3::new(0.76, 0.11, 0.2), 0.15, mat::DASH),
            (Vec3::new(0.38, 0.98, 3.1), Vec3::new(0.17, 0.04, 0.08), 0.0, mat::DASH),
        ],
    );
    solid(
        "tunnel",
        &[(Vec3::new(0.0, 0.45, 2.75), Vec3::new(0.11, 0.12, 0.42), 0.0, mat::DASH)],
    );
    // Steering wheel: a torus on the column, tilted toward the driver.
    if let Some(g) = gi("column") {
        let mut m = Mesh::default();
        let c = Vec3::new(0.38, 0.95, 2.97);
        let axis = Vec3::new(0.0, 0.45, -0.89).normalize();
        m.lathe(
            c,
            axis,
            Vec3::X,
            &[(0.168, 0.0), (0.185, 0.017), (0.202, 0.0), (0.185, -0.017), (0.168, 0.0)],
            36,
            |_, _| mat::DASH,
            |_| 0.35,
            false,
        );
        m.block(c - axis * 0.03, Vec3::X, axis.cross(Vec3::X), axis, Vec3::new(0.06, 0.05, 0.04), mat::DASH, 0.8);
        out.push(Piece {
            name: "steering".into(),
            mesh: m,
            bind: Bind::Group(g as u16),
        });
    }
    out
}

fn rest_edges(pos: &[Vec3], t: [u32; 3]) -> [f32; 2] {
    let p0 = pos[t[0] as usize];
    [
        pos[t[1] as usize].distance(p0),
        pos[t[2] as usize].distance(p0),
    ]
}

/// A material's base colour (linear RGB).
pub fn color(m: u8, paint: [f32; 3]) -> [f32; 3] {
    match m {
        mat::PAINT => paint,
        mat::GLASS => [0.02, 0.025, 0.03],
        mat::CHROME => [0.75, 0.76, 0.78],
        mat::TRIM => [0.02, 0.02, 0.022],
        mat::HEADLIGHT => [0.85, 0.88, 0.9],
        mat::TAILLIGHT => [0.5, 0.015, 0.015],
        mat::TYRE => [0.025, 0.025, 0.025],
        mat::RIM => [0.55, 0.57, 0.6],
        mat::GAP => [0.1, 0.004, 0.004],
        mat::UNDERBODY => [0.04, 0.04, 0.04],
        mat::PLATE => [0.85, 0.85, 0.8],
        mat::GRILLE => [0.015, 0.015, 0.015],
        mat::SEAT => [0.05, 0.045, 0.04],
        mat::DASH => [0.03, 0.03, 0.032],
        mat::DOORCARD => [0.1, 0.095, 0.09],
        mat::HEADLINER => [0.42, 0.4, 0.37],
        mat::CARPET => [0.05, 0.05, 0.05],
        mat::LENS => [0.9, 0.92, 0.95],
        mat::DRL => [1.0, 0.98, 0.95],
        mat::REFLECTOR => [0.8, 0.8, 0.82],
        mat::TAIL_LENS => [0.55, 0.02, 0.02],
        mat::TAIL_LED => [1.0, 0.06, 0.04],
        mat::BRAKE => [0.32, 0.32, 0.33],
        mat::CALIPER => [0.5, 0.03, 0.02],
        mat::TAIL_BACK => [0.12, 0.01, 0.01],
        mat::SCREEN => [0.02, 0.025, 0.03],
        _ => [1.0, 0.0, 1.0],
    }
}

/// Per solver node its skin normal (zero off the skin).
pub fn node_normals(v: &ShowcaseVisual, nodes: &[Vec3]) -> Vec<Vec3> {
    let mut out = vec![Vec3::ZERO; nodes.len()];
    for i in 0..nodes.len().min(v.node_tri_start.len().saturating_sub(1)) {
        let mut acc = Vec3::ZERO;
        for &ti in &v.node_tri_list[v.node_tri_start[i] as usize..v.node_tri_start[i + 1] as usize]
        {
            let t = v.normal_tris[ti as usize];
            let (a, b, c) = (
                nodes[t[0] as usize],
                nodes[t[1] as usize],
                nodes[t[2] as usize],
            );
            acc += (b - a).cross(c - a);
        }
        out[i] = acc.normalize_or_zero();
    }
    out
}

/// Deform the rest mesh by node positions (the solver's order): the GPU
/// shader's formula on the CPU, for tests and offline renders.
pub fn deform(v: &ShowcaseVisual, nodes: &[Vec3]) -> (Vec<Vec3>, Vec<Vec3>) {
    let nn = node_normals(v, nodes);
    let mut pos = Vec::with_capacity(v.positions.len());
    let mut nrm = Vec::with_capacity(v.positions.len());
    for i in 0..v.positions.len() {
        let [a, b, c] = v.bind_nodes[i];
        let (p0, p1, p2) = (nodes[a as usize], nodes[b as usize], nodes[c as usize]);
        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let n = e1.cross(e2).normalize_or_zero();
        let [al, be, h] = v.bind_coords[i];
        let [r1, r2] = v.bind_rest[i];
        let t = e1.normalize_or_zero();
        let bb = n.cross(t);
        let rigid = v.rigid.get(i).copied().unwrap_or(0) == 1;
        if e1.length() > TEAR_STRETCH * r1 || e2.length() > TEAR_STRETCH * r2 {
            // Torn: the renderer cuts its triangles away (NaN here).
            pos.push(Vec3::splat(f32::NAN));
            nrm.push(Vec3::Y);
            continue;
        }
        if rigid {
            pos.push(p0 + t * al + bb * be + n * h);
        } else {
            pos.push(p0 + e1 * al + e2 * be + n * h);
        }
        let [x, y, z] = v.bind_normal[i];
        let nf = (t * x + bb * y + n * z).normalize_or_zero();
        if v.smooth[i] == 1 && !rigid {
            let w = [1.0 - al - be, al, be];
            let mut d = Vec3::from_array(v.normals[i]);
            for (k, &node) in [a, b, c].iter().enumerate() {
                d += w[k]
                    * (nn[node as usize] - Vec3::from_array(v.node_rest_normal[node as usize]));
            }
            let ns = d.normalize_or_zero();
            // A kink (smooth and frame normals > ~15° apart) keeps the facet.
            let k = super::parts::smoothstep(0.97, 0.90, ns.dot(nf));
            nrm.push(ns.lerp(nf, k).normalize_or_zero());
            continue;
        }
        nrm.push(nf);
    }
    (pos, nrm)
}
