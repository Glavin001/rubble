//! Convex hull chunks: geometry, faces and mass properties.
//!
//! A hull chunk is the convex hull of up to [`MAX_HULL_VERTICES`] points given in the
//! chunk's frame, centred on its centre of mass. Faces are planar convex polygons
//! (coplanar triangles merged), wound counter-clockwise seen from outside.
//!
//! Construction is incremental: a tetrahedron, then every point outside the current
//! hull replaces the triangles it sees by a fan to their horizon, so the surface stays a
//! closed triangle mesh whatever the input's rounding. Faces are then grown from those
//! triangles: neighbours whose corners lie on a face's plane (within the tolerance of f32
//! input) join it, and each face is the convex polygon of its corners. Hull points
//! usually come from f32 data (PhysX cooks f32 hulls; asset formats store f32), whose
//! faces are planar only to rounding, and can be visibly non-planar (thin fragments cut
//! by a solver): a face that is not planar stays several faces, and the surface remains
//! closed either way. Mass properties integrate the polyhedron exactly (Mirtich's
//! divergence theorem form, as in Eberly, "Polyhedral Mass Properties").

use crate::math::{Mat3, Vec3};

/// PhysX cooks hulls with at most 64 vertices; the stress solver uses the same limit.
pub const MAX_HULL_VERTICES: usize = 64;

/// One planar face: `normal . x = offset` for points `x` on it, normal outward.
#[derive(Clone, Debug)]
pub struct HullFace {
    /// Vertex indices, counter-clockwise seen from outside.
    pub vertices: Vec<usize>,
    pub normal: Vec3,
    pub offset: f64,
    pub area: f64,
    pub centroid: Vec3,
}

#[derive(Clone, Debug)]
pub struct ConvexHull {
    /// Hull vertices (input points that are corners of the hull).
    pub vertices: Vec<Vec3>,
    pub faces: Vec<HullFace>,
    /// The closed triangle mesh of the surface (vertex indices, counter-clockwise seen
    /// from outside), which the mass properties integrate exactly.
    pub triangles: Vec<[usize; 3]>,
}

/// Mass properties of a solid of unit density.
#[derive(Clone, Copy, Debug)]
pub struct UnitMass {
    pub volume: f64,
    pub centroid: Vec3,
    /// Inertia about the centroid, per unit density.
    pub inertia: Mat3,
}

impl ConvexHull {
    /// The convex hull of `points` (at least 4, not coplanar, at most
    /// [`MAX_HULL_VERTICES`]).
    pub fn new(points: &[Vec3]) -> Result<ConvexHull, String> {
        if points.len() < 4 {
            return Err(format!("a hull needs at least 4 points, got {}", points.len()));
        }
        if points.len() > MAX_HULL_VERTICES {
            return Err(format!("a hull has at most {MAX_HULL_VERTICES} points, got {}", points.len()));
        }
        let size = points.iter().map(|p| p.norm()).fold(0.0, f64::max).max(1e-12);
        // Points within 16 f32 ulps of the coordinates' size of a plane are on it.
        let tol = 16.0 * f32::EPSILON as f64 * size;
        // Coincident points (repeated corners are common in exported hulls) count once:
        // a repeat would make zero-area triangles, whose planes are rounding noise.
        let mut unique: Vec<Vec3> = Vec::with_capacity(points.len());
        for p in points {
            if !unique.iter().any(|q| (*q - *p).norm() <= tol) {
                unique.push(*p);
            }
        }
        let points = &unique[..];
        let n = points.len();
        if n < 4 {
            return Err(format!("a hull needs at least 4 distinct points, got {n}"));
        }
        // Initial tetrahedron: extreme point, farthest from it, farthest from that line,
        // farthest from that plane.
        let i0 = (0..n).min_by(|&a, &b| points[a].x.total_cmp(&points[b].x)).unwrap();
        let far = |f: &dyn Fn(Vec3) -> f64| (0..n).max_by(|&a, &b| f(points[a]).total_cmp(&f(points[b]))).unwrap();
        let i1 = far(&|p| (p - points[i0]).norm());
        let dir = points[i1] - points[i0];
        if dir.norm() <= tol {
            return Err("hull points coincide".into());
        }
        let dir = dir.normalized();
        let i2 = far(&|p| (p - points[i0]).cross(dir).norm());
        if (points[i2] - points[i0]).cross(dir).norm() <= tol {
            return Err("hull points are collinear".into());
        }
        let normal = dir.cross(points[i2] - points[i0]).normalized();
        let i3 = far(&|p| normal.dot(p - points[i0]).abs());
        if normal.dot(points[i3] - points[i0]).abs() <= tol {
            return Err("hull points are coplanar".into());
        }
        // Outward plane of a triangle (counter-clockwise seen from outside); a degenerate
        // triangle has none and sees nothing.
        let plane = |t: &[usize; 3]| {
            let c = (points[t[1]] - points[t[0]]).cross(points[t[2]] - points[t[0]]);
            let len = c.norm();
            (len > 0.0).then(|| (c / len, c.dot(points[t[0]]) / len))
        };
        // Each face wound so that the opposite corner is behind it (exactly).
        let coord = |p: Vec3| robust::Coord3D { x: p.x, y: p.y, z: p.z };
        let mut tris: Vec<[usize; 3]> = [([i0, i1, i2], i3), ([i0, i1, i3], i2), ([i0, i2, i3], i1), ([i1, i2, i3], i0)]
            .into_iter()
            .map(|(t, o)| if robust::orient3d(coord(points[t[0]]), coord(points[t[1]]), coord(points[t[2]]), coord(points[o])) < 0.0 { [t[0], t[2], t[1]] } else { t })
            .collect();
        // Points are added farthest first (well-shaped triangles early). Whether a point
        // sees a triangle is decided by the exact orientation predicate, so the mesh is
        // exactly convex at every step: the triangles a point sees form one connected cap,
        // and replacing them by a fan from the point to the cap's horizon keeps the mesh
        // closed and consistently wound. Points on a face's plane or inside see nothing.
        let sees = |t: &[usize; 3], p: usize| robust::orient3d(coord(points[t[0]]), coord(points[t[1]]), coord(points[t[2]]), coord(points[p])) < 0.0;
        let mut used = vec![false; n];
        for i in [i0, i1, i2, i3] {
            used[i] = true;
        }
        loop {
            let mut best: Option<(f64, usize)> = None;
            for p in (0..n).filter(|&p| !used[p]) {
                let d = tris
                    .iter()
                    .filter(|t| sees(t, p))
                    .map(|t| plane(t).map_or(0.0, |(nrm, off)| nrm.dot(points[p]) - off))
                    .fold(f64::NEG_INFINITY, f64::max);
                if d > f64::NEG_INFINITY && best.is_none_or(|b| d > b.0) {
                    best = Some((d, p));
                }
            }
            let Some((_, p)) = best else { break };
            used[p] = true;
            let visible: Vec<bool> = tris.iter().map(|t| sees(t, p)).collect();
            // The visible triangles' edges in triangle order (not the set's hash order, which
            // differs between processes and would reorder the faces and their sums).
            let edges: Vec<(usize, usize)> =
                tris.iter().zip(&visible).filter(|(_, v)| **v).flat_map(|(t, _)| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]).collect();
            let seen: std::collections::HashSet<(usize, usize)> = edges.iter().copied().collect();
            let horizon: Vec<(usize, usize)> = edges.into_iter().filter(|&(a, b)| !seen.contains(&(b, a))).collect();
            let mut kept: Vec<[usize; 3]> = tris.iter().zip(&visible).filter(|(_, v)| !**v).map(|(t, _)| *t).collect();
            kept.extend(horizon.into_iter().map(|(a, b)| [a, b, p]));
            tris = kept;
        }
        // Faces: grow each from a seed triangle over neighbours whose corners lie on the
        // seed's plane.
        let mut across: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();
        for (k, t) in tris.iter().enumerate() {
            for e in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                across.insert(e, k);
            }
        }
        let mut face_of = vec![usize::MAX; tris.len()];
        let mut groups: Vec<Vec<usize>> = Vec::new();
        // Seeds largest first: a face takes the plane of its best-conditioned triangle.
        let area = |t: &[usize; 3]| (points[t[1]] - points[t[0]]).cross(points[t[2]] - points[t[0]]).norm();
        let mut seeds: Vec<usize> = (0..tris.len()).collect();
        seeds.sort_by(|&a, &b| area(&tris[b]).total_cmp(&area(&tris[a])));
        for seed in seeds {
            if face_of[seed] != usize::MAX {
                continue;
            }
            let Some((nrm, off)) = plane(&tris[seed]) else { continue };
            let g = groups.len();
            face_of[seed] = g;
            let (mut group, mut stack) = (vec![seed], vec![seed]);
            while let Some(k) = stack.pop() {
                let t = tris[k];
                for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    if let Some(&o) = across.get(&(b, a)) {
                        // Joins if its plane is the face's within `tol` over its own
                        // extent: corners on the plane, normal tilted by at most `tol` over
                        // its longest edge (a sliver's normal is rounding noise: it stays a
                        // face of its own, whose plane still supports the hull).
                        let joins = || {
                            let u = tris[o];
                            let longest = (0..3).map(|e| (points[u[(e + 1) % 3]] - points[u[e]]).norm()).fold(0.0, f64::max);
                            u.iter().all(|&q| (nrm.dot(points[q]) - off).abs() <= tol)
                                && plane(&u).is_some_and(|(m, _)| m.cross(nrm).norm() * longest <= tol && m.dot(nrm) > 0.0)
                        };
                        if face_of[o] == usize::MAX && joins() {
                            face_of[o] = g;
                            group.push(o);
                            stack.push(o);
                        }
                    }
                }
            }
            groups.push(group);
        }
        // Vertices: the triangles' corners, indexed in order of use.
        let mut index = vec![usize::MAX; n];
        let mut vertices: Vec<Vec3> = Vec::new();
        for t in &tris {
            for &q in t {
                if index[q] == usize::MAX {
                    index[q] = vertices.len();
                    vertices.push(points[q]);
                }
            }
        }
        let mut faces = Vec::with_capacity(groups.len());
        for group in groups {
            // Area-weighted normal; the plane through the hull's support point along it.
            let normal = group.iter().fold(Vec3::ZERO, |a, &k| {
                let t = tris[k];
                a + (points[t[1]] - points[t[0]]).cross(points[t[2]] - points[t[0]])
            });
            let normal = normal.normalized();
            let offset = vertices.iter().map(|v| normal.dot(*v)).fold(f64::NEG_INFINITY, f64::max);
            let mut corners: Vec<usize> = group.iter().flat_map(|&k| tris[k]).collect();
            corners.sort_unstable();
            corners.dedup();
            let on: Vec<Vec3> = corners.iter().map(|&q| points[q]).collect();
            let polygon = convex_polygon(&on, normal);
            // A sliver can drop below three corners in the plane's 2-D hull: it stays the
            // triangle it is (its plane still bounds the hull).
            let polygon = if polygon.len() >= 3 { polygon } else { tris[group[0]].iter().map(|&q| points[q]).collect() };
            let ids: Vec<usize> = polygon.iter().map(|p| index[corners[on.iter().position(|q| q == p).unwrap()]]).collect();
            let (area, centroid) = polygon_area_centroid(&polygon, normal);
            faces.push(HullFace { vertices: ids, normal, offset, area, centroid });
        }
        if faces.len() < 4 {
            return Err("hull points are coplanar".into());
        }
        let triangles = tris.iter().map(|t| t.map(|q| index[q])).collect();
        Ok(ConvexHull { vertices, faces, triangles })
    }

    /// A hull given as its polygon faces (vertex indices into `points`, counter-clockwise
    /// seen from outside), taken as given: the faces are the hull's faces, and its mass
    /// properties integrate their fans. Checked, not recomputed: the surface must be
    /// closed (every edge met once in each direction, vertices matched by position). A
    /// face's plane is its Newell normal through the hull's support point along it, so
    /// the planes bound the hull whatever the input's rounding; a face that is not planar
    /// within f32 rounding is its fan triangles. Returned with the hull: how far the
    /// surface departs from convex (the largest distance of a face's corner behind its
    /// plane), which is how much the planes overstate the hull for contact.
    pub fn from_faces(points: &[Vec3], faces: &[Vec<usize>]) -> Result<(ConvexHull, f64), String> {
        if points.len() > MAX_HULL_VERTICES {
            return Err(format!("a hull has at most {MAX_HULL_VERTICES} points, got {}", points.len()));
        }
        if faces.len() < 4 || faces.iter().any(|f| f.len() < 3 || f.iter().any(|&i| i >= points.len())) {
            return Err("a hull needs at least 4 faces of at least 3 valid vertex indices".into());
        }
        let size = points.iter().map(|p| p.norm()).fold(0.0, f64::max).max(1e-12);
        let tol = 16.0 * f32::EPSILON as f64 * size;
        // One vertex per distinct position.
        let mut vertices: Vec<Vec3> = Vec::new();
        let index: Vec<usize> = points
            .iter()
            .map(|p| match vertices.iter().position(|v| v == p) {
                Some(i) => i,
                None => {
                    vertices.push(*p);
                    vertices.len() - 1
                }
            })
            .collect();
        let polygons: Vec<Vec<usize>> = faces
            .iter()
            .map(|f| {
                let mut g: Vec<usize> = f.iter().map(|&i| index[i]).collect();
                g.dedup();
                while g.len() > 1 && g[0] == g[g.len() - 1] {
                    g.pop();
                }
                g
            })
            .filter(|g| g.len() >= 3)
            .collect();
        let mut edges = std::collections::HashMap::new();
        for g in &polygons {
            for k in 0..g.len() {
                *edges.entry((g[k], g[(k + 1) % g.len()])).or_insert(0usize) += 1;
            }
        }
        if let Some((e, _)) = edges.iter().find(|(&(a, b), &n)| n != 1 || edges.get(&(b, a)) != Some(&1)) {
            return Err(format!("hull faces are not a closed surface (edge {e:?})"));
        }
        // Newell's normal (exact for a planar polygon) and the plane through the hull's
        // support point along it; how far the polygon's own corners fall behind it.
        let plane = |g: &[usize]| {
            let mut normal = Vec3::ZERO;
            for k in 0..g.len() {
                let (a, b) = (vertices[g[k]], vertices[g[(k + 1) % g.len()]]);
                normal += Vec3::new((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y));
            }
            let normal = normal.normalized();
            let offset = vertices.iter().map(|v| normal.dot(*v)).fold(f64::NEG_INFINITY, f64::max);
            let behind = g.iter().map(|&q| offset - normal.dot(vertices[q])).fold(0.0, f64::max);
            (normal, offset, behind)
        };
        let mut out = Vec::with_capacity(polygons.len());
        let mut triangles = Vec::new();
        let mut departure: f64 = 0.0;
        for g in polygons {
            let fan: Vec<Vec<usize>> = (1..g.len() - 1).map(|k| vec![g[0], g[k], g[k + 1]]).collect();
            triangles.extend(fan.iter().map(|t| [t[0], t[1], t[2]]));
            // A face planar only to a few tens of micrometres (a clipped polytope's
            // rounding) is its fan triangles, each exactly planar.
            let pieces = if plane(&g).2 <= tol { vec![g] } else { fan };
            for g in pieces {
                let (normal, offset, behind) = plane(&g);
                if !normal.is_finite() {
                    return Err("a hull face has no area".into());
                }
                departure = departure.max(behind);
                let polygon: Vec<Vec3> = g.iter().map(|&q| vertices[q]).collect();
                let (area, centroid) = polygon_area_centroid(&polygon, normal);
                out.push(HullFace { vertices: g, normal, offset, area, centroid });
            }
        }
        Ok((ConvexHull { vertices, faces: out, triangles }, departure))
    }

    /// A part's hull: its given faces when it has them (`from_faces`), else the convex
    /// hull of its points.
    pub fn of_part(points: &[Vec3], faces: Option<&[Vec<usize>]>) -> Result<ConvexHull, String> {
        match faces {
            Some(f) => ConvexHull::from_faces(points, f).map(|(h, _)| h),
            None => ConvexHull::new(points),
        }
    }

    /// Exact volume, centroid and inertia (about the centroid) at unit density.
    pub fn unit_mass(&self) -> UnitMass {
        // Integrals of 1, x, y, z, x^2, y^2, z^2, xy, yz, zx over the solid, from the
        // surface triangles (Eberly's form of Mirtich's algorithm).
        let mut intg = [0.0f64; 10];
        for t in &self.triangles {
            let v0 = self.vertices[t[0]];
            {
                let (v1, v2) = (self.vertices[t[1]], self.vertices[t[2]]);
                let d = (v1 - v0).cross(v2 - v0);
                let sub = |a: f64, b: f64, c: f64| {
                    let t0 = a + b;
                    let f1 = t0 + c;
                    let t1 = a * a;
                    let t2 = t1 + b * t0;
                    let f2 = t2 + c * f1;
                    let f3 = a * t1 + b * t2 + c * f2;
                    (f1, f2, f3, f2 + a * (f1 + a), f2 + b * (f1 + b), f2 + c * (f1 + c))
                };
                let (f1x, f2x, f3x, g0x, g1x, g2x) = sub(v0.x, v1.x, v2.x);
                let (_, f2y, f3y, g0y, g1y, g2y) = sub(v0.y, v1.y, v2.y);
                let (_, f2z, f3z, g0z, g1z, g2z) = sub(v0.z, v1.z, v2.z);
                intg[0] += d.x * f1x;
                intg[1] += d.x * f2x;
                intg[2] += d.y * f2y;
                intg[3] += d.z * f2z;
                intg[4] += d.x * f3x;
                intg[5] += d.y * f3y;
                intg[6] += d.z * f3z;
                intg[7] += d.x * (v0.y * g0x + v1.y * g1x + v2.y * g2x);
                intg[8] += d.y * (v0.z * g0y + v1.z * g1y + v2.z * g2y);
                intg[9] += d.z * (v0.x * g0z + v1.x * g1z + v2.x * g2z);
            }
        }
        let k = [1.0 / 6.0, 1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0, 1.0 / 60.0, 1.0 / 60.0, 1.0 / 60.0, 1.0 / 120.0, 1.0 / 120.0, 1.0 / 120.0];
        for (v, k) in intg.iter_mut().zip(k) {
            *v *= k;
        }
        let volume = intg[0];
        let c = Vec3::new(intg[1], intg[2], intg[3]) * (1.0 / volume);
        let ixx = intg[5] + intg[6] - volume * (c.y * c.y + c.z * c.z);
        let iyy = intg[4] + intg[6] - volume * (c.z * c.z + c.x * c.x);
        let izz = intg[4] + intg[5] - volume * (c.x * c.x + c.y * c.y);
        let ixy = -(intg[7] - volume * c.x * c.y);
        let iyz = -(intg[8] - volume * c.y * c.z);
        let izx = -(intg[9] - volume * c.z * c.x);
        let inertia = Mat3 { m: [[ixx, ixy, izx], [ixy, iyy, iyz], [izx, iyz, izz]] };
        UnitMass { volume, centroid: c, inertia }
    }

    /// The hull moved by `-shift` (to put its centroid at the origin).
    pub fn translated(&self, shift: Vec3) -> ConvexHull {
        let vertices = self.vertices.iter().map(|v| *v - shift).collect();
        let faces = self
            .faces
            .iter()
            .map(|f| HullFace { offset: f.offset - f.normal.dot(shift), centroid: f.centroid - shift, ..f.clone() })
            .collect();
        ConvexHull { vertices, faces, triangles: self.triangles.clone() }
    }

    /// Half extents of the axis-aligned box about the origin that contains the hull.
    pub fn half_extents(&self) -> Vec3 {
        self.vertices.iter().fold(Vec3::ZERO, |h, v| h.component_max(v.abs()))
    }

    /// Distance of `p` (hull frame) outside the hull's nearest face plane, and that
    /// face: negative inside (minus the depth to the nearest face).
    pub fn signed_distance(&self, p: Vec3) -> (f64, usize) {
        self.faces
            .iter()
            .enumerate()
            .map(|(i, f)| (f.normal.dot(p) - f.offset, i))
            .fold((f64::NEG_INFINITY, 0), |a, b| if b.0 > a.0 { b } else { a })
    }

    /// Contact sample points (hull frame): the corners pulled 10% towards the centre
    /// plus the face centroids, as for boxes.
    pub fn samples(&self) -> impl Iterator<Item = Vec3> + '_ {
        self.vertices.iter().map(|v| *v * 0.9).chain(self.faces.iter().map(|f| f.centroid))
    }
}

/// The contact patch shared by two touching chunks: where a face of `a` and a face of
/// `b` lie in one plane facing each other, the overlap of the two face polygons.
#[derive(Clone, Copy, Debug)]
pub struct Patch {
    pub centroid: Vec3,
    /// Unit normal from `a` towards `b`.
    pub normal: Vec3,
    pub area: f64,
    /// Principal in-plane direction and the widths of the rectangle with the patch's
    /// area-moments about its principal axes (`width[0]` along `tangent`).
    pub tangent: Vec3,
    pub width: [f64; 2],
}

/// The patch between hull `a` at `pa` and hull `b` at `pb` (same orientation, world
/// offsets), for faces within `gap` of each other and facing along `normal` (a to b).
pub fn contact_patch(a: &ConvexHull, pa: Vec3, b: &ConvexHull, pb: Vec3, normal: Vec3, gap: f64) -> Option<Patch> {
    let u = normal.any_perpendicular().normalized();
    let v = normal.cross(u);
    let mut best: Option<Patch> = None;
    for fa in a.faces.iter().filter(|f| f.normal.dot(normal) > 0.995) {
        for fb in b.faces.iter().filter(|f| f.normal.dot(-normal) > 0.995) {
            let (oa, ob) = (fa.offset + fa.normal.dot(pa), fb.offset + fb.normal.dot(pb));
            if (oa + ob).abs() > gap {
                continue;
            }
            let flat = |h: &ConvexHull, f: &HullFace, p: Vec3| -> Vec<(f64, f64)> {
                let mut poly: Vec<(f64, f64)> = f.vertices.iter().map(|&i| ((h.vertices[i] + p).dot(u), (h.vertices[i] + p).dot(v))).collect();
                if signed_area(&poly) < 0.0 {
                    poly.reverse();
                }
                poly
            };
            let overlap = clip(&flat(a, fa, pa), &flat(b, fb, pb));
            let Some((area, cu, cv, iuu, ivv, iuv)) = moments(&overlap) else { continue };
            if best.is_some_and(|p| p.area >= area) {
                continue;
            }
            // Principal axes of the in-plane second moments.
            let angle = 0.5 * (2.0 * iuv).atan2(iuu - ivv);
            let (c, s) = (angle.cos(), angle.sin());
            let i0 = iuu * c * c + 2.0 * iuv * c * s + ivv * s * s;
            let i1 = iuu * s * s - 2.0 * iuv * c * s + ivv * c * c;
            let plane = 0.5 * (oa - ob);
            best = Some(Patch {
                centroid: u * cu + v * cv + normal * plane,
                normal,
                area,
                tangent: (u * c + v * s).normalized(),
                width: [(12.0 * i0.max(0.0) / area).sqrt(), (12.0 * i1.max(0.0) / area).sqrt()],
            });
        }
    }
    best
}

fn signed_area(p: &[(f64, f64)]) -> f64 {
    0.5 * (0..p.len()).map(|i| p[i].0 * p[(i + 1) % p.len()].1 - p[(i + 1) % p.len()].0 * p[i].1).sum::<f64>()
}

/// Intersection of two counter-clockwise convex polygons (Sutherland-Hodgman).
fn clip(subject: &[(f64, f64)], window: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut out = subject.to_vec();
    for i in 0..window.len() {
        let (e0, e1) = (window[i], window[(i + 1) % window.len()]);
        let side = |p: (f64, f64)| (e1.0 - e0.0) * (p.1 - e0.1) - (e1.1 - e0.1) * (p.0 - e0.0);
        let input = std::mem::take(&mut out);
        for j in 0..input.len() {
            let (p, q) = (input[j], input[(j + 1) % input.len()]);
            let (sp, sq) = (side(p), side(q));
            if sp >= 0.0 {
                out.push(p);
            }
            if (sp >= 0.0) != (sq >= 0.0) {
                let t = sp / (sp - sq);
                out.push((p.0 + t * (q.0 - p.0), p.1 + t * (q.1 - p.1)));
            }
        }
        if out.is_empty() {
            break;
        }
    }
    out
}

/// Area, centroid and second moments about the centroid (`int u^2`, `int v^2`,
/// `int uv`) of a counter-clockwise polygon; `None` if degenerate.
fn moments(p: &[(f64, f64)]) -> Option<(f64, f64, f64, f64, f64, f64)> {
    if p.len() < 3 {
        return None;
    }
    let (mut a, mut cx, mut cy, mut ixx, mut iyy, mut ixy) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..p.len() {
        let ((x0, y0), (x1, y1)) = (p[i], p[(i + 1) % p.len()]);
        let cr = x0 * y1 - x1 * y0;
        a += cr;
        cx += (x0 + x1) * cr;
        cy += (y0 + y1) * cr;
        ixx += (x0 * x0 + x0 * x1 + x1 * x1) * cr;
        iyy += (y0 * y0 + y0 * y1 + y1 * y1) * cr;
        ixy += (x0 * y1 + 2.0 * x0 * y0 + 2.0 * x1 * y1 + x1 * y0) * cr;
    }
    a *= 0.5;
    if a <= 1e-12 {
        return None;
    }
    let (cx, cy) = (cx / (6.0 * a), cy / (6.0 * a));
    let (ixx, iyy, ixy) = (ixx / 12.0 - a * cx * cx, iyy / 12.0 - a * cy * cy, ixy / 24.0 - a * cx * cy);
    Some((a, cx, cy, ixx, iyy, ixy))
}

/// Counter-clockwise (about `normal`) convex polygon through the coplanar `points`.
fn convex_polygon(points: &[Vec3], normal: Vec3) -> Vec<Vec3> {
    let u = normal.any_perpendicular().normalized();
    let v = normal.cross(u);
    let mut pts: Vec<(f64, f64, Vec3)> = points.iter().map(|p| (p.dot(u), p.dot(v), *p)).collect();
    pts.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    pts.dedup_by(|a, b| (a.2 - b.2).norm() < 1e-12);
    let cross = |o: &(f64, f64, Vec3), a: &(f64, f64, Vec3), b: &(f64, f64, Vec3)| (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0);
    // Andrew's monotone chain: lower then upper hull, collinear points dropped.
    let mut hull: Vec<(f64, f64, Vec3)> = Vec::new();
    for pass in 0..2 {
        let start = hull.len();
        let iter: Box<dyn Iterator<Item = &(f64, f64, Vec3)>> = if pass == 0 { Box::new(pts.iter()) } else { Box::new(pts.iter().rev()) };
        for p in iter {
            while hull.len() >= start + 2 && cross(&hull[hull.len() - 2], &hull[hull.len() - 1], p) <= 1e-18 {
                hull.pop();
            }
            hull.push(*p);
        }
        hull.pop();
    }
    hull.into_iter().map(|p| p.2).collect()
}

/// Area and centroid of a planar convex polygon.
fn polygon_area_centroid(poly: &[Vec3], normal: Vec3) -> (f64, Vec3) {
    let o = poly[0];
    let (mut area, mut c) = (0.0, Vec3::ZERO);
    for w in poly[1..].windows(2) {
        let a = 0.5 * (w[0] - o).cross(w[1] - o).dot(normal);
        area += a;
        c += (o + w[0] + w[1]) * (a / 3.0);
    }
    (area, c * (1.0 / area))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_points(h: Vec3) -> Vec<Vec3> {
        (0..8).map(|i| Vec3::new(if i & 1 == 0 { -h.x } else { h.x }, if i & 2 == 0 { -h.y } else { h.y }, if i & 4 == 0 { -h.z } else { h.z })).collect()
    }

    #[test]
    fn box_hull_has_six_quads_and_exact_mass_properties() {
        let h = Vec3::new(0.3, 0.2, 0.1);
        let hull = ConvexHull::new(&box_points(h)).unwrap();
        assert_eq!(hull.faces.len(), 6);
        assert!(hull.faces.iter().all(|f| f.vertices.len() == 4));
        let m = hull.unit_mass();
        let v = 8.0 * h.x * h.y * h.z;
        assert!((m.volume - v).abs() < 1e-12 * v);
        assert!(m.centroid.norm() < 1e-12);
        let ixx = v * (4.0 * h.y * h.y + 4.0 * h.z * h.z) / 12.0;
        assert!((m.inertia.m[0][0] - ixx).abs() < 1e-12 * ixx);
        assert!(m.inertia.m[0][1].abs() < 1e-15);
        let total: f64 = hull.faces.iter().map(|f| f.area).sum();
        assert!((total - 8.0 * (h.x * h.y + h.y * h.z + h.z * h.x)).abs() < 1e-12);
    }

    #[test]
    fn stacked_boxes_share_their_full_face_as_the_patch() {
        let lower = ConvexHull::new(&box_points(Vec3::new(0.3, 0.2, 0.1))).unwrap();
        let upper = ConvexHull::new(&box_points(Vec3::new(0.2, 0.2, 0.1))).unwrap();
        // Upper box offset by 0.1 in x: overlap is x in [-0.1, 0.3] x y in [-0.2, 0.2].
        let p = contact_patch(&lower, Vec3::ZERO, &upper, Vec3::new(0.1, 0.0, 0.2), Vec3::Z, 1e-9).unwrap();
        assert!((p.area - 0.4 * 0.4).abs() < 1e-12, "{}", p.area);
        assert!((p.centroid - Vec3::new(0.1, 0.0, 0.1)).norm() < 1e-12, "{:?}", p.centroid);
        let w = [p.width[0].max(p.width[1]), p.width[0].min(p.width[1])];
        assert!((w[0] - 0.4).abs() < 1e-9 && (w[1] - 0.4).abs() < 1e-9, "{w:?}");
        assert!(contact_patch(&lower, Vec3::ZERO, &upper, Vec3::new(0.0, 0.0, 0.25), Vec3::Z, 1e-3).is_none());
    }

    #[test]
    fn tetrahedron_and_interior_points() {
        let mut pts = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
        pts.push(Vec3::splat(0.1)); // interior: dropped
        let hull = ConvexHull::new(&pts).unwrap();
        assert_eq!((hull.vertices.len(), hull.faces.len()), (4, 4));
        let m = hull.unit_mass();
        assert!((m.volume - 1.0 / 6.0).abs() < 1e-12);
        assert!((m.centroid - Vec3::splat(0.25)).norm() < 1e-12);
        let (d, _) = hull.signed_distance(Vec3::splat(0.2));
        assert!(d < 0.0);
        assert!(hull.signed_distance(Vec3::splat(1.0)).0 > 0.0);
    }

    /// A fragment hull exported by Shardwright (f32 coordinates in the asset frame, a
    /// thin slab with repeated corners and faces planar only to rounding): the same
    /// solid in any frame, with the volume of its exact hull (scipy/qhull: 0.0156395 m^3).
    #[test]
    fn f32_hull_is_frame_independent() {
        let pts: Vec<Vec3> = [
            [7.930295467376709, 6.399499893188477, 8.132719993591309],
            [8.108762741088867, 6.399499893188477, 7.913223743438721],
            [8.1087646484375, 6.399499893188477, 7.91260290145874],
            [8.031707763671875, 6.399499893188477, 7.810866832733154],
            [7.933857440948486, 6.399499893188477, 7.784193515777588],
            [7.686007022857666, 6.399499893188477, 7.9024834632873535],
            [7.691296577453613, 6.399499893188477, 7.987177848815918],
            [7.921714305877685, 6.200500011444092, 8.086097717285156],
            [8.069902420043945, 6.200500011444092, 7.903841495513916],
            [7.666600704193115, 6.261756896972656, 7.869978904724121],
            [7.659938335418701, 6.200500011444092, 7.887030124664307],
            [7.66251277923584, 6.200500011444092, 7.928252220153809],
            [7.699060440063477, 6.200500011444092, 7.835912704467773],
            [7.763654708862305, 6.200500011444092, 7.805084228515625],
            [8.069958686828613, 6.200500011444092, 7.888580322265625],
            [8.069902420043945, 6.200500011444092, 7.903841495513916],
            [7.921714305877685, 6.200500011444092, 8.086097717285156],
            [8.08887004852295, 6.296735763549805, 7.8611955642700195],
            [8.069958686828613, 6.200500011444092, 7.888580322265625],
            [8.08887004852295, 6.296735763549805, 7.8611955642700195]
        ]
        .iter()
        .map(|p| Vec3::new(p[0], p[1], p[2]))
        .collect();
        let a = ConvexHull::new(&pts).unwrap();
        let rot = crate::math::Quat::from_axis_angle(Vec3::new(0.3, -0.5, 0.8).normalized(), 0.7).to_mat3();
        let b = ConvexHull::new(&pts.iter().map(|p| rot * *p + Vec3::new(-3.0, 1.0, 2.0)).collect::<Vec<_>>()).unwrap();
        let (ua, ub) = (a.unit_mass(), b.unit_mass());
        assert!((ua.volume - 0.01563953536248381).abs() < 1e-12, "volume {}", ua.volume);
        assert!((ua.volume - ub.volume).abs() < 1e-12 * ua.volume);
        assert!((rot * ua.centroid + Vec3::new(-3.0, 1.0, 2.0) - ub.centroid).norm() < 1e-9);
        // Every input point is inside every face plane.
        for p in &pts {
            assert!(a.faces.iter().all(|f| f.normal.dot(*p) - f.offset < 1e-9), "point outside a face");
        }
    }

    /// Faces taken as given match the hull of the points; a duplicated (overlapping)
    /// face is not a closed surface.
    #[test]
    fn hull_from_given_faces() {
        let h = Vec3::new(0.3, 0.2, 0.1);
        let pts = box_points(h);
        // Corner i has x from bit 0, y from bit 1, z from bit 2; faces counter-clockwise
        // seen from outside.
        let faces = vec![vec![0, 2, 3, 1], vec![4, 5, 7, 6], vec![0, 1, 5, 4], vec![2, 6, 7, 3], vec![0, 4, 6, 2], vec![1, 3, 7, 5]];
        let (given, departure) = ConvexHull::from_faces(&pts, &faces).unwrap();
        let built = ConvexHull::new(&pts).unwrap();
        assert_eq!(given.faces.len(), 6);
        assert!(departure < 1e-15);
        assert!((given.unit_mass().volume - built.unit_mass().volume).abs() < 1e-15);
        assert!((given.unit_mass().inertia.m[0][0] - built.unit_mass().inertia.m[0][0]).abs() < 1e-15);
        let mut doubled = faces.clone();
        doubled.push(vec![0, 2, 3]);
        assert!(ConvexHull::from_faces(&pts, &doubled).is_err());
    }
}
