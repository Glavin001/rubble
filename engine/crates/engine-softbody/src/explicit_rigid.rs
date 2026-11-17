//! Explicit vehicles against rigid bodies (`engine/docs/explicit-vehicles-in-world.md`):
//! the CPU reference of the two-way coupling.
//!
//! A [`ProxySet`] is the rigid bodies (and their collider shapes) one vehicle may touch in
//! one World window. Set it on [`ExplicitBeams::rigid`] and step the vehicle as usual: each
//! step, before the node pass, the set counts the nodes inside each body's shapes (mass
//! splitting); the node pass, after the static shapes, adds the two-body form of the
//! contact rule for every node inside a proxy shape ([`ProxySet::node_contact`]); after
//! the node pass the bodies integrate the reaction ([`ProxySet::end_step`]). Per shape the
//! window's impulse and its moment are kept for the World ([`ProxyShape::impulse`]).
//!
//! The per-step sums of a shape's reaction are integer fixed point (1/32 N and 1/32 N m,
//! as the GPU's): integer sums do not depend on their order, so the GPU (atomics) and the
//! CPU sum the same values.
//!
//! [`ExplicitBeams::rigid`]: crate::explicit_beams::ExplicitBeams::rigid

use crate::explicit_beams::{
    contact_force, shape_face, ContactSurface, Node, StaticShape, NODE_NO_GROUND, STEP,
};
use crate::explicit_surface::SurfaceShape;
use engine_schema::feature_query as fq;
use glam::{Mat3, Quat, Vec3};

/// A convex hull's vertices and edges from its planes (n · x ≤ d inside): every
/// triple of planes meeting at a point inside all of them (merged within 10 µm), and
/// an edge between two vertices that share two planes.
pub fn hull_from_planes(planes: &[(Vec3, f32)]) -> (Vec<Vec3>, Vec<[u32; 2]>) {
    let mut verts: Vec<(Vec3, Vec<usize>)> = Vec::new();
    let n = planes.len();
    for i in 0..n {
        for j in i + 1..n {
            for k in j + 1..n {
                let (a, b, c) = (planes[i].0, planes[j].0, planes[k].0);
                let det = a.dot(b.cross(c));
                if det.abs() < 1e-9 {
                    continue;
                }
                let x = (b.cross(c) * planes[i].1 + c.cross(a) * planes[j].1 + a.cross(b) * planes[k].1) / det;
                if planes.iter().any(|&(pn, pd)| pn.dot(x) - pd > 1e-5) {
                    continue;
                }
                let on: Vec<usize> = (0..n).filter(|&p| (planes[p].0.dot(x) - planes[p].1).abs() <= 1e-5).collect();
                match verts.iter_mut().find(|(v, _)| v.distance(x) <= 1e-5) {
                    Some((_, set)) => {
                        for p in on {
                            if !set.contains(&p) {
                                set.push(p);
                            }
                        }
                    }
                    None => verts.push((x, on)),
                }
            }
        }
    }
    let mut edges = Vec::new();
    for a in 0..verts.len() {
        for b in a + 1..verts.len() {
            let shared = verts[a].1.iter().filter(|p| verts[b].1.contains(p)).count();
            if shared >= 2 {
                edges.push([a as u32, b as u32]);
            }
        }
    }
    (verts.into_iter().map(|(v, _)| v).collect(), edges)
}

/// Fixed-point scale of the per-step sums (the explicit kernel's `FIXED`).
pub const FIXED: f32 = 32.0;

/// A proxy shape's geometry, in its own frame.
#[derive(Clone, Debug, PartialEq)]
pub enum ProxyGeom {
    /// Half extents.
    Box { half: Vec3 },
    Sphere { radius: f32 },
    /// Along the shape's local Y, segment ends at ±`half_len`.
    Capsule { radius: f32, half_len: f32 },
    /// Convex hull: outward face planes (normal, offset) in the shape frame (scale
    /// applied): inside when `n · x − d < 0` for every plane.
    /// `radius` bounds the hull about the shape's origin (its cull sphere).
    Hull { planes: Vec<(Vec3, f32)>, radius: f32 },
}

/// One collider of a proxy body.
#[derive(Clone, Debug, PartialEq)]
pub struct ProxyShape {
    /// The body it belongs to (index in [`ProxySet::bodies`]).
    pub body: u32,
    /// Its pose in the body's frame (centre of mass, principal axes).
    pub local_pos: Vec3,
    pub local_rot: Quat,
    pub geom: ProxyGeom,
    /// The engine collider it stands for, and its stress chunk (none: not destructible).
    pub collider: u32,
    pub chunk: Option<u32>,
    /// The window's impulse on it (N s) and its moment about `reference` (N m s).
    pub impulse: Vec3,
    pub moment: Vec3,
    /// Its world position at the window's start (the moment's reference).
    pub reference: Vec3,
    /// The largest one-step force on it this window (N).
    pub peak: f32,
    /// This step's fixed-point sums: force, moment about `reference`.
    sum: [i64; 6],
    /// A hull's vertices (scale applied) and edges in its own frame: the World's hull
    /// records ([`ProxyShape::with_hull_geometry`]), or else from its planes, once.
    hull_local: Option<(Vec<Vec3>, Vec<[u32; 2]>)>,
    /// This step's world geometry and cull box.
    world: Option<(WorldGeom, Vec3, Vec3)>,
    /// This step's body velocity when the body is fast ([`ProxySet::cast_velocities`]).
    fast: Option<Vec3>,
}

#[derive(Clone, Debug, PartialEq)]
enum WorldGeom {
    Static(StaticShape),
    /// Planes in the world frame.
    Hull(Vec<(Vec3, f32)>),
}

/// A rigid body as one vehicle's frame sees it.
#[derive(Clone, Debug, PartialEq)]
pub struct RigidProxy {
    /// The engine body.
    pub body: u32,
    /// Centre of mass and principal-axes orientation (world).
    pub pos: Vec3,
    pub rot: Quat,
    pub vel: Vec3,
    pub ang_vel: Vec3,
    /// 0: static, anchored or kinematic (infinite mass).
    pub inv_mass: f32,
    /// Principal inverse inertia (body frame).
    pub inv_inertia: Vec3,
    /// Vehicles holding this body as a proxy this window (mass splitting), ≥ 1.
    pub share: f32,
    /// This step's contacts (mass splitting).
    pub count: u32,
}

impl RigidProxy {
    pub fn is_static(&self) -> bool {
        self.inv_mass == 0.0
    }

    /// World inverse inertia applied to `v`.
    pub fn inv_inertia_mul(&self, v: Vec3) -> Vec3 {
        self.rot * (self.inv_inertia * (self.rot.conjugate() * v))
    }

    /// Linear and angular momentum about the world origin (finite mass only).
    pub fn momentum(&self) -> (Vec3, Vec3) {
        if self.is_static() {
            return (Vec3::ZERO, Vec3::ZERO);
        }
        let m = 1.0 / self.inv_mass;
        let inertia = |w: Vec3| {
            let local = self.rot.conjugate() * w;
            let i = Vec3::new(
                if self.inv_inertia.x > 0.0 { 1.0 / self.inv_inertia.x } else { 0.0 },
                if self.inv_inertia.y > 0.0 { 1.0 / self.inv_inertia.y } else { 0.0 },
                if self.inv_inertia.z > 0.0 { 1.0 / self.inv_inertia.z } else { 0.0 },
            );
            self.rot * (i * local)
        };
        let p = self.vel * m;
        (p, self.pos.cross(p) + inertia(self.ang_vel))
    }

    /// Kinetic energy (finite mass only).
    pub fn kinetic_energy(&self) -> f64 {
        if self.is_static() {
            return 0.0;
        }
        let m = 1.0 / f64::from(self.inv_mass);
        let w = self.rot.conjugate() * self.ang_vel;
        let rot: f64 = [
            (w.x, self.inv_inertia.x),
            (w.y, self.inv_inertia.y),
            (w.z, self.inv_inertia.z),
        ]
        .iter()
        .map(|&(w, ii)| if ii > 0.0 { f64::from(w) * f64::from(w) / f64::from(ii) } else { 0.0 })
        .sum();
        0.5 * m * f64::from(self.vel.length_squared()) + 0.5 * rot
    }
}

/// The bodies and shapes one vehicle meets in a window.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProxySet {
    pub bodies: Vec<RigidProxy>,
    pub shapes: Vec<ProxyShape>,
    /// Integrate the bodies inside the frame (default). Off: they move with their
    /// window-start velocity only (a measurement of the coupling's substep effect).
    pub frozen: bool,
}

fn fixed(x: f32) -> i64 {
    (x * FIXED).round() as i64
}

fn unfixed(x: i64) -> f32 {
    x as f32 / FIXED
}

impl ProxyShape {
    pub fn new(body: u32, local_pos: Vec3, local_rot: Quat, geom: ProxyGeom, collider: u32, chunk: Option<u32>) -> Self {
        Self {
            body,
            local_pos,
            local_rot,
            geom,
            collider,
            chunk,
            impulse: Vec3::ZERO,
            moment: Vec3::ZERO,
            reference: Vec3::ZERO,
            peak: 0.0,
            sum: [0; 6],
            hull_local: None,
            world: None,
            fast: None,
        }
    }

    /// A hull's vertices (in its frame, the collider's scale applied) and edges (vertex
    /// pairs) as the World's hull records hold them (`engine_schema::convex`), for the
    /// surface contact; without them they are found from the planes.
    pub fn with_hull_geometry(mut self, verts: Vec<Vec3>, edges: Vec<[u32; 2]>) -> Self {
        self.hull_local = Some((verts, edges));
        self
    }

    /// A hull's local vertices and edges when given or found already (the GPU records
    /// carry them then, for the surface contact).
    pub fn hull_geometry_known(&self) -> Option<&(Vec<Vec3>, Vec<[u32; 2]>)> {
        self.hull_local.as_ref()
    }

    /// A hull's local vertices and edges (from its planes when not given).
    pub fn hull_geometry(&mut self) -> Option<&(Vec<Vec3>, Vec<[u32; 2]>)> {
        let ProxyGeom::Hull { planes, .. } = &self.geom else {
            return None;
        };
        if self.hull_local.is_none() {
            self.hull_local = Some(hull_from_planes(planes));
        }
        self.hull_local.as_ref()
    }

    /// The window's angular impulse about point `c` (N m s).
    pub fn angular_impulse_about(&self, c: Vec3) -> Vec3 {
        self.moment + (self.reference - c).cross(self.impulse)
    }
}

/// The shape's world pose from its body's.
fn shape_pose(b: &RigidProxy, s: &ProxyShape) -> (Vec3, Quat) {
    (b.pos + b.rot * s.local_pos, b.rot * s.local_rot)
}

/// The face of a hull (world planes) a point strictly inside is pushed out through:
/// the plane nearest the point (the first of equals), as the box rule.
pub fn hull_face(planes: &[(Vec3, f32)], x: Vec3) -> Option<Vec3> {
    let mut best: Option<(f32, Vec3)> = None;
    for &(n, d) in planes {
        let s = n.x * x.x + n.y * x.y + n.z * x.z - d;
        if !(s < 0.0) {
            return None;
        }
        if best.is_none_or(|(b, _)| s > b) {
            best = Some((s, n));
        }
    }
    best.map(|(_, n)| n)
}

impl ProxySet {
    /// Start a window: per-shape totals cleared, the moments' reference points set.
    pub fn begin_window(&mut self) {
        for i in 0..self.shapes.len() {
            let (p, _) = shape_pose(&self.bodies[self.shapes[i].body as usize], &self.shapes[i]);
            let s = &mut self.shapes[i];
            s.impulse = Vec3::ZERO;
            s.moment = Vec3::ZERO;
            s.peak = 0.0;
            s.reference = p;
        }
    }

    fn place_shapes(&mut self) {
        for s in &mut self.shapes {
            let b = &self.bodies[s.body as usize];
            let (p, q) = shape_pose(b, s);
            let (geom, lo, hi) = match &s.geom {
                ProxyGeom::Box { half } => {
                    let sh = StaticShape::oriented_box(p, *half, q);
                    let bd = sh.bounds();
                    (WorldGeom::Static(sh), bd.lo, bd.hi)
                }
                ProxyGeom::Sphere { radius } => {
                    let sh = StaticShape::sphere(p, *radius);
                    let bd = sh.bounds();
                    (WorldGeom::Static(sh), bd.lo, bd.hi)
                }
                ProxyGeom::Capsule { radius, half_len } => {
                    let axis = q * Vec3::Y * *half_len;
                    let sh = StaticShape::capsule(p - axis, p + axis, *radius);
                    let bd = sh.bounds();
                    (WorldGeom::Static(sh), bd.lo, bd.hi)
                }
                ProxyGeom::Hull { planes, radius } => {
                    let m = Mat3::from_quat(q);
                    let world: Vec<(Vec3, f32)> = planes
                        .iter()
                        .map(|&(n, d)| {
                            let wn = m * n;
                            (wn, d + wn.dot(p))
                        })
                        .collect();
                    let r = *radius + 0.01;
                    (WorldGeom::Hull(world), p - Vec3::splat(r), p + Vec3::splat(r))
                }
            };
            s.world = Some((geom, lo, hi));
        }
    }

    fn face(s: &ProxyShape, x: Vec3) -> Option<Vec3> {
        let (g, lo, hi) = s.world.as_ref()?;
        match g {
            WorldGeom::Static(sh) => shape_face(sh, &crate::explicit_beams::StaticBox { lo: *lo, hi: *hi }, x),
            WorldGeom::Hull(planes) => {
                if x.cmple(*lo).any() || x.cmpge(*hi).any() {
                    return None;
                }
                hull_face(planes, x)
            }
        }
    }

    /// Before the node pass: shapes placed at the bodies' poses, and each body's contact
    /// count (nodes inside its shapes, on the positions the node pass tests).
    pub fn begin_step(&mut self, origin: Vec3, nodes: &[Node]) {
        self.place_shapes();
        let fast = self.cast_velocities_now();
        for (s, f) in self.shapes.iter_mut().zip(fast) {
            s.fast = f;
        }
        for b in &mut self.bodies {
            b.count = 0;
        }
        for n in nodes {
            if n.has(NODE_NO_GROUND) {
                continue;
            }
            let x = origin + n.pos;
            for s in &self.shapes {
                if Self::face(s, x).is_some() {
                    self.bodies[s.body as usize].count += 1;
                }
            }
        }
        for s in &mut self.shapes {
            s.sum = [0; 6];
        }
    }

    /// The contact force on a node at world `x` (force `n.force` so far) from every proxy
    /// shape it is strictly inside, in shape order; each one's reaction is summed into its
    /// shape.
    pub fn node_contact(&mut self, x: Vec3, n: &Node) -> Vec3 {
        self.node_contact_with(x, n, false, &ContactSurface::BOX)
    }

    /// [`Self::node_contact`]; with `fast_side` (the surface contact's casts, §38.4) a
    /// node inside a fast body's shape is pushed out along the body's motion relative to
    /// it, the side it entered by, instead of through the nearest face (a node a fast ball
    /// carried past its centre would otherwise leave by the far side and push the ball
    /// on). Slow bodies are unchanged. `stat`: the vehicle's static shape rule, used
    /// against an infinite-mass (static or kinematic) body: the node pass's own boxes'
    /// and shapes' surface (`ContactSurface::BOX`, exact and dissipative on the BeamNG
    /// profile: jbeam-semantics-spec.md §37.1).
    pub fn node_contact_with(&mut self, x: Vec3, n: &Node, fast_side: bool, stat: &ContactSurface) -> Vec3 {
        let mut total = Vec3::ZERO;
        let h = STEP;
        for k in 0..self.shapes.len() {
            let Some(mut normal) = Self::face(&self.shapes[k], x) else {
                continue;
            };
            if let Some(vb) = self.shapes[k].fast.filter(|_| fast_side) {
                // Only while the body overtakes the node (their relative motion along
                // the body's own): a node outrunning the body leaves by its nearest face.
                let d = vb - n.vel;
                if d.length_squared() > 0.0 && d.dot(vb) > 0.0 {
                    normal = d.normalize();
                }
            }
            let s = &self.shapes[k];
            let b = &self.bodies[s.body as usize];
            let f = n.force + total;
            let fc = if b.is_static() {
                // Bit for bit the node pass's static shape rule (the body still moves:
                // kinematic), on the vehicle's surface.
                let vb = b.vel + b.ang_vel.cross(x - b.pos);
                let v = if vb == Vec3::ZERO { n.vel } else { n.vel - vb };
                contact_force(f, v, n.mass, normal, h, stat, 0.0, n.friction)
            } else {
                let r = x - b.pos;
                let rn = r.cross(normal);
                let split = b.count.max(1) as f32 * b.share.max(1.0);
                let wb = split * (b.inv_mass + rn.dot(b.inv_inertia_mul(rn)));
                let wn = 1.0 / n.mass;
                let m_red = 1.0 / (wn + wb);
                let share = wn / (wn + wb);
                let vb = b.vel + b.ang_vel.cross(r);
                contact_force(f * share, n.vel - vb, m_red, normal, h, &ContactSurface::BOX, 0.0, n.friction)
            };
            total += fc;
            // The reaction on the body: −fc at x.
            let arm = x - s.reference;
            let react = -fc;
            let m = arm.cross(react);
            let s = &mut self.shapes[k];
            for (slot, v) in s.sum.iter_mut().zip([react.x, react.y, react.z, m.x, m.y, m.z]) {
                *slot += fixed(v);
            }
        }
        total
    }

    /// After the node pass: each shape's step sums into its window totals, and each body
    /// integrates its shapes' reaction (symplectic Euler, as the nodes).
    pub fn end_step(&mut self) {
        let h = STEP;
        let nb = self.bodies.len();
        let mut force = vec![Vec3::ZERO; nb];
        let mut torque = vec![Vec3::ZERO; nb];
        for s in &mut self.shapes {
            let f = Vec3::new(unfixed(s.sum[0]), unfixed(s.sum[1]), unfixed(s.sum[2]));
            let m = Vec3::new(unfixed(s.sum[3]), unfixed(s.sum[4]), unfixed(s.sum[5]));
            s.impulse += f * h;
            s.moment += m * h;
            s.peak = s.peak.max(f.length());
            let b = &self.bodies[s.body as usize];
            force[s.body as usize] += f;
            torque[s.body as usize] += m + (s.reference - b.pos).cross(f);
        }
        for (i, b) in self.bodies.iter_mut().enumerate() {
            if !b.is_static() && !self.frozen {
                b.vel += force[i] * (b.inv_mass * h);
                b.ang_vel += b.inv_inertia_mul(torque[i]) * h;
            }
            b.pos += b.vel * h;
            if b.ang_vel != Vec3::ZERO {
                let w = b.ang_vel;
                let dq = Quat::from_xyzw(w.x, w.y, w.z, 0.0) * b.rot;
                b.rot = Quat::from_xyzw(
                    b.rot.x + 0.5 * h * dq.x,
                    b.rot.y + 0.5 * h * dq.y,
                    b.rot.z + 0.5 * h * dq.z,
                    b.rot.w + 0.5 * h * dq.w,
                )
                .normalize();
            }
        }
    }

    /// The surface contact's view of the shapes at the bodies' poses now
    /// ([`crate::explicit_surface`]): each shape in shape order with its cull box. The
    /// poses are the shared module's strict ones (`fq::shape_pose`), as the GPU's.
    pub fn surface_shapes(&mut self) -> Vec<(SurfaceShape, Vec3, Vec3)> {
        let mut out = Vec::with_capacity(self.shapes.len());
        for k in 0..self.shapes.len() {
            let b = &self.bodies[self.shapes[k].body as usize];
            let (bp, bq) = (b.pos, b.rot);
            let shape = surface_shape(bp, bq, &mut self.shapes[k]);
            let (lo, hi) = fq::bounds(&shape);
            out.push((shape, lo, hi));
        }
        out
    }

    /// Adds the surface contact's force `f` on shape `k` at world point `at` to this
    /// step's sums (fixed point, as the node reactions), integrated by this step's
    /// [`Self::end_step`]. The moment is the strict (at − reference) × f.
    pub fn add_surface_reaction(&mut self, k: usize, f: Vec3, at: Vec3) {
        let s = &mut self.shapes[k];
        let m = fq::cross3(at - s.reference, f);
        for (slot, v) in s.sum.iter_mut().zip([f.x, f.y, f.z, m.x, m.y, m.z]) {
            *slot += fixed(v);
        }
    }

    /// Per shape, its body's velocity when the body is fast for the surface contact's
    /// casts (§38.4): it moves more than half the shape's least extent in a step,
    /// counting its spin at the shape's bounding radius, (|v| + |ω| R) h > e / 2, with
    /// e a box's least half extent, a sphere's or capsule's radius, a hull's least
    /// plane offset (its inscribed radius about its origin; at least 1 cm). Else none.
    pub fn cast_velocities(&self) -> Vec<Option<Vec3>> {
        self.cast_velocities_now()
    }

    fn cast_velocities_now(&self) -> Vec<Option<Vec3>> {
        self.shapes
            .iter()
            .map(|s| {
                let b = &self.bodies[s.body as usize];
                let (e, radius) = match &s.geom {
                    ProxyGeom::Box { half } => (half.x.min(half.y).min(half.z), half.length()),
                    ProxyGeom::Sphere { radius } => (*radius, *radius),
                    ProxyGeom::Capsule { radius, half_len } => (*radius, radius + half_len),
                    ProxyGeom::Hull { planes, radius } => {
                        (planes.iter().fold(f32::INFINITY, |m, p| m.min(p.1)).max(0.01), *radius)
                    }
                };
                let len = |v: Vec3| fq::dot3(v, v).sqrt();
                let travel = len(b.vel) * STEP + (len(b.ang_vel) * (radius + 0.01)) * STEP;
                (travel > 0.5 * e).then_some(b.vel)
            })
            .collect()
    }

    /// Per body, the window's impulse and angular impulse about its centre of mass now.
    pub fn body_impulses(&self) -> Vec<(Vec3, Vec3)> {
        let mut out = vec![(Vec3::ZERO, Vec3::ZERO); self.bodies.len()];
        for s in &self.shapes {
            let b = &self.bodies[s.body as usize];
            let o = &mut out[s.body as usize];
            o.0 += s.impulse;
            o.1 += s.angular_impulse_about(b.pos);
        }
        out
    }
}

/// Proxy shape `s` of a body at (`bp`, `bq`) as the surface contact meets it: the strict
/// pose (`fq::shape_pose`), a box's axes, a capsule's ends along its local y, a hull's
/// world vertices, edges and planes (`fq::hull_world`). The GPU's `cplSurfShape` is
/// the same arithmetic.
pub fn surface_shape(bp: Vec3, bq: Quat, s: &mut ProxyShape) -> SurfaceShape {
    let (p, r) = fq::shape_pose(bp, bq, s.local_pos, s.local_rot);
    let a = fq::axes(r);
    match s.geom.clone() {
        ProxyGeom::Box { half } => SurfaceShape::Box { c: p, axes: a, half },
        ProxyGeom::Sphere { radius } => SurfaceShape::Sphere { c: p, radius },
        ProxyGeom::Capsule { radius, half_len } => {
            let e = a[1] * half_len;
            SurfaceShape::Capsule {
                a: p - e,
                b: p + e,
                radius,
            }
        }
        ProxyGeom::Hull { planes, .. } => {
            let (verts, edges) = s.hull_geometry().cloned().unwrap_or_default();
            fq::hull_world(p, &a, &verts, &edges, &planes)
        }
    }
}

/// The coupling's per-frame candidate list (a Verlet list with a checked skin; the GPU's
/// `explicit_beams_rigid.slang` `cplList*`, the same arithmetic). The GPU tests a node
/// against a proxy shape only while the node may be inside the shape's cull sphere
/// (centre `p`, grown radius `r`, [`candidate_list::cull_sphere`]): a node outside it is
/// rejected by the face test, with no force and no count. The list is the work-saving
/// form of that test, lossless:
///
/// * **Build** (a node's first step of a frame, or a step its list fails the check): the
///   node scans every shape, as the exhaustive path does, and lists the shapes within
///   `r + m` of it (`m` its skin, [`candidate_list::margin`]), in shape order. It keeps
///   its position `x0` and its limit `m − E` (`E` below, at the build).
/// * **Check** (every later step of the frame): the list holds while
///   `|x − x0| + E < limit`, with `E` the frame's running maximum, over every shape of the
///   vehicle, of the shape's centre's distance from its centre at the frame's first step.
///   Then an unlisted shape k (`|x0 − p_k(b)| ≥ r_k(b) + m` at the build step b) stays at
///   `|x − p_k| ≥ r_k(b) + m − |x − x0| − |p_k − p_k(b)| > r_k(b)` (the triangle
///   inequality; `|p_k − p_k(b)| ≤ E(s) + E(b)`), so the node is outside its true cull
///   radius (`r` is grown at least 1 cm and 0.1 % past it, well past the rounding of
///   these sums): the face test would reject it. Listed shapes are tested as before, in
///   shape order, so every force, count and sum is the same, bit for bit.
/// * A node whose list would pass [`candidate_list::CAP`] shapes keeps no list
///   (overflow): it scans every shape every step, as before.
///
/// Nothing about the motion is assumed: a node that accelerates, a body thrown at the
/// car, a thin shape or a shape that comes within reach mid-frame only make the check
/// fail sooner (the node rebuilds its own list), never a contact missed. The skin only
/// sets how often a node rebuilds.
pub mod candidate_list {
    use glam::Vec3;

    /// Shapes a node's list holds at most (the GPU's 16 half-words a node).
    pub const CAP: usize = 16;
    /// The skin's floor and ceiling (m).
    pub const MIN_MARGIN: f32 = 0.02;
    pub const MAX_MARGIN: f32 = 4.0;

    /// A shape's cull sphere as the GPU's `cplSpheresBy` writes it: its centre `p` and its
    /// radius grown past the true bounding radius `radius` (`pr.z`) and the face test's
    /// rounding: `radius · 1.001 + 0.01 + 1e-4 · max(|p|∞, radius)`. Returns the radius
    /// squared (3e38 when not finite: every node tests the shape).
    pub fn cull_sphere(p: Vec3, radius: f32) -> f32 {
        let big = p.x.abs().max(p.y.abs()).max(p.z.abs().max(radius.abs()));
        let r = radius * 1.001 + 0.01 + 1.0e-4 * big;
        let finite = p.abs().cmplt(Vec3::splat(1.0e30)).all() && r.abs() < 1.0e18;
        if finite {
            r * r
        } else {
            3.0e38
        }
    }

    /// The skin of a node with speed `speed` (m/s), the shapes' fastest point speed
    /// `shape_speed`, `left` steps to the frame's end: the distance both may close over
    /// the rest of the frame, a quarter more, within [`MIN_MARGIN`, `MAX_MARGIN`]. Only
    /// the rebuild rate depends on it.
    pub fn margin(speed: f32, shape_speed: f32, left: u32) -> f32 {
        let m = 1.25 * crate::explicit_beams::STEP * left as f32 * (speed + shape_speed) + MIN_MARGIN;
        // NaN (a non-finite speed) takes the ceiling.
        if m < MAX_MARGIN {
            m.max(MIN_MARGIN)
        } else {
            MAX_MARGIN
        }
    }

    /// One node's list.
    #[derive(Clone, Debug, PartialEq)]
    pub struct NodeList {
        pub x0: Vec3,
        pub limit: f32,
        /// The listed shapes in shape order; `None` past [`CAP`] (every shape).
        pub shapes: Option<Vec<u16>>,
    }

    /// Build a node's list at `x` against the shapes' spheres (`(centre, radius squared)`
    /// in shape order), skin `m`, the frame's running maximum shape travel `e`.
    pub fn build(x: Vec3, spheres: &[(Vec3, f32)], m: f32, e: f32) -> NodeList {
        let mut shapes = Some(Vec::new());
        for (k, &(p, r2)) in spheres.iter().enumerate() {
            let d = x - p;
            let l = r2.sqrt() + m;
            // NaN lists the shape.
            if !(d.dot(d) >= l * l) {
                match &mut shapes {
                    Some(s) if s.len() < CAP => s.push(k as u16),
                    _ => shapes = None,
                }
            }
        }
        NodeList { x0: x, limit: m - e, shapes }
    }

    /// Whether `list` still holds every shape the node at `x` may be inside, the frame's
    /// running maximum shape travel now `e` (never on a frame's first step: the caller
    /// builds there).
    pub fn holds(list: &NodeList, x: Vec3, e: f32) -> bool {
        (x - list.x0).length() + e < list.limit
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// A small deterministic generator (no dependency).
        struct Lcg(u64);
        impl Lcg {
            fn f(&mut self) -> f32 {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
            }
            fn v(&mut self, s: f32) -> Vec3 {
                Vec3::new(self.f() - 0.5, self.f() - 0.5, self.f() - 0.5) * (2.0 * s)
            }
        }

        /// The lossless claim, against the exhaustive test: at every step a node inside a
        /// shape's true bounding sphere (where the face test can pass) has that shape in
        /// its list (or keeps no list). Adversarial frames: nodes that jump speed mid-frame
        /// (impacts, up to 300 m/s), bodies thrown through the car (up to 200 m/s), thin
        /// shapes (radius 1 cm), shapes that start far out of reach and enter mid-frame,
        /// shapes far from the origin (rounding), skins forced to the floor and the
        /// ceiling. Returns (rebuilds, checks held).
        fn run(seed: u64, skin: Option<f32>) -> (u32, u32) {
            let mut g = Lcg(seed);
            let steps = 34u32;
            let h = crate::explicit_beams::STEP;
            let (mut rebuilds, mut held) = (0, 0);
            for _frame in 0..40 {
                let far = if g.f() < 0.3 { 2000.0 } else { 0.0 };
                let centre = Vec3::new(far, 0.5 * far, -far);
                let ns = 1 + (g.f() * 40.0) as usize;
                let mut shapes: Vec<(Vec3, Vec3, f32)> = (0..ns)
                    .map(|_| {
                        let radius = if g.f() < 0.3 { 0.01 } else { 0.05 + 3.0 * g.f() };
                        let speed = if g.f() < 0.2 { 200.0 } else { 5.0 * g.f() };
                        // Some start well out of reach, moving at the nodes.
                        let at = if g.f() < 0.3 { g.v(15.0) } else { g.v(3.0) };
                        (centre + at, -at.normalize_or_zero() * speed + g.v(1.0), radius)
                    })
                    .collect();
                let nn = 30;
                let mut nodes: Vec<(Vec3, Vec3)> =
                    (0..nn)
                        .map(|_| {
                            let x = centre + g.v(2.0);
                            let s = if g.f() < 0.5 { 30.0 } else { 1.0 };
                            (x, g.v(s))
                        })
                        .collect();
                let mut p0 = Vec::new();
                let mut e = 0.0f32;
                let mut lists: Vec<Option<NodeList>> = vec![None; nn];
                for s in 0..steps {
                    let spheres: Vec<(Vec3, f32)> = shapes.iter().map(|&(p, _, r)| (p, cull_sphere(p, r))).collect();
                    if s == 0 {
                        p0 = shapes.iter().map(|s| s.0).collect();
                    }
                    let mut vs = 0.0f32;
                    for (k, sh) in shapes.iter().enumerate() {
                        e = e.max((sh.0 - p0[k]).length());
                        vs = vs.max(sh.1.length());
                    }
                    for (i, &(x, v)) in nodes.iter().enumerate() {
                        let ok = s > 0 && lists[i].as_ref().is_some_and(|l| holds(l, x, e));
                        if ok {
                            held += 1;
                        } else {
                            rebuilds += 1;
                            let m = skin.unwrap_or_else(|| margin(v.length(), vs, steps - s));
                            lists[i] = Some(build(x, &spheres, m, e));
                        }
                        let l = lists[i].as_ref().unwrap();
                        for (k, &(p, _, radius)) in shapes.iter().enumerate() {
                            let d = x - p;
                            // Where the face test can pass (the true bounding sphere).
                            if d.dot(d) < radius * radius {
                                if let Some(ls) = &l.shapes {
                                    assert!(ls.contains(&(k as u16)), "seed {seed} step {s}: node {i} inside shape {k}, not listed");
                                }
                            }
                        }
                    }
                    // Motion: impacts change the nodes' speed mid-frame, bodies keep theirs.
                    for n in &mut nodes {
                        if g.f() < 0.05 {
                            n.1 = g.v(300.0);
                        }
                        n.0 += n.1 * h;
                    }
                    for sh in &mut shapes {
                        if g.f() < 0.02 {
                            sh.1 = g.v(200.0);
                        }
                        sh.0 += sh.1 * h;
                    }
                }
            }
            (rebuilds, held)
        }

        #[test]
        fn a_listed_node_meets_no_unlisted_shape() {
            let (mut r, mut h) = (0, 0);
            for seed in 0..60 {
                let (a, b) = run(seed, None);
                r += a;
                h += b;
            }
            // The lists are used (most checks hold), and they are rebuilt.
            assert!(h > r && r > 0, "rebuilds {r}, held {h}");
        }

        #[test]
        fn the_skin_at_its_floor_and_ceiling_misses_nothing() {
            for seed in 0..20 {
                run(seed, Some(MIN_MARGIN));
                run(seed, Some(0.0));
                run(seed, Some(MAX_MARGIN));
            }
        }

        #[test]
        fn non_finite_positions_list_every_shape() {
            let spheres = [(Vec3::ZERO, 1.0), (Vec3::splat(f32::NAN), 3.0e38), (Vec3::splat(10.0), 1.0)];
            let l = build(Vec3::splat(f32::NAN), &spheres, 0.1, 0.0);
            assert_eq!(l.shapes, Some(vec![0, 1, 2]));
            assert!(!holds(&l, Vec3::ZERO, 0.0));
            let l = build(Vec3::ZERO, &spheres, 0.1, 0.0);
            assert_eq!(l.shapes, Some(vec![0, 1]));
            assert!(!holds(&l, Vec3::ZERO, f32::NAN));
        }
    }
}
