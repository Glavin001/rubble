//! Surface contact (opt-in, [`ExplicitBeams::surface_contact`]): a vehicle's collision
//! triangles against rigid collider shapes (static boxes and shapes, and the rigid
//! bodies of [`crate::explicit_rigid`]), so a shape that fits between nodes (a 20 cm
//! pole, the edge of a ram) is caught by the skin, not only by its nodes. The spec is
//! `engine/docs/jbeam-semantics-spec.md` §38.
//!
//! **Features.** Node contact (the node pass's static and proxy contact) keeps every
//! node; a triangle takes the contacts whose nearest feature on the triangle is an
//! edge or its face: a shape's edge, corner or curved side against the face, and edge
//! against edge. Per (triangle, shape) the narrow phase ([`triangle_contact`])
//! collects
//! * V: each triangle vertex outside the shape, its exact distance to the shape;
//! * X: the shape's features against the triangle (closest points), each with its
//!   point on the triangle (barycentric weights), its point on the shape and its gap.
//!
//! When a vertex is no farther than the nearest X candidate (within [`TIE`]) the
//! contact is the node's, and the triangle takes none; else the X candidates within
//! [`TIE`] of the nearest are one contact at their mean (two parallel edges of a pole
//! give its face's middle). A contact whose weights fall on one vertex is the node's.
//!
//! **Response.** The two-body rule of vehicle contact (§10.3, §37): the shape is the
//! contacter (infinite mass when static), the triangle's point is the other body with
//! inverse mass Σ w²·k/m (k: the surface hits of the step that node takes part in),
//! the dissipative law (no positive work) with the position step, in the vehicle's
//! band ([`ExplicitBeams::collision_range`]): a gap g ≤ r is a contact of depth
//! min(r − g, r). A dynamic body takes the equal and opposite force at the contact
//! point (its shape's surface sums, integrated by the next step's end).
//!
//! Every operation is single precision in a fixed order (sums left to right, no
//! fusing): the GPU kernel (`explicit_beams.slang`, `surfaceContact`) does the same.

use crate::explicit_beams::{
    contact_response, no_positive_work_body, shift_depth, surface_response, to_fixed,
    ContactSurface, ExplicitBeams, ExplicitKind, Node, StaticShape, CONTACT_FIXED, NODE_IMMOVABLE,
    NODE_NO_GROUND, STEP,
};
// The geometry is the shared strict closest-feature module (its Slang twin is
// `slang/physics/feature_query.slang`).
use engine_schema::feature_query as fq;
pub use engine_schema::feature_query::{
    closest_on_triangle, closest_seg_seg, cross3, dot3, near, overlap_axis, point_contact,
    seg_near, seg_overlap_axis, segment_contact, separated, triangle_contact, SurfaceHit,
    SurfaceShape, PIERCE, TIE,
};
use glam::Vec3;

/// The shape of a static collider (an axis-aligned box is a box with the world axes).
pub fn shape_from_static(s: &StaticShape) -> SurfaceShape {
    match *s {
        StaticShape::OrientedBox {
            centre, half, axes, ..
        } => SurfaceShape::Box {
            c: centre,
            axes,
            half,
        },
        StaticShape::Cylinder {
            centre,
            radius,
            half_height,
        } => SurfaceShape::Cylinder {
            c: centre,
            radius,
            half_height,
        },
        StaticShape::Capsule { a, b, radius } => SurfaceShape::Capsule { a, b, radius },
        StaticShape::Sphere { centre, radius } => SurfaceShape::Sphere { c: centre, radius },
    }
}

/// A beam the surface pass meets shapes with ([`ExplicitBeams::surface_segments`]):
/// the kinds that are a physical member between their two nodes (not triggers,
/// couplers, torsion, LBEAM, pressure or aero records).
pub fn segment_kind(k: ExplicitKind) -> bool {
    matches!(
        k,
        ExplicitKind::Plain
            | ExplicitKind::LimitedSpring
            | ExplicitKind::Support
            | ExplicitKind::Rope
            | ExplicitKind::Shock2
            | ExplicitKind::JbeamPlain
            | ExplicitKind::JbeamSupport
            | ExplicitKind::JbeamBounded
            | ExplicitKind::JbeamAniso
            | ExplicitKind::JbeamPressured
            | ExplicitKind::JbeamRope
    )
}

/// Measurement only (CPU reference): `RUBBLE_SURFACE_FRICTION=ror` takes RoR's contact
/// law ([`contact_response`]) for the surface pass's friction instead of
/// [`surface_response`]'s Coulomb stick (§38.8), to measure what it changed.
pub fn ror_friction() -> bool {
    static ROR: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ROR.get_or_init(|| std::env::var("RUBBLE_SURFACE_FRICTION").as_deref() == Ok("ror"))
}

/// Measurement only: `RUBBLE_SURFACE_CAST=0` turns the fast shapes' casts off (the
/// discrete contact alone, as before §38.4), to report their effect.
pub fn casts_off() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("RUBBLE_SURFACE_CAST").as_deref() == Ok("0"))
}

/// A fast shape's cast (§38.4) against element `x` (a triangle, a segment or a point)
/// over its relative motion `d` this step: within the swept cull box, the earliest
/// fraction t > 0 of the motion at which the element comes within the band `r`
/// ([`fq::toi`]). None when the element is not met (or is met at the start: the
/// discrete contact's).
pub fn cast_toi(x: &[Vec3], s: &SurfaceShape, lo: Vec3, hi: Vec3, d: Vec3, r: f32) -> Option<f32> {
    let (elo, ehi) = x.iter().fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(a, b), &v| (a.min(v), b.max(v)),
    );
    let (slo, shi) = (lo + d.min(Vec3::ZERO), hi + d.max(Vec3::ZERO));
    let (elo, ehi) = (elo - Vec3::splat(r), ehi + Vec3::splat(r));
    if !(elo.cmple(shi).all() && ehi.cmpge(slo).all()) {
        return None;
    }
    let t = fq::toi(x, s, d, r)?;
    (t > 0.0).then_some(t)
}

/// The casts of a step (§38.4): the first pass gathers each fast body's earliest time of
/// impact over every element (`tmin`); the second places a contact only for the
/// elements met then (within [`TIE`] of it along their motion), each at its own time of
/// impact against the shape moved that far, on the side it came from: an element the
/// body would reach only after passing through another is not met this step.
struct Casts {
    /// Per slot its body's velocity (fast) and body index.
    fast: Vec<Option<(Vec3, usize)>>,
    tmin: Vec<f32>,
    gather: bool,
    /// The last placed cast's shape motion ([`Found::shift`]).
    shift: Vec3,
}

impl Casts {
    fn hit<T>(
        &mut self,
        slot: usize,
        x: &[Vec3],
        ve: Vec3,
        s: &SurfaceShape,
        lo: Vec3,
        hi: Vec3,
        r: f32,
        contact: impl Fn(&SurfaceShape) -> Option<T>,
    ) -> Option<T> {
        let (vb, body) = self.fast[slot]?;
        let d = (vb - ve) * STEP;
        let t = cast_toi(x, s, lo, hi, d, r)?;
        if self.gather {
            self.tmin[body] = self.tmin[body].min(t);
            return None;
        }
        let len = fq::dot3(d, d).sqrt();
        if !(t <= self.tmin[body] + TIE / len) {
            return None;
        }
        let shift = d * t;
        let c = contact(&fq::translated(s, shift))?;
        self.shift = shift;
        Some(c)
    }
}

/// One surface hit found by the pass: its element (a triangle, or past the
/// triangles a segment of [`surface_segments`]), its nodes (a segment's third is its
/// first, weight 0), shape slot, the contact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Found {
    pub tri: u32,
    pub nodes: [u32; 3],
    /// Static slot (boxes, then shapes) or, at or past `static_slots`, proxy shape.
    pub slot: u32,
    pub hit: SurfaceHit,
    /// A cast's (§38.4) shape motion to its time of impact: the body takes the contact at
    /// `hit.q − shift`, its own point at its pose now (zero for a discrete contact).
    pub shift: Vec3,
}

/// The beams surface contact tests as segments (beam-edge contact): every beam of a
/// physical kind ([`segment_kind`]) between two distinct collidable nodes (not
/// `NODE_NO_GROUND`) that is no edge of a collision triangle the surface pass meets
/// (not anti-clip only): a triangle owns its edges, a node its ends. In beam order;
/// fixed at spawn (a triangle's break does not hand its edges to the beams).
pub fn surface_segments(s: &ExplicitBeams) -> Vec<u32> {
    let mut edges = std::collections::HashSet::new();
    for (t, tri) in s.triangles.iter().enumerate() {
        if s.tri_anti_only(t) {
            continue;
        }
        let n = tri.nodes;
        for k in 0..3 {
            let (a, b) = (n[k], n[(k + 1) % 3]);
            edges.insert((a.min(b), a.max(b)));
        }
    }
    let mut out = Vec::new();
    for (j, b) in s.beams.iter().enumerate() {
        if !segment_kind(b.kind) || b.a == b.b {
            continue;
        }
        if s.nodes[b.a as usize].has(NODE_NO_GROUND) || s.nodes[b.b as usize].has(NODE_NO_GROUND) {
            continue;
        }
        if edges.contains(&(b.a.min(b.b), b.a.max(b.b))) {
            continue;
        }
        out.push(j as u32);
    }
    out
}

impl ExplicitBeams {
    /// The world shapes the surface pass meets, in slot order: the static boxes, the
    /// static shapes, then the rigid proxies' shapes at their bodies' poses now; and
    /// each one's cull box.
    pub fn surface_shapes(&mut self) -> Vec<(SurfaceShape, Vec3, Vec3)> {
        let mut out = Vec::new();
        for b in &self.boxes {
            let c = (b.lo + b.hi) * 0.5;
            let half = (b.hi - b.lo) * 0.5;
            out.push((
                SurfaceShape::Box {
                    c,
                    axes: [Vec3::X, Vec3::Y, Vec3::Z],
                    half,
                },
                b.lo,
                b.hi,
            ));
        }
        for s in &self.shapes {
            // An unturned box is an axis-aligned box slot on the GPU (lo, hi only).
            if let Some(b) = s.as_aabb() {
                let c = (b.lo + b.hi) * 0.5;
                let half = (b.hi - b.lo) * 0.5;
                let axes = [Vec3::X, Vec3::Y, Vec3::Z];
                out.push((SurfaceShape::Box { c, axes, half }, b.lo, b.hi));
                continue;
            }
            let bd = s.bounds();
            out.push((shape_from_static(s), bd.lo, bd.hi));
        }
        if let Some(r) = self.rigid.as_mut() {
            out.extend(r.surface_shapes());
        }
        out
    }

    /// The surface pass (opt-in, [`Self::surface_contact`]), after self-collision and
    /// before the slide nodes: every live collision triangle against every shape
    /// ([`Self::surface_shapes`]) near it, the step's hits found first (positions
    /// only), then applied in triangle and shape order with mass splitting.
    pub fn surface_pass(&mut self) {
        let shapes = self.surface_shapes();
        let statics = (self.boxes.len() + self.shapes.len()) as u32;
        let r = self.collision_range;
        // Fast proxy bodies (§38.4): per slot its body's velocity when it moves more than
        // half its least extent in a step (cast along its motion), else none.
        let mut casts = Casts {
            fast: vec![None; statics as usize],
            tmin: Vec::new(),
            gather: false,
            shift: Vec3::ZERO,
        };
        if let Some(rs) = self.rigid.as_ref() {
            casts.fast.extend(
                rs.cast_velocities()
                    .into_iter()
                    .zip(&rs.shapes)
                    .map(|(v, sh)| v.map(|v| (v, sh.body as usize))),
            );
            casts.tmin = vec![f32::INFINITY; rs.bodies.len()];
        }
        if casts_off() {
            casts.fast.iter_mut().for_each(|c| *c = None);
        }
        let any_fast = casts.fast.iter().any(Option::is_some);
        let cull = self.surface_separation;
        let mut queries = 0u32;
        if self.surface_segs.is_none() {
            self.surface_segs = Some(surface_segments(self));
        }
        let segs = self.surface_segs.take().unwrap_or_default();
        let mut found: Vec<Found> = Vec::new();
        // With a fast body: a gathering pass first (its hits dropped), then the pass.
        for gather in [true, false] {
            if gather && !any_fast {
                continue;
            }
            casts.gather = gather;
            found.clear();
            let nodes = &self.nodes;
            let vel = |i: u32| nodes[i as usize].vel;
            // Retired debris is no element (the GPU's surfTriRetired, surfSegRetired).
            let gone = |i: u32| self.is_retired(i as usize);
            for t in 0..self.triangles.len() {
                if (!self.tri_dead.is_empty() && self.tri_dead[t]) || self.tri_anti_only(t) {
                    continue;
                }
                let [i0, i1, i2] = self.triangles[t].nodes;
                if gone(i0) || gone(i1) || gone(i2) {
                    continue;
                }
                let x = [i0, i1, i2].map(|i| self.origin + self.nodes[i as usize].pos);
                for (k, (s, lo, hi)) in shapes.iter().enumerate() {
                    let hit = if near(&x, r, *lo, *hi) && !(cull && separated(&x, s, r)) {
                        queries += u32::from(!gather);
                        triangle_contact(&x, s, r)
                    } else {
                        None
                    };
                    casts.shift = Vec3::ZERO;
                    let hit = hit.or_else(|| {
                        let ve = ((vel(i0) + vel(i1)) + vel(i2)) / 3.0;
                        casts.hit(k, &x, ve, s, *lo, *hi, r, |s| triangle_contact(&x, s, r))
                    });
                    if let Some(hit) = hit {
                        found.push(Found {
                            tri: t as u32,
                            nodes: [i0, i1, i2],
                            slot: k as u32,
                            hit,
                            shift: casts.shift,
                        });
                    }
                }
            }
            // Beam-edge contact: the segments, after the triangles.
            let nt = self.triangles.len() as u32;
            for (k, &j) in segs.iter().enumerate() {
                let b = &self.beams[j as usize];
                if b.broken || gone(b.a) || gone(b.b) {
                    continue;
                }
                let (a, bb) = (b.a, b.b);
                let x = [a, bb].map(|i| self.origin + self.nodes[i as usize].pos);
                for (slot, (s, lo, hi)) in shapes.iter().enumerate() {
                    let pair = if seg_near(&x, r, *lo, *hi) && !(cull && separated(&x, s, r)) {
                        queries += u32::from(!gather);
                        segment_contact(&x, s, r)
                    } else {
                        None
                    };
                    casts.shift = Vec3::ZERO;
                    let pair = pair.or_else(|| {
                        let ve = (vel(a) + vel(bb)) * 0.5;
                        casts.hit(slot, &x, ve, s, *lo, *hi, r, |s| segment_contact(&x, s, r))
                    });
                    if let Some((hit, second)) = pair {
                        for hit in std::iter::once(hit).chain(second) {
                            found.push(Found {
                                tri: nt + k as u32,
                                nodes: [a, bb, a],
                                slot: slot as u32,
                                hit,
                                shift: casts.shift,
                            });
                        }
                    }
                }
            }
            // Point contact in the band: every collidable node outside a shape and within
            // the band of it (a node strictly inside is node contact's), after the segments.
            let ns = nt + segs.len() as u32;
            for i in 0..self.nodes.len() {
                if self.nodes[i].has(NODE_NO_GROUND) || gone(i as u32) {
                    continue;
                }
                let x = self.origin + self.nodes[i].pos;
                for (slot, (s, lo, hi)) in shapes.iter().enumerate() {
                    let hit = if seg_near(&[x, x], r, *lo, *hi) {
                        point_contact(x, s, r)
                    } else {
                        None
                    };
                    casts.shift = Vec3::ZERO;
                    let hit = hit.or_else(|| {
                        casts.hit(slot, &[x], vel(i as u32), s, *lo, *hi, r, |s| {
                            point_contact(x, s, r)
                        })
                    });
                    if let Some(hit) = hit {
                        let n = i as u32;
                        found.push(Found {
                            tri: ns + n,
                            nodes: [n, n, n],
                            slot: slot as u32,
                            hit,
                            shift: casts.shift,
                        });
                    }
                }
            }
        }
        self.surface_segs = Some(segs);
        self.surface_last = found.len() as u32;
        self.surface_queries = queries;
        self.surface_force = glam::DVec3::ZERO;
        if found.is_empty() {
            return;
        }
        self.apply_surface_hits(&found, statics);
    }

    fn apply_surface_hits(&mut self, found: &[Found], statics: u32) {
        let exact = self.exact_friction;
        let mut split = vec![0u32; self.nodes.len()];
        for f in found {
            let tri = f.nodes;
            for (j, &i) in tri.iter().enumerate() {
                if f.hit.w[j] > 0.0 {
                    split[i as usize] += 1;
                }
            }
        }
        // Proxy bodies: their surface hits this step (mass splitting).
        let mut body_hits: Vec<u32> = Vec::new();
        if let Some(rs) = self.rigid.as_ref() {
            body_hits = vec![0; rs.bodies.len()];
            for f in found.iter().filter(|f| f.slot >= statics) {
                body_hits[rs.shapes[(f.slot - statics) as usize].body as usize] += 1;
            }
        }
        let pass: Vec<Vec3> = self.forces_now();
        let im = |n: &Node, k: u32| {
            if n.has(NODE_IMMOVABLE) {
                0.0
            } else {
                k as f32 / n.mass
            }
        };
        let acc = |f: Vec3, n: &Node| {
            if n.has(NODE_IMMOVABLE) {
                Vec3::ZERO
            } else {
                f / n.mass
            }
        };
        let surface = ContactSurface::BOX.exact_if(exact).dissipative_if(true);
        let r = self.collision_range;
        for f in found {
            let tri = f.nodes;
            let nd = tri.map(|i| self.nodes[i as usize]);
            let w = f.hit.w;
            let nn = f.hit.n;
            let ims = [0, 1, 2].map(|j| im(&nd[j], split[tri[j] as usize]));
            let term = |j: usize| {
                if w[j] > 0.0 {
                    (w[j] * w[j]) * ims[j]
                } else {
                    0.0
                }
            };
            let imt = (term(0) + term(1)) + term(2);
            let vt = (nd[0].vel * w[0] + nd[1].vel * w[1]) + nd[2].vel * w[2];
            let at = (acc(pass[tri[0] as usize], &nd[0]) * w[0]
                + acc(pass[tri[1] as usize], &nd[1]) * w[1])
                + acc(pass[tri[2] as usize], &nd[2]) * w[2];
            // The shape: static (infinite mass, its velocity zero), or a proxy body (a
            // dynamic one's response for the friction cap: its split, 1/M, arm, rotation,
            // local 1/I).
            let mut body = None;
            let (ims_shape, vs) = match self.rigid.as_ref().filter(|_| f.slot >= statics) {
                Some(rs) => {
                    let sh = &rs.shapes[(f.slot - statics) as usize];
                    let b = &rs.bodies[sh.body as usize];
                    let arm = (f.hit.q - f.shift) - b.pos;
                    let vb = b.vel + cross3(b.ang_vel, arm);
                    if b.is_static() {
                        (0.0, vb)
                    } else {
                        // Strict, as the GPU (explicit_beams_surface.slang surfApplyTo).
                        let rn = cross3(arm, nn);
                        let conj = glam::Quat::from_xyzw(-b.rot.x, -b.rot.y, -b.rot.z, b.rot.w);
                        let irn = fq::rot(b.rot, b.inv_inertia * fq::rot(conj, rn));
                        let k = body_hits[sh.body as usize].max(1) as f32 * b.share.max(1.0);
                        body = Some((k, b.inv_mass, arm, b.rot, conj, b.inv_inertia));
                        (k * (b.inv_mass + dot3(rn, irn)), vb)
                    }
                }
                None => (0.0, Vec3::ZERO),
            };
            let inv = ims_shape + imt;
            let m = if inv > 0.0 { 1.0 / inv } else { 0.0 };
            let fh = (Vec3::ZERO - at) * m;
            let friction = (nd[0].friction * w[0] + nd[1].friction * w[1]) + nd[2].friction * w[2];
            let p = (r - f.hit.gap).min(r);
            let (fc, budget) = if ror_friction() {
                contact_response(fh, vs - vt, m, nn, STEP, &surface, p, friction)
            } else {
                surface_response(fh, vs - vt, m, nn, STEP, &surface, p, friction)
            };
            if fc == Vec3::ZERO {
                continue;
            }
            // A dynamic body turns: the friction capped with its full response (§38.4).
            let fc = match body {
                Some((k, im, arm, rot, conj, ii)) => {
                    no_positive_work_body(fc, fh, vs - vt, m, nn, STEP, |x| {
                        let l = fq::rot(conj, cross3(arm, x));
                        let w = fq::rot(rot, Vec3::new(ii.x * l.x, ii.y * l.y, ii.z * l.z));
                        x * imt + (x * im + cross3(w, arm)) * k
                    })
                }
                None => fc,
            };
            // §37: the position step (the triangle's share; a rigid body takes none).
            let delta = shift_depth(p, budget, dot3(fh, nn));
            if delta > 0.0 {
                let qd = delta * m;
                for j in 0..3 {
                    if w[j] > 0.0 {
                        self.add_shift(tri[j] as usize, nn * -(w[j] * (qd * ims[j])));
                    }
                }
            }
            for j in 0..3 {
                let f = -(fc * w[j]);
                self.add_contact(tri[j] as usize, f);
                // (The statistic: as the node takes it, its fixed-point sum's.)
                self.surface_force += if self.fixed_contact {
                    glam::DVec3::from(
                        f.to_array()
                            .map(|x| f64::from(to_fixed(x)) / f64::from(CONTACT_FIXED)),
                    )
                } else {
                    f.as_dvec3()
                };
            }
            if let Some(l) = self.contact_force.as_mut() {
                for j in 0..3 {
                    l.add(tri[j] as usize, fc * w[j]);
                }
            }
            if f.slot >= statics {
                if let Some(rs) = self.rigid.as_mut() {
                    rs.add_surface_reaction((f.slot - statics) as usize, fc, f.hit.q - f.shift);
                }
            }
        }
    }
}
