//! AVBD vehicles, CPU reference (`engine/docs/avbd-vehicles.md`): an explicit
//! node–beam vehicle ([`ExplicitBeams`], JBeam or `.truck` content) stepped by
//! an implicit block-descent solve at a World rate (60–240 Hz) instead of the
//! 2 kHz explicit step. Opt-in per vehicle ([`ExplicitBeams::avbd`]); with it
//! `None` (the default) nothing here runs and the explicit path is unchanged.
//!
//! One step of length `H` (backward Euler, the AVBD/VBD energy):
//!
//! 1. forces that stay explicit (gravity, air drag, triangle aerodynamics) are
//!    taken at the step's start;
//! 2. the inertial target `y = x₀ + H v₀ + H² f_ext / m`;
//! 3. `iterations` Gauss–Seidel sweeps over the nodes in graph-colour order:
//!    each node takes one Newton step on its 3×3 block (inertia, its beams,
//!    torsion elements and couplers, its contacts). A beam's force is the
//!    explicit solver's own force law (`jbeam_force_filtered`,
//!    `jbeam_ext_force_with`, `beam_force`, `coupler_force`, `torsion_force`)
//!    evaluated on a copy at the iterate, with the velocity `(x − x₀)/H`, so the
//!    spring, the damping, the bounds, support and rope one-sidedness and the
//!    yield cap act inside the solve (a yielding beam is perfectly plastic in the
//!    iteration); its stiffness is the law's slope along the beam (a finite
//!    difference) plus the tension's geometric stiffness;
//! 4. contacts are augmented-Lagrangian inequality constraints: the ground,
//!    static boxes and shapes (planes found at the iterate, a stiff penalty and
//!    a multiplier updated after each sweep, lagged Coulomb friction), and
//!    self-contact as node–triangle constraints found at the step's start (the
//!    node kept on the side it started, frictionless);
//! 5. `v = (x − x₀)/H`, then the explicit beam pass runs once at the converged
//!    state with its forces discarded: plasticity, `deformLimit*`, breaking,
//!    break groups, deform groups and torsion breaks commit by exactly the
//!    explicit rules (the soft solver's `soft_plastic` placement).
//!
//! Forward-only: nothing is undone or re-solved. Content with an element the
//! mapping does not cover fails [`AvbdState::new`] with an error naming it.

use crate::explicit_beams::{
    beam_force, coupler_force, jbeam_force_filtered, torsion_force, Beam, ContactSurface,
    ExplicitBeams, ExplicitKind, Node, StaticShape, DRAG, GRAVITY, NODE_IMMOVABLE, NODE_NO_GROUND,
    STEP,
};
use crate::explicit_jbeam_ext::{aero_tri_force, jbeam_ext_force_with};
use glam::{DMat3, DVec3, Vec3};

/// The AVBD vehicle mode's settings (engine-wide in a World; per vehicle here).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AvbdSettings {
    /// Step rate (Hz).
    pub hz: f32,
    /// Gauss–Seidel sweeps per step.
    pub iterations: u32,
    /// Contact penalty as a multiple of the node's m/H².
    pub contact_scale: f32,
    /// Self-contact (node–triangle) constraints.
    pub self_contact: bool,
    /// Chebyshev semi-iterative acceleration (spectral radius estimate; 0 off).
    pub chebyshev_rho: f32,
    /// The largest correction one node takes in one sweep (m; divergence guard).
    pub max_correction: f32,
}

impl Default for AvbdSettings {
    fn default() -> Self {
        Self {
            hz: 120.0,
            iterations: 20,
            contact_scale: 1.0e3,
            self_contact: true,
            chebyshev_rho: 0.0,
            max_correction: 0.2,
        }
    }
}

/// How a beam record takes part in the implicit solve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    /// A two-node element in the node blocks.
    Implicit,
    /// A zero-length isotropic coupler in the node blocks.
    Coupler,
    /// A four-node torsion element in the node blocks (its pair record follows).
    Torsion,
    /// Its force is taken at the step start (aerodynamics).
    Explicit,
    /// No force of its own (second records, triggers).
    None,
}

#[derive(Clone, Copy, Debug)]
struct PlaneContact {
    node: u32,
    /// 0 ground, 1 + box, 1 + boxes + shape.
    source: u32,
    n: Vec3,
    offset: f32,
    k: f64,
    lambda: f64,
    mu_s: f32,
    mu_c: f32,
    /// The explicit solver's friction law for this surface (`contact_friction`'s curve).
    sf: ContactSurface,
    fr: f32,
}

#[derive(Clone, Copy, Debug)]
struct SelfContact {
    p: u32,
    tri: [u32; 3],
    w: [f32; 3],
    n: Vec3,
    target: f32,
    k: f64,
    lambda: f64,
}

/// Running statistics (measurement).
#[derive(Clone, Copy, Debug, Default)]
pub struct AvbdStats {
    pub steps: u64,
    /// Last sweep's largest node correction (m), summed over steps.
    pub last_dx_sum: f64,
    /// Largest last-sweep correction seen (m).
    pub last_dx_max: f32,
    pub self_contacts: u64,
    pub plane_contacts: u64,
    /// Node corrections cut to [`AvbdSettings::max_correction`].
    pub clamped: u64,
}

/// Per-vehicle AVBD state ([`ExplicitBeams::avbd`]).
#[derive(Clone, Debug)]
pub struct AvbdState {
    pub settings: AvbdSettings,
    /// Explicit time not yet stepped (s).
    acc: f32,
    roles: Vec<Role>,
    /// Nodes in colour order (movable only).
    order: Vec<u32>,
    /// Per node its elements (beam record indices of implicit, coupler and torsion roles).
    elems: Vec<Vec<u32>>,
    pub stats: AvbdStats,
}

impl AvbdState {
    /// The mode for `s`, or an error naming what the mapping does not cover.
    pub fn new(s: &ExplicitBeams, settings: AvbdSettings) -> Result<Self, String> {
        let mut bad = Vec::new();
        if s.drive.is_some() {
            bad.push("raycast wheels/drive".to_string());
        }
        if !s.wings.is_empty() || s.fusedrag.is_some() {
            bad.push("wings/fusedrag".into());
        }
        if s.rigid.is_some() {
            bad.push("rigid proxies".into());
        }
        if !s.tread.is_empty() {
            bad.push("tread contact".into());
        }
        let mut roles = Vec::with_capacity(s.beams.len());
        for (j, b) in s.beams.iter().enumerate() {
            use ExplicitKind as K;
            let r = match b.kind {
                K::Plain
                | K::LimitedSpring
                | K::Support
                | K::Rope
                | K::JbeamPlain
                | K::JbeamSupport
                | K::JbeamBounded
                | K::JbeamAniso
                | K::JbeamRope => Role::Implicit,
                K::JbeamCoupler => Role::Coupler,
                K::JbeamTorsion => Role::Torsion,
                K::JbeamAeroTri => Role::Explicit,
                K::JbeamAeroTri2 | K::JbeamTorsionPair | K::Trigger => Role::None,
                k => {
                    bad.push(format!("beam {j} kind {k:?}"));
                    Role::None
                }
            };
            roles.push(r);
        }
        if !bad.is_empty() {
            bad.truncate(8);
            return Err(format!(
                "AVBD vehicle mode does not cover: {}",
                bad.join(", ")
            ));
        }
        let n = s.nodes.len();
        let mut elems = vec![Vec::new(); n];
        let mut edges = Vec::new();
        for (j, b) in s.beams.iter().enumerate() {
            let nodes: Vec<u32> = match roles[j] {
                Role::Implicit | Role::Coupler => vec![b.a, b.b],
                Role::Torsion => vec![b.a, b.b, s.beams[j + 1].a, s.beams[j + 1].b],
                _ => continue,
            };
            for (k, &x) in nodes.iter().enumerate() {
                if !elems[x as usize].contains(&(j as u32)) {
                    elems[x as usize].push(j as u32);
                }
                for &y in &nodes[k + 1..] {
                    if x != y {
                        edges.push((x, y));
                    }
                }
            }
        }
        let (colors, count) = crate::color::color_graph(n, &edges);
        let mut order = Vec::with_capacity(n);
        for c in 0..count {
            for i in 0..n {
                if colors[i] == c && !s.nodes[i].has(NODE_IMMOVABLE) {
                    order.push(i as u32);
                }
            }
        }
        Ok(Self {
            settings,
            acc: 0.0,
            roles,
            order,
            elems,
            stats: AvbdStats::default(),
        })
    }

    pub fn h(&self) -> f32 {
        1.0 / self.settings.hz
    }
}

/// One explicit-step tick of a vehicle in the AVBD mode: the AVBD step runs
/// when a whole `H` has accumulated; in between the state holds (the harness
/// samples at the explicit 0.5 ms).
pub(crate) fn tick(s: &mut ExplicitBeams) {
    let mut st = s.avbd.take().expect("AVBD mode");
    st.acc += STEP;
    let h = st.h();
    if st.acc + 1e-7 >= h {
        st.acc -= h;
        step(s, &mut st, h);
    }
    s.steps += 1;
    s.avbd = Some(st);
}

/// A node copy for a force law at the iterate.
#[inline]
fn node_at(s: &ExplicitBeams, x: &[Vec3], x0: &[Vec3], h: f32, i: u32) -> Node {
    let mut n = s.nodes[i as usize];
    n.pos = x[i as usize];
    n.vel = if n.has(NODE_IMMOVABLE) {
        Vec3::ZERO
    } else {
        (x[i as usize] - x0[i as usize]) / h
    };
    n
}

/// The force on end `a` of two-node record `j` (beam or coupler) at the
/// iterate, its law applied to a copy (nothing committed).
fn pair_force(s: &ExplicitBeams, j: usize, na: &Node, nb: &Node, coupler: bool) -> Vec3 {
    let mut beam: Beam = s.beams[j];
    if coupler {
        return coupler_force(&mut beam, na, nb);
    }
    let out = if !s.ext_of.is_empty() && s.ext_of[j] != 0 {
        let mut ext = s.beam_ext[s.ext_of[j] as usize - 1];
        let mut filt = match beam.shock {
            0 => None,
            i => Some(s.damp_filters[i as usize - 1]),
        };
        jbeam_ext_force_with(
            &mut beam,
            &mut ext,
            na,
            nb,
            filt.as_mut(),
            !s.plastic_strength_drop,
        )
    } else if beam.kind.is_jbeam() {
        let mut filt = match beam.shock {
            0 => None,
            i => Some(s.damp_filters[i as usize - 1]),
        };
        jbeam_force_filtered(
            &mut beam,
            na,
            nb,
            false,
            filt.as_mut(),
            !s.plastic_strength_drop,
        )
    } else {
        beam_force(&mut beam, na, nb, false)
    };
    if out.check_break || beam.broken {
        return Vec3::ZERO;
    }
    out.d * (out.force * out.inv)
}

fn outer(a: DVec3, b: DVec3) -> DMat3 {
    DMat3::from_cols(a * b.x, a * b.y, a * b.z)
}

/// Node `i`'s force and stiffness block from element `j`.
fn element_block(
    s: &ExplicitBeams,
    roles: &[Role],
    j: usize,
    i: u32,
    x: &mut [Vec3],
    x0: &[Vec3],
    h: f32,
) -> (DVec3, DMat3) {
    let b = s.beams[j];
    match roles[j] {
        Role::Implicit | Role::Coupler => {
            let coupler = roles[j] == Role::Coupler;
            let force_on_i = |x: &[Vec3]| {
                let na = node_at(s, x, x0, h, b.a);
                let nb = node_at(s, x, x0, h, b.b);
                let fa = pair_force(s, j, &na, &nb, coupler);
                if i == b.a {
                    fa
                } else {
                    -fa
                }
            };
            let f = force_on_i(x);
            if coupler {
                if f == Vec3::ZERO && b.k > 0.0 {
                    return (DVec3::ZERO, DMat3::ZERO);
                }
                let k = f64::from(b.k) + f64::from(b.c) / f64::from(h);
                return (f.as_dvec3(), DMat3::from_diagonal(DVec3::splat(k)));
            }
            let o = if i == b.a { b.b } else { b.a };
            let d = x[i as usize] - x[o as usize];
            let len = d.length();
            if len < 1e-6 {
                return (f.as_dvec3(), DMat3::ZERO);
            }
            let e = d / len;
            let delta = 1.0e-4f32;
            let xi = x[i as usize];
            x[i as usize] = xi + e * delta;
            let f2 = force_on_i(x);
            x[i as usize] = xi;
            let k_ax = (-(f64::from(f2.dot(e)) - f64::from(f.dot(e))) / f64::from(delta)).max(0.0);
            // Tension T pulls node i toward o: f·e = −T.
            let t = (-f64::from(f.dot(e))).max(0.0);
            let ed = e.as_dvec3();
            let p = outer(ed, ed);
            let k = p * k_ax + (DMat3::IDENTITY - p) * (t / f64::from(len));
            (f.as_dvec3(), k)
        }
        Role::Torsion => {
            let ids = [b.a, b.b, s.beams[j + 1].a, s.beams[j + 1].b];
            let slot = ids.iter().position(|&q| q == i).unwrap();
            let force = |x: &[Vec3]| {
                let mut beam = b;
                let nodes = ids.map(|q| node_at(s, x, x0, h, q));
                let f = torsion_force(&mut beam, &nodes);
                if beam.broken {
                    Vec3::ZERO
                } else {
                    f[slot]
                }
            };
            let f = force(x);
            let delta = 1.0e-4f32;
            let xi = x[i as usize];
            let mut diag = DVec3::ZERO;
            for ax in 0..3 {
                let mut dx = Vec3::ZERO;
                dx[ax] = delta;
                x[i as usize] = xi + dx;
                let f2 = force(x);
                diag[ax] = (-(f64::from(f2[ax]) - f64::from(f[ax])) / f64::from(delta)).max(0.0);
            }
            x[i as usize] = xi;
            (f.as_dvec3(), DMat3::from_diagonal(diag))
        }
        _ => (DVec3::ZERO, DMat3::ZERO),
    }
}

/// Node `i`'s force and (diagonal) stiffness from slide node `k`'s rail spring.
fn slide_block(s: &ExplicitBeams, k: usize, i: u32, x: &mut [Vec3]) -> (DVec3, DMat3) {
    let sl = s.slides[k];
    let rb = s.beams[s.segments[sl.segment.unwrap() as usize].beam as usize];
    let force = |x: &[Vec3]| {
        let xn = x[sl.node as usize];
        let (q, _, ratio) =
            crate::explicit_triggers::segment_point(x[rb.a as usize], x[rb.b as usize], xn);
        let mut c = sl;
        let f = c.force(q, xn, ratio);
        if i == sl.node {
            f.node
        } else if i == rb.a {
            f.a
        } else {
            f.b
        }
    };
    let f = force(x);
    let delta = 1.0e-4f32;
    let xi = x[i as usize];
    let mut diag = DVec3::ZERO;
    for ax in 0..3 {
        let mut dx = Vec3::ZERO;
        dx[ax] = delta;
        x[i as usize] = xi + dx;
        let f2 = force(x);
        diag[ax] = (-(f64::from(f2[ax]) - f64::from(f[ax])) / f64::from(delta)).max(0.0);
    }
    x[i as usize] = xi;
    (f.as_dvec3(), DMat3::from_diagonal(diag))
}

/// Depth of `x` inside shape `sh` along its face normal `n` (bisection to the surface).
fn shape_depth(sh: &StaticShape, bd: &crate::explicit_beams::StaticBox, x: Vec3, n: Vec3) -> f32 {
    let (mut lo, mut hi) = (0.0f32, 0.05f32);
    while crate::explicit_beams::shape_face(sh, bd, x + n * hi).is_some() && hi < 8.0 {
        lo = hi;
        hi *= 2.0;
    }
    for _ in 0..24 {
        let mid = 0.5 * (lo + hi);
        if crate::explicit_beams::shape_face(sh, bd, x + n * mid).is_some() {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi
}

fn box_depth(b: &crate::explicit_beams::StaticBox, x: Vec3, n: Vec3) -> f32 {
    // Distance to the face with outward normal n.
    if n.x > 0.5 {
        b.hi.x - x.x
    } else if n.x < -0.5 {
        x.x - b.lo.x
    } else if n.y > 0.5 {
        b.hi.y - x.y
    } else if n.y < -0.5 {
        x.y - b.lo.y
    } else if n.z > 0.5 {
        b.hi.z - x.z
    } else {
        x.z - b.lo.z
    }
}

fn step(s: &mut ExplicitBeams, st: &mut AvbdState, h: f32) {
    let n = s.nodes.len();
    let set = st.settings;
    let x0: Vec<Vec3> = s.nodes.iter().map(|q| q.pos).collect();
    let v0: Vec<Vec3> = s.nodes.iter().map(|q| q.vel).collect();
    // 1. Explicit forces at the step start.
    let mut fext: Vec<Vec3> = s
        .nodes
        .iter()
        .map(|q| {
            let mut f = Vec3::new(0.0, q.mass * GRAVITY, 0.0);
            if s.settings.drag {
                f += -DRAG * q.vel.length() * q.vel;
            }
            f
        })
        .collect();
    for (j, b) in s.beams.iter().enumerate() {
        if b.broken || st.roles[j] != Role::Explicit {
            continue;
        }
        let ext = s.beam_ext[s.ext_of[j] as usize - 1];
        let n2 = ext.index(0) as usize;
        let (a, bb) = (b.a as usize, b.b as usize);
        let f = aero_tri_force(
            [s.nodes[a].pos, s.nodes[bb].pos, s.nodes[n2].pos],
            [s.nodes[a].vel, s.nodes[bb].vel, s.nodes[n2].vel],
            &ext,
        );
        fext[a] += f;
        fext[bb] += f;
        fext[n2] += f;
    }
    // Slide nodes (spec §3): the rail segments are chosen at the step start (the
    // explicit pass, its forces discarded); each spring is then implicit.
    let mut node_slides: Vec<Vec<u32>> = vec![Vec::new(); n];
    if !s.slides.is_empty() {
        let saved: Vec<Vec3> = s.nodes.iter().map(|q| q.force).collect();
        s.slide_pass();
        s.slide_hold.clear();
        for (q, f) in s.nodes.iter_mut().zip(saved) {
            q.force = f;
        }
        for (k, sl) in s.slides.iter().enumerate() {
            let Some(seg) = sl.segment else { continue };
            let rb = s.beams[s.segments[seg as usize].beam as usize];
            if sl.broken_away || rb.broken {
                continue;
            }
            for v in [sl.node, rb.a, rb.b] {
                node_slides[v as usize].push(k as u32);
            }
        }
    }
    // 2. Inertial target and initial guess.
    let y: Vec<Vec3> = (0..n)
        .map(|i| {
            let q = &s.nodes[i];
            if q.has(NODE_IMMOVABLE) {
                x0[i]
            } else {
                x0[i] + v0[i] * h + fext[i] / q.mass * (h * h)
            }
        })
        .collect();
    let mut x = y.clone();
    // Contacts.
    let origin = s.origin;
    let ground = s.ground - origin.y;
    let boxes_local: Vec<crate::explicit_beams::StaticBox> = s
        .boxes
        .iter()
        .map(|b| crate::explicit_beams::StaticBox {
            lo: b.lo - origin,
            hi: b.hi - origin,
        })
        .collect();
    let shape_bounds: Vec<_> = s.shapes.iter().map(StaticShape::bounds).collect();
    let mut planes: Vec<PlaneContact> = Vec::new();
    let mut node_planes: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut selfc: Vec<SelfContact> = Vec::new();
    let mut node_self: Vec<Vec<u32>> = vec![Vec::new(); n];
    let kc = |m: f32| f64::from(set.contact_scale) * f64::from(m) / f64::from(h * h);
    if set.self_contact && !s.triangles.is_empty() && std::env::var("AVBD_NO_SELF").is_err() {
        let r = s.collision_range;
        let vmax_all = v0.iter().map(|v| v.length()).fold(0.0, f32::max);
        for (t, tri) in s.triangles.iter().enumerate() {
            if s.anti_only_tris.get(t).copied().unwrap_or(false)
                || s.tri_dead.get(t).copied().unwrap_or(false)
            {
                continue;
            }
            let [a, b, c] = tri.nodes;
            let (pa, pb, pc) = (x0[a as usize], x0[b as usize], x0[c as usize]);
            let nrm = (pb - pa).cross(pc - pa);
            let area2 = nrm.length();
            if area2 < 1e-9 {
                continue;
            }
            let nrm = nrm / area2;
            let band = r + 2.0 * h * vmax_all + 0.01;
            for &p in &s.candidates {
                if p == a || p == b || p == c {
                    continue;
                }
                if !s.self_groups.is_empty() {
                    let g = s.self_groups[p as usize];
                    if g != 0
                        && s.self_groups[a as usize] == g
                        && s.self_groups[b as usize] == g
                        && s.self_groups[c as usize] == g
                    {
                        continue;
                    }
                }
                if s.tri_excl.get(t).is_some_and(|e| e.contains(&p)) {
                    continue;
                }
                let xp = x0[p as usize];
                let d0 = nrm.dot(xp - pa);
                if d0.abs() > band {
                    continue;
                }
                // Barycentric weights of the projection.
                let q = xp - nrm * d0;
                let (e0, e1, e2) = (pb - pa, pc - pa, q - pa);
                let (d00, d01, d11, d20, d21) =
                    (e0.dot(e0), e0.dot(e1), e1.dot(e1), e2.dot(e0), e2.dot(e1));
                let den = d00 * d11 - d01 * d01;
                if den.abs() < 1e-12 {
                    continue;
                }
                let wv = (d11 * d20 - d01 * d21) / den;
                let ww = (d00 * d21 - d01 * d20) / den;
                let wu = 1.0 - wv - ww;
                if wu < 0.0 || wv < 0.0 || ww < 0.0 {
                    continue;
                }
                let side = if d0 >= 0.0 { 1.0 } else { -1.0 };
                let k = kc(s.nodes[p as usize].mass);
                let id = selfc.len() as u32;
                selfc.push(SelfContact {
                    p,
                    tri: [a, b, c],
                    w: [wu, wv, ww],
                    n: nrm * side,
                    target: r.min(d0.abs()),
                    k,
                    lambda: 0.0,
                });
                node_self[p as usize].push(id);
                for v in [a, b, c] {
                    node_self[v as usize].push(id);
                }
            }
        }
    }
    st.stats.self_contacts += selfc.len() as u64;
    let surface_g = ContactSurface::GROUND;
    let surface_b = ContactSurface::BOX;
    let mut detect =
        |x: &[Vec3], planes: &mut Vec<PlaneContact>, node_planes: &mut Vec<Vec<u32>>| {
            for i in 0..n {
                let q = &s.nodes[i];
                if q.has(NODE_NO_GROUND) || q.has(NODE_IMMOVABLE) {
                    continue;
                }
                let xi = x[i];
                let mut add = |src: u32, nrm: Vec3, offset: f32, sf: &ContactSurface| {
                    if node_planes[i]
                        .iter()
                        .any(|&c| planes[c as usize].source == src)
                    {
                        return;
                    }
                    node_planes[i].push(planes.len() as u32);
                    planes.push(PlaneContact {
                        node: i as u32,
                        source: src,
                        n: nrm,
                        offset,
                        k: kc(q.mass),
                        lambda: 0.0,
                        mu_s: sf.ms * sf.strength * q.friction,
                        mu_c: sf.mc * sf.strength * q.friction,
                        sf: *sf,
                        fr: q.friction,
                    });
                };
                if xi.y < ground {
                    add(0, Vec3::Y, ground, &surface_g);
                }
                for (bi, bx) in boxes_local.iter().enumerate() {
                    if let Some(nrm) = crate::explicit_beams::box_face(bx, xi) {
                        let d = box_depth(bx, xi, nrm);
                        add(1 + bi as u32, nrm, nrm.dot(xi) + d, &surface_b);
                    }
                }
                for (si, (sh, bd)) in s.shapes.iter().zip(&shape_bounds).enumerate() {
                    let xw = xi + origin;
                    if let Some(nrm) = crate::explicit_beams::shape_face(sh, bd, xw) {
                        let d = shape_depth(sh, bd, xw, nrm);
                        add(
                            1 + boxes_local.len() as u32 + si as u32,
                            nrm,
                            nrm.dot(xi) + d,
                            &surface_b,
                        );
                    }
                }
            }
        };
    detect(&x0, &mut planes, &mut node_planes);
    let hh = f64::from(h) * f64::from(h);
    let no_torsion = std::env::var("AVBD_NO_TORSION").is_ok();
    let no_friction = std::env::var("AVBD_NO_FRICTION").is_ok();
    let stick = std::env::var("AVBD_STICK").is_ok();
    let rho2 = set.chebyshev_rho * set.chebyshev_rho;
    let (mut prev1, mut prev2) = (x.clone(), x.clone());
    let mut omega = 1.0f32;
    let mut last_dx = 0.0f32;
    let mut clamped = 0u64;
    for it in 0..set.iterations {
        detect(&x, &mut planes, &mut node_planes);
        last_dx = 0.0;
        for &iu in &st.order {
            let i = iu as usize;
            let m = f64::from(s.nodes[i].mass);
            let xi = x[i];
            let mut hm = DMat3::from_diagonal(DVec3::splat(m / hh));
            let mut g = (xi - y[i]).as_dvec3() * (m / hh);
            for &j in &st.elems[i] {
                if no_torsion && st.roles[j as usize] == Role::Torsion {
                    continue;
                }
                if s.beams[j as usize].broken {
                    continue;
                }
                let (f, k) = element_block(s, &st.roles, j as usize, iu, &mut x, &x0, h);
                g -= f;
                hm += k;
            }
            for &k in &node_slides[i] {
                let (f, kd) = slide_block(s, k as usize, iu, &mut x);
                g -= f;
                hm += kd;
            }
            for &c in &node_planes[i] {
                let pc = planes[c as usize];
                let nd = pc.n.as_dvec3();
                let pen = f64::from(pc.offset) - nd.dot(xi.as_dvec3());
                let fnm = pc.k * pen + pc.lambda;
                if fnm <= 0.0 {
                    continue;
                }
                g -= nd * fnm;
                hm += outer(nd, nd) * pc.k;
                // Lagged Coulomb friction against the step's tangential slip.
                let slip = (xi - x0[i]).as_dvec3();
                let slip = slip - nd * nd.dot(slip);
                let sl = slip.length();
                let ks = pc.k;
                let cap_s = f64::from(pc.mu_s) * fnm;
                let tt = DMat3::IDENTITY - outer(nd, nd);
                if no_friction {
                } else if !stick {
                    // The explicit law's magnitude at the step's slip speed σ (RoR's
                    // curve: no true stick, adhesion ms(1 − e^(−σ/va)) under va),
                    // against the slip; its secant slope as the stiffness.
                    if sl > 1e-12 {
                        let sigma = sl / f64::from(h);
                        let sf = pc.sf;
                        let mu = if sigma < f64::from(sf.va) {
                            f64::from(sf.ms) * (1.0 - (-sigma / f64::from(sf.va)).exp())
                        } else {
                            let e = (-(sigma / f64::from(sf.vs)).powf(f64::from(sf.alpha))).exp();
                            f64::from(sf.mc)
                                + f64::from(sf.ms - sf.mc) * e
                                + (f64::from(sf.t2) * sigma).min(5.0)
                        };
                        let cap = mu * f64::from(sf.strength * pc.fr) * fnm;
                        g += slip / sl * cap;
                        hm += tt * (cap / sl);
                    }
                } else if ks * sl <= cap_s {
                    g += slip * ks;
                    hm += tt * ks;
                } else {
                    let cap = f64::from(pc.mu_c) * fnm;
                    g += slip / sl * cap;
                    hm += tt * (cap / sl);
                }
            }
            for &c in &node_self[i] {
                let sc = selfc[c as usize];
                let coef = if sc.p == iu {
                    1.0f64
                } else {
                    let k = sc.tri.iter().position(|&v| v == iu).unwrap();
                    -f64::from(sc.w[k])
                };
                let pt = sc.w[0] * x[sc.tri[0] as usize]
                    + sc.w[1] * x[sc.tri[1] as usize]
                    + sc.w[2] * x[sc.tri[2] as usize];
                let cval = f64::from(sc.n.dot(x[sc.p as usize] - pt)) - f64::from(sc.target);
                let fnm = sc.lambda - sc.k * cval;
                if fnm <= 0.0 {
                    continue;
                }
                let nd = sc.n.as_dvec3();
                g -= nd * (fnm * coef);
                hm += outer(nd, nd) * (sc.k * coef * coef);
            }
            let det = hm.determinant();
            if det.abs() < 1e-30 {
                continue;
            }
            let dx = -(hm.inverse() * g);
            if dx.is_finite() {
                let mut d = dx.as_vec3();
                // Divergence guard (a measurement-visible safeguard, counted): one
                // node's Newton correction is at most `max_correction` per sweep.
                let dl = d.length();
                if dl > set.max_correction {
                    d *= set.max_correction / dl;
                    clamped += 1;
                }
                x[i] = xi + d;
                last_dx = last_dx.max(d.length());
            }
        }
        if rho2 > 0.0 {
            omega = match it {
                0 => 1.0,
                1 => 2.0 / (2.0 - rho2),
                _ => 4.0 / (4.0 - rho2 * omega),
            };
            if it > 0 {
                for &iu in &st.order {
                    let i = iu as usize;
                    if node_planes[i].is_empty() && node_self[i].is_empty() {
                        x[i] = prev2[i] + (x[i] - prev2[i]) * omega;
                    }
                }
            }
            prev2.clone_from(&prev1);
            prev1.clone_from(&x);
        }
        // Dual updates.
        for pc in &mut planes {
            let pen = f64::from(pc.offset) - f64::from(pc.n.dot(x[pc.node as usize]));
            pc.lambda = (pc.k * pen + pc.lambda).max(0.0);
        }
        for sc in &mut selfc {
            let pt = sc.w[0] * x[sc.tri[0] as usize]
                + sc.w[1] * x[sc.tri[1] as usize]
                + sc.w[2] * x[sc.tri[2] as usize];
            let cval = f64::from(sc.n.dot(x[sc.p as usize] - pt)) - f64::from(sc.target);
            sc.lambda = (sc.lambda - sc.k * cval).max(0.0);
        }
    }
    st.stats.plane_contacts += planes.len() as u64;
    st.stats.clamped += clamped;
    st.stats.steps += 1;
    if std::env::var("AVBD_DEBUG").is_ok() && st.stats.steps <= 40 {
        let (mut vm, mut im) = (0.0f32, 0usize);
        for i in 0..n {
            let v = (x[i] - x0[i]).length() / h;
            if v > vm {
                vm = v;
                im = i;
            }
        }
        let mom: Vec3 = (0..n).map(|i| (x[i] - x0[i]) / h * s.nodes[i].mass).sum();
        let mom0: Vec3 = (0..n).map(|i| v0[i] * s.nodes[i].mass).sum();
        let fx: Vec3 = fext.iter().copied().sum();
        // Contact impulse at the converged state (normal + friction, as the block solve).
        let mut fc = DVec3::ZERO;
        let mut ff = DVec3::ZERO;
        for pc in &planes {
            let i = pc.node as usize;
            let nd = pc.n.as_dvec3();
            let pen = f64::from(pc.offset) - nd.dot(x[i].as_dvec3());
            let fnm = pc.k * pen + pc.lambda;
            if fnm <= 0.0 {
                continue;
            }
            fc += nd * fnm;
            let slip = (x[i] - x0[i]).as_dvec3();
            let slip = slip - nd * nd.dot(slip);
            let sl = slip.length();
            if pc.k * sl <= f64::from(pc.mu_s) * fnm {
                ff -= slip * pc.k;
            } else {
                ff -= slip / sl * f64::from(pc.mu_c) * fnm;
            }
        }
        let dm = (mom - mom0 - fx * h).as_dvec3();
        eprintln!("  momentum {mom:?} (start {mom0:?}); dP - ext {dm:?}, contact normal {:?} friction {:?}", fc * f64::from(h), ff * f64::from(h));
        eprintln!(
            "avbd step {}: last_dx {:.2e}, vmax {:.3} at node {} (mass {:.2}, elems {}), planes {}, self {}",
            st.stats.steps,
            last_dx,
            vm,
            im,
            s.nodes[im].mass,
            st.elems[im].len(),
            planes.len(),
            selfc.len()
        );
    }
    st.stats.last_dx_sum += f64::from(last_dx);
    st.stats.last_dx_max = st.stats.last_dx_max.max(last_dx);
    // 5. Velocities, then the explicit rules commit at the converged state.
    for i in 0..n {
        let q = &mut s.nodes[i];
        if q.has(NODE_IMMOVABLE) {
            q.vel = Vec3::ZERO;
            continue;
        }
        q.vel = (x[i] - x0[i]) / h;
        q.pos = x[i];
    }
    s.beam_pass();
    for q in &mut s.nodes {
        q.force = Vec3::new(0.0, q.mass * GRAVITY, 0.0);
    }
}

/// Mechanical energy of a vehicle (J): kinetic, gravity above the ground and the
/// beams' elastic ½k(ℓ − L)² (plain law; a measurement).
pub fn energy(s: &ExplicitBeams) -> (f64, f64, f64) {
    let mut ke = 0.0f64;
    let mut pe = 0.0f64;
    for q in &s.nodes {
        ke += 0.5 * f64::from(q.mass) * f64::from(q.vel.length_squared());
        pe += -f64::from(q.mass) * f64::from(GRAVITY) * f64::from(q.pos.y + s.origin.y - s.ground);
    }
    let mut el = 0.0f64;
    for b in &s.beams {
        if b.broken
            || !matches!(
                b.kind,
                ExplicitKind::JbeamPlain
                    | ExplicitKind::Plain
                    | ExplicitKind::JbeamSupport
                    | ExplicitKind::Support
            )
        {
            continue;
        }
        let l = (s.nodes[b.a as usize].pos - s.nodes[b.b as usize].pos).length();
        let e = l - b.rest;
        if matches!(b.kind, ExplicitKind::JbeamSupport | ExplicitKind::Support) && e > 0.0 {
            continue;
        }
        el += 0.5 * f64::from(b.k) * f64::from(e) * f64::from(e);
    }
    (ke, pe, el)
}
