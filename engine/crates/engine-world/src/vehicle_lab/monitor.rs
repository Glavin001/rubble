//! The vehicle lab's measurements, from what a host can read each World window: the
//! car's nodes and beams, and the rigid bodies the coupling held as the car's proxies
//! (their poses after the window, the impulse each shape took). The same code measures
//! the GPU World (the records read back) and the CPU reference (its `ProxySet`).
//!
//! Everything is in the car's origin frame (node positions as the solver holds them,
//! proxies relative to the same origin), so the numbers keep their precision kilometres
//! from the World's origin.

use engine_softbody::explicit_beams::{ExplicitBeams, GRAVITY};
use engine_webgpu::CouplingLayout;
use glam::{Mat3, Quat, Vec3};

/// A proxy shape's geometry in its own frame.
#[derive(Clone, Debug, PartialEq)]
pub enum ViewGeom {
    Box {
        half: Vec3,
    },
    Sphere {
        radius: f32,
    },
    /// Along local y.
    Capsule {
        radius: f32,
        half_len: f32,
    },
    /// Outward planes (normal, offset).
    Hull {
        planes: Vec<(Vec3, f32)>,
    },
}

/// A proxy body after a window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyView {
    /// Relative to the car's origin.
    pub pos: Vec3,
    pub rot: Quat,
    pub vel: Vec3,
    pub ang_vel: Vec3,
    /// 0: static or kinematic.
    pub inv_mass: f32,
}

/// A proxy shape after a window.
#[derive(Clone, Debug, PartialEq)]
pub struct ShapeView {
    pub collider: u32,
    pub body: usize,
    pub geom: ViewGeom,
    /// Its centre relative to the car's origin, and its rotation.
    pub pos: Vec3,
    pub rot: Quat,
    /// The window's impulse on it (N s) and the largest one-step force (N).
    pub impulse: Vec3,
    pub peak: f32,
}

/// The proxies of one window.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProxyFrame {
    pub bodies: Vec<BodyView>,
    pub shapes: Vec<ShapeView>,
    pub planes: u32,
}

/// The coupling records of one vehicle (`words`: `layout.recs()` records from
/// `layout.vehicle(v)`) as a [`ProxyFrame`]. Counts past the layout's room are an error
/// (a corrupt header), never read.
pub fn decode_coupling(layout: &CouplingLayout, words: &[u32]) -> Result<ProxyFrame, String> {
    let cap = layout.cap;
    if words.len() < 4 * layout.recs() as usize {
        return Err(format!(
            "vehicle lab: {} coupling words, {} expected",
            words.len(),
            4 * layout.recs()
        ));
    }
    let u = |i: usize| -> [u32; 4] {
        [
            words[4 * i],
            words[4 * i + 1],
            words[4 * i + 2],
            words[4 * i + 3],
        ]
    };
    let f = |i: usize| -> [f32; 4] { u(i).map(f32::from_bits) };
    let [nb, ns, planes, _] = u(0);
    if nb > cap.bodies || ns > cap.shapes || planes > cap.planes {
        return Err(format!("vehicle lab: coupling header ({nb} bodies, {ns} shapes, {planes} planes) past its room {cap:?}"));
    }
    let mut out = ProxyFrame {
        planes,
        ..Default::default()
    };
    for j in 0..nb as usize {
        let at = 1 + j * CouplingLayout::BODY_RECS as usize;
        let (p, q, v, w) = (f(at), f(at + 1), f(at + 2), f(at + 3));
        out.bodies.push(BodyView {
            pos: Vec3::new(p[0], p[1], p[2]),
            rot: Quat::from_xyzw(q[0], q[1], q[2], q[3]),
            vel: Vec3::new(v[0], v[1], v[2]),
            ang_vel: Vec3::new(w[0], w[1], w[2]),
            inv_mass: p[3],
        });
    }
    let shape0 = 1 + (cap.bodies * CouplingLayout::BODY_RECS) as usize;
    let plane0 = shape0 + (cap.shapes * CouplingLayout::SHAPE_RECS) as usize;
    for k in 0..ns as usize {
        let at = shape0 + k * CouplingLayout::SHAPE_RECS as usize;
        let head = u(at);
        let (lp, lq, pr, imp) = (f(at + 1), f(at + 2), f(at + 3), f(at + 4));
        let body = head[1] as usize;
        let Some(b) = out.bodies.get(body) else {
            return Err(format!(
                "vehicle lab: proxy shape {k} names body {body} of {nb}"
            ));
        };
        let geom = match head[0] {
            0 => ViewGeom::Box {
                half: Vec3::new(lp[3], pr[0], pr[1]),
            },
            1 => ViewGeom::Sphere { radius: lp[3] },
            2 => ViewGeom::Capsule {
                radius: lp[3],
                half_len: pr[0],
            },
            3 => {
                let (first, n) = ((head[3] & 0xFFFF) as usize, (head[3] >> 16) as usize);
                if first + n > cap.planes as usize {
                    return Err(format!(
                        "vehicle lab: proxy shape {k}'s planes past the planes room"
                    ));
                }
                ViewGeom::Hull {
                    planes: (0..n)
                        .map(|i| {
                            let p = f(plane0 + first + i);
                            (Vec3::new(p[0], p[1], p[2]), p[3])
                        })
                        .collect(),
                }
            }
            other => {
                return Err(format!(
                    "vehicle lab: proxy shape {k} of unknown kind {other}"
                ))
            }
        };
        let lq = Quat::from_xyzw(lq[0], lq[1], lq[2], lq[3]);
        out.shapes.push(ShapeView {
            collider: head[2],
            body,
            geom,
            pos: b.pos + b.rot * Vec3::new(lp[0], lp[1], lp[2]),
            rot: (b.rot * lq).normalize(),
            impulse: Vec3::new(imp[0], imp[1], imp[2]),
            peak: imp[3],
        });
    }
    Ok(out)
}

impl ShapeView {
    /// A radius about `pos` holding the shape.
    pub fn radius(&self) -> f32 {
        match &self.geom {
            ViewGeom::Box { half } => half.length(),
            ViewGeom::Sphere { radius } => *radius,
            ViewGeom::Capsule { radius, half_len } => radius + half_len,
            ViewGeom::Hull { planes } => {
                // The hull lies inside the box of its planes' offsets only loosely; its
                // vertices give the true reach.
                let (v, _) = engine_softbody::explicit_rigid::hull_from_planes(planes);
                v.iter().fold(0.0f32, |m, p| m.max(p.length()))
            }
        }
    }

    /// How deep `x` (the car's origin frame) lies inside the shape (m; none: outside):
    /// its distance to the nearest face.
    pub fn depth(&self, x: Vec3) -> Option<f32> {
        let l = self.rot.conjugate() * (x - self.pos);
        let d = match &self.geom {
            ViewGeom::Box { half } => {
                let e = *half - l.abs();
                e.x.min(e.y).min(e.z)
            }
            ViewGeom::Sphere { radius } => radius - l.length(),
            ViewGeom::Capsule { radius, half_len } => {
                let y = l.y.clamp(-half_len, *half_len);
                radius - (l - Vec3::new(0.0, y, 0.0)).length()
            }
            ViewGeom::Hull { planes } => planes
                .iter()
                .fold(f32::INFINITY, |m, (n, d)| m.min(d - n.dot(l))),
        };
        (d > 0.0 && d.is_finite()).then_some(d)
    }

    /// Points of the shape's surface and its centre (the car's origin frame): what the
    /// "rigid body inside the car" test samples.
    pub fn sample_points(&self) -> Vec<Vec3> {
        let mut local = vec![Vec3::ZERO];
        match &self.geom {
            ViewGeom::Box { half } => {
                for k in 0..8 {
                    local.push(Vec3::new(
                        if k & 1 == 0 { -half.x } else { half.x },
                        if k & 2 == 0 { -half.y } else { half.y },
                        if k & 4 == 0 { -half.z } else { half.z },
                    ));
                }
            }
            ViewGeom::Sphere { radius } => {
                for a in [Vec3::X, Vec3::Y, Vec3::Z] {
                    local.push(a * *radius);
                    local.push(-a * *radius);
                }
            }
            ViewGeom::Capsule { radius, half_len } => {
                local.push(Vec3::Y * (half_len + radius));
                local.push(-Vec3::Y * (half_len + radius));
            }
            ViewGeom::Hull { planes } => {
                local.extend(engine_softbody::explicit_rigid::hull_from_planes(planes).0);
            }
        }
        local.into_iter().map(|p| self.pos + self.rot * p).collect()
    }
}

/// The solid angle triangle (a, b, c) subtends at the origin (signed).
fn solid_angle(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let (la, lb, lc) = (a.length(), b.length(), c.length());
    let det = a.dot(b.cross(c));
    let den = la * lb * lc + a.dot(b) * lc + b.dot(c) * la + c.dot(a) * lb;
    2.0 * det.atan2(den)
}

/// The closest point of triangle (a, b, c) to `p` (Ericson, Real-Time Collision
/// Detection §5.1.5).
fn closest_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denom = 1.0 / (va + vb + vc);
    a + ab * (vb * denom) + ac * (vc * denom)
}

/// The rotation nearest the 3 × 3 map `a` (Müller et al. 2016, "A robust method to
/// extract the rotational part of deformations"), from `q`.
fn nearest_rotation(a: Mat3, mut q: Quat) -> Quat {
    for _ in 0..30 {
        let r = Mat3::from_quat(q);
        let num = r.x_axis.cross(a.x_axis) + r.y_axis.cross(a.y_axis) + r.z_axis.cross(a.z_axis);
        let den =
            (r.x_axis.dot(a.x_axis) + r.y_axis.dot(a.y_axis) + r.z_axis.dot(a.z_axis)).abs() + 1e-9;
        let omega = num / den;
        let w = omega.length();
        if w < 1e-7 {
            break;
        }
        q = (Quat::from_axis_angle(omega / w, w) * q).normalize();
    }
    q
}

/// What the car is made of, taken once from the vehicle as it was added.
#[derive(Clone, Debug)]
pub struct CarModel {
    pub masses: Vec<f32>,
    pub mass: f32,
    /// Rest positions about the rest centre of mass.
    rest: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
    pub beams: Vec<[u32; 2]>,
    pub rest0: Vec<f32>,
    /// Named parts (beam lists): the showcase car's damage units, a JBeam car's none.
    pub parts: Vec<(String, Vec<u32>)>,
    /// The car's length along its heading at rest (m) and the two nodes that give the
    /// heading (rearmost, foremost).
    pub length: f32,
    pub head: [usize; 2],
}

impl CarModel {
    pub fn new(sim: &ExplicitBeams, parts: Vec<(String, Vec<u32>)>) -> Self {
        let masses: Vec<f32> = sim.nodes.iter().map(|n| n.mass).collect();
        let mass: f32 = masses.iter().sum();
        let c = sim.nodes.iter().fold(Vec3::ZERO, |a, n| a + n.pos * n.mass) / mass;
        let (mut lo, mut hi) = (0usize, 0usize);
        for (i, n) in sim.nodes.iter().enumerate() {
            if n.pos.z < sim.nodes[lo].pos.z {
                lo = i;
            }
            if n.pos.z > sim.nodes[hi].pos.z {
                hi = i;
            }
        }
        Self {
            rest: sim.nodes.iter().map(|n| n.pos - c).collect(),
            mass,
            masses,
            triangles: sim.triangles.iter().map(|t| t.nodes).collect(),
            beams: sim.beams.iter().map(|b| [b.a, b.b]).collect(),
            rest0: sim.beams.iter().map(|b| b.rest0).collect(),
            parts,
            length: sim.nodes[hi].pos.z - sim.nodes[lo].pos.z,
            head: [lo, hi],
        }
    }
}

/// One window's reading of the car (the origin frame).
pub struct CarRead<'a> {
    pub pos: &'a [Vec3],
    pub vel: &'a [Vec3],
    /// Per beam (broken, rest length now); none: not read this window.
    pub beams: Option<&'a [(bool, f32)]>,
    /// The car's origin (world).
    pub origin: glam::DVec3,
}

/// A logged event.
#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub t: f64,
    pub text: String,
}

/// Per named part: beams, broken, yielded, and whether it has left the main body.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PartState {
    pub beams: u32,
    pub broken: u32,
    pub yielded: u32,
    pub detached: bool,
}

/// The lab's running measurements of the car.
#[derive(Clone, Debug, Default)]
pub struct CarStats {
    pub t: f64,
    /// Centre of mass (world), its velocity and heading (unit, horizontal).
    pub centre: glam::DVec3,
    pub velocity: Vec3,
    pub heading: Vec3,
    /// The car's node box (the origin frame) and origin.
    pub lo: Vec3,
    pub hi: Vec3,
    pub origin: glam::DVec3,
    pub speed: f32,
    pub kinetic: f64,
    pub potential: f64,
    /// Kinetic + gravitational energy at the first reading, and the least seen since.
    pub mech0: f64,
    /// Centre-of-mass acceleration over the last window (g) and its peak.
    pub g_now: f32,
    pub g_peak: f32,
    pub beams_total: u32,
    pub beams_broken: u32,
    pub beams_yielded: u32,
    pub parts: Vec<PartState>,
    /// Largest node displacement from the rest shape's best rigid fit (m), now and ever,
    /// and nodes more than 10 cm from it.
    pub deform: f32,
    pub deform_peak: f32,
    pub nodes_moved: u32,
    /// The share of the car's mass still joined to its largest piece by unbroken beams.
    pub attached: f32,
    /// Penetration: the deepest node inside any proxy shape (m; now and ever, with the
    /// collider), nodes inside, and proxy points inside the car's collision hull (count,
    /// deepest now and ever).
    pub node_depth: f32,
    pub node_depth_peak: f32,
    pub node_depth_collider: u32,
    pub nodes_inside: u32,
    pub body_points_inside: u32,
    pub body_depth: f32,
    pub body_depth_peak: f32,
    pub shapes_checked: u32,
    /// Coupling use this window and the most seen.
    pub proxies: [u32; 3],
    pub proxies_peak: [u32; 3],
    pub non_finite: bool,
    pub readings: u64,
    /// (t, broken, yielded) twice a second.
    pub damage_series: Vec<(f32, u32, u32)>,
    pub events: Vec<Event>,
    fit: Option<Quat>,
    last_v: Option<(f64, Vec3)>,
    /// Per node: joined to the largest piece at the last beam reading (empty: all).
    main: Vec<bool>,
    main_mass: f32,
    /// Per shape of the last reading: (collider, deepest node depth, centre inside the
    /// car's hull).
    pub per_shape: Vec<(u32, f32, bool)>,
}

impl CarStats {
    fn log(&mut self, text: String) {
        if self.events.len() < 400 {
            self.events.push(Event { t: self.t, text });
        }
    }

    /// Take one window's reading. `dt`: the time since the last one.
    pub fn read(&mut self, model: &CarModel, car: &CarRead, frame: &ProxyFrame, t: f64) {
        let n = model.masses.len().min(car.pos.len());
        self.t = t;
        self.readings += 1;
        self.origin = car.origin;
        if car.pos[..n]
            .iter()
            .chain(&car.vel[..n])
            .any(|p| !p.is_finite())
        {
            if !self.non_finite {
                self.log("non-finite node state (NaN)".into());
            }
            self.non_finite = true;
            return;
        }
        // The pieces the unbroken beams leave: the car is its largest (a wheel or a panel
        // left behind up the mountain is not where the car is).
        fn find(p: &mut [u32], mut x: u32) -> u32 {
            while p[x as usize] != x {
                p[x as usize] = p[p[x as usize] as usize];
                x = p[x as usize];
            }
            x
        }
        let mut parent: Vec<u32> = (0..n as u32).collect();
        let mut main_root = 0u32;
        if let Some(beams) = car.beams {
            let nb = model.beams.len().min(beams.len());
            for j in 0..nb {
                if !beams[j].0 {
                    let [x, y] = model.beams[j];
                    if (x as usize) < n && (y as usize) < n {
                        let (rx, ry) = (find(&mut parent, x), find(&mut parent, y));
                        if rx != ry {
                            parent[rx as usize] = ry;
                        }
                    }
                }
            }
            let mut piece = vec![0.0f32; n];
            for i in 0..n {
                let r = find(&mut parent, i as u32) as usize;
                piece[r] += model.masses[i];
            }
            main_root = (0..n).fold(0, |b, i| if piece[i] > piece[b] { i } else { b }) as u32;
            self.main = (0..n)
                .map(|i| find(&mut parent, i as u32) == main_root)
                .collect();
            self.main_mass = piece[main_root as usize];
            self.attached = self.main_mass / model.mass;
        }
        if self.main.len() != n {
            self.main = vec![true; n];
            self.main_mass = model.mass;
            self.attached = 1.0;
        }
        // Centre of mass and velocity (the main piece), box and energy (every node).
        let (mut c, mut v, mut ke, mut pe) = (glam::DVec3::ZERO, glam::DVec3::ZERO, 0.0f64, 0.0f64);
        let (mut lo, mut hi) = (Vec3::MAX, Vec3::MIN);
        for i in 0..n {
            let m = f64::from(model.masses[i]);
            ke += 0.5 * m * f64::from(car.vel[i].length_squared());
            pe += m * f64::from(-GRAVITY) * (car.origin.y + f64::from(car.pos[i].y));
            if self.main[i] {
                c += car.pos[i].as_dvec3() * m;
                v += car.vel[i].as_dvec3() * m;
                lo = lo.min(car.pos[i]);
                hi = hi.max(car.pos[i]);
            }
        }
        let mass = f64::from(self.main_mass);
        let (c, v) = ((c / mass).as_vec3(), (v / mass).as_vec3());
        self.centre = car.origin + c.as_dvec3();
        self.velocity = v;
        self.speed = v.length();
        (self.lo, self.hi) = (lo, hi);
        self.kinetic = ke;
        self.potential = pe;
        if self.readings == 1 {
            self.mech0 = ke + pe;
        }
        if let Some((t0, v0)) = self.last_v {
            let dt = (t - t0) as f32;
            if dt > 1e-6 {
                // Gravity is not felt: free fall reads 0 g, standing 1 g.
                let a = (v - v0) / dt - Vec3::new(0.0, GRAVITY, 0.0);
                self.g_now = a.length() / -GRAVITY;
                self.g_peak = self.g_peak.max(self.g_now);
            }
        }
        self.last_v = Some((t, v));
        let d = car.pos[model.head[1]] - car.pos[model.head[0]];
        let h = Vec3::new(d.x, 0.0, d.z);
        if h.length() > 1e-3 {
            self.heading = h.normalize();
        }
        // Deformation: against the rest shape's best rigid fit.
        let mut rc = Vec3::ZERO;
        for i in (0..n).filter(|i| self.main[*i]) {
            rc += model.rest[i] * model.masses[i];
        }
        let rc = rc / self.main_mass;
        let mut a = Mat3::ZERO;
        for i in (0..n).filter(|i| self.main[*i]) {
            let (p, r) = ((car.pos[i] - c) * model.masses[i], model.rest[i] - rc);
            a += Mat3::from_cols(p * r.x, p * r.y, p * r.z);
        }
        // From the last fit; a tumbling car can be half a turn from it, where the
        // iteration stalls: the heading's turn is tried too and the better kept.
        let fit = |q: Quat| -> f32 {
            (0..n)
                .filter(|i| self.main[*i])
                .map(|i| {
                    model.masses[i] * (car.pos[i] - c - q * (model.rest[i] - rc)).length_squared()
                })
                .sum()
        };
        let mut q = nearest_rotation(a, self.fit.unwrap_or(Quat::IDENTITY));
        for start in [
            Quat::from_rotation_arc(
                Vec3::Z,
                if self.heading.length() > 0.5 {
                    self.heading
                } else {
                    Vec3::Z
                },
            ),
            Quat::from_rotation_x(std::f32::consts::PI),
            Quat::from_rotation_z(std::f32::consts::PI),
            Quat::from_rotation_y(std::f32::consts::PI),
        ] {
            let alt = nearest_rotation(a, start);
            if fit(alt) < 0.99 * fit(q) {
                q = alt;
            }
        }
        self.fit = Some(q);
        let (mut worst, mut moved) = (0.0f32, 0u32);
        for i in (0..n).filter(|i| self.main[*i]) {
            let e = (car.pos[i] - c - q * (model.rest[i] - rc)).length();
            worst = worst.max(e);
            moved += u32::from(e > 0.1);
        }
        self.deform = worst;
        self.deform_peak = self.deform_peak.max(worst);
        self.nodes_moved = moved;
        // Beams, parts, what is still attached.
        if let Some(beams) = car.beams {
            let nb = model.beams.len().min(beams.len());
            let bent = |j: usize| {
                !beams[j].0 && (beams[j].1 - model.rest0[j]).abs() > 1e-3 * model.rest0[j].abs()
            };
            let (broken, yielded) = (0..nb).fold((0u32, 0u32), |(b, y), j| {
                (b + u32::from(beams[j].0), y + u32::from(bent(j)))
            });
            if self.beams_total == 0 {
                self.parts = vec![PartState::default(); model.parts.len()];
            }
            if broken > self.beams_broken && self.beams_broken == 0 {
                self.log(format!("first beams broken ({broken})"));
            }
            self.beams_total = nb as u32;
            self.beams_broken = broken;
            self.beams_yielded = yielded;
            let main = main_root;
            for (k, (name, list)) in model.parts.iter().enumerate() {
                let mut s = PartState {
                    beams: list.len() as u32,
                    ..Default::default()
                };
                let (mut nodes, mut off) = (0u32, 0u32);
                for &j in list {
                    let j = j as usize;
                    if j >= nb {
                        continue;
                    }
                    s.broken += u32::from(beams[j].0);
                    s.yielded += u32::from(bent(j));
                    for x in model.beams[j] {
                        if (x as usize) < n {
                            nodes += 1;
                            off += u32::from(find(&mut parent, x) != main);
                        }
                    }
                }
                s.detached = nodes > 0 && 2 * off > nodes;
                let was = self.parts[k].clone();
                if s.broken > 0 && was.broken == 0 {
                    self.log(format!("{name}: first beam broken"));
                }
                if s.detached && !was.detached {
                    self.log(format!("{name}: detached from the main body"));
                }
                self.parts[k] = s;
            }
            if self
                .damage_series
                .last()
                .is_none_or(|l| t as f32 - l.0 >= 0.5)
                && self.damage_series.len() < 4000
            {
                self.damage_series.push((t as f32, broken, yielded));
            }
        }
        // Penetration: nodes inside proxy shapes; proxy points inside the car's hull.
        self.proxies = [
            frame.bodies.len() as u32,
            frame.shapes.len() as u32,
            frame.planes,
        ];
        for k in 0..3 {
            self.proxies_peak[k] = self.proxies_peak[k].max(self.proxies[k]);
        }
        let (mut deepest, mut inside, mut which) = (0.0f32, 0u32, u32::MAX);
        let (mut pts_in, mut pts_depth, mut checked) = (0u32, 0.0f32, 0u32);
        self.per_shape.clear();
        for s in &frame.shapes {
            let r = s.radius();
            if (s.pos + Vec3::splat(r)).cmplt(lo).any() || (s.pos - Vec3::splat(r)).cmpgt(hi).any()
            {
                self.per_shape.push((s.collider, 0.0, false));
                continue;
            }
            checked += 1;
            let mut worst = 0.0f32;
            for p in &car.pos[..n] {
                if (*p - s.pos).length_squared() > r * r {
                    continue;
                }
                if let Some(d) = s.depth(*p) {
                    inside += 1;
                    worst = worst.max(d);
                }
            }
            if worst > deepest {
                deepest = worst;
                which = s.collider;
            }
            // The mountain's ground and walls are far larger than the car: a point of
            // theirs is never "inside the car" in a way a winding number can tell.
            let mut centre_in = false;
            if r < 4.0 {
                for (i, p) in s.sample_points().into_iter().enumerate() {
                    if p.cmplt(lo).any() || p.cmpgt(hi).any() {
                        continue;
                    }
                    let mut w = 0.0f32;
                    for t in &model.triangles {
                        let [a, b, c] = t.map(|k| car.pos[k as usize] - p);
                        w += solid_angle(a, b, c);
                    }
                    if w.abs() > 2.0 * std::f32::consts::PI {
                        pts_in += 1;
                        centre_in |= i == 0;
                        let mut near = f32::INFINITY;
                        for t in &model.triangles {
                            let [a, b, c] = t.map(|k| car.pos[k as usize]);
                            near = near.min((closest_on_triangle(p, a, b, c) - p).length());
                        }
                        if near.is_finite() {
                            pts_depth = pts_depth.max(near);
                        }
                    }
                }
            }
            self.per_shape.push((s.collider, worst, centre_in));
        }
        self.node_depth = deepest;
        self.nodes_inside = inside;
        if deepest > self.node_depth_peak {
            self.node_depth_peak = deepest;
            self.node_depth_collider = which;
        }
        self.body_points_inside = pts_in;
        self.body_depth = pts_depth;
        self.body_depth_peak = self.body_depth_peak.max(pts_depth);
        self.shapes_checked = checked;
    }

    pub fn json(&self, model: &CarModel) -> serde_json::Value {
        let r3 = |x: f32| (f64::from(x) * 1000.0).round() / 1000.0;
        serde_json::json!({
            "t": self.t,
            "centre": [self.centre.x, self.centre.y, self.centre.z],
            "origin": [self.origin.x, self.origin.y, self.origin.z],
            "lo": [self.lo.x, self.lo.y, self.lo.z],
            "hi": [self.hi.x, self.hi.y, self.hi.z],
            "velocity": [self.velocity.x, self.velocity.y, self.velocity.z],
            "heading": [self.heading.x, self.heading.y, self.heading.z],
            "speed": r3(self.speed),
            "kmh": r3(self.speed * 3.6),
            "mass": r3(model.mass),
            "kineticJ": self.kinetic.round(),
            "potentialJ": self.potential.round(),
            "mechanicalLostJ": (self.mech0 - self.kinetic - self.potential).round(),
            "g": r3(self.g_now),
            "gPeak": r3(self.g_peak),
            "beams": { "total": self.beams_total, "broken": self.beams_broken, "yielded": self.beams_yielded },
            "parts": model.parts.iter().zip(&self.parts).filter(|(_, s)| s.broken + s.yielded > 0 || s.detached).map(|((name, _), s)| serde_json::json!({
                "name": name, "beams": s.beams, "broken": s.broken, "yielded": s.yielded, "detached": s.detached })).collect::<Vec<_>>(),
            "partsDetached": self.parts.iter().filter(|s| s.detached).count(),
            "deform": r3(self.deform),
            "deformPeak": r3(self.deform_peak),
            "nodesMoved10cm": self.nodes_moved,
            "nodes": model.masses.len(),
            "attachedMass": r3(self.attached),
            "penetration": {
                "nodeDepth": r3(self.node_depth), "nodeDepthPeak": r3(self.node_depth_peak), "nodeDepthCollider": self.node_depth_collider,
                "nodesInside": self.nodes_inside, "bodyPointsInside": self.body_points_inside,
                "bodyDepth": r3(self.body_depth), "bodyDepthPeak": r3(self.body_depth_peak), "shapesChecked": self.shapes_checked,
            },
            "coupling": { "bodies": self.proxies[0], "shapes": self.proxies[1], "planes": self.proxies[2], "peak": self.proxies_peak },
            "nonFinite": self.non_finite,
            "readings": self.readings,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(geom: ViewGeom) -> ShapeView {
        ShapeView {
            collider: 0,
            body: 0,
            geom,
            pos: Vec3::new(1.0, 2.0, 3.0),
            rot: Quat::from_rotation_y(0.5),
            impulse: Vec3::ZERO,
            peak: 0.0,
        }
    }

    #[test]
    fn depth_is_the_distance_to_the_nearest_face() {
        let b = shape(ViewGeom::Box {
            half: Vec3::new(1.0, 0.5, 2.0),
        });
        let inside = b.pos + b.rot * Vec3::new(0.9, 0.0, 0.0);
        assert!((b.depth(inside).unwrap() - 0.1).abs() < 1e-5);
        assert!(b.depth(b.pos + b.rot * Vec3::new(1.1, 0.0, 0.0)).is_none());
        let s = shape(ViewGeom::Sphere { radius: 0.5 });
        assert!((s.depth(s.pos + Vec3::new(0.2, 0.0, 0.0)).unwrap() - 0.3).abs() < 1e-5);
        let planes = vec![
            (Vec3::X, 1.0),
            (-Vec3::X, 1.0),
            (Vec3::Y, 1.0),
            (-Vec3::Y, 1.0),
            (Vec3::Z, 1.0),
            (-Vec3::Z, 1.0),
        ];
        let h = shape(ViewGeom::Hull { planes });
        assert!((h.depth(h.pos + h.rot * Vec3::new(0.0, 0.75, 0.0)).unwrap() - 0.25).abs() < 1e-5);
    }

    #[test]
    fn the_winding_number_tells_inside_from_outside_a_closed_box() {
        let v = [
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
        ];
        let tris = [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [0, 4, 7],
            [0, 7, 3],
        ];
        let w = |p: Vec3| {
            tris.iter()
                .map(|t| solid_angle(v[t[0]] - p, v[t[1]] - p, v[t[2]] - p))
                .sum::<f32>()
        };
        assert!((w(Vec3::new(0.2, 0.1, -0.3)).abs() - 4.0 * std::f32::consts::PI).abs() < 1e-3);
        assert!(w(Vec3::new(3.0, 0.1, -0.3)).abs() < 1e-3);
    }

    #[test]
    fn the_nearest_rotation_recovers_a_rotation() {
        let q = Quat::from_euler(glam::EulerRot::XYZ, 0.7, -1.1, 2.0);
        let pts = [
            Vec3::X,
            Vec3::Y * 2.0,
            Vec3::Z * 0.5,
            Vec3::new(1.0, 1.0, 1.0),
        ];
        let mut a = Mat3::ZERO;
        for r in pts {
            let p = q * r;
            a += Mat3::from_cols(p * r.x, p * r.y, p * r.z);
        }
        let got = nearest_rotation(a, Quat::IDENTITY);
        assert!(got.dot(q).abs() > 0.9999, "{got:?} vs {q:?}");
    }
}
