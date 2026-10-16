//! Principles any faithful simulation satisfies, whatever its internals. Each test
//! fails for a workaround (a cap, a regularisation, a hand-picked constant) even when
//! every benchmark gate passes, because it checks an invariance or a textbook law
//! rather than a calibrated number:
//!
//! * **Dimensional similarity**: rescaling every unit (lengths x2, times x4, masses
//!   x16, every derived quantity accordingly) rescales every result and changes no
//!   dimensionless one. A constant with physical units hidden in the code (a velocity,
//!   a force, a length), or a formula adding quantities of different units, breaks it.
//!   Powers of two, squares for times and masses, keep the arithmetic and its square
//!   roots exact: the run must be reproduced bit for bit.
//! * **Frame invariance**: a uniform velocity added to everything (Galilean) or a rigid
//!   rotation of the whole scene changes nothing relative.
//! * **Step independence**: results converge as the substep and the frame step shrink;
//!   a bound proportional to `1/dt` does not.
//! * **Resting contact**: a box set down flat stays flat, without rocking, and sinks
//!   only by the elastic compression of the two bodies.
//! * **Coulomb friction**: a block on an incline sticks below the friction angle (no
//!   creep) and slides above it with `g (sin t - mu cos t)`; a sliding block stops after
//!   `v^2 / (2 mu g)`; both independent of the block's mass and of the substep.

mod common;

use stress_ref::builders::*;
use stress_ref::math::{Quat, Vec3};
use stress_ref::scene::*;
use stress_ref::snapshot::{BondStatus, Snapshot};
use stress_ref::world::World;

// ------------------------------------------------------------------ scenes

const G: f64 = 9.81;

/// A free cube resting on the ground under gravity tilted by `slope` (radians) about
/// y: the ground acts as an incline. `mu` is the contact friction, `density` sets the
/// block's mass, `safety` the Courant fraction.
fn incline(slope: f64, mu: f64, density: f64, safety: f64, duration: f64) -> Scene {
    let mut s = new_scene("incline", "free block on an incline (tilted gravity)");
    s.gravity = [G * slope.sin(), 0.0, -G * slope.cos()];
    let mut m = oracle_concrete(None);
    m.density = density;
    m.friction = mu;
    s.materials.insert("block".into(), m.clone());
    s.materials.insert("ground".into(), m);
    let h = 0.1;
    s.bodies.push(body("block", vec![chunk(Vec3::new(0.0, 0.0, h), Vec3::splat(h), "block")], vec![]));
    s.ground = Some(GroundDesc { height: 0.0, friction: mu, material: "ground".into() });
    s.sim.duration = duration;
    s.sim.frame_dt = 1e-3;
    s.sim.courant_safety = safety;
    s.sim.gravity_prestress = false;
    s
}

/// A block sliding on flat ground at `speed` along x.
fn sliding(speed: f64, mu: f64, density: f64, safety: f64, duration: f64) -> Scene {
    let mut s = incline(0.0, mu, density, safety, duration);
    s.name = "sliding".into();
    s.bodies[0].linear_velocity = [speed, 0.0, 0.0];
    s
}

/// A small anchored concrete wall on the ground, struck by a steel box: impact,
/// fracture, crack contact, friction and gravity in one short run.
fn small_impact() -> Scene {
    let mut s = new_scene("small_impact", "anchored wall struck by a steel box");
    s.materials.insert("concrete".into(), oracle_concrete(None));
    s.materials.insert("steel".into(), elastic_steel());
    let mut chunks = grid(Vec3::new(-0.3, 0.0, 0.0), [6, 2, 6], Vec3::splat(0.1), "concrete");
    for c in chunks.iter_mut().take(12) {
        c.support = Support::Fixed;
    }
    let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
    s.bodies.push(body("wall", chunks, bonds));
    s.impactors.push(ImpactorDesc {
        name: "ram".into(),
        shape: ImpactorShape::Box { half_extents: [0.08, 0.08, 0.08] },
        mass: 40.0,
        position: [0.05, -0.081, 0.35],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [0.0, 12.0, 0.0],
        angular_velocity: [0.0; 3],
        material: "steel".into(),
        crush: None,
    });
    s.ground = Some(GroundDesc { height: 0.0, friction: 0.6, material: "concrete".into() });
    s.sim.duration = 0.02;
    s.sim.frame_dt = 1e-3;
    s
}

/// A free wall in zero gravity struck by a sphere (no ground): the scene for the
/// frame-invariance checks, where nothing may single out an origin, a velocity or an
/// axis.
fn free_impact(fracture: bool) -> Scene {
    let mut s = new_scene("free_impact", "free wall struck by a sphere in zero gravity");
    s.gravity = [0.0; 3];
    s.materials.insert("concrete".into(), oracle_concrete(None));
    s.materials.insert("steel".into(), elastic_steel());
    let chunks = grid(Vec3::new(-0.3, 0.0, -0.3), [6, 2, 6], Vec3::splat(0.1), "concrete");
    let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
    s.bodies.push(body("wall", chunks, bonds));
    s.impactors.push(ImpactorDesc {
        name: "ball".into(),
        shape: ImpactorShape::Sphere { radius: 0.08 },
        mass: 20.0,
        position: [0.05, -0.081, 0.05],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [0.0, 15.0, 0.0],
        angular_velocity: [0.0; 3],
        material: "steel".into(),
        crush: None,
    });
    s.sim.duration = 0.015;
    s.sim.frame_dt = 1e-3;
    s.sim.fracture = fracture;
    s.sim.gravity_prestress = false;
    s
}

// ------------------------------------------------------------------ helpers

/// Runs `scene` and returns a snapshot after every frame (the first is t = 0).
fn frames(scene: &Scene) -> Vec<Snapshot> {
    let mut w = World::new(scene);
    let mut out = vec![w.snapshot()];
    while w.solver.time < scene.sim.duration - 0.5 * scene.sim.frame_dt {
        w.step_frame();
        out.push(w.snapshot());
    }
    out
}

fn v(a: [f64; 3]) -> Vec3 {
    Vec3::from_array(a)
}

/// Broken bonds, as (body, a, b), sorted.
fn broken(s: &Snapshot) -> Vec<(usize, usize, usize)> {
    let mut out: Vec<_> = s.bonds.iter().filter(|b| b.status == BondStatus::Broken).map(|b| (b.body, b.a, b.b)).collect();
    out.sort();
    out
}

/// Units of a rescaling: lengths, times and masses multiplied by `l`, `t`, `m`.
#[derive(Clone, Copy)]
struct Units {
    l: f64,
    t: f64,
    m: f64,
}

impl Units {
    fn vel(self) -> f64 {
        self.l / self.t
    }
    fn acc(self) -> f64 {
        self.l / (self.t * self.t)
    }
    fn force(self) -> f64 {
        self.m * self.acc()
    }
    fn stress(self) -> f64 {
        self.force() / (self.l * self.l)
    }
    fn energy(self) -> f64 {
        self.force() * self.l
    }
}

fn scale3(a: [f64; 3], k: f64) -> [f64; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}

fn scale_time_fn(f: &TimeFunction, u: Units, value: f64) -> TimeFunction {
    match f {
        TimeFunction::Constant { value: x } => TimeFunction::Constant { value: x * value },
        TimeFunction::Ramp { t0, t1, value: x } => TimeFunction::Ramp { t0: t0 * u.t, t1: t1 * u.t, value: x * value },
        TimeFunction::HalfSine { start, duration, peak } => TimeFunction::HalfSine { start: start * u.t, duration: duration * u.t, peak: peak * value },
        TimeFunction::Friedlander { arrival, peak, duration, decay } => {
            TimeFunction::Friedlander { arrival: arrival * u.t, peak: peak * value, duration: duration * u.t, decay: *decay }
        }
        TimeFunction::Table { points } => TimeFunction::Table { points: points.iter().map(|p| [p[0] * u.t, p[1] * value]).collect() },
    }
}

fn scale_region(r: &Region, u: Units) -> Region {
    Region { min: scale3(r.min, u.l), max: scale3(r.max, u.l) }
}

fn scale_selector(c: &ChunkSelector, u: Units) -> ChunkSelector {
    match c {
        ChunkSelector::Region(r) => ChunkSelector::Region(scale_region(r, u)),
        other => other.clone(),
    }
}

/// The same physical scene in other units: every dimensional quantity is multiplied by
/// its unit's factor, every dimensionless one kept.
fn rescale(s: &Scene, u: Units) -> Scene {
    let mut s = s.clone();
    s.gravity = scale3(s.gravity, u.acc());
    for m in s.materials.values_mut() {
        m.density *= u.m / u.l.powi(3);
        m.youngs_modulus *= u.stress();
        m.tensile_strength *= u.stress();
        m.compressive_strength *= u.stress();
        m.cohesion *= u.stress();
        m.shear_cap = m.shear_cap.map(|x| x * u.stress());
        // Energy per unit area.
        let ga = u.energy() / (u.l * u.l);
        m.fracture_energy.tension *= ga;
        m.fracture_energy.shear *= ga;
        m.fracture_energy.compression *= ga;
        if let Some(d) = &mut m.dif {
            d.reference_rate /= u.t;
            d.transition_rate /= u.t;
        }
        if let Some(f) = &mut m.static_fatigue {
            f.test_time *= u.t;
        }
    }
    for b in &mut s.bodies {
        b.position = scale3(b.position, u.l);
        b.linear_velocity = scale3(b.linear_velocity, u.vel());
        b.angular_velocity = scale3(b.angular_velocity, 1.0 / u.t);
        for c in &mut b.chunks {
            c.center = scale3(c.center, u.l);
            c.half_extents = scale3(c.half_extents, u.l);
            if let Some(h) = &mut c.hull {
                h.iter_mut().for_each(|p| *p = scale3(*p, u.l));
            }
        }
        for bd in &mut b.bonds {
            bd.centroid = scale3(bd.centroid, u.l);
            bd.area *= u.l * u.l;
            bd.width = [bd.width[0] * u.l, bd.width[1] * u.l];
            bd.buckling_length = bd.buckling_length.map(|x| x * u.l);
            if let Some(r) = &mut bd.rebar {
                r.area *= u.l * u.l;
            }
            bd.derived = None;
        }
    }
    for i in &mut s.impactors {
        i.shape = match &i.shape {
            ImpactorShape::Sphere { radius } => ImpactorShape::Sphere { radius: radius * u.l },
            ImpactorShape::Box { half_extents } => ImpactorShape::Box { half_extents: scale3(*half_extents, u.l) },
        };
        i.mass *= u.m;
        i.position = scale3(i.position, u.l);
        i.velocity = scale3(i.velocity, u.vel());
        i.angular_velocity = scale3(i.angular_velocity, 1.0 / u.t);
        if let Some(c) = &mut i.crush {
            c.max_force *= u.force();
            c.energy *= u.energy();
        }
    }
    if let Some(g) = &mut s.ground {
        g.height *= u.l;
    }
    for load in &mut s.loads {
        *load = match load {
            LoadDesc::PointForce { body, chunk, direction, magnitude, point } => LoadDesc::PointForce {
                body: body.clone(),
                chunk: *chunk,
                direction: *direction,
                magnitude: scale_time_fn(magnitude, u, u.force()),
                point: point.map(|p| scale3(p, u.l)),
            },
            LoadDesc::Pressure { body, chunks, face_normal, pressure } => LoadDesc::Pressure {
                body: body.clone(),
                chunks: scale_selector(chunks, u),
                face_normal: *face_normal,
                pressure: scale_time_fn(pressure, u, u.stress()),
            },
            LoadDesc::Blast { .. } => panic!("blast loads follow an empirical, unit-bearing law (Kingery-Bulmash): not rescalable"),
        };
    }
    for e in &mut s.events {
        *e = match e {
            EventDesc::RemoveChunks { time, duration, body, chunks } => {
                EventDesc::RemoveChunks { time: *time * u.t, duration: *duration * u.t, body: body.clone(), chunks: scale_selector(chunks, u) }
            }
            EventDesc::RemoveSupports { time, duration, body, chunks } => {
                EventDesc::RemoveSupports { time: *time * u.t, duration: *duration * u.t, body: body.clone(), chunks: scale_selector(chunks, u) }
            }
        };
    }
    let sim = &mut s.sim;
    sim.duration *= u.t;
    sim.frame_dt *= u.t;
    sim.sample_interval = sim.sample_interval.map(|x| x * u.t);
    sim.max_substep = sim.max_substep.map(|x| x * u.t);
    sim.mass_scaling_dt = sim.mass_scaling_dt.map(|x| x * u.t);
    sim.implicit_dt = sim.implicit_dt.map(|x| x * u.t);
    // Probes and metrics are not compared here.
    s.probes.clear();
    s.metrics.clear();
    s
}

/// Largest deviation, as a fraction of `reference`, of `b` from `a` after mapping `b`
/// back to `a`'s frame (`map`), over every chunk and impactor of every frame; and
/// whether the broken-bond sets agree at every frame.
fn compare_runs(a: &[Snapshot], b: &[Snapshot], map_pos: impl Fn(f64, Vec3) -> Vec3, map_vel: impl Fn(Vec3) -> Vec3, reference: (f64, f64)) -> (f64, f64, Option<f64>) {
    assert_eq!(a.len(), b.len(), "same number of frames");
    let (mut dx, mut dv) = (0.0f64, 0.0f64);
    let mut first_split = None;
    for (fa, fb) in a.iter().zip(b) {
        let key = |s: &Snapshot| {
            let mut m: Vec<_> = s.chunks.iter().map(|c| ((c.body, c.chunk), (v(c.center), v(c.velocity)))).collect();
            m.sort_by_key(|e| e.0);
            m
        };
        let (ka, kb) = (key(fa), key(fb));
        assert_eq!(ka.len(), kb.len(), "same active chunks at t = {}", fa.time);
        for ((ia, (xa, va)), (ib, (xb, vb))) in ka.iter().zip(&kb) {
            assert_eq!(ia, ib);
            dx = dx.max((map_pos(fa.time, *xb) - *xa).norm() / reference.0);
            dv = dv.max((map_vel(*vb) - *va).norm() / reference.1);
        }
        for (ia, ib) in fa.impactors.iter().zip(&fb.impactors) {
            dx = dx.max((map_pos(fa.time, v(ib.center)) - v(ia.center)).norm() / reference.0);
            dv = dv.max((map_vel(v(ib.velocity)) - v(ia.velocity)).norm() / reference.1);
        }
        if first_split.is_none() && broken(fa) != broken(fb) {
            first_split = Some(fa.time);
        }
    }
    (dx, dv, first_split)
}

// ------------------------------------------------------------------ dimensional similarity

/// Rescaling every unit reproduces the run: positions scale by L, velocities by L/T,
/// the same bonds break at the same scaled times. The factors are powers of two (and
/// squares where square roots are taken), so a solver built only from the scene's own
/// quantities with dimensionally consistent formulas reproduces it bit for bit; any
/// constant with physical units in the code shows up as a difference.
#[test]
fn rescaling_every_unit_reproduces_the_run() {
    let u = Units { l: 2.0, t: 4.0, m: 16.0 };
    let cases: Vec<(&str, Scene)> = vec![
        ("impact with fracture, contact and friction", small_impact()),
        ("block sticking on an incline", incline(20f64.to_radians(), 0.5, 2400.0, 0.5, 0.1)),
        ("block sliding to rest", sliding(1.0, 0.5, 2400.0, 0.5, 0.1)),
        ("masonry wall struck by a ball", {
            let mut s = masonry_wall(15.0, "b7");
            s.sim.duration = 0.02;
            s
        }),
        ("frame losing a column", {
            let mut s = frame_column_removal(0.0, "b8");
            s.sim.duration = 0.1;
            s
        }),
    ];
    let mut failures = Vec::new();
    for (label, scene) in &cases {
        let a = frames(scene);
        let b = frames(&rescale(scene, u));
        let size = a[0].chunks.iter().map(|c| v(c.half_extents).norm()).fold(0.0, f64::max).max(1e-3);
        let speed = a.iter().flat_map(|f| f.chunks.iter().map(|c| v(c.velocity).norm())).fold(0.0, f64::max).max(1e-3);
        let (dx, dv, split) = compare_runs(&a, &b, |_, x| x / u.l, |w| w / u.vel(), (size, speed));
        println!("{label}: position {dx:.2e} of a chunk size, velocity {dv:.2e} of the peak speed, broken bonds differ from {split:?}");
        if dx > 0.0 || dv > 0.0 || split.is_some() {
            failures.push(format!("{label}: position {dx:.2e}, velocity {dv:.2e}, broken bonds differ from t = {split:?}"));
        }
    }
    assert!(failures.is_empty(), "results depend on the choice of units (a dimensional constant in the code):\n{}", failures.join("\n"));
}

// ------------------------------------------------------------------ frame invariance

/// The scales at which a change of frame rounds the state: positions at `reach` (the
/// largest coordinate the transformed run attains), linear velocities at `speed`, every
/// other vector at its own length (a rotation mixes its components, so each carries the
/// rounding of the whole vector) and orientations at 1.
#[derive(Clone, Copy)]
struct FrameScale {
    reach: f64,
    speed: f64,
}

/// Runs `scene` with the substep pinned (half the stable one, a whole fraction of the
/// frame) and one substep per world frame, returning a snapshot at every scene frame:
/// bit-identical to the scene's own frames by the partition invariance shown below. With
/// `jiggle = Some((seed, scale))`, after every substep each stored coordinate of the
/// moving state (clusters' poses and velocities, chunks' local deformation and
/// velocities, impactors' poses and velocities) moves by one unit in the last place at
/// its `FrameScale`, up or down by a fixed pseudo-random sign: the rounding a change of
/// frame injects, every substep.
fn pinned_frames(scene: &Scene, jiggle: Option<(u64, FrameScale)>) -> Vec<Snapshot> {
    let n = (2.0 * scene.sim.frame_dt / World::new(scene).substep_dt()).ceil();
    let dt = scene.sim.frame_dt / n;
    let mut s = scene.clone();
    s.sim.max_substep = Some(dt);
    s.sim.frame_dt = dt;
    let mut w = World::new(&s);
    let mut out = vec![w.snapshot()];
    let mut seed = jiggle.map_or(0, |(k, _)| 0x9e37_79b9_7f4a_7c15u64 ^ (k + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9));
    let mut nudge = |x: &mut f64, scale: f64| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ulp(x.abs().max(scale));
        *x += if (seed >> 63) == 1 { u } else { -u };
    };
    let total = (scene.sim.duration / scene.sim.frame_dt).round() as u64 * n as u64;
    while w.solver.substeps < total {
        w.step_frame();
        if let Some((_, f)) = jiggle {
            let mut vec3 = |x: &mut Vec3, scale: f64| {
                let scale = scale.max(x.norm());
                nudge(&mut x.x, scale);
                nudge(&mut x.y, scale);
                nudge(&mut x.z, scale);
            };
            for c in &mut w.solver.clusters {
                if c.anchored {
                    continue;
                }
                vec3(&mut c.pose.position, f.reach);
                vec3(&mut c.velocity, f.speed);
                vec3(&mut c.angular_velocity, 0.0);
            }
            for st in w.solver.chunks.iter_mut().flatten() {
                vec3(&mut st.u, 0.0);
                vec3(&mut st.th, 0.0);
                vec3(&mut st.v, 0.0);
                vec3(&mut st.w, 0.0);
            }
            for i in &mut w.impactors {
                vec3(&mut i.pose.position, f.reach);
                vec3(&mut i.velocity, f.speed);
                vec3(&mut i.angular_velocity, 0.0);
            }
            let mut quat = |q: &mut Quat| {
                for c in [&mut q.w, &mut q.x, &mut q.y, &mut q.z] {
                    nudge(c, 1.0);
                }
            };
            for c in &mut w.solver.clusters {
                if !c.anchored {
                    quat(&mut c.pose.rotation);
                }
            }
            for i in &mut w.impactors {
                quat(&mut i.pose.rotation);
            }
        }
        if w.solver.substeps % n as u64 == 0 {
            out.push(w.snapshot());
        }
    }
    out
}

/// How far rounding alone moves a run: the largest deviation, from the unperturbed
/// `reference`, of `K` runs of `scene` with the rounding of a change of frame injected
/// every substep (`pinned_frames`, `K` fixed seeds), and whether any of them broke
/// different bonds. A run in another frame must stay inside it: if the change of frame
/// is no more than such rounding, the chance it lies beyond the largest of `K`
/// exchangeable draws is `1 / (K + 1)` (5 % for `K = 19`; the seeds are fixed, so the
/// test is deterministic). A dependence on the frame (an absolute velocity or axis in a
/// force law) lands orders of magnitude outside.
fn rounding_envelope(scene: &Scene, reference: &[Snapshot], scale: FrameScale) -> (f64, f64, bool) {
    const K: u64 = 19;
    let (mut ex, mut ev, mut split) = (0.0f64, 0.0f64, false);
    for k in 0..K {
        let (dx, dv, s) = compare_runs(reference, &pinned_frames(scene, Some((k, scale))), |_, x| x, |w| w, (1.0, 1.0));
        ex = ex.max(dx);
        ev = ev.max(dv);
        split |= s.is_some();
    }
    (ex, ev, split)
}

/// Checks a change of frame against `rounding_envelope`: the same bonds break (unless
/// rounding alone changes which), and the motion deviates no more than rounding does.
fn assert_frame_change(label: &str, original: &Scene, a: &[Snapshot], dx: f64, dv: f64, split: Option<f64>, scale: FrameScale) {
    let (lx, lv, rounding_splits) = rounding_envelope(original, a, scale);
    println!("{label}: position {dx:.2e} m (rounding moves it up to {lx:.2e}), velocity {dv:.2e} m/s (up to {lv:.2e}), bonds differ from {split:?}");
    assert!(split.is_none() || rounding_splits, "{label}: the change of frame changed which bonds break (from t = {split:?})");
    assert!(dx <= lx && dv <= lv, "{label}: the change of frame moved the run beyond rounding: position {dx:.2e} > {lx:.2e} or velocity {dv:.2e} > {lv:.2e}");
}

/// `scene` turned rigidly by `q` about the origin.
fn rotated(scene: &Scene, q: Quat) -> Scene {
    let mut turned = scene.clone();
    let rot = |a: [f64; 3]| q.rotate(v(a)).to_array();
    for b in &mut turned.bodies {
        b.position = rot(b.position);
        b.orientation = (q * Quat::from_wxyz(b.orientation)).to_wxyz();
        b.linear_velocity = rot(b.linear_velocity);
        b.angular_velocity = rot(b.angular_velocity);
    }
    for i in &mut turned.impactors {
        i.position = rot(i.position);
        i.orientation = (q * Quat::from_wxyz(i.orientation)).to_wxyz();
        i.velocity = rot(i.velocity);
        i.angular_velocity = rot(i.angular_velocity);
    }
    turned
}

/// A uniform velocity added to every body changes nothing relative (no ground, no
/// gravity: nothing singles out a frame), beyond what rounding alone does: floating
/// point cannot make it exact (`(v_a + d) - (v_b + d)` rounds), so the drifting run is
/// held to `rounding_envelope` at the scale its coordinates reach.
#[test]
fn a_uniform_velocity_changes_nothing_relative() {
    for fracture in [false, true] {
        let scene = free_impact(fracture);
        let drift = Vec3::new(3.0, -2.0, 1.5);
        let mut moved = scene.clone();
        for b in &mut moved.bodies {
            b.linear_velocity = (v(b.linear_velocity) + drift).to_array();
        }
        for i in &mut moved.impactors {
            i.velocity = (v(i.velocity) + drift).to_array();
        }
        let (a, b) = (pinned_frames(&scene, None), pinned_frames(&moved, None));
        let (dx, dv, split) = compare_runs(&a, &b, |t, x| x - drift * t, |w| w - drift, (1.0, 1.0));
        let speed = a.iter().flat_map(|f| f.impactors.iter().map(|i| v(i.velocity).norm())).fold(0.0, f64::max);
        let scale = FrameScale { reach: 1.0 + drift.norm() * scene.sim.duration, speed: speed + drift.norm() };
        assert_frame_change(&format!("uniform velocity, fracture {fracture}"), &scene, &a, dx, dv, split, scale);
    }
}

/// A half turn of the whole scene about a coordinate axis turns the result, bit for
/// bit. The turn only flips the signs of two coordinates, which floating point does
/// exactly (the quaternion is `(0, e_k)` exactly, and rotating by it reduces to
/// `v - 2 v_perp`); every product and sum the solver forms then rounds identically up
/// to sign, so a solver that never singles out an axis reproduces the run exactly, with
/// fracture, contact and friction. Anything that consults a world axis, or the sign of
/// a coordinate, to choose a basis, a branch or an order shows up here.
#[test]
fn a_half_turn_of_the_scene_turns_the_result_bit_for_bit() {
    let mut failures = Vec::new();
    for (name, q) in [("x", Quat { w: 0.0, x: 1.0, y: 0.0, z: 0.0 }), ("y", Quat { w: 0.0, x: 0.0, y: 1.0, z: 0.0 }), ("z", Quat { w: 0.0, x: 0.0, y: 0.0, z: 1.0 })] {
        for fracture in [false, true] {
            let scene = free_impact(fracture);
            let (a, b) = (frames(&scene), frames(&rotated(&scene, q)));
            let (dx, dv, split) = compare_runs(&a, &b, |_, x| q.rotate(x), |w| q.rotate(w), (1.0, 1.0));
            println!("half turn about {name}, fracture {fracture}: position {dx:.2e} m, velocity {dv:.2e} m/s, broken bonds differ from {split:?}");
            if dx != 0.0 || dv != 0.0 || split.is_some() {
                failures.push(format!("about {name}, fracture {fracture}: position {dx:.2e} m, velocity {dv:.2e} m/s, broken bonds differ from {split:?}"));
            }
        }
    }
    assert!(failures.is_empty(), "a half turn of the scene changes the result:\n{}", failures.join("\n"));
}

/// Rotating the whole scene by a generic angle rotates the result (nothing singles out an
/// axis), beyond what rounding alone does (`rounding_envelope`); the half turns above
/// hold it bit for bit.
#[test]
fn a_rigid_rotation_of_the_scene_rotates_the_result() {
    let q = Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0).normalized(), 0.7);
    for fracture in [false, true] {
        let scene = free_impact(fracture);
        let (a, b) = (pinned_frames(&scene, None), pinned_frames(&rotated(&scene, q), None));
        let back = q.conjugate();
        let (dx, dv, split) = compare_runs(&a, &b, |_, x| back.rotate(x), |w| back.rotate(w), (1.0, 1.0));
        let speed = a.iter().flat_map(|f| f.impactors.iter().map(|i| v(i.velocity).norm())).fold(0.0, f64::max);
        assert_frame_change(&format!("rotation, fracture {fracture}"), &scene, &a, dx, dv, split, FrameScale { reach: 1.0, speed });
    }
}

// ------------------------------------------------------------------ resting contact

/// A box set down on a flat support (released just touching) settles to the static
/// equilibrium of the elastic layer: its sink and tilt are the exact solution of the
/// contact law, not "no tilt". With gravity tilted 5 degrees, friction at the base and
/// gravity at the centre make a couple, and the box leans by
/// `phi ~ F_t h / (k I - m g h)` (~8e-8 rad for the middle density). On the ground the
/// target is solved from the exact geometry (`rest_state`); on an anchored chunk from
/// the pair law's exact two-field energy (`slab_energy`, `slab_rest_state`). A face
/// supported at one point, or a contact that does not damp rocking, misses it.
///
/// Settled (`settle`): over a whole rocking period, sampled 8 times a bounce period,
/// the centre and tilt stay within the bounds below; reached within the decay of the
/// slowest mode (rocking about the base, rate `zeta w_v / 5` for a cube,
/// `w_v = sqrt(k A / m)`, from amplitude `delta` down to `ulp(z)`), the run cut off at
/// ten times that. Agreement: positions are stored in doubles (half an ulp
/// each), and the forces the equilibrium balances are computed from them along a path
/// of at most eight roundings at that scale (coordinate difference, clip interpolation,
/// centroid, arm): sink within 4 ulp of the centre's height, tilt within
/// `4 (ulp(z) / 2h + eps)` (the same over the face's width, plus the rotation's own).
#[test]
fn a_box_set_down_flat_rests_at_the_exact_static_equilibrium() {
    let mut failures = Vec::new();
    let h: f64 = 0.1;
    let slope = 5f64.to_radians();
    for on_chunk in [false, true] {
        for density in [24.0, 2400.0, 240_000.0] {
            let mut s = incline(slope, 0.5, density, 0.5, 1.0);
            if on_chunk {
                s.ground = None;
                s.bodies[0].chunks[0].center = [0.0, 0.0, 3.0 * h];
                let mut slab = chunk(Vec3::new(0.0, 0.0, h), Vec3::new(4.0 * h, 4.0 * h, h), "ground");
                slab.support = Support::Fixed;
                s.bodies.push(body("slab", vec![slab], vec![]));
            }
            let top = if on_chunk { 2.0 * h } else { 0.0 };
            let mat = &s.materials["block"];
            let e = mat.youngs_modulus / (1.0 - mat.poisson_ratio * mat.poisson_ratio);
            let k = 1.0 / (h / e + h / e);
            let m = density * (2.0 * h).powi(3);
            let (z_eq, phi_eq) = if on_chunk { slab_rest_state(h, m, s.gravity, e, top) } else { rest_state(h, m, s.gravity, k, top) };
            let zeta = {
                let l = s.sim.contact_restitution.ln();
                -l / (std::f64::consts::PI.powi(2) + l * l).sqrt()
            };
            let bounce = (k * 4.0 * h * h / m).sqrt();
            let rocking = zeta * bounce / 5.0;
            let settle_estimate = ((top + h - z_eq) / ulp(z_eq)).ln() / rocking;
            s.sim.frame_dt = 2.0 * std::f64::consts::PI / bounce / 8.0;
            let b = Block { h, m, k, k_t: 0.0, zeta };
            let mut w = World::new(&s);
            let settled = settle(&mut w, &b, 10.0 * settle_estimate);
            let z = w.solver.chunk_position(0, 0).z;
            let r = w.solver.clusters[w.solver.chunks[0][0].cluster].rotation();
            let phi = r.m[0][2].atan2(r.m[2][2]);
            let (dz, dphi) = (z - z_eq, phi - phi_eq);
            let (bz, bphi) = (4.0 * ulp(z_eq), 4.0 * (ulp(z_eq) / (2.0 * h) + f64::EPSILON));
            let label = format!("{} density {density}", if on_chunk { "on an anchored chunk," } else { "on the ground," });
            println!("{label}: settled {settled} at {:.3} s (decay estimate {settle_estimate:.3} s); sink {:.6e} (exact {:.6e}, off {:.1} ulp); tilt {phi:.6e} (exact {phi_eq:.6e}, off {:.2e} of the bound)", w.solver.time, top + h - z, top + h - z_eq, dz / ulp(z_eq), dphi.abs() / bphi);
            if !settled || dz.abs() > bz || dphi.abs() > bphi {
                failures.push(format!("{label}: settled {settled}; sink off by {dz:.2e} m (bound {bz:.1e}), tilt off by {dphi:.2e} rad (bound {bphi:.1e})"));
            }
        }
    }
    assert!(failures.is_empty(), "a box set down flat does not rest at the exact equilibrium:\n{}", failures.join("\n"));
}

/// Properties of the incline scenes' block and its ground contact.
struct Block {
    h: f64,
    m: f64,
    /// Normal and shear layer moduli per area (`1 / (h / E' + h / E')`, `1 / (2h / G)`).
    k: f64,
    k_t: f64,
    zeta: f64,
}

fn block_of(s: &Scene) -> Block {
    let h = 0.1;
    let mat = &s.materials["block"];
    let e = mat.youngs_modulus / (1.0 - mat.poisson_ratio * mat.poisson_ratio);
    let g = mat.shear_modulus();
    let l = s.sim.contact_restitution.ln();
    Block { h, m: mat.density * (2.0 * h).powi(3), k: 1.0 / (h / e + h / e), k_t: 1.0 / (h / g + h / g), zeta: -l / (std::f64::consts::PI.powi(2) + l * l).sqrt() }
}

impl Block {
    fn area(&self) -> f64 {
        4.0 * self.h * self.h
    }
    /// Bounce and rocking (about the base) frequencies, and the slowest decay rate of
    /// the contact's modes (rocking, `zeta w_v / 5` for a cube; sliding on the shear
    /// layer decays at `zeta w_s`, `w_s = sqrt(k_t A / m)`, faster).
    fn bounce(&self) -> f64 {
        (self.k * self.area() / self.m).sqrt()
    }
    fn slowest_decay(&self) -> f64 {
        (self.zeta * self.bounce() / 5.0).min(self.zeta * (self.k_t * self.area() / self.m).sqrt())
    }
}

/// Places the block of an incline scene at the exact equilibrium of its contact under
/// the contact force `-m g_contact` (sink and tilt from `rest_state`), so the textbook
/// motion applies from the start; returns the equilibrium (centre height, tilt).
fn place_at_equilibrium(s: &mut Scene, g_contact: [f64; 3]) -> (f64, f64) {
    let b = block_of(s);
    let (z, phi) = rest_state(b.h, b.m, g_contact, b.k, 0.0);
    s.bodies[0].chunks[0].center = [0.0, 0.0, 0.0];
    s.bodies[0].position = [0.0, 0.0, z];
    s.bodies[0].orientation = Quat::from_axis_angle(Vec3::Y, phi).to_wxyz();
    (z, phi)
}

// ------------------------------------------------------------------ Coulomb friction

/// Units in the last place of `x`.
fn ulp(x: f64) -> f64 {
    x.abs().next_up() - x.abs()
}

/// Steps `w` until the block has been still, to the rounding of its state, for a whole
/// rocking period, or until `limit`; returns whether it settled. Still: over the
/// period, its centre's coordinates move within `4 ulp` of themselves and its tilt
/// within `4 (ulp(z) / 2h + eps)` (the bounds of `a_box_set_down_flat_rests_at_the_
/// exact_static_equilibrium`). Frames must sample the bounce and rocking at least 8
/// times a period: an oscillation of amplitude `A` then shows a range of at least
/// `2 A cos(pi / 8)`, so the window bounds `A` within the same rounding. Positions, not
/// velocities: a velocity whose step `v dt` is below half an ulp of the position is
/// rounded away by the integrator and moves nothing (it persists, harmless), and an
/// oscillation passes zero velocity twice a period.
fn settle(w: &mut World, b: &Block, limit: f64) -> bool {
    let period = 2.0 * std::f64::consts::PI / (b.bounce() / 5f64.sqrt());
    let tilt_of = |w: &World| {
        let r = w.solver.clusters[w.solver.chunks[0][0].cluster].rotation();
        r.m[0][2].atan2(r.m[2][2])
    };
    let mut window: Option<(f64, [f64; 3], [f64; 3])> = None;
    while w.solver.time < limit {
        w.step_frame();
        let c = w.solver.chunk_position(0, 0);
        let now = [c.x, c.z, tilt_of(w)];
        let (t0, lo, hi) = window.get_or_insert((w.solver.time, now, now));
        for i in 0..3 {
            lo[i] = lo[i].min(now[i]);
            hi[i] = hi[i].max(now[i]);
        }
        if w.solver.time - *t0 >= period {
            let bound = [4.0 * ulp(c.x.abs().max(b.h)), 4.0 * ulp(c.z), 4.0 * (ulp(c.z) / (2.0 * b.h) + f64::EPSILON)];
            if (0..3).all(|i| hi[i] - lo[i] <= bound[i]) {
                return true;
            }
            window = None;
        }
    }
    false
}

/// Below the friction angle a block sticks: it settles at its elastic shear deflection
/// and then does not move (no creep), whatever its mass and the substep. Coulomb
/// friction holds any tangential force up to `mu N` with zero slip; a regularised
/// (viscous) law creeps.
///
/// The block starts at its exact contact equilibrium (sink and tilt, `rest_state`) with
/// the shear layer unloaded, so the friction is a step load. At 10 degrees the cone
/// holds 2.8 times it: the layer's damped response (sliding and rocking) stays inside,
/// no slip, and the block settles where the shear layer carries the load:
/// `e = m g sin(t) / (k_t A)`, to round-off (`4 ulp` at the coordinates' scale `h`,
/// as for the resting box). That holds while the second-order effect of the contact
/// point moving with the rocking (`~e^2 / h`) is below round-off: the light blocks. At
/// 20 degrees (2/3 of the friction angle, 1.37 times the load) the step can slip
/// briefly, a path-dependent offset; every block, at both slopes, must then hold still:
/// over the second half of the run after settling, it moves less than the rounding of
/// its position (`4 ulp(h)`): a creep of 1e-15 m/s would show.
#[test]
fn a_block_below_the_friction_angle_sticks_without_creep() {
    let mu = 0.5;
    let mut failures = Vec::new();
    for slope_deg in [10.0, 20.0] {
        let slope = f64::to_radians(slope_deg);
        for density in [24.0, 2400.0, 240_000.0] {
            for safety in [0.5, 0.1] {
                let mut s = incline(slope, mu, density, safety, 0.0);
                let gravity = s.gravity;
                place_at_equilibrium(&mut s, gravity);
                let b = block_of(&s);
                s.sim.frame_dt = 2.0 * std::f64::consts::PI / b.bounce() / 8.0;
                let e = b.m * G * slope.sin() / (b.k_t * b.area());
                let decay = (e.max(ulp(b.h)) / ulp(b.h)).ln() / b.slowest_decay();
                let mut w = World::new(&s);
                let settled = settle(&mut w, &b, 10.0 * decay + 1e-3);
                let x1 = w.solver.chunk_position(0, 0).x;
                let t1 = w.solver.time;
                while w.solver.time < 2.0 * t1 {
                    w.step_frame();
                }
                let x2 = w.solver.chunk_position(0, 0).x;
                let creep = x2 - x1;
                let exact = slope_deg == 10.0 && e * e / b.h < ulp(b.h);
                println!("{slope_deg} deg, density {density}, safety {safety}: settled {settled} at {t1:.4} s; offset {x1:.6e} (shear deflection {e:.6e}); then moved {creep:.1e} m in {t1:.4} s");
                if !settled || creep.abs() > 4.0 * ulp(b.h) || (exact && (x1 - e).abs() > 4.0 * ulp(b.h)) {
                    failures.push(format!("{slope_deg} deg, density {density} kg/m3, safety {safety}: settled {settled}, offset {x1:.6e} vs {e:.6e}, creep {creep:.1e} m"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "a sticking block creeps or settles off its elastic deflection:\n{}", failures.join("\n"));
}

/// Above the friction angle a block slides with `a = g (sin t - mu cos t)`, whatever
/// its mass and the substep.
///
/// The block starts at the contact equilibrium of steady sliding (normal force
/// `m g cos t`, friction `mu` of it: `rest_state`), at rest. Friction is the cap from the
/// first substep on (the layer's dashpot alone exceeds it at once), so the integrator
/// (which adds the force's `dt / m` to the velocity each substep, exactly) gives the
/// Coulomb acceleration exactly but for rounding, once the rocking excited by the
/// start has decayed (at the slowest rate, from 1 to `eps`). Per frame, the rounding
/// is that of the normal force, the overlap's volume from coordinates (`4 ulp(z)` of the
/// sink `delta`, as for the resting box), times `mu g cos t`, plus the velocity's own
/// (`ulp(v)` per substep, over a frame `2 ulp(v) / dt`).
#[test]
fn a_block_above_the_friction_angle_slides_at_the_coulomb_acceleration() {
    let (mu, slope) = (0.3, 30f64.to_radians());
    let expected = G * (slope.sin() - mu * slope.cos());
    let mut failures = Vec::new();
    for density in [24.0, 2400.0, 240_000.0] {
        for safety in [0.5, 0.1] {
            let mut s = incline(slope, mu, density, safety, 0.0);
            let (z, _) = place_at_equilibrium(&mut s, [mu * G * slope.cos(), 0.0, -G * slope.cos()]);
            let b = block_of(&s);
            let start = (1.0 / f64::EPSILON).ln() / b.slowest_decay();
            s.sim.duration = start + 0.05;
            let mut w = World::new(&s);
            let dt = w.substep_dt();
            let delta = b.h - z;
            let (mut worst, mut bound_used) = (0.0f64, 0.0f64);
            let mut prev: Option<(f64, f64)> = None;
            while w.solver.time < s.sim.duration - 0.5 * s.sim.frame_dt {
                w.step_frame();
                let v = w.solver.chunk_velocity(0, 0).0.x;
                if w.solver.time >= start {
                    if let Some((t0, v0)) = prev {
                        let a = (v - v0) / (w.solver.time - t0);
                        let bound = mu * G * slope.cos() * 4.0 * ulp(z) / delta + 2.0 * ulp(v) / dt;
                        if (a - expected).abs() / bound > worst / bound_used.max(1e-300) {
                            worst = (a - expected).abs();
                            bound_used = bound;
                        }
                    }
                    prev = Some((w.solver.time, v));
                }
            }
            println!("density {density}, safety {safety}: after {start:.3} s, |a - Coulomb| up to {worst:.2e} m/s2 (bound {bound_used:.2e}; Coulomb {expected:.6})");
            if worst > bound_used {
                failures.push(format!("density {density} kg/m3, safety {safety}: a off Coulomb's by {worst:.2e} m/s2 (bound {bound_used:.2e})"));
            }
        }
    }
    assert!(failures.is_empty(), "sliding acceleration is not Coulomb's:\n{}", failures.join("\n"));
}

/// A block sliding on flat ground stops after `v^2 / (2 mu g)` in `v / (mu g)` and then
/// stays put, whatever its mass and the substep.
///
/// It starts at the contact equilibrium of sliding (`rest_state` under `(mu g, 0, -g)`),
/// and friction is the cap from the first substep (the dashpot alone exceeds it), so
/// the integrator's velocity falls by exactly `a dt` per substep (`a = mu g`) and its
/// position, advanced with the new velocity (semi-implicit Euler), is the discrete sum
/// `dt sum_k (v0 - k a dt)`, up to the substep where the velocity would turn: the stop
/// is in substep `K = floor(v0 / (a dt))` or the next. Checked one substep per frame:
/// velocities and the position at the stop to the exact discrete values, to rounding
/// (per substep, `ulp(v)` and the friction's, as for the incline); and the discrete
/// stop against Coulomb's continuous one, which it differs from by the scheme's
/// `v0 dt / 2` plus at most the last partial substep `a dt^2` (distance), and `dt`
/// (time). After it stops, the friction layer's shear and the block's tilt relax (the
/// spring holds exactly `mu N` at the stop, so the release itself may slip, a
/// path-dependent amount); once settled, the block holds still (`4 ulp(h)`).
#[test]
fn a_sliding_block_stops_after_the_coulomb_distance() {
    let (mu, speed) = (0.5, 1.0);
    let a = mu * G;
    let (distance, time) = (speed * speed / (2.0 * a), speed / a);
    let mut failures = Vec::new();
    for density in [24.0, 2400.0, 240_000.0] {
        for safety in [0.5, 0.1] {
            let mut s = sliding(speed, mu, density, safety, 0.0);
            let (z, _) = place_at_equilibrium(&mut s, [a, 0.0, -G]);
            let b = block_of(&s);
            let dt = World::new(&s).substep_dt();
            s.sim.frame_dt = dt;
            let delta = b.h - z;
            let mut w = World::new(&s);
            assert_eq!(w.substep_dt(), dt, "one substep per frame");
            let x0 = w.solver.chunk_position(0, 0).x;
            let k_turn = (speed / (a * dt)).floor();
            let (mut k, mut x_sum, mut worst_v, mut worst_x, mut stop) = (0.0, 0.0, 0.0f64, 0.0f64, None);
            let per_step = mu * G * 4.0 * ulp(z) / delta * dt;
            while stop.is_none() && k < k_turn + 2.0 {
                w.step_frame();
                k += 1.0;
                let (v, x) = (w.solver.chunk_velocity(0, 0).0.x, w.solver.chunk_position(0, 0).x - x0);
                if v <= 0.0 {
                    stop = Some((k, x_sum, w.solver.time));
                    break;
                }
                let v_exact = speed - k * a * dt;
                x_sum += v_exact * dt;
                // Rounding: each substep adds the friction's and the velocity's own.
                worst_v = worst_v.max((v - v_exact).abs() / (k * (per_step + ulp(speed))));
                worst_x = worst_x.max((x - x_sum).abs() / (k * (per_step * dt * k + ulp(distance))));
            }
            let Some((k_stop, x_stop, t_stop)) = stop else {
                failures.push(format!("density {density}, safety {safety}: did not stop by substep {}", k_turn + 2.0));
                continue;
            };
            let dx = x_stop - distance;
            let settled = settle(&mut w, &b, t_stop + 10.0 * (1.0 / f64::EPSILON).ln() / b.slowest_decay());
            let x1 = w.solver.chunk_position(0, 0).x;
            let t1 = w.solver.time;
            while w.solver.time < t1 + 0.05 {
                w.step_frame();
            }
            let creep = w.solver.chunk_position(0, 0).x - x1;
            println!(
                "density {density}, safety {safety}: dt {dt:.3e}; stopped in substep {k_stop} (K {k_turn}); velocities and position to the discrete scheme within {worst_v:.2} and {worst_x:.2} of rounding; stop {x_stop:.9} m vs Coulomb {distance:.9} (off {dx:.2e}, scheme -v0 dt/2 = {:.2e}); at {t_stop:.6} s vs {time:.6}; settled {settled}, then moved {creep:.1e} m",
                -speed * dt / 2.0
            );
            if worst_v > 1.0 || worst_x > 1.0 || (k_stop - k_turn - 0.5).abs() > 0.5 + 1e-9 || (dx + speed * dt / 2.0).abs() > a * dt * dt + 4.0 * ulp(distance) || (t_stop - time).abs() > dt || !settled || creep.abs() > 4.0 * ulp(b.h) {
                failures.push(format!("density {density} kg/m3, safety {safety}: stop substep {k_stop} (K {k_turn}), velocity/position off the scheme by {worst_v:.2}/{worst_x:.2} of rounding, stop off Coulomb by {dx:.2e} (scheme {:.2e}), time {t_stop:.6} vs {time:.6}, settled {settled}, creep {creep:.1e}", -speed * dt / 2.0));
            }
        }
    }
    assert!(failures.is_empty(), "sliding to rest is not Coulomb's:\n{}", failures.join("\n"));
}

// ------------------------------------------------------------------ step independence

/// An impact converges as the substep shrinks, and does not depend on how the
/// substeps are grouped into frames.
///
/// Substep: halving it four times, the ram's rebound velocity changes by less at each
/// halving than at the one before (the scheme converges; elastic and with fracture,
/// crack contact and friction), and the number of broken bonds changes by no more.
/// Monotone convergence is the criterion: no tolerance. (Measured: elastic differences
/// shrink by 0.57 then 0.50, the scheme's first order; with fracture 0.37 then 0.67.)
///
/// Frame: the frame only groups substeps (loads, contacts and fracture act per
/// substep), so with the substep pinned, frames of any number of substeps give the
/// same run, bit for bit, at every common time (16, 4 and 1 compared every 16
/// substeps).
#[test]
fn an_impact_converges_as_the_substep_shrinks_and_does_not_depend_on_the_frame() {
    let run = |fracture: bool, safety: f64| {
        let mut s = small_impact();
        s.sim.fracture = fracture;
        s.sim.courant_safety = safety;
        let f = frames(&s);
        let last = f.last().unwrap();
        (broken(last).len() as f64, v(last.impactors[0].velocity).y)
    };
    for fracture in [false, true] {
        let outcomes: Vec<(f64, f64)> = [0.5, 0.25, 0.125, 0.0625].iter().map(|&k| run(fracture, k)).collect();
        println!("fracture {fracture}, safety 0.5 to 0.0625: {outcomes:?}");
        for k in 1..outcomes.len() - 1 {
            let (d0, d1) = ((outcomes[k].1 - outcomes[k - 1].1).abs(), (outcomes[k + 1].1 - outcomes[k].1).abs());
            let (b0, b1) = ((outcomes[k].0 - outcomes[k - 1].0).abs(), (outcomes[k + 1].0 - outcomes[k].0).abs());
            assert!(d1 < d0, "fracture {fracture}: the ram's velocity does not converge with the substep: {outcomes:?}");
            assert!(b1 <= b0, "fracture {fracture}: broken bonds do not converge with the substep: {outcomes:?}");
        }
        let base = {
            let mut s = small_impact();
            s.sim.fracture = fracture;
            s
        };
        let dt = World::new(&base).substep_dt() / 2.0;
        // The state at every 16th substep: chunks' and the ram's positions and
        // velocities, and the broken bonds.
        let partition = |n: f64| {
            let mut s = base.clone();
            s.sim.max_substep = Some(dt);
            s.sim.frame_dt = n * dt;
            let mut w = World::new(&s);
            let mut states = Vec::new();
            let total = ((base.sim.duration / dt / 16.0).ceil() * 16.0) as u64;
            while w.solver.substeps < total {
                w.step_frame();
                if w.solver.substeps % 16 == 0 {
                    let mut x: Vec<f64> = Vec::new();
                    for c in 0..w.solver.structures[0].chunks.len() {
                        let (p, (vl, va)) = (w.solver.chunk_position(0, c), w.solver.chunk_velocity(0, c));
                        x.extend([p.x, p.y, p.z, vl.x, vl.y, vl.z, va.x, va.y, va.z]);
                    }
                    let i = &w.impactors[0];
                    x.extend([i.pose.position.x, i.pose.position.y, i.pose.position.z, i.velocity.x, i.velocity.y, i.velocity.z]);
                    x.push(w.solver.broken_bond_count() as f64);
                    states.push(x);
                }
            }
            states
        };
        let coarse = partition(16.0);
        for n in [4.0, 1.0] {
            let fine = partition(n);
            let same = coarse.len() == fine.len() && coarse.iter().zip(&fine).all(|(a, b)| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()));
            assert!(same, "fracture {fracture}: frames of {n} substeps change the run");
        }
    }
}

/// The static equilibrium of a cube (half-size `h`, mass `m`) resting on the ground
/// plane `z = top` under gravity `g` (in the x-z plane), turned by `phi` about y: the
/// layer force `k V` straight up at the overlap's centroid (the ground's field is one
/// cell, and the cube's thickness along it does not turn: `k` is fixed), friction on the
/// ground plane at the same point, the overlap the turned square cut by the plane,
/// extruded over the cube's depth, all exact. Solved by Newton for the height of the
/// centre and the tilt; returns (centre height, tilt).
fn rest_state(h: f64, m: f64, g: [f64; 3], k: f64, top: f64) -> (f64, f64) {
    let overlap = |z: f64, phi: f64| -> (f64, f64, f64) {
        // (area, centroid x, centroid z) of the turned square below z = top.
        let (c, s) = (phi.cos(), phi.sin());
        let corners: Vec<(f64, f64)> = [(-h, -h), (h, -h), (h, h), (-h, h)].iter().map(|&(x, y)| (x * c + y * s, z - x * s + y * c)).collect();
        let mut poly = Vec::new();
        for i in 0..4 {
            let (p, q) = (corners[i], corners[(i + 1) % 4]);
            let (sp, sq) = (p.1 - top, q.1 - top);
            if sp <= 0.0 {
                poly.push(p);
            }
            if (sp < 0.0) != (sq < 0.0) && sp != 0.0 && sq != 0.0 {
                let t = sp / (sp - sq);
                poly.push((p.0 + (q.0 - p.0) * t, p.1 + (q.1 - p.1) * t));
            }
        }
        let (mut a, mut cx, mut cz) = (0.0, 0.0, 0.0);
        for i in 0..poly.len() {
            let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
            let cr = p.0 * q.1 - q.0 * p.1;
            a += cr / 2.0;
            cx += (p.0 + q.0) * cr / 6.0;
            cz += (p.1 + q.1) * cr / 6.0;
        }
        (a, cx / a, cz / a)
    };
    let gv = Vec3::from_array(g);
    let residual = |z: f64, phi: f64| -> (f64, f64) {
        let (a, px, pz) = overlap(z, phi);
        let v = a * 2.0 * h;
        let force = Vec3::Z * (k * v);
        // Force balance along the layer force, and no moment about the centre: the
        // contact point lies on the line of gravity through the centre.
        let along = force.norm() + m * gv.dot(force / force.norm());
        let lever = Vec3::new(px, 0.0, pz - z);
        (along / (m * gv.norm()), lever.cross(gv).y / (h * gv.norm()))
    };
    let (mut z, mut phi) = (top + h - m * gv.norm() / (k * 4.0 * h * h), 0.0);
    for _ in 0..60 {
        let (r0, r1) = residual(z, phi);
        let (dz, dp) = (1e-6 * h * 1e-3, 1e-9);
        let (a0, a1) = residual(z + dz, phi);
        let (b0, b1) = residual(z, phi + dp);
        let (j00, j10, j01, j11) = ((a0 - r0) / dz, (a1 - r1) / dz, (b0 - r0) / dp, (b1 - r1) / dp);
        let det = j00 * j11 - j01 * j10;
        let (sz, sp) = ((r0 * j11 - r1 * j01) / det, (j00 * r1 - j10 * r0) / det);
        z -= sz;
        phi -= sp;
    }
    (z, phi)
}

/// A number with its derivative (forward-mode differentiation), for exact gradients of
/// the closed-form energies below.
#[derive(Clone, Copy, Debug)]
struct Dual {
    v: f64,
    d: f64,
}

impl Dual {
    fn var(v: f64) -> Dual {
        Dual { v, d: 1.0 }
    }
    fn c(v: f64) -> Dual {
        Dual { v, d: 0.0 }
    }
    fn sin(self) -> Dual {
        Dual { v: self.v.sin(), d: self.d * self.v.cos() }
    }
    fn cos(self) -> Dual {
        Dual { v: self.v.cos(), d: -self.d * self.v.sin() }
    }
    fn sqrt(self) -> Dual {
        let r = self.v.sqrt();
        Dual { v: r, d: self.d / (2.0 * r) }
    }
    fn powi(self, n: i32) -> Dual {
        Dual { v: self.v.powi(n), d: self.d * n as f64 * self.v.powi(n - 1) }
    }
}

impl std::ops::Add for Dual {
    type Output = Dual;
    fn add(self, o: Dual) -> Dual {
        Dual { v: self.v + o.v, d: self.d + o.d }
    }
}
impl std::ops::Sub for Dual {
    type Output = Dual;
    fn sub(self, o: Dual) -> Dual {
        Dual { v: self.v - o.v, d: self.d - o.d }
    }
}
impl std::ops::Mul for Dual {
    type Output = Dual;
    fn mul(self, o: Dual) -> Dual {
        Dual { v: self.v * o.v, d: self.d * o.v + self.v * o.d }
    }
}
impl std::ops::Div for Dual {
    type Output = Dual;
    fn div(self, o: Dual) -> Dual {
        Dual { v: self.v / o.v, d: (self.d * o.v - self.v * o.d) / (o.v * o.v) }
    }
}

/// The pair contact's elastic energy for a cube (half-size `h`, plane-strain modulus
/// `e`) resting on a wider slab (half-thicknesses `4h` sideways, `h` vertically, same
/// material), tilted by `phi` about y, `delta` the overlap thickness under the cube's
/// centre. The pair law (`world::pair_elastic`) is `E = (E_slab + E_cube) / 2`, each
/// field `sum_f k_f int s dV` over the overlap's cells nearest that body's faces, `k_f`
/// from the two bodies' half-thicknesses along the face's normal.
///
/// The slab's field is one cell, its top: depth `top - z`, integrated in vertical
/// columns over the cube's footprint, plus the slivers where the cube's leaning side
/// faces pass (low side) or fall short of (high side) the footprint.
///
/// The cube's field, in the cube's frame: over its bottom face at `(x, y)` the overlap
/// is `T(x) = (delta + x sin phi) / cos phi` thick (to the slab's top); a point at
/// height `z` is in the cell of the nearest of the bottom (`z`), the x-sides
/// (`a = h - |x|`) and the y-sides (`v = h - |y|`). Integrating `z`, then `y`, in
/// closed form leaves polynomials of degree at most 4 in `x` between the kinks
/// (`x = 0`, `T = a`), integrated exactly by 3-point Gauss-Legendre. Exact but for the
/// slab's slivers' own `O(phi t^4)` ends, relatively `1e-14` here.
fn slab_energy(h: f64, e: f64, delta: Dual, phi: Dual) -> Dual {
    let (s, c) = (phi.sin(), phi.cos());
    let hh = Dual::c(h);
    let one = Dual::c(1.0);
    let k_top = Dual::c(1.0 / (h / e + h / e));
    let k_bottom = one / (Dual::c(h / e) + (Dual::c(16.0 * h * h) * s * s + Dual::c(h * h) * c * c).sqrt() / Dual::c(e));
    let k_xside = one / (Dual::c(h / e) + (Dual::c(16.0 * h * h) * c * c + Dual::c(h * h) * s * s).sqrt() / Dual::c(e));
    let k_yside = Dual::c(1.0 / (h / e + 4.0 * h / e));
    // The slab's field: vertical thickness t = delta + x sin(phi) over the footprint
    // (dx_world = cos(phi) dx), plus the slivers int s w ds = tan(phi) t^3 / 6.
    let squares = hh * (Dual::c(2.0) * hh * delta * delta + Dual::c(2.0 / 3.0) * hh.powi(3) * s * s);
    let (t_plus, t_minus) = (delta + hh * s, delta - hh * s);
    let sliver = Dual::c(2.0) * hh * s / c * (t_plus.powi(3) - t_minus.powi(3)) / Dual::c(6.0);
    let slab_field = k_top * (c * squares + sliver);
    // The cube's field, per unit x after integrating z and y.
    let column = |x: Dual| -> Dual {
        let t = (delta + x * s) / c;
        let a = if x.v >= 0.0 { hh - x } else { hh + x };
        let m0 = if t.v < a.v { t } else { a };
        let bottom = hh * m0 * m0 - Dual::c(2.0 / 3.0) * m0.powi(3);
        let xside = if a.v < t.v { a * (t - a) * Dual::c(2.0) * (hh - a) } else { Dual::c(0.0) };
        let yside = Dual::c(2.0) * (t * m0 * m0 / Dual::c(2.0) - m0.powi(3) / Dual::c(3.0));
        k_bottom * bottom + k_xside * xside + k_yside * yside
    };
    // Kinks: x = 0 and T(x) = h - |x| on each side.
    let k_plus = (c * hh - delta) / (s + c);
    let k_minus = (delta - c * hh) / (c - s);
    let nodes = [(-0.774_596_669_241_483_4, 5.0 / 9.0), (0.0, 8.0 / 9.0), (0.774_596_669_241_483_4, 5.0 / 9.0)];
    let mut cube_field = Dual::c(0.0);
    let cuts = [Dual::c(-h), k_minus, Dual::c(0.0), k_plus, Dual::c(h)];
    for w in cuts.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        let (mid, half) = ((lo + hi) / Dual::c(2.0), (hi - lo) / Dual::c(2.0));
        for (xi, wi) in nodes {
            cube_field = cube_field + column(mid + half * Dual::c(xi)) * half * Dual::c(wi);
        }
    }
    // Depth 2h over y is inside `column` (the y-integral); E per field, halved.
    (slab_field + cube_field) / Dual::c(2.0)
}

/// Height of the point where the pair contact's forces act (friction among them): the
/// stiffness-weighted centroid of all cells of both fields (`world::pair_elastic`). The
/// slab's one cell is the whole overlap; the cube's cells tile it, so their moments
/// (per unit x, as in `slab_energy`, integrated exactly) give every cell's.
fn slab_contact_height(h: f64, e: f64, delta: f64, phi: f64, z_centre: f64) -> f64 {
    let (s, c) = (phi.sin(), phi.cos());
    let k_top = 1.0 / (h / e + h / e);
    let k_bottom = 1.0 / (h / e + (16.0 * h * h * s * s + h * h * c * c).sqrt() / e);
    let k_xside = 1.0 / (h / e + (16.0 * h * h * c * c + h * h * s * s).sqrt() / e);
    let k_yside = 1.0 / (h / e + 4.0 * h / e);
    // Per cell, per unit x: (volume, int zeta, ), zeta the height above the bottom face.
    let column = |x: f64| -> [(f64, f64, f64); 3] {
        let t = (delta + x * s) / c;
        let a = h - x.abs();
        let m0 = t.min(a);
        let bottom = (2.0 * h * m0 - m0 * m0, h * m0 * m0 - 2.0 / 3.0 * m0.powi(3), k_bottom);
        let xside = if a < t { ((t - a) * 2.0 * (h - a), (t * t - a * a) / 2.0 * 2.0 * (h - a), k_xside) } else { (0.0, 0.0, k_xside) };
        let yside = (2.0 * (t * m0 - m0 * m0 / 2.0), t * t * m0 - m0.powi(3) / 3.0, k_yside);
        [bottom, xside, yside]
    };
    let k_plus = (c * h - delta) / (s + c);
    let k_minus = (delta - c * h) / (c - s);
    let nodes = [(-0.774_596_669_241_483_4, 5.0 / 9.0), (0.0, 8.0 / 9.0), (0.774_596_669_241_483_4, 5.0 / 9.0)];
    // Totals: sum w V, sum w int z dV (world z of a point: z_centre - s x + c (zeta - h)).
    let (mut wv, mut wz, mut v_all, mut z_all) = (0.0, 0.0, 0.0, 0.0);
    for w in [-h, k_minus, 0.0, k_plus, h].windows(2) {
        let (mid, half) = ((w[0] + w[1]) / 2.0, (w[1] - w[0]) / 2.0);
        for (xi, wi) in nodes {
            let x = mid + half * xi;
            for (vol, zeta, k) in column(x) {
                let z = (z_centre - s * x - c * h) * vol + c * zeta;
                wv += 0.5 * k * vol * half * wi;
                wz += 0.5 * k * z * half * wi;
                v_all += vol * half * wi;
                z_all += z * half * wi;
            }
        }
    }
    // The slab's cell: the whole overlap, at weight k_top / 2.
    (wz + 0.5 * k_top * z_all) / (wv + 0.5 * k_top * v_all)
}

/// The resting state of the cube on the slab under gravity `g`: the elastic force
/// `-dE/dz` balances gravity's normal part, friction (at the overlap's centroid, on the
/// contact plane) its tangential part, and the moments about the centre balance. The
/// energy does not change with a sideways shift (the slab is uniform), so the elastic
/// force is vertical. Returns (centre height, tilt).
fn slab_rest_state(h: f64, m: f64, g: [f64; 3], e: f64, top: f64) -> (f64, f64) {
    let gv = Vec3::from_array(g);
    // delta = top - z + h cos(phi): the overlap thickness on the cube's axis.
    let residual = |z: f64, phi: f64| -> (f64, f64) {
        let delta = top - z + h * phi.cos();
        let de_ddelta = slab_energy(h, e, Dual::var(delta), Dual::c(phi)).d;
        // dE/dphi at fixed z: delta moves with phi.
        let de_dphi = slab_energy(h, e, Dual { v: delta, d: -h * phi.sin() }, Dual::var(phi)).d;
        let lift = de_ddelta; // -dE/dz
        // Where friction acts, relative to the centre.
        let arm = slab_contact_height(h, e, delta, phi, z) - z;
        // Friction is -m g_x at the arm: its moment about y is arm * (-m g_x).
        let moment = -de_dphi - arm * m * gv.x;
        ((lift + m * gv.z) / (m * gv.norm()), moment / (m * gv.norm() * h))
    };
    let (mut z, mut phi) = (top + h - m * gv.norm() / (e / (2.0 * h) * 4.0 * h * h), 0.0);
    for _ in 0..60 {
        let (r0, r1) = residual(z, phi);
        let (dz, dp) = (1e-9 * h * 1e-3, 1e-12);
        let (a0, a1) = residual(z + dz, phi);
        let (b0, b1) = residual(z, phi + dp);
        let (j00, j10, j01, j11) = ((a0 - r0) / dz, (a1 - r1) / dz, (b0 - r0) / dp, (b1 - r1) / dp);
        let det = j00 * j11 - j01 * j10;
        z -= (r0 * j11 - r1 * j01) / det;
        phi -= (j00 * r1 - j10 * r0) / det;
    }
    (z, phi)
}
