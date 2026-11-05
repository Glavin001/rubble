//! Shared builders, measures and test bodies of the small analytic invariant
//! tests (`engine/docs/vehicle-test-strategy.md`, layer 1). Each body runs one
//! small system on a [`Sim`] (the CPU reference here, the GPU kernel in
//! `engine-webgpu/tests/invariants_small_gpu.rs`, which includes this file) and
//! compares it with a closed form. Every tolerance is derived in its comment.
#![allow(dead_code)]

use engine_softbody::explicit_beams::{
    Beam, ExplicitBeams, ExplicitBeamsSettings, ExplicitKind, ExplicitType, JbeamWheels, Node,
    NODE_IMMOVABLE, NODE_NO_GROUND, STEP,
};
use engine_softbody::jbeam::load_jbeam;
use glam::{DVec3, Vec3};
use std::f64::consts::PI;

pub const H: f64 = STEP as f64;

/// The analytic tests' profile: exact lengths, no air (the closed forms have
/// neither the fast inverse square root's 0.2 % length error nor drag).
pub fn exact() -> ExplicitBeamsSettings {
    ExplicitBeamsSettings {
        fast_inv_sqrt: false,
        drag: false,
        turbulence: false,
        triangle_throttle: true,
    }
}

/// A simulation the bodies drive: the CPU reference or the GPU kernel.
pub trait Sim {
    fn steps(&mut self, n: u32);
    fn pos(&self) -> Vec<Vec3>;
    fn vel(&self) -> Vec<Vec3>;
    /// Per beam (broken, rest length).
    fn beams(&self) -> Vec<(bool, f32)>;
    fn set_vel(&mut self, node: u32, v: Vec3);
    fn label(&self) -> &'static str;
}

pub struct Cpu(pub ExplicitBeams);

impl Sim for Cpu {
    fn steps(&mut self, n: u32) {
        for _ in 0..n {
            self.0.step();
        }
    }
    fn pos(&self) -> Vec<Vec3> {
        self.0.positions()
    }
    fn vel(&self) -> Vec<Vec3> {
        self.0.nodes.iter().map(|n| n.vel).collect()
    }
    fn beams(&self) -> Vec<(bool, f32)> {
        self.0.beams.iter().map(|b| (b.broken, b.rest)).collect()
    }
    fn set_vel(&mut self, node: u32, v: Vec3) {
        self.0.nodes[node as usize].vel = v;
    }
    fn label(&self) -> &'static str {
        "CPU"
    }
}

pub type Maker<'a> = &'a dyn Fn(ExplicitBeams) -> Option<Box<dyn Sim>>;

pub fn cpu(s: ExplicitBeams) -> Option<Box<dyn Sim>> {
    Some(Box::new(Cpu(s)))
}

pub fn node(pos: Vec3, mass: f32, flags: u32) -> Node {
    Node {
        pos,
        vel: Vec3::ZERO,
        force: Vec3::ZERO,
        mass,
        friction: 1.0,
        flags: flags | NODE_NO_GROUND,
    }
}

/// A plain (RoR-rule) beam, deformable, with the given thresholds.
pub fn plain(a: u32, b: u32, k: f32, c: f32, rest: f32, deform: f32, strength: f32) -> Beam {
    Beam::new(
        a,
        b,
        ExplicitKind::Plain,
        ExplicitType::Deformable,
        k,
        c,
        rest,
        deform,
        strength,
        0.0,
    )
}

/// A JBeam NORMAL beam (spec §3) with no deform limits or hardening cap.
pub fn jplain(a: u32, b: u32, k: f32, c: f32, rest: f32, deform: f32, strength: f32) -> Beam {
    let mut b = Beam::new(
        a,
        b,
        ExplicitKind::JbeamPlain,
        ExplicitType::Deformable,
        k,
        c,
        rest,
        deform,
        strength,
        0.0,
    );
    // Spec §2: the tests start at G = min(deform, S).
    b.gate = deform.min(strength);
    b.short_bound = 0.0;
    b.long_bound = f32::INFINITY;
    b.limit_k = f32::INFINITY;
    b
}

pub fn system(nodes: Vec<Node>, beams: Vec<Beam>) -> ExplicitBeams {
    let mut s = ExplicitBeams::empty(exact());
    s.ground = -1.0e3;
    s.nodes = nodes;
    s.beams = beams;
    s.finish();
    s
}

// --- Closed forms of the solver's step (symplectic Euler, exact) ---------------
//
// The node pass integrates the force of the last beam pass: v' = v + h·F(x, v)/m,
// x' = x + h·v'. For one degree of freedom of mass m on spring k and dashpot c
// the step is the linear map A = [[1 − h²k/m, h(1 − hc/m)], [−hk/m, 1 − hc/m]]:
// det A = 1 − hc/m and tr A = 2 − h²k/m − hc/m, so its eigenvalues r·e^{±iθ}
// have r = √(1 − hc/m) per step and 2r·cos θ = tr A.

/// The discrete angular frequency (rad/s) of the step's oscillator.
pub fn discrete_omega(k: f64, c: f64, m: f64) -> f64 {
    let r = (1.0 - H * c / m).sqrt();
    let tr = 2.0 - H * H * k / m - H * c / m;
    (tr / (2.0 * r)).acos() / H
}

/// The discrete decay rate (1/s) of the step's oscillator: −ln r / h.
pub fn discrete_sigma(c: f64, m: f64) -> f64 {
    -(1.0 - H * c / m).sqrt().ln() / H
}

/// Upward zero crossings of `y` (times by linear interpolation, in steps).
pub fn crossings(y: &[f64]) -> Vec<f64> {
    let mut out = Vec::new();
    for i in 1..y.len() {
        if y[i - 1] < 0.0 && y[i] >= 0.0 {
            out.push((i - 1) as f64 + y[i - 1] / (y[i - 1] - y[i]));
        }
    }
    out
}

/// Maxima of |y| between sign changes, refined by a parabola through the three
/// samples (time in steps, value).
pub fn peaks(y: &[f64]) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    for i in 1..y.len() - 1 {
        let (a, b, c) = (y[i - 1].abs(), y[i].abs(), y[i + 1].abs());
        if b >= a
            && b > c
            && y[i - 1].signum() == y[i].signum()
            && y[i + 1].signum() == y[i].signum()
        {
            let d = a - 2.0 * b + c;
            let off = if d != 0.0 { 0.5 * (a - c) / d } else { 0.0 };
            out.push((i as f64 + off, b - 0.25 * (a - c) * off));
        }
    }
    out
}

/// Least-squares slope of (x, y).
pub fn slope(p: &[(f64, f64)]) -> f64 {
    let n = p.len() as f64;
    let (sx, sy) = p.iter().fold((0.0, 0.0), |(a, b), &(x, y)| (a + x, b + y));
    let (mx, my) = (sx / n, sy / n);
    let (num, den) = p.iter().fold((0.0, 0.0), |(a, b), &(x, y)| {
        (a + (x - mx) * (y - my), b + (x - mx) * (x - mx))
    });
    num / den
}

pub fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-30)
}

/// One row of the report table (`vehicle-test-strategy.md`).
pub fn row(name: &str, sim: &str, measured: String, expected: String, pass: bool) {
    eprintln!(
        "| {name} | {sim} | {measured} | {expected} | {} |",
        if pass { "pass" } else { "FAIL" }
    );
}

// --- 1. A single beam ----------------------------------------------------------

/// The showcase sedan's body skin (`vehicle-authoring.md`: skin k 8.7e5 N/m on
/// 1.53 kg nodes) and the BX's measured body damping ratio ζ = 0.0275.
pub const SKIN_K: f32 = 8.7e5;
pub const SKIN_M: f32 = 1.53;
pub const BX_ZETA_BODY: f64 = 0.0275;
pub const BX_ZETA_LID: f64 = 0.0567;

/// A node of mass `m` hanging from an immovable anchor on a beam of stiffness k,
/// damping c and rest 1 m, pulled `amp` below its equilibrium (mg/k below rest).
pub fn hanging(k: f32, c: f32, m: f32, amp: f32) -> ExplicitBeams {
    let g = -engine_softbody::explicit_beams::GRAVITY;
    let top = Vec3::new(0.0, 3.0, 0.0);
    let eq = 1.0 + m * g / k;
    let mut bottom = node(top - Vec3::Y * (eq + amp), m, 0);
    // The spawn force (the node pass integrates the last beam pass's): gravity and
    // the beam's pull, so the first step is the oscillator's.
    bottom.force = Vec3::new(0.0, m * -g + k * (eq + amp - 1.0), 0.0);
    system(
        vec![node(top, 1.0, NODE_IMMOVABLE), bottom],
        vec![plain(0, 1, k, c, 1.0, f32::INFINITY, f32::INFINITY)],
    )
}

/// Records the hanging node's height about its equilibrium for `n` steps.
pub fn record_hanging(sim: &mut dyn Sim, k: f32, m: f32, n: usize) -> Vec<f64> {
    let g = -engine_softbody::explicit_beams::GRAVITY as f64;
    let eq = 3.0 - 1.0 - m as f64 * g / k as f64;
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        sim.steps(1);
        y.push(sim.pos()[1].y as f64 - eq);
    }
    y
}

pub struct OscOut {
    pub omega: f64,
    pub omega_discrete: f64,
    pub omega_continuous: f64,
    pub sigma: f64,
    pub sigma_discrete: f64,
    pub sigma_continuous: f64,
}

/// Natural frequency and decay of the hanging node (k, ζ, m).
pub fn oscillator(make: Maker, k: f32, zeta: f64, m: f32) -> Option<OscOut> {
    let c = (2.0 * zeta * ((k * m) as f64).sqrt()) as f32;
    let mut sim = make(hanging(k, c, m, 0.01))?;
    // 0.25 s: 30 periods at 121 Hz.
    let y = record_hanging(sim.as_mut(), k, m, 500);
    // Crossings while the amplitude is above 1 % of its start (a decayed signal's
    // late crossings sit near the f32 rounding of positions at y ≈ 2 m).
    let x: Vec<f64> = crossings(&y)
        .into_iter()
        .filter(|&t| (-(zeta * (k as f64 / m as f64).sqrt()) * t * H).exp() > 1e-2)
        .collect();
    let period = (x[x.len() - 1] - x[0]) / (x.len() - 1) as f64 * H;
    let pk: Vec<(f64, f64)> = peaks(&y)
        .into_iter()
        .filter(|p| p.1 > 1e-6)
        .map(|(t, a)| (t * H, a.ln()))
        .collect();
    let sigma = if zeta > 0.0 { -slope(&pk) } else { 0.0 };
    let (k, c, m) = (k as f64, c as f64, m as f64);
    Some(OscOut {
        omega: 2.0 * PI / period,
        omega_discrete: discrete_omega(k, c, m),
        omega_continuous: (k / m).sqrt(),
        sigma,
        sigma_discrete: discrete_sigma(c, m),
        sigma_continuous: zeta * (k / m).sqrt(),
    })
}

/// A heavy anchor and a heavy driver (1e9 kg: a beam force of 10⁵ N moves them by
/// 10⁻¹⁶ m/s per step, kinematic to rounding) on `beam` along x, the driver at
/// `v` m/s: displacement control on the CPU and the GPU alike. Both fall together
/// under gravity, which leaves their separation alone.
pub fn driven(beam: Beam, len: f32, v: f32) -> ExplicitBeams {
    let mut a = node(Vec3::new(0.0, 3.0, 0.0), 1.0e9, 0);
    let mut b = node(Vec3::new(len, 3.0, 0.0), 1.0e9, 0);
    a.vel = Vec3::ZERO;
    b.vel = Vec3::new(v, 0.0, 0.0);
    let g = engine_softbody::explicit_beams::GRAVITY;
    a.force = Vec3::new(0.0, a.mass * g, 0.0);
    b.force = a.force;
    system(vec![a, b], vec![beam])
}

/// (extension from the spawn length, rest length − spawn rest, elastic force k(ℓ −
/// L) as tension positive, broken) every `every` steps.
pub fn pull(
    sim: &mut dyn Sim,
    k: f32,
    l0: f32,
    rest0: f32,
    v: f32,
    to: f32,
    every: u32,
) -> Vec<(f64, f64, f64, bool)> {
    let mut out = Vec::new();
    // (Along x only: the pair falls together, keep its fall.)
    let mut u = sim.vel()[1];
    u.x = v;
    sim.set_vel(1, u);
    let n = ((to / v).abs() / STEP / every as f32).ceil() as usize;
    for _ in 0..n {
        sim.steps(every);
        let p = sim.pos();
        let len = (p[1] - p[0]).length() as f64;
        let (broken, rest) = sim.beams()[0];
        out.push((
            len - l0 as f64,
            rest as f64 - rest0 as f64,
            k as f64 * (len - rest as f64),
            broken,
        ));
        if broken {
            break;
        }
    }
    out
}

/// Yield, plastic set and unloading under displacement control (tension).
/// Closed form with the solver's hardening (spec §3: Tt ← Tt·L/L_old, so
/// Tt = Fy·L/L₀ while yielding): the rest length after a pull to extension x is
/// L = (k·(L₀ + x))/(k + Fy/L₀), the plastic set L − L₀ = (k·x − Fy)/(k + Fy/L₀),
/// (F − Fy)/k to first order in Fy/(k·L₀). Unloading is elastic (slope k) and
/// passes zero force at the set.
pub struct YieldOut {
    pub set: f64,
    pub set_closed: f64,
    pub unload_slope: f64,
    pub zero_force_at: f64,
    pub yield_force: f64,
}

pub fn yield_unload(make: Maker, jbeam: bool) -> Option<YieldOut> {
    let (k, fy, l0) = (1.0e5f32, 1000.0f32, 1.0f32);
    let beam = if jbeam {
        jplain(0, 1, k, 0.0, l0, fy, 1.0e9)
    } else {
        plain(0, 1, k, 0.0, l0, fy, 1.0e9)
    };
    let v = 0.05;
    let mut sim = make(driven(beam, l0, v))?;
    let x_max = 0.03f32;
    let load = pull(sim.as_mut(), k, l0, l0, v, x_max, 1);
    let x_reached = load.last().unwrap().0;
    // The first yield: the elastic force where the rest length first moves.
    let first = load.iter().find(|r| r.1.abs() > 1e-7).unwrap();
    let unload = pull(sim.as_mut(), k, l0, l0, -v, 0.015, 1);
    let set = unload.last().unwrap().1;
    // Slope over the first 5 mm of unloading (elastic).
    let pts: Vec<(f64, f64)> = unload
        .iter()
        .filter(|r| r.0 > x_reached - 0.005)
        .map(|r| (r.0, r.2))
        .collect();
    let zero = unload
        .windows(2)
        .find(|w| w[0].2 > 0.0 && w[1].2 <= 0.0)
        .map(|w| w[0].0 + (w[1].0 - w[0].0) * w[0].2 / (w[0].2 - w[1].2))
        .unwrap_or(f64::NAN);
    let (kd, fyd, l0d) = (k as f64, fy as f64, l0 as f64);
    Some(YieldOut {
        set,
        set_closed: (kd * x_reached - fyd) / (kd + fyd / l0d),
        unload_slope: slope(&pts),
        zero_force_at: zero,
        yield_force: first.2,
    })
}

/// Break under displacement control: (extension at break, force at break,
/// closed-form extension, closed-form force).
/// - JBeam NORMAL, deform ≥ strength S (brittle): breaks when |F| first passes S,
///   at x = S/k.
/// - RoR plain, deform Fy < S, in tension: it yields, hardens (Tt = Fy·L/L₀) and
///   its strength drops by k × its plastic stretch (S − k·(L − L₀)); it breaks
///   where they meet, L − L₀ = (S − Fy)/(k + Fy/L₀), at force Fy·L/L₀ and extension
///   (L − L₀) + Fy·(L/L₀)/k.
pub fn break_force(make: Maker, ductile: bool) -> Option<(f64, f64, f64, f64)> {
    let (k, l0, s) = (1.0e5f32, 1.0f32, 2000.0f32);
    let (beam, fy) = if ductile {
        (plain(0, 1, k, 0.0, l0, 1000.0, s), 1000.0f64)
    } else {
        (jplain(0, 1, k, 0.0, l0, f32::INFINITY, s), f64::INFINITY)
    };
    let v = 0.05;
    let mut sim = make(driven(beam, l0, v))?;
    let rows = pull(sim.as_mut(), k, l0, l0, v, 0.05, 1);
    let last = rows.iter().rev().find(|r| !r.3)?;
    let brk = rows.last()?;
    if !brk.3 {
        return Some((f64::NAN, f64::NAN, 0.0, 0.0));
    }
    let (kd, sd, l0d) = (k as f64, s as f64, l0 as f64);
    let (x, f) = if ductile {
        let dl = (sd - fy) / (kd + fy / l0d);
        let f = fy * (l0d + dl) / l0d;
        (dl + f / kd, f)
    } else {
        (sd / kd, sd)
    };
    Some((brk.0, last.2, x, f))
}

/// The elastic energy stored in a pulled beam against F²/2k, and the work done on
/// it against the stored plus the plastic dissipation ∫Tt dL = Fy·(L² − L₀²)/(2L₀)
/// after a yield (Tt = Fy·L/L₀). (stored, F²/2k, work, stored + plastic).
pub fn energy_stored(make: Maker) -> Option<(f64, f64, f64, f64)> {
    let (k, fy, l0) = (1.0e5f32, 1000.0f32, 1.0f32);
    let v = 0.05;
    let mut sim = make(driven(plain(0, 1, k, 0.0, l0, fy, 1.0e9), l0, v))?;
    // Elastic part: to 0.8 of the yield extension.
    let rows = pull(sim.as_mut(), k, l0, l0, v, 0.8 * fy / k, 1);
    let r = rows.last().unwrap();
    let stored = 0.5 * k as f64 * (r.2 / k as f64).powi(2);
    let f2 = r.2 * r.2 / (2.0 * k as f64);
    // Then on to 3 cm: the work as the trapezoid of the force over each step.
    let more = pull(sim.as_mut(), k, l0, l0, v, 0.03 - 0.8 * fy / k, 1);
    let mut work = 0.0;
    let mut prev = (r.0, r.2);
    for q in &more {
        work += 0.5 * (prev.1 + q.2) * (q.0 - prev.0);
        prev = (q.0, q.2);
    }
    let work = stored + work;
    let end = more.last().unwrap();
    let l_end = l0 as f64 + end.1;
    let plastic = fy as f64 * (l_end * l_end - (l0 as f64).powi(2)) / (2.0 * l0 as f64);
    let u_end = end.2 * end.2 / (2.0 * k as f64);
    Some((stored, f2, work, u_end + plastic))
}

// --- 2. Beam types (JBeam text through the loader and the spawn) ---------------

/// A two-node JBeam vehicle: nodes a (x = 0) and b (x = len) along JBeam X, the
/// beam row's options `opts` (a JSON object's body), spawned 3 m up with no
/// collision.
pub fn jbeam_pair(opts: &str, len: f32) -> ExplicitBeams {
    let text = format!(
        r#"{{ "t": {{ "information": {{"name": "pair"}}, "slotType": "main",
  "nodes": [["id", "posX", "posY", "posZ"], {{"nodeWeight": 1, "collision": false, "selfCollision": false}},
            ["a", 0, 0, 0], ["b", {len}, 0, 0]],
  "beams": [["id1:", "id2:"], {{{opts}}}, ["a", "b"]] }} }}"#
    );
    spawn_jbeam(&text)
}

pub fn spawn_jbeam(text: &str) -> ExplicitBeams {
    let d = load_jbeam(&[("t.jbeam", text)], None).unwrap();
    let wheels = JbeamWheels::Soft(engine_softbody::truck_export::TyreDefaults::for_vehicle(&d));
    let (s, _) = ExplicitBeams::from_desc_jbeam_with(
        &d,
        wheels,
        Vec3::new(0.0, 3.0, 0.0),
        Vec3::ZERO,
        exact(),
    )
    .unwrap();
    s
}

/// Makes nodes 0 and 1 of `s` the heavy anchor and driver of [`driven`] (the
/// beam's x axis is the spawn's: JBeam X is Rubble X).
pub fn make_driven(mut s: ExplicitBeams) -> ExplicitBeams {
    let g = engine_softbody::explicit_beams::GRAVITY;
    for n in s.nodes.iter_mut().take(2) {
        n.mass = 1.0e9;
        n.flags |= NODE_NO_GROUND;
        n.force = Vec3::new(0.0, n.mass * g, 0.0);
    }
    s
}

/// The force law of a two-node beam swept quasi-statically from extension
/// `from` to `to` (m): (extension, force as tension positive from the CPU's beam
/// record) at each step. CPU only (the GPU keeps no force record).
pub fn sweep(mut s: ExplicitBeams, from: f32, to: f32, v: f32) -> Vec<(f64, f64, bool)> {
    let l0 = (s.nodes[1].pos - s.nodes[0].pos).length();
    s.nodes[1].pos.x = s.nodes[0].pos.x + l0 + from;
    let v = v.copysign(to - from);
    s.nodes[1].vel = Vec3::new(v, 0.0, 0.0);
    let n = ((to - from).abs() / v.abs() / STEP).round() as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s.step();
        let len = (s.nodes[1].pos - s.nodes[0].pos).length();
        out.push((
            (len - l0) as f64,
            -s.beams[0].force as f64,
            s.beams[0].broken,
        ));
        if s.beams[0].broken {
            break;
        }
    }
    out
}

// --- Measures shared with the crash invariants ------------------------------------

pub fn momentum(s: &ExplicitBeams) -> DVec3 {
    s.nodes
        .iter()
        .map(|n| n.vel.as_dvec3() * n.mass as f64)
        .sum()
}

pub fn kinetic(nodes: &[Node]) -> f64 {
    nodes
        .iter()
        .map(|n| 0.5 * n.mass as f64 * n.vel.length_squared() as f64)
        .sum()
}

// --- Dynamic bodies shared with the GPU ------------------------------------------

/// Two free nodes of mass m on an ANISOTROPIC beam (k in compression, kₑ in
/// extension, no transition zone), pulled `amp` apart: a bilinear oscillator. Its
/// relative coordinate has mass μ = m/2 and its period is half of each side's,
/// T = π√(μ/k) + π√(μ/kₑ) (uniform gravity leaves the separation alone).
/// (measured period, closed form, closed form of the step: π/ω_d(k) + π/ω_d(kₑ)).
pub fn anisotropic_period(make: Maker) -> Option<(f64, f64, f64)> {
    let (k, ke, m) = (2.0e5f32, 8.0e5f32, 1.0f32);
    let mut s = jbeam_pair(
        &format!(
            r#""beamType": "|ANISOTROPIC", "beamSpring": {k}, "beamDamp": 0, "springExpansion": {ke},
               "dampExpansion": 0, "transitionZone": 0, "beamDeform": "FLT_MAX", "beamStrength": "FLT_MAX""#
        ),
        0.5,
    );
    for n in &mut s.nodes {
        n.mass = m;
        n.flags |= NODE_NO_GROUND;
    }
    s.nodes[1].pos.x += 0.005;
    let mut sim = make(s)?;
    let mut y = Vec::new();
    for _ in 0..2000 {
        sim.steps(1);
        let p = sim.pos();
        y.push(((p[1] - p[0]).length() - 0.5) as f64);
    }
    let x = crossings(&y);
    let period = (x[x.len() - 1] - x[0]) / (x.len() - 1) as f64 * H;
    let mu = m as f64 / 2.0;
    let (k, ke) = (k as f64, ke as f64);
    Some((
        period,
        PI * (mu / k).sqrt() + PI * (mu / ke).sqrt(),
        PI / discrete_omega(k, 0.0, mu) + PI / discrete_omega(ke, 0.0, mu),
    ))
}

/// A bar: `n` free nodes of mass m every `l` metres along x on beams k, c, its
/// first node a heavy piston (1e9 kg) moving at v0 into it.
pub fn bar(n: usize, k: f32, c: f32, m: f32, l: f32, v0: f32) -> ExplicitBeams {
    let g = engine_softbody::explicit_beams::GRAVITY;
    let mut nodes: Vec<Node> = (0..n)
        .map(|i| {
            let mut q = node(Vec3::new(i as f32 * l, 3.0, 0.0), m, 0);
            q.force = Vec3::new(0.0, m * g, 0.0);
            q
        })
        .collect();
    nodes[0].mass = 1.0e9;
    nodes[0].force = Vec3::new(0.0, 1.0e9 * g, 0.0);
    nodes[0].vel = Vec3::new(v0, 0.0, 0.0);
    let beams = (0..n - 1)
        .map(|i| {
            plain(
                i as u32,
                i as u32 + 1,
                k,
                c,
                l,
                f32::INFINITY,
                f32::INFINITY,
            )
        })
        .collect();
    system(nodes, beams)
}

/// The wave speed of a struck bar: the time each node's velocity first passes half
/// the piston's, fitted against its distance. The discrete bar's long-wave speed is
/// c₀ = l·√(k/m) (k·l = EA, m/l = ρA: c₀ = √(E/ρ)); its step front is centred
/// there (an Airy front, width ∝ j^{1/3}, around j = c₀t/l).
/// (measured speed, c₀).
pub fn wave_speed(make: Maker, k: f32, m: f32, l: f32) -> Option<(f64, f64)> {
    let n = 121;
    let v0 = 1.0f32;
    let c0 = l as f64 * (k as f64 / m as f64).sqrt();
    let mut sim = make(bar(n, k, 0.0, m, l, v0))?;
    let probes = [30usize, 50, 70, 90];
    let mut arrive = [f64::NAN; 4];
    let horizon = ((100.0 * l as f64 / c0) / H * 1.5) as usize;
    let mut prev = vec![0.0f64; probes.len()];
    for step in 1..=horizon {
        sim.steps(1);
        let v = sim.vel();
        for (q, &j) in probes.iter().enumerate() {
            let x = v[j].x as f64;
            if arrive[q].is_nan() && x >= 0.5 * v0 as f64 {
                let f = (0.5 * v0 as f64 - prev[q]) / (x - prev[q]);
                arrive[q] = (step as f64 - 1.0 + f) * H;
            }
            prev[q] = x;
        }
        if arrive.iter().all(|a| !a.is_nan()) {
            break;
        }
    }
    let pts: Vec<(f64, f64)> = probes
        .iter()
        .zip(arrive)
        .map(|(&j, t)| (t, j as f64 * l as f64))
        .collect();
    Some((slope(&pts), c0))
}

/// The lowest mode of a bar of `n` free nodes between two immovable ends, beams k
/// and c (stiffness-proportional damping C = (c/k)·K): ω₁² = (4k/m)·sin²(π/(2(n+1)))
/// and ζ₁ = (c/2k)·ω₁. With the beams' own ratio ζ_b = c/(2√(k·m/2)) on their pair
/// mode ω_p = √(2k/m) (the BX's definition, `vehicle-authoring.md`), ζ₁ = ζ_b·ω₁/ω_p:
/// a bar's ripple is damped far less than its beams' ratio.
/// Excited in its first mode shape. (ω measured, ω closed form of the step, σ
/// measured, σ of the step, ζ₁ continuous, ζ_b).
pub fn chain_mode(make: Maker, n: usize, k: f32, zeta_b: f64, m: f32, l: f32) -> Option<[f64; 6]> {
    let c = (2.0 * zeta_b * (k as f64 * m as f64 / 2.0).sqrt()) as f32;
    let g = engine_softbody::explicit_beams::GRAVITY;
    let total = n + 2;
    let amp = 0.002f32;
    let nodes: Vec<Node> = (0..total)
        .map(|i| {
            let end = i == 0 || i == total - 1;
            let shape = (std::f32::consts::PI * i as f32 / (n + 1) as f32).sin();
            // Transverse modes of an unstressed bar have no stiffness: excite
            // the axial mode (displacement along the bar).
            // Heavy free ends (1e9 kg) rather than immovable ones: everything
            // falls together, so gravity never sags the bar.
            let mass = if end { 1.0e9 } else { m };
            let mut q = node(Vec3::new(i as f32 * l + amp * shape, 3.0, 0.0), mass, 0);
            q.force = Vec3::new(0.0, mass * g, 0.0);
            q
        })
        .collect();
    let beams = (0..total - 1)
        .map(|i| {
            plain(
                i as u32,
                i as u32 + 1,
                k,
                c,
                l,
                f32::INFINITY,
                f32::INFINITY,
            )
        })
        .collect();
    let mut sim = make(system(nodes, beams))?;
    let mid = total / 2;
    let x0 = mid as f64 * l as f64;
    let (kd, cd, md) = (k as f64, c as f64, m as f64);
    let s1 = (PI / (2.0 * (n as f64 + 1.0))).sin();
    let w1 = (4.0 * kd / md).sqrt() * s1;
    // 12 periods of the lowest mode.
    let steps = (12.0 * 2.0 * PI / w1 / H) as usize;
    let mut y = Vec::with_capacity(steps);
    for _ in 0..steps {
        sim.steps(1);
        let p = sim.pos();
        y.push((p[mid].x - p[0].x) as f64 - x0);
    }
    let x = crossings(&y);
    let period = (x[x.len() - 1] - x[0]) / (x.len() - 1) as f64 * H;
    let pk: Vec<(f64, f64)> = peaks(&y)
        .into_iter()
        .map(|(t, a)| (t * H, a.ln()))
        .collect();
    // The mode as one oscillator of mass m: k₁ = m·ω₁², c₁ = (c/k)·k₁.
    let k1 = md * w1 * w1;
    let c1 = cd / kd * k1;
    Some([
        2.0 * PI / period,
        discrete_omega(k1, c1, md),
        -slope(&pk),
        discrete_sigma(c1, md),
        cd / (2.0 * kd) * w1,
        zeta_b,
    ])
}

/// One node dropped onto the ground at `v` (m/s) from `h0` above it: (deepest
/// penetration, rebound speed, impact speed: its speed the step before it first
/// passed the ground). The rule (`contact_force`): an approaching node's
/// normal force is cancelled and its normal velocity becomes 0.2·v + 0.2·p/h (p
/// its depth); a node at rest or leaving keeps its velocity. Crossing at depth φ·v·h
/// (0 < φ ≤ 1) it leaves at 0.2·φ·v: restitution ≤ 0.2, and its depth never
/// passes v·h.
pub fn ground_drop(make: Maker, v: f32, h0: f32) -> Option<(f64, f64, f64)> {
    let g = engine_softbody::explicit_beams::GRAVITY;
    let mut q = node(Vec3::new(0.0, h0, 0.0), 1.0, 0);
    q.flags &= !NODE_NO_GROUND;
    q.vel = Vec3::new(0.0, -v, 0.0);
    q.force = Vec3::new(0.0, g, 0.0);
    let mut s = system(vec![q], vec![]);
    s.ground = 0.0;
    let mut sim = make(s)?;
    let (mut deepest, mut rebound, mut touched, mut impact) = (0.0f64, 0.0f64, false, v as f64);
    for _ in 0..((h0 / v) / STEP) as usize + 400 {
        sim.steps(1);
        let (p, u) = (sim.pos()[0], sim.vel()[0]);
        if p.y < 0.0 {
            if !touched {
                impact = -u.y as f64;
            }
            touched = true;
            deepest = deepest.max(-p.y as f64);
        }
        if touched && p.y >= 0.0 {
            rebound = u.y as f64;
            break;
        }
    }
    Some((deepest, rebound, impact))
}

/// A stiff plate (a 1 m × 0.8 m rectangle, corners 0..3, a latch node 4 on its
/// far edge, the load point 5, braced into a rigid body by an apex above and
/// below its centre, 6 and 7; 1 kg nodes, beams 3·10⁵ N/m) held up by three
/// supports, each a vertical beam to an immovable anchor 0.2 m below plus a
/// level in-plane tie (no vertical share) in the same break group: hinge A at
/// corner (0, 0), hinge B at (0, b), the latch at (a, b/2). A heavy driver above
/// the load point (p, b/2) pulls it up through a soft spring (k_s) at a steady
/// rate. A rigid plate on three supports is statically determinate: the latch
/// carries F·p/a and each hinge F·(1 − p/a)/2 whatever their stiffness, so the
/// latch (strength S) lets go at F = S·a/p (plus its share of the plate's weight).
/// Returns (pull force when the latch broke, its closed form, every latch beam
/// broken, hinge beams broken).
pub fn latch_release(make: Maker) -> Option<(f64, f64, bool, usize)> {
    let (a, b, p) = (1.0f32, 0.8f32, 0.5f32);
    let (ks, s_latch) = (2.0e4f32, 2000.0f32);
    let g = engine_softbody::explicit_beams::GRAVITY;
    let y = 3.0;
    let c = Vec3::new(0.5 * a, y, 0.5 * b);
    let pos = [
        Vec3::new(0.0, y, 0.0),
        Vec3::new(0.0, y, b),
        Vec3::new(a, y, 0.0),
        Vec3::new(a, y, b),
        Vec3::new(a, y, 0.5 * b), // latch node, 4
        Vec3::new(p, y, 0.5 * b), // load point, 5
        c + Vec3::Y * 0.3,        // apexes 6, 7
        c - Vec3::Y * 0.3,
    ];
    let np = pos.len();
    let mut nodes: Vec<Node> = pos
        .iter()
        .map(|&q| {
            let mut n = node(q, 1.0, 0);
            n.force = Vec3::new(0.0, g, 0.0);
            n
        })
        .collect();
    // Anchors under hinge A, hinge B and the latch (8, 9, 10), and 0.2 m along −x
    // level with them for the in-plane ties (11, 12, 13).
    for q in [pos[0], pos[1], pos[4]] {
        nodes.push(node(q - Vec3::Y * 0.2, 1.0, NODE_IMMOVABLE));
    }
    for q in [pos[0], pos[1], pos[4]] {
        nodes.push(node(q - Vec3::X * 0.2, 1.0, NODE_IMMOVABLE));
    }
    // The driver (14), heavy, 0.1 m above the load point, rising at 0.2 m/s.
    let drv = nodes.len() as u32;
    let mut d = node(pos[5] + Vec3::Y * 0.1, 1.0e9, 0);
    d.vel = Vec3::new(0.0, 0.2, 0.0);
    d.force = Vec3::new(0.0, 1.0e9 * g, 0.0);
    nodes.push(d);
    let stiff = |i: u32, j: u32| {
        let l = pos[i as usize].distance(pos[j as usize]);
        plain(i, j, 3.0e5, 100.0, l, f32::INFINITY, f32::INFINITY)
    };
    let mut beams = Vec::new();
    for (i, j) in [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (2, 4),
        (3, 4),
        (0, 5),
        (1, 5),
        (4, 5),
    ] {
        beams.push(stiff(i, j));
    }
    for apex in [6u32, 7] {
        for i in 0..6u32 {
            beams.push(stiff(i, apex));
        }
    }
    beams.push(stiff(6, 7));
    // Supports: (plate node, anchor, tie anchor, group, strength).
    let first_support = beams.len();
    let base = np as u32;
    for (q, an, dg, grp, st) in [
        (0u32, base, base + 3, 1i32, 1.0e9f32),
        (1, base + 1, base + 4, 2, 1.0e9),
        (4, base + 2, base + 5, 3, s_latch),
    ] {
        let mut v = plain(q, an, 1.0e5, 50.0, 0.2, f32::INFINITY, st);
        v.gate = st;
        v.group = grp;
        let mut w = plain(q, dg, 1.0e5, 50.0, 0.2, f32::INFINITY, 1.0e9);
        w.gate = 1.0e9;
        w.group = grp;
        beams.push(v);
        beams.push(w);
    }
    let spring = beams.len();
    beams.push(plain(5, drv, ks, 50.0, 0.1, f32::INFINITY, f32::INFINITY));
    let mut sim = make(system(nodes, beams))?;
    let mut released = f64::NAN;
    let mut last_f = 0.0f64;
    for _ in 0..4000 {
        // (The driver's speed is reset every step: gravity would slow it.)
        sim.set_vel(drv, Vec3::new(0.0, 0.2, 0.0));
        sim.steps(1);
        let q = sim.pos();
        let bs = sim.beams();
        if bs[first_support + 4].0 {
            released = last_f;
            sim.steps(2);
            break;
        }
        last_f = ks as f64 * ((q[drv as usize] - q[5]).length() - bs[spring].1) as f64;
    }
    let bs = sim.beams();
    let latch_broken = bs[first_support + 4].0 && bs[first_support + 5].0;
    let hinges = (0..4).filter(|&i| bs[first_support + i].0).count();
    // The plate's weight (8 kg at x̄) loads the supports the other way: the latch
    // takes m·g·x̄/a of it, so it lets go at F = (S + m·g·x̄/a)·a/p.
    let xbar = pos.iter().map(|q| q.x as f64).sum::<f64>() / np as f64;
    let w = np as f64 * -g as f64;
    let f_closed = (s_latch as f64 + w * xbar / a as f64) * a as f64 / p as f64;
    Some((released, f_closed, latch_broken, hinges))
}

/// Internal mechanical energy: kinetic energy relative to the centre of mass plus
/// the linear beams' ½k·e² (`explicit_contact::elastic_energy`). Uniform gravity
/// does no work on it (the centre of mass's free fall, and the step's O(h)
/// error of it, cancel out), so in free flight it is conserved by the
/// continuous equations and only oscillates by the step's bounded error.
pub fn internal_energy(v: &ExplicitBeams) -> f64 {
    let mass: f64 = v.nodes.iter().map(|n| n.mass as f64).sum();
    let vc = momentum(v) / mass;
    v.nodes
        .iter()
        .map(|n| 0.5 * n.mass as f64 * (n.vel.as_dvec3() - vc).length_squared())
        .sum::<f64>()
        + engine_softbody::explicit_contact::elastic_energy(v)
}

/// A rigorous upper bound on the highest angular frequency (Gershgorin):
/// ω_max² ≤ max_i Σ_j k_ij·(1/m_i + 1/√(m_i·m_j)) (the authoring's stability rows,
/// `Skeleton::bound_rows`, with damping left out).
pub fn omega_max(v: &ExplicitBeams) -> f64 {
    let mut row = vec![0.0f64; v.nodes.len()];
    for b in v.beams.iter().filter(|b| !b.broken && b.k > 0.0) {
        let (ma, mb) = (
            v.nodes[b.a as usize].mass as f64,
            v.nodes[b.b as usize].mass as f64,
        );
        let k = b.k as f64;
        row[b.a as usize] += k * (1.0 / ma + 1.0 / (ma * mb).sqrt());
        row[b.b as usize] += k * (1.0 / mb + 1.0 / (ma * mb).sqrt());
    }
    row.into_iter().fold(0.0, f64::max).sqrt()
}
