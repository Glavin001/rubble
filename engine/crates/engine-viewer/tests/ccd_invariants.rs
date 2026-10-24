//! CCD invariants (engine/docs/ccd.md, "Invariants"), checked step by step over
//! scenario families × speeds × mass ratios, in the fracture lab's own step plan
//! (the browser lab replays the same plan through wasm).
//!
//! Every scenario is built in code (no generated assets) and run one tick at a
//! time with the bodies read back after each. The invariants:
//!
//! - I1 no tunnelling: a CCD ball's centre path in a tick never meets a body
//!   that did not move that tick (exact segment-hull distance);
//! - I2 bounded sink: at the end of a tick a CCD ball overlaps a still body by
//!   at most half its radius;
//! - I3 no stall: a ball that keeps its speed through a tick also moves by it
//!   (the post-solve pull-back keeping the velocity was the stop motion);
//! - I4 momentum: with no gravity and nothing static, total momentum is kept;
//! - I5 energy: total kinetic energy never grows (no gravity);
//! - I6 sleep invariance: targets that start asleep give the ball the same
//!   outcome as targets that start awake;
//! - I7 reference: the ball's outcome at the lab's tick matches the same scene
//!   at a 16× finer tick;
//! - I8 determinism: the same run twice is bit-identical.
//!
//! Each scenario also runs with CCD off where that makes the check sensitive
//! (the controls must fail I1). Prints a table; fails listing every violation.
//!
//!   cargo test --release -p engine-viewer --test ccd_invariants -- --nocapture
//!   CCD_INV_ONLY=fence cargo test ...   (one family)

use engine_fracture::scene::*;
use engine_schema::convex::HullArena;
use engine_viewer::fracture::FractureSim;
use metal::Device;

#[path = "support/phase.rs"]
mod phase;

type V = [f64; 3];
fn sub(a: V, b: V) -> V {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V, b: V) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn len(a: V) -> f64 {
    dot(a, a).sqrt()
}
fn vec3(s: &[f32], b: usize, k: usize) -> V {
    [0, 1, 2].map(|i| s[b * 28 + k + i] as f64)
}

// ---------------------------------------------------------------- scenes ----

struct World {
    bodies: Vec<Body>,
    colliders: Vec<Collider>,
    asleep: bool,
}

impl World {
    /// Body 0: the ground plane, or (`ground` false) a static anchor far away.
    fn new(ground: bool, asleep: bool) -> Self {
        let mut w = World { bodies: vec![Body::fixed([0.0, if ground { 0.0 } else { -5000.0 }, 0.0])], colliders: vec![], asleep };
        if ground {
            w.colliders.push(Collider::ground());
        } else {
            w.colliders.push(Collider::new(0, SHAPE_BOX, 0.0, [0.5, 0.5, 0.5, 0.0]));
        }
        w
    }
    fn dynamic_box(&mut self, p: V, h: V, mass: f64) -> u32 {
        let id = self.bodies.len() as u32;
        let sq = |x: f64| x * x;
        self.bodies.push(Body {
            p,
            inv_mass: 1.0 / mass,
            inv_i: [
                3.0 / (mass * (sq(h[1]) + sq(h[2]))),
                3.0 / (mass * (sq(h[0]) + sq(h[2]))),
                3.0 / (mass * (sq(h[0]) + sq(h[1]))),
            ],
            flags: FLAG_DYNAMIC | if self.asleep { FLAG_SLEEPING } else { FLAG_AWAKE },
            ..Body::fixed(p)
        });
        self.colliders.push(Collider::new(id, SHAPE_BOX, 0.0, [h[0], h[1], h[2], 0.0]));
        id
    }
    fn static_box(&mut self, p: V, h: V) {
        let id = self.bodies.len() as u32;
        self.bodies.push(Body::fixed(p));
        self.colliders.push(Collider::new(id, SHAPE_BOX, 0.0, [h[0], h[1], h[2], 0.0]));
    }
    fn scene(self, gravity: f64, ball_r: f64, ball_m: f64, dt_div: f64) -> Scene {
        let params = SceneParams { gravity_y: gravity as f32, dt: (1.0 / 120.0 / dt_div) as f32, ..SceneParams::lab(LAB_FLAGS) };
        let mut s = Scene::from_parts(self.bodies, self.colliders, &HullArena::new(), params);
        add_projectiles(&mut s, ball_r, 3.0);
        s.projectile_mass = ball_m;
        s.projectile_speed = Some(1.0);
        s.projectile_arc = false;
        s.extent = 20.0;
        s.extent_x = 20.0;
        s.extent_z = 20.0;
        s.center = [0.0, 1.0, 0.0];
        // Bricks touch several neighbours (as the lab's procedural houses).
        s.capacity.candidates_per_collider = 16;
        s.capacity.contacts_per_collider = 32;
        s
    }
}

#[derive(Clone, Copy)]
struct Shot {
    tick: usize,
    origin: V,
    dir: V,
    speed: f64,
}

#[derive(Clone)]
struct Case {
    family: &'static str,
    label: String,
    /// Builds the world (asleep: targets start asleep) at a tick divisor.
    build: fn(bool, f64, &Params) -> Scene,
    params: Params,
    shots: Vec<Shot>,
    ticks: usize,
    /// Which invariants apply.
    zero_g: bool,
    sleep_check: bool,
    reference: bool,
}

#[derive(Clone, Copy)]
struct Params {
    speed: f64,
    ball_m: f64,
    piece_m: f64,
    ball_r: f64,
}

// Picket fence: thin light pickets standing on the ground in a row along x;
// the ball flies along the row at picket height, clipping them one by one.
fn fence(asleep: bool, div: f64, p: &Params) -> Scene {
    let mut w = World::new(true, asleep);
    for k in 0..16 {
        w.dynamic_box([2.0 + 0.5 * k as f64, 0.6, 0.0], [0.03, 0.6, 0.15], p.piece_m);
    }
    w.scene(-9.81, p.ball_r, p.ball_m, div)
}

// Brick wall: 8 × 5 bricks in running bond, standing on the ground, facing -x.
fn wall(asleep: bool, div: f64, p: &Params) -> Scene {
    let mut w = World::new(true, asleep);
    for row in 0..5 {
        for k in 0..8 {
            let z = -1.6 + 0.4 * k as f64 + if row % 2 == 1 { 0.2 } else { 0.0 };
            w.dynamic_box([4.0, 0.1 + 0.2 * row as f64, z], [0.1, 0.1, 0.2], p.piece_m);
        }
    }
    w.scene(-9.81, p.ball_r, p.ball_m, div)
}

// Zero-g lattice: 4 × 4 × 4 free boxes, nothing static near.
fn lattice(asleep: bool, div: f64, p: &Params) -> Scene {
    let mut w = World::new(false, asleep);
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                w.dynamic_box([3.0 + 0.6 * i as f64, -0.9 + 0.6 * j as f64, -0.9 + 0.6 * k as f64], [0.15, 0.15, 0.15], p.piece_m);
            }
        }
    }
    w.scene(0.0, p.ball_r, p.ball_m, div)
}

// Thin plates: six 2 cm plates stacked 0.4 m apart (free, zero-g), the ball
// shot down through them.
fn plates(asleep: bool, div: f64, p: &Params) -> Scene {
    let mut w = World::new(false, asleep);
    for k in 0..6 {
        w.dynamic_box([0.0, 1.0 - 0.4 * k as f64, 0.0], [1.0, 0.01, 1.0], p.piece_m);
    }
    w.scene(0.0, p.ball_r, p.ball_m, div)
}

// Pinball: a closed room of thin static walls, zero-g; the ball rattles inside.
fn pinball(asleep: bool, div: f64, p: &Params) -> Scene {
    let mut w = World::new(false, asleep);
    let (r, t) = (3.0, 0.02);
    for a in 0..3 {
        for s in [-1.0, 1.0] {
            let mut c = [0.0; 3];
            c[a] = s * (r + t);
            let mut h = [r + 2.0 * t; 3];
            h[a] = t;
            w.static_box(c, h);
        }
    }
    w.scene(0.0, p.ball_r, p.ball_m, div)
}

fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    let only = std::env::var("CCD_INV_ONLY").ok();
    let mut add = |c: Case| {
        if only.as_deref().is_none_or(|o| o.split(',').any(|f| f == c.family)) {
            out.push(c);
        }
    };
    for &speed in &[30.0, 120.0, 400.0, 1200.0] {
        for &(ball_m, piece_m) in &[(10.0, 10.0), (1000.0, 1.0), (200000.0, 1.0)] {
            let p = Params { speed, ball_m, piece_m, ball_r: 0.3 };
            let tag = format!("{speed} m/s, ball {ball_m} kg / piece {piece_m} kg");
            let ticks = 120;
            add(Case { family: "fence", label: tag.clone(), build: fence, params: p, shots: vec![Shot { tick: 0, origin: [-2.0, 0.6, 0.12], dir: [1.0, 0.0, 0.0], speed }], ticks, zero_g: false, sleep_check: true, reference: speed <= 400.0 });
            add(Case { family: "wall", label: tag.clone(), build: wall, params: p, shots: vec![Shot { tick: 0, origin: [-2.0, 0.5, 0.05], dir: [1.0, 0.0, 0.0], speed }], ticks, zero_g: false, sleep_check: true, reference: speed <= 400.0 });
            add(Case { family: "lattice", label: tag.clone(), build: lattice, params: p, shots: vec![Shot { tick: 0, origin: [-2.0, 0.05, 0.03], dir: [1.0, 0.0, 0.0], speed }, Shot { tick: 3, origin: [4.0, 4.0, 0.33], dir: [0.0, -1.0, 0.0], speed }], ticks, zero_g: true, sleep_check: true, reference: speed <= 400.0 });
            add(Case { family: "plates", label: tag.clone(), build: plates, params: p, shots: vec![Shot { tick: 0, origin: [0.07, 4.0, 0.05], dir: [0.0, -1.0, 0.0], speed }], ticks, zero_g: true, sleep_check: true, reference: speed <= 400.0 });
        }
        let p = Params { speed, ball_m: 10.0, piece_m: 1.0, ball_r: 0.3 };
        add(Case { family: "pinball", label: format!("{speed} m/s"), build: pinball, params: p, shots: vec![Shot { tick: 0, origin: [0.0, 0.0, 0.0], dir: { let d = [1.0, 0.37, 0.61]; let l = len(d); d.map(|x| x / l) }, speed }], ticks: 600, zero_g: true, sleep_check: false, reference: false });
    }
    out
}

// ---------------------------------------------------------------- runs ----

struct Run {
    snaps: Vec<Vec<f32>>,
    /// (tick after which the shot is in flight, body)
    launches: Vec<(usize, usize)>,
    clamped_ticks: Vec<usize>,
}

fn run(device: &Device, scene: &Scene, shots: &[Shot], ticks: usize, ccd: bool, div: usize) -> Run {
    let queue = device.new_command_queue();
    let mut sim = FractureSim::new(device, &queue, scene).unwrap();
    sim.plan.set_projectile_ccd(ccd);
    let n = scene.bodies.len();
    let mut snaps = vec![sim.snapshot(n).unwrap()];
    let mut launches = Vec::new();
    for t in 0..ticks * div {
        for s in shots.iter().filter(|s| s.tick * div == t) {
            sim.plan.set_projectile_speed_scale(s.speed / sim.plan.shot_speed()).unwrap();
            let id = sim.launch(s.origin, s.dir).unwrap() as usize;
            launches.push((snaps.len() - 1, id));
            // The pre-tick pose of the ball is its launch pose.
            *snaps.last_mut().unwrap() = sim.snapshot(n).unwrap();
        }
        sim.step(1).unwrap();
        assert_eq!(sim.last.overflow, 0, "capacity overflow");
        snaps.push(sim.snapshot(n).unwrap());
    }
    Run { snaps, launches, clamped_ticks: vec![] }
}

// ------------------------------------------------------------- checks ----

#[derive(Default)]
struct Report {
    tunnels: Vec<String>,
    max_sink: f64,
    stalls: Vec<String>,
    momentum_drift: f64,
    energy_gain: f64,
}

fn check(scene: &Scene, r: &Run, ball_r: f64, zero_g: bool) -> Report {
    let pieces = phase::pieces(scene);
    let mut rep = Report::default();
    let dt = scene.params.dt as f64;
    let last_launch = r.launches.iter().map(|l| l.0).max().unwrap_or(0);
    let dynamic: Vec<usize> = (1..scene.bodies.len()).filter(|&b| scene.bodies[b].inv_mass > 0.0 || b >= scene.projectile_start as usize).collect();
    let mass = |s: &[f32], b: usize| {
        let im = s[b * 28 + 3] as f64;
        if im > 0.0 { 1.0 / im } else { 0.0 }
    };
    let momentum = |s: &[f32]| dynamic.iter().fold([0.0; 3], |acc, &b| { let v = vec3(s, b, 12); let m = mass(s, b); [acc[0] + m * v[0], acc[1] + m * v[1], acc[2] + m * v[2]] });
    let energy = |s: &[f32]| dynamic.iter().map(|&b| { let v = vec3(s, b, 12); 0.5 * mass(s, b) * dot(v, v) }).sum::<f64>();
    let (p0, e0) = (momentum(&r.snaps[last_launch]), energy(&r.snaps[last_launch]));
    for t in 1..r.snaps.len() {
        let (prev, now) = (&r.snaps[t - 1], &r.snaps[t]);
        for &(from, ball) in &r.launches {
            if t <= from {
                continue;
            }
            if let Some(f) = phase::crossing(scene, &pieces, prev, now, ball) {
                rep.tunnels.push(format!("tick {}: {f}", t - 1));
            }
            // I3: kept its speed (within 20%) but moved under half of it.
            let (x0, x1) = (vec3(prev, ball, 0), vec3(now, ball, 0));
            let (v0, v1) = (len(vec3(prev, ball, 12)), len(vec3(now, ball, 12)));
            let dx = len(sub(x1, x0));
            if v0 > 5.0 && v1 > 0.8 * v0 && dx < 0.5 * v1 * dt {
                rep.stalls.push(format!("tick {}: |v| {v0:.1}->{v1:.1} m/s, moved {dx:.3} m of {:.3}", t - 1, v1 * dt));
            }
            // I2: sink into still bodies at the end of the tick.
            for p in &pieces {
                let (y0, q0) = phase::pose(prev, p.body);
                let (y1, q1) = phase::pose(now, p.body);
                if len(sub(y1, y0)) > 1e-3 || len(sub(y1, x1)) > p.bound + ball_r || (q0[3] - q1[3]).abs() > 1e-6 {
                    continue;
                }
                let d = phase::segment_hull_distance(x1, x1, &phase::piece_vertices(scene, &scene.colliders[p.collider], y1, q1));
                rep.max_sink = rep.max_sink.max(ball_r - d);
            }
        }
        if zero_g && t > last_launch {
            let p = momentum(now);
            rep.momentum_drift = rep.momentum_drift.max(len(sub(p, p0)) / len(p0).max(1e-9));
            rep.energy_gain = rep.energy_gain.max(energy(now) / e0.max(1e-9) - 1.0);
        }
    }
    rep
}

/// The ball of the first shot: final velocity.
fn ball_outcome(r: &Run) -> V {
    let ball = r.launches[0].1;
    vec3(r.snaps.last().unwrap(), ball, 12)
}

#[test]
#[ignore = "invariant harness: written, not yet run and triaged (expected to report violations above ~300 m/s)"]
fn ccd_invariants() {
    let device = Device::system_default().expect("Metal device");
    let mut failures = Vec::new();
    let mut controls_ok = 0;
    let mut controls = 0;
    for c in cases() {
        let p = c.params;
        let scene = (c.build)(true, 1.0, &p);
        let r = run(&device, &scene, &c.shots, c.ticks, true, 1);
        let rep = check(&scene, &r, p.ball_r, c.zero_g);
        let mut notes = Vec::new();
        let mut fail = |what: String| failures.push(format!("{} [{}]: {what}", c.family, c.label));
        if !rep.tunnels.is_empty() {
            fail(format!("I1 tunnelled {}x, first {}", rep.tunnels.len(), rep.tunnels[0]));
        }
        if rep.max_sink > 0.5 * p.ball_r {
            fail(format!("I2 sank {:.3} m into a still body (bound {:.3})", rep.max_sink, 0.5 * p.ball_r));
        }
        if !rep.stalls.is_empty() {
            fail(format!("I3 stalled {} ticks, first {}", rep.stalls.len(), rep.stalls[0]));
        }
        if c.zero_g {
            if rep.momentum_drift > 0.02 {
                fail(format!("I4 momentum drifted {:.1}%", 100.0 * rep.momentum_drift));
            }
            if rep.energy_gain > 0.01 {
                fail(format!("I5 kinetic energy grew {:.1}%", 100.0 * rep.energy_gain));
            }
        }
        let v = ball_outcome(&r);
        if c.sleep_check {
            let awake = (c.build)(false, 1.0, &p);
            let ra = run(&device, &awake, &c.shots, c.ticks, true, 1);
            let va = ball_outcome(&ra);
            let dv = len(sub(v, va)) / p.speed;
            notes.push(format!("asleep vs awake Δv {:.0}%", 100.0 * dv));
            if dv > 0.1 {
                fail(format!("I6 sleeping targets changed the ball's outcome by {:.0}% of its speed ({v:.1?} vs awake {va:.1?})", 100.0 * dv));
            }
        }
        if c.reference {
            let fine = (c.build)(true, 16.0, &p);
            let rf = run(&device, &fine, &c.shots, c.ticks, true, 16);
            let vf = ball_outcome(&rf);
            let dv = len(sub(v, vf)) / p.speed;
            notes.push(format!("vs 16x finer tick Δv {:.0}%", 100.0 * dv));
            if dv > 0.15 {
                fail(format!("I7 outcome differs from a 16x finer tick by {:.0}% of its speed ({v:.1?} vs {vf:.1?})", 100.0 * dv));
            }
        }
        // I8: determinism.
        let r2 = run(&device, &scene, &c.shots, c.ticks, true, 1);
        if r2.snaps.last() != r.snaps.last() {
            fail("I8 two runs differ".into());
        }
        // Control: without CCD, fast cases should tunnel (the detector is sensitive).
        if p.speed >= 400.0 && c.family != "pinball" {
            controls += 1;
            let rc = run(&device, &scene, &c.shots, c.ticks, false, 1);
            if !check(&scene, &rc, p.ball_r, c.zero_g).tunnels.is_empty() {
                controls_ok += 1;
            }
        }
        println!(
            "{:<8} {:<44} tunnels {:>2}  sink {:>6.3}  stalls {:>3}  {}{}",
            c.family,
            c.label,
            rep.tunnels.len(),
            rep.max_sink.max(0.0),
            rep.stalls.len(),
            if c.zero_g { format!("Δp {:>5.1}%  ΔKE {:>+6.1}%  ", 100.0 * rep.momentum_drift, 100.0 * rep.energy_gain) } else { String::new() },
            notes.join("  ")
        );
    }
    println!("controls (CCD off, ≥400 m/s): {controls_ok} of {controls} tunnel");
    assert!(failures.is_empty(), "{} CCD invariant violations:\n{}", failures.len(), failures.join("\n"));
}
