//! Parametric scene construction and the benchmark scene catalogue.
//!
//! The catalogue is the single source of every benchmark scene: `stress-ref
//! gen-scenes` writes it to `scenes/*.json`, the oracle exporters read those files,
//! and the tests rebuild the same scenes in memory.

use std::collections::{BTreeMap, HashMap};

use crate::material::Material;
use crate::math::Vec3;
use crate::scene::*;

pub fn new_scene(name: &str, description: &str) -> Scene {
    Scene {
        format: SCENE_FORMAT.into(),
        name: name.into(),
        description: description.into(),
        benchmark: None,
        gravity: [0.0, 0.0, -9.81],
        materials: BTreeMap::new(),
        bodies: Vec::new(),
        impactors: Vec::new(),
        ground: None,
        loads: Vec::new(),
        events: Vec::new(),
        sim: SimDesc::default(),
        probes: Vec::new(),
        metrics: Vec::new(),
        oracle: BTreeMap::new(),
        blast: None,
    }
}

pub fn chunk(center: Vec3, half: Vec3, material: &str) -> ChunkDesc {
    ChunkDesc {
        center: center.to_array(),
        half_extents: half.to_array(),
        orientation: [1.0, 0.0, 0.0, 0.0],
        material: material.into(),
        support: Support::None,
        groups: Vec::new(),
        level: 0,
        parent: None,
        hull: None,
        compound: None,
    }
}

/// A regular grid of equal boxes; index = `i + nx (j + ny k)`.
pub fn grid(min_corner: Vec3, counts: [usize; 3], size: Vec3, material: &str) -> Vec<ChunkDesc> {
    let mut out = Vec::with_capacity(counts.iter().product());
    for k in 0..counts[2] {
        for j in 0..counts[1] {
            for i in 0..counts[0] {
                let c = min_corner + Vec3::new((i as f64 + 0.5) * size.x, (j as f64 + 0.5) * size.y, (k as f64 + 0.5) * size.z);
                out.push(chunk(c, size * 0.5, material));
            }
        }
    }
    out
}

pub fn grid_index(counts: [usize; 3], i: usize, j: usize, k: usize) -> usize {
    i + counts[0] * (j + counts[1] * k)
}

/// Bonds between every pair of face-touching axis-aligned boxes on the same level.
/// The patch is the overlap rectangle; `joint(a, b)` names the joint material.
pub fn auto_bonds(chunks: &[ChunkDesc], joint: impl Fn(&ChunkDesc, &ChunkDesc) -> String) -> Vec<BondDesc> {
    let max_h = chunks.iter().flat_map(|c| c.half_extents).fold(0.0f64, f64::max);
    // Wider than any centre spacing of touching boxes, so neighbours are at most one cell apart.
    let cell = 2.5 * max_h;
    let key = |p: [f64; 3]| -> (i64, i64, i64) {
        ((p[0] / cell).floor() as i64, (p[1] / cell).floor() as i64, (p[2] / cell).floor() as i64)
    };
    let mut hash: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (i, c) in chunks.iter().enumerate() {
        hash.entry(key(c.center)).or_default().push(i);
    }
    let scale = max_h.max(1e-9);
    let tol = 1e-9 * scale;
    let mut bonds = Vec::new();
    for (i, a) in chunks.iter().enumerate() {
        let (kx, ky, kz) = key(a.center);
        let mut cand = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(v) = hash.get(&(kx + dx, ky + dy, kz + dz)) {
                        cand.extend(v.iter().copied().filter(|&j| j > i));
                    }
                }
            }
        }
        cand.sort_unstable();
        for j in cand {
            let b = &chunks[j];
            if a.level != b.level {
                continue;
            }
            for axis in 0..3 {
                let gap = (b.center[axis] - a.center[axis]).abs() - (a.half_extents[axis] + b.half_extents[axis]);
                if gap.abs() > tol {
                    continue;
                }
                let (o1, o2) = ((axis + 1) % 3, (axis + 2) % 3);
                let lo = |o: usize| (a.center[o] - a.half_extents[o]).max(b.center[o] - b.half_extents[o]);
                let hi = |o: usize| (a.center[o] + a.half_extents[o]).min(b.center[o] + b.half_extents[o]);
                let (w1, w2) = (hi(o1) - lo(o1), hi(o2) - lo(o2));
                if w1 <= tol || w2 <= tol {
                    continue;
                }
                let sign = if b.center[axis] > a.center[axis] { 1.0 } else { -1.0 };
                let mut normal = [0.0; 3];
                normal[axis] = sign;
                let mut tangent = [0.0; 3];
                tangent[o1] = 1.0;
                let mut centroid = [0.0; 3];
                centroid[axis] = a.center[axis] + sign * a.half_extents[axis];
                centroid[o1] = 0.5 * (lo(o1) + hi(o1));
                centroid[o2] = 0.5 * (lo(o2) + hi(o2));
                bonds.push(BondDesc {
                    a: i,
                    b: j,
                    centroid,
                    normal,
                    tangent,
                    area: w1 * w2,
                    width: [w1, w2],
                    material: joint(a, b),
                    rebar: None,
                    buckling_length: None,
                    level: a.level,
                    groups: Vec::new(),
                    derived: None,
                });
            }
        }
    }
    bonds
}

pub fn body(name: &str, chunks: Vec<ChunkDesc>, bonds: Vec<BondDesc>) -> BodyDesc {
    BodyDesc {
        name: name.into(),
        position: [0.0; 3],
        orientation: [1.0, 0.0, 0.0, 0.0],
        linear_velocity: [0.0; 3],
        angular_velocity: [0.0; 3],
        chunks,
        bonds,
    }
}

pub fn probe(name: &str, kind: ProbeKind) -> ProbeDesc {
    ProbeDesc { name: name.into(), kind }
}

pub fn metric(name: &str, kind: MetricKind, tolerance: Tolerance, expected: Option<serde_json::Value>, oracles: &[&str]) -> MetricDesc {
    MetricDesc { name: name.into(), kind, tolerance, oracle_tolerance: None, oracles: oracles.iter().map(|s| s.to_string()).collect(), expected }
}

fn num(x: f64) -> Option<serde_json::Value> {
    Some(serde_json::json!(x))
}

// ======================================================================== benchmark 1

/// Benchmark 1, tension: two cubes, one held; the other is pulled by a slow force ramp
/// through the bond centroid. Analytic: failure force `f_t A`, dissipated `G_f A`.
pub fn bond_tension() -> Scene {
    let m = Material::analytic_test();
    let h = 0.05;
    let mut s = new_scene("b1_bond_tension", "Single bond pulled in tension to failure (force ramp).");
    s.benchmark = Some(1);
    s.gravity = [0.0; 3];
    let mut chunks = vec![chunk(Vec3::ZERO, Vec3::splat(h), "rock"), chunk(Vec3::new(2.0 * h, 0.0, 0.0), Vec3::splat(h), "rock")];
    chunks[0].support = Support::Fixed;
    let bonds = auto_bonds(&chunks, |_, _| "rock".into());
    let area = 4.0 * h * h;
    s.materials.insert("rock".into(), m.clone());
    s.bodies.push(body("pair", chunks, bonds));
    s.loads.push(LoadDesc::PointForce {
        body: "pair".into(),
        chunk: 1,
        direction: [1.0, 0.0, 0.0],
        magnitude: TimeFunction::Ramp { t0: 0.0, t1: 0.5, value: 2.0 * m.tensile_strength * area },
        point: None,
    });
    s.sim.duration = 0.6;
    s.probes.push(probe(
        "axial",
        ProbeKind::SectionForce { body: "pair".into(), point: [h, 0.0, 0.0], normal: [1.0, 0.0, 0.0], region: None, component: SectionComponent::Normal },
    ));
    // Exact for the bond model (1%: probe sampling); continuum oracles within 5%.
    s.metrics.push(
        metric("failure_force", MetricKind::ProbeMax { probe: "axial".into() }, Tolerance::Relative(0.01), num(m.tensile_strength * area), &[])
            .with_oracle_tolerance(Tolerance::Relative(0.05)),
    );
    s.metrics.push(
        metric(
            "dissipated_energy",
            MetricKind::Value { key: "bond_dissipation".into() },
            Tolerance::Relative(0.01),
            num(m.fracture_energy.tension * area),
            &[],
        )
        .with_oracle_tolerance(Tolerance::Relative(0.05)),
    );
    s.metrics.push(metric("broke", MetricKind::Flag { flag: "any_bond_broken".into() }, Tolerance::Exact, Some(serde_json::json!(true)), &[]));
    s
}

/// Two cubes of size `2h`, the first held, the second pulled along +x through the
/// bond centroid by `magnitude(t)`; probe `axial` is the bond's normal force.
pub fn bond_pull(name: &str, m: Material, h: f64, magnitude: TimeFunction, duration: f64) -> Scene {
    let mut s = new_scene(name, "Single bond pulled along its normal.");
    s.gravity = [0.0; 3];
    let mut chunks = vec![chunk(Vec3::ZERO, Vec3::splat(h), "rock"), chunk(Vec3::new(2.0 * h, 0.0, 0.0), Vec3::splat(h), "rock")];
    chunks[0].support = Support::Fixed;
    let bonds = auto_bonds(&chunks, |_, _| "rock".into());
    s.materials.insert("rock".into(), m);
    s.bodies.push(body("pair", chunks, bonds));
    s.loads.push(LoadDesc::PointForce { body: "pair".into(), chunk: 1, direction: [1.0, 0.0, 0.0], magnitude, point: None });
    s.sim.duration = duration;
    s.probes.push(probe(
        "axial",
        ProbeKind::SectionForce { body: "pair".into(), point: [h, 0.0, 0.0], normal: [1.0, 0.0, 0.0], region: None, component: SectionComponent::Normal },
    ));
    s
}

/// Benchmark 1, shear: compressive preload `sigma_n`, then a shear force ramp through
/// the bond centroid. Analytic Mohr-Coulomb failure force `(c + mu sigma_n) A`.
pub fn bond_shear() -> Scene {
    let m = Material::analytic_test();
    let h = 0.05;
    let area = 4.0 * h * h;
    let sigma_n = 2.0e6;
    let mut s = new_scene("b1_bond_shear", "Single bond under compressive preload, sheared to failure (Mohr-Coulomb).");
    s.benchmark = Some(1);
    s.gravity = [0.0; 3];
    let mut chunks = vec![chunk(Vec3::ZERO, Vec3::splat(h), "rock"), chunk(Vec3::new(2.0 * h, 0.0, 0.0), Vec3::splat(h), "rock")];
    chunks[0].support = Support::Fixed;
    let bonds = auto_bonds(&chunks, |_, _| "rock".into());
    s.materials.insert("rock".into(), m.clone());
    s.bodies.push(body("pair", chunks, bonds));
    s.loads.push(LoadDesc::PointForce {
        body: "pair".into(),
        chunk: 1,
        direction: [-1.0, 0.0, 0.0],
        magnitude: TimeFunction::Constant { value: sigma_n * area },
        point: None,
    });
    let v_fail = (m.cohesion + m.friction * sigma_n) * area;
    s.loads.push(LoadDesc::PointForce {
        body: "pair".into(),
        chunk: 1,
        direction: [0.0, 1.0, 0.0],
        magnitude: TimeFunction::Ramp { t0: 0.0, t1: 0.5, value: 1.6 * v_fail },
        point: Some([h, 0.0, 0.0]),
    });
    s.sim.duration = 0.45;
    s.probes.push(probe(
        "shear",
        ProbeKind::SectionForce {
            body: "pair".into(),
            point: [h, 0.0, 0.0],
            normal: [1.0, 0.0, 0.0],
            region: None,
            component: SectionComponent::Force([0.0, -1.0, 0.0]),
        },
    ));
    s.metrics.push(metric("failure_force", MetricKind::ProbeMax { probe: "shear".into() }, Tolerance::Relative(0.01), num(v_fail), &[]));
    s.metrics.push(metric("broke", MetricKind::Flag { flag: "any_bond_broken".into() }, Tolerance::Exact, Some(serde_json::json!(true)), &[]));
    s
}

// ======================================================================== benchmark 2

/// Cantilever geometry shared by the scene and its analytic expectations.
pub struct Cantilever {
    pub n: usize,
    pub h: f64,
    pub load: f64,
}

impl Cantilever {
    pub fn second_moment(&self) -> f64 {
        self.h.powi(4) / 12.0
    }
    /// Load point distance from the clamp (anchored chunk centre).
    pub fn arm(&self) -> f64 {
        self.n as f64 * self.h
    }
    /// Euler-Bernoulli span for the free vibration: clamp to the free end.
    pub fn span(&self) -> f64 {
        (self.n as f64 + 0.5) * self.h
    }
}

/// Benchmark 2: a chunked cantilever (N chunks plus a clamped root chunk) under a tip
/// load, statically (prestress), then released into free vibration.
pub fn cantilever(n: usize, name: &str) -> Scene {
    let m = Material::analytic_test();
    let cfg = Cantilever { n, h: 0.1 * 20.0 / n as f64, load: 30.0 };
    let h = cfg.h;
    let mut s = new_scene(name, "Chunked cantilever: static tip load, then free vibration (Euler-Bernoulli).");
    s.benchmark = Some(2);
    s.gravity = [0.0; 3];
    s.materials.insert("beam".into(), m.clone());
    let mut chunks = grid(Vec3::new(-0.5 * h, -0.05, -0.05), [n + 1, 1, 1], Vec3::new(h, 0.1, 0.1), "beam");
    chunks[0].support = Support::Fixed;
    // Square section 0.1 x 0.1 regardless of N (the chunk length changes).
    let bonds = auto_bonds(&chunks, |_, _| "beam".into());
    s.bodies.push(body("beam", chunks, bonds));
    let release = 0.05;
    s.loads.push(LoadDesc::PointForce {
        body: "beam".into(),
        chunk: n,
        direction: [0.0, 0.0, -1.0],
        magnitude: TimeFunction::Table { points: vec![[0.0, cfg.load], [release, cfg.load], [release + 1e-9, 0.0]] },
        point: None,
    });
    s.sim.duration = 2.0;
    s.sim.fracture = false;
    s.sim.sample_interval = Some(1e-3);
    s.probes.push(probe("tip", ProbeKind::ChunkDisplacement { body: "beam".into(), chunk: n, axis: [0.0, 0.0, 1.0] }));
    s.probes.push(probe(
        "root_moment",
        ProbeKind::SectionForce {
            body: "beam".into(),
            point: [0.5 * h, 0.0, 0.0],
            normal: [1.0, 0.0, 0.0],
            region: None,
            component: SectionComponent::Moment([0.0, 1.0, 0.0]),
        },
    ));
    let i = 0.1f64.powi(4) / 12.0;
    let area = 0.01;
    let e = m.youngs_modulus;
    let a = cfg.arm();
    let tip = -cfg.load * a.powi(3) / (3.0 * e * i);
    let root_moment = -cfg.load * (a - 0.5 * h);
    let l = cfg.span();
    let f1 = 1.875_104_07f64.powi(2) / (2.0 * std::f64::consts::PI) * (e * i / (m.density * area * l.powi(4))).sqrt();
    // Euler-Bernoulli (and OpenSees' elastic beam elements) within 1%: the chunk model
    // adds the bonds' shear flexibility and an O(h^2) midpoint error of the rotations.
    s.metrics.push(metric("tip_deflection", MetricKind::ProbeAt { probe: "tip".into(), time: 0.0 }, Tolerance::Relative(0.01), num(tip), &["opensees"]));
    // The exact discrete model (rigid chunks, bond springs EI/h and GA/h at the joints):
    // tip = P (a^3/3 - a h^2/12) / EI + P a / (G A), at the static solver's tolerance.
    let g = m.shear_modulus();
    let discrete = -cfg.load * (a.powi(3) / 3.0 - a * h * h / 12.0) / (e * i) - cfg.load * a / (g * area);
    s.metrics.push(metric("tip_deflection_discrete", MetricKind::ProbeAt { probe: "tip".into(), time: 0.0 }, Tolerance::Relative(1e-6), num(discrete), &["none"]));
    s.metrics.push(
        metric("root_moment", MetricKind::ProbeAt { probe: "root_moment".into(), time: 0.0 }, Tolerance::Relative(1e-6), num(root_moment), &["opensees"])
            .with_oracle_tolerance(Tolerance::Relative(0.01)),
    );
    s.metrics.push(metric(
        "first_frequency",
        MetricKind::ProbeFrequency { probe: "tip".into(), after: release + 0.01 },
        Tolerance::Relative(0.01),
        num(f1),
        &["opensees"],
    ));
    s
}

// ======================================================================== benchmark 3

/// Benchmark 3: a free bar struck by a half-sine force pulse; the compressive wave
/// travels at `sqrt(E_eff / rho)` and reflects from the free end as tension, where the
/// particle velocity doubles.
pub fn bar_wave(stiffness_scale: f64, name: &str) -> Scene {
    let m = Material::analytic_test();
    let n = 100;
    let h = 0.05;
    let area = 0.05 * 0.05;
    let mut s = new_scene(name, "Free bar, half-sine end pulse: wave speed and free-end reflection.");
    s.benchmark = Some(3);
    s.gravity = [0.0; 3];
    s.materials.insert("bar".into(), m.clone());
    let chunks = grid(Vec3::new(0.0, -0.025, -0.025), [n, 1, 1], Vec3::new(h, 0.05, 0.05), "bar");
    let bonds = auto_bonds(&chunks, |_, _| "bar".into());
    s.bodies.push(body("bar", chunks, bonds));
    let c = (m.youngs_modulus * stiffness_scale / m.density).sqrt();
    // Pulse long against the chunk size (low dispersion), short against the bar.
    let duration = 30.0 * h / c;
    let peak_force = 1000.0;
    s.loads.push(LoadDesc::PointForce {
        body: "bar".into(),
        chunk: 0,
        direction: [1.0, 0.0, 0.0],
        magnitude: TimeFunction::HalfSine { start: 0.0, duration, peak: peak_force },
        point: None,
    });
    s.sim.stiffness_scale = stiffness_scale;
    s.sim.fracture = false;
    s.sim.gravity_prestress = false;
    s.sim.frame_dt = duration / 10.0;
    s.sim.sample_interval = Some(duration / 200.0);
    // Long enough for the reflected pulse to pass the middle again.
    s.sim.duration = 2.4 * n as f64 * h / c;
    let v_particle = peak_force / area / (m.density * c);
    for (name, k) in [("v20", 20), ("v50", 50), ("v80", 80), ("v_end", n - 1)] {
        s.probes.push(probe(name, ProbeKind::ChunkVelocity { body: "bar".into(), chunk: k, axis: [1.0, 0.0, 0.0] }));
    }
    s.probes.push(probe(
        "axial_mid",
        ProbeKind::SectionForce { body: "bar".into(), point: [50.0 * h, 0.0, 0.0], normal: [1.0, 0.0, 0.0], region: None, component: SectionComponent::Normal },
    ));
    s.metrics.push(metric(
        "wave_speed",
        MetricKind::WaveSpeed { probe_a: "v20".into(), probe_b: "v80".into(), distance: 60.0 * h, threshold: 0.5 * v_particle },
        Tolerance::Relative(0.01),
        num(c),
        &["opencourant"],
    )
    .with_oracle_tolerance(Tolerance::Relative(0.05)));
    s.metrics.push(metric(
        "free_end_velocity_ratio",
        MetricKind::ProbePeakRatioOf { probe: "v_end".into(), reference: "v50".into() },
        Tolerance::Relative(0.01),
        num(2.0),
        &["opencourant"],
    )
    .with_oracle_tolerance(Tolerance::Relative(0.05)));
    s.metrics.push(metric(
        "incident_compression",
        MetricKind::ProbeMin { probe: "axial_mid".into() },
        Tolerance::Relative(0.01),
        num(-peak_force),
        &["opencourant"],
    )
    .with_oracle_tolerance(Tolerance::Relative(0.05)));
    s.metrics.push(metric(
        "reflected_tension",
        MetricKind::ProbeMax { probe: "axial_mid".into() },
        Tolerance::Relative(0.01),
        num(peak_force),
        &["opencourant"],
    )
    .with_oracle_tolerance(Tolerance::Relative(0.05)));
    s
}

// ======================================================================== benchmark 4

/// Benchmark 4: a block held between a lower and an upper support shares its weight
/// equally; the upper support is removed (suddenly or over `duration`). The lower
/// support's reaction jumps to the full weight with a dynamic overshoot: the
/// single-mass-spring solution gives an amplification of 2 for sudden loss.
pub fn support_loss(duration: f64, name: &str) -> Scene {
    let mut m = Material::analytic_test();
    m.damping_ratio = 0.02;
    let h = 0.05;
    let mut s = new_scene(name, "Mass between two supports; one support removed (sudden or gradual).");
    s.benchmark = Some(4);
    s.materials.insert("rock".into(), m.clone());
    let mut chunks = grid(Vec3::new(-h, -h, 0.0), [1, 1, 3], Vec3::splat(2.0 * h), "rock");
    chunks[0].support = Support::Fixed;
    chunks[2].support = Support::Fixed;
    chunks[0].groups.push("lower".into());
    chunks[2].groups.push("upper".into());
    let bonds = auto_bonds(&chunks, |_, _| "rock".into());
    s.bodies.push(body("stack", chunks, bonds));
    let t_remove = 0.01;
    s.events.push(EventDesc::RemoveChunks { time: t_remove, duration, body: "stack".into(), chunks: ChunkSelector::Group("upper".into()) });
    s.sim.fracture = false;
    s.sim.duration = 0.3;
    s.sim.frame_dt = 1e-3;
    s.sim.sample_interval = Some(2e-5);
    s.probes.push(probe("reaction", ProbeKind::Reaction { body: "stack".into(), chunks: ChunkSelector::Group("lower".into()), axis: [0.0, 0.0, 1.0] }));
    // Sudden: the damped single-degree-of-freedom peak 1 + exp(-zeta pi / sqrt(1 - zeta^2))
    // (2 without damping); gradual (much slower than the period): 1.
    let zeta = m.damping_ratio;
    let expected = if duration == 0.0 { 1.0 + (-zeta * std::f64::consts::PI / (1.0 - zeta * zeta).sqrt()).exp() } else { 1.0 };
    s.metrics.push(metric(
        "dynamic_amplification",
        MetricKind::DynamicAmplification { probe: "reaction".into(), before: t_remove - 1e-6 },
        Tolerance::Relative(0.01),
        num(expected),
        &[],
    ));
    s
}

// ======================================================================== oracle scenes

/// Concrete without rate effects or static fatigue, so continuum oracles that
/// lack them run the identical material. `weibull` enables randomized strengths.
pub fn oracle_concrete(weibull: Option<f64>) -> Material {
    Material { dif: None, static_fatigue: None, weibull_modulus: weibull, ..Material::concrete() }
}

/// A steel that stays elastic in these scenes (impactor bodies).
pub fn elastic_steel() -> Material {
    Material { tensile_strength: 5e9, compressive_strength: 5e9, cohesion: 5e9, shear_cap: None, ..Material::steel() }
}

fn region(min: [f64; 3], max: [f64; 3]) -> Region {
    Region { min, max }
}

/// Benchmark 5: a 2 m x 2 m x 0.2 m concrete wall on a fixed foundation course, struck
/// at mid-height by a 1000 kg rigid ram at `speed`. Slow pushes the wall over at its
/// base; fast punches a hole.
pub fn wall_impact(speed: f64, name: &str) -> Scene {
    let mut s = new_scene(name, "Concrete wall struck by a rigid ram (speed sweep): hole vs push-over.");
    s.benchmark = Some(5);
    s.materials.insert("concrete".into(), oracle_concrete(None));
    s.materials.insert("steel".into(), elastic_steel());
    let counts = [20, 2, 21];
    let mut chunks = grid(Vec3::new(-1.0, 0.0, -0.1), counts, Vec3::splat(0.1), "concrete");
    for i in 0..counts[0] {
        for j in 0..counts[1] {
            let c = &mut chunks[grid_index(counts, i, j, 0)];
            c.support = Support::Fixed;
            c.groups.push("foundation".into());
        }
    }
    let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
    s.bodies.push(body("wall", chunks, bonds));
    s.impactors.push(ImpactorDesc {
        name: "ram".into(),
        shape: ImpactorShape::Box { half_extents: [0.2, 0.25, 0.2] },
        mass: 1000.0,
        position: [0.0, -0.251, 1.0],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [0.0, speed, 0.0],
        angular_velocity: [0.0; 3],
        material: "steel".into(),
        crush: None,
    });
    s.sim.duration = if speed >= 20.0 { 0.04 } else if speed >= 8.0 { 0.1 } else { 0.3 };
    s.sim.sample_interval = Some(5e-4);
    s.probes.push(probe("ram_velocity", ProbeKind::ImpactorVelocity { impactor: "ram".into(), axis: [0.0, 1.0, 0.0] }));
    s.probes.push(probe(
        "base_shear",
        ProbeKind::Reaction { body: "wall".into(), chunks: ChunkSelector::Group("foundation".into()), axis: [0.0, 1.0, 0.0] },
    ));
    let impact = region([-0.35, -1.0, 0.65], [0.35, 1.0, 1.35]);
    let o = &["opencourant"];
    s.metrics.push(metric("failure_mode", MetricKind::FailureMode { body: "wall".into(), impact_region: impact }, Tolerance::Exact, None, o));
    s.metrics.push(metric(
        "hole_area",
        MetricKind::DetachedArea { body: "wall".into(), region: Some(region([-0.6, -1.0, 0.4], [0.6, 1.0, 1.6])), axis: [0.0, 1.0, 0.0] },
        Tolerance::Relative(0.30),
        None,
        o,
    ));
    s.metrics.push(metric("speed_lost", MetricKind::ProbeDrop { probe: "ram_velocity".into() }, Tolerance::Relative(0.30), None, o));
    if speed <= 2.0 {
        // The slow push-over by momentum alone: the wall (above the foundation, 2 m x
        // 0.2 m x 2 m) as a rigid body turning about its toe, `I = m (h^2 + t^2) / 3`, the
        // ram striking at `r = 1` m, perfectly plastic: the ram's angular momentum about
        // the toe is shared, `v0 m_r r^2 / (I + m_r r^2)` remains. An idealisation (rigid
        // wall, no rebound, no local damage), so reported against, not gated.
        let (m_w, h, t, r, m_r) = (2400.0 * 2.0 * 0.2 * 2.0, 2.0, 0.2, 1.0, 1000.0);
        let i_toe = m_w * (h * h + t * t) / 3.0;
        let lost = speed * i_toe / (i_toe + m_r * r * r);
        s.metrics.push(metric("speed_lost_rigid_push", MetricKind::ProbeDrop { probe: "ram_velocity".into() }, Tolerance::Report, num(lost), &[]));
    }
    s.metrics.push(metric("peak_base_shear", MetricKind::ProbePeak { probe: "base_shear".into() }, Tolerance::Report, None, o));
    s
}

/// Benchmark 6: a 0.5 m thick, 1.2 m wide concrete slab hit face-on by a 0.1 m steel
/// plate covering its whole face, so the stress wave is planar. The compressive pulse
/// (`sigma = v Zs Zc / (Zs + Zc)`, about 21 MPa at 3 m/s: below the crushing strength)
/// reflects from the free back face as tension far above the tensile strength and
/// spalls a layer about half a pulse length thick, which flies off at about twice the
/// particle velocity. The front face is not damaged.
pub fn spall(speed: f64, name: &str) -> Scene {
    let mut s = new_scene(name, "Wide concrete slab, full-face steel plate impact: back-face spall (planar wave).");
    s.benchmark = Some(6);
    s.gravity = [0.0; 3];
    let concrete = oracle_concrete(None);
    let steel = elastic_steel();
    s.materials.insert("concrete".into(), concrete.clone());
    s.materials.insert("steel".into(), steel.clone());
    let counts = [12, 10, 12];
    let size = Vec3::new(0.1, 0.05, 0.1);
    let block = grid(Vec3::new(-0.6, 0.0, -0.6), counts, size, "concrete");
    let bonds = auto_bonds(&block, |_, _| "concrete".into());
    s.bodies.push(body("block", block, bonds));
    // Four layers through the plate thickness resolve its 40 us pulse (2 L / c).
    let plate = grid(Vec3::new(-0.6, -0.1 - 1e-4, -0.6), [12, 4, 12], Vec3::new(0.1, 0.025, 0.1), "steel");
    let pb = auto_bonds(&plate, |_, _| "steel".into());
    let mut plate_body = body("plate", plate, pb);
    plate_body.linear_velocity = [0.0, speed, 0.0];
    s.bodies.push(plate_body);
    s.sim.gravity_prestress = false;
    // Elastic contact: the plate-slab interface must transmit the wave, not damp it.
    s.sim.contact_restitution = 1.0;
    s.sim.duration = 1.0e-3;
    s.sim.frame_dt = 1e-4;
    s.sim.sample_interval = Some(2e-6);
    s.probes.push(probe("back_face_velocity", ProbeKind::ChunkVelocity { body: "block".into(), chunk: grid_index(counts, 6, 9, 6), axis: [0.0, 1.0, 0.0] }));
    s.probes.push(probe("front_face_velocity", ProbeKind::ChunkVelocity { body: "block".into(), chunk: grid_index(counts, 6, 0, 6), axis: [0.0, 1.0, 0.0] }));
    // Analytic 1-D expectation for the central region (lateral release arrives later).
    let zc = concrete.density * concrete.bar_wave_speed();
    let zs = steel.density * steel.bar_wave_speed();
    let sigma = speed * zs * zc / (zs + zc);
    let v_free = 2.0 * sigma / zc;
    // Spall is judged in the central region, where the 1-D estimate holds until the
    // release waves from the side faces arrive; corner ligaments may still connect the
    // layer at the end (they do in the continuum oracle), so the criterion is the layer
    // separating, not chunk connectivity (`spall_detached` reports the latter).
    let back_central = region([-0.35, 0.4, -0.35], [0.35, 0.5, 0.35]);
    let inner_central = region([-0.35, 0.2, -0.35], [0.35, 0.35, 0.35]);
    let back = Some(back_central);
    let front = Some(region([-0.6, 0.0, -0.6], [0.6, 0.1, 0.6]));
    let o = &["opencourant"];
    let y = [0.0, 1.0, 0.0];
    s.metrics.push(metric(
        "spall_occurs",
        MetricKind::Separating { body: "block".into(), outer: back_central, inner: inner_central, axis: y, threshold: 0.1 * v_free },
        Tolerance::Exact,
        Some(serde_json::json!(true)),
        o,
    ));
    s.metrics.push(metric("spall_speed", MetricKind::RegionSpeed { body: "block".into(), region: back_central, axis: y }, Tolerance::Relative(0.30), None, o));
    s.metrics.push(metric("front_face_detached", MetricKind::DetachedAny { body: "block".into(), region: front }, Tolerance::Exact, Some(serde_json::json!(false)), o));
    s.metrics.push(metric("peak_back_face_velocity", MetricKind::ProbeMax { probe: "back_face_velocity".into() }, Tolerance::Relative(0.30), num(v_free), o));
    s.metrics.push(metric("spall_detached", MetricKind::DetachedAny { body: "block".into(), region: back }, Tolerance::Report, None, o));
    s.metrics.push(metric("spall_mass", MetricKind::DetachedMass { body: "block".into(), region: back }, Tolerance::Report, None, o));
    s
}

/// Benchmark 7: a running-bond brick wall (mortar joints) on a fixed footing, hit by a
/// rigid sphere. Breach yes/no and debris speeds.
pub fn masonry_wall(speed: f64, name: &str) -> Scene {
    let mut s = new_scene(name, "Running-bond masonry wall with mortar joints, sphere impact: breach and debris.");
    s.benchmark = Some(7);
    s.materials.insert("brick".into(), Material::brick());
    s.materials.insert("mortar".into(), Material::mortar());
    s.materials.insert("steel".into(), elastic_steel());
    let (bl, bt, bh) = (0.2, 0.1, 0.1);
    let (courses, width) = (15usize, 2.0f64);
    let mut chunks = Vec::new();
    // Footing course: fixed, full-length bricks.
    for i in 0..(width / bl) as usize {
        let mut c = chunk(Vec3::new(-1.0 + (i as f64 + 0.5) * bl, 0.5 * bt, -0.5 * bh), Vec3::new(0.5 * bl, 0.5 * bt, 0.5 * bh), "brick");
        c.support = Support::Fixed;
        c.groups.push("footing".into());
        chunks.push(c);
    }
    for k in 0..courses {
        let z = (k as f64 + 0.5) * bh;
        let mut edges: Vec<f64> = Vec::new();
        let mut x = -1.0;
        if k % 2 == 1 {
            edges.push(x);
            x += 0.5 * bl;
        }
        while x < 1.0 - 1e-9 {
            edges.push(x);
            x += bl;
        }
        edges.push(1.0);
        edges.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        for w in edges.windows(2) {
            let (x0, x1) = (w[0], w[1].min(1.0));
            if x1 - x0 < 1e-9 {
                continue;
            }
            chunks.push(chunk(Vec3::new(0.5 * (x0 + x1), 0.5 * bt, z), Vec3::new(0.5 * (x1 - x0), 0.5 * bt, 0.5 * bh), "brick"));
        }
    }
    let bonds = auto_bonds(&chunks, |_, _| "mortar".into());
    s.bodies.push(body("wall", chunks, bonds));
    s.impactors.push(ImpactorDesc {
        name: "ball".into(),
        shape: ImpactorShape::Sphere { radius: 0.15 },
        mass: 30.0,
        position: [0.0, -0.151, 0.75],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [0.0, speed, 0.0],
        angular_velocity: [0.0; 3],
        material: "steel".into(),
        crush: None,
    });
    s.sim.duration = 0.1;
    s.sim.sample_interval = Some(5e-4);
    s.probes.push(probe("ball_velocity", ProbeKind::ImpactorVelocity { impactor: "ball".into(), axis: [0.0, 1.0, 0.0] }));
    let impact = Some(region([-0.4, -1.0, 0.35], [0.4, 1.0, 1.15]));
    let o = &["lmgc90", "kratos", "opencourant"];
    s.metrics.push(metric("breach", MetricKind::DetachedAny { body: "wall".into(), region: impact }, Tolerance::Exact, None, o));
    s.metrics.push(metric(
        "debris_speed",
        MetricKind::DetachedSpeed { body: "wall".into(), region: None, axis: [0.0, 1.0, 0.0], max: false },
        Tolerance::Relative(0.30),
        None,
        o,
    ));
    s.metrics.push(metric("debris_mass", MetricKind::DetachedMass { body: "wall".into(), region: None }, Tolerance::Report, None, o));
    s.metrics.push(metric("ball_speed_lost", MetricKind::ProbeDrop { probe: "ball_velocity".into() }, Tolerance::Relative(0.30), None, o));
    s
}

/// Frame layout shared by the scene and the OpenSees exporter: 3 bays x 2 storeys,
/// 3 m bays and storeys, 0.3 m square members chunked at 0.3 m.
pub struct FrameLayout;

impl FrameLayout {
    pub const CELL: f64 = 0.3;
    pub const COLUMNS: [usize; 4] = [0, 10, 20, 30];
    pub const BEAM_ROWS: [usize; 2] = [10, 20];
    pub const NX: usize = 31;
    pub const NZ: usize = 21;
}

/// Benchmark 8: a plain-concrete 2-D frame under self weight and a floor load loses its
/// second ground-floor column, suddenly or over `duration`. Sudden loss overloads the
/// beams over the lost column (dynamic amplification); gradual loss does not.
pub fn frame_column_removal(duration: f64, name: &str) -> Scene {
    frame_column_removal_with(duration, true, name)
}

/// The frame scene; with `fracture = false` it is the elastic variant whose peak
/// forces an elastic oracle can match (a fracturing frame's peaks after the first
/// break are not comparable with an elastic analysis).
pub fn frame_column_removal_with(duration: f64, fracture: bool, name: &str) -> Scene {
    let mut s = new_scene(name, "3-bay 2-storey concrete frame, ground column removed (sudden or gradual).");
    s.benchmark = Some(8);
    s.materials.insert("concrete".into(), oracle_concrete(None));
    let c = FrameLayout::CELL;
    let mut chunks = Vec::new();
    let mut index = std::collections::HashMap::new();
    for k in 0..FrameLayout::NZ {
        for i in 0..FrameLayout::NX {
            let col = FrameLayout::COLUMNS.iter().position(|&x| x == i);
            let row = FrameLayout::BEAM_ROWS.iter().position(|&z| z == k);
            if col.is_none() && row.is_none() {
                continue;
            }
            let mut ch = chunk(Vec3::new((i as f64 + 0.5) * c, 0.0, (k as f64 + 0.5) * c), Vec3::splat(0.5 * c), "concrete");
            match (col, row) {
                (Some(ci), Some(r)) => ch.groups.push(format!("joint_{ci}_{r}")),
                (Some(ci), None) => {
                    let storey = if k < FrameLayout::BEAM_ROWS[0] { 0 } else { 1 };
                    ch.groups.push(format!("column_{ci}_{storey}"));
                }
                (None, Some(r)) => {
                    let bay = i / 10;
                    ch.groups.push(format!("beam_{r}_{bay}"));
                }
                _ => unreachable!(),
            }
            index.insert((i, k), chunks.len());
            chunks.push(ch);
        }
    }
    for (ci, &i) in FrameLayout::COLUMNS.iter().enumerate() {
        let mut ch = chunk(Vec3::new((i as f64 + 0.5) * c, 0.0, -0.5 * c), Vec3::splat(0.5 * c), "concrete");
        ch.support = Support::Fixed;
        ch.groups.push(format!("footing_{ci}"));
        ch.groups.push("footing".into());
        chunks.push(ch);
    }
    let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
    s.bodies.push(body("frame", chunks, bonds));
    // Floor load: 1.5 kN/m on every beam chunk (on top of self weight).
    let beam_chunks: Vec<usize> = (0..s.bodies[0].chunks.len())
        .filter(|&i| s.bodies[0].chunks[i].groups.iter().any(|g| g.starts_with("beam_") || g.starts_with("joint_")))
        .collect();
    for ch in beam_chunks {
        s.loads.push(LoadDesc::PointForce {
            body: "frame".into(),
            chunk: ch,
            direction: [0.0, 0.0, -1.0],
            magnitude: TimeFunction::Constant { value: 1.5e3 * c },
            point: None,
        });
    }
    let t_remove = 0.05;
    s.events.push(EventDesc::RemoveChunks { time: t_remove, duration, body: "frame".into(), chunks: ChunkSelector::Group("column_1_0".into()) });
    s.sim.duration = t_remove + duration + 0.6;
    s.sim.frame_dt = 1e-3;
    s.sim.sample_interval = Some(1e-3);
    // Axial forces in the neighbouring ground columns, and the beam moment next to the lost column.
    for (name, ci) in [("axial_col0", 0usize), ("axial_col2", 2usize)] {
        let x = (FrameLayout::COLUMNS[ci] as f64 + 0.5) * c;
        s.probes.push(probe(
            name,
            ProbeKind::SectionForce {
                body: "frame".into(),
                point: [x, 0.0, 1.5],
                normal: [0.0, 0.0, 1.0],
                region: Some(region([x - 0.2, -1.0, 0.0], [x + 0.2, 1.0, 3.0])),
                component: SectionComponent::Normal,
            },
        ));
    }
    let x_face = (FrameLayout::COLUMNS[1] as f64 + 1.0) * c;
    s.probes.push(probe(
        "beam_moment_at_lost_column",
        ProbeKind::SectionForce {
            body: "frame".into(),
            point: [x_face, 0.0, 3.15],
            normal: [1.0, 0.0, 0.0],
            region: Some(region([x_face - 0.2, -1.0, 2.9], [x_face + 0.2, 1.0, 3.4])),
            component: SectionComponent::Moment([0.0, 1.0, 0.0]),
        },
    ));
    let o = &["opensees"];
    let before = t_remove - 1e-3;
    s.metrics.push(metric("axial_col0_before", MetricKind::ProbeAt { probe: "axial_col0".into(), time: before }, Tolerance::Relative(0.20), None, o));
    s.metrics.push(metric("axial_col2_before", MetricKind::ProbeAt { probe: "axial_col2".into(), time: before }, Tolerance::Relative(0.20), None, o));
    s.metrics.push(metric(
        "beam_moment_before",
        MetricKind::ProbeAt { probe: "beam_moment_at_lost_column".into(), time: before },
        Tolerance::Relative(0.20),
        None,
        o,
    ));
    s.sim.fracture = fracture;
    let peaks_comparable = !fracture || duration > 0.0;
    if peaks_comparable {
        s.metrics.push(metric("axial_col0_peak", MetricKind::ProbePeak { probe: "axial_col0".into() }, Tolerance::Relative(0.20), None, o));
        s.metrics.push(metric("axial_col2_peak", MetricKind::ProbePeak { probe: "axial_col2".into() }, Tolerance::Relative(0.20), None, o));
        s.metrics.push(metric(
            "beam_moment_peak",
            MetricKind::ProbePeak { probe: "beam_moment_at_lost_column".into() },
            Tolerance::Relative(0.20),
            None,
            o,
        ));
    }
    // Damage onset anywhere: the criterion an elastic analysis can evaluate.
    s.metrics.push(metric("first_failure", MetricKind::Flag { flag: "any_bond_cracked".into() }, Tolerance::Exact, None, o));
    if fracture {
        s.metrics.push(metric("collapse", MetricKind::Flag { flag: "collapse".into() }, Tolerance::Report, None, &[]));
    }
    s
}

/// Benchmark 9: a 2 m x 2 m x 0.1 m concrete panel spanning between fixed top and
/// bottom edge strips, loaded on its front face by a Friedlander pressure pulse.
pub fn pressure_panel(peak: f64, name: &str) -> Scene {
    pressure_panel_sized(peak, 0.1, name)
}

/// The pressure panel with chunks of size `cell` (0.1 m in the benchmark): the same
/// 2 m x 2 m x 0.1 m panel, edge strips and pulse for refinement studies.
pub fn pressure_panel_sized(peak: f64, cell: f64, name: &str) -> Scene {
    let mut s = new_scene(name, "One-way concrete panel under a Friedlander pressure pulse: breach and fragment speeds.");
    s.benchmark = Some(9);
    s.materials.insert("concrete".into(), oracle_concrete(None));
    let n = (2.0 / cell).round() as usize;
    let nt = (0.1 / cell).round().max(1.0) as usize;
    let counts = [n, nt, n + 2];
    let mut chunks = grid(Vec3::new(-1.0, 0.0, -cell), counts, Vec3::new(cell, 0.1 / nt as f64, cell), "concrete");
    for i in 0..counts[0] {
        for j in 0..counts[1] {
        for k in [0, counts[2] - 1] {
            let c = &mut chunks[grid_index(counts, i, j, k)];
            c.support = Support::Fixed;
            c.groups.push("support".into());
        }
        }
    }
    let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
    s.bodies.push(body("panel", chunks, bonds));
    let centre = grid_index(counts, n / 2, 0, n / 2 + 1);
    s.loads.push(LoadDesc::Pressure {
        body: "panel".into(),
        chunks: ChunkSelector::Region(region([-1.0, -1.0, 0.0], [1.0, 1.0, 2.0])),
        face_normal: [0.0, -1.0, 0.0],
        pressure: TimeFunction::Friedlander { arrival: 1e-3, peak, duration: 10e-3, decay: 1.0 },
    });
    s.sim.duration = 0.06;
    s.sim.frame_dt = 1e-3;
    s.sim.sample_interval = Some(2.5e-4);
    s.probes.push(probe(
        "center_displacement",
        ProbeKind::ChunkDisplacement { body: "panel".into(), chunk: centre, axis: [0.0, 1.0, 0.0] },
    ));
    let o = &["opencourant"];
    s.metrics.push(metric("breach", MetricKind::DetachedAny { body: "panel".into(), region: None }, Tolerance::Exact, None, o));
    s.metrics.push(metric(
        "fragment_speed",
        MetricKind::DetachedSpeed { body: "panel".into(), region: None, axis: [0.0, 1.0, 0.0], max: false },
        Tolerance::Relative(0.30),
        None,
        o,
    ));
    s.metrics.push(metric("first_failure", MetricKind::Flag { flag: "any_bond_broken".into() }, Tolerance::Exact, None, o));
    s.metrics.push(metric("peak_center_displacement", MetricKind::ProbeMax { probe: "center_displacement".into() }, Tolerance::Relative(0.30), None, o));
    s
}

/// Benchmark 9 with Weibull-distributed bond strengths (modulus 8): compared as a
/// distribution over seeds (`stress-ref seeds`, oracle `_seedN` goldens).
pub fn pressure_panel_weibull(peak: f64, name: &str) -> Scene {
    let mut s = pressure_panel(peak, name);
    s.description = "One-way concrete panel, Friedlander pulse, Weibull bond strengths: distributions over seeds.".into();
    s.materials.insert("concrete".into(), oracle_concrete(Some(8.0)));
    s
}

/// A grid of coarse chunks, each pre-fractured into `2 x 2 x 2` finer children
/// (level 1), with bonds on both levels: the two-scale representation.
pub fn two_level_grid(min_corner: Vec3, counts: [usize; 3], size: Vec3, material: &str) -> (Vec<ChunkDesc>, Vec<BondDesc>) {
    let mut chunks = grid(min_corner, counts, size, material);
    let n0 = chunks.len();
    for p in 0..n0 {
        let c = Vec3::from_array(chunks[p].center);
        let h = size * 0.25;
        for k in 0..8 {
            let o = Vec3::new(
                if k & 1 == 0 { -h.x } else { h.x },
                if k & 2 == 0 { -h.y } else { h.y },
                if k & 4 == 0 { -h.z } else { h.z },
            );
            let mut ch = chunk(c + o, h, material);
            ch.level = 1;
            ch.parent = Some(p);
            chunks.push(ch);
        }
    }
    let bonds = auto_bonds(&chunks, |_, _| material.to_string());
    (chunks, bonds)
}

/// Explosion beside a building: two parallel anchored walls, the charge in front of the
/// first. The facing wall takes the reflected pulse; the wall behind it is shadowed.
pub fn blast_two_walls(tnt: f64, name: &str) -> Scene {
    let mut s = new_scene(name, "Charge beside two parallel walls: facing wall breaches, shadowed wall survives.");
    s.materials.insert("concrete".into(), oracle_concrete(None));
    for (bname, y0) in [("front", 0.0), ("back", 2.0)] {
        let counts = [20, 2, 21];
        let mut chunks = grid(Vec3::new(-1.0, y0, -0.1), counts, Vec3::splat(0.1), "concrete");
        for i in 0..counts[0] {
            for j in 0..counts[1] {
                chunks[grid_index(counts, i, j, 0)].support = Support::Fixed;
            }
        }
        let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
        s.bodies.push(body(bname, chunks, bonds));
    }
    s.loads.push(LoadDesc::Blast { position: [0.0, -2.0, 1.0], tnt_mass: tnt, time: 0.0 });
    s.sim.duration = 0.03;
    s.sim.frame_dt = 1e-3;
    s.metrics.push(metric("front_breach", MetricKind::DetachedAny { body: "front".into(), region: None }, Tolerance::Report, None, &[]));
    s.metrics.push(metric("back_breach", MetricKind::DetachedAny { body: "back".into(), region: None }, Tolerance::Report, None, &[]));
    s
}

/// Every scene of the catalogue.
pub fn catalog() -> Vec<Scene> {
    vec![
        bond_tension(),
        bond_shear(),
        cantilever(20, "b2_cantilever"),
        cantilever(10, "b2_cantilever_n10"),
        cantilever(40, "b2_cantilever_n40"),
        bar_wave(1.0, "b3_bar_wave"),
        bar_wave(0.01, "b3_bar_wave_scaled"),
        support_loss(0.0, "b4_support_loss_sudden"),
        support_loss(0.1, "b4_support_loss_gradual"),
        wall_impact(2.0, "b5_wall_impact_v02"),
        wall_impact(10.0, "b5_wall_impact_v10"),
        wall_impact(40.0, "b5_wall_impact_v40"),
        spall(3.0, "b6_spall"),
        masonry_wall(4.0, "b7_masonry_v04"),
        masonry_wall(15.0, "b7_masonry_v15"),
        frame_column_removal(0.0, "b8_frame_sudden"),
        frame_column_removal(1.0, "b8_frame_gradual"),
        frame_column_removal_with(0.0, false, "b8_frame_sudden_elastic"),
        pressure_panel(10e3, "b9_panel_low"),
        pressure_panel(300e3, "b9_panel_high"),
        pressure_panel_weibull(300e3, "b9_panel_high_weibull"),
    ]
}
