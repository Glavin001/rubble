//! The accuracy gate against stress-ref, with the reference's runs cached on disk.
//!
//! Each case's reference set is the reference run, the run at half the substep and the
//! runs with the inputs perturbed by +-1e-4 (the size of the GPU's f32 drift). The GPU
//! passes where its probes stay within 1e-3 of their range or within twice the
//! reference's own spread, and its broken bonds and fragments within 10% or twice the
//! spread's deviation (`gate`).
//!
//! The reference runs are slow (minutes for the larger scenes, at full duration) and
//! never change unless stress-ref or the scene does, so they are computed once and kept
//! under `target/stress-gate-cache/`, keyed by the scene's JSON, the perturbation and a
//! fingerprint of stress-ref's sources. A GPU check then costs only the GPU runs.

use std::path::{Path, PathBuf};

use stress_ref::math::Vec3;
use stress_ref::observation::Observation;
use stress_ref::scene::{ImpactorDesc, ImpactorShape, ProbeDesc, ProbeKind, Scene, SolveMode};
use stress_ref::world::World;

pub const KEYS: [&str; 6] = ["broken_bonds", "fragments", "max_failure_index", "first_crack_time", "bond_dissipation", "damping_dissipation"];

/// Worst probe error (relative to the reference's probe range) and the outcome values.
pub fn difference(reference: &Observation, other: &Observation) -> (f64, Vec<String>) {
    let mut worst = 0.0f64;
    let mut notes = Vec::new();
    for (name, r) in &reference.probes {
        let Some(g) = other.probes.get(name) else {
            notes.push(format!("{name}: missing"));
            worst = f64::INFINITY;
            continue;
        };
        let range = r.v.iter().filter(|x| x.is_finite()).fold(0.0f64, |m, x| m.max(x.abs())).max(1e-30);
        let n = r.v.len().min(g.v.len());
        let mut err = 0.0f64;
        let mut at = 0;
        for i in 0..n {
            let (a, b) = (r.v[i], g.v[i]);
            let e = if a.is_nan() && b.is_nan() { 0.0 } else if a.is_nan() || b.is_nan() { f64::INFINITY } else { (a - b).abs() / range };
            if e > err {
                err = e;
                at = i;
            }
        }
        if r.v.len() != g.v.len() {
            notes.push(format!("{name}: samples {} vs {}", r.v.len(), g.v.len()));
        }
        let detail = if n > 0 && err > 1e-3 { format!(" at t={:.5} ({:.4e} vs {:.4e}, range {:.3e})", r.t[at], r.v[at], g.v[at], range) } else { String::new() };
        notes.push(format!("{name} {err:.1e}{detail}"));
        worst = worst.max(err);
    }
    for key in KEYS {
        let (a, b) = (reference.values.get(key).copied(), other.values.get(key).copied());
        if a != b {
            let fmt = |x: Option<f64>| x.map_or("-".into(), |v| format!("{v:.6}"));
            notes.push(format!("{key} {}|{}", fmt(a), fmt(b)));
        }
    }
    (worst, notes)
}

/// The scene with its impactor and body velocities and gravity scaled by `1 + eps`.
pub fn perturbed_by(scene: &Scene, eps: f64) -> Scene {
    let mut s = scene.clone();
    let k = 1.0 + eps;
    s.gravity = s.gravity.map(|g| g * k);
    for imp in &mut s.impactors {
        imp.velocity = imp.velocity.map(|v| v * k);
    }
    for b in &mut s.bodies {
        b.linear_velocity = b.linear_velocity.map(|v| v * k);
    }
    s
}

/// The input perturbations of the spread runs: +-1e-6, +-1e-5, +-1e-4
/// (`STRESS_GPU_PERTURB=e` restricts them to +-e). One pair is not enough: on a chaotic
/// cascade (b8_frame_sudden, adaptive) the reference's broken bonds range over 18-35
/// for input changes from 1e-7 to 1e-3, while +-1e-4 alone happened to give 22-24.
pub fn perturbations() -> Vec<f64> {
    match std::env::var("STRESS_GPU_PERTURB").ok().and_then(|v| v.parse::<f64>().ok()) {
        Some(e) => vec![e, -e],
        None => vec![1e-6, -1e-6, 1e-5, -1e-5, 1e-4, -1e-4],
    }
}

/// The reference's sensitivity to its inputs: the runs with its inputs perturbed
/// (`perturbations`). That is the size of the GPU's own perturbation: f32 arithmetic
/// moves even smooth, intact runs by 1e-5 to 5e-4 of their range over thousands of
/// substeps, so wherever the reference separates under such changes (fracture
/// cascades, fatigue and settling thresholds), the GPU may separate as far.
pub fn spread_run(scene: &Scene, _reference: &Observation) -> Vec<Observation> {
    perturbations().iter().map(|&e| World::new(&perturbed_by(scene, e)).run()).collect()
}

/// The largest of the spread runs' differences from the reference (with its notes).
pub fn spread_difference(reference: &Observation, spreads: &[Observation]) -> (f64, Vec<String>) {
    spreads.iter().map(|o| difference(reference, o)).fold((0.0, Vec::new()), |a, b| if b.0 > a.0 || a.1.is_empty() { b } else { a })
}

/// The scene at half the reference's substep (its Courant safety halved).
pub fn refined(scene: &Scene) -> Scene {
    let mut s = scene.clone();
    s.sim.scale_safety(0.5);
    s
}

fn value(o: &Observation, key: &str) -> f64 {
    o.values.get(key).copied().unwrap_or(0.0)
}

/// The accuracy gate (the analysis's G2, judged against the reference's own spread):
/// probes within 1e-3 of their range, or within twice the largest difference of the
/// reference's timestep-halving and input-perturbation runs; broken bonds and fragments
/// within 10% (at least one), or within twice those runs' largest deviation (fracture
/// cascades scatter their counts: b5 v10 adaptive gives 41-50 fragments over input
/// changes of 1e-6 to 2e-4). Returns what fails.
pub fn gate(reference: &Observation, ours: &Observation, half: &Observation, spreads: &[Observation]) -> Vec<String> {
    let mut fails = Vec::new();
    let (gpu_err, _) = difference(reference, ours);
    let allowed = allowed_error(reference, half, spreads);
    if !(gpu_err <= allowed) {
        fails.push(format!("probes {gpu_err:.1e} > {allowed:.1e}"));
    }
    for key in ["broken_bonds", "fragments"] {
        let r = value(reference, key);
        let d = (value(ours, key) - r).abs();
        let deviation = spreads.iter().chain(std::iter::once(half)).map(|o| (value(o, key) - r).abs()).fold(0.0, f64::max);
        let allowed = (0.1 * r).max(1.0).max(2.0 * deviation);
        if d > allowed {
            fails.push(format!("{key} {} vs {r} (allowed {allowed})", value(ours, key)));
        }
    }
    fails
}

/// The counts of the reference's ensemble: the reference run, the half step and the
/// perturbed runs.
fn ensemble_counts(refs: &ReferenceSet, key: &str) -> Vec<f64> {
    std::iter::once(&refs.reference).chain(std::iter::once(&refs.half)).chain(refs.spreads.iter()).map(|o| value(o, key)).collect()
}

/// A chaotic case: the reference's own ensemble scatters its broken bonds or fragments
/// by at least 10% (and 3). One GPU run against one reference run is then a draw from
/// each distribution, not a comparison (b8_frame_sudden adaptive: the reference gives
/// 18-35 broken bonds and the GPU 16-33 under 1e-7..1e-3 input changes, while single
/// runs gave 23 and 43).
pub fn is_chaotic(refs: &ReferenceSet) -> bool {
    ["broken_bonds", "fragments"].iter().any(|k| {
        let r = value(&refs.reference, k);
        let dev = ensemble_counts(refs, k).iter().map(|c| (c - r).abs()).fold(0.0, f64::max);
        dev >= (0.1 * r).max(3.0)
    })
}

/// The ensemble judgement of a chaotic case: the GPU run on the same input changes as
/// the reference's ensemble (`perturbations`, plus the unperturbed run); its median
/// broken bonds and fragments must lie within the reference ensemble's range, and its
/// median probe error within the allowed error. Returns what fails.
pub fn ensemble_gate(refs: &ReferenceSet, gpu: &[Observation]) -> Vec<String> {
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let mut fails = Vec::new();
    for key in ["broken_bonds", "fragments"] {
        let c = ensemble_counts(refs, key);
        let (lo, hi) = (c.iter().copied().fold(f64::INFINITY, f64::min), c.iter().copied().fold(f64::NEG_INFINITY, f64::max));
        let m = median(gpu.iter().map(|o| value(o, key)).collect());
        if m < lo || m > hi {
            fails.push(format!("{key}: GPU median {m} outside the reference's {lo}-{hi}"));
        }
    }
    let allowed = allowed_error(&refs.reference, &refs.half, &refs.spreads);
    let err = median(gpu.iter().map(|o| difference(&refs.reference, o).0).collect());
    if !(err <= allowed) {
        fails.push(format!("median probes {err:.1e} > {allowed:.1e}"));
    }
    fails
}

/// The probe error the gate allows: twice the reference's own spread, at least 1e-3.
pub fn allowed_error(reference: &Observation, half: &Observation, spreads: &[Observation]) -> f64 {
    let (half_err, _) = difference(reference, half);
    let (spread_err, _) = spread_difference(reference, spreads);
    (2.0 * half_err.max(spread_err)).max(1e-3)
}

// ------------------------------------------------------------------ cases

/// A gate case: a scene at the duration it is judged over.
#[derive(Clone)]
pub struct Case {
    pub name: String,
    pub scene: Scene,
}

/// The bounds of a scene's chunks (world).
pub fn bounds(scene: &Scene) -> (Vec3, Vec3) {
    let mut lo = Vec3::splat(f64::INFINITY);
    let mut hi = Vec3::splat(f64::NEG_INFINITY);
    for c in scene.bodies.iter().flat_map(|b| b.chunks.iter()) {
        let p = Vec3::from_array(c.center);
        lo = lo.component_min(p);
        hi = hi.component_max(p);
    }
    (lo, hi)
}

/// An impactor aimed at the middle of the structure's -x face, from 0.2 m outside it,
/// with probes on its velocity and on a few chunks' displacement.
pub fn with_impactor(mut scene: Scene, shape: ImpactorShape, mass: f64, speed: f64) -> Scene {
    let (lo, hi) = bounds(&scene);
    let mid = (lo + hi) * 0.5;
    let reach = match shape {
        ImpactorShape::Sphere { radius } => radius,
        ImpactorShape::Box { half_extents } => half_extents[0],
    };
    let material = scene.bodies[0].chunks[0].material.clone();
    scene.impactors.push(ImpactorDesc {
        name: "ram".into(),
        shape,
        mass,
        position: [lo.x - 0.2 - reach, mid.y, mid.z],
        orientation: [0.0, 0.0, 0.0, 1.0],
        velocity: [speed, 0.0, 0.0],
        angular_velocity: [0.0; 3],
        material,
        crush: None,
    });
    scene.probes.push(ProbeDesc { name: "ram_velocity".into(), kind: ProbeKind::ImpactorVelocity { impactor: "ram".into(), axis: [1.0, 0.0, 0.0] } });
    with_chunk_probes(scene)
}

/// Displacement probes on the chunks nearest a few points of the structure.
pub fn with_chunk_probes(mut scene: Scene) -> Scene {
    let (lo, hi) = bounds(&scene);
    let body = scene.bodies[0].name.clone();
    for (i, f) in [[0.0, 0.5, 0.5], [0.5, 0.5, 1.0], [1.0, 0.5, 0.5]].iter().enumerate() {
        let target = lo + Vec3::new((hi.x - lo.x) * f[0], (hi.y - lo.y) * f[1], (hi.z - lo.z) * f[2]);
        let nearest = (0..scene.bodies[0].chunks.len())
            .min_by(|&a, &b| {
                let d = |k: usize| (Vec3::from_array(scene.bodies[0].chunks[k].center) - target).norm();
                d(a).total_cmp(&d(b))
            })
            .unwrap();
        for (axis, name) in [([1.0, 0.0, 0.0], "x"), ([0.0, 0.0, 1.0], "z")] {
            scene.probes.push(ProbeDesc { name: format!("chunk{i}_{name}"), kind: ProbeKind::ChunkDisplacement { body: body.clone(), chunk: nearest, axis } });
        }
    }
    scene
}

/// The authored scene packs' directory (sibling asset checkout).
pub fn packs_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packs")
}

/// Every gate case: each catalogue and showcase scene as authored, at its full duration;
/// a set of them in the adaptive and quasi-static modes; the authored packs that stand
/// at rest (villa-savoye is over strength under its own weight as authored), struck.
pub fn cases() -> Vec<Case> {
    let catalogue: Vec<Scene> = stress_ref::builders::catalog().into_iter().chain(stress_ref::showcases::catalog()).collect();
    let mut out: Vec<Case> = catalogue.iter().map(|s| Case { name: s.name.clone(), scene: s.clone() }).collect();
    let modes: [(&str, SolveMode); 8] = [
        ("b2_cantilever", SolveMode::QuasiStatic),
        ("b8_frame_sudden", SolveMode::QuasiStatic),
        ("s_floor_static", SolveMode::QuasiStatic),
        ("b8_frame_sudden", SolveMode::Adaptive),
        ("b9_panel_high", SolveMode::Adaptive),
        ("b5_wall_impact_v10", SolveMode::Adaptive),
        ("s_house_car", SolveMode::Adaptive),
        ("s_overhang_thin", SolveMode::Adaptive),
    ];
    for (name, mode) in modes {
        let mut s = catalogue.iter().find(|s| s.name == name).expect("catalogue scene").clone();
        s.sim.solve_mode = mode;
        out.push(Case { name: format!("{name} ({mode:?})"), scene: s });
    }
    let packs: [(&str, &str, f64); 4] = [("rig-portal", "box", 0.25), ("comp-wall-bay", "sphere", 0.25), ("rig-toppled", "box", 0.25), ("house-1story", "box", 0.1)];
    for (pack, load, duration) in packs {
        let path = packs_dir().join(format!("{pack}.json"));
        let Ok(base) = stress_ref::scene_pack::import(&path, pack, "structure") else { continue };
        let mut scene = match load {
            "box" => with_impactor(base, ImpactorShape::Box { half_extents: [0.15, 0.3, 0.3] }, 400.0, 12.0),
            _ => with_impactor(base, ImpactorShape::Sphere { radius: 0.25 }, 300.0, 15.0),
        };
        scene.sim.duration = duration;
        out.push(Case { name: format!("{pack} ({load})"), scene });
    }
    out
}

/// The quick tier: a representative subset that runs in a few minutes (elastic
/// benchmarks, fracture, contact, the solve modes, a pack), for iterating; the full
/// set before a commit.
/// Fuzz scene seeds that once failed (`stress-gpu-gate --fuzz` prints them ready to paste):
/// always run with `--micro` (`micro::fuzz_scene`).
pub const FUZZ_REGRESSIONS: &[u64] = &[
    // Displacement probes rounded to the f32 ulp of the chunk position (fixed in record_probes):
    0x5e8c87468dcf66d3, // stone wall, prestressed sag only
    0x814cac4f3e95b960, // timber, box impactor
    0x34cac5489fdc078a, // steel, adaptive
    // A probe the reference holds at exactly zero (f32 noise of 1e-14 m judged absolutely):
    0x3c31cba91af8fb74, // glass slab resting on the ground
];

pub use crate::micro::{fuzz_scene, micro_cases, micro_reference, with_knobs, Knobs, MicroCase};

pub const QUICK: [&str; 13] = [
    "b1_bond_tension",
    "b2_cantilever",
    "b3_bar_wave",
    "b4_support_loss_sudden",
    "b7_masonry_v15",
    "b8_frame_sudden",
    "b9_panel_high",
    "s_car_brick",
    "s_floor_static",
    "s_house_car",
    "b8_frame_sudden (QuasiStatic)",
    "b9_panel_high (Adaptive)",
    "rig-portal (box)",
];

// ------------------------------------------------------------------ the cache

/// A case's reference runs.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct ReferenceSet {
    pub reference: Observation,
    pub half: Observation,
    pub spreads: Vec<Observation>,
    /// Host seconds the reference run took (when it was computed).
    pub reference_seconds: f64,
}

/// 64-bit FNV-1a (stable across runs and toolchains, unlike `DefaultHasher`).
pub(crate) fn fnv(bytes: &[u8], mut h: u64) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// A fingerprint of stress-ref's sources (and manifest): every cached result is stale
/// once they change. Computed once per process.
pub fn reference_fingerprint() -> u64 {
    static FINGERPRINT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *FINGERPRINT.get_or_init(|| {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../stress-ref");
        let mut files = Vec::new();
        let mut stack = vec![root.join("src"), root.join("Cargo.toml")];
        while let Some(p) = stack.pop() {
            if p.is_dir() {
                for e in std::fs::read_dir(&p).into_iter().flatten().flatten() {
                    stack.push(e.path());
                }
            } else {
                files.push(p);
            }
        }
        files.sort();
        let mut h = 0xcbf29ce484222325;
        for f in files {
            // Relative paths: every checkout of the same sources shares the cache.
            h = fnv(f.strip_prefix(&root).unwrap_or(&f).to_string_lossy().as_bytes(), h);
            h = fnv(&std::fs::read(&f).unwrap_or_default(), h);
        }
        h
    })
}

static REFRESH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Recompute (and overwrite) every cached result this process asks for: `--refresh`,
/// or `STRESS_GATE_REFRESH=1`.
pub fn set_refresh(on: bool) {
    REFRESH.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn refresh() -> bool {
    REFRESH.load(std::sync::atomic::Ordering::Relaxed) || std::env::var("STRESS_GATE_REFRESH").is_ok_and(|v| v == "1")
}

/// Where cached results live (`target/stress-gate-cache`, or `STRESS_GATE_CACHE`).
pub fn cache_dir() -> PathBuf {
    std::env::var_os("STRESS_GATE_CACHE").map(PathBuf::from).unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/stress-gate-cache"))
}

/// The cache file of a computation keyed by `key` (with stress-ref's fingerprint), for
/// tools outside the gate that cache reference runs (`label` is readable).
pub fn cached_path(label: &str, key: &str) -> PathBuf {
    cache_path(label, &key)
}

/// The cache file of a computation: `label` (readable), its inputs `key` (serialized,
/// hashed with stress-ref's fingerprint).
fn cache_path<K: serde::Serialize>(label: &str, key: &K) -> PathBuf {
    let json = serde_json::to_string(key).expect("cache key json");
    let mut h = fnv(json.as_bytes(), 0xcbf29ce484222325);
    h = fnv(&reference_fingerprint().to_le_bytes(), h);
    let safe: String = label.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    cache_dir().join(format!("{safe}-{h:016x}.json"))
}

/// A cached result: `compute` runs only on a miss (or with refresh on), and its result
/// is stored. The key must hold every input of the computation besides stress-ref's
/// sources (the fingerprint covers those): the same inputs on the same reference build
/// reproduce the result.
pub fn cached<K, T>(label: &str, key: &K, compute: impl FnOnce() -> T) -> T
where
    K: serde::Serialize,
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    lookup(label, key).unwrap_or_else(|| {
        let value = compute();
        store(label, key, &value);
        value
    })
}

/// The cached result, if present and fresh (refresh on: none).
pub fn lookup<K: serde::Serialize, T: serde::de::DeserializeOwned>(label: &str, key: &K) -> Option<T> {
    lookup_at(&cache_path(label, key))
}

fn lookup_at<T: serde::de::DeserializeOwned>(path: &Path) -> Option<T> {
    if refresh() {
        return None;
    }
    serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
}

/// Store a result.
pub fn store<K: serde::Serialize, T: serde::Serialize>(label: &str, key: &K, value: &T) {
    store_at(&cache_path(label, key), value);
}

/// Written, then renamed: a cache file is complete or absent.
fn store_at<T: serde::Serialize>(path: &Path, value: &T) {
    std::fs::create_dir_all(cache_dir()).ok();
    let tmp = path.with_extension("tmp");
    if std::fs::write(&tmp, serde_json::to_string(value).expect("cache json")).is_ok() {
        std::fs::rename(&tmp, &path).ok();
    }
}

/// The reference set's cache file: the scene's JSON, the perturbations and stress-ref's
/// fingerprint.
fn reference_path(scene: &Scene) -> PathBuf {
    let json = serde_json::to_string(scene).expect("scene json");
    let mut h = fnv(json.as_bytes(), 0xcbf29ce484222325);
    for e in perturbations() {
        h = fnv(&e.to_bits().to_le_bytes(), h);
    }
    h = fnv(&reference_fingerprint().to_le_bytes(), h);
    cache_dir().join(format!("{}-{h:016x}.json", scene.name))
}

/// JSON has no NaN or infinity (serde writes null, which does not read back as f64):
/// such samples and values (a chunk gone, an unbounded failure index) are stored as
/// these values.
const NAN_SENTINEL: f64 = -3.0e307;
const INF_SENTINEL: f64 = 3.1e307;
const NEG_INF_SENTINEL: f64 = -3.1e307;

fn map_observation(o: &mut Observation, f: impl Fn(f64) -> f64) {
    for p in o.probes.values_mut() {
        for x in p.t.iter_mut().chain(p.v.iter_mut()).chain(p.lo.iter_mut()).chain(p.hi.iter_mut()) {
            *x = f(*x);
        }
    }
    for x in o.values.values_mut() {
        *x = f(*x);
    }
}

fn map_set(set: &mut ReferenceSet, f: impl Fn(f64) -> f64 + Copy) {
    map_observation(&mut set.reference, f);
    map_observation(&mut set.half, f);
    for o in &mut set.spreads {
        map_observation(o, f);
    }
}

/// A value as stored in a cache file (NaN and infinities as sentinels).
pub(crate) fn enc(x: f64) -> f64 {
    if x.is_nan() {
        NAN_SENTINEL
    } else if x == f64::INFINITY {
        INF_SENTINEL
    } else if x == f64::NEG_INFINITY {
        NEG_INF_SENTINEL
    } else {
        x
    }
}

/// A value read from a cache file (`enc` undone).
pub(crate) fn dec(x: f64) -> f64 {
    if x == NAN_SENTINEL {
        f64::NAN
    } else if x == INF_SENTINEL {
        f64::INFINITY
    } else if x == NEG_INF_SENTINEL {
        f64::NEG_INFINITY
    } else {
        x
    }
}

pub(crate) fn encode(mut set: ReferenceSet) -> ReferenceSet {
    map_set(&mut set, enc);
    set
}

pub(crate) fn decode(mut set: ReferenceSet) -> ReferenceSet {
    map_set(&mut set, dec);
    set
}

/// The case's reference set from the cache, or computed (its runs in parallel) and
/// stored. `compute`: false returns None on a miss instead of running the reference.
pub fn reference_set(scene: &Scene, compute: bool) -> Option<ReferenceSet> {
    let path = reference_path(scene);
    if let Some(r) = lookup_at::<ReferenceSet>(&path) {
        return Some(decode(r));
    }
    if !compute {
        return None;
    }
    let eps = perturbations();
    let (reference, half, spreads) = std::thread::scope(|sc| {
        let r = sc.spawn(|| {
            let t = web_time::Instant::now();
            let o = World::new(scene).run();
            (o, t.elapsed().as_secs_f64())
        });
        let h = sc.spawn(|| World::new(&refined(scene)).run());
        let ps: Vec<_> = eps.iter().map(|&e| sc.spawn(move || World::new(&perturbed_by(scene, e)).run())).collect();
        (r.join().unwrap(), h.join().unwrap(), ps.into_iter().map(|p| p.join().unwrap()).collect::<Vec<_>>())
    });
    let set = ReferenceSet { reference: reference.0, half, spreads, reference_seconds: reference.1 };
    let set = encode(set);
    store_at(&path, &set);
    Some(decode(set))
}
