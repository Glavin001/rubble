//! Head-to-head on PhysX's own systems: a solve the native destruction stage captured (its
//! diagnostic runtime, `stressgpu/detail/StressProblemCapture.cuh` version 2, as vibe-land's
//! `scripts/perf/build-stress-capture-sdk.sh` records them) solved again by rubble's GPU native
//! solve from the same warm start, and both against the converged f64 answer.
//!
//!   cargo run --release -p stress-gpu --example blast_capture -- <capture dir or solve .json>...
//!
//! A capture holds per node its `D` weights, right-hand side `b` and component; per bond its
//! nodes, offsets (length units), health, column scale and warm forces; per component what the
//! stage decided (anchored, rotations, iterations, converged); and the forces it ended with.
//! The capture is input data read at run time, never a test fixture.
//!
//! `STRESS_BLAST_ITERATIONS` (64, vibe-land's garage cap), `STRESS_BLAST_FORCE_TOL` (0),
//! `STRESS_BLAST_FIRST=identity|polynomial|warm` (`BLAST_STRESS_FIRST_PRECONDITIONER`).

use std::path::{Path, PathBuf};

use stress_gpu::blast::GpuBlastSolver;
use stress_gpu::gpu::Gpu;
use stress_ref::blast_bond::*;
use stress_ref::math::Vec3;

// The solve does not read materials; one to satisfy the solver.
const MATERIAL: BlastMaterial = BlastMaterial {
    compression_elastic: 1.0,
    compression_fatal: 2.0,
    tension_elastic: 1.0,
    tension_fatal: 2.0,
    shear_elastic: 1.0,
    shear_fatal: 2.0,
    residual_area_fraction: 0.0,
    modulus: 0.0,
};

fn words(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn f(w: u32) -> f64 {
    f32::from_bits(w) as f64
}

struct Component {
    id: u32,
    anchored: bool,
    rotations: u32,
    iterations: u32,
    converged: bool,
    failed: bool,
}

struct Capture {
    name: String,
    /// Per node: (d_a, d_l), b (angular, linear), component.
    weight: Vec<(f64, f64)>,
    rhs: Vec<AngLin>,
    component: Vec<u32>,
    /// Per bond: nodes, offsets, health, column scale, warm forces.
    ends: Vec<[usize; 2]>,
    o0: Vec<Vec3>,
    o1: Vec<Vec3>,
    health: Vec<f64>,
    scale: Vec<f64>,
    warm: Vec<AngLin>,
    components: Vec<Component>,
    /// The forces the stage ended with.
    lambda: Vec<AngLin>,
    tolerance: f64,
    /// Per-bond rotational stiffness `A` (packed xx yy zz xy xz yz), when captured.
    rotation: Vec<[f64; 6]>,
}

fn v3(w: &[u32]) -> Vec3 {
    Vec3::new(f(w[0]), f(w[1]), f(w[2]))
}

fn load(meta: &Path) -> Capture {
    let text = std::fs::read_to_string(meta).expect("capture metadata");
    let m: serde_json::Value = serde_json::from_str(&text).expect("capture json");
    assert_eq!(
        m["version"].as_u64(),
        Some(2),
        "{}: capture version 2 expected",
        meta.display()
    );
    assert_eq!(m["record_bytes"].as_u64(), Some(64));
    let base = meta
        .to_str()
        .unwrap()
        .strip_suffix(".json")
        .unwrap()
        .to_string();
    let part = |ext: &str| words(Path::new(&format!("{base}.{ext}.bin")));
    let (nodes, bonds) = (part("nodes"), part("bonds"));
    let n = m["node_count"].as_u64().unwrap() as usize;
    let mb = m["bond_count"].as_u64().unwrap() as usize;
    assert!(
        nodes.len() == 16 * n && bonds.len() == 16 * mb,
        "truncated capture"
    );
    let mut c = Capture {
        name: meta.file_name().unwrap().to_string_lossy().into_owned(),
        weight: Vec::new(),
        rhs: Vec::new(),
        component: Vec::new(),
        ends: Vec::new(),
        o0: Vec::new(),
        o1: Vec::new(),
        health: Vec::new(),
        scale: Vec::new(),
        warm: Vec::new(),
        components: Vec::new(),
        lambda: Vec::new(),
        tolerance: 0.0,
        rotation: Vec::new(),
    };
    for r in nodes.chunks_exact(16) {
        c.weight.push((f(r[0]), f(r[1])));
        c.rhs.push(AngLin {
            ang: v3(&r[2..5]),
            lin: v3(&r[5..8]),
        });
        c.component.push(r[15]);
    }
    for r in bonds.chunks_exact(16) {
        c.ends.push([r[0] as usize, r[1] as usize]);
        c.o0.push(v3(&r[2..5]));
        c.o1.push(v3(&r[5..8]));
        c.health.push(f(r[8]));
        c.scale.push(f(r[9]));
        c.warm.push(AngLin {
            ang: v3(&r[10..13]),
            lin: v3(&r[13..16]),
        });
    }
    for r in part("components").chunks_exact(16) {
        c.components.push(Component {
            id: r[0],
            anchored: r[2] != 0,
            rotations: r[3],
            iterations: r[5],
            converged: r[6] != 0,
            failed: r[8] != 0,
        });
        // tolerance2 = tol^2 |b|^2 over the component.
        let b2: f64 = (0..n)
            .filter(|&i| c.component[i] == r[0])
            .map(|i| c.rhs[i].ang.norm2() + c.rhs[i].lin.norm2())
            .sum();
        if b2 > 0.0 && c.tolerance == 0.0 {
            c.tolerance = (f(r[9]) / b2).sqrt();
        }
    }
    if m.get("rotation").is_some_and(|r| !r.is_null()) {
        c.rotation = part("rotation")
            .chunks_exact(6)
            .map(|r| std::array::from_fn(|k| f(r[k])))
            .collect();
        assert_eq!(c.rotation.len(), mb, "truncated rotation capture");
    }
    for r in part("lambda").chunks_exact(8) {
        c.lambda.push(AngLin {
            ang: v3(&r[0..3]),
            lin: v3(&r[4..7]),
        });
    }
    c
}

/// The capture as a support graph: length scale 1 (offsets are already in its units), the
/// stage's weights and column scales, health as area.
fn graph(c: &Capture, settings: BlastSettings) -> BlastGraph {
    let nodes: Vec<BlastNode> = c
        .weight
        .iter()
        .map(|w| BlastNode {
            mass: if w.1 > 0.0 { 1.0 } else { 0.0 },
            inertia: 1.0,
            volume: 1.0,
            ..Default::default()
        })
        .collect();
    let bonds: Vec<BlastBond> = (0..c.ends.len())
        .map(|j| BlastBond {
            nodes: c.ends[j],
            area: c.health[j],
            material: 0,
            ..Default::default()
        })
        .collect();
    let mut g = BlastGraph::new(nodes, bonds, settings).expect("capture graph");
    g.length_scale = 1.0;
    g.mass_scale = 1.0;
    g.offset0 = c.o0.clone();
    g.offset1 = c.o1.clone();
    g.weight = c
        .weight
        .iter()
        .map(|&(ang, lin)| Weight { ang, lin })
        .collect();
    g.col_scale = c.scale.clone();
    // The stage's A as captured (its offsets already the centroids'), W = A^2.
    g.rotation = c.rotation.iter().map(|a| [*a, sym_square(a)]).collect();
    g
}

/// a = -D b (length scale 1): the input whose right-hand side is the captured b.
fn accel(c: &Capture) -> Vec<AngLin> {
    (0..c.weight.len())
        .map(|i| {
            let (wa, wl) = c.weight[i];
            AngLin {
                ang: c.rhs[i].ang * -wa,
                lin: c.rhs[i].lin * -wl,
            }
        })
        .collect()
}

/// |a - b| / |b| over the live bonds.
fn relative(a: &[AngLin], b: &[AngLin], live: &[bool]) -> f64 {
    let (mut d, mut s) = (0.0, 0.0);
    for j in 0..b.len() {
        if live[j] {
            d += (a[j].ang - b[j].ang).norm2() + (a[j].lin - b[j].lin).norm2();
            s += b[j].ang.norm2() + b[j].lin.norm2();
        }
    }
    if s > 0.0 {
        (d / s).sqrt()
    } else {
        d.sqrt()
    }
}

fn solves(args: &[String]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for a in args {
        let p = PathBuf::from(a);
        if p.is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(&p)
                .unwrap()
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    p.to_string_lossy().contains(".solve-")
                        && p.extension().is_some_and(|e| e == "json")
                })
                .collect();
            // By structure, then solve number (solve-2 before solve-10).
            let key = |p: &PathBuf| {
                let name = p.file_name().unwrap().to_string_lossy().into_owned();
                let (stem, rest) = name.split_once(".solve-").unwrap_or((&name, ""));
                (
                    stem.to_string(),
                    rest.trim_end_matches(".json").parse::<u64>().unwrap_or(0),
                )
            };
            v.sort_by_key(key);
            out.extend(v);
        } else {
            out.push(p);
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(
        !args.is_empty(),
        "usage: blast_capture <capture dir or solve .json>..."
    );
    let gpu = Gpu::new().expect("a GPU");
    let iterations: u32 =
        std::env::var("STRESS_BLAST_ITERATIONS").map_or(64, |v| v.parse().expect("iterations"));
    let force_tolerance: f64 = std::env::var("STRESS_BLAST_FORCE_TOL")
        .map_or(0.0, |v| v.parse().expect("force tolerance"));
    let limit: usize =
        std::env::var("STRESS_CAPTURE_LIMIT").map_or(usize::MAX, |v| v.parse().expect("limit"));
    println!("solve\tcomponents\tframe(anch,rot) same\titer physx\titer rubble\tconv physx\tconv rubble\t|rubble-physx|/|physx|\t|physx-exact|\t|rubble-exact|");
    let (mut worst, mut frames_differ) = (0.0f64, 0usize);
    // STRESS_CAPTURE_SEQUENCE=1: one solver across consecutive solves of the same structure (same
    // health), each warm from rubble's own last solve, as the stage runs (the Krylov carry needs it).
    let sequence = std::env::var("STRESS_CAPTURE_SEQUENCE").is_ok_and(|v| v == "1");
    let mut kept: Option<(GpuBlastSolver, Vec<f64>)> = None;
    // STRESS_CAPTURE_RESTART=n: start solve n afresh from its captured warm start (what a
    // topology generation change does to the stage's carry), for diagnosis.
    let restart: Option<usize> = std::env::var("STRESS_CAPTURE_RESTART")
        .ok()
        .map(|v| v.parse().expect("restart"));
    for (index, meta) in solves(&args).into_iter().take(limit).enumerate() {
        let c = load(&meta);
        let settings = BlastSettings {
            max_iterations: iterations,
            tolerance: c.tolerance,
            force_tolerance,
            first_preconditioner: match std::env::var("STRESS_BLAST_FIRST").as_deref() {
                Ok("polynomial") => FirstPreconditioner::Polynomial,
                Ok("warm") => FirstPreconditioner::Warm,
                _ => FirstPreconditioner::Identity,
            },
            carry_krylov: std::env::var("STRESS_BLAST_CARRY").ok().map(|v| v == "1"),
            ..BlastSettings::native()
        };
        let g = graph(&c, settings);
        let live: Vec<bool> = c.health.iter().map(|&h| h > 0.0).collect();
        let a = accel(&c);
        let mut solver = match kept.take() {
            Some((s, health)) if sequence && health == c.health && restart != Some(index) => s,
            _ => {
                let mut s =
                    GpuBlastSolver::with_wide_min(&gpu, g.clone(), vec![MATERIAL], u32::MAX);
                s.set_solution(&c.warm);
                s
            }
        };
        let reports = solver.solve(&a, false);
        let ours = solver.solution();
        let frames = solver.rigid_frames();
        let islands = solver.islands();
        // The converged answer, f64, from the same warm start.
        let mut exact = BlastSolver::new(graph(
            &c,
            BlastSettings {
                max_iterations: 20_000,
                tolerance: 1e-9,
                force_tolerance: 0.0,
                ..settings
            },
        ));
        exact.set_solution(c.warm.clone());
        exact.solve(&a, false);
        // Components by their lowest node.
        let (mut same, mut it_p, mut it_r, mut conv_p, mut conv_r) =
            (0usize, 0u32, 0u32, 0usize, 0usize);
        for comp in &c.components {
            let Some(k) = (0..islands.count).find(|&k| {
                islands.node.iter().position(|&x| x == k as u32) == Some(comp.id as usize)
            }) else {
                continue;
            };
            let fr = &frames[k];
            let rotations = if fr.anchored { 0 } else { fr.rotations as u32 };
            let theirs = if comp.anchored { 0 } else { comp.rotations };
            if fr.anchored == comp.anchored && rotations == theirs {
                same += 1;
            } else {
                frames_differ += 1;
                eprintln!("{}: component {} frame differs: physx anchored {} rotations {}, rubble anchored {} rotations {}", c.name, comp.id, comp.anchored, comp.rotations, fr.anchored, fr.rotations);
            }
            it_p += comp.iterations;
            it_r += reports[k].iterations;
            conv_p += (comp.converged && !comp.failed) as usize;
            conv_r += reports[k].converged as usize;
        }
        let d = relative(&ours, &c.lambda, &live);
        worst = worst.max(d);
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3e}\t{:.3e}\t{:.3e}",
            c.name,
            c.components.len(),
            same,
            it_p,
            it_r,
            conv_p,
            conv_r,
            d,
            relative(&c.lambda, &exact.x, &live),
            relative(&ours, &exact.x, &live)
        );
        if sequence {
            kept = Some((solver, c.health.clone()));
        }
    }
    println!("worst |rubble - physx| / |physx|: {worst:.3e}; components whose frame differs: {frames_differ}");
}
