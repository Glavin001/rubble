//! Gate for the modal pre-window (`world::modal`, `findings/modal-prewindow.md`).
//!
//! Each case runs twice from the same scene:
//! * **explicit** (the truth): the reference world in explicit mode. It is stepped
//!   substep by substep up to the time the modal window ended (timed: the span the
//!   window replaced), then to the scene's end.
//! * **modal**: the modal pre-window from t = 0; on onset the world is handed over to
//!   the explicit solver, which runs to the scene's end.
//!
//! Compared: first damage onset (first `Cracked` event), broken bonds, fragments,
//! impactor exit velocity along its initial direction, and the damage verdict.
//!
//! ```sh
//! cargo run --release --features modal --example modal_prewindow -- [--methods established|proven] \
//!     [--case NAME]... [--json out.json] [--serial]
//! ```
//! `--serial` runs the modal window on one thread (faer sequential, no rayon), for a
//! one-core comparison with the explicit solver (which is serial at these sizes).

use std::collections::HashMap;
use std::time::Instant;

use stress_ref::builders::*;
use stress_ref::math::Vec3;
use stress_ref::scene::*;
use stress_ref::solver::SolverEvent;
use stress_ref::world::modal::{self, ModalBasis, ModalConfig, Verdict, WindowReport};
use stress_ref::world::World;

/// A 200 kg ram at 40 m/s into a free 528 kg concrete wall (`rubble_references.rs`).
fn ram_into_free_wall() -> Scene {
    let mut s = new_scene("ram_into_free_wall", "200 kg steel ram at 40 m/s into a free concrete wall");
    s.materials.insert("concrete".into(), oracle_concrete(None));
    s.materials.insert("steel".into(), elastic_steel());
    let chunks = grid(Vec3::new(-0.5, 0.0, -0.1), [10, 2, 11], Vec3::splat(0.1), "concrete");
    let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
    s.bodies.push(body("wall", chunks, bonds));
    s.impactors.push(ImpactorDesc {
        name: "ram".into(),
        shape: ImpactorShape::Box { half_extents: [0.1, 0.1, 0.1] },
        mass: 200.0,
        position: [0.0, -0.15, 0.5],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [0.0, 40.0, 0.0],
        angular_velocity: [0.0; 3],
        material: "steel".into(),
        crush: None,
    });
    s.sim.duration = 0.5;
    s
}

/// b5's wall struck by a light piece of debris: a `size` m cube of `material` at `speed`.
fn debris(name: &str, material: &str, size: f64, speed: f64) -> Scene {
    let mut s = wall_impact(speed, name);
    let mat = match material {
        "steel" => elastic_steel(),
        _ => oracle_concrete(None),
    };
    s.materials.insert(material.into(), mat.clone());
    let h = 0.5 * size;
    s.impactors[0] = ImpactorDesc {
        name: "debris".into(),
        shape: ImpactorShape::Box { half_extents: [h, h, h] },
        mass: mat.density * size * size * size,
        position: [0.0, -h - 0.001, 1.0],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [0.0, speed, 0.0],
        angular_velocity: [0.0; 3],
        material: material.into(),
        crush: None,
    };
    s.probes.clear();
    s.metrics.clear();
    s.sim.duration = 0.05;
    s
}

struct Case {
    name: String,
    /// Scenes with the same asset share one modal basis.
    asset: &'static str,
    scene: Scene,
}

fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for v in [10.0, 15.0, 20.0, 30.0, 40.0] {
        out.push(Case { name: format!("b5 {v} m/s"), asset: "b5", scene: wall_impact(v, &format!("b5_v{v}")) });
    }
    out.push(Case { name: "free wall ram 40 m/s".into(), asset: "free_wall", scene: ram_into_free_wall() });
    for v in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0] {
        out.push(Case { name: format!("b5 {v} m/s"), asset: "b5", scene: wall_impact(v, &format!("b5_v{v}")) });
    }
    out.push(Case { name: "debris 10 cm concrete 1 m/s".into(), asset: "b5", scene: debris("debris_c1", "concrete", 0.1, 1.0) });
    out.push(Case { name: "debris 2 cm steel 5 m/s".into(), asset: "b5", scene: debris("debris_s5", "steel", 0.02, 5.0) });
    out.push(Case { name: "debris 10 cm concrete 5 m/s".into(), asset: "b5", scene: debris("debris_c5", "concrete", 0.1, 5.0) });
    out.push(Case { name: "debris 10 cm concrete 20 m/s".into(), asset: "b5", scene: debris("debris_c20", "concrete", 0.1, 20.0) });
    out.push(Case { name: "debris 5 cm steel 10 m/s".into(), asset: "b5", scene: debris("debris_s10", "steel", 0.05, 10.0) });
    out
}

#[derive(Clone, Debug, Default, serde::Serialize)]
struct Outcome {
    first_crack: Option<f64>,
    broken: usize,
    fragments: usize,
    exit_velocity: f64,
    max_utilization: f64,
    substeps: u64,
    seconds: f64,
}

fn outcome(world: &World, dir: Vec3, substeps: u64, seconds: f64) -> Outcome {
    let first_crack = world.solver.events.iter().find_map(|e| match e {
        SolverEvent::Cracked { time, .. } => Some(*time),
        _ => None,
    });
    Outcome {
        first_crack,
        broken: world.solver.broken_bond_count(),
        fragments: world.solver.clusters.len(),
        exit_velocity: world.impactors[0].velocity.dot(dir),
        max_utilization: world.solver.max_utilization,
        substeps,
        seconds,
    }
}

#[derive(Debug, serde::Serialize)]
struct Row {
    case: String,
    methods: String,
    truth: Outcome,
    /// The explicit solver with the impactor 1 um further back (its own sensitivity).
    perturbed: Outcome,
    /// The explicit solver at 0.9 times the Courant safety.
    perturbed_dt: Outcome,
    /// Where the window ended: relative hidden-displacement and -velocity difference
    /// from the explicit solver at the same time, and impactor velocity difference (m/s).
    state_diff: (f64, f64, f64),
    /// Explicit substeps and seconds over the span the modal window covered.
    truth_span_substeps: u64,
    truth_span_seconds: f64,
    modal: Outcome,
    verdict: String,
    window_end: f64,
    window_steps: u64,
    window_seconds: f64,
    window_breakdown: [f64; 8],
    window_max_utilization: f64,
    window_contact_steps: u64,
    certificate_bound: f64,
    max_displacement: f64,
    onset_diff_ms: Option<f64>,
    verdict_match: bool,
    false_negative: bool,
}

fn fmt_opt(x: Option<f64>) -> String {
    x.map(|t| format!("{:.4}", t * 1e3)).unwrap_or_else(|| "none".into())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut methods = Methods::established();
    let mut methods_name = "established".to_string();
    let mut filter: Vec<String> = Vec::new();
    let mut json: Option<String> = None;
    let mut serial = false;
    let mut window_only = false;
    let mut zoh = false;
    let mut safety_scale = 1.0;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--methods" => {
                methods_name = args[i + 1].clone();
                methods = match methods_name.as_str() {
                    "proven" => Methods::proven(),
                    _ => Methods::established(),
                };
                i += 1;
            }
            "--case" => {
                filter.push(args[i + 1].clone());
                i += 1;
            }
            "--json" => {
                json = Some(args[i + 1].clone());
                i += 1;
            }
            "--serial" => serial = true,
            // The contact as a constant force over each step instead of an impulse.
            "--zoh" => zoh = true,
            // Compare only up to the window's end (no run to the scene's end).
            "--window-only" => window_only = true,
            // Scale the Courant safety (stress and contact steps) of both runs.
            "--safety-scale" => {
                safety_scale = args[i + 1].parse().unwrap();
                i += 1;
            }
            other => panic!("unknown argument {other}"),
        }
        i += 1;
    }
    if serial {
        faer::set_global_parallelism(faer::Par::Seq);
    }
    let cfg = ModalConfig { parallel: !serial, contact_impulse: !zoh, ..ModalConfig::default() };
    let mut bases: HashMap<&'static str, ModalBasis> = HashMap::new();
    let mut rows: Vec<Row> = Vec::new();
    println!("methods: {methods_name}, modal window {}", if serial { "serial" } else { "parallel" });
    for case in cases() {
        if !filter.is_empty() && !filter.iter().any(|f| case.name.contains(f.as_str())) {
            continue;
        }
        let mut scene = case.scene.clone();
        scene.sim.methods = methods;
        scene.sim.courant_safety *= safety_scale;
        if let Some(x) = scene.sim.stress_courant_safety.as_mut() {
            *x *= safety_scale;
        }
        let dir = Vec3::from_array(scene.impactors[0].velocity).normalized();

        // Modal path.
        let mut world = World::new(&scene);
        let basis = bases.entry(case.asset).or_insert_with(|| {
            let b = ModalBasis::new(&world, 0).expect("modal basis");
            let t = &b.timings;
            println!(
                "basis {}: {} DOFs, {} flexible, {} modal coordinates, {:.0} MB; omega {:.3e}..{:.3e} rad/s, zeta {:.4}..{:.4}, non-classical {:.3e}, residuals {:.1e} {:.1e} {:.1e}",
                case.asset,
                b.n,
                b.nf,
                b.modes(),
                b.bytes() as f64 / 1e6,
                b.omega_range.0,
                b.omega_range.1,
                b.zeta_range.0,
                b.zeta_range.1,
                b.nonclassical,
                b.residuals[0],
                b.residuals[1],
                b.residuals[2]
            );
            println!(
                "  precompute {:.2} s: assemble {:.2}, undamped eigen {:.2}, projection {:.2}, state-space eigen {:.2}, inverse {:.2}, outputs {:.2}",
                t.total, t.assemble, t.undamped, t.project, t.state_space, t.inverse, t.outputs
            );
            b
        });
        assert!(basis.matches(&world, 0), "{}: basis of another cluster", case.name);
        let t = Instant::now();
        let report: WindowReport = modal::run(&mut world, basis, &cfg).expect("modal window");
        let frames = (scene.sim.duration / scene.sim.frame_dt).round() as u64;
        if report.verdict.damages() && !window_only {
            while world.frame < frames {
                world.step_frame();
            }
        }
        // The window's own time (the rest is the explicit solver after the handover).
        let window_seconds = report.timings.total;
        let modal = outcome(&world, dir, world.solver.substeps, t.elapsed().as_secs_f64());

        // Explicit truth, timed over the window's span, then to the end.
        let mut truth_world = World::new(&scene);
        let t = Instant::now();
        let (span_steps, span_seconds, truth_state) = modal::explicit_until(&mut truth_world, report.verdict.time());
        while truth_world.frame < frames && !window_only {
            truth_world.step_frame();
        }
        let truth_seconds = t.elapsed().as_secs_f64();
        let truth = outcome(&truth_world, dir, truth_world.solver.substeps, truth_seconds);
        let state_diff = report.end_state.as_ref().map(|m| m.compare(&truth_state, 0.1)).unwrap_or((f64::NAN, f64::NAN, f64::NAN));
        if std::env::var("MODAL_DEBUG").is_ok() {
            if let Some(m) = &report.end_state {
                let mut worst = (0.0, 0, 0);
                for (c, (a, b)) in m.chunks.iter().zip(&truth_state.chunks).enumerate() {
                    for q in 0..4 {
                        let d = (a[q] - b[q]).norm();
                        if d > worst.0 {
                            worst = (d, c, q);
                        }
                    }
                }
                let (d, c, q) = worst;
                eprintln!("worst diff {d:.3e} at chunk {c} field {q}: modal {:?} explicit {:?}; t {} vs {}", m.chunks[c][q], truth_state.chunks[c][q], m.time, truth_state.time);
            }
        }
        // The explicit solver's own sensitivity: the impactor 1 um further back.
        let mut perturbed_scene = scene.clone();
        let p = &mut perturbed_scene.impactors[0].position;
        for k in 0..3 {
            p[k] -= 1e-6 * dir.to_array()[k];
        }
        let mut pw = World::new(&perturbed_scene);
        let t = Instant::now();
        while pw.frame < frames && !window_only {
            pw.step_frame();
        }
        let perturbed = outcome(&pw, dir, pw.solver.substeps, t.elapsed().as_secs_f64());
        // And at a 10% smaller step (the size of difference the window makes: the explicit
        // solver's own time-discretisation error).
        let mut dt_scene = scene.clone();
        dt_scene.sim.courant_safety *= 0.9;
        if let Some(x) = dt_scene.sim.stress_courant_safety.as_mut() {
            *x *= 0.9;
        }
        let mut dw = World::new(&dt_scene);
        let t = Instant::now();
        while dw.frame < frames && !window_only {
            dw.step_frame();
        }
        let perturbed_dt = outcome(&dw, dir, dw.solver.substeps, t.elapsed().as_secs_f64());

        let onset_diff_ms = match (truth.first_crack, modal.first_crack) {
            (Some(a), Some(b)) => Some((a - b).abs() * 1e3),
            _ => None,
        };
        let truth_damages = truth.first_crack.is_some();
        let modal_damages = modal.first_crack.is_some();
        let tm = &report.timings;
        let verdict = match &report.verdict {
            Verdict::Handover { time, bond, crossing } => format!("handover at {:.4} ms (bond {bond}, crossing {:.4} ms)", time * 1e3, crossing * 1e3),
            Verdict::Certified { time, bound } => format!("certified no damage at {:.3} ms (bound {bound:.3})", time * 1e3),
            Verdict::NoOnset { time } => format!("no onset to the end ({:.3} ms)", time * 1e3),
        };
        let row = Row {
            case: case.name.clone(),
            methods: methods_name.clone(),
            truth_span_substeps: span_steps,
            truth_span_seconds: span_seconds,
            verdict,
            window_end: report.verdict.time(),
            window_steps: report.steps,
            window_seconds,
            window_breakdown: [tm.setup, tm.contact, tm.modal, tm.bodies, tm.checks_gemm, tm.checks_law, tm.certificate, tm.handover],
            window_max_utilization: report.max_utilization,
            window_contact_steps: report.contact_steps,
            certificate_bound: report.last_certificate_bound,
            max_displacement: report.max_displacement,
            onset_diff_ms,
            verdict_match: truth_damages == modal_damages,
            false_negative: truth_damages && !modal_damages,
            truth,
            perturbed,
            perturbed_dt,
            state_diff,
            modal,
        };
        println!(
            "\n{} [{}]\n  modal: {}; {} steps ({} with contact) in {:.3} s; max utilization {:.3}; max contact-row displacement {:.2e} m\n  window time: setup {:.3}, contact {:.3}, modal {:.3}, bodies {:.3}, checks gemm {:.3} + law {:.3}, certificate {:.3}, handover {:.3}\n  explicit over the same span: {} substeps in {:.3} s\n  {:<10} {:>12} {:>7} {:>9} {:>11} {:>8} {:>10} {:>9}\n  {:<10} {:>12} {:>7} {:>9} {:>11.4} {:>8.3} {:>10} {:>9.2}\n  {:<10} {:>12} {:>7} {:>9} {:>11.4} {:>8.3} {:>10} {:>9.2}\n  explicit, ram 1 um back: onset {} ms, {} broken, {} fragments, exit {:.4} m/s\n  explicit, step x 0.9: onset {} ms, {} broken, {} fragments, exit {:.4} m/s\n  state at window end vs explicit: displacement {:.2e}, velocity {:.2e} (relative), impactor {:.2e} m/s\n  onset diff {} ms, verdict match {}, false negative {}",
            row.case,
            row.methods,
            row.verdict,
            row.window_steps,
            row.window_contact_steps,
            row.window_seconds,
            row.window_max_utilization,
            row.max_displacement,
            tm.setup,
            tm.contact,
            tm.modal,
            tm.bodies,
            tm.checks_gemm,
            tm.checks_law,
            tm.certificate,
            tm.handover,
            row.truth_span_substeps,
            row.truth_span_seconds,
            "",
            "onset (ms)",
            "broken",
            "fragments",
            "exit (m/s)",
            "max util",
            "substeps",
            "time (s)",
            "explicit",
            fmt_opt(row.truth.first_crack),
            row.truth.broken,
            row.truth.fragments,
            row.truth.exit_velocity,
            row.truth.max_utilization,
            row.truth.substeps,
            row.truth.seconds,
            "modal",
            fmt_opt(row.modal.first_crack),
            row.modal.broken,
            row.modal.fragments,
            row.modal.exit_velocity,
            row.modal.max_utilization,
            row.modal.substeps,
            row.modal.seconds,
            fmt_opt(row.perturbed.first_crack),
            row.perturbed.broken,
            row.perturbed.fragments,
            row.perturbed.exit_velocity,
            fmt_opt(row.perturbed_dt.first_crack),
            row.perturbed_dt.broken,
            row.perturbed_dt.fragments,
            row.perturbed_dt.exit_velocity,
            row.state_diff.0,
            row.state_diff.1,
            row.state_diff.2,
            row.onset_diff_ms.map(|d| format!("{d:.4}")).unwrap_or_else(|| "-".into()),
            row.verdict_match,
            row.false_negative
        );
        rows.push(row);
    }
    let all_match = rows.iter().all(|r| r.verdict_match);
    let no_fn = rows.iter().all(|r| !r.false_negative);
    let onset_ok = rows.iter().all(|r| r.onset_diff_ms.map_or(true, |d| d <= 0.1));
    println!("\nverdicts match: {all_match}, false negatives: {}, onset within 0.1 ms: {onset_ok}", !no_fn);
    if let Some(path) = json {
        std::fs::write(&path, serde_json::to_string_pretty(&rows).unwrap()).unwrap();
        println!("wrote {path}");
    }
}
