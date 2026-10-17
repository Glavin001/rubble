//! Diagnostic (not a gate): native repro of the web "Grid Boxes" scene, which
//! visibly explodes (boxes flung far from a ~2-unit footprint). Deep stacks
//! (NY=24) inject energy; shallow (NY=6) settle. This harness runs the T1-T6
//! discriminating series to isolate impact-transient vs. static-load-propagation
//! vs. convergence failure, and to characterize the onset.
//!
//! Run: cargo test -p rubble3d --test grid_explosion_diagnostic --release -- --ignored --nocapture

mod support;

use glam::Vec3;
use rubble3d::{RigidBodyDesc, ShapeDesc, SimConfig, World};
use support::{add_tracked_body, compute_metrics, snapshot_all, try_world, TrackedBody3D};

#[derive(Clone, Copy)]
enum Floor {
    Plane,
    Box,
}

#[derive(Clone, Copy)]
struct GridSpec {
    nx: usize,
    ny: usize,
    nz: usize,
    floor: Floor,
    /// Vertical gap between stacked layers. 0.08 = the web scene (pile-driver
    /// drop); ~0 = pre-settled (boxes already touching).
    vgap: f32,
    /// Gap from the bottom layer to the floor. ~0 = resting on the floor.
    clearance: f32,
}

impl GridSpec {
    fn n(&self) -> usize {
        self.nx * self.ny * self.nz
    }
}

fn build_grid(spec: GridSpec, cfg: SimConfig) -> Option<(World, Vec<TrackedBody3D>, Vec3)> {
    let gravity = cfg.gravity;
    let mut world = try_world(SimConfig {
        max_bodies: spec.n() + 4,
        ..cfg
    })?;

    match spec.floor {
        Floor::Plane => {
            world.add_body(&RigidBodyDesc {
                position: Vec3::ZERO,
                mass: 0.0,
                shape: ShapeDesc::Plane {
                    normal: Vec3::Y,
                    distance: 0.0,
                },
                ..Default::default()
            });
        }
        Floor::Box => {
            world.add_body(&RigidBodyDesc {
                position: Vec3::new(0.0, -0.5, 0.0),
                mass: 0.0,
                friction: 0.5,
                shape: ShapeDesc::Box {
                    half_extents: Vec3::new(50.0, 0.5, 50.0),
                },
                ..Default::default()
            });
        }
    }

    // Web scene_grid_boxes geometry (jitter removed for determinism). Horizontal
    // gap fixed at 0.08 (the web scene); vertical gap + clearance parameterized.
    let side = 0.42_f32;
    let hpitch = side + 0.08;
    let vpitch = side + spec.vgap;
    let half = side * 0.5;
    let ox = -((spec.nx - 1) as f32 * hpitch) * 0.5;
    let oz = -((spec.nz - 1) as f32 * hpitch) * 0.5;
    let base_y = half + spec.clearance;

    let mut tracked = Vec::new();
    for j in 0..spec.ny {
        for i in 0..spec.nx {
            for k in 0..spec.nz {
                let x = ox + i as f32 * hpitch;
                let y = base_y + j as f32 * vpitch;
                let z = oz + k as f32 * hpitch;
                tracked.push(add_tracked_body(
                    &mut world,
                    "box",
                    RigidBodyDesc {
                        position: Vec3::new(x, y, z),
                        mass: 1.0,
                        friction: 0.5,
                        shape: ShapeDesc::Box {
                            half_extents: Vec3::splat(half),
                        },
                        ..Default::default()
                    },
                ));
            }
        }
    }
    Some((world, tracked, gravity))
}

/// Trajectory summary: max radius, max speed, mean/max height over time. Returns
/// the peak speed seen across all sampled frames (the explosion signature).
fn run(label: &str, spec: GridSpec, cfg: SimConfig, steps: usize) -> f32 {
    let (iters, substeps) = (cfg.solver_iterations, cfg.substeps);
    let Some((mut world, tracked, _gravity)) = build_grid(spec, cfg) else {
        eprintln!("SKIP: No GPU adapter found");
        return 0.0;
    };
    let n = tracked.len();
    println!(
        "\n=== {label}: {}x{}x{}={n} boxes (vgap={:.2} clearance={:.2} iters={} substeps={}) ===",
        spec.nx, spec.ny, spec.nz, spec.vgap, spec.clearance, iters, substeps
    );
    println!("step,max_radius_xz,max_speed,mean_y,max_y");

    let mut peak_speed = 0.0f32;
    for s in 0..steps {
        world.step();
        let snaps = snapshot_all(&world, &tracked);
        let mut max_speed = 0.0f32;
        for b in &snaps {
            max_speed = max_speed.max(b.linear_velocity.length());
        }
        peak_speed = peak_speed.max(max_speed);
        if s % 30 == 0 || s == steps - 1 {
            let mut max_r = 0.0f32;
            let mut sum_y = 0.0f32;
            let mut max_y = f32::NEG_INFINITY;
            for b in &snaps {
                let r = (b.position.x * b.position.x + b.position.z * b.position.z).sqrt();
                max_r = max_r.max(r);
                sum_y += b.position.y;
                max_y = max_y.max(b.position.y);
            }
            println!(
                "{},{:.2},{:.2},{:.2},{:.2}",
                s,
                max_r,
                max_speed,
                sum_y / n as f32,
                max_y
            );
        }
    }
    println!("PEAK SPEED: {peak_speed:.2} m/s");
    peak_speed
}

fn default_cfg() -> SimConfig {
    SimConfig::default()
}

fn cfg_with(iters: u32, substeps: u32) -> SimConfig {
    SimConfig {
        solver_iterations: iters,
        substeps,
        // Diagnostics for the fixed-substep ladders want EXACTLY `substeps`,
        // not adaptive escalation.
        max_substeps: substeps,
        ..Default::default()
    }
}

fn cfg_adaptive(max_substeps: u32, travel: f32) -> SimConfig {
    SimConfig {
        max_substeps,
        max_substep_travel: travel,
        ..Default::default()
    }
}

fn cfg_floor(floor: u32, max_substeps: u32, travel: f32) -> SimConfig {
    SimConfig {
        substeps: floor,
        max_substeps: max_substeps.max(floor),
        max_substep_travel: travel,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Baseline (from prior investigation): floor type irrelevant, depth triggers.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "diagnostic instrumentation, not a gate"]
fn grid_explosion_baseline() {
    let dropped = |nx, ny, nz, floor| GridSpec {
        nx,
        ny,
        nz,
        floor,
        vgap: 0.08,
        clearance: 0.03,
    };
    run(
        "plane shallow",
        dropped(6, 6, 6, Floor::Plane),
        default_cfg(),
        300,
    );
    run(
        "box shallow",
        dropped(6, 6, 6, Floor::Box),
        default_cfg(),
        300,
    );
    run(
        "plane deep",
        dropped(6, 24, 6, Floor::Plane),
        default_cfg(),
        300,
    );
    run(
        "box deep",
        dropped(6, 24, 6, Floor::Box),
        default_cfg(),
        300,
    );
}

// ---------------------------------------------------------------------------
// T1 — Statics vs. impact. Deep grid spawned pre-settled (zero gaps, touching).
// Stable => failure is purely the impact transient (H1/H3). Explodes => static
// load-propagation problem in the solver (much more serious).
// ---------------------------------------------------------------------------

#[test]
#[ignore = "diagnostic instrumentation, not a gate"]
fn t1_statics_vs_impact() {
    let settled = GridSpec {
        nx: 6,
        ny: 24,
        nz: 6,
        floor: Floor::Plane,
        vgap: 0.0,
        clearance: 0.0,
    };
    let dropped = GridSpec {
        vgap: 0.08,
        clearance: 0.03,
        ..settled
    };
    let s = run("T1 pre-settled deep", settled, default_cfg(), 300);
    let d = run("T1 dropped deep (control)", dropped, default_cfg(), 300);
    println!("\nT1 VERDICT: pre-settled peak={s:.1} m/s, dropped peak={d:.1} m/s");
}

// ---------------------------------------------------------------------------
// T2 — Minimal repro. Single 1x24x1 column with the drop gaps. Explodes => a
// ~24-body repro. Stable => lateral neighbor contacts are essential.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "diagnostic instrumentation, not a gate"]
fn t2_minimal_column() {
    let col = GridSpec {
        nx: 1,
        ny: 24,
        nz: 1,
        floor: Floor::Plane,
        vgap: 0.08,
        clearance: 0.03,
    };
    run("T2 single column dropped", col, default_cfg(), 300);
}

// ---------------------------------------------------------------------------
// T3 — Iteration ladder. NY=24 at 12/24/48/96 iterations. Explosion vanishing
// with iterations => convergence failure (H2). Persisting at 96 => not.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "diagnostic instrumentation, not a gate"]
fn t3_iteration_ladder() {
    let deep = GridSpec {
        nx: 6,
        ny: 24,
        nz: 6,
        floor: Floor::Plane,
        vgap: 0.08,
        clearance: 0.03,
    };
    let mut results = Vec::new();
    for iters in [12u32, 24, 48, 96] {
        let peak = run(&format!("T3 iters={iters}"), deep, cfg_with(iters, 1), 300);
        results.push((iters, peak));
    }
    println!("\nT3 VERDICT (iters -> peak speed):");
    for (i, p) in results {
        println!("  {i:>3} iters -> {p:.1} m/s");
    }
}

// ---------------------------------------------------------------------------
// T4 — Substep ladder. substeps 1/2/4 (halves/quarters effective dt). If 2
// substeps fixes it, confirms the impact-timescale mechanisms (H1/H3).
// ---------------------------------------------------------------------------

#[test]
#[ignore = "diagnostic instrumentation, not a gate"]
fn t4_substep_ladder() {
    let deep = GridSpec {
        nx: 6,
        ny: 24,
        nz: 6,
        floor: Floor::Plane,
        vgap: 0.08,
        clearance: 0.03,
    };
    let mut results = Vec::new();
    for substeps in [1u32, 2, 4] {
        let peak = run(
            &format!("T4 substeps={substeps}"),
            deep,
            cfg_with(12, substeps),
            300,
        );
        results.push((substeps, peak));
    }
    println!("\nT4 VERDICT (substeps -> peak speed):");
    for (s, p) in results {
        println!("  {s} substeps -> {p:.1} m/s");
    }
}

// ---------------------------------------------------------------------------
// T5 — Depth ladder. NY = 6/10/14/18/24. Sharp threshold => chain-length /
// convergence; gradual => accumulated impact energy.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "diagnostic instrumentation, not a gate"]
fn t5_depth_ladder() {
    let mut results = Vec::new();
    for ny in [6usize, 10, 14, 18, 24] {
        let spec = GridSpec {
            nx: 6,
            ny,
            nz: 6,
            floor: Floor::Plane,
            vgap: 0.08,
            clearance: 0.03,
        };
        let peak = run(&format!("T5 ny={ny}"), spec, default_cfg(), 300);
        results.push((ny, peak));
    }
    println!("\nT5 VERDICT (depth -> peak speed):");
    for (ny, p) in results {
        println!("  ny={ny:>2} -> {p:.1} m/s");
    }
}

// ---------------------------------------------------------------------------
// T6 — Frame-zero energy audit. Per-frame total energy, max penetration depth,
// contact count, to pinpoint the exact frame injection starts.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// FIX CHECK — does substeps=4 stabilize the three failing regimes (pre-settled
// static deep, single column, dropped deep)? Confirms substeps are load-bearing
// for the STATIC case too (not just the impact transient), which decides whether
// a velocity-triggered adaptive scheme is sufficient.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "diagnostic instrumentation, not a gate"]
fn fix_check_substeps4() {
    let presettled = GridSpec {
        nx: 6,
        ny: 24,
        nz: 6,
        floor: Floor::Plane,
        vgap: 0.0,
        clearance: 0.0,
    };
    let column = GridSpec {
        nx: 1,
        ny: 24,
        nz: 1,
        floor: Floor::Plane,
        vgap: 0.08,
        clearance: 0.03,
    };
    let dropped = GridSpec {
        vgap: 0.08,
        clearance: 0.03,
        ..presettled
    };
    let ps = run(
        "FIX pre-settled deep (ss=4)",
        presettled,
        cfg_with(12, 4),
        300,
    );
    let cl = run("FIX single column (ss=4)", column, cfg_with(12, 4), 300);
    let dr = run("FIX dropped deep (ss=4)", dropped, cfg_with(12, 4), 300);
    println!("\nFIX-CHECK VERDICT (ss=4): pre-settled={ps:.1} column={cl:.1} dropped={dr:.1} m/s");
}

// ---------------------------------------------------------------------------
// SUBSTEP INSTRUMENTATION — print the actual substep count chosen per frame so
// we can tell whether adaptive escalation is happening, and isolate floor vs.
// escalation as the load-bearing variable.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "diagnostic instrumentation, not a gate"]
fn substep_choice_probe() {
    let dropped = GridSpec {
        nx: 6,
        ny: 24,
        nz: 6,
        floor: Floor::Plane,
        vgap: 0.08,
        clearance: 0.03,
    };
    // Pure adaptive: floor 1, escalate to 8 at travel 0.01.
    let cfg = cfg_adaptive(8, 0.01);
    let Some((mut world, tracked, _g)) = build_grid(dropped, cfg) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };
    println!("\n=== substep probe: pure adaptive floor=1 cap=8 travel=0.01 ===");
    println!("step,substeps,max_speed");
    let mut peak = 0.0f32;
    for s in 0..120 {
        world.step();
        let ss = world.last_substeps();
        let snaps = snapshot_all(&world, &tracked);
        let v = snaps
            .iter()
            .map(|b| b.linear_velocity.length())
            .fold(0.0f32, f32::max);
        peak = peak.max(v);
        if s % 5 == 0 {
            println!("{s},{ss},{v:.2}");
        }
    }
    println!("PEAK SPEED (120): {peak:.2}");
}

// ---------------------------------------------------------------------------
// ADAPTIVE TUNING — sweep max_substep_travel to find the threshold that matches
// fixed-substeps=4 stability while keeping the escalation cap at 8. Tests the
// hardest regimes: pre-settled static deep + dropped deep + single column.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "diagnostic instrumentation, not a gate"]
fn adaptive_travel_sweep() {
    let presettled = GridSpec {
        nx: 6,
        ny: 24,
        nz: 6,
        floor: Floor::Plane,
        vgap: 0.0,
        clearance: 0.0,
    };
    let dropped = GridSpec {
        vgap: 0.08,
        clearance: 0.03,
        ..presettled
    };
    let column = GridSpec {
        nx: 1,
        ny: 24,
        nz: 1,
        floor: Floor::Plane,
        vgap: 0.08,
        clearance: 0.03,
    };
    let mut rows = Vec::new();
    for travel in [0.02f32, 0.015, 0.01] {
        let ps = run(
            &format!("ADAPT t={travel} pre-settled"),
            presettled,
            cfg_adaptive(8, travel),
            300,
        );
        let dr = run(
            &format!("ADAPT t={travel} dropped"),
            dropped,
            cfg_adaptive(8, travel),
            300,
        );
        let cl = run(
            &format!("ADAPT t={travel} column"),
            column,
            cfg_adaptive(8, travel),
            300,
        );
        rows.push((travel, ps, dr, cl));
    }
    println!("\nADAPTIVE SWEEP VERDICT (travel -> pre-settled / dropped / column peak m/s):");
    for (t, ps, dr, cl) in rows {
        println!("  t={t:.3} -> {ps:.1} / {dr:.1} / {cl:.1}");
    }
}

// ---------------------------------------------------------------------------
// FLOOR LADDER — minimum sustained substep floor (with escalation to 8) that
// gives LONG-TERM stability of the 24-deep stack, static + dropped, over 300
// frames. Finds the cheapest reliable default.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "diagnostic instrumentation, not a gate"]
fn floor_ladder() {
    let presettled = GridSpec {
        nx: 6,
        ny: 24,
        nz: 6,
        floor: Floor::Plane,
        vgap: 0.0,
        clearance: 0.0,
    };
    let dropped = GridSpec {
        vgap: 0.08,
        clearance: 0.03,
        ..presettled
    };
    let mut rows = Vec::new();
    for floor in [1u32, 2, 3, 4] {
        let ps = run(
            &format!("FLOOR {floor} pre-settled"),
            presettled,
            cfg_floor(floor, 8, 0.02),
            300,
        );
        let dr = run(
            &format!("FLOOR {floor} dropped"),
            dropped,
            cfg_floor(floor, 8, 0.02),
            300,
        );
        rows.push((floor, ps, dr));
    }
    println!("\nFLOOR LADDER VERDICT (floor -> pre-settled / dropped peak m/s over 300 frames):");
    for (f, ps, dr) in rows {
        println!("  floor={f} -> {ps:.1} / {dr:.1}");
    }
}

// ---------------------------------------------------------------------------
// ENERGY-GAIN MEASUREMENT — the pyramid injection metric (max rise of total
// energy KE+PE above its running minimum) for the grid across configs. This
// separates the BUG (solver adds energy -> boxes fired upward) from expected
// physics (under-converged deep stack slowly collapses, converting PE->KE).
// ---------------------------------------------------------------------------

fn energy_gain(spec: GridSpec, cfg: SimConfig, steps: usize) -> (f32, f32, f32) {
    let gravity = cfg.gravity;
    let Some((mut world, tracked, _g)) = build_grid(spec, cfg) else {
        eprintln!("SKIP: No GPU adapter found");
        return (0.0, 0.0, 0.0);
    };
    let mut running_min = f32::INFINITY;
    let mut max_gain = 0.0f32;
    let mut peak_speed = 0.0f32;
    let mut final_ke = 0.0f32;
    for _ in 0..steps {
        world.step();
        let snaps = snapshot_all(&world, &tracked);
        let m = compute_metrics(&snaps, gravity);
        let ke = m.translational_ke + m.rotational_ke;
        running_min = running_min.min(m.total_energy);
        max_gain = max_gain.max(m.total_energy - running_min);
        peak_speed = peak_speed.max(m.max_speed);
        final_ke = ke;
    }
    (max_gain, peak_speed, final_ke)
}

#[test]
#[ignore = "diagnostic instrumentation, not a gate"]
fn energy_gain_by_config() {
    let dropped = GridSpec {
        nx: 6,
        ny: 24,
        nz: 6,
        floor: Floor::Plane,
        vgap: 0.08,
        clearance: 0.03,
    };
    let configs: [(&str, SimConfig); 4] = [
        ("substeps=1 (no adaptive)", cfg_with(12, 1)),
        ("adaptive floor=1 cap=8", cfg_adaptive(8, 0.03)),
        ("adaptive floor=1 cap=8 t=0.01", cfg_adaptive(8, 0.01)),
        ("adaptive floor=4 cap=8", cfg_floor(4, 8, 0.02)),
    ];
    let mut rows = Vec::new();
    for (label, cfg) in configs {
        let (gain, peak, fke) = energy_gain(dropped, cfg, 300);
        rows.push((label, gain, peak, fke));
    }
    println!(
        "\nENERGY-GAIN VERDICT (config -> max_total_energy_gain / peak_speed / final_ke over 300):"
    );
    for (label, gain, peak, fke) in rows {
        println!("  {label:32} -> gain={gain:8.1} J  peak={peak:5.1}  final_ke={fke:8.1} J");
    }
}

// ---------------------------------------------------------------------------
// PYRAMID REGRESSION PROBE — does substepping destabilize the (already-stable)
// pyramid? Tests substeps=1 (baseline), fixed 2/4, and adaptive, measuring the
// settle-tail energy gain. Tells us whether substeps are a safe per-scene knob
// or introduce their own injection.
// ---------------------------------------------------------------------------

fn build_pyramid_local(cfg: SimConfig) -> Option<(World, Vec<TrackedBody3D>, Vec3)> {
    const SIZE: usize = 16;
    let gravity = cfg.gravity;
    let mut world = try_world(SimConfig {
        max_bodies: 256,
        ..cfg
    })?;
    let mut tracked = Vec::new();
    tracked.push(add_tracked_body(
        &mut world,
        "floor",
        RigidBodyDesc {
            position: Vec3::new(0.0, -0.5, 0.0),
            mass: 0.0,
            friction: 0.5,
            shape: ShapeDesc::Box {
                half_extents: Vec3::new(50.0, 0.5, 50.0),
            },
            ..Default::default()
        },
    ));
    for y in 0..SIZE {
        for x in 0..(SIZE - y) {
            tracked.push(add_tracked_body(
                &mut world,
                "box",
                RigidBodyDesc {
                    position: Vec3::new(
                        x as f32 * 1.01 + y as f32 * 0.5 - SIZE as f32 / 2.0,
                        y as f32 * 0.85 + 0.5,
                        0.0,
                    ),
                    mass: 1.0,
                    friction: 0.5,
                    shape: ShapeDesc::Box {
                        half_extents: Vec3::new(0.5, 0.25, 0.25),
                    },
                    ..Default::default()
                },
            ));
        }
    }
    Some((world, tracked, gravity))
}

fn pyramid_energy_gain(cfg: SimConfig) -> (f32, f32) {
    let gravity = cfg.gravity;
    let Some((mut world, tracked, _g)) = build_pyramid_local(cfg) else {
        eprintln!("SKIP: No GPU adapter found");
        return (0.0, 0.0);
    };
    let mut running_min = f32::INFINITY;
    let mut max_gain = 0.0f32;
    let mut final_ke = 0.0f32;
    for i in 0..600 {
        world.step();
        let snaps = snapshot_all(&world, &tracked);
        let m = compute_metrics(&snaps, gravity);
        if i >= 420 {
            running_min = running_min.min(m.total_energy);
            max_gain = max_gain.max(m.total_energy - running_min);
            final_ke = m.translational_ke + m.rotational_ke;
        }
    }
    (max_gain, final_ke)
}

#[test]
#[ignore = "diagnostic instrumentation, not a gate"]
fn pyramid_regression_probe() {
    let configs: [(&str, SimConfig); 4] = [
        ("substeps=1 (baseline)", cfg_with(12, 1)),
        ("fixed substeps=2", cfg_with(12, 2)),
        ("fixed substeps=4", cfg_with(12, 4)),
        ("adaptive floor=1 cap=8 t=0.01", cfg_adaptive(8, 0.01)),
    ];
    let mut rows = Vec::new();
    for (label, cfg) in configs {
        let (gain, fke) = pyramid_energy_gain(cfg);
        rows.push((label, gain, fke));
    }
    println!("\nPYRAMID PROBE VERDICT (config -> settle-tail energy_gain / final_ke):");
    for (label, gain, fke) in rows {
        println!("  {label:32} -> gain={gain:8.2} J  final_ke={fke:8.2} J");
    }
}

#[test]
#[ignore = "diagnostic instrumentation, not a gate"]
fn t6_energy_audit() {
    let deep = GridSpec {
        nx: 6,
        ny: 24,
        nz: 6,
        floor: Floor::Plane,
        vgap: 0.08,
        clearance: 0.03,
    };
    let cfg = default_cfg();
    let Some((mut world, tracked, gravity)) = build_grid(deep, cfg) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };
    println!("\n=== T6 energy audit: 6x24x6 dropped deep ===");
    println!("step,total_energy,d_total,ke,pe,max_speed,contacts,max_depth");
    let mut prev_total = f32::NAN;
    for s in 0..300 {
        world.step();
        let snaps = snapshot_all(&world, &tracked);
        let m = compute_metrics(&snaps, gravity);
        let contacts = world.fetch_last_contacts().to_vec();
        let max_depth = contacts
            .iter()
            .map(|c| c.depth().abs())
            .fold(0.0f32, f32::max);
        let total = m.total_energy;
        let d_total = if prev_total.is_nan() {
            0.0
        } else {
            total - prev_total
        };
        // Print every frame in the first second (impact window), then decimate.
        if s < 90 || s % 10 == 0 {
            println!(
                "{},{:.2},{:+.2},{:.2},{:.2},{:.2},{},{:.4}",
                s,
                total,
                d_total,
                m.translational_ke + m.rotational_ke,
                m.potential_energy,
                m.max_speed,
                contacts.len(),
                max_depth
            );
        }
        prev_total = total;
    }
}
