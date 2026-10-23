//! Acceptance test: lab balls never phase through the fractured tower.
//!
//! The vibe-city `fracturedTower` pack (3,711 Voronoi pieces, 15 floors) in the
//! fracture lab's own step plan on Metal, with CCD balls (`Planner::set_projectile_ccd`,
//! the lab's "CCD balls" checkbox). Balls are fired at the tower from its four sides
//! at three heights, at 1×, 4×, 16× (the lab's Ball speed maximum) and 64× the scene's
//! shot speed (up to ~1,150 m/s).
//!
//! Phase-through, exactly: in some step, the ball's centre path (the segment from its
//! pose before the step to its pose after) meets a piece that did not move during
//! that step (GJK distance between the segment and the piece's hull is zero). A ball
//! that hits a piece either stops at its surface (its centre stays a radius away) or
//! moves it; a ball whose centre crosses a still piece went through it without a
//! contact. Every shot must have none. A control without CCD must have some (the
//! detector is sensitive).
//!
//! Needs the generated pack: `bun run export.ts fracturedTower` in
//! engine/scenes/vibe-city (skips with a message when missing).
//!
//!   cargo test --release -p engine-viewer --test fracture_tower_ccd -- --nocapture

use engine_fracture::pack::build_pack_scene;
use engine_fracture::scene::{collider_vertices, Scene, SHAPE_CONVEX};
use engine_viewer::fracture::FractureSim;
use metal::Device;
use std::path::PathBuf;

type V = [f64; 3];
fn sub(a: V, b: V) -> V {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: V, b: V) -> V {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn mul(a: V, s: f64) -> V {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: V, b: V) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Closest point to the origin of the simplex `pts` (1–4 points): every face
/// subset, keeping only affine combinations with non-negative weights.
fn simplex_closest(pts: &[V]) -> (V, Vec<usize>) {
    let n = pts.len();
    let mut best: Option<(f64, V, Vec<usize>)> = None;
    for mask in 1u32..(1 << n) {
        let idx: Vec<usize> = (0..n).filter(|i| mask & (1 << i) != 0).collect();
        // Solve min |sum w_i p_i| with sum w_i = 1: Gram system on p_i - p_0.
        let p0 = pts[idx[0]];
        let e: Vec<V> = idx[1..].iter().map(|&i| sub(pts[i], p0)).collect();
        let m = e.len();
        let mut a = [[0.0f64; 3]; 3];
        let mut rhs = [0.0f64; 3];
        for r in 0..m {
            for c in 0..m {
                a[r][c] = dot(e[r], e[c]);
            }
            rhs[r] = -dot(e[r], p0);
        }
        // Gaussian elimination (m <= 3).
        let mut w = [0.0f64; 3];
        let mut ok = true;
        let mut aa = a;
        let mut bb = rhs;
        for col in 0..m {
            let piv = (col..m).max_by(|&x, &y| aa[x][col].abs().total_cmp(&aa[y][col].abs())).unwrap();
            if aa[piv][col].abs() < 1e-14 {
                ok = false;
                break;
            }
            aa.swap(col, piv);
            bb.swap(col, piv);
            for r in 0..m {
                if r != col {
                    let f = aa[r][col] / aa[col][col];
                    for c in 0..m {
                        aa[r][c] -= f * aa[col][c];
                    }
                    bb[r] -= f * bb[col];
                }
            }
        }
        if !ok {
            continue;
        }
        for r in 0..m {
            w[r] = bb[r] / aa[r][r];
        }
        let w0 = 1.0 - w[..m].iter().sum::<f64>();
        if w0 < -1e-12 || w[..m].iter().any(|&x| x < -1e-12) {
            continue;
        }
        let mut p = mul(p0, w0);
        for r in 0..m {
            p = add(p, mul(pts[idx[r + 1]], w[r]));
        }
        let d = dot(p, p);
        if best.as_ref().is_none_or(|b| d < b.0) {
            best = Some((d, p, idx));
        }
    }
    let (_, p, idx) = best.expect("a vertex is always a candidate");
    (p, idx)
}

/// GJK distance between the segment [s0, s1] and the convex hull of `verts`.
fn segment_hull_distance(s0: V, s1: V, verts: &[V]) -> f64 {
    // Minkowski difference hull − segment: support(d) = max_v v·d − min_s s·d.
    let support = |d: V| -> V {
        let v = *verts.iter().max_by(|a, b| dot(**a, d).total_cmp(&dot(**b, d))).unwrap();
        let s = if dot(s0, d) < dot(s1, d) { s0 } else { s1 };
        sub(v, s)
    };
    let mut simplex = vec![support([1.0, 0.0, 0.0])];
    let mut closest = simplex[0];
    for _ in 0..64 {
        let dist2 = dot(closest, closest);
        if dist2 < 1e-18 {
            return 0.0;
        }
        let p = support(mul(closest, -1.0));
        // No progress toward the origin: `closest` is the answer.
        if dist2 - dot(closest, p) <= 1e-12 * dist2.max(1.0) {
            return dist2.sqrt();
        }
        simplex.push(p);
        let (c, keep) = simplex_closest(&simplex);
        simplex = keep.iter().map(|&i| simplex[i]).collect();
        closest = c;
        if simplex.len() == 4 {
            return 0.0; // the origin is inside the tetrahedron
        }
    }
    dot(closest, closest).sqrt()
}

#[test]
fn segment_hull_distance_is_exact_on_a_cube() {
    let cube: Vec<V> = (0..8).map(|i| [(i & 1) as f64, ((i >> 1) & 1) as f64, ((i >> 2) & 1) as f64]).collect();
    let d = |a: V, b: V| segment_hull_distance(a, b, &cube);
    assert!(d([-1.0, 0.5, 0.5], [2.0, 0.5, 0.5]) < 1e-9, "through the cube");
    assert!((d([-1.0, 2.0, 0.5], [2.0, 2.0, 0.5]) - 1.0).abs() < 1e-9, "1 m above it");
    assert!((d([-3.0, 0.5, 0.5], [-2.0, 0.5, 0.5]) - 2.0).abs() < 1e-9, "2 m in front");
    assert!(d([0.2, 0.2, 0.2], [0.3, 0.3, 0.3]) < 1e-9, "inside");
    let diag = d([2.0, 2.0, 0.5], [3.0, 3.0, 0.5]);
    assert!((diag - 2f64.sqrt()).abs() < 1e-9, "diagonal {diag}");
}

/// One convex piece: its body and hull (local vertices, as `collider_vertices` builds
/// them at a pose).
struct Piece {
    body: usize,
    collider: usize,
}

fn pose(snap: &[f32], b: usize) -> (V, [f64; 4]) {
    let o = b * 28;
    let f = |k: usize| snap[o + k] as f64;
    ([f(0), f(1), f(2)], [f(4), f(5), f(6), f(7)])
}

struct Shot {
    origin: V,
    dir: V,
    label: String,
}

#[derive(Default, Debug)]
struct Outcome {
    phase_throughs: usize,
    first: Option<String>,
    max_depth: f64,
    hit: bool,
}

fn run_shot(sim: &mut FractureSim, scene: &Scene, pieces: &[Piece], shot: &Shot, n: usize, steps: usize) -> Outcome {
    let start = scene.projectile_start as usize;
    let r = scene.projectile_radius;
    let id = sim.launch(shot.origin, shot.dir).unwrap() as usize;
    let mut prev = sim.snapshot(n).unwrap();
    let mut out = Outcome::default();
    for step in 0..steps {
        sim.step(1).unwrap();
        assert_eq!(sim.last.overflow, 0, "{}: GPU capacity overflow", shot.label);
        let now = sim.snapshot(n).unwrap();
        let (b0, _) = pose(&prev, id);
        let (b1, _) = pose(&now, id);
        let (lo, hi) = ([0, 1, 2].map(|k| b0[k].min(b1[k])), [0, 1, 2].map(|k| b0[k].max(b1[k])));
        for p in pieces {
            let (x0, q0) = pose(&prev, p.body);
            let (x1, q1) = pose(&now, p.body);
            let moved = dot(sub(x1, x0), sub(x1, x0)).sqrt() > 1e-3
                || (q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3]).abs() < 1.0 - 1e-6;
            if moved {
                out.hit |= p.body >= 1 && p.body < start;
                continue;
            }
            let c = &scene.colliders[p.collider];
            let verts = collider_vertices(&scene.hull_data, c, x0, q0);
            // AABB prefilter (the segment's box against the hull's).
            let mut vlo = [f64::INFINITY; 3];
            let mut vhi = [f64::NEG_INFINITY; 3];
            for v in &verts {
                for k in 0..3 {
                    vlo[k] = vlo[k].min(v[k]);
                    vhi[k] = vhi[k].max(v[k]);
                }
            }
            if (0..3).any(|k| hi[k] < vlo[k] || lo[k] > vhi[k]) {
                continue;
            }
            let d = segment_hull_distance(b0, b1, &verts);
            if d <= 1e-6 {
                out.phase_throughs += 1;
                if out.first.is_none() {
                    out.first = Some(format!(
                        "step {step}: ball centre {:?} -> {:?} crossed still piece {} (hull centre {:?})",
                        b0.map(|x| (x * 100.0).round() / 100.0),
                        b1.map(|x| (x * 100.0).round() / 100.0),
                        p.body,
                        x0.map(|x| (x * 100.0).round() / 100.0),
                    ));
                }
            } else if d < r {
                out.max_depth = out.max_depth.max(r - d);
            }
        }
        prev = now;
        // Stopped, or far past the tower: nothing more to check.
        let v = &prev[id * 28 + 12..id * 28 + 15];
        if (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() < 0.5 {
            break;
        }
    }
    out
}

fn tower_shots(scene: &Scene) -> (Vec<Shot>, V, V) {
    let start = scene.projectile_start as usize;
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for b in &scene.bodies[1..start] {
        for k in 0..3 {
            lo[k] = lo[k].min(b.p[k]);
            hi[k] = hi[k].max(b.p[k]);
        }
    }
    let c = [0, 1, 2].map(|k| 0.5 * (lo[k] + hi[k]));
    let back = 0.5 * (hi[0] - lo[0]).max(hi[2] - lo[2]) + 6.0;
    let height = |f: f64| lo[1] + f * (hi[1] - lo[1]);
    let norm = |d: V| mul(d, 1.0 / dot(d, d).sqrt());
    let mut shots = Vec::new();
    // Four sides at three heights, aimed horizontally at the tower's axis.
    for (side, d) in [("+x", [-1.0, 0.0, 0.0]), ("-x", [1.0, 0.0, 0.0]), ("+z", [0.0, 0.0, -1.0]), ("-z", [0.0, 0.0, 1.0])] {
        for (h, f) in [("low", 0.12), ("mid", 0.5), ("high", 0.88)] {
            let y = height(f);
            shots.push(Shot { origin: [c[0] - d[0] * back, y, c[2] - d[2] * back], dir: d, label: format!("{side} {h}") });
        }
    }
    // Four corners (edges and column pieces), mid height.
    for (name, d) in [("+x+z", [-1.0, 0.0, -1.0]), ("-x+z", [1.0, 0.0, -1.0]), ("+x-z", [-1.0, 0.0, 1.0]), ("-x-z", [1.0, 0.0, 1.0])] {
        let d = norm(d);
        let o = [c[0] - d[0] * back * 1.42, height(0.5), c[2] - d[2] * back * 1.42];
        shots.push(Shot { origin: o, dir: d, label: format!("corner {name}") });
    }
    // Plunging 45° into the roof and the top floors.
    for (name, d) in [("roof +x", [-1.0, -1.0, 0.0]), ("roof -z", [0.0, -1.0, 1.0])] {
        let d = norm(d);
        let top = [c[0], hi[1], c[2]];
        shots.push(Shot { origin: sub(top, mul(d, back)), dir: d, label: name.into() });
    }
    (shots, lo, hi)
}

fn tower_scene() -> Option<(Scene, usize)> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../engine-webgpu/generated/vibe-city/fracturedTower.json");
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!("skip: {} missing (bun run export.ts fracturedTower in engine/scenes/vibe-city)", path.display());
        return None;
    };
    let (scene, info) = build_pack_scene(&bytes).unwrap();
    Some((scene, info.nodes))
}

/// Every shot of `scales` × the scene's shot speed against the tower; returns the
/// phase-throughs per shot (CCD) and in total (control, CCD off).
fn tower_run(label: &str, scene: &Scene, ccd: bool, scales: &[f64], only: usize) -> (Vec<String>, usize) {
    let start = scene.projectile_start as usize;
    let n = scene.bodies.len();
    let pieces: Vec<Piece> = scene
        .colliders
        .iter()
        .enumerate()
        .filter(|(_, c)| c.shape == SHAPE_CONVEX && (c.body as usize) >= 1 && (c.body as usize) < start)
        .map(|(i, c)| Piece { body: c.body as usize, collider: i })
        .collect();
    let (shots, lo, hi) = tower_shots(scene);
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let mut sim = FractureSim::new(&device, &queue, scene).unwrap();
    let base_speed = sim.plan.shot_speed();
    let mut failures = Vec::new();
    let mut total = 0;
    for &scale in scales {
        for shot in shots.iter().take(only) {
            sim.reset();
            sim.step(2).unwrap();
            sim.plan.set_projectile_ccd(ccd);
            sim.plan.set_projectile_speed_scale(scale).unwrap();
            let speed = base_speed * scale;
            // Long enough to cross the tower at the shot's speed (capped).
            let span = (hi[0] - lo[0]).max(hi[1] - lo[1]) + 20.0;
            let steps = (span / (speed * scene.params.dt as f64)).ceil().clamp(30.0, 240.0) as usize;
            let o = run_shot(&mut sim, scene, &pieces, shot, n, steps);
            println!(
                "{label} ccd {ccd} ×{scale:>2} ({speed:>6.0} m/s) {:<12}: {} phase-throughs, deepest overlap {:.3} m{}",
                shot.label,
                o.phase_throughs,
                o.max_depth,
                o.first.as_ref().map(|f| format!(" — {f}")).unwrap_or_default()
            );
            total += o.phase_throughs;
            if ccd {
                assert!(o.hit, "{label} ×{scale} {}: the ball should hit the tower", shot.label);
                if o.phase_throughs > 0 {
                    failures.push(format!("{label} ×{scale} {}: {}", shot.label, o.first.unwrap()));
                }
            }
        }
    }
    (failures, total)
}

/// Bond mode none (the lab's default: pieces sleep until hit), 120 Hz: every shot at
/// 1×, 4×, 16× (the Ball speed maximum) and 64×. The control (CCD off, 64×) must
/// phase through, or the detector would not be sensitive.
#[test]
fn ccd_balls_never_phase_through_the_fractured_tower() {
    let Some((scene, nodes)) = tower_scene() else { return };
    let scene = scene.without_joints();
    println!(
        "fracturedTower: {nodes} nodes, ball radius {:.2} m, shot speed {:.1} m/s, dt {:.4}",
        scene.projectile_radius,
        engine_fracture::plan::default_shot_speed(scene.extent, scene.projectile_speed),
        scene.params.dt
    );
    let (_, control) = tower_run("none 120 Hz", &scene, false, &[64.0], 4);
    assert!(control > 0, "control: without CCD the 64× balls should phase through pieces (the detector is not sensitive)");
    let (failures, _) = tower_run("none 120 Hz", &scene, true, &[1.0, 4.0, 16.0, 64.0], usize::MAX);
    assert!(failures.is_empty(), "CCD balls phased through the tower:\n{}", failures.join("\n"));
}

/// The lab's Physics 60 Hz setting: twice the travel per step.
#[test]
fn ccd_balls_never_phase_through_the_fractured_tower_at_60_hz() {
    let Some((scene, _)) = tower_scene() else { return };
    let mut scene = scene.without_joints();
    scene.params.dt = 1.0 / 60.0;
    let (failures, _) = tower_run("none 60 Hz", &scene, true, &[4.0, 16.0, 64.0], usize::MAX);
    assert!(failures.is_empty(), "CCD balls phased through the tower:\n{}", failures.join("\n"));
}

/// Bond mode "Breakable D6 joints": the welded tower.
#[test]
fn ccd_balls_never_phase_through_the_welded_tower() {
    let Some((scene, _)) = tower_scene() else { return };
    let (failures, _) = tower_run("d6 120 Hz", &scene, true, &[4.0, 16.0, 64.0], usize::MAX);
    assert!(failures.is_empty(), "CCD balls phased through the tower:\n{}", failures.join("\n"));
}
