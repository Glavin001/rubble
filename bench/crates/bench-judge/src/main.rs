//! Engine-neutral quality of a benchmark run, from its pose dumps alone.
//!
//! `bench-judge --result <run>.json [--scene <id>.scene.json] [--out <run>.judge.json]`
//!
//! Every engine is measured by this one piece of code, on the same definitions:
//! - `initial`: checkpoint 0 against the scene file: did the engine load the scene we gave it?
//! - `nonfinite`, `escaped`: NaN/Inf; dynamic bodies through the floor (below the ground top
//!   − 1 m while over it) or 200 m above the scene. `off_edge` counts bodies pushed past the
//!   ground's edge (legitimate motion, reported only).
//! - `pen_*`: penetration depth of every touching pair (distance < 5 mm, joined pairs
//!   excluded), from exact shape-shape queries (parry3d) on the dumped poses. A body made of
//!   several convex hulls counts once per pair of bodies, at its deepest hull pair.
//! - `joint_gap_*`: distance between each ball joint's two world anchors.
//! - per group (`dynamic` or a scene group): drift from the initial pose, |Δy|, horizontal
//!   drift, mean position, speed, displacement.
//! Gates from the scene file are evaluated on these and the run passes when all of them do.

use anyhow::{bail, Context, Result};
use bench_core::protocol::read_poses;
use bench_core::{BodySample, Scene, Shape};
use parry3d::math::{Pose, Rotation, Vector};
use parry3d::shape::{Ball, ConvexPolyhedron, Cuboid, HalfSpace, Shape as ParryShape};
use rayon::prelude::*;
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

/// Pairs closer than this count as touching for the penetration statistics.
const TOUCH: f32 = 0.005;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let (mut result, mut scene, mut out) = (None, None, None);
    while let Some(a) = args.next() {
        let mut val = || args.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--result" => result = Some(PathBuf::from(val()?)),
            "--scene" => scene = Some(PathBuf::from(val()?)),
            "--out" => out = Some(PathBuf::from(val()?)),
            _ => bail!("unknown argument {a}; usage: bench-judge --result run.json [--scene s.scene.json] [--out run.judge.json]"),
        }
    }
    let result_path = result.context("--result is required")?;
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(&result_path)?)?;
    let scene_id = doc["scene"]["id"].as_str().context("result has no scene id")?;
    let scene_path = scene.unwrap_or_else(|| {
        let here = std::env::current_exe().ok().and_then(|p| p.parent().map(|p| p.to_path_buf())).unwrap_or_default();
        here.join(format!("../../scenes/out/{scene_id}.scene.json"))
    });
    let scene = Scene::load(&scene_path)?;
    if doc["scene"]["sha256"].as_str() != Some(scene.meta.bin_sha256.as_str()) {
        bail!("result was run on a different scene file than {}", scene_path.display());
    }
    let poses_path = result_path.with_file_name(doc["poses"].as_str().context("result has no pose file")?);
    let checkpoints = read_poses(&poses_path)?;
    let judge = Judge::new(&scene);
    let mut metrics = Vec::new();
    for (step, samples) in &checkpoints {
        if samples.len() != scene.bodies.len() {
            bail!("checkpoint {step}: {} bodies, scene has {}", samples.len(), scene.bodies.len());
        }
        let mut m = judge.measure(samples);
        if *step == 0 {
            m.insert("initial".into(), judge.initial_match(samples));
        }
        metrics.push((*step, m));
    }
    // A --smoke run proves the plumbing only: the scene loaded identically and stepping stayed finite
    let smoke = doc["protocol"]["smoke"].as_bool().unwrap_or(false);
    let gates = if smoke { Vec::new() } else { judge.gates(&metrics) };
    let finite_end = metrics.last().map(|(_, m)| m["nonfinite"].as_u64() == Some(0)).unwrap_or(false);
    let initial_ok = metrics
        .first()
        .and_then(|(s, m)| (*s == 0).then(|| m["initial"]["ok"].as_bool().unwrap_or(false)))
        .unwrap_or(false);
    let passed = initial_ok && if smoke { finite_end } else { gates.iter().all(|g| g["status"] == "pass") };
    let report = json!({
        "format": "rubble-bench-judge",
        "version": 1,
        "result": result_path.file_name().unwrap().to_string_lossy(),
        "scene": scene.meta.id,
        "initialMatches": initial_ok,
        "passed": passed,
        "smoke": smoke,
        "gates": gates,
        "checkpoints": metrics.iter().map(|(s, m)| json!({ "step": s, "metrics": m })).collect::<Vec<_>>(),
    });
    let out = out.unwrap_or_else(|| result_path.with_extension("judge.json"));
    std::fs::write(&out, serde_json::to_string_pretty(&report)? + "\n")?;
    let failed: Vec<String> = gates
        .iter()
        .filter(|g| g["status"] != "pass")
        .map(|g| format!("{}@{}{} = {} ({})", g["metric"].as_str().unwrap(), g["at"], g["group"].as_str().map(|s| format!("[{s}]")).unwrap_or_default(), g["value"], g["status"].as_str().unwrap()))
        .collect();
    eprintln!(
        "{} {}: {}{}{}",
        doc["engine"]["engine"].as_str().unwrap_or("?"),
        scene.meta.id,
        match (smoke, passed) {
            (true, true) => "SMOKE OK (loads identically, steps finite; no quality or timing claim)",
            (true, false) => "SMOKE FAIL",
            (false, true) => "PASS",
            (false, false) => "FAIL",
        },
        if initial_ok { "" } else { " (initial state differs from the scene)" },
        if failed.is_empty() { String::new() } else { format!(" — {}", failed.join("; ")) }
    );
    Ok(())
}

struct Judge<'a> {
    scene: &'a Scene,
    /// Each body's convex pieces (one for a box or sphere, one per hull for a compound).
    shapes: Vec<Vec<Box<dyn ParryShape>>>,
    dynamic: Vec<u32>,
    joined: HashSet<(u32, u32)>,
    floor_top: f32,
    ground_half: f32,
    top0: f32,
    cell: f32,
    large: Vec<u32>,
}

fn pose(s: &BodySample) -> Pose {
    Pose::from_parts(Vector::from_array(s.position), Rotation::from_xyzw(s.rotation[0], s.rotation[1], s.rotation[2], s.rotation[3]).normalize())
}

fn stats(mut v: Vec<f32>) -> (f64, f64, f64, usize) {
    if v.is_empty() {
        return (0.0, 0.0, 0.0, 0);
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    (v[n - 1] as f64, v[((n - 1) as f64 * 0.99).round() as usize] as f64, mean, n)
}

impl<'a> Judge<'a> {
    fn new(scene: &'a Scene) -> Self {
        let shapes: Vec<Vec<Box<dyn ParryShape>>> = scene
            .bodies
            .iter()
            .enumerate()
            .map(|(i, b)| -> Vec<Box<dyn ParryShape>> {
                match b.shape {
                    Shape::Box(h) => vec![Box::new(Cuboid::new(Vector::from_array(h)))],
                    Shape::Sphere(r) => vec![Box::new(Ball::new(r))],
                    Shape::Plane => vec![Box::new(HalfSpace::new(Vector::Y))],
                    Shape::Hulls { first, count, .. } => (first..first + count)
                        .map(|h| -> Box<dyn ParryShape> {
                            let pts: Vec<Vector> = scene.hulls[h as usize].iter().map(|v| Vector::from_array(*v)).collect();
                            Box::new(ConvexPolyhedron::from_convex_hull(&pts).unwrap_or_else(|| panic!("body {i}, hull {h}: not a solid convex hull")))
                        })
                        .collect(),
                }
            })
            .collect();
        let dynamic: Vec<u32> = (0..scene.bodies.len() as u32).filter(|&i| !scene.bodies[i as usize].is_static).collect();
        let joined = scene.joints.iter().filter(|j| !j.collide).map(|j| (j.body_a.min(j.body_b), j.body_a.max(j.body_b))).collect();
        // The ground: the static body with the widest footprint
        let ground = scene
            .bodies
            .iter()
            .filter(|b| b.is_static)
            .max_by(|a, b| a.radius().total_cmp(&b.radius()))
            .expect("scene has a static ground");
        let (floor_top, ground_half) = match ground.shape {
            Shape::Box(h) => (ground.position[1] + h[1], h[0].min(h[2])),
            Shape::Sphere(r) => (ground.position[1] + r, r),
            // Horizontal plane through its body position (the suite never tilts it)
            Shape::Plane => (ground.position[1], f32::INFINITY),
            Shape::Hulls { radius, .. } => (ground.position[1] + radius, radius),
        };
        let top0 = dynamic.iter().map(|&i| scene.bodies[i as usize].position[1]).fold(f32::MIN, f32::max);
        // Grid cell: twice the largest "small" radius; bigger bodies are tested against all
        let mut radii: Vec<f32> = scene.bodies.iter().map(|b| b.radius()).collect();
        radii.sort_by(|a, b| a.total_cmp(b));
        let median = radii[radii.len() / 2];
        let limit = 4.0 * median;
        let small_max = scene.bodies.iter().map(|b| b.radius()).filter(|&r| r <= limit).fold(0.0f32, f32::max);
        let large = (0..scene.bodies.len() as u32).filter(|&i| scene.bodies[i as usize].radius() > limit).collect();
        Self { scene, shapes, dynamic, joined, floor_top, ground_half, top0, cell: 2.0 * small_max.max(1e-3), large }
    }

    fn initial_match(&self, s: &[BodySample]) -> Value {
        let (mut dp, mut dq, mut dv) = (0.0f32, 0.0f32, 0.0f32);
        let mut worst = 0usize;
        for (i, (b, x)) in self.scene.bodies.iter().zip(s).enumerate() {
            let p = (0..3).map(|k| (b.position[k] - x.position[k]).abs()).fold(0.0, f32::max);
            // Rotation angle between the two, from |q1 ∓ q2| (well conditioned near zero, unlike acos)
            let diff = |sign: f32| (0..4).map(|k| (b.rotation[k] - sign * x.rotation[k]).powi(2)).sum::<f32>().sqrt();
            let q = 4.0 * (diff(1.0).min(diff(-1.0)) / 2.0).min(1.0).asin();
            let v = if b.is_static { 0.0 } else { (0..3).map(|k| (b.linvel[k] - x.linvel[k]).abs().max((b.angvel[k] - x.angvel[k]).abs())).fold(0.0, f32::max) };
            if p > dp {
                worst = i;
            }
            dp = dp.max(p);
            dq = dq.max(q);
            dv = dv.max(v);
        }
        // f32 scene values: allow a few ulps at 250 m and an engine's quaternion renormalisation
        let ok = dp <= 1e-4 && dq <= 1e-3 && dv <= 1e-3;
        json!({ "ok": ok, "maxPositionError": dp, "maxRotationErrorRad": dq, "maxVelocityError": dv, "worstBody": worst })
    }

    fn measure(&self, s: &[BodySample]) -> Map<String, Value> {
        let mut m = Map::new();
        let finite = |x: &BodySample| x.position.iter().chain(&x.rotation).chain(&x.linvel).chain(&x.angvel).all(|v| v.is_finite());
        let nonfinite = self.dynamic.iter().filter(|&&i| !finite(&s[i as usize])).count();
        // Escaped: through the floor (below its top − 1 m while over the ground) or flung 200 m
        // up. A body pushed off the ground's edge is legitimate motion, counted separately.
        let over_ground = |p: [f32; 3]| p[0].abs() <= self.ground_half && p[2].abs() <= self.ground_half;
        let escaped = self
            .dynamic
            .iter()
            .filter(|&&i| {
                let p = s[i as usize].position;
                !finite(&s[i as usize]) || (p[1] < self.floor_top - 1.0 && over_ground(p)) || p[1] > self.top0 + 200.0
            })
            .count();
        let off_edge = self.dynamic.iter().filter(|&&i| !over_ground(s[i as usize].position)).count();
        m.insert("nonfinite".into(), json!(nonfinite));
        m.insert("escaped".into(), json!(escaped));
        m.insert("off_edge".into(), json!(off_edge));
        let (pmax, p99, pmean, pairs) = stats(self.penetrations(s));
        m.insert("pen_max".into(), json!(pmax));
        m.insert("pen_p99".into(), json!(p99));
        m.insert("pen_mean".into(), json!(pmean));
        m.insert("touching_pairs".into(), json!(pairs));
        if !self.scene.joints.is_empty() {
            let gaps = self
                .scene
                .joints
                .iter()
                .map(|j| {
                    let a = pose(&s[j.body_a as usize]) * Vector::from_array(j.anchor_a);
                    let b = pose(&s[j.body_b as usize]) * Vector::from_array(j.anchor_b);
                    let d = (a - b).length();
                    if d.is_finite() { d } else { f32::MAX }
                })
                .collect();
            let (gmax, g99, gmean, _) = stats(gaps);
            m.insert("joint_gap_max".into(), json!(gmax));
            m.insert("joint_gap_p99".into(), json!(g99));
            m.insert("joint_gap_mean".into(), json!(gmean));
        }
        let mut groups = Map::new();
        groups.insert("dynamic".into(), Value::Object(self.group_metrics(&self.dynamic, s)));
        for (name, ids) in &self.scene.meta.groups {
            groups.insert(name.clone(), Value::Object(self.group_metrics(ids, s)));
        }
        m.insert("groups".into(), Value::Object(groups));
        // Kinetic energy of everything (J), with the scene's masses
        let ke: f64 = self
            .dynamic
            .iter()
            .map(|&i| {
                let (b, x) = (&self.scene.bodies[i as usize], &s[i as usize]);
                let q = Rotation::from_xyzw(x.rotation[0], x.rotation[1], x.rotation[2], x.rotation[3]).normalize();
                let wl = q.inverse() * Vector::from_array(x.angvel);
                let v2: f32 = x.linvel.iter().map(|v| v * v).sum();
                0.5 * (b.mass * v2 + b.inertia[0] * wl.x * wl.x + b.inertia[1] * wl.y * wl.y + b.inertia[2] * wl.z * wl.z) as f64
            })
            .sum();
        m.insert("ke".into(), json!(ke));
        m
    }

    fn group_metrics(&self, ids: &[u32], s: &[BodySample]) -> Map<String, Value> {
        let mut m = Map::new();
        if ids.is_empty() {
            return m;
        }
        let n = ids.len() as f64;
        let (mut max_drift, mut sum_drift, mut max_dy, mut sum_dy, mut max_h, mut sum_h) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
        let (mut max_speed, mut sum_speed, mut min_disp) = (0f64, 0f64, f64::MAX);
        let mut mean = [0f64; 3];
        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
        for &i in ids {
            let (b, x) = (&self.scene.bodies[i as usize], &s[i as usize]);
            let d: Vec<f64> = (0..3).map(|k| (x.position[k] - b.position[k]) as f64).collect();
            let drift = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let h = (d[0] * d[0] + d[2] * d[2]).sqrt();
            let speed = x.linvel.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
            let finite = |v: f64| if v.is_finite() { v } else { f64::MAX };
            max_drift = max_drift.max(finite(drift));
            sum_drift += finite(drift);
            max_dy = max_dy.max(finite(d[1].abs()));
            sum_dy += finite(d[1].abs());
            max_h = max_h.max(finite(h));
            sum_h += finite(h);
            max_speed = max_speed.max(finite(speed));
            sum_speed += finite(speed);
            min_disp = min_disp.min(finite(drift));
            for k in 0..3 {
                let p = x.position[k] as f64;
                mean[k] += p / n;
                lo[k] = lo[k].min(p);
                hi[k] = hi[k].max(p);
            }
        }
        let put = |m: &mut Map<String, Value>, k: &str, v: f64| {
            m.insert(k.into(), json!(if v.is_finite() { v } else { f64::MAX }));
        };
        put(&mut m, "max_drift", max_drift);
        put(&mut m, "mean_drift", sum_drift / n);
        put(&mut m, "max_abs_dy", max_dy);
        put(&mut m, "mean_abs_dy", sum_dy / n);
        put(&mut m, "max_horizontal_drift", max_h);
        put(&mut m, "mean_horizontal_drift", sum_h / n);
        put(&mut m, "max_speed", max_speed);
        put(&mut m, "mean_speed", sum_speed / n);
        put(&mut m, "min_displacement", min_disp);
        for (k, axis) in ["x", "y", "z"].iter().enumerate() {
            put(&mut m, &format!("pos_{axis}"), mean[k]);
            put(&mut m, &format!("min_pos_{axis}"), lo[k]);
            put(&mut m, &format!("max_pos_{axis}"), hi[k]);
        }
        m
    }

    /// Depths (≥ 0) of every pair closer than TOUCH.
    fn penetrations(&self, s: &[BodySample]) -> Vec<f32> {
        let n = s.len();
        let poses: Vec<Pose> = s.iter().map(pose).collect();
        let ok: Vec<bool> = s.iter().map(|x| x.position.iter().chain(&x.rotation).all(|v| v.is_finite())).collect();
        let large: HashSet<u32> = self.large.iter().copied().collect();
        let radius: Vec<f32> = self.scene.bodies.iter().map(|b| b.radius()).collect();
        let inv = 1.0 / self.cell;
        let key = |p: [f32; 3]| [(p[0] * inv).floor() as i32, (p[1] * inv).floor() as i32, (p[2] * inv).floor() as i32];
        let mut grid: HashMap<[i32; 3], Vec<u32>> = HashMap::new();
        for i in 0..n as u32 {
            if ok[i as usize] && !large.contains(&i) {
                grid.entry(key(s[i as usize].position)).or_default().push(i);
            }
        }
        let depth = |a: u32, b: u32| -> Option<f32> {
            let (ba, bb) = (&self.scene.bodies[a as usize], &self.scene.bodies[b as usize]);
            if (ba.is_static && bb.is_static) || self.joined.contains(&(a.min(b), a.max(b))) {
                return None;
            }
            let (pa, pb) = (s[a as usize].position, s[b as usize].position);
            let d2: f32 = (0..3).map(|k| (pa[k] - pb[k]).powi(2)).sum();
            let r = radius[a as usize] + radius[b as usize] + TOUCH;
            if r.is_finite() && d2 > r * r {
                return None;
            }
            // Closest hull pair between the two bodies; `None` if no piece is within TOUCH
            let mut dist: Option<f32> = None;
            for ga in &self.shapes[a as usize] {
                for gb in &self.shapes[b as usize] {
                    if let Ok(Some(c)) = parry3d::query::contact(&poses[a as usize], &**ga, &poses[b as usize], &**gb, TOUCH) {
                        dist = Some(dist.map_or(c.dist, |d: f32| d.min(c.dist)));
                    }
                }
            }
            Some((-dist?).max(0.0))
        };
        // Small-small pairs from the grid (each pair once: b > a), then large against everything
        let mut out: Vec<f32> = (0..n as u32)
            .into_par_iter()
            .filter(|&a| ok[a as usize] && !large.contains(&a))
            .flat_map_iter(|a| {
                let c = key(s[a as usize].position);
                let mut found = Vec::new();
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            if let Some(cell) = grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) {
                                for &b in cell {
                                    if b > a {
                                        if let Some(d) = depth(a, b) {
                                            found.push(d);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                found
            })
            .collect();
        for (k, &a) in self.large.iter().enumerate() {
            if !ok[a as usize] {
                continue;
            }
            let more: Vec<f32> = (0..n as u32)
                .into_par_iter()
                .filter(|&b| b != a && ok[b as usize] && !(large.contains(&b) && self.large[..k].contains(&b)))
                .filter_map(|b| depth(a, b))
                .collect();
            out.extend(more);
        }
        out
    }

    fn gates(&self, metrics: &[(u32, Map<String, Value>)]) -> Vec<Value> {
        self.scene
            .meta
            .gates
            .iter()
            .map(|g| {
                let at = metrics.iter().find(|(s, _)| *s == g.at).map(|(_, m)| m);
                let value = at.and_then(|m| match &g.group {
                    Some(group) => m["groups"][group.as_str()][g.metric.as_str()].as_f64(),
                    None => m.get(&g.metric).and_then(|v| v.as_f64()),
                });
                let status = match (at, value) {
                    (None, _) => "not reached",
                    (Some(_), None) => "no metric",
                    (Some(_), Some(v)) if g.max.is_some_and(|mx| !(v <= mx)) || g.min.is_some_and(|mn| !(v >= mn)) => "fail",
                    _ => "pass",
                };
                json!({ "at": g.at, "metric": g.metric, "group": g.group, "max": g.max, "min": g.min, "value": value, "status": status, "why": g.why })
            })
            .collect()
    }
}
